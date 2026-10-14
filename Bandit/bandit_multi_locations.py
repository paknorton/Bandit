#!/usr/bin/env python3

from __future__ import annotations

import datetime
import html
import json
import logging
import os
import queue
import shutil
import subprocess
import sys
import threading

from cyclopts import App, Parameter
from pathlib import Path
from rich.progress import (BarColumn, MofNCompleteColumn, Progress, SpinnerColumn,
                           TaskProgressColumn, TextColumn, TimeElapsedColumn,
                           TimeRemainingColumn)
from rich.table import Table
from typing import Annotated, NamedTuple, Union

from pyPRMS.base.console import get_console_instance   # type: ignore

from Bandit import bandit_cfg as bc
from Bandit.exceptions import BanditError

__author__ = 'Parker Norton (pnorton@usgs.gov)'

"""Generate model extractions for named locations in the NHM.

Each location has one or more stream segment outlets and, optionally, upstream
segment cutoffs and non-routed HRUs. The outlets file has the format::

    name,outlets
    mill,1242
    norwalk,1244,1245
    quinnipiac,868

The location name in the first column may be a string (e.g. ``mill``) or an
integer (e.g. a headwater area ID). The optional cutoffs and non-routed HRU
files use the same first-column location key followed by integer segment or
HRU IDs.
"""

# Rich console for user-facing output
con = get_console_instance()

# Default arguments passed through to each bandit extraction when the
# user does not supply any on the command line.
DEFAULT_BANDIT_ARGS = ['--keep_hru_order']

# bandit uses return codes >= this value to report extraction errors that do
# not require shutting the whole batch down. Smaller non-zero codes indicate a
# failure that should stop the batch.
BANDIT_ERROR_FLOOR = 200

# Arguments that this driver sets for each extraction and so must not be
# supplied as pass-through arguments.
RESERVED_BANDIT_ARGS = {'--job-dir', '-j', '--config-file'}

# A location key may be a string name or an integer ID.
LocationId = Union[str, int]


class Result(NamedTuple):
    """Result of running one bandit extraction in a worker thread."""

    thread: str          # Name of the worker thread that ran the command
    cmd: list[str]       # The command that was run
    retcode: int         # Process return code (-1 if it failed to launch)


# Setup logging
bandit_log = logging.getLogger('bandit')


class WorkerThread(threading.Thread):
    """A worker thread that pulls bandit commands off a queue and runs them.

    Input is done by placing commands (as lists of tokens) into the queue
    passed in ``input_q``.

    Output is done by placing tuples into the queue passed in ``result_q``.
    Each tuple is (thread name, command, return code).

    Ask the thread to stop by calling its :meth:`join` method.
    """

    def __init__(self, input_q: queue.Queue, result_q: queue.Queue):
        """Init method for WorkerThread.

        :param input_q: Queue of commands to run
        :param result_q: Queue to place command results into
        """
        super().__init__()
        # Daemon threads so an unhandled interrupt in the main thread cannot
        # be blocked indefinitely waiting on a worker to finish.
        self.daemon = True
        self.input_q = input_q
        self.result_q = result_q
        self.stoprequest = threading.Event()

    def run(self):
        """Pull commands from the input queue and run them until asked to stop.

        The tasks are taken with a blocking ``get`` given a short timeout so
        that ``stoprequest`` is always checked, even when the queue is empty.
        """
        while not self.stoprequest.is_set():
            try:
                thecmd = self.input_q.get(True, 0.05)
                retcode = self.run_cmd(thecmd)
                self.result_q.put(Result(thread=self.name, cmd=thecmd, retcode=retcode))
            except queue.Empty:
                continue

    def join(self, timeout=None):
        """Request the thread stop and wait for it to finish.

        :param timeout: Optional timeout in seconds
        """
        self.stoprequest.set()
        super().join(timeout)

    def run_cmd(self, thecmd: list[str]) -> int:
        """Run a single bandit command in a subprocess.

        :param thecmd: Command to run, as a list of argument tokens
        :returns: Return code from the subprocess (-1 on failure to launch)
        """
        retcode = 0

        try:
            retcode = subprocess.call(thecmd)
        except Exception as err:   # noqa: BLE001
            # Error launching the command; shut this worker down
            bandit_log.error(f'run_cmd() failed to launch {thecmd}: {err}')
            retcode = -1
            self.stoprequest.set()
        return retcode


def _location_key(raw: str) -> LocationId:
    """Normalize a raw location key to an int when possible, else a string.

    :param raw: Location key as read from the CSV first column
    :returns: Integer location ID if ``raw`` is an integer, otherwise the string
    """
    try:
        return int(raw)
    except ValueError:
        return raw


def read_outlets(filename: str | Path) -> 'dict[LocationId, list[int]]':
    """Read a CSV of a location ID followed by one or more integer values.

    The file has a header row, then one location ID per row followed by one or
    more integer segment outlet IDs::

        name,outlets
        mill,1242
        norwalk,1244,1245

    The location ID may be a string name or an integer. Blank lines and
    surrounding whitespace are ignored. Insertion order is preserved.

    :param filename: File to read
    :returns: Mapping of location ID to its list of integer values
    :raises BanditError: If a non-header row has no values, a non-integer value,
                         or a duplicate location ID
    """

    values_by_id: 'dict[LocationId, list[int]]' = {}

    with open(filename, 'r') as src_file:
        # Skip the header row
        src_file.readline()

        for line in src_file:
            cols = [cc.strip() for cc in line.strip().split(',')]

            if not cols or not cols[0]:
                # Skip blank lines (and lines with no location ID)
                continue

            loc = _location_key(cols[0])

            if loc in values_by_id:
                raise BanditError(f'Duplicate location {loc} in {filename}', exit_code=1)

            if len(cols) < 2 or not any(cols[1:]):
                raise BanditError(f'Location {loc} in {filename} has no values', exit_code=1)

            try:
                values_by_id[loc] = [int(xx) for xx in cols[1:] if xx]
            except ValueError as err:
                raise BanditError(f'Non-integer value in {filename} for location {loc}: {err}',
                                  exit_code=1)

    return values_by_id


def read_values(filename: str | Path) -> 'dict[LocationId, list[int]]':
    """Read a CSV of a location ID followed by zero or more integer values.

    Used for the optional upstream segment cutoff and non-routed HRU files,
    which share the outlets-file format but may legitimately have no values for
    a location.

    :param filename: File to read
    :returns: Mapping of location ID to its list of integer values
    :raises BanditError: If a non-header row contains a non-integer value or a
                         duplicate location ID
    """

    values_by_id: 'dict[LocationId, list[int]]' = {}

    with open(filename, 'r') as src_file:
        # Skip the header row
        src_file.readline()

        for line in src_file:
            cols = [cc.strip() for cc in line.strip().split(',')]

            if not cols or not cols[0]:
                # Skip blank lines (and lines with no location ID)
                continue

            loc = _location_key(cols[0])

            if loc in values_by_id:
                raise BanditError(f'Duplicate location {loc} in {filename}', exit_code=1)

            try:
                values_by_id[loc] = [int(xx) for xx in cols[1:] if xx]
            except ValueError as err:
                raise BanditError(f'Non-integer value in {filename} for location {loc}: {err}',
                                  exit_code=1)

    return values_by_id


def location_dirname(loc: LocationId, prefix: str | None) -> str:
    """Build the output directory name for a location.

    Integer location IDs are zero-padded to four digits (e.g. ``0003``) to
    match the historical headwater-area naming; string names are used as-is.

    :param loc: Location ID (string name or integer)
    :param prefix: Optional directory prefix to prepend
    :returns: Directory name for the location's extraction
    """
    if isinstance(loc, int):
        base = f'{loc:04d}'
    else:
        base = f'{loc}'

    return base if prefix is None else f'{prefix}{base}'


def stop_pool(pool: list[WorkerThread], cmd_q: queue.Queue):
    """Stop a pool of worker threads, abandoning any un-started work.

    Any commands still queued are discarded so that no new extractions start.
    Each worker is then asked to stop and briefly joined. Workers are daemon
    threads, so an extraction still running past the join timeout does not
    block process exit (its output directory may be left incomplete).

    :param pool: Worker threads to stop
    :param cmd_q: Command queue to drain of un-started work
    """

    # Discard any queued commands that have not been picked up yet
    while True:
        try:
            cmd_q.get_nowait()
        except queue.Empty:
            break

    for thread in pool:
        thread.join(timeout=5.0)


# A summary row: (location, HRUs, segments, POIs, status). Count fields are
# ints for completed runs, otherwise a placeholder string such as '-'.
SummaryRow = tuple[str, 'int | str', 'int | str', 'int | str', str]


def collect_summary_rows(dir_by_loc: 'dict[LocationId, str]',
                         job_dir: str | Path) -> list[SummaryRow]:
    """Collect per-location summary rows for the roll-up report.

    Each extraction's counts are read from the ``summary.json`` written by
    bandit into that run's output directory. A location whose run did not
    produce a summary (e.g. it failed or was never started) is marked as not
    completed.

    :param dir_by_loc: Mapping of location ID to its output directory name
    :param job_dir: Top-level job directory the extractions ran under
    :returns: List of summary rows
    """

    rows: list[SummaryRow] = []
    for loc, cdir in dir_by_loc.items():
        summary_file = Path(job_dir) / cdir / 'summary.json'

        num_hru: int | str
        num_seg: int | str
        num_poi: int | str
        status: str

        if summary_file.is_file():
            try:
                with open(summary_file, 'r') as fhdl:
                    data = json.load(fhdl)
                num_hru = data.get('num_hru', '?')
                num_seg = data.get('num_segments', '?')
                num_poi = data.get('num_poi', '?')
                status = 'completed'
            except (OSError, json.JSONDecodeError):
                num_hru = num_seg = num_poi = '-'
                status = 'unreadable summary'
        else:
            num_hru = num_seg = num_poi = '-'
            status = 'not completed'

        rows.append((str(loc), num_hru, num_seg, num_poi, status))

    return rows


def print_summary_table(rows: list[SummaryRow]):
    """Print a summary table of the batch extraction to the console.

    :param rows: Summary rows from :func:`collect_summary_rows`
    """

    table = Table(title='Bandit extraction summary')
    table.add_column('Location')
    table.add_column('HRUs', justify='right')
    table.add_column('Segments', justify='right')
    table.add_column('POIs', justify='right')
    table.add_column('Status')

    for loc, num_hru, num_seg, num_poi, status in rows:
        if status == 'completed':
            style = 'green4'
        elif status.startswith('skipped'):
            style = 'dark_orange3'
        else:
            style = 'red'

        table.add_row(str(loc), str(num_hru), str(num_seg), str(num_poi),
                      f'[{style}]{status}[/]')

    con.print(table)


def write_summary_report(report_path: str | Path, rows: list[SummaryRow]):
    """Write an HTML table summarizing each location extraction.

    :param report_path: Path of the HTML report to write
    :param rows: Summary rows from :func:`collect_summary_rows`
    """

    generated = datetime.datetime.now().isoformat(timespec='seconds')

    lines = ['<!DOCTYPE html>',
             '<html lang="en">',
             '<head>',
             '<meta charset="utf-8">',
             '<title>Bandit extraction summary</title>',
             '<style>',
             'body { font-family: sans-serif; margin: 2em; }',
             'table { border-collapse: collapse; }',
             'th, td { border: 1px solid #ccc; padding: 4px 10px; text-align: right; }',
             'th { background: #f0f0f0; }',
             'td.loc, td.status { text-align: left; }',
             '</style>',
             '</head>',
             '<body>',
             '<h1>Bandit extraction summary</h1>',
             f'<p>Generated {html.escape(generated)}</p>',
             '<table>',
             '<thead>',
             '<tr><th>Location</th><th>HRUs</th><th>Segments</th>'
             '<th>POIs</th><th>Status</th></tr>',
             '</thead>',
             '<tbody>']

    for loc, num_hru, num_seg, num_poi, status in rows:
        lines.append(f'<tr><td class="loc">{html.escape(str(loc))}</td>'
                     f'<td>{html.escape(str(num_hru))}</td>'
                     f'<td>{html.escape(str(num_seg))}</td>'
                     f'<td>{html.escape(str(num_poi))}</td>'
                     f'<td class="status">{html.escape(status)}</td></tr>')

    lines.extend(['</tbody>', '</table>', '</body>', '</html>', ''])

    with open(report_path, 'w') as fhdl:
        fhdl.write('\n'.join(lines))


app = App(default_parameter=Parameter(negative=()))


@app.default
def multi_locations(seg_outlets: Annotated[Path, Parameter(name=['--seg_outlets', '-s'])],
                    *bandit_args: Annotated[str, Parameter(allow_leading_hyphen=True)],
                    seg_cutoffs: Annotated[Path | None, Parameter(name=['--seg_cutoffs', '-c'])] = None,
                    nr_hrus: Annotated[Path | None, Parameter(name=['--nr_hrus', '-n'])] = None,
                    prefix: Annotated[str | None, Parameter(name=['--prefix', '-p'])] = None,
                    num_threads: Annotated[int, Parameter(name=['--num-threads', '-N'])] = 8,
                    verbose: Annotated[bool, Parameter(name=['--verbose', '-v'])] = False):
    """Batch Bandit extractions driven by a file of named locations.

    Each location is extracted to its own directory and may have one or more
    stream segment outlets. Per-location upstream segment cutoffs and non-routed
    HRUs can optionally be supplied as CSV files (see ``seg_cutoffs`` and
    ``nr_hrus``) and are applied to the matching location.

    Any extra tokens supplied after the known options are passed through,
    unchanged, to each ``bandit`` extraction. Use the ``--`` delimiter to
    separate pass-through flags from this command's own options, e.g.::

        bandit_multi_locations -s basin_outlets.csv -- --param_netcdf

    When no pass-through arguments are supplied, the default ``--keep_hru_order``
    is used.

    :param seg_outlets: File containing segment outlets by location
    :param bandit_args: Additional arguments passed through to each bandit run
    :param seg_cutoffs: File containing upstream cutoff segments by location
    :param nr_hrus: File containing non-routed HRUs by location
    :param prefix: Directory prefix to add to each extraction output directory
    :param num_threads: Number of concurrent bandit extractions to run
    :param verbose: Output additional information
    """

    seg_outlets = Path(seg_outlets)

    if not seg_outlets.is_file():
        raise BanditError(f'Segment outlets file does not exist: {seg_outlets}', exit_code=1)

    if num_threads < 1:
        raise BanditError(f'num_threads must be at least 1 (got {num_threads})', exit_code=1)

    # Arguments to pass through to each bandit extraction
    passthrough_args = list(bandit_args) if bandit_args else list(DEFAULT_BANDIT_ARGS)

    # Reject pass-through arguments that this driver controls for each run.
    reserved_used = [aa for aa in passthrough_args if aa in RESERVED_BANDIT_ARGS]
    if reserved_used:
        raise BanditError(f'These bandit arguments are set by bandit_multi_locations and '
                          f'cannot be passed through: {" ".join(reserved_used)}', exit_code=1)

    if verbose:
        con.print(f'Pass-through bandit arguments: {" ".join(passthrough_args)}')

    # Should be in the current job directory
    job_dir = os.getcwd()

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Fail fast: validate everything that can go wrong before spawning any
    # worker threads. Nothing below should start the thread pool until there
    # is confirmed work to queue.
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    cmd_bandit = shutil.which('bandit')

    if not cmd_bandit:
        raise BanditError('Unable to find bandit executable', exit_code=1)

    # Read the default configuration file
    config_file = Path(job_dir) / 'bandit.cfg'

    if not config_file.is_file():
        raise BanditError(f'Configuration file does not exist: {config_file}', exit_code=1)

    config = bc.Cfg(config_file)

    # Update control_filename to include the main extraction path
    config.update_value('control_filename', f'{job_dir}/control.default')

    # Read the segment outlets by location
    segments_by_loc = read_outlets(seg_outlets)

    if not segments_by_loc:
        raise BanditError('No locations found; nothing to extract', exit_code=1)

    if seg_cutoffs is None:
        uscutoff_seg_by_loc: dict = {}
    else:
        seg_cutoffs = Path(seg_cutoffs)
        if not seg_cutoffs.is_file():
            raise BanditError(f'Upstream segment cutoff file does not exist: {seg_cutoffs}', exit_code=1)
        uscutoff_seg_by_loc = read_values(seg_cutoffs)

    if nr_hrus is None:
        noroute_hrus_by_loc: dict = {}
    else:
        nr_hrus = Path(nr_hrus)
        if not nr_hrus.is_file():
            raise BanditError(f'Non-routed HRUs file does not exist: {nr_hrus}', exit_code=1)
        noroute_hrus_by_loc = read_values(nr_hrus)

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Setup validated and there is work to do: initialize the worker pool.
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    cmd_q: queue.Queue = queue.Queue()
    result_q: queue.Queue = queue.Queue()

    # No point starting more workers than there are extractions to run
    pool_size = min(num_threads, len(segments_by_loc))

    # Create pool of threads
    pool = [WorkerThread(input_q=cmd_q, result_q=result_q) for __ in range(pool_size)]

    # Start the threads
    for thread in pool:
        thread.start()

    work_count = 0

    # Map each location to the directory its extraction runs in, so the
    # per-run summaries can be collected into a roll-up report afterward.
    dir_by_loc: 'dict[LocationId, str]' = {}

    for kk, vv in segments_by_loc.items():
        cdir = location_dirname(kk, prefix)

        dir_by_loc[kk] = cdir

        # Create the location directory if needed
        if not os.path.exists(cdir):
            try:
                os.makedirs(cdir)
            except OSError as err:
                raise BanditError(f'Error creating directory: {err}', exit_code=1)

        # Update the outlets, cutoffs, non-routed HRUs, and output_dir config
        # variables. outlets is a list of one or more segments.
        config.update_value('outlets', vv)
        config.update_value('cutoffs', uscutoff_seg_by_loc.get(kk, []))
        config.update_value('hru_noroute', noroute_hrus_by_loc.get(kk, []))
        config.update_value('output_dir', f'{job_dir}/{cdir}')

        config.write(f'{cdir}/bandit.cfg')

        # Add bandit job to the processing queue
        work_count += 1
        cmd = [cmd_bandit] + passthrough_args + ['--job-dir', f'{job_dir}/{cdir}']

        cmd_q.put(cmd)

    total_jobs = work_count
    con.print(f'Total number of locations: {total_jobs}')

    # Collect results, displaying a progress bar as extractions complete
    progress = Progress(SpinnerColumn(),
                        TextColumn('[progress.description]{task.description}'),
                        BarColumn(),
                        MofNCompleteColumn(),
                        TaskProgressColumn(),
                        TextColumn('•'),
                        TimeElapsedColumn(),
                        TextColumn('elapsed •'),
                        TimeRemainingColumn(),
                        TextColumn('remaining'),
                        console=con)

    aborted = False

    with progress:
        task = progress.add_task('Extracting', total=total_jobs)

        remaining = total_jobs
        while remaining > 0:
            try:
                result = result_q.get()
            except KeyboardInterrupt:
                con.print('\n[gold3]Interrupted[/]: stopping remaining extractions')
                aborted = True
                break

            progress.advance(task)
            remaining -= 1

            if result.retcode != 0 and result.retcode < BANDIT_ERROR_FLOOR:
                # An error occurred running the command. Return codes at or
                # above BANDIT_ERROR_FLOOR are generated by bandit extraction
                # errors that don't necessitate shutting the entire job down.
                con.print(f'[red]ERROR[/]: Thread {result.thread} return code = {result.retcode} '
                          f'({" ".join(result.cmd)})')
                aborted = True
                break

    # Stop the workers, discarding any extractions that have not yet started.
    # Extractions already running are allowed to finish.
    stop_pool(pool, cmd_q)

    # Build a roll-up report from the per-run summaries. This runs even on an
    # abort so the report reflects whatever extractions did complete.
    rows = collect_summary_rows(dir_by_loc, job_dir)
    print_summary_table(rows)

    report = Path(job_dir) / 'extraction_summary.html'
    write_summary_report(report, rows)
    con.print(f'Summary report: {report}')

    if aborted:
        raise BanditError('Batch extraction stopped before all locations were processed',
                          exit_code=1)


def main():
    """CLI entry point for batch Bandit extractions by location."""
    try:
        app()
    except BanditError as err:
        bandit_log.error(str(err))
        con.print(f'[red]ERROR[/]: {err}')
        sys.exit(err.exit_code)


if __name__ == '__main__':
    main()
