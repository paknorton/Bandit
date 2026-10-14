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

from collections import OrderedDict
from cyclopts import App, Parameter
from pathlib import Path
from rich.progress import (BarColumn, MofNCompleteColumn, Progress, SpinnerColumn,
                           TaskProgressColumn, TextColumn, TimeElapsedColumn,
                           TimeRemainingColumn)
from rich.table import Table
from typing import Annotated, NamedTuple

from pyPRMS.base.console import get_console_instance   # type: ignore
from pyPRMS.metadata.metadata import MetaData   # type: ignore
from pyPRMS import ParamDb   # type: ignore

from Bandit import bandit_cfg as bc
from Bandit.exceptions import BanditError

__author__ = 'Parker Norton (pnorton@usgs.gov)'

# Rich console for user-facing output
con = get_console_instance()

# Default arguments passed through to each bandit extraction when the
# user does not supply any on the command line.
DEFAULT_BANDIT_ARGS = ['--no_filter_params', '--keep_hru_order']

# bandit uses return codes >= this value to report extraction errors that do
# not require shutting the whole batch down. Smaller non-zero codes indicate a
# failure that should stop the batch.
BANDIT_ERROR_FLOOR = 200

# Arguments that this driver sets for each extraction and so must not be
# supplied as pass-through arguments.
RESERVED_BANDIT_ARGS = {'--job-dir', '-j', '--config-file'}


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


def read_file(filename: str | Path) -> dict:
    """Read a csv with single ID followed by one or more values

    The format of the file is CSV with a comma separator with one poi_id (string) per row
    followed by one or more integer HRU IDs or segment IDs
        poi_id,ids
        00000000,4,67,8,...

    :param filename: File containing one ID per line"""

    values_by_id = {}

    src_file = open(filename, 'r')

    # Skip the header row
    src_file.readline()

    # Read in the non-routed HRUs or segment cutoffs by POI
    for line in src_file:
        cols = line.strip().replace(' ', '').split(',')
        values_by_id[cols[0]] = [int(xx) for xx in cols[1:]]

    return values_by_id


def get_streamgage_segments(filename: str | Path,
                            poi_id_to_seg: dict) -> tuple['OrderedDict[str, int]',
                                                          'OrderedDict[str, str]']:
    """Map streamgage POI IDs from a file to NHM POI segments.

    The input file contains one streamgage POI ID per line. Blank lines and
    surrounding whitespace are ignored. Streamgages that are not POIs in the
    parameter database, that have ``poi_gage_segment == 0``, or that are
    assigned multiple segments are reported and skipped.

    :param filename: File containing streamgage POI IDs (one per line)
    :param poi_id_to_seg: Mapping of poi_gage_id to poi_gage_segment
    :returns: A tuple of (valid streamgage ID -> NHM POI segment,
              skipped streamgage ID -> reason it was skipped)
    """

    with open(filename, 'r') as fhdl:
        streamgages = [ll.strip() for ll in fhdl.read().splitlines() if ll.strip()]

    # Create dictionary of streamgage IDs to nhm_seg for POI
    segs_by_poi: 'OrderedDict[str, int]' = OrderedDict()

    # Streamgages that could not be used, mapped to the reason they were skipped
    skipped: 'OrderedDict[str, str]' = OrderedDict()

    for kk in streamgages:
        poiseg = poi_id_to_seg.get(kk, None)

        if poiseg is None:
            reason = 'not a POI in the parameter database'
            con.print(f'Streamgage {kk} is {reason}; skipping', style='dark_orange3')
            skipped[kk] = reason
        elif poiseg == 0:
            reason = 'poi_gage_segment = 0'
            con.print(f'Streamgage {kk} has {reason}; skipping', style='dark_orange3')
            skipped[kk] = reason
        elif kk in segs_by_poi:
            reason = 'multiple assigned segments in the parameter database'
            con.print(f'Streamgage {kk} has {reason}; skipping', style='red')
            skipped[kk] = reason
        else:
            segs_by_poi[kk] = poiseg

    return segs_by_poi, skipped


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


# A summary row: (streamgage, HRUs, segments, POIs, status). Count fields are
# ints for completed runs, otherwise a placeholder string such as '-'.
SummaryRow = tuple[str, 'int | str', 'int | str', 'int | str', str]


def collect_summary_rows(dir_by_gage: 'OrderedDict[str, str]',
                         job_dir: str | Path,
                         skipped: 'OrderedDict[str, str] | None' = None) -> list[SummaryRow]:
    """Collect per-streamgage summary rows for the roll-up report.

    Each extraction's counts are read from the ``summary.json`` written by
    bandit into that run's output directory. A gage whose run did not produce
    a summary (e.g. it failed or was never started) is marked as not completed.
    Streamgages skipped before extraction (invalid POIs) are listed with the
    reason they were skipped.

    :param dir_by_gage: Mapping of streamgage ID to its output directory name
    :param job_dir: Top-level job directory the extractions ran under
    :param skipped: Mapping of skipped streamgage ID to the reason it was skipped
    :returns: List of summary rows, completed/failed first then skipped
    """

    rows: list[SummaryRow] = []
    for gage, cdir in dir_by_gage.items():
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

        rows.append((gage, num_hru, num_seg, num_poi, status))

    # Append rows for streamgages that were skipped before any extraction ran
    for gage, reason in (skipped or {}).items():
        rows.append((gage, '-', '-', '-', f'skipped: {reason}'))

    return rows


def print_summary_table(rows: list[SummaryRow]):
    """Print a summary table of the batch extraction to the console.

    :param rows: Summary rows from :func:`collect_summary_rows`
    """

    table = Table(title='Bandit extraction summary')
    table.add_column('Streamgage')
    table.add_column('HRUs', justify='right')
    table.add_column('Segments', justify='right')
    table.add_column('POIs', justify='right')
    table.add_column('Status')

    for gage, num_hru, num_seg, num_poi, status in rows:
        if status == 'completed':
            style = 'green4'
        elif status.startswith('skipped'):
            style = 'dark_orange3'
        else:
            style = 'red'

        table.add_row(str(gage), str(num_hru), str(num_seg), str(num_poi),
                      f'[{style}]{status}[/]')

    con.print(table)


def write_summary_report(report_path: str | Path, rows: list[SummaryRow]):
    """Write an HTML table summarizing each streamgage extraction.

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
             'td.gage, td.status { text-align: left; }',
             '</style>',
             '</head>',
             '<body>',
             '<h1>Bandit extraction summary</h1>',
             f'<p>Generated {html.escape(generated)}</p>',
             '<table>',
             '<thead>',
             '<tr><th>Streamgage</th><th>HRUs</th><th>Segments</th>'
             '<th>POIs</th><th>Status</th></tr>',
             '</thead>',
             '<tbody>']

    for gage, num_hru, num_seg, num_poi, status in rows:
        lines.append(f'<tr><td class="gage">{html.escape(str(gage))}</td>'
                     f'<td>{html.escape(str(num_hru))}</td>'
                     f'<td>{html.escape(str(num_seg))}</td>'
                     f'<td>{html.escape(str(num_poi))}</td>'
                     f'<td class="status">{html.escape(status)}</td></tr>')

    lines.extend(['</tbody>', '</table>', '</body>', '</html>', ''])

    with open(report_path, 'w') as fhdl:
        fhdl.write('\n'.join(lines))


app = App(default_parameter=Parameter(negative=()))


@app.default
def by_streamgage(streamgages: Annotated[Path, Parameter(name=['--streamgages', '-s'])],
                  *bandit_args: Annotated[str, Parameter(allow_leading_hyphen=True)],
                  seg_cutoffs: Annotated[Path | None, Parameter(name=['--seg_cutoffs', '-c'])] = None,
                  nr_hrus: Annotated[Path | None, Parameter(name=['--nr_hrus', '-n'])] = None,
                  prefix: Annotated[str | None, Parameter(name=['--prefix', '-p'])] = None,
                  num_threads: Annotated[int, Parameter(name=['--num-threads', '-N'])] = 8,
                  verbose: Annotated[bool, Parameter(name=['--verbose', '-v'])] = False):
    """Batch Bandit extractions driven by a file of streamgage POI IDs.

    Any extra tokens supplied after the known options are passed through,
    unchanged, to each ``bandit`` extraction. Use the ``--`` delimiter to
    separate pass-through flags from this command's own options, e.g.::

        bandit_by_streamgage -s sample.csv -- --param_netcdf --streamflow_netcdf

    When no pass-through arguments are supplied, the defaults
    ``--no_filter_params --keep_hru_order`` are used.

    :param streamgages: File containing streamgage POI IDs (one per line)
    :param bandit_args: Additional arguments passed through to each bandit run
    :param seg_cutoffs: File containing upstream cutoff segments for model by POI
    :param nr_hrus: File containing non-route HRUs in model by POI
    :param prefix: Directory prefix to add to each extraction output directory
    :param num_threads: Number of concurrent bandit extractions to run
    :param verbose: Output additional information
    """

    streamgages = Path(streamgages)

    if not streamgages.is_file():
        raise BanditError(f'Streamgage file does not exist: {streamgages}', exit_code=1)

    if num_threads < 1:
        raise BanditError(f'num_threads must be at least 1 (got {num_threads})', exit_code=1)

    # Arguments to pass through to each bandit extraction
    passthrough_args = list(bandit_args) if bandit_args else list(DEFAULT_BANDIT_ARGS)

    # Reject pass-through arguments that this driver controls for each run.
    reserved_used = [aa for aa in passthrough_args if aa in RESERVED_BANDIT_ARGS]
    if reserved_used:
        raise BanditError(f'These bandit arguments are set by bandit_by_streamgage and '
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

    # List of upstream cutoffs (not allowed with by-streamgage extractions)
    config.update_value('cutoffs', [])

    # Load PRMS metadata
    prms_meta = MetaData(verbose=False).metadata

    # Get the POI-to-segment mappings from the parameter database
    pdb = ParamDb(config.paramdb_dir, metadata=prms_meta, verbose=verbose)

    # Get dictionary which maps poi_gage_id to poi_gage_segment
    poi_id_to_seg = pdb.poi_to_seg

    # Read streamgage file and map IDs to NHM POI segments
    segs_by_poi, skipped_gages = get_streamgage_segments(streamgages, poi_id_to_seg)

    if not segs_by_poi:
        raise BanditError('No valid streamgages found; nothing to extract', exit_code=1)

    if seg_cutoffs is None:
        us_seg_cutoffs = {}
    else:
        if not seg_cutoffs.is_file():
            raise BanditError(f'Upstream segment cutoff file does not exist: {seg_cutoffs}', exit_code=1)
        else:
            us_seg_cutoffs = read_file(seg_cutoffs)

    if nr_hrus is None:
        nonrouted_hrus = {}
    else:
        if not nr_hrus.is_file():
            raise BanditError(f'Non-routed HRUs file does not exist: {nr_hrus}', exit_code=1)
        else:
            nonrouted_hrus = read_file(nr_hrus)

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Setup validated and there is work to do: initialize the worker pool.
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    cmd_q: queue.Queue = queue.Queue()
    result_q: queue.Queue = queue.Queue()

    # No point starting more workers than there are extractions to run
    pool_size = min(num_threads, len(segs_by_poi))

    # Create pool of threads
    pool = [WorkerThread(input_q=cmd_q, result_q=result_q) for __ in range(pool_size)]

    # Start the threads
    for thread in pool:
        thread.start()

    work_count = 0

    # Map each streamgage to the directory its extraction runs in, so the
    # per-run summaries can be collected into a roll-up report afterward.
    dir_by_gage: 'OrderedDict[str, str]' = OrderedDict()

    for kk, vv in segs_by_poi.items():
        if prefix is None:
            cdir = kk
        else:
            cdir = f'{prefix}{kk}'

        dir_by_gage[kk] = cdir

        # Create the directory if needed
        if not os.path.exists(cdir):
            try:
                os.makedirs(cdir)
            except OSError as err:
                raise BanditError(f'Error creating directory: {err}', exit_code=1)

        # Update the outlets and output_dir config variables.
        # outlets is a list of segments; a streamgage maps to a single segment.
        config.update_value('outlets', [vv])
        config.update_value('cutoffs',  us_seg_cutoffs.get(kk, []))
        config.update_value('hru_noroute', nonrouted_hrus.get(kk, []))
        config.update_value('output_dir', f'{job_dir}/{cdir}')

        config.write(f'{cdir}/bandit.cfg')

        # Add bandit job to the processing queue
        work_count += 1
        cmd = [cmd_bandit] + passthrough_args + ['--job-dir', f'{job_dir}/{cdir}']

        cmd_q.put(cmd)

    total_jobs = work_count
    con.print(f'Total number of streamgages: {total_jobs}')

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
    rows = collect_summary_rows(dir_by_gage, job_dir, skipped_gages)
    print_summary_table(rows)

    report = Path(job_dir) / 'extraction_summary.html'
    write_summary_report(report, rows)
    con.print(f'Summary report: {report}')

    if aborted:
        raise BanditError('Batch extraction stopped before all streamgages were processed',
                          exit_code=1)


def main():
    """CLI entry point for batch Bandit extractions by streamgage."""
    try:
        app()
    except BanditError as err:
        bandit_log.error(str(err))
        con.print(f'[red]ERROR[/]: {err}')
        sys.exit(err.exit_code)


if __name__ == '__main__':
    main()
