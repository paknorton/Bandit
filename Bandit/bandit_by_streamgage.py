#!/usr/bin/env python3

from __future__ import annotations

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
from typing import Annotated

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
                self.result_q.put((self.name, thecmd, retcode))
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


def get_streamgage_segments(filename: str | Path,
                            poi_id_to_seg: dict) -> 'OrderedDict[str, int]':
    """Map streamgage POI IDs from a file to NHM POI segments.

    The input file contains one streamgage POI ID per line. Blank lines and
    surrounding whitespace are ignored. Streamgages that are not POIs in the
    parameter database, that have ``poi_gage_segment == 0``, or that are
    assigned multiple segments are reported and skipped.

    :param filename: File containing streamgage POI IDs (one per line)
    :param poi_id_to_seg: Mapping of poi_gage_id to poi_gage_segment
    :returns: Ordered mapping of valid streamgage ID to NHM POI segment
    """

    with open(filename, 'r') as fhdl:
        streamgages = [ll.strip() for ll in fhdl.read().splitlines() if ll.strip()]

    # Create dictionary of streamgage IDs to nhm_seg for POI
    segs_by_poi: 'OrderedDict[str, int]' = OrderedDict()

    for kk in streamgages:
        poiseg = poi_id_to_seg.get(kk, None)

        if poiseg is None:
            con.print(f'Streamgage {kk} is not a POI in the parameter database; skipping',
                      style='dark_orange3')
        elif poiseg == 0:
            con.print(f'Streamgage {kk} has poi_gage_segment = 0; skipping',
                      style='dark_orange3')
        elif kk in segs_by_poi:
            con.print(f'Streamgage {kk} has multiple assigned segments in the parameter database; skipping',
                      style='red')
        else:
            segs_by_poi[kk] = poiseg

    return segs_by_poi


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


app = App(default_parameter=Parameter(negative=()))


@app.default
def by_streamgage(streamgages: Annotated[Path, Parameter(name=['--streamgages', '-s'])],
                  *bandit_args: Annotated[str, Parameter(allow_leading_hyphen=True)],
                  prefix: Annotated[str | None, Parameter(name=['--prefix', '-p'])] = None,
                  num_threads: Annotated[int, Parameter(name=['--num-threads', '-n'])] = 8,
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
    :param prefix: Directory prefix to add to each extraction output directory
    :param num_threads: Number of concurrent bandit extractions to run
    :param verbose: Output additional information
    """

    streamgages = Path(streamgages)

    if not streamgages.is_file():
        raise BanditError(f'Streamgage file does not exist: {streamgages}', exit_code=1)

    # Arguments to pass through to each bandit extraction
    passthrough_args = list(bandit_args) if bandit_args else list(DEFAULT_BANDIT_ARGS)

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
    segs_by_poi = get_streamgage_segments(streamgages, poi_id_to_seg)

    if not segs_by_poi:
        raise BanditError('No valid streamgages found; nothing to extract', exit_code=1)

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Setup validated and there is work to do: initialize the worker pool.
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    cmd_q: queue.Queue = queue.Queue()
    result_q: queue.Queue = queue.Queue()

    # Create pool of threads
    pool = [WorkerThread(input_q=cmd_q, result_q=result_q) for __ in range(num_threads)]

    # Start the threads
    for thread in pool:
        thread.start()

    work_count = 0

    for kk, vv in segs_by_poi.items():
        if prefix is None:
            cdir = kk
        else:
            cdir = f'{prefix}{kk}'

        # Create the directory if needed
        if not os.path.exists(cdir):
            try:
                os.makedirs(cdir)
            except OSError as err:
                raise BanditError(f'Error creating directory: {err}', exit_code=1)

        # Update the outlets and output_dir config variables.
        # outlets is a list of segments; a streamgage maps to a single segment.
        config.update_value('outlets', [vv])
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

            if result[2] != 0 and result[2] < 200:
                # An error occurred running the command.
                # Return-codes greater than 200 are generated by bandit errors
                # that don't necessitate shutting the entire job down.
                con.print(f'[red]ERROR[/]: Thread {result[0]} return code = {result[2]} '
                          f'({" ".join(result[1])})')
                aborted = True
                break

    # Stop the workers, discarding any extractions that have not yet started.
    # Extractions already running are allowed to finish.
    stop_pool(pool, cmd_q)

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
