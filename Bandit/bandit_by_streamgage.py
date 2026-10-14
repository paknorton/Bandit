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

    # Read the default configuration file
    config = bc.Cfg(f'{job_dir}/bandit.cfg')

    # Update control_filename to include the main extraction path
    config.update_value('control_filename', f'{job_dir}/control.default')

    # List of upstream cutoffs (not allowed with by-streamgage extractions)
    config.update_value('cutoffs', [])

    cmd_bandit = shutil.which('bandit')

    if not cmd_bandit:
        raise BanditError('Unable to find bandit executable', exit_code=1)

    # ****************************************************************************
    # Initialize the threads
    cmd_q: queue.Queue = queue.Queue()
    result_q: queue.Queue = queue.Queue()

    # Create pool of threads
    pool = [WorkerThread(input_q=cmd_q, result_q=result_q) for __ in range(num_threads)]

    # Start the threads
    for thread in pool:
        try:
            thread.start()
        except (KeyboardInterrupt, SystemExit):
            # Shutdown the threads when the program is terminated
            con.print('Program terminated.')
            thread.join()
            sys.exit(1)

    # Load PRMS metadata
    prms_meta = MetaData(verbose=False).metadata

    # Get the POI-to-segment mappings from the parameter database
    pdb = ParamDb(config.paramdb_dir, metadata=prms_meta, verbose=verbose)

    # Get dictionary which maps poi_gage_id to poi_gage_segment
    poi_id_to_seg = pdb.poi_to_seg

    # Read streamgage file and map IDs to NHM POI segments
    segs_by_poi = get_streamgage_segments(streamgages, poi_id_to_seg)

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

    con.print(f'Total number of streamgages: {work_count}')

    # Output results
    while work_count > 0:
        result = result_q.get()

        sys.stdout.write(f'\rExtractions remaining: {work_count:4d}')
        sys.stdout.flush()

        work_count -= 1

        if result[2] != 0 and result[2] < 200:
            # An error occurred running the command
            # Return-codes greater than 200 are generated by bandit errors that
            # don't necessitate shutting the entire job down.
            con.print(f'\nThread {result[0]} return code = {result[2]} ({" ".join(result[1])})')
            work_count = 0

    # Ask the threads to die and wait for them to do it
    for thread in pool:
        thread.join()

    # Finish the in-place progress line with a newline
    sys.stdout.write('\n')
    sys.stdout.flush()


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
