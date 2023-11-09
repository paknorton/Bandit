#!/usr/bin/env python3

from Bandit import bandit_cfg as bc
from pyPRMS.metadata.metadata import MetaData
from pyPRMS import ParamDb

from collections import OrderedDict
import os
import queue
import subprocess
import sys
import threading

from rich.console import Console
from rich import pretty
pretty.install()

con = Console()

class WorkerThread(threading.Thread):
    """ A worker thread that takes directory names from a queue, finds all
        files in them recursively and reports the result.

        Input is done by placing directory names (as strings) into the
        Queue passed in dir_q.

        Output is done by placing tuples into the Queue passed in result_q.
        Each tuple is (thread name, dirname, [list of files]).

        Ask the thread to stop by calling its join() method.
    """
    def __init__(self, input_q, result_q):
        super(WorkerThread, self).__init__()
        self.input_q = input_q
        self.result_q = result_q
        self.stoprequest = threading.Event()

    def run(self):
        # As long as we weren't asked to stop, try to take new tasks from the
        # queue. The tasks are taken with a blocking 'get', so no CPU
        # cycles are wasted while waiting.
        # Also, 'get' is given a timeout, so stoprequest is always checked,
        # even if there's nothing in the queue.
        while not self.stoprequest.is_set():
            try:
                thecmd = self.input_q.get(True, 0.05)
                retcode = self.run_cmd(thecmd)
                self.result_q.put((self.name, thecmd, retcode))
            except queue.Empty:
                continue

    def join(self, timeout=None):
        self.stoprequest.set()
        super(WorkerThread, self).join(timeout)

    def run_cmd(self, thecmd):
        retcode = 0

        try:
            retcode = subprocess.call(thecmd, shell=True)
        except:
            # Error running command
            print("Error: run_cmd() shutting down")
            print(retcode)
            retcode = -1
            self.stoprequest.set()
        return retcode


def get_streamgage_segments(filename, poi_id_to_seg):
    """Returns dictionary mapping streamgages to NHM POI segments"""

    with open(filename, 'r') as fhdl:
        streamgages = fhdl.read().splitlines()

    # Create dictionary of streamgage IDs to nhm_seg for POI
    segs_by_poi = OrderedDict()

    for kk in streamgages:
        poiseg = poi_id_to_seg.setdefault(kk, None)

        if poiseg is None:
            con.print(f'Streamgage {kk} is not a POI in the parameter database; skipping', style='dark_orange3')
        elif poiseg == 0:
            con.print(f'Streamgage {kk} has poi_gage_segment = 0; skipping', style='dark_orange3')
        elif kk in segs_by_poi:
            con.print(f'Streamgage {kk} has multiple assigned segments in the parameter database; skipping', style='red')
            print(f'    {kk} -> {segs_by_poi[kk]}')
            print(f'    {kk} -> {poiseg}')
        else:
            segs_by_poi[kk] = poiseg

    return segs_by_poi


def main():
    import argparse
    from distutils.spawn import find_executable

    # Command line arguments
    parser = argparse.ArgumentParser(description='Batch script for Bandit extractions')

    parser.add_argument('-s', '--streamgages', help='File containing streamgage POI IDs', default='', type=str)
    parser.add_argument('-p', '--prefix', help='Directory prefix to add')
    parser.add_argument('-v', '--verbose', help='Output additional information', action='store_true')

    args = parser.parse_args()

    if not args.streamgages:
        print('ERROR: Must specify the streamgage POI ID file.')
        exit(1)

    # Should be in the current job directory
    job_dir = os.getcwd()

    # Read the default configuration file
    config = bc.Cfg(f'{job_dir}/bandit.cfg')

    # Update control_filename to include the main extraction path
    config.update_value('control_filename', f'{job_dir}/control.default')

    # List of upstream cutoffs (not allowed with by-streamgage extractions)
    config.cutoffs = []

    cmd_bandit = find_executable('bandit')

    if not cmd_bandit:
        print('ERROR: Unable to find bandit executable')
        exit(1)

    num_threads = 8

    # ****************************************************************************
    # Initialize the threads
    cmd_q = queue.Queue()
    result_q = queue.Queue()

    # Create pool of threads
    pool = [WorkerThread(input_q=cmd_q, result_q=result_q) for __ in range(num_threads)]

    # Start the threads
    for thread in pool:
        try:
            thread.start()
        except (KeyboardInterrupt, SystemExit):
            # Shutdown the threads when the program is terminated
            print('Program terminated.')
            thread.join()
            sys.exit(1)

    # Load PRMS metadata
    prms_meta = MetaData(verbose=False).metadata

    # Get the POI-to-segment mappings from the parameter database
    pdb = ParamDb(config.paramdb_dir, metadata=prms_meta, verbose=args.verbose)
    nhm_params = pdb.parameters

    # Get dictionary which maps poi_gage_id to poi_gage_segment
    poi_id_to_seg = nhm_params.poi_to_seg

    # Read streamgage file and map IDs to NHM POI segments
    segs_by_poi = get_streamgage_segments(args.streamgages, poi_id_to_seg)

    work_count = 0

    for kk, vv in segs_by_poi.items():
        cdir = kk

        # Create the directory if needed
        if not os.path.exists(cdir):
            try:
                os.makedirs(cdir)
            except OSError as err:
                con.print(f'\tError creating directory: {err}', style='red')
                exit(1)

        # Update the outlets and output_dir config variables
        config.update_value('outlets', vv.item())
        config.update_value('output_dir', f'{job_dir}/{cdir}')

        config.write(f'{cdir}/bandit.cfg')

        # Add bandit job to the processing queue
        work_count += 1
        cmd = f'{cmd_bandit} --no_filter_params --keep_hru_order -j {job_dir}/{cdir}'

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
            # Return-codes greater than 200 are generated by bandit errors, that
            # don't necessitate shutting the entire job down.
            print(f'\nThread {result[0]} return code = {result[2]} ({result[1]})')
            work_count = 0

    # Ask for the threads to die and wait for them to do it
    for thread in pool:
        thread.join()


if __name__ == '__main__':
    main()
