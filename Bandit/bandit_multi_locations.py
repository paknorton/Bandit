#!/usr/bin/env python3

from Bandit import bandit_cfg as bc

import os
import sys
from collections import OrderedDict
import threading
import queue

import subprocess


"""Example of code used to generate model extractions for headwaters in the NHM

Used a file with the following format:

hwAreaId, seg_id_nat, [variable length list of seg_id_nat]
0,13, 8, 9
1,14, 4, 3, 12, 11, 7, 10, 1, 5, 2, 6
2,18, 16

"""


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
        while not self.stoprequest.isSet():
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


def read_file(filename):
    """Read a csv with single ID followed by one or more values"""

    values_by_id = OrderedDict()

    src_file = open(filename, 'r')
    src_file.readline()

    # Read in the non-routed HRUs by location
    for line in src_file:
        cols = line.strip().replace(' ', '').split(',')
        try:
            # Assume first column is a number
            cols = [int(xx) for xx in cols]
            values_by_id[cols[0]] = cols[1:]
        except ValueError:
            # First column is probably a string
            values_by_id[cols[0]] = [int(xx) for xx in cols[1:]]
    return values_by_id


def main():
    import argparse
    from distutils.spawn import find_executable

    # Command line arguments
    parser = argparse.ArgumentParser(description='Batch script for Bandit extractions')

    # parser.add_argument('-j', '--jobdir', help='Job directory to work in')
    parser.add_argument('-s', '--segoutlets', help='File containing segment outlets by location')
    parser.add_argument('-n', '--nrhrus', help='File containing non-routed HRUs by location')
    parser.add_argument('-p', '--prefix', help='Directory prefix to add')

    # parser.add_argument('--check_DAG', help='Verify the streamflow network', action='store_true')

    args = parser.parse_args()

    # Should be in the current job directory
    job_dir = os.getcwd()

    # Read the default configuration file
    config = bc.Cfg(f'{job_dir}/bandit.cfg')

    if not args.segoutlets:
        print('ERROR: Must specify the segment outlets file.')
        exit(1)

    if args.nrhrus:
        noroute_hrus_by_loc = read_file(f'{job_dir}/{args.nrhrus}')
    else:
        noroute_hrus_by_loc = None

    segments_by_loc = read_file(f'{job_dir}/{args.segoutlets}')

    # jobdir = '/media/scratch/PRMS/bandit/jobs/hw_jobs'
    # default_config_file = '{}/bandit.cfg'.format(jobdir)

    cmd_bandit = find_executable('bandit_v2')

    if not cmd_bandit:
        print('ERROR: Unable to find bandit.py')
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

    # ### For each head_waters
    # - create directory hw_# (where # is the hwAreaId)
    # - copy default bandit.cfg into directory
    # - run bandit on the directory

    if not os.path.exists(job_dir):
        try:
            os.makedirs(job_dir)
        except OSError as err:
            print(f'\tError creating directory: {err}')
            exit(1)

    # st_dir = os.getcwd()
    os.chdir(job_dir)

    # Read the default configuration file
    # config = bc.Cfg(default_config_file)

    work_count = 0

    for kk, vv in segments_by_loc.items():
        try:
            # Try for integer formatted output directories first
            if args.prefix:
                cdir = f'{args.prefix}{kk:04d}'
            else:
                cdir = f'{kk:04d}'
        except ValueError:
            cdir = f'{kk}'

        # Create the headwater directory if needed
        if not os.path.exists(cdir):
            try:
                os.makedirs(cdir)
            except OSError as err:
                print(f'\tError creating directory: {err}')
                exit(1)

        # Update the outlets in the basin.cfg file and write into the headwater directory
        config.update_value('outlets', vv)

        if noroute_hrus_by_loc is not None and kk in noroute_hrus_by_loc:
            config.update_value('hru_noroute', noroute_hrus_by_loc[kk])

        # TODO: This causes the control_filename to be rewritten in the parent
        #       directory; so this happens for each location. Need to fix.
        config.update_value('control_filename', f'{job_dir}/control.default')
        config.update_value('output_dir', f'{job_dir}/{cdir}')
        config.write(f'{cdir}/bandit.cfg')

        # Run bandit
        # Add the command to queue for processing
        work_count += 1
        cmd = f'{cmd_bandit} --no_filter_params --keep_hru_order -j {job_dir}/{cdir}'

        os.chdir(cdir)
        cmd_q.put(cmd)
        os.chdir(job_dir)

    print(f'work_count = {work_count:4d}')

    # Output results
    while work_count > 0:
        result = result_q.get()

        sys.stdout.write(f'\rwork_count: {work_count:4d}')
        sys.stdout.flush()

        # print "Thread %s return code = %d" % (result[0], result[2])
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
