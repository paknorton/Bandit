#!/usr/bin/env python
# coding: utf-8
from __future__ import (absolute_import, division, print_function)
from future.utils import iteritems

from Bandit import bandit_cfg as bc

import os
import sys
from collections import OrderedDict
# import time
import threading
from queue import Queue

if os.name == 'posix' and sys.version_info[0] < 3:
    import subprocess32 as subprocess
else:
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
            except Queue.Empty:
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
    config = bc.Cfg('{}/bandit.cfg'.format(job_dir))

    if args.segoutlets:
        seg_src = '{}/{}'.format(job_dir, args.segoutlets)
    else:
        print('ERROR: Must specify the segment outlets file.')
        exit(1)

    if args.nrhrus:
        nrhru_src = '{}/{}'.format(job_dir, args.nrhrus)
    else:
        nrhru_src = None

    # jobdir = '/media/scratch/PRMS/bandit/jobs/hw_jobs'
    # default_config_file = '{}/bandit.cfg'.format(jobdir)

    # cmd_bandit = '/media/scratch/PRMS/bandit/Bandit/bandit.py'
    cmd_bandit = find_executable('bandit')

    if not cmd_bandit:
        print('ERROR: Unable to find bandit.py')
        exit(1)

    seg_file = open(seg_src, 'r')

    # Skip the header information
    # NOTE: If file has no header the first entry will be skipped
    seg_file.readline()
    # seg_file.next()

    # First column is hwAreaId
    # Second and following columns are seg_id_nat
    segments_by_loc = OrderedDict()

    # Read the segment outlets by location
    for line in seg_file:
        cols = line.strip().replace(" ", "").split(',')
        try:
            # Assume first column is a number
            cols = [int(xx) for xx in cols]
            segments_by_loc[cols[0]] = cols[1:]
        except ValueError:
            # First column is probably a string
            segments_by_loc[cols[0]] = [int(xx) for xx in cols[1:]]

    if nrhru_src:
        nrhru_file = open(nrhru_src, 'r')
        nrhru_file.next()

        noroute_hrus_by_loc = OrderedDict()

        # Read in the non-routed HRUs by location
        for line in nrhru_file:
            cols = line.strip().replace(" ", "").split(',')
            try:
                # Assume first column is a number
                cols = [int(xx) for xx in cols]
                noroute_hrus_by_loc[cols[0]] = cols[1:]
            except ValueError:
                # First column is probably a string
                noroute_hrus_by_loc[cols[0]] = [int(xx) for xx in cols[1:]]

    num_threads = 8

    # ****************************************************************************
    # Initialize the threads
    cmd_q = Queue.Queue()
    result_q = Queue.Queue()

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
    #

    if not os.path.exists(job_dir):
        try:
            os.makedirs(job_dir)
        except OSError as err:
            print("\tError creating directory: {}".format(err))
            exit(1)

    # st_dir = os.getcwd()
    os.chdir(job_dir)

    # Read the default configuration file
    # config = bc.Cfg(default_config_file)

    work_count = 0

    for kk, vv in iteritems(segments_by_loc):
        try:
            # Try for integer formatted output directories first
            if args.prefix:
                cdir = '{}{:04d}'.format(args.prefix, kk)
            else:
                cdir = '{:04d}'.format(kk)
        except ValueError:
            cdir = '{}'.format(kk)

        # Create the headwater directory if needed
        if not os.path.exists(cdir):
            try:
                os.makedirs(cdir)
            except OSError as err:
                print("\tError creating directory: {}".format(err))
                exit(1)

        # Update the outlets in the basin.cfg file and write into the headwater directory
        config.update_value('outlets', vv)

        if nrhru_src and kk in noroute_hrus_by_loc:
            config.update_value('hru_noroute', noroute_hrus_by_loc[kk])

        config.update_value('output_dir', '{}/{}'.format(job_dir, cdir))
        config.write('{}/bandit.cfg'.format(cdir))

        # Run bandit
        # Add the command to queue for processing
        work_count += 1
        cmd = '{} -j {}/{}'.format(cmd_bandit, job_dir, cdir)

        os.chdir(cdir)
        cmd_q.put(cmd)
        os.chdir(job_dir)

    print("work_count = {:d}".format(work_count))

    # Output results
    while work_count > 0:
        result = result_q.get()

        sys.stdout.write("\rwork_count: {:4d}".format(work_count))
        sys.stdout.flush()

    #     print "Thread %s return code = %d" % (result[0], result[2])
        work_count -= 1

        if result[2] != 0 and result[2] < 200:
            # An error occurred running the command
            # Returncodes greater than 200 are generated by bandit errors, that
            # don't necessitate shutting the entire job down.
            print("\nThread %s return code = %d (%s)" % (result[0], result[2], result[1]))
            work_count = 0

    # Ask for the threads to die and wait for them to do it
    for thread in pool:
        thread.join()


if __name__ == '__main__':
    main()
