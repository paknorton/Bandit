#!/usr/bin/env python

from __future__ import (absolute_import, division, print_function)

import argparse
import os
import shutil

import Bandit.bandit_cfg as bc

__author__ = 'Parker Norton (pnorton@usgs.gov)'


def main():
    parser = argparse.ArgumentParser(description='Setup new job for Bandit extraction')
    parser.add_argument('jobdir', help='Name of new job directory')

    args = parser.parse_args()

    config = bc.Cfg('bandit.cfg')

    print('Creating new job for Bandit extraction')

    # Check that the various required directories and files defined in bandit.cfg exist
    if not os.path.exists(config.cbh_dir):
        print("Location of the CBH files (cbh_dir) does not exist!")
        exit(2)
    elif not os.path.exists(config.paramdb_dir):
        print("Location of the NHM parameter database (paramdb_dir) does not exist!")
        exit(2)
    elif not os.path.exists(config.merged_paramdb_dir):
        print("Location of the merged parameters database (merged_paramdb_dir) does not exist!")
        exit(2)
    elif not os.path.exists(config.geodatabase_filename):
        print("The geodatabase file (geodatabase_filename) does not exist!")
        exit(2)
    elif not os.path.exists(config.output_dir):
        print("The main jobs directory (output_dir) does not exist!")
        exit(2)

    # Define the path to the new job directory
    tl_jobsdir = config.output_dir

    # check for / create output directory
    new_job_dir = '{}/{}'.format(tl_jobsdir, args.jobdir)

    try:
        os.mkdir(new_job_dir)
        print('\tJob directory created: {}'.format(new_job_dir))
    except OSError as err:
        if err.errno == 2:
            print('\tThe top-level jobs directory does not exist: {}'.format(tl_jobsdir))
            exit(2)
        elif err.errno == 13:
            print('\tYou have insufficient privileges to create: {}'.format(new_job_dir))
            exit(2)
        elif err.errno == 17:
            print('\tFile/Directory already exists: {}'.format(new_job_dir))
            print('\tNew bandit.cfg and control.default files will be copied here.')
        else:
            print('\tOther error')
            print(err)
            raise

    # Copy bandit.cfg to job directory
    print('\tCreating bandit.cfg file for new job')
    # config.update_value('output_dir', new_job_dir)
    config.write('{}/bandit.cfg'.format(new_job_dir))

    # Copy the control.default file to the job directory
    print('\tCopying control.default to new job')
    shutil.copy('{}/control.default'.format(tl_jobsdir), '{}/control.default'.format(new_job_dir))

    print('\nNew job directory has been created.')
    print('Make sure to update outlets, cutoffs, and hru_noroute parameters as needed in bandit.cfg' +
          'before running bandit.')


if __name__ == '__main__':
    main()
