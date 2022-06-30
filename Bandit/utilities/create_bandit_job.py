#!/usr/bin/env python3

import argparse
import os
import shutil

import Bandit.bandit_cfg as bc

__author__ = 'Parker Norton (pnorton@usgs.gov)'


def main():
    parser = argparse.ArgumentParser(description='Setup new job for Bandit extraction')
    parser.add_argument('-c', '--config', help='Name of configuration file', nargs='?', default='bandit.cfg', type=str)
    parser.add_argument('jobdir', help='Name of new job directory')

    args = parser.parse_args()

    config = bc.Cfg(args.config)

    print('Creating new job for Bandit extraction')
    print(f'Config file: {args.config}')

    # Check that the various required directories and files defined in bandit.cfg exist
    if os.path.splitext(config.cbh_dir)[1] == '.nc':
        print('INFO: Using netCDF format for CBH files')
        cbh_dir_tmp = os.path.split(config.cbh_dir)[0]
    else:
        cbh_dir_tmp = config.cbh_dir

    if not os.path.exists(cbh_dir_tmp):
        print("Location of the CBH files (cbh_dir) does not exist!")
        exit(2)
    elif not os.path.exists(config.paramdb_dir):
        print("Location of the NHM parameter database (paramdb_dir) does not exist!")
        exit(2)
    elif not os.path.exists(config.geodatabase_filename):
        print("The geodatabase file (geodatabase_filename) does not exist!")
        exit(2)
    elif not os.path.exists(config.output_dir):
        print("The main jobs directory (output_dir) does not exist!")
        exit(2)

    # Define the path to the new job directory
    tl_jobsdir = config.output_dir

    # Create output directory if necessary
    new_job_dir = f'{tl_jobsdir}/{args.jobdir}'

    try:
        os.mkdir(new_job_dir)
        print(f'\tJob directory created: {new_job_dir}')
    except OSError as err:
        if err.errno == 2:
            print(f'\tThe top-level jobs directory does not exist: {tl_jobsdir}')
            exit(2)
        elif err.errno == 13:
            print(f'\tYou have insufficient privileges to create: {new_job_dir}')
            exit(2)
        elif err.errno == 17:
            print(f'\tFile/Directory already exists: {new_job_dir}')
            print('\tNew bandit.cfg and control.default files will be copied here.')
        else:
            print('\tOther error')
            print(err)
            raise

    # Copy bandit.cfg to job directory
    print('\tCreating bandit.cfg file for new job')
    config.update_value('output_dir', new_job_dir)
    config.write(f'{new_job_dir}/bandit.cfg')

    # Copy the control.default file to the job directory
    print('\tCopying control.default to new job')
    shutil.copy(f'{tl_jobsdir}/control.default', f'{new_job_dir}/control.default')

    print('\nNew job directory has been created.')
    print('Make sure to update outlets, cutoffs, and hru_noroute parameters as needed in bandit.cfg ' +
          'before running bandit.')


if __name__ == '__main__':
    main()
