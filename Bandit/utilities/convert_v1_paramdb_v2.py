#!/usr/bin/env python3

import argparse
import os

import Bandit.bandit_cfg as bc
from Bandit.git_version import git_commit_url
from pyPRMS.ParamDbRegion import ParamDbRegion

__author__ = 'Parker Norton (pnorton@usgs.gov)'


def main():
    parser = argparse.ArgumentParser(description='Utility to convert paramDb by to region to monolithic paramDb')
    parser.add_argument('-c', '--config', help='Name of configuration file', nargs='?', default='bandit.cfg', type=str)

    args = parser.parse_args()

    config = bc.Cfg(args.config)

    paramdb_dir = config.paramdb_dir
    merged_paramdb_dir = config.merged_paramdb_dir

    # check for output directory
    try:
        os.makedirs(merged_paramdb_dir)
        print(f'Creating directory for merged parameters: {merged_paramdb_dir}')
    except OSError:
        print(f'\tUsing existing directory for merged parameters: {merged_paramdb_dir}')

    # Write the git revision number of the NhmParamDb repo to a file in the merged params directory
    with open(f'{merged_paramdb_dir}/00-REVISION', 'w') as fhdl:
        fhdl.write(f'{git_commit_url(paramdb_dir)}\n')

    # Create NhmParamDb object and retrieve the parameters
    pdb = ParamDbRegion(paramdb_dir, verbose=True, verify=True)

    # Overwrite the data for tosegment and hru_segment with their respective
    # NHM counterparts.
    pdb.parameters['tosegment'].data = pdb.parameters['tosegment_nhm'].data
    pdb.parameters['hru_segment'].data = pdb.parameters['hru_segment_nhm'].data

    # =======================================================================
    # Process all the parameters, skipping special-handling cases
    pdb.write_paramdb(merged_paramdb_dir)

    # Write the parameters.xml and dimensions.xml files to the merged_db directory
    pdb.write_parameters_xml(merged_paramdb_dir)
    pdb.write_dimensions_xml(merged_paramdb_dir)


if __name__ == '__main__':
    main()
