#!/usr/bin/env python

from __future__ import (absolute_import, division, print_function)
from future.utils import iteritems    # , iterkeys

# import argparse
import msgpack
import os
import sys

import bandit_cfg as bc
from pyPRMS.NhmParamDb import NhmParamDb

# from dimension_class import Parameter
from helpers import git_version
# from pr_util import print_warning, print_error

__author__ = 'Parker Norton (pnorton@usgs.gov)'
__version__ = '0.3'

config = bc.Cfg('bandit.cfg')


def main():
    # TODO: Automatically update the paramdb from git before creating merged params

    paramdb_dir = config.paramdb_dir
    merged_paramdb_dir = config.merged_paramdb_dir

    # check for / create output directory
    try:
        os.makedirs(merged_paramdb_dir)
        print('Creating directory for merged parameters: {}'.format(merged_paramdb_dir))
    except OSError:
        print("\tUsing existing directory for merged parameters: {}".format(merged_paramdb_dir))

    # Write the git revision number of the NhmParamDb repo to a file in the merged params directory
    with open('{}/00-REVISION'.format(merged_paramdb_dir), 'w') as fhdl:
        fhdl.write('{}\n'.format(git_version(paramdb_dir)))

    # Create NhmParamDb object and retrieve the parameters
    pdb = NhmParamDb(paramdb_dir)
    param_info = pdb.available_parameters

    # =======================================================================
    # Process all the parameters, skipping those that need to be handled
    # specially.
    for pp in param_info:
        sys.stdout.write('\r                                       ')
        sys.stdout.write('\rProcessing {}'.format(pp))
        sys.stdout.flush()

        cparam = pdb.parameters.get(pp)

        # write the serialized param to a file
        with open('{}/{}.msgpack'.format(merged_paramdb_dir, pp), 'wb') as ff:
            msgpack.dump(cparam.tostructure(), ff)

    # Write the parameters.xml and dimensions.xml files to the merged_db directory
    pdb.write_parameters_xml(merged_paramdb_dir)
    pdb.write_dimensions_xml(merged_paramdb_dir)

    # =======================================================================
    # Lastly there are a few non-parameter mapping variables that are needed
    # during the checkout process. It's easier/faster to create them once
    # here rather than create them on the fly during checkout.

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Process nhm_seg related mappings
    # pp = 'nhm_seg'
    sys.stdout.write('\r                                       ')
    sys.stdout.write('\rProcessing segment mappings')
    sys.stdout.flush()

    # write the serialized segment mappings to a file
    # with open('{}/segment_nhm_to_local.msgpack'.format(merged_paramdb_dir), 'wb') as ff:
    #     msgpack.dump(segment_nhm_to_local, ff)

    with open('{}/segment_nhm_to_region.msgpack'.format(merged_paramdb_dir), 'wb') as ff:
        msgpack.dump(pdb.segment_nhm_to_region, ff)

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Process nhm_id related mappings
    sys.stdout.write('\r                                       ')
    sys.stdout.write('\rProcessing hru mappings')

    # write the serialized segment mappings to a file
    with open('{}/hru_nhm_to_local.msgpack'.format(merged_paramdb_dir), 'wb') as ff:
        msgpack.dump(pdb.hru_nhm_to_local, ff)

    with open('{}/hru_nhm_to_region.msgpack'.format(merged_paramdb_dir), 'wb') as ff:
        msgpack.dump(pdb.hru_nhm_to_region, ff)


if __name__ == '__main__':
    main()
