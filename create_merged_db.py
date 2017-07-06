#!/usr/bin/env python

from __future__ import (absolute_import, division, print_function)
from future.utils import iteritems, iterkeys

import argparse
import xml.etree.ElementTree as xmlET
import msgpack
from collections import OrderedDict
import sys

# from bandit import git_version

import bandit_cfg as bc
import pyPRMS.NhmParamDb as nhmdb

from dimension_class import Parameter
from helpers import git_version
from pr_util import print_warning, print_error

__author__ = 'Parker Norton (pnorton@usgs.gov)'
__version__ = '0.1'

REGIONS = ['r01', 'r02', 'r03', 'r04', 'r05', 'r06', 'r07', 'r08', 'r09',
           'r10L', 'r10U', 'r11', 'r12', 'r13', 'r14', 'r15', 'r16', 'r17', 'r18']

config = bc.Cfg('bandit.cfg')


def main():
    # TODO: Automatically update the paramdb from git before creating merged params

    paramdb_dir = config.paramdb_dir
    merged_paramdb_dir = config.merged_paramdb_dir

    # Write the git revision number of the NhmParamDb repo to a file in the merged params directory
    with open('{}/00-REVISION'.format(merged_paramdb_dir), 'w') as fhdl:
        fhdl.write('{}\n'.format(git_version(paramdb_dir)))

    # global_dims_file = '{}/dimensions.xml'.format(paramdb_dir)
    # global_params_file = '{}/parameters.xml'.format(paramdb_dir)

    param = nhmdb.NhmParamDb(paramdb_dir)
    param_info = param.available_parameters
    # param_info = get_global_params(global_params_file)
    # dimension_info = get_global_dimensions(param_info, REGIONS, workdir)

    skip_params = ['poi_gage_segment', 'hru_deplcrv']

    # =======================================================================
    # Process all the parameters, skipping those that need to be handled
    # specially.
    for pp, pv in iteritems(param_info):
        sys.stdout.write('\r                                       ')
        sys.stdout.write('\rProcessing {}'.format(pp))

        if pp in skip_params:
            continue

        cparam = param.get(pp)

        # write the serialized param to a file
        with open('{}/{}.msgpack'.format(merged_paramdb_dir, pp), 'wb') as ff:
            msgpack.dump(cparam.tostructure(), ff)

    # =======================================================================
    # Now process the parameters we skipped earlier
    # These are parameters where some additional processing is needed to
    # create their CONUS paramdb form.
    for pp in skip_params:
        sys.stdout.write('\r                                       ')
        sys.stdout.write('\rProcessing {}'.format(pp))

        param = Parameter(name=pp, datatype=param_info[pp]['type'])

        crv_offset = 0  # only used by hru_deplcrv

        for rr in REGIONS:
            sys.stdout.write('\rProcessing {}: {} '.format(pp, rr))
            sys.stdout.flush()

            # Read parameter information
            cdir = '{}/{}/{}'.format(paramdb_dir, pp, rr)

            if not param.ndims:
                param.add_dimensions_from_xml('{}/{}.xml'.format(cdir, pp))

            if pp == 'poi_gage_segment':
                # When processing poi_gage_segment we also need nhm_seg to
                # translate the segment entries from region to national IDs.
                seghdl = open('{}/nhm_seg/{}/nhm_seg.csv'.format(paramdb_dir, rr))
                segdata = seghdl.read().splitlines()
                seghdl.close()
                seg_it = iter(segdata)
                next(seg_it)

                nhm_seg = []
                for seg_rec in seg_it:
                    seg = int(seg_rec.split(',')[1])
                    nhm_seg.append(seg)
                nhm_seg.insert(0, 0)

            # Read the parameter data
            fhdl = open('{}/{}.csv'.format(cdir, pp))
            rawdata = fhdl.read().splitlines()
            fhdl.close()
            it = iter(rawdata)
            next(it)    # Skip the header row

            tmp_data = []

            # Read the parameter values
            for rec in it:
                idx, val = rec.split(',')

                if pp == 'poi_gage_segment':
                    if int(val) == 0:
                        print_warning('{}: {} for index {} is zero'.format(rr, pp, idx))
                    tmp_data.append(nhm_seg[int(val)])
                elif pp == 'hru_deplcrv':
                    tmp_data.append(int(val)+crv_offset)
                else:
                    tmp_data.append(val)

            if pp == 'hru_deplcrv':
                crv_offset += len(tmp_data)

            param.append_paramdb_data(tmp_data)

        # write the serialized param to a file
        with open('{}/{}.msgpack'.format(merged_paramdb_dir, pp), 'wb') as ff:
            msgpack.dump(param.tostructure(), ff)

    # =======================================================================
    # Lastly there are a few non-parameter mapping variables that are needed
    # during the checkout process. It's easier/faster to create them once
    # here rather than create them on the fly during checkout.

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Process nhm_seg related mappings
    pp = 'nhm_seg'
    sys.stdout.write('\r                                       ')
    sys.stdout.write('\rProcessing {} mappings')

    segment_nhm_to_local = OrderedDict()
    segment_nhm_to_region = OrderedDict()

    for rr in REGIONS:
        sys.stdout.write('\rProcessing {} mappings: {} '.format(pp, rr))
        sys.stdout.flush()

        # Read parameter information
        cdir = '{}/{}/{}'.format(paramdb_dir, pp, rr)

        # Read the data
        fhdl = open('{}/{}.csv'.format(cdir, pp))
        rawdata = fhdl.read().splitlines()
        fhdl.close()
        it = iter(rawdata)
        next(it)  # Skip the header row

        tmp_data = []

        # Read the parameter values
        for rec in it:
            idx, val = rec.split(',')

            if int(val) in segment_nhm_to_local:
                print_warning('{} has duplicate entry in segment_nhm_to_local'.format(pp))
            segment_nhm_to_local[int(val)] = int(idx)

            tmp_data.append(int(val))
        segment_nhm_to_region[rr] = [min(tmp_data), max(tmp_data)]

    # write the serialized segment mappings to a file
    with open('{}/segment_nhm_to_local.msgpack'.format(merged_paramdb_dir), 'wb') as ff:
        msgpack.dump(segment_nhm_to_local, ff)

    with open('{}/segment_nhm_to_region.msgpack'.format(merged_paramdb_dir), 'wb') as ff:
        msgpack.dump(segment_nhm_to_region, ff)

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Process nhm_id related mappings
    pp = 'nhm_id'
    sys.stdout.write('\r                                       ')
    sys.stdout.write('\rProcessing {} mappings')

    # NOTE: There really is no point to using Ordered dictionaries
    #       because msgpack does not honor the ordered dicionary type.
    #       These will end up being read into a standard dictionary
    #       and the desired order will be gone.
    hru_nhm_to_local = OrderedDict()
    hru_nhm_to_region = OrderedDict()

    for rr in REGIONS:
        sys.stdout.write('\rProcessing {} mappings: {} '.format(pp, rr))
        sys.stdout.flush()

        # Read parameter information
        cdir = '{}/{}/{}'.format(paramdb_dir, pp, rr)

        # Read the data
        fhdl = open('{}/{}.csv'.format(cdir, pp))
        rawdata = fhdl.read().splitlines()
        fhdl.close()
        it = iter(rawdata)
        next(it)  # Skip the header row

        tmp_data = []

        # Read the parameter values
        for rec in it:
            idx, val = rec.split(',')

            if int(val) in hru_nhm_to_local:
                print_warning('{} has duplicate entry in hru_nhm_to_local'.format(pp))
            hru_nhm_to_local[int(val)] = int(idx)

            tmp_data.append(int(val))
        hru_nhm_to_region[rr] = [min(tmp_data), max(tmp_data)]

    # write the serialized segment mappings to a file
    with open('{}/hru_nhm_to_local.msgpack'.format(merged_paramdb_dir), 'wb') as ff:
        msgpack.dump(hru_nhm_to_local, ff)

    with open('{}/hru_nhm_to_region.msgpack'.format(merged_paramdb_dir), 'wb') as ff:
        msgpack.dump(hru_nhm_to_region, ff)


if __name__ == '__main__':
    main()
