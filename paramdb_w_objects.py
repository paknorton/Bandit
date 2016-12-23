#!/usr/bin/env python

from __future__ import (absolute_import, division,
                        print_function)
# , unicode_literals)
from future.utils import iteritems, iterkeys

import xml.etree.ElementTree as xmlET
import msgpack
from collections import OrderedDict
import sys

from dimension_class import Parameter
from pr_util import print_warning, print_error

REGIONS = ['r01', 'r02', 'r03', 'r04', 'r05', 'r06', 'r07', 'r08', 'r09',
           'r10L', 'r10U', 'r11', 'r12', 'r13', 'r14', 'r15', 'r16', 'r17', 'r18']


def get_global_dimensions(params, regions, workdir):
    # This builds a dictionary of total dimension sizes for the concatenated parameters
    dimension_info = {}
    is_populated = {}

    # Loop through the xml files for each parameter and define the total size and dimensions
    for pp in iterkeys(params):
        for rr in regions:
            cdim_tree = xmlET.parse('{}/{}/{}/{}.xml'.format(workdir, pp, rr, pp))
            cdim_root = cdim_tree.getroot()

            for cdim in cdim_root.findall('./dimensions/dimension'):
                dim_name = cdim.get('name')
                dim_size = int(cdim.get('size'))

                is_populated.setdefault(dim_name, False)

                if not is_populated[dim_name]:
                    if dim_name in ['nmonths', 'ndays', 'one']:
                        # Non-additive dimensions
                        dimension_info[dim_name] = dimension_info.get(dim_name, dim_size)
                    else:
                        # Other dimensions are additive
                        dimension_info[dim_name] = dimension_info.get(dim_name, 0) + dim_size

        # Update is_populated to reflect dimension size(s) don't need to be re-computed
        for kk, vv in iteritems(is_populated):
            if not vv:
                is_populated[kk] = True
    return dimension_info


def get_global_params(params_file):
    # Get the parameters available from the parameter database
    # Returns a dictionary of parameters and associate units and types

    # Read in the parameters.xml file
    params_tree = xmlET.parse(params_file)
    params_root = params_tree.getroot()

    params = {}

    for param in params_root.findall('parameter'):
        params[param.get('name')] = {}
        params[param.get('name')]['type'] = param.get('type')
        params[param.get('name')]['units'] = param.get('units')

    return params


def main():
    workdir = '/Users/pnorton/Projects/National_Hydrology_Model/paramDb/nhmparamdb'
    outdir = '/Users/pnorton/Projects/National_Hydrology_Model/paramDb/merged_params2'

    global_dims_file = '{}/dimensions.xml'.format(workdir)
    global_params_file = '{}/parameters.xml'.format(workdir)

    param_info = get_global_params(global_params_file)
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

        param = Parameter(name=pp, datatype=pv['type'])

        for rr in REGIONS:
            sys.stdout.write('\rProcessing {}: {} '.format(pp, rr))
            sys.stdout.flush()

            # Read parameter information
            cdir = '{}/{}/{}'.format(workdir, pp, rr)

            if not param.ndims:
                param.add_dimensions_from_xml('{}/{}.xml'.format(cdir, pp))

            # Read the data
            fhdl = open('{}/{}.csv'.format(cdir, pp))
            rawdata = fhdl.read().splitlines()
            fhdl.close()
            it = iter(rawdata)
            next(it)    # Skip the header row

            tmp_data = []

            # Read the parameter values
            for rec in it:
                idx, val = rec.split(',')
                tmp_data.append(val)

            param.append_paramdb_data(tmp_data)

        # write the serialized param to a file
        with open('{}/{}.msgpack'.format(outdir, pp), 'wb') as ff:
            msgpack.dump(param.tostructure(), ff)

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
            cdir = '{}/{}/{}'.format(workdir, pp, rr)

            if not param.ndims:
                param.add_dimensions_from_xml('{}/{}.xml'.format(cdir, pp))

            if pp == 'poi_gage_segment':
                # When processing poi_gage_segment we also need nhm_seg to
                # translate the segment entries from region to national IDs.
                seghdl = open('{}/nhm_seg/{}/nhm_seg.csv'.format(workdir, rr))
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
        with open('{}/{}.msgpack'.format(outdir, pp), 'wb') as ff:
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
        cdir = '{}/{}/{}'.format(workdir, pp, rr)

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

            tmp_data.append(val)
        segment_nhm_to_region[rr] = [min(tmp_data), max(tmp_data)]

    # write the serialized segment mappings to a file
    with open('{}/segment_nhm_to_local.msgpack'.format(outdir), 'wb') as ff:
        msgpack.dump(segment_nhm_to_local, ff)

    with open('{}/segment_nhm_to_region.msgpack'.format(outdir), 'wb') as ff:
        msgpack.dump(segment_nhm_to_region, ff)

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Process nhm_id related mappings
    pp = 'nhm_id'
    sys.stdout.write('\r                                       ')
    sys.stdout.write('\rProcessing {} mappings')

    hru_nhm_to_local = OrderedDict()
    hru_nhm_to_region = OrderedDict()

    for rr in REGIONS:
        sys.stdout.write('\rProcessing {} mappings: {} '.format(pp, rr))
        sys.stdout.flush()

        # Read parameter information
        cdir = '{}/{}/{}'.format(workdir, pp, rr)

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

            tmp_data.append(val)
        hru_nhm_to_region[rr] = [min(tmp_data), max(tmp_data)]

    # write the serialized segment mappings to a file
    with open('{}/hru_nhm_to_local.msgpack'.format(outdir), 'wb') as ff:
        msgpack.dump(hru_nhm_to_local, ff)

    with open('{}/hru_nhm_to_region.msgpack'.format(outdir), 'wb') as ff:
        msgpack.dump(hru_nhm_to_region, ff)


if __name__ == '__main__':
    main()
