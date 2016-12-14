#!/usr/bin/env python

from __future__ import (absolute_import, division,
                        print_function)
# , unicode_literals)
from future.utils import iteritems, iterkeys

import xml.etree.ElementTree as xmlET
import numpy as np
import pickle as pickle
from collections import OrderedDict
import sys

from pr_util import colorize, heading, print_info, print_warning, print_error

REGIONS = ['r01', 'r02', 'r03', 'r04', 'r05', 'r06', 'r07', 'r08', 'r09',
           'r10L', 'r10U', 'r11', 'r12', 'r13', 'r14', 'r15', 'r16', 'r17', 'r18']

workdir = '/Users/pnorton/Projects/National_Hydrology_Model/paramDb/nhmparamdb'
outdir = '/Users/pnorton/Projects/National_Hydrology_Model/paramDb/merged_params'

dims_file = '{}/dimensions.xml'.format(workdir)
params_file = '{}/parameters.xml'.format(workdir)


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


def get_dim_size(dimensions, position=1):
    for dd in dimensions:
        if dimensions[dd]['position'] == position:
            return dimensions[dd]['size']
    # If we get to here raise an exception


def get_ndims(dimensions):
    return len(dimensions)


def get_param_info(xml_file, param):
    # Opens the xml file for a param and populated the dimensions key with
    # the dimensions names, sizes, and positions
    cdim_tree = xmlET.parse(xml_file)
    cdim_root = cdim_tree.getroot()

    for cdim in cdim_root.findall('./dimensions/dimension'):
        dim_name = cdim.get('name')
        dim_size = int(cdim.get('size'))

        param['dimensions'].setdefault(dim_name, {})
        param['dimensions'][dim_name]['position'] = int(cdim.get('position'))

        if dim_name in ['nmonths', 'ndays', 'one']:
            # Non-additive dimensions
            param['dimensions'][dim_name]['size'] = param['dimensions'][dim_name].get('size', dim_size)
        else:
            # Other dimensions are additive
            param['dimensions'][dim_name]['size'] = param['dimensions'][dim_name].get('size', 0) + dim_size


# For a parameter we should store: parameter name, parameter type, dimension name(s) and size(s), and the data
# param_data{'name': 'blah',
#            'type': 'integer',
#            'dimensions': {<name>: {<position>: val, <size>: val},
#            'data': []}
# params = ['tmax_allsnow']

# ~~~~~~~~~~~~~~~~~~~~~~~
# hru_deplcrv
# There is no guarantee that hru_deplcrv values are unique and monontonic in any
# given region. So how do you read the index values and transform them to a NHM
# index?
# ~~~~~~~~~~~~~~~~~~~~~~~~

def main():
    param_info = get_global_params(params_file)
    # dimension_info = get_global_dimensions(param_info, REGIONS, workdir)

    for pp, pv in iteritems(param_info):
        sys.stdout.write('\r                                       ')
        sys.stdout.write('\rProcessing {}'.format(pp))
        # sys.stdout.flush()

        dim_total = 1
        param = {'name': pp, 'type': pv['type'], 'dimensions': {}, 'data': []}
        tmp_arr = None

        if pp == 'nhm_seg':
            segment_nhm_to_local = OrderedDict()
            segment_nhm_to_region = OrderedDict()
        elif pp == 'nhm_id':
            hru_nhm_to_local = OrderedDict()
            hru_nhm_to_region = OrderedDict()

        if pp == 'hru_deplcrv':
            crv_offset = 0

        for rr in REGIONS:
            sys.stdout.write('\rProcessing {}: {} '.format(pp, rr))
            sys.stdout.flush()

            tmp_list = []   # Used only for nhm_seg and nhm_id

            # Read parameter information
            cdir = '{}/{}/{}'.format(workdir, pp, rr)
            get_param_info('{}/{}.xml'.format(cdir, pp), param)

            # When processing poi_gage_segment we also need to have access to nhm_seg to translate
            # the segment entries from region to national IDs.
            if pp == 'poi_gage_segment':
                seghdl = open('{}/nhm_seg/{}/nhm_seg.csv'.format(workdir, rr))
                crap = seghdl.read().splitlines()
                seghdl.close()
                seg_it = iter(crap)
                next(seg_it)

                nhm_seg = {}
                seg_idx = 1
                for seg_rec in seg_it:
                    seg = int(seg_rec.split(',')[1])
                    nhm_seg[seg_idx] = seg
                    seg_idx += 1

            fhdl = open('{}/{}.csv'.format(cdir, pp))

            rawdata = fhdl.read().splitlines()
            fhdl.close()
            it = iter(rawdata)
            next(it)    # Skip the header row

            tmp_data = []
            ndims = get_ndims(param['dimensions'])

            # Read the parameter values
            for rec in it:
                if not ('one' in param['dimensions'] and tmp_arr):
                    # Append the data from current region
                    # The above conditional prevents 'one' dimensioned parameters
                    # from appending values beyond the first region. This assumes
                    # the values are all the same across the regions.
                    idx, val = rec.split(',')

                    if pp == 'poi_gage_segment':
                        if int(val) == 0:
                            print_error('{}: poi_gage_segment for {} is zero'.format(rr, idx))
                        else:
                            val = nhm_seg[int(val)]
                    elif pp == 'nhm_seg':
                        if int(val) in segment_nhm_to_local:
                            print_error('{} has a duplicate entry'.format(pp))
                        segment_nhm_to_local[int(val)] = int(idx)
                        tmp_list.append(int(val))
                    elif pp == 'nhm_id':
                        if int(val) in hru_nhm_to_local:
                            print_error('{} {} has a duplicate entry'.format(pp, int(val)))
                        hru_nhm_to_local[int(val)] = int(idx)
                        tmp_list.append(int(val))

                    if pv['type'] == 'F':
                        tmp_data.append(float(val))
                    elif pv['type'] == 'I':
                        if pp == 'hru_deplcrv':
                            tmp_data.append(int(val)+crv_offset)
                        else:
                            tmp_data.append(int(val))
                    elif pv['type'] == 'S':
                        tmp_data.append(val)
                    else:
                        print('ERROR: no datatype for {}'.format(pp))

            # The following provides the range of nhm_seg and nhm_id values for each region
            if pp == 'nhm_seg':
                segment_nhm_to_region[rr] = [min(tmp_list), max(tmp_list)]
            elif pp == 'nhm_id':
                hru_nhm_to_region[rr] = [min(tmp_list), max(tmp_list)]
            elif pp == 'hru_deplcrv':
                crv_offset += len(tmp_data)

            if ndims == 2:
                # 2D array
                tmpD = np.array(tmp_data).reshape((-1, get_dim_size(param['dimensions'], 2)), order='F')
            elif ndims == 1:
                tmpD = np.array(tmp_data)
            else:
                print_error('Parameter {} in {} did not create tmpD correctly'.format(pp, rr))
                exit(1)

            if rr == 'r01':
                tmp_arr = tmpD
            else:
                if 'one' not in param['dimensions']:
                    tmp_arr = np.concatenate((tmp_arr, tmpD))

        param['data'] = tmp_arr.ravel(order='F').tolist()

        for ss in param['dimensions']:
            dim_total *= param['dimensions'][ss]['size']

        if len(param['data']) != dim_total:
            print('WARNING: Declared size of {} ({}) is different from number of values read ({}).'.format(pp, dim_total,
                                                                                            len(param['data'])))

        # Write out additional mapping variables for nhm_seg and nhm_id
        if pp == 'nhm_seg':
            with open('{}/{}.pickle'.format(outdir, 'segment_nhm_to_local'), 'wb') as ff:
                pickle.dump(segment_nhm_to_local, ff)
            with open('{}/{}.pickle'.format(outdir, 'segment_nhm_to_region'), 'wb') as ff:
                pickle.dump(segment_nhm_to_region, ff)
        elif pp == 'nhm_id':
            with open('{}/{}.pickle'.format(outdir, 'hru_nhm_to_local'), 'wb') as ff:
                pickle.dump(hru_nhm_to_local, ff)
            with open('{}/{}.pickle'.format(outdir, 'hru_nhm_to_region'), 'wb') as ff:
                pickle.dump(hru_nhm_to_region, ff)

        # write the serialized param to a file
        with open('{}/{}.pickle'.format(outdir, pp), 'wb') as ff:
            pickle.dump(param, ff)


if __name__ == '__main__':
    main()
