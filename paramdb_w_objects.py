#!/usr/bin/env python

from __future__ import (absolute_import, division,
                        print_function)
# , unicode_literals)
from future.utils import iteritems, iterkeys

import xml.etree.ElementTree as xmlET
import numpy as np
# import pickle as pickle
import msgpack
from collections import OrderedDict
import sys

from dimension_class import Parameter, Dimensions, Dimension
from pr_util import colorize, heading, print_info, print_warning, print_error

REGIONS = ['r01', 'r02', 'r03', 'r04', 'r05', 'r06', 'r07', 'r08', 'r09',
           'r10L', 'r10U', 'r11', 'r12', 'r13', 'r14', 'r15', 'r16', 'r17', 'r18']

workdir = '/Users/pnorton/Projects/National_Hydrology_Model/paramDb/nhmparamdb'
outdir = '/Users/pnorton/Projects/National_Hydrology_Model/paramDb/merged_params2'

global_dims_file = '{}/dimensions.xml'.format(workdir)
global_params_file = '{}/parameters.xml'.format(workdir)


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


def main():
    param_info = get_global_params(global_params_file)
    # dimension_info = get_global_dimensions(param_info, REGIONS, workdir)

    for pp, pv in iteritems(param_info):
        sys.stdout.write('\r                                       ')
        sys.stdout.write('\rProcessing {}'.format(pp))

        if pp in ['poi_gage_segment']:
            # Just skip for now
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


if __name__ == '__main__':
    main()
