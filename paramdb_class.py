#!/usr/bin/env python2.7

from __future__ import (absolute_import, division, print_function)
# , unicode_literals)
from future.utils import iteritems

import xml.etree.ElementTree as xmlET
import numpy as np
import msgpack
import sys

from pr_util import colorize, heading, print_info, print_warning, print_error


REGIONS = ['r01', 'r02', 'r03', 'r04', 'r05', 'r06', 'r07', 'r08', 'r09',
           'r10L', 'r10U', 'r11', 'r12', 'r13', 'r14', 'r15', 'r16', 'r17', 'r18']


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


def get_dimension_sizes(params, regions, workdir):
    # This builds a dictionary of total dimension sizes for the concatenated parameters
    dimension_info = {}
    is_populated = {}

    # Loop through the xml files for each parameter and define the total size and dimensions
    for pp in params.iterkeys():
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
        for kk, vv in is_populated.iteritems():
            if not vv:
                is_populated[kk] = True
    return dimension_info


def get_params(params_file):
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


class ParamDb(object):
    def __init__(self, paramdb_dir=None, conusdb_dir=None):
        # Needs:

        self.__paramdb_dir = paramdb_dir    # Location of official parameter database
        self.__conusdb_dir = conusdb_dir    # Location of CONUS parameter database

        # Always default the global dimensions and parameter filenames
        # They can be overridden later
        self.__global_dims_file = '{}/dimensions.xml'.format(self.__workdir)
        self.__global_params_file = '{}/parameters.xml'.format(self.__workdir)

        self.__global_parameters = None
        self.__parameters = {}

    @property
    def global_dims_file(self):
        return self.__global_dims_file

    @property
    def global_params_file(self):
        return self.__global_params_file

    def get_parameter(self, param_name):
        # Return the given parameter from the CONUS paramdb
        if param_name not in self.__parameters:
            try:
                with open('{}/{}.msgpack'.format(self.__conusdb_dir, param_name), 'rb') as ff:
                    self.__parameters[param_name] = msgpack.load(ff, use_list=False)
            except:
                # ??possible errors: no such file, ??more??
                print('Error occurred' + sys.exc_info()[0])
        return self.__parameters[param_name]

    def create_conusdb(self):
        # Create the CONUS parameter database from the official parameter database
        self.__global_parameters = get_global_params(self.__global_params_file)

