#!/usr/bin/env python2.7

from __future__ import (absolute_import, division,
                        print_function)
# , unicode_literals)
from future.utils import iteritems

import xml.etree.ElementTree as xmlET
import numpy as np
import cPickle as pickle
import sys

from pr_util import colorize, heading, print_info, print_warning, print_error


REGIONS = ['r01', 'r02', 'r03', 'r04', 'r05', 'r06', 'r07', 'r08', 'r09',
           'r10L', 'r10U', 'r11', 'r12', 'r13', 'r14', 'r15', 'r16', 'r17', 'r18']


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
    def __init__(self, workdir=None, outdir=None):
        self.__workdir = workdir
        self.__outdir = outdir

        self.__dims_file = '{}/dimensions.xml'.format(self.__workdir)
        self.__params_file = '{}/parameters.xml'.format(self.__workdir)

        param_info = get_params(self.__params_file)