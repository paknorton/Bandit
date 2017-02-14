#!/usr/bin/env python2.7

#      Author: Parker Norton (pnorton@usgs.gov)
#        Date: 2017-02-14
# Description: Configuration class for Model Bandit
#              YAML is used for the backend

from __future__ import (absolute_import, division, print_function)
from future.utils import iteritems

import yaml


class Cfg(object):
    def __init__(self, filename, cmdline=None):
        self.__cfgdict = None
        self.__cmdline = cmdline
        self.load(filename)

    def __getattr__(self, item):
        # Undefined attributes will look up the given configuration item
        return self.get_value(item)

    def get_value(self, varname):
        """Return the value for a given config variable"""
        try:
            return self.__cfgdict[varname]
        except KeyError:
            print('Configuration variable, {}, does not exist'.format(varname))
            return None

    def list_config_items(self):
        for (kk, vv) in iteritems(self.__cfgdict):
            print('{0:s}:'.format(kk)),

            if isinstance(vv, list):
                for ll in vv:
                    print(ll),
                print()
            else:
                print(vv)

    def load(self, filename):
        tmp = yaml.load(open(filename, 'r'))
        self.__cfgdict = tmp

    def update_value(self, variable, newval):
        """Update an existing configuration variable with a new value"""
        if variable in self.__cfgdict:
            self.__cfgdict[variable] = newval
        else:
            raise KeyError("Configuration variable, {}, does not exist".format(variable))
