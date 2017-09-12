#!/usr/bin/env python2.7

#      Author: Parker Norton (pnorton@usgs.gov)
#        Date: 2017-02-14
# Description: Configuration class for Model Bandit
#              YAML is used for the backend

from __future__ import (absolute_import, division, print_function)
from future.utils import iteritems

import yaml


# Following class from: https://stackoverflow.com/questions/34667108/ignore-dates-and-times-while-parsing-yaml
class NoDatesSafeLoader(yaml.SafeLoader):
    @classmethod
    def remove_implicit_resolver(cls, tag_to_remove):
        """
        Remove implicit resolvers for a particular tag

        Takes care not to modify resolvers in super classes.

        We want to load datetimes as strings, not dates, because we
        go on to serialise as json which doesn't have the advanced types
        of yaml, and leads to incompatibilities down the track.
        """
        if 'yaml_implicit_resolvers' not in cls.__dict__:
            cls.yaml_implicit_resolvers = cls.yaml_implicit_resolvers.copy()

        for first_letter, mappings in cls.yaml_implicit_resolvers.items():
            cls.yaml_implicit_resolvers[first_letter] = [(tag, regexp)
                                                         for tag, regexp in mappings
                                                         if tag != tag_to_remove]


NoDatesSafeLoader.remove_implicit_resolver('tag:yaml.org,2002:timestamp')


class Cfg(object):
    def __init__(self, filename, cmdline=None):
        self.__cfgdict = None
        self.__cmdline = cmdline
        self.load(filename)

    def __str__(self):
        """Pretty-print the configuration items"""
        outstr = ''

        for (kk, vv) in iteritems(self.__cfgdict):
            outstr += '{0:s}: '.format(kk)

            if isinstance(vv, list):
                for ll in vv:
                    outstr += '{}, '.format(ll)
                outstr = outstr.strip(', ') + '\n'
            else:
                outstr += '{}\n'.format(vv)
        return outstr

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

    def load(self, filename):
        tmp = yaml.load(open(filename, 'r'), Loader=NoDatesSafeLoader)
        self.__cfgdict = tmp

    def update_value(self, variable, newval):
        """Update an existing configuration variable with a new value"""
        if variable in self.__cfgdict:
            self.__cfgdict[variable] = newval
        else:
            raise KeyError("Configuration variable, {}, does not exist".format(variable))

    def write(self, filename):
        """"Write the configuration out to a file"""
        outfile = open(filename, 'w')
        yaml.dump(self.__cfgdict, outfile)
