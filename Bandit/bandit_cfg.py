#!/usr/bin/env python2.7

#      Author: Parker Norton (pnorton@usgs.gov)
#        Date: 2017-02-14
# Description: Configuration class for Model Bandit
#              YAML is used for the backend

from __future__ import (absolute_import, division, print_function)
from future.utils import iteritems

from ruamel.yaml import YAML
# import yaml
# import yaml.constructor


# class OrderedDictYAMLLoader(yaml.Loader):
#     """
#     A YAML loader that loads mappings into ordered dictionaries.
#     """
#
#     # FROM: https://stackoverflow.com/questions/5121931/in-python-how-can-you-load-yaml-mappings-as-ordereddicts
#     def __init__(self, *args, **kwargs):
#         yaml.Loader.__init__(self, *args, **kwargs)
#
#         self.add_constructor(u'tag:yaml.org,2002:map', type(self).construct_yaml_map)
#         self.add_constructor(u'tag:yaml.org,2002:omap', type(self).construct_yaml_map)
#
#     def construct_yaml_map(self, node):
#         data = OrderedDict()
#         yield data
#         value = self.construct_mapping(node)
#         data.update(value)
#
#     def construct_mapping(self, node, deep=False):
#         if isinstance(node, yaml.MappingNode):
#             self.flatten_mapping(node)
#         else:
#             raise yaml.constructor.ConstructorError(None, None,
#                                                     'expected a mapping node, but found %s' % node.id, node.start_mark)
#
#         mapping = OrderedDict()
#         for key_node, value_node in node.value:
#             key = self.construct_object(key_node, deep=deep)
#             try:
#                 hash(key)
#             except TypeError as exc:
#                 raise yaml.constructor.ConstructorError('while constructing a mapping',
#                                                         node.start_mark, 'found unacceptable key (%s)' % exc,
#                                                         key_node.start_mark)
#             value = self.construct_object(value_node, deep=deep)
#             mapping[key] = value
#         return mapping


# Following class from: https://stackoverflow.com/questions/34667108/ignore-dates-and-times-while-parsing-yaml
# class NoDatesSafeLoader(yaml.SafeLoader):
#     @classmethod
#     def remove_implicit_resolver(cls, tag_to_remove):
#         """
#         Remove implicit resolvers for a particular tag
#
#         Takes care not to modify resolvers in super classes.
#
#         We want to load datetimes as strings, not dates, because we
#         go on to serialise as json which doesn't have the advanced types
#         of yaml, and leads to incompatibilities down the track.
#         """
#         if 'yaml_implicit_resolvers' not in cls.__dict__:
#             cls.yaml_implicit_resolvers = cls.yaml_implicit_resolvers.copy()
#
#         for first_letter, mappings in cls.yaml_implicit_resolvers.items():
#             cls.yaml_implicit_resolvers[first_letter] = [(tag, regexp)
#                                                          for tag, regexp in mappings
#                                                          if tag != tag_to_remove]
#
#
# NoDatesSafeLoader.remove_implicit_resolver('tag:yaml.org,2002:timestamp')


class Cfg(object):
    """Configuration class for the Bandit NHM extraction program."""

    def __init__(self, filename, cmdline=None):
        """Init method for Cfg class.

        Args:
            filename (str): The configuration filename.
            cmdline (str, optional): Currently unused. Defaults to None.

        """

        # yaml.add_representer(OrderedDict, dict_representer)
        # yaml.add_constructor(_mapping_tag, dict_constructor)
        self.yaml = YAML()

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
        """Return the value for a given config variable.

        Args:
            varname (str): Configuration variable.

        Returns:
             The value of the configuration variable or None if variable does not exist.

        """

        try:
            return self.__cfgdict[varname]
        except KeyError:
            print('Configuration variable, {}, does not exist'.format(varname))
            return None

    def load(self, filename):
        """Load the YAML-format configuration file.

        Args:
            filename (str): Name of the configuration file.

        """
        # tmp = yaml.load(open(filename, 'r'), Loader=NoDatesSafeLoader)
        tmp = self.yaml.load(open(filename, 'r'))
        self.__cfgdict = tmp

    def update_value(self, variable, newval):
        """Update an existing configuration variable with a new value

        Args:
            variable (str): The configuration variable to update.
            newval (str): The value to assign to the variable.

        Raises:
            KeyError: If configuration variable does not exist.

        """

        if variable in self.__cfgdict:
            self.__cfgdict[variable] = newval
        else:
            raise KeyError("Configuration variable, {}, does not exist".format(variable))

    def write(self, filename):
        """"Write the configuration out to a file.

        Args:
            filename (str): Name of file to write configuration to.

        """

        outfile = open(filename, 'w')
        self.yaml.dump(self.__cfgdict, outfile)
