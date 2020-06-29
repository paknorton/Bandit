#!/usr/bin/env python3

#      Author: Parker Norton (pnorton@usgs.gov)
#        Date: 2017-02-14
# Description: Configuration class for Model Bandit
#              YAML is used for the backend

from ruamel.yaml import YAML


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

        for (kk, vv) in self.__cfgdict.items():
            outstr += f'{kk}: '

            if isinstance(vv, list):
                for ll in vv:
                    outstr += f'{ll}, '
                outstr = outstr.strip(', ') + '\n'
            else:
                outstr += f'{vv}\n'
        return outstr

    def __getattr__(self, item):
        # Undefined attributes will look up the given configuration item
        return self.get_value(item)

    def exists(self, name):
        return name in self.__cfgdict

    def is_empty(self, name):
        cval = self.get_value(name)

        if isinstance(cval, list):
            return len(cval) == 0
        if isinstance(cval, str):
            return len(cval) == 0
        if isinstance(cval, bool):
            return False
        return True

    def get_value(self, varname):
        """Return the value for a given config variable.

        Args:
            varname (str): Configuration variable.

        Returns:
             The value of the configuration variable or raise KeyError if variable does not exist.

        """

        try:
            return self.__cfgdict[varname]
        except KeyError:
            raise KeyError(f'Configuration variable, {varname}, does not exist') from None
            # return None

    def load(self, filename):
        """Load the YAML-format configuration file.

        Args:
            filename (str): Name of the configuration file.

        """

        tmp = self.yaml.load(open(filename, 'r'))
        self.__cfgdict = tmp

    def update_value(self, variable, newval):
        """Update an existing configuration variable with a new value

        Args:
            variable (str): The configuration variable to update.
            newval (str, list): The value to assign to the variable.

        Raises:
            KeyError: If configuration variable does not exist.

        """

        if variable in self.__cfgdict:
            self.__cfgdict[variable] = newval
        else:
            raise KeyError(f'Configuration variable, {variable}, does not exist')

    def write(self, filename):
        """"Write the configuration out to a file.

        Args:
            filename (str): Name of file to write configuration to.

        """

        outfile = open(filename, 'w')
        self.yaml.dump(self.__cfgdict, outfile)
