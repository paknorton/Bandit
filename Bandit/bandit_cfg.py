#!/usr/bin/env python3

#      Author: Parker Norton (pnorton@usgs.gov)
#        Date: 2017-02-14
# Description: Configuration class for Model Bandit
#              YAML is used for the backend

from ruamel.yaml import YAML
from typing import Dict, List, Optional, Union

default_values = dict(start_date='1980-01-01',
                      end_date='2010-12-31',
                      check_DAG='False',
                      poi_dir='',
                      output_dir='',
                      control_filename='control.default',
                      param_filename='myparam.param',
                      paramdb_dir='',
                      dyn_params_dir='',
                      outlets=[],
                      cutoffs=[],
                      hru_noroute=[],
                      include_model_output=False,
                      output_vars=[],
                      output_vars_dir='',
                      output_cbh=False,
                      cbh_dir='',
                      cbh_var_map={},
                      output_streamflow=False,
                      streamflow_filename='sf_data',
                      output_shapefiles=False,
                      geodatabase_filename='',
                      hru_gis_layer=None,
                      hru_gis_id=None,
                      seg_gis_layer=None,
                      seg_gis_id=None)


class Cfg(object):
    """Configuration class for the Bandit NHM extraction program."""

    def __init__(self, filename: str,
                 cmdline: Optional[str]=None):
        """Init method for Cfg class.

        :param filename: Configuration filename
        :param cmdline: Currently unused.
        """

        # yaml.add_representer(OrderedDict, dict_representer)
        # yaml.add_constructor(_mapping_tag, dict_constructor)
        self.yaml = YAML()

        self.__cfgdict = None
        self.__cmdline = cmdline
        self.load(filename)

    def __str__(self) -> str:
        """Pretty-print the configuration items

        :returns: String of configuration parameters and values
        """
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

    def __getattr__(self, item: str) -> Union[int, float, str, List]:
        """Get value for a configuration item.

        :returns: Configuration parameter value
        """
        # Undefined attributes will look up the given configuration item
        return self.get_value(item)

    def exists(self, name: str) -> bool:
        """Tests if configuration parameter exists

        :param name: Name of configuration parameter to check
        :returns: True if parameter exists
        """
        return name in self.__cfgdict or name in default_values

    def is_empty(self, name: str) -> bool:
        """Check if list or string parameter values are empty

        :param name: Name of configuration parameter to check
        :returns: True is values is an empty list or string
        """
        cval = self.get_value(name)

        if isinstance(cval, list):
            return len(cval) == 0
        if isinstance(cval, str):
            return len(cval) == 0
        if isinstance(cval, bool):
            return False
        return True

    def get_value(self, name: str) -> Union[str, int, float, List, Dict]:
        """Return the value for a given config variable.

        :param name: Name of configuration parameter
        :returns: Value(s) for the configuration parameter
        """
        try:
            return self.__cfgdict.get(name, default_values[name])
            # return self.__cfgdict[varname]
        except KeyError:
            raise KeyError(f'Configuration variable, {name}, does not exist') from None
            # return None

    def load(self, filename: str):
        """Load the YAML-format configuration file.

        :param filename: Name of YAML configuration filele.
        """
        tmp = self.yaml.load(open(filename, 'r'))
        self.__cfgdict = tmp

    def update_value(self, name: str, newval: Union[List, str]):
        """Update an existing configuration variable with a new value

        :param name: Name of configuration parameter
        :param newval: New value for parameter
        """

        if name in self.__cfgdict:
            self.__cfgdict[name] = newval
        else:
            raise KeyError(f'Configuration variable, {name}, does not exist')

    def write(self, filename: str):
        """Write the current configuration out to a file.

        :param filename: Name of file to write configuration to
        """

        outfile = open(filename, 'w')
        self.yaml.dump(self.__cfgdict, outfile)
