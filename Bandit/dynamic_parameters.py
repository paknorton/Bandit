
from collections import namedtuple
from typing import Dict, List, Optional, Union

import datetime
import netCDF4 as nc
import numpy as np
import xarray as xr


class DynamicParameters(object):

    """Class for handling dynamic parameters.
    """

    def __init__(self, filename: str,
                 varname: str,
                 startdate: Optional[datetime.datetime]=None,
                 enddate: Optional[datetime.datetime]=None,
                 nhm_hrus: Optional[List[int]]=None):
        """Create the DynamicParameters object.

        :param filename: name of source dynamic parameter file
        :param varname: name of variable to extract
        :param startdate: start date of extraction
        :param enddate: end date of extraction
        :param nhm_hrus: list of NHM HRUs to extract
        """

        self.__filename = filename
        self.__varname = varname
        self.__stdate = startdate
        self.__endate = enddate
        self.__nhm_hrus = nhm_hrus
        self.__data = None
        self.ds = None

    @property
    def data(self) -> xr.Dataset:
        """Get the data for dynamic parameter

        :returns: dynamic parameter data
        """

        return self.__data

    @staticmethod
    def nearest(items, pivot):
        return min(items, key=lambda x: abs(x - pivot))

    # def read(self):
    #     """Read a netCDF dynamic parameter file."""
    #
    #     fhdl = nc.Dataset(self.__filename)
    #
    #     recdim = namedtuple('recdim', 'name size')
    #
    #     for xx, yy in fhdl.dimensions.items():
    #         if yy.isunlimited():
    #             print(f'{xx}: {len(yy)} (unlimited)')
    #             recdim.name = xx
    #             recdim.size = len(yy)
    #         else:
    #             print(f'{xx}: {len(yy)}')
    #
    #     d0units = fhdl.variables[recdim.name].units
    #     d0calendar = fhdl.variables[recdim.name].calendar
    #
    #     timelist = nc.num2date(fhdl.variables[recdim.name][:], units=d0units, calendar=d0calendar)
    #
    #     print('-'*50)
    #     for xx in fhdl.variables:
    #         print(xx)
    #
    #     # Get indices for date closest to the start and end dates
    #     st_idx = np.where(timelist == self.nearest(timelist, self.__stdate))[0][0]
    #     en_idx = np.where(timelist == self.nearest(timelist, self.__endate))[0][0]
    #
    #     print(f'st_idx: {st_idx}')
    #     print(f'en_idx: {en_idx}')
    #
    #     # print('en_idx: {}'.format(self.nearest(timelist, self.__enddate)))
    #     print('-' * 50)
    #     print(f'time.units: {d0units}')
    #     print(f'time.calendar: {d0calendar}')
    #     print(timelist[st_idx:en_idx])
    #
    #     # Extract columns of data for date range
    #     if self.__nhm_hrus:
    #         self.__data = fhdl.variables[self.__varname][st_idx:en_idx, self.__nhm_hrus]
    #     else:
    #         self.__data = fhdl.variables[self.__varname][st_idx:en_idx, :]
    #     # print(self.__data)
    #     fhdl.close()

    def read_netcdf(self):
        """Read dynamic parameter files stored in netCDF format.
        """

        ds = None

        if self.__nhm_hrus:
            ds = xr.open_dataarray(self.__filename, chunks={'hru': 1000})

            # Subset to given HRUs and convert to pandas DataFrame
            self.__data = ds.loc[:, self.__nhm_hrus].to_pandas()

            if self.__stdate is not None and self.__endate is not None:
                # Restrict dataframe to the given date range
                self.__data = self.__data[self.__stdate:self.__endate]

        # Split the date into separate columns
        self.__data['year'] = self.__data.index.year
        self.__data['month'] = self.__data.index.month
        self.__data['day'] = self.__data.index.day
        self.ds = ds
