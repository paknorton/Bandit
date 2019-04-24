from __future__ import (absolute_import, division, print_function)
# from future.utils import iteritems

from collections import namedtuple
import netCDF4 as nc
import numpy as np
import xarray as xr


# def __init__(self, filename=None, st_date=None, en_date=None, indices=None, nhm_hrus=None, mapping=None,
#              var=None, regions=REGIONS):
#     # def __init__(self, cbhdb_dir, st_date=None, en_date=None, indices=None, nhm_hrus=None, mapping=None):
#     #     self.__cbhdb_dir = cbhdb_dir
#     self.__filename = filename
#
#     # self.__indices = [str(kk) for kk in indices]
#     self.__indices = indices  # OrdereDict: nhm_ids -> local_ids
#
#     self.__stdate = st_date
#     self.__endate = en_date
#     self.__nhm_hrus = nhm_hrus
#     self.__mapping = mapping
#     self.__date_range = None
#     self.__data = None
#     self.__final_outorder = None
#     self.__var = var
#     self.__regions = regions


class DynamicParameters(object):
    def __init__(self, filename, varname, startdate=None, enddate=None, nhm_hrus=None):
        self.__filename = filename
        self.__varname = varname
        self.__stdate = startdate
        self.__endate = enddate
        self.__nhm_hrus = nhm_hrus
        self.__data = None

    @property
    def data(self):
        return self.__data

    def nearest(self, items, pivot):
        return min(items, key=lambda x: abs(x - pivot))

    def read(self):
        """Read a netcdf dynamic parameter file"""

        fhdl = nc.Dataset(self.__filename)

        recdim = namedtuple('recdim', 'name size')

        for xx, yy in fhdl.dimensions.items():
            if yy.isunlimited():
                print('{}: {} (unlimited)'.format(xx, len(yy)))
                recdim.name = xx
                recdim.size = len(yy)
            else:
                print('{}: {}'.format(xx, len(yy)))

        d0units = fhdl.variables[recdim.name].units
        d0calendar = fhdl.variables[recdim.name].calendar

        timelist = nc.num2date(fhdl.variables[recdim.name][:], units=d0units, calendar=d0calendar)

        print('-'*50)
        for xx in fhdl.variables:
            print(xx)

        # Get indices for date closest to the start and end dates
        st_idx = np.where(timelist == self.nearest(timelist, self.__stdate))[0][0]
        en_idx = np.where(timelist == self.nearest(timelist, self.__endate))[0][0]

        print('st_idx: {}'.format(st_idx))
        print('en_idx: {}'.format(en_idx))

        # print('en_idx: {}'.format(self.nearest(timelist, self.__enddate)))
        print('-' * 50)
        print('time.units: {}'.format(d0units))
        print('time.calendar: {}'.format(d0calendar))
        print(timelist[st_idx:en_idx])

        # Extract columns of data for date range
        if self.__nhm_hrus:
            self.__data = fhdl.variables[self.__varname][st_idx:en_idx, self.__nhm_hrus]
        else:
            self.__data = fhdl.variables[self.__varname][st_idx:en_idx, :]
        # print(self.__data)
        fhdl.close()

    def read_netcdf(self):
        """Read dynamic parameter files stored in netCDF format"""
        if self.__nhm_hrus:
            # print('\t\tOpen dataarray')
            ds = xr.open_dataarray(self.__filename, chunks={'hru': 1000})
            # ds = xr.open_dataarray(self.__filename)

            # print('\t\tConvert subset to pandas dataframe')
            self.__data = ds.loc[:, self.__nhm_hrus].to_pandas()

            # print('\t\tRestrict to date range')
            if self.__stdate is not None and self.__endate is not None:
                # Restrict dataframe to the given date range
                self.__data = self.__data[self.__stdate:self.__endate]

            # self.__data = self.__data[self.__indices]

        # print('\t\tInsert date info')
        self.__data['year'] = self.__data.index.year
        self.__data['month'] = self.__data.index.month
        self.__data['day'] = self.__data.index.day
        self.ds = ds
        # self.__data['hour'] = 0
        # self.__data['minute'] = 0
        # self.__data['second'] = 0
