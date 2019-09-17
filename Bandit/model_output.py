from __future__ import (absolute_import, division, print_function)
# from future.utils import iteritems

from collections import namedtuple
import netCDF4 as nc
import numpy as np
import xarray as xr


class ModelOutput(object):
    def __init__(self, filename, varname, startdate=None, enddate=None, nhm_hrus=None, nhm_segs=None):
        self.__filename = filename
        self.__varname = varname
        self.__stdate = startdate
        self.__endate = enddate
        self.__nhm_hrus = nhm_hrus
        self.__nhm_segs = nhm_segs
        self.__data = None
        self.__dataset = None

        self.read_netcdf()

    @property
    def data(self):
        return self.__data

    @property
    def dataset(self):
        return self.__dataset

    def nearest(self, items, pivot):
        return min(items, key=lambda x: abs(x - pivot))

    def get_var(self, varname):
        if self.__stdate is not None and self.__endate is not None:
            if self.__nhm_hrus:
                data = self.__dataset[varname].loc[self.__stdate:self.__endate, self.__nhm_hrus].to_pandas()
            elif self.__nhm_segs:
                data = self.__dataset[varname].loc[self.__stdate:self.__endate, self.__nhm_segs].to_pandas()
        else:
            if self.__nhm_hrus:
                data = self.__dataset[varname].loc[:, self.__nhm_hrus].to_pandas()
            elif self.__nhm_segs:
                data = self.__dataset[varname].loc[:, self.__nhm_segs].to_pandas()

        return data

    def read_netcdf(self):
        """Read model output file stored in netCDF format"""
        if self.__nhm_hrus:
            # print('\t\tOpen dataarray')

            self.__dataset = xr.open_dataset(self.__filename, chunks={'hru': 1000})
            # self.__dataset = xr.open_dataarray(self.__filename, chunks={'hru': 1000})

            # print('\t\tConvert subset to pandas dataframe')
            # self.__data = ds.loc[:, self.__nhm_hrus].to_pandas()
            #
            # # print('\t\tRestrict to date range')
            # if self.__stdate is not None and self.__endate is not None:
            #     # Restrict dataframe to the given date range
            #     self.__data = self.__data[self.__stdate:self.__endate]
            #
            # self.__dataarray = ds.loc[:, self.__nhm_hrus]
            #
        elif self.__nhm_segs:
            self.__dataset = xr.open_dataset(self.__filename, chunks={'segment': 1000})
        # print('\t\tInsert date info')
        # self.__data['year'] = self.__data.index.year
        # self.__data['month'] = self.__data.index.month
        # self.__data['day'] = self.__data.index.day
        # self.ds = ds
        # self.__data['hour'] = 0
        # self.__data['minute'] = 0
        # self.__data['second'] = 0

    def write_csv(self, pathname=None, fileprefix=None):

        data = self.get_var(self.__varname)

        if self.__nhm_hrus:
            data.to_csv(f'{pathname}/{self.__varname}.csv', columns=self.__nhm_hrus,
                        sep=',', index=True, header=True, chunksize=50)
        elif self.__nhm_segs:
            data.to_csv(f'{pathname}/{self.__varname}.csv', columns=self.__nhm_segs,
                        sep=',', index=True, header=True, chunksize=50)

    def write_netcdf(self, pathname=None):
        # Write output variable to a netcdf file

        if self.__nhm_hrus:
            ss = self.__dataset[self.__varname].loc[self.__stdate:self.__endate, self.__nhm_hrus]
            ss.to_netcdf(f'{pathname}/{self.__varname}.nc', mode='w', format='NETCDF4',
                         encoding = {'time': {'dtype': 'float32', 'calendar': 'standard', '_FillValue': None},
                                     'hru': {'_FillValue': None}})
        elif self.__nhm_segs:
            ss = self.__dataset[self.__varname].loc[self.__stdate:self.__endate, self.__nhm_segs]
            ss.to_netcdf(f'{pathname}/{self.__varname}.nc', mode='w', format='NETCDF4',
                         encoding={'time': {'dtype': 'float32', 'calendar': 'standard', '_FillValue': None},
                                   'segment': {'_FillValue': None}})