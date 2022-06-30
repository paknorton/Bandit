from typing import List, Optional

import datetime
import pandas as pd
import xarray as xr


class ModelOutput(object):
    def __init__(self, filename: str,
                 varname: str,
                 startdate: Optional[datetime.datetime]=None,
                 enddate: Optional[datetime.datetime]=None,
                 nhm_hrus: Optional[List[int]]=None,
                 nhm_segs: Optional[List[int]]=None):
        """Initialize the model output object.

        :param filename: Name of model output netCDF file
        :param varname: Name of variable to extract
        :param startdate: Start date for extraction
        :param enddate: End date for extraction
        :param nhm_hrus: List of NHM HRUs to extract
        :param nhm_segs: List of NHM segments to extract"""
        self.__filename = filename
        self.__varname = varname
        self.__stdate = startdate
        self.__endate = enddate
        self.__nhm_hrus = nhm_hrus
        self.__nhm_segs = nhm_segs
        self.__data = None

        self.read_netcdf()

    @property
    def data(self) -> pd.DataFrame:
        """Returns the source model output.

        :returns: Model output dataframe
        """
        return self.__data

    @staticmethod
    def nearest(items, pivot):
        return min(items, key=lambda x: abs(x - pivot))

    def get_var(self, varname: str) -> pd.DataFrame:
        """Get the data subset for a given variable.

        :param varname: Name of model output variable
        :returns: Model output dataframe
        """
        data = None

        if self.__stdate is not None and self.__endate is not None:
            if self.__nhm_hrus:
                data = self.__data[varname].loc[self.__stdate:self.__endate, self.__nhm_hrus].to_pandas()
            elif self.__nhm_segs:
                data = self.__data[varname].loc[self.__stdate:self.__endate, self.__nhm_segs].to_pandas()
        else:
            if self.__nhm_hrus:
                data = self.__data[varname].loc[:, self.__nhm_hrus].to_pandas()
            elif self.__nhm_segs:
                data = self.__data[varname].loc[:, self.__nhm_segs].to_pandas()

        return data

    def read_netcdf(self):
        """Read model output file stored in netCDF format."""
        if self.__nhm_hrus:
            self.__data = xr.open_dataset(self.__filename, chunks={})
            self.__data = self.__data.assign_coords(nhru=self.__data.nhm_id)
            # try:
            #     self.__data = xr.open_dataset(self.__filename, chunks={'hru': 1000})
            # except ValueError:
            #     self.__data = xr.open_dataset(self.__filename, chunks={'nhru': 1000})
            #     self.__data = self.__data.assign_coords(nhru=(self.__data.nhm_id))
        elif self.__nhm_segs:
            self.__data = xr.open_dataset(self.__filename, decode_coords=True, chunks={})
            self.__data = self.__data.assign_coords(nsegment=self.__data.nhm_seg)
            # try:
            #     print('first')
            #     self.__data = xr.open_dataset(self.__filename, decode_coords=True, chunks={'segment': 1000})
            # except ValueError:
            #     print('second')
            # self.__data = xr.open_dataset(self.__filename, decode_coords=True, chunks={'nsegment': 1000})
            # self.__data = self.__data.assign_coords(nsegment=(self.__data.nhm_seg))

    def write_csv(self, pathname: str):
        """Write model output subset to PRMS CSV file.

        :param pathname: location to write file to (filename is based on variable)
        """
        data = self.get_var(self.__varname)

        if self.__nhm_hrus:
            data.to_csv(f'{pathname}/{self.__varname}.csv', columns=self.__nhm_hrus,
                        sep=',', index=True, header=True, chunksize=50)
        elif self.__nhm_segs:
            data.to_csv(f'{pathname}/{self.__varname}.csv', columns=self.__nhm_segs,
                        sep=',', index=True, header=True, chunksize=50)

    def write_netcdf(self, pathname: str):
        """Write model output subset to netCDF file.

        :param pathname: location to write file to (filename is based on variable)
        """
        if self.__nhm_hrus:
            ss = self.__data[self.__varname].loc[self.__stdate:self.__endate, self.__nhm_hrus]
            ss.to_netcdf(f'{pathname}/{self.__varname}.nc', mode='w', format='NETCDF4',
                         encoding={'time': {'dtype': 'float32', 'calendar': 'standard', '_FillValue': None},
                                   'hru': {'_FillValue': None}})
        elif self.__nhm_segs:
            ss = self.__data[self.__varname].loc[self.__stdate:self.__endate, self.__nhm_segs]
            ss.to_netcdf(f'{pathname}/{self.__varname}.nc', mode='w', format='NETCDF4',
                         encoding={'time': {'dtype': 'float32', 'calendar': 'standard', '_FillValue': None},
                                   'segment': {'_FillValue': None}})
