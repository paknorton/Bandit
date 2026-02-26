from typing import List, Optional, Union

import numpy as np
import pandas as pd   # type: ignore
import xarray as xr

from pathlib import Path


class ModelOutput(object):
    def __init__(self, filename: Union[str, Path, List[Union[str, Path]]]):
        """Initialize the model output object.

        :param filename: Name of model output kerchunk JSON or netCDF file
        """

        if isinstance(filename, str):
            filename = Path(filename)

        self.__filename = filename

        # map local coordinate dimensions to global dimension variables
        self.__coord_dims = dict(nhru='nhm_id', nsegment='nhm_seg')
        self.__data = None

        self.__data = xr.open_mfdataset(self.__filename, chunks={}, coords="none", data_vars="minimal",
                                        compat='override', parallel=True)
        self.__data = self.__data.assign_coords(nhru=self.__data.nhm_id)
        self.__data = self.__data.assign_coords(nsegment=self.__data.nhm_seg)

    @property
    def data(self) -> xr.Dataset:
        """Returns the source model output.

        :returns: Model output dataframe
        """
        return self.__data

    @staticmethod
    def nearest(items, pivot):
        return min(items, key=lambda x: abs(x - pivot))

    def write_csv(self, pathname: Union[str, Path],
                  variables: Optional[Union[str, List[str]]] = None,
                  time_slice: Optional[Union[list, slice]] = None,
                  hru_ids: Optional[Union[list, np.ndarray]] = None,
                  seg_ids: Optional[Union[list, np.ndarray]] = None):
        """Write model output subset to PRMS CSV file. If more than one variable
        is selected, a separate CSV file will be created for each variable.

        :param pathname: location to write file to (filename is based on variable)
        :param variables: list of variables to write (list of variables will write one variable per file)
        :param time_slice: time slice to write (default is all time steps)
        :param hru_ids: list of NHM HRU IDs to write (default is all HRUs)
        :param seg_ids: list of NHM segment IDs to write (default is all segments)
        """

        ds = self._select_data(variables=variables,
                               time_slice=time_slice,
                               hru_ids=hru_ids,
                               seg_ids=seg_ids)

        for cvar in ds.data_vars:
            if cvar not in self.__coord_dims.values():
                ds[cvar].to_pandas().to_csv(f'{pathname}/{cvar}.csv',
                                            sep=',',
                                            index=True,
                                            header=True,
                                            chunksize=50)

    def write_netcdf(self, filename: Union[str, Path],
                     variables: Optional[Union[str, List[str]]] = None,
                     time_slice: Optional[Union[list, slice]] = None,
                     hru_ids: Optional[Union[list, np.ndarray]] = None,
                     seg_ids: Optional[Union[list, np.ndarray]] = None):
        """Write model output subset to netCDF file.

        :param filename: name of netCDF output file
        :param variables: list of variables to write
        :param time_slice: time slice to write (default is all time steps)
        :param hru_ids: list of NHM HRU IDs to write (default is all HRUs)
        :param seg_ids: list of NHM segment IDs to write (default is all segments)
        """

        ds = self._select_data(variables=variables,
                               time_slice=time_slice,
                               hru_ids=hru_ids,
                               seg_ids=seg_ids)

        # Add local model IDs
        if 'nhru' in ds.dims:
            # Change the nhru coordinate variable values to reflect the local model HRU IDs
            ds = ds.assign_coords(nhru=np.arange(1, ds.nhru.values.size+1, dtype=ds.nhru.dtype))
            ds['nhru'].attrs['long_name'] = 'Local model Hydrologic Response Unit ID (HRU)'
            ds['nhru'].attrs['cf_role'] = 'timeseries_id'

        if 'nsegment' in ds.dims:
            # Change the nsegment coordinate variable values to reflect the local model HRU IDs
            ds = ds.assign_coords(nsegment=np.arange(1, ds.nsegment.values.size+1, dtype=ds.nsegment.dtype))
            ds['nsegment'].attrs['long_name'] = 'Local model segment ID'
            ds['nsegment'].attrs['cf_role'] = 'timeseries_id'

        # Set the encoding required for the output netcdf file
        encoding = {}

        for cvar in ds.coords:
            encoding[cvar] = dict(_FillValue=None, contiguous=True)

        for cvar in ds.data_vars:
            if cvar in self.__coord_dims.values():
                encoding[cvar] = dict(_FillValue=None, contiguous=True)
            else:
                # Add coordinates attribute to the data variable
                # For single variable files this allows the data variable to be
                # opened as an xarray dataarray.
                ds[cvar].attrs['coordinates'] = f'time {self.__coord_dims[ds[cvar].dims[-1]]}'

                # Set the encoding for the data variable
                encoding[cvar] = dict(_FillValue=ds[cvar].encoding['_FillValue'],
                                      compression='zlib',
                                      complevel=2,
                                      fletcher32=True)

        ds.to_netcdf(filename, engine='netcdf4', format='NETCDF4', encoding=encoding)

    def _select_data(self,
                     variables: Optional[Union[str, List[str]]] = None,
                     time_slice: Optional[Union[list, slice]] = None,
                     hru_ids: Optional[Union[list, np.ndarray]] = None,
                     seg_ids: Optional[Union[list, np.ndarray]] = None):
        """Select a subset of the model output data.

        :param variables: list of variables to select (default is all variables)
        :param time_slice: time slice to select (default is all time steps)
        :param hru_ids: list of NHM HRU IDs to select (default is all HRUs)
        :param seg_ids: list of NHM segment IDs to select (default is all segments)
        """

        if isinstance(variables, str):
            variables = [variables]
        if variables is None:
            # Select all variables
            variables = list(self.__data.data_vars)

        if time_slice is None:
            # Return all time steps if time_slice is not provided
            time_slice = slice(self.__data['time'][0].values,
                               self.__data['time'][-1].values)

        if isinstance(time_slice, list):
            time_slice = slice(time_slice[0], time_slice[-1])

        addl_vars = set([self.__coord_dims[self.__data[cvar].dims[-1]] for cvar in variables])
        # print(f'addl_vars: {addl_vars}')

        for cvar in addl_vars:
            # Add the national IDs
            variables.append(cvar)

        # What dimensions are we using?
        used_dims = [kk for kk, vv in self.__coord_dims.items() if vv in addl_vars]

        # Remove dimensions not needed for the selected variables
        rem_dims = [kk for kk, vv in self.__coord_dims.items() if vv not in addl_vars]
        # print(f'rem_dims: {rem_dims}')

        ds = self.__data.drop_dims(rem_dims)

        sel_criteria = dict(time=time_slice)
        if 'nhru' in used_dims and hru_ids is not None:
            sel_criteria['nhru'] = hru_ids
        if 'nsegment' in used_dims and seg_ids is not None:
            sel_criteria['nsegment'] = seg_ids
        # print(f'sel_criteria: {sel_criteria}')

        return ds[variables].sel(sel_criteria)
