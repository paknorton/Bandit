import datetime
import logging
import netCDF4 as nc  # type: ignore
import numpy as np
import pandas as pd  # type: ignore
import sys
import xarray as xr

from typing import List, Optional, Union

from Bandit.bandit_helpers import set_date

logger = logging.getLogger(__name__)

class POI:
    """Class for accessing point-of-interest observations."""

    def __init__(self, src_path: Optional[str]=None,
                 gage_ids: Optional[List[str]]=None,
                 st_date: Optional[datetime.datetime]=None,
                 en_date: Optional[datetime.datetime]=None,
                 verbose: Optional[bool]=False):
        """Create the POI object.

        :param src_path: path to POI netcdf files
        :param gage_ids: list of streamgages to retrieve
        :param st_date: start date for retrieving streamgage observations
        :param en_date: end date for retrieving streamgage observations
        :param bool verbose: output additional debuggin information
        """

        logger.info('POI netcdf instance')

        self.__src_path = src_path
        self.__stdate = None
        self.__endate = None
        self.__gageids = None

        self.start_date = st_date
        self.end_date = en_date
        self.gage_ids = gage_ids
        self.__outdata = None
        self.__date_range = None
        self.__final_outorder = None
        self.__verbose = verbose

        self.read()

    @property
    def data(self) -> xr.Dataset:
        """Returns the source netcdf dataset.

        :returns: source netCDF xarray Dataset
        """
        return self.__outdata

    @property
    def start_date(self) -> Union[datetime.datetime, None]:
        """Get the start date.

        :returns: start date of retrieved observations
        """

        return self.__stdate

    @start_date.setter
    def start_date(self, st_date: Union[datetime.datetime, str]):
        """Set the start date for observations.

        :param st_date: set start date for observations (either a datetime object or a string of the form YYYY-MM-DD)
        """

        # As written this will clear any streamgage observations that have been downloaded.
        self.__stdate = set_date(st_date)
        self.__outdata = None

    @property
    def end_date(self) -> Union[datetime.datetime, None]:
        """Get the end date for observations.

        :returns: end date of retrieved observations
        """

        return self.__endate

    @end_date.setter
    def end_date(self, en_date: Union[datetime.datetime, None]):
        """Set the end date for observations.

        :param en_date: end date for observations (either a datetime object or a string of the form YYYY-MM-DD)
        """
        self.__endate = set_date(en_date)
        self.__outdata = None

    @property
    def gage_ids(self) -> List[str]:
        """Get list of streamgage IDs for retrieval.

        :returns: list of streamgage IDs
        """

        return self.__gageids

    @gage_ids.setter
    def gage_ids(self, gage_ids: Union[List[str], str]):
        """Set the streamgage ID(s) to retrieve.

        :param gage_ids: streamgage ID(s)
        """

        # Set the gage ids for retrieval this will clear any downloaded observations
        if isinstance(gage_ids, (list, tuple)):
            self.__gageids = gage_ids
        else:
            # Assuming a single value, so convert to a list
            self.__gageids = [gage_ids]
        self.__outdata = None

    def read(self):
        """Read POI files stored in netCDF format."""

        if self.__gageids:
            # print('\t\tOpen dataset')
            self.__outdata = xr.open_mfdataset(self.__src_path,
                                               chunks={}, combine='nested',
                                               # chunks={'poi_id': 1040}, combine='nested',
                                               concat_dim='poi_id', decode_cf=True,
                                               engine='netcdf4')
            # NOTE: With a multi-file dataset the time attributes 'units' and
            #       'calendar' are lost.
            #       see https://github.com/pydata/xarray/issues/2436
        else:
            logger.warning('No poi_ids were specified.')

    def get(self, var: str) -> pd.DataFrame:
        """Get a subset of data for a given variable.

        :param var: Name of variable from netCDF file
        :returns: Pandas DataFrame of extracted data
        """
        if 'time' in self.__outdata[var].dims:
            if self.__stdate is not None and self.__endate is not None:
                try:
                    data = self.__outdata[var].loc[self.__gageids, self.__stdate:self.__endate].to_pandas()
                except IndexError:
                    print(f'ERROR: Indices (time, poi_id) were used to subset {var} which expects' +
                          f'indices ({" ".join(map(str, self.__outdata[var].coords))})')
                    raise
            else:
                data = self.__outdata[var].loc[self.__gageids, :].to_pandas()
        else:
            data = self.__outdata[var].loc[self.__gageids].to_pandas()
        return data

    def write_ascii(self, filename: str):
        """Writes POI observations to a file in PRMS format.

        :param filename: name of the file to create
        """

        out_order = [kk for kk in self.__gageids]
        for cc in ['second', 'minute', 'hour', 'day', 'month', 'year']:
            out_order.insert(0, cc)

        data = self.get('discharge').T

        # Create the year, month, day, hour, minute, second columns
        try:
            data['year'] = data.index.year
            data['month'] = data.index.month
            data['day'] = data.index.day
            data['hour'] = data.index.hour
            data['minute'] = data.index.minute
            data['second'] = data.index.second
            data.fillna(-999, inplace=True)

        except AttributeError:
            print('AttributeError')
            print(data.head())
            print(data.info())

        outhdl = open(filename, 'w')
        outhdl.write('Created by Bandit\n')
        outhdl.write('/////////////////////////////////////////////////////////////////////////\n')
        outhdl.write('// Station IDs for runoff:\n')
        outhdl.write('// ID\n')

        if not self.__gageids:
            outhdl.write('// 00000000\n')
        else:
            for gg in self.__gageids:
                outhdl.write(f'// {gg}\n')

        outhdl.write('/////////////////////////////////////////////////////////////////////////\n')
        outhdl.write('// Unit: runoff = cfs\n')
        outhdl.write('/////////////////////////////////////////////////////////////////////////\n')
        outhdl.write(f'runoff {len(self.__gageids)}\n')
        outhdl.write('#########################################################\n')

        data.to_csv(outhdl, sep=' ', columns=out_order, index=False, header=False)
        outhdl.close()

        if self.__verbose:
            sys.stdout.write('\r                                       ')
            sys.stdout.write(f'\r\tStreamflow data written to: {filename}\n')
            sys.stdout.flush()

    def write_netcdf(self, filename: str):
        """Write POI streamflow to netcdf format file.

        :param filename: name of the netCDF file to create
        """

        poiname_list = self.get('poi_name').tolist()

        max_poiid_len = len(max(self.__gageids, key=len))
        max_poiname_len = len(max(poiname_list, key=len))

        # Create a netCDF file for the CBH data
        nco = nc.Dataset(filename, 'w', clobber=True)

        # Create the dimensions
        nco.createDimension('poiid_nchars', max_poiid_len)
        nco.createDimension('poi_id', len(self.__gageids))
        nco.createDimension('poiname_nchars', max_poiname_len)
        nco.createDimension('time', None)

        reference_time = self.__stdate.strftime('%Y-%m-%d %H:%M:%S')
        cal_type = 'standard'

        # Create the variables
        timeo = nco.createVariable('time', 'f4', 'time')
        timeo.calendar = cal_type
        timeo.units = f'days since {reference_time}'

        poiido = nco.createVariable('poi_id', 'S1', ('poi_id', 'poiid_nchars'), zlib=True)
        poiido.long_name = 'Point-of-Interest ID'
        poiido.cf_role = 'timeseries_id'
        poiido._Encoding = 'ascii'

        poinameo = nco.createVariable('poi_name', 'S1', ('poi_id', 'poiname_nchars'), zlib=True)
        poinameo.long_name = 'Name of POI station'

        lato = nco.createVariable('latitude', 'f4', 'poi_id', zlib=True)
        lato.long_name = 'Latitude'
        lato.units = 'degrees_north'

        lono = nco.createVariable('longitude', 'f4', 'poi_id', zlib=True)
        lono.long_name = 'Longitude'
        lono.units = 'degrees_east'

        draino = nco.createVariable('drainage_area', 'f4', 'poi_id',
                                    fill_value=nc.default_fillvals['f4'], zlib=True)
        draino.long_name = 'Drainage Area'
        draino.units = 'mi2'

        draineffo = nco.createVariable('drainage_area_contrib', 'f4', 'poi_id',
                                       fill_value=nc.default_fillvals['f4'], zlib=True)
        draineffo.long_name = 'Effective drainage area'
        draineffo.units = 'mi2'

        varo = nco.createVariable('discharge', 'f4', ('poi_id', 'time'),
                                  fill_value=nc.default_fillvals['f4'], zlib=True)
        varo.long_name = 'discharge'
        varo.units = 'ft3 s-1'

        nco.setncattr('Description', 'POI data for PRMS')
        nco.setncattr('FeatureType', 'timeSeries')
        # nco.setncattr('Bandit_version', __version__)
        # nco.setncattr('NHM_version', nhmparamdb_revision)

        data = self.get('discharge')

        # Write the Streamgage IDs
        poiido[:] = nc.stringtochar(np.array(self.__gageids).astype('S'))

        timeo[:] = nc.date2num(pd.to_datetime(data.T.index).tolist(),
                               units=f'days since {reference_time}',
                               calendar=cal_type)

        # Write the streamgage observations
        varo[:, :] = data.to_numpy(dtype=float)

        poinameo[:] = nc.stringtochar(np.array(poiname_list).astype('S'))
        lato[:] = self.get('latitude').to_numpy(dtype=float)
        lono[:] = self.get('longitude').to_numpy(dtype=float)
        draino[:] = self.get('drainage_area').to_numpy(dtype=float)
        draineffo[:] = self.get('drainage_area_contrib').to_numpy(dtype=float)
        nco.close()
