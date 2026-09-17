import datetime
import logging
import netCDF4 as nc  # type: ignore
import numpy as np
import pandas as pd  # type: ignore
import sys
import xarray as xr

from pathlib import Path
from typing import Dict, List, Optional, Union

from pyPRMS.prms_helpers import set_date   # type: ignore

from Bandit import WDFN

logger = logging.getLogger(__name__)

# Mapping of POI metadata variable names to the keys returned by
# WDFN.get_monitoring_locations() for ad-hoc streamgage lookups.
_WDFN_META_VARS = ('poi_name', 'latitude', 'longitude',
                   'drainage_area', 'drainage_area_contrib')


class POI:
    """Class for accessing point-of-interest observations."""

    def __init__(self, src_path: Optional[str] = None,
                 gage_ids: Optional[List[str]] = None,
                 st_date: Optional[datetime.datetime] = None,
                 en_date: Optional[datetime.datetime] = None,
                 online_lookup: Optional[bool] = False,
                 api_key: Optional[str] = None,
                 verbose: Optional[bool] = False):
        """Create the POI object.

        :param src_path: path to POI netcdf files
        :param gage_ids: list of streamgages to retrieve
        :param st_date: start date for retrieving streamgage observations
        :param en_date: end date for retrieving streamgage observations
        :param online_lookup: attempt an online WDFN lookup for gages missing
            from the source files (e.g. ad-hoc gages), falling back to NaN
        :param api_key: optional USGS Water Data API key for the WDFN lookup
        :param bool verbose: output additional debuggin information
        """

        logger.info('POI netcdf instance')

        self.__src_path = src_path
        self.__stdate = None
        self.__endate = None
        self.__gageids = None

        self.start_date = st_date
        self.end_date = en_date
        self.gage_ids: Optional[List[str]] = gage_ids
        self.__outdata = None
        self.__date_range = None
        self.__final_outorder = None
        self.__online_lookup = online_lookup
        self.__api_key = api_key
        self.__verbose = verbose

        # Overlays populated for gages missing from the source files.
        self.__missing_ids: List[str] = []
        self.__discharge_overlay: Optional[pd.DataFrame] = None
        self.__meta_overlay: Dict[str, Dict] = {}

        self.read()
        self._resolve_missing_gages()

    @property
    def data(self) -> Optional[xr.Dataset]:
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
    def gage_ids(self) -> Optional[List[str]]:
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
                                               join='outer',
                                               # chunks={'poi_id': 1040}, combine='nested',
                                               concat_dim='poi_id', decode_cf=True,
                                               engine='netcdf4')
            # NOTE: With a multi-file dataset the time attributes 'units' and
            #       'calendar' are lost.
            #       see https://github.com/pydata/xarray/issues/2436
        else:
            logger.warning('No poi_ids were specified.')

    def _resolve_missing_gages(self):
        """Identify requested gages missing from the source files.

        Any requested gage ID that is not present in the source POI netCDF
        files is treated as an ad-hoc streamgage. Each such gage is noted in
        the log. When online lookup is enabled, daily streamflow and
        monitoring-location metadata are retrieved from the USGS Water Data
        API (WDFN); gages that cannot be retrieved fall back to NaN entries.
        """

        if self.__outdata is None or not self.__gageids:
            return

        source_ids = set(self.__outdata['poi_id'].values.tolist())
        self.__missing_ids = [gg for gg in self.__gageids if gg not in source_ids]

        if not self.__missing_ids:
            return

        for gg in self.__missing_ids:
            logger.info(f'Ad-hoc streamgage {gg}: not found in POI source files')

        if not self.__online_lookup:
            for gg in self.__missing_ids:
                logger.info(f'Ad-hoc streamgage {gg}: online lookup disabled; '
                            f'writing NaN streamflow')
            return

        # Attempt an online WDFN lookup for the missing (ad-hoc) gages.
        logger.info(f'Attempting online WDFN lookup for {len(self.__missing_ids)} '
                    f'ad-hoc streamgage(s): {self.__missing_ids}')

        try:
            self.__discharge_overlay = WDFN.get_daily_streamflow(gage_ids=self.__missing_ids,
                                                                 st_date=self.__stdate,
                                                                 en_date=self.__endate,
                                                                 api_key=self.__api_key)
        except Exception as err:   # noqa: BLE001 - fall back to NaN on any failure
            logger.warning(f'WDFN streamflow lookup failed for ad-hoc gages '
                           f'{self.__missing_ids}: {err}')
            self.__discharge_overlay = None

        try:
            self.__meta_overlay = WDFN.get_monitoring_locations(gage_ids=self.__missing_ids,
                                                                api_key=self.__api_key)
        except Exception as err:   # noqa: BLE001 - metadata is best-effort
            logger.warning(f'WDFN metadata lookup failed for ad-hoc gages '
                           f'{self.__missing_ids}: {err}')
            self.__meta_overlay = {}

        # Log the outcome for each ad-hoc gage.
        for gg in self.__missing_ids:
            has_data = False
            if self.__discharge_overlay is not None and gg in self.__discharge_overlay.columns:
                has_data = bool(self.__discharge_overlay[gg].notna().any())

            if has_data:
                logger.info(f'Ad-hoc streamgage {gg}: streamflow retrieved from WDFN')
            else:
                logger.info(f'Ad-hoc streamgage {gg}: no WDFN streamflow available; '
                            f'writing NaN streamflow')

    def get(self, var: str) -> pd.DataFrame:
        """Get a subset of data for a given variable.

        Requested gage IDs that are not present in the source netCDF files
        (e.g. ad-hoc streamgages added via ``--add-gages``) are filled with
        NaN entries. The requested order of the gage IDs is preserved.

        :param var: Name of variable from netCDF file
        :returns: Pandas DataFrame of extracted data
        """

        # Reindex on poi_id so that any requested gage IDs missing from the
        # source data are inserted as NaN while preserving the requested order.
        # This avoids a KeyError ("not all values found in index 'poi_id'")
        # when ad-hoc streamgages have no observations in the cached source files.
        subset = self.__outdata[var].reindex(poi_id=self.__gageids)

        if 'time' in subset.dims:
            if self.__stdate is not None and self.__endate is not None:
                try:
                    data = subset.loc[:, self.__stdate:self.__endate].to_pandas()
                except IndexError:
                    print(f'ERROR: Indices (time, poi_id) were used to subset {var} which expects' +
                          f'indices ({" ".join(map(str, self.__outdata[var].coords))})')
                    raise
            else:
                data = subset.loc[:, :].to_pandas()
        else:
            data = subset.to_pandas()

        # Overlay any online WDFN data retrieved for ad-hoc gages missing from
        # the source files. Gages with no online data remain NaN.
        data = self._overlay_missing(var, data)
        return data

    def _overlay_missing(self, var: str, data: pd.DataFrame) -> pd.DataFrame:
        """Overlay online WDFN data for ad-hoc gages onto reindexed results.

        :param var: name of the variable being retrieved
        :param data: reindexed data (NaN rows for missing gages) to update
        :returns: data with any available online values filled in
        """

        if not self.__missing_ids:
            return data

        if var == 'discharge':
            if self.__discharge_overlay is None:
                return data

            # data is indexed by poi_id (rows) x time (columns). The overlay is
            # indexed by date (rows) x gage (columns), so align by transposing.
            for gg in self.__missing_ids:
                if gg in self.__discharge_overlay.columns and gg in data.index:
                    series = self.__discharge_overlay[gg]
                    series = series.reindex(data.columns)
                    data.loc[gg, :] = series.to_numpy()
            return data

        if var in _WDFN_META_VARS:
            for gg in self.__missing_ids:
                if gg in self.__meta_overlay and gg in data.index:
                    value = self.__meta_overlay[gg].get(var)
                    if value is not None:
                        data.loc[gg] = value
            return data

        return data

    def write_ascii(self, filename: Union[str, Path]):
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

    def write_netcdf(self, filename: Union[str, Path]):
        """Write POI streamflow to netcdf format file.

        :param filename: name of the netCDF file to create
        """

        # Ad-hoc streamgages missing from the source files return NaN for
        # poi_name; replace those with an empty string so string handling works.
        poiname_list = ['' if isinstance(nn, float) and np.isnan(nn) else nn
                        for nn in self.get('poi_name').tolist()]

        max_poiid_len = len(max(self.__gageids, key=len))
        max_poiname_len = max(len(max(poiname_list, key=len)), 1)

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
