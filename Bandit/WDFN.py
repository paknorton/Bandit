"""Client for the USGS Water Data for the Nation (WDFN) OGC API.

This module provides a small, self-contained client for retrieving daily
mean streamflow observations and monitoring-location metadata from the new
USGS Water Data API (``api.waterdata.usgs.gov/ogcapi/v0``). It replaces the
legacy ``waterservices.usgs.gov/nwis`` endpoints which are being retired.

Within Bandit this is used to attempt an online lookup of ad-hoc streamgages
(added via ``--add-gages``) that have no observations in the cached POI
source files. It is intentionally scoped for a small number of gages; the
CONUS-scale caching workflow lives in the ``nhm_utilities`` project.

References:
- API docs: https://api.waterdata.usgs.gov/docs/ogcapi/
- Daily values: https://api.waterdata.usgs.gov/ogcapi/v0/collections/daily/items
- Monitoring locations:
  https://api.waterdata.usgs.gov/ogcapi/v0/collections/monitoring-locations/items
- API keys: https://api.waterdata.usgs.gov/docs/ogcapi/keys/

Adapted from the ``water_data_api`` module developed for the USGS
nhm_utilities project by Parker Norton.
"""

import datetime
import json
import logging
import time

from pathlib import Path
from typing import Dict, List, Optional, Union
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import netCDF4 as nc  # type: ignore
import numpy as np
import pandas as pd  # type: ignore

logger = logging.getLogger(__name__)

# Placeholder gage ID used when a model subset contains no POIs. PRMS still
# expects a runoff file, so a single all-missing streamgage is written.
DUMMY_GAGE_ID = '00000000'

# ===========================================================================
# Configuration
# ===========================================================================
BASE_URL = 'https://api.waterdata.usgs.gov/ogcapi/v0'
DAILY_ITEMS_URL = f'{BASE_URL}/collections/daily/items'
MONITORING_LOCATIONS_URL = f'{BASE_URL}/collections/monitoring-locations/items'

# Maximum number of records per page (API maximum is 50000)
MAX_LIMIT = 50000

# Daily mean discharge is parameter_code 00060, statistic_id 00003
DISCHARGE_PARAMETER_CODE = '00060'
MEAN_STATISTIC_ID = '00003'

# Retry/backoff settings
MAX_RETRIES = 5
INITIAL_BACKOFF = 3.0  # seconds
BACKOFF_MULTIPLIER = 2.0
MAX_BACKOFF = 60.0  # cap the maximum wait between retries
REQUEST_TIMEOUT = 120  # seconds


def get_daily_streamflow(gage_ids: List[str],
                         st_date: datetime.datetime,
                         en_date: datetime.datetime,
                         api_key: Optional[str] = None) -> pd.DataFrame:
    """Retrieve daily mean streamflow for a list of NWIS gages.

    Downloads daily mean discharge (parameter_code 00060, statistic_id 00003)
    for each gage between ``st_date`` and ``en_date``. The result is a
    DataFrame with a ``DatetimeIndex`` covering the full requested date range
    and one column per gage (columns named by the bare NWIS site number). Gages
    that return no data are present as all-NaN columns. Columns are ordered to
    match ``gage_ids``.

    :param gage_ids: list of NWIS gage IDs (without agency prefix)
    :param st_date: start date for retrieval
    :param en_date: end date for retrieval
    :param api_key: optional USGS Water Data API key
    :returns: DataFrame indexed by date with one discharge column per gage
    """

    date_range = pd.date_range(start=st_date, end=en_date, freq='D')
    result_df = pd.DataFrame(index=date_range)
    result_df.index.name = 'date'

    records = _fetch_daily(gage_ids, st_date, en_date, api_key=api_key)

    if records:
        df = pd.DataFrame(records)
        df['site_no'] = df['monitoring_location_id'].str.replace(r'^USGS-', '', regex=True)
        df['date'] = pd.to_datetime(df['time'])
        df['value'] = pd.to_numeric(df['value'], errors='coerce')

        df_wide = df.pivot_table(index='date', columns='site_no',
                                 values='value', aggfunc='first')
        result_df = result_df.join(df_wide, how='left')

    # Ensure every requested gage has a column, ordered to match gage_ids.
    # reindex inserts all-NaN columns for any gage that returned no data.
    result_df = result_df.reindex(columns=gage_ids).copy()

    return result_df


def get_monitoring_locations(gage_ids: List[str],
                             api_key: Optional[str] = None) -> Dict[str, Dict]:
    """Retrieve monitoring-location metadata for a list of NWIS gages.

    :param gage_ids: list of NWIS gage IDs (without agency prefix)
    :param api_key: optional USGS Water Data API key
    :returns: dict mapping gage ID to a metadata dict with keys ``poi_name``,
        ``latitude``, ``longitude``, ``drainage_area`` and
        ``drainage_area_contrib``. Gages not found are omitted from the dict.
    """

    monitoring_ids = ','.join([f'USGS-{gg}' for gg in gage_ids])

    params: Dict[str, object] = {'id': monitoring_ids, 'limit': MAX_LIMIT}
    if api_key:
        params['api_key'] = api_key

    url = f'{MONITORING_LOCATIONS_URL}?{urlencode(params)}'
    features = _fetch_all_pages(url, api_key=api_key)

    meta: Dict[str, Dict] = {}
    for feature in features:
        props = feature.get('properties', {})
        geom = feature.get('geometry', {})
        coords = geom.get('coordinates', [None, None]) if geom else [None, None]

        mlid = props.get('monitoring_location_id', '') or feature.get('id', '')
        site_no = mlid.replace('USGS-', '') if mlid else ''

        if not site_no:
            continue

        meta[site_no] = {'poi_name': props.get('monitoring_location_name', ''),
                         'latitude': coords[1] if coords else np.nan,
                         'longitude': coords[0] if coords else np.nan,
                         'drainage_area': props.get('drainage_area', np.nan),
                         'drainage_area_contrib': props.get('contributing_drainage_area', np.nan)}

    return meta


# ===========================================================================
# Internal helpers
# ===========================================================================

def _fetch_daily(gage_ids: List[str],
                 st_date: datetime.datetime,
                 en_date: datetime.datetime,
                 api_key: Optional[str] = None) -> List[Dict]:
    """Fetch daily discharge records for the given gages and date range.

    :returns: list of record dicts with keys ``monitoring_location_id``,
        ``time`` and ``value``. An empty list is returned on failure.
    """

    monitoring_ids = ','.join([f'USGS-{gg}' for gg in gage_ids])
    datetime_range = f'{st_date.strftime("%Y-%m-%d")}/{en_date.strftime("%Y-%m-%d")}'

    params: Dict[str, object] = {'monitoring_location_id': monitoring_ids,
                                 'parameter_code': DISCHARGE_PARAMETER_CODE,
                                 'statistic_id': MEAN_STATISTIC_ID,
                                 'datetime': datetime_range,
                                 'properties': 'monitoring_location_id,time,value',
                                 'limit': MAX_LIMIT}
    if api_key:
        params['api_key'] = api_key

    url = f'{DAILY_ITEMS_URL}?{urlencode(params)}'

    try:
        features = _fetch_all_pages(url, api_key=api_key)
    except RuntimeError as err:
        logger.warning(f'WDFN daily values request failed: {err}')
        return []

    records = []
    for feat in features:
        props = feat.get('properties', {})
        records.append({'monitoring_location_id': props.get('monitoring_location_id', ''),
                        'time': props.get('time', ''),
                        'value': props.get('value')})

    return records


def _fetch_all_pages(initial_url: str,
                     api_key: Optional[str] = None) -> List[Dict]:
    """Fetch all pages of a paginated GeoJSON response.

    Follows ``next`` links until all pages are consumed and returns the
    accumulated list of ``features``.

    :raises RuntimeError: if the first page fails after all retries
    """

    all_features: List[Dict] = []
    url: Optional[str] = initial_url
    page_num = 0

    while url:
        page_num += 1
        data = _request_with_retry(url)

        if data is None:
            if page_num == 1:
                raise RuntimeError(f'All retries exhausted for initial request: {url[:200]}')
            logger.warning(f'WDFN page {page_num} failed; returning '
                           f'{len(all_features)} features collected so far')
            break

        all_features.extend(data.get('features', []))

        url = _get_next_link(data)
        if url and api_key and 'api_key' not in url:
            separator = '&' if '?' in url else '?'
            url = f'{url}{separator}api_key={api_key}'

    return all_features


def _get_next_link(data: Dict) -> Optional[str]:
    """Extract the ``next`` pagination link from a GeoJSON response."""

    for link in data.get('links', []):
        if link.get('rel') == 'next':
            return link.get('href')
    return None


def _request_with_retry(url: str) -> Optional[Dict]:
    """Make an HTTP GET request with retry and exponential backoff.

    Handles HTTP 429 (rate limited), transient server errors and network
    errors. Returns the parsed JSON dict, or None on failure.
    """

    backoff = INITIAL_BACKOFF

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            req = Request(url)
            req.add_header('Accept', 'application/geo+json')

            response = urlopen(req, timeout=REQUEST_TIMEOUT)
            encoding = response.info().get_param('charset', failobj='utf-8')
            body = response.read().decode(encoding)
            return json.loads(body)
        except HTTPError as err:
            if err.code == 429:
                retry_after = err.headers.get('Retry-After')
                wait_time = float(retry_after) if retry_after else backoff
                wait_time = min(wait_time, MAX_BACKOFF)
                logger.warning(f'WDFN rate limited (429); waiting {wait_time:.1f}s '
                               f'(attempt {attempt}/{MAX_RETRIES})')
                time.sleep(wait_time)
                backoff = min(backoff * BACKOFF_MULTIPLIER, MAX_BACKOFF)
            elif err.code == 404:
                logger.warning(f'WDFN not found (404) for URL: {url[:200]}')
                return None
            elif err.code >= 500:
                logger.warning(f'WDFN server error ({err.code}); retrying in '
                               f'{backoff:.1f}s (attempt {attempt}/{MAX_RETRIES})')
                time.sleep(backoff)
                backoff = min(backoff * BACKOFF_MULTIPLIER, MAX_BACKOFF)
            else:
                logger.error(f'WDFN HTTP error ({err.code}) for URL: {url[:200]}')
                if attempt == MAX_RETRIES:
                    return None
                time.sleep(backoff)
                backoff = min(backoff * BACKOFF_MULTIPLIER, MAX_BACKOFF)
        except (URLError, TimeoutError, ConnectionResetError, OSError) as err:
            logger.warning(f'WDFN network error: {err}; retrying in {backoff:.1f}s '
                           f'(attempt {attempt}/{MAX_RETRIES})')
            time.sleep(backoff)
            backoff = min(backoff * BACKOFF_MULTIPLIER, MAX_BACKOFF)

    logger.error(f'WDFN gave up after {MAX_RETRIES} retries for URL: {url[:200]}')
    return None


# ===========================================================================
# Streamflow writer
# ===========================================================================

class WDFNStreamflow:
    """Daily streamflow observations retrieved from the USGS WDFN service.

    This is the online-only replacement for the retired NWIS web service
    (``waterservices.usgs.gov``). It retrieves daily mean discharge and
    monitoring-location metadata for a set of streamgages from the USGS Water
    Data API and writes them to PRMS-format ASCII or netCDF files, matching the
    output produced by :class:`Bandit.points_of_interest.POI`.

    When no streamgages are supplied (a model subset with no POIs), a single
    dummy streamgage of all-missing values is produced so a valid runoff file
    is still written.
    """

    def __init__(self,
                 gage_ids: Optional[List[str]],
                 st_date: datetime.datetime,
                 en_date: datetime.datetime,
                 api_key: Optional[str] = None,
                 verbose: bool = False):
        """Create the WDFNStreamflow object and retrieve observations.

        :param gage_ids: list of NWIS gage IDs to retrieve; an empty list or
            None produces a single dummy streamgage of missing values
        :param st_date: start date for retrieving streamgage observations
        :param en_date: end date for retrieving streamgage observations
        :param api_key: optional USGS Water Data API key for the WDFN lookup
        :param verbose: output additional information
        """

        logger.info('WDFNStreamflow instance')

        self.__stdate = st_date
        self.__endate = en_date
        self.__api_key = api_key
        self.__verbose = verbose

        self.__date_range = pd.date_range(start=st_date, end=en_date, freq='D')

        # An empty/None gage list means the model subset has no POIs; write a
        # single dummy streamgage so PRMS still gets a valid runoff file.
        self.__is_dummy = not gage_ids
        self.__gageids: List[str] = list(gage_ids) if gage_ids else [DUMMY_GAGE_ID]

        # discharge: date (rows) x gage (columns); metadata: gage -> dict
        self.__discharge: pd.DataFrame
        self.__meta: Dict[str, Dict]

        self._retrieve()

    def _retrieve(self):
        """Retrieve discharge and metadata from WDFN (unless this is a dummy)."""

        if self.__is_dummy:
            logger.warning('No streamgages provided - dummy entry created.')
            self.__discharge = pd.DataFrame(index=self.__date_range,
                                            columns=self.__gageids)
            self.__discharge.index.name = 'date'
            self.__meta = {}
            return

        if self.__verbose:
            logger.info(f'Retrieving {len(self.__gageids)} streamgage(s) from WDFN')

        # Daily discharge: all-NaN columns for any gage WDFN has no data for.
        # Any failure falls back to an all-NaN frame so a file is still written.
        try:
            self.__discharge = get_daily_streamflow(gage_ids=self.__gageids,
                                                    st_date=self.__stdate,
                                                    en_date=self.__endate,
                                                    api_key=self.__api_key)
        except Exception as err:   # noqa: BLE001 - fall back to NaN on any failure
            logger.warning(f'WDFN streamflow lookup failed for {self.__gageids}: {err}')
            self.__discharge = pd.DataFrame(index=self.__date_range,
                                            columns=self.__gageids)
            self.__discharge.index.name = 'date'

        # Monitoring-location metadata is best-effort.
        try:
            self.__meta = get_monitoring_locations(gage_ids=self.__gageids,
                                                   api_key=self.__api_key)
        except Exception as err:   # noqa: BLE001 - metadata is best-effort
            logger.warning(f'WDFN metadata lookup failed for {self.__gageids}: {err}')
            self.__meta = {}

    @property
    def data(self) -> pd.DataFrame:
        """Get the discharge dataframe (date rows x gage columns).

        :returns: DataFrame of daily discharge indexed by date
        """
        return self.__discharge

    def _discharge_by_date(self) -> pd.DataFrame:
        """Return discharge as a date-indexed frame with one column per gage.

        Columns are ordered to match ``gage_ids`` and the index spans the full
        requested date range, so every day is present (NaN where missing).

        :returns: DataFrame indexed by date with one discharge column per gage
        """

        data = self.__discharge.reindex(index=self.__date_range,
                                        columns=self.__gageids)
        data.index.name = 'date'
        return data

    def _meta_value(self, gage_id: str, key: str, default):
        """Look up a metadata value for a gage, falling back to a default.

        :param gage_id: streamgage ID
        :param key: metadata key (e.g. ``latitude``)
        :param default: value to use when the gage or key is missing
        :returns: the metadata value or the default
        """

        value = self.__meta.get(gage_id, {}).get(key)
        if value is None:
            return default
        if isinstance(value, float) and np.isnan(value):
            return default
        return value

    def write_ascii(self, filename: Union[str, Path]):
        """Write streamgage observations to a file in PRMS format.

        :param filename: name of the file to create
        """

        data = self._discharge_by_date()

        # Prepend the date component columns in the order PRMS expects.
        out_order = list(self.__gageids)
        for cc in ['second', 'minute', 'hour', 'day', 'month', 'year']:
            out_order.insert(0, cc)

        data['year'] = data.index.year
        data['month'] = data.index.month
        data['day'] = data.index.day
        data['hour'] = data.index.hour
        data['minute'] = data.index.minute
        data['second'] = data.index.second
        data.fillna(-999, inplace=True)

        with open(filename, 'w') as outhdl:
            outhdl.write('Created by Bandit\n')
            outhdl.write('/////////////////////////////////////////////////////////////////////////\n')
            outhdl.write('// Station IDs for runoff:\n')
            outhdl.write('// ID\n')

            for gg in self.__gageids:
                outhdl.write(f'// {gg}\n')

            outhdl.write('/////////////////////////////////////////////////////////////////////////\n')
            outhdl.write('// Unit: runoff = cfs\n')
            outhdl.write('/////////////////////////////////////////////////////////////////////////\n')
            outhdl.write(f'runoff {len(self.__gageids)}\n')
            outhdl.write('#########################################################\n')

            data.to_csv(outhdl, sep=' ', columns=out_order, index=False, header=False)

        if self.__verbose:
            logger.info(f'Streamflow data written to: {filename}')

    def write_netcdf(self, filename: Union[str, Path]):
        """Write streamgage observations to a netCDF format file.

        :param filename: name of the netCDF file to create
        """

        data = self._discharge_by_date()

        poiname_list = [str(self._meta_value(gg, 'poi_name', '')) for gg in self.__gageids]

        max_poiid_len = len(max(self.__gageids, key=len))
        max_poiname_len = max(len(max(poiname_list, key=len)), 1)

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

        # Write the Streamgage IDs
        poiido[:] = nc.stringtochar(np.array(self.__gageids).astype('S'))

        timeo[:] = nc.date2num(pd.to_datetime(data.index).tolist(),
                               units=f'days since {reference_time}',
                               calendar=cal_type)

        # Discharge is poi_id (rows) x time (columns) in the netCDF layout.
        varo[:, :] = data.to_numpy(dtype=float).T

        poinameo[:] = nc.stringtochar(np.array(poiname_list).astype('S'))
        lato[:] = np.array([self._meta_value(gg, 'latitude', np.nan)
                            for gg in self.__gageids], dtype=float)
        lono[:] = np.array([self._meta_value(gg, 'longitude', np.nan)
                            for gg in self.__gageids], dtype=float)
        draino[:] = np.array([self._meta_value(gg, 'drainage_area', np.nan)
                              for gg in self.__gageids], dtype=float)
        draineffo[:] = np.array([self._meta_value(gg, 'drainage_area_contrib', np.nan)
                                 for gg in self.__gageids], dtype=float)
        nco.close()

        if self.__verbose:
            logger.info(f'Streamflow data written to: {filename}')
