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

from typing import Dict, List, Optional
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import numpy as np
import pandas as pd  # type: ignore

logger = logging.getLogger(__name__)

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
