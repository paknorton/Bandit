#!/usr/bin/env python3

import argparse
import datetime
import logging
import netCDF4 as nc
import numpy as np
import os
import pandas as pd
import platform
import re
import sys

from collections import OrderedDict
from io import StringIO
from typing import Dict, List, Optional, Tuple, Union

from urllib.request import urlopen  # , Request
from urllib.error import HTTPError

import Bandit.prms_nwis as prms_nwis
from Bandit.bandit_helpers import set_date

__version__ = 0.2
__author__ = 'Parker Norton (pnorton@usgs.gov)'

# URLs can be generated/tested at: http://waterservices.usgs.gov/rest/Site-Test-Tool.html
base_url = 'https://waterservices.usgs.gov/nwis'

t1 = re.compile('^#.*$\n?', re.MULTILINE)   # remove comment lines
t2 = re.compile('^5s.*$\n?', re.MULTILINE)  # remove field length lines

# Setup logging
pgm_log = logging.getLogger('create_NWIS_netcdf')
pgm_log.setLevel(logging.DEBUG)

log_fmt = logging.Formatter('%(levelname)s: %(name)s: %(message)s')

# Handler for file logs
flog = logging.FileHandler(f'create_NWIS_netcdf.log')
flog.setLevel(logging.DEBUG)
flog.setFormatter(log_fmt)

# Handler for console logs
clog = logging.StreamHandler()
clog.setLevel(logging.ERROR)
clog.setFormatter(log_fmt)

pgm_log.addHandler(flog)
pgm_log.addHandler(clog)


def read_csv(filename: str) -> List[str]:
    """Read paramdb CSV file and return list of data values

    :param filename: name of paramdb CSV file to read
    :returns: list of data values from file
    """
    # Opens a paramdb csv file and returns the second field
    fhdl = open(filename)
    rawdata = fhdl.read().splitlines()
    fhdl.close()
    it = iter(rawdata)

    data = []
    next(it)

    for lh in it:
        data.append(lh.split(',')[1])

    return data


def get_nhm_nwis_pois(filename: str) -> List[str]:
    """Get list of valid NWIS streamgages.

    :param filename: name of file to read
    :returns: list of NWIS streamgages
    """
    fhdl = open(filename, 'r')
    nhm_gages = []

    for row in fhdl:
        flds = row.strip().split(',')
        gage_id = flds[1]
        gage_src = flds[3]

        if gage_src != 'EC':
            try:
                _ = int(gage_id)

                if len(gage_id) > 15:
                    pgm_log.warning(f'{gage_id} is not a USGS gage')
                    print(f'{gage_id} is not a USGS gage')
                else:
                    nhm_gages.append(gage_id)
            except ValueError:
                if len(gage_id) == 7:
                    pgm_log.warning(f'  {gage_id} incorrectly sourced to {gage_src}')
                    print(f'  {gage_id} incorrectly sourced to {gage_src}')
        elif gage_src == 'EC':
            if len(gage_id) > 7:
                pgm_log.warning(f'{gage_id} incorrectly sourced to {gage_src}')
                print(f'  {gage_id} incorrectly sourced to {gage_src}')
                if len(gage_id) <= 15:
                    nhm_gages.append(gage_id)
        elif gage_src == '0':
            pgm_log.error(f'{gage_id} incorrectly sourced as 0')
            print(f'  {gage_id} incorrectly sourced as 0')
            nhm_gages.append(gage_id)

    nhm_gages.sort()
    return nhm_gages


def get_nwis_site_fields() -> Dict:
    """Get NWIS streamgage field information.

    :returns: dictionary of field_name, datatype pairs
    """
    # Retrieve a single station and pull out the field names and data types
    stn_url = f'{base_url}/site/?format=rdb&sites=01646500&siteOutput=expanded&' + \
              'siteStatus=active&parameterCd=00060&siteType=ST'

    response = urlopen(stn_url)
    encoding = response.info().get_param('charset', failobj='utf8')
    streamgage_site_page = response.read().decode(encoding)

    # Strip the comment lines and field length lines from the result
    streamgage_site_page = t1.sub('', streamgage_site_page, 0)

    # nwis_dtypes = t2.findall(streamgage_site_page)[0].strip('\n').split('\t')
    nwis_fields = StringIO(streamgage_site_page).getvalue().split('\n')[0].split('\t')

    nwis_final = {}
    for fld in nwis_fields:
        code = fld[-2:]
        if code in ['cd', 'no', 'nm', 'dt']:
            nwis_final[fld] = np.str_
        elif code in ['va']:
            nwis_final[fld] = np.float32
        else:
            nwis_final[fld] = np.str_

    return nwis_final


def _retrieve_from_nwis(url: str) -> str:
    """Get streamgage site page for given URL.

    :param url: URL to streamgage page on NWIS
    :returns: streamgage site page
    """
    response = urlopen(url)
    encoding = response.info().get_param('charset', failobj='utf8')
    streamgage_site_page = response.read().decode(encoding)

    # Strip the comment lines and field length lines from the result
    streamgage_site_page = t1.sub('', streamgage_site_page, 0)
    streamgage_site_page = t2.sub('', streamgage_site_page, 0)

    return streamgage_site_page


def get_nwis_sites(stdate: datetime.datetime,
                   endate: datetime.datetime,
                   sites: Optional[List, Tuple, str] = None,
                   regions: Optional[List, Tuple, str] = None) -> pd.DataFrame:
    """Get NWIS streamgage site information

    :param stdate: start date for extraction
    :param endate: end date for extraction
    :param sites: streamgage(s) to pull from NWIS
    :param regions: region(s) to pull from NWIS
    :returns: dataframe of streamgage site information
    """
    cols = get_nwis_site_fields()

    # Columns to include in the final dataframe
    include_cols = ['agency_cd', 'site_no', 'station_nm', 'dec_lat_va', 'dec_long_va', 'dec_coord_datum_cd',
                    'alt_va', 'alt_datum_cd', 'huc_cd', 'drain_area_va', 'contrib_drain_area_va']

    # Start with an empty dataframe
    nwis_sites = pd.DataFrame(columns=include_cols)

    url_pieces = OrderedDict()
    url_pieces['format'] = 'rdb'
    url_pieces['startDT'] = stdate.strftime('%Y-%m-%d')
    url_pieces['endDT'] = endate.strftime('%Y-%m-%d')
    # url_pieces['huc'] = None
    url_pieces['siteOutput'] = 'expanded'
    url_pieces['siteStatus'] = 'all'
    url_pieces['parameterCd'] = '00060'  # Discharge
    url_pieces['siteType'] = 'ST'
    # url_pieces['hasDataTypeCd'] = 'dv'

    # NOTE: If both sites and regions parameters are specified the sites
    #       parameter takes precedence.
    if sites is None:
        # No sites specified so default to HUC02-based retrieval
        url_pieces['huc'] = None

        if regions is None:
            # Default to HUC02 regions 1 thru 18
            regions = list(range(1, 19))
        if isinstance(regions, (list, tuple)):
            pass
        else:
            # Single region
            regions = [regions]
    else:
        # One or more sites were specified
        url_pieces['sites'] = None

        if isinstance(sites, (list, tuple)):
            pass
        else:
            # Single string, convert to list of sites
            sites = [sites]

    if 'huc' in url_pieces:
        # for region in range(19):
        for region in regions:
            sys.stdout.write(f'\r  Region: {region:02}')
            sys.stdout.flush()

            url_pieces['huc'] = f'{region:02}'
            url_final = '&'.join([f'{kk}={vv}' for kk, vv in url_pieces.items()])
            stn_url = f'{base_url}/site/?{url_final}'

            streamgage_site_page = _retrieve_from_nwis(stn_url)

            # Read the rdb file into a dataframe
            df = pd.read_csv(StringIO(streamgage_site_page), sep='\t', dtype=cols, usecols=include_cols)

            # pandas append deprecated since v1.4
            # nwis_sites = nwis_sites.append(df, ignore_index=True)
            nwis_sites = pd.concat([nwis_sites, df], ignore_index=True)
            sys.stdout.write('\r                      \r')
    else:
        for site in sites:
            sys.stdout.write(f'\r  Site: {site} ')
            sys.stdout.flush()

            url_pieces['sites'] = site
            url_final = '&'.join([f'{kk}={vv}' for kk, vv in url_pieces.items()])

            stn_url = f'{base_url}/site/?{url_final}'

            try:
                streamgage_site_page = _retrieve_from_nwis(stn_url)

                # Read the rdb file into a dataframe
                df = pd.read_csv(StringIO(streamgage_site_page), sep='\t', dtype=cols, usecols=include_cols)

                # pandas append deprecated since v1.4
                # nwis_sites = nwis_sites.append(df, ignore_index=True)
                nwis_sites = pd.concat([nwis_sites, df], ignore_index=True)
            except HTTPError as err:
                if err.code == 404:
                    sys.stdout.write(f'HTTPError: {err.code}, site does not meet criteria - SKIPPED\n')
            sys.stdout.write('\r                      \r')

    field_map = {'agency_cd': 'poi_agency',
                 'site_no': 'poi_id',
                 'station_nm': 'poi_name',
                 'dec_lat_va': 'latitude',
                 'dec_long_va': 'longitude',
                 'alt_va': 'elevation',
                 'drain_area_va': 'drainage_area',
                 'contrib_drain_area_va': 'drainage_area_contrib'}

    nwis_sites.rename(columns=field_map, inplace=True)
    nwis_sites.set_index('poi_id', inplace=True)
    nwis_sites = nwis_sites.sort_index()

    return nwis_sites


def write_nwis_netcdf(df_stn: pd.DataFrame,
                      df_streamflow: pd.DataFrame,
                      filename: str):
    """Write daily streamflow for select sites to netCDF file.

    :param df_stn: dataframe of streamgage information
    :param df_streamflow: dateframe of daily streamflow
    :param filename: name of netCDF file to write streamflow to
    """
    site_list = df_streamflow.columns.tolist()
    stdate = min(df_streamflow.index.tolist())

    poiname_list = df_stn['poi_name'].tolist()

    max_poiid_len = len(max(site_list, key=len))
    max_poiname_len = len(max(poiname_list, key=len))

    # Create a netCDF file for the CBH data
    nco = nc.Dataset(filename, 'w', clobber=True)

    # Create the dimensions
    nco.createDimension('poiid_nchars', max_poiid_len)
    nco.createDimension('poi_id', len(site_list))
    nco.createDimension('poiname_nchars', max_poiname_len)
    nco.createDimension('time', None)

    reference_time = stdate.strftime('%Y-%m-%d %H:%M:%S')
    cal_type = 'standard'

    # Create the variables
    timeo = nco.createVariable('time', 'f4', 'time')
    timeo.calendar = cal_type
    timeo.units = f'days since {reference_time}'

    poiido = nco.createVariable('poi_id', 'S1', ('poi_id', 'poiid_nchars'), zlib=True)
    poiido.long_name = 'Point-of-Interest ID'
    poiido.cf_role = 'timeseries_id'
    poiido._Encoding = 'ascii'

    poinameo = nco.createVariable('poi_name', 'S1', ('poi_id', 'poiname_nchars'),
                                  fill_value=nc.default_fillvals['S1'], zlib=True)
    poinameo.long_name = 'Name of POI station'
    poinameo._Encoding = 'ascii'

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

    nco.setncattr('Description', 'Streamflow data for PRMS')
    nco.setncattr('FeatureType', 'timeSeries')
    # nco.setncattr('Bandit_version', __version__)
    # nco.setncattr('NHM_version', nhmparamdb_revision)

    # Write the Streamgage IDs
    poiido[:] = nc.stringtochar(np.array(site_list).astype('S'))

    timeo[:] = nc.date2num(pd.to_datetime(df_streamflow.index).tolist(),
                           units=f'days since {reference_time}',
                           calendar=cal_type)

    # Write the station information
    poinameo[:] = nc.stringtochar(np.array(poiname_list).astype('S'))
    lato[:] = df_stn['latitude'].to_numpy(dtype=float)
    lono[:] = df_stn['longitude'].to_numpy(dtype=float)
    draino[:] = df_stn['drainage_area'].to_numpy(dtype=float)
    draineffo[:] = df_stn['drainage_area_contrib'].to_numpy(dtype=float)

    # Write the streamgage observations
    varo[:, :] = df_streamflow.to_numpy(dtype=float).T

    nco.close()


def main():
    parser = argparse.ArgumentParser(description='Setup new job for Bandit extraction')
    parser.add_argument('-d', '--dstdir', help='Destination directory for netCDF files')
    parser.add_argument('--daterange',
                        help='Starting and ending calendar date (YYYY-MM-DD YYYY-MM-DD)',
                        nargs=2, metavar=('startDate', 'endDate'), required=True)
    parser.add_argument('-p', '--poi', help='POI agency file derived from GF')

    args = parser.parse_args()

    # Location of poi_agency file (derived from GF shapefile attributes)
    poi_filename = args.poi

    # netCDF output filename
    netcdf_filename = f'{args.dstdir}/NWIS_pois.nc'

    st_date = set_date(args.daterange[0])
    en_date = set_date(args.daterange[1])

    # =========================================================================
    pgm_log.info(f'========== START {datetime.datetime.now().isoformat()} ==========')
    pgm_log.info(" ".join(sys.argv))
    pgm_log.info(f'Script version: {__version__}')
    pgm_log.info(f'Script directory: {os.path.dirname(os.path.abspath(__file__))}')
    pgm_log.info(f'Python: {platform.python_implementation()} ({platform.python_version()})')
    pgm_log.info(f'Host: {platform.node()}')
    pgm_log.info('-'*70)
    pgm_log.info(f'Current directory: {os.getcwd()}')
    pgm_log.info(f'Date range: {args.daterange[0]} to {args.daterange[1]}')
    pgm_log.info(f'poi_id source: {args.poi}')
    pgm_log.info(f'Destination dir: {args.dstdir}')
    # =========================================================================

    print('Get NWIS streamgages that are used by the NHM')
    # nhm_gages = get_nhm_nwis_pois(agency_filename)
    nhm_gages = read_csv(poi_filename)
    pgm_log.info(f'Number of POIs read from poi_id file: {len(nhm_gages)}')

    nhm_nwis_gages = []
    for xx in nhm_gages:
        if len(xx) != 7:
            nhm_nwis_gages.append(xx)

    pgm_log.info(f'Number of NWIS POIs in poi_id file: {len(nhm_nwis_gages)}')

    print('Reading station information')
    nwis_sites = get_nwis_sites(stdate=st_date, endate=en_date)
    # nwis_sites = get_nwis_sites(stdate=st_date, endate=en_date, regions=17)
    # nwis_sites = get_nwis_sites(stdate=st_date, endate=en_date, sites='06469400')

    # Reduce dataframe to sites in the NHM
    nwis_sites = nwis_sites[nwis_sites.index.isin(nhm_nwis_gages)]
    # nwis_daily[~nwis_daily['site_no'].isin(nwis_multiple)]

    site_list = nwis_sites.index.tolist()
    print(f'Number of matching NWIS streamgages: {len(site_list)}')
    pgm_log.info(f'Number of matching NWIS streamgages: {len(site_list)}')

    missing_sites = list(set(nhm_nwis_gages) - set(site_list))

    print(f'Sites not pulled: {missing_sites}')
    pgm_log.info(f'Sites not pulled: {missing_sites}')

    # Try to retrieve any missing sites
    if len(missing_sites) > 0:
        print('Trying to retrieve missing sites')
        addl_sites = get_nwis_sites(stdate=st_date, endate=en_date, sites=missing_sites)
        nwis_sites = pd.concat([nwis_sites, addl_sites])

    # Reduce dataframe to sites in the NHM
    nwis_sites = nwis_sites[nwis_sites.index.isin(nhm_nwis_gages)]
    # nwis_daily[~nwis_daily['site_no'].isin(nwis_multiple)]

    site_list = nwis_sites.index.tolist()
    print(f'Number of matching NWIS streamgages and addl sites: {len(site_list)}')
    pgm_log.info(f'Number of matching NWIS streamgages and addl: {len(site_list)}')

    missing_sites = list(set(nhm_nwis_gages) - set(site_list))
    if len(missing_sites) > 0:
        print(f'ERROR: Missing sites from NWIS, {missing_sites}')
        exit(200)

    print('Retrieving daily streamflow observations')
    streamflow = prms_nwis.NWIS(gage_ids=site_list, st_date=st_date, en_date=en_date,
                                verbose=True)
    streamflow.get_daily_streamgage_observations()

    print('Writing streamflow to netcdf')
    write_nwis_netcdf(nwis_sites, streamflow.data, filename=netcdf_filename)


if __name__ == '__main__':
    main()
