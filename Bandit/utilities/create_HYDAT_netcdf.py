#!/usr/bin/env python3

import argparse
import calendar
import datetime
import netCDF4 as nc
import numpy as np
import pandas as pd
import sqlite3
import sys

from Bandit.bandit_helpers import set_date

__author__ = 'Parker Norton (pnorton@usgs.gov)'


def get_hydat_daily_streamflow(dbcon, stn_list, stdate, endate):
    flow_query = f'SELECT * FROM DLY_FLOWS'

    # Read daily flows from sqlite3 database into pandas
    df = pd.read_sql_query(flow_query, dbcon)

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Reformat the data into a more usable format
    data = {'poi_id': [],
            'date': [],
            'discharge': [],
            'flag': []}

    for idx, stn in enumerate(stn_list):
        sys.stdout.write(f'\r  {idx}: {stn}')
        sys.stdout.flush()

        df_tmp = df[df['STATION_NUMBER'] == stn]

        for xx in df_tmp.index:
            cyear = df_tmp['YEAR'][xx]
            cmonth = df_tmp['MONTH'][xx]

            for dd in range(calendar.monthrange(cyear, cmonth)[1]):
                data['poi_id'].append(df_tmp['STATION_NUMBER'][xx])
                data['date'].append(datetime.datetime(cyear, cmonth, dd+1))
                data['discharge'].append(df_tmp[f'FLOW{dd+1}'][xx])
                data['flag'].append(df_tmp[f'FLOW_SYMBOL{dd+1}'][xx])

        sys.stdout.write(f'\r               ')
    print('')

    # Create a dataframe from the results
    df2 = pd.DataFrame(data)

    # Convert the discharge from cms to cfs
    df2['discharge'] = df2['discharge'] * 35.3146667

    # Pivot the dataframe so site_no's are the columns
    df3 = df2.pivot(index='date', columns='poi_id', values='discharge')

    # Fill in missing days with NaN
    df4 = df3.resample('D').mean()
    df4 = df4[stdate:endate]
    return df4


def get_hydat_stations(dbcon):
    # Get stations information from HYDAT database
    stn_query = 'SELECT STATION_NUMBER, STATION_NAME, LATITUDE, ' + \
                'LONGITUDE, DRAINAGE_AREA_GROSS, DRAINAGE_AREA_EFFECT FROM STATIONS'
    df_stn = pd.read_sql_query(stn_query, dbcon)

    field_map = {'STATION_NUMBER': 'poi_id',
                 'STATION_NAME': 'poi_name',
                 'LATITUDE': 'latitude',
                 'LONGITUDE': 'longitude',
                 'DRAINAGE_AREA_GROSS': 'drainage_area',
                 'DRAINAGE_AREA_EFFECT': 'drainage_area_contrib'}

    df_stn.rename(columns=field_map, inplace=True)
    df_stn.set_index('poi_id', verify_integrity=True, inplace=True)

    # Convert the original area units of square kilometers to square miles
    df_stn.loc[:, 'drainage_area'] = df_stn.loc[:, 'drainage_area'].apply(lambda x: x * 0.386102)
    df_stn.loc[:, 'drainage_area_contrib'] = df_stn.loc[:, 'drainage_area_contrib'].apply(lambda x: x * 0.386102)
    df_stn.sort_index(inplace=True)
    return df_stn


def get_nhm_hydat_pois(filename):
    # Build ordered dictionary of geospatial fabric POIs
    col_names = ['GNIS_Name', 'Type_Gage', 'Type_Ref', 'Gage_Source', 'poi_segment_v1_1']
    col_types = [np.str_, np.str_, np.str_, np.str_, np.int]

    cols = dict(zip(col_names, col_types))

    df_poi = pd.read_csv(filename, sep='\t', dtype=cols, index_col=1)
    gf_pois_srcs = df_poi.loc[:, 'Gage_Source'].to_dict()
    ec_gages = []

    for gage_id, gage_src in gf_pois_srcs.items():
        if gage_src != 'EC':
            if gage_src == '0':
                print(f'{gage_id} in PARAMDB but has no gage_src')
            try:
                _ = int(gage_id)

                if len(gage_id) > 15:
                    print(f'{gage_id} is not a USGS gage')
            except ValueError:
                print(f'{gage_id} incorrectly sourced to {gage_src}')
        elif gage_src == 'EC':
            if len(gage_id) > 7:
                print(f'{gage_id} incorrectly sourced to {gage_src}')
            else:
                ec_gages.append(gage_id)
    ec_gages.sort()
    return ec_gages


def write_hydat_netcdf(df_stn, df_streamflow, filename):
    site_list = df_streamflow.columns.tolist()
    stdate = min(df_streamflow.index.tolist())

    poiname_list = df_stn['poi_name'].tolist()

    # max_poiid_len = len(max(site_list, key=len))
    max_poiid_len = 10
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

    poinameo = nco.createVariable('poi_name', 'S1', ('poi_id', 'poiname_nchars'), zlib=True)
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
    print(np.array(site_list).astype('S').shape)
    poiido[:] = nc.stringtochar(np.array(site_list, dtype='S10'))

    timeo[:] = nc.date2num(pd.to_datetime(df_streamflow.index).tolist(),
                           units=f'days since {reference_time}',
                           calendar=cal_type)

    # Write the station information
    poinameo[:] = nc.stringtochar(np.array(poiname_list).astype('S'))
    lato[:] = df_stn['latitude'].to_numpy(dtype=np.float)
    lono[:] = df_stn['longitude'].to_numpy(dtype=np.float)
    draino[:] = df_stn['drainage_area'].to_numpy(dtype=np.float)
    draineffo[:] = df_stn['drainage_area_contrib'].to_numpy(dtype=np.float)

    # Write the streamgage observations
    varo[:, :] = df_streamflow.to_numpy(dtype=np.float).T

    nco.close()


def main():
    parser = argparse.ArgumentParser(description='Setup new job for Bandit extraction')
    parser.add_argument('-d', '--dstdir', help='Destination directory for netCDF files')
    parser.add_argument('--daterange',
                        help='Starting and ending calendar date (YYYY-MM-DD YYYY-MM-DD)',
                        nargs=2, metavar=('startDate', 'endDate'), required=True)
    parser.add_argument('-p', '--poi', help='POI agency file derived from GF')
    parser.add_argument('-s', '--src', help='HYDAT sqlite3 database')

    args = parser.parse_args()

    # NOTE: POI agency file was created from the GF v1.1 with the following command:
    # ogr2ogr -f CSV poi_agency.csv GFv1.1.gdb -dialect sqlite \
    #          -sql "select GNIS_Name,Type_Gage,Type_Ref,Gage_Source,poi_segment_v1_1 from POIs_v1_1 where not Type_Gage = '0'"

    # Location of poi_agency file (derived from GF shapefile attributes)
    agency_filename = args.poi

    # netCDF output filename
    netcdf_filename = f'{args.dstdir}/HYDAT_pois.nc'

    # Location of the HYDAT sqlite3 database
    src_db = args.src

    st_date = set_date(args.daterange[0])
    en_date = set_date(args.daterange[1])

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Create list of HYDAT streamgages used in the NHM
    print('Get HYDAT streamgages that are used by the NHM')
    ec_gages = get_nhm_hydat_pois(agency_filename)

    # Open connection to HYDAT database
    connection = sqlite3.connect(src_db)

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Get streamgage information from HYDAT
    print('Build dataframe of station information')
    df_stn = get_hydat_stations(connection)

    # Restrict stations to those included in the NHM
    df_stn = df_stn[df_stn.index.isin(ec_gages)]

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Get daily streamflow from HYDAT database
    print('Build dataframe of daily streamflow')
    df_streamflow = get_hydat_daily_streamflow(connection, ec_gages,
                                               stdate=st_date, endate=en_date)

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Create a netcdf file containing the discharge values by site number
    print('Write HYDAT netcdf format file')
    write_hydat_netcdf(df_stn, df_streamflow, filename=netcdf_filename)

    print('Done.')


if __name__ == '__main__':
    main()
