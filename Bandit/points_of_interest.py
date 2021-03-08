# from abc import ABCMeta, abstractmethod
# from typing import Callable

import logging
import netCDF4 as nc
import numpy as np
import pandas as pd
# import re
import sys
import xarray as xr

from datetime import datetime

from Bandit.bandit_helpers import set_date
# from Bandit.pr_util import print_error


class POI:
    """Class for accessing point-of-interest observations."""

    def __init__(self, src_path=None, gage_ids=None, st_date=None, en_date=None, verbose=False):
        """Create the POI object.

        :param list[str] gage_ids: list of streamgages to retrieve
        :param st_date: start date for retrieving streamgage observations
        :type st_date: None or datetime
        :param en_date: end date for retrieving streamgage observations
        :type en_date: None or datetime
        :param bool verbose: output additional debuggin information
        """

        self.logger = logging.getLogger('bandit.NWIS')
        self.logger.info('NWIS instance')

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

        self.thredds_server = 'http://gdp-netcdfdev.cr.usgs.gov:8080'
        self.base_opendap = f'{self.thredds_server}/thredds/dodsC/NHM_POIS/files'

        # base_url is used to get a list of files for a product
        # Until xarray supports ncml parsing the list of files will have to be manually built
        self.base_url = f'{self.thredds_server}/thredds/catalog/NHM_POIS/files/catalog.html'

        self.read_thredds()

    @property
    def data(self):
        return self.__outdata

    @property
    def start_date(self):
        """Get the start date.

        :returns: start date
        :rtype: None or datetime
        """

        return self.__stdate

    @start_date.setter
    def start_date(self, st_date):
        """Set the start date.

        :param st_date: start date (either a datetime object or a string of the form YYYY-MM-DD)
        :type st_date: datetime or str
        """

        # Set the starting date for retrieval
        # As written this will clear any streamgage observations that have been downloaded.
        self.__stdate = set_date(st_date)
        self.__outdata = None

    @property
    def end_date(self):
        """Get the end date.

        :returns: end date
        :rtype: None or datetime
        """

        return self.__endate

    @end_date.setter
    def end_date(self, en_date):
        """Set the end date.

        :param en_date: end date (either a datetime object or a string of the form YYYY-MM-DD)
        :type en_date: datetime or str
        """
        self.__endate = set_date(en_date)
        self.__outdata = None

    @property
    def gage_ids(self):
        """Get list of streamgage IDs for retrieval.

        :returns: list of streamgage IDs
        :rtype: list[str]
        """

        return self.__gageids

    @gage_ids.setter
    def gage_ids(self, gage_ids):
        """Set the streamgage ID(s) to retrieve from NWIS.

        :param gage_ids: streamgage ID(s)
        :type gage_ids: list or tuple or str
        """

        # Set the gage ids for retrieval this will clear any downloaded observations
        if isinstance(gage_ids, (list, tuple)):
            self.__gageids = gage_ids
        else:
            # Assuming a single value, so convert to a list
            self.__gageids = [gage_ids]
        self.__outdata = None

    def get(self, var: str):
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

    def read(self):
        """Read POI files stored in netCDF format"""

        if self.__gageids:
            # print('\t\tOpen dataset')
            self.__outdata = xr.open_mfdataset(self.__src_path,
                                               chunks={'poi_id': 1040}, combine='nested',
                                               concat_dim='poi_id', decode_cf=True,
                                               engine='netcdf4')
            # NOTE: With a multi-file dataset the time attributes 'units' and
            #       'calendar' are lost.
            #       see https://github.com/pydata/xarray/issues/2436
        else:
            self.logger.warning('No poi_ids were specified.')

    def read_thredds(self):
        """Read POI files stored in netCDF format"""

        if self.__gageids:
            # print('\t\tOpen dataset')
            full_file_list = pd.read_html(self.base_url, skiprows=1)[0]['Files']

            # Only include files ending in .nc (sometimes the .ncml files are included and we don't want those)
            flist = full_file_list[full_file_list.str.match('.*nc$')].tolist()

            # Create list of file URLs
            # The list looks something like:
            # ['http://gdp-netcdfdev.cr.usgs.gov:8080/thredds/dodsC/NHM_NWIS/files/NWIS_pois.nc',
            #  'http://gdp-netcdfdev.cr.usgs.gov:8080/thredds/dodsC/NHM_NWIS/files/HYDAT_pois.nc']
            xfiles = [f'{self.base_opendap}/{xx}' for xx in flist]

            self.__outdata = xr.open_mfdataset(xfiles,
                                               chunks={'poi_id': 1040}, combine='nested',
                                               concat_dim='poi_id', decode_cf=True,
                                               engine='netcdf4')
            # NOTE: With a multi-file dataset the time attributes 'units' and
            #       'calendar' are lost.
            #       see https://github.com/pydata/xarray/issues/2436
        else:
            self.logger.warning('No poi_ids were specified.')

    def write_ascii(self, filename):
        """Writes POI observations to a file in PRMS format.

        :param str filename: name of the file to create
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

    def write_netcdf(self, filename):
        """Write POI streamflow to netcdf format file"""

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
        varo[:, :] = data.to_numpy(dtype=np.float)

        poinameo[:] = nc.stringtochar(np.array(poiname_list).astype('S'))
        lato[:] = self.get('latitude').to_numpy(dtype=np.float)
        lono[:] = self.get('longitude').to_numpy(dtype=np.float)
        draino[:] = self.get('drainage_area').to_numpy(dtype=np.float)
        draineffo[:] = self.get('drainage_area_contrib').to_numpy(dtype=np.float)
        nco.close()


# class DataSourceFactory:
#     registry = {}
#
#     def register(self, name, connector):
#         self.registry[name] = connector
#         # print(connector)
#
#     def create_service(self, name: str, **kwargs):
#         svc_class = self.registry.get(name)
#
#         if not svc_class:
#             raise ValueError(name)
#
#         # print(f'DataFactory.create_service(): {name} service retrieved')
#         # print(svc_class)
#         return svc_class(**kwargs)
#
#
# class ServiceBase(metaclass=ABCMeta):
#     """Base class for datasource services"""
#     def __init__(self, **kwargs):
#         pass
#
#     @abstractmethod
#     def get_obs(self):
#         # print('Getting some data')
#         pass
#
#
# class ServiceHYDAT(ServiceBase):
#     """Service class for accessing streamflow originally from the HYDAT database"""
#
#     def __init__(self, db_hdl, **kwargs):
#         super().__init__(**kwargs)
#         self._db_hdl = db_hdl
#         print(f'ServiceHYDAT.__init__(): dbl_hdl = {self._db_hdl}')
#
#     def get_obs(self):
#         print('Get HYDAT data')
#
#
# class ServiceHYDATConnector:
#     """Connector class for a persistent connection to a datasource"""
#     def __init__(self, **kwargs):
#         # print('ServiceHYDAT_connector.__init__()')
#         self._instance = None
#
#     def __call__(self, HYDAT_db_filename, **_ignored):
#         # print('ServiceHYDAT_connector.__call__()')
#         if not self._instance:
#             # print('    *new* ServiceHYDAT instance')
#             db_hdl = self.connect(HYDAT_db_filename)
#             self._instance = ServiceHYDAT(db_hdl)
#         return self._instance
#
#     def connect(self, db_filename):
#         # This would connect to the sqlite3 db and return the handle
#         # print(f'Connecting to {db_filename}')
#         return 'mydb_hdl'
#
#
# class POI(object):
#     """Points of Interest class"""
#     def __init__(self, poi_ids=None, st_date=None, en_date=None, verbose=False):
#
#         # Create ordereddict of poi_id -> plugin (e.g. NWIS, HYDAT)
#         # Instantiate datasource plugins
#         #   ? read from a plugins directory?
#
#         # Iterate over poi_ids
#         #   - add current poi_id to ordereddict pointing to matching object
#         #   * could just iterate and call the object functions directly
#
#         # Provides lookup for poi agency and other variables
#         self.__poi_info = None
#
#         self.__stdate = st_date
#         self.__endate = en_date
#         self.__poi_ids = poi_ids
#         self.__verbose = verbose
#
#     def read(self):
#         """Read the observations for all POIs"""
#         pass
#
#     def write(self):
#         """Write POI observations to a file"""
#
#         # Can write to ASCII, netCDF, others?
#         pass
#
#
# class DataDriver(object):
#     """Abstract class for datasources"""
#     def __init__(self):
#         self.__driver_name = 'Base class'
#         pass
#
#     def read(self):
#         """Abstract read method"""
#         assert False, 'DataDriver.read() must be defined by child class'
#
#     def write(self):
#         """Method to write data to output file"""
#
#
# class DataDriverNWIS(DataDriver):
#     """Driver for accessing NWIS REST service"""
#     def __init__(self):
#         pass
#
#     def read(self):
#         # Read streamflow from NWIS
#         pass
#
#
# class DataDriverHYDAT(DataDriver):
#     """Driver for accessing HYDAT streamflow"""
#     def __init__(self):
#         # Init connection to HYDAT database
#
#         datasource = 'HYDAT.sqlite3'
#
#         field_map = {}
#         pass
#
#     def read(self):
#         # Read streamflow from HYDAT database
#         pass
