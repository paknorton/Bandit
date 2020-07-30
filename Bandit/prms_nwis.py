#!/usr/bin/env python3

from collections import OrderedDict
from datetime import datetime

import logging
import netCDF4 as nc
import numpy as np
import pandas as pd
import re
import socket
import sys
import time

from typing import Union, Dict, List, OrderedDict as OrderedDictType, Sequence

from Bandit.pr_util import print_error

from io import StringIO
from urllib.request import urlopen
from urllib.error import HTTPError, URLError
from html.parser import HTMLParser


class NWISErrorParser(HTMLParser):
    # This is a quick 'n dirty error message parser for NWIS
    inBody = False
    inPara = False
    inBold = False
    lsStartTags = list()
    lsEndTags = list()
    lsStartEndTags = list()
    lsComments = list()
    curr_key = None
    error_info = {}

    # HTML Parser Methods
    def handle_starttag(self, start_tag, attrs):
        if self.inBody and start_tag != 'h1':
            self.lsStartTags.append(start_tag)
            # print('Start tag: ', start_tag)

        if start_tag == 'body':
            self.inBody = True

        if start_tag == 'p':
            self.inPara = True

        if start_tag == 'b':
            self.inBold = True

    def handle_endtag(self, end_tag):
        if end_tag == 'body':
            self.inBody = False

        if self.inBody and end_tag != 'h1':
            self.lsEndTags.append(end_tag)
            # print('End tag: ', end_tag)

        if end_tag == 'p':
            self.inPara = False
        if end_tag == 'b':
            self.inBold = False

    def handle_startendtag(self, startend_tag, attrs):
        self.lsStartEndTags.append(startend_tag)

    def handle_data(self, data):
        if self.inBody:
            # print("Encountered some data  :", data)

            if self.inPara and self.inBold:
                self.curr_key = data
            elif self.inPara:
                if self.curr_key == 'message':
                    self.error_info[self.curr_key] = data.split(',')[0]
                else:
                    self.error_info[self.curr_key] = data

    def handle_comment(self, data):
        self.lsComments.append(data)


# URLs can be generated/tested at: http://waterservices.usgs.gov/rest/Site-Test-Tool.html
BASE_NWIS_URL = 'https://waterservices.usgs.gov/nwis'
RETRIES = 3

nwis_logger = logging.getLogger('bandit.NWIS')


class NWIS:

    """Class for accessing and manipulating streamflow information from the
    National Water Information System (NWIS; https://waterdata.usgs.gov/) provided by the
    United States Geological Survey (USGS; https://www.usgs.gov/).
    """

    # As written this class provides fucntions for downloading daily streamgage observations
    # Additional functionality (e.g. monthyly, annual, other statistics) may be added at a future time.

    def __init__(self, gage_ids=None, st_date=None, en_date=None, verbose=False):
        """Create the NWIS object.

        :param list[str] gage_ids: list of streamgages to retrieve
        :param st_date: start date for retrieving streamgage observations
        :type st_date: None or datetime
        :param en_date: end date for retrieving streamgage observations
        :type en_date: None or datetime
        :param bool verbose: output additional debuggin information
        """

        self.logger = logging.getLogger('bandit.NWIS')
        self.logger.info('NWIS instance')

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

        # Regex's for stripping unneeded clutter from the rdb file
        self.__t1 = re.compile('^#.*$\n?', re.MULTILINE)  # remove comment lines
        self.__t2 = re.compile('^5s.*$\n?', re.MULTILINE)  # remove field length lines

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
        if isinstance(st_date, datetime):
            self.__stdate = st_date
        else:
            try:
                # Assume a string of form 'YYYY-MM-DD' was provided
                self.__stdate = datetime(*[int(xx) for xx in re.split('-| |:', st_date)])
            except ValueError as dt_err:
                # Wrong form of date was provided
                print_error('Date must be either a datetime or of form "YYYY-MM-DD"')
                print(dt_err)
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

        if isinstance(en_date, datetime):
            self.__endate = en_date
        else:
            try:
                # Assume a string of form 'YYYY-MM-DD' was provided
                self.__endate = datetime(*[int(xx) for xx in re.split('-| |:', en_date)])
            except ValueError as dt_err:
                # Wrong form of date was provided
                print_error('Date must be either a datetime or of form "YYYY-MM-DD"')
                print(dt_err)
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

    def check_for_flag(self, pat, data, col_id):
        """Check for a given pattern in supplied data.

        Checks for a given pattern in the data and log a warning if it occurs.

        :param str pat: pattern to find
        :param data: data for pattern matching
        :param str col_id: column to search in data
        """

        # Check for pattern in data, log error if found and remove from data
        pat_count = data[col_id].str.contains(pat).sum()

        if pat_count > 0:
            pat_first_date = data[data[col_id].str.contains(pat)].index[0].strftime('%Y-%m-%d')

            self.logger.warning(f'{col_id} has {pat_count} records marked {pat}. ' +
                                f'First occurrence at {pat_first_date}. Suffix removed from values')
            data[col_id].replace(pat, '', regex=True, inplace=True)

    def initialize_dataframe(self):
        """Clears downloaded data and initializes the output dataframe.
        """

        if not self.__endate:
            self.__endate = datetime.today()
        if not self.__stdate:
            self.__stdate = datetime(1890, 1, 1)

        # Create an initial dataframe that contains all dates in the date range.
        # Any streamgage missing date(s) will have a NaN value for each missing date.
        # Otherwise it is possible to have dates missing in the output.
        self.__date_range = pd.date_range(start=self.__stdate, end=self.__endate, freq='D')
        self.__outdata = pd.DataFrame(index=self.__date_range)
        self.__final_outorder = ['year', 'month', 'day', 'hour', 'minute', 'second']

    def get_daily_streamgage_observations(self):
        """Retrieves daily observations.

        If gage_ids is set then retrieve observations for those streamgages; otherwise,
        return a single dummy dataset. If st_date and en_date are set then observations
        are restricted to the given date range.
        """

        if not self.__outdata:
            self.initialize_dataframe()

        # Set timeout in seconds - if not set defaults to infinite time for response
        timeout = 30
        socket.setdefaulttimeout(timeout)

        url_pieces = OrderedDict()
        url_pieces['?format'] = 'rdb'
        url_pieces['sites'] = ''
        url_pieces['startDT'] = self.__stdate.strftime('%Y-%m-%d')
        url_pieces['endDT'] = self.__endate.strftime('%Y-%m-%d')
        url_pieces['statCd'] = '00003'  # Mean values
        url_pieces['siteStatus'] = 'all'
        url_pieces['parameterCd'] = '00060'  # Discharge
        url_pieces['siteType'] = 'ST'
        # url_pieces['access'] = '3'  # Allows download of observations for restricted sites/parameters

        if not self.__gageids:
            # If no streamgages are provided then create a single dummy column filled with noData
            self.logger.warning('No streamgages provided - dummy entry created.')
            df = pd.DataFrame(index=self.__date_range, columns=['00000000'])
            df.index.name = 'date'

            self.__outdata = pd.merge(self.__outdata, df, how='left', left_index=True, right_index=True)
            self.__final_outorder.append('00000000')

        # Iterate over new_poi_gage_id and retrieve daily streamflow data from NWIS
        for gidx, gg in enumerate(self.__gageids):
            if self.__verbose:
                sys.stdout.write(f'\rStreamgage: {gg} ({gidx + 1}/{len(self.__gageids)}) ')
                sys.stdout.flush()

            url_pieces['sites'] = gg
            url_final = '&'.join([f'{kk}={vv}' for kk, vv in url_pieces.items()])

            # Read site data from NWIS
            streamgage_obs_page = None
            attempts = 0
            while attempts < RETRIES:
                try:
                    response = urlopen(f'{BASE_NWIS_URL}/dv/{url_final}')
                    encoding = response.info().get_param('charset', failobj='utf8')
                    streamgage_obs_page = response.read().decode(encoding)

                    break
                except (HTTPError, URLError) as err:
                    if err.code == 400:
                        err_parser = NWISErrorParser()
                        err_parser.feed(str(err.read().decode("utf8", 'ignore')))
                        self.logger.warning(f'HTTPError: {err.code}, Site: {gg}, {err_parser.error_info["message"]}')

                        break
                    else:
                        attempts += 1
                        self.logger.warning(f'HTTPError: {err}, Try {attempts} of {RETRIES}')
                        # print('HTTPError: {}, Try {} of {}'.format(err, attempts, RETRIES))
                except ConnectionResetError as err:
                    attempts += 1
                    self.logger.warning(f'ConnectionResetError: {err}, Try {attempts} of {RETRIES}')
                    time.sleep(10)

            if streamgage_obs_page is None:
                # Create a dummy dataframe
                df = pd.DataFrame(index=self.__date_range, columns=[gg])
                df.index.name = 'date'
            elif streamgage_obs_page.splitlines()[0] == '#  No sites found matching all criteria':
                # No observations are available for the streamgage
                # Create a dummy dataset to output
                self.logger.warning(f'{gg} has no data for ' + self.__stdate.strftime('%Y-%m-%d') +
                                    ' to ' + self.__endate.strftime('%Y-%m-%d'))

                df = pd.DataFrame(index=self.__date_range, columns=[gg])
                df.index.name = 'date'
            else:
                streamgage_observations = streamgage_obs_page
                # streamgage_observations = streamgage_obs_page.read()

                # Strip the comment lines and field length lines from the result using regex
                streamgage_observations = self.__t1.sub('', streamgage_observations, count=0)
                streamgage_observations = self.__t2.sub('', streamgage_observations, count=0)

                # Have to enforce site_no as string/text
                col_names = ['site_no']
                col_types = [np.str_]
                cols = dict(zip(col_names, col_types))

                # Read the rdb file into a dataframe
                # TODO: Handle empty datasets from NWIS by creating dummy data and providing a warning
                df = pd.read_csv(StringIO(streamgage_observations), sep='\t', dtype=cols,
                                 parse_dates={'date': ['datetime']}, index_col='date')

                # Conveniently the columns we want to drop contain '_cd' in their names
                drop_cols = [col for col in df.columns if '_cd' in col]
                drop_cols.append('site_no')
                df.drop(drop_cols, axis=1, inplace=True)

                # There should now only be date, site_no, and a Q column named *_00060_00003
                # We will rename the *_00060_00003 to the site_no value
                rename_col = [col for col in df.columns if '_00060_00003' in col]

                if len(rename_col) > 1:
                    self.logger.warning(f'{gg} had more than one Q-col returned; empty dataset used.')
                    df = pd.DataFrame(index=self.__date_range, columns=[gg])
                    df.index.name = 'date'

                    # self.logger.warning('{} had more than one Q-col returned; using {}'.format(gg, rename_col[0]))
                    #
                    # # Keep the first TS column and drop the others
                    # while len(rename_col) > 1:
                    #     curr_col = rename_col.pop()
                    #     df.drop([curr_col], axis=1, inplace=True)
                else:
                    df.rename(columns={rename_col[0]: gg}, inplace=True)

                    try:
                        # If no flags are present the column should already be float
                        df[gg] = pd.to_numeric(df[gg], errors='raise', downcast='float')
                    except ValueError:
                        self.logger.warning(f'{gg} had one or more flagged values; flagged values converted to NaN.')
                        df[gg] = pd.to_numeric(df[gg], errors='coerce', downcast='float')

                    # Check for discontinued gage records
                    # if df[gg].dtype == np.object_:
                    #     # If the datatype of the streamgage values is np.object_ that
                    #     # means some string is appended to one or more of the values.
                    #
                    #     # Common bad data: set(['Eqp', 'Ice', 'Ssn', 'Rat', 'Bkw', '***', 'Dis'])
                    #
                    #     # Check for discontinued flagged records
                    #     self.check_for_flag('_?Dis', df, gg)
                    #
                    #     # Check for ice-flagged records
                    #     self.check_for_flag('_?Ice', df, gg)
                    #
                    #     # Check for eqp-flagged records (Equipment malfunction)
                    #     self.check_for_flag('_?Eqp', df, gg)
                    #
                    #     # Check for _Ssn (parameter monitored seasonally)
                    #     self.check_for_flag('_?Ssn', df, gg)
                    #
                    #     # Check for _Rat (rating being developed)
                    #     self.check_for_flag('_?Rat', df, gg)
                    #
                    #     # Check for _Bkw (Value is affected by backwater at the measurement site)
                    #     self.check_for_flag('_?Bkw', df, gg)
                    #
                    #     # Check for 1 or more astericks
                    #     self.check_for_flag('_?\*+', df, gg)

                    # Resample to daily to fill in the missing days with NaN
                    # df = df.resample('D').mean()

            self.__outdata = pd.merge(self.__outdata, df, how='left', left_index=True, right_index=True)
            self.__final_outorder.append(gg)
            sys.stdout.write('\r                                       \r')

    def write_ascii(self, filename):
        """Writes streamgage observations to a file in PRMS format.

        :param str filename: name of the file to create
        """

        # Create the year, month, day, hour, minute, second columns
        try:
            self.__outdata['year'] = self.__outdata.index.year
            self.__outdata['month'] = self.__outdata.index.month
            self.__outdata['day'] = self.__outdata.index.day
            self.__outdata['hour'] = self.__outdata.index.hour
            self.__outdata['minute'] = self.__outdata.index.minute
            self.__outdata['second'] = self.__outdata.index.second
            self.__outdata.fillna(-999, inplace=True)

            # TODO: 2019-04-03 PAN - Make sure all streamflow data is of type float
        except AttributeError:
            print('AttributeError')
            print(self.__outdata.head())
            print(self.__outdata.info())

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

        self.__outdata.to_csv(outhdl, sep=' ', columns=self.__final_outorder, index=False, header=False)
        outhdl.close()

        if self.__verbose:
            sys.stdout.write('\r                                       ')
            sys.stdout.write(f'\r\tStreamflow data written to: {filename}\n')
            sys.stdout.flush()

    def write_netcdf(self, filename):
        """Write NWIS streamflow to netcdf format file"""

        max_gageid_len = len(max(self.__gageids, key=len))

        # Create a netCDF file for the CBH data
        nco = nc.Dataset(filename, 'w', clobber=True)

        # Create the dimensions
        nco.createDimension('gageid_nchars', max_gageid_len)
        nco.createDimension('gageid', len(self.__gageids))
        nco.createDimension('time', None)

        reference_time = self.__stdate.strftime('%Y-%m-%d %H:%M:%S')
        cal_type = 'standard'

        # Create the variables
        timeo = nco.createVariable('time', 'f4', ('time'))
        timeo.calendar = cal_type
        timeo.units = f'days since {reference_time}'

        gageido = nco.createVariable('gageid', 'S1', ('gageid', 'gageid_nchars'), zlib=True)
        gageido.long_name = 'Streamgage ID'
        gageido.cf_role = 'timeseries_id'

        varo = nco.createVariable('discharge', 'f4', ('gageid', 'time'), fill_value=-999.0, zlib=True)
        varo.long_name = 'discharge'
        varo.units = 'cfs'

        nco.setncattr('Description', 'Streamflow data for PRMS')
        nco.setncattr('FeatureType', 'timeSeries')
        # nco.setncattr('Bandit_version', __version__)
        # nco.setncattr('NHM_version', nhmparamdb_revision)

        # Write the Streamgage IDs
        gageido[:] = nc.stringtochar(np.array(self.__gageids).astype('S'))

        timeo[:] = nc.date2num(pd.to_datetime(self.__outdata.index).tolist(),
                               units=f'days since {reference_time}',
                               calendar=cal_type)

        # Write the streamgage observations
        varo[:, :] = self.__outdata.to_numpy(dtype=np.float).T

        nco.close()
