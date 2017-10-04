#!/usr/bin/env python

from __future__ import (absolute_import, division, print_function)
# , unicode_literals)
from future.utils import iteritems

from collections import OrderedDict
from datetime import datetime

import logging
import numpy as np
import pandas as pd
import re
import sys

from Bandit.pr_util import print_warning, print_error

try:
    # Python 2.x
    from StringIO import StringIO
except ImportError:
    # Python 3.x
    from io import StringIO

try:
    # Try importing assuming Python 3.x first
    # from urllib.parse import urlparse, urlencode
    from urllib.request import urlopen, Request
    from urllib.error import HTTPError
except ImportError:
    # Otherwise fallback to Python 2.x
    # from urlparse import urlparse
    # from urllib import urlencode
    from urllib2 import urlopen, Request, HTTPError, URLError

# URLs can be generated/tested at: http://waterservices.usgs.gov/rest/Site-Test-Tool.html
BASE_NWIS_URL = 'http://waterservices.usgs.gov/nwis'
RETRIES = 3

nwis_logger = logging.getLogger('bandit.NWIS')


class NWIS(object):
    """Class for accessing and manipulating streamflow information from the
    National Water Information System (NWIS; https://waterdata.usgs.gov/) provided by the
    United States Geological Survey (USGS; https://www.usgs.gov/).

    """

    # Class for NWIS streamgage observations
    # As written this class provides fucntions for downloading daily streamgage observations
    # Additional functionality (e.g. monthyly, annual, other statistics) may be added at a future time.

    def __init__(self, gage_ids=None, st_date=None, en_date=None):
        """Init method for NWIS class.

        Args:
            gage_ids (:obj:`list` of :obj:`str`): String or list of strings of streamgage IDs.
            st_date (:obj:`datetime`, optional: Starting date for date range to retrieve streamgage observations.
            en_date (:obj:`datetime`, optional: Ending date for date range to retrieve streamgage observations.
        """

        self.logger = logging.getLogger('bandit.NWIS')
        self.logger.info('NWIS instance')

        self.__stdate = st_date
        self.__endate = en_date
        self.__gageids = gage_ids
        self.__outdata = None
        self.__date_range = None
        self.__final_outorder = None

        # Regex's for stripping unneeded clutter from the rdb file
        self.__t1 = re.compile('^#.*$\n?', re.MULTILINE)  # remove comment lines
        self.__t2 = re.compile('^5s.*$\n?', re.MULTILINE)  # remove field length lines

    @property
    def start_date(self):
        """:obj:`datetime`: The starting date of a date range for retrieving streamflow observations.

        """

        return self.__stdate

    @start_date.setter
    def start_date(self, st_date):
        # Set the starting date for retrieval
        # As written this will clear any streamgage observations that have been downloaded.
        if isinstance(st_date, datetime.datetime):
            self.__stdate = st_date
        else:
            try:
                # Assume a string of form 'YYYY-MM-DD' was provided
                self.__stdate = datetime(*[int(xx) for xx in re.split('-| |:', st_date)])
            except ValueError as dt_err:
                # Wrong form for date was provided
                print_error('Date must be either a datetime or of form "YYYY-MM-DD"')
                print(dt_err)
        self.__outdata = None

    @property
    def end_date(self):
        """:obj:`datetime`: The ending date of a date range for retrieving streamflow observations.

        """

        return self.__endate

    @end_date.setter
    def end_date(self, en_date):
        if isinstance(en_date, datetime.datetime):
            self.__endate = en_date
        else:
            try:
                # Assume a string of form 'YYYY-MM-DD' was provided
                self.__endate = datetime(*[int(xx) for xx in re.split('-| |:', en_date)])
            except ValueError as dt_err:
                # Wrong form for date was provided
                print_error('Date must be either a datetime or of form "YYYY-MM-DD"')
                print(dt_err)
        self.__outdata = None

    @property
    def gage_ids(self):
        """:obj:`list` of :obj:`str`: Streamgage IDs for retrieval.

        """

        return self.__gageids

    @gage_ids.setter
    def gage_ids(self, gage_ids):
        # Set the gage ids for retrieval this will clear any downloaded observations
        if isinstance(gage_ids, (list, tuple)):
            self.__gageids = gage_ids
        else:
            # Assuming a single value, so convert to a list
            self.__gageids = [gage_ids]
            print(self.__gageids)
        self.__outdata = None

    def initialize_dataframe(self):
        """Clears downloaded data and initializes the output dataframe"""
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
        """Retrieves daily observations for a given date range and set of streamgage IDs"""
        if not self.__outdata:
            if not self.__gageids:
                print_error('No streamgages have been specified')
                return
            self.initialize_dataframe()

        url_pieces = OrderedDict()
        url_pieces['?format'] = 'rdb'
        url_pieces['sites'] = ''
        url_pieces['startDT'] = self.__stdate.strftime('%Y-%m-%d')
        url_pieces['endDT'] = self.__endate.strftime('%Y-%m-%d')
        url_pieces['statCd'] = '00003'  # Mean values
        url_pieces['siteStatus'] = 'all'
        url_pieces['parameterCd'] = '00060'  # Discharge
        url_pieces['siteType'] = 'ST'
        url_pieces['access'] = '3'  # Allows download of observations for restricted sites/parameters

        if not self.__gageids:
            # If do streamgages are provided then create a single dummy column filled with noData
            self.logger.warning('No streamgages provided - dummy entry created.')
            df = pd.DataFrame(index=self.__date_range, columns=['00000000'])
            df.index.name = 'date'

            self.__outdata = pd.merge(self.__outdata, df, how='left', left_index=True, right_index=True)
            self.__final_outorder.append('00000000')

        # Iterate over new_poi_gage_id and retrieve daily streamflow data from NWIS
        for gidx, gg in enumerate(self.__gageids):
            sys.stdout.write('\r                                       ')
            sys.stdout.write('\rStreamgage: {} ({}/{}) '.format(gg, gidx + 1, len(self.__gageids)))
            sys.stdout.flush()

            url_pieces['sites'] = gg
            url_final = '&'.join(['{}={}'.format(kk, vv) for kk, vv in iteritems(url_pieces)])

            # Read site data from NWIS
            attempts = 0
            while attempts < RETRIES:
                try:
                    streamgage_obs_page = urlopen('{}/dv/{}'.format(BASE_NWIS_URL, url_final))
                    break
                except (HTTPError, URLError) as err:
                    attempts += 1
                    self.logger.warning('HTTPError: {}, Try {} of {}'.format(err, attempts, RETRIES))
                    # print('HTTPError: {}, Try {} of {}'.format(err, attempts, RETRIES))

            if streamgage_obs_page.readline().strip() == '#  No sites found matching all criteria':
                # No observations are available for the streamgage
                # Create a dummy dataset to output
                self.logger.warning('{} has no data for {} to {}'.format(gg,
                                                                         self.__stdate.strftime('%Y-%m-%d'),
                                                                         self.__endate.strftime('%Y-%m-%d')))

                df = pd.DataFrame(index=self.__date_range, columns=[gg])
                df.index.name = 'date'
            else:
                streamgage_observations = streamgage_obs_page.read()

                # Strip the comment lines and field length lines from the result using regex
                streamgage_observations = self.__t1.sub('', streamgage_observations, 0)
                streamgage_observations = self.__t2.sub('', streamgage_observations, 0)

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
                df.drop(drop_cols, axis=1, inplace=True)

                # There should now only be date, site_no, and a Q column named *_00060_00003
                # We will rename the *_00060_00003 to mean_val
                rename_col = [col for col in df.columns if '_00060_00003' in col]

                if len(rename_col) > 1:
                    self.logger.warning('{} had more than one Q-col returned; using {}'.format(gg, rename_col[0]))

                    # Keep the first TS column and drop the others
                    while len(rename_col) > 1:
                        curr_col = rename_col.pop()
                        df.drop([curr_col], axis=1, inplace=True)

                df.rename(columns={rename_col[0]: gg}, inplace=True)

                # Resample to daily to fill in the missing days with NaN
                # df = df.resample('D').mean()

            self.__outdata = pd.merge(self.__outdata, df, how='left', left_index=True, right_index=True)
            self.__final_outorder.append(gg)

    def write_prms_data(self, filename):
        """Writes streamgage observations that have been downloaded to a file in PRMS format

        Args:
            filename: The name of the file for writing streamgage observations."""
        # Create the year, month, day, hour, minute, second columns
        try:
            self.__outdata['year'] = self.__outdata.index.year
            self.__outdata['month'] = self.__outdata.index.month
            self.__outdata['day'] = self.__outdata.index.day
            self.__outdata['hour'] = self.__outdata.index.hour
            self.__outdata['minute'] = self.__outdata.index.minute
            self.__outdata['second'] = self.__outdata.index.second
            self.__outdata.fillna(-999, inplace=True)
        except AttributeError:
            print('AttributeError')
            print(self.__outdata.head())
            print(self.__outdata.info())

        outhdl = open(filename, 'w')
        outhdl.write('Created by Bandit\n')
        outhdl.write('/////////////////////////////////////////////////////////////////////////\n')
        outhdl.write('// Station IDs for runoff:\n')
        outhdl.write('// ID\n')

        for gg in self.__gageids:
            outhdl.write('// {}\n'.format(gg))

        outhdl.write('/////////////////////////////////////////////////////////////////////////\n')
        outhdl.write('// Unit: runoff = cfs\n')
        outhdl.write('/////////////////////////////////////////////////////////////////////////\n')
        outhdl.write('runoff {}\n'.format(len(self.__gageids)))
        outhdl.write('#########################################################\n')

        self.__outdata.to_csv(outhdl, sep=' ', columns=self.__final_outorder, index=False, header=False)
        outhdl.close()
        sys.stdout.write('\r                                       ')
        sys.stdout.write('\r\tStreamflow data written to: {}\n'.format(filename))
        sys.stdout.flush()
