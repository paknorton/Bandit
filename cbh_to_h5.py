#!/usr/bin/env python

from __future__ import (absolute_import, division, print_function)
# , unicode_literals)
# from future.utils import iteritems

# import os
import calendar
import pandas as pd
from datetime import datetime


workdir = '/Users/pnorton/Projects/National_Hydrology_Model/datasets/daymet'

CBH_VARS = ['prcp', 'tmax', 'tmin']

REGIONS = ['r01', 'r02', 'r03', 'r04', 'r05', 'r06', 'r07', 'r08', 'r09',
           'r10L', 'r10U', 'r11', 'r12', 'r13', 'r14', 'r15', 'r16', 'r17', 'r18']


# NOTE: This is not currently used -- more work is needed to get it to function properly.

def dparse(*dstr):
    dint = [int(x) for x in dstr]

    if len(dint) == 2:
        # For months we want the last day of each month
        dint.append(calendar.monthrange(*dint)[1])
    if len(dint) == 1:
        # For annual we want the last day of the year
        dint.append(12)
        dint.append(calendar.monthrange(*dint)[1])

    return datetime(*dint)


for rr in REGIONS:
    for vv in CBH_VARS:
        hdf_filename = '{}/{}_{}.h5'.format(workdir, rr, vv)
        cbh_filename = '{}/{}_{}.cbh.gz'.format(workdir, rr, vv)
        print('Processing {}_{}'.format(rr, vv))

        # Create hdf5 file for output
        hdf_out = pd.HDFStore(hdf_filename)

        # Load the CBH file
        cbh_in = pd.read_csv(cbh_filename, sep=' ', skipinitialspace=True, skiprows=3, engine='c',
                             date_parser=dparse, parse_dates={'thedate': [0, 1, 2, 3, 4, 5]},
                             index_col='thedate', header=None, na_values=[-99.0, -999.0])

        # Renumber/rename columns to reflect local HRU number
        cbh_in.rename(columns=lambda xx: cbh_in.columns.get_loc(xx) + 1, inplace=True)
        print(cbh_in.head())
        print(cbh_in.info())

        # Copy the data to the hdf5 file
        hdf_out.put(vv, cbh_in.ix[:, 0:1000], format='t', complib='zlib', complevel=1)
        hdf_out.close()
        # cbh_in.to_hdf(hdf_filename, vv, mode='w', format='t', data_columns=True, complevel=1, complib='zlib',
        #               fletcher32=True)
        exit()
