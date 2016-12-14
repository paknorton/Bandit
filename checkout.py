#!/usr/bin/env python

from __future__ import (absolute_import, division,
                        print_function)
# , unicode_literals)
from future.utils import iteritems

import networkx as nx
import numpy as np
import pandas as pd
import pickle as pickle
import re


from collections import OrderedDict
from datetime import datetime

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
    from urllib2 import urlopen, Request, HTTPError

from paramdb import get_global_params, get_global_dimensions
from pr_util import colorize, heading, print_info, print_warning, print_error
import cbh

REGIONS = ['r01', 'r02', 'r03', 'r04', 'r05', 'r06', 'r07', 'r08', 'r09',
           'r10L', 'r10U', 'r11', 'r12', 'r13', 'r14', 'r15', 'r16', 'r17', 'r18']

HRU_DIMS = ['nhru', 'ngw', 'nssr']  # These dimensions are related and should have same size

# Output directory
outdir = '/Users/pnorton/Projects/National_Hydrology_Model/regions/subset_testing'

# Output parameter filename
param_filename = 'crap.param'

# Output observation data filename
obs_filename = 'sf_data'

# Location of CONUS NHM parameter files
srcdir = '/Users/pnorton/Projects/National_Hydrology_Model/paramDb/merged_params'

# Specify downstream-most stream segment for extracting an upstream subset of NHM model
dsmost_seg = 31126  # 31380  # 31392    # 10

# Location of global NHM parameter xml file
params_file = '/Users/pnorton/Projects/National_Hydrology_Model/paramDb/nhmparamdb/parameters.xml'

# Location of NHM parameter database
workdir = '/Users/pnorton/Projects/National_Hydrology_Model/paramDb/nhmparamdb'

# Location of CBH files by region
cbh_dir = '/Users/pnorton/Projects/National_Hydrology_Model/datasets/daymet'

# Date range for pulling NWIS streamgage observations
st_date = datetime(1979, 10, 1)
en_date = datetime(2015, 9, 30)

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Read hru_nhm_to_local and hru_nhm_to_region
# Create segment_nhm_to_local and segment_nhm_to_region
print('-'*10 + 'Reading hru_nhm_to_region')
with open('{}/hru_nhm_to_region.pickle'.format(srcdir), 'rb') as ff:
    hru_nhm_to_region = pickle.load(ff)

print('-'*10 + 'Reading hru_nhm_to_local')
with open('{}/hru_nhm_to_local.pickle'.format(srcdir), 'rb') as ff:
    hru_nhm_to_local = pickle.load(ff)


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Read tosegment_nhm
print('-'*10 + 'Reading tosegment_nhm')
with open('{}/tosegment_nhm.pickle'.format(srcdir), 'rb') as ff:
    tosegment = pickle.load(ff)['data']

print('\tGenerating DAG of tosegments')
DAG_ds = nx.DiGraph()
for ii, vv in enumerate(tosegment):
    #     DAG_ds.add_edge(ii+1, vv)
    if vv == 0:
        DAG_ds.add_edge(ii + 1, 'Out_{}'.format(ii + 1))
    else:
        DAG_ds.add_edge(ii + 1, vv)

# nx.draw_networkx(DAG_ds)
print('\tNumber of nodes: {}'.format(DAG_ds.number_of_nodes()))
print('\tNumber of edges: {}'.format(DAG_ds.number_of_edges()))

if not nx.is_directed_acyclic_graph(DAG_ds):
    print('='*40)
    print_warning('Cycles and/or loops found in stream network')
    for xx in nx.simple_cycles(DAG_ds):
        print(xx)
    print('-'*40)

# Create the upstream graph
print('-'*10 + 'Creating U/S DAG')
DAG_us = DAG_ds.reverse()

# =======================================
# Given a d/s segment...
# dsmost_seg = 31126  # 31380  # 31392    # 10

print('-'*10 + 'Generating subset')
print('\tdsmost_seg:', dsmost_seg)

# then get all segments u/s of the starting segment
ss1 = nx.dfs_predecessors(DAG_us, dsmost_seg)

# Get a set of unique node values from the keys and values
uniq_seg_us = set(ss1.keys()).union(set(ss1.values()))
print('\tuniq_seg_us nodes:', uniq_seg_us)
print('\tSize of uniq_seg_us: {}'.format(len(uniq_seg_us)))

# Get a subgraph in the DAG_ds graph and return the edges
DAG_ds_subset = DAG_ds.subgraph(uniq_seg_us)

# Add the downstream segment that exits the subgraph
# We'll keep the original downstream tosegment value for now
# DAG_subds.add_edge(dsmost_seg, DAG_ds.neighbors(dsmost_seg)[0])
DAG_ds_subset.add_edge(dsmost_seg, 'Out_{}'.format(dsmost_seg))

print('-'*10 + 'DAG_subds Edges')
print(DAG_ds_subset.edges())

# Create list of toseg ids for the model subset
toseg_idx = list(set(xx[0] for xx in DAG_ds_subset.edges_iter()))
toseg_idx0 = [xx-1 for xx in toseg_idx]  # 0-based version of toseg_idx

print('-'*10 + 'toseg_idx')
print(toseg_idx)
print('len(toseg_idx): {}'.format(len(toseg_idx)))

print('Reading hru_segment_nhm')
with open('{}/hru_segment_nhm.pickle'.format(srcdir), 'rb') as ff:
    hru_segment = pickle.load(ff)['data']

print('Size of original hru_segment: {}'.format(len(hru_segment)))

# Create a dictionary mapping segments to HRUs
seg_to_hru = {}
for ii, vv in enumerate(hru_segment):
    seg_to_hru.setdefault(vv, []).append(ii + 1)

# Get HRU ids ordered by the segments in the model subset
print('-'*10 + 'seg_to_hru')
hru_order_subset = []
for xx in toseg_idx:
    if xx in seg_to_hru:
        print(xx, seg_to_hru[xx])

        for yy in seg_to_hru[xx]:
            hru_order_subset.append(yy)
    else:
        raise ValueError('Stream segment has no HRUs connected to it.')

hru_order_subset0 = [xx - 1 for xx in hru_order_subset]

print('Size of hru_order: {}'.format(len(hru_order_subset)))
print(hru_order_subset)

# Use hru_order to pull selected indices for parameters with nhru dimensions
# hru_order contains the in-order indices for the subset of hru_segments
# toseg_idx contains the in-order indices for the subset of tosegments (the indices reflect the original tosegment size)

# Renumber the tosegment list
new_tosegment = []

print('-'*10 + 'Mapping old DAG_subds indices to new')
for xx in toseg_idx:
    if DAG_ds_subset.neighbors(xx)[0] in toseg_idx:
        print('old: ({}, {}) '.format(xx, DAG_ds_subset.neighbors(xx)[0]) +
              'new: ({}, {})'.format(toseg_idx.index(xx) + 1, toseg_idx.index(DAG_ds_subset.neighbors(xx)[0]) + 1))
        new_tosegment.append(toseg_idx.index(DAG_ds_subset.neighbors(xx)[0]) + 1)
    else:
        # Outlets should be assigned zero
        print('old: ({}, {}) '.format(xx, DAG_ds_subset.neighbors(xx)[0]) +
              'new: ({}, {})'.format(toseg_idx.index(xx) + 1, 0))
        new_tosegment.append(0)

print('-'*10 + 'New tosegment indices')
print(new_tosegment)

# Renumber the hru_segments for the subset
new_hru_segment = []

for xx in toseg_idx:
    # if DAG_subds.neighbors(xx)[0] in toseg_idx:
    for yy in seg_to_hru[xx]:
        # The new indices should be 1-based from PRMS
        new_hru_segment.append(toseg_idx.index(xx)+1)

print('-'*10 + 'New hru_segment indices')
print(new_hru_segment)
print('Size of new_hru_segment: {}'.format(len(new_hru_segment)))

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Subset hru_deplcrv
print('-'*10 + 'Reading hru_deplcrv')
with open('{}/hru_deplcrv.pickle'.format(srcdir), 'rb') as ff:
    hru_deplcrv = pickle.load(ff)['data']

print('Size of original hru_deplcrv: {}'.format(len(hru_deplcrv)))

# Get subset of hru_deplcrv using hru_order
# A single snarea_curve can be referenced by multiple HRUs
hru_deplcrv_subset = np.array(hru_deplcrv)[tuple(hru_order_subset0),]
uniq_deplcrv = list(set(hru_deplcrv_subset))
uniq_deplcrv0 = [xx - 1 for xx in uniq_deplcrv]
print('-'*10 + 'uniq_deplcrv')
print(uniq_deplcrv)

# Create new hru_deplcrv and renumber
new_hru_deplcrv = [uniq_deplcrv.index(cc)+1 for cc in hru_deplcrv_subset]


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Subset poi_gage_segment
print('-'*10 + 'Reading poi_gage_segment')
with open('{}/poi_gage_segment.pickle'.format(srcdir), 'rb') as ff:
    poi_gage_segment = pickle.load(ff)['data']

print('Size of original poi_gage_segment: {}'.format(len(poi_gage_segment)))

with open('{}/poi_gage_id.pickle'.format(srcdir), 'rb') as ff:
    poi_gage_id = pickle.load(ff)['data']

with open('{}/poi_type.pickle'.format(srcdir), 'rb') as ff:
    poi_type = pickle.load(ff)['data']

# We want to get the indices of the poi_gage_segments that match the
# segments that are part of the subset. We can then use these
# indices to subset poi_gage_id and poi_type.
# The poi_gage_segment will need to be renumbered for the subset of segments.

# To subset poi_gage_segment we have to lookup each segment in the subset
new_poi_gage_segment = []
new_poi_gage_id = []
new_poi_type = []

for ss in uniq_seg_us:
    if ss in poi_gage_segment:
        new_poi_gage_segment.append(toseg_idx.index(ss)+1)
        new_poi_gage_id.append(poi_gage_id[poi_gage_segment.index(ss)])
        new_poi_type.append(poi_type[poi_gage_segment.index(ss)])

if len(poi_gage_segment) == 0:
    print_warning('No poi gages found for subset')


# ==================================================================
# ==================================================================
# Process the parameters and create a parameter file for the subset
# TODO: We should have the list of params and dimensions in the merged_params directory
params = get_global_params(params_file)

dims = get_global_dimensions(params, REGIONS, workdir)

# Map parameter datatypes for output to parameter file
param_types = {'I': 1, 'F': 2, 'D': 3, 'S': 4}

# Resize dimensions to the model subset
crap_dims = dims.copy()  # need a copy since we modify dims
for dd, dv in iteritems(crap_dims):
    # dimensions 'nmonths' and 'one' are never changed
    if dd in HRU_DIMS:
        dims[dd] = len(hru_order_subset0)
    elif dd == 'nsegment':
        dims[dd] = len(toseg_idx0)
    elif dd == 'ndeplval':
        dims[dd] = len(uniq_deplcrv0) * 11
        if 'ndepl' not in dims:
            dims['ndepl'] = len(uniq_deplcrv0)
    elif dd == 'npoigages':
        dims[dd] = len(new_poi_gage_segment)

# Open output file
outhdl = open('{}/{}'.format(outdir, param_filename), 'w')

# Write header lines
outhdl.write('Subset from NHM written by Skein\n')
outhdl.write('0.5\n')

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Write out dimensions section
outhdl.write('** Dimensions **\n')

for dd, dv in iteritems(dims):
    outhdl.write('####\n')
    outhdl.write('{}\n'.format(dd))
    outhdl.write('{}\n'.format(dv))

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Write out parameters section
outhdl.write('** Parameters **\n')

for pp, pv in iteritems(params):
    with open('{}/{}.pickle'.format(srcdir, pp), 'rb') as ff:
        cparam = pickle.load(ff)

    ndims = len(cparam['dimensions'])
    print('Parameter: {}'.format(cparam['name']))

    # Get order of dimensions and total size for parameter
    dim_order = [None] * ndims
    dim_total_size = 1

    for dd, dv in iteritems(cparam['dimensions']):
        dim_order[dv['position']-1] = dd
        dim_total_size *= dims[dd]

    outhdl.write('####\n')
    outhdl.write('{}\n'.format(cparam['name']))     # Parameter name
    outhdl.write('{}\n'.format(ndims))  # Number of dimensions

    # Write out dimension names in order
    for dd in dim_order:
        outhdl.write('{}\n'.format(dd))

    # Write out the total size for the parameter
    outhdl.write('{}\n'.format(dim_total_size))

    # Write out the datatype for the parameter
    outhdl.write('{}\n'.format(param_types[cparam['type']]))

    first_dimension = dim_order[0]
    second_dimension = dim_order[1]

    # Write out the data for the parameter
    if ndims == 1:
        # 1D Parameters
        if first_dimension == 'one':
            outdata = np.array(cparam['data'])
        elif first_dimension == 'nsegment':
            if pp in ['tosegment']:
                outdata = np.array(new_tosegment)
            else:
                outdata = np.array(cparam['data'])[tuple(toseg_idx0), ]
        elif first_dimension == 'ndeplval':
            # This is really a 2D in disguise, however, it is stored in C-order unlike
            # other 2D arrays
            outdata = np.array(cparam['data']).reshape((-1, 11))[tuple(uniq_deplcrv0), :]
        elif first_dimension == 'npoigages':
            if pp == 'poi_gage_segment':
                outdata = np.array(new_poi_gage_segment)
            elif pp == 'poi_gage_id':
                outdata = np.array(new_poi_gage_id)
            elif pp == 'poi_type':
                outdata = np.array(new_poi_type)
            else:
                print_warning('Unknown parameter, {}, with dimension {}'.format(pp, first_dimension))
        elif first_dimension in HRU_DIMS:
            if pp == 'hru_deplcrv':
                outdata = np.array(new_hru_deplcrv)
            elif pp == 'hru_segment':
                outdata = np.array(new_hru_segment)
            else:
                outdata = np.array(cparam['data'])[tuple(hru_order_subset0), ]
        else:
            print_warning('Code not written for dimension {}'.format(first_dimension))
    elif ndims == 2:
        # 2D Parameters
        outdata = np.array(cparam['data']).reshape((-1, dims[second_dimension]), order='F')

        if first_dimension == 'nsegment':
            outdata = outdata[tuple(toseg_idx0), :]
        elif first_dimension in HRU_DIMS:
            outdata = outdata[tuple(hru_order_subset0), :]
        else:
            print_warning('Code not written for 2D parameter, {}, which contains dimension {}'.format(pp, first_dimension))

    # Convert outdata to a list for writing
    if first_dimension == 'ndeplval':
        outlist = outdata.ravel().tolist()
    else:
        outlist = outdata.ravel(order='F').tolist()

    for xx in outlist:
        outhdl.write('{}\n'.format(xx))

outhdl.close()


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Subset the cbh files for the selected HRUs
CBH_VARS = ['tmax', 'tmin', 'prcp']
hru_order_ss = OrderedDict()

# Subset hru_nhm_to_local mapping
for xx in hru_order_subset:
    hru_order_ss[xx] = hru_nhm_to_local[xx]

print('hru_order_ss')
print(hru_order_ss)

for vv in CBH_VARS:
    print('Writing {} CBH subset'.format(vv))

    # For out_order the first six columns contain the time information and
    # are always output for the cbh files
    out_order = [0, 1, 2, 3, 4, 5]

    if not out_order:
        raise NameError('CBH column order is empty!')

    outdata = None
    first = True

    for rr, rvals in iteritems(hru_nhm_to_region):
        # print('Examining {} ({} to {})'.format(rr, rvals[0], rvals[1]))
        idx_retrieve = {}

        for yy in hru_order_ss.keys():
            if rvals[0] <= yy <= rvals[1]:
                # print('\tMatching region {}, HRU: {} ({})'.format(rr, yy, hru_order_ss[yy]))
                idx_retrieve[yy] = hru_order_ss[yy]
        if len(idx_retrieve) > 0:
            cc1 = cbh.Cbh('{}/{}_{}.cbh.gz'.format(cbh_dir, rr, vv), idx_retrieve)
            cc1.read_cbh()
            if first:
                outdata = cc1.data.copy()
                first = False
            else:
                outdata = pd.merge(outdata, cc1.data, on=[0, 1, 2, 3, 4, 5])

    # Append the HRUs as ordered for the subset
    out_order.extend(hru_order_subset)

    out_cbh = open('{}/{}.cbh'.format(outdir, vv), 'w')
    out_cbh.write('Written by skein\n')
    out_cbh.write('{} {}\n'.format(vv, len(hru_order_subset)))
    out_cbh.write('########################################\n')
    outdata.to_csv(out_cbh, columns=out_order, sep=' ', index=False, header=True)
    out_cbh.close()


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Download the streamgage information from NWIS

# URLs can be generated/tested at: http://waterservices.usgs.gov/rest/Site-Test-Tool.html
base_url = 'http://waterservices.usgs.gov/nwis'

url_pieces = OrderedDict()
url_pieces['?format'] = 'rdb'
url_pieces['sites'] = ''
url_pieces['startDT'] = st_date.strftime('%Y-%m-%d')
url_pieces['endDT'] = en_date.strftime('%Y-%m-%d')
url_pieces['siteStatus'] = 'active'
url_pieces['parameterCd'] = '00060'
url_pieces['siteType'] = 'ST'

# Regex's for stripping unneeded clutter from the rdb file
t1 = re.compile('^#.*$\n?', re.MULTILINE)   # remove comment lines
t2 = re.compile('^5s.*$\n?', re.MULTILINE)  # remove field length lines

first = True
outdata = None
out_order = ['year', 'month', 'day', 'hour', 'minute', 'second']

# Iterate over new_poi_gage_id and retrieve daily streamflow data from NWIS
for gg in new_poi_gage_id:
    url_pieces['sites'] = gg
    url_final = '&'.join(['{}={}'.format(kk, vv) for kk, vv in iteritems(url_pieces)])

    # Read site data
    streamgageObsPage = urlopen('{}/dv/{}'.format(base_url, url_final))
    streamgageObservations = streamgageObsPage.read()

    # Strip the comment lines and field length lines from the result using regex
    streamgageObservations = t1.sub('', streamgageObservations, 0)
    streamgageObservations = t2.sub('', streamgageObservations, 0)

    # Have to enforce site_no as string/text
    col_names = ['site_no']
    col_types = [np.str_]
    cols = dict(zip(col_names, col_types))

    # Read the rdb file into a dataframe
    df = pd.read_csv(StringIO(streamgageObservations), sep='\t', dtype=cols, parse_dates={'date': ['datetime']},
                     index_col='date')

    # Conveniently the columns we want to drop contain '_cd' in their names
    drop_cols = [col for col in df.columns if '_cd' in col]
    df.drop(drop_cols, axis=1, inplace=True)

    # There should now only be date, site_no, and a Q column named *_00060_00003
    # We will rename the *_00060_00003 to mean_val
    rename_col = [col for col in df.columns if '_00060_00003' in col]

    if len(rename_col) > 1:
        print('ERROR: more than one Q-col returned')
    else:
        df.rename(columns={rename_col[0]: 'mean_val'}, inplace=True)

    # Resample to daily to fill in the missing days with NaN
    df = df.resample('D').mean()
    df.fillna(-999, inplace=True)

    # Rename mean_va to site number
    df.rename(columns={'mean_va': gg}, inplace=True)

    if first:
        outdata = df.copy()
        first = False
    else:
        outdata = pd.merge(outdata, df, left_index=True, right_index=True)

    out_order.append(gg)

# Create the year, month, day, hour, minute, second columns
outdata['year'] = outdata.index.year
outdata['month'] = outdata.index.month
outdata['day'] = outdata.index.day
outdata['hour'] = outdata.index.hour
outdata['minute'] = outdata.index.minute
outdata['second'] = outdata.index.second

outhdl = open('{}/{}'.format(outdir, obs_filename), 'w')
outhdl.write('Created by skein\n')
outhdl.write('/////////////////////////////////////////////////////////////////////////\n')
outhdl.write('// Station IDs for runoff:\n')
outhdl.write('// ID\n')

for gg in new_poi_gage_id:
    outhdl.write('// {}\n'.format(gg))

outhdl.write('/////////////////////////////////////////////////////////////////////////\n')
outhdl.write('// Unit: runoff = cfs\n')
outhdl.write('/////////////////////////////////////////////////////////////////////////\n')
outhdl.write('runoff {}\n'.format(len(new_poi_gage_id)))
outhdl.write('#########################################################\n')

outdata.to_csv(outhdl, sep=' ', columns=out_order, index=False, header=False)
outhdl.close()
