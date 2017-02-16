#!/usr/bin/env python

from __future__ import (absolute_import, division, print_function)
# , unicode_literals)
from future.utils import iteritems

import argparse
import networkx as nx
import numpy as np
import pandas as pd
import re
import msgpack
import sys

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

import bandit_cfg as bc
from paramdb import get_global_params, get_global_dimensions
from pr_util import colorize, heading, print_info, print_warning, print_error
import cbh
import prms_nwis
import prms_geo

__author__ = 'Parker Norton (pnorton@usgs.gov)'
__version__ = '0.2'

REGIONS = ['r01', 'r02', 'r03', 'r04', 'r05', 'r06', 'r07', 'r08', 'r09',
           'r10L', 'r10U', 'r11', 'r12', 'r13', 'r14', 'r15', 'r16', 'r17', 'r18']

HRU_DIMS = ['nhru', 'ngw', 'nssr']  # These dimensions are related and should have same size

# Command line arguments
parser = argparse.ArgumentParser(description='Extract model subsets from the National Hydrologic Model')
parser.add_argument('-O', '--output_dir', help='Output directory for subset')
parser.add_argument('-p', '--param_filename', help='Name of output parameter file')
parser.add_argument('-s', '--streamflow_filename', help='Name of streamflow data file')
parser.add_argument('-P', '--paramdb_dir', help='Location of parameter database')
parser.add_argument('-M', '--merged_paramdb_dir', help='Location of merged parameter database')
parser.add_argument('-C', '--cbh_dir', help='Location of CBH files')
parser.add_argument('-g', '--geodatabase_filename', help='Full path to NHM geodatabase')
parser.add_argument('--check_DAG', help='Verify the streamflow network', action='store_true')
parser.add_argument('--output_cbh', help='Output CBH files for subset', action='store_true')
parser.add_argument('--output_shapefiles', help='Output shapefiles for subset', action='store_true')
parser.add_argument('--output_streamflow', help='Output streamflows for subset', action='store_true')

args = parser.parse_args()
print(args)

config = bc.Cfg('bandit.cfg')

# Override configuration variables with any command line parameters
for kk, vv in iteritems(args.__dict__):
    if vv:
        print("Overriding configuration for {} with {}".format(kk, vv))
        config.update_value(kk, vv)

# Where to output the subset
outdir = config.output_dir

# What to name the output parameter file
param_filename = config.param_filename

# What to name the streamflow output file
obs_filename = config.streamflow_filename

# Location of the NHM parameter database
paramdb_dir = config.paramdb_dir

# Location of the merged parameter database
merged_paramdb_dir = config.merged_paramdb_dir

# Location of the NHM CBH files
cbh_dir = config.cbh_dir

# Full path and filename to the geodatabase to use for outputting shapefile subsets
geo_file = config.geodatabase_filename

# List of outlets
dsmost_seg = config.outlets

# List of upstream cutoffs
uscutoff_seg = config.cutoffs

# List of additional HRUs (have no route to segment within subset)
hru_noroute = config.hru_noroute

# Control what is checked and output for subset
check_dag = config.check_DAG
output_cbh = config.output_cbh
output_streamflow = config.output_streamflow
output_shapefiles = config.output_shapefiles

# Date range for pulling NWIS streamgage observations
st_date = datetime(*[int(x) for x in re.split('-| |:', config.start_date)])
en_date = datetime(*[int(x) for x in re.split('-| |:', config.end_date)])

params_file = '{}/parameters.xml'.format(paramdb_dir)

# Specify downstream-most stream segment for extracting an upstream subset of NHM model

# GCPO outlet segments
# dsmost_seg = [4288, 4289, 4308, 4309, 4335, 4466, 4467, 4601, 4659, 4662, 4734, 4764,
#               4772, 4782, 4989, 5387, 7150, 7152, 7192, 7193, 7347, 7720, 7802, 8310,
#               8350, 8357, 8362, 8432, 8433, 8696, 8701, 8716, 8717, 8729, 9773, 9885,
#               21378, 21382, 21810, 21825, 21832, 22944, 23136, 23137, 23140, 23201,
#               23202, 23203, 23205, 23206, 23207, 23245, 23272, 23273, 38360, 38761,
#               38764, 38771, 38772, 38781, 38786, 38803, 38876, 41702,
#               22363]
# dsmost_seg = [31126, ]  # 31380  # 31392    # 10
# dsmost_seg = (36382, 36383, 22795)    # Red River of the South

# Specify the upstream-most stream segments to remove from the final subset

# GCPO cutoff segments
# hold: 12368, 38416, 39973, 40159
# uscutoff_seg = [12364, 12369, 16955, 20406, 24635, 24963,
#                 25304, 25506, 33700, 33812, 33829, 33700,
#                 36766, 38408, 39971, 40160]
# uscutoff_seg = [12364, 12369, 16954, 17608, 20360, 20391, 24618, 24619,
#                 24961, 24962, 25302, 25303, 25503, 25504, 33705, 33811,
#                 33826, 33837, 34262, 34910, 34912, 36763, 36764, 38408, 39971, 40160,
#                 8074]

# uscutoff_seg = [31113, ]    # cutoff for 31126


def get_parameter(filename):
    with open(filename, 'rb') as ff:
        return msgpack.load(ff, use_list=False)


def main():
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Read hru_nhm_to_local and hru_nhm_to_region
    # Create segment_nhm_to_local and segment_nhm_to_region
    print('-'*10 + 'Reading hru_nhm_to_region')
    hru_nhm_to_region = get_parameter('{}/hru_nhm_to_region.msgpack'.format(merged_paramdb_dir))

    print('-'*10 + 'Reading hru_nhm_to_local')
    hru_nhm_to_local = get_parameter('{}/hru_nhm_to_local.msgpack'.format(merged_paramdb_dir))

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Read tosegment_nhm
    print('-'*10 + 'Reading tosegment_nhm')
    tosegment = get_parameter('{}/tosegment_nhm.msgpack'.format(merged_paramdb_dir))['data']

    print('\tGenerating DAG of tosegments')
    dag_ds = nx.DiGraph()
    for ii, vv in enumerate(tosegment):
        #     dag_ds.add_edge(ii+1, vv)
        if vv == 0:
            dag_ds.add_edge(ii + 1, 'Out_{}'.format(ii + 1))
        else:
            dag_ds.add_edge(ii + 1, vv)

    # nx.draw_networkx(dag_ds)
    print('\tNumber of nodes: {}'.format(dag_ds.number_of_nodes()))
    print('\tNumber of edges: {}'.format(dag_ds.number_of_edges()))

    if check_dag:
        if not nx.is_directed_acyclic_graph(dag_ds):
            print('='*40)
            print_warning('Cycles and/or loops found in stream network')
            for xx in nx.simple_cycles(dag_ds):
                print(xx)
            print('-'*40)

    # Create the upstream graph
    print('-'*10 + 'Creating U/S DAG')
    dag_us = dag_ds.reverse()
    print('\tNumber of nodes: {}'.format(dag_us.number_of_nodes()))
    print('\tNumber of edges: {}'.format(dag_us.number_of_edges()))

    # Trim the u/s graph to remove segments above the u/s cutoff segments
    print('-'*10 + 'Trimming U/S DAG segments')
    for xx in uscutoff_seg:
        dag_us.remove_nodes_from(nx.dfs_predecessors(dag_us, xx))

        # Also remove the cutoff segment itself
        dag_us.remove_node(xx)

    print('\tNumber of nodes: {}'.format(dag_us.number_of_nodes()))
    print('\tNumber of edges: {}'.format(dag_us.number_of_edges()))

    # =======================================
    # Given a d/s segment (dsmost_seg) create a subset of u/s segments
    print('-'*10 + 'Generating subset')
    # print('\tdsmost_seg:', dsmost_seg)

    # Get all unique segments u/s of the starting segment
    uniq_seg_us = set()
    for xx in dsmost_seg:
        pred = nx.dfs_predecessors(dag_us, xx)
        uniq_seg_us = uniq_seg_us.union(set(pred.keys()).union(set(pred.values())))

    print('\tSize of uniq_seg_us: {}'.format(len(uniq_seg_us)))

    # Get a subgraph in the dag_ds graph and return the edges
    dag_ds_subset = dag_ds.subgraph(uniq_seg_us)

    # Add the downstream segments that exit the subgraph
    for xx in dsmost_seg:
        dag_ds_subset.add_edge(xx, 'Out_{}'.format(xx))

    print('-'*10 + 'DAG_subds Edges')

    # Create list of toseg ids for the model subset
    toseg_idx = list(set(xx[0] for xx in dag_ds_subset.edges_iter()))
    toseg_idx0 = [xx-1 for xx in toseg_idx]  # 0-based version of toseg_idx

    print('-'*10 + 'toseg_idx')
    # print(toseg_idx)
    print('len(toseg_idx): {}'.format(len(toseg_idx)))

    print('Reading hru_segment_nhm')
    hru_segment = get_parameter('{}/hru_segment_nhm.msgpack'.format(merged_paramdb_dir))['data']

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
            # print(xx, seg_to_hru[xx])

            for yy in seg_to_hru[xx]:
                hru_order_subset.append(yy)
        else:
            print_warning('Stream segment {} has no HRUs connected to it.'.format(xx))
            # raise ValueError('Stream segment has no HRUs connected to it.')

    # Append the additional non-routed HRUs to the list
    if len(hru_noroute) > 0:
        print("Adding additional non-routed HRUs")
        hru_order_subset.extend(hru_noroute)

    hru_order_subset0 = [xx - 1 for xx in hru_order_subset]

    print('Size of hru_order: {}'.format(len(hru_order_subset)))
    # print(hru_order_subset)

    # Use hru_order_subset to pull selected indices for parameters with nhru dimensions
    # hru_order_subset contains the in-order indices for the subset of hru_segments
    # toseg_idx contains the in-order indices for the subset of tosegments

    # Renumber the tosegment list
    new_tosegment = []

    print('-'*10 + 'Mapping old DAG_subds indices to new')
    for xx in toseg_idx:
        if dag_ds_subset.neighbors(xx)[0] in toseg_idx:
            # print('old: ({}, {}) '.format(xx, dag_ds_subset.neighbors(xx)[0]) +
            #       'new: ({}, {})'.format(toseg_idx.index(xx) + 1, toseg_idx.index(dag_ds_subset.neighbors(xx)[0]) + 1))
            new_tosegment.append(toseg_idx.index(dag_ds_subset.neighbors(xx)[0]) + 1)
        else:
            # Outlets should be assigned zero
            # print('old: ({}, {}) '.format(xx, dag_ds_subset.neighbors(xx)[0]) +
            #       'new: ({}, {})'.format(toseg_idx.index(xx) + 1, 0))
            new_tosegment.append(0)

    print('-'*10 + 'New tosegment indices')
    # print(new_tosegment)

    # Renumber the hru_segments for the subset
    new_hru_segment = []

    for xx in toseg_idx:
        # if DAG_subds.neighbors(xx)[0] in toseg_idx:
        if xx in seg_to_hru:
            for _ in seg_to_hru[xx]:
                # The new indices should be 1-based from PRMS
                new_hru_segment.append(toseg_idx.index(xx)+1)

    # Append zeroes to new_hru_segment for each additional non-routed HRU
    if len(hru_noroute) > 0:
        print("Adding additional non-routed HRUs to new_hru_segment")
        for _ in hru_noroute:
            new_hru_segment.append(0)

    print('-'*10 + 'New hru_segment indices')
    # print(new_hru_segment)
    print('Size of new_hru_segment: {}'.format(len(new_hru_segment)))

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Subset hru_deplcrv
    print('-'*10 + 'Reading hru_deplcrv')
    hru_deplcrv = get_parameter('{}/hru_deplcrv.msgpack'.format(merged_paramdb_dir))['data']

    print('Size of original hru_deplcrv: {}'.format(len(hru_deplcrv)))

    # Get subset of hru_deplcrv using hru_order
    # A single snarea_curve can be referenced by multiple HRUs
    hru_deplcrv_subset = np.array(hru_deplcrv)[tuple(hru_order_subset0), ]
    uniq_deplcrv = list(set(hru_deplcrv_subset))
    uniq_deplcrv0 = [xx - 1 for xx in uniq_deplcrv]
    print('-'*10 + 'uniq_deplcrv')
    # print(uniq_deplcrv)

    # Create new hru_deplcrv and renumber
    new_hru_deplcrv = [uniq_deplcrv.index(cc)+1 for cc in hru_deplcrv_subset]

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Subset poi_gage_segment
    print('-'*10 + 'Reading poi_gage_segment, poi_gage_id, poi_type')
    poi_gage_segment = get_parameter('{}/poi_gage_segment.msgpack'.format(merged_paramdb_dir))['data']
    print('Size of original poi_gage_segment: {}'.format(len(poi_gage_segment)))

    poi_gage_id = get_parameter('{}/poi_gage_id.msgpack'.format(merged_paramdb_dir))['data']
    poi_type = get_parameter('{}/poi_type.msgpack'.format(merged_paramdb_dir))['data']

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

    dims = get_global_dimensions(params, REGIONS, paramdb_dir)

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
        cparam = get_parameter('{}/{}.msgpack'.format(merged_paramdb_dir, pp))

        ndims = len(cparam['dimensions'])
        sys.stdout.write('\r                                       ')
        sys.stdout.write('\rProcessing {} '.format(cparam['name']))
        sys.stdout.flush()
        # print('Parameter: {}'.format(cparam['name']))

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

        if ndims == 2:
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
    sys.stdout.write('\r                                       ')
    sys.stdout.write('\rParameter file written: {}\n'.format('{}/{}'.format(outdir, param_filename)))
    sys.stdout.flush()

    if output_cbh:
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Subset the cbh files for the selected HRUs
        CBH_VARS = ['tmax', 'tmin', 'prcp']

        # Subset the hru_nhm_to_local mapping
        hru_order_ss = OrderedDict((kk, hru_nhm_to_local[kk]) for kk in hru_order_subset)

        print('Processing CBH files')
        for vv in CBH_VARS:
            # For out_order the first six columns contain the time information and
            # are always output for the cbh files
            out_order = [0, 1, 2, 3, 4, 5]

            if not out_order:
                raise NameError('CBH column out order is empty!')

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
                    cc1 = cbh.Cbh(filename='{}/{}_{}.cbh.gz'.format(cbh_dir, rr, vv), indices=idx_retrieve)
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
            print('\t{} written to: {}'.format(vv, '{}/{}.cbh'.format(outdir, vv)))

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Download the streamgage information from NWIS
    if output_streamflow:
        print('Downloading NWIS streamgage observations for {} stations'.format(len(new_poi_gage_id)))
        streamflow = prms_nwis.NWIS(gage_ids=new_poi_gage_id, st_date=st_date, en_date=en_date)
        streamflow.get_daily_streamgage_observations()
        streamflow.write_prms_data(filename='{}/{}'.format(outdir, obs_filename))

    # *******************************************
    # Create a shapefile of the selected HRUs
    if output_shapefiles:
        print('-'*40)
        print('Writing shapefiles for model subset')
        geo_shp = prms_geo.Geo(geo_file)

        # Output a shapefile of the selected HRUs
        print('\tHRUs')
        geo_shp.select_layer('nhruNationalIdentifier')
        geo_shp.write_shapefile('{}/HRU_subset.shp'.format(outdir), 'hru_id_nat', hru_order_subset)
        # geo_shp.filter_by_attribute('hru_id_nat', hru_order_subset)
        # geo_shp.write_shapefile2('{}/HRU_subset.shp'.format(outdir))
        # geo_shp.write_kml('{}/HRU_subset.kml'.format(outdir))

        # Output a shapefile of the selected stream segments
        print('\tSegments')
        geo_shp.select_layer('nsegmentNationalIdentifier')
        geo_shp.write_shapefile('{}/Segments_subset.shp'.format(outdir), 'seg_id_nat', toseg_idx)
        # geo_shp.filter_by_attribute('seg_id_nat', uniq_seg_us)
        # geo_shp.write_shapefile2('{}/Segments_subset.shp'.format(outdir))

        del geo_shp


if __name__ == '__main__':
    main()

