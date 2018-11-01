#!/usr/bin/env python

from __future__ import (absolute_import, division, print_function)
# , unicode_literals)
from future.utils import iteritems, iterkeys

import argparse
import errno
import logging
import networkx as nx
import msgpack
import netCDF4
import numpy as np
import os
import re
import sys
import xml.etree.ElementTree as xmlET

from collections import OrderedDict
import datetime

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

import Bandit.bandit_cfg as bc
import Bandit.prms_geo as prms_geo
import Bandit.prms_nwis as prms_nwis
from Bandit.git_version import git_version
from Bandit import __version__

from pyPRMS.constants import REGIONS, HRU_DIMS, PARAMETERS_XML
from pyPRMS.Cbh import Cbh, CBH_VARNAMES

__author__ = 'Parker Norton (pnorton@usgs.gov)'


def get_global_dimensions(params, regions, workdir):
    """Build dictionary of global dimensions for given list of parameters from the NHM parameter database.

    Args:
        params (:obj:`list` of :obj:`str`: List of parameters.
        regions (:obj:`list` of :obj:`str`: List of regions to include.
        workdir (str): Location of NHM parameter database.

    Returns:
        Dictionary of dimensions and sizes.

    """

    # This builds a dictionary of total dimension sizes for the concatenated parameters
    dimension_info = {}
    is_populated = {}

    # Loop through the xml files for each parameter and define the total size and dimensions
    for pp in iterkeys(params):
        for rr in regions:
            cdim_tree = xmlET.parse('{}/{}/{}/{}.xml'.format(workdir, pp, rr, pp))
            cdim_root = cdim_tree.getroot()

            for cdim in cdim_root.findall('./dimensions/dimension'):
                dim_name = cdim.get('name')
                dim_size = int(cdim.get('size'))

                is_populated.setdefault(dim_name, False)

                if not is_populated[dim_name]:
                    if dim_name in ['nmonths', 'ndays', 'one']:
                        # Non-additive dimensions
                        dimension_info[dim_name] = dimension_info.get(dim_name, dim_size)
                    else:
                        # Other dimensions are additive
                        dimension_info[dim_name] = dimension_info.get(dim_name, 0) + dim_size

        # Update is_populated to reflect dimension size(s) don't need to be re-computed
        for kk, vv in iteritems(is_populated):
            if not vv:
                is_populated[kk] = True
    return dimension_info


def get_global_params(params_file):
    """Retrieve dictionary of parameters with associated unit and type information.

    Args:
        params_file (str): Name of parameter xml file from the NHM parameter database.

    Returns:
        Dictionary of parameters with associated unit and type information.

    """

    # Read in the parameters.xml file
    params_tree = xmlET.parse(params_file)
    params_root = params_tree.getroot()

    params = {}

    for param in params_root.findall('parameter'):
        params[param.get('name')] = {}
        params[param.get('name')]['type'] = param.get('type')
        params[param.get('name')]['units'] = param.get('units')

    return params


def get_parameter(filename):
    """Load a msgpack file.

    Args:
        filename (str): Name of msgpack file.

    Returns:
        Object from msgpack file.

    """

    with open(filename, 'rb') as ff:
        return msgpack.load(ff, use_list=False)


def main():
    # Command line arguments
    parser = argparse.ArgumentParser(description='Extract model subsets from the National Hydrologic Model')
    parser.add_argument('-O', '--output_dir', help='Output directory for subset')
    parser.add_argument('-p', '--param_filename', help='Name of output parameter file')
    parser.add_argument('-s', '--streamflow_filename', help='Name of streamflow data file')
    parser.add_argument('-P', '--paramdb_dir', help='Location of parameter database')
    parser.add_argument('-M', '--merged_paramdb_dir', help='Location of merged parameter database')
    parser.add_argument('-C', '--cbh_dir', help='Location of CBH files')
    parser.add_argument('-g', '--geodatabase_filename', help='Full path to NHM geodatabase')
    parser.add_argument('-j', '--job', help='Job directory to work in')
    parser.add_argument('-v', '--verbose', help='Output additional information', action='store_true')
    parser.add_argument('--check_DAG', help='Verify the streamflow network', action='store_true')
    parser.add_argument('--output_cbh', help='Output CBH files for subset', action='store_true')
    parser.add_argument('--output_shapefiles', help='Output shapefiles for subset', action='store_true')
    parser.add_argument('--output_streamflow', help='Output streamflows for subset', action='store_true')
    parser.add_argument('--cbh_netcdf', help='Enable netCDF output for CBH files', action='store_true')

    args = parser.parse_args()

    stdir = os.getcwd()

    if args.job:
        if os.path.exists(args.job):
            # Change into job directory before running extraction
            os.chdir(args.job)
            # print('Working in directory: {}'.format(args.job))
        else:
            print('ERROR: Invalid jobs directory: {}'.format(args.job))
            exit(1)

    # Setup the logging
    bandit_log = logging.getLogger('bandit')
    bandit_log.setLevel(logging.DEBUG)

    log_fmt = logging.Formatter('%(levelname)s: %(name)s: %(message)s')

    # Handler for file logs
    flog = logging.FileHandler('bandit.log')
    flog.setLevel(logging.DEBUG)
    flog.setFormatter(log_fmt)

    # Handler for console logs
    clog = logging.StreamHandler()
    clog.setLevel(logging.ERROR)
    clog.setFormatter(log_fmt)

    bandit_log.addHandler(flog)
    bandit_log.addHandler(clog)

    bandit_log.info('========== START {} =========='.format(datetime.datetime.now().isoformat()))

    config = bc.Cfg('bandit.cfg')

    # Override configuration variables with any command line parameters
    for kk, vv in iteritems(args.__dict__):
        if kk not in ['job', 'verbose', 'cbh_netcdf']:
            if vv:
                bandit_log.info('Overriding configuration for {} with {}'.format(kk, vv))
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
    if isinstance(config.start_date, datetime.date):
        st_date = config.start_date
    else:
        st_date = datetime.datetime(*[int(x) for x in re.split('-| |:', config.start_date)])

    if isinstance(config.end_date, datetime.date):
        en_date = config.end_date
    else:
        en_date = datetime.datetime(*[int(x) for x in re.split('-| |:', config.end_date)])

    # ===============================================================
    params_file = '{}/{}'.format(merged_paramdb_dir, PARAMETERS_XML)

    # Output revision of NhmParamDb and the revision used by merged paramdb
    bandit_log.debug('Current NhmParamDb revision: {}'.format(git_version(paramdb_dir)))
    with open('{}/00-REVISION'.format(merged_paramdb_dir), 'r') as fhdl:
        nhmparamdb_revision = fhdl.readline().strip()
        bandit_log.info('Parameters based on NhmParamDb revision: {}'.format(nhmparamdb_revision))

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Read hru_nhm_to_local and hru_nhm_to_region
    # Create segment_nhm_to_local and segment_nhm_to_region
    hru_nhm_to_region = get_parameter('{}/hru_nhm_to_region.msgpack'.format(merged_paramdb_dir))
    hru_nhm_to_local = get_parameter('{}/hru_nhm_to_local.msgpack'.format(merged_paramdb_dir))

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Read tosegment_nhm
    tosegment = get_parameter('{}/tosegment_nhm.msgpack'.format(merged_paramdb_dir))['data']

    if args.verbose:
        print('Generating stream network from tosegment_nhm')

    # First check is any of the requested stream segments exist in the NHM.
    # An intersection of 0 elements can occur when all stream segments are
    # not included in the NHM (e.g. segments in Alaska).
    if len(set(dsmost_seg).intersection(tosegment)) == 0:
        bandit_log.error('None of the requested stream segments exist in the NHM paramDb')
        exit(200)

    dag_ds = nx.DiGraph()
    for ii, vv in enumerate(tosegment):
        #     dag_ds.add_edge(ii+1, vv)
        if vv == 0:
            dag_ds.add_edge(ii + 1, 'Out_{}'.format(ii + 1))
        else:
            dag_ds.add_edge(ii + 1, vv)

    # nx.draw_networkx(dag_ds)
    bandit_log.debug('Number of NHM downstream nodes: {}'.format(dag_ds.number_of_nodes()))
    bandit_log.debug('Number of NHM downstream edges: {}'.format(dag_ds.number_of_edges()))

    if check_dag:
        if not nx.is_directed_acyclic_graph(dag_ds):
            bandit_log.error('Cycles and/or loops found in stream network')

            for xx in nx.simple_cycles(dag_ds):
                bandit_log.error('Cycle found for segment {}'.format(xx))

    # Create the upstream graph
    dag_us = dag_ds.reverse()
    bandit_log.debug('Number of NHM upstream nodes: {}'.format(dag_us.number_of_nodes()))
    bandit_log.debug('Number of NHM upstream edges: {}'.format(dag_us.number_of_edges()))

    # Trim the u/s graph to remove segments above the u/s cutoff segments
    try:
        for xx in uscutoff_seg:
            try:
                dag_us.remove_nodes_from(nx.dfs_predecessors(dag_us, xx))

                # Also remove the cutoff segment itself
                dag_us.remove_node(xx)
            except KeyError:
                        print('WARNING: nhm_segment {} does not exist in stream network'.format(xx))
    except TypeError:
        bandit_log.error('\nSelected cutoffs should at least be an empty list instead of NoneType. ({})'.format(outdir))
        exit(200)

    bandit_log.debug('Number of NHM upstream nodes (trimmed): {}'.format(dag_us.number_of_nodes()))
    bandit_log.debug('Number of NHM upstream edges (trimmed): {}'.format(dag_us.number_of_edges()))

    # =======================================
    # Given a d/s segment (dsmost_seg) create a subset of u/s segments
    if args.verbose:
        print('\tExtracting model subset')

    # Get all unique segments u/s of the starting segment
    uniq_seg_us = set()
    if dsmost_seg:
        for xx in dsmost_seg:
            try:
                pred = nx.dfs_predecessors(dag_us, xx)
                uniq_seg_us = uniq_seg_us.union(set(pred.keys()).union(set(pred.values())))
            except KeyError:
                bandit_log.error('KeyError: Segment {} does not exist in stream network'.format(xx))
                # print('\nKeyError: Segment {} does not exist in stream network'.format(xx))

        # Get a subgraph in the dag_ds graph and return the edges
        dag_ds_subset = dag_ds.subgraph(uniq_seg_us)

        # 2018-02-13 PAN: It is possible to have outlets specified which are not truly
        #                 outlets in the most conservative sense (e.g. a point where
        #                 the stream network exits the study area). This occurs when
        #                 doing headwater extractions where all segments for a headwater
        #                 are specified in the configuration file. Instead of creating
        #                 output edges for all specified 'outlets' the set difference
        #                 between the specified outlets and nodes in the graph subset
        #                 which have no edges is performed first to reduce the number of
        #                 outlets to the 'true' outlets of the system.
        node_outlets = [ee[0] for ee in dag_ds_subset.edges()]
        true_outlets = set(dsmost_seg).difference(set(node_outlets))
        bandit_log.debug('node_outlets: {}'.format(','.join(map(str, node_outlets))))
        bandit_log.debug('true_outlets: {}'.format(','.join(map(str, true_outlets))))

        # Add the downstream segments that exit the subgraph
        for xx in true_outlets:
            dag_ds_subset.add_edge(xx, 'Out_{}'.format(xx))
    else:
        # No outlets specified so pull the CONUS
        dag_ds_subset = dag_ds

    # Create list of toseg ids for the model subset
    toseg_idx = list(set(xx[0] for xx in dag_ds_subset.edges_iter()))
    toseg_idx0 = [xx-1 for xx in toseg_idx]  # 0-based version of toseg_idx

    bandit_log.info('Number of segments in subset: {}'.format(len(toseg_idx)))

    # if len(toseg_idx) == 0:
    #     print('\nERROR: No segments were selected for extraction. ({})'.format(outdir))
    #     exit(200)

    # print('edges')
    # for xx in dag_ds_subset.adjacency_iter():
    #    print(xx)

    # print('toseg_idx')
    # print(toseg_idx)

    hru_segment = get_parameter('{}/hru_segment_nhm.msgpack'.format(merged_paramdb_dir))['data']

    bandit_log.info('Number of NHM hru_segment entries: {}'.format(len(hru_segment)))

    # Create a dictionary mapping segments to HRUs
    seg_to_hru = {}
    for ii, vv in enumerate(hru_segment):
        seg_to_hru.setdefault(vv, []).append(ii + 1)

    # Get HRU ids ordered by the segments in the model subset
    hru_order_subset = []
    for xx in toseg_idx:
        if xx in seg_to_hru:
            for yy in seg_to_hru[xx]:
                hru_order_subset.append(yy)
        else:
            bandit_log.warning('Stream segment {} has no HRUs connected to it.'.format(xx))
            # raise ValueError('Stream segment has no HRUs connected to it.')

    # print('hru_order_subset')
    # print(hru_order_subset)

    # Append the additional non-routed HRUs to the list
    if len(hru_noroute) > 0:
        for xx in hru_noroute:
            if hru_segment[xx-1] == 0:
                bandit_log.info('User-supplied HRU {} is not connected to any stream segment'.format(xx))
                hru_order_subset.append(xx)
            else:
                bandit_log.error('User-supplied HRU {} routes to stream segment {} - Skipping.'.format(xx,
                                                                                                       hru_segment[xx-1]))

    hru_order_subset0 = [xx - 1 for xx in hru_order_subset]

    bandit_log.info('Number of HRUs in subset: {}'.format(len(hru_order_subset)))

    # Use hru_order_subset to pull selected indices for parameters with nhru dimensions
    # hru_order_subset contains the in-order indices for the subset of hru_segments
    # toseg_idx contains the in-order indices for the subset of tosegments

    # Renumber the tosegment list
    new_tosegment = []

    # Map old DAG_subds indices to new
    for xx in toseg_idx:
        if dag_ds_subset.neighbors(xx)[0] in toseg_idx:
            new_tosegment.append(toseg_idx.index(dag_ds_subset.neighbors(xx)[0]) + 1)
        else:
            # Outlets should be assigned zero
            new_tosegment.append(0)

    # print('new_tosegment')
    # print(new_tosegment)

    # Renumber the hru_segments for the subset
    new_hru_segment = []

    for xx in toseg_idx:
        # if DAG_subds.neighbors(xx)[0] in toseg_idx:
        if xx in seg_to_hru:
            for yy in seg_to_hru[xx]:
                # The new indices should be 1-based from PRMS
                new_hru_segment.append(toseg_idx.index(xx)+1)
                # print(xx, yy)

    # print('new_hru_segment')
    # print(new_hru_segment)

    # Append zeroes to new_hru_segment for each additional non-routed HRU
    if len(hru_noroute) > 0:
        for xx in hru_noroute:
            if hru_segment[xx-1] == 0:
                new_hru_segment.append(0)

    bandit_log.info('Size of hru_segment for subset: {}'.format(len(new_hru_segment)))

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Subset hru_deplcrv
    hru_deplcrv = get_parameter('{}/hru_deplcrv.msgpack'.format(merged_paramdb_dir))['data']

    bandit_log.info('Size of NHM hru_deplcrv: {}'.format(len(hru_deplcrv)))

    # Get subset of hru_deplcrv using hru_order
    # A single snarea_curve can be referenced by multiple HRUs
    hru_deplcrv_subset = np.array(hru_deplcrv)[tuple(hru_order_subset0), ]
    uniq_deplcrv = list(set(hru_deplcrv_subset))
    uniq_deplcrv0 = [xx - 1 for xx in uniq_deplcrv]

    # print('hru_deplcrv_subset')
    # print(hru_deplcrv_subset)

    # Create new hru_deplcrv and renumber
    new_hru_deplcrv = [uniq_deplcrv.index(cc)+1 for cc in hru_deplcrv_subset]
    bandit_log.info('Size of hru_deplcrv for subset: {}'.format(len(new_hru_deplcrv)))

    # print('new_hru_deplcrv')
    # print(new_hru_deplcrv)

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Subset poi_gage_segment
    poi_gage_segment = get_parameter('{}/poi_gage_segment.msgpack'.format(merged_paramdb_dir))['data']
    bandit_log.info('Size of NHM poi_gage_segment: {}'.format(len(poi_gage_segment)))

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

    # for ss in uniq_seg_us:
    for ss in nx.nodes_iter(dag_ds_subset):
        if ss in poi_gage_segment:
            new_poi_gage_segment.append(toseg_idx.index(ss)+1)
            new_poi_gage_id.append(poi_gage_id[poi_gage_segment.index(ss)])
            new_poi_type.append(poi_type[poi_gage_segment.index(ss)])

    # ==================================================================
    # ==================================================================
    # Process the parameters and create a parameter file for the subset
    # TODO: We should have the list of params and dimensions in the merged_params directory
    params = get_global_params(params_file)

    # Remove the POI-related parameters if we have no POIs
    if len(new_poi_gage_segment) == 0:
        bandit_log.warning('No POI gages found for subset; removing POI-related parameters.')

        for rp in ['poi_gage_id', 'poi_gage_segment', 'poi_type']:
            params.pop(rp, None)

    dims = get_global_dimensions(params, REGIONS, paramdb_dir)

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
    outhdl.write('Written by Bandit version {}\n'.format(__version__))
    outhdl.write('NhmParamDb revision: {}\n'.format(nhmparamdb_revision))

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Write out dimensions section
    outhdl.write('** Dimensions **\n')

    for dd, dv in iteritems(dims):
        outhdl.write('####\n')
        outhdl.write('{}\n'.format(dd))
        outhdl.write('{}\n'.format(dv))

        if dd == 'npoigages':
            # 20170217 PAN: nobs is missing from the paramdb but is necessary
            outhdl.write('####\n')
            outhdl.write('{}\n'.format('nobs'))
            outhdl.write('{}\n'.format(dv))

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Write out parameters section
    outhdl.write('** Parameters **\n')

    # for pp, pv in iteritems(params):
    params = params.keys()
    params.sort()
    for pp in params:
        cparam = get_parameter('{}/{}.msgpack'.format(merged_paramdb_dir, pp))

        ndims = len(cparam['dimensions'])
        if args.verbose:
            sys.stdout.write('\r                                       ')
            sys.stdout.write('\rProcessing {} '.format(cparam['name']))
            sys.stdout.flush()

        # Get order of dimensions and total size for parameter
        dim_order = [None] * ndims
        dim_total_size = 1

        for dd, dv in iteritems(cparam['dimensions']):
            dim_order[dv['position']] = dd
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
        # outhdl.write('{}\n'.format(param_types[cparam['datatype']]))
        outhdl.write('{}\n'.format(cparam['datatype']))

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
                    bandit_log.error('Unkown parameter, {}, with dimensions {}'.format(pp, first_dimension))
            elif first_dimension in HRU_DIMS:
                if pp == 'hru_deplcrv':
                    outdata = np.array(new_hru_deplcrv)
                elif pp == 'hru_segment':
                    outdata = np.array(new_hru_segment)
                else:
                    outdata = np.array(cparam['data'])[tuple(hru_order_subset0), ]
            else:
                bandit_log.error('No rules to handle dimension {}'.format(first_dimension))
        elif ndims == 2:
            # 2D Parameters
            outdata = np.array(cparam['data']).reshape((-1, dims[second_dimension]), order='F')

            if first_dimension == 'nsegment':
                outdata = outdata[tuple(toseg_idx0), :]
            elif first_dimension in HRU_DIMS:
                outdata = outdata[tuple(hru_order_subset0), :]
            else:
                bandit_log.error('No rules to handle 2D parameter, {}, which contains dimension {}'.format(pp,
                                                                                                           first_dimension))

        # Convert outdata to a list for writing
        if first_dimension == 'ndeplval':
            outlist = outdata.ravel().tolist()
        else:
            outlist = outdata.ravel(order='F').tolist()

        for xx in outlist:
            if cparam['datatype'] in [2, 3]:
                # Float and double types have to be formatted specially so
                # they aren't written in exponential notation or with
                # extraneous zeroes
                tmp = '{:<20f}'.format(xx).rstrip('0 ')
                if tmp[-1] == '.':
                    tmp += '0'

                outhdl.write('{}\n'.format(tmp))
                # outhdl.write('{:f}\n'.format(xx))
            else:
                outhdl.write('{}\n'.format(xx))

    outhdl.close()

    if args.verbose:
        sys.stdout.write('\r                                       ')
        sys.stdout.write('\r\tParameter file written: {}\n'.format('{}/{}'.format(outdir, param_filename)))
        sys.stdout.flush()

    if output_cbh:
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Subset the cbh files for the selected HRUs

        if len(hru_order_subset) > 0:
            # Subset the hru_nhm_to_local mapping
            hru_order_ss = OrderedDict()
            for kk in hru_order_subset:
                hru_order_ss[kk] = hru_nhm_to_local[kk]

            if args.verbose:
                print('Processing CBH files')

            # NetCDF-related variables
            NC_FILL_DOUBLE = 9.9692099683868690e+36
            NC_FILL_FLOAT = 9.9692099683868690e+36
            var_desc = {'tmax': 'Maximum Temperature', 'tmin': 'Minimum temperature', 'prcp': 'Precipitation'}
            var_units = {'tmax': 'C', 'tmin': 'C', 'prcp': 'inches'}

            # for vv in ['prcp']:
            for vv in CBH_VARNAMES:
                if args.verbose:
                    print(vv)

                cbh_hdl = Cbh(indices=hru_order_ss, mapping=hru_nhm_to_region, var=vv,
                              st_date=st_date, en_date=en_date)

                if args.verbose:
                    print('\tReading {}'.format(vv))

                cbh_hdl.read_cbh_multifile(cbh_dir)

                if args.verbose:
                    print('\tWriting {} CBH file'.format(vv))

                if args.cbh_netcdf:
                    out_order = [kk for kk in hru_order_subset]

                    # Get a pandas dataframe in the order originally specified
                    # The extraneous yr, mon, day, etc columns won't be in the
                    # resulting dataframe so it's not necessary to drop them.
                    df2 = cbh_hdl.data[out_order]
                    # print(df2.head())

                    # df2.drop(columns=['second', 'minute', 'hour', 'day', 'month', 'year'], inplace=True)
                    # df2.drop(['second', 'minute', 'hour', 'day', 'month', 'year'], axis=1, inplace=True)

                    # Create a netCDF file for the CBH data
                    nco = netCDF4.Dataset('{}/{}.nc'.format(outdir, vv), 'w', clobber=True)
                    nco.createDimension('hru', len(df2.columns))
                    nco.createDimension('time', None)

                    timeo = nco.createVariable('time', 'f4', ('time'))
                    hruo = nco.createVariable('hru', 'i4', ('hru'))

                    varo = nco.createVariable(vv, 'f4', ('time', 'hru'), fill_value=NC_FILL_FLOAT, zlib=True)

                    nco.setncattr('Description', 'Climate by HRU')
                    nco.setncattr('Bandit_version', __version__)
                    nco.setncattr('NHM_version', nhmparamdb_revision)

                    timeo.calendar = 'standard'
                    # timeo.bounds = 'time_bnds'
                    timeo.units = 'days since 1980-01-01 00:00:00'

                    hruo.long_name = 'Hydrologic Response Unit ID (HRU)'

                    varo.long_name = var_desc[vv]
                    varo.units = var_units[vv]

                    # Write the HRU ids
                    hruo[:] = df2.columns.values

                    timeo[:] = netCDF4.date2num(df2.index.tolist(), units='days since 1980-01-01 00:00:00',
                                                calendar='standard')

                    # Write the CBH values
                    varo[:, :] = df2.values

                    nco.close()
                else:
                    # For out_order the first six columns contain the time information and
                    # are always output for the cbh files
                    out_order = [kk for kk in hru_order_subset]
                    for cc in ['second', 'minute', 'hour', 'day', 'month', 'year']:
                        out_order.insert(0, cc)

                    # Output ASCII CBH files
                    out_cbh = open('{}/{}.cbh'.format(outdir, vv), 'w')
                    out_cbh.write('Written by Bandit\n')
                    out_cbh.write('{} {}\n'.format(vv, len(hru_order_subset)))
                    out_cbh.write('########################################\n')
                    cbh_hdl.data.to_csv(out_cbh, columns=out_order, sep=' ', index=False, header=False,
                                        encoding=None, chunksize=50)
                    out_cbh.close()
                    bandit_log.info('{} written to: {}'.format(vv, '{}/{}.cbh'.format(outdir, vv)))
        else:
            bandit_log.error('No HRUs associated with the segments')

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Download the streamgage information from NWIS
    if output_streamflow:
        if args.verbose:
            print('Downloading NWIS streamgage observations for {} stations'.format(len(new_poi_gage_id)))

        streamflow = prms_nwis.NWIS(gage_ids=new_poi_gage_id, st_date=st_date, en_date=en_date)
        streamflow.get_daily_streamgage_observations()
        streamflow.write_prms_data(filename='{}/{}'.format(outdir, obs_filename))

    # *******************************************
    # Create a shapefile of the selected HRUs
    if output_shapefiles:
        if args.verbose:
            print('-'*40)
            print('Writing shapefiles for model subset')

        if not os.path.isdir(geo_file):
            bandit_log.error('File geodatabase, {}, does not exist. Shapefiles will not be created'.format(geo_file))
        else:
            geo_shp = prms_geo.Geo(geo_file)

            # Create GIS sub-directory if it doesn't already exist
            gis_dir = '{}/GIS'.format(outdir)
            try:
                os.makedirs(gis_dir)
            except OSError as exception:
                if exception.errno != errno.EEXIST:
                    raise
                else:
                    pass

            # Output a shapefile of the selected HRUs
            # print('\tHRUs')
            # geo_shp.select_layer('nhruNationalIdentifier')
            geo_shp.select_layer('nhru')
            geo_shp.write_shapefile('{}/GIS/HRU_subset.shp'.format(outdir), 'hru_id_nat', hru_order_subset)

            # geo_shp.write_shapefile3('{}/GIS/HRU_subset.gdb'.format(outdir), 'hru_id_nat', hru_order_subset)

            # geo_shp.filter_by_attribute('hru_id_nat', hru_order_subset)
            # geo_shp.write_shapefile2('{}/HRU_subset.shp'.format(outdir))
            # geo_shp.write_kml('{}/HRU_subset.kml'.format(outdir))

            # Output a shapefile of the selected stream segments
            # print('\tSegments')
            geo_shp.select_layer('nsegmentNationalIdentifier')
            geo_shp.write_shapefile('{}/GIS/Segments_subset.shp'.format(outdir), 'seg_id_nat', toseg_idx)

            # geo_shp.filter_by_attribute('seg_id_nat', uniq_seg_us)
            # geo_shp.write_shapefile2('{}/Segments_subset.shp'.format(outdir))

            del geo_shp

    bandit_log.info('========== END {} =========='.format(datetime.datetime.now().isoformat()))

    os.chdir(stdir)


if __name__ == '__main__':
    main()
