#!/usr/bin/env python

from __future__ import (absolute_import, division, print_function)
# , unicode_literals)
from future.utils import iteritems, iterkeys

import argparse
import errno
import glob
import logging
import networkx as nx
import numpy as np
import os
import re
import sys
# import xml.etree.ElementTree as xmlET

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
import Bandit.dynamic_parameters as dyn_params
from Bandit.git_version import git_version
from Bandit import __version__

from pyPRMS.ParamDb import ParamDb
# from pyPRMS.ParameterFile import ParameterFile
from pyPRMS.constants import REGIONS, HRU_DIMS, PARAMETERS_XML
from pyPRMS.CbhNetcdf import CbhNetcdf
from pyPRMS.CbhAscii import CbhAscii
from pyPRMS.ControlFile import ControlFile
from pyPRMS.ParameterSet import ParameterSet
from pyPRMS.ValidParams import ValidParams

__author__ = 'Parker Norton (pnorton@usgs.gov)'


def parse_gage(s):
    """Parse a streamgage key-value pair.

    Parse a streamgage key-value pair, separated by '='; that's the reverse of ShellArgs.
    On the command line (argparse) a declaration will typically look like::
        foo=hello or foo="hello world"

    :param s: tuple(key, value)
    """

    # Adapted from: https://gist.github.com/fralau/061a4f6c13251367ef1d9a9a99fb3e8d
    items = s.split('=')
    key = items[0].strip()  # we remove blanks around keys, as is logical
    if len(items) > 1:
        # rejoin the rest:
        value = '='.join(items[1:])
    return (key, value)


def parse_gages(items):
    """Parse a list of key-value pairs and return a dictionary.

    :param list[(str, str)] items: list of key-value tuples

    :returns: key-value dictionary
    :rtype: dict[str, str]
    """

    # Adapted from: https://gist.github.com/fralau/061a4f6c13251367ef1d9a9a99fb3e8d
    d = {}

    if items:
        for item in items:
            key, value = parse_gage(item)
            d[key] = int(value)
    return d


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
    parser.add_argument('--param_netcdf', help='Enable netCDF output for parameter file', action='store_true')
    parser.add_argument('--add_gages', metavar="KEY=VALUE", nargs='+', help='Add arbitrary streamgages to POIs of form gage_id=segment. Segment must exist in the model subset. Additional streamgages are marked as poi_type=0.')
    parser.add_argument('--no_filter_params', help='Output all parameters regardless of modules selected', action='store_true')
    args = parser.parse_args()

    stdir = os.getcwd()

    if args.job:
        if os.path.exists(args.job):
            # Change into job directory before running extraction
            os.chdir(args.job)
            # print('Working in directory: {}'.format(args.job))
        else:
            print('ERROR: Invalid jobs directory: {}'.format(args.job))
            exit(-1)

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

    addl_gages = None
    if args.add_gages:
        addl_gages = parse_gages(args.add_gages)
        bandit_log.info('Additionals streamgages specified on command line')

    config = bc.Cfg('bandit.cfg')

    # Override configuration variables with any command line parameters
    for kk, vv in iteritems(args.__dict__):
        if kk not in ['job', 'verbose', 'cbh_netcdf', 'add_gages', 'param_netcdf', 'no_filter_params']:
            if vv:
                bandit_log.info('Overriding configuration for {} with {}'.format(kk, vv))
                config.update_value(kk, vv)

    # Where to output the subset
    outdir = config.output_dir

    # The control file to use
    control_filename = config.control_filename

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

    # Load the control file
    ctl = ControlFile(control_filename)

    if ctl.has_dynamic_parameters:
        if config.dyn_params_dir:
            if os.path.exists(config.dyn_params_dir):
                dyn_params_dir = config.dyn_params_dir
            else:
                bandit_log.error('dyn_params_dir: {}, does not exist.'.format(config.dyn_params_dir))
                exit(2)
        else:
            bandit_log.error('Control file has dynamic parameters but dyn_params_dir is not specified in the config file')
            exit(2)

    # Load master list of valid parameters
    vpdb = ValidParams()

    # Build list of parameters required for the selected control file modules
    required_params = vpdb.get_params_for_modules(modules=ctl.modules.values())

    # TODO: make sure dynamic parameter filenames are correct
    # Write an updated control file
    # ctl.write('somefile')

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
    nhmparamdb_revision = git_version(paramdb_dir)
    bandit_log.info('Parameters based on NhmParamDb revision: {}'.format(nhmparamdb_revision))

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Read hru_nhm_to_local and hru_nhm_to_region
    # Create segment_nhm_to_local and segment_nhm_to_region

    # TODO: since hru_nhm_to_region and nhru_nhm_to_local are only needed for
    #       CBH files we should 'soft-fail' if the files are missing and just
    #       output a warning and turn off CBH output if it was selected.
    # hru_nhm_to_region = get_parameter('{}/hru_nhm_to_region.msgpack'.format(cbh_dir))
    # hru_nhm_to_local = get_parameter('{}/hru_nhm_to_local.msgpack'.format(cbh_dir))

    # Load the NHMparamdb
    print('Loading NHM ParamDb')
    pdb = ParamDb(merged_paramdb_dir)
    nhm_params = pdb.parameters
    nhm_global_dimensions = pdb.dimensions

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Get tosegment_nhm
    # NOTE: tosegment is now tosegment_nhm and the regional tosegment is gone.
    tosegment = nhm_params.get('tosegment').data
    nhm_seg = nhm_params.get('nhm_seg').data

    if args.verbose:
        print('Generating stream network from tosegment_nhm')

    # First check if any of the requested stream segments exist in the NHM.
    # An intersection of 0 elements can occur when all stream segments are
    # not included in the NHM (e.g. segments in Alaska).
    # NOTE: It's possible to have a stream segment that does not exist in
    #       tosegment but does exist in nhm_seg (e.g. standalone segment). So
    #       we use nhm_seg to verify at least one of the given segment(s) exist.
    if dsmost_seg and len(set(dsmost_seg).intersection(nhm_seg)) == 0:
        bandit_log.error('None of the requested stream segments exist in the NHM paramDb')
        exit(200)

    # Build the stream network
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
        dag_ds_subset = dag_ds.subgraph(uniq_seg_us).copy()

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
    try:
        # networkx 1.x
        toseg_idx = list(set(xx[0] for xx in dag_ds_subset.edges_iter()))
    except AttributeError:
        # networkx 2.x
        toseg_idx = list(set(xx[0] for xx in dag_ds_subset.edges))

    toseg_idx0 = [xx-1 for xx in toseg_idx]  # 0-based version of toseg_idx

    bandit_log.info('Number of segments in subset: {}'.format(len(toseg_idx)))

    # NOTE: With monolithic nhmParamDb files hru_segment becomes hru_segment_nhm and the regional hru_segments are gone.
    # 2019-09-16 PAN: This initially assumed hru_segment in the monolithic paramdb was ALWAYS
    #                 ordered 1..nhru. This is not always the case so the nhm_id parameter
    #                 needs to be loaded and used to map the nhm HRU ids to their
    #                 respective indices.
    hru_segment = nhm_params.get('hru_segment').data
    nhm_id = nhm_params.get('nhm_id').data

    nhm_id_to_idx = {}
    for ii, vv in enumerate(nhm_id):
        # keys are 1-based, values are 0-based
        nhm_id_to_idx[vv] = ii

    bandit_log.info('Number of NHM hru_segment entries: {}'.format(len(hru_segment)))

    # Create a dictionary mapping segments to HRUs
    seg_to_hru = {}
    for ii, vv in enumerate(hru_segment):
        # keys are 1-based, values in arrays are 1-based
        seg_to_hru.setdefault(vv, []).append(ii + 1)

    # Get HRU ids ordered by the segments in the model subset - entries are 1-based
    hru_order_subset = []
    for xx in toseg_idx:
        if xx in seg_to_hru:
            for yy in seg_to_hru[xx]:
                hru_order_subset.append(yy)
        else:
            bandit_log.warning('Stream segment {} has no HRUs connected to it.'.format(xx))
            # raise ValueError('Stream segment has no HRUs connected to it.')

    # Append the additional non-routed HRUs to the list
    if len(hru_noroute) > 0:
        for xx in hru_noroute:
            if hru_segment[nhm_id_to_idx[xx]] == 0:
            # if hru_segment[xx-1] == 0:
                bandit_log.info('User-supplied HRU {} is not connected to any stream segment'.format(xx))
                hru_order_subset.append(nhm_id_to_idx[xx] + 1)
                # hru_order_subset.append(xx)
            else:
                bandit_log.error('User-supplied HRU {} routes to stream segment {} - Skipping.'.format(xx,
                                                                                                       hru_segment[nhm_id_to_idx[xx]]))

    hru_order_subset0 = [xx - 1 for xx in hru_order_subset]

    bandit_log.info('Number of HRUs in subset: {}'.format(len(hru_order_subset)))

    # Use hru_order_subset to pull selected indices for parameters with nhru dimensions
    # hru_order_subset contains the in-order indices for the subset of hru_segments
    # toseg_idx contains the in-order indices for the subset of tosegments

    # Renumber the tosegment list
    new_tosegment = []

    # Map old DAG_subds indices to new
    for xx in toseg_idx:
        if list(dag_ds_subset.neighbors(xx))[0] in toseg_idx:
            new_tosegment.append(toseg_idx.index(list(dag_ds_subset.neighbors(xx))[0]) + 1)
        else:
            # Outlets should be assigned zero
            new_tosegment.append(0)

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
        for xx in hru_noroute:
            # if hru_segment[xx-1] == 0:
            if hru_segment[nhm_id_to_idx[xx]] == 0:
                new_hru_segment.append(0)

    bandit_log.info('Size of hru_segment for subset: {}'.format(len(new_hru_segment)))

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Subset hru_deplcrv
    hru_deplcrv = nhm_params.get('hru_deplcrv').data

    bandit_log.info('Size of NHM hru_deplcrv: {}'.format(len(hru_deplcrv)))

    # Get subset of hru_deplcrv using hru_order
    # A single snarea_curve can be referenced by multiple HRUs
    hru_deplcrv_subset = np.array(hru_deplcrv)[tuple(hru_order_subset0), ]
    uniq_deplcrv = list(set(hru_deplcrv_subset))
    uniq_deplcrv0 = [xx - 1 for xx in uniq_deplcrv]

    # Create new hru_deplcrv and renumber
    new_hru_deplcrv = [uniq_deplcrv.index(cc)+1 for cc in hru_deplcrv_subset]
    bandit_log.info('Size of hru_deplcrv for subset: {}'.format(len(new_hru_deplcrv)))

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Subset poi_gage_segment
    new_poi_gage_segment = []
    new_poi_gage_id = []
    new_poi_type = []

    if nhm_params.exists('poi_gage_segment'):
        poi_gage_segment = nhm_params.get('poi_gage_segment').tolist()
        bandit_log.info('Size of NHM poi_gage_segment: {}'.format(len(poi_gage_segment)))

        poi_gage_id = nhm_params.get('poi_gage_id').data
        poi_type = nhm_params.get('poi_type').data

        # We want to get the indices of the poi_gage_segments that match the
        # segments that are part of the subset. We can then use these
        # indices to subset poi_gage_id and poi_type.
        # The poi_gage_segment will need to be renumbered for the subset of segments.

        # To subset poi_gage_segment we have to lookup each segment in the subset

        # for ss in uniq_seg_us:
        try:
            # networkx 1.x
            for ss in nx.nodes_iter(dag_ds_subset):
                if ss in poi_gage_segment:
                    new_poi_gage_segment.append(toseg_idx.index(ss)+1)
                    new_poi_gage_id.append(poi_gage_id[poi_gage_segment.index(ss)])
                    new_poi_type.append(poi_type[poi_gage_segment.index(ss)])
        except AttributeError:
            # networkx 2.x
            for ss in dag_ds_subset.nodes:
                if ss in poi_gage_segment:
                    new_poi_gage_segment.append(toseg_idx.index(ss)+1)
                    new_poi_gage_id.append(poi_gage_id[poi_gage_segment.index(ss)])
                    new_poi_type.append(poi_type[poi_gage_segment.index(ss)])

        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Add any valid user-specified streamgage, nhm_seg pairs
        if addl_gages:
            for ss, vv in iteritems(addl_gages):
                if ss in new_poi_gage_id:
                    idx = new_poi_gage_id.index(ss)
                    bandit_log.warning('Existing NHM POI, {}, overridden on commandline (was {}, now {})'.format(ss, new_poi_gage_segment[idx],
                                                                                                              toseg_idx.index(vv)+1))
                    new_poi_gage_segment[idx] = toseg_idx.index(vv)+1
                    new_poi_type[idx] = 0
                elif toseg_idx.index(vv)+1 in new_poi_gage_segment:
                    sidx = new_poi_gage_segment.index(toseg_idx.index(vv)+1)
                    bandit_log.warning('User-specified streamgage ({}) has same nhm_seg ({}) as existing POI ({}), replacing streamgage ID'.format(ss, toseg_idx.index(vv)+1, new_poi_gage_id[sidx]))
                    new_poi_gage_id[sidx] = ss
                    new_poi_type[sidx] = 0
                elif vv not in seg_to_hru.keys():
                    bandit_log.warning('User-specified streamgage ({}) has nhm_seg={} which is not part of the model subset - Skipping.'.format(ss, vv))
                else:
                    new_poi_gage_id.append(ss)
                    new_poi_gage_segment.append(toseg_idx.index(vv)+1)
                    new_poi_type.append(0)
                    bandit_log.info('Added user-specified POI streamgage ({}) at nhm_seg={}'.format(ss, vv))

    # ==================================================================
    # ==================================================================
    # Process the parameters and create a parameter file for the subset
    params = list(nhm_params.keys())

    # Remove the POI-related parameters if we have no POIs
    if len(new_poi_gage_segment) == 0:
        bandit_log.warning('No POI gages found for subset; removing POI-related parameters.')

        for rp in ['poi_gage_id', 'poi_gage_segment', 'poi_type']:
            # params.pop(rp, None)
            try:
                params.remove(rp)
            except ValueError:
                print('ERROR: unable to remove {}'.format(rp))
                pass

    params.sort()

    dims = {}
    for kk in nhm_global_dimensions.values():
        dims[kk.name] = kk.size

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
            # if 'ndepl' not in dims:
            dims['ndepl'] = len(uniq_deplcrv0)
        elif dd == 'npoigages':
            dims[dd] = len(new_poi_gage_segment)

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Build a ParameterSet for output
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    new_ps = ParameterSet()

    for dd, dv in iteritems(dims):
        new_ps.dimensions.add(dd, dv)

        if dd == 'npoigages':
            # 20170217 PAN: nobs is missing from the paramdb but is necessary
            new_ps.dimensions.add('nobs', dv)

    new_params = list(required_params)

    # WARNING: 2019-04-23 PAN
    #          Very hacky way to remove parameters that shouldn't always get
    #          included. Need to figure out a better way.
    check_list = ['basin_solsta', 'gvr_hru_id', 'hru_solsta', 'humidity_percent',
                  'irr_type', 'obsout_segment', 'rad_conv', 'rain_code', 'hru_lon']

    for xx in check_list:
        if xx in new_params:
            if xx in ['basin_solsta', 'hru_solsta', 'rad_conv']:
                if not new_ps.dimensions.exists('nsol'):
                    new_params.remove(xx)
                elif new_ps.dimensions.get('nsol') == 0:
                    new_params.remove(xx)
            elif xx == 'humidity_percent':
                if not new_ps.dimensions.exists('nhumid'):
                    new_params.remove(xx)
                elif new_ps.dimensions.get('nhumid') == 0:
                    new_params.remove(xx)
            elif xx == 'irr_type':
                if not new_ps.dimensions.exists('nwateruse'):
                    new_params.remove(xx)
                elif new_ps.dimensions.get('nwateruse') == 0:
                    new_params.remove(xx)
            elif xx == 'gvr_hru_id':
                if ctl.get('mapOutON_OFF').values == 0:
                    new_params.remove(xx)
            elif xx in ['hru_lat', 'hru_lon', ]:
                if not nhm_params.exists(xx):
                    new_params.remove(xx)

    new_params.sort()
    for pp in params:
        if pp in new_params or args.no_filter_params:
            cparam = nhm_params.get(pp).tostructure()

            new_ps.parameters.add(cparam['name'])

            ndims = len(cparam['dimensions'])
            if args.verbose:
                sys.stdout.write('\r                                       ')
                sys.stdout.write('\rProcessing {} '.format(cparam['name']))
                sys.stdout.flush()

            # Get order of dimensions and total size for parameter
            dim_order = [None] * ndims

            for dd, dv in iteritems(cparam['dimensions']):
                dim_order[dv['position']] = dd

            for dd in dim_order:
                # self.parameters.get(varname).dimensions.add(dd, self.dimensions.get(dd).size)
                new_ps.parameters.get(cparam['name']).dimensions.add(dd, new_ps.dimensions.get(dd).size)

                new_ps.parameters.get(cparam['name']).datatype = cparam['datatype']

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

            new_ps.parameters.get(cparam['name']).data = outlist

    # Write the new parameter file
    header = ['Written by Bandit version {}'.format(__version__),
              'NhmParamDb revision: {}'.format(nhmparamdb_revision)]
    if args.param_netcdf:
        base_filename = os.path.splitext(param_filename)[0]
        param_filename = '{}.nc'.format(base_filename)
        new_ps.write_netcdf('{}/{}'.format(outdir, param_filename))
    else:
        new_ps.write_parameter_file('{}/{}'.format(outdir, param_filename), header=header)

    ctl.get('param_file').values = param_filename

    if args.verbose:
        sys.stdout.write('\n')
    #     sys.stdout.write('\r                                       ')
    #     sys.stdout.write('\r\tParameter file written: {}\n'.format('{}/{}'.format(outdir, param_filename)))
    sys.stdout.flush()

    # 2019-09-16 PAN: Nasty hack to handle parameter databases that may not have
    #                 a one-to-one match between index value and nhm_id.
    cparam = nhm_params.get('nhm_id').tostructure()
    hru_order_subset_nhm_id = np.array(cparam['data'])[tuple(hru_order_subset0),].ravel(order='F').tolist()


    if output_cbh:
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Subset the cbh files for the selected HRUs
        if len(hru_order_subset) > 0:
            if args.verbose:
                print('Processing CBH files')

            if os.path.splitext(cbh_dir)[1] == '.nc':
                cbh_hdl = CbhNetcdf(src_path=cbh_dir, st_date=st_date, en_date=en_date,
                                    nhm_hrus=hru_order_subset_nhm_id)
                                    # nhm_hrus=hru_order_subset)
            else:
                # Subset the hru_nhm_to_local mapping
                # TODO: This section will not work with the monolithic paramdb - remove
                hru_order_ss = OrderedDict()
                for kk in hru_order_subset:
                    hru_order_ss[kk] = hru_nhm_to_local[kk]

                cbh_hdl = CbhAscii(src_path=cbh_dir, st_date=st_date, en_date=en_date,
                                   nhm_hrus=hru_order_subset, indices=hru_order_ss,
                                   mapping=hru_nhm_to_region)

            if args.cbh_netcdf:
                # Pull the filename prefix off of the first file found in the
                # source netcdf CBH directory.
                file_it = glob.iglob(cbh_dir)
                cbh_prefix = os.path.basename(next(file_it)).split('_')[0]

                cbh_outfile = '{}/{}.nc'.format(outdir, cbh_prefix)
                cbh_hdl.write_netcdf(cbh_outfile)
                ctl.get('tmax_day').values = os.path.basename(cbh_outfile)
                ctl.get('tmin_day').values = os.path.basename(cbh_outfile)
                ctl.get('precip_day').values = os.path.basename(cbh_outfile)
            else:
                cbh_hdl.write_ascii()
            # bandit_log.info('{} written to: {}'.format(vv, '{}/{}.cbh'.format(outdir, vv)))
        else:
            bandit_log.error('No HRUs associated with the segments')

    if ctl.has_dynamic_parameters:
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Add dynamic parameters
        for cparam in ctl.dynamic_parameters:
            param_name = 'dyn_{}'.format(cparam)
            input_file = '{}/{}.nc'.format(dyn_params_dir, param_name)
            output_file = '{}/{}.param'.format(outdir, param_name)

            if not os.path.exists(input_file):
                bandit_log.warning('WARNING: CONUS dynamic parameter file: {}, does not exist... skipping'.format(input_file))
            else:
                if args.verbose:
                    print('Writing dynamic parameter {}'.format(cparam))

                mydyn = dyn_params.DynamicParameters(input_file, cparam, st_date, en_date, hru_order_subset_nhm_id)
                # mydyn = dyn_params.DynamicParameters(input_file, cparam, st_date, en_date, hru_order_subset)

                mydyn.read_netcdf()
                out_order = [kk for kk in hru_order_subset_nhm_id]
                # out_order = [kk for kk in hru_order_subset]
                for cc in ['day', 'month', 'year']:
                    out_order.insert(0, cc)

                header = ' '.join(map(str, out_order))

                # Output ASCII files
                out_ascii = open(output_file, 'w')
                out_ascii.write('{}\n'.format(cparam))
                out_ascii.write('{}\n'.format(header))
                out_ascii.write('####\n')
                mydyn.data.to_csv(out_ascii, columns=out_order, na_rep='-999',
                                  sep=' ', index=False, header=False, encoding=None, chunksize=50)
                out_ascii.close()

    # Write an updated control file to the output directory
    ctl.write('{}.bandit'.format(control_filename))

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Output model output variables
    # outvar = 'tmaxf'
    # workdir = '/Users/pnorton/Projects/National_Hydrology_Model/datasets/NHM_output'
    # input_file = '{}/{}.nc'.format(workdir, outvar)
    # output_file = '{}/{}.nc'.format(outdir, outvar)
    #
    # if not os.path.exists(input_file):
    #     bandit_log.warning('WARNING: CONUS output variable file: {}, does not exist... skipping'.format(input_file))
    # else:
    #     if args.verbose:
    #         print('Writing NHM model output variable {}'.format(outvar))
    #
    #     model_out_var = ModelOutput(input_file, outvar, st_date, en_date, hru_order_subset)
    #     model_out_var.read_netcdf()
    #     out_order = [kk for kk in hru_order_subset]
    #
    #     model_out_var.data.to_csv('{}/{}.csv'.format(outdir, outvar), columns=out_order, na_rep='-999',
    #                       sep=',', index=True, header=True, encoding=None, chunksize=50)
    #
    #     model_out_var.dataarray.to_netcdf('{}/{}.nc'.format(outdir, outvar),
    #                                       encoding={'time': {'_FillValue': None,
    #                                                          'calendar': 'standard'}})

    if output_streamflow:
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Download the streamgage information from NWIS
        if args.verbose:
            print('Downloading NWIS streamgage observations for {} stations'.format(len(new_poi_gage_id)))

        streamflow = prms_nwis.NWIS(gage_ids=new_poi_gage_id, st_date=st_date, en_date=en_date,
                                    verbose=args.verbose)
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
            # geo_shp.write_shapefile('{}/GIS/HRU_subset.shp'.format(outdir), 'hru_id_nat', hru_order_subset,
            geo_shp.write_shapefile('{}/GIS/HRU_subset.shp'.format(outdir), 'hru_id_nat', hru_order_subset_nhm_id,
                                    included_fields=['nhm_id', 'model_idx', 'region', 'hru_id_nat'])

            # geo_shp.write_shapefile3('{}/GIS/HRU_subset.gdb'.format(outdir), 'hru_id_nat', hru_order_subset)

            # geo_shp.filter_by_attribute('hru_id_nat', hru_order_subset)
            # geo_shp.write_shapefile2('{}/HRU_subset.shp'.format(outdir))
            # geo_shp.write_kml('{}/HRU_subset.kml'.format(outdir))

            # Output a shapefile of the selected stream segments
            # print('\tSegments')
            geo_shp.select_layer('nsegmentNationalIdentifier')
            geo_shp.write_shapefile('{}/GIS/Segments_subset.shp'.format(outdir), 'seg_id_nat', toseg_idx,
                                    included_fields=['seg_id_nat', 'model_idx', 'region'])

            # geo_shp.filter_by_attribute('seg_id_nat', uniq_seg_us)
            # geo_shp.write_shapefile2('{}/Segments_subset.shp'.format(outdir))

            del geo_shp

    bandit_log.info('========== END {} =========='.format(datetime.datetime.now().isoformat()))

    os.chdir(stdir)


if __name__ == '__main__':
    main()
