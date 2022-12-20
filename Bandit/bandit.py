#!/usr/bin/env python3

import argparse
import datetime
import errno
# import glob
import logging
import networkx as nx   # type: ignore
import numpy as np
import os
import sys

from collections import OrderedDict
from typing import List

import Bandit.bandit_cfg as bc
import Bandit.dynamic_parameters as dyn_params
import Bandit.prms_geo as prms_geo
import Bandit.prms_nwis as prms_nwis

from Bandit import __version__
from Bandit.bandit_helpers import parse_gages, set_date, subset_stream_network
from Bandit.git_version import git_commit, git_repo, git_branch, git_commit_url
from Bandit.model_output import ModelOutput
from Bandit.points_of_interest import POI

from pyPRMS.constants import HRU_DIMS
from pyPRMS.CbhNetcdf import CbhNetcdf
from pyPRMS.ControlFile import ControlFile
from pyPRMS.ParamDb import ParamDb
from pyPRMS.ParameterSet import ParameterSet
# from pyPRMS.ValidParams import ValidParams

__author__ = 'Parker Norton (pnorton@usgs.gov)'

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


def get_hru_and_seg_subset_maps(orig_hru_segment, orig_nhm_id, nhm_seg_subset, hru_noroute):
    # Create a dictionary mapping hru_segment segments to hru_segment 1-based indices filtered by
    # new_nhm_seg and hru_noroute.
    seg_to_hru = OrderedDict()
    hru_to_seg = OrderedDict()

    for ii, vv in enumerate(orig_hru_segment):
        # Contains both new_nhm_seg values and non-routed HRU values
        # keys are 1-based, values in arrays are 1-based
        hid = orig_nhm_id[ii]
        if vv in nhm_seg_subset:
            seg_to_hru.setdefault(vv, []).append(hid)
            hru_to_seg[hid] = vv
        elif hid in hru_noroute:
            if vv != 0:
                err_txt = f'User-supplied non-routed HRU {hid} that routes to stream segment {vv}; skipping.'
                bandit_log.error(err_txt)
            else:
                seg_to_hru.setdefault(vv, []).append(hid)
                hru_to_seg[hid] = vv

    return seg_to_hru, hru_to_seg


def get_output_order(hru_to_seg, seg_to_hru, orig_hru_segment, orig_nhm_id_to_idx,
                     new_nhm_seg, new_nhm_seg_to_idx1, hru_noroute, keep_hru_order=False):
    # HRU-related parameters can either be output with the legacy, segment-oriented order
    # or can be output maintaining their original HRU-relative order from the parameter database.
    if keep_hru_order:
        hru_order_subset = [kk for kk in hru_to_seg.keys()]

        new_hru_segment = [new_nhm_seg_to_idx1[kk] if kk in new_nhm_seg else 0 if kk == 0 else -1 for kk in
                           hru_to_seg.values()]
    else:
        # Get NHM HRU ids ordered by the segments in the model subset - indices are 1-based
        hru_order_subset = []
        for xx in new_nhm_seg:
            if xx in seg_to_hru:
                for yy in seg_to_hru[xx]:
                    hru_order_subset.append(yy)
            else:
                print(f'Segment {xx} has no HRUs connected to it')
                bandit_log.warning(f'Stream segment {xx} has no HRUs connected to it.')

        # Append the additional non-routed HRUs to the list
        if len(hru_noroute) > 0:
            for xx in hru_noroute:
                if orig_hru_segment[orig_nhm_id_to_idx[xx]] == 0:
                    bandit_log.info(f'User-supplied HRU {xx} is not connected to any stream segment')
                    hru_order_subset.append(xx)
                else:
                    err_txt = f'User-supplied HRU {xx} routes to stream segment ' + \
                              f'{orig_hru_segment[orig_nhm_id_to_idx[xx]]} - Skipping.'
                    bandit_log.error(err_txt)

        # Renumber the hru_segments for the subset
        new_hru_segment = []

        for xx in new_nhm_seg:
            if xx in seg_to_hru:
                for _ in seg_to_hru[xx]:
                    # The new indices should be 1-based from PRMS
                    new_hru_segment.append(new_nhm_seg_to_idx1[xx])

        # Append zeroes to new_hru_segment for each additional non-routed HRU
        if len(hru_noroute) > 0:
            for xx in hru_noroute:
                if orig_hru_segment[orig_nhm_id_to_idx[xx]] == 0:
                    new_hru_segment.append(0)

    return hru_order_subset, new_hru_segment


def resize_dims(src_global_dims, num_hru, num_seg, num_deplcrv, num_poi):
    dims = {kk.name: kk.size for kk in src_global_dims}

    # Resize dimensions to the model subset
    crap_dims = dims.copy()   # need a copy since we modify dims
    for dd, dv in crap_dims.items():
        # dimensions 'nmonths' and 'one' are never changed
        if dd in HRU_DIMS:
            dims[dd] = num_hru
        elif dd == 'nsegment':
            dims[dd] = num_seg
        elif dd == 'ndeplval':
            dims[dd] = num_deplcrv * 11
            dims['ndepl'] = num_deplcrv
        elif dd == 'npoigages':
            dims[dd] = num_poi
            dims['nobs'] = num_poi

    return dims


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
    parser.add_argument('--streamflow_netcdf', help='Enable netCDF output for streamflow file', action='store_true')
    parser.add_argument('--add_gages', metavar="KEY=VALUE", nargs='+',
                        help='Add arbitrary streamgages to POIs of form gage_id=segment. Segment must ' +
                             'exist in the model subset. Additional streamgages are marked as poi_type=0.')
    parser.add_argument('--no_filter_params',
                        help='Output all parameters regardless of modules selected', action='store_true')
    parser.add_argument('--keep_hru_order',
                        help='Keep HRUs in the relative order they occur in the paramdb', action='store_true')
    parser.add_argument('--hru_gis_id', help='Name of key/id for HRUs in geodatabase', nargs='?', type=str)
    parser.add_argument('--seg_gis_id', help='Name of key/id for segments in geodatabase', nargs='?', type=str)
    parser.add_argument('--hru_gis_layer', help='Name of geodatabase layer containing HRUs', nargs='?', type=str)
    parser.add_argument('--seg_gis_layer', help='Name of geodatabase layer containing Segments', nargs='?', type=str)
    parser.add_argument('--prms_version', help='Write PRMS version 5 or 6 parameter file', nargs='?',
                        default=5, type=int)
    args = parser.parse_args()

    stdir = os.getcwd()

    if args.job:
        if os.path.exists(args.job):
            # Change into job directory before running extraction
            os.chdir(args.job)
            # print('Working in directory: {}'.format(args.job))
        else:
            print(f'ERROR: Invalid jobs directory: {args.job}')
            exit(-1)

    bandit_log.info(f'========== START {datetime.datetime.now().isoformat()} ==========')

    addl_gages = None
    if args.add_gages:
        addl_gages = parse_gages(args.add_gages)
        bandit_log.info('Additionals streamgages specified on command line')

    config = bc.Cfg('bandit.cfg')

    # Override configuration variables with any command line parameters
    for kk, vv in args.__dict__.items():
        if config.exists(kk):
            if vv:
                bandit_log.info(f'Overriding configuration for {kk} with {vv}')
                config.update_value(kk, vv)

    # Where to output the subset
    outdir = config.output_dir

    # The control file to use
    # control_filename = config.control_filename

    # What to name the output parameter file
    param_filename = config.param_filename

    # Location of the NHM parameter database
    paramdb_dir = config.paramdb_dir

    # List of outlets
    dsmost_seg = config.outlets

    # List of upstream cutoffs
    uscutoff_seg = config.cutoffs

    # List of additional HRUs (have no route to segment within subset)
    hru_noroute = config.hru_noroute

    if args.prms_version == 6:
        args.cbh_netcdf = True
        args.param_netcdf = True
        args.streamflow_netcdf = True

    # Load the control file
    ctl = ControlFile(config.control_filename, version=args.prms_version)

    # dyn_params_dir = ''
    if ctl.has_dynamic_parameters:
        if config.dyn_params_dir:
            if not os.path.exists(config.dyn_params_dir):
                bandit_log.error(f'dyn_params_dir: {config.dyn_params_dir}, does not exist.')
                exit(2)
        else:
            bandit_log.error('Control file has dynamic parameters but dyn_params_dir is not specified ' +
                             'in the config file')
            exit(2)

    # Date range for pulling NWIS streamgage observations and CBH data
    st_date = set_date(config.start_date)
    en_date = set_date(config.end_date)

    # Adjust the start and end dates in the control file to reflect
    # date range from bandit config file
    ctl.get('start_time').values = [st_date.year, st_date.month, st_date.day, 0, 0, 0]
    ctl.get('end_time').values = [en_date.year, en_date.month, en_date.day, 0, 0, 0]

    # ===============================================================
    # params_file = '{}/{}'.format(merged_paramdb_dir, PARAMETERS_XML)

    # Output revision of NhmParamDb and the revision used by merged paramdb
    git_url = git_commit_url(paramdb_dir)
    nhmparamdb_revision = git_commit(paramdb_dir, length=7)
    bandit_log.info(f'Using parameter database from: {git_repo(paramdb_dir)}')
    bandit_log.info(f'Repo branch: {git_branch(paramdb_dir)}')
    bandit_log.info(f'Repo commit: {nhmparamdb_revision}')

    # Load the NHMparamdb
    print('Loading NHM ParamDb')
    pdb = ParamDb(paramdb_dir, verify=True)
    pdb.control = ctl

    if not args.no_filter_params:
        # Reduce the parameters to those required by the selected modules
        pdb.reduce_by_modules()

    # Default the various *ON_OFF variables to 0 (off)
    # The original values are needed to reduce parameters by module,
    # but it's best to disable them in the final control file since
    # no output variables are defined for them.
    disable_vars = ['basinOutON_OFF', 'mapOutON_OFF', 'nhruOutON_OFF',
                    'nsegmentOutON_OFF', 'nsubOutON_OFF']
    for vv in disable_vars:
        ctl.get(vv).values = '0'

    nhm_params = pdb.parameters
    nhm_global_dimensions = pdb.dimensions

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Get tosegment_nhm
    # NOTE: tosegment is now tosegment_nhm and the regional tosegment is gone.
    # Convert to list for fastest access to array
    # tosegment = nhm_params.get('tosegment_nhm').tolist()
    nhm_seg = nhm_params.get('nhm_seg').tolist()

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
    dag_ds = pdb.parameters.stream_network(tosegment='tosegment_nhm', seg_id='nhm_seg')

    bandit_log.debug('Number of NHM downstream nodes: {}'.format(dag_ds.number_of_nodes()))
    bandit_log.debug('Number of NHM downstream edges: {}'.format(dag_ds.number_of_edges()))

    if config.check_DAG:
        if not nx.is_directed_acyclic_graph(dag_ds):
            bandit_log.error('Cycles and/or loops found in stream network')

            for xx in nx.simple_cycles(dag_ds):
                bandit_log.error(f'Cycle found for segment {xx}')

    if args.verbose:
        print('\tExtracting model subset')

    dag_ds_subset = subset_stream_network(dag_ds, uscutoff_seg, dsmost_seg)

    # Segments in model subset
    new_nhm_seg = [ee[0] for ee in dag_ds_subset.edges]
    bandit_log.info(f'Number of segments in subset: {len(new_nhm_seg)}')

    # Using a dictionary mapping nhm_seg to 1-based index for speed
    new_nhm_seg_to_idx1 = OrderedDict((ss, ii+1) for ii, ss in enumerate(new_nhm_seg))

    # Generate the renumbered local tosegments (1-based with zero being an outlet)
    new_tosegment = [new_nhm_seg_to_idx1[ee[1]] if ee[1] in new_nhm_seg_to_idx1
                     else 0 for ee in dag_ds_subset.edges]

    # NOTE: With monolithic nhmParamDb files hru_segment becomes hru_segment_nhm and the
    #       regional hru_segments are gone.
    # 2019-09-16 PAN: This initially assumed hru_segment in the monolithic paramdb was ALWAYS
    #                 ordered 1..nhru. This is not always the case so the nhm_id parameter
    #                 needs to be loaded and used to map the nhm HRU ids to their
    #                 respective indices.
    hru_segment = nhm_params.get('hru_segment_nhm').tolist()
    nhm_id = nhm_params.get('nhm_id').tolist()
    nhm_id_to_idx = nhm_params.get('nhm_id').index_map
    bandit_log.info(f'Number of NHM hru_segment entries: {len(hru_segment)}')

    # Create a dictionaries mapping hru_segment segments to hru_segment 1-based indices filtered by
    # new_nhm_seg and hru_noroute.
    seg_to_hru, hru_to_seg = get_hru_and_seg_subset_maps(hru_segment, nhm_id, new_nhm_seg, hru_noroute)

    if set(hru_to_seg.values()) == set(hru_noroute):
        # This occurs when there are no ROUTED HRUs for any of the stream segments
        bandit_log.error('No HRUs associated with any of the segments; exiting.')
        exit(2)

    # HRU-related parameters can either be output with the legacy, segment-oriented order
    # or can be output maintaining their original HRU-relative order from the parameter database.
    hru_order_subset, new_hru_segment = get_output_order(hru_to_seg, seg_to_hru, hru_segment,
                                                         nhm_id_to_idx, new_nhm_seg,
                                                         new_nhm_seg_to_idx1, hru_noroute,
                                                         keep_hru_order=args.keep_hru_order)

    bandit_log.info(f'Number of HRUs in subset: {len(hru_order_subset)}')
    bandit_log.info(f'Size of hru_segment for subset: {len(new_hru_segment)}')

    # Use hru_order_subset to pull selected indices for parameters with nhru dimensions
    # hru_order_subset contains the in-order indices for the subset of hru_segments
    # toseg_idx contains the in-order indices for the subset of tosegments
    # --------------------------------------------------------------------------

    # ==========================================================================
    # ==========================================================================
    # Get subset of hru_deplcrv using hru_order
    # A single snarea_curve can be referenced by multiple HRUs
    hru_deplcrv_subset = nhm_params.get_subset('hru_deplcrv', hru_order_subset)

    # noinspection PyTypeChecker
    uniq_deplcrv: List = np.unique(hru_deplcrv_subset).tolist()  # type: ignore

    # @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Subset poi_gage_segment
    new_poi_gage_segment = []
    new_poi_gage_id = []
    new_poi_type = []

    if nhm_params.exists('poi_gage_segment'):
        poi_gage_segment = nhm_params.get('poi_gage_segment').tolist()
        bandit_log.info(f'Size of NHM poi_gage_segment: {len(poi_gage_segment)}')

        poi_gage_id = nhm_params.get('poi_gage_id').tolist()
        poi_type = nhm_params.get('poi_type').tolist()

        # We want to get the indices of the poi_gage_segments that match the
        # segments that are part of the subset. We can then use these
        # indices to subset poi_gage_id and poi_type.
        # The poi_gage_segment will need to be renumbered for the subset of segments.

        # To subset poi_gage_segment we have to look up each segment in the subset
        nhm_seg_dict = nhm_params.get('nhm_seg').index_map
        poi_gage_dict = nhm_params.get('poi_gage_segment').index_map

        for ss in new_nhm_seg:
            sidx = nhm_seg_dict[ss] + 1
            if sidx in poi_gage_segment:
                # print('   {}'.format(poi_gage_segment.index(sidx)))
                new_poi_gage_segment.append(new_nhm_seg_to_idx1[sidx])
                new_poi_gage_id.append(poi_gage_id[poi_gage_dict[sidx]])
                new_poi_type.append(poi_type[poi_gage_dict[sidx]])

        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Add any valid user-specified streamgage, nhm_seg pairs
        if addl_gages:
            for ss, vv in addl_gages.items():
                if ss in new_poi_gage_id:
                    idx = new_poi_gage_id.index(ss)
                    warn_txt = f'Existing NHM POI, {ss}, overridden on commandline ' + \
                               f'(was {new_poi_gage_segment[idx]}, now {new_nhm_seg_to_idx1[vv]})'
                    bandit_log.warning(warn_txt)
                    new_poi_gage_segment[idx] = new_nhm_seg_to_idx1[vv]
                    new_poi_type[idx] = 0
                elif new_nhm_seg_to_idx1[vv] in new_poi_gage_segment:
                    sidx = new_poi_gage_segment.index(new_nhm_seg_to_idx1[vv])
                    warn_txt = f'User-specified streamgage ({ss}) ' + \
                               f'has same nhm_seg ({new_nhm_seg_to_idx1[vv]}) ' + \
                               f'as existing POI ({new_poi_gage_id[sidx]}); replacing streamgage ID'
                    bandit_log.warning(warn_txt)
                    new_poi_gage_id[sidx] = ss
                    new_poi_type[sidx] = 0
                elif vv not in seg_to_hru.keys():
                    warn_txt = f'User-specified streamgage ({ss}) has nhm_seg={vv} which is not part ' + \
                               f'of the model subset; skipping.'
                    bandit_log.warning(warn_txt)
                else:
                    new_poi_gage_id.append(ss)
                    new_poi_gage_segment.append(new_nhm_seg_to_idx1[vv])
                    new_poi_type.append(0)
                    bandit_log.info(f'Added user-specified POI streamgage ({ss}) at nhm_seg={vv}')

    # ==================================================================
    # ==================================================================
    # Process the parameters and create a parameter file for the subset
    params = list(nhm_params.keys())

    # Remove the POI-related parameters if we have no POIs
    if len(new_poi_gage_segment) == 0:
        bandit_log.warning('No POI gages found for subset; removing POI-related parameters.')

        for rp in ['poi_gage_id', 'poi_gage_segment', 'poi_type']:
            if rp in params:
                params.remove(rp)

    params.sort()

    # Build dictionary of resized dimensions for the model subset
    dims = resize_dims(src_global_dims=nhm_global_dimensions.values(),
                       num_hru=len(hru_order_subset),
                       num_seg=len(new_nhm_seg),
                       num_deplcrv=len(uniq_deplcrv),
                       num_poi=len(new_poi_gage_segment))

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Build a ParameterSet for extracted model
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    new_ps = ParameterSet()
    new_global_dims = new_ps.dimensions

    # Add the global dimensions
    for dd, dv in dims.items():
        new_ps.dimensions.add(dd, dv)

    for pp in params:
        src_param = nhm_params.get(pp)

        if args.verbose:
            sys.stdout.write('\r                                       ')
            sys.stdout.write(f'\rProcessing {src_param.name} ')
            sys.stdout.flush()

        new_ps.parameters.add(name=pp, info=src_param)
        cnew_param = new_ps.parameters.get(pp)

        ndims = src_param.ndims
        dim_order = list(src_param.dimensions.keys())

        for dd in dim_order:
            cnew_param.dimensions.add(dd, new_global_dims.get(dd).size)

        first_dimension = dim_order[0]
        outdata = None

        # Write out the data for the parameter
        if ndims == 1:
            # 1D Parameters
            if first_dimension == 'one':
                outdata = src_param.data
            elif first_dimension == 'nsegment':
                if pp in ['tosegment']:
                    outdata = np.array(new_tosegment)
                else:
                    outdata = nhm_params.get_subset(pp, new_nhm_seg)
            elif first_dimension == 'ndeplval':
                # snarea_thresh - this is really a 2D in disguise, however,
                # it is stored in C-order unlike other 2D arrays
                outdata = nhm_params.get_subset(pp, hru_order_subset)
            elif first_dimension == 'npoigages':
                if pp == 'poi_gage_segment':
                    outdata = np.array(new_poi_gage_segment)
                elif pp == 'poi_gage_id':
                    outdata = np.array(new_poi_gage_id)
                elif pp == 'poi_type':
                    outdata = np.array(new_poi_type)
                else:
                    bandit_log.error(f'Unkown parameter, {pp}, with dimensions {first_dimension}')
            elif first_dimension in HRU_DIMS:
                if pp == 'hru_deplcrv':
                    outdata = nhm_params.get_subset(pp, hru_order_subset)
                elif pp == 'hru_segment':
                    outdata = np.array(new_hru_segment)
                else:
                    outdata = nhm_params.get_subset(pp, hru_order_subset)
            else:
                bandit_log.error(f'No rules to handle dimension {first_dimension}')
        elif ndims == 2:
            # 2D Parameters
            if first_dimension == 'nsegment':
                outdata = nhm_params.get_subset(pp, new_nhm_seg)
            elif first_dimension in HRU_DIMS:
                outdata = nhm_params.get_subset(pp, hru_order_subset)
            else:
                err_txt = f'No rules to handle 2D parameter, {pp}, which contains dimension {first_dimension}'
                bandit_log.error(err_txt)

        cnew_param.data = outdata

    # Write the new parameter file
    header = [f'Written by Bandit version {__version__}',
              f'ParamDb revision: {git_url}']
    if args.param_netcdf:
        base_filename = os.path.splitext(param_filename)[0]
        param_filename = f'{base_filename}.nc'
        new_ps.write_netcdf(f'{outdir}/{param_filename}')
    else:
        print(f'\nWriting version {args.prms_version} parameter file')
        new_ps.write_parameter_file(f'{outdir}/{param_filename}', header=header, prms_version=args.prms_version)

    ctl.get('param_file').values = param_filename

    if args.verbose:
        sys.stdout.write('\n')
    #     sys.stdout.write('\r                                       ')
    #     sys.stdout.write('\r\tParameter file written: {}\n'.format('{}/{}'.format(outdir, param_filename)))
    sys.stdout.flush()

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Write CBH files
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    if config.output_cbh:
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Subset the cbh files for the selected HRUs
        if args.verbose:
            print('Processing CBH files')

        # Read the CBH source file
        if os.path.splitext(config.cbh_dir)[1] == '.nc':
            cbh_hdl = CbhNetcdf(src_path=config.cbh_dir, st_date=st_date, en_date=en_date,
                                nhm_hrus=hru_order_subset)
        else:
            raise ValueError('Missing netcdf CBH files')

        if args.cbh_netcdf:
            # Pull the filename prefix off of the first file found in the
            # source netcdf CBH directory.
            # file_it = glob.iglob(config.cbh_dir)
            # cbh_prefix = os.path.basename(next(file_it)).split('_')[0]

            # cbh_outfile = f'{outdir}/{cbh_prefix}.nc'
            cbh_outfile = f'{outdir}/cbh.nc'
            cbh_hdl.write_netcdf(cbh_outfile, variables=list(config.cbh_var_map.keys()))

            # Set the control file variables for the CBH files
            for cfv in config.cbh_var_map.values():
                ctl.get(cfv).values = os.path.basename(cbh_outfile)

        else:
            for cvar, cfv in config.cbh_var_map.items():
                print(f'--- {cvar}')
                cfile = ctl.get(cfv).values
                cbh_hdl.write_ascii(cfile, variable=cvar)
            # cbh_hdl.write_ascii(variables=list(config.cbh_var_map.keys()))

            # Set the control file variables for the CBH files
            # for cbhvar, cfv in config.cbh_var_map.items():
            #     ctl.get(cfv).values = f'{cbhvar}.cbh'

        # bandit_log.info('{} written to: {}'.format(vv, '{}/{}.cbh'.format(outdir, vv)))

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Write output variables
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # 2019-08-07 PAN: first prototype for extractions of output variables
    if config.include_model_output:
        # TODO: 2020-03-12 PAN - this is brittle, fix it.
        # TODO: 2022-06-30 PAN - add option to write netCDF format
        seg_vars = ['seginc_gwflow', 'seginc_potet', 'seginc_sroff', 'seginc_ssflow',
                    'seginc_swrad', 'segment_delta_flow', 'seg_gwflow', 'seg_inflow',
                    'seg_lateral_inflow', 'seg_outflow', 'seg_sroff', 'seg_ssflow',
                    'seg_upstream_inflow']

        # if len(hru_order_subset) > 0 or len(new_nhm_seg) > 0:
        try:
            os.makedirs(f'{outdir}/model_output')
            print('Creating directory model_output, for model output variables')
        except OSError:
            print('Using existing model_output directory for output variables')

        for vv in config.output_vars:
            if args.verbose:
                sys.stdout.write('\r                                                  ')
                sys.stdout.write(f'\rProcessing output variable: {vv} ')
                sys.stdout.flush()

            filename = f'{config.output_vars_dir}/{vv}.nc'

            try:
                if vv in seg_vars:
                    mod_out = ModelOutput(filename=filename, varname=vv, startdate=st_date, enddate=en_date,
                                          nhm_segs=new_nhm_seg)
                    mod_out.write_csv(f'{outdir}/model_output')
                else:
                    mod_out = ModelOutput(filename=filename, varname=vv, startdate=st_date, enddate=en_date,
                                          nhm_hrus=hru_order_subset)
                    mod_out.write_csv(f'{outdir}/model_output')
            except FileNotFoundError:
                bandit_log.warning(f'Model output variable, {vv}, does not exist; skipping.')

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Write dynamic parameters
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    if ctl.has_dynamic_parameters:
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Add dynamic parameters
        for cparam in ctl.dynamic_parameters:
            param_name = f'dyn_{cparam}'
            input_file = f'{config.dyn_params_dir}/{param_name}.nc'
            output_file = f'{outdir}/{param_name}.param'

            if not os.path.exists(input_file):
                warn_txt = f'WARNING: CONUS dynamic parameter file: {input_file}, does not exist... skipping'
                bandit_log.warning(warn_txt)
            else:
                if args.verbose:
                    print(f'Writing dynamic parameter {cparam}')

                mydyn = dyn_params.DynamicParameters(input_file, cparam, st_date, en_date, hru_order_subset)
                # mydyn = dyn_params.DynamicParameters(input_file, cparam, st_date, en_date, hru_order_subset)

                mydyn.read_netcdf()
                out_order = [kk for kk in hru_order_subset]
                # out_order = [kk for kk in hru_order_subset]
                for cc in ['day', 'month', 'year']:
                    out_order.insert(0, cc)

                header = ' '.join(map(str, out_order))

                # Output ASCII files
                out_ascii = open(output_file, 'w')
                out_ascii.write(f'{cparam}\n')
                out_ascii.write(f'{header}\n')
                out_ascii.write('####\n')
                mydyn.data.to_csv(out_ascii, columns=out_order, na_rep='-999',
                                  sep=' ', index=False, header=False, encoding=None, chunksize=50)
                out_ascii.close()

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Write control file
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    ctl.write(f'{config.control_filename}.bandit')

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Write streamflow
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    if config.output_streamflow:
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Download the streamgage information from NWIS
        if len(new_poi_gage_id) > 0:
            if args.verbose:
                print(f'Retrieving streamgage observations for {len(new_poi_gage_id)} stations')

            if config.exists('poi_dir') and config.poi_dir != '':
                bandit_log.info('Retrieving POIs from local HYDAT and NWIS netcdf files')
                streamflow = POI(src_path=config.poi_dir, st_date=st_date, en_date=en_date,
                                 gage_ids=new_poi_gage_id, verbose=args.verbose)
            else:
                # Default to retrieving only NWIS stations from waterservices.usgs.gov
                bandit_log.info('No poi_dir: retrieving only NWIS POIs from online NWIS service.')
                streamflow = prms_nwis.NWIS(gage_ids=new_poi_gage_id, st_date=st_date, en_date=en_date,
                                            verbose=args.verbose)
                streamflow.get_daily_streamgage_observations()

            if args.streamflow_netcdf:
                streamflow.write_netcdf(filename=f'{outdir}/{config.streamflow_filename}.nc')
            else:
                streamflow.write_ascii(filename=f'{outdir}/{config.streamflow_filename}')
        else:
            # TODO: 2021-03-30 PAN - prms_nwis has a mechanism for creating the sf_data file
            #       with a dummy record when no POIs are provided
            print('Writing dummy streamflow data file')
            streamflow = prms_nwis.NWIS(gage_ids=None, st_date=st_date, en_date=en_date, verbose=args.verbose)
            streamflow.get_daily_streamgage_observations()
            streamflow.write_ascii(filename=f'{config.streamflow_filename}')
            bandit_log.info(f'No POIs exist in model subset; dummy data written.')

    # *******************************************
    # Create a shapefile of the selected HRUs
    if config.output_shapefiles:
        if args.verbose:
            print('-'*40)
            print('Writing shapefiles for model subset')

        if not os.path.exists(config.geodatabase_filename):
            bandit_log.error(f'Source GIS file, {config.geodatabase_filename}, '
                             f'does not exist. Shapefiles will not be created')
        else:
            geo_shp = prms_geo.Geo(config.geodatabase_filename)

            # Create GIS subdirectory if it doesn't already exist
            gis_dir = f'{outdir}/GIS'
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
            if args.verbose:
                print(f'Layers: {config.hru_gis_layer}, {config.seg_gis_layer}')
                print(f'IDs: {config.hru_gis_id}, {config.seg_gis_id}')

            # add nhm_id to included fields to include v1.0 national ids
            geo_shp.select_layer(config.hru_gis_layer)
            geo_shp.write_shapefile(f'{outdir}/GIS/HRU_subset.shp', config.hru_gis_id, hru_order_subset,
                                    included_fields=['model_idx', config.hru_gis_id])

            geo_shp.select_layer(config.seg_gis_layer)
            geo_shp.write_shapefile(f'{outdir}/GIS/Segments_subset.shp', config.seg_gis_id, new_nhm_seg,
                                    included_fields=[config.seg_gis_id, 'model_idx'])

            # Original code
            # geo_shp.select_layer('nhru')
            #        geo_shp.write_shapefile('{}/GIS/HRU_subset.shp'.format(outdir), 'hru_id_nat', hru_order_subset,
            # geo_shp.write_shapefile('{}/GIS/HRU_subset.shp'.format(outdir), 'hru_id_nat', hru_order_subset,
            #                         included_fields=['nhm_id', 'model_idx', 'region', 'hru_id_nat'])

            #        geo_shp.write_shapefile3('{}/GIS/HRU_subset.gdb'.format(outdir), 'hru_id_nat', hru_order_subset)

            #        geo_shp.filter_by_attribute('hru_id_nat', hru_order_subset)
            #        geo_shp.write_shapefile2('{}/HRU_subset.shp'.format(outdir))
            #        geo_shp.write_kml('{}/HRU_subset.kml'.format(outdir))

            # Output a shapefile of the selected stream segments
            # print('\tSegments')
            # geo_shp.select_layer('nsegmentNationalIdentifier')
            # geo_shp.write_shapefile('{}/GIS/Segments_subset.shp'.format(outdir), 'seg_id_nat', new_nhm_seg,
            #                         included_fields=['seg_id_nat', 'model_idx', 'region'])

            #       geo_shp.filter_by_attribute('seg_id_nat', uniq_seg_us)
            #       geo_shp.write_shapefile2('{}/Segments_subset.shp'.format(outdir))

            del geo_shp

    bandit_log.info(f'========== END {datetime.datetime.now().isoformat()} ==========')

    os.chdir(stdir)


if __name__ == '__main__':
    main()
