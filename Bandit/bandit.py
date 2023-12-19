#!/usr/bin/env python3

import argparse
import datetime
import errno
import logging
import networkx as nx   # type: ignore
import numpy as np
import os
import sys
import time

from collections import OrderedDict
from typing import List

from rich.console import Console
from rich import pretty

import Bandit.bandit_cfg as bc
import Bandit.dynamic_parameters as dyn_params
import Bandit.prms_nwis as prms_nwis

from Bandit import __version__
from Bandit.bandit_helpers import (parse_gages, set_date, subset_stream_network, get_hru_and_seg_subset_maps,
                                   get_output_order, get_poi_subset, resize_dims)
from Bandit.git_version import git_commit, git_repo, git_branch, git_commit_url
from Bandit.model_output import ModelOutput
from Bandit.points_of_interest import POI

from pyPRMS.constants import HRU_DIMS
from pyPRMS.metadata.metadata import MetaData
from pyPRMS import CbhNetcdf
from pyPRMS import ControlFile
from pyPRMS import ParamDb
from pyPRMS import Parameters

import pyogrio as pyg  # type: ignore
import warnings
# warnings.filterwarnings('ignore', category=RuntimeWarning)
warnings.filterwarnings('ignore', message=r'.*Measured \(M\) geometry types are not supported.*')
warnings.filterwarnings('ignore',
                        message='.*Column names longer than 10 characters will be truncated when saved to ESRI Shapefile*')
warnings.filterwarnings('ignore', message='.*Slicing with an out-of-order index is generating 10 times more chunks.*')
warnings.filterwarnings('ignore', message=r'.*organizePolygons\(\) received a polygon with more than 100 parts.*')
# from pyogrio import list_drivers, list_layers, read_info, read_dataframe, write_dataframe

# Rich library
pretty.install()
con = Console()

__author__ = 'Parker Norton (pnorton@usgs.gov)'

# Setup the logging
bandit_log = logging.getLogger('bandit')
bandit_log.setLevel(logging.DEBUG)

log_fmt = logging.Formatter('%(levelname)s: %(name)s: %(message)s')

# Handler for console logs
clog = logging.StreamHandler()
clog.setLevel(logging.ERROR)
clog.setFormatter(log_fmt)

bandit_log.addHandler(clog)


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

    # Handler for logging to file
    flog = logging.FileHandler(f'{os.getcwd()}/bandit.log')
    flog.setLevel(logging.DEBUG)
    flog.setFormatter(log_fmt)

    bandit_log.addHandler(flog)

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

    # Load PRMS metadata
    prms_meta = MetaData(verbose=False).metadata

    # Load the control file
    ctl = ControlFile(config.control_filename, metadata=prms_meta, version=args.prms_version)

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
    ctl.get('start_time').values = st_date
    ctl.get('end_time').values = en_date

    # Output revision of NhmParamDb
    git_url = git_commit_url(paramdb_dir)
    nhmparamdb_revision = git_commit(paramdb_dir, length=7)
    bandit_log.info(f'Using parameter database from: {git_repo(paramdb_dir)}')
    bandit_log.info(f'Repo branch: {git_branch(paramdb_dir)}')
    bandit_log.info(f'Repo commit: {nhmparamdb_revision}')

    # Load the NHMparamdb
    if args.verbose:
        con.print('Loading NHM ParamDb', style='green4')

    pdb = ParamDb(paramdb_dir=paramdb_dir, metadata=prms_meta, verbose=args.verbose)
    pdb.control = ctl

    # Add defaults for parameters that are missing but required for the select modules
    pdb.add_missing_parameters()

    if not args.no_filter_params:
        # Reduce the parameters to those required by the selected modules
        pdb.remove(pdb.unneeded_parameters)

    # Default the various *ON_OFF variables to 0 (off)
    # The original values are needed to reduce parameters by module,
    # but it's best to disable them in the final control file since
    # no output variables are defined for them.
    disable_vars = ['basinOutON_OFF', 'mapOutON_OFF', 'nhruOutON_OFF',
                    'nsegmentOutON_OFF', 'nsubOutON_OFF']
    for vv in disable_vars:
        ctl.get(vv).values = 0

    nhm_global_dimensions = pdb.dimensions

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Get tosegment_nhm
    # Convert to list for fastest access to array
    nhm_seg = pdb.get('nhm_seg').tolist()

    if args.verbose:
        con.print('Generating stream network', style='green4')

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
    dag_ds = pdb.stream_network(tosegment='tosegment_nhm', seg_id='nhm_seg')

    bandit_log.debug(f'Number of NHM downstream nodes: {dag_ds.number_of_nodes()}')
    bandit_log.debug(f'Number of NHM downstream edges: {dag_ds.number_of_edges()}')

    if config.check_DAG:
        if not nx.is_directed_acyclic_graph(dag_ds):
            bandit_log.error('Cycles and/or loops found in stream network')

            for xx in nx.simple_cycles(dag_ds):
                bandit_log.error(f'Cycle found for segment {xx}')

    if args.verbose:
        con.print('    Extracting model subset', style='blue')

    dag_ds_subset = subset_stream_network(dag_ds, uscutoff_seg, dsmost_seg)

    # Segments in model subset
    new_nhm_seg = [ee[0] for ee in dag_ds_subset.edges]
    bandit_log.info(f'Number of segments in subset: {len(new_nhm_seg)}')

    # Using a dictionary mapping nhm_seg to 1-based index for speed
    new_nhm_seg_to_idx1 = OrderedDict((ss, ii+1) for ii, ss in enumerate(new_nhm_seg))

    # Generate the renumbered local tosegments (1-based with zero being an outlet)
    new_tosegment = [new_nhm_seg_to_idx1[ee[1]] if ee[1] in new_nhm_seg_to_idx1
                     else 0 for ee in dag_ds_subset.edges]

    # 2019-09-16 PAN: This initially assumed hru_segment in the monolithic paramdb was ALWAYS
    #                 ordered 1..nhru. This is not always the case so the nhm_id parameter
    #                 needs to be loaded and used to map the nhm HRU ids to their
    #                 respective indices.
    hru_segment = pdb.get('hru_segment_nhm').tolist()
    nhm_id = pdb.get('nhm_id').tolist()
    nhm_id_to_idx = pdb.get('nhm_id').index_map
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
    # new_hru_segment contains the in-order indices for the subset of tosegments
    # --------------------------------------------------------------------------

    # ==========================================================================
    # ==========================================================================
    # Get subset of hru_deplcrv using hru_order_subset
    # A single snarea_curve can be referenced by multiple HRUs
    hru_deplcrv_subset = pdb.get_subset('hru_deplcrv', hru_order_subset)

    # noinspection PyTypeChecker
    uniq_deplcrv: List = np.unique(hru_deplcrv_subset).tolist()  # type: ignore

    # @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Subset poi_gage_segment
    new_poi_gage_segment, new_poi_gage_id, new_poi_type = get_poi_subset(pdb, new_nhm_seg,
                                                                         new_nhm_seg_to_idx1,
                                                                         seg_to_hru,
                                                                         addl_gages=addl_gages)

    # ==================================================================
    # ==================================================================
    # Process the parameters and create a parameter file for the subset
    params = list(pdb.keys())

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
    # Build Parameters for extracted model
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    new_ps = Parameters(metadata=prms_meta)

    # Add the global dimensions
    for dd, dv in dims.items():
        new_ps.dimensions.add(dd, dv)

    for pp in params:
        src_param = pdb.get(pp)

        new_ps.add(name=pp)
        cnew_param = new_ps.get(pp)

        ndims = src_param.ndim
        dim_order = list(src_param.dimensions.keys())

        first_dimension = dim_order[0]
        outdata = None

        # Write out the data for the parameter
        if ndims == 0:
            # Scalar parameters
            outdata = src_param.data
        elif ndims == 1:
            # 1D Parameters
            # if first_dimension == 'one':
            #     outdata = src_param.data
            if first_dimension == 'nsegment':
                if pp in ['tosegment']:
                    outdata = np.array(new_tosegment)
                else:
                    outdata = pdb.get_subset(pp, new_nhm_seg)
            elif first_dimension == 'ndeplval':
                # snarea_thresh - this is really a 2D in disguise, however,
                # it is stored in C-order unlike other 2D arrays
                outdata = pdb.get_subset(pp, hru_order_subset)
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
                    outdata = pdb.get_subset(pp, hru_order_subset)
                elif pp == 'hru_segment':
                    outdata = np.array(new_hru_segment)
                else:
                    outdata = pdb.get_subset(pp, hru_order_subset)
            else:
                bandit_log.error(f'No rules to handle dimension {first_dimension}')
        elif ndims == 2:
            # 2D Parameters
            if first_dimension == 'nsegment':
                outdata = pdb.get_subset(pp, new_nhm_seg)
            elif first_dimension in HRU_DIMS:
                outdata = pdb.get_subset(pp, hru_order_subset)
            else:
                err_txt = f'No rules to handle 2D parameter, {pp}, which contains dimension {first_dimension}'
                bandit_log.error(err_txt)

        cnew_param.data = outdata

    # Write the new parameter file
    header = [f'Written by Bandit version {__version__}',
              f'ParamDb revision: {git_url}']
    if args.param_netcdf:
        # TODO: 2023-11-13 PAN - add version info and prms version as global attributes
        base_filename = os.path.splitext(param_filename)[0]
        param_filename = f'{base_filename}.nc'
        new_ps.write_parameter_netcdf(f'{outdir}/{param_filename}')
    else:
        if args.verbose:
            con.print(f'\nWriting version {args.prms_version} parameter file', style='green4')
        new_ps.write_parameter_file(f'{outdir}/{param_filename}', header=header, prms_version=args.prms_version)

    ctl.get('param_file').values = param_filename

    if args.verbose:
        sys.stdout.write('\n')
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
            con.print('Processing CBH files', style='green4')

        # Read the CBH source file
        if os.path.splitext(config.cbh_dir)[1] == '.nc':
            cbh_hdl = CbhNetcdf(src_path=config.cbh_dir, st_date=st_date, en_date=en_date,
                                nhm_hrus=hru_order_subset)
        else:
            raise ValueError('Missing netcdf CBH files')

        if args.cbh_netcdf:
            cbh_outfile = f'{outdir}/cbh.nc'

            global_attrs = dict(bandit_version=__version__, paramdb_url=git_url)
            cbh_hdl.write_netcdf(cbh_outfile, variables=list(config.cbh_var_map.keys()),
                                 global_attrs=global_attrs)

            # Set the control file variables for the CBH files
            for cfv in config.cbh_var_map.values():
                ctl.get(cfv).values = os.path.basename(cbh_outfile)

        else:
            for cvar, cfv in config.cbh_var_map.items():
                if args.verbose:
                    con.print(f'--- {cvar}')

                cfile = ctl.get(cfv).values
                cbh_hdl.write_ascii(cfile, variable=cvar)

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
                    'seg_upstream_inflow', 'seg_cum_gwflow', 'seg_cum_potet',
                    'seg_cum_sroff', 'seg_cum_ssflow']

        try:
            os.makedirs(f'{outdir}/model_output')
            bandit_log.info('Creating directory model_output, for model output variables')
        except OSError:
            bandit_log.info('Using existing model_output directory for output variables')

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

                mydyn.read_netcdf()
                out_order = [kk for kk in hru_order_subset]

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
    ctl.write(f'{os.path.basename(config.get_value("control_filename"))}.bandit')

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Write streamflow
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    if config.output_streamflow:
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Download the streamgage information from NWIS
        if len(new_poi_gage_id) > 0:
            if args.verbose:
                con.print(f'Retrieving streamgage observations for {len(new_poi_gage_id)} stations', style='green4')

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
            if args.verbose:
                con.print('No POIs exist in model subset; writing dummy data', style='gold3')
            streamflow = prms_nwis.NWIS(gage_ids=None, st_date=st_date, en_date=en_date, verbose=args.verbose)
            streamflow.get_daily_streamgage_observations()
            streamflow.write_ascii(filename=f'{config.streamflow_filename}')
            bandit_log.info(f'No POIs exist in model subset; dummy data written.')

    # *******************************************
    # Create a shapefile of the selected HRUs
    if config.output_shapefiles:
        stime = time.time()

        if args.verbose:
            print('-'*40)
            con.print('Writing shapefiles for model subset', style='green4')

        if len(config.gis) == 0 or not os.path.exists(config.gis['src_filename']):
            bandit_log.error(f'Source GIS file'
                             f'does not exist. Shapefiles will not be created')
        else:
            # Create GIS subdirectory if it doesn't already exist
            gis_dir = f'{outdir}/GIS'
            try:
                os.makedirs(gis_dir)
            except OSError as exception:
                if exception.errno != errno.EEXIST:
                    raise
                else:
                    pass

            geo_outfile = f'{gis_dir}/model_layers.{config.gis["dst_extension"]}'

            for kk, vv in config.gis['layers'].items():
                vv['include_fields'].extend([vv['key']])

                if vv['type'] == 'nhru':
                    geo_file = pyg.read_dataframe(config.gis['src_filename'], layer=vv['layer'],
                                                  columns=vv['include_fields'], force_2d=True,
                                                  where=f'{vv["key"]} >= {min(hru_order_subset)} AND {vv["key"]} <= {max(hru_order_subset)}')
                    bb = geo_file[geo_file[vv['key']].isin(hru_order_subset)]
                    bb = bb.rename(columns={vv['key']: 'nhm_id'})
                    local_ids = new_ps.get_dataframe('nhm_id').reset_index()
                    bb = bb.merge(local_ids, on='nhm_id')

                    if config.gis["dst_extension"] == 'gpkg':
                        bb.to_file(geo_outfile, layer=vv['type'], driver='GPKG')
                    else:
                        geo_outfile = f'{gis_dir}/model_{vv["type"]}.{config.gis["dst_extension"]}'
                        bb.to_file(geo_outfile)
                elif vv['type'] == 'nsegment':
                    geo_file = pyg.read_dataframe(config.gis['src_filename'], layer=vv['layer'],
                                                  columns=vv['include_fields'], force_2d=True,
                                                  where=f'{vv["key"]} >= {min(new_nhm_seg)} AND {vv["key"]} <= {max(new_nhm_seg)}')
                    bb = geo_file[geo_file[vv['key']].isin(new_nhm_seg)]
                    bb = bb.rename(columns={vv['key']: 'nhm_seg'})
                    local_ids = new_ps.get_dataframe('nhm_seg').reset_index()
                    bb = bb.merge(local_ids, on='nhm_seg')

                    if config.gis["dst_extension"] == 'gpkg':
                        bb.to_file(geo_outfile, layer=vv['type'], driver='GPKG')
                    else:
                        geo_outfile = f'{gis_dir}/model_{vv["type"]}.{config.gis["dst_extension"]}'
                        bb.to_file(geo_outfile)
                elif vv['type'] == 'npoigages':
                    if len(new_poi_gage_id) > 0:
                        geo_file = pyg.read_dataframe(config.gis['src_filename'], layer=vv['layer'],
                                                      columns=vv['include_fields'], force_2d=True)

                        bb = geo_file[geo_file[vv['key']].isin(new_poi_gage_id)]
                        bb = bb.rename(columns={vv['key']: 'gage_id', vv['include_fields'][0]: 'nhm_seg'})

                        if config.gis["dst_extension"] == 'gpkg':
                            bb.to_file(geo_outfile, layer=vv['type'], driver='GPKG')
                        else:
                            geo_outfile = f'{gis_dir}/model_{vv["type"]}.{config.gis["dst_extension"]}'
                            bb.to_file(geo_outfile)
                    else:
                        bandit_log.info('No POIs in model subset so POI GIS layer not written.')
                else:
                    bandit_log.warning(f'Layer, {kk}, has unknown type, {vv["type"]}; skipping.')

        if args.verbose:
            print(f'Geo write time: {time.time() - stime:0.3f} s', flush=True)

    bandit_log.info(f'========== END {datetime.datetime.now().isoformat()} ==========')

    os.chdir(stdir)


if __name__ == '__main__':
    main()
