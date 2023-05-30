#!/usr/bin/env python3

import argparse
import datetime
import errno
import glob
import logging
import numpy as np
import os
import sys
import time

from typing import List

import Bandit.bandit_cfg as bc
# import Bandit.dynamic_parameters as dyn_params
# import Bandit.prms_geo as prms_geo
import Bandit.prms_nwis as prms_nwis

from Bandit import __version__
from Bandit.bandit_helpers import set_date, resize_dims
from Bandit.git_version import git_commit, git_repo, git_branch, git_commit_url
from Bandit.model_output import ModelOutput
# from Bandit.points_of_interest import POI

from pyPRMS.constants import HRU_DIMS
from pyPRMS import CbhNetcdf
from pyPRMS import ControlFile
from pyPRMS import ParamDb
from pyPRMS import ParameterSet

# os.environ['USE_PYGEOS'] = '0'
# import geopandas as gpd

import pyogrio as pyg  # type: ignore
import warnings
warnings.filterwarnings("ignore", message=".*Measured \(M\) geometry types are not supported.*")

# from pyogrio import list_drivers, list_layers, read_info, read_dataframe, write_dataframe

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


def main():
    # Command line arguments
    parser = argparse.ArgumentParser(description='Extract model subsets from the National Hydrologic Model')
    parser.add_argument('-O', '--output_dir', help='Output directory for subset')
    parser.add_argument('-p', '--param_filename', help='Name of output parameter file')
    parser.add_argument('-P', '--paramdb_dir', help='Location of parameter database')
    parser.add_argument('-C', '--cbh_dir', help='Location of CBH files')
    parser.add_argument('-g', '--geodatabase_filename', help='Full path to NHM geodatabase')
    parser.add_argument('-j', '--job', help='Job directory to work in')
    parser.add_argument('-v', '--verbose', help='Output additional information', action='store_true')
    parser.add_argument('--output_cbh', help='Output CBH files for subset', action='store_true')
    parser.add_argument('--output_shapefiles', help='Output shapefiles for subset', action='store_true')
    parser.add_argument('--cbh_netcdf', help='Enable netCDF output for CBH files', action='store_true')
    parser.add_argument('--param_netcdf', help='Enable netCDF output for parameter file', action='store_true')
    parser.add_argument('--no_filter_params',
                        help='Output all parameters regardless of modules selected', action='store_true')
    parser.add_argument('--hru_gis_id', help='Name of key/id for HRUs in geodatabase', nargs='?', type=str)
    parser.add_argument('--hru_gis_layer', help='Name of geodatabase layer containing HRUs', nargs='?', type=str)
    parser.add_argument('--include_stream', help='Include the stream segment the HRU connects to', action='store_true')
    parser.add_argument('--prms_version', help='Write PRMS version 5 or 6 parameter file', nargs='?',
                        default=5, type=int)
    parser.add_argument('--prefix', help='Prefix to append to each HRU directory', type=str)
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

    # List of additional HRUs (have no route to segment within subset)
    # TODO: make sure the len of the array is equal to one
    hru_noroute = config.hru_noroute

    if args.prms_version == 6:
        args.cbh_netcdf = True
        args.param_netcdf = True
        args.streamflow_netcdf = True

    # Load the control file
    ctl = ControlFile(config.control_filename, version=args.prms_version)

    # dyn_params_dir = ''
    # if ctl.has_dynamic_parameters:
    #     if config.dyn_params_dir:
    #         if os.path.exists(config.dyn_params_dir):
    #             dyn_params_dir = config.dyn_params_dir
    #         else:
    #             bandit_log.error(f'dyn_params_dir: {config.dyn_params_dir}, does not exist.')
    #             exit(2)
    #     else:
    #         bandit_log.error('Control file has dynamic parameters but dyn_params_dir is not specified ' +
    #                          'in the config file')
    #         exit(2)

    # Date range for pulling NWIS streamgage observations and CBH data
    st_date = set_date(config.start_date)
    en_date = set_date(config.end_date)

    # Adjust the start and end dates in the control file to reflect
    # date range from bandit config file
    ctl.get('start_time').values = [st_date.year, st_date.month, st_date.day, 0, 0, 0]
    ctl.get('end_time').values = [en_date.year, en_date.month, en_date.day, 0, 0, 0]

    # Output revision of NhmParamDb
    git_url = git_commit_url(paramdb_dir)
    nhmparamdb_revision = git_commit(paramdb_dir, length=7)
    bandit_log.info(f'Using parameter database from: {git_repo(paramdb_dir)}')
    bandit_log.info(f'Repo branch: {git_branch(paramdb_dir)}')
    bandit_log.info(f'Repo commit: {nhmparamdb_revision}')

    # Load the NHMparamdb
    if args.verbose:
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

    # Trim paramdb parameters for single-HRU extractions
    params = list(nhm_params.keys())

    # Initial list of parameters not included in single-hru extractions
    if args.include_stream:
        remove_params = []
    else:
        remove_params = ['hru_segment', 'hru_segment_nhm', 'obsout_segment']

    # Add segment- and poi-related parameters
    for pp in params:
        src_param = nhm_params.get(pp)

        if 'nsegment' in src_param.dimensions.keys():
            if args.include_stream:
                pass
                # if pp not in ['K_coef', 'nhm_seg', 'segment_type', 'tosegment',
                #               'tosegment_nhm', 'x_coef']:
                #     bandit_log.info(f'INFO: Removed nsegment parameter, {pp}')
                #     remove_params.append(pp)
            else:
                bandit_log.info(f'INFO: Removed nsegment parameter, {pp}')
                remove_params.append(pp)
        elif 'npoigages' in src_param.dimensions.keys():
            bandit_log.info(f'INFO: Removed npoigages parameter, {pp}')
            remove_params.append(pp)

    print('-'*20)
    print(remove_params)
    print('-'*20)
    # Now remove those parameters
    for rp in remove_params:
        if rp in params:
            params.remove(rp)

    params.sort()

    # ====================================================================
    # ====================================================================
    # By definition single-hru extractions are individual non-routed HRUs
    # Loop over provided list of non-routed HRUs generating an extraction

    for chru in hru_noroute:
        print(f'Working on HRU {chru}')
        bandit_log.info(f'HRU {chru}')

        # Set the output directory name
        if args.prefix:
            sg_dir = f'{outdir}/{args.prefix}{chru:06d}'
        else:
            # Style used historically in the byHRU calibration workflow
            sg_dir = f'{outdir}/HRU{chru}'

        hru_order_subset = [chru]
        bandit_log.info(f'HRU_{chru}: Number of HRUs in subset: {len(hru_order_subset)}')

        if args.include_stream:
            new_nhm_seg = [nhm_params.hru_to_seg[chru]]
        else:
            new_nhm_seg = []

        # ==========================================================================
        # Get subset of hru_deplcrv using hru_order_subset
        # A single snarea_curve can be referenced by multiple HRUs
        hru_deplcrv_subset = nhm_params.get_subset('hru_deplcrv', hru_order_subset)

        # noinspection PyTypeChecker
        uniq_deplcrv: List = np.unique(hru_deplcrv_subset).tolist()  # type: ignore

        uniq_dict = {}
        for ii, xx in enumerate(uniq_deplcrv):
            uniq_dict[xx] = ii + 1

        # Create new hru_deplcrv and renumber
        new_hru_deplcrv = [uniq_dict[xx] for xx in hru_deplcrv_subset]
        bandit_log.info(f'Size of hru_deplcrv for subset: {len(new_hru_deplcrv)}')

        # ==================================================================
        # ==================================================================
        # Process the parameters and create a parameter file for the subset
        params = list(nhm_params.keys())

        # Remove the POI-related parameters
        for rp in remove_params:
            if rp in params:
                params.remove(rp)

        params.sort()

    # Build dictionary of resized dimensions for the model subset
        dims = resize_dims(src_global_dims=nhm_global_dimensions.values(),
                           num_hru=len(hru_order_subset),
                           num_seg=len(new_nhm_seg),
                           num_deplcrv=len(uniq_deplcrv),
                           num_poi=0)

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
                elif args.include_stream and first_dimension == 'nsegment':
                    if pp in ['tosegment']:
                        outdata = np.array(0)
                    else:
                        outdata = nhm_params.get_subset(pp, new_nhm_seg)
                elif first_dimension == 'ndeplval':
                    # snarea_thresh - this is really a 2D in disguise, however,
                    # it is stored in C-order unlike other 2D arrays
                    outdata = nhm_params.get_subset(pp, hru_order_subset)
                elif first_dimension in HRU_DIMS:
                    if pp == 'hru_deplcrv':
                        outdata = nhm_params.get_subset(pp, hru_order_subset)
                    elif pp == 'hru_segment':
                        if args.include_stream:
                            outdata = np.array(1)
                        else:
                            print(f'ERROR: {src_param.name} should not be here')
                            pass
                    else:
                        outdata = nhm_params.get_subset(pp, hru_order_subset)
                else:
                    bandit_log.error(f'No rules to handle dimension {first_dimension}')
            elif ndims == 2:
                # 2D Parameters
                if first_dimension == 'nsegment':
                    if args.include_stream:
                        outdata = nhm_params.get_subset(pp, new_nhm_seg)
                    else:
                        print(f'ERROR: {src_param.name} should not be here')
                elif first_dimension in HRU_DIMS:
                    outdata = nhm_params.get_subset(pp, hru_order_subset)
                else:
                    err_txt = f'No rules to handle 2D parameter, {pp}, which contains dimension {first_dimension}'
                    bandit_log.error(err_txt)

            cnew_param.data = outdata

        # We're far enough along without error to go ahead and make the directory
        try:
            os.makedirs(sg_dir)
        except OSError as exception:
            if exception.errno != errno.EEXIST:
                raise
            else:
                # Directory already exists
                pass

        # Write the new parameter file
        header = [f'Written by Bandit version {__version__}; HRU={chru}',
                  f'ParamDb revision: {git_url}']
        if args.param_netcdf:
            base_filename = os.path.splitext(param_filename)[0]
            param_filename = f'{base_filename}.nc'
            new_ps.write_netcdf(f'{sg_dir}/{param_filename}')
        else:
            print(f'\nWriting version {args.prms_version} parameter file')
            new_ps.write_parameter_file(f'{sg_dir}/{param_filename}', header=header, prms_version=args.prms_version)

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
                cbh_outfile = f'{sg_dir}/cbh.nc'
                cbh_hdl.write_netcdf(cbh_outfile, variables=list(config.cbh_var_map.keys()))

                # Set the control file variables for the CBH files
                for cfv in config.cbh_var_map.values():
                    ctl.get(cfv).values = os.path.basename(cbh_outfile)

            else:
                for cvar, cfv in config.cbh_var_map.items():
                    if args.verbose:
                        print(f'--- {cvar}')

                    cfile = f'{sg_dir}/{ctl.get(cfv).values}'
                    cbh_hdl.write_ascii(cfile, variable=cvar)

        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Write output variables
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # 2019-08-07 PAN: first prototype for extractions of output variables
        if config.include_model_output:
            # TODO: 2020-03-12 PAN - this is brittle, fix it.
            seg_vars = ['seginc_gwflow', 'seginc_potet', 'seginc_sroff', 'seginc_ssflow',
                        'seginc_swrad', 'segment_delta_flow', 'seg_gwflow', 'seg_inflow',
                        'seg_lateral_inflow', 'seg_outflow', 'seg_sroff', 'seg_ssflow',
                        'seg_upstream_inflow']

            try:
                os.makedirs(f'{sg_dir}/model_output')
                bandit_log.info('Creating directory model_output, for model output variables')
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
                        print('ERROR: segment-related output variables not allowed in single-hru extraction')
                        # mod_out = ModelOutput(filename=filename, varname=vv, startdate=st_date, enddate=en_date,
                        #                       nhm_segs=new_nhm_seg)
                        # mod_out.write_csv(f'{outdir}/model_output')
                    else:
                        mod_out = ModelOutput(filename=filename, varname=vv, startdate=st_date, enddate=en_date,
                                              nhm_hrus=hru_order_subset)
                        mod_out.write_csv(f'{sg_dir}/model_output')
                except FileNotFoundError:
                    bandit_log.warning(f'Model output variable, {vv}, does not exist; skipping.')
            print()

        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Write dynamic parameters
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # if ctl.has_dynamic_parameters:
        #     # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        #     # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        #     # Add dynamic parameters
        #     for cparam in ctl.dynamic_parameters:
        #         param_name = f'dyn_{cparam}'
        #         input_file = f'{dyn_params_dir}/{param_name}.nc'
        #         output_file = f'{outdir}/{param_name}.param'
        #
        #         if not os.path.exists(input_file):
        #             warn_txt = f'WARNING: CONUS dynamic parameter file: {input_file}, does not exist... skipping'
        #             bandit_log.warning(warn_txt)
        #         else:
        #             if args.verbose:
        #                 print(f'Writing dynamic parameter {cparam}')
        #
        #             mydyn = dyn_params.DynamicParameters(input_file, cparam, st_date, en_date, hru_order_subset)
        #             # mydyn = dyn_params.DynamicParameters(input_file, cparam, st_date, en_date, hru_order_subset)
        #
        #             mydyn.read_netcdf()
        #             out_order = [kk for kk in hru_order_subset]
        #             # out_order = [kk for kk in hru_order_subset]
        #             for cc in ['day', 'month', 'year']:
        #                 out_order.insert(0, cc)
        #
        #             header = ' '.join(map(str, out_order))
        #
        #             # Output ASCII files
        #             out_ascii = open(output_file, 'w')
        #             out_ascii.write(f'{cparam}\n')
        #             out_ascii.write(f'{header}\n')
        #             out_ascii.write('####\n')
        #             mydyn.data.to_csv(out_ascii, columns=out_order, na_rep='-999',
        #                               sep=' ', index=False, header=False, encoding=None, chunksize=50)
        #             out_ascii.close()

        # Write an updated control file to the output directory
        ctl.write(f'{sg_dir}/{os.path.basename(config.get_value("control_filename"))}.bandit')

        if config.output_streamflow:
            print('Writing dummy streamflow data file')
            streamflow = prms_nwis.NWIS(gage_ids=None, st_date=st_date, en_date=en_date, verbose=args.verbose)
            streamflow.get_daily_streamgage_observations()
            streamflow.write_ascii(filename=f'{sg_dir}/{config.streamflow_filename}')

        # *******************************************
        # Create a shapefile of the selected HRUs
        if config.output_shapefiles:
            stime = time.time()

            if args.verbose:
                print('-'*40)
                print('Writing shapefiles for model subset')

            if len(config.gis) == 0 or not os.path.exists(config.gis['src_filename']):
                bandit_log.error(f'Source GIS file'
                                 f'does not exist. Shapefiles will not be created')
            else:
                # Create GIS subdirectory if it doesn't already exist
                gis_dir = f'{sg_dir}/GIS'
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
                        local_ids = new_ps.parameters.get_dataframe('nhm_id').reset_index()
                        bb = bb.merge(local_ids, on='nhm_id')

                        if config.gis["dst_extension"] == 'gpkg':
                            bb.to_file(geo_outfile, layer=vv['type'], driver='GPKG')
                        else:
                            geo_outfile = f'{gis_dir}/model_{vv["type"]}.{config.gis["dst_extension"]}'
                            bb.to_file(geo_outfile)
                    elif vv['type'] == 'nsegment' and args.include_stream:
                        geo_file = pyg.read_dataframe(config.gis['src_filename'], layer=vv['layer'],
                                                      columns=vv['include_fields'], force_2d=True,
                                                      where=f'{vv["key"]} >= {min(new_nhm_seg)} AND {vv["key"]} <= {max(new_nhm_seg)}')
                        bb = geo_file[geo_file[vv['key']].isin(new_nhm_seg)]
                        bb = bb.rename(columns={vv['key']: 'nhm_seg'})
                        local_ids = new_ps.parameters.get_dataframe('nhm_seg').reset_index()
                        bb = bb.merge(local_ids, on='nhm_seg')

                        if config.gis["dst_extension"] == 'gpkg':
                            bb.to_file(geo_outfile, layer=vv['type'], driver='GPKG')
                        else:
                            geo_outfile = f'{gis_dir}/model_{vv["type"]}.{config.gis["dst_extension"]}'
                            bb.to_file(geo_outfile)
                    # elif vv['type'] == 'npoigages':
                    #     geo_file = pyg.read_dataframe(config.gis['src_filename'], layer=vv['layer'],
                    #                                   columns=vv['include_fields'], force_2d=True)
                    #     bb = geo_file[geo_file['Type_Gage'].isin(new_poi_gage_id)]
                    #     bb = bb.rename(columns={vv['key']: 'gage_id', vv['include_fields'][0]: 'nhm_seg'})
                    #
                    #     if config.gis["dst_extension"] == 'gpkg':
                    #         bb.to_file(geo_outfile, layer=vv['type'], driver='GPKG')
                    #     else:
                    #         geo_outfile = f'{gis_dir}/model_{vv["type"]}.{config.gis["dst_extension"]}'
                    #         bb.to_file(geo_outfile)

                        # bb.to_file(geo_outfile, layer='POIs', driver='GPKG')
                    else:
                        bandit_log.warning(f'Layer, {kk}, has unknown type, {vv["type"]}; skipping.')

            if args.verbose:
                print(f'Geo write time: {time.time() - stime:0.3f} s', flush=True)

    bandit_log.info(f'========== END {datetime.datetime.now().isoformat()} ==========')

    os.chdir(stdir)


if __name__ == '__main__':
    main()
