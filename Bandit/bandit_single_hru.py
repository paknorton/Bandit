#!/usr/bin/env python3

import argparse
import datetime
import errno
import glob
import logging
import numpy as np
import os
import sys

# from collections import OrderedDict

import Bandit.bandit_cfg as bc
# import Bandit.dynamic_parameters as dyn_params
import Bandit.prms_geo as prms_geo
import Bandit.prms_nwis as prms_nwis

from Bandit import __version__
from Bandit.bandit_helpers import set_date
from Bandit.git_version import git_commit, git_repo, git_branch, git_commit_url
from Bandit.model_output import ModelOutput

from pyPRMS.constants import HRU_DIMS
from pyPRMS import CbhNetcdf
from pyPRMS import ControlFile
from pyPRMS import ParamDb
from pyPRMS import ParameterSet
from pyPRMS import ValidParams

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
        if kk not in ['job', 'verbose', 'cbh_netcdf', 'add_gages', 'param_netcdf', 'no_filter_params',
                      'prms_version', 'prefix', 'include_stream']:
            if vv:
                bandit_log.info(f'Overriding configuration for {kk} with {vv}')
                config.update_value(kk, vv)

    # Where to output the subset
    outdir = config.output_dir

    # The control file to use
    control_filename = config.control_filename

    # What to name the output parameter file
    param_filename = config.param_filename

    # Location of the NHM parameter database
    paramdb_dir = config.paramdb_dir

    # List of additional HRUs (have no route to segment within subset)
    # TODO: make sure the len of the array is equal to one
    hru_noroute = config.hru_noroute

    # List of output variables to subset
    try:
        include_model_output = config.include_model_output
        output_vars = config.output_vars
        output_vars_dir = config.output_vars_dir
    except KeyError:
        include_model_output = False
        output_vars = []
        output_vars_dir = ''

    try:
        output_cbh = config.output_cbh

        # Location of the NHM CBH files
        cbh_dir = config.cbh_dir
    except KeyError:
        output_cbh = False
        cbh_dir = ''

    try:
        output_streamflow = config.output_streamflow

        # What to name the streamflow output file
        obs_filename = config.streamflow_filename
    except KeyError:
        output_streamflow = False
        obs_filename = ''

    try:
        output_shapefiles = config.output_shapefiles

        # Full path and filename to the geodatabase to use for outputting shapefile subsets
        geo_file = config.geodatabase_filename
        hru_gis_layer = config.hru_gis_layer
        hru_gis_id = config.hru_gis_id

        if args.include_stream:
            seg_gis_layer = config.seg_gis_layer
            seg_gis_id = config.seg_gis_id
    except KeyError:
        output_shapefiles = False
        geo_file = ''
        hru_gis_layer = None
        hru_gis_id = None
        seg_gis_layer = None
        seg_gis_id = None

    # Load the control file
    ctl = ControlFile(control_filename)

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

    # Output revision of NhmParamDb and the revision used by merged paramdb
    git_url = git_commit_url(paramdb_dir)
    nhmparamdb_revision = git_commit(paramdb_dir, length=7)
    bandit_log.info(f'Using parameter database from: {git_repo(paramdb_dir)}')
    bandit_log.info(f'Repo branch: {git_branch(paramdb_dir)}')
    bandit_log.info(f'Repo commit: {nhmparamdb_revision}')

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Load master list of valid parameters
    vpdb = ValidParams()

    # Build list of parameters required for the selected control file modules
    required_params = vpdb.get_params_for_modules(modules=list(ctl.modules.values()))

    # Load the NHMparamdb
    print('Loading NHM ParamDb')
    pdb = ParamDb(paramdb_dir)
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

        # ==========================================================================
        # Get subset of hru_deplcrv using hru_order_subset
        # A single snarea_curve can be referenced by multiple HRUs
        hru_deplcrv_subset = nhm_params.get_subset('hru_deplcrv', hru_order_subset)

        uniq_deplcrv = list(set(hru_deplcrv_subset))
        uniq_deplcrv0 = [xx - 1 for xx in uniq_deplcrv]

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

        # Remove the POI-related parameters if we have no POIs
        if len(new_poi_gage_segment) == 0:
            bandit_log.warning('No POI gages found for subset; removing POI-related parameters.')

            for rp in ['poi_gage_id', 'poi_gage_segment', 'poi_type']:
                if rp in params:
                    params.remove(rp)

        params.sort()

        # Resize dimensions to the model subset
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
        if output_cbh:
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Subset the cbh files for the selected HRUs
            if args.verbose:
                print('Processing CBH files')

            # Read the CBH source file
            if os.path.splitext(cbh_dir)[1] == '.nc':
                cbh_hdl = CbhNetcdf(src_path=cbh_dir, st_date=st_date, en_date=en_date,
                                    nhm_hrus=hru_order_subset)
            else:
                raise ValueError('Missing netcdf CBH files')

            if args.cbh_netcdf:
                # Pull the filename prefix off of the first file found in the
                # source netcdf CBH directory.
                file_it = glob.iglob(cbh_dir)
                cbh_prefix = os.path.basename(next(file_it)).split('_')[0]

                cbh_outfile = f'{sg_dir}/{cbh_prefix}.nc'
                cbh_hdl.write_netcdf(cbh_outfile, variables=list(config.cbh_var_map.keys()))

                # Set the control file variables for the CBH files
                for cfv in config.cbh_var_map.values():
                    ctl.get(cfv).values = os.path.basename(cbh_outfile)

            else:
                cbh_hdl.write_ascii(pathname=sg_dir, variables=list(config.cbh_var_map.keys()))

                # Set the control file variables for the CBH files
                for cbhvar, cfv in config.cbh_var_map.items():
                    ctl.get(cfv).values = f'{cbhvar}.cbh'

            # bandit_log.info('{} written to: {}'.format(vv, '{}/{}.cbh'.format(outdir, vv)))

        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Write output variables
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # 2019-08-07 PAN: first prototype for extractions of output variables
        if include_model_output:
            # TODO: 2020-03-12 PAN - this is brittle, fix it.
            seg_vars = ['seginc_gwflow', 'seginc_potet', 'seginc_sroff', 'seginc_ssflow',
                        'seginc_swrad', 'segment_delta_flow', 'seg_gwflow', 'seg_inflow',
                        'seg_lateral_inflow', 'seg_outflow', 'seg_sroff', 'seg_ssflow',
                        'seg_upstream_inflow']

            try:
                os.makedirs(f'{sg_dir}/model_output')
                print('Creating directory model_output, for model output variables')
            except OSError:
                print('Using existing model_output directory for output variables')

            for vv in output_vars:
                if args.verbose:
                    sys.stdout.write('\r                                                  ')
                    sys.stdout.write(f'\rProcessing output variable: {vv} ')
                    sys.stdout.flush()

                filename = f'{output_vars_dir}/{vv}.nc'

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
        ctl.write(f'{sg_dir}/{control_filename}.bandit')

        if output_streamflow:
            print('Writing dummy streamflow data file')
            streamflow = prms_nwis.NWIS(gage_ids=None, st_date=st_date, en_date=en_date, verbose=args.verbose)
            streamflow.get_daily_streamgage_observations()
            streamflow.write_ascii(filename=f'{sg_dir}/{obs_filename}')

        # *******************************************
        # Create a shapefile of the selected HRUs
        if output_shapefiles:
            if args.verbose:
                # print('-'*40)
                print('Writing shapefiles for model subset')

            if not os.path.isdir(geo_file):
                bandit_log.error(f'File geodatabase, {geo_file}, does not exist. Shapefiles will not be created')
            else:
                geo_shp = prms_geo.Geo(geo_file)

                # Create GIS subdirectory if it doesn't already exist
                gis_dir = f'{sg_dir}/GIS'
                try:
                    os.makedirs(gis_dir)
                except OSError as exception:
                    if exception.errno != errno.EEXIST:
                        raise
                    else:
                        pass

                # Output a shapefile of the selected HRUs
                if args.verbose:
                    print(f'Layers: {hru_gis_layer}')
                    print(f'IDs: {hru_gis_id}')

                geo_shp.select_layer(hru_gis_layer)
                geo_shp.write_shapefile(f'{sg_dir}/GIS/HRU_subset.shp', hru_gis_id, hru_order_subset,
                                        included_fields=['nhm_id', 'model_idx', hru_gis_id])

                if args.include_stream:
                    geo_shp.select_layer(seg_gis_layer)
                    geo_shp.write_shapefile(f'{sg_dir}/GIS/Segments_subset.shp', seg_gis_id, new_nhm_seg,
                                            included_fields=[seg_gis_id, 'model_idx'])
                del geo_shp

        if args.verbose:
            print('-'*60)

    bandit_log.info(f'========== END {datetime.datetime.now().isoformat()} ==========')

    os.chdir(stdir)


if __name__ == '__main__':
    main()
