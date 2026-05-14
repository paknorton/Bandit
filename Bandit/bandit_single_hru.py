#!/usr/bin/env python3

import datetime
import errno
import logging
import numpy as np
import os
import time

from cyclopts import App, Parameter, validators
from packaging.version import Version
from pathlib import Path
from typing import Annotated, List, Optional, Union

from pyPRMS.base.console import get_console_instance

from Bandit import __version__
from Bandit.bandit_helpers import create_parameter_subset, set_date
from Bandit.git_version import git_commit, git_repo, git_branch, git_commit_url
from Bandit.model_output import ModelOutput
import Bandit.bandit_cfg as bc   # type: ignore
import Bandit.prms_nwis as prms_nwis   # type: ignore

from pyPRMS.constants import HRU_DIMS, PRMS_VERSION   # type: ignore
from pyPRMS.metadata.metadata import MetaData   # type: ignore
from pyPRMS import Cbh   # type: ignore
from pyPRMS import ControlFile   # type: ignore
from pyPRMS import ParamDb   # type: ignore
from pyPRMS import Parameters   # type: ignore

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
con = get_console_instance(record=True)

__author__ = 'Parker Norton (pnorton@usgs.gov)'

# Setup the logging
bandit_log = logging.getLogger('bandit_single_hru')
root = logging.getLogger()
root.setLevel(logging.INFO)

log_fmt = logging.Formatter('%(levelname)s: %(name)s: %(message)s')

# Handler for console logs
clog = logging.StreamHandler()
clog.setLevel(logging.ERROR)
clog.setFormatter(log_fmt)

# Handler for logging to file
flog = logging.FileHandler(f'{os.getcwd()}/bandit.log')
flog.setLevel(logging.INFO)
flog.setFormatter(log_fmt)

root.addHandler(flog)
bandit_log.addHandler(clog)

app = App(default_parameter=Parameter(negative=()))

@app.default
def extract(config_file: Annotated[Path, Parameter(validator=validators.Path(exists=True))] = Path('bandit.cfg'),
            job_dir: Union[str, Path] = None,
            verbose: bool = False,
            cbh_netcdf: bool = False,
            model_output_netcdf: bool = False,
            param_netcdf: bool = False,
            no_filter_params: bool = False,
            prms_version: Union[str, Version] = PRMS_VERSION,
            include_stream: bool = False,
            prefix: Optional[str] = None):
    """Extract model subsets from the National Hydrologic Model parameter database

    :param config_file: Name of config file to use for extraction
    :param job_dir: Name of job directory to work in
    :param verbose: Output additional debugging information
    :param cbh_netcdf: Output CBH forcings in netCDF format
    :param model_output_netcdf: Output model output variables in netCDF format
    :param param_netcdf: Output parameter file in netCDF format
    :param no_filter_params: Output all parameters regardless of the control file options
    :param prms_version: Output control variables and parameters supported by the version number of PRMS
    :param include_stream: Include the stream segment the HRU connects to
    :param prefix: Prefix to append to each HRU directory
    """

    if isinstance(prms_version, str):
        prms_version: Version = Version(prms_version)

    stdir = os.getcwd()

    if job_dir is not None:
        if isinstance(job_dir, str):
            job_dir = Path(job_dir)

        if job_dir.is_dir():
            # Change into job directory before running extraction
            os.chdir(job_dir)
        else:
            print(f'ERROR: Invalid jobs directory: {str(job_dir)}')
            exit(-1)

    bandit_log.info(f'========== START {datetime.datetime.now().isoformat()} ==========')

    config = bc.Cfg(config_file)

    outdir = Path(config.output_dir)   # Where to output the subset
    param_filename = Path(config.param_filename)   # Name of the output parameter file
    paramdb_dir = Path(config.paramdb_dir)   # Location of the NHM parameter database
    cbh_dir = Path(config.cbh_dir)
    hru_noroute = np.array(config.hru_noroute)   # Array of additional HRUs (have no route to segment within subset)

    # if prms_version == 6:
    #     cbh_netcdf = True
    #     param_netcdf = True
    #     streamflow_netcdf = True

    # Load PRMS metadata
    con.print(f'[green4]INFO[/]: Loading PRMS metadata for version {prms_version}')
    prms_meta = MetaData(version=prms_version, verbose=verbose).metadata

    # Load the control file
    ctl = ControlFile(config.control_filename, metadata=prms_meta, verbose=verbose, include_missing=True)

    # Date range for pulling NWIS streamgage observations and CBH data
    st_date = set_date(config.start_date)
    en_date = set_date(config.end_date)

    # Adjust the start and end dates in the control file to reflect
    # date range from bandit config file
    ctl.get('start_time').values = st_date
    ctl.get('end_time').values = en_date

    # Default the various *ON_OFF variables to 0 (off)
    # The original values are needed to reduce parameters by module,
    # but it's best to disable them in the final control file since
    # no output variables are defined for them.
    disable_vars = ['basinOutON_OFF', 'csvON_OFF', 'mapOutON_OFF', 'nhruOutON_OFF',
                    'nsegmentOutON_OFF', 'nsubOutON_OFF']
    for vv in disable_vars:
        ctl.get(vv).values = 0

    # Output revision of NhmParamDb
    git_url = git_commit_url(paramdb_dir)
    nhmparamdb_revision = git_commit(paramdb_dir, length=7)
    bandit_log.info(f'Parameter database: {git_repo(paramdb_dir)}')
    bandit_log.info(f'Branch: {git_branch(paramdb_dir)}')
    bandit_log.info(f'Commit: {nhmparamdb_revision}')

    # Load the NHMparamdb
    if verbose:
        con.print(f'[green4]INFO[/]: Parameter database: {git_repo(paramdb_dir)}')
        con.print(f'[green4]INFO[/]: Branch: {git_branch(paramdb_dir)}')
        con.print(f'[green4]INFO[/]: Commit: {nhmparamdb_revision}')

    pdb = ParamDb(paramdb_dir=paramdb_dir, metadata=prms_meta, verbose=verbose)
    pdb.control = ctl

    # Add defaults for parameters that are missing but required for the select modules
    pdb.add_missing_parameters()

    if not no_filter_params:
        # Reduce the parameters to those required by the selected modules
        pdb.remove(pdb.unneeded_parameters)

    # Trim paramdb parameters for single-HRU extractions
    params = list(pdb.parameters.keys())

    # Initial list of parameters not included in single-hru extractions
    if include_stream:
        remove_params = []
    else:
        remove_params = ['hru_segment', 'hru_segment_nhm', 'obsout_segment']

    # Add segment- and poi-related parameters
    for pp in params:
        src_param = pdb.get(pp)

        if 'nsegment' in src_param.dimensions.keys():
            if include_stream:
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
            pdb.remove(rp)

    # ====================================================================
    # ====================================================================
    # By definition single-hru extractions are individual non-routed HRUs
    # Loop over provided list of non-routed HRUs generating an extraction
    for chru in hru_noroute:
        print(f'Working on HRU {chru}')
        bandit_log.info(f'HRU {chru}')

        # Set the output directory name
        if prefix:
            sg_dir = outdir / f'{prefix}{chru:06d}'
        else:
            # Style used historically in the byHRU calibration workflow
            sg_dir = outdir / f'HRU{chru}'

        hru_order_subset = [chru]
        bandit_log.info(f'HRU_{chru}: Number of HRUs in subset: {len(hru_order_subset)}')

        if include_stream:
            new_nhm_seg = [pdb.hru_to_seg[chru]]
        else:
            new_nhm_seg = []

        # ==========================================================================
        # Get subset of hru_deplcrv using hru_order_subset
        # A single snarea_curve can be referenced by multiple HRUs
        hru_deplcrv_subset = pdb.get_subset('hru_deplcrv', hru_order_subset)

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

        # The following 5 variables are not used for single-HRU extractions
        new_hru_segment = []
        new_poi_gage_id = []
        new_poi_gage_segment = []
        new_poi_type = []
        new_tosegment = []
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Process the parameters and create a parameter file for the subset
        new_ps = create_parameter_subset(prms_meta, pdb, hru_order_subset,
                                         new_hru_segment, new_nhm_seg, new_poi_gage_id,
                                         new_poi_gage_segment, new_poi_type, new_tosegment)

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
        if verbose:
            con.print(f'[green4]INFO[/]: Writing parameter file for PRMS {prms_version}')

        header = [f'Written by Bandit version {__version__} for PRMS {prms_version}; HRU={chru}',
                  f'ParamDb revision: {git_url}']
        if param_netcdf:
            # TODO: 2023-11-13 PAN - add version info and prms version as global attributes
            param_filename = Path(f'{param_filename.stem}.nc')
            new_ps.write_parameter_netcdf(sg_dir / param_filename)
        else:
            new_ps.write_parameter_file(sg_dir / param_filename, header=header)

        ctl.get('param_file').values = str(param_filename)

        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Write CBH files
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        if config.output_cbh:
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Subset the cbh files for the selected HRUs
            if verbose:
                con.print('Processing CBH files', style='green4')

            # Read the CBH source file
            if cbh_dir.suffix == '.nc':
                cbh_hdl = Cbh(cbh_dir, metadata=prms_meta, engine='netcdf')
            elif cbh_dir.suffix == '.zarr':
                cbh_hdl = Cbh(cbh_dir, metadata=prms_meta, engine='zarr')
            else:
                raise ValueError('Missing CBH files')

            # Add the global NHM IDs
            cbh_hdl.set_nhm_id(pdb.get('nhm_id').data)

            if cbh_netcdf:
                cbh_outfile = sg_dir / 'cbh.nc'

                global_attrs = dict(bandit_version=__version__, paramdb_url=git_url)
                cbh_hdl.write_netcdf(cbh_outfile, variables=list(config.cbh_var_map.keys()),
                                     global_attrs=global_attrs, time_slice=slice(st_date, en_date),
                                     hru_ids=hru_order_subset)

                # Set the control file variables for the CBH files
                for cfv in config.cbh_var_map.values():
                    ctl.get(cfv).values = cbh_outfile.name

            else:
                for cvar, cfv in config.cbh_var_map.items():
                    if verbose:
                        con.print(f'--- {cvar}')

                    cfile = f'{sg_dir}/{ctl.get(cfv).values}'
                    cbh_hdl.write_ascii(cfile, variable=cvar, time_slice=slice(st_date, en_date), hru_ids=hru_order_subset)

        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Write output variables
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # 2019-08-07 PAN: first prototype for extractions of output variables
        if config.include_model_output:
            src_model_output = Path(config.output_vars_dir)
            model_output_dir = sg_dir / 'model_output'
            model_output_dir.mkdir(exist_ok=True)

            model_output_ds = ModelOutput(filename=src_model_output)

            for cvar in config.output_vars:
                if model_output_netcdf:
                    model_output_ds.write_netcdf(filename=model_output_dir / f'{cvar}.nc',
                                                 variables=cvar,
                                                 time_slice=slice(st_date, en_date),
                                                 hru_ids=hru_order_subset,
                                                 seg_ids=new_nhm_seg)
                else:
                    model_output_ds.write_csv(pathname=model_output_dir,
                                              variables=cvar,
                                              time_slice=slice(st_date, en_date),
                                              hru_ids=hru_order_subset,
                                              seg_ids=new_nhm_seg)

        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Write control file
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        ctl.write((sg_dir / config.control_filename).with_suffix('.bandit'))
        # ctl.write(str(Path(config.control_filename).with_suffix('.bandit')))

        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Write streamflow
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        if config.output_streamflow:
            if verbose:
                con.print('Writing dummy streamflow data file', style='green4')
            streamflow = prms_nwis.NWIS(gage_ids=None, st_date=st_date, en_date=en_date, verbose=verbose)
            streamflow.get_daily_streamgage_observations()
            streamflow.write_ascii(filename=sg_dir / f'{config.streamflow_filename}')

        # *******************************************
        # Create a shapefile of the selected HRUs
        if config.output_shapefiles:
            stime = time.time()

            src_gis = Path(config.gis['src_filename'])

            if verbose:
                print('-'*40)
                con.print('Writing shapefiles for model subset', style='green4')

            if len(config.gis) == 0 or not src_gis.exists():
                bandit_log.error(f'Source GIS file'
                                 f'does not exist. Shapefiles will not be created')
            else:
                # Create GIS subdirectory if it doesn't already exist
                gis_dir = sg_dir / 'GIS'
                gis_dir.mkdir(exist_ok=True)

                dst_gis_type = config.gis["dst_extension"]
                geo_outfile = gis_dir / f'model_layers.{dst_gis_type}'

                for kk, vv in config.gis['layers'].items():
                    vv['include_fields'].extend([vv['key']])

                    if vv['type'] == 'nhru':
                        geo_file = pyg.read_dataframe(src_gis, layer=vv['layer'],
                                                      columns=vv['include_fields'], force_2d=True,
                                                      where=f'{vv["key"]} >= {min(hru_order_subset)} AND {vv["key"]} <= {max(hru_order_subset)}')
                        bb = geo_file[geo_file[vv['key']].isin(hru_order_subset)]
                        bb = bb.rename(columns={vv['key']: 'nhm_id'})
                        local_ids = new_ps.get_dataframe('nhm_id').reset_index()
                        bb = bb.merge(local_ids, on='nhm_id')

                        # Buffer the HRUs to make them visible to reduce/remove artifacts
                        # in the dissolved layer caused by tiny gaps between HRUs
                        bb2 = bb.copy()
                        # bb2['geometry'] = bb2['geometry'].buffer(0.0002)
                        domain_layer = bb2.dissolve(aggfunc={'nhm_id': 'count'})
                        domain_layer.rename(columns={'nhm_id': 'num_hrus'}, inplace=True)

                        if dst_gis_type == 'gpkg':
                            bb.to_file(geo_outfile, layer=vv['type'], driver='GPKG')
                            domain_layer.to_file(geo_outfile, layer='domain', driver='GPKG')
                        else:
                            geo_outfile = gis_dir / f'model_{vv["type"]}.{dst_gis_type}'
                            bb.to_file(geo_outfile)

                            domain_outfile = gis_dir / f'model_domain.{dst_gis_type}'
                            domain_layer.to_file(domain_outfile)
                    elif vv['type'] == 'nsegment' and include_stream:
                        geo_file = pyg.read_dataframe(src_gis, layer=vv['layer'],
                                                      columns=vv['include_fields'], force_2d=True,
                                                      where=f'{vv["key"]} >= {min(new_nhm_seg)} AND {vv["key"]} <= {max(new_nhm_seg)}')
                        bb = geo_file[geo_file[vv['key']].isin(new_nhm_seg)]
                        bb = bb.rename(columns={vv['key']: 'nhm_seg'})
                        local_ids = new_ps.get_dataframe('nhm_seg').reset_index()
                        bb = bb.merge(local_ids, on='nhm_seg')

                        if dst_gis_type == 'gpkg':
                            bb.to_file(geo_outfile, layer=vv['type'], driver='GPKG')
                        else:
                            geo_outfile = gis_dir / f'model_{vv["type"]}.{dst_gis_type}'
                            bb.to_file(geo_outfile)
                    else:
                        bandit_log.warning(f'Layer, {kk}, has unknown type, {vv["type"]}; skipping.')

            if verbose:
                con.print(f'[green4]INFO[/]: Geo write time: {time.time() - stime:0.3f} s')

    bandit_log.info(f'========== END {datetime.datetime.now().isoformat()} ==========')

    os.chdir(stdir)


def main():
    app()


if __name__ == '__main__':
    app()
