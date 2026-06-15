#!/usr/bin/env python3

import datetime
import logging
import networkx as nx   # type: ignore
import numpy as np
import os
import pyogrio as pyg  # type: ignore
import sys
import time

from cyclopts import App, Parameter, validators
from dask.distributed import Client
# from numpy.typing import NDArray
from packaging.version import Version
from pathlib import Path
from rich.rule import Rule
from typing import Annotated, List, Optional, Union

from pyPRMS.base.console import get_console_instance
from pyPRMS import Cbh   # type: ignore
from pyPRMS import ControlFile   # type: ignore
from pyPRMS import ParamDb   # type: ignore
from pyPRMS import Parameters   # type: ignore
from pyPRMS.constants import HRU_DIMS, PRMS_VERSION   # type: ignore
from pyPRMS.metadata.metadata import MetaData   # type: ignore
from pyPRMS.prms_helpers import get_streamnet_subset, set_date   # type: ignore

from Bandit import __version__
from Bandit.bandit_helpers import parse_gages, write_gis
from Bandit.config_validator import ConfigValidator
from Bandit.exceptions import BanditError
from Bandit.git_version import git_commit, git_repo, git_branch, git_commit_url
from Bandit.model_output import ModelOutput
from Bandit.points_of_interest import POI   # type: ignore
import Bandit.bandit_cfg as bc   # type: ignore
import Bandit.dynamic_parameters as dyn_params
import Bandit.prms_nwis as prms_nwis   # type: ignore

import warnings
warnings.filterwarnings('ignore', message=r'.*Measured \(M\) geometry types are not supported.*')
warnings.filterwarnings('ignore',
                        message='.*Column names longer than 10 characters will be truncated when saved to ESRI Shapefile*')
warnings.filterwarnings('ignore', message='.*Slicing with an out-of-order index is generating 10 times more chunks.*')
warnings.filterwarnings('ignore', message=r'.*organizePolygons\(\) received a polygon with more than 100 parts.*')

# Rich library
con = get_console_instance(record=True)

__author__ = 'Parker Norton (pnorton@usgs.gov)'

# Setup the logging
bandit_log = logging.getLogger('bandit')
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
            streamflow_netcdf: bool = False,
            no_filter_params: bool = False,
            keep_hru_order: bool = False,
            prms_version: Union[str, Version] = PRMS_VERSION,
            add_gages: Optional[List[str]] = None):
    """Extract model subsets from the National Hydrologic Model parameter database

    :param config_file: Name of config file to use for extraction
    :param job_dir: Name of job directory to work in
    :param verbose: Output additional debugging information
    :param cbh_netcdf: Output CBH forcings in netCDF format
    :param model_output_netcdf: Output model output variables in netCDF format
    :param param_netcdf: Output parameter file in netCDF format
    :param streamflow_netcdf: Output streamflow observations in netCDF format
    :param no_filter_params: Output all parameters regardless of the control file options
    :param keep_hru_order: Keep the HRU order relative to the source parameter database order
    :param prms_version: Output control variables and parameters supported by the version number of PRMS
    :param add_gages: Include additional ad-hoc streamgages as POIs using the form gage_id=segment.
                      Each gage_id, segment pair should be separated with a space. Segment must
                      exist in the model subset and not be used by existing POIs. These ad-hoc gages
                      are marked in the subset parameter file as poi_type=0
    """

    if isinstance(prms_version, str):
        prms_version: Version = Version(prms_version)

    stdir = os.getcwd()
    chdir_needed = False

    if job_dir is not None:
        if isinstance(job_dir, str):
            job_dir = Path(job_dir)

        if job_dir.is_dir():
            # Change into job directory before running extraction
            os.chdir(job_dir)
            chdir_needed = True
        else:
            con.print(f'[red]ERROR[/]: Invalid jobs directory: {str(job_dir)}')
            bandit_log.error(f'Invalid jobs directory: {str(job_dir)}')
            raise BanditError(f'Invalid jobs directory: {str(job_dir)}', exit_code=-1)

    try:
        bandit_log.info(f'========== START {datetime.datetime.now().isoformat()} ==========')

        addl_gages = None
        if add_gages:
            addl_gages = parse_gages(add_gages)
            if verbose:
                con.print(f'Additional streamgages: {addl_gages}')
            bandit_log.info('Additionals streamgages specified on command line')

        config = bc.Cfg(config_file)

        validator = ConfigValidator(config)
        errors = validator.validate()
        if errors:
            for err in errors:
                bandit_log.error(err)
                con.print(f'[red]ERROR[/]: {err}')
            con.print(f'[red]Configuration has {len(errors)} error(s). Fix the above issues and retry.[/]')
            raise BanditError(f'Configuration has {len(errors)} error(s). Fix the above issues and retry.', exit_code=2)

        outdir = Path(config.output_dir)   # Where to output the subset
        param_filename = Path(config.param_filename)   # Name of the output parameter file
        paramdb_dir = Path(config.paramdb_dir)   # Location of the NHM parameter database
        cbh_dir = Path(config.cbh_dir)
        dsmost_seg = config.outlets   # List of outlets
        uscutoff_seg = config.cutoffs   # List of upstream cutoffs
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

        # Output revision of NhmParamDb
        git_url = git_commit_url(paramdb_dir)
        nhmparamdb_revision = git_commit(paramdb_dir, length=7)
        bandit_log.info(f'Parameter database: {git_repo(paramdb_dir)}')
        bandit_log.info(f'Branch: {git_branch(paramdb_dir)}')
        bandit_log.info(f'Commit: {nhmparamdb_revision}')

        # client = Client(threads_per_worker=1)
        with Client() as client:
            dash_link = client.dashboard_link
            con.print(f'Dask dashboard: {dash_link}')

            # Load the NHMparamdb
            if verbose:
                con.print(f'[green4]INFO[/]: Parameter database: {git_repo(paramdb_dir)}')
                con.print(f'[green4]INFO[/]: Branch: {git_branch(paramdb_dir)}')
                con.print(f'[green4]INFO[/]: Commit: {nhmparamdb_revision}')

            pdb = ParamDb(paramdb_dir=paramdb_dir, metadata=prms_meta, verbose=verbose)
            pdb.control = ctl

            if pdb.dimensions.exists('npoigages') and not pdb.dimensions.exists('nobs'):
                # If we have poi gages and nobs is missing then add it
                if verbose:
                    con.print('[gold3]WARNING[/]: Added missing nobs dimension')
                pdb.dimensions.add('nobs', size=pdb.dimensions.get('npoigages').size)

            # Add defaults for parameters that are missing but required for the selected modules
            # WARNING: 20240726 PAN - if hru_segment_nhm is missing from the paramdb no warning is issued,
            #                         and the wrong number of HRUs will likely be output for the extraction.
            pdb.add_missing_parameters()

            if not no_filter_params:
                # Reduce the parameters to those required by the selected modules
                pdb.remove(pdb.unneeded_parameters)

            if not pdb.exists('poi_gage_segment'):
                con.print('[gold3]WARNING[/]: Missing POI-related parameters. To include POIs, set csvON_OFF > 0 in the control file')

            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Get tosegment_nhm
            # Convert to list for fastest access to array
            nhm_seg = pdb.get('nhm_seg').data

            # First check if any of the requested stream segments exist in the NHM.
            # An intersection of 0 elements can occur when all stream segments are
            # not included in the NHM (e.g. segments in Alaska).
            # NOTE: It's possible to have a stream segment that does not exist in
            #       tosegment but does exist in nhm_seg (e.g. standalone segment). So
            #       we use nhm_seg to verify at least one of the given segment(s) exist.
            if dsmost_seg and len(set(dsmost_seg).intersection(set(nhm_seg))) == 0:
                con.print(f'[red]ERROR[/]: None of the requested stream segments exist in the NHM')
                bandit_log.error('None of the requested stream segments exist in the NHM paramDb')
                raise BanditError('None of the requested stream segments exist in the NHM', exit_code=200)

            # Build the stream network
            dag_ds = pdb.stream_network(tosegment='tosegment_nhm', seg_id='nhm_seg')
            assert dag_ds is not None

            bandit_log.debug(f'Number of NHM downstream nodes: {dag_ds.number_of_nodes()}')
            bandit_log.debug(f'Number of NHM downstream edges: {dag_ds.number_of_edges()}')

            if config.check_DAG:
                if not nx.is_directed_acyclic_graph(dag_ds):
                    con.print('[red]ERROR[/]: Cycles and/or loops found in stream network')
                    bandit_log.error('Cycles and/or loops found in stream network')

                    for xx in nx.simple_cycles(dag_ds):
                        con.print(f'[red]ERROR[/]: Cycle found for segment {xx}')
                        bandit_log.error(f'Cycle found for segment {xx}')

            dag_ds_subset = get_streamnet_subset(dag_ds, uscutoff_seg, dsmost_seg)

            # Segments in model subset
            new_nhm_seg = np.array([ee[0] for ee in dag_ds_subset.edges])
            con.print(f'[green4]INFO[/]: Number of stream segments in model subset: {len(new_nhm_seg)}')
            bandit_log.info(f'Number of segments in model subset: {len(new_nhm_seg)}')
            if verbose:
                con.print(f'segments: {new_nhm_seg}')

            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Extract the parameter subset from the stream network subgraph
            try:
                new_ps = pdb.extract_subset(dag_ds_subset, hru_noroute,
                                            keep_hru_order=keep_hru_order,
                                            addl_gages=addl_gages)
            except ValueError as err:
                con.print(f'[red]ERROR[/]: {err}')
                bandit_log.error(str(err))
                raise BanditError(str(err), exit_code=2)

            # Derive HRU order from the resulting subset (needed for CBH, dynamic params, GIS)
            hru_order_subset = new_ps.get('nhm_id').data.tolist()

            # Derive POI gage IDs from the subset (needed for streamgage downloads, GIS)
            new_poi_gage_id = new_ps.get('poi_gage_id').tolist() if new_ps.exists('poi_gage_id') else []

            num_hru = len(hru_order_subset)
            num_poi = len(new_poi_gage_id)

            con.print(f'[green4]INFO[/]: Number of HRUs in model subset: {num_hru}')
            bandit_log.info(f'Number of HRUs in subset: {num_hru}')
            con.print(f'[green4]INFO[/]: Number of POI gages in model subset: {num_poi}')

            # Write the new parameter file
            if verbose:
                con.print(f'[green4]INFO[/]: Writing parameter file for PRMS {prms_version}')

            header = [f'Written by Bandit version {__version__} for PRMS {prms_version}',
                      f'ParamDb revision: {git_url}']
            if param_netcdf:
                # TODO: 2023-11-13 PAN - add version info and prms version as global attributes
                param_filename = Path(f'{param_filename.stem}.nc')
                new_ps.write_parameter_netcdf(outdir / param_filename)
            else:
                new_ps.write_parameter_file(outdir / param_filename, header=header)

            ctl.get('param_file').values = str(param_filename)

            if verbose:
                con.print('')
    
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

                # Add the global NHM IDs from source parameter database
                cbh_hdl.set_nhm_id(pdb.get('nhm_id').data)

                if cbh_netcdf:
                    cbh_outfile = outdir / 'cbh.nc'

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

                        cfile = ctl.get(cfv).values
                        cbh_hdl.write_ascii(cfile, variable=cvar, time_slice=slice(st_date, en_date), hru_ids=hru_order_subset)

            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Write output variables
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            if config.include_model_output:
                src_model_output = Path(config.output_vars_dir)
                model_output_dir = outdir / 'model_output'
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
            # Write dynamic parameters
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            if ctl.has_dynamic_parameters:
                if config.dyn_params_dir is None:
                    con.print('[red]ERROR[/]: Control file has dynamic parameters but dyn_params_dir is not specified in the config file')
                    bandit_log.error('Control file has dynamic parameters but dyn_params_dir is not specified ' +
                                     'in the config file')
                    raise BanditError('Control file has dynamic parameters but dyn_params_dir is not specified in the config file', exit_code=2)
                else:
                    dyn_params_dir = Path(config.dyn_params_dir)

                    if not dyn_params_dir.is_dir():
                        con.print(f'[red]ERROR[/]: dyn_params_dir: {config.dyn_params_dir}, does not exist.')
                        bandit_log.error(f'dyn_params_dir: {config.dyn_params_dir}, does not exist.')
                        raise BanditError(f'dyn_params_dir: {config.dyn_params_dir}, does not exist.', exit_code=2)

                    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                    # Add dynamic parameters
                    for cparam in ctl.dynamic_parameters:
                        param_name = f'dyn_{cparam}'
                        input_file = dyn_params_dir / f'{param_name}.nc'
                        output_file = outdir / f'{param_name}.param'

                        if not input_file.is_file():
                            warn_txt = f'WARNING: CONUS dynamic parameter file: {input_file}, does not exist... skipping'
                            bandit_log.warning(warn_txt)
                        else:
                            if verbose:
                                con.print(f'[green4]INFO[/]: Writing dynamic parameter {cparam}')

                            mydyn = dyn_params.DynamicParameters(str(input_file), cparam, st_date, en_date, hru_order_subset)

                            mydyn.read_netcdf()
                            out_order = [kk for kk in hru_order_subset]

                            out_order = ['year', 'month', 'day'] + out_order
                            # for cc in ['day', 'month', 'year']:
                            #     out_order.insert(0, cc)

                            header = ' '.join(map(str, out_order))   # type: ignore

                            # Output ASCII files
                            with open(output_file, 'w') as out_ascii:
                                out_ascii.write(f'{cparam}\n')
                                out_ascii.write(f'{header}\n')
                                out_ascii.write('####\n')
                                mydyn.data.to_csv(out_ascii, columns=out_order, na_rep='-999',
                                                  sep=' ', index=False, header=False, encoding=None, chunksize=50)

            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Write control file
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Default the various *ON_OFF variables to 0 (off)
            # The original values are needed to reduce parameters by module,
            # but it's best to disable them in the final control file since
            # no output variables are defined for them.
            disable_vars = ['basinOutON_OFF', 'csvON_OFF', 'mapOutON_OFF', 'nhruOutON_OFF',
                            'nsegmentOutON_OFF', 'nsubOutON_OFF']
            for vv in disable_vars:
                ctl.get(vv).values = 0

            ctl.write(str(Path(config.control_filename).with_suffix('.bandit')))

            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Write streamflow
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            if config.output_streamflow:
                # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                # Download the streamgage information from NWIS
                if len(new_poi_gage_id) > 0:
                    if verbose:
                        con.print(f'[green4]INFO[/]: Retrieving streamgage observations for {len(new_poi_gage_id)} stations', style='green4')

                    if config.exists('poi_dir') and config.poi_dir != '':
                        bandit_log.info('Retrieving POIs from local HYDAT and NWIS netcdf files')
                        streamflow = POI(src_path=config.poi_dir, st_date=st_date, en_date=en_date,
                                         gage_ids=new_poi_gage_id, verbose=verbose)
                    else:
                        # Default to retrieving only NWIS stations from waterservices.usgs.gov
                        bandit_log.info('No poi_dir: retrieving only NWIS POIs from online NWIS service.')
                        streamflow = prms_nwis.NWIS(gage_ids=new_poi_gage_id, st_date=st_date, en_date=en_date,
                                                    verbose=verbose)
                        streamflow.get_daily_streamgage_observations()

                    if streamflow_netcdf:
                        streamflow.write_netcdf(filename=outdir / f'{config.streamflow_filename}.nc')
                    else:
                        streamflow.write_ascii(filename=outdir / f'{config.streamflow_filename}')
                else:
                    # TODO: 2025-03-25 PAN - this should write a netcdf file if that option was selected
                    if verbose:
                        con.print('[green4]WARNING[/]: No POIs exist in model subset; writing dummy data', style='gold3')
                    streamflow = prms_nwis.NWIS(gage_ids=None, st_date=st_date, en_date=en_date, verbose=verbose)
                    streamflow.get_daily_streamgage_observations()
                    streamflow.write_ascii(filename=outdir / f'{config.streamflow_filename}')
                    bandit_log.info(f'No POIs exist in model subset; dummy data written.')

            # *******************************************
            # Create a shapefile of the selected HRUs
            if config.output_shapefiles:
                stime = time.time()

                src_gis = Path(config.gis['src_filename'])

                if verbose:
                    con.print(Rule())
                    con.print('Writing shapefiles for model subset', style='green4')

                if len(config.gis) == 0 or not src_gis.exists():
                    bandit_log.error(f'Source GIS file does not exist. Shapefiles will not be created')
                else:
                    # Create GIS subdirectory if it doesn't already exist
                    gis_dir = outdir / 'GIS'
                    gis_dir.mkdir(exist_ok=True)

                    dst_gis_type = config.gis["dst_extension"]

                    for kk, vv in config.gis['layers'].items():
                        vv['include_fields'].extend([vv['key']])
                        src_include_flds = vv['include_fields']
                        src_layer = vv['layer']
                        src_key = vv['key']
                        dst_layer = vv['type']

                        if dst_layer == 'nhru':
                            geo_file = pyg.read_dataframe(src_gis, layer=src_layer,
                                                          columns=src_include_flds, force_2d=True,
                                                          where=f'{src_key} >= {min(hru_order_subset)} AND {src_key} <= {max(hru_order_subset)}')

                            bb = geo_file[geo_file[src_key].isin(hru_order_subset)]
                            bb = bb.rename(columns={src_key: 'nhm_id'})
                            local_ids = new_ps.get_dataframe('nhm_id').reset_index()
                            bb = bb.merge(local_ids, on='nhm_id')
                            write_gis(gdf=bb, gis_dir=gis_dir, gis_type=dst_gis_type, layer=dst_layer)

                            # Create a domain outline from the subset HRUs
                            # Buffer the HRUs to make them visible to reduce/remove artifacts
                            # in the dissolved layer caused by tiny gaps between HRUs
                            bb2 = bb.copy()
                            # bb2['geometry'] = bb2['geometry'].buffer(0.0002)
                            domain_layer = bb2.dissolve(aggfunc={'nhm_id': 'count'})
                            domain_layer.rename(columns={'nhm_id': 'num_hrus'}, inplace=True)
                            write_gis(gdf=domain_layer, gis_dir=gis_dir, gis_type=dst_gis_type, layer='domain')
                        elif dst_layer == 'nsegment':
                            geo_file = pyg.read_dataframe(src_gis, layer=src_layer,
                                                          columns=src_include_flds, force_2d=True,
                                                          where=f'{src_key} >= {min(new_nhm_seg)} AND {src_key} <= {max(new_nhm_seg)}')

                            bb = geo_file[geo_file[src_key].isin(new_nhm_seg)]
                            bb = bb.rename(columns={src_key: 'nhm_seg'})
                            local_ids = new_ps.get_dataframe('nhm_seg').reset_index()
                            bb = bb.merge(local_ids, on='nhm_seg')
                            write_gis(gdf=bb, gis_dir=gis_dir, gis_type=dst_gis_type, layer=dst_layer)
                        elif dst_layer == 'npoigages':
                            if len(new_poi_gage_id) > 0:
                                geo_file = pyg.read_dataframe(src_gis, layer=src_layer,
                                                              columns=src_include_flds, force_2d=True)

                                bb = geo_file[geo_file[src_key].isin(new_poi_gage_id)]
                                bb = bb.rename(columns={src_key: 'gage_id', src_include_flds[0]: 'nhm_seg'})
                                write_gis(gdf=bb, gis_dir=gis_dir, gis_type=dst_gis_type, layer=dst_layer)
                            else:
                                bandit_log.info('No POIs in model subset so POI GIS layer not written.')
                        else:
                            bandit_log.warning(f'Layer, {kk}, has unknown type, {dst_layer}; skipping.')

                if verbose:
                    con.print(f'[green4]INFO[/]: Geo write time: {time.time() - stime:0.3f} s')

            bandit_log.info(f'========== END {datetime.datetime.now().isoformat()} ==========')
    finally:
        if chdir_needed:
            os.chdir(stdir)


def main():
    try:
        app()
    except BanditError as err:
        bandit_log.error(str(err))
        con.print(f'[red]ERROR[/]: {err}')
        sys.exit(err.exit_code)


if __name__ == '__main__':
    app()
