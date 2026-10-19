#!/usr/bin/env python3

"""Batch Bandit extractions driven by a file of streamgage POI IDs.

The input file lists one streamgage POI ID per line. Each streamgage is mapped
to its NHM POI segment via the parameter database and extracted into its own
output directory. Streamgages that are not POIs, that have no assigned segment,
or that map to multiple segments are reported and skipped.

Per-streamgage upstream segment cutoffs and non-routed HRUs can optionally be
supplied as CSV files. The per-run summaries are collected into a console table
and an HTML roll-up report.
"""

import os
import shutil

from cyclopts import App, Parameter
from pathlib import Path
from typing import Annotated

from pyPRMS.metadata.metadata import MetaData   # type: ignore
from pyPRMS import ParamDb   # type: ignore

from Bandit import bandit_cfg as bc
from Bandit.cli_shared import (check_paramdb_revision, collect_summary_rows, con,
                               print_summary_table, RESERVED_BANDIT_ARGS, run_cli,
                               run_worker_pool, write_summary_report)
from Bandit.exceptions import BanditError

__author__ = 'Parker Norton (pnorton@usgs.gov)'

# Default arguments passed through to each bandit extraction when the
# user does not supply any on the command line.
DEFAULT_BANDIT_ARGS = ['--no_filter_params', '--keep_hru_order']


def read_file(filename: str | Path) -> dict:
    """Read a CSV of a POI ID followed by one or more integer values.

    The file has a header row, then one POI ID (string) per row followed by one
    or more integer HRU IDs or segment IDs::

        poi_id,ids
        00000000,4,67,8,...

    Blank lines and surrounding whitespace are ignored.

    :param filename: File to read
    :returns: Mapping of POI ID to its list of integer values
    :raises BanditError: If a non-header row contains a non-integer value
    """

    values_by_id = {}

    with open(filename, 'r') as src_file:
        # Skip the header row
        src_file.readline()

        # Read in the non-routed HRUs or segment cutoffs by POI
        for line in src_file:
            cols = [cc.strip() for cc in line.strip().split(',')]

            if not cols or not cols[0]:
                # Skip blank lines (and lines with no POI ID)
                continue

            try:
                values_by_id[cols[0]] = [int(xx) for xx in cols[1:]]
            except ValueError as err:
                raise BanditError(f'Non-integer value in {filename} for POI {cols[0]}: {err}',
                                  exit_code=1)

    return values_by_id


def get_streamgage_segments(filename: str | Path,
                            poi_id_to_seg: dict) -> tuple['dict[str, int]',
                                                          'dict[str, str]']:
    """Map streamgage POI IDs from a file to NHM POI segments.

    The input file contains one streamgage POI ID per line. Blank lines and
    surrounding whitespace are ignored. Streamgages that are not POIs in the
    parameter database, that have ``poi_gage_segment == 0``, or that are
    assigned multiple segments are reported and skipped.

    :param filename: File containing streamgage POI IDs (one per line)
    :param poi_id_to_seg: Mapping of poi_gage_id to poi_gage_segment
    :returns: A tuple of (valid streamgage ID -> NHM POI segment,
              skipped streamgage ID -> reason it was skipped)
    """

    with open(filename, 'r') as fhdl:
        streamgages = [ll.strip() for ll in fhdl.read().splitlines() if ll.strip()]

    # Create dictionary of streamgage IDs to nhm_seg for POI
    segs_by_poi: 'dict[str, int]' = {}

    # Streamgages that could not be used, mapped to the reason they were skipped
    skipped: 'dict[str, str]' = {}

    for kk in streamgages:
        poiseg = poi_id_to_seg.get(kk, None)

        if poiseg is None:
            reason = 'not a POI in the parameter database'
            con.print(f'Streamgage {kk} is {reason}; skipping', style='dark_orange3')
            skipped[kk] = reason
        elif poiseg == 0:
            reason = 'poi_gage_segment = 0'
            con.print(f'Streamgage {kk} has {reason}; skipping', style='dark_orange3')
            skipped[kk] = reason
        elif kk in segs_by_poi:
            reason = 'multiple assigned segments in the parameter database'
            con.print(f'Streamgage {kk} has {reason}; skipping', style='red')
            skipped[kk] = reason
        else:
            segs_by_poi[kk] = poiseg

    return segs_by_poi, skipped


app = App(default_parameter=Parameter(negative=()))


@app.default
def by_streamgage(streamgages: Annotated[Path, Parameter(name=['--streamgages', '-s'])],
                  *bandit_args: Annotated[str, Parameter(allow_leading_hyphen=True)],
                  seg_cutoffs: Annotated[Path | None, Parameter(name=['--seg_cutoffs', '-c'])] = None,
                  nr_hrus: Annotated[Path | None, Parameter(name=['--nr_hrus', '-n'])] = None,
                  prefix: Annotated[str | None, Parameter(name=['--prefix', '-p'])] = None,
                  num_threads: Annotated[int, Parameter(name=['--num-threads', '-N'])] = 8,
                  verbose: Annotated[bool, Parameter(name=['--verbose', '-v'])] = False):
    """Batch Bandit extractions driven by a file of streamgage POI IDs.

    Each streamgage is extracted to its own directory. Per-POI upstream segment
    cutoffs and non-routed HRUs can optionally be supplied as CSV files (see
    ``seg_cutoffs`` and ``nr_hrus``) and are applied to the matching streamgage.

    Any extra tokens supplied after the known options are passed through,
    unchanged, to each ``bandit`` extraction. Use the ``--`` delimiter to
    separate pass-through flags from this command's own options, e.g.::

        bandit_by_streamgage -s sample.csv -- --param_netcdf --streamflow_netcdf

    When no pass-through arguments are supplied, the defaults
    ``--no_filter_params --keep_hru_order`` are used.

    :param streamgages: File containing streamgage POI IDs (one per line)
    :param bandit_args: Additional arguments passed through to each bandit run
    :param seg_cutoffs: File containing upstream cutoff segments for model by POI
    :param nr_hrus: File containing non-route HRUs in model by POI
    :param prefix: Directory prefix to add to each extraction output directory
    :param num_threads: Number of concurrent bandit extractions to run
    :param verbose: Output additional information
    """

    streamgages = Path(streamgages)

    if not streamgages.is_file():
        raise BanditError(f'Streamgage file does not exist: {streamgages}', exit_code=1)

    if num_threads < 1:
        raise BanditError(f'num_threads must be at least 1 (got {num_threads})', exit_code=1)

    # Arguments to pass through to each bandit extraction
    passthrough_args = list(bandit_args) if bandit_args else list(DEFAULT_BANDIT_ARGS)

    # Reject pass-through arguments that this driver controls for each run.
    reserved_used = [aa for aa in passthrough_args if aa in RESERVED_BANDIT_ARGS]
    if reserved_used:
        raise BanditError(f'These bandit arguments are set by bandit_by_streamgage and '
                          f'cannot be passed through: {" ".join(reserved_used)}', exit_code=1)

    if verbose:
        con.print(f'Pass-through bandit arguments: {" ".join(passthrough_args)}')

    # Should be in the current job directory
    job_dir = os.getcwd()

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Fail fast: validate everything that can go wrong before spawning any
    # worker threads. Nothing below should start the thread pool until there
    # is confirmed work to queue.
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    cmd_bandit = shutil.which('bandit')

    if not cmd_bandit:
        raise BanditError('Unable to find bandit executable', exit_code=1)

    # Read the default configuration file
    config_file = Path(job_dir) / 'bandit.cfg'

    if not config_file.is_file():
        raise BanditError(f'Configuration file does not exist: {config_file}', exit_code=1)

    config = bc.Cfg(config_file)

    # Update control_filename to include the main extraction path
    config.update_value('control_filename', f'{job_dir}/control.default')

    # When the config requests a specific parameter database revision, verify
    # the paramdb is actually checked out at that branch, commit, or tag before
    # launching any extractions.
    check_paramdb_revision(config)

    # Load PRMS metadata
    prms_meta = MetaData(verbose=False).metadata

    # Get the POI-to-segment mappings from the parameter database
    pdb = ParamDb(config.paramdb_dir, metadata=prms_meta, verbose=verbose)

    # Get dictionary which maps poi_gage_id to poi_gage_segment
    poi_id_to_seg = pdb.poi_to_seg

    # Read streamgage file and map IDs to NHM POI segments
    segs_by_poi, skipped_gages = get_streamgage_segments(streamgages, poi_id_to_seg)

    if not segs_by_poi:
        raise BanditError('No valid streamgages found; nothing to extract', exit_code=1)

    if seg_cutoffs is None:
        us_seg_cutoffs = {}
    else:
        if not seg_cutoffs.is_file():
            raise BanditError(f'Upstream segment cutoff file does not exist: {seg_cutoffs}', exit_code=1)
        else:
            us_seg_cutoffs = read_file(seg_cutoffs)

    if nr_hrus is None:
        nonrouted_hrus = {}
    else:
        if not nr_hrus.is_file():
            raise BanditError(f'Non-routed HRUs file does not exist: {nr_hrus}', exit_code=1)
        else:
            nonrouted_hrus = read_file(nr_hrus)

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Setup validated and there is work to do: build the per-streamgage
    # commands, writing each run's config, then run them through the pool.
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    commands: list[list[str]] = []

    # Map each streamgage to the directory its extraction runs in, so the
    # per-run summaries can be collected into a roll-up report afterward.
    dir_by_gage: 'dict[str, str]' = {}

    for kk, vv in segs_by_poi.items():
        if prefix is None:
            cdir = kk
        else:
            cdir = f'{prefix}{kk}'

        dir_by_gage[kk] = cdir

        # Create the directory if needed
        if not os.path.exists(cdir):
            try:
                os.makedirs(cdir)
            except OSError as err:
                raise BanditError(f'Error creating directory: {err}', exit_code=1)

        # Update the outlets and output_dir config variables.
        # outlets is a list of segments; a streamgage maps to a single segment.
        config.update_value('outlets', [vv])
        config.update_value('cutoffs',  us_seg_cutoffs.get(kk, []))
        config.update_value('hru_noroute', nonrouted_hrus.get(kk, []))
        config.update_value('output_dir', f'{job_dir}/{cdir}')

        config.write(f'{cdir}/bandit.cfg')

        commands.append([cmd_bandit] + passthrough_args + ['--job-dir', f'{job_dir}/{cdir}'])

    con.print(f'Total number of streamgages: {len(commands)}')

    aborted = run_worker_pool(commands, num_threads)

    # Build a roll-up report from the per-run summaries. This runs even on an
    # abort so the report reflects whatever extractions did complete.
    rows = collect_summary_rows(dir_by_gage, job_dir, skipped_gages)
    print_summary_table(rows, label='Streamgage')

    report = Path(job_dir) / 'extraction_summary.html'
    write_summary_report(report, rows, label='Streamgage')
    con.print(f'Summary report: {report}')

    if aborted:
        raise BanditError('Batch extraction stopped before all streamgages were processed',
                          exit_code=1)


def main():
    """CLI entry point for batch Bandit extractions by streamgage."""
    run_cli(app)


if __name__ == '__main__':
    main()
