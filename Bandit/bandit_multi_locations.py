#!/usr/bin/env python3

"""Batch Bandit extractions driven by a file of named locations.

Each location has one or more stream segment outlets and, optionally, upstream
segment cutoffs and non-routed HRUs. The outlets file has the format::

    name,outlets
    mill,1242
    norwalk,1244,1245
    quinnipiac,868

The location name in the first column may be a string (e.g. ``mill``) or an
integer (e.g. a headwater area ID). The optional cutoffs and non-routed HRU
files use the same first-column location key followed by integer segment or
HRU IDs.

Each location is extracted into its own output directory and the per-run
summaries are collected into a console table and an HTML roll-up report.
"""

import os
import shutil

from cyclopts import App, Parameter
from pathlib import Path
from typing import Annotated, Union

from Bandit import bandit_cfg as bc
from Bandit.cli_shared import (check_paramdb_revision, collect_summary_rows, con,
                               print_summary_table, RESERVED_BANDIT_ARGS, run_cli,
                               run_worker_pool, write_summary_report)
from Bandit.exceptions import BanditError

__author__ = 'Parker Norton (pnorton@usgs.gov)'

# Default arguments passed through to each bandit extraction when the
# user does not supply any on the command line.
DEFAULT_BANDIT_ARGS = ['--keep_hru_order']

# A location key may be a string name or an integer ID.
LocationId = Union[str, int]


def _location_key(raw: str) -> LocationId:
    """Normalize a raw location key to an int when possible, else a string.

    :param raw: Location key as read from the CSV first column
    :returns: Integer location ID if ``raw`` is an integer, otherwise the string
    """
    try:
        return int(raw)
    except ValueError:
        return raw


def read_outlets(filename: str | Path) -> 'dict[LocationId, list[int]]':
    """Read a CSV of a location ID followed by one or more integer values.

    The file has a header row, then one location ID per row followed by one or
    more integer segment outlet IDs::

        name,outlets
        mill,1242
        norwalk,1244,1245

    The location ID may be a string name or an integer. Blank lines and
    surrounding whitespace are ignored. Insertion order is preserved.

    :param filename: File to read
    :returns: Mapping of location ID to its list of integer values
    :raises BanditError: If a non-header row has no values, a non-integer value,
                         or a duplicate location ID
    """

    values_by_id: 'dict[LocationId, list[int]]' = {}

    with open(filename, 'r') as src_file:
        # Skip the header row
        src_file.readline()

        for line in src_file:
            cols = [cc.strip() for cc in line.strip().split(',')]

            if not cols or not cols[0]:
                # Skip blank lines (and lines with no location ID)
                continue

            loc = _location_key(cols[0])

            if loc in values_by_id:
                raise BanditError(f'Duplicate location {loc} in {filename}', exit_code=1)

            if len(cols) < 2 or not any(cols[1:]):
                raise BanditError(f'Location {loc} in {filename} has no values', exit_code=1)

            try:
                values_by_id[loc] = [int(xx) for xx in cols[1:] if xx]
            except ValueError as err:
                raise BanditError(f'Non-integer value in {filename} for location {loc}: {err}',
                                  exit_code=1)

    return values_by_id


def read_values(filename: str | Path) -> 'dict[LocationId, list[int]]':
    """Read a CSV of a location ID followed by zero or more integer values.

    Used for the optional upstream segment cutoff and non-routed HRU files,
    which share the outlets-file format but may legitimately have no values for
    a location.

    :param filename: File to read
    :returns: Mapping of location ID to its list of integer values
    :raises BanditError: If a non-header row contains a non-integer value or a
                         duplicate location ID
    """

    values_by_id: 'dict[LocationId, list[int]]' = {}

    with open(filename, 'r') as src_file:
        # Skip the header row
        src_file.readline()

        for line in src_file:
            cols = [cc.strip() for cc in line.strip().split(',')]

            if not cols or not cols[0]:
                # Skip blank lines (and lines with no location ID)
                continue

            loc = _location_key(cols[0])

            if loc in values_by_id:
                raise BanditError(f'Duplicate location {loc} in {filename}', exit_code=1)

            try:
                values_by_id[loc] = [int(xx) for xx in cols[1:] if xx]
            except ValueError as err:
                raise BanditError(f'Non-integer value in {filename} for location {loc}: {err}',
                                  exit_code=1)

    return values_by_id


def location_dirname(loc: LocationId, prefix: str | None) -> str:
    """Build the output directory name for a location.

    Integer location IDs are zero-padded to four digits (e.g. ``0003``) to
    match the historical headwater-area naming; string names are used as-is.

    :param loc: Location ID (string name or integer)
    :param prefix: Optional directory prefix to prepend
    :returns: Directory name for the location's extraction
    """
    if isinstance(loc, int):
        base = f'{loc:04d}'
    else:
        base = f'{loc}'

    return base if prefix is None else f'{prefix}{base}'


app = App(default_parameter=Parameter(negative=()))


@app.default
def multi_locations(seg_outlets: Annotated[Path, Parameter(name=['--seg_outlets', '-s'])],
                    *bandit_args: Annotated[str, Parameter(allow_leading_hyphen=True)],
                    seg_cutoffs: Annotated[Path | None, Parameter(name=['--seg_cutoffs', '-c'])] = None,
                    nr_hrus: Annotated[Path | None, Parameter(name=['--nr_hrus', '-n'])] = None,
                    prefix: Annotated[str | None, Parameter(name=['--prefix', '-p'])] = None,
                    num_threads: Annotated[int, Parameter(name=['--num-threads', '-N'])] = 8,
                    verbose: Annotated[bool, Parameter(name=['--verbose', '-v'])] = False):
    """Batch Bandit extractions driven by a file of named locations.

    Each location is extracted to its own directory and may have one or more
    stream segment outlets. Per-location upstream segment cutoffs and non-routed
    HRUs can optionally be supplied as CSV files (see ``seg_cutoffs`` and
    ``nr_hrus``) and are applied to the matching location.

    Any extra tokens supplied after the known options are passed through,
    unchanged, to each ``bandit`` extraction. Use the ``--`` delimiter to
    separate pass-through flags from this command's own options, e.g.::

        bandit_multi_locations -s basin_outlets.csv -- --param_netcdf

    When no pass-through arguments are supplied, the default ``--keep_hru_order``
    is used.

    :param seg_outlets: File containing segment outlets by location
    :param bandit_args: Additional arguments passed through to each bandit run
    :param seg_cutoffs: File containing upstream cutoff segments by location
    :param nr_hrus: File containing non-routed HRUs by location
    :param prefix: Directory prefix to add to each extraction output directory
    :param num_threads: Number of concurrent bandit extractions to run
    :param verbose: Output additional information
    """

    seg_outlets = Path(seg_outlets)

    if not seg_outlets.is_file():
        raise BanditError(f'Segment outlets file does not exist: {seg_outlets}', exit_code=1)

    if num_threads < 1:
        raise BanditError(f'num_threads must be at least 1 (got {num_threads})', exit_code=1)

    # Arguments to pass through to each bandit extraction
    passthrough_args = list(bandit_args) if bandit_args else list(DEFAULT_BANDIT_ARGS)

    # Reject pass-through arguments that this driver controls for each run.
    reserved_used = [aa for aa in passthrough_args if aa in RESERVED_BANDIT_ARGS]
    if reserved_used:
        raise BanditError(f'These bandit arguments are set by bandit_multi_locations and '
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

    # Read the segment outlets by location
    segments_by_loc = read_outlets(seg_outlets)

    if not segments_by_loc:
        raise BanditError('No locations found; nothing to extract', exit_code=1)

    if seg_cutoffs is None:
        uscutoff_seg_by_loc: dict = {}
    else:
        seg_cutoffs = Path(seg_cutoffs)
        if not seg_cutoffs.is_file():
            raise BanditError(f'Upstream segment cutoff file does not exist: {seg_cutoffs}', exit_code=1)
        uscutoff_seg_by_loc = read_values(seg_cutoffs)

    if nr_hrus is None:
        noroute_hrus_by_loc: dict = {}
    else:
        nr_hrus = Path(nr_hrus)
        if not nr_hrus.is_file():
            raise BanditError(f'Non-routed HRUs file does not exist: {nr_hrus}', exit_code=1)
        noroute_hrus_by_loc = read_values(nr_hrus)

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Setup validated and there is work to do: build the per-location commands,
    # writing each run's config, then run them through the pool.
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    commands: list[list[str]] = []

    # Map each location to the directory its extraction runs in, so the
    # per-run summaries can be collected into a roll-up report afterward.
    dir_by_loc: 'dict[LocationId, str]' = {}

    for kk, vv in segments_by_loc.items():
        cdir = location_dirname(kk, prefix)

        dir_by_loc[kk] = cdir

        # Create the location directory if needed
        if not os.path.exists(cdir):
            try:
                os.makedirs(cdir)
            except OSError as err:
                raise BanditError(f'Error creating directory: {err}', exit_code=1)

        # Update the outlets, cutoffs, non-routed HRUs, and output_dir config
        # variables. outlets is a list of one or more segments.
        config.update_value('outlets', vv)
        config.update_value('cutoffs', uscutoff_seg_by_loc.get(kk, []))
        config.update_value('hru_noroute', noroute_hrus_by_loc.get(kk, []))
        config.update_value('output_dir', f'{job_dir}/{cdir}')

        config.write(f'{cdir}/bandit.cfg')

        commands.append([cmd_bandit] + passthrough_args + ['--job-dir', f'{job_dir}/{cdir}'])

    con.print(f'Total number of locations: {len(commands)}')

    aborted = run_worker_pool(commands, num_threads)

    # Build a roll-up report from the per-run summaries. This runs even on an
    # abort so the report reflects whatever extractions did complete.
    rows = collect_summary_rows(dir_by_loc, job_dir)
    print_summary_table(rows, label='Location')

    report = Path(job_dir) / 'extraction_summary.html'
    write_summary_report(report, rows, label='Location')
    con.print(f'Summary report: {report}')

    if aborted:
        raise BanditError('Batch extraction stopped before all locations were processed',
                          exit_code=1)


def main():
    """CLI entry point for batch Bandit extractions by location."""
    run_cli(app)


if __name__ == '__main__':
    main()
