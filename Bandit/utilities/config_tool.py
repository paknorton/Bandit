#!/usr/bin/env python3

#      Author: Parker Norton (pnorton@usgs.gov)
# Description: Command-line utility for inspecting and maintaining Bandit
#              YAML configuration files. Provides three commands:
#                verify  - check that paths (and GIS layers) in a config exist
#                diff    - side-by-side comparison of two configs by variable
#                update  - update a target config from a source config, while
#                          leaving run-specific variables untouched

import glob
import logging
from pathlib import Path
from typing import Annotated, Any, List, Optional, Set, Tuple

import pyogrio as pyg   # type: ignore

from cyclopts import App, Parameter, validators
from pyPRMS.base.console import get_console_instance   # type: ignore
from rich.table import Table

import Bandit.bandit_cfg as bc
from Bandit.exceptions import BanditError

import warnings
warnings.filterwarnings('ignore', message=r'.*Measured \(M\) geometry types are not supported.*')

__author__ = 'Parker Norton (pnorton@usgs.gov)'

# Rich console for user-facing output
con = get_console_instance()

# Logging
cfgtool_log = logging.getLogger('config_tool')


# ---------------------------------------------------------------------------
# Variable classification
# ---------------------------------------------------------------------------

# Configuration variables that point at a filesystem path and therefore can be
# checked for existence by the verify command. The value may be a directory, a
# file, or (for poi_dir) a glob pattern.
PATH_VARS = [
    'output_dir',
    'poi_dir',
    'control_filename',
    'paramdb_dir',
    'dyn_params_dir',
    'output_vars_dir',
    'cbh_dir',
    'streamgage_file',
]

# Variables that are protected from the update command by default. These are
# specific to an individual extraction/run rather than to the shared data
# environment, so they are preserved in the target config unless the user names
# them explicitly via --include. The nested GIS key is referenced in dotted
# form ('gis.dst_extension').
PROTECTED_VARS = [
    'output_dir',
    'control_filename',
    'param_filename',
    'outlets',
    'cutoffs',
    'hru_noroute',
    'check_DAG',
    'include_model_output',
    'output_vars',
    'output_cbh',
    'output_shapefiles',
    'output_streamflow',
    'start_date',
    'end_date',
    'gis.dst_extension',
]

# Date variables get special reporting: when a date differs but is not included
# for update, a NOTE is emitted rather than silently leaving it unchanged.
DATE_VARS = [
    'start_date',
    'end_date',
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_config(filename: Path) -> bc.Cfg:
    """Load a Bandit configuration file.

    :param filename: Path to the YAML configuration file
    :returns: Loaded Cfg instance
    """
    try:
        return bc.Cfg(filename)
    except Exception as err:   # noqa: BLE001 - surface any load error to the CLI
        raise BanditError(f'Unable to load config file {filename}: {err}', exit_code=2)


def _config_keys(config: bc.Cfg) -> List[str]:
    """Return the top-level variable names present in a config file.

    :param config: Loaded Cfg instance
    :returns: Sorted list of top-level config variable names
    """
    # The underlying dict is name-mangled on the Cfg instance.
    cfgdict = getattr(config, '_Cfg__cfgdict', {})
    return list(cfgdict.keys())


def _resolve_include(include: Optional[List[str]]) -> Set[str]:
    """Parse and validate the --include option for the update command.

    Accepts values given either as a comma-separated list in a single option
    or by repeating the option. Each name must be one of the protected
    variables; an unknown name raises a BanditError so typos do not silently
    skip an intended update.

    :param include: Raw values collected from the --include option, or None
    :returns: Set of validated protected-variable names to update
    """
    if not include:
        return set()

    names: Set[str] = set()
    for raw in include:
        for part in str(raw).split(','):
            name = part.strip()
            if name:
                names.add(name)

    unknown = sorted(n for n in names if n not in PROTECTED_VARS)
    if unknown:
        allowed = ', '.join(PROTECTED_VARS)
        raise BanditError(
            f"--include has unknown variable(s): {', '.join(unknown)}. "
            f"Allowed values: {allowed}",
            exit_code=2)

    return names


# Sentinel used to distinguish "variable not present" from a real value such
# as None or an empty string when comparing two configs.
_MISSING = object()


def _diff_gis(src_gis: dict, other_gis: dict) -> List[Tuple[str, Any, Any]]:
    """Return per-key differences between two gis sections.

    Produces one (variable, src_value, other_value) tuple for each top-level
    gis key whose value differs, named as 'gis.<key>'. Keys present in only
    one of the two sections are reported with the _MISSING sentinel.

    :param src_gis: The 'gis' mapping from the source config
    :param other_gis: The 'gis' mapping from the other config
    :returns: List of (variable, src_value, other_value) tuples
    """
    rows: List[Tuple[str, Any, Any]] = []

    keys: List[str] = list(src_gis.keys())
    for key in other_gis.keys():
        if key not in keys:
            keys.append(key)

    for key in keys:
        src_val = src_gis.get(key, _MISSING)
        other_val = other_gis.get(key, _MISSING)

        if src_val is _MISSING or other_val is _MISSING or src_val != other_val:
            rows.append((f'gis.{key}', src_val, other_val))

    return rows


def _format_value(value: Any) -> str:
    """Render a config value as a readable single string for table output.

    :param value: Any config value (scalar, list, or dict)
    :returns: String representation
    """
    if value is None:
        return ''
    if isinstance(value, dict):
        return '{' + ', '.join(f'{kk}: {_format_value(vv)}' for kk, vv in value.items()) + '}'
    if isinstance(value, list):
        return '[' + ', '.join(_format_value(vv) for vv in value) + ']'
    return str(value)


def _against_base(path_str: str, base_dir: Path) -> str:
    """Resolve a possibly-relative path against the config file's directory.

    Absolute paths are returned unchanged. Relative paths (e.g. a bare
    'control.default') are resolved against the directory the config file
    lives in, which is how Bandit uses them at run time.

    :param path_str: Raw path string from the config
    :param base_dir: Directory of the config file being checked
    :returns: Resolved path string
    """
    p = Path(path_str)
    if p.is_absolute():
        return path_str
    return str(base_dir / p)


def _resolve_path_status(name: str, value: Any, base_dir: Path) -> Optional[str]:
    """Check a path-type config value and return a problem message, or None.

    :param name: Name of the config variable
    :param value: Value of the config variable
    :param base_dir: Directory of the config file (for resolving relative paths)
    :returns: Problem description string, or None when the path is OK or empty
    """
    # Empty path entries are allowed (optional features); nothing to verify.
    if value is None or (isinstance(value, str) and value.strip() == ''):
        return None

    path_str = str(value)

    # poi_dir is commonly a glob pattern (e.g. '/path/*_pois.nc'); a config is
    # valid as long as the pattern matches at least one file.
    if any(ch in path_str for ch in ['*', '?', '[']):
        matches = glob.glob(_against_base(path_str, base_dir))
        if not matches:
            return f'No files match pattern: {path_str}'
        return None

    if not Path(_against_base(path_str, base_dir)).exists():
        return f'Path does not exist: {path_str}'

    return None


def _verify_gis(gis: Any, base_dir: Path) -> List[Tuple[str, str]]:
    """Verify the GIS section: source file existence and declared layers.

    :param gis: The value of the 'gis' config variable
    :param base_dir: Directory of the config file (for resolving relative paths)
    :returns: List of (variable, problem) tuples; empty when GIS is valid
    """
    problems: List[Tuple[str, str]] = []

    if not isinstance(gis, dict) or len(gis) == 0:
        problems.append(('gis', 'GIS section is missing or empty'))
        return problems

    src_filename = gis.get('src_filename')
    if not src_filename or str(src_filename).strip() == '':
        problems.append(('gis.src_filename', 'GIS source file is missing or empty'))
        # Without a source file the layers cannot be checked.
        return problems

    src_path = Path(_against_base(str(src_filename), base_dir))
    if not src_path.exists():
        problems.append(('gis.src_filename', f'GIS source does not exist: {src_filename}'))
        return problems

    # Enumerate the layers actually present in the GIS source.
    try:
        available = {str(row[0]) for row in pyg.list_layers(src_path)}
    except Exception as err:   # noqa: BLE001 - GDAL/pyogrio read errors
        problems.append(('gis.src_filename', f'Unable to read GIS layers: {err}'))
        return problems

    layers = gis.get('layers')
    if not isinstance(layers, dict) or len(layers) == 0:
        problems.append(('gis.layers', 'No GIS layers are defined'))
        return problems

    for entry_name, layer_def in layers.items():
        if not isinstance(layer_def, dict):
            problems.append((f'gis.layers.{entry_name}', 'Layer definition is not a mapping'))
            continue

        layer_name = layer_def.get('layer')
        if not layer_name:
            problems.append((f'gis.layers.{entry_name}.layer', 'Layer name is missing'))
            continue

        if str(layer_name) not in available:
            problems.append((f'gis.layers.{entry_name}.layer',
                             f"Layer '{layer_name}' not found in GIS source"))

    return problems


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

app = App(default_parameter=Parameter(negative=()))


@app.command
def verify(config_file: Annotated[Path, Parameter(validator=validators.Path(exists=True))]):
    """Verify that paths and GIS layers in a config file are valid.

    Checks every path-type configuration variable for existence and, when GIS
    output is configured, verifies that the GIS source file exists and that
    each declared layer is present in it. Problems are reported in a table,
    one row per variable.

    :param config_file: Name of the config file to verify
    """
    config = _load_config(config_file)
    base_dir = config_file.resolve().parent

    problems: List[Tuple[str, str]] = []

    # Check path-type variables.
    for name in PATH_VARS:
        if not config.exists(name):
            continue
        msg = _resolve_path_status(name, config.get_value(name), base_dir)
        if msg is not None:
            problems.append((name, msg))

    # Check the GIS section when it is present and non-empty.
    if config.exists('gis'):
        gis = config.get_value('gis')
        if isinstance(gis, dict) and len(gis) > 0:
            problems.extend(_verify_gis(gis, base_dir))

    if not problems:
        con.print(f'[green4]OK[/]: No problems found in {config_file}')
        return

    table = Table(title=f'Configuration problems in {config_file}',
                  show_lines=True, title_style='bold red')
    table.add_column('Variable', style='cyan', no_wrap=True)
    table.add_column('Problem', style='red')

    for name, msg in problems:
        table.add_row(name, msg)

    con.print(table)
    raise BanditError(f'{len(problems)} problem(s) found in {config_file}', exit_code=2)


@app.command
def diff(source_config: Annotated[Path, Parameter(validator=validators.Path(exists=True))],
         config_file: Annotated[Path, Parameter(validator=validators.Path(exists=True))]):
    """Show a side-by-side comparison of the differences between two configs.

    Compares every variable found in either config and reports those whose
    values differ, one row per variable, with the source value and the other
    config's value side-by-side.

    :param source_config: Name of the source config file
    :param config_file: Name of the other config file to compare against
    """
    src = _load_config(source_config)
    other = _load_config(config_file)

    # Union of variables present in either config, preserving a stable order.
    names: List[str] = _config_keys(src)
    for name in _config_keys(other):
        if name not in names:
            names.append(name)

    rows: List[Tuple[str, Any, Any]] = []

    for name in names:
        src_val = src.get_value(name) if src.exists(name) else _MISSING
        other_val = other.get_value(name) if other.exists(name) else _MISSING

        if src_val is _MISSING or other_val is _MISSING or src_val != other_val:
            # For the nested gis section, report only the specific keys that
            # differ rather than dumping the whole dict.
            if name == 'gis' and isinstance(src_val, dict) and isinstance(other_val, dict):
                rows.extend(_diff_gis(src_val, other_val))
            else:
                rows.append((name, src_val, other_val))

    if not rows:
        con.print(f'[green4]OK[/]: {source_config} and {config_file} have no differences')
        return

    table = Table(title=f'Differences: {source_config} vs {config_file}',
                  show_lines=True, title_style='bold')
    table.add_column('Variable', style='cyan', no_wrap=True)
    table.add_column(f'{source_config.name} (source)', style='green4', overflow='fold')
    table.add_column(f'{config_file.name}', style='yellow', overflow='fold')

    for name, src_val, other_val in rows:
        src_disp = '[dim](not set)[/]' if src_val is _MISSING else _format_value(src_val)
        other_disp = '[dim](not set)[/]' if other_val is _MISSING else _format_value(other_val)
        table.add_row(name, src_disp, other_disp)

    con.print(table)


@app.command
def update(source_config: Annotated[Path, Parameter(validator=validators.Path(exists=True))],
           config_file: Annotated[Path, Parameter(validator=validators.Path(exists=True))],
           include: Optional[List[str]] = None):
    """Update a config file from a source config file.

    Copies values from the source config into the target config, while leaving
    run-specific variables untouched by default (output_dir, control_filename,
    param_filename, outlets, cutoffs, hru_noroute, check_DAG,
    include_model_output, output_vars, output_cbh, output_shapefiles,
    output_streamflow, start_date, end_date, and gis.dst_extension).

    Use --include to allow one or more of these protected variables to be
    updated as well, as a comma-separated list or by repeating the option, e.g.
    '--include output_dir,start_date,end_date' or '--include output_dir
    --include start_date'. The nested GIS key is named in dotted form
    (gis.dst_extension). A protected date that differs but is not included for
    update produces a NOTE instead of being changed.

    Updated variables are reported in a table showing the original and new
    values. The target config file is rewritten in place.

    :param source_config: Name of the source config file to copy values from
    :param config_file: Name of the config file to update
    :param include: Protected variables to also update (comma-separated or repeated)
    """
    src = _load_config(source_config)
    target = _load_config(config_file)

    include_set = _resolve_include(include)

    updated: List[Tuple[str, Any, Any]] = []

    # Report date mismatches that are not being updated, so the user is aware
    # of the difference even though the target is left unchanged.
    for name in DATE_VARS:
        if name in include_set:
            continue
        if src.exists(name) and target.exists(name):
            src_val = src.get_value(name)
            target_val = target.get_value(name)
            if src_val != target_val:
                con.print(f'[gold3]NOTE[/]: {name} differs - '
                          f'source={_format_value(src_val)}, '
                          f'target={_format_value(target_val)} (not updated)')

    for name in _config_keys(src):
        # Skip protected variables unless the user opted them in via --include.
        if name in PROTECTED_VARS and name not in include_set:
            continue

        src_val = src.get_value(name)

        # Handle the GIS section specially so gis.dst_extension is preserved
        # (unless included) while the other GIS keys are updated.
        if name == 'gis':
            new_gis, gis_updates = _merge_gis(target, src_val, include_set)
            if gis_updates:
                target.update_value('gis', new_gis)
                updated.extend(gis_updates)
            continue

        if not target.exists(name):
            # Variable not present in the target; skip rather than inject new keys.
            continue

        target_val = target.get_value(name)
        if src_val != target_val:
            target.update_value(name, src_val)
            updated.append((name, target_val, src_val))

    if not updated:
        con.print(f'[green4]OK[/]: Nothing to update in {config_file}')
        return

    # Prevent ruamel from wrapping long plain scalars (e.g. absolute paths)
    # onto a second indented line, which would otherwise change how simple
    # hand-authored config entries look after a rewrite.
    target.yaml.width = 4096
    target.write(str(config_file))

    table = Table(title=f'Updated variables in {config_file}',
                  show_lines=True, title_style='bold')
    table.add_column('Variable', style='cyan', no_wrap=True)
    table.add_column('Original value', style='red', overflow='fold')
    table.add_column('Updated value', style='green4', overflow='fold')

    for name, old_val, new_val in updated:
        table.add_row(name, _format_value(old_val), _format_value(new_val))

    con.print(table)
    con.print(f'[green4]OK[/]: Updated {len(updated)} variable(s) in {config_file}')


def _merge_gis(target: bc.Cfg, src_gis: Any,
               include_set: Set[str]) -> Tuple[Any, List[Tuple[str, Any, Any]]]:
    """Merge source GIS settings into the target, preserving protected keys.

    :param target: The target Cfg instance being updated
    :param src_gis: The 'gis' value from the source config
    :param include_set: Protected variable names the user opted in for update
    :returns: Tuple of (new gis value, list of (variable, old, new) updates)
    """
    updates: List[Tuple[str, Any, Any]] = []

    if not target.exists('gis'):
        return src_gis, updates

    target_gis = target.get_value('gis')

    # If either side is not a dict, fall back to a straight comparison.
    if not isinstance(src_gis, dict) or not isinstance(target_gis, dict):
        if src_gis != target_gis:
            updates.append(('gis', target_gis, src_gis))
            return src_gis, updates
        return target_gis, updates

    # Work on the existing target gis object so ruamel formatting is preserved.
    new_gis = target_gis

    for key, src_val in src_gis.items():
        dotted = f'gis.{key}'
        # Skip protected nested keys unless the user opted them in via --include.
        if dotted in PROTECTED_VARS and dotted not in include_set:
            continue

        old_val = new_gis.get(key)
        if old_val != src_val:
            new_gis[key] = src_val
            updates.append((dotted, old_val, src_val))

    return new_gis, updates


def main():
    """CLI entry point."""
    try:
        app()
    except BanditError as err:
        con.print(f'[red]ERROR[/]: {err}')
        raise SystemExit(err.exit_code)


if __name__ == '__main__':
    main()
