"""Configuration validator for Bandit.

Validates a Bandit Cfg object before extraction begins, collecting all
errors in a single pass so the modeler can fix every issue at once.
"""

import datetime
from typing import List, Set

from Bandit.bandit_cfg import Cfg


class ConfigValidator:
    """Validates a Bandit Cfg object before extraction begins."""

    CORE_FIELDS = [
        'output_dir',
        'param_filename',
        'paramdb_dir',
        'control_filename',
        'start_date',
        'end_date',
    ]

    def __init__(self, config: Cfg):
        """Initialize the validator.

        :param config: A loaded Cfg instance to validate.
        """
        self._config = config
        self._errors: List[str] = []

    def validate(self) -> List[str]:
        """Run all validation checks and return a list of error messages.

        Returns an empty list if the configuration is valid.
        """
        self._errors = []

        failed_fields = self._validate_core_fields()
        self._validate_dates()
        self._validate_paths(failed_fields)
        self._validate_cbh_fields()
        self._validate_model_output_fields()
        self._validate_streamflow_fields()
        self._validate_gis_fields()
        self._validate_list_types()

        return self._errors

    def _validate_core_fields(self) -> Set[str]:
        """Check that core required fields are present and non-empty.

        Returns the set of field names that failed validation, so
        downstream path checks can skip them.
        """
        failed: Set[str] = set()

        for name in self.CORE_FIELDS:
            if not self._config.exists(name) or self._config.is_empty(name):
                self._errors.append(f"Required field '{name}' is missing or empty")
                failed.add(name)

        return failed

    def _validate_dates(self) -> None:
        """Check date format (YYYY-MM-DD) and that start_date < end_date."""
        parsed_dates = {}

        for name in ('start_date', 'end_date'):
            if not self._config.exists(name):
                # Field doesn't exist at all; skip (caught by core field validation).
                continue

            value = self._config.get_value(name)

            if isinstance(value, datetime.date) and not isinstance(value, datetime.datetime):
                # ruamel.yaml may parse bare dates as datetime.date objects.
                parsed_dates[name] = value
                continue

            # Skip truly empty values — they'll be caught by core field validation.
            if value is None or (isinstance(value, str) and value.strip() == ''):
                continue

            # Value should be a string — try to parse it.
            try:
                parsed_dates[name] = datetime.datetime.strptime(str(value), '%Y-%m-%d').date()
            except (ValueError, TypeError):
                self._errors.append(
                    f"Field '{name}' has invalid date format '{value}'; expected YYYY-MM-DD"
                )

        # Only check ordering if both dates were successfully parsed.
        if 'start_date' in parsed_dates and 'end_date' in parsed_dates:
            if parsed_dates['start_date'] >= parsed_dates['end_date']:
                self._errors.append(
                    f"'start_date' ({parsed_dates['start_date']}) must precede "
                    f"'end_date' ({parsed_dates['end_date']})"
                )

    def _validate_paths(self, skip_fields: Set[str]) -> None:
        """Check that required paths exist on disk.

        Skips any field in skip_fields (already flagged as missing).
        Checks cbh_dir only when output_cbh is true.
        """
        from pathlib import Path

        # Check paramdb_dir exists as a directory
        if 'paramdb_dir' not in skip_fields:
            paramdb_dir = self._config.get_value('paramdb_dir')
            if not Path(paramdb_dir).is_dir():
                self._errors.append(
                    f"Path for 'paramdb_dir' does not exist: {paramdb_dir}"
                )

        # Check control_filename exists as a file
        if 'control_filename' not in skip_fields:
            control_filename = self._config.get_value('control_filename')
            if not Path(control_filename).is_file():
                self._errors.append(
                    f"Path for 'control_filename' does not exist: {control_filename}"
                )

        # Check cbh_dir exists on disk when output_cbh is true
        if self._config.get_value('output_cbh') and 'cbh_dir' not in skip_fields:
            cbh_dir = self._config.get_value('cbh_dir')
            if not Path(cbh_dir).exists():
                self._errors.append(
                    f"Path for 'cbh_dir' does not exist: {cbh_dir}"
                )

    def _validate_cbh_fields(self) -> None:
        """When output_cbh is true, check cbh_dir and cbh_var_map."""
        if not self._config.get_value('output_cbh'):
            return

        # Check cbh_dir is present and non-empty
        if not self._config.exists('cbh_dir') or self._config.is_empty('cbh_dir'):
            self._errors.append(
                "Field 'cbh_dir' is required when 'output_cbh' is enabled"
            )

        # Check cbh_var_map is a non-empty dictionary
        cbh_var_map = self._config.get_value('cbh_var_map')
        if not isinstance(cbh_var_map, dict) or len(cbh_var_map) == 0:
            self._errors.append(
                "Field 'cbh_var_map' must be a non-empty dictionary when 'output_cbh' is enabled"
            )

    def _validate_model_output_fields(self) -> None:
        """When include_model_output is true, check output_vars_dir and output_vars."""
        if not self._config.get_value('include_model_output'):
            return

        # Check output_vars_dir is a non-empty string
        if not self._config.exists('output_vars_dir') or self._config.is_empty('output_vars_dir'):
            self._errors.append(
                "Field 'output_vars_dir' is required when 'include_model_output' is enabled"
            )

        # Check output_vars is a non-empty list
        if not self._config.exists('output_vars') or self._config.is_empty('output_vars'):
            self._errors.append(
                "Field 'output_vars' is required when 'include_model_output' is enabled"
            )

    def _validate_streamflow_fields(self) -> None:
        """When output_streamflow is true, check streamflow_filename."""
        if not self._config.get_value('output_streamflow'):
            return

        # Check streamflow_filename is a non-empty string
        if not self._config.exists('streamflow_filename') or self._config.is_empty('streamflow_filename'):
            self._errors.append(
                "Field 'streamflow_filename' is required when 'output_streamflow' is enabled"
            )

    def _validate_gis_fields(self) -> None:
        """When output_shapefiles is true, check gis structure."""
        if not self._config.get_value('output_shapefiles'):
            return

        # Check gis is a non-empty dictionary
        gis = self._config.get_value('gis')
        if not isinstance(gis, dict) or len(gis) == 0:
            self._errors.append(
                "Field 'gis' must be a non-empty dictionary when 'output_shapefiles' is enabled"
            )
            return

        # Check gis contains required keys
        required_keys = ['src_filename', 'dst_extension', 'layers']
        for key in required_keys:
            if key not in gis:
                self._errors.append(
                    f"GIS config is missing required key '{key}'"
                )

        # Check gis['src_filename'] is a non-empty string
        if 'src_filename' in gis:
            src = gis['src_filename']
            if not isinstance(src, str) or len(src.strip()) == 0:
                self._errors.append(
                    "GIS config 'src_filename' is missing or empty"
                )

        # Check gis['layers'] is a non-empty dictionary
        if 'layers' in gis:
            layers = gis['layers']
            if not isinstance(layers, dict) or len(layers) == 0:
                self._errors.append(
                    "GIS config 'layers' must be a non-empty dictionary"
                )

    def _validate_list_types(self) -> None:
        """Check that outlets, cutoffs, hru_noroute contain only integers."""
        import numpy as np

        for name in ('outlets', 'cutoffs', 'hru_noroute'):
            if not self._config.exists(name) or self._config.is_empty(name):
                continue

            values = self._config.get_value(name)
            if not isinstance(values, list):
                continue

            for elem in values:
                if not isinstance(elem, (int, np.integer)):
                    self._errors.append(
                        f"Field '{name}' contains non-integer element: {elem}"
                    )
