"""Tests for ConfigValidator."""

import datetime
from unittest.mock import MagicMock

import pytest

from Bandit.config_validator import ConfigValidator


def _make_config(**overrides):
    """Create a mock Cfg with sensible defaults, overridden by kwargs.

    All core fields are valid by default so tests can focus on the
    specific behaviour under test.
    """
    defaults = {
        'output_dir': '/tmp/output',
        'param_filename': 'myparam.param',
        'paramdb_dir': '/tmp/paramdb',
        'control_filename': 'control.default',
        'start_date': '1980-01-01',
        'end_date': '2010-12-31',
        'outlets': [],
        'cutoffs': [],
        'hru_noroute': [],
        'output_cbh': False,
        'cbh_dir': '',
        'cbh_var_map': {},
        'include_model_output': False,
        'output_vars_dir': '',
        'output_vars': [],
        'output_streamflow': False,
        'streamflow_filename': 'sf_data',
        'output_shapefiles': False,
        'gis': {},
    }
    defaults.update(overrides)

    cfg = MagicMock()
    cfg.exists.side_effect = lambda name: name in defaults
    cfg.is_empty.side_effect = lambda name: (
        defaults[name] == '' or defaults[name] == [] or defaults[name] == {}
        if isinstance(defaults[name], (str, list, dict)) else False
    )
    cfg.get_value.side_effect = lambda name: defaults[name]
    return cfg


class TestValidateDates:
    """Tests for _validate_dates method."""

    def test_valid_dates_no_errors(self):
        cfg = _make_config(start_date='2000-01-01', end_date='2010-12-31')
        v = ConfigValidator(cfg)
        errors = v.validate()
        date_errors = [e for e in errors if 'date' in e.lower()]
        assert date_errors == []

    def test_invalid_start_date_format(self):
        cfg = _make_config(start_date='not-a-date', end_date='2010-12-31')
        v = ConfigValidator(cfg)
        errors = v.validate()
        assert any("Field 'start_date' has invalid date format 'not-a-date'" in e for e in errors)

    def test_invalid_end_date_format(self):
        cfg = _make_config(start_date='2000-01-01', end_date='31-12-2010')
        v = ConfigValidator(cfg)
        errors = v.validate()
        assert any("Field 'end_date' has invalid date format '31-12-2010'" in e for e in errors)

    def test_start_date_equals_end_date(self):
        cfg = _make_config(start_date='2010-06-15', end_date='2010-06-15')
        v = ConfigValidator(cfg)
        errors = v.validate()
        assert any("'start_date'" in e and "must precede" in e for e in errors)

    def test_start_date_after_end_date(self):
        cfg = _make_config(start_date='2020-01-01', end_date='2010-01-01')
        v = ConfigValidator(cfg)
        errors = v.validate()
        assert any("'start_date' (2020-01-01) must precede 'end_date' (2010-01-01)" in e for e in errors)

    def test_datetime_date_objects_accepted(self):
        """ruamel.yaml may parse bare dates as datetime.date objects."""
        cfg = _make_config(
            start_date=datetime.date(2000, 1, 1),
            end_date=datetime.date(2010, 12, 31),
        )
        v = ConfigValidator(cfg)
        errors = v.validate()
        date_errors = [e for e in errors if 'date' in e.lower()]
        assert date_errors == []

    def test_datetime_date_ordering_check(self):
        """datetime.date objects should also be checked for ordering."""
        cfg = _make_config(
            start_date=datetime.date(2020, 1, 1),
            end_date=datetime.date(2010, 1, 1),
        )
        v = ConfigValidator(cfg)
        errors = v.validate()
        assert any("must precede" in e for e in errors)

    def test_empty_dates_skipped(self):
        """Empty date fields are caught by core validation, not date validation."""
        cfg = _make_config(start_date='', end_date='')
        v = ConfigValidator(cfg)
        errors = v.validate()
        # Should have "missing or empty" errors but no date format errors
        format_errors = [e for e in errors if 'invalid date format' in e]
        assert format_errors == []

    def test_both_dates_invalid_format(self):
        cfg = _make_config(start_date='abc', end_date='xyz')
        v = ConfigValidator(cfg)
        errors = v.validate()
        format_errors = [e for e in errors if 'invalid date format' in e]
        assert len(format_errors) == 2

    def test_only_start_invalid_no_ordering_check(self):
        """If start_date is unparseable, ordering check should be skipped."""
        cfg = _make_config(start_date='bad', end_date='2010-12-31')
        v = ConfigValidator(cfg)
        errors = v.validate()
        assert any("invalid date format" in e for e in errors)
        assert not any("must precede" in e for e in errors)


# ---------------------------------------------------------------------------
# Property-based tests (Hypothesis)
# ---------------------------------------------------------------------------
from hypothesis import given, settings, assume, HealthCheck
from hypothesis import strategies as st


# Feature: config-validation, Property 1: Core field presence detection
class TestPropertyCoreFieldPresence:
    """Property 1: For any subset of core fields that are missing or empty,
    the validator SHALL return an error for each such field, and the error
    message SHALL contain the field name.

    **Validates: Requirements 1.1, 1.2, 1.3**
    """

    CORE_FIELDS = [
        'output_dir', 'param_filename', 'paramdb_dir',
        'control_filename', 'start_date', 'end_date',
    ]

    @given(
        omit_fields=st.lists(
            st.sampled_from([
                'output_dir', 'param_filename', 'paramdb_dir',
                'control_filename', 'start_date', 'end_date',
            ]),
            unique=True,
            min_size=0,
            max_size=6,
        ),
        use_empty=st.booleans(),
    )
    @settings(max_examples=100)
    def test_core_field_presence(self, omit_fields, use_empty):
        """Each omitted or emptied core field produces an error containing its name."""
        overrides = {}
        for field in omit_fields:
            if use_empty:
                # Set to empty string (is_empty returns True for '')
                overrides[field] = ''
            else:
                # Set to empty string as well — the mock's exists still returns True
                # but is_empty returns True for ''
                overrides[field] = ''

        cfg = _make_config(**overrides)
        v = ConfigValidator(cfg)
        errors = v.validate()

        # Filter to only "missing or empty" errors
        missing_errors = [e for e in errors if 'missing or empty' in e]

        # Each omitted field should produce exactly one error containing its name
        for field in omit_fields:
            matching = [e for e in missing_errors if field in e]
            assert len(matching) == 1, (
                f"Expected exactly 1 'missing or empty' error for '{field}', "
                f"got {len(matching)}: {matching}"
            )

        # Fields NOT in omit_fields should NOT produce missing errors
        for field in self.CORE_FIELDS:
            if field not in omit_fields:
                matching = [e for e in missing_errors if field in e]
                assert len(matching) == 0, (
                    f"Unexpected 'missing or empty' error for '{field}': {matching}"
                )


# Feature: config-validation, Property 2: Date format validation
class TestPropertyDateFormatValidation:
    """Property 2: For any string value assigned to start_date or end_date,
    the validator SHALL produce a date-format error if and only if the string
    is not parseable as a valid YYYY-MM-DD date.

    **Validates: Requirements 3.1, 3.2, 3.3**
    """

    @staticmethod
    def _is_valid_date_str(s):
        """Check if a string is a valid YYYY-MM-DD date."""
        try:
            datetime.datetime.strptime(s, '%Y-%m-%d')
            return True
        except (ValueError, TypeError):
            return False

    # Strategy: valid YYYY-MM-DD date strings
    valid_date_st = st.dates(
        min_value=datetime.date(1900, 1, 1),
        max_value=datetime.date(2100, 12, 31),
    ).map(lambda d: d.strftime('%Y-%m-%d'))

    # Strategy: invalid date strings (not parseable as YYYY-MM-DD)
    invalid_date_st = st.one_of(
        st.text(min_size=1, max_size=20).filter(
            lambda s: not __class__._is_valid_date_str(s) and s.strip() != ''
        ),
        st.just('2024-13-01'),   # invalid month
        st.just('2024-02-30'),   # invalid day
        st.just('01-01-2024'),   # wrong format
        st.just('not-a-date'),
    )

    # Strategy: datetime.date objects (YAML type coercion)
    date_object_st = st.dates(
        min_value=datetime.date(1900, 1, 1),
        max_value=datetime.date(2100, 12, 31),
    )

    # Combined strategy for date values
    date_value_st = st.one_of(valid_date_st, invalid_date_st, date_object_st)

    @given(date_value=st.one_of(
        st.dates(
            min_value=datetime.date(1900, 1, 1),
            max_value=datetime.date(2100, 12, 31),
        ).map(lambda d: d.strftime('%Y-%m-%d')),
        st.one_of(
            st.text(min_size=1, max_size=20),
            st.just('2024-13-01'),
            st.just('2024-02-30'),
            st.just('01-01-2024'),
            st.just('not-a-date'),
        ),
        st.dates(
            min_value=datetime.date(1900, 1, 1),
            max_value=datetime.date(2100, 12, 31),
        ),
    ))
    @settings(max_examples=100)
    def test_date_format_validation(self, date_value):
        """Error iff date value is not parseable as YYYY-MM-DD (or a datetime.date)."""
        # Skip empty strings — those are caught by core field validation, not date format
        if isinstance(date_value, str) and date_value.strip() == '':
            assume(False)

        # Use a valid end_date far in the future so we only test start_date format
        cfg = _make_config(start_date=date_value, end_date='2099-12-31')
        v = ConfigValidator(cfg)
        errors = v.validate()

        format_errors = [e for e in errors if "invalid date format" in e and "start_date" in e]

        if isinstance(date_value, datetime.date):
            # datetime.date objects are always valid
            assert len(format_errors) == 0, (
                f"datetime.date {date_value} should not produce format error: {format_errors}"
            )
        elif isinstance(date_value, str) and self._is_valid_date_str(date_value):
            assert len(format_errors) == 0, (
                f"Valid date string '{date_value}' should not produce format error: {format_errors}"
            )
        else:
            assert len(format_errors) == 1, (
                f"Invalid date value '{date_value}' should produce exactly 1 format error, "
                f"got {len(format_errors)}: {format_errors}"
            )
            assert str(date_value) in format_errors[0]


# Feature: config-validation, Property 3: Date ordering
class TestPropertyDateOrdering:
    """Property 3: For any two valid YYYY-MM-DD date strings, the validator
    SHALL produce a date-ordering error if and only if start_date >= end_date.

    **Validates: Requirements 3.4**
    """

    @given(
        start=st.dates(
            min_value=datetime.date(1900, 1, 1),
            max_value=datetime.date(2100, 12, 31),
        ),
        end=st.dates(
            min_value=datetime.date(1900, 1, 1),
            max_value=datetime.date(2100, 12, 31),
        ),
    )
    @settings(max_examples=100)
    def test_date_ordering(self, start, end):
        """Error iff start_date >= end_date."""
        cfg = _make_config(
            start_date=start.strftime('%Y-%m-%d'),
            end_date=end.strftime('%Y-%m-%d'),
        )
        v = ConfigValidator(cfg)
        errors = v.validate()

        ordering_errors = [e for e in errors if 'must precede' in e]

        if start >= end:
            assert len(ordering_errors) == 1, (
                f"start={start}, end={end}: expected ordering error, got {ordering_errors}"
            )
            assert str(start) in ordering_errors[0]
            assert str(end) in ordering_errors[0]
        else:
            assert len(ordering_errors) == 0, (
                f"start={start}, end={end}: unexpected ordering error: {ordering_errors}"
            )


# Feature: config-validation, Property 4: Conditional field validation
class TestPropertyConditionalFieldValidation:
    """Property 4: For any conditional toggle and its associated required fields,
    the validator SHALL produce an error for a field if and only if the toggle
    is true and the field is missing or empty.

    **Validates: Requirements 4.1, 4.2, 4.3, 4.4, 5.1, 5.2, 5.3, 5.4, 6.1, 6.2**
    """

    # Mapping of toggles to their required fields
    TOGGLE_FIELDS = {
        'output_cbh': ['cbh_dir', 'cbh_var_map'],
        'include_model_output': ['output_vars_dir', 'output_vars'],
        'output_streamflow': ['streamflow_filename'],
    }

    # Non-empty values for each conditional field
    NON_EMPTY_VALUES = {
        'cbh_dir': '/tmp/cbh',
        'cbh_var_map': {'tmax': 'tmax.cbh', 'tmin': 'tmin.cbh'},
        'output_vars_dir': '/tmp/output_vars',
        'output_vars': ['seg_outflow'],
        'streamflow_filename': 'sf_data',
    }

    # Empty values for each conditional field
    EMPTY_VALUES = {
        'cbh_dir': '',
        'cbh_var_map': {},
        'output_vars_dir': '',
        'output_vars': [],
        'streamflow_filename': '',
    }

    @given(
        toggle_states=st.fixed_dictionaries({
            'output_cbh': st.booleans(),
            'include_model_output': st.booleans(),
            'output_streamflow': st.booleans(),
        }),
        empty_fields=st.lists(
            st.sampled_from([
                'cbh_dir', 'cbh_var_map', 'output_vars_dir',
                'output_vars', 'streamflow_filename',
            ]),
            unique=True,
            min_size=0,
            max_size=5,
        ),
    )
    @settings(max_examples=100)
    def test_conditional_field_validation(self, toggle_states, empty_fields):
        """Error iff toggle is true and associated field is empty."""
        overrides = dict(toggle_states)

        # Set field values: empty or non-empty
        for field in self.NON_EMPTY_VALUES:
            if field in empty_fields:
                overrides[field] = self.EMPTY_VALUES[field]
            else:
                overrides[field] = self.NON_EMPTY_VALUES[field]

        cfg = _make_config(**overrides)
        v = ConfigValidator(cfg)
        errors = v.validate()

        # Check each toggle/field combination
        for toggle, fields in self.TOGGLE_FIELDS.items():
            for field in fields:
                # Find errors mentioning this field and its toggle
                field_errors = [
                    e for e in errors
                    if field in e and toggle.replace('_', '') in e.replace('_', '')
                    or (field in e and 'required when' in e)
                    or (field in e and 'non-empty dictionary when' in e)
                ]

                should_error = toggle_states[toggle] and field in empty_fields

                if should_error:
                    assert len(field_errors) >= 1, (
                        f"toggle={toggle}=True, field={field} empty: "
                        f"expected error, got none. All errors: {errors}"
                    )
                elif not toggle_states[toggle]:
                    # When toggle is off, no error for this field from conditional checks
                    conditional_errors = [
                        e for e in errors
                        if field in e and ('required when' in e or 'non-empty dictionary when' in e)
                    ]
                    assert len(conditional_errors) == 0, (
                        f"toggle={toggle}=False: unexpected conditional error for "
                        f"'{field}': {conditional_errors}"
                    )


# Feature: config-validation, Property 5: GIS structure validation
class TestPropertyGISStructureValidation:
    """Property 5: For any configuration where output_shapefiles is true,
    the validator SHALL produce an error if gis is not a non-empty dictionary,
    or if gis is missing any of the required keys.

    **Validates: Requirements 7.1, 7.2, 7.3, 7.4, 7.5**
    """

    REQUIRED_KEYS = ['src_filename', 'dst_extension', 'layers']

    @given(
        shapefiles_enabled=st.booleans(),
        present_keys=st.lists(
            st.sampled_from(['src_filename', 'dst_extension', 'layers']),
            unique=True,
            min_size=0,
            max_size=3,
        ),
        src_empty=st.booleans(),
        layers_empty=st.booleans(),
    )
    @settings(max_examples=100)
    def test_gis_structure_validation(self, shapefiles_enabled, present_keys,
                                      src_empty, layers_empty):
        """Errors match missing/empty keys when output_shapefiles is true."""
        if shapefiles_enabled and len(present_keys) > 0:
            # Build a gis dict with only the present keys
            gis = {}
            for key in present_keys:
                if key == 'src_filename':
                    gis[key] = '' if src_empty else 'source.shp'
                elif key == 'layers':
                    gis[key] = {} if layers_empty else {'nhru': 'nhru_layer'}
                elif key == 'dst_extension':
                    gis[key] = '.shp'
        elif shapefiles_enabled:
            # Empty dict
            gis = {}
        else:
            gis = {}

        cfg = _make_config(output_shapefiles=shapefiles_enabled, gis=gis)
        v = ConfigValidator(cfg)
        errors = v.validate()

        gis_errors = [e for e in errors if 'gis' in e.lower() or 'GIS' in e]

        if not shapefiles_enabled:
            # No GIS errors when shapefiles disabled
            assert len(gis_errors) == 0, (
                f"Shapefiles disabled but got GIS errors: {gis_errors}"
            )
        elif len(present_keys) == 0:
            # Empty gis dict → "must be a non-empty dictionary" error
            assert any('non-empty dictionary' in e for e in gis_errors), (
                f"Empty gis dict should produce 'non-empty dictionary' error: {gis_errors}"
            )
        else:
            # Check missing keys produce errors
            for key in self.REQUIRED_KEYS:
                missing_key_errors = [e for e in gis_errors if f"'{key}'" in e]
                if key not in present_keys:
                    assert len(missing_key_errors) >= 1, (
                        f"Missing key '{key}' should produce error: {gis_errors}"
                    )

            # Check src_filename empty produces error
            if 'src_filename' in present_keys and src_empty:
                src_errors = [e for e in gis_errors if 'src_filename' in e and ('empty' in e or 'missing' in e)]
                assert len(src_errors) >= 1, (
                    f"Empty src_filename should produce error: {gis_errors}"
                )

            # Check layers empty produces error
            if 'layers' in present_keys and layers_empty:
                layers_errors = [e for e in gis_errors if 'layers' in e and 'non-empty dictionary' in e]
                assert len(layers_errors) >= 1, (
                    f"Empty layers should produce error: {gis_errors}"
                )


# Feature: config-validation, Property 6: List element type validation
class TestPropertyListElementTypeValidation:
    """Property 6: For any of the list fields that contain at least one element,
    the validator SHALL produce an error for each non-integer element.

    **Validates: Requirements 8.1, 8.2, 8.3, 8.4**
    """

    # Strategy for list elements: mix of ints and non-ints
    int_element = st.integers(min_value=-10000, max_value=10000)
    non_int_element = st.one_of(
        st.text(min_size=1, max_size=10),
        st.floats(allow_nan=False, allow_infinity=False),
        st.just(None),
        st.just(True),
    )
    mixed_element = st.one_of(int_element, non_int_element)

    @given(
        field_name=st.sampled_from(['outlets', 'cutoffs', 'hru_noroute']),
        elements=st.lists(
            st.one_of(
                st.integers(min_value=-10000, max_value=10000),
                st.text(min_size=1, max_size=10),
                st.floats(allow_nan=False, allow_infinity=False),
                st.just(None),
            ),
            min_size=1,
            max_size=10,
        ),
    )
    @settings(max_examples=100)
    def test_list_element_types(self, field_name, elements):
        """Error for each non-integer element in list fields."""
        import numpy as np

        cfg = _make_config(**{field_name: elements})
        v = ConfigValidator(cfg)
        errors = v.validate()

        type_errors = [e for e in errors if 'non-integer element' in e and field_name in e]

        # Count expected non-integer elements (booleans are excluded from int check
        # by the isinstance check in the validator since bool is subclass of int,
        # but the validator uses isinstance(elem, (int, np.integer)) which includes bool)
        expected_non_ints = [
            elem for elem in elements
            if not isinstance(elem, (int, np.integer))
        ]

        assert len(type_errors) == len(expected_non_ints), (
            f"Field '{field_name}', elements={elements}: "
            f"expected {len(expected_non_ints)} type errors, got {len(type_errors)}. "
            f"Errors: {type_errors}"
        )

        # Each non-integer element should be mentioned in an error
        for elem in expected_non_ints:
            matching = [e for e in type_errors if str(elem) in e]
            assert len(matching) >= 1, (
                f"Non-integer element {elem!r} should appear in an error: {type_errors}"
            )


# Feature: config-validation, Property 7: Path existence validation
class TestPropertyPathExistenceValidation:
    """Property 7: For any configuration where path fields are non-empty,
    the validator SHALL produce an error if the path does not exist on disk.
    When a path field was already flagged as missing/empty, no path-existence
    error SHALL be produced for it.

    **Validates: Requirements 2.1, 2.2, 2.3, 2.4**
    """

    @given(
        paramdb_exists=st.booleans(),
        control_exists=st.booleans(),
        cbh_enabled=st.booleans(),
        cbh_exists=st.booleans(),
    )
    @settings(max_examples=100, suppress_health_check=[HealthCheck.function_scoped_fixture])
    def test_path_existence_validation(self, tmp_path, paramdb_exists,
                                       control_exists, cbh_enabled, cbh_exists):
        """Error iff path doesn't exist on disk and field was not flagged as missing."""
        import tempfile
        # Use a unique temp directory per hypothesis example to avoid state leakage
        run_dir = tempfile.mkdtemp(dir=tmp_path)
        from pathlib import Path
        run_path = Path(run_dir)

        paramdb_dir = str(run_path / 'paramdb')
        control_file = str(run_path / 'control.default')
        cbh_dir = str(run_path / 'cbh')

        if paramdb_exists:
            (run_path / 'paramdb').mkdir(exist_ok=True)
        if control_exists:
            (run_path / 'control.default').touch()
        if cbh_exists:
            (run_path / 'cbh').mkdir(exist_ok=True)

        overrides = {
            'paramdb_dir': paramdb_dir,
            'control_filename': control_file,
            'output_cbh': cbh_enabled,
            'cbh_dir': cbh_dir if cbh_enabled else '',
        }

        # If cbh is enabled, provide a non-empty cbh_var_map to avoid conditional errors
        if cbh_enabled:
            overrides['cbh_var_map'] = {'tmax': 'tmax.cbh'}

        cfg = _make_config(**overrides)
        v = ConfigValidator(cfg)
        errors = v.validate()

        path_errors = [e for e in errors if 'does not exist' in e]

        # Check paramdb_dir
        paramdb_path_errors = [e for e in path_errors if 'paramdb_dir' in e]
        if paramdb_exists:
            assert len(paramdb_path_errors) == 0, (
                f"paramdb_dir exists but got error: {paramdb_path_errors}"
            )
        else:
            assert len(paramdb_path_errors) == 1, (
                f"paramdb_dir missing but no error: {path_errors}"
            )

        # Check control_filename
        control_path_errors = [e for e in path_errors if 'control_filename' in e]
        if control_exists:
            assert len(control_path_errors) == 0, (
                f"control_filename exists but got error: {control_path_errors}"
            )
        else:
            assert len(control_path_errors) == 1, (
                f"control_filename missing but no error: {path_errors}"
            )

        # Check cbh_dir (only when enabled)
        cbh_path_errors = [e for e in path_errors if 'cbh_dir' in e]
        if cbh_enabled:
            if cbh_exists:
                assert len(cbh_path_errors) == 0, (
                    f"cbh_dir exists but got error: {cbh_path_errors}"
                )
            else:
                assert len(cbh_path_errors) == 1, (
                    f"cbh_dir missing but no error: {path_errors}"
                )
        else:
            assert len(cbh_path_errors) == 0, (
                f"cbh disabled but got cbh_dir path error: {cbh_path_errors}"
            )


# ---------------------------------------------------------------------------
# Unit tests for integration and edge cases (Task 6.1)
# ---------------------------------------------------------------------------
class TestConfigValidatorUnit:
    """Unit tests for ConfigValidator edge cases and integration.

    **Validates: Requirements 10.1, 3.1, 3.2**
    """

    def test_fully_valid_config_no_errors(self, tmp_path):
        """A fully valid config with existing paths returns an empty error list."""
        # Create real paths
        paramdb = tmp_path / 'paramdb'
        paramdb.mkdir()
        control = tmp_path / 'control.default'
        control.touch()

        cfg = _make_config(
            output_dir=str(tmp_path / 'output'),
            param_filename='myparam.param',
            paramdb_dir=str(paramdb),
            control_filename=str(control),
            start_date='2000-01-01',
            end_date='2010-12-31',
        )
        v = ConfigValidator(cfg)
        errors = v.validate()
        assert errors == [], f"Expected no errors for valid config, got: {errors}"

    def test_datetime_date_objects_handled_correctly(self, tmp_path):
        """YAML type coercion: datetime.date objects are handled without errors."""
        paramdb = tmp_path / 'paramdb'
        paramdb.mkdir()
        control = tmp_path / 'control.default'
        control.touch()

        cfg = _make_config(
            paramdb_dir=str(paramdb),
            control_filename=str(control),
            start_date=datetime.date(2000, 1, 1),
            end_date=datetime.date(2010, 12, 31),
        )
        v = ConfigValidator(cfg)
        errors = v.validate()
        date_errors = [e for e in errors if 'date' in e.lower()]
        assert date_errors == [], f"datetime.date objects should not produce date errors: {date_errors}"

    def test_multiple_errors_across_categories_collected(self):
        """Multiple errors across different categories are all collected in a single pass."""
        cfg = _make_config(
            output_dir='',                          # core field error
            start_date='not-a-date',                # date format error
            output_cbh=True,                        # conditional toggle
            cbh_dir='',                             # conditional field error
            cbh_var_map={},                         # conditional field error
            outlets=[1, 'bad', 2],                  # list type error
        )
        v = ConfigValidator(cfg)
        errors = v.validate()

        # Should have errors from multiple categories
        has_core_error = any('missing or empty' in e and 'output_dir' in e for e in errors)
        has_date_error = any('invalid date format' in e for e in errors)
        has_conditional_error = any('required when' in e or 'non-empty dictionary when' in e for e in errors)
        has_list_error = any('non-integer element' in e for e in errors)

        assert has_core_error, f"Expected core field error in: {errors}"
        assert has_date_error, f"Expected date format error in: {errors}"
        assert has_conditional_error, f"Expected conditional field error in: {errors}"
        assert has_list_error, f"Expected list type error in: {errors}"

    def test_error_count_matches_expected_failures(self, tmp_path):
        """Error count matches the expected number of individual failures."""
        paramdb = tmp_path / 'paramdb'
        paramdb.mkdir()
        control = tmp_path / 'control.default'
        control.touch()

        # Create config with exactly 3 known errors:
        # 1. output_dir empty → "Required field 'output_dir' is missing or empty"
        # 2. start_date invalid → "Field 'start_date' has invalid date format..."
        # 3. outlets has non-int → "Field 'outlets' contains non-integer element: bad"
        cfg = _make_config(
            output_dir='',
            paramdb_dir=str(paramdb),
            control_filename=str(control),
            start_date='invalid',
            end_date='2010-12-31',
            outlets=[1, 'bad'],
        )
        v = ConfigValidator(cfg)
        errors = v.validate()

        # Count specific expected errors
        core_errors = [e for e in errors if "Required field 'output_dir' is missing or empty" in e]
        date_errors = [e for e in errors if "invalid date format" in e and "start_date" in e]
        list_errors = [e for e in errors if "non-integer element" in e and "outlets" in e]

        assert len(core_errors) == 1, f"Expected 1 core error, got: {core_errors}"
        assert len(date_errors) == 1, f"Expected 1 date error, got: {date_errors}"
        assert len(list_errors) == 1, f"Expected 1 list error, got: {list_errors}"

        # Total should be exactly 3
        expected_count = 3
        assert len(errors) == expected_count, (
            f"Expected exactly {expected_count} errors, got {len(errors)}: {errors}"
        )
