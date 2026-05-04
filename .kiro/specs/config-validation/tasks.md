# Implementation Plan: Config Validation

## Overview

Implement a `ConfigValidator` class in `Bandit/config_validator.py` that validates the Bandit YAML configuration before heavy processing begins. The validator accumulates all errors and returns them as a list of strings. Integration into `Bandit/bandit.py` is a thin block after `config = bc.Cfg(config_file)` that logs errors and exits if any are found. Property-based tests with Hypothesis verify each correctness property.

## Tasks

- [x] 1. Create `ConfigValidator` class with core field validation
  - [x] 1.1 Create `Bandit/config_validator.py` with `ConfigValidator` class skeleton
    - Define `__init__(self, config: Cfg)` storing the config and an empty `_errors: List[str]`
    - Define `validate(self) -> List[str]` that calls all `_validate_*` methods in order and returns `_errors`
    - Define `_validate_core_fields(self) -> Set[str]` that checks `output_dir`, `param_filename`, `paramdb_dir`, `control_filename`, `start_date`, `end_date` are present and non-empty
    - Return the set of field names that failed so downstream path checks can skip them
    - Error format: `"Required field '{name}' is missing or empty"`
    - _Requirements: 1.1, 1.2, 1.3, 10.1_

  - [x] 1.2 Implement `_validate_dates` method
    - Check that `start_date` and `end_date` are parseable as `YYYY-MM-DD` dates
    - Handle both `str` and `datetime.date` inputs (YAML type coercion)
    - Check that `start_date < end_date`; produce error if `start_date >= end_date`
    - Error formats: `"Field '{name}' has invalid date format '{value}'; expected YYYY-MM-DD"` and `"'start_date' ({start}) must precede 'end_date' ({end})"`
    - _Requirements: 3.1, 3.2, 3.3, 3.4_

  - [x] 1.3 Implement `_validate_paths` method
    - Accept `skip_fields: Set[str]` parameter to skip fields already flagged as missing
    - Check `paramdb_dir` exists as a directory on disk
    - Check `control_filename` exists as a file on disk
    - When `output_cbh` is true and `cbh_dir` not in skip_fields, check `cbh_dir` exists on disk
    - Error format: `"Path for '{name}' does not exist: {path}"`
    - _Requirements: 2.1, 2.2, 2.3, 2.4_

- [x] 2. Implement conditional field validation methods
  - [x] 2.1 Implement `_validate_cbh_fields` method
    - When `output_cbh` is true, check `cbh_dir` is present and non-empty
    - When `output_cbh` is true, check `cbh_var_map` is a non-empty dictionary
    - Error format: `"Field '{name}' is required when 'output_cbh' is enabled"` / `"Field '{name}' must be a non-empty dictionary when 'output_cbh' is enabled"`
    - _Requirements: 4.1, 4.2, 4.3, 4.4_

  - [x] 2.2 Implement `_validate_model_output_fields` method
    - When `include_model_output` is true, check `output_vars_dir` is a non-empty string
    - When `include_model_output` is true, check `output_vars` is a non-empty list
    - Error format: `"Field '{name}' is required when 'include_model_output' is enabled"`
    - _Requirements: 5.1, 5.2, 5.3, 5.4_

  - [x] 2.3 Implement `_validate_streamflow_fields` method
    - When `output_streamflow` is true, check `streamflow_filename` is a non-empty string
    - Error format: `"Field '{name}' is required when 'output_streamflow' is enabled"`
    - _Requirements: 6.1, 6.2_

  - [x] 2.4 Implement `_validate_gis_fields` method
    - When `output_shapefiles` is true, check `gis` is a non-empty dictionary
    - Check `gis` contains keys `src_filename`, `dst_extension`, `layers`
    - Check `gis['src_filename']` is a non-empty string
    - Check `gis['layers']` is a non-empty dictionary
    - Error format: `"GIS config is missing required key '{key}'"` and related messages
    - _Requirements: 7.1, 7.2, 7.3, 7.4, 7.5_

  - [x] 2.5 Implement `_validate_list_types` method
    - Check `outlets`, `cutoffs`, `hru_noroute` contain only integers when non-empty
    - Error format: `"Field '{name}' contains non-integer element: {element}"`
    - _Requirements: 8.1, 8.2, 8.3, 8.4_

- [x] 3. Integrate validator into `extract` function
  - [x] 3.1 Add validation call in `Bandit/bandit.py`
    - Add `from Bandit.config_validator import ConfigValidator` import
    - After `config = bc.Cfg(config_file)` and before any `ParamDb`/`ControlFile`/network operations, add validation block
    - Log each error at ERROR level via `bandit_log.error()`
    - Print each error to console via `con.print()` with red formatting
    - Print summary count and call `exit(2)` if errors exist
    - _Requirements: 9.1, 9.2, 10.2, 10.3, 10.4_

- [x] 4. Checkpoint - Ensure all tests pass
  - Ensure all tests pass, ask the user if questions arise.

- [x] 5. Property-based tests for `ConfigValidator`
  - [ ]* 5.1 Write property test for core field presence detection
    - **Property 1: Core field presence detection**
    - Generate random subsets of core fields to omit or set empty; verify each missing field produces an error containing the field name
    - Use `@settings(max_examples=100)`
    - **Validates: Requirements 1.1, 1.2, 1.3**

  - [ ]* 5.2 Write property test for date format validation
    - **Property 2: Date format validation**
    - Generate random strings (valid YYYY-MM-DD, invalid formats, edge dates, `datetime.date` objects); verify error iff not parseable as YYYY-MM-DD
    - Use `@settings(max_examples=100)`
    - **Validates: Requirements 3.1, 3.2, 3.3**

  - [ ]* 5.3 Write property test for date ordering
    - **Property 3: Date ordering**
    - Generate random pairs of valid dates; verify error iff `start_date >= end_date`
    - Use `@settings(max_examples=100)`
    - **Validates: Requirements 3.4**

  - [ ]* 5.4 Write property test for conditional field validation
    - **Property 4: Conditional field validation**
    - Generate random toggle states (`output_cbh`, `include_model_output`, `output_streamflow`) and field values; verify error iff toggle is true and field is missing/empty
    - Use `@settings(max_examples=100)`
    - **Validates: Requirements 4.1, 4.2, 4.3, 4.4, 5.1, 5.2, 5.3, 5.4, 6.1, 6.2**

  - [ ]* 5.5 Write property test for GIS structure validation
    - **Property 5: GIS structure validation**
    - Generate random `gis` dicts with subsets of required keys; verify errors match missing/empty keys when `output_shapefiles` is true
    - Use `@settings(max_examples=100)`
    - **Validates: Requirements 7.1, 7.2, 7.3, 7.4, 7.5**

  - [ ]* 5.6 Write property test for list element type validation
    - **Property 6: List element type validation**
    - Generate lists with random mixes of ints and non-ints for `outlets`, `cutoffs`, `hru_noroute`; verify error for each non-integer element
    - Use `@settings(max_examples=100)`
    - **Validates: Requirements 8.1, 8.2, 8.3, 8.4**

  - [ ]* 5.7 Write property test for path existence validation
    - **Property 7: Path existence validation**
    - Use `tmp_path` fixture to create/omit paths; verify error iff path doesn't exist on disk and field was not already flagged as missing
    - Use `@settings(max_examples=100)`
    - **Validates: Requirements 2.1, 2.2, 2.3, 2.4**

- [x] 6. Unit tests for integration and edge cases
  - [ ]* 6.1 Write unit tests in `tests/test_config_validator.py`
    - Test that a fully valid config returns an empty error list
    - Test YAML type coercion edge case (`datetime.date` objects handled correctly)
    - Test that multiple errors across different categories are all collected in a single pass
    - Test error count matches expected number of failures
    - _Requirements: 10.1, 3.1, 3.2_

- [x] 7. Final checkpoint - Ensure all tests pass
  - Ensure all tests pass, ask the user if questions arise.

## Notes

- Tasks marked with `*` are optional and can be skipped for faster MVP
- Each task references specific requirements for traceability
- Checkpoints ensure incremental validation
- Property tests validate universal correctness properties from the design document
- Unit tests validate specific examples and edge cases
- The design uses Python, so all code examples and implementations use Python
- Hypothesis property tests use `@settings(max_examples=100)` as specified in the design
- All tests go in `tests/test_config_validator.py`
