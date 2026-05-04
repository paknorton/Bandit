# Requirements Document

## Introduction

The `extract` function in `Bandit/bandit.py` reads numerous configuration attributes (e.g., `config.output_dir`, `config.paramdb_dir`, `config.gis['src_filename']`) without any upfront validation. When the configuration is malformed or missing required fields, errors surface deep in the extraction process with cryptic messages that are difficult to diagnose. This feature adds early validation of required configuration fields at the start of the `extract` function, providing clear, actionable error messages before any heavy processing begins.

## Glossary

- **Config_Validator**: The validation component that checks configuration fields for presence, correct types, and structural completeness before extraction begins.
- **Cfg**: The existing configuration class (`Bandit.bandit_cfg.Cfg`) that loads YAML configuration files and provides attribute-style access to configuration values.
- **Extract_Function**: The main `extract` function in `Bandit/bandit.py` that performs NHM model subset extraction.
- **Required_Field**: A configuration field that must be present and non-empty for the extraction to proceed.
- **Conditional_Field**: A configuration field that is required only when a related boolean toggle is enabled (e.g., `cbh_var_map` is required only when `output_cbh` is true).
- **GIS_Config**: The nested dictionary configuration under the `gis` key, containing `src_filename`, `dst_extension`, and `layers` sub-keys.

## Requirements

### Requirement 1: Validate Required Core Fields

**User Story:** As a modeler, I want the extraction process to validate that all required core configuration fields are present and non-empty before processing begins, so that I receive clear error messages instead of cryptic failures deep in the workflow.

#### Acceptance Criteria

1. WHEN the `extract` function is called, THE Config_Validator SHALL check that each of the following fields is present and non-empty in the loaded configuration: `output_dir`, `param_filename`, `paramdb_dir`, `control_filename`, `start_date`, `end_date`.
2. IF a required core field is missing or empty, THEN THE Config_Validator SHALL raise an error that names the missing field and states that it is required.
3. IF multiple required core fields are missing or empty, THEN THE Config_Validator SHALL report all missing fields in a single error message rather than failing on the first missing field.

### Requirement 2: Validate Required Path Fields Exist on Disk

**User Story:** As a modeler, I want the validator to check that required directory and file paths actually exist on disk, so that I learn about missing paths before the extraction attempts to read from them.

#### Acceptance Criteria

1. WHEN the configuration passes field-presence validation, THE Config_Validator SHALL verify that the path specified by `paramdb_dir` exists as a directory on disk.
2. WHEN the configuration passes field-presence validation, THE Config_Validator SHALL verify that the path specified by `control_filename` exists as a file on disk.
3. WHILE `output_cbh` is true, THE Config_Validator SHALL verify that the path specified by `cbh_dir` exists on disk as a file or directory.
4. IF a required path does not exist on disk, THEN THE Config_Validator SHALL raise an error that names the field and includes the non-existent path value.

### Requirement 3: Validate Date Fields

**User Story:** As a modeler, I want the validator to check that date fields contain parseable date strings and that the start date precedes the end date, so that I avoid confusing date-related errors during processing.

#### Acceptance Criteria

1. WHEN the configuration passes field-presence validation, THE Config_Validator SHALL verify that `start_date` is a string parseable as a date in `YYYY-MM-DD` format.
2. WHEN the configuration passes field-presence validation, THE Config_Validator SHALL verify that `end_date` is a string parseable as a date in `YYYY-MM-DD` format.
3. IF `start_date` or `end_date` is not parseable as a `YYYY-MM-DD` date, THEN THE Config_Validator SHALL raise an error that names the field and includes the invalid value.
4. IF `start_date` is equal to or later than `end_date`, THEN THE Config_Validator SHALL raise an error stating that `start_date` must precede `end_date` and include both date values.

### Requirement 4: Validate Conditional CBH Fields

**User Story:** As a modeler, I want the validator to check CBH-related fields when CBH output is enabled, so that missing CBH configuration is caught before the extraction reaches the CBH processing stage.

#### Acceptance Criteria

1. WHILE `output_cbh` is true, THE Config_Validator SHALL verify that `cbh_dir` is present and non-empty.
2. WHILE `output_cbh` is true, THE Config_Validator SHALL verify that `cbh_var_map` is a non-empty dictionary.
3. IF `output_cbh` is true and `cbh_dir` is missing or empty, THEN THE Config_Validator SHALL raise an error stating that `cbh_dir` is required when `output_cbh` is enabled.
4. IF `output_cbh` is true and `cbh_var_map` is missing or empty, THEN THE Config_Validator SHALL raise an error stating that `cbh_var_map` is required when `output_cbh` is enabled.

### Requirement 5: Validate Conditional Model Output Fields

**User Story:** As a modeler, I want the validator to check model-output-related fields when model output is enabled, so that missing output configuration is caught early.

#### Acceptance Criteria

1. WHILE `include_model_output` is true, THE Config_Validator SHALL verify that `output_vars_dir` is a non-empty string.
2. WHILE `include_model_output` is true, THE Config_Validator SHALL verify that `output_vars` is a non-empty list.
3. IF `include_model_output` is true and `output_vars_dir` is missing or empty, THEN THE Config_Validator SHALL raise an error stating that `output_vars_dir` is required when `include_model_output` is enabled.
4. IF `include_model_output` is true and `output_vars` is missing or empty, THEN THE Config_Validator SHALL raise an error stating that `output_vars` is required when `include_model_output` is enabled.

### Requirement 6: Validate Conditional Streamflow Fields

**User Story:** As a modeler, I want the validator to check streamflow-related fields when streamflow output is enabled, so that missing streamflow configuration is caught early.

#### Acceptance Criteria

1. WHILE `output_streamflow` is true, THE Config_Validator SHALL verify that `streamflow_filename` is a non-empty string.
2. IF `output_streamflow` is true and `streamflow_filename` is missing or empty, THEN THE Config_Validator SHALL raise an error stating that `streamflow_filename` is required when `output_streamflow` is enabled.

### Requirement 7: Validate Conditional GIS Fields

**User Story:** As a modeler, I want the validator to check GIS-related fields when shapefile output is enabled, so that missing or malformed GIS configuration is caught before the shapefile writing stage.

#### Acceptance Criteria

1. WHILE `output_shapefiles` is true, THE Config_Validator SHALL verify that `gis` is a non-empty dictionary.
2. WHILE `output_shapefiles` is true, THE Config_Validator SHALL verify that `gis` contains the keys `src_filename`, `dst_extension`, and `layers`.
3. WHILE `output_shapefiles` is true, THE Config_Validator SHALL verify that `gis['src_filename']` is a non-empty string.
4. WHILE `output_shapefiles` is true, THE Config_Validator SHALL verify that `gis['layers']` is a non-empty dictionary.
5. IF `output_shapefiles` is true and any required GIS sub-key is missing or empty, THEN THE Config_Validator SHALL raise an error that names the missing sub-key within the `gis` configuration.

### Requirement 8: Validate List Fields Have Correct Element Types

**User Story:** As a modeler, I want the validator to check that list-typed configuration fields contain elements of the expected type, so that type errors do not surface as cryptic failures during array operations.

#### Acceptance Criteria

1. WHEN `outlets` is non-empty, THE Config_Validator SHALL verify that each element in `outlets` is an integer.
2. WHEN `cutoffs` is non-empty, THE Config_Validator SHALL verify that each element in `cutoffs` is an integer.
3. WHEN `hru_noroute` is non-empty, THE Config_Validator SHALL verify that each element in `hru_noroute` is an integer.
4. IF a list field contains an element of an unexpected type, THEN THE Config_Validator SHALL raise an error that names the field and identifies the invalid element.

### Requirement 9: Run Validation Before Heavy Processing

**User Story:** As a modeler, I want all configuration validation to complete before any parameter database loading, network construction, or file I/O begins, so that I get fast feedback on configuration problems.

#### Acceptance Criteria

1. WHEN the `extract` function is called, THE Config_Validator SHALL execute all validation checks after the `Cfg` object is constructed and before any call to `ParamDb`, `ControlFile`, `Cbh`, network operations, or file writes.
2. IF any validation check fails, THEN THE Config_Validator SHALL prevent the extraction from proceeding and exit with a non-zero status code.

### Requirement 10: Aggregate Validation Errors

**User Story:** As a modeler, I want to see all configuration errors at once rather than fixing them one at a time, so that I can resolve all issues in a single pass.

#### Acceptance Criteria

1. THE Config_Validator SHALL collect all validation errors across all checks (Requirements 1 through 8) before reporting them.
2. WHEN one or more validation errors exist, THE Config_Validator SHALL log each error message to the bandit log at the ERROR level.
3. WHEN one or more validation errors exist, THE Config_Validator SHALL print a summary to the console listing all errors.
4. WHEN one or more validation errors exist, THE Config_Validator SHALL exit with a non-zero status code after reporting all errors.
