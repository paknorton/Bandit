# Requirements Document

## Introduction

The `extract` function in `Bandit/bandit.py` currently uses a mix of four output methods: Python's `logging` module (`bandit_log`), Rich console (`con.print()`), bare `print()` statements, and direct `sys.stdout.write()`/`sys.stdout.flush()` calls. This inconsistency makes output behavior unpredictable across environments (e.g., piped output, log aggregation) and complicates maintenance.

This feature standardizes all output in `bandit.py` on two channels:
- **Rich console (`con`)** for user-facing status messages with markup
- **Python logger (`bandit_log`)** for persistent log records

Bare `print()` and `sys.stdout.write/flush` calls are eliminated entirely.

## Glossary

- **Bandit**: The model extraction application defined in `Bandit/bandit.py`
- **Extract_Function**: The main `extract()` function that performs NHM parameter database subsetting
- **Rich_Console**: The Rich library console instance (`con`) obtained from `get_console_instance()`
- **Logger**: The Python logging instance (`bandit_log`) configured for the Bandit application
- **User_Facing_Message**: A message intended to be seen by the operator running Bandit interactively
- **Log_Record**: A message intended to be persisted in the log file for auditing or debugging

## Requirements

### Requirement 1: Replace bare print error messages with Rich console and logger

**User Story:** As a developer, I want error messages to use the Rich console and logger consistently, so that errors are both visible to the user with proper formatting and recorded in the log file.

#### Acceptance Criteria

1. WHEN an error condition is detected, THE Extract_Function SHALL output the error message via Rich_Console using `[red]ERROR[/]` markup
2. WHEN an error condition is detected, THE Extract_Function SHALL record the error message via Logger at the ERROR level
3. THE Extract_Function SHALL NOT use bare `print()` calls to output error messages

### Requirement 2: Replace bare print informational messages with Rich console

**User Story:** As a developer, I want informational and verbose messages to use the Rich console, so that all user-facing output has consistent formatting and respects Rich console configuration.

#### Acceptance Criteria

1. WHEN verbose mode is active and a status update is generated, THE Extract_Function SHALL output the message via Rich_Console with appropriate styling
2. THE Extract_Function SHALL NOT use bare `print()` calls to output informational or status messages
3. WHEN a visual separator is needed, THE Extract_Function SHALL use Rich_Console with a Rule or styled separator instead of bare `print()` with repeated characters

### Requirement 3: Eliminate sys.stdout.write and sys.stdout.flush calls

**User Story:** As a developer, I want all direct stdout manipulation removed, so that output routing is fully controlled by the Rich console and logger abstractions.

#### Acceptance Criteria

1. THE Extract_Function SHALL NOT call `sys.stdout.write()` for any output purpose
2. THE Extract_Function SHALL NOT call `sys.stdout.flush()` for any output purpose
3. WHERE a blank line or spacing is needed between output sections, THE Extract_Function SHALL use `con.print()` with an empty string or appropriate Rich markup

### Requirement 4: Dual-channel output for significant events

**User Story:** As a developer, I want significant operational events (errors, warnings, key milestones) to appear in both the console and the log file, so that interactive users see immediate feedback and the log file retains a complete record.

#### Acceptance Criteria

1. WHEN an error occurs, THE Extract_Function SHALL output the message to both Rich_Console and Logger at ERROR level
2. WHEN a warning condition is detected, THE Extract_Function SHALL output the message to both Rich_Console with `[gold3]WARNING[/]` markup and Logger at WARNING level
3. WHEN a key processing milestone is reached, THE Extract_Function SHALL output the message to Rich_Console with `[green4]INFO[/]` markup and Logger at INFO level

### Requirement 5: Preserve existing logger and Rich console patterns

**User Story:** As a developer, I want the existing correct uses of `bandit_log` and `con.print()` to remain unchanged, so that the refactoring only addresses the inconsistent output methods without disrupting working code.

#### Acceptance Criteria

1. THE Extract_Function SHALL retain all existing `bandit_log` calls that currently use appropriate log levels
2. THE Extract_Function SHALL retain all existing `con.print()` calls that currently use appropriate Rich markup
3. THE Extract_Function SHALL NOT alter the log format, log level configuration, or Rich console instantiation

### Requirement 6: Remove sys import dependency for output

**User Story:** As a developer, I want the `sys` module import to be removed if it is no longer needed after eliminating stdout calls, so that the module's imports accurately reflect its dependencies.

#### Acceptance Criteria

1. IF `sys` is no longer referenced after removing `sys.stdout.write()` and `sys.stdout.flush()`, THEN THE Extract_Function module SHALL remove the `sys` import statement
2. IF `sys` is still referenced for other purposes (e.g., `sys.exit()`), THEN THE Extract_Function module SHALL retain the `sys` import statement
