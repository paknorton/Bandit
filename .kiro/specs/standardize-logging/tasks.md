# Implementation Plan: Standardize Logging

## Overview

Replace five non-standard output calls in `Bandit/bandit.py` (bare `print()`, `sys.stdout.write()`, `sys.stdout.flush()`) with the existing Rich console (`con.print()`) and Python logger (`bandit_log`) channels. Add `from rich.rule import Rule` import and remove the unused `import sys`. All changes are in a single file with no new modules or abstractions.

## Tasks

- [x] 1. Update imports in `Bandit/bandit.py`
  - [x] 1.1 Add `from rich.rule import Rule` import near the other Rich-related imports
    - This is needed for the separator replacement in Change 4
    - _Requirements: 2.3_

  - [x] 1.2 Remove `import sys` from the imports
    - After all `sys.stdout` calls are removed, `sys` has no remaining references
    - _Requirements: 6.1_

- [x] 2. Replace non-standard output calls in `extract()`
  - [x] 2.1 Replace bare `print()` for invalid job directory error with dual-channel output
    - Change `print(f'ERROR: Invalid jobs directory: {str(job_dir)}')` to `con.print(f'[red]ERROR[/]: Invalid jobs directory: {str(job_dir)}')` and add `bandit_log.error(f'Invalid jobs directory: {str(job_dir)}')`
    - _Requirements: 1.1, 1.2, 1.3, 4.1_

  - [x] 2.2 Replace `sys.stdout.write`/`sys.stdout.flush` block with `con.print('')`
    - Replace the `if verbose: sys.stdout.write('\n')` and `sys.stdout.flush()` block (after parameter file write) with `if verbose: con.print('')`
    - _Requirements: 3.1, 3.2, 3.3_

  - [x] 2.3 Replace bare `print()` for dynamic parameter verbose message
    - Change `print(f'Writing dynamic parameter {cparam}')` to `con.print(f'[green4]INFO[/]: Writing dynamic parameter {cparam}')`
    - _Requirements: 2.1, 2.2_

  - [x] 2.4 Replace bare `print('-'*40)` separator with Rich Rule
    - Change `print('-'*40)` to `con.print(Rule())`
    - _Requirements: 2.2, 2.3_

- [x] 3. Checkpoint - Verify refactoring is correct
  - Ensure all tests pass, ask the user if questions arise.

- [x] 4. Write tests in `tests/test_standardize_logging.py`
  - [x] 4.1 Write AST-based tests to verify prohibited patterns are absent
    - Test that `extract` function contains no bare `print()` calls
    - Test that `bandit.py` contains no `sys.stdout.write()` calls
    - Test that `bandit.py` contains no `sys.stdout.flush()` calls
    - Test that `bandit.py` does not import `sys`
    - _Requirements: 1.3, 2.2, 3.1, 3.2, 6.1_

  - [ ]* 4.2 Write example-based unit tests for the replaced call sites
    - Test invalid job_dir triggers `con.print` with `[red]ERROR[/]` and `bandit_log.error`
    - Test verbose dynamic parameter message uses `con.print` with `[green4]INFO[/]`
    - Test shapefile separator uses `con.print(Rule())`
    - Test verbose blank line uses `con.print('')`
    - _Requirements: 1.1, 1.2, 2.1, 2.3, 3.3, 4.1, 4.3_

- [x] 5. Final checkpoint - Ensure all tests pass
  - Ensure all tests pass, ask the user if questions arise.

## Notes

- Tasks marked with `*` are optional and can be skipped for faster MVP
- Each task references specific requirements for traceability
- Checkpoints ensure incremental validation
- No property-based tests — this is a mechanical refactoring with no input-dependent logic to fuzz
- All five changes are in `Bandit/bandit.py`; tests go in `tests/test_standardize_logging.py`
