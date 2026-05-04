# Implementation Plan: error-handling-cleanup

## Overview

Replace bare `exit()` calls in `Bandit/bandit.py` with a custom `BanditError` exception, wrap the Dask `Client` in a context manager, guard `os.chdir()` with `try/finally`, and catch `BanditError` in `main()` to call `sys.exit()`. All new tests go in `tests/test_error_handling.py`.

## Tasks

- [x] 1. Create the BanditError exception class
  - [x] 1.1 Create `Bandit/exceptions.py` with the `BanditError` class
    - Subclass `Exception` with `__init__(self, message: str, exit_code: int = 1)`
    - Store `exit_code` as an instance attribute
    - Default `exit_code` to `1` when not provided
    - `str(error)` returns the message via standard `Exception.__str__`
    - _Requirements: 1.1, 1.2, 1.3, 1.4, 1.5_

  - [x] 1.2 Write property test for BanditError construction round-trip
    - **Property 1: BanditError construction round-trip**
    - For any integer `exit_code` and any non-empty string `message`, constructing `BanditError(message, exit_code)` produces an object where `error.exit_code == exit_code` and `str(error) == message`
    - Use Hypothesis to generate random `(exit_code: int, message: str)` pairs, minimum 100 examples
    - **Validates: Requirements 1.2, 1.3, 1.4**

  - [x] 1.3 Write unit tests for BanditError
    - Test that `BanditError` is a subclass of `Exception`
    - Test that default `exit_code` is `1` when not provided
    - Test that `str(error)` returns the message
    - _Requirements: 1.1, 1.4, 1.5_

- [x] 2. Replace bare exit() calls with BanditError raises in extract()
  - [x] 2.1 Add `from Bandit.exceptions import BanditError` import to `bandit.py`
    - Add the import alongside existing Bandit imports
    - _Requirements: 2.1, 2.2, 2.3, 2.4, 2.5, 2.6_

  - [x] 2.2 Replace the invalid job directory `exit(-1)` with `raise BanditError`
    - Replace `exit(-1)` with `raise BanditError(f'Invalid jobs directory: {str(job_dir)}', exit_code=-1)`
    - Preserve existing `con.print()` and `bandit_log.error()` calls before the raise
    - _Requirements: 2.1, 6.1, 6.2, 6.3, 7.1_

  - [x] 2.3 Replace the configuration validation `exit(2)` with `raise BanditError`
    - Replace `exit(2)` with `raise BanditError(f'Configuration has {len(errors)} error(s). Fix the above issues and retry.', exit_code=2)`
    - Preserve existing error logging and console output
    - _Requirements: 2.2, 6.1, 6.2, 6.3, 7.2_

  - [x] 2.4 Replace the missing stream segments `exit(200)` with `raise BanditError`
    - Replace `exit(200)` with `raise BanditError('None of the requested stream segments exist in the NHM', exit_code=200)`
    - Preserve existing `con.print()` and `bandit_log.error()` calls
    - _Requirements: 2.3, 6.1, 6.2, 6.3, 7.3_

  - [x] 2.5 Replace the no-HRUs `exit(2)` with `raise BanditError`
    - Replace `exit(2)` with `raise BanditError('No HRUs associated with any of the segments', exit_code=2)`
    - Preserve existing error logging and console output
    - _Requirements: 2.4, 6.1, 6.2, 6.3, 7.4_

  - [x] 2.6 Replace the two dynamic parameters `exit(2)` calls with `raise BanditError`
    - Replace `exit(2)` for missing `dyn_params_dir` config with `raise BanditError('Control file has dynamic parameters but dyn_params_dir is not specified in the config file', exit_code=2)`
    - Replace `exit(2)` for non-existent `dyn_params_dir` path with `raise BanditError(f'dyn_params_dir: {config.dyn_params_dir}, does not exist.', exit_code=2)`
    - Add `con.print()` calls before each raise to match the pattern of other error sites
    - _Requirements: 2.5, 2.6, 6.1, 6.2, 6.3, 7.5, 7.6_

  - [x] 2.7 Write a static inspection unit test verifying no bare exit() calls remain
    - Read `Bandit/bandit.py` source and assert zero calls to `exit()` (excluding `sys.exit`)
    - _Requirements: 2.7_

- [x] 3. Wrap Dask Client in context manager and guard os.chdir() with try/finally
  - [x] 3.1 Wrap the Dask `Client()` call with `with Client() as client:`
    - Replace `client = Client()` with `with Client() as client:` and indent the block that uses `client`
    - Remove the explicit `client.close()` call at the end of `extract()`
    - _Requirements: 4.1, 4.2, 4.3, 4.4_

  - [x] 3.2 Add `try/finally` guard for `os.chdir()` restoration
    - Save `stdir = os.getcwd()` and track `chdir_needed` flag
    - Wrap the function body after the chdir decision in `try/finally`
    - In the `finally` block, restore `os.chdir(stdir)` only if `chdir_needed` is `True`
    - Remove the explicit `os.chdir(stdir)` at the end of `extract()`
    - _Requirements: 5.1, 5.2, 5.3, 5.4_

- [x] 4. Checkpoint
  - Ensure all changes compile and the existing test suite passes. Ask the user if questions arise.

- [x] 5. Update main() to catch BanditError and call sys.exit()
  - [x] 5.1 Add `import sys` to `bandit.py` imports
    - Add `import sys` at the top of the file with the other standard library imports
    - _Requirements: 3.1_

  - [x] 5.2 Wrap `app()` call in `main()` with try/except for BanditError
    - Catch `BanditError`, log the message at ERROR level via `bandit_log.error()`, print to console via `con.print()`, and call `sys.exit(err.exit_code)`
    - Allow normal completion (no BanditError) to exit with code 0
    - _Requirements: 3.1, 3.2, 3.3_

  - [x] 5.3 Write property test for CLI entry point BanditError translation
    - **Property 2: CLI entry point translates BanditError to sys.exit**
    - For any `BanditError` with an arbitrary integer `exit_code` and string `message`, when `app()` raises that error, `main()` calls `sys.exit()` with exactly that `exit_code`
    - Mock `app()` to raise generated `BanditError` instances, verify `sys.exit()` receives the correct exit code
    - Use Hypothesis, minimum 100 examples
    - **Validates: Requirements 3.1, 3.3, 7.1, 7.2, 7.3, 7.4, 7.5, 7.6**

  - [x] 5.4 Write unit tests for main() behavior
    - Test that `main()` does not call `sys.exit()` when no error occurs (mock `app()` to succeed)
    - Test that `main()` logs the error message at ERROR level before exiting
    - _Requirements: 3.1, 3.2, 3.3_

- [x] 6. Final checkpoint
  - Ensure all tests pass, ask the user if questions arise.

## Notes

- Tasks marked with `*` are optional and can be skipped for faster MVP
- Each task references specific requirements for traceability
- Checkpoints ensure incremental validation
- Property tests validate universal correctness properties from the design document
- Unit tests validate specific examples and edge cases
- All new tests go in `tests/test_error_handling.py` using pytest and Hypothesis (already project dependencies)
