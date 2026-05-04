# Requirements Document

## Introduction

The `extract()` function in `Bandit/bandit.py` has three related resource-management and error-handling problems that make the code fragile, untestable, and unsuitable for use as a library:

1. **Bare `exit()` calls** — The function calls `exit(-1)`, `exit(200)`, and `exit(2)` in several error paths. These terminate the Python interpreter immediately, which prevents callers from catching errors, makes the function impossible to unit-test, and blocks any use of Bandit as an imported library.

2. **Unguarded Dask Client** — A `dask.distributed.Client` is created mid-function but is only closed at the very end (`client.close()`). Any early return, exception, or `exit()` call before that line leaks the client and its associated threads and network resources.

3. **Unguarded `os.chdir()`** — The function changes the working directory with `os.chdir(job_dir)` near the top and only restores it with `os.chdir(stdir)` at the very end. Any error between those two points leaves the process in the wrong directory for all subsequent operations.

This feature replaces the bare `exit()` calls with a custom `BanditError` exception, wraps the Dask Client in proper resource cleanup, and protects the working-directory change with a context manager or `try/finally` block. The CLI entry point (`main()`) catches `BanditError` and calls `sys.exit()` with the appropriate code.

## Glossary

- **Extract_Function**: The `extract()` function in `Bandit/bandit.py` that performs NHM model subset extraction.
- **BanditError**: A custom exception class that carries an integer exit code and a human-readable message, raised in place of bare `exit()` calls.
- **CLI_Entry_Point**: The `main()` function in `Bandit/bandit.py` that serves as the command-line entry point and delegates to `extract()` via the cyclopts `App`.
- **Dask_Client**: An instance of `dask.distributed.Client` used for parallel computation during extraction.
- **Working_Directory_Guard**: A mechanism (context manager or `try/finally`) that saves the current working directory before changing it and restores it when the guarded block exits, regardless of how it exits.

## Requirements

### Requirement 1: Define a Custom BanditError Exception

**User Story:** As a developer, I want a custom exception class that carries an exit code and message, so that error conditions can be signaled without killing the interpreter.

#### Acceptance Criteria

1. THE BanditError SHALL be a subclass of `Exception`.
2. THE BanditError SHALL accept an integer exit code and a string message at construction time.
3. THE BanditError SHALL expose the exit code via an `exit_code` attribute.
4. THE BanditError SHALL expose the message via the standard `str()` representation.
5. WHEN no exit code is provided, THE BanditError SHALL default the exit code to 1.

### Requirement 2: Replace All Bare exit() Calls with BanditError

**User Story:** As a developer, I want every bare `exit()` call in the `extract()` function replaced with a `raise BanditError(...)`, so that the function can be called from tests and library code without terminating the interpreter.

#### Acceptance Criteria

1. WHEN an invalid job directory is provided, THE Extract_Function SHALL raise a BanditError with exit code -1 and a message that includes the invalid directory path.
2. WHEN configuration validation fails, THE Extract_Function SHALL raise a BanditError with exit code 2 and a message that includes the number of configuration errors.
3. WHEN none of the requested stream segments exist in the NHM, THE Extract_Function SHALL raise a BanditError with exit code 200 and a message stating that no requested segments exist.
4. WHEN no HRUs are associated with any of the stream segments, THE Extract_Function SHALL raise a BanditError with exit code 2 and a message stating that no HRUs are associated with the segments.
5. WHEN the control file has dynamic parameters but `dyn_params_dir` is not specified, THE Extract_Function SHALL raise a BanditError with exit code 2 and a message stating that `dyn_params_dir` is required.
6. WHEN `dyn_params_dir` does not exist on disk, THE Extract_Function SHALL raise a BanditError with exit code 2 and a message that includes the non-existent path.
7. THE Extract_Function SHALL contain zero calls to the built-in `exit()` function after this change.

### Requirement 3: CLI Entry Point Catches BanditError and Calls sys.exit()

**User Story:** As a CLI user, I want the command-line entry point to translate BanditError exceptions into proper process exit codes, so that shell scripts and CI pipelines receive the correct exit status.

#### Acceptance Criteria

1. WHEN the Extract_Function raises a BanditError, THE CLI_Entry_Point SHALL catch the exception, print the error message to the console, and call `sys.exit()` with the exit code from the BanditError.
2. WHEN the Extract_Function completes without raising a BanditError, THE CLI_Entry_Point SHALL allow the process to exit normally with code 0.
3. THE CLI_Entry_Point SHALL log the BanditError message to the bandit log at the ERROR level before exiting.

### Requirement 4: Wrap the Dask Client in Resource Cleanup

**User Story:** As a developer, I want the Dask Client to be properly closed regardless of how the extraction exits, so that threads, network connections, and scheduler resources are never leaked.

#### Acceptance Criteria

1. WHEN the Dask_Client is created, THE Extract_Function SHALL ensure the Dask_Client is closed when the guarded block exits normally.
2. IF an exception is raised after the Dask_Client is created, THEN THE Extract_Function SHALL close the Dask_Client before the exception propagates.
3. IF a BanditError is raised after the Dask_Client is created, THEN THE Extract_Function SHALL close the Dask_Client before the BanditError propagates.
4. THE Extract_Function SHALL use either a context manager (`with Client() as client:`) or a `try/finally` block to guarantee Dask_Client cleanup.

### Requirement 5: Guard the Working Directory Change

**User Story:** As a developer, I want the working directory to be restored to its original value regardless of how the extraction exits, so that callers and subsequent operations are never left in an unexpected directory.

#### Acceptance Criteria

1. WHEN the Extract_Function changes the working directory via `os.chdir()`, THE Working_Directory_Guard SHALL save the original working directory before the change.
2. WHEN the guarded block exits normally, THE Working_Directory_Guard SHALL restore the original working directory.
3. IF an exception is raised inside the guarded block, THEN THE Working_Directory_Guard SHALL restore the original working directory before the exception propagates.
4. IF no `job_dir` is provided, THEN THE Extract_Function SHALL not change the working directory and the Working_Directory_Guard SHALL have no effect.

### Requirement 6: Preserve Existing Error Messages and Logging

**User Story:** As a modeler, I want the same error messages and log entries I see today, so that my existing workflows and log-parsing scripts continue to work.

#### Acceptance Criteria

1. WHEN a BanditError is raised, THE Extract_Function SHALL log the same error message to the bandit log that was logged before the refactor.
2. WHEN a BanditError is raised, THE Extract_Function SHALL print the same console message (including Rich markup) that was printed before the refactor.
3. THE Extract_Function SHALL preserve the existing log-level (ERROR) for all error paths that previously called `exit()`.

### Requirement 7: Maintain Backward-Compatible Exit Codes

**User Story:** As a pipeline operator, I want the same exit codes for the same error conditions, so that my automation scripts continue to detect failures correctly.

#### Acceptance Criteria

1. WHEN an invalid job directory is provided, THE CLI_Entry_Point SHALL exit with code -1.
2. WHEN configuration validation fails, THE CLI_Entry_Point SHALL exit with code 2.
3. WHEN no requested stream segments exist in the NHM, THE CLI_Entry_Point SHALL exit with code 200.
4. WHEN no HRUs are associated with any segments, THE CLI_Entry_Point SHALL exit with code 2.
5. WHEN `dyn_params_dir` is required but not specified, THE CLI_Entry_Point SHALL exit with code 2.
6. WHEN `dyn_params_dir` does not exist on disk, THE CLI_Entry_Point SHALL exit with code 2.
