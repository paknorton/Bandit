#!/usr/bin/env python3

"""Shared scaffolding for the Bandit command-line entry points.

The individual entry points (``bandit``, ``bandit_single_hru``,
``bandit_by_streamgage``, ``bandit_multi_locations``) historically carried
copy-pasted implementations of the same concerns: logging setup, the
top-level ``BanditError`` handler, the parameter-database revision check, and
(for the two batch drivers) the worker-thread pool and the per-run summary
reporting. This module collects those so each entry point shares one
implementation.

User-facing console output is written explicitly with ``con.print()``; the
``bandit`` logger records messages in bandit.log without echoing them to the
console. The two are kept separate so a logged error is not also printed to
the console as a duplicate of the ``con.print()`` message.
"""

import datetime
import html
import json
import logging
import os
import queue
import subprocess
import sys
import threading

from pathlib import Path
from typing import Any, Callable, List, Mapping, NamedTuple, Optional, Tuple, Union

from pyPRMS.base.console import get_console_instance   # type: ignore

from Bandit import bandit_cfg as bc
from Bandit.exceptions import BanditError
from Bandit.git_version import Repo

__author__ = 'Parker Norton (pnorton@usgs.gov)'

# Shared Rich console (a process-wide singleton) for user-facing output.
con = get_console_instance()

# bandit uses return codes >= this value to report extraction errors that do
# not require shutting the whole batch down. Smaller non-zero codes indicate a
# failure that should stop the batch.
BANDIT_ERROR_FLOOR = 200

# Arguments that the batch drivers set for each extraction and so must not be
# supplied as pass-through arguments.
RESERVED_BANDIT_ARGS = {'--job-dir', '-j', '--config-file'}


# ---------------------------------------------------------------------------
# Logging and CLI error handling
# ---------------------------------------------------------------------------

def get_logger(name: str = 'bandit') -> logging.Logger:
    """Return the named logger with a NullHandler attached.

    The NullHandler keeps the logging framework from falling back to its
    last-resort stderr handler when no file handler is attached, so logged
    messages are never echoed to the console. Attaching it more than once (the
    entry points share the ``bandit`` logger) is harmless.

    :param name: logger name
    :returns: the configured logger
    """

    logger = logging.getLogger(name)
    logger.addHandler(logging.NullHandler())
    return logger


# Module logger used by the shared batch scaffolding below.
bandit_log = get_logger('bandit')


def run_cli(app: Callable[[], None], logger: Optional[logging.Logger] = None) -> None:
    """Run a cyclopts app, translating BanditError to a clean exit.

    This is the common body of every entry point's ``main()``: a BanditError
    raised anywhere in the command is logged (to bandit.log) and printed once
    to the console, then the process exits with the error's exit code.

    :param app: the cyclopts App to invoke
    :param logger: logger to record the error with (defaults to the bandit logger)
    """

    log = logger if logger is not None else bandit_log
    try:
        app()
    except BanditError as err:
        log.error(str(err))
        con.print(f'[red]ERROR[/]: {err}')
        sys.exit(err.exit_code)


def check_paramdb_revision(config: bc.Cfg) -> None:
    """Verify the parameter database is at the revision named in the config.

    When the config's ``revision`` variable is set, the parameter database at
    ``paramdb_dir`` must currently be checked out at that branch, commit (full
    or partial), or tag. An empty ``revision`` skips the check.

    :param config: Loaded Bandit configuration
    :raises BanditError: If the paramdb is not at the requested revision
    """

    cfg_revision = config.revision if config.exists('revision') else ''
    if not cfg_revision:
        return

    pdb_repo = Repo(config.paramdb_dir)
    if not pdb_repo.matches_revision(cfg_revision):
        # The BanditError is surfaced to the console and log by run_cli().
        raise BanditError(f"Parameter database revision mismatch: config 'revision' is "
                          f"'{cfg_revision}' but paramdb is on branch '{pdb_repo.branch}' "
                          f"(commit {pdb_repo.commit})", exit_code=2)


# ---------------------------------------------------------------------------
# Batch-driver worker pool
# ---------------------------------------------------------------------------

class Result(NamedTuple):
    """Result of running one bandit extraction in a worker thread."""

    thread: str          # Name of the worker thread that ran the command
    cmd: List[str]       # The command that was run
    retcode: int         # Process return code (-1 if it failed to launch)


class WorkerThread(threading.Thread):
    """A worker thread that pulls bandit commands off a queue and runs them.

    Input is done by placing commands (as lists of tokens) into the queue
    passed in ``input_q``.

    Output is done by placing tuples into the queue passed in ``result_q``.
    Each tuple is (thread name, command, return code).

    Ask the thread to stop by calling its :meth:`join` method.
    """

    def __init__(self, input_q: queue.Queue, result_q: queue.Queue):
        """Init method for WorkerThread.

        :param input_q: Queue of commands to run
        :param result_q: Queue to place command results into
        """
        super().__init__()
        # Daemon threads so an unhandled interrupt in the main thread cannot
        # be blocked indefinitely waiting on a worker to finish.
        self.daemon = True
        self.input_q = input_q
        self.result_q = result_q
        self.stoprequest = threading.Event()

    def run(self):
        """Pull commands from the input queue and run them until asked to stop.

        The tasks are taken with a blocking ``get`` given a short timeout so
        that ``stoprequest`` is always checked, even when the queue is empty.
        """
        while not self.stoprequest.is_set():
            try:
                thecmd = self.input_q.get(True, 0.05)
                retcode = self.run_cmd(thecmd)
                self.result_q.put(Result(thread=self.name, cmd=thecmd, retcode=retcode))
            except queue.Empty:
                continue

    def join(self, timeout=None):
        """Request the thread stop and wait for it to finish.

        :param timeout: Optional timeout in seconds
        """
        self.stoprequest.set()
        super().join(timeout)

    def run_cmd(self, thecmd: List[str]) -> int:
        """Run a single bandit command in a subprocess.

        :param thecmd: Command to run, as a list of argument tokens
        :returns: Return code from the subprocess (-1 on failure to launch)
        """
        retcode = 0

        try:
            retcode = subprocess.call(thecmd)
        except Exception as err:   # noqa: BLE001
            # Error launching the command; shut this worker down
            bandit_log.error(f'run_cmd() failed to launch {thecmd}: {err}')
            retcode = -1
            self.stoprequest.set()
        return retcode


def stop_pool(pool: List[WorkerThread], cmd_q: queue.Queue):
    """Stop a pool of worker threads, abandoning any un-started work.

    Any commands still queued are discarded so that no new extractions start.
    Each worker is then asked to stop and briefly joined. Workers are daemon
    threads, so an extraction still running past the join timeout does not
    block process exit (its output directory may be left incomplete).

    :param pool: Worker threads to stop
    :param cmd_q: Command queue to drain of un-started work
    """

    # Discard any queued commands that have not been picked up yet
    while True:
        try:
            cmd_q.get_nowait()
        except queue.Empty:
            break

    for thread in pool:
        thread.join(timeout=5.0)


# ---------------------------------------------------------------------------
# Per-run summary reporting
# ---------------------------------------------------------------------------

# A summary row: (name, HRUs, segments, POIs, status). Count fields are ints
# for completed runs, otherwise a placeholder string such as '-'.
SummaryRow = Tuple[str, Union[int, str], Union[int, str], Union[int, str], str]


def collect_summary_rows(dir_by_name: Mapping[Any, str],
                         job_dir: Union[str, Path],
                         skipped: Optional[Mapping[Any, str]] = None) -> List[SummaryRow]:
    """Collect per-run summary rows for the roll-up report.

    Each extraction's counts are read from the ``summary.json`` written by
    bandit into that run's output directory. A run that did not produce a
    summary (e.g. it failed or was never started) is marked as not completed.
    Entries skipped before extraction are listed with the reason they were
    skipped.

    :param dir_by_name: Mapping of run name (streamgage or location) to its
        output directory name
    :param job_dir: Top-level job directory the extractions ran under
    :param skipped: Optional mapping of skipped run name to the skip reason
    :returns: List of summary rows, completed/failed first then skipped
    """

    rows: List[SummaryRow] = []
    for name, cdir in dir_by_name.items():
        summary_file = Path(job_dir) / cdir / 'summary.json'

        num_hru: Union[int, str]
        num_seg: Union[int, str]
        num_poi: Union[int, str]
        status: str

        if summary_file.is_file():
            try:
                with open(summary_file, 'r') as fhdl:
                    data = json.load(fhdl)
                num_hru = data.get('num_hru', '?')
                num_seg = data.get('num_segments', '?')
                num_poi = data.get('num_poi', '?')
                status = 'completed'
            except (OSError, json.JSONDecodeError):
                num_hru = num_seg = num_poi = '-'
                status = 'unreadable summary'
        else:
            num_hru = num_seg = num_poi = '-'
            status = 'not completed'

        rows.append((str(name), num_hru, num_seg, num_poi, status))

    # Append rows for entries that were skipped before any extraction ran
    for name, reason in (skipped or {}).items():
        rows.append((str(name), '-', '-', '-', f'skipped: {reason}'))

    return rows


def print_summary_table(rows: List[SummaryRow], label: str = 'Location'):
    """Print a summary table of the batch extraction to the console.

    :param rows: Summary rows from :func:`collect_summary_rows`
    :param label: heading for the first column (e.g. 'Streamgage' or 'Location')
    """

    from rich.table import Table

    table = Table(title='Bandit extraction summary')
    table.add_column(label)
    table.add_column('HRUs', justify='right')
    table.add_column('Segments', justify='right')
    table.add_column('POIs', justify='right')
    table.add_column('Status')

    for name, num_hru, num_seg, num_poi, status in rows:
        if status == 'completed':
            style = 'green4'
        elif status.startswith('skipped'):
            style = 'dark_orange3'
        else:
            style = 'red'

        table.add_row(str(name), str(num_hru), str(num_seg), str(num_poi),
                      f'[{style}]{status}[/]')

    con.print(table)


def write_summary_report(report_path: Union[str, Path], rows: List[SummaryRow],
                         label: str = 'Location'):
    """Write an HTML table summarizing each extraction.

    :param report_path: Path of the HTML report to write
    :param rows: Summary rows from :func:`collect_summary_rows`
    :param label: heading for the first column (e.g. 'Streamgage' or 'Location')
    """

    generated = datetime.datetime.now().isoformat(timespec='seconds')

    lines = ['<!DOCTYPE html>',
             '<html lang="en">',
             '<head>',
             '<meta charset="utf-8">',
             '<title>Bandit extraction summary</title>',
             '<style>',
             'body { font-family: sans-serif; margin: 2em; }',
             'table { border-collapse: collapse; }',
             'th, td { border: 1px solid #ccc; padding: 4px 10px; text-align: right; }',
             'th { background: #f0f0f0; }',
             'td.name, td.status { text-align: left; }',
             '</style>',
             '</head>',
             '<body>',
             '<h1>Bandit extraction summary</h1>',
             f'<p>Generated {html.escape(generated)}</p>',
             '<table>',
             '<thead>',
             f'<tr><th>{html.escape(label)}</th><th>HRUs</th><th>Segments</th>'
             '<th>POIs</th><th>Status</th></tr>',
             '</thead>',
             '<tbody>']

    for name, num_hru, num_seg, num_poi, status in rows:
        lines.append(f'<tr><td class="name">{html.escape(str(name))}</td>'
                     f'<td>{html.escape(str(num_hru))}</td>'
                     f'<td>{html.escape(str(num_seg))}</td>'
                     f'<td>{html.escape(str(num_poi))}</td>'
                     f'<td class="status">{html.escape(status)}</td></tr>')

    lines.extend(['</tbody>', '</table>', '</body>', '</html>', ''])

    with open(report_path, 'w') as fhdl:
        fhdl.write('\n'.join(lines))


def run_worker_pool(commands: List[List[str]], num_threads: int) -> bool:
    """Run a list of bandit commands through a worker-thread pool.

    Starts up to ``num_threads`` workers (never more than there are commands),
    queues every command, and displays a Rich progress bar as extractions
    complete. The pool is stopped afterward, abandoning any un-started work.

    An extraction whose return code is non-zero and below
    :data:`BANDIT_ERROR_FLOOR`, or a keyboard interrupt, aborts the batch:
    remaining queued work is discarded and the function returns ``True``.

    :param commands: list of bandit commands (each a list of argument tokens)
    :param num_threads: maximum number of concurrent extractions
    :returns: True if the batch was aborted before all commands completed
    """

    from rich.progress import (BarColumn, MofNCompleteColumn, Progress, SpinnerColumn,
                                TaskProgressColumn, TextColumn, TimeElapsedColumn,
                                TimeRemainingColumn)

    cmd_q: queue.Queue = queue.Queue()
    result_q: queue.Queue = queue.Queue()

    # No point starting more workers than there are extractions to run.
    pool_size = min(num_threads, len(commands))
    pool = [WorkerThread(input_q=cmd_q, result_q=result_q) for __ in range(pool_size)]

    for thread in pool:
        thread.start()

    for cmd in commands:
        cmd_q.put(cmd)

    total_jobs = len(commands)

    progress = Progress(SpinnerColumn(),
                        TextColumn('[progress.description]{task.description}'),
                        BarColumn(),
                        MofNCompleteColumn(),
                        TaskProgressColumn(),
                        TextColumn('•'),
                        TimeElapsedColumn(),
                        TextColumn('elapsed •'),
                        TimeRemainingColumn(),
                        TextColumn('remaining'),
                        console=con)

    aborted = False

    with progress:
        task = progress.add_task('Extracting', total=total_jobs)

        remaining = total_jobs
        while remaining > 0:
            try:
                result = result_q.get()
            except KeyboardInterrupt:
                con.print('\n[gold3]Interrupted[/]: stopping remaining extractions')
                aborted = True
                break

            progress.advance(task)
            remaining -= 1

            if result.retcode != 0 and result.retcode < BANDIT_ERROR_FLOOR:
                # An error occurred running the command. Return codes at or
                # above BANDIT_ERROR_FLOOR are generated by bandit extraction
                # errors that don't necessitate shutting the entire job down.
                con.print(f'[red]ERROR[/]: Thread {result.thread} return code = {result.retcode} '
                          f'({" ".join(result.cmd)})')
                aborted = True
                break

    # Stop the workers, discarding any extractions that have not yet started.
    # Extractions already running are allowed to finish.
    stop_pool(pool, cmd_q)

    return aborted
