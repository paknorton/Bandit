"""Tests for the shared CLI scaffolding (Bandit/cli_shared.py).

Covers the summary-row collection, the HTML roll-up report, and the
parameter-database revision check. The worker-thread pool and run_cli() are
exercised indirectly elsewhere and are not unit-tested here.
"""

import json
import subprocess
from unittest.mock import MagicMock

import pytest

from Bandit.cli_shared import (check_paramdb_revision, collect_summary_rows,
                               print_summary_table, write_summary_report)
from Bandit.exceptions import BanditError


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_run(job_dir, name, summary=None):
    """Create a run subdirectory under job_dir, optionally with a summary.json.

    :returns: the directory name (relative) used for the run
    """
    run_dir = job_dir / name
    run_dir.mkdir()
    if summary is not None:
        (run_dir / 'summary.json').write_text(json.dumps(summary))
    return name


# ---------------------------------------------------------------------------
# collect_summary_rows
# ---------------------------------------------------------------------------


class TestCollectSummaryRows:
    """collect_summary_rows reads each run's summary.json into a status row."""

    def test_completed_run(self, tmp_path):
        _make_run(tmp_path, 'g1', {'num_hru': 5, 'num_segments': 3, 'num_poi': 1})
        rows = collect_summary_rows({'g1': 'g1'}, tmp_path)
        assert rows == [('g1', 5, 3, 1, 'completed')]

    def test_missing_summary_is_not_completed(self, tmp_path):
        _make_run(tmp_path, 'g1', summary=None)
        rows = collect_summary_rows({'g1': 'g1'}, tmp_path)
        assert rows[0][0] == 'g1'
        assert rows[0][4] == 'not completed'

    def test_unreadable_summary(self, tmp_path):
        run_dir = tmp_path / 'g1'
        run_dir.mkdir()
        (run_dir / 'summary.json').write_text('{ not valid json')
        rows = collect_summary_rows({'g1': 'g1'}, tmp_path)
        assert rows[0][4] == 'unreadable summary'

    def test_skipped_rows_appended_after(self, tmp_path):
        _make_run(tmp_path, 'g1', {'num_hru': 1, 'num_segments': 1, 'num_poi': 0})
        rows = collect_summary_rows({'g1': 'g1'}, tmp_path,
                                    skipped={'g2': 'not a POI'})
        assert rows[0][0] == 'g1' and rows[0][4] == 'completed'
        assert rows[1][0] == 'g2'
        assert rows[1][4] == 'skipped: not a POI'

    def test_integer_location_key(self, tmp_path):
        """Location IDs may be integers; the row name is stringified."""
        _make_run(tmp_path, '0003', {'num_hru': 2, 'num_segments': 2, 'num_poi': 0})
        rows = collect_summary_rows({3: '0003'}, tmp_path)
        assert rows[0][0] == '3'


# ---------------------------------------------------------------------------
# write_summary_report
# ---------------------------------------------------------------------------


class TestWriteSummaryReport:
    """write_summary_report emits an HTML table with the given column label."""

    def test_writes_html_with_label_and_rows(self, tmp_path):
        rows = [('g1', 5, 3, 1, 'completed'),
                ('g2', '-', '-', '-', 'skipped: not a POI')]
        report = tmp_path / 'summary.html'
        write_summary_report(report, rows, label='Streamgage')

        html = report.read_text()
        assert '<th>Streamgage</th>' in html
        assert 'g1' in html and 'g2' in html
        assert 'completed' in html
        assert 'skipped: not a POI' in html

    def test_label_defaults_to_location(self, tmp_path):
        report = tmp_path / 'summary.html'
        write_summary_report(report, [], label='Location')
        assert '<th>Location</th>' in report.read_text()

    def test_html_escapes_values(self, tmp_path):
        """Values are HTML-escaped so stray markup cannot break the table."""
        rows = [('a<b>', 1, 1, 0, 'completed')]
        report = tmp_path / 'summary.html'
        write_summary_report(report, rows, label='Location')
        html = report.read_text()
        assert 'a&lt;b&gt;' in html
        assert '<b>' not in html.split('<tbody>')[1]


# ---------------------------------------------------------------------------
# print_summary_table (smoke)
# ---------------------------------------------------------------------------


class TestPrintSummaryTable:
    """print_summary_table renders to the console without raising."""

    def test_runs_without_error(self):
        rows = [('g1', 5, 3, 1, 'completed'),
                ('g2', '-', '-', '-', 'not completed'),
                ('g3', '-', '-', '-', 'skipped: reason')]
        # Should not raise regardless of row status styling.
        print_summary_table(rows, label='Streamgage')


# ---------------------------------------------------------------------------
# check_paramdb_revision
# ---------------------------------------------------------------------------


def _git(repo, *args):
    subprocess.run(['git', *args], cwd=repo, check=True, capture_output=True, text=True)


@pytest.fixture
def paramdb_repo(tmp_path):
    """A git repo on branch 'main' with one commit, standing in for a paramdb."""
    repo = tmp_path / 'paramdb'
    repo.mkdir()
    _git(repo, 'init', '-b', 'main')
    _git(repo, 'config', 'user.email', 'test@example.com')
    _git(repo, 'config', 'user.name', 'Test User')
    (repo / 'param.txt').write_text('x\n')
    _git(repo, 'add', '.')
    _git(repo, 'commit', '-m', 'first')
    return repo


def _config(paramdb_dir, revision):
    """Build a minimal mock Cfg exposing paramdb_dir and revision."""
    values = {'paramdb_dir': str(paramdb_dir), 'revision': revision}
    cfg = MagicMock()
    cfg.exists.side_effect = lambda name: name in values
    cfg.paramdb_dir = values['paramdb_dir']
    cfg.revision = values['revision']
    return cfg


class TestCheckParamdbRevision:
    """check_paramdb_revision raises only on a real revision mismatch."""

    def test_empty_revision_is_a_noop(self, paramdb_repo):
        # Must not raise and must not even need a valid repo check.
        check_paramdb_revision(_config(paramdb_repo, ''))

    def test_matching_branch_passes(self, paramdb_repo):
        check_paramdb_revision(_config(paramdb_repo, 'main'))

    def test_mismatch_raises(self, paramdb_repo):
        with pytest.raises(BanditError) as exc_info:
            check_paramdb_revision(_config(paramdb_repo, 'some-other-branch'))
        assert exc_info.value.exit_code == 2

    def test_revision_absent_is_a_noop(self, paramdb_repo):
        """When the config has no 'revision' key at all, nothing is checked."""
        values = {'paramdb_dir': str(paramdb_repo)}
        cfg = MagicMock()
        cfg.exists.side_effect = lambda name: name in values
        cfg.paramdb_dir = values['paramdb_dir']
        check_paramdb_revision(cfg)
