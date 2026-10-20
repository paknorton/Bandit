"""Tests for the config_tool utility (Bandit/utilities/config_tool.py).

Covers the pure helper functions and the verify/diff/update commands driven
against temporary config files. None of this needs a parameter database; the
GIS-verification path (which requires a readable GIS source via pyogrio) is
exercised only in its "section missing" form.
"""

from pathlib import Path

import pytest

import Bandit.bandit_cfg as bc
from Bandit.exceptions import BanditError
from Bandit.utilities import config_tool as ct


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _write(path, text):
    path.write_text(text)
    return path


# A valid config whose path-type variables all point at files/dirs created by
# the fixtures below, so `verify` reports no problems.
def _valid_cfg_text(paramdb_dir, control_file):
    return (f'start_date: 1990-01-01\n'
            f'end_date: 2000-12-31\n'
            f'paramdb_dir: {paramdb_dir}\n'
            f'control_filename: {control_file}\n'
            f'output_dir: {paramdb_dir}\n'
            f'param_filename: myparam.param\n'
            f'outlets:\n  - 1\n'
            f'output_cbh: false\n'
            f'output_shapefiles: false\n')


@pytest.fixture
def valid_config(tmp_path):
    """Create a config plus the paramdb dir and control file it references."""
    paramdb = tmp_path / 'paramdb'
    paramdb.mkdir()
    control = tmp_path / 'control.default'
    control.write_text('control\n')
    cfg = _write(tmp_path / 'bandit.cfg', _valid_cfg_text(paramdb, control))
    return cfg


# ---------------------------------------------------------------------------
# _format_value
# ---------------------------------------------------------------------------


class TestFormatValue:
    def test_none_is_empty_string(self):
        assert ct._format_value(None) == ''

    def test_scalar(self):
        assert ct._format_value(42) == '42'
        assert ct._format_value('hi') == 'hi'

    def test_list(self):
        assert ct._format_value([1, 2, 3]) == '[1, 2, 3]'

    def test_dict(self):
        assert ct._format_value({'a': 1, 'b': 2}) == '{a: 1, b: 2}'

    def test_nested(self):
        assert ct._format_value({'layers': [1, 2]}) == '{layers: [1, 2]}'


# ---------------------------------------------------------------------------
# _against_base
# ---------------------------------------------------------------------------


class TestAgainstBase:
    def test_absolute_path_unchanged(self):
        assert ct._against_base('/abs/path', Path('/base')) == '/abs/path'

    def test_relative_resolved_against_base(self):
        assert ct._against_base('control.default', Path('/base')) == '/base/control.default'


# ---------------------------------------------------------------------------
# _resolve_path_status
# ---------------------------------------------------------------------------


class TestResolvePathStatus:
    def test_empty_value_is_ok(self, tmp_path):
        assert ct._resolve_path_status('poi_dir', '', tmp_path) is None
        assert ct._resolve_path_status('poi_dir', None, tmp_path) is None

    def test_existing_absolute_path_ok(self, tmp_path):
        f = tmp_path / 'thing'
        f.mkdir()
        assert ct._resolve_path_status('paramdb_dir', str(f), tmp_path) is None

    def test_missing_path_reports_problem(self, tmp_path):
        msg = ct._resolve_path_status('paramdb_dir', str(tmp_path / 'nope'), tmp_path)
        assert msg is not None and 'does not exist' in msg

    def test_relative_path_resolved_against_base(self, tmp_path):
        (tmp_path / 'control.default').write_text('x')
        assert ct._resolve_path_status('control_filename', 'control.default', tmp_path) is None

    def test_glob_pattern_with_match_ok(self, tmp_path):
        (tmp_path / 'a_pois.nc').write_text('x')
        assert ct._resolve_path_status('poi_dir', str(tmp_path / '*_pois.nc'), tmp_path) is None

    def test_glob_pattern_without_match_reports(self, tmp_path):
        msg = ct._resolve_path_status('poi_dir', str(tmp_path / '*_pois.nc'), tmp_path)
        assert msg is not None and 'No files match' in msg


# ---------------------------------------------------------------------------
# _resolve_include
# ---------------------------------------------------------------------------


class TestResolveInclude:
    def test_none_returns_empty_set(self):
        assert ct._resolve_include(None) == set()

    def test_comma_separated(self):
        result = ct._resolve_include(['output_dir,start_date'])
        assert result == {'output_dir', 'start_date'}

    def test_repeated_option(self):
        result = ct._resolve_include(['output_dir', 'end_date'])
        assert result == {'output_dir', 'end_date'}

    def test_whitespace_trimmed(self):
        assert ct._resolve_include([' output_dir , end_date ']) == {'output_dir', 'end_date'}

    def test_nested_gis_key_allowed(self):
        assert ct._resolve_include(['gis.dst_extension']) == {'gis.dst_extension'}

    def test_unknown_name_raises(self):
        with pytest.raises(BanditError) as exc_info:
            ct._resolve_include(['not_a_protected_var'])
        assert exc_info.value.exit_code == 2


# ---------------------------------------------------------------------------
# _diff_gis
# ---------------------------------------------------------------------------


class TestDiffGis:
    def test_identical_sections_no_rows(self):
        gis = {'src_filename': '/a.gpkg', 'dst_extension': 'shp'}
        assert ct._diff_gis(dict(gis), dict(gis)) == []

    def test_differing_value_reported(self):
        src = {'dst_extension': 'shp'}
        other = {'dst_extension': 'gpkg'}
        rows = ct._diff_gis(src, other)
        assert rows == [('gis.dst_extension', 'shp', 'gpkg')]

    def test_key_only_in_one_side_uses_sentinel(self):
        rows = ct._diff_gis({'src_filename': '/a.gpkg'}, {})
        assert len(rows) == 1
        var, src_val, other_val = rows[0]
        assert var == 'gis.src_filename'
        assert src_val == '/a.gpkg'
        assert other_val is ct._MISSING


# ---------------------------------------------------------------------------
# _config_keys
# ---------------------------------------------------------------------------


class TestConfigKeys:
    def test_returns_top_level_keys(self, valid_config):
        cfg = bc.Cfg(valid_config)
        keys = ct._config_keys(cfg)
        assert 'paramdb_dir' in keys
        assert 'start_date' in keys
        assert 'outlets' in keys


# ---------------------------------------------------------------------------
# verify command
# ---------------------------------------------------------------------------


class TestVerify:
    def test_valid_config_passes(self, valid_config):
        # No problems -> returns without raising.
        ct.verify(valid_config)

    def test_missing_paramdb_raises(self, tmp_path):
        control = tmp_path / 'control.default'
        control.write_text('x')
        cfg = _write(tmp_path / 'bandit.cfg',
                     f'paramdb_dir: {tmp_path / "missing"}\n'
                     f'control_filename: {control}\n')
        with pytest.raises(BanditError) as exc_info:
            ct.verify(cfg)
        assert exc_info.value.exit_code == 2


# ---------------------------------------------------------------------------
# diff command
# ---------------------------------------------------------------------------


class TestDiff:
    def test_identical_configs_no_raise(self, tmp_path):
        text = 'paramdb_dir: /data/pdb\nstart_date: 1990-01-01\n'
        a = _write(tmp_path / 'a.cfg', text)
        b = _write(tmp_path / 'b.cfg', text)
        # diff reports to the console and returns None; should not raise.
        ct.diff(a, b)

    def test_differing_configs_no_raise(self, tmp_path):
        a = _write(tmp_path / 'a.cfg', 'paramdb_dir: /data/pdb_a\n')
        b = _write(tmp_path / 'b.cfg', 'paramdb_dir: /data/pdb_b\n')
        ct.diff(a, b)


# ---------------------------------------------------------------------------
# update command
# ---------------------------------------------------------------------------


class TestUpdate:
    def test_updates_unprotected_variable(self, tmp_path):
        src = _write(tmp_path / 'src.cfg',
                     'paramdb_dir: /new/pdb\noutput_dir: /src/out\n')
        target = _write(tmp_path / 'target.cfg',
                        'paramdb_dir: /old/pdb\noutput_dir: /target/out\n')

        ct.update(src, target)

        reloaded = bc.Cfg(target)
        # paramdb_dir is not protected: it gets updated from the source.
        assert reloaded.get_value('paramdb_dir') == '/new/pdb'

    def test_protected_variable_not_updated_by_default(self, tmp_path):
        src = _write(tmp_path / 'src.cfg',
                     'paramdb_dir: /new/pdb\noutput_dir: /src/out\n')
        target = _write(tmp_path / 'target.cfg',
                        'paramdb_dir: /old/pdb\noutput_dir: /target/out\n')

        ct.update(src, target)

        reloaded = bc.Cfg(target)
        # output_dir is protected: the target keeps its own value.
        assert reloaded.get_value('output_dir') == '/target/out'

    def test_protected_variable_updated_when_included(self, tmp_path):
        src = _write(tmp_path / 'src.cfg',
                     'paramdb_dir: /new/pdb\noutput_dir: /src/out\n')
        target = _write(tmp_path / 'target.cfg',
                        'paramdb_dir: /old/pdb\noutput_dir: /target/out\n')

        ct.update(src, target, include=['output_dir'])

        reloaded = bc.Cfg(target)
        assert reloaded.get_value('output_dir') == '/src/out'

    def test_variable_absent_in_target_is_not_injected(self, tmp_path):
        src = _write(tmp_path / 'src.cfg',
                     'paramdb_dir: /new/pdb\ndyn_params_dir: /src/dyn\n')
        target = _write(tmp_path / 'target.cfg', 'paramdb_dir: /old/pdb\n')

        ct.update(src, target)

        reloaded = bc.Cfg(target)
        # dyn_params_dir was not in the target file, so it must not be added.
        assert 'dyn_params_dir' not in ct._config_keys(reloaded)
