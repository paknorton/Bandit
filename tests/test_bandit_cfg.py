"""Tests for the YAML-backed configuration class (Bandit/bandit_cfg.py)."""

import datetime

import pytest

from Bandit.bandit_cfg import Cfg, default_values


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _write_cfg(tmp_path, text):
    """Write *text* to a bandit.cfg under tmp_path and return the path."""
    cfg_file = tmp_path / 'bandit.cfg'
    cfg_file.write_text(text)
    return cfg_file


# A small, valid configuration covering scalars, a list, and a nested mapping.
SAMPLE_CFG = """\
start_date: 1990-01-01
end_date: 2000-12-31
paramdb_dir: /data/paramdb
output_dir: /data/output
outlets:
  - 1234
  - 5678
gis:
  src_filename: /data/gis.gpkg
  dst_extension: shp
"""


# ---------------------------------------------------------------------------
# Loading and get_value / attribute access
# ---------------------------------------------------------------------------


class TestLoadAndGet:
    """Loading a file and reading values back."""

    def test_scalar_value(self, tmp_path):
        cfg = Cfg(_write_cfg(tmp_path, SAMPLE_CFG))
        assert cfg.get_value('paramdb_dir') == '/data/paramdb'

    def test_attribute_access_delegates_to_get_value(self, tmp_path):
        cfg = Cfg(_write_cfg(tmp_path, SAMPLE_CFG))
        assert cfg.paramdb_dir == '/data/paramdb'
        assert cfg.output_dir == '/data/output'

    def test_list_value(self, tmp_path):
        cfg = Cfg(_write_cfg(tmp_path, SAMPLE_CFG))
        assert list(cfg.outlets) == [1234, 5678]

    def test_nested_mapping_value(self, tmp_path):
        cfg = Cfg(_write_cfg(tmp_path, SAMPLE_CFG))
        gis = cfg.gis
        assert gis['src_filename'] == '/data/gis.gpkg'
        assert gis['dst_extension'] == 'shp'

    def test_bare_date_parses_as_date(self, tmp_path):
        """ruamel parses a bare YYYY-MM-DD value as a datetime.date."""
        cfg = Cfg(_write_cfg(tmp_path, SAMPLE_CFG))
        assert cfg.get_value('start_date') == datetime.date(1990, 1, 1)

    def test_missing_value_falls_back_to_default(self, tmp_path):
        """A variable absent from the file resolves to its built-in default."""
        cfg = Cfg(_write_cfg(tmp_path, 'paramdb_dir: /data/paramdb\n'))
        # streamflow_filename is not in the file; comes from default_values.
        assert cfg.get_value('streamflow_filename') == default_values['streamflow_filename']

    def test_unknown_variable_raises_keyerror(self, tmp_path):
        cfg = Cfg(_write_cfg(tmp_path, SAMPLE_CFG))
        with pytest.raises(KeyError):
            cfg.get_value('no_such_variable')

    def test_blank_scalar_parses_as_none(self, tmp_path):
        """A key with no value (e.g. 'poi_dir:') is parsed as None, not ''.

        This is the behavior behind the historical poi_dir handling: callers
        must treat None and '' alike when deciding whether a path was set.
        """
        cfg = Cfg(_write_cfg(tmp_path, 'poi_dir:\nparamdb_dir: /data/paramdb\n'))
        assert cfg.poi_dir is None


# ---------------------------------------------------------------------------
# exists
# ---------------------------------------------------------------------------


class TestExists:
    """exists() reports presence in the file or in the built-in defaults."""

    def test_present_in_file(self, tmp_path):
        cfg = Cfg(_write_cfg(tmp_path, SAMPLE_CFG))
        assert cfg.exists('paramdb_dir') is True

    def test_present_only_as_default(self, tmp_path):
        """A variable not in the file but with a default still 'exists'."""
        cfg = Cfg(_write_cfg(tmp_path, 'paramdb_dir: /data/paramdb\n'))
        assert cfg.exists('cbh_dir') is True

    def test_absent_everywhere(self, tmp_path):
        cfg = Cfg(_write_cfg(tmp_path, SAMPLE_CFG))
        assert cfg.exists('no_such_variable') is False


# ---------------------------------------------------------------------------
# is_empty
# ---------------------------------------------------------------------------


class TestIsEmpty:
    """is_empty() interprets emptiness per value type."""

    def test_empty_list(self, tmp_path):
        cfg = Cfg(_write_cfg(tmp_path, 'outlets: []\nparamdb_dir: /x\n'))
        assert cfg.is_empty('outlets') is True

    def test_non_empty_list(self, tmp_path):
        cfg = Cfg(_write_cfg(tmp_path, SAMPLE_CFG))
        assert cfg.is_empty('outlets') is False

    def test_empty_string(self, tmp_path):
        cfg = Cfg(_write_cfg(tmp_path, "output_dir: ''\nparamdb_dir: /x\n"))
        assert cfg.is_empty('output_dir') is True

    def test_non_empty_string(self, tmp_path):
        cfg = Cfg(_write_cfg(tmp_path, SAMPLE_CFG))
        assert cfg.is_empty('paramdb_dir') is False

    def test_bool_is_never_empty(self, tmp_path):
        cfg = Cfg(_write_cfg(tmp_path, 'output_cbh: false\nparamdb_dir: /x\n'))
        assert cfg.is_empty('output_cbh') is False

    def test_date_is_never_empty(self, tmp_path):
        cfg = Cfg(_write_cfg(tmp_path, SAMPLE_CFG))
        assert cfg.is_empty('start_date') is False


# ---------------------------------------------------------------------------
# update_value
# ---------------------------------------------------------------------------


class TestUpdateValue:
    """update_value() replaces values present in the loaded config."""

    def test_update_scalar(self, tmp_path):
        cfg = Cfg(_write_cfg(tmp_path, SAMPLE_CFG))
        cfg.update_value('paramdb_dir', '/new/paramdb')
        assert cfg.get_value('paramdb_dir') == '/new/paramdb'

    def test_update_list_with_list(self, tmp_path):
        cfg = Cfg(_write_cfg(tmp_path, SAMPLE_CFG))
        cfg.update_value('outlets', [9999])
        assert list(cfg.get_value('outlets')) == [9999]

    def test_update_list_with_scalar_wraps_in_sequence(self, tmp_path):
        """Assigning a scalar to a list variable stores it as a flow sequence."""
        cfg = Cfg(_write_cfg(tmp_path, SAMPLE_CFG))
        cfg.update_value('outlets', 42)
        assert list(cfg.get_value('outlets')) == [42]

    def test_update_absent_variable_raises(self, tmp_path):
        """update_value only updates keys present in the loaded file."""
        cfg = Cfg(_write_cfg(tmp_path, SAMPLE_CFG))
        # cbh_dir has a default but is not in this file, so it is not updatable.
        with pytest.raises(KeyError):
            cfg.update_value('cbh_dir', '/data/cbh')


# ---------------------------------------------------------------------------
# write round-trip
# ---------------------------------------------------------------------------


class TestWriteRoundTrip:
    """Writing a config and reloading it preserves values."""

    def test_round_trip_preserves_values(self, tmp_path):
        cfg = Cfg(_write_cfg(tmp_path, SAMPLE_CFG))
        cfg.update_value('paramdb_dir', '/updated/paramdb')
        cfg.update_value('outlets', [1, 2, 3])

        out_file = tmp_path / 'out.cfg'
        cfg.write(str(out_file))

        reloaded = Cfg(out_file)
        assert reloaded.get_value('paramdb_dir') == '/updated/paramdb'
        assert list(reloaded.get_value('outlets')) == [1, 2, 3]
        assert reloaded.get_value('end_date') == datetime.date(2000, 12, 31)
