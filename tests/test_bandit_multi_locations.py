"""Tests for the pure helpers in Bandit/bandit_multi_locations.py.

Covers the location-key normalization, the two CSV readers, and the output
directory naming - none of which require a parameter database or extraction.
"""

import pytest

from Bandit.bandit_multi_locations import (_location_key, location_dirname,
                                           read_outlets, read_values)
from Bandit.exceptions import BanditError


# ---------------------------------------------------------------------------
# _location_key
# ---------------------------------------------------------------------------


class TestLocationKey:
    """_location_key normalizes a raw key to int when possible, else str."""

    def test_integer_string_becomes_int(self):
        assert _location_key('42') == 42
        assert isinstance(_location_key('42'), int)

    def test_name_stays_string(self):
        assert _location_key('mill') == 'mill'
        assert isinstance(_location_key('mill'), str)


# ---------------------------------------------------------------------------
# read_outlets
# ---------------------------------------------------------------------------


class TestReadOutlets:
    """read_outlets requires at least one value per location."""

    def test_string_and_integer_keys(self, tmp_path):
        f = tmp_path / 'outlets.csv'
        f.write_text('name,outlets\nmill,1242\nnorwalk,1244,1245\n3,868\n')
        result = read_outlets(f)
        assert result == {'mill': [1242], 'norwalk': [1244, 1245], 3: [868]}

    def test_blank_lines_ignored(self, tmp_path):
        f = tmp_path / 'outlets.csv'
        f.write_text('name,outlets\n\nmill,1242\n  \n')
        assert read_outlets(f) == {'mill': [1242]}

    def test_duplicate_location_raises(self, tmp_path):
        f = tmp_path / 'outlets.csv'
        f.write_text('name,outlets\nmill,1242\nmill,1243\n')
        with pytest.raises(BanditError):
            read_outlets(f)

    def test_location_with_no_values_raises(self, tmp_path):
        f = tmp_path / 'outlets.csv'
        f.write_text('name,outlets\nmill\n')
        with pytest.raises(BanditError):
            read_outlets(f)

    def test_non_integer_value_raises(self, tmp_path):
        f = tmp_path / 'outlets.csv'
        f.write_text('name,outlets\nmill,abc\n')
        with pytest.raises(BanditError):
            read_outlets(f)


# ---------------------------------------------------------------------------
# read_values
# ---------------------------------------------------------------------------


class TestReadValues:
    """read_values is like read_outlets but allows zero values per location."""

    def test_location_with_no_values_is_allowed(self, tmp_path):
        f = tmp_path / 'cutoffs.csv'
        f.write_text('name,ids\nmill\nnorwalk,5,6\n')
        assert read_values(f) == {'mill': [], 'norwalk': [5, 6]}

    def test_duplicate_location_raises(self, tmp_path):
        f = tmp_path / 'cutoffs.csv'
        f.write_text('name,ids\nmill,1\nmill,2\n')
        with pytest.raises(BanditError):
            read_values(f)

    def test_non_integer_value_raises(self, tmp_path):
        f = tmp_path / 'cutoffs.csv'
        f.write_text('name,ids\nmill,x\n')
        with pytest.raises(BanditError):
            read_values(f)


# ---------------------------------------------------------------------------
# location_dirname
# ---------------------------------------------------------------------------


class TestLocationDirname:
    """location_dirname zero-pads integer IDs and leaves names as-is."""

    def test_integer_zero_padded(self):
        assert location_dirname(3, None) == '0003'

    def test_string_name_unchanged(self):
        assert location_dirname('mill', None) == 'mill'

    def test_prefix_applied_to_integer(self):
        assert location_dirname(3, 'run_') == 'run_0003'

    def test_prefix_applied_to_name(self):
        assert location_dirname('mill', 'run_') == 'run_mill'
