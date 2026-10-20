"""Tests for the pure helpers in Bandit/bandit_by_streamgage.py.

These cover the CSV reader and the streamgage-to-segment mapping, neither of
which needs a parameter database or a running extraction.
"""

import pytest

from Bandit.bandit_by_streamgage import get_streamgage_segments, read_file
from Bandit.exceptions import BanditError


# ---------------------------------------------------------------------------
# read_file
# ---------------------------------------------------------------------------


class TestReadFile:
    """read_file parses a header + 'id,int,int,...' rows into a dict."""

    def test_basic(self, tmp_path):
        f = tmp_path / 'vals.csv'
        f.write_text('poi_id,ids\n00000001,4,67,8\n00000002,12\n')
        assert read_file(f) == {'00000001': [4, 67, 8], '00000002': [12]}

    def test_header_is_skipped(self, tmp_path):
        f = tmp_path / 'vals.csv'
        f.write_text('poi_id,ids\n00000001,5\n')
        result = read_file(f)
        assert 'poi_id' not in result
        assert result == {'00000001': [5]}

    def test_blank_lines_ignored(self, tmp_path):
        f = tmp_path / 'vals.csv'
        f.write_text('poi_id,ids\n\n00000001,5\n   \n')
        assert read_file(f) == {'00000001': [5]}

    def test_id_with_no_values(self, tmp_path):
        """A POI ID with no trailing values maps to an empty list."""
        f = tmp_path / 'vals.csv'
        f.write_text('poi_id,ids\n00000001\n')
        assert read_file(f) == {'00000001': []}

    def test_non_integer_value_raises(self, tmp_path):
        f = tmp_path / 'vals.csv'
        f.write_text('poi_id,ids\n00000001,abc\n')
        with pytest.raises(BanditError):
            read_file(f)


# ---------------------------------------------------------------------------
# get_streamgage_segments
# ---------------------------------------------------------------------------


class TestGetStreamgageSegments:
    """get_streamgage_segments maps gage IDs to segments, skipping invalid ones."""

    def _write_gages(self, tmp_path, ids):
        f = tmp_path / 'gages.txt'
        f.write_text('\n'.join(ids) + '\n')
        return f

    def test_valid_gages_mapped(self, tmp_path):
        gages = self._write_gages(tmp_path, ['111', '222'])
        poi_to_seg = {'111': 10, '222': 20}
        segs, skipped = get_streamgage_segments(gages, poi_to_seg)
        assert segs == {'111': 10, '222': 20}
        assert skipped == {}

    def test_gage_not_a_poi_is_skipped(self, tmp_path):
        gages = self._write_gages(tmp_path, ['111', '999'])
        poi_to_seg = {'111': 10}
        segs, skipped = get_streamgage_segments(gages, poi_to_seg)
        assert segs == {'111': 10}
        assert '999' in skipped
        assert 'not a POI' in skipped['999']

    def test_gage_with_zero_segment_is_skipped(self, tmp_path):
        gages = self._write_gages(tmp_path, ['111'])
        poi_to_seg = {'111': 0}
        segs, skipped = get_streamgage_segments(gages, poi_to_seg)
        assert segs == {}
        assert skipped['111'] == 'poi_gage_segment = 0'

    def test_blank_lines_ignored(self, tmp_path):
        f = tmp_path / 'gages.txt'
        f.write_text('111\n\n   \n222\n')
        segs, skipped = get_streamgage_segments(f, {'111': 10, '222': 20})
        assert segs == {'111': 10, '222': 20}
        assert skipped == {}
