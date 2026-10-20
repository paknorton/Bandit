"""Tests for WDFNStreamflow (Bandit/WDFN.py).

The online retrieval functions ``get_daily_streamflow`` and
``get_monitoring_locations`` are patched so no network access occurs; the
tests focus on how WDFNStreamflow assembles and writes the data (dummy record,
PRMS ASCII, and netCDF).
"""

import datetime
from unittest.mock import patch

import netCDF4 as nc   # type: ignore
import numpy as np
import pandas as pd   # type: ignore
import pytest

from Bandit.WDFN import DUMMY_GAGE_ID, WDFNStreamflow


ST_DATE = datetime.datetime(2020, 1, 1)
EN_DATE = datetime.datetime(2020, 1, 5)   # 5 days inclusive
N_DAYS = 5


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _data_rows(ascii_path):
    """Return the data rows (after the '####' header line) of a PRMS file."""
    lines = ascii_path.read_text().splitlines()
    idx = next(i for i, ll in enumerate(lines) if ll.startswith('####'))
    return [ll for ll in lines[idx + 1:] if ll.strip()]


def _read_strings(variable):
    """Read a netCDF char/string variable as a list of Python strings."""
    vals = variable[:]
    return [vv if isinstance(vv, str) else nc.chartostring(vv).item() for vv in vals]


def _fake_daily(gage_ids, st_date, en_date, api_key=None):
    """Stand-in for get_daily_streamflow: gage order preserved, NaN for misses."""
    idx = pd.date_range(st_date, en_date, freq='D')
    df = pd.DataFrame(index=idx)
    df.index.name = 'date'
    # First gage gets real values; any others are all-NaN (no data).
    for pos, gg in enumerate(gage_ids):
        if pos == 0:
            df[gg] = [10.0, 11.0, 12.0, 13.0, 14.0]
        else:
            df[gg] = np.nan
    return df.reindex(columns=gage_ids)


# ---------------------------------------------------------------------------
# Dummy (no-POI) path
# ---------------------------------------------------------------------------


class TestWDFNDummy:
    """With no gage IDs, a single dummy streamgage of missing values is used."""

    def test_dummy_gage_id_and_all_nan(self):
        sf = WDFNStreamflow(gage_ids=None, st_date=ST_DATE, en_date=EN_DATE)
        data = sf.data
        assert list(data.columns) == [DUMMY_GAGE_ID]
        assert data[DUMMY_GAGE_ID].isna().all()

    def test_empty_list_is_also_dummy(self):
        sf = WDFNStreamflow(gage_ids=[], st_date=ST_DATE, en_date=EN_DATE)
        assert list(sf.data.columns) == [DUMMY_GAGE_ID]

    def test_dummy_ascii_output(self, tmp_path):
        sf = WDFNStreamflow(gage_ids=None, st_date=ST_DATE, en_date=EN_DATE)
        out = tmp_path / 'sf_data'
        sf.write_ascii(out)

        text = out.read_text()
        assert 'runoff 1' in text
        assert f'// {DUMMY_GAGE_ID}' in text

        rows = _data_rows(out)
        assert len(rows) == N_DAYS
        # Each row is: year month day hour minute second value, value == -999
        assert rows[0].split() == ['2020', '1', '1', '0', '0', '0', '-999']
        assert all(r.split()[-1] == '-999' for r in rows)

    def test_dummy_netcdf_output(self, tmp_path):
        sf = WDFNStreamflow(gage_ids=None, st_date=ST_DATE, en_date=EN_DATE)
        out = tmp_path / 'sf_data.nc'
        sf.write_netcdf(out)

        with nc.Dataset(out) as ds:
            assert _read_strings(ds['poi_id']) == [DUMMY_GAGE_ID]
            assert ds.dimensions['time'].size == N_DAYS
            assert ds['discharge'].shape == (1, N_DAYS)


# ---------------------------------------------------------------------------
# Online (WDFN) path
# ---------------------------------------------------------------------------


class TestWDFNOnline:
    """With gage IDs, discharge and metadata are pulled from the WDFN helpers."""

    def _make(self, gage_ids, meta=None):
        with patch('Bandit.WDFN.get_daily_streamflow', side_effect=_fake_daily), \
             patch('Bandit.WDFN.get_monitoring_locations', return_value=(meta or {})):
            return WDFNStreamflow(gage_ids=gage_ids, st_date=ST_DATE, en_date=EN_DATE)

    def test_columns_match_requested_order(self):
        sf = self._make(['11111111', '22222222'])
        assert list(sf.data.columns) == ['11111111', '22222222']

    def test_ascii_values_and_missing(self, tmp_path):
        sf = self._make(['11111111', '22222222'])
        out = tmp_path / 'sf_data'
        sf.write_ascii(out)

        assert 'runoff 2' in out.read_text()
        rows = _data_rows(out)
        assert len(rows) == N_DAYS

        # Columns after the 6 date fields: gage '11111111' has data, the
        # second gage is missing (-999).
        first = rows[0].split()
        assert first[6] == '10.0'
        assert first[7] == '-999.0'

    def test_netcdf_variables_and_metadata(self, tmp_path):
        meta = {'11111111': {'poi_name': 'Test River', 'latitude': 44.5,
                             'longitude': -110.1, 'drainage_area': 123.4,
                             'drainage_area_contrib': 120.0}}
        sf = self._make(['11111111', '22222222'], meta=meta)
        out = tmp_path / 'sf_data.nc'
        sf.write_netcdf(out)

        with nc.Dataset(out) as ds:
            assert _read_strings(ds['poi_id']) == ['11111111', '22222222']
            assert _read_strings(ds['poi_name']) == ['Test River', '']
            assert ds['discharge'].shape == (2, N_DAYS)

            # Gage with metadata keeps its latitude; the other is fill/NaN.
            lat = ds['latitude'][:]
            assert lat[0] == pytest.approx(44.5, abs=1e-4)

            # Gage 0 has real discharge; gage 1 is entirely fill.
            disch = ds['discharge'][:]
            assert disch[0, 0] == pytest.approx(10.0, abs=1e-4)

    def test_streamflow_lookup_failure_falls_back_to_nan(self, tmp_path):
        """A WDFN discharge failure yields an all-NaN frame, not an exception."""
        with patch('Bandit.WDFN.get_daily_streamflow', side_effect=RuntimeError('boom')), \
             patch('Bandit.WDFN.get_monitoring_locations', return_value={}):
            sf = WDFNStreamflow(gage_ids=['11111111'], st_date=ST_DATE, en_date=EN_DATE)

        assert list(sf.data.columns) == ['11111111']
        assert sf.data['11111111'].isna().all()

        # A file is still written, with missing values.
        out = tmp_path / 'sf_data'
        sf.write_ascii(out)
        rows = _data_rows(out)
        assert len(rows) == N_DAYS
        assert all(r.split()[-1] == '-999' for r in rows)
