#!/usr/bin/env python3
"""Generate per-streamgage HTML inspection reports for a Bandit extraction.

This module produces the same style of report as the standalone
``inspect_nwis_station_cache`` tool, but it is driven by the in-memory
streamflow object that Bandit builds during an extraction (either a
:class:`Bandit.points_of_interest.POI` backed by cached netCDF/zarr source
files, or a :class:`Bandit.WDFN.WDFNStreamflow` object populated from the
online USGS Water Data (WDFN) service).

For each POI in the model subset a self-contained HTML file is written
containing:

1. A duplicate-timestamp check (always clean for the dense daily POI grid,
   retained for parity with the cache-based report).
2. The daily series and period-of-record summary.
3. A table of missing-record gaps (longest first).
4. A per-year coverage table.
5. An interactive Plotly streamflow plot with a legend (Provisional in gold,
   Revised in dark orange, and colored qualifier tick marks for missing days).

When the streamflow source carries latitude/longitude metadata (the POI
source does), an ``index.html`` Leaflet map is also written. Each POI is
color-coded by its percent of missing days and links to its per-station
report.

``plotly`` (a Bandit dependency) is required to render the plots. If it is
somehow unavailable an informative :class:`ReportError` is raised rather than
aborting the whole extraction.
"""

import ast
import datetime
import html
import logging

import numpy as np
import pandas as pd   # type: ignore

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

logger = logging.getLogger(__name__)

# Discharge units for axis labels / hover text.
UNITS = 'cfs'

# Colors for known NWIS daily-value qualifier codes. Any qualifier not listed
# here falls back to QUALIFIER_DEFAULT_COLOR.
QUALIFIER_COLORS = {'ICE':                'deepskyblue',
                    'ESTIMATED':          'darkorange',
                    'EQUIP':              'purple',
                    'MAINT':              'brown',
                    'DRY':                'saddlebrown',
                    'ZEROFLOW':           'black',
                    'BACKWATER':          'teal',
                    'REGULATED':          'darkgreen',
                    'DEBRIS':             'olive',
                    'SEASONAL':           'goldenrod',
                    'TIDE':               'navy',
                    'PUMP':               'magenta',
                    'DISCONTINUED':       'gray',
                    'DATUMCHANGE':        'darkred',
                    'EVENT':              'hotpink',
                    'FORCEINTERPOLATION': 'slateblue',
                    'LESSTHAN':           'cadetblue',
                    'UNKNOWNREGULATION':  'dimgray'}
QUALIFIER_DEFAULT_COLOR = 'orchid'

# Color ramp for the index map, ordered by increasing percent-missing. Each
# entry is (upper_bound_inclusive, color, label). A station with pct_missing
# <= upper_bound takes that color. Stations with no data are drawn in
# NODATA_COLOR.
MISSING_PCT_BINS = [(0.0,   '#1a9850', '0% (complete)'),
                    (1.0,   '#66bd63', '0-1%'),
                    (5.0,   '#a6d96a', '1-5%'),
                    (10.0,  '#d9ef8b', '5-10%'),
                    (25.0,  '#fee08b', '10-25%'),
                    (50.0,  '#fdae61', '25-50%'),
                    (75.0,  '#f46d43', '50-75%'),
                    (100.0, '#d73027', '75-100%')]
NODATA_COLOR = '#000000'
NODATA_LABEL = 'No data'


class ReportError(Exception):
    """Raised when a streamgage report cannot be generated."""


def _require_plotly():
    """Import and return the plotly graph_objects module.

    :returns: the ``plotly.graph_objects`` module
    :raises ReportError: if plotly is not installed
    """

    try:
        import plotly.graph_objects as go   # type: ignore
    except ImportError as err:   # noqa: F841
        raise ReportError("The streamgage report requires the 'plotly' "
                          'package, which is a Bandit dependency but could '
                          "not be imported. Reinstall Bandit (e.g. 'pip "
                          "install -e .') or run 'pip install plotly'.") from None
    return go


def _qualifier_tokens(q: Any) -> List[str]:
    """Return a list of qualifier code strings from a qualifier field value.

    The POI source stores the qualifier per day either as a numpy array of
    codes (e.g. ``array(['ICE'])``), or - when read back from zarr/netCDF - as
    the *string repr* of such an array (e.g. ``"['ESTIMATED' 'REVISED']"``).
    Scalars, ``None`` and empty strings are also handled.

    :param q: qualifier field value for a single day
    :returns: list of qualifier code strings (may be empty)
    """

    if q is None:
        return []

    if isinstance(q, (list, tuple, np.ndarray)):
        return [str(x) for x in q if x is not None and str(x) != '']

    if isinstance(q, float) and np.isnan(q):
        return []

    text = str(q).strip()
    if not text:
        return []

    # Stringified list/array form, e.g. "['ESTIMATED' 'REVISED']" or
    # "['ICE']". numpy prints array elements space-separated without commas,
    # so a literal_eval of the raw text usually fails; normalize first.
    if text[0] in '[(' and text[-1] in '])':
        inner = text[1:-1].strip()
        if not inner:
            return []
        try:
            # Insert commas between quoted tokens so ast can parse the
            # numpy-style space-separated repr.
            normalized = '[' + ', '.join(_split_quoted_tokens(inner)) + ']'
            parsed = ast.literal_eval(normalized)
            return [str(x) for x in parsed if str(x) != '']
        except (ValueError, SyntaxError):
            # Fall back to a best-effort token split.
            return [t.strip("'\" ") for t in inner.split() if t.strip("'\" ")]

    return [text]


def _split_quoted_tokens(inner: str) -> List[str]:
    """Split the inner text of a stringified array into quoted token strings.

    Handles single- or double-quoted tokens separated by spaces and/or commas,
    e.g. ``"'ESTIMATED' 'REVISED'"`` -> ``["'ESTIMATED'", "'REVISED'"]``.

    :param inner: inner text (without the enclosing brackets)
    :returns: list of individually quoted token strings
    """

    tokens: List[str] = []
    i = 0
    n = len(inner)
    while i < n:
        ch = inner[i]
        if ch in '\'"':
            quote = ch
            j = i + 1
            while j < n and inner[j] != quote:
                j += 1
            tokens.append(f"'{inner[i + 1:j]}'")
            i = j + 1
        else:
            i += 1
    return tokens


# ---------------------------------------------------------------------------
# Normalization: adapt the POI / NWIS streamflow objects to a common shape
# ---------------------------------------------------------------------------

class _GageSeries:
    """Per-gage daily discharge plus optional approval/qualifier flags."""

    def __init__(self, gage_id: str,
                 discharge: pd.Series,
                 approval: Optional[pd.Series] = None,
                 qualifier: Optional[pd.Series] = None,
                 meta: Optional[Dict[str, Any]] = None):
        """Create a per-gage series container.

        :param gage_id: streamgage id
        :param discharge: daily discharge series indexed by datetime
        :param approval: optional daily approval-status series (aligned to
            discharge)
        :param qualifier: optional daily qualifier series (aligned to
            discharge); each value is parsed with :func:`_qualifier_tokens`
        :param meta: optional metadata dict (poi_name, latitude, longitude,
            drainage_area, drainage_area_contrib)
        """

        self.gage_id = gage_id
        self.discharge = discharge
        self.approval = approval
        self.qualifier = qualifier
        self.meta = meta or {}


def _clip_series(s: Optional[pd.Series],
                 st_date: Optional[datetime.datetime],
                 en_date: Optional[datetime.datetime]) -> Optional[pd.Series]:
    """Clip a datetime-indexed series to the inclusive [st_date, en_date] range.

    :param s: series to clip (returned unchanged when None)
    :param st_date: inclusive start bound, or None for no lower bound
    :param en_date: inclusive end bound, or None for no upper bound
    :returns: the clipped series (a view/copy)
    """

    if s is None or (st_date is None and en_date is None):
        return s
    return s.loc[st_date:en_date]


def _normalize_streamflow(streamflow: Any,
                          gage_ids: List[str],
                          st_date: Optional[datetime.datetime] = None,
                          en_date: Optional[datetime.datetime] = None) -> Tuple[List[_GageSeries], str]:
    """Build per-gage daily series from a POI or NWIS streamflow object.

    :param streamflow: a POI or NWIS object (must expose ``.data``)
    :param gage_ids: the ordered list of gage ids in the subset
    :param st_date: optional inclusive start date to clip the series to
    :param en_date: optional inclusive end date to clip the series to
    :returns: a tuple of (list of :class:`_GageSeries`, source-kind string)
    :raises ReportError: if the streamflow object cannot be interpreted
    """

    data = getattr(streamflow, 'data', None)
    if data is None:
        raise ReportError('Streamflow object has no data to report on.')

    # xarray Dataset => POI source (carries approval_status / qualifier / meta)
    if hasattr(data, 'data_vars'):
        return _normalize_from_poi(data, gage_ids, st_date, en_date), 'POI'

    # pandas DataFrame => NWIS source (discharge only)
    if isinstance(data, pd.DataFrame):
        return _normalize_from_nwis(data, gage_ids, st_date, en_date), 'NWIS'

    raise ReportError(f'Unsupported streamflow data type: {type(data)!r}')


def _normalize_from_poi(ds: Any,
                        gage_ids: List[str],
                        st_date: Optional[datetime.datetime] = None,
                        en_date: Optional[datetime.datetime] = None) -> List[_GageSeries]:
    """Build per-gage series from a POI xarray Dataset.

    :param ds: POI source dataset (dims poi_id x time)
    :param gage_ids: ordered list of gage ids in the subset
    :param st_date: optional inclusive start date to clip the series to
    :param en_date: optional inclusive end date to clip the series to
    :returns: list of :class:`_GageSeries`
    """

    series_list: List[_GageSeries] = []
    source_ids = set(ds['poi_id'].values.tolist())
    time_index = pd.to_datetime(ds['time'].values)

    for gg in gage_ids:
        if gg not in source_ids:
            # Ad-hoc gage missing from the source files => no data.
            series_list.append(_GageSeries(gg, pd.Series(dtype=float)))
            continue

        sub = ds.sel(poi_id=gg)

        discharge = _clip_series(pd.Series(np.asarray(sub['discharge'].values, dtype=float),
                                           index=time_index, name=gg),
                                 st_date, en_date)
        assert discharge is not None   # _clip_series only returns None for a None input

        approval = None
        if 'approval_status' in ds:
            approval = _clip_series(pd.Series(sub['approval_status'].values, index=time_index),
                                    st_date, en_date)

        qualifier = None
        if 'qualifier' in ds:
            qualifier = _clip_series(pd.Series(sub['qualifier'].values, index=time_index),
                                     st_date, en_date)

        meta = {}
        for key in ('poi_name', 'latitude', 'longitude',
                    'drainage_area', 'drainage_area_contrib', 'poi_agency'):
            if key in ds:
                val = sub[key].values
                meta[key] = val.item() if np.ndim(val) == 0 else val

        series_list.append(_GageSeries(gg, discharge, approval, qualifier, meta))

    return series_list


def _normalize_from_nwis(df: pd.DataFrame,
                         gage_ids: List[str],
                         st_date: Optional[datetime.datetime] = None,
                         en_date: Optional[datetime.datetime] = None) -> List[_GageSeries]:
    """Build per-gage series from an NWIS DataFrame (discharge only).

    :param df: NWIS output dataframe (date index, one column per gage)
    :param gage_ids: ordered list of gage ids in the subset
    :param st_date: optional inclusive start date to clip the series to
    :param en_date: optional inclusive end date to clip the series to
    :returns: list of :class:`_GageSeries`
    """

    series_list: List[_GageSeries] = []
    index = pd.to_datetime(df.index)

    for gg in gage_ids:
        if gg in df.columns:
            discharge = _clip_series(pd.Series(pd.to_numeric(df[gg], errors='coerce').to_numpy(dtype=float),
                                               index=index, name=gg),
                                     st_date, en_date)
            assert discharge is not None   # _clip_series only returns None for a None input
        else:
            discharge = pd.Series(dtype=float)
        series_list.append(_GageSeries(gg, discharge))

    return series_list


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------

def _analyze(gage: _GageSeries) -> dict:
    """Compute derived series and summaries for a single gage.

    :param gage: per-gage series container
    :returns: dict of computed artifacts used to build the HTML report
    :raises ValueError: if the gage has no usable observations
    """

    daily = gage.discharge.dropna()
    full_dates = gage.discharge.index

    if len(full_dates) == 0:
        raise ValueError('no observations for this streamgage')

    daily_full = gage.discharge.copy()
    daily_full.index = pd.to_datetime(daily_full.index).normalize()
    daily_full = daily_full.groupby(level=0).mean().sort_index()

    por_start = daily_full.index.min()
    por_end = daily_full.index.max()
    if pd.isna(por_start) or pd.isna(por_end):
        raise ValueError('no usable observations for this streamgage')

    full_index = pd.date_range(por_start, por_end, freq='D')
    daily_full = daily_full.reindex(full_index)
    daily_full.index.name = 'date'

    n_total_days = len(full_index)
    n_present = int(daily_full.notna().sum())
    n_missing = int(daily_full.isna().sum())
    pct_missing = (n_missing / n_total_days * 100.0) if n_total_days else 0.0

    if n_present == 0:
        raise ValueError('no non-missing discharge values for this streamgage')

    # --- Missing-record gaps ---------------------------------------------
    missing_mask = daily_full.isna()
    gaps = []
    if missing_mask.any():
        group_id = (missing_mask != missing_mask.shift()).cumsum()
        for _, grp in daily_full.groupby(group_id):
            if bool(missing_mask.loc[grp.index[0]]):
                gaps.append({'gap_start': grp.index[0].date(),
                             'gap_end':   grp.index[-1].date(),
                             'days':      len(grp)})
    df_gaps = pd.DataFrame(gaps)
    if not df_gaps.empty:
        df_gaps = df_gaps.sort_values('days', ascending=False).reset_index(drop=True)

    # --- Per-year coverage -----------------------------------------------
    by_year = pd.DataFrame({'present': daily_full.notna().groupby(daily_full.index.year).sum(),   # type: ignore[attr-defined]
                            'missing': daily_full.isna().groupby(daily_full.index.year).sum()})   # type: ignore[attr-defined]
    by_year['days_in_year'] = by_year['present'] + by_year['missing']
    by_year['pct_complete'] = (by_year['present'] / by_year['days_in_year'] * 100.0).round(1)
    by_year.index.name = 'year'
    by_year = by_year.reset_index()

    return {'daily_full':    daily_full,
            'missing_mask':  missing_mask,
            'por_start':     por_start,
            'por_end':       por_end,
            'n_total_days':  n_total_days,
            'n_present':     n_present,
            'n_missing':     n_missing,
            'pct_missing':   pct_missing,
            'df_gaps':       df_gaps,
            'by_year':       by_year}


def _build_figure(gage: _GageSeries,
                  analysis: dict):
    """Build the interactive Plotly streamflow figure for a gage.

    steelblue daily line, gold Provisional overlay, dark orange Revised
    overlay, crimson missing-day ticks, and colored per-qualifier ticks for
    Provisional missing days.

    :param gage: per-gage series container
    :param analysis: output of :func:`_analyze`
    :returns: a Plotly Figure
    """

    go = _require_plotly()

    daily_full = analysis['daily_full']
    missing_mask = analysis['missing_mask']
    n_missing = analysis['n_missing']
    pct_missing = analysis['pct_missing']
    por_start = analysis['por_start']
    por_end = analysis['por_end']

    # Daily-aligned approval_status / qualifier flags (normalized to day).
    prov_by_day = pd.Series(False, index=daily_full.index)
    rev_by_day = pd.Series(False, index=daily_full.index)

    approval_day = None
    if gage.approval is not None:
        approval_day = gage.approval.copy()
        approval_day.index = pd.to_datetime(approval_day.index).normalize()
        approval_day = approval_day.groupby(level=0).first()

        prov_days = approval_day.index[approval_day == 'Provisional']
        prov_by_day.loc[prov_by_day.index.isin(prov_days)] = True

    qualifier_day = None
    if gage.qualifier is not None:
        qualifier_day = gage.qualifier.copy()
        qualifier_day.index = pd.to_datetime(qualifier_day.index).normalize()
        qualifier_day = qualifier_day.groupby(level=0).first()

        rev_days = [d for d, q in qualifier_day.items() if 'REVISED' in _qualifier_tokens(q)]
        rev_by_day.loc[rev_by_day.index.isin(rev_days)] = True

    # Qualifier codes for Provisional missing days
    qualifier_days_by_code: Dict[str, list] = {}
    qualified_missing_days: set = set()

    if approval_day is not None and qualifier_day is not None:
        for cal_day in daily_full.index[missing_mask]:
            if approval_day.get(cal_day) != 'Provisional':
                continue
            codes = _qualifier_tokens(qualifier_day.get(cal_day))
            if not codes:
                continue
            qualified_missing_days.add(cal_day)
            for code in codes:
                qualifier_days_by_code.setdefault(code, []).append(cal_day)

    missing_dates = daily_full.index[missing_mask]
    plain_missing_dates = [d for d in missing_dates if d not in qualified_missing_days]

    y_max = float(np.nanmax(daily_full.values)) if analysis['n_present'] else 1.0
    tick_y = y_max * 1.02 if y_max > 0 else 1.0

    fig = go.Figure()

    fig.add_trace(go.Scattergl(x=daily_full.index,
                               y=daily_full.values,
                               mode='lines',
                               line=dict(color='steelblue', width=1),
                               name='Daily discharge',
                               hovertemplate='%{x|%Y-%m-%d}<br>%{y:.2f} ' + UNITS + '<extra></extra>'))

    if prov_by_day.any():
        y_prov = daily_full.where(prov_by_day.values).values
        fig.add_trace(go.Scattergl(x=daily_full.index,
                                   y=y_prov,
                                   mode='lines',
                                   line=dict(color='gold', width=1.5),
                                   name='Provisional',
                                   hovertemplate='%{x|%Y-%m-%d}<br>%{y:.2f} ' + UNITS + '<extra>Provisional</extra>'))

    if rev_by_day.any():
        y_rev = daily_full.where(rev_by_day.values).values
        fig.add_trace(go.Scattergl(x=daily_full.index,
                                   y=y_rev,
                                   mode='lines',
                                   line=dict(color='darkorange', width=1.5),
                                   name='Revised',
                                   hovertemplate='%{x|%Y-%m-%d}<br>%{y:.2f} ' + UNITS + '<extra>Revised</extra>'))

    if len(plain_missing_dates) > 0:
        fig.add_trace(go.Scattergl(x=plain_missing_dates,
                                   y=[tick_y] * len(plain_missing_dates),
                                   mode='markers',
                                   marker=dict(symbol='line-ns', size=16, color='crimson',
                                               line=dict(color='crimson', width=2)),
                                   name='Missing day',
                                   hovertemplate='%{x|%Y-%m-%d}<extra>Missing</extra>'))

    for code in sorted(qualifier_days_by_code):
        code_days = qualifier_days_by_code[code]
        color = QUALIFIER_COLORS.get(code, QUALIFIER_DEFAULT_COLOR)
        fig.add_trace(go.Scattergl(x=code_days,
                                   y=[tick_y] * len(code_days),
                                   mode='markers',
                                   marker=dict(symbol='line-ns', size=16, color=color,
                                               line=dict(color=color, width=2)),
                                   name=f'Missing: {code}',
                                   hovertemplate='%{x|%Y-%m-%d}<extra>Missing: ' + code + '</extra>'))

    fig.update_layout(height=450,
                      margin=dict(l=40, r=20, t=50, b=40),
                      title=(f'{gage.gage_id} daily streamflow  |  '
                             f'{por_start.date()} to {por_end.date()}  |  '
                             f'{n_missing} missing days ({pct_missing:.1f}%)'),
                      xaxis_title='Date',
                      yaxis_title=f'Discharge ({UNITS})',
                      legend=dict(x=0.01, y=0.99))

    return fig


# ---------------------------------------------------------------------------
# HTML rendering
# ---------------------------------------------------------------------------

def _df_to_html(df: pd.DataFrame,
                index: bool = False) -> str:
    """Render a DataFrame as an HTML table with sensible defaults.

    :param df: DataFrame to render
    :param index: whether to include the index column
    :returns: HTML table fragment
    """

    return df.to_html(index=index, border=0, na_rep='',
                      classes='report-table', justify='left')


def _qualifier_legend_html(color_map: Dict[str, str],
                           default_color: str) -> str:
    """Render the qualifier color map as an HTML swatch table.

    :param color_map: mapping of qualifier code to color
    :param default_color: fallback color for unlisted qualifiers
    :returns: HTML fragment for the legend
    """

    cells = []
    for code in sorted(color_map):
        color = color_map[code]
        cells.append(f'<div style="display:flex;align-items:center;margin:2px 14px 2px 0;">'
                     f'<span style="display:inline-block;width:26px;height:12px;'
                     f'background:{color};border:1px solid #888;margin-right:6px;"></span>'
                     f'<span style="font-family:monospace;">{html.escape(code)}</span></div>')

    cells.append(f'<div style="display:flex;align-items:center;margin:2px 14px 2px 0;">'
                 f'<span style="display:inline-block;width:26px;height:12px;'
                 f'background:{default_color};border:1px solid #888;margin-right:6px;"></span>'
                 f'<span style="font-family:monospace;">(other)</span></div>')

    return '<div style="display:flex;flex-wrap:wrap;">' + ''.join(cells) + '</div>'


def _meta_section_html(gage: _GageSeries) -> str:
    """Render the gage metadata (name, location, drainage area) as HTML.

    :param gage: per-gage series container
    :returns: HTML fragment (empty when no metadata is available)
    """

    meta = gage.meta
    if not meta:
        return ''

    def _fmt(key: str, fmt: str = '{}') -> Optional[str]:
        val = meta.get(key)
        if val is None:
            return None
        if isinstance(val, float) and np.isnan(val):
            return None
        try:
            return fmt.format(val)
        except (ValueError, TypeError):
            return str(val)

    rows = []
    name = _fmt('poi_name')
    if name:
        rows.append(f'<li>Name: <b>{html.escape(str(name))}</b></li>')

    agency = _fmt('poi_agency')
    if agency:
        rows.append(f'<li>Agency: <b>{html.escape(str(agency))}</b></li>')

    lat = _fmt('latitude', '{:.5f}')
    lon = _fmt('longitude', '{:.5f}')
    if lat and lon:
        rows.append(f'<li>Location: <b>{lat}, {lon}</b></li>')

    da = _fmt('drainage_area', '{:.1f}')
    if da:
        rows.append(f'<li>Drainage area: <b>{da}</b> mi&sup2;</li>')

    da_c = _fmt('drainage_area_contrib', '{:.1f}')
    if da_c:
        rows.append(f'<li>Contributing drainage area: <b>{da_c}</b> mi&sup2;</li>')

    if not rows:
        return ''

    return '<h2>Station Metadata</h2>\n<ul>\n' + '\n'.join(rows) + '\n</ul>'


def _render_report_html(gage: _GageSeries,
                        analysis: dict,
                        plot_html: str,
                        source_kind: str) -> str:
    """Assemble the full HTML report for a single gage.

    :param gage: per-gage series container
    :param analysis: output of :func:`_analyze`
    :param plot_html: embedded (dynamic) Plotly figure HTML
    :param source_kind: 'POI' or 'NWIS' (shown in the report metadata)
    :returns: a complete HTML document as a string
    """

    station = gage.gage_id
    df_gaps = analysis['df_gaps']
    by_year = analysis['by_year']

    has_flags = gage.approval is not None or gage.qualifier is not None

    # --- Gaps section -----------------------------------------------------
    if not df_gaps.empty:
        gaps_section = (f'<p>Number of distinct missing-data gaps: '
                        f'<b>{len(df_gaps)}</b>. Longest gap: '
                        f'<b>{int(df_gaps["days"].iloc[0])}</b> days. '
                        f'Showing the longest 25 gaps.</p>'
                        + _df_to_html(df_gaps.head(25)))
    else:
        gaps_section = '<p class="ok">No missing days in the period of record.</p>'

    meta_section = _meta_section_html(gage)

    if has_flags:
        plot_desc = ('<p>Daily discharge over the full period of record. '
                     'Provisional observations are overlaid in gold and Revised '
                     'observations in dark orange. Missing days are marked with '
                     'vertical ticks along the top; Provisional missing days '
                     'carrying a qualifier code use the colors below.</p>'
                     '<h3>Qualifier Color Legend</h3>'
                     + _qualifier_legend_html(QUALIFIER_COLORS, QUALIFIER_DEFAULT_COLOR))
    else:
        plot_desc = ('<p>Daily discharge over the full period of record. '
                     'Missing days are marked with vertical ticks along the top. '
                     '(Approval-status and qualifier flags are not available for '
                     'this streamflow source.)</p>')

    return f'''<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Streamgage {html.escape(station)} report</title>
<style>
  body {{ font-family: -apple-system, Segoe UI, Roboto, Helvetica, Arial, sans-serif;
          margin: 24px; color: #222; }}
  h1 {{ margin-bottom: 4px; }}
  h2 {{ margin-top: 32px; border-bottom: 2px solid #ddd; padding-bottom: 4px; }}
  .meta {{ color: #555; font-size: 0.9em; }}
  .warn {{ color: #b8860b; font-weight: 600; }}
  .ok {{ color: #2e7d32; font-weight: 600; }}
  table.report-table {{ border-collapse: collapse; margin: 8px 0; font-size: 0.9em; }}
  table.report-table th, table.report-table td {{
      border: 1px solid #ddd; padding: 4px 10px; text-align: left; }}
  table.report-table th {{ background: #f4f4f4; }}
  table.report-table tr:nth-child(even) td {{ background: #fafafa; }}
</style>
</head>
<body>
<h1>Streamgage {html.escape(station)}</h1>
<p class="meta">Streamflow source: <b>{html.escape(source_kind)}</b></p>

{meta_section}

<h2>Daily Series and Period of Record</h2>
<ul>
  <li>Period of record: <b>{analysis['por_start'].date()}</b> to <b>{analysis['por_end'].date()}</b></li>
  <li>Total days in period: <b>{analysis['n_total_days']}</b></li>
  <li>Days with data: <b>{analysis['n_present']}</b></li>
  <li>Missing days: <b>{analysis['n_missing']}</b> (<b>{analysis['pct_missing']:.2f}%</b>)</li>
</ul>

<h2>Missing-Record Gaps</h2>
{gaps_section}

<h2>Per-Year Coverage</h2>
<p>Years spanned: <b>{len(by_year)}</b>. Partial first and last years are expected.</p>
{_df_to_html(by_year)}

<h2>Streamflow Plot</h2>
{plot_desc}
{plot_html}
</body>
</html>
'''


# ---------------------------------------------------------------------------
# Index map (built directly from in-memory lat/lon metadata)
# ---------------------------------------------------------------------------

def _missing_pct_color(pct: Optional[float]) -> str:
    """Return the marker color for a station given its percent-missing.

    :param pct: percent of missing days, or None for a nodata station
    :returns: hex color string
    """

    if pct is None:
        return NODATA_COLOR
    for upper, color, _ in MISSING_PCT_BINS:
        if pct <= upper:
            return color
    return MISSING_PCT_BINS[-1][1]


def _map_legend_html() -> str:
    """Render the index-map color legend as an HTML fragment.

    :returns: HTML fragment describing the percent-missing color bins
    """

    rows = []
    for _, color, label in MISSING_PCT_BINS:
        rows.append(f'<div style="display:flex;align-items:center;margin:2px 0;">'
                    f'<span style="display:inline-block;width:16px;height:16px;border-radius:50%;'
                    f'background:{color};border:1px solid #555;margin-right:8px;"></span>'
                    f'<span>{label}</span></div>')

    rows.append(f'<div style="display:flex;align-items:center;margin:2px 0;">'
                f'<span style="display:inline-block;width:16px;height:16px;border-radius:50%;'
                f'background:{NODATA_COLOR};border:1px solid #555;margin-right:8px;"></span>'
                f'<span>{NODATA_LABEL}</span></div>')

    return ('<div style="font:13px -apple-system,Segoe UI,Roboto,Helvetica,Arial,sans-serif;">'
            '<div style="font-weight:600;margin-bottom:4px;">% missing days</div>'
            + ''.join(rows) + '</div>')


# Styling (Leaflet path options) for each optional context overlay. Keys are
# the Bandit subset layer names written by the shapefile/geopackage step.
_CONTEXT_LAYER_STYLE = {'domain':    {'color': '#d7301f', 'weight': 2,   'fillColor': '#d7301f',
                                      'fillOpacity': 0.05, 'label': 'Domain outline', 'default_on': True},
                        'nhru': {'color': '#7f7f7f', 'weight': 0.6, 'fillColor': '#7f7f7f',
                                 'fillOpacity': 0.08, 'label': 'HRUs', 'default_on': False},
                        'nsegment': {'color': '#1f6fb4', 'weight': 1.2, 'fillColor': '#1f6fb4',
                                     'fillOpacity': 0.0, 'label': 'Stream segments', 'default_on': True}}

# Order (bottom-to-top) in which context overlays are drawn / listed.
_CONTEXT_LAYER_ORDER = ['nhru', 'domain', 'nsegment']


def _context_layer_path(gis_dir: Path, gis_type: str, layer: str) -> Optional[Path]:
    """Resolve the on-disk path for a subset GIS layer.

    Mirrors the naming used by ``Bandit.bandit_helpers.write_gis``: shapefiles
    are written one-per-layer as ``model_<layer>.shp`` while any other format
    (e.g. geopackage) is a single ``model_layers.<ext>`` holding named layers.

    :param gis_dir: directory the subset GIS layers were written to
    :param gis_type: destination GIS extension (e.g. ``shp`` or ``gpkg``)
    :param layer: subset layer name (``nhru`` / ``nsegment`` / ``domain``)
    :returns: path to the file holding the layer, or None if it does not exist
    """

    if gis_type == 'shp':
        candidate = gis_dir / f'model_{layer}.{gis_type}'
    else:
        candidate = gis_dir / f'model_layers.{gis_type}'
    return candidate if candidate.exists() else None


def _reproject_geographic(gdf: Any) -> Any:
    """Reproject a GeoDataFrame to geographic lon/lat degrees for Leaflet.

    Leaflet expects lon/lat coordinates. WGS84 (EPSG:4326) is preferred, but
    some PROJ builds fail the NAD83 datum-ensemble transform for projected
    source CRSs (e.g. CONUS Albers / EPSG:5070), yielding non-finite
    coordinates. When that happens this falls back to NAD83 geographic
    (EPSG:4269), which differs from WGS84 by well under a meter and is
    indistinguishable at web-map scale.

    :param gdf: a GeoDataFrame (with a defined CRS) to reproject
    :returns: the reprojected GeoDataFrame (unchanged if already geographic
        lon/lat); may still contain non-finite coordinates if every attempt
        fails
    """

    import numpy as _np

    if gdf.crs is None:
        return gdf

    # Already geographic lon/lat (4326 or 4269) => nothing to do.
    epsg = gdf.crs.to_epsg()
    if epsg in (4326, 4269):
        return gdf

    out = gdf.to_crs(4326)
    if _np.isfinite(out.total_bounds).all():
        return out

    # WGS84 transform produced non-finite coordinates; fall back to NAD83.
    logger.info('WGS84 reprojection produced non-finite coordinates; '
                'falling back to NAD83 (EPSG:4269) for the index map')
    return gdf.to_crs(4269)


def _load_context_layers(gis_dir: Optional[Path],
                         gis_type: Optional[str],
                         simplify_tolerance_m: float = 100.0,
                         con: Any = None) -> List[Tuple[str, str, dict]]:
    """Read subset GIS layers and return them as WGS84 GeoJSON for the map.

    Reads the ``nhru``, ``nsegment`` and ``domain`` subset layers (whichever
    exist), reprojects them to WGS84, lightly simplifies polygon/line geometry
    to keep the embedded GeoJSON small, and returns one entry per layer.

    :param gis_dir: directory holding the subset GIS layers (None disables
        context overlays)
    :param gis_type: destination GIS extension (e.g. ``shp`` or ``gpkg``)
    :param simplify_tolerance_m: simplification tolerance in meters (applied in
        the layer's projected CRS); set to 0 to disable simplification
    :param con: optional rich Console for warnings
    :returns: list of (layer_name, geojson_str, style_dict) tuples in draw
        order; empty when no layers could be read
    """

    if gis_dir is None or gis_type is None:
        return []

    try:
        import pyogrio   # type: ignore
    except ImportError:
        return []

    layers: List[Tuple[str, str, dict]] = []

    for layer in _CONTEXT_LAYER_ORDER:
        path = _context_layer_path(gis_dir, gis_type, layer)
        if path is None:
            continue

        try:
            # read_dataframe returns a GeoDataFrame (geopandas); the geopackage
            # case needs the layer name, the shapefile case does not.
            if gis_type == 'shp':
                gdf = pyogrio.read_dataframe(path)
            else:
                gdf = pyogrio.read_dataframe(path, layer=layer)
        except Exception as err:   # noqa: BLE001 - context layers are best-effort
            logger.warning(f'Could not read context layer {layer!r} from {path}: {err}')
            if con is not None:
                con.print(f'[gold3]WARNING[/]: Could not read context layer {layer!r}: {err}')
            continue

        if gdf is None or len(gdf) == 0:
            continue

        # Keep only geometry to minimize embedded size.
        gdf = gdf[['geometry']].copy()

        # Simplify in the projected source CRS (tolerance is in meters there),
        # then reproject to geographic lon/lat for Leaflet.
        try:
            if (simplify_tolerance_m and gdf.crs is not None and gdf.crs.is_projected):
                gdf['geometry'] = gdf.geometry.simplify(simplify_tolerance_m,
                                                        preserve_topology=True)

            gdf = _reproject_geographic(gdf)
            geojson = gdf.to_json()
        except Exception as err:   # noqa: BLE001
            logger.warning(f'Could not process context layer {layer!r}: {err}')
            if con is not None:
                con.print(f'[gold3]WARNING[/]: Could not process context layer {layer!r}: {err}')
            continue

        style = _CONTEXT_LAYER_STYLE.get(layer, {'color': '#888', 'weight': 1,
                                                 'fillColor': '#888', 'fillOpacity': 0.05,
                                                 'label': layer, 'default_on': False})
        layers.append((layer, geojson, style))

    return layers


def _build_index_map(features: List[dict],
                     outdir: Path,
                     context_layers: Optional[List[Tuple[str, str, dict]]] = None) -> Optional[Path]:
    """Build a self-contained Leaflet index map and write it to ``index.html``.

    :param features: list of GeoJSON-style feature dicts (one per gage with a
        valid location)
    :param outdir: directory the reports (and index.html) live in
    :param context_layers: optional list of (layer_name, geojson_str, style)
        tuples drawn as selectable overlays (nhru / nsegment / domain)
    :returns: path to the written ``index.html``, or None when no mappable
        features exist
    """

    import json

    if not features:
        return None

    context_layers = context_layers or []

    lats = [f['geometry']['coordinates'][1] for f in features]
    lons = [f['geometry']['coordinates'][0] for f in features]
    center_lat = sum(lats) / len(lats)
    center_lon = sum(lons) / len(lons)

    poi_geojson = json.dumps({'type': 'FeatureCollection', 'features': features})
    legend_html = _map_legend_html()
    n_stations = len(features)
    n_nodata = sum(1 for f in features if f['properties']['color'] == NODATA_COLOR)
    n_missing_report = sum(1 for f in features if not f['properties']['href'])

    # Build the JS that creates each context overlay (nhru / nsegment / domain)
    # as its own toggleable Leaflet layer, and the overlay entries for the
    # layer control. Layers flagged default_on are added to the map initially.
    context_js_parts: List[str] = []
    overlay_entries: List[str] = []
    default_on_adds: List[str] = []

    for idx, (layer, geojson, style) in enumerate(context_layers):
        var = f'ctxLayer{idx}'
        path_style = {'color':       style.get('color', '#888'),
                      'weight':      style.get('weight', 1),
                      'fillColor':   style.get('fillColor', style.get('color', '#888')),
                      'fillOpacity': style.get('fillOpacity', 0.05)}
        label = style.get('label', layer)

        context_js_parts.append(f'  var {var}Data = {geojson};\n'
                                f'  var {var} = L.geoJSON({var}Data, {{ style: {json.dumps(path_style)}, '
                                f'interactive: false }});')
        overlay_entries.append(f'{json.dumps(label)}: {var}')

        if style.get('default_on'):
            default_on_adds.append(f'  {var}.addTo(map);')

    context_js = '\n'.join(context_js_parts)
    default_on_js = '\n'.join(default_on_adds)
    overlays_js = '{' + ', '.join(overlay_entries) + '}'

    html_doc = f'''<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Streamgage index map</title>
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css"
      integrity="sha256-p4NxAoJBhIIN+hmNHrzRCf9tD/miZyoHS5obTRR9BMY=" crossorigin=""/>
<script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"
        integrity="sha256-20nQCchB9co0qIjJZRGuk2/Z9VM+kNiyxNV1lvTlZBo=" crossorigin=""></script>
<style>
  html, body {{ height: 100%; margin: 0; }}
  #map {{ height: 100%; width: 100%; }}
  .info-box {{ font: 13px -apple-system, Segoe UI, Roboto, Helvetica, Arial, sans-serif;
               background: rgba(255,255,255,0.9); padding: 8px 10px; border-radius: 6px;
               box-shadow: 0 1px 4px rgba(0,0,0,0.3); max-width: 260px; }}
  .legend {{ line-height: 18px; color: #333; }}
  .title-box {{ font-weight: 600; }}
</style>
</head>
<body>
<div id="map"></div>
<script>
  var poiData = {poi_geojson};

  var map = L.map('map').setView([{center_lat}, {center_lon}], 9);

  var esriTopo = L.tileLayer(
      'https://server.arcgisonline.com/ArcGIS/rest/services/World_Topo_Map/MapServer/tile/{{z}}/{{y}}/{{x}}', {{
      maxZoom: 19,
      attribution: 'Tiles &copy; Esri'
  }});
  var esriImagery = L.tileLayer(
      'https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{{z}}/{{y}}/{{x}}', {{
      maxZoom: 19,
      attribution: 'Tiles &copy; Esri'
  }});

  esriTopo.addTo(map);

  // Optional context overlays (nhru / nsegment / domain) from the subset GIS.
{context_js}

  // Add default-on context overlays beneath the POI markers.
{default_on_js}

  L.control.layers({{'Esri Topographic': esriTopo,
                     'Esri Imagery': esriImagery}},
                   {overlays_js},
                   {{ position: 'topleft', collapsed: false }}).addTo(map);

  var poiLayer = L.geoJSON(poiData, {{
      pointToLayer: function(feature, latlng) {{
          var p = feature.properties;
          var marker = L.circleMarker(latlng, {{
              radius: 7,
              color: '#333',
              weight: 1,
              fillColor: p.color,
              fillOpacity: 0.9
          }});

          var popup = '<b>' + p.station + '</b><br>Missing: ' + p.pct_text;
          if (p.name) {{ popup = '<b>' + p.station + '</b><br>' + p.name + '<br>Missing: ' + p.pct_text; }}
          if (p.href) {{
              popup += '<br><a href="' + p.href + '">Open report</a>';
              marker.on('click', function() {{ window.location.href = p.href; }});
              marker.on('mouseover', function() {{ this.setStyle({{ weight: 3 }}); }});
              marker.on('mouseout', function() {{ this.setStyle({{ weight: 1 }}); }});
          }}
          marker.bindTooltip(p.station + ' (' + p.pct_text + ')');
          marker.bindPopup(popup);
          return marker;
      }}
  }}).addTo(map);

  map.fitBounds(poiLayer.getBounds().pad(0.1));

  var legend = L.control({{ position: 'bottomright' }});
  legend.onAdd = function() {{
      var div = L.DomUtil.create('div', 'info-box legend');
      div.innerHTML = {json.dumps(legend_html)};
      return div;
  }};
  legend.addTo(map);

  var info = L.control({{ position: 'topright' }});
  info.onAdd = function() {{
      var div = L.DomUtil.create('div', 'info-box');
      div.innerHTML = '<div class="title-box">Streamgage index</div>' +
                      'Stations shown: {n_stations}<br>' +
                      'No data: {n_nodata}<br>' +
                      'Without report: {n_missing_report}';
      return div;
  }};
  info.addTo(map);
</script>
</body>
</html>
'''

    outfile = outdir / 'index.html'
    outfile.write_text(html_doc, encoding='utf-8')
    return outfile


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def generate_report(streamflow: Any,
                    gage_ids: List[str],
                    outdir: Union[str, Path],
                    st_date: Optional[datetime.datetime] = None,
                    en_date: Optional[datetime.datetime] = None,
                    offline: bool = False,
                    gis_dir: Optional[Union[str, Path]] = None,
                    gis_type: Optional[str] = None,
                    con: Any = None,
                    verbose: bool = False) -> Dict[str, Any]:
    """Generate per-streamgage HTML reports (and an index map when possible).

    :param streamflow: the in-memory POI or NWIS streamflow object
    :param gage_ids: ordered list of gage ids in the model subset
    :param outdir: directory to write the HTML reports into (created if needed)
    :param st_date: optional inclusive start date; the report is clipped to
        this bound so it reflects the extraction's date range rather than the
        full period of record in the source files
    :param en_date: optional inclusive end date (see ``st_date``)
    :param offline: embed the full plotly.js library in each HTML file so the
        plots render without internet access (larger files). By default
        plotly.js is loaded from a CDN.
    :param gis_dir: optional directory holding the subset GIS layers; when
        provided (together with ``gis_type``) the index map gains selectable
        nhru / nsegment / domain overlays
    :param gis_type: destination GIS extension of the subset layers (e.g.
        ``shp`` or ``gpkg``)
    :param con: optional rich Console for progress / status messages
    :param verbose: emit additional per-gage messages
    :returns: a summary dict with keys ``written``, ``failed``, ``index`` and
        ``outdir``
    :raises ReportError: if plotly is unavailable or the streamflow object
        cannot be interpreted
    """

    # Fail fast with a clear message if plotly is missing.
    _require_plotly()

    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    include_plotlyjs: Union[bool, str] = True if offline else 'cdn'

    series_list, source_kind = _normalize_streamflow(streamflow, gage_ids, st_date, en_date)

    def _msg(text: str):
        if con is not None:
            con.print(text)

    written: List[Path] = []
    failed: List[str] = []
    map_features: List[dict] = []

    for gage in series_list:
        station = gage.gage_id
        try:
            analysis = _analyze(gage)
            fig = _build_figure(gage, analysis)
            plot_html = fig.to_html(full_html=False, include_plotlyjs=include_plotlyjs)
            report_html = _render_report_html(gage, analysis, plot_html, source_kind)

            outfile = outdir / f'{station}.html'
            outfile.write_text(report_html, encoding='utf-8')
            written.append(outfile)

            pct = float(analysis['pct_missing'])
            has_report = True
            if verbose:
                _msg(f'  [green4]report[/] {station}: {pct:.1f}% missing')
        except (ValueError, KeyError) as err:
            failed.append(station)
            pct = None   # type: ignore
            has_report = False
            logger.info(f'Streamgage {station}: no report ({err})')
            if verbose:
                _msg(f'  [orange3]no data[/] {station}: {err}')
        except Exception as err:   # noqa: BLE001
            failed.append(station)
            pct = None   # type: ignore
            has_report = False
            logger.warning(f'Streamgage {station}: report failed ({err})')
            _msg(f'  [red]failed[/] {station}: {err}')

        # Collect a map feature when we have a usable location.
        lat = gage.meta.get('latitude')
        lon = gage.meta.get('longitude')
        if (lat is not None and lon is not None
                and not (isinstance(lat, float) and np.isnan(lat))
                and not (isinstance(lon, float) and np.isnan(lon))):
            map_features.append({'type': 'Feature',
                                 'geometry': {'type': 'Point',
                                              'coordinates': [float(lon), float(lat)]},
                                 'properties': {'station':  station,
                                                'name':     str(gage.meta.get('poi_name', '')),
                                                'pct_text': 'no data' if pct is None else f'{pct:.2f}%',
                                                'color':    _missing_pct_color(pct),
                                                'href':     f'{station}.html' if has_report else ''}})

    # Load optional context overlays (nhru / nsegment / domain) from the
    # subset GIS layers so the index map can toggle them.
    context_layers = _load_context_layers(Path(gis_dir) if gis_dir else None,
                                          gis_type, con=con)
    if context_layers and verbose:
        _msg(f'[green4]INFO[/]: Index map context layers: '
             f'{", ".join(name for name, _, _ in context_layers)}')

    index_file = _build_index_map(map_features, outdir, context_layers)

    return {'written': written,
            'failed':  failed,
            'index':   index_file,
            'outdir':  outdir}
