"""Shared app-level helpers used by the extracted page modules (site revamp Batch 3).

Moved byte-identical from app.py. `_season_week_controls` takes `df` as a parameter
(app.py closed over the module-level df); each page binds its own df via a thin closure
so the extracted tab bodies stay byte-identical. app.py keeps its own inline copies until
the swap (3e) removes the tab layer — temporary duplication by design.
"""
import importlib
import json
import os
from pathlib import Path

import streamlit as st

from dashboard_utils import sanitize_agent_analysis

_HERE = Path(__file__).resolve().parents[1]


def load_agent_analysis(week: int, season: int) -> dict:
    """Agent cache for one week, with unprovenanced market claims stripped.

    Every public read of an agent artifact goes through here, so the provenance gate
    cannot be bypassed by adding a new caller. Sanitisation is FAIL-CLOSED: an artifact
    without verifiable market provenance loses its Sharp Money / Line Movement lines
    entirely — they are removed, never replaced with a placeholder.
    """
    cache_file = str(_HERE / "betting" / f"agent_analysis_{season}_week{week}.json")
    if os.path.exists(cache_file):
        try:
            with open(cache_file, 'r', encoding='utf-8') as f:
                payload = json.load(f)
        except (json.JSONDecodeError, OSError, UnicodeDecodeError):
            return None
        clean, _report = sanitize_agent_analysis(payload)
        return clean
    return None


def _season_week_controls(df, cols_container, key_prefix, with_week=True, with_edge=False, default_week=None):
    """Render a tab's own Season (+ optional Week, Min-edge) controls in its body.
    Returns (season, week, edge_threshold). df-based options preserve prior behavior."""
    _seasons = sorted(df['season'].unique(), reverse=True)
    _season = cols_container[0].selectbox("Season", _seasons, key=f"{key_prefix}_season")
    _week = None
    _edge = 0.0
    _i = 1
    if with_week:
        _weeks = sorted(df[df['season'] == _season]['week'].unique(), reverse=True)
        if default_week is not None and default_week in set(_weeks):
            _dwk = _weeks.index(default_week)
        else:
            _dwk = next((i for i, w in enumerate(_weeks) if w == 10), 0)
        _week = cols_container[_i].selectbox("Week", _weeks, index=_dwk,
                                             key=f"{key_prefix}_week")
        _i += 1
    if with_edge:
        _edge = cols_container[_i].slider(
            "Min Edge (pts)", min_value=0.0, max_value=5.0, value=0.0, step=0.5,
            key=f"{key_prefix}_edge",
            help="Only show games where model disagrees with spread by at least this many points")
    return _season, _week, _edge


_MODE_BADGE_COLORS = {
    'monday':   '#ffd600',
    'thursday': '#ff9800',
    'sunday':   '#00c853',
    'backfill': '#3D95CE',
    'matchup':  '#888888',
}

# ATS blurb — moved byte-identical from the retired sidebar onto the Betting pages.
ATS_BLURB = """
    <div style="padding: 2px 4px 6px 4px;">
        <p style="font-size:12px;color:#aaa;line-height:1.65;margin:0">
            ML model trained on NFL data since 2014. Predicts each game vs the Vegas spread.
            <b style="color:#3D95CE">52.4% ATS</b> is break-even.
            2026 uses the Tuesday HIGH book. 2025 weeks on this site are a demo.
        </p>
    </div>
    """


def reload_if_stale(module):
    """Reload a module when Streamlit Cloud copies a new file into a live process.

    ``_lazy_render`` already does this for the selected page. Sibling helpers
    imported from that page (``league_insights_view``, ``fantasy.league_intelligence``)
    stay pinned in ``sys.modules`` unless they get the same mtime check.
    """
    path = getattr(module, "__file__", None)
    if not path:
        return module
    try:
        mtime = Path(path).stat().st_mtime_ns
    except OSError:
        return module
    if getattr(module, "__joscho_source_mtime_ns__", None) != mtime:
        importlib.invalidate_caches()
        source = Path(path)
        cache_dir = source.parent / "__pycache__"
        if cache_dir.is_dir():
            for pyc in cache_dir.glob(f"{source.stem}*.pyc"):
                try:
                    pyc.unlink()
                except OSError:
                    pass
        module = importlib.reload(module)
    module.__joscho_source_mtime_ns__ = mtime
    return module


_PLOTLY_TOUCH = {"displayModeBar": False, "scrollZoom": False}


def unlabeled_scatter_copy(fig):
    """Keep hover text, drop on-chart labels. Phone copy of a labeled scatter."""
    import plotly.graph_objects as go

    phone = go.Figure(fig.to_dict())
    for trace in phone.data:
        mode = str(getattr(trace, "mode", None) or "")
        if "text" not in mode.split("+"):
            continue
        kept = [part for part in mode.split("+") if part != "text"]
        trace.mode = "+".join(kept) if kept else "markers"
    return phone


def plotly_phone_desktop(desktop_fig, phone_fig, *, slug: str) -> None:
    """Show desktop_fig above 640px and phone_fig at phone width.

    A phone figure with an explicit layout.width keeps that size (no Plotly
    autoshrink). Streamlit still stretch-wraps so the page itself does not
    grow; CSS on the keyed container is what pans sideways.
    """
    phone_fixed = phone_fig.layout.width not in (None, 0)
    phone_cfg = {**_PLOTLY_TOUCH, "responsive": False} if phone_fixed else _PLOTLY_TOUCH
    with st.container(key=f"jsa-scatter-desktop-{slug}"):
        st.plotly_chart(desktop_fig, width="stretch", config=_PLOTLY_TOUCH)
    with st.container(key=f"jsa-scatter-phone-{slug}"):
        st.plotly_chart(phone_fig, width="stretch", config=phone_cfg)


def plotly_labeled_scatter(fig, *, slug: str) -> None:
    """Desktop keeps names. Phone is dots only. CSS shows one copy."""
    plotly_phone_desktop(fig, unlabeled_scatter_copy(fig), slug=slug)
