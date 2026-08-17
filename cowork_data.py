"""cowork_OS workspace data roots for JoSchoAnalytics local pipelines.

Streamlit pages keep reading the in-repo hardlinks. This module is for scripts
that should write the canonical workspace copy.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path


def _load_ws():
    for parent in Path(__file__).resolve().parents:
        cand = parent / "workspace" / "paths.py"
        if cand.exists():
            spec = importlib.util.spec_from_file_location("cowork_workspace_paths", cand)
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            return mod
    raise FileNotFoundError("workspace/paths.py not found")


WS = _load_ws()
FUTURES_DATA = WS.FUTURES_RAW
BETTING_LINES_XLSX = WS.NFL_RAW / "betting_lines" / "nfl.xlsx"
SPREAD_MODELS = WS.NFL_MODELS / "spread"
SPREAD_TRACKER = WS.GAME_SPREAD_TRACKER
TOTALS_TRACKER = WS.GAME_TOTALS_TRACKER
