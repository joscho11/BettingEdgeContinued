"""2026 live Tuesday-model display rules. Not the 2025 3-voter demo.

HIGH is |predicted home margin - Tuesday spread| >= 3, last REG week skipped.
A later line can drop HIGH. It cannot create HIGH. No MEDIUM tier.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

LIVE_SEASON = 2026
HIGH_GAP = 3.0
LAST_REG_WEEK = 18
SLATE_NAME = "slate_2026.csv"

# Walk-forward skip-HIGH book vs the Tuesday line, 2021-2025. One-sided 95% Wilson.
LIVE_HIGH_WINS = 201
LIVE_HIGH_N = 349
LIVE_HIGH_ATS = LIVE_HIGH_WINS / LIVE_HIGH_N
LIVE_HIGH_WILSON_Z = 1.64485
LIVE_HIGH_WILSON_LOWER = 0.5320
BREAKEVEN = 0.524
TRACKER_2025_MD5 = "88d526ca46e8cbb9f1eea77a3d96fa08"


def is_live_season(season) -> bool:
    return int(season) == LIVE_SEASON


def is_finale_week(season, week, game_type: str = "REG") -> bool:
    if str(game_type) != "REG":
        return False
    last = 17 if int(season) <= 2020 else LAST_REG_WEEK
    return int(week) == last


def tuesday_high(pred, tuesday_spread) -> bool:
    if pred is None or tuesday_spread is None:
        return False
    if pd.isna(pred) or pd.isna(tuesday_spread):
        return False
    return abs(float(pred) - float(tuesday_spread)) >= HIGH_GAP


def display_high(pred, tuesday_spread, live_spread=None, *, season=LIVE_SEASON, week=1, game_type="REG") -> bool:
    """True only if the Tuesday ticket was HIGH and the live line still is."""
    if is_finale_week(season, week, game_type):
        return False
    if not tuesday_high(pred, tuesday_spread):
        return False
    line = tuesday_spread if live_spread is None or pd.isna(live_spread) else live_spread
    return abs(float(pred) - float(line)) >= HIGH_GAP


def high_dropped(pred, tuesday_spread, live_spread, *, season=LIVE_SEASON, week=1, game_type="REG") -> bool:
    """Tuesday ticket was HIGH; live line shrank it under 3."""
    if is_finale_week(season, week, game_type):
        return False
    if not tuesday_high(pred, tuesday_spread):
        return False
    if live_spread is None or pd.isna(live_spread):
        return False
    return abs(float(pred) - float(live_spread)) < HIGH_GAP


def has_pick(row) -> bool:
    pred = row_pred(row)
    return pred is not None and not pd.isna(pred)


def _num(val):
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return None
    try:
        if pd.isna(val):
            return None
    except (TypeError, ValueError):
        return None
    return float(val)


def row_pred(row):
    pred = row.get("ens_predicted_margin")
    if pred is None or pd.isna(pred):
        pred = row.get("predicted_margin")
    return pred


def row_tuesday_spread(row):
    tue = _num(row.get("tuesday_spread_line"))
    if tue is not None:
        return tue
    return _num(row.get("spread_line"))


def row_live_spread(row):
    return _num(row.get("live_spread_line"))


def row_game_type(row) -> str:
    gt = row.get("game_type")
    if gt is None or pd.isna(gt) or str(gt) == "nan":
        return "REG"
    return str(gt)


def row_display_high(row) -> bool:
    return display_high(
        row_pred(row),
        row_tuesday_spread(row),
        row_live_spread(row),
        season=int(row.get("season", LIVE_SEASON)),
        week=int(row.get("week", 1)),
        game_type=row_game_type(row),
    )


def row_high_dropped(row) -> bool:
    return high_dropped(
        row_pred(row),
        row_tuesday_spread(row),
        row_live_spread(row),
        season=int(row.get("season", LIVE_SEASON)),
        week=int(row.get("week", 1)),
        game_type=row_game_type(row),
    )


def attach_slate(tracker: pd.DataFrame, base_dir) -> pd.DataFrame:
    """Keep tracker rows. Add 2026 matchups whose game_id is not already logged."""
    path = Path(base_dir) / "betting" / SLATE_NAME
    if not path.is_file():
        return tracker
    slate = pd.read_csv(path)
    if slate.empty:
        return tracker
    if "season" in slate.columns:
        slate["season"] = slate["season"].astype(int)
    if "week" in slate.columns:
        slate["week"] = slate["week"].astype(int)
    if tracker is None or tracker.empty:
        return slate
    extra = slate.loc[~slate["game_id"].isin(tracker["game_id"])]
    if extra.empty:
        return tracker
    return pd.concat([tracker, extra], ignore_index=True, sort=False)
