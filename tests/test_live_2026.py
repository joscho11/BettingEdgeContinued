"""2026 live Tuesday-model display rules and slate merge.

2025 predictions_tracker.csv stays byte-identical. 2026 matchups live in slate_2026.csv.
"""
import hashlib
import sys
from pathlib import Path

import pandas as pd

_HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_HERE))
sys.path.insert(0, str(_HERE / "betting"))

from live_2026 import (  # noqa: E402
    LIVE_HIGH_N,
    LIVE_HIGH_WILSON_LOWER,
    LIVE_HIGH_WILSON_Z,
    LIVE_HIGH_WINS,
    TRACKER_2025_MD5,
    attach_slate,
    display_high,
    high_dropped,
    is_finale_week,
    is_live_season,
    leftover_to_home_margin,
    row_display_high,
    row_high_dropped,
    sportsbook_to_nflverse,
    tracker_2025_payload,
    tuesday_high,
)

_TRACKER = _HERE / "betting" / "predictions_tracker.csv"
_SLATE = _HERE / "betting" / "slate_2026.csv"


def _wilson_one_sided_lower(wins: int, n: int, z: float) -> float:
    p = wins / n
    z2 = z * z
    denom = 1.0 + z2 / n
    center = (p + z2 / (2 * n)) / denom
    margin = (z * ((p * (1 - p) + z2 / (4 * n)) / n) ** 0.5) / denom
    return center - margin


def test_2025_tracker_byte_identical():
    digest = hashlib.md5(tracker_2025_payload(_TRACKER)).hexdigest()
    assert digest == TRACKER_2025_MD5, digest
    if b",2026," not in _TRACKER.read_bytes():
        assert hashlib.md5(_TRACKER.read_bytes()).hexdigest() == TRACKER_2025_MD5


def test_slate_is_matchups_only():
    df = pd.read_csv(_SLATE)
    assert len(df) == 272
    assert set(df["week"].astype(int)) == set(range(1, 19))
    assert df["predicted_margin"].isna().all()
    assert df["ens_predicted_margin"].isna().all()
    assert df["spread_line"].isna().all()
    assert df["tuesday_spread_line"].isna().all()
    assert (df["mode"] == "matchup").all()
    w1 = df[df["week"] == 1]
    assert (w1["away_team"] == "NE").any()
    assert (w1["home_team"] == "SEA").any()
    assert "2026-09-09" in set(w1["gameday"].astype(str))


def test_attach_slate_does_not_duplicate_tracker_ids():
    tracker = pd.read_csv(_TRACKER)
    out = attach_slate(tracker, _HERE)
    assert out["game_id"].is_unique
    assert len(out) == len(tracker) + 272
    assert set(tracker["game_id"]).issubset(set(out["game_id"]))
    assert (out["season"] == 2025).sum() == len(tracker)
    assert (out["season"] == 2026).sum() == 272


def test_attach_slate_skips_already_logged_2026_row(tmp_path):
    tracker = pd.DataFrame(
        {"game_id": ["2026_01_NE_SEA"], "season": [2026], "week": [1]}
    )
    out = attach_slate(tracker, _HERE)
    assert (out["game_id"] == "2026_01_NE_SEA").sum() == 1
    assert len(out) == 272


def test_display_high_demote_only():
    assert tuesday_high(10, 7)
    assert display_high(10, 7, None)
    assert display_high(10, 7, 6.5)
    assert not display_high(10, 7, 8)
    assert high_dropped(10, 7, 8)
    assert not high_dropped(10, 7, None)
    assert not display_high(10, 8, 6)
    assert not high_dropped(10, 8, 6)


def test_finale_week_never_high():
    assert is_finale_week(2026, 18, "REG")
    assert not display_high(12, 7, None, season=2026, week=18)
    assert not is_finale_week(2026, 18, "POST")


def test_row_helpers_and_live_season():
    assert is_live_season(2026)
    assert not is_live_season(2025)
    row = pd.Series(
        {
            "ens_predicted_margin": 10.0,
            "tuesday_spread_line": 7.0,
            "live_spread_line": 8.5,
            "season": 2026,
            "week": 1,
            "game_type": "REG",
        }
    )
    assert not row_display_high(row)
    assert row_high_dropped(row)


def test_one_sided_wilson_claim_matches_locked_book():
    assert LIVE_HIGH_WINS == 155
    assert LIVE_HIGH_N == 290
    lo = _wilson_one_sided_lower(LIVE_HIGH_WINS, LIVE_HIGH_N, LIVE_HIGH_WILSON_Z)
    assert round(lo, 4) == LIVE_HIGH_WILSON_LOWER
    assert lo < 0.524


def test_leftover_converts_to_site_home_margin():
    leftover = 4.0
    sportsbook = -7.0
    home = leftover_to_home_margin(leftover, sportsbook)
    nflverse = sportsbook_to_nflverse(sportsbook)
    assert home == 11.0
    assert nflverse == 7.0
    assert abs(home - nflverse) == abs(leftover)
    assert tuesday_high(home, nflverse)
