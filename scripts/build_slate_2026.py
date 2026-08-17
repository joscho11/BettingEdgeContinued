"""Write 2026 REG matchups for the site. No picks. No nflverse spread as our line."""
from __future__ import annotations

import sys
from pathlib import Path

import nflreadpy as nfl
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT / "betting" / "slate_2026.csv"
TRACKER_COLS = [
    "game_id",
    "home_team",
    "away_team",
    "gameday",
    "spread_line",
    "predicted_margin",
    "model_edge",
    "recommendation",
    "season",
    "week",
    "logged_at",
    "actual_margin",
    "home_covered",
    "model_correct",
    "mode",
    "home_score",
    "away_score",
    "ens_model_edge",
    "ens_recommendation",
    "consensus_tier",
    "ens_model_correct",
    "ens_predicted_margin",
    "ridge_predicted_margin",
    "ridge_model_edge",
    "ridge_recommendation",
    "ridge_model_correct",
    "lgbm_predicted_margin",
    "lgbm_model_edge",
    "lgbm_recommendation",
    "lgbm_model_correct",
    "pick_line",
    "closing_line",
    "clv",
    "gametime",
    "game_type",
    "live_spread_line",
    "tuesday_spread_line",
]


def build() -> pd.DataFrame:
    sched = nfl.load_schedules([2026]).to_pandas()
    reg = sched.loc[sched["game_type"].eq("REG")].copy()
    if len(reg) != 272:
        raise RuntimeError(f"2026 REG slate is {len(reg)} games, want 272")
    rows = []
    for _, g in reg.iterrows():
        row = {c: None for c in TRACKER_COLS}
        row["game_id"] = g["game_id"]
        row["home_team"] = g["home_team"]
        row["away_team"] = g["away_team"]
        row["gameday"] = str(g["gameday"])[:10]
        row["gametime"] = g["gametime"]
        row["season"] = 2026
        row["week"] = int(g["week"])
        row["game_type"] = "REG"
        row["mode"] = "matchup"
        rows.append(row)
    out = pd.DataFrame(rows, columns=TRACKER_COLS)
    out = out.sort_values(["week", "gameday", "gametime", "game_id"]).reset_index(drop=True)
    if out["predicted_margin"].notna().any() or out["spread_line"].notna().any():
        raise RuntimeError("slate must not carry picks or lines")
    return out


def main() -> int:
    out = build()
    DEST.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(DEST, index=False)
    print(f"wrote {DEST} n={len(out)} weeks={int(out.week.min())}-{int(out.week.max())}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
