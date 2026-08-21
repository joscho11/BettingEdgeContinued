"""Freeze the 2025 W10-W16 injury context consumed by matchup demo pages.

The deploy cannot read cowork_OS/workspace. This script selects only final weekly
game-status rows from the local nflverse cache and writes a small, deterministic
website artifact. Run it again only when intentionally refreshing the demo source.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

STATUS_ORDER = {"Out": 0, "Doubtful": 1, "Questionable": 2}


def build(source: Path) -> dict:
    frame = pd.read_parquet(source)
    selected = frame[
        pd.to_numeric(frame["season"], errors="coerce").eq(2025)
        & pd.to_numeric(frame["week"], errors="coerce").between(10, 16)
        & frame["report_status"].isin(STATUS_ORDER)
    ].copy()
    selected["season"] = selected["season"].astype(int)
    selected["week"] = selected["week"].astype(int)
    selected = selected.sort_values(
        ["season", "week", "team", "report_status", "position", "full_name"],
        key=lambda values: values.map(STATUS_ORDER) if values.name == "report_status" else values,
    )
    teams = {}
    for (season, week, team), group in selected.groupby(["season", "week", "team"], sort=True):
        players = []
        for row in group.drop_duplicates("gsis_id", keep="last").itertuples():
            players.append(
                {
                    "player": str(row.full_name),
                    "position": str(row.position),
                    "injury": str(row.report_primary_injury or row.practice_primary_injury or "Not listed"),
                    "status": str(row.report_status),
                    "practice_status": None if pd.isna(row.practice_status) else str(row.practice_status),
                }
            )
        counts = {status: sum(player["status"] == status for player in players) for status in STATUS_ORDER}
        teams[f"{int(season)}_{int(week):02d}_{team}"] = {
            "counts": counts,
            "players": players,
        }
    return {
        "schema_version": 1,
        "season": 2025,
        "weeks": list(range(10, 17)),
        "source": "nflverse weekly injury reports",
        "source_url": "https://github.com/nflverse/nflverse-data/releases/tag/injuries",
        "timing": "Archived final weekly report; snapshot time unavailable",
        "teams": teams,
    }


def main(argv=None) -> int:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source",
        type=Path,
        default=root.parent / "workspace" / "nfl" / "raw" / "nflverse" / "weekly_cache" / "injuries.parquet",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=root / "data" / "matchups" / "injuries_2025_weeks10_16.json",
    )
    args = parser.parse_args(argv)
    payload = build(args.source)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"wrote {len(payload['teams'])} team-week reports to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
