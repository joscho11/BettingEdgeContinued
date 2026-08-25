"""Daily Yahoo ADP refresh for the shipped 2026 Draft Board. MARKET DATA ONLY.

Mirrors refresh_board_espn_adp.py. The 180-player universe and Model Proj stay frozen.
This script pulls Yahoo ADP, joins it onto that exact 180 via the committed yahoo_id
map, recomputes within-180 Yahoo position ranks, and writes one regenerable overlay.

It never writes Sleeper or ESPN overlays, phase4_band_2026.csv, talent artifacts, or
the season dataset. Unmatched rows stay blank: Yahoo prices are never filled from
Sleeper or ESPN.

Coverage gate: overall AND every position must clear the same floors as the
Sleeper refresh (coverage_floor on the live n). A failed pull or a coverage
breach aborts before writing and leaves the previous Yahoo overlay on disk.

In-season pause: same SEASON_START / BOARD_REFRESH_FORCE contract as Sleeper.

Run:   python fantasy/seasonal_projections/refresh_board_yahoo_adp.py [--force]
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import date, datetime, timezone
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(HERE))

from apply_board_labels import nmz
from fetch_yahoo_adp import ID_MAP_CSV, fetch_yahoo_adp, load_yahoo_id_map
from refresh_board_adp import (
    MIN_PULL_PLAYERS,
    MIN_UNIVERSE,
    _atomic_write,
    _forced,
    _season_start,
    check_coverage,
    coverage_floor,
    load_board_universe,
)

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

OVERLAY = HERE / "board_yahoo_adp_live_2026.csv"
LOGS_DIR = HERE / "adp_logs"
LEDGER = LOGS_DIR / "yahoo_refresh_ledger.jsonl"

OVERLAY_CORE_COLS = [
    "player_id", "player", "position", "yahoo_id", "yahoo_adp", "yahoo_pos_rank",
    "refreshed_at",
]
OVERLAY_META_COLS = ["adp_source", "adp_matched"]


def _prior_snapshot(today_name: str) -> pd.DataFrame | None:
    if not LOGS_DIR.exists():
        return None
    snaps = sorted(
        p for p in LOGS_DIR.glob("board_yahoo_adp_*.csv") if p.name != today_name
    )
    if not snaps:
        return None
    return pd.read_csv(snaps[-1])


def build_yahoo_overlay_full(
    universe: pd.DataFrame,
    fresh: pd.DataFrame,
    id_map: pd.DataFrame,
    source_date: str,
):
    """Refresh Yahoo prices for the fixed 180. Blank where Yahoo did not match.

    Returns (overlay, coverage). adp_source is "fresh" or "unmatched". Unmatched
    rows keep a blank yahoo_adp; they are never filled from Sleeper or ESPN.
    """
    u = universe.copy()
    u["player_id"] = u["player_id"].astype("string")
    u["nn"] = u["player"].map(nmz)
    ids = id_map.copy()
    ids["player_id"] = ids["player_id"].astype("string")
    ids["yahoo_id"] = ids["yahoo_id"].astype("string")
    m = u.merge(ids[["player_id", "yahoo_id"]], on="player_id", how="left")

    f = fresh.copy()
    f["yahoo_id"] = f["yahoo_id"].astype("string")
    f["nn"] = f["player"].map(nmz)
    by_id = f.drop_duplicates("yahoo_id")[["yahoo_id", "yahoo_adp"]].rename(
        columns={"yahoo_adp": "adp_by_id"}
    )
    m = m.merge(by_id, on="yahoo_id", how="left")

    by_name = f.drop_duplicates(["nn", "position"])[["nn", "position", "yahoo_adp"]].rename(
        columns={"yahoo_adp": "adp_by_name"}
    )
    m = m.merge(by_name, on=["nn", "position"], how="left")
    m["yahoo_adp"] = m["adp_by_id"].where(m["adp_by_id"].notna(), m["adp_by_name"])
    m["adp_matched"] = m["yahoo_adp"].notna()
    m["adp_source"] = m["adp_matched"].map({True: "fresh", False: "unmatched"})

    m = m.sort_values(["yahoo_adp", "player_id"], kind="stable", na_position="last")
    m["yahoo_pos_rank"] = (
        m.groupby("position")["yahoo_adp"].rank(method="min", ascending=True).astype("Int64")
    )
    m["refreshed_at"] = source_date
    overlay = (
        m[OVERLAY_CORE_COLS + OVERLAY_META_COLS]
        .sort_values(["position", "player"], kind="stable")
        .reset_index(drop=True)
    )

    by_position = {}
    for pos, grp in m.groupby("position"):
        n = int(len(grp))
        k = int(grp["adp_matched"].sum())
        by_position[str(pos)] = {
            "n": n,
            "matched": k,
            "coverage": k / n if n else 0.0,
            "floor": coverage_floor(n),
        }
    n_all = int(len(m))
    matched = int(m["adp_matched"].sum())
    coverage = {
        "n": n_all,
        "matched": matched,
        "coverage": matched / n_all if n_all else 0.0,
        "floor": coverage_floor(n_all),
        "by_position": by_position,
    }
    return overlay, coverage


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--force",
        action="store_true",
        help="run even in-season (set by workflow_dispatch)",
    )
    args = parser.parse_args()

    run_ts = datetime.now(timezone.utc).isoformat(timespec="seconds")
    source_date = datetime.now(timezone.utc).date().isoformat()

    def ledger(row: dict) -> None:
        LOGS_DIR.mkdir(exist_ok=True)
        with open(LEDGER, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(row) + "\n")

    if date.today() >= _season_start() and not _forced(args.force):
        msg = "in-season: pre-draft board frozen, refresh paused"
        print(msg)
        ledger({
            "run_ts": run_ts, "source_date": source_date,
            "status": f"paused ({msg})", "pull_players": None,
            "matched": None, "mean_abs_rank_change": None, "movers": [],
        })
        return 0

    try:
        fresh = fetch_yahoo_adp()
    except Exception as exc:
        reason = f"aborted: pull failed ({type(exc).__name__}: {exc})"
        print(reason)
        ledger({
            "run_ts": run_ts, "source_date": source_date, "status": reason,
            "pull_players": 0, "matched": None,
            "mean_abs_rank_change": None, "movers": [],
        })
        return 1

    if fresh is None or fresh.empty or "yahoo_adp" not in fresh.columns \
            or len(fresh) < MIN_PULL_PLAYERS:
        n = 0 if fresh is None else len(fresh)
        reason = f"aborted: unhealthy pull ({n} players < {MIN_PULL_PLAYERS} floor)"
        print(reason)
        ledger({
            "run_ts": run_ts, "source_date": source_date, "status": reason,
            "pull_players": n, "matched": None,
            "mean_abs_rank_change": None, "movers": [],
        })
        return 1

    universe = load_board_universe()
    if len(universe) < MIN_UNIVERSE:
        reason = (
            f"aborted: board universe is {len(universe)} rows, below the "
            f"{MIN_UNIVERSE}-row V2 floor"
        )
        print(reason)
        ledger({
            "run_ts": run_ts, "source_date": source_date, "status": reason,
            "pull_players": int(len(fresh)), "matched": None,
            "coverage": None, "mean_abs_rank_change": None, "movers": [],
        })
        return 1

    try:
        id_map = load_yahoo_id_map(ID_MAP_CSV)
    except Exception as exc:
        reason = f"aborted: yahoo id map failed ({type(exc).__name__}: {exc})"
        print(reason)
        ledger({
            "run_ts": run_ts, "source_date": source_date, "status": reason,
            "pull_players": int(len(fresh)), "matched": None,
            "mean_abs_rank_change": None, "movers": [],
        })
        return 1

    overlay, coverage = build_yahoo_overlay_full(universe, fresh, id_map, source_date)
    matched = coverage["matched"]
    cov_by_pos = {p: round(s["coverage"], 4) for p, s in coverage["by_position"].items()}

    failures = check_coverage(coverage)
    if failures:
        reason = "aborted: coverage below floor (" + "; ".join(failures) + ")"
        print(reason)
        print(f"  nothing written; {OVERLAY.name} left untouched")
        ledger({
            "run_ts": run_ts, "source_date": source_date, "status": reason,
            "pull_players": int(len(fresh)), "matched": matched,
            "coverage": round(coverage["coverage"], 4),
            "coverage_by_position": cov_by_pos,
            "mean_abs_rank_change": None, "movers": [],
        })
        return 1

    today_name = f"board_yahoo_adp_{source_date}.csv"
    prior = _prior_snapshot(today_name)
    mean_abs = None
    movers = []
    if prior is not None and "yahoo_pos_rank" in prior.columns:
        joined = overlay.merge(
            prior[["player_id", "yahoo_pos_rank"]],
            on="player_id", how="inner", suffixes=("", "_prev"),
        )
        joined["delta"] = joined["yahoo_pos_rank"] - joined["yahoo_pos_rank_prev"]
        if len(joined):
            mean_abs = round(float(joined["delta"].abs().mean()), 3)
            names = universe.set_index("player_id")[["player", "position"]]
            top = joined.reindex(
                joined["delta"].abs().sort_values(ascending=False).index
            ).head(5)
            for _, row in top.iterrows():
                pid = row["player_id"]
                movers.append({
                    "player_id": pid,
                    "player": str(names.loc[pid, "player"]) if pid in names.index else "",
                    "position": str(names.loc[pid, "position"]) if pid in names.index else "",
                    "rank_delta": int(row["delta"]) if pd.notna(row["delta"]) else None,
                })

    _atomic_write(overlay, OVERLAY)
    LOGS_DIR.mkdir(exist_ok=True)
    _atomic_write(overlay, LOGS_DIR / today_name)
    ledger({
        "run_ts": run_ts, "source_date": source_date, "status": "success",
        "pull_players": int(len(fresh)), "matched": matched,
        "coverage": round(coverage["coverage"], 4),
        "coverage_by_position": cov_by_pos,
        "mean_abs_rank_change": mean_abs, "movers": movers,
    })

    pos_str = ", ".join(
        f"{p} {s['matched']}/{s['n']}"
        for p, s in sorted(coverage["by_position"].items())
    )
    print(
        f"yahoo refresh OK: {matched}/{len(universe)} matched to Yahoo ADP "
        f"({coverage['coverage']:.1%}, floor {coverage['floor']:.0%}); {pos_str}; "
        f"source {source_date}; mean|d rank| {mean_abs}; "
        f"wrote {OVERLAY.name} + snapshot + ledger row"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
