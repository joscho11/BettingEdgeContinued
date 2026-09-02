"""Fetch ESPN fantasy ADP for the 2026 Draft Board.

Source (unofficial, stable in the wild):
    GET https://lm-api-reads.fantasy.espn.com/apis/v3/games/ffl/seasons/{season}/segments/0/leaguedefaults/{scoring_id}?view=kona_player_info
    Header X-Fantasy-Filter must include players.limit and a sort (sortPercOwned).

Scoring IDs 1 (standard) and 3 (PPR) returned IDENTICAL averageDraftPosition
values on 2026-08-20 (1,027 / 1,027). Scoring ID 4 (often cited as half-PPR)
404s. ESPN publishes one ADP series, not a half-PPR ranking. Label it ESPN ADP.

The ADP lives at player.ownership.averageDraftPosition. Join to the frozen 180
via the committed espn_id map (board_espn_ids_2026.csv), never via a live
nflverse parquet (Streamlit Cloud and the refresh CI image do not have it).

Output of a live refresh is board_espn_adp_live_2026.csv (see refresh_board_espn_adp.py).
This module is fetch + parse + the frozen id map only.

Run:
    python fantasy/seasonal_projections/fetch_espn_adp.py
    python fantasy/seasonal_projections/fetch_espn_adp.py --write-id-map
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd
import requests

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _utils import SKILL_POSITIONS, norm_name

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent.parent
V2_SOURCE = REPO_ROOT / "fantasy" / "projections" / "results" / "independent_half_ppr_points_2026.csv"
ID_MAP_CSV = HERE / "board_espn_ids_2026.csv"
PLAYERS_PARQUET = (
    REPO_ROOT.parent / "workspace" / "nfl" / "raw" / "nflverse" / "players.parquet"
)

SEASON = 2026
SCORING_ID = 1
ESPN_URL = (
    "https://lm-api-reads.fantasy.espn.com/apis/v3/games/ffl/seasons/"
    f"{SEASON}/segments/0/leaguedefaults/{SCORING_ID}?view=kona_player_info"
)
HEADERS = {
    "User-Agent": "Mozilla/5.0 (JoSchoAnalytics draft board ESPN ADP)",
    "X-Fantasy-Filter": json.dumps(
        {"players": {"limit": 2000, "sortPercOwned": {"sortPriority": 1, "sortAsc": False}}}
    ),
}
POS_BY_ID = {1: "QB", 2: "RB", 3: "WR", 4: "TE"}
SKILL = set(SKILL_POSITIONS)

# V2 board placeholder ids that are not nflverse gsis_id. Measured 2026-08-20:
# WAS797326 is Mike Washington Jr. (nflverse 00-0040878, ESPN 4686658).
BOARD_ESPN_ID_ALIASES = {
    "WAS797326": "4686658",
    "LAN311008": "4870847",  # Ja'Kobi Lane, nflverse placeholder gsis
}


def parse_espn_payload(payload: dict) -> pd.DataFrame:
    """Turn an ESPN kona_player_info JSON object into a skill-position ADP table."""
    rows = []
    for rec in payload.get("players") or []:
        if not isinstance(rec, dict):
            continue
        player = rec.get("player") or {}
        own = player.get("ownership") or {}
        adp = own.get("averageDraftPosition")
        if adp is None:
            continue
        try:
            adp = float(adp)
        except (TypeError, ValueError):
            continue
        if adp <= 0:
            continue
        pos = POS_BY_ID.get(player.get("defaultPositionId"))
        if pos not in SKILL:
            continue
        espn_id = rec.get("id")
        if espn_id is None:
            espn_id = player.get("id")
        if espn_id is None:
            continue
        espn_id = str(espn_id)
        if espn_id.endswith(".0"):
            espn_id = espn_id[:-2]
        name = player.get("fullName") or ""
        if not name:
            continue
        rows.append({
            "espn_id": espn_id,
            "player": name,
            "norm_name": norm_name(name),
            "position": pos,
            "espn_adp": adp,
        })
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out = out.sort_values(["espn_adp", "espn_id"], kind="stable")
    out = out.drop_duplicates("espn_id", keep="first").reset_index(drop=True)
    return out


def fetch_espn_adp(timeout: int = 60) -> pd.DataFrame:
    """Live pull. Raises on HTTP errors. Does not write disk."""
    response = requests.get(ESPN_URL, headers=HEADERS, timeout=timeout)
    response.raise_for_status()
    payload = response.json()
    if not isinstance(payload, dict):
        raise ValueError("ESPN ADP payload is not a JSON object")
    return parse_espn_payload(payload)


def load_espn_id_map(path: Path | None = None) -> pd.DataFrame:
    path = path or ID_MAP_CSV
    ids = pd.read_csv(path, dtype={"player_id": "string", "espn_id": "string"})
    required = {"player_id", "player", "position", "espn_id"}
    missing = required.difference(ids.columns)
    if missing:
        raise ValueError(f"ESPN id map missing columns: {sorted(missing)}")
    ids["espn_id"] = ids["espn_id"].astype("string").str.replace(r"\.0$", "", regex=True)
    ids["player_id"] = ids["player_id"].astype("string")
    if ids["player_id"].duplicated().any() or ids["espn_id"].duplicated().any():
        raise ValueError("ESPN id map has duplicate player_id or espn_id")
    if ids["espn_id"].isna().any():
        raise ValueError("ESPN id map has blank espn_id rows")
    return ids


def build_espn_id_map(universe: pd.DataFrame, players: pd.DataFrame) -> pd.DataFrame:
    """player_id -> espn_id for the frozen 180. Uses nflverse, then explicit aliases."""
    u = universe.copy()
    u["player_id"] = u["player_id"].astype("string")
    bio = players.copy()
    bio["gsis_id"] = bio["gsis_id"].astype("string")
    bio["espn_id"] = (
        bio["espn_id"].astype("string").str.replace(r"\.0$", "", regex=True)
    )
    bio = bio[bio["espn_id"].notna() & bio["gsis_id"].notna()]
    bio = bio.drop_duplicates("gsis_id", keep="first")
    mapped = u.merge(
        bio[["gsis_id", "espn_id"]],
        left_on="player_id",
        right_on="gsis_id",
        how="left",
    )
    mapped["espn_id"] = mapped["espn_id"].astype("string")
    for player_id, espn_id in BOARD_ESPN_ID_ALIASES.items():
        hit = mapped["player_id"].eq(player_id)
        mapped.loc[hit, "espn_id"] = espn_id
    missing = mapped.loc[mapped["espn_id"].isna(), "player"].tolist()
    if missing:
        raise ValueError(f"ESPN id map still blank for: {missing}")
    if mapped["espn_id"].duplicated().any():
        dups = mapped.loc[mapped["espn_id"].duplicated(keep=False), "player"].tolist()
        raise ValueError(f"duplicate ESPN ids after map: {dups}")
    return mapped[["player_id", "player", "position", "espn_id"]].reset_index(drop=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--write-id-map",
        action="store_true",
        help="rebuild board_espn_ids_2026.csv from nflverse + aliases (local only)",
    )
    args = parser.parse_args()
    if args.write_id_map:
        if not PLAYERS_PARQUET.exists():
            print(f"players parquet missing: {PLAYERS_PARQUET}")
            return 1
        universe = pd.read_csv(
            V2_SOURCE, usecols=["player_id", "player", "position"]
        )
        players = pd.read_parquet(
            PLAYERS_PARQUET, columns=["gsis_id", "espn_id", "display_name"]
        )
        ids = build_espn_id_map(universe, players)
        ids.to_csv(ID_MAP_CSV, index=False)
        print(f"wrote {ID_MAP_CSV.name}: {len(ids)} rows")
        return 0

    fresh = fetch_espn_adp()
    print(f"ESPN ADP pull: {len(fresh)} skill players")
    print(fresh.nsmallest(5, "espn_adp")[["player", "position", "espn_adp"]].to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
