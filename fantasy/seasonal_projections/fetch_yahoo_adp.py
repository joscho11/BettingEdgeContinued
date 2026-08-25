"""Fetch Yahoo fantasy ADP for the 2026 Draft Board.

Source (unofficial, public read-only, same class as ESPN's kona feed):
    GET https://pub-api-ro.fantasysports.yahoo.com/fantasy/v2/league/{game}.l.public
        ;out=settings/players;position=ALL;start={n};count=200;sort=AR;out=draft_analysis?format=json

Game key 470 is Yahoo's 2026 NFL game (verified against /fantasy/v2/game/nfl).
The ADP lives at player.draft_analysis.average_pick. Yahoo publishes one ADP
series, not a half-PPR ranking. Label it Yahoo ADP.

Join to the frozen 180 via the committed yahoo_id map (board_yahoo_ids_2026.csv).
nflverse players.parquet has no yahoo_id column, so the map is name+position
against Yahoo's own roster, then frozen. Refresh never hits nflverse.

Output of a live refresh is board_yahoo_adp_live_2026.csv
(see refresh_board_yahoo_adp.py). This module is fetch + parse + the frozen
id map only.

Run:
    python fantasy/seasonal_projections/fetch_yahoo_adp.py
    python fantasy/seasonal_projections/fetch_yahoo_adp.py --write-id-map
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
import requests

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _utils import SKILL_POSITIONS, norm_name
from apply_board_labels import nmz

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent.parent
V2_SOURCE = REPO_ROOT / "fantasy" / "projections" / "results" / "independent_half_ppr_points_2026.csv"
ID_MAP_CSV = HERE / "board_yahoo_ids_2026.csv"

SEASON = 2026
GAME_KEY = "470"
LEAGUE_KEY = f"{GAME_KEY}.l.public"
PAGE_SIZE = 200
YAHOO_PLAYERS_URL = (
    "https://pub-api-ro.fantasysports.yahoo.com/fantasy/v2/league/"
    f"{LEAGUE_KEY};out=settings/players;position=ALL;start={{start}};count={PAGE_SIZE}"
    ";sort=AR;out=draft_analysis?format=json"
)
HEADERS = {
    "User-Agent": "Mozilla/5.0 (JoSchoAnalytics draft board Yahoo ADP)",
    "Accept": "*/*",
    "Origin": "https://football.fantasysports.yahoo.com",
}
SKILL = set(SKILL_POSITIONS)

# V2 board placeholder ids that are not nflverse gsis_id. Measured 2026-08-24:
# WAS797326 is Mike Washington Jr. (Yahoo 42744). Name match already covers
# him because nmz strips Jr; this alias is a pin, not the join path.
BOARD_YAHOO_ID_ALIASES = {
    "WAS797326": "42744",
}


def _flatten_yahoo(node, out=None) -> dict:
    """Collapse Yahoo's list-of-singleton-dicts JSON into one dict."""
    if out is None:
        out = {}
    if isinstance(node, dict):
        for key, value in node.items():
            if key == "player":
                _flatten_yahoo(value, out)
            else:
                out[key] = value
        return out
    if isinstance(node, list):
        for item in node:
            _flatten_yahoo(item, out)
        return out
    return out


def _league_players(payload: dict) -> dict:
    league = (payload or {}).get("fantasy_content", {}).get("league")
    if isinstance(league, dict):
        players = league.get("players")
        return players if isinstance(players, dict) else {}
    if isinstance(league, list):
        for item in league:
            if isinstance(item, dict) and "players" in item:
                players = item.get("players")
                return players if isinstance(players, dict) else {}
    return {}


def _player_count(players: dict) -> int:
    try:
        return int(players.get("count") or 0)
    except (TypeError, ValueError):
        return 0


def _numeric_adp(value) -> float | None:
    if value in (None, "", "-"):
        return None
    try:
        adp = float(value)
    except (TypeError, ValueError):
        return None
    if adp <= 0:
        return None
    return adp


def _skill_position(flat: dict) -> str | None:
    raw = flat.get("primary_position") or flat.get("display_position") or ""
    pos = str(raw).split(",")[0].strip().upper()
    return pos if pos in SKILL else None


def iter_yahoo_player_records(payload: dict):
    """Yield flattened skill-position player dicts from one Yahoo JSON page."""
    players = _league_players(payload)
    for key, rec in players.items():
        if key == "count" or not isinstance(rec, dict):
            continue
        flat = _flatten_yahoo(rec.get("player"))
        pos = _skill_position(flat)
        if pos is None:
            continue
        yahoo_id = flat.get("player_id")
        if yahoo_id is None:
            continue
        yahoo_id = str(yahoo_id)
        if yahoo_id.endswith(".0"):
            yahoo_id = yahoo_id[:-2]
        name_obj = flat.get("name")
        name = name_obj.get("full") if isinstance(name_obj, dict) else ""
        if not name:
            continue
        analysis = flat.get("draft_analysis")
        if isinstance(analysis, list):
            analysis = _flatten_yahoo(analysis)
        adp = None
        if isinstance(analysis, dict):
            adp = _numeric_adp(analysis.get("average_pick"))
        yield {
            "yahoo_id": yahoo_id,
            "player": name,
            "norm_name": norm_name(name),
            "position": pos,
            "yahoo_adp": adp,
        }


def parse_yahoo_roster(payload: dict) -> pd.DataFrame:
    """Skill-position Yahoo roster. yahoo_adp may be blank."""
    rows = list(iter_yahoo_player_records(payload))
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out = out.sort_values(["yahoo_id"], kind="stable")
    out = out.drop_duplicates("yahoo_id", keep="first").reset_index(drop=True)
    return out


def parse_yahoo_payload(payload: dict) -> pd.DataFrame:
    """Skill-position Yahoo ADP table. Drops players with no numeric ADP."""
    rows = [row for row in iter_yahoo_player_records(payload) if row["yahoo_adp"] is not None]
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out = out.sort_values(["yahoo_adp", "yahoo_id"], kind="stable")
    out = out.drop_duplicates("yahoo_id", keep="first").reset_index(drop=True)
    return out


def _get_yahoo_page(start: int, timeout: int) -> dict:
    url = YAHOO_PLAYERS_URL.format(start=start)
    response = requests.get(url, headers=HEADERS, timeout=timeout)
    response.raise_for_status()
    payload = response.json()
    if not isinstance(payload, dict):
        raise ValueError("Yahoo ADP payload is not a JSON object")
    return payload


def fetch_yahoo_pages(timeout: int = 60) -> list[dict]:
    """Paginate the public league player list. Does not write disk."""
    pages = []
    start = 0
    while True:
        payload = _get_yahoo_page(start, timeout)
        pages.append(payload)
        n_raw = _player_count(_league_players(payload))
        if n_raw < PAGE_SIZE:
            break
        start += PAGE_SIZE
        if start > 5000:
            raise ValueError("Yahoo ADP pagination exceeded 5000 players")
    return pages


def _concat_skill(pages: list[dict], require_adp: bool) -> pd.DataFrame:
    frames = [
        parse_yahoo_payload(page) if require_adp else parse_yahoo_roster(page)
        for page in pages
    ]
    frames = [frame for frame in frames if frame is not None and not frame.empty]
    if not frames:
        return pd.DataFrame(
            columns=["yahoo_id", "player", "norm_name", "position", "yahoo_adp"]
        )
    typed = []
    for frame in frames:
        frame = frame.copy()
        frame["yahoo_adp"] = pd.to_numeric(frame["yahoo_adp"], errors="coerce")
        typed.append(frame)
    out = pd.concat(typed, ignore_index=True)
    out = out.drop_duplicates("yahoo_id", keep="first")
    if require_adp:
        out = out.sort_values(["yahoo_adp", "yahoo_id"], kind="stable")
    else:
        out = out.sort_values(["yahoo_id"], kind="stable")
    return out.reset_index(drop=True)


def fetch_yahoo_adp(timeout: int = 60) -> pd.DataFrame:
    """Live ADP pull. Raises on HTTP errors. Does not write disk."""
    return _concat_skill(fetch_yahoo_pages(timeout=timeout), require_adp=True)


def fetch_yahoo_roster(timeout: int = 60) -> pd.DataFrame:
    """Live skill roster (ADP may be blank). Used only to rebuild the id map."""
    return _concat_skill(fetch_yahoo_pages(timeout=timeout), require_adp=False)


def load_yahoo_id_map(path: Path | None = None) -> pd.DataFrame:
    path = path or ID_MAP_CSV
    ids = pd.read_csv(path, dtype={"player_id": "string", "yahoo_id": "string"})
    required = {"player_id", "player", "position", "yahoo_id"}
    missing = required.difference(ids.columns)
    if missing:
        raise ValueError(f"Yahoo id map missing columns: {sorted(missing)}")
    ids["yahoo_id"] = ids["yahoo_id"].astype("string").str.replace(r"\.0$", "", regex=True)
    ids["player_id"] = ids["player_id"].astype("string")
    if ids["player_id"].duplicated().any() or ids["yahoo_id"].duplicated().any():
        raise ValueError("Yahoo id map has duplicate player_id or yahoo_id")
    if ids["yahoo_id"].isna().any():
        raise ValueError("Yahoo id map has blank yahoo_id rows")
    return ids


def build_yahoo_id_map(universe: pd.DataFrame, roster: pd.DataFrame) -> pd.DataFrame:
    """player_id -> yahoo_id for the frozen 180. Name+position, then aliases."""
    u = universe.copy()
    u["player_id"] = u["player_id"].astype("string")
    u["nn"] = u["player"].map(nmz)
    bio = roster.copy()
    bio["yahoo_id"] = bio["yahoo_id"].astype("string").str.replace(r"\.0$", "", regex=True)
    bio["nn"] = bio["player"].map(nmz)
    bio = bio[bio["yahoo_id"].notna() & bio["nn"].ne("")]
    bio = bio.drop_duplicates(["nn", "position"], keep="first")
    mapped = u.merge(
        bio[["nn", "position", "yahoo_id"]],
        on=["nn", "position"],
        how="left",
    )
    mapped["yahoo_id"] = mapped["yahoo_id"].astype("string")
    for player_id, yahoo_id in BOARD_YAHOO_ID_ALIASES.items():
        hit = mapped["player_id"].eq(player_id)
        mapped.loc[hit, "yahoo_id"] = yahoo_id
    missing = mapped.loc[mapped["yahoo_id"].isna(), "player"].tolist()
    if missing:
        raise ValueError(f"Yahoo id map still blank for: {missing}")
    if mapped["yahoo_id"].duplicated().any():
        dups = mapped.loc[mapped["yahoo_id"].duplicated(keep=False), "player"].tolist()
        raise ValueError(f"duplicate Yahoo ids after map: {dups}")
    return mapped[["player_id", "player", "position", "yahoo_id"]].reset_index(drop=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--write-id-map",
        action="store_true",
        help="rebuild board_yahoo_ids_2026.csv from a live Yahoo roster (local only)",
    )
    args = parser.parse_args()
    if args.write_id_map:
        universe = pd.read_csv(
            V2_SOURCE, usecols=["player_id", "player", "position"]
        )
        roster = fetch_yahoo_roster()
        ids = build_yahoo_id_map(universe, roster)
        ids.to_csv(ID_MAP_CSV, index=False)
        print(f"wrote {ID_MAP_CSV.name}: {len(ids)} rows")
        return 0

    fresh = fetch_yahoo_adp()
    print(f"Yahoo ADP pull: {len(fresh)} skill players")
    print(fresh.nsmallest(5, "yahoo_adp")[["player", "position", "yahoo_adp"]].to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
