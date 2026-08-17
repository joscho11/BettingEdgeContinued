"""Snapshot ESPN core-API NFL futures that match FUTURES_CAPTURE_TARGETS.md.

Public JSON GET (no sportsbook app, no HTML scrape):
  GET /v2/sports/football/leagues/nfl/seasons/2026/futures
Team and athlete names are $ref-only in that payload, so this script
resolves each unique id once and caches athletes on disk.

Only markets mapped to the locked target list are written to the tidy CSV.

Run:  python futures/snapshot_espn_futures.py
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from cowork_data import FUTURES_DATA
DATA = FUTURES_DATA / "espn"
RAW = DATA / "raw"
CACHE = DATA / "cache"
TIDY = DATA / "nfl_futures.csv"
MANIFEST = REPO / "futures" / "artifacts" / "espn_futures_snapshot.json"
SEASON = 2026
BASE = "https://sports.core.api.espn.com/v2/sports/football/leagues/nfl"
UA = "JoSchoAnalytics futures archive (local research snapshot)"
ID_RE = re.compile(r"/(?:athletes|teams)/(\d+)")

def norm_name(s: str) -> str:
    return " ".join((s or "").split())


# ESPN market name (whitespace-normalized) -> (target_id, target_name, extra label)
MARKET_MAP = {
    "NFL - Super Bowl Winner": (1, "Super Bowl winner", None),
    "Pro Football (A) Conference Winner": (4, "AFC winner", None),
    "Pro Football (N) Conference - Winner": (5, "NFC winner", None),
    "Pro Football (A) East Division - Winner": (6, "Division winner", "AFC East"),
    "Pro Football (A) North Division - Winner": (6, "Division winner", "AFC North"),
    "Pro Football (A) South Division - Winner": (6, "Division winner", "AFC South"),
    "Pro Football (A) West Division - Winner": (6, "Division winner", "AFC West"),
    "Pro Football (N) East Division - Winner": (6, "Division winner", "NFC East"),
    "Pro Football (N) North Division - Winner": (6, "Division winner", "NFC North"),
    "Pro Football (N) South Division": (6, "Division winner", "NFC South"),
    "Pro Football (N) West Division": (6, "Division winner", "NFC West"),
    "Most Regular Season Passing Yards": (17, "Passing yards leader", None),
    "Most Regular Season Rushing Yards": (18, "Rushing yards leader", None),
    "Most Regular Season Receiving Yards": (19, "Receiving yards leader", None),
    "Regular Season MVP": (32, "MVP", None),
    "Offensive Player of the Year": (33, "Offensive Player of the Year", None),
    "Defensive Player of the Year": (34, "Defensive Player of the Year", None),
    "Offensive Rookie of the Year": (35, "Offensive Rookie of the Year", None),
    "Defensive Rookie of the Year": (36, "Defensive Rookie of the Year", None),
    "Coach of the Year": (37, "Coach of the Year", None),
    "Comeback Player of the Year": (38, "Comeback Player of the Year", None),
    "Protector of the Year": (39, "Protector of the Year", None),
    "NFL - Team To Win Most Games (reg. season)": (40, "Team to win most regular-season games", None),
}

# ESPN abbreviation -> nflverse
TEAM_ABBR = {
    "WSH": "WAS", "WAS": "WAS", "LAR": "LA", "LA": "LA", "LAR ": "LA",
}


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def stamp(dt: datetime) -> str:
    return dt.strftime("%Y%m%dT%H%M%SZ")


def get_json(url: str) -> dict:
    req = Request(url, headers={"User-Agent": UA, "Accept": "application/json"})
    try:
        with urlopen(req, timeout=30) as r:
            return json.loads(r.read().decode("utf-8"))
    except HTTPError as e:
        body = e.read().decode("utf-8", "replace")
        raise SystemExit(f"HTTP {e.code} {url}: {body[:300]}") from e
    except URLError as e:
        raise SystemExit(f"network error {url}: {e.reason}") from e


def parse_id(ref: str | None) -> str | None:
    if not ref:
        return None
    m = ID_RE.search(ref)
    return m.group(1) if m else None


def parse_american(value) -> int | None:
    if value is None:
        return None
    s = str(value).replace(",", "").strip()
    if s in {"", "EVEN", "even"}:
        return 100
    try:
        return int(s)
    except ValueError:
        return None


def load_athlete_cache() -> dict:
    path = CACHE / "athletes.json"
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def save_athlete_cache(cache: dict) -> None:
    CACHE.mkdir(parents=True, exist_ok=True)
    (CACHE / "athletes.json").write_text(
        json.dumps(cache, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def resolve_teams() -> dict[str, dict]:
    out: dict[str, dict] = {}
    url = f"{BASE}/seasons/{SEASON}/teams?limit=50"
    payload = get_json(url)
    for item in payload.get("items") or []:
        ref = item.get("$ref") if isinstance(item, dict) else None
        if not ref:
            continue
        team = get_json(ref)
        tid = str(team.get("id") or parse_id(ref))
        raw_abbr = (team.get("abbreviation") or "").upper()
        abbr = TEAM_ABBR.get(raw_abbr, raw_abbr)
        if abbr == "LAR":
            abbr = "LA"
        out[tid] = {
            "name": team.get("displayName") or team.get("name"),
            "abbr": abbr,
        }
        time.sleep(0.05)
    return out


def resolve_athletes(ids: set[str], cache: dict) -> dict:
    missing = [i for i in sorted(ids) if i not in cache]
    for i, aid in enumerate(missing):
        url = f"{BASE}/seasons/{SEASON}/athletes/{aid}?lang=en&region=us"
        try:
            a = get_json(url)
        except SystemExit:
            cache[aid] = {"name": None, "error": "fetch_failed"}
            continue
        cache[aid] = {
            "name": a.get("displayName") or a.get("fullName") or a.get("shortName"),
            "position": ((a.get("position") or {}).get("abbreviation")),
        }
        if (i + 1) % 40 == 0:
            save_athlete_cache(cache)
            print(f"  resolved {i + 1}/{len(missing)} new athletes")
        time.sleep(0.08)
    if missing:
        save_athlete_cache(cache)
    return cache


def flatten(payload: dict, captured_at: str, teams: dict, athletes: dict) -> tuple[list[dict], list[str]]:
    rows = []
    unmapped = []
    for item in payload.get("items") or []:
        espn_name = item.get("name") or ""
        mapped = MARKET_MAP.get(norm_name(espn_name))
        if not mapped:
            unmapped.append(norm_name(espn_name))
            continue
        target_id, target_name, group = mapped
        for fut in item.get("futures") or []:
            provider = (fut.get("provider") or {}).get("name")
            for book in fut.get("books") or []:
                team_ref = (book.get("team") or {}).get("$ref")
                ath_ref = (book.get("athlete") or {}).get("$ref")
                team_id = parse_id(team_ref)
                ath_id = parse_id(ath_ref)
                team_meta = teams.get(team_id or "", {})
                ath_meta = athletes.get(ath_id or "", {})
                rows.append({
                    "captured_at": captured_at,
                    "season": SEASON,
                    "source": "espn_core_api_v2",
                    "target_id": target_id,
                    "target_name": target_name,
                    "market_group": group,
                    "espn_market_id": item.get("id"),
                    "espn_market_name": espn_name,
                    "book": provider,
                    "outcome": ath_meta.get("name") or team_meta.get("name"),
                    "team": team_meta.get("abbr"),
                    "athlete_id": ath_id,
                    "team_id": team_id,
                    "price_american": parse_american(book.get("value")),
                    "raw_value": book.get("value"),
                })
    return rows, unmapped


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--raw", type=Path, default=None,
                   help="rebuild tidy CSV from an existing raw futures JSON")
    args = p.parse_args()

    DATA.mkdir(parents=True, exist_ok=True)
    RAW.mkdir(parents=True, exist_ok=True)
    CACHE.mkdir(parents=True, exist_ok=True)
    now = utc_now()
    captured_at = now.isoformat()
    tag = stamp(now)
    url = f"{BASE}/seasons/{SEASON}/futures?limit=50&lang=en&region=us"

    if args.raw:
        raw_path = args.raw.resolve()
        payload = json.loads(raw_path.read_text(encoding="utf-8"))
        raw_bytes = raw_path.read_bytes()
        print("rebuilding from", raw_path)
    else:
        print("fetching", url)
        payload = get_json(url)
        raw_bytes = json.dumps(payload, ensure_ascii=False, indent=2).encode("utf-8")
        raw_path = RAW / f"{tag}_futures.json"
        raw_path.write_bytes(raw_bytes)
        print(f"raw {len(payload.get('items') or [])} markets -> {raw_path.relative_to(REPO)}")

    print("resolving teams")
    teams = resolve_teams()
    print(f"  {len(teams)} teams")

    athlete_ids: set[str] = set()
    for item in payload.get("items") or []:
        if norm_name(item.get("name") or "") not in MARKET_MAP:
            continue
        for fut in item.get("futures") or []:
            for book in fut.get("books") or []:
                aid = parse_id((book.get("athlete") or {}).get("$ref"))
                if aid:
                    athlete_ids.add(aid)
    cache = load_athlete_cache()
    print(f"resolving {len(athlete_ids)} athletes ({sum(1 for i in athlete_ids if i not in cache)} new)")
    athletes = resolve_athletes(athlete_ids, cache)

    rows, unmapped = flatten(payload, captured_at, teams, athletes)
    new = pd.DataFrame(rows)
    if args.raw or not TIDY.exists():
        combined = new
    else:
        old = pd.read_csv(TIDY)
        combined = pd.concat([old, new], ignore_index=True)
    combined.to_csv(TIDY, index=False)
    print(f"wrote {len(new)} rows -> {TIDY.relative_to(REPO)} (file now {len(combined)})")

    by_target = (
        new.groupby(["target_id", "target_name"], as_index=False)
        .agg(n_quotes=("price_american", "size"), n_outcomes=("outcome", "nunique"))
        .sort_values("target_id")
    )
    print(by_target.to_string(index=False))
    print("unmapped ESPN markets:", unmapped)

    missing_names = int(new["outcome"].isna().sum()) if len(new) else 0
    manifest = {
        "captured_at": captured_at,
        "season": SEASON,
        "source_url": url,
        "raw_file": str(raw_path.relative_to(REPO)).replace("\\", "/"),
        "sha256": hashlib.sha256(raw_bytes).hexdigest(),
        "n_rows": int(len(new)),
        "n_targets_logged": int(new["target_id"].nunique()) if len(new) else 0,
        "missing_outcome_names": missing_names,
        "n_teams_resolved": len(teams),
        "n_athletes_resolved": len(athlete_ids),
        "unmapped_espn_markets": unmapped,
        "by_target": by_target.to_dict(orient="records"),
    }
    MANIFEST.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print("wrote", MANIFEST.relative_to(REPO))
    if missing_names:
        raise SystemExit(f"{missing_names} quotes missing resolved names")


if __name__ == "__main__":
    main()
