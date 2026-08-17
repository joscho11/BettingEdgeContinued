"""Snapshot NFL futures from The Odds API into dated files.

WHAT THIS CAN AND CANNOT GET
----------------------------
The Odds API's only NFL future is Super Bowl winner
(`americanfootball_nfl_super_bowl_winner`, has_outrights=true).
Team season win totals are not a sport key and not a market on
`americanfootball_nfl`. Those stay in
`futures/data/win_totals_2026_named_books.csv` (DraftKings primary;
2026-08-17 featured O/U).

/sports is free. Each outrights fetch costs 1 credit per region
(us = 1). Empty responses do not bill.

Run:  python futures/snapshot_odds_api_futures.py
"""
from __future__ import annotations

import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "betting"))
import odds_client as oc  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from cowork_data import FUTURES_DATA
DATA = FUTURES_DATA / "odds_api"
RAW = DATA / "raw"
TIDY = DATA / "nfl_outrights.csv"
MANIFEST = REPO / "futures" / "artifacts" / "odds_api_futures_snapshot.json"

SEASON = 2026
REGION = "us"
NFL_PREFIX = "americanfootball_nfl"


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def stamp(dt: datetime) -> str:
    return dt.strftime("%Y%m%dT%H%M%SZ")


def discover_nfl_outrights() -> tuple[list[dict], dict, list[dict]]:
    sports, hdr = oc.api_get("/sports/", all="true")
    nfl = [
        s for s in sports
        if str(s.get("key") or "").startswith(NFL_PREFIX)
    ]
    outrights = [s for s in nfl if s.get("has_outrights")]
    return outrights, hdr, nfl


def flatten_outrights(payload: object, sport_key: str, captured_at: str) -> list[dict]:
    events = payload if isinstance(payload, list) else [payload]
    rows = []
    for ev in events:
        if not isinstance(ev, dict):
            continue
        commence = ev.get("commence_time")
        for bk in ev.get("bookmakers") or []:
            book = bk.get("title") or bk.get("key")
            for mk in bk.get("markets") or []:
                market = mk.get("key")
                last = mk.get("last_update") or bk.get("last_update")
                for oc_row in mk.get("outcomes") or []:
                    name = oc_row.get("name")
                    rows.append({
                        "captured_at": captured_at,
                        "season": SEASON,
                        "sport_key": sport_key,
                        "event_id": ev.get("id"),
                        "commence_time": commence,
                        "market": market,
                        "book": book,
                        "book_key": bk.get("key"),
                        "outcome": name,
                        "team": oc.NFL_TEAMS.get(name),
                        "price_american": oc_row.get("price"),
                        "point": oc_row.get("point"),
                        "description": oc_row.get("description"),
                        "last_update": last,
                    })
    return rows


def sha256_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def main() -> None:
    DATA.mkdir(parents=True, exist_ok=True)
    RAW.mkdir(parents=True, exist_ok=True)
    now = utc_now()
    captured_at = now.isoformat()
    tag = stamp(now)

    outrights, sports_hdr, nfl_keys = discover_nfl_outrights()
    print("quota after /sports (free): remaining", sports_hdr.get("remaining"),
          "used", sports_hdr.get("used"))
    print("NFL sport keys:")
    for s in sorted(nfl_keys, key=lambda x: x.get("key") or ""):
        print(f"  {s.get('key'):45s} active={s.get('active')} "
              f"outrights={s.get('has_outrights')}")

    fetches = []
    all_rows: list[dict] = []
    last_hdr = sports_hdr

    for s in outrights:
        key = s["key"]
        payload, hdr = oc.api_get(
            f"/sports/{key}/odds/",
            regions=REGION,
            markets="outrights",
            oddsFormat="american",
        )
        last_hdr = hdr
        raw_bytes = json.dumps(payload, ensure_ascii=False, indent=2).encode("utf-8")
        raw_path = RAW / f"{tag}_{key}.json"
        raw_path.write_bytes(raw_bytes)
        rows = flatten_outrights(payload, key, captured_at)
        all_rows.extend(rows)
        n_events = len(payload) if isinstance(payload, list) else 0
        n_books = len({r["book"] for r in rows})
        n_teams = len({r["outcome"] for r in rows})
        fetches.append({
            "sport_key": key,
            "title": s.get("title"),
            "description": s.get("description"),
            "raw_file": str(raw_path.relative_to(REPO)).replace("\\", "/"),
            "sha256": sha256_bytes(raw_bytes),
            "n_events": n_events,
            "n_rows": len(rows),
            "n_books": n_books,
            "n_outcomes": n_teams,
            "quota_remaining": hdr.get("remaining"),
            "quota_used": hdr.get("used"),
        })
        print(f"fetched {key}: {len(rows)} quotes, {n_books} books, "
              f"{n_teams} outcomes; quota remaining {hdr.get('remaining')}")

    if all_rows:
        new = pd.DataFrame(all_rows)
        if TIDY.exists():
            old = pd.read_csv(TIDY)
            combined = pd.concat([old, new], ignore_index=True)
        else:
            combined = new
        combined.to_csv(TIDY, index=False)
        print(f"appended {len(new)} rows -> {TIDY.relative_to(REPO)} "
              f"(file now {len(combined)} rows)")
    else:
        print("no outright quotes returned")

    manifest = {
        "captured_at": captured_at,
        "season": SEASON,
        "region": REGION,
        "nfl_sport_keys": [
            {
                "key": s.get("key"),
                "title": s.get("title"),
                "active": s.get("active"),
                "has_outrights": s.get("has_outrights"),
            }
            for s in sorted(nfl_keys, key=lambda x: x.get("key") or "")
        ],
        "win_totals_on_odds_api": False,
        "win_totals_note": (
            "The Odds API does not expose NFL team season win totals. "
            "Named-book 2026 lines remain in "
            "futures/data/win_totals_2026_named_books.csv."
        ),
        "fetches": fetches,
        "quota": last_hdr,
    }
    MANIFEST.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print("wrote", MANIFEST.relative_to(REPO))


if __name__ == "__main__":
    main()
