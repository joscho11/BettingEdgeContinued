"""Parse a DraftKings futures page paste (Joseph, 2026-08-17).

Ignores nav, account balances, promo parlays, and bet-slip chrome.
Writes named-book CSVs under futures/data/manual/.
"""
from __future__ import annotations

import re
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from cowork_data import FUTURES_DATA
MANUAL = FUTURES_DATA / "manual"
RAW = MANUAL / "draftkings_2026-08-17.raw.txt"

SHORT = {
    "ARI Cardinals": "ARI", "ATL Falcons": "ATL", "BAL Ravens": "BAL",
    "BUF Bills": "BUF", "CAR Panthers": "CAR", "CHI Bears": "CHI",
    "CIN Bengals": "CIN", "CLE Browns": "CLE", "DAL Cowboys": "DAL",
    "DEN Broncos": "DEN", "DET Lions": "DET", "GB Packers": "GB",
    "HOU Texans": "HOU", "IND Colts": "IND", "JAX Jaguars": "JAX",
    "KC Chiefs": "KC", "LA Chargers": "LAC", "LA Rams": "LA",
    "LV Raiders": "LV", "MIA Dolphins": "MIA", "MIN Vikings": "MIN",
    "NE Patriots": "NE", "NO Saints": "NO", "NY Giants": "NYG",
    "NY Jets": "NYJ", "PHI Eagles": "PHI", "PIT Steelers": "PIT",
    "SEA Seahawks": "SEA", "SF 49ers": "SF", "TB Buccaneers": "TB",
    "TEN Titans": "TEN", "WAS Commanders": "WAS",
}
FULL = {
    "ARI": "Arizona Cardinals", "ATL": "Atlanta Falcons", "BAL": "Baltimore Ravens",
    "BUF": "Buffalo Bills", "CAR": "Carolina Panthers", "CHI": "Chicago Bears",
    "CIN": "Cincinnati Bengals", "CLE": "Cleveland Browns", "DAL": "Dallas Cowboys",
    "DEN": "Denver Broncos", "DET": "Detroit Lions", "GB": "Green Bay Packers",
    "HOU": "Houston Texans", "IND": "Indianapolis Colts", "JAX": "Jacksonville Jaguars",
    "KC": "Kansas City Chiefs", "LAC": "Los Angeles Chargers", "LA": "Los Angeles Rams",
    "LV": "Las Vegas Raiders", "MIA": "Miami Dolphins", "MIN": "Minnesota Vikings",
    "NE": "New England Patriots", "NO": "New Orleans Saints", "NYG": "New York Giants",
    "NYJ": "New York Jets", "PHI": "Philadelphia Eagles", "PIT": "Pittsburgh Steelers",
    "SEA": "Seattle Seahawks", "SF": "San Francisco 49ers", "TB": "Tampa Bay Buccaneers",
    "TEN": "Tennessee Titans", "WAS": "Washington Commanders",
}
DIVISIONS = [
    ("AFC East", "NFL 2026/27 - AFC East Winner"),
    ("AFC North", "NFL 2026/27 - AFC North Winner"),
    ("AFC South", "NFL 2026/27 - AFC South Winner"),
    ("AFC West", "NFL 2026/27 - AFC West Winner"),
    ("NFC East", "NFL 2026/27 - NFC East Winner"),
    ("NFC North", "NFL 2026/27 - NFC North Winner"),
    ("NFC South", "NFL 2026/27 - NFC South Winner"),
    ("NFC West", "NFL 2026/27 - NFC West Winner"),
]
PLAYER_THRESHOLD_HEADER = re.compile(
    r"^Player to Have (\d+)\+ Regular Season .+$"
)
PASS_YD_HEADER = re.compile(
    r"^Player to Have (\d+)\+ Regular Season Passing Yards$"
)
PASS_TD_HEADER = re.compile(
    r"^Player to Have (\d+)\+ Regular Season Passing TDs$"
)
REC_YD_HEADER = re.compile(
    r"^Player to Have (\d+)\+ Regular Season Receiving Yards$"
)
REC_TD_HEADER = re.compile(
    r"^Player to Have (\d+)\+ Regular Season Receiving TDs$"
)
RUSH_YD_HEADER = re.compile(
    r"^Player to Have (\d+)\+ Regular Season Rushing Yards$"
)
RUSH_TD_HEADER = re.compile(
    r"^Player to Have (\d+)\+ Regular Season Rushing TDs$"
)
SACKS_HEADER = re.compile(
    r"^Player to Have (\d+)\+ Regular Season Sacks$"
)
OU_SIDE_LINE = re.compile(r"^(Over|Under)\s+(\d+\.\d+)$")
PLAYER_OU_HEADER = re.compile(
    r"^NFL 2026/27 - (.+) Regular Season Passing Yards$"
)
PASS_TD_OU_HEADER = re.compile(
    r"^NFL 2026/27 - (.+) Regular Season Passing TDs$"
)
REC_YD_OU_HEADER = re.compile(
    r"^NFL 2026/27 - (.+) Regular Season Receiving Yards$"
)
REC_TD_OU_HEADER = re.compile(
    r"^NFL 2026/27 - (.+) Regular Season Receiving TDs$"
)
REC_OU_HEADER = re.compile(
    r"^NFL 2026/27 - (.+) Regular Season Receptions$"
)
RUSH_YD_OU_HEADER = re.compile(
    r"^NFL 2026/27 - (.+) Regular Season Rushing Yards$"
)
RUSH_TD_OU_HEADER = re.compile(
    r"^NFL 2026/27 - (.+) Regular Season Rushing TDs$"
)
SACKS_OU_HEADER = re.compile(
    r"^NFL 2026/27 - (.+) Regular Season Sacks$"
)
REC_YD_MILESTONE_HEADER = re.compile(
    r"^NFL 2026/27 - (.+) Regular Season Receiving Yards Milestones$"
)
REC_TD_MILESTONE_HEADER = re.compile(
    r"^NFL 2026/27 - (.+) Regular Season Receiving TDs Milestones$"
)
RUSH_YD_MILESTONE_HEADER = re.compile(
    r"^NFL 2026/27 - (.+) Regular Season Rushing Yards Milestones$"
)
RUSH_TD_MILESTONE_HEADER = re.compile(
    r"^NFL 2026/27 - (.+) Regular Season Rushing TDs Milestones$"
)
SACKS_MILESTONE_HEADER = re.compile(
    r"^NFL 2026/27 - (.+) Regular Season Sacks Milestones$"
)
INT_MILESTONE_HEADER = re.compile(
    r"^NFL 2026/27 - (.+) Regular Season Def Interceptions Milestones$"
)
MILESTONE_RE = re.compile(r"^(\d+)\+$")
PRICE_RE = re.compile(r"^[+−\-](\d+)$")
LINE_RE = re.compile(r"^\d+\.\d$")
CONCAT_LINE = re.compile(r"^\d+\.\d\d+\.\d$")
CONCAT_PRICE = re.compile(r"^([+−\-]\d+)\1$")


def norm_price(s: str) -> int | None:
    s = s.strip().replace("−", "-")
    m = CONCAT_PRICE.match(s.replace("−", "-"))
    if m:
        s = m.group(1)
    if PRICE_RE.match(s.replace("−", "-")) or re.fullmatch(r"[+\-]\d+", s.replace("−", "-")):
        return int(s.replace("−", "-").replace("+", ""))
    return None


def lines(text: str) -> list[str]:
    return [ln.strip() for ln in text.splitlines()]


def pair_teams(seq: list[str]) -> list[tuple[str, int]]:
    out = []
    i = 0
    while i < len(seq) - 1:
        name, pr = seq[i], seq[i + 1]
        if name in SHORT:
            p = norm_price(pr)
            if p is not None:
                out.append((name, p))
                i += 2
                continue
        i += 1
    return out


def meta(captured: str) -> dict:
    return {
        "captured_at": captured,
        "season": 2026,
        "source": "manual_paste_joseph",
        "book": "DraftKings",
        "as_of_date": "2026-08-17",
        "market_source": "DraftKings sportsbook (manual paste by Joseph)",
        "server_time_on_paste": "2026-08-17 13:36:06 ET",
    }


def pair_named_prices(seq: list[str]) -> list[tuple[str, int]]:
    """Name then American price. Skips team shorts and board headers."""
    skip = set(SHORT) | {
        "NFL 2026/27 - Player Props",
        "NFL 2026/27 – Player Props",
        "To Make Playoffs",
    }
    out = []
    i = 0
    while i < len(seq) - 1:
        name, pr = seq[i], seq[i + 1]
        if (
            name in skip
            or PLAYER_THRESHOLD_HEADER.match(name)
            or name.startswith("NFL 2026")
        ):
            i += 1
            continue
        p = norm_price(pr)
        if p is not None and not PRICE_RE.match(name.replace("−", "-")):
            out.append((name, p))
            i += 2
            continue
        i += 1
    return out


def player_threshold_rows(
    text: str,
    captured: str,
    header: re.Pattern,
    target_id: int,
    target_name: str,
    stat_label: str,
) -> list[dict]:
    ls = lines(text.replace("–", "-").replace("−", "-"))
    rows = []
    i = 0
    while i < len(ls):
        m = header.match(ls[i])
        if not m:
            i += 1
            continue
        threshold = int(m.group(1))
        j = i + 1
        while j < len(ls) and not header.match(ls[j]) and not ls[j].startswith("NFL 2026"):
            j += 1
        for player, price in pair_named_prices(ls[i + 1:j]):
            rows.append({
                **meta(captured),
                "target_id": target_id,
                "target_name": target_name,
                "market_group": f"{threshold}+",
                "market_type": "threshold_yes",
                "outcome": player,
                "team": "",
                "threshold": threshold,
                "price_american": price,
                "raw_value": f"{price:+d}",
                "settlement_note": (
                    f"Yes to {threshold}+ regular-season {stat_label}; "
                    "DK board had no Under"
                ),
            })
        i = j
    return rows


def pass_yd_threshold_rows(text: str, captured: str) -> list[dict]:
    return player_threshold_rows(
        text, captured, PASS_YD_HEADER, 10,
        "Regular-season passing yards", "passing yards",
    )


def pass_td_threshold_rows(text: str, captured: str) -> list[dict]:
    return player_threshold_rows(
        text, captured, PASS_TD_HEADER, 11,
        "Regular-season passing TDs", "passing TDs",
    )


def rec_yd_threshold_rows(text: str, captured: str) -> list[dict]:
    return player_threshold_rows(
        text, captured, REC_YD_HEADER, 12,
        "Regular-season receiving yards", "receiving yards",
    )


def rec_td_threshold_rows(text: str, captured: str) -> list[dict]:
    return player_threshold_rows(
        text, captured, REC_TD_HEADER, 13,
        "Regular-season receiving TDs", "receiving TDs",
    )


def rush_yd_threshold_rows(text: str, captured: str) -> list[dict]:
    return player_threshold_rows(
        text, captured, RUSH_YD_HEADER, 14,
        "Regular-season rushing yards", "rushing yards",
    )


def rush_td_threshold_rows(text: str, captured: str) -> list[dict]:
    return player_threshold_rows(
        text, captured, RUSH_TD_HEADER, 15,
        "Regular-season rushing TDs", "rushing TDs",
    )


def sacks_threshold_rows(text: str, captured: str) -> list[dict]:
    return player_threshold_rows(
        text, captured, SACKS_HEADER, 44,
        "Regular-season sacks", "sacks",
    )


def rec_yd_milestone_rows(text: str, captured: str) -> list[dict]:
    return player_milestone_rows(
        text, captured, REC_YD_MILESTONE_HEADER, 12,
        "Regular-season receiving yards", "receiving yards",
    )


def rec_td_milestone_rows(text: str, captured: str) -> list[dict]:
    return player_milestone_rows(
        text, captured, REC_TD_MILESTONE_HEADER, 13,
        "Regular-season receiving TDs", "receiving TDs",
    )


def rush_yd_milestone_rows(text: str, captured: str) -> list[dict]:
    return player_milestone_rows(
        text, captured, RUSH_YD_MILESTONE_HEADER, 14,
        "Regular-season rushing yards", "rushing yards",
    )


def rush_td_milestone_rows(text: str, captured: str) -> list[dict]:
    return player_milestone_rows(
        text, captured, RUSH_TD_MILESTONE_HEADER, 15,
        "Regular-season rushing TDs", "rushing TDs",
    )


def sacks_milestone_rows(text: str, captured: str) -> list[dict]:
    return player_milestone_rows(
        text, captured, SACKS_MILESTONE_HEADER, 44,
        "Regular-season sacks", "sacks",
    )


def int_milestone_rows(text: str, captured: str) -> list[dict]:
    return player_milestone_rows(
        text, captured, INT_MILESTONE_HEADER, 45,
        "Regular-season interceptions", "interceptions",
    )


def player_milestone_rows(
    text: str,
    captured: str,
    header: re.Pattern,
    target_id: int,
    target_name: str,
    stat_label: str,
) -> list[dict]:
    ls = lines(text.replace("–", "-").replace("−", "-"))
    rows = []
    i = 0
    while i < len(ls):
        m = header.match(ls[i])
        if not m:
            i += 1
            continue
        player = m.group(1).strip()
        j = i + 1
        while j < len(ls) and not header.match(ls[j]):
            j += 1
        block = ls[i + 1:j]
        n = 0
        while n < len(block) - 1:
            mm = MILESTONE_RE.match(block[n])
            p = norm_price(block[n + 1])
            if mm and p is not None:
                threshold = int(mm.group(1))
                rows.append({
                    **meta(captured),
                    "target_id": target_id,
                    "target_name": target_name,
                    "market_group": f"{threshold}+",
                    "market_type": "threshold_yes",
                    "outcome": player,
                    "team": "",
                    "threshold": threshold,
                    "price_american": p,
                    "raw_value": f"{p:+d}",
                    "settlement_note": (
                        f"Yes to {threshold}+ regular-season {stat_label}; "
                        "DK player milestone board"
                    ),
                })
                n += 2
                continue
            n += 1
        i = j
    return rows


def player_ou_rows(
    text: str,
    captured: str,
    header: re.Pattern,
    target_id: int,
    target_name: str,
    stat_label: str,
) -> list[dict]:
    ls = lines(text.replace("–", "-").replace("−", "-"))
    rows = []
    i = 0
    while i < len(ls):
        m = header.match(ls[i])
        if not m:
            i += 1
            continue
        player = m.group(1).strip()
        j = i + 1
        while j < len(ls) and not header.match(ls[j]) and not ls[j].startswith("NFL 2026"):
            j += 1
        quotes: dict[float, dict] = {}
        block = ls[i + 1:j]
        n = 0
        while n < len(block) - 1:
            m_side = OU_SIDE_LINE.match(block[n])
            if m_side:
                p = norm_price(block[n + 1])
                if p is not None:
                    line = float(m_side.group(2))
                    quotes.setdefault(line, {})[m_side.group(1).lower()] = p
                    n += 2
                    continue
            if (
                block[n] in {"Over", "Under"}
                and n + 2 < len(block)
                and LINE_RE.match(block[n + 1])
            ):
                p = norm_price(block[n + 2])
                if p is not None:
                    line = float(block[n + 1])
                    quotes.setdefault(line, {})[block[n].lower()] = p
                    n += 3
                    continue
            n += 1
        if not quotes:
            i = j
            continue
        line = next(iter(quotes))
        if len(quotes) == 1:
            d = quotes[line]
        else:
            complete = [ln for ln, d in quotes.items() if "over" in d and "under" in d]
            line = complete[0] if complete else line
            d = quotes[line]
        note = f"Regular-season {stat_label} O/U"
        if d.get("over") is None or d.get("under") is None:
            note += "; paste missing a side"
        rows.append({
            **meta(captured),
            "target_id": target_id,
            "target_name": target_name,
            "market_type": "ou",
            "outcome": player,
            "team": "",
            "line": line,
            "price_over": d.get("over"),
            "price_under": d.get("under"),
            "settlement_note": note,
        })
        i = j
    return rows


def pass_yd_ou_rows(text: str, captured: str) -> list[dict]:
    return player_ou_rows(
        text, captured, PLAYER_OU_HEADER, 10,
        "Regular-season passing yards", "passing yards",
    )


def pass_td_ou_rows(text: str, captured: str) -> list[dict]:
    return player_ou_rows(
        text, captured, PASS_TD_OU_HEADER, 11,
        "Regular-season passing TDs", "passing TDs",
    )


def rec_yd_ou_rows(text: str, captured: str) -> list[dict]:
    return player_ou_rows(
        text, captured, REC_YD_OU_HEADER, 12,
        "Regular-season receiving yards", "receiving yards",
    )


def rec_td_ou_rows(text: str, captured: str) -> list[dict]:
    return player_ou_rows(
        text, captured, REC_TD_OU_HEADER, 13,
        "Regular-season receiving TDs", "receiving TDs",
    )


def rec_ou_rows(text: str, captured: str) -> list[dict]:
    return player_ou_rows(
        text, captured, REC_OU_HEADER, 16,
        "Regular-season receptions", "receptions",
    )


def rush_yd_ou_rows(text: str, captured: str) -> list[dict]:
    return player_ou_rows(
        text, captured, RUSH_YD_OU_HEADER, 14,
        "Regular-season rushing yards", "rushing yards",
    )


def rush_td_ou_rows(text: str, captured: str) -> list[dict]:
    return player_ou_rows(
        text, captured, RUSH_TD_OU_HEADER, 15,
        "Regular-season rushing TDs", "rushing TDs",
    )


def sacks_ou_rows(text: str, captured: str) -> list[dict]:
    return player_ou_rows(
        text, captured, SACKS_OU_HEADER, 44,
        "Regular-season sacks", "sacks",
    )


def player_leader_rows(
    text: str,
    captured: str,
    title: str,
    target_id: int,
    target_name: str,
) -> list[dict]:
    ls = lines(text.replace("–", "-").replace("−", "-"))
    try:
        start = next(i for i, x in enumerate(ls) if x == title)
    except StopIteration:
        return []
    end = start + 1
    while end < len(ls) and not ls[end].startswith("Most Regular Season") and not (
        ls[end].startswith("NFL 2026") and "Leaders" not in ls[end]
    ):
        # stop at a later leader board title if present
        if ls[end].startswith("Most Regular Season") and ls[end] != title:
            break
        end += 1
    rows = []
    for player, price in pair_named_prices(ls[start + 1:end]):
        rows.append({
            **meta(captured),
            "target_id": target_id,
            "target_name": target_name,
            "market_group": "",
            "market_type": "winner",
            "outcome": player,
            "team": "",
            "price_american": price,
            "raw_value": f"{price:+d}",
        })
    return rows


def pass_yds_leader_rows(text: str, captured: str) -> list[dict]:
    return player_leader_rows(
        text, captured, "Most Regular Season Passing Yards", 17,
        "Passing yards leader",
    )


def pass_tds_leader_rows(text: str, captured: str) -> list[dict]:
    return player_leader_rows(
        text, captured, "Most Regular Season Passing Touchdowns", 20,
        "Passing TDs leader",
    )


def rush_yds_leader_rows(text: str, captured: str) -> list[dict]:
    return player_leader_rows(
        text, captured, "Most Regular Season Rushing Yards", 18,
        "Rushing yards leader",
    )


def rush_tds_leader_rows(text: str, captured: str) -> list[dict]:
    return player_leader_rows(
        text, captured, "Most Regular Season Rushing Touchdowns", 21,
        "Rushing TDs leader",
    )


def rec_yds_leader_rows(text: str, captured: str) -> list[dict]:
    return player_leader_rows(
        text, captured, "Most Regular Season Receiving Yards", 19,
        "Receiving yards leader",
    )


def rec_tds_leader_rows(text: str, captured: str) -> list[dict]:
    return player_leader_rows(
        text, captured, "Most Regular Season Receiving Touchdowns", 22,
        "Receiving TDs leader",
    )


def rec_leader_rows(text: str, captured: str) -> list[dict]:
    return player_leader_rows(
        text, captured, "Most Regular Season Receptions", 23,
        "Receptions leader",
    )


def sacks_leader_rows(text: str, captured: str) -> list[dict]:
    return player_leader_rows(
        text, captured, "Most Regular Season Sacks", 30,
        "Sacks leader",
    )


def ints_thrown_leader_rows(text: str, captured: str) -> list[dict]:
    return player_leader_rows(
        text, captured, "Most Regular Season Interceptions Thrown", 46,
        "Interceptions thrown leader",
    )


def winner_rows(pairs, target_id, target_name, captured, group=None):
    base = meta(captured)
    rows = []
    for name, price in pairs:
        abbr = SHORT[name]
        rows.append({
            **base,
            "target_id": target_id,
            "target_name": target_name,
            "market_group": group,
            "outcome": FULL[abbr],
            "team": abbr,
            "price_american": price,
            "raw_value": f"{price:+d}",
        })
    return rows


def parse_win_totals(block: str, captured: str) -> tuple[list[dict], list[str]]:
    notes = []
    rows = []
    chunks = re.split(r"\n(?=[A-Z]{2,3} .+ Regular Season Wins 2026/27)", block)
    for chunk in chunks:
        chunk = chunk.strip()
        if "Regular Season Wins" not in chunk:
            continue
        header = chunk.split("Regular Season Wins", 1)[0].strip().splitlines()[-1].strip()
        if header not in SHORT:
            notes.append(f"unmapped win-total header: {header!r}")
            continue
        abbr = SHORT[header]
        ls = lines(chunk)
        quotes: dict[float, dict] = {}
        i = 0
        while i < len(ls) - 2:
            side = ls[i]
            if side in {"Over", "Under"} and LINE_RE.match(ls[i + 1]):
                p = norm_price(ls[i + 2])
                if p is not None:
                    line = float(ls[i + 1])
                    quotes.setdefault(line, {})[side.lower()] = p
                    i += 3
                    continue
            i += 1
        if not quotes:
            notes.append(f"{abbr}: no parseable O/U lines")
            continue
        # Featured: unique line whose over price matches the first Over price
        # in the header (possibly concatenated). Fallback: line nearest 8.5
        # among complete pairs, which is not used if a price match exists.
        first_over = None
        for i, x in enumerate(ls):
            if x == "Over":
                for j in range(i + 1, min(i + 4, len(ls))):
                    p = norm_price(ls[j])
                    if p is not None:
                        first_over = p
                        break
                break
        featured = None
        if first_over is not None:
            hits = [ln for ln, d in quotes.items() if d.get("over") == first_over]
            if len(hits) == 1:
                featured = hits[0]
        if featured is None:
            complete = [ln for ln, d in quotes.items() if "over" in d and "under" in d]
            featured = sorted(complete, key=lambda ln: abs(ln - 8.5))[0] if complete else None
        if featured is None:
            notes.append(f"{abbr}: could not pick featured line")
            continue
        d = quotes[featured]
        if "over" not in d or "under" not in d:
            notes.append(f"{abbr}: featured {featured} missing a side")
        rows.append({
            **meta(captured),
            "target_id": 3,
            "target_name": "Regular-season win totals",
            "team": abbr,
            "outcome": FULL[abbr],
            "win_total_line": featured,
            "price_over": d.get("over"),
            "price_under": d.get("under"),
            "n_alt_lines": len(quotes),
            "alt_lines": ",".join(str(x) for x in sorted(quotes)),
        })
    return rows, notes


def main() -> None:
    text = RAW.read_text(encoding="utf-8")
    captured = datetime.now(timezone.utc).isoformat()
    ls = lines(text)
    joined = "\n".join(ls)

    east_i = next(i for i, x in enumerate(ls) if x.startswith("NFL 2026/27 - AFC East Winner"))
    sb_pairs = pair_teams(ls[max(0, east_i - 80):east_i])
    sb_pairs = sb_pairs[-32:]

    div_rows = []
    for label, header in DIVISIONS:
        i = next(j for j, x in enumerate(ls) if x == header)
        div_rows.extend(winner_rows(pair_teams(ls[i + 1:i + 12])[:4], 6, "Division winner", captured, label))

    nfc_west_end = next(i for i, x in enumerate(ls) if x == "NFL 2026/27 - NFC West Winner")
    after = pair_teams(ls[nfc_west_end + 1:])
    # 4 division teams, then 16 NFC, then 16 AFC
    nfc_conf = after[4:20]
    afc_conf = after[20:36]

    make_i = next(i for i, x in enumerate(ls) if x == "To Make Playoffs")
    make_pairs = pair_teams(ls[make_i + 1:])[:32]

    wt_start = joined.find("ARI Cardinals Regular Season Wins")
    wt_end = joined.find("To Make Playoffs")
    wt_rows, wt_notes = parse_win_totals(joined[wt_start:wt_end], captured)

    sb = pd.DataFrame(winner_rows(sb_pairs, 1, "Super Bowl winner", captured))
    div = pd.DataFrame(div_rows)
    nfc = pd.DataFrame(winner_rows(nfc_conf, 5, "NFC winner", captured))
    afc = pd.DataFrame(winner_rows(afc_conf, 4, "AFC winner", captured))
    make = pd.DataFrame(winner_rows(make_pairs, 8, "Make playoffs", captured))
    wt = pd.DataFrame(wt_rows)

    sb.to_csv(MANUAL / "super_bowl_winner_2026_draftkings.csv", index=False)
    pd.concat([afc, nfc, div], ignore_index=True).to_csv(
        MANUAL / "conference_division_2026_draftkings.csv", index=False
    )
    make.to_csv(MANUAL / "make_playoffs_2026_draftkings.csv", index=False)
    wt.to_csv(MANUAL / "win_totals_2026_draftkings.csv", index=False)

    print("super_bowl", len(sb), "teams", sb.team.nunique() if len(sb) else 0)
    print("division", len(div), "groups", sorted(div.market_group.unique()) if len(div) else [])
    print("nfc", len(nfc), "afc", len(afc))
    print("make", len(make), "teams", make.team.nunique() if len(make) else 0)
    print("win_totals", len(wt), "teams", sorted(wt.team) if len(wt) else [])
    missing_wt = sorted(set(SHORT.values()) - set(wt.team if len(wt) else []))
    print("win_totals missing", missing_wt)
    for n in wt_notes:
        print("NOTE", n)
    if len(sb):
        print("SB fav", sb.nsmallest(3, "price_american")[["team", "price_american"]].to_string(index=False))
    if len(wt):
        print(wt[["team", "win_total_line", "price_over", "price_under", "n_alt_lines"]].to_string(index=False))


if __name__ == "__main__":
    main()
