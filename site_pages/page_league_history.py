"""League History page backed by Sleeper's public league-history endpoints."""
import concurrent.futures as _cf
import glob
import html as _html
import itertools as _it
import json
import os
from datetime import datetime as dt
from pathlib import Path

import pandas as pd
import requests as req
import streamlit as st

import dashboard_data
import page_common
from dashboard_utils import metric_card, get_confidence, _md_to_html
from dashboard_chrome import _OFFLINE, TABLE_HEIGHT

_HERE = Path(__file__).resolve().parents[1]

# Sleeper league IDs in the local history fixture are 18--19 digit snowflakes. This is
# deliberately a plausibility gate, not a claim that every ID in this range exists.
_MIN_LEAGUE_ID_DIGITS = 15
_MAX_LEAGUE_ID_DIGITS = 20
_SLEEPER_GET_CACHE_ENTRIES = 128
_HISTORY_CACHE_ENTRIES = 8
_MATCHUP_FETCH_WORKERS = 6
_MAX_HISTORY_SEASONS = 10
_SEASON_CACHE_ENTRIES = _HISTORY_CACHE_ENTRIES * _MAX_HISTORY_SEASONS
_LEAGUE_HISTORY_TABS = (
    "🧠 Draft & Roster Insights",
    "🏆 All-Time Leaderboard",
    "🎖️ Hall of Fame",
    "⚔️ Rivalries",
    "📋 Report Cards",
    "📊 Consistency & Luck",
    "📈 Score Trends",
)

# Rivalries color is semantic, not decoration. Classic = established (brand green),
# Maximum Drama = close fights (rose), Fresh Blood = new pairings (sky). Score
# bands: green 70+, yellow 50-69, red below 50. Locked matchups take blue.
# Full hairline borders only; no side stripes.
_RIVALRY_MODE_SWATCH = {
    "Classic Rivalries": ("#35D08A", "rgba(53,208,138,0.14)"),
    "Maximum Drama": ("#FB7185", "rgba(251,113,133,0.16)"),
    "Fresh Blood": ("#38BDF8", "rgba(56,189,248,0.14)"),
}


def _rivalry_score_swatch(score: float, locked: bool = False) -> tuple[str, str, str]:
    """Return (ink, border, fill) for a rivalry-week card."""
    if locked:
        return "#60A5FA", "rgba(96,165,250,0.70)", "rgba(96,165,250,0.14)"
    if score >= 70:
        return "#35D08A", "rgba(53,208,138,0.55)", "rgba(53,208,138,0.12)"
    if score >= 50:
        return "#FACC15", "rgba(250,204,21,0.55)", "rgba(250,204,21,0.12)"
    return "#F87171", "rgba(248,113,113,0.55)", "rgba(248,113,113,0.12)"


def _rivalry_slate_card_html(row) -> str:
    manager_a = _html.escape(str(row["manager_a"]))
    manager_b = _html.escape(str(row["manager_b"]))
    reason = _html.escape(str(row.get("reason") or ""))
    locked = bool(row.get("locked"))
    score = float(row["rivalry_score"])
    ink, border, fill = _rivalry_score_swatch(score, locked)
    lock_mark = " 🔒" if locked else ""
    return (
        "<div class='jsa-lh-card' style='background:" + fill + ";border:1px solid " + border
        + ";border-radius:12px;padding:14px 16px;margin:0 0 10px 0;'>"
        "<div class='jsa-lh-card-row' style='display:flex;justify-content:space-between;"
        "gap:16px;align-items:flex-start;'>"
        "<div class='jsa-lh-card-copy' style='min-width:0;'>"
        "<div style='font-size:18px;font-weight:700;color:#E7ECF3;letter-spacing:-0.02em;"
        "overflow-wrap:anywhere;'>"
        + manager_a + " vs " + manager_b + lock_mark + "</div>"
        "<div style='font-size:13px;color:#E7ECF3;margin-top:6px;line-height:1.45;"
        "overflow-wrap:anywhere;'>"
        + reason + "</div>"
        "</div>"
        "<div class='jsa-lh-score' style='text-align:right;flex:0 0 auto;'>"
        "<div style='font-size:11px;font-weight:700;letter-spacing:0.08em;color:"
        + ink + ";'>RIVALRY SCORE</div>"
        "<div style='font-size:28px;font-weight:800;color:" + ink
        + ";font-variant-numeric:tabular-nums;line-height:1.1;'>"
        + f"{score:.1f}" + "</div>"
        "</div></div></div>"
    )


def _rivalry_score_legend_html() -> str:
    chips = (
        ("#35D08A", "rgba(53,208,138,0.16)", "70+ fit"),
        ("#FACC15", "rgba(250,204,21,0.16)", "50-69"),
        ("#F87171", "rgba(248,113,113,0.16)", "below 50"),
        ("#60A5FA", "rgba(96,165,250,0.16)", "Locked"),
    )
    parts = []
    for ink, fill, label in chips:
        parts.append(
            "<span style='display:inline-block;padding:3px 9px;border-radius:999px;"
            "border:1px solid " + ink + ";background:" + fill + ";color:" + ink
            + ";font-size:11px;font-weight:700;letter-spacing:0.04em;'>"
            + label + "</span>"
        )
    return (
        "<div class='jsa-lh-legend' style='display:flex;gap:8px;flex-wrap:wrap;margin:4px 0 12px 0;'>"
        + "".join(parts) + "</div>"
    )


def _lh_plotly(fig) -> None:
    """Stretch charts and keep phone scroll from being stolen by Plotly."""
    st.plotly_chart(
        fig,
        width="stretch",
        config={"displayModeBar": False, "scrollZoom": False},
    )


def _league_id_error(raw_league_id: str) -> str | None:
    league_id = raw_league_id.strip()
    if not league_id:
        return "Enter your Sleeper league ID to load your league history."
    if not league_id.isdigit():
        return "Sleeper league IDs contain digits only."
    if not _MIN_LEAGUE_ID_DIGITS <= len(league_id) <= _MAX_LEAGUE_ID_DIGITS:
        return (
            f"That does not look like a Sleeper league ID. Enter a "
            f"{_MIN_LEAGUE_ID_DIGITS}-{_MAX_LEAGUE_ID_DIGITS} digit ID from your league URL."
        )
    return None


@st.cache_data(ttl=3600, max_entries=_SLEEPER_GET_CACHE_ENTRIES)
def _sleeper_get(url: str):
    if _OFFLINE:
        return None
    try:
        r = req.get(url, timeout=15)
        r.raise_for_status()
        return r.json()
    except Exception:
        return None


@st.cache_data(ttl=3600, max_entries=_HISTORY_CACHE_ENTRIES)
def _league_history_chain(start_league_id: str) -> list[dict]:
    """Walk previous_league_id links. Newest season first."""
    current_id = start_league_id.strip()
    seen = set()
    chain: list[dict] = []
    while current_id and current_id not in {"0", ""} and current_id not in seen:
        if len(seen) >= _MAX_HISTORY_SEASONS:
            break
        seen.add(current_id)
        info = _sleeper_get(f"https://api.sleeper.app/v1/league/{current_id}")
        if not info or not isinstance(info, dict) or not info.get("season"):
            break
        chain.append({
            "league_id": current_id,
            "season": str(info.get("season")),
            "name": info.get("name") or "League",
        })
        previous_id = info.get("previous_league_id")
        current_id = previous_id if previous_id and previous_id != "0" else ""
    return chain


def _league_history_season_count(start_league_id: str) -> int:
    """Count the linked Sleeper seasons before the heavier weekly pulls begin."""
    return len(_league_history_chain(start_league_id))


def _history_load_estimate(season_count: int) -> tuple[int, int]:
    """Return a conservative first-load range calibrated to the public fetch path."""
    seasons = max(1, min(int(season_count or 1), _MAX_HISTORY_SEASONS))
    return max(5, seasons * 2), max(10, seasons * 4)


def _fetch_league_weeks(league_id: str, resource: str) -> dict:
    """Fetch weeks 1-18 for matchups or transactions with bounded concurrency."""
    def _fetch_wk(wk):
        try:
            r = req.get(
                f"https://api.sleeper.app/v1/league/{league_id}/{resource}/{wk}",
                timeout=15,
            )
            r.raise_for_status()
            return wk, r.json()
        except Exception:
            return wk, None

    with _cf.ThreadPoolExecutor(max_workers=_MATCHUP_FETCH_WORKERS) as pool:
        return dict(pool.map(_fetch_wk, range(1, 19)))


@st.cache_data(ttl=86400, max_entries=1)
def _fetch_player_directory() -> dict:
    """Sleeper asks consumers to cache the large NFL player directory."""
    payload = _sleeper_get("https://api.sleeper.app/v1/players/nfl")
    return payload if isinstance(payload, dict) else {}


@st.cache_data(ttl=3600, max_entries=_SEASON_CACHE_ENTRIES)
def _fetch_one_season(league_id: str):
    """Return (season, payload) for one Sleeper league-season, or None."""
    current_id = league_id.strip()
    info = _sleeper_get(f"https://api.sleeper.app/v1/league/{current_id}")
    if not info or not isinstance(info, dict):
        return None
    yr = info.get("season")
    if not yr:
        return None

    users_raw = _sleeper_get(f"https://api.sleeper.app/v1/league/{current_id}/users") or []
    rosters_raw = _sleeper_get(f"https://api.sleeper.app/v1/league/{current_id}/rosters") or []
    bracket_raw = _sleeper_get(f"https://api.sleeper.app/v1/league/{current_id}/winners_bracket") or []
    draft_id = str(info.get("draft_id") or "")
    draft_picks = (
        _sleeper_get(f"https://api.sleeper.app/v1/draft/{draft_id}/picks") or []
        if draft_id else []
    )

    user_map = {
        u["user_id"]: {
            "username": u.get("display_name") or "—",
            "team_name": (u.get("metadata") or {}).get("team_name") or "",
        }
        for u in users_raw if isinstance(u, dict)
    }

    playoff_finish = {}
    champion_rid = None
    runner_up_rid = None

    if bracket_raw and info.get("status") == "complete":
        valid = [m for m in bracket_raw if isinstance(m, dict)]
        max_r = max((m.get("r", 0) for m in valid), default=0)
        for m in valid:
            if m.get("r") != max_r or m.get("w") is None or m.get("l") is None:
                continue
            w, l, p = str(m["w"]), str(m["l"]), m.get("p")
            if p == 1:
                champion_rid = w
                runner_up_rid = l
            if p:
                if w not in playoff_finish or p < playoff_finish[w]:
                    playoff_finish[w] = p
                if l not in playoff_finish or p + 1 < playoff_finish[l]:
                    playoff_finish[l] = p + 1

    standings = []
    for ro in rosters_raw:
        if not isinstance(ro, dict):
            continue
        rid = str(ro.get("roster_id", ""))
        owner_id = ro.get("owner_id")
        u = user_map.get(owner_id, {"username": "—", "team_name": ""})
        s = ro.get("settings") or {}
        fpts = s.get("fpts", 0) + s.get("fpts_decimal", 0) / 100
        fpts_ag = s.get("fpts_against", 0) + s.get("fpts_against_decimal", 0) / 100
        standings.append({
            "roster_id": ro.get("roster_id"),
            "owner_id": owner_id,
            "username": u["username"],
            "team_name": u["team_name"],
            "wins": s.get("wins", 0),
            "losses": s.get("losses", 0),
            "fpts": round(fpts, 2),
            "fpts_against": round(fpts_ag, 2),
            "playoff_finish": playoff_finish.get(rid),
        })

    standings.sort(key=lambda x: (x["playoff_finish"] or 99, -x["wins"], -x["fpts"]))

    def _by_rid(rid_str):
        for row in standings:
            if str(row["roster_id"]) == rid_str:
                return row
        return {"username": "?", "team_name": ""}

    champ = _by_rid(champion_rid) if champion_rid else {"username": "?", "team_name": ""}
    ruup = _by_rid(runner_up_rid) if runner_up_rid else {"username": "?", "team_name": ""}

    _lg_settings = info.get("settings") or {}
    _playoff_start = int(_lg_settings.get("playoff_week_start") or 15)
    _wk_data = _fetch_league_weeks(current_id, "matchups")

    _matchups_season: list = []
    _roster_entries_season: list = []
    for _wk in range(1, 19):
        _wk_raw = _wk_data.get(_wk)
        if not _wk_raw or not isinstance(_wk_raw, list):
            continue
        _grps: dict = {}
        for _entry in _wk_raw:
            if not isinstance(_entry, dict):
                continue
            _mid = _entry.get("matchup_id")
            _roster_entries_season.append({
                "season": yr,
                "week": _wk,
                "roster_id": _entry.get("roster_id"),
                "matchup_id": _mid,
                "players": list(_entry.get("players") or []),
                "starters": list(_entry.get("starters") or []),
                "players_points": dict(_entry.get("players_points") or {}),
            })
            if _mid is None:
                continue
            _grps.setdefault(_mid, []).append(_entry)
        for _mid2, _ents in _grps.items():
            if len(_ents) == 2:
                _ma, _mb = _ents[0], _ents[1]
                _sa = float(_ma.get("points") or 0)
                _sb = float(_mb.get("points") or 0)
                if _sa == 0 and _sb == 0:
                    continue
                _matchups_season.append({
                    "season": yr,
                    "week": _wk,
                    "is_playoff": _wk >= _playoff_start,
                    "rid_a": str(_ma.get("roster_id", "")),
                    "score_a": _sa,
                    "rid_b": str(_mb.get("roster_id", "")),
                    "score_b": _sb,
                })

    payload = {
        "league_id": current_id,
        "draft_id": draft_id,
        "status": info.get("status"),
        "champion": {"username": champ["username"], "team_name": champ.get("team_name", "")},
        "runner_up": {"username": ruup["username"], "team_name": ruup.get("team_name", "")},
        "standings": standings,
        "matchups": _matchups_season,
        "draft_picks": draft_picks if isinstance(draft_picks, list) else [],
        "roster_entries": _roster_entries_season,
        "league_settings": {
            "total_rosters": info.get("total_rosters"),
            "roster_positions": list(info.get("roster_positions") or []),
            "scoring_settings": dict(info.get("scoring_settings") or {}),
            "waiver_type": _lg_settings.get("waiver_type"),
            "waiver_budget": _lg_settings.get("waiver_budget"),
        },
    }
    return str(yr), payload


@st.cache_data(ttl=3600, max_entries=_SEASON_CACHE_ENTRIES)
def _fetch_season_transactions(league_id: str) -> list:
    """All complete-looking weekly transactions for one Sleeper season."""
    weeks = _fetch_league_weeks(league_id.strip(), "transactions")
    rows: list = []
    for week in range(1, 19):
        payload = weeks.get(week)
        if isinstance(payload, list):
            rows.extend(payload)
    return rows


def _fetch_sleeper_history(start_league_id: str) -> dict:
    """Compose the cached chain plus per-season payloads. Used by tests."""
    chain = _league_history_chain(start_league_id)
    seasons = {}
    league_name = chain[0]["name"] if chain else "League"
    for item in chain:
        fetched = _fetch_one_season(item["league_id"])
        if not fetched:
            continue
        yr, payload = fetched
        seasons[yr] = payload
        if league_name == "League":
            league_name = item.get("name") or league_name
    return {"league_name": league_name or "League", "seasons": seasons}


def _render_load_form():
    with st.form("lh_load_form", border=False):
        league_id_input = st.text_input(
            "Sleeper League ID",
            value="",
            placeholder="e.g. 1255197436951932928",
            help="Find it in your Sleeper league URL: sleeper.com/leagues/{ID}/league",
            key="lh_league_id",
        )
        load_requested = st.form_submit_button("Load league history", type="primary")
    return league_id_input, load_requested


def _load_history_with_status(league_id: str) -> tuple[dict, str | None]:
    """Fetch with a visible per-season status panel. Returns (history, error)."""
    with st.status("Finding linked Sleeper seasons…", expanded=True) as status:
        chain = _league_history_chain(league_id)
        if not chain:
            status.update(
                label="Sleeper did not return a league for that ID.",
                state="error",
                expanded=True,
            )
            return {"league_name": "League", "seasons": {}}, (
                "Sleeper did not return a league for that ID. "
                "Check the number in your league URL."
            )
        n = len(chain)
        low, high = _history_load_estimate(n)
        years = ", ".join(item["season"] for item in chain)
        season_word = "season" if n == 1 else "seasons"
        st.write(f"Found {n} {season_word}: {years}.")
        st.write(f"First load usually takes {low} to {high} seconds. Reloads of this ID are instant for an hour.")
        seasons = {}
        league_name = chain[0]["name"]
        for i, item in enumerate(chain, 1):
            yr = item["season"]
            status.update(label=f"Loading {yr} (season {i} of {n})")
            st.write(f"{yr}: standings, draft board, weekly scores")
            fetched = _fetch_one_season(item["league_id"])
            if fetched:
                fetched_year, payload = fetched
                seasons[fetched_year] = payload
        if not seasons:
            status.update(
                label="This league exists, but no season history came back.",
                state="error",
                expanded=True,
            )
            return {"league_name": league_name, "seasons": {}}, (
                "This league exists, but no season history came back. "
                "It may be too new or still empty."
            )
        status.update(
            label=f"Loaded {len(seasons)} {season_word}.",
            state="complete",
            expanded=False,
        )
        return {"league_name": league_name or "League", "seasons": seasons}, None


def render():
    st.title("🏅 Fantasy League History")

    # A form batches text entry. Without it, every numeric keystroke could start a
    # multi-season public API crawl on the next Streamlit rerun.
    _league_id_input, _load_requested = _render_load_form()

    _form_error = None
    if _load_requested:
        _submitted_lid = _league_id_input.strip()
        _form_error = _league_id_error(_submitted_lid)
        if _form_error is None:
            st.session_state["lh_loaded_league_id"] = _submitted_lid
            st.session_state.pop("lh_acq_league_id", None)
            st.session_state.pop("lh_history_ready_for", None)

    # Keep a successfully loaded result visible while users change page controls or
    # prepare a different ID. Only an explicit, valid Load submits a new public request.
    _lid = st.session_state.get("lh_loaded_league_id", "")
    if _form_error:
        st.warning(_form_error)

    if not _lid:
        if not _form_error:
            st.info(
                "Enter your Sleeper league ID, then select Load league history.  \n\n"
                "First load walks every linked season (standings, drafts, weekly scores). "
                "A 3-year league is usually about 10-20 seconds. A 10-year league can take "
                "about 40. The same ID is instant for an hour after that.  \n\n"
                "Find the ID in your league URL: sleeper.com/leagues/{ID}/league"
            )
    elif _OFFLINE:
        st.info("League history needs a live connection to Sleeper and is "
                "unavailable offline.")
    else:
        if st.session_state.get("lh_history_ready_for") == _lid:
            _lh = _fetch_sleeper_history(_lid)
            _load_error = None if _lh["seasons"] else (
                "This league exists, but no season history came back. "
                "It may be too new or still empty."
            )
        else:
            _lh, _load_error = _load_history_with_status(_lid)
            if _load_error is None and _lh["seasons"]:
                st.session_state["lh_history_ready_for"] = _lid

        if _load_error:
            st.error(_load_error)
        elif not _lh["seasons"]:
            st.error(
                "This league exists, but no season history came back. "
                "It may be too new or still empty."
            )
        else:
            st.header(_lh["league_name"])
            st.caption(
                f"Loaded {len(_lh['seasons'])} "
                f"{'season' if len(_lh['seasons']) == 1 else 'seasons'}. "
                "Cached for an hour."
            )

            # Season filter
            _seasons_list = sorted(_lh["seasons"].keys())
            _season_filter = st.selectbox(
                "Season",
                ["All Time"] + _seasons_list,
                key="lh_season_filter",
            )

            # Build cross-season helpers
            from fantasy import league_intelligence as _league_intel

            _identity_names = _league_intel.manager_identity_map(_lh["seasons"])
            _identity_labels = _league_intel.manager_display_labels(_identity_names)
            _rid_to_user: dict = {}
            _rid_to_owner: dict = {}
            for _yr0, _sd0 in _lh["seasons"].items():
                for _row0 in _sd0["standings"]:
                    _rid_to_user[(_yr0, str(_row0["roster_id"]))] = _row0["username"]
                    _owner0 = str(_row0.get("owner_id") or "").strip()
                    if _owner0:
                        _rid_to_owner[(_yr0, str(_row0["roster_id"]))] = _owner0

            _all_matchups = [
                _m for _sd0 in _lh["seasons"].values()
                for _m in _sd0.get("matchups", [])
            ]

            def _guser(_yr_g, _rid_g):
                return _rid_to_user.get((_yr_g, _rid_g), "?")

            def _gowner(_yr_g, _rid_g):
                return _rid_to_owner.get((_yr_g, _rid_g), "")

            # Expand matchups into per-player game records (full, unfiltered)
            _game_records = []
            _stable_h2h_records = []
            for _m0 in _all_matchups:
                _ua0 = _guser(_m0["season"], _m0["rid_a"])
                _ub0 = _guser(_m0["season"], _m0["rid_b"])
                if "?" in (_ua0, _ub0):
                    continue
                _sa0, _sb0 = _m0["score_a"], _m0["score_b"]
                _game_records += [
                    {"season": _m0["season"], "week": _m0["week"],
                     "is_playoff": _m0["is_playoff"],
                     "username": _ua0, "score": _sa0, "won": _sa0 > _sb0,
                     "opp": _ub0, "opp_score": _sb0},
                    {"season": _m0["season"], "week": _m0["week"],
                     "is_playoff": _m0["is_playoff"],
                     "username": _ub0, "score": _sb0, "won": _sb0 > _sa0,
                     "opp": _ua0, "opp_score": _sa0},
                ]
                _oa0 = _gowner(_m0["season"], _m0["rid_a"])
                _ob0 = _gowner(_m0["season"], _m0["rid_b"])
                _la0 = _identity_labels.get(_oa0)
                _lb0 = _identity_labels.get(_ob0)
                if _oa0 and _ob0 and _la0 and _lb0 and _oa0 != _ob0:
                    _stable_h2h_records += [
                        {"season": _m0["season"], "week": _m0["week"],
                         "is_playoff": _m0["is_playoff"],
                         "username": _la0, "score": _sa0, "won": _sa0 > _sb0,
                         "opp": _lb0, "opp_score": _sb0},
                        {"season": _m0["season"], "week": _m0["week"],
                         "is_playoff": _m0["is_playoff"],
                         "username": _lb0, "score": _sb0, "won": _sb0 > _sa0,
                         "opp": _la0, "opp_score": _sa0},
                    ]

            _latest_season_key = max(
                _lh["seasons"],
                key=lambda value: (
                    pd.to_numeric(value, errors="coerce")
                    if pd.notna(pd.to_numeric(value, errors="coerce")) else -1
                ),
            )
            _active_rivalry_managers = []
            for _standing0 in _lh["seasons"][_latest_season_key].get("standings", []):
                _active_owner0 = str(_standing0.get("owner_id") or "").strip()
                _active_label0 = _identity_labels.get(_active_owner0)
                if _active_label0 and _active_label0 not in _active_rivalry_managers:
                    _active_rivalry_managers.append(_active_label0)
            _active_rivalry_managers.sort()

            # Apply season filter
            if _season_filter == "All Time":
                _filt_records  = _game_records
                _filt_matchups = _all_matchups
                _filt_seasons  = _lh["seasons"]
            else:
                _filt_records  = [r for r in _game_records  if r["season"] == _season_filter]
                _filt_matchups = [m for m in _all_matchups  if m["season"] == _season_filter]
                _filt_seasons  = {k: v for k, v in _lh["seasons"].items() if k == _season_filter}

            _all_managers = sorted(set(r["username"] for r in _game_records))

            # Compute manager list once, before sub-tabs (used in both C and D)
            _h2h_managers = sorted(set(r["username"] for r in _filt_records)) if _season_filter != "All Time" else _all_managers

            # Sub-tabs
            _lhG, _lhA, _lhB, _lhC, _lhD, _lhE, _lhF = st.tabs(
                _LEAGUE_HISTORY_TABS
            )

            # ── Sub-tab A: All-Time Leaderboard ────────────────────────────────
            with _lhA:
                import plotly.graph_objects as go

                from fantasy import league_intelligence as _league_intel

                _rec_label = _season_filter if _season_filter != "All Time" else "All-Time"
                st.subheader(f"{_rec_label} Leaderboard")
                st.caption(
                    "Regular-season record plus weekly scoring adjusted to the league's "
                    "own scoring environment. Bubble size represents seasons played."
                )

                _leader_df = _league_intel.manager_leaderboard_frame(
                    _filt_seasons, _filt_records
                )
                if _leader_df.empty:
                    st.info("No manager records are available for this selection yet.")
                else:
                    _played = _leader_df[_leader_df["win_pct"].notna()].copy()
                    _scored = _leader_df[_leader_df["avg_above_league"].notna()].copy()
                    _eligible = _played
                    if _season_filter == "All Time":
                        _established = _played[_played["seasons"].ge(2)]
                        if not _established.empty:
                            _eligible = _established

                    _top_titles = _leader_df.sort_values(
                        ["titles", "win_pct"], ascending=[False, False], na_position="last"
                    ).iloc[0]
                    _top_win = (
                        _eligible.sort_values("win_pct", ascending=False).iloc[0]
                        if not _eligible.empty else None
                    )
                    _top_score = (
                        _scored.sort_values("avg_above_league", ascending=False).iloc[0]
                        if not _scored.empty else None
                    )
                    _top_seasons = _leader_df.sort_values(
                        ["seasons", "wins"], ascending=[False, False]
                    ).iloc[0]

                    _m1, _m2, _m3, _m4 = st.columns(4)
                    with _m1:
                        if int(_top_titles["titles"]) > 0:
                            st.metric(
                                "Most Titles", _top_titles["manager"],
                                f'{int(_top_titles["titles"])} championships',
                                delta_color="off", delta_arrow="off", border=True,
                            )
                        else:
                            st.metric(
                                "Most Titles", "No champion yet", "0 championships",
                                delta_color="off", delta_arrow="off", border=True,
                            )
                    with _m2:
                        st.metric(
                            "Best Win %", _top_win["manager"] if _top_win is not None else "No games yet",
                            (
                                f'{_top_win["win_pct"]:.1f}% · '
                                f'{int(_top_win["wins"])}-{int(_top_win["losses"])}'
                                if _top_win is not None else None
                            ),
                            delta_color="off", delta_arrow="off", border=True,
                        )
                    with _m3:
                        st.metric(
                            "Best Adjusted Scorer",
                            _top_score["manager"] if _top_score is not None else "No scores yet",
                            (
                                f'{_top_score["avg_above_league"]:+.2f} pts/week vs league'
                                if _top_score is not None else None
                            ),
                            delta_color="off", delta_arrow="off", border=True,
                        )
                    with _m4:
                        st.metric(
                            "Most Seasons", _top_seasons["manager"],
                            f'{int(_top_seasons["seasons"])} seasons',
                            delta_color="off", delta_arrow="off", border=True,
                        )

                    if not _scored.empty:
                        _bubble = _scored[_scored["win_pct"].notna()].copy()
                        _bubble_sizes = 15 + _bubble["seasons"].astype(float) * 5
                        _fig_map = go.Figure(go.Scatter(
                            x=_bubble["win_pct"],
                            y=_bubble["avg_above_league"],
                            text=_bubble["manager"],
                            customdata=_bubble[[
                                "titles", "seasons", "wins", "losses", "avg_score",
                            ]],
                            mode="markers+text",
                            textposition="top center",
                            marker={
                                "size": _bubble_sizes,
                                "color": _bubble["titles"],
                                "colorscale": [[0, "#38bdf8"], [1, "#fbbf24"]],
                                "cmin": 0,
                                "cmax": max(1, int(_bubble["titles"].max())),
                                "colorbar": {"title": "Titles"},
                                "line": {"color": "rgba(255,255,255,0.75)", "width": 1},
                                "opacity": 0.9,
                            },
                            hovertemplate=(
                                "<b>%{text}</b><br>Win rate: %{x:.1f}%<br>"
                                "Points vs league: %{y:+.2f}/week<br>Avg score: %{customdata[4]:.2f}<br>"
                                "Record: %{customdata[2]}-%{customdata[3]}<br>"
                                "Seasons: %{customdata[1]}<br>Titles: %{customdata[0]}<extra></extra>"
                            ),
                        ))
                        _x_min = max(0, float(_bubble["win_pct"].min()) - 8)
                        _x_max = min(100, float(_bubble["win_pct"].max()) + 8)
                        if _x_max - _x_min < 16:
                            _x_mid = (_x_min + _x_max) / 2
                            _x_min, _x_max = max(0, _x_mid - 8), min(100, _x_mid + 8)
                        _y_span = max(2.0, float(_bubble["avg_above_league"].abs().max()) + 2)
                        _fig_map.add_vline(x=50, line_dash="dot", line_color="#94a3b8")
                        _fig_map.add_hline(y=0, line_dash="dot", line_color="#94a3b8")
                        _fig_map.update_layout(
                            title="Win Rate vs Scoring Performance",
                            height=500,
                            template="plotly_dark",
                            paper_bgcolor="rgba(0,0,0,0)",
                            plot_bgcolor="rgba(15,23,42,0.36)",
                            margin={"l": 55, "r": 45, "t": 70, "b": 55},
                            showlegend=False,
                            xaxis={
                                "title": "Regular-Season Win Rate",
                                "ticksuffix": "%",
                                "range": [_x_min, _x_max],
                                "gridcolor": "rgba(148,163,184,0.16)",
                            },
                            yaxis={
                                "title": "Avg Points Above League / Week",
                                "range": [-_y_span, _y_span],
                                "gridcolor": "rgba(148,163,184,0.16)",
                                "zeroline": False,
                            },
                        )
                        _lh_plotly(_fig_map)

                        def _names(_frame):
                            return ", ".join(_frame.sort_values(
                                "avg_above_league", ascending=False
                            )["manager"].tolist())

                        _upper_right = _bubble[
                            _bubble["win_pct"].ge(50) & _bubble["avg_above_league"].ge(0)
                        ]
                        _upper_left = _bubble[
                            _bubble["win_pct"].lt(50) & _bubble["avg_above_league"].ge(0)
                        ]
                        _lower_right = _bubble[
                            _bubble["win_pct"].ge(50) & _bubble["avg_above_league"].lt(0)
                        ]
                        _map_takeaways = []
                        if not _upper_right.empty:
                            _map_takeaways.append(
                                f"**{_names(_upper_right)}** paired above-average scoring with a winning "
                                "record—the strongest evidence of repeatable roster quality."
                            )
                        if not _upper_left.empty:
                            _map_takeaways.append(
                                f"**{_names(_upper_left)}** scored above the league but won fewer than half "
                                "their games, suggesting matchup timing cost them wins more than weak scoring did."
                            )
                        if not _lower_right.empty:
                            _map_takeaways.append(
                                f"**{_names(_lower_right)}** won despite below-average scoring; that record may "
                                "be harder to repeat without stronger weekly production."
                            )
                        if _top_score is not None:
                            _season_equiv = float(_top_score["avg_above_league"]) * 14
                            _map_takeaways.append(
                                f"**{_top_score['manager']}** leads adjusted scoring at "
                                f"{_top_score['avg_above_league']:+.2f} points per week—roughly "
                                f"{_season_equiv:+.1f} points over a 14-game regular season versus an average "
                                "team in the same weeks."
                            )
                        st.markdown(" **What it means:** " + " ".join(_map_takeaways))
                    else:
                        st.info("No completed regular-season scores are available for this selection yet.")

                    if not _played.empty:
                        _ranked = _played.sort_values("win_pct", ascending=True).copy()
                        _rank_text = [
                            f'{pct:.1f}%  ' + (f'🏆 × {int(titles)}' if titles else "")
                            for pct, titles in zip(_ranked["win_pct"], _ranked["titles"])
                        ]
                        _fig_rank = go.Figure(go.Bar(
                            x=_ranked["win_pct"],
                            y=_ranked["manager"],
                            orientation="h",
                            text=_rank_text,
                            textposition="outside",
                            cliponaxis=False,
                            marker={
                                "color": [
                                    "#fbbf24" if titles else "#38bdf8"
                                    for titles in _ranked["titles"]
                                ],
                                "line": {"color": "rgba(255,255,255,0.45)", "width": 1},
                            },
                            customdata=_ranked[["wins", "losses", "seasons", "titles"]],
                            hovertemplate=(
                                "<b>%{y}</b><br>Win rate: %{x:.1f}%<br>"
                                "Record: %{customdata[0]}-%{customdata[1]}<br>"
                                "Seasons: %{customdata[2]}<br>Titles: %{customdata[3]}<extra></extra>"
                            ),
                        ))
                        _fig_rank.add_vline(x=50, line_dash="dot", line_color="#94a3b8")
                        _fig_rank.update_layout(
                            title="Regular-Season Win Rate Ranking",
                            height=max(380, 42 * len(_ranked) + 105),
                            template="plotly_dark",
                            paper_bgcolor="rgba(0,0,0,0)",
                            plot_bgcolor="rgba(15,23,42,0.36)",
                            margin={"l": 25, "r": 95, "t": 70, "b": 50},
                            xaxis={
                                "title": "Win Rate",
                                "range": [0, 100],
                                "ticksuffix": "%",
                                "gridcolor": "rgba(148,163,184,0.16)",
                            },
                            yaxis={"title": "", "automargin": True},
                            showlegend=False,
                        )
                        _lh_plotly(_fig_rank)

                        _rank_desc = _played.sort_values("win_pct", ascending=False).reset_index(drop=True)
                        _rank_leader = _rank_desc.iloc[0]
                        _runner_gap = (
                            float(_rank_leader["win_pct"] - _rank_desc.iloc[1]["win_pct"])
                            if len(_rank_desc) > 1 else 0.0
                        )
                        _sample_note = (
                            f" Their rate covers only {int(_rank_leader['seasons'])} season, so treat it as "
                            "an early lead rather than an established edge."
                            if int(_rank_leader["seasons"]) == 1 and _season_filter == "All Time"
                            else ""
                        )
                        st.markdown(
                            f" **What it means:** **{_rank_leader['manager']}** has the best raw win rate at "
                            f"{_rank_leader['win_pct']:.1f}%, {_runner_gap:.1f} percentage points ahead of second."
                            f"{_sample_note} The 50% line separates managers who have won more often than they lost."
                        )

                    _details = _leader_df.copy()
                    _details["Record"] = (
                        _details["wins"].astype(int).astype(str)
                        + "-" + _details["losses"].astype(int).astype(str)
                    )
                    _details["Best Finish"] = _details["best_finish"].apply(
                        lambda value: str(int(value)) if pd.notna(value) else "DNQ"
                    )
                    _details = _details.rename(columns={
                        "manager": "Manager",
                        "titles": "Titles",
                        "finals": "Finals",
                        "seasons": "Seasons",
                        "win_pct": "Win %",
                        "avg_score": "Avg Weekly Score",
                        "avg_above_league": "Pts Above League Avg",
                    })[[
                        "Manager", "Titles", "Finals", "Seasons", "Record", "Win %",
                        "Avg Weekly Score", "Pts Above League Avg", "Best Finish",
                    ]].sort_values(["Titles", "Win %"], ascending=[False, False])
                    with st.expander("View complete manager records"):
                        st.dataframe(
                            _details,
                            width="stretch",
                            hide_index=True,
                            column_config={
                                "Titles": st.column_config.NumberColumn(
                                    "Titles", help="Championship wins"
                                ),
                                "Finals": st.column_config.NumberColumn(
                                    "Finals", help="Championship appearances"
                                ),
                                "Win %": st.column_config.NumberColumn("Win %", format="%.1f%%"),
                                "Avg Weekly Score": st.column_config.NumberColumn(
                                    "Avg Weekly Score", format="%.2f"
                                ),
                                "Pts Above League Avg": st.column_config.NumberColumn(
                                    "Pts Above League Avg", format="%+.2f",
                                    help="Average weekly score minus that same league-week's average",
                                ),
                            },
                        )

            # ── Sub-tab B: Hall of Fame / Shame ───────────────────────────────
            with _lhB:
                import plotly.graph_objects as go

                from fantasy import league_intelligence as _league_intel

                _hof_scope = _season_filter if _season_filter != "All Time" else "all seasons"
                st.subheader("League Record Book")
                st.caption(
                    f"The best, worst, closest, and strangest games from {_hof_scope}, "
                    "including playoffs. Scores of 5 points or fewer are treated as incomplete."
                )

                if not _filt_records:
                    st.info("No weekly matchup data available.")
                else:
                    _played_recs = [
                        r for r in _filt_records
                        if r["score"] > 5 and r["opp_score"] > 5
                    ]
                    _best_score  = max(_played_recs, key=lambda r: r["score"]) if _played_recs else None
                    _worst_score = min(_played_recs, key=lambda r: r["score"]) if _played_recs else None
                    _losses_recs = [r for r in _played_recs if r["score"] < r["opp_score"]]
                    _best_loss   = max(_losses_recs, key=lambda r: r["score"]) if _losses_recs else None
                    _matchup_df = _league_intel.matchup_record_frame(_filt_records)
                    if _matchup_df.empty or not _played_recs:
                        st.info("No completed matchup scores are available for this selection.")
                    else:
                        _blowout = _matchup_df.loc[_matchup_df["margin"].idxmax()].to_dict()
                        _closest = _matchup_df.loc[_matchup_df["margin"].idxmin()].to_dict()
                        _hi_combined = _matchup_df.loc[
                            _matchup_df["combined"].idxmax()
                        ].to_dict()
                        _lo_combined = _matchup_df.loc[
                            _matchup_df["combined"].idxmin()
                        ].to_dict()
                        _luck_pool = _matchup_df[
                            _matchup_df["all_play_win_pct"].notna()
                        ].sort_values(
                            ["all_play_win_pct", "winner_score"], ascending=[True, True]
                        )
                        _luck_win = _luck_pool.iloc[0].to_dict() if not _luck_pool.empty else None

                        def _matchup_text(record):
                            if record.get("is_tie"):
                                return f"{record['team_a']} tied {record['team_b']}"
                            return f"{record['winner']} def. {record['loser']}"

                        _hero1, _hero2, _hero3, _hero4 = st.columns(4)
                        with _hero1:
                            st.metric(
                                "🏆 Highest Score", f"{_best_score['score']:.2f} pts",
                                f"{_best_score['username']} · {_best_score['season']} Wk {_best_score['week']}",
                                delta_color="off", delta_arrow="off", border=True,
                            )
                        with _hero2:
                            st.metric(
                                "😤 Most Painful Loss",
                                f"{_best_loss['score']:.2f} pts" if _best_loss else "No losses",
                                (
                                    f"{_best_loss['username']} lost by "
                                    f"{_best_loss['opp_score'] - _best_loss['score']:.2f}"
                                    if _best_loss else None
                                ),
                                delta_color="off", delta_arrow="off", border=True,
                            )
                        with _hero3:
                            st.metric(
                                "💥 Biggest Blowout", f"{_blowout['margin']:.2f} pts",
                                f"{_matchup_text(_blowout)} · {_blowout['season']} Wk {_blowout['week']}",
                                delta_color="off", delta_arrow="off", border=True,
                            )
                        with _hero4:
                            st.metric(
                                "🤝 Closest Game", f"{_closest['margin']:.2f} pts",
                                f"{_matchup_text(_closest)} · {_closest['season']} Wk {_closest['week']}",
                                delta_color="off", delta_arrow="off", border=True,
                            )

                        _matchup_df = _matchup_df.copy()
                        _matchup_df["matchup_label"] = _matchup_df.apply(
                            _matchup_text, axis=1
                        )
                        _fig_chaos = go.Figure()
                        _season_colors = [
                            "#38bdf8", "#a78bfa", "#34d399", "#fb7185",
                            "#fbbf24", "#22d3ee", "#f472b6", "#a3e635",
                        ]
                        for _color_i, (_season, _season_games) in enumerate(
                            _matchup_df.groupby("season", sort=True)
                        ):
                            _fig_chaos.add_trace(go.Scatter(
                                x=_season_games["combined"],
                                y=_season_games["margin"],
                                name=str(_season),
                                mode="markers",
                                marker={
                                    "size": 10,
                                    "color": _season_colors[_color_i % len(_season_colors)],
                                    "opacity": 0.72,
                                    "line": {"color": "rgba(255,255,255,0.38)", "width": 1},
                                },
                                customdata=_season_games[[
                                    "matchup_label", "winner_score", "loser_score", "week",
                                ]],
                                hovertemplate=(
                                    "<b>%{customdata[0]}</b><br>Score: %{customdata[1]:.2f}–"
                                    "%{customdata[2]:.2f}<br>Week %{customdata[3]}<br>"
                                    "Combined: %{x:.2f}<br>Margin: %{y:.2f}<extra>%{fullData.name}</extra>"
                                ),
                            ))

                        _highlight_labels: dict[int, list[str]] = {}
                        for _row_index, _label in (
                            (_matchup_df["combined"].idxmax(), "Highest total"),
                            (_matchup_df["margin"].idxmax(), "Biggest blowout"),
                            (_matchup_df["margin"].idxmin(), "Closest game"),
                        ):
                            _highlight_labels.setdefault(_row_index, []).append(_label)
                        _highlighted = _matchup_df.loc[list(_highlight_labels)].copy()
                        _highlighted["record_label"] = [
                            " / ".join(_highlight_labels[index]) for index in _highlighted.index
                        ]
                        _fig_chaos.add_trace(go.Scatter(
                            x=_highlighted["combined"],
                            y=_highlighted["margin"],
                            text=_highlighted["record_label"],
                            customdata=_highlighted[["matchup_label", "season", "week"]],
                            mode="markers+text",
                            textposition="top center",
                            name="Records",
                            marker={
                                "size": 17, "symbol": "diamond-open",
                                "color": "#f8fafc", "line": {"width": 2.5},
                            },
                            hovertemplate=(
                                "<b>%{text}</b><br>%{customdata[0]}<br>"
                                "%{customdata[1]} Week %{customdata[2]}<br>"
                                "Combined: %{x:.2f}<br>Margin: %{y:.2f}<extra></extra>"
                            ),
                        ))
                        _median_total = float(_matchup_df["combined"].median())
                        _median_margin = float(_matchup_df["margin"].median())
                        _fig_chaos.add_vline(
                            x=_median_total, line_dash="dot", line_color="#64748b"
                        )
                        _fig_chaos.add_hline(
                            y=_median_margin, line_dash="dot", line_color="#64748b"
                        )
                        _fig_chaos.update_layout(
                            title="The Chaos Map: Every Matchup",
                            height=520,
                            template="plotly_dark",
                            paper_bgcolor="rgba(0,0,0,0)",
                            plot_bgcolor="rgba(15,23,42,0.36)",
                            margin={"l": 55, "r": 35, "t": 70, "b": 55},
                            xaxis={
                                "title": "Combined Points",
                                "gridcolor": "rgba(148,163,184,0.16)",
                            },
                            yaxis={
                                "title": "Victory Margin",
                                "gridcolor": "rgba(148,163,184,0.16)",
                                "rangemode": "tozero",
                            },
                            legend={"title": "Season", "orientation": "h", "y": -0.18},
                        )
                        _lh_plotly(_fig_chaos)

                        _shootout_type = (
                            "a true shootout—both teams contributed to the record total"
                            if float(_hi_combined["margin"]) <= _median_margin
                            else "high scoring but one-sided, so one team drove much of the record total"
                        )
                        _blowout_multiple = (
                            float(_blowout["margin"]) / _median_margin
                            if _median_margin > 0 else 0
                        )
                        st.markdown(
                            f" **What it means:** **{_matchup_text(_hi_combined)}** was {_shootout_type}. "
                            f"The **{_blowout['winner']}** blowout was {_blowout_multiple:.1f}× the typical "
                            f"margin, making it a genuine outlier rather than merely a comfortable win. "
                            f"The dotted lines mark the typical combined score and margin, so distance from "
                            "their intersection shows how unusual each matchup was."
                        )

                        _played_df = pd.DataFrame(_played_recs)
                        _range_group = "season" if _season_filter == "All Time" else "week"
                        _range_label = "Season" if _range_group == "season" else "Week"
                        _range_rows = []
                        for _period, _period_scores in _played_df.groupby(
                            _range_group, sort=True
                        ):
                            _high_row = _period_scores.loc[_period_scores["score"].idxmax()]
                            _low_row = _period_scores.loc[_period_scores["score"].idxmin()]
                            _range_rows.append({
                                _range_label: str(_period) if _range_group == "season" else int(_period),
                                "High": float(_high_row["score"]),
                                "Average": float(_period_scores["score"].mean()),
                                "Low": float(_low_row["score"]),
                                "High Manager": _high_row["username"],
                                "Low Manager": _low_row["username"],
                            })
                        _range_df = pd.DataFrame(_range_rows)
                        _range_df["Spread"] = _range_df["High"] - _range_df["Low"]

                        _fig_range = go.Figure()
                        _fig_range.add_trace(go.Scatter(
                            x=_range_df[_range_label], y=_range_df["Low"],
                            customdata=_range_df[["Low Manager"]],
                            name="Low", mode="lines+markers",
                            line={"color": "#fb7185", "width": 2},
                            hovertemplate=(
                                f"{_range_label} %{{x}}<br>Low: %{{y:.2f}}<br>"
                                "%{customdata[0]}<extra></extra>"
                            ),
                        ))
                        _fig_range.add_trace(go.Scatter(
                            x=_range_df[_range_label], y=_range_df["High"],
                            customdata=_range_df[["High Manager"]],
                            name="High", mode="lines+markers", fill="tonexty",
                            fillcolor="rgba(56,189,248,0.12)",
                            line={"color": "#fbbf24", "width": 2},
                            hovertemplate=(
                                f"{_range_label} %{{x}}<br>High: %{{y:.2f}}<br>"
                                "%{customdata[0]}<extra></extra>"
                            ),
                        ))
                        _fig_range.add_trace(go.Scatter(
                            x=_range_df[_range_label], y=_range_df["Average"],
                            name="League average", mode="lines+markers",
                            line={"color": "#38bdf8", "width": 3},
                            hovertemplate=(
                                f"{_range_label} %{{x}}<br>Average: %{{y:.2f}}<extra></extra>"
                            ),
                        ))
                        _range_title = (
                            "Season-by-Season Scoring Range"
                            if _range_group == "season"
                            else f"Weekly Scoring Range — {_season_filter}"
                        )
                        _fig_range.update_layout(
                            title=_range_title,
                            height=430,
                            template="plotly_dark",
                            paper_bgcolor="rgba(0,0,0,0)",
                            plot_bgcolor="rgba(15,23,42,0.36)",
                            margin={"l": 55, "r": 35, "t": 70, "b": 50},
                            xaxis={
                                "title": _range_label,
                                "gridcolor": "rgba(148,163,184,0.16)",
                                **({"dtick": 1} if _range_group == "week" else {}),
                            },
                            yaxis={
                                "title": "Team Score",
                                "gridcolor": "rgba(148,163,184,0.16)",
                            },
                            legend={"orientation": "h", "y": -0.18},
                            hovermode="x unified",
                        )
                        _lh_plotly(_fig_range)

                        _widest = _range_df.loc[_range_df["Spread"].idxmax()]
                        _highest_environment = _range_df.loc[
                            _range_df["Average"].idxmax()
                        ]
                        _period_word = "season" if _range_group == "season" else "week"
                        st.markdown(
                            f" **What it means:** **{_range_label} {_widest[_range_label]}** had the widest "
                            f"scoring spread at {_widest['Spread']:.2f} points, meaning lineup outcomes were "
                            f"far more volatile than usual in that {_period_word}. **{_range_label} "
                            f"{_highest_environment[_range_label]}** had the highest average scoring environment; "
                            "records from that period deserve that context rather than being compared as if every "
                            "season or week scored identically."
                        )

                        st.markdown("#### More Records")
                        _more1, _more2, _more3, _more4 = st.columns(4)
                        with _more1:
                            st.metric(
                                "💀 Lowest Score", f"{_worst_score['score']:.2f} pts",
                                f"{_worst_score['username']} · {_worst_score['season']} Wk {_worst_score['week']}",
                                delta_color="off", delta_arrow="off", border=True,
                            )
                        with _more2:
                            st.metric(
                                "🍀 Luckiest Win (All-Play)",
                                f"{_luck_win['winner_score']:.2f} pts" if _luck_win else "Unavailable",
                                (
                                    f"{_luck_win['winner']} · beats "
                                    f"{int(_luck_win['all_play_wins'])}/{int(_luck_win['all_play_opponents'])} teams"
                                    if _luck_win else None
                                ),
                                delta_color="off", delta_arrow="off", border=True,
                            )
                        with _more3:
                            st.metric(
                                "🔥 Highest-Scoring Game",
                                f"{_hi_combined['combined']:.2f} pts",
                                f"{_matchup_text(_hi_combined)} · {_hi_combined['season']} Wk {_hi_combined['week']}",
                                delta_color="off", delta_arrow="off", border=True,
                            )
                        with _more4:
                            st.metric(
                                "🧊 Lowest-Scoring Game",
                                f"{_lo_combined['combined']:.2f} pts",
                                f"{_matchup_text(_lo_combined)} · {_lo_combined['season']} Wk {_lo_combined['week']}",
                                delta_color="off", delta_arrow="off", border=True,
                            )
                        st.caption(
                            "All-play luck compares the winner's score with every other team in that same "
                            "league-week. A winner that would have beaten very few teams received the most "
                            "favorable matchup timing."
                        )

            # ── Sub-tab C: Rivalries ──────────────────────────────────────────
            with _lhC:
                import plotly.graph_objects as go

                from fantasy import league_intelligence as _league_intel

                _all_h2h_matchups = _league_intel.matchup_record_frame(
                    _stable_h2h_records
                )

                st.subheader("Rivalries")
                st.caption(
                    "Build a rivalry week, inspect one matchup, or compare the full "
                    "league. Only the selected view is shown."
                )
                _rivalry_view = st.radio(
                    "Rivalry view",
                    ("Build a Week", "Explore a Matchup", "League Matrix"),
                    horizontal=True,
                    key="lh_rivalry_view",
                    label_visibility="collapsed",
                )

                if _rivalry_view == "Build a Week":
                    st.markdown("#### Rivalry Week Builder")
                    st.caption(
                        "Generate a complete upcoming-season slate for the league's current "
                        "managers. Scores describe historical rivalry fit, not predictions."
                    )

                    _builder_left, _builder_right = st.columns(2)
                    with _builder_left:
                        _rivalry_mode = st.selectbox(
                            "Slate style",
                            list(_league_intel.RIVALRY_WEEK_MODES),
                            key="lh_rivalry_mode",
                            help=(
                                "Classic rewards established series and playoff history; Maximum "
                                "Drama emphasizes close, back-and-forth games; Fresh Blood favors "
                                "underplayed opponents with similar historical results."
                            ),
                        )
                        _mode_ink, _mode_fill = _RIVALRY_MODE_SWATCH.get(
                            _rivalry_mode, ("#93A0B1", "rgba(147,160,177,0.12)")
                        )
                        st.markdown(
                            "<div class='jsa-lh-mode' style='display:inline-block;margin:2px 0 8px 0;"
                            "padding:4px 10px;"
                            "border-radius:999px;background:" + _mode_fill
                            + ";border:1px solid " + _mode_ink + ";color:" + _mode_ink
                            + ";font-size:12px;font-weight:700;letter-spacing:0.04em;'>"
                            + _html.escape(_rivalry_mode) + "</div>",
                            unsafe_allow_html=True,
                        )
                    with _builder_right:
                        _rivalry_history = st.selectbox(
                            "History window",
                            ["All history", "Last 3 completed seasons"],
                            key="lh_rivalry_history",
                        )

                    _builder_matchups = _all_h2h_matchups.copy()
                    if (
                        _rivalry_history == "Last 3 completed seasons"
                        and not _builder_matchups.empty
                    ):
                        _played_seasons = sorted(
                            _builder_matchups["season"].astype(str).unique().tolist(),
                            key=lambda value: (
                                pd.to_numeric(value, errors="coerce")
                                if pd.notna(pd.to_numeric(value, errors="coerce")) else -1
                            ),
                        )
                        _recent_seasons = set(_played_seasons[-3:])
                        _builder_matchups = _builder_matchups[
                            _builder_matchups["season"].astype(str).isin(_recent_seasons)
                        ].copy()

                    _builder_signature = "|".join([
                        _lid,
                        _rivalry_mode,
                        _rivalry_history,
                        *_active_rivalry_managers,
                    ])
                    if st.session_state.get("lh_rivalry_signature") != _builder_signature:
                        st.session_state["lh_rivalry_signature"] = _builder_signature
                        st.session_state["lh_rivalry_locked_pairs"] = []
                        st.session_state["lh_rivalry_avoided_pairs"] = []
                        st.session_state.pop("lh_rivalry_locked_choices", None)

                    _pair_scores = _league_intel.rivalry_pair_score_frame(
                        _builder_matchups,
                        _active_rivalry_managers,
                        mode=_rivalry_mode,
                    )
                    _saved_locks = [
                        tuple(pair)
                        for pair in st.session_state.get("lh_rivalry_locked_pairs", [])
                    ]
                    _saved_avoids = [
                        tuple(pair)
                        for pair in st.session_state.get("lh_rivalry_avoided_pairs", [])
                    ]
                    _rivalry_slate = _league_intel.rivalry_week_slate_frame(
                        _pair_scores,
                        locked_pairs=_saved_locks,
                        avoided_pairs=_saved_avoids,
                    )

                    _lock_lookup = {}
                    if not _rivalry_slate.empty:
                        for _, _slate_row in _rivalry_slate.iterrows():
                            if pd.isna(_slate_row.get("manager_b")):
                                continue
                            _slate_pair = tuple(sorted((
                                str(_slate_row["manager_a"]),
                                str(_slate_row["manager_b"]),
                            )))
                            _slate_label = f"{_slate_pair[0]} vs {_slate_pair[1]}"
                            _lock_lookup[_slate_label] = _slate_pair

                    _selected_lock_labels = st.multiselect(
                        "Lock matchups before generating another slate",
                        list(_lock_lookup),
                        key="lh_rivalry_locked_choices",
                        disabled=not _lock_lookup,
                    )
                    _requested_locks = [
                        _lock_lookup[label]
                        for label in _selected_lock_labels
                        if label in _lock_lookup
                    ]
                    _alternative_requested = st.button(
                        "Generate another slate",
                        key="lh_rivalry_regenerate",
                        disabled=len(_active_rivalry_managers) < 4,
                    )
                    if _alternative_requested:
                        _requested_lock_set = {
                            tuple(sorted(pair)) for pair in _requested_locks
                        }
                        _new_avoid_set = {
                            tuple(sorted(pair)) for pair in _saved_avoids
                        }.difference(_requested_lock_set)
                        for _, _slate_row in _rivalry_slate.iterrows():
                            if pd.isna(_slate_row.get("manager_b")):
                                continue
                            _slate_pair = tuple(sorted((
                                str(_slate_row["manager_a"]),
                                str(_slate_row["manager_b"]),
                            )))
                            if _slate_pair not in _requested_lock_set:
                                _new_avoid_set.add(_slate_pair)
                        _new_avoids = sorted(_new_avoid_set)
                        st.session_state["lh_rivalry_locked_pairs"] = [
                            list(pair) for pair in _requested_locks
                        ]
                        st.session_state["lh_rivalry_avoided_pairs"] = [
                            list(pair) for pair in _new_avoids
                        ]
                        _rivalry_slate = _league_intel.rivalry_week_slate_frame(
                            _pair_scores,
                            locked_pairs=_requested_locks,
                            avoided_pairs=_new_avoids,
                        )

                    if len(_active_rivalry_managers) < 2:
                        st.info(
                            "At least two current managers with stable Sleeper owner IDs are "
                            "needed to build a rivalry slate."
                        )
                    elif _rivalry_slate.empty:
                        st.info("No rivalry slate is available for the current league yet.")
                    else:
                        _scored_slate = pd.to_numeric(
                            _rivalry_slate["rivalry_score"], errors="coerce"
                        ).dropna()
                        _matchup_count = int(
                            _rivalry_slate["manager_b"].notna().sum()
                        )
                        _average_score = (
                            f"{_scored_slate.mean():.1f}/100"
                            if not _scored_slate.empty else "—"
                        )
                        st.caption(
                            f"{_matchup_count} matchups for "
                            f"{len(_active_rivalry_managers)} current managers · "
                            f"average score {_average_score}. "
                            "Green is 70+ fit, yellow is 50-69, red is below 50. "
                            "Locked matchups stay blue."
                        )
                        st.markdown(
                            _rivalry_score_legend_html(),
                            unsafe_allow_html=True,
                        )

                        for _, _slate_row in _rivalry_slate.iterrows():
                            if pd.isna(_slate_row.get("manager_b")):
                                st.warning(
                                    f"**{_slate_row['manager_a']}** has the open slot because "
                                    "the league currently has an odd number of managers."
                                )
                                continue
                            st.markdown(
                                _rivalry_slate_card_html(_slate_row),
                                unsafe_allow_html=True,
                            )

                elif _rivalry_view == "Explore a Matchup":
                    _h2h_scope = _season_filter if _season_filter != "All Time" else "all-time"
                    st.markdown("#### Matchup Explorer")
                    st.caption(
                        f"Inspect one completed series {_h2h_scope}. Includes playoffs; "
                        "scores of 5 points or fewer are excluded."
                    )

                    _h2h_matchups = _all_h2h_matchups
                    if _season_filter != "All Time":
                        _h2h_matchups = _h2h_matchups[
                            _h2h_matchups["season"].astype(str).eq(str(_season_filter))
                        ].copy()
                    _rivalries = _league_intel.rivalry_summary_frame(_h2h_matchups)
                    if _rivalries.empty:
                        st.info("No completed head-to-head matchups are available for this selection.")
                    else:
                        _rivalry_managers = sorted(set(_rivalries["manager_a"]).union(
                            _rivalries["manager_b"]
                        ))
                        _manager_meetings = {manager: 0 for manager in _rivalry_managers}
                        for _, _series in _rivalries.iterrows():
                            _manager_meetings[_series["manager_a"]] += int(_series["games"])
                            _manager_meetings[_series["manager_b"]] += int(_series["games"])
                        _default_a = sorted(
                            _rivalry_managers,
                            key=lambda manager: (-_manager_meetings[manager], manager),
                        )[0]
                        if st.session_state.get("lh_h2h_manager_a") not in _rivalry_managers:
                            st.session_state.pop("lh_h2h_manager_a", None)
                        _select_a, _select_b = st.columns(2)
                        with _select_a:
                            _manager_a = st.selectbox(
                                "Manager A",
                                _rivalry_managers,
                                index=_rivalry_managers.index(_default_a),
                                key="lh_h2h_manager_a",
                            )

                        _opponent_series = _rivalries[
                            (_rivalries["manager_a"] == _manager_a)
                            | (_rivalries["manager_b"] == _manager_a)
                        ].copy()
                        _opponent_series["opponent"] = _opponent_series.apply(
                            lambda row: (
                                row["manager_b"]
                                if row["manager_a"] == _manager_a else row["manager_a"]
                            ),
                            axis=1,
                        )
                        _opponent_series["series_gap"] = (
                            _opponent_series["manager_a_wins"]
                            - _opponent_series["manager_b_wins"]
                        ).abs()
                        _opponent_series = _opponent_series.sort_values(
                            ["games", "series_gap", "opponent"],
                            ascending=[False, True, True],
                        )
                        _manager_b_options = sorted(
                            _opponent_series["opponent"].unique().tolist()
                        )
                        _default_b = _opponent_series.iloc[0]["opponent"]
                        if st.session_state.get("lh_h2h_manager_b") not in _manager_b_options:
                            st.session_state.pop("lh_h2h_manager_b", None)
                        with _select_b:
                            _manager_b = st.selectbox(
                                "Manager B",
                                _manager_b_options,
                                index=_manager_b_options.index(_default_b),
                                key="lh_h2h_manager_b",
                            )

                        _selected_series = _rivalries[
                            ((_rivalries["manager_a"] == _manager_a)
                             & (_rivalries["manager_b"] == _manager_b))
                            | ((_rivalries["manager_a"] == _manager_b)
                               & (_rivalries["manager_b"] == _manager_a))
                        ]
                        if _selected_series.empty:
                            st.info(
                                f"{_manager_a} and {_manager_b} did not play during this selection."
                            )
                        else:
                            _series = _selected_series.iloc[0]
                            if _series["manager_a"] == _manager_a:
                                _a_wins = int(_series["manager_a_wins"])
                                _b_wins = int(_series["manager_b_wins"])
                                _a_avg = float(_series["manager_a_avg_score"])
                                _b_avg = float(_series["manager_b_avg_score"])
                                _a_diff = float(_series["avg_point_diff"])
                            else:
                                _a_wins = int(_series["manager_b_wins"])
                                _b_wins = int(_series["manager_a_wins"])
                                _a_avg = float(_series["manager_b_avg_score"])
                                _b_avg = float(_series["manager_a_avg_score"])
                                _a_diff = -float(_series["avg_point_diff"])
                            _ties = int(_series["ties"])
                            if _a_wins > _b_wins:
                                _series_leader = _manager_a
                            elif _b_wins > _a_wins:
                                _series_leader = _manager_b
                            else:
                                _series_leader = "Series tied"
                            if _a_diff > 0:
                                _point_edge_manager = _manager_a
                            elif _a_diff < 0:
                                _point_edge_manager = _manager_b
                            else:
                                _point_edge_manager = None
                            _streak_manager = _series["current_streak_manager"]
                            _streak_count = int(_series["current_streak"])
                            _streak_value = (
                                "Latest game tied"
                                if _streak_manager == "Tie"
                                else f"{_streak_manager} · W{_streak_count}"
                            )

                            _series_ink = (
                                "#93A0B1" if _series_leader == "Series tied"
                                else "#35D08A"
                            )
                            st.markdown(
                                "<div class='jsa-lh-series' style='font-size:22px;font-weight:700;color:"
                                + _series_ink + ";letter-spacing:-0.02em;margin:4px 0 12px 0;"
                                "overflow-wrap:anywhere;'>"
                                + _html.escape(str(_manager_a)) + " vs "
                                + _html.escape(str(_manager_b)) + "</div>",
                                unsafe_allow_html=True,
                            )
                            _r1, _r2, _r3, _r4 = st.columns(4)
                            with _r1:
                                st.metric(
                                    "Series Record",
                                    f"{_a_wins}–{_b_wins}" + (f"–{_ties}T" if _ties else ""),
                                    (
                                        "Series tied"
                                        if _series_leader == "Series tied"
                                        else f"{_series_leader} leads"
                                    ),
                                    delta_color="off", delta_arrow="off", border=True,
                                )
                            with _r2:
                                st.metric(
                                    f"{_manager_a} Avg Score", f"{_a_avg:.1f} pts",
                                    (
                                        f"{_a_diff:+.2f} per meeting"
                                        if _a_diff else "Even per meeting"
                                    ),
                                    delta_color="off", delta_arrow="off", border=True,
                                )
                            with _r3:
                                st.metric(
                                    f"{_manager_b} Avg Score", f"{_b_avg:.1f} pts",
                                    (
                                        f"{-_a_diff:+.2f} per meeting"
                                        if _a_diff else "Even per meeting"
                                    ),
                                    delta_color="off", delta_arrow="off", border=True,
                                )
                            with _r4:
                                st.metric(
                                    "Current Streak", _streak_value,
                                    f"{int(_series['playoff_meetings'])} playoff meetings",
                                    delta_color="off", delta_arrow="off", border=True,
                                )

                            _pair_mask = (
                                ((_h2h_matchups["team_a"] == _manager_a)
                                 & (_h2h_matchups["team_b"] == _manager_b))
                                | ((_h2h_matchups["team_a"] == _manager_b)
                                   & (_h2h_matchups["team_b"] == _manager_a))
                            )
                            _rivalry_games = _h2h_matchups[_pair_mask].copy()
                            _rivalry_games["season_sort"] = pd.to_numeric(
                                _rivalry_games["season"], errors="coerce"
                            ).fillna(0)
                            _rivalry_games = _rivalry_games.sort_values(
                                ["season_sort", "week"]
                            ).reset_index(drop=True)

                            def _score_for_manager(row, manager):
                                if row["is_tie"]:
                                    return float(row["winner_score"])
                                return float(
                                    row["winner_score"]
                                    if row["winner"] == manager else row["loser_score"]
                                )

                            _rivalry_games["manager_a_score"] = _rivalry_games.apply(
                                lambda row: _score_for_manager(row, _manager_a), axis=1
                            )
                            _rivalry_games["manager_b_score"] = _rivalry_games.apply(
                                lambda row: _score_for_manager(row, _manager_b), axis=1
                            )
                            _rivalry_games["signed_margin"] = (
                                _rivalry_games["manager_a_score"]
                                - _rivalry_games["manager_b_score"]
                            )
                            _rivalry_games["game_label"] = _rivalry_games.apply(
                                lambda row: f"{row['season']} W{int(row['week'])}", axis=1
                            )
                            _bar_colors = [
                                "#38bdf8" if margin > 0 else "#fb7185" if margin < 0 else "#94a3b8"
                                for margin in _rivalry_games["signed_margin"]
                            ]
                            _fig_rivalry = go.Figure(go.Bar(
                                x=_rivalry_games["game_label"],
                                y=_rivalry_games["signed_margin"],
                                marker={
                                    "color": _bar_colors,
                                    "line": {"color": "rgba(255,255,255,0.45)", "width": 1},
                                },
                                customdata=_rivalry_games[[
                                    "manager_a_score", "manager_b_score", "winner", "is_playoff",
                                ]],
                                hovertemplate=(
                                    f"<b>%{{x}}</b><br>{_manager_a}: %{{customdata[0]:.2f}}<br>"
                                    f"{_manager_b}: %{{customdata[1]:.2f}}<br>"
                                    "Winner: %{customdata[2]}<br>Margin: %{y:+.2f}"
                                    "<br>Playoff: %{customdata[3]}<extra></extra>"
                                ),
                            ))
                            _playoff_games = _rivalry_games[_rivalry_games["is_playoff"]]
                            if not _playoff_games.empty:
                                _fig_rivalry.add_trace(go.Scatter(
                                    x=_playoff_games["game_label"],
                                    y=_playoff_games["signed_margin"],
                                    mode="markers",
                                    name="Playoff meeting",
                                    marker={
                                        "symbol": "diamond", "size": 11,
                                        "color": "#fbbf24",
                                        "line": {"color": "#f8fafc", "width": 1},
                                    },
                                    hovertemplate="Playoff meeting<extra></extra>",
                                ))
                            _fig_rivalry.add_hline(
                                y=0, line_color="#94a3b8", line_width=1
                            )
                            _fig_rivalry.update_layout(
                                title=f"{_manager_a} vs {_manager_b}: Game-by-Game Margin",
                                height=460,
                                template="plotly_dark",
                                paper_bgcolor="rgba(0,0,0,0)",
                                plot_bgcolor="rgba(15,23,42,0.36)",
                                margin={"l": 55, "r": 35, "t": 70, "b": 70},
                                showlegend=not _playoff_games.empty,
                                xaxis={
                                    "title": "Meeting",
                                    "tickangle": -35,
                                    "gridcolor": "rgba(148,163,184,0.08)",
                                },
                                yaxis={
                                    "title": f"Margin: {_manager_a} (+) / {_manager_b} (–)",
                                    "gridcolor": "rgba(148,163,184,0.16)",
                                    "zeroline": False,
                                },
                                legend={"orientation": "h", "y": -0.28},
                            )
                            _lh_plotly(_fig_rivalry)

                            _largest_game = _rivalry_games.loc[
                                _rivalry_games["signed_margin"].abs().idxmax()
                            ]
                            if _series_leader == "Series tied":
                                _series_meaning = (
                                    "Neither manager has established a repeatable advantage in the win column"
                                )
                            elif _point_edge_manager == _series_leader:
                                _series_meaning = (
                                    f"{_series_leader} leads both the series and average scoring, so the edge is "
                                    "supported by weekly production rather than only close-game timing"
                                )
                            else:
                                _series_meaning = (
                                    f"{_series_leader} leads the series, but {_point_edge_manager} owns the average "
                                    "scoring edge; the record is therefore less dominant than the W–L line looks"
                                )
                            _recent_meaning = (
                                "the latest meeting ended level"
                                if _streak_manager == "Tie"
                                else (
                                    f"{_streak_manager} has won the last {_streak_count} meetings, creating recent momentum"
                                    if _streak_count > 1
                                    else f"{_streak_manager} won the latest meeting, but there is no multi-game streak"
                                )
                            )
                            _largest_winner = (
                                _manager_a
                                if _largest_game["signed_margin"] > 0 else _manager_b
                            )
                            st.markdown(
                                f" **What it means:** {_series_meaning}. {_recent_meaning.capitalize()}. "
                                f"The rivalry's biggest separation was {_largest_winner}'s "
                                f"{abs(_largest_game['signed_margin']):.2f}-point win in "
                                f"{_largest_game['game_label']}; the rest of the bars show whether that result was "
                                "typical or an isolated blowout."
                            )

                else:
                    _h2h_scope = (
                        _season_filter
                        if _season_filter != "All Time" else "all-time"
                    )
                    st.markdown("#### League Matrix")
                    st.caption(
                        f"Compare every completed series {_h2h_scope}. Includes playoffs; "
                        "scores of 5 points or fewer are excluded. "
                        "Green means the row manager's win rate is higher. "
                        "Rose means the column manager's is."
                    )
                    _h2h_matchups = _all_h2h_matchups
                    if _season_filter != "All Time":
                        _h2h_matchups = _h2h_matchups[
                            _h2h_matchups["season"].astype(str).eq(
                                str(_season_filter)
                            )
                        ].copy()
                    _rivalries = _league_intel.rivalry_summary_frame(_h2h_matchups)
                    if _rivalries.empty:
                        st.info(
                            "No completed head-to-head matchups are available for "
                            "this selection."
                        )
                    else:
                        _rivalry_managers = sorted(
                            set(_rivalries["manager_a"]).union(
                                _rivalries["manager_b"]
                            )
                        )
                        _pair_lookup = {
                            frozenset((row["manager_a"], row["manager_b"])): row
                            for _, row in _rivalries.iterrows()
                        }

                        def _record_for(manager, opponent):
                            row = _pair_lookup.get(frozenset((manager, opponent)))
                            if row is None:
                                return 0, 0, 0
                            if row["manager_a"] == manager:
                                return (
                                    int(row["manager_a_wins"]),
                                    int(row["manager_b_wins"]),
                                    int(row["ties"]),
                                )
                            return (
                                int(row["manager_b_wins"]),
                                int(row["manager_a_wins"]),
                                int(row["ties"]),
                            )

                        _manager_h2h_stats = {}
                        for _manager in _rivalry_managers:
                            _wins = _losses = _ties_total = _games_total = 0
                            for _opponent in _rivalry_managers:
                                if _manager == _opponent:
                                    continue
                                _w, _l, _t = _record_for(_manager, _opponent)
                                _wins += _w
                                _losses += _l
                                _ties_total += _t
                                _games_total += _w + _l + _t
                            _manager_h2h_stats[_manager] = (
                                (_wins + 0.5 * _ties_total) / _games_total
                                if _games_total else 0,
                                _games_total,
                            )
                        _mgrs_sorted = sorted(
                            _rivalry_managers,
                            key=lambda manager: (
                                -_manager_h2h_stats[manager][0],
                                -_manager_h2h_stats[manager][1],
                                manager,
                            ),
                        )
                        _heat_values = []
                        _heat_text = []
                        _heat_games = []
                        _matrix_rows = []
                        for _row_manager in _mgrs_sorted:
                            _z_row = []
                            _text_row = []
                            _games_row = []
                            _matrix_row = {"Manager": _row_manager}
                            for _col_manager in _mgrs_sorted:
                                if _row_manager == _col_manager:
                                    _z_row.append(None)
                                    _text_row.append("—")
                                    _games_row.append(0)
                                    _matrix_row[_col_manager] = "—"
                                    continue
                                _wins, _losses, _ties = _record_for(
                                    _row_manager, _col_manager
                                )
                                _games = _wins + _losses + _ties
                                _edge = (
                                    ((_wins + 0.5 * _ties) / _games * 100) - 50
                                    if _games else None
                                )
                                _record_text = f"{_wins}–{_losses}" if _games else "—"
                                if _games and _ties:
                                    _record_text += f"–{_ties}T"
                                _z_row.append(_edge)
                                _text_row.append(_record_text)
                                _games_row.append(_games)
                                _matrix_row[_col_manager] = _record_text
                            _heat_values.append(_z_row)
                            _heat_text.append(_text_row)
                            _heat_games.append(_games_row)
                            _matrix_rows.append(_matrix_row)

                        _fig_h2h = go.Figure(go.Heatmap(
                            z=_heat_values,
                            x=_mgrs_sorted,
                            y=_mgrs_sorted,
                            text=_heat_text,
                            customdata=_heat_games,
                            texttemplate="%{text}",
                            textfont={"size": 11},
                            hoverongaps=False,
                            zmin=-50,
                            zmax=50,
                            zmid=0,
                            colorscale=[
                                [0, "#9F1239"],
                                [0.5, "#121821"],
                                [1, "#15803D"],
                            ],
                            xgap=2,
                            ygap=2,
                            colorbar={
                                "title": "Win-rate edge",
                                "ticksuffix": " pp",
                            },
                            hovertemplate=(
                                "<b>%{y} vs %{x}</b><br>Record: %{text}<br>"
                                "Win-rate edge: %{z:+.1f} pp<br>Meetings: %{customdata}<extra></extra>"
                            ),
                        ))
                        _fig_h2h.update_layout(
                            title="League-Wide Head-to-Head Dominance",
                            height=max(520, 42 * len(_mgrs_sorted) + 170),
                            template="plotly_dark",
                            paper_bgcolor="rgba(0,0,0,0)",
                            plot_bgcolor="rgba(15,23,42,0.36)",
                            margin={"l": 72, "r": 24, "t": 64, "b": 96},
                            xaxis={
                                "title": "Opponent",
                                "tickangle": -45,
                                "side": "bottom",
                                "automargin": True,
                                "tickfont": {"size": 10},
                            },
                            yaxis={
                                "title": "Manager",
                                "autorange": "reversed",
                                "automargin": True,
                                "tickfont": {"size": 10},
                            },
                        )
                        _lh_plotly(_fig_h2h)

                        _analysis_pairs = _rivalries.copy()
                        _analysis_pairs["series_gap"] = (
                            _analysis_pairs["manager_a_wins"]
                            - _analysis_pairs["manager_b_wins"]
                        ).abs()
                        _analysis_pairs["dominance"] = _analysis_pairs["series_gap"].div(
                            _analysis_pairs["games"]
                        )
                        _established_pairs = _analysis_pairs[
                            _analysis_pairs["games"].ge(3)
                        ]
                        if _established_pairs.empty:
                            _established_pairs = _analysis_pairs
                        _dominant = _established_pairs.sort_values(
                            ["dominance", "games"], ascending=[False, False]
                        ).iloc[0]
                        _competitive = _established_pairs.sort_values(
                            ["series_gap", "games"], ascending=[True, False]
                        ).iloc[0]
                        _dominant_leader = (
                            _dominant["manager_a"]
                            if _dominant["manager_a_wins"] > _dominant["manager_b_wins"]
                            else _dominant["manager_b"]
                        )
                        _dominant_wins = max(
                            int(_dominant["manager_a_wins"]),
                            int(_dominant["manager_b_wins"]),
                        )
                        _dominant_losses = min(
                            int(_dominant["manager_a_wins"]),
                            int(_dominant["manager_b_wins"]),
                        )
                        _winning_opponents = {}
                        for _manager in _mgrs_sorted:
                            _winning_opponents[_manager] = sum(
                                _record_for(_manager, opponent)[0]
                                > _record_for(_manager, opponent)[1]
                                for opponent in _mgrs_sorted if opponent != _manager
                            )
                        _broadest = sorted(
                            _mgrs_sorted,
                            key=lambda manager: (-_winning_opponents[manager], manager),
                        )[0]
                        st.markdown(
                            f" **What it means:** **{_broadest}** owns a winning record against "
                            f"{_winning_opponents[_broadest]} different managers, the broadest matchup edge in "
                            f"this view. The most one-sided established series is **{_dominant_leader}** over "
                            f"**{_dominant['manager_b'] if _dominant_leader == _dominant['manager_a'] else _dominant['manager_a']}** "
                            f"at {_dominant_wins}–{_dominant_losses}, showing a repeated opponent-specific advantage. "
                            f"**{_competitive['manager_a']} vs {_competitive['manager_b']}** is the most balanced "
                            f"established rivalry at {int(_competitive['manager_a_wins'])}–"
                            f"{int(_competitive['manager_b_wins'])}; neither side has separated consistently. "
                            "Green cells favor the row manager, red cells favor the column opponent."
                        )

                        _h2h_df = pd.DataFrame(_matrix_rows).set_index("Manager")
                        with st.expander("View complete head-to-head record matrix"):
                            st.dataframe(_h2h_df, width="stretch")

            # ── Sub-tab D: Report Cards ───────────────────────────────────────
            with _lhD:
                import plotly.graph_objects as go
                from plotly.subplots import make_subplots

                from fantasy import league_intelligence as _league_intel

                _rc_scope = _season_filter if _season_filter != "All Time" else "all-time"
                st.subheader("Manager Report Cards")
                st.caption(
                    f"Peer-ranked regular-season performance for {_rc_scope}. Head-to-head "
                    "records exclude Sleeper median-game bonuses; postseason results are shown separately."
                )

                _manager_performance = _league_intel.manager_performance_frame(_filt_records)
                _report_managers = (
                    sorted(_manager_performance["manager"].tolist())
                    if not _manager_performance.empty else []
                )
                if not _report_managers:
                    st.info("No completed manager games are available for this filter.")
                    _sel_mgr = None
                else:
                    _preferred_manager = st.session_state.get("lh_h2h_manager_a")
                    if _preferred_manager not in _report_managers:
                        _preferred_manager = _manager_performance.sort_values(
                            ["games", "avg_above_league"], ascending=[False, False]
                        ).iloc[0]["manager"]
                    if st.session_state.get("lh_manager") not in _report_managers:
                        st.session_state.pop("lh_manager", None)
                    _sel_mgr = st.selectbox(
                        "Manager",
                        _report_managers,
                        index=_report_managers.index(_preferred_manager),
                        key="lh_manager",
                    )

                _mgr_games_all = [
                    r for r in _filt_records
                    if r["username"] == _sel_mgr
                    and r["score"] > 5 and r["opp_score"] > 5
                ] if _sel_mgr else []

                # Season history always shows full career (not filtered)
                _mgr_season_rows: dict = {}
                for _yr3, _sd3 in _lh["seasons"].items():
                    for _row3 in _sd3["standings"]:
                        if _row3["username"] == _sel_mgr:
                            _mgr_season_rows[_yr3] = _row3

                if not _mgr_games_all:
                    st.info(f"No data for this manager in {_rc_scope}.")
                else:
                    _manager_performance = _manager_performance.copy()
                    _manager_performance["win_rank"] = _manager_performance["win_pct"].rank(
                        method="min", ascending=False
                    ).astype(int)
                    _manager_performance["scoring_rank"] = _manager_performance[
                        "avg_above_league"
                    ].rank(method="min", ascending=False).astype(int)
                    _manager_performance["consistency_rank"] = _manager_performance[
                        "std_dev"
                    ].rank(method="min", ascending=True).astype(int)
                    _profile = _manager_performance[
                        _manager_performance["manager"] == _sel_mgr
                    ].iloc[0]
                    _peer_count = len(_manager_performance)
                    _titles = sum(1 for _sd4 in _filt_seasons.values()
                                  if _sd4["champion"]["username"] == _sel_mgr)
                    _runner_ups = sum(1 for _sd4 in _filt_seasons.values()
                                     if _sd4["runner_up"]["username"] == _sel_mgr)
                    _playoff_apps = sum(
                        1 for _yr3p, _s4 in _mgr_season_rows.items()
                        if _s4.get("playoff_finish") is not None
                        and (_season_filter == "All Time" or _yr3p == _season_filter)
                    )

                    _d1, _d2, _d3, _d4 = st.columns(4)
                    with _d1:
                        _record_value = (
                            f"{int(_profile['wins'])}–{int(_profile['losses'])}"
                            + (f"–{int(_profile['ties'])}T" if _profile["ties"] else "")
                        )
                        st.metric(
                            "Regular-Season Record", _record_value,
                            f"{_profile['win_pct']:.1f}% · rank #{int(_profile['win_rank'])}/{_peer_count}",
                            delta_color="off", delta_arrow="off", border=True,
                        )
                    with _d2:
                        st.metric(
                            "Scoring vs League",
                            f"{_profile['avg_above_league']:+.2f} pts/wk",
                            f"rank #{int(_profile['scoring_rank'])}/{_peer_count}",
                            delta_color="off", delta_arrow="off", border=True,
                        )
                    with _d3:
                        st.metric(
                            "Consistency", f"{_profile['std_dev']:.2f} SD",
                            f"rank #{int(_profile['consistency_rank'])}/{_peer_count} · lower is steadier",
                            delta_color="off", delta_arrow="off", border=True,
                        )
                    with _d4:
                        st.metric(
                            "Postseason Résumé", f"{_titles} titles",
                            f"{_titles + _runner_ups} finals · {_playoff_apps} playoff apps",
                            delta_color="off", delta_arrow="off", border=True,
                        )

                    if _season_filter == "All Time":
                        _trajectory_rows = []
                        for _season in sorted(_filt_seasons):
                            _season_records = [
                                record for record in _game_records
                                if record["season"] == _season
                            ]
                            _season_performance = _league_intel.manager_performance_frame(
                                _season_records
                            )
                            _season_profile = _season_performance[
                                _season_performance["manager"] == _sel_mgr
                            ]
                            if _season_profile.empty:
                                continue
                            _season_profile = _season_profile.iloc[0]
                            _trajectory_rows.append({
                                "Season": str(_season),
                                "Avg Score": float(_season_profile["avg_score"]),
                                "Pts vs League": float(_season_profile["avg_above_league"]),
                                "Win %": float(_season_profile["win_pct"]),
                                "Record": (
                                    f"{int(_season_profile['wins'])}–"
                                    f"{int(_season_profile['losses'])}"
                                ),
                            })
                        _trajectory_df = pd.DataFrame(_trajectory_rows)
                        if not _trajectory_df.empty:
                            _fig_trajectory = make_subplots(
                                specs=[[{"secondary_y": True}]]
                            )
                            _fig_trajectory.add_trace(go.Bar(
                                x=_trajectory_df["Season"],
                                y=_trajectory_df["Pts vs League"],
                                name="Points vs league / week",
                                marker={
                                    "color": [
                                        "#38bdf8" if value >= 0 else "#fb7185"
                                        for value in _trajectory_df["Pts vs League"]
                                    ],
                                    "line": {"color": "rgba(255,255,255,0.45)", "width": 1},
                                },
                                customdata=_trajectory_df[["Avg Score", "Record"]],
                                hovertemplate=(
                                    "<b>%{x}</b><br>Points vs league: %{y:+.2f}/week<br>"
                                    "Avg score: %{customdata[0]:.2f}<br>"
                                    "Record: %{customdata[1]}<extra></extra>"
                                ),
                            ), secondary_y=False)
                            _fig_trajectory.add_trace(go.Scatter(
                                x=_trajectory_df["Season"],
                                y=_trajectory_df["Win %"],
                                name="Head-to-head win rate",
                                mode="lines+markers",
                                line={"color": "#fbbf24", "width": 3},
                                marker={"size": 9},
                                hovertemplate=(
                                    "<b>%{x}</b><br>Win rate: %{y:.1f}%<extra></extra>"
                                ),
                            ), secondary_y=True)
                            _fig_trajectory.add_hline(
                                y=0, line_dash="dot", line_color="#94a3b8",
                                secondary_y=False,
                            )
                            _fig_trajectory.update_layout(
                                title=f"{_sel_mgr}'s Season-by-Season Trajectory",
                                height=470,
                                template="plotly_dark",
                                paper_bgcolor="rgba(0,0,0,0)",
                                plot_bgcolor="rgba(15,23,42,0.36)",
                                margin={"l": 55, "r": 55, "t": 70, "b": 55},
                                legend={"orientation": "h", "y": -0.2},
                                hovermode="x unified",
                            )
                            _fig_trajectory.update_xaxes(
                                title_text="Season",
                                gridcolor="rgba(148,163,184,0.12)",
                            )
                            _fig_trajectory.update_yaxes(
                                title_text="Avg Points vs League / Week",
                                gridcolor="rgba(148,163,184,0.16)",
                                secondary_y=False,
                            )
                            _fig_trajectory.update_yaxes(
                                title_text="Win Rate", range=[0, 100], ticksuffix="%",
                                secondary_y=True,
                            )
                            _lh_plotly(_fig_trajectory)

                            _best_scoring_season = _trajectory_df.loc[
                                _trajectory_df["Pts vs League"].idxmax()
                            ]
                            _best_winning_season = _trajectory_df.loc[
                                _trajectory_df["Win %"].idxmax()
                            ]
                            _first_season = _trajectory_df.iloc[0]
                            _latest_season = _trajectory_df.iloc[-1]
                            _trend_delta = (
                                _latest_season["Pts vs League"]
                                - _first_season["Pts vs League"]
                            )
                            if _best_scoring_season["Season"] == _best_winning_season["Season"]:
                                _peak_meaning = (
                                    f"the scoring peak in {_best_scoring_season['Season']} translated directly "
                                    "into the best win rate"
                                )
                            else:
                                _peak_meaning = (
                                    f"the scoring peak came in {_best_scoring_season['Season']}, but the best "
                                    f"win rate came in {_best_winning_season['Season']}; matchup timing and close "
                                    "games changed the record"
                                )
                            _trend_meaning = (
                                f"weekly scoring relative to the league improved by {_trend_delta:+.2f} points "
                                "from the first completed season to the latest"
                                if _trend_delta >= 0 else
                                f"weekly scoring relative to the league fell by {abs(_trend_delta):.2f} points "
                                "from the first completed season to the latest"
                            )
                            st.markdown(
                                f" **What it means:** For **{_sel_mgr}**, {_peak_meaning}. Across the full "
                                f"timeline, {_trend_meaning}. Bars measure performance against each season's "
                                "own scoring environment, so the direction is not inflated by league-wide scoring changes."
                            )
                    else:
                        _valid_regular = [
                            record for record in _filt_records
                            if not record["is_playoff"]
                            and record["score"] > 5 and record["opp_score"] > 5
                        ]
                        _week_scores: dict = {}
                        for _record in _valid_regular:
                            _week_scores.setdefault(_record["week"], []).append(
                                _record["score"]
                            )
                        _week_averages = {
                            week: sum(scores) / len(scores)
                            for week, scores in _week_scores.items() if scores
                        }
                        _weekly_rows = []
                        for _record in _valid_regular:
                            if _record["username"] != _sel_mgr:
                                continue
                            _result = (
                                "Win" if _record["score"] > _record["opp_score"]
                                else "Loss" if _record["score"] < _record["opp_score"]
                                else "Tie"
                            )
                            _weekly_rows.append({
                                "Week": int(_record["week"]),
                                "Score": float(_record["score"]),
                                "League Avg": float(_week_averages[_record["week"]]),
                                "Vs League": float(
                                    _record["score"] - _week_averages[_record["week"]]
                                ),
                                "Opponent": _record["opp"],
                                "Opponent Score": float(_record["opp_score"]),
                                "Result": _result,
                            })
                        _weekly_df = pd.DataFrame(_weekly_rows).sort_values("Week")
                        if not _weekly_df.empty:
                            _weekly_colors = _weekly_df["Result"].map({
                                "Win": "#38bdf8", "Loss": "#fb7185", "Tie": "#94a3b8",
                            })
                            _fig_weekly = go.Figure(go.Bar(
                                x=_weekly_df["Week"],
                                y=_weekly_df["Vs League"],
                                marker={
                                    "color": _weekly_colors,
                                    "line": {"color": "rgba(255,255,255,0.45)", "width": 1},
                                },
                                customdata=_weekly_df[[
                                    "Score", "League Avg", "Opponent", "Opponent Score", "Result",
                                ]],
                                hovertemplate=(
                                    "<b>Week %{x}</b><br>%{customdata[4]} vs %{customdata[2]}<br>"
                                    "Score: %{customdata[0]:.2f}–%{customdata[3]:.2f}<br>"
                                    "League average: %{customdata[1]:.2f}<br>"
                                    "Points vs league: %{y:+.2f}<extra></extra>"
                                ),
                            ))
                            _fig_weekly.add_hline(
                                y=0, line_dash="dot", line_color="#94a3b8"
                            )
                            _fig_weekly.update_layout(
                                title=f"{_sel_mgr}'s Weekly Performance — {_season_filter}",
                                height=450,
                                template="plotly_dark",
                                paper_bgcolor="rgba(0,0,0,0)",
                                plot_bgcolor="rgba(15,23,42,0.36)",
                                margin={"l": 55, "r": 35, "t": 70, "b": 55},
                                showlegend=False,
                                xaxis={
                                    "title": "Week", "dtick": 1,
                                    "gridcolor": "rgba(148,163,184,0.08)",
                                },
                                yaxis={
                                    "title": "Points Above / Below League Average",
                                    "gridcolor": "rgba(148,163,184,0.16)",
                                },
                            )
                            _lh_plotly(_fig_weekly)

                            _above_weeks = int(_weekly_df["Vs League"].gt(0).sum())
                            _lucky_wins = len(_weekly_df[
                                _weekly_df["Result"].eq("Win")
                                & _weekly_df["Vs League"].lt(0)
                            ])
                            _unlucky_losses = len(_weekly_df[
                                _weekly_df["Result"].eq("Loss")
                                & _weekly_df["Vs League"].gt(0)
                            ])
                            _best_week = _weekly_df.loc[_weekly_df["Vs League"].idxmax()]
                            st.markdown(
                                f" **What it means:** **{_sel_mgr}** scored above the league average in "
                                f"{_above_weeks} of {len(_weekly_df)} completed weeks. The strongest relative "
                                f"performance was Week {int(_best_week['Week'])} at "
                                f"{_best_week['Vs League']:+.2f} points versus the league. "
                                f"{_lucky_wins} below-average wins and {_unlucky_losses} above-average losses "
                                "show how often matchup timing changed the record beyond the manager's scoring output."
                            )

                    _opponent_rows = []
                    for _opponent, _games in pd.DataFrame(_mgr_games_all).groupby("opp"):
                        _wins = int((_games["score"] > _games["opp_score"]).sum())
                        _losses = int((_games["score"] < _games["opp_score"]).sum())
                        _ties = int((_games["score"] == _games["opp_score"]).sum())
                        _opponent_rows.append({
                            "Opponent": _opponent,
                            "Wins": _wins,
                            "Losses": _losses,
                            "Ties": _ties,
                            "Record": f"{_wins}–{_losses}" + (f"–{_ties}T" if _ties else ""),
                            "Meetings": len(_games),
                            "Avg Score": float(_games["score"].mean()),
                            "Avg Point Diff": float(
                                (_games["score"] - _games["opp_score"]).mean()
                            ),
                            "Playoff Meetings": int(_games["is_playoff"].sum()),
                        })
                    _opponent_df = pd.DataFrame(_opponent_rows).sort_values(
                        "Avg Point Diff", ascending=True
                    )
                    if not _opponent_df.empty:
                        _opponent_colors = [
                            "#38bdf8" if value >= 0 else "#fb7185"
                            for value in _opponent_df["Avg Point Diff"]
                        ]
                        _fig_opponents = go.Figure(go.Bar(
                            x=_opponent_df["Avg Point Diff"],
                            y=_opponent_df["Opponent"],
                            orientation="h",
                            text=_opponent_df["Record"],
                            textposition="outside",
                            cliponaxis=False,
                            marker={
                                "color": _opponent_colors,
                                "line": {"color": "rgba(255,255,255,0.45)", "width": 1},
                            },
                            customdata=_opponent_df[[
                                "Meetings", "Avg Score", "Playoff Meetings", "Record",
                            ]],
                            hovertemplate=(
                                "<b>%{y}</b><br>Record: %{customdata[3]}<br>"
                                "Avg point differential: %{x:+.2f}<br>"
                                "Avg score: %{customdata[1]:.2f}<br>"
                                "Meetings: %{customdata[0]}<br>"
                                "Playoff meetings: %{customdata[2]}<extra></extra>"
                            ),
                        ))
                        _opponent_span = max(
                            5.0, float(_opponent_df["Avg Point Diff"].abs().max()) * 1.25
                        )
                        _fig_opponents.add_vline(
                            x=0, line_color="#94a3b8", line_width=1
                        )
                        _fig_opponents.update_layout(
                            title=f"Where {_sel_mgr} Gains and Loses Ground",
                            height=max(420, 38 * len(_opponent_df) + 115),
                            template="plotly_dark",
                            paper_bgcolor="rgba(0,0,0,0)",
                            plot_bgcolor="rgba(15,23,42,0.36)",
                            margin={"l": 25, "r": 80, "t": 70, "b": 55},
                            showlegend=False,
                            xaxis={
                                "title": "Average Point Differential vs Opponent",
                                "range": [-_opponent_span, _opponent_span],
                                "gridcolor": "rgba(148,163,184,0.16)",
                            },
                            yaxis={"title": "", "automargin": True},
                        )
                        _lh_plotly(_fig_opponents)

                        _best_matchup = _opponent_df.loc[
                            _opponent_df["Avg Point Diff"].idxmax()
                        ]
                        _worst_matchup = _opponent_df.loc[
                            _opponent_df["Avg Point Diff"].idxmin()
                        ]
                        _best_conversion = (
                            "and that scoring advantage has converted into a winning record"
                            if _best_matchup["Wins"] > _best_matchup["Losses"]
                            else "but it has not produced a winning record, so close-game timing worked against them"
                        )
                        _worst_conversion = (
                            "the negative scoring margin also shows up in the W–L record"
                            if _worst_matchup["Losses"] > _worst_matchup["Wins"]
                            else "the manager has survived the negative scoring margin in the W–L record"
                        )
                        st.markdown(
                            f" **What it means:** **{_best_matchup['Opponent']}** is {_sel_mgr}'s most favorable "
                            f"scoring matchup at {_best_matchup['Avg Point Diff']:+.2f} points per meeting, "
                            f"{_best_conversion}. **{_worst_matchup['Opponent']}** is the toughest matchup at "
                            f"{_worst_matchup['Avg Point Diff']:+.2f} points per meeting; {_worst_conversion}. "
                            "This separates a true opponent-specific scoring edge from a record created by a few close finishes."
                        )

                    _s_hist = []
                    for _yr4 in sorted(_mgr_season_rows):
                        _row4 = _mgr_season_rows[_yr4]
                        _pf4  = _row4.get("playoff_finish")
                        _fin4 = {1: "🥇 Champion", 2: "🥈 Runner-up", 3: "🥉 3rd"}.get(
                            _pf4, f"{_pf4}th" if _pf4 else "DNQ")
                        _season_records = [
                            record for record in _game_records
                            if record["season"] == _yr4
                        ]
                        _season_performance = _league_intel.manager_performance_frame(
                            _season_records
                        )
                        _season_profile = _season_performance[
                            _season_performance["manager"] == _sel_mgr
                        ]
                        _season_profile = (
                            _season_profile.iloc[0] if not _season_profile.empty else None
                        )
                        _s_hist.append({
                            "Season":    _yr4,
                            "Team Name": _row4.get("team_name") or "—",
                            "Official Record": f"{_row4['wins']}–{_row4['losses']}",
                            "H2H Win %": (
                                float(_season_profile["win_pct"])
                                if _season_profile is not None else None
                            ),
                            "Avg Score": (
                                float(_season_profile["avg_score"])
                                if _season_profile is not None else None
                            ),
                            "Vs League / Wk": (
                                float(_season_profile["avg_above_league"])
                                if _season_profile is not None else None
                            ),
                            "PF":        _row4["fpts"],
                            "PA":        _row4["fpts_against"],
                            "Finish":    _fin4,
                        })
                    with st.expander("View complete career season history"):
                        st.dataframe(
                            pd.DataFrame(_s_hist),
                            hide_index=True,
                            width="stretch",
                            column_config={
                                "H2H Win %": st.column_config.NumberColumn(
                                    "H2H Win %", format="%.1f%%"
                                ),
                                "Avg Score": st.column_config.NumberColumn(
                                    "Avg Score", format="%.2f"
                                ),
                                "Vs League / Wk": st.column_config.NumberColumn(
                                    "Vs League / Wk", format="%+.2f"
                                ),
                                "PF": st.column_config.NumberColumn("PF", format="%.2f"),
                                "PA": st.column_config.NumberColumn("PA", format="%.2f"),
                            },
                        )
                    with st.expander("View complete opponent breakdown"):
                        st.dataframe(
                            _opponent_df.sort_values(
                                "Avg Point Diff", ascending=False
                            ),
                            hide_index=True,
                            width="stretch",
                            column_config={
                                "Avg Score": st.column_config.NumberColumn(
                                    "Avg Score", format="%.2f"
                                ),
                                "Avg Point Diff": st.column_config.NumberColumn(
                                    "Avg Point Diff", format="%+.2f"
                                ),
                            },
                        )

            # ── Sub-tab E: Consistency & Luck ─────────────────────────────────
            with _lhE:
                import plotly.graph_objects as go

                from fantasy import league_intelligence as _league_intel

                _cl_scope = _season_filter if _season_filter != "All Time" else "all seasons"
                st.subheader("Consistency & Schedule Luck")
                st.caption(
                    f"Regular season only · {_cl_scope}. Expected wins use all-play probability: "
                    "how often each weekly score would have beaten every other team that week."
                )

                _cl_df = _league_intel.consistency_luck_frame(_filt_records)
                if _cl_df.empty:
                    st.info("No completed regular-season scores are available for this selection.")
                else:
                    _sample_floor = min(3, int(_cl_df["games"].max()))
                    _eligible_cl = _cl_df[_cl_df["games"].ge(_sample_floor)].copy()
                    _most_consistent = _eligible_cl.loc[
                        _eligible_cl["volatility"].idxmin()
                    ]
                    _most_volatile = _eligible_cl.loc[
                        _eligible_cl["volatility"].idxmax()
                    ]
                    _most_fortunate = _eligible_cl.loc[
                        _eligible_cl["luck_delta"].idxmax()
                    ]
                    _most_unfortunate = _eligible_cl.loc[
                        _eligible_cl["luck_delta"].idxmin()
                    ]

                    _c1, _c2, _c3, _c4 = st.columns(4)
                    with _c1:
                        st.metric(
                            "Most Consistent", _most_consistent["manager"],
                            f"{_most_consistent['volatility']:.2f} adjusted SD",
                            delta_color="off", delta_arrow="off", border=True,
                        )
                    with _c2:
                        st.metric(
                            "Most Volatile", _most_volatile["manager"],
                            f"{_most_volatile['volatility']:.2f} adjusted SD",
                            delta_color="off", delta_arrow="off", border=True,
                        )
                    with _c3:
                        st.metric(
                            "Most Fortunate", _most_fortunate["manager"],
                            f"{_most_fortunate['luck_delta']:+.2f} wins vs expected",
                            delta_color="off", delta_arrow="off", border=True,
                        )
                    with _c4:
                        st.metric(
                            "Most Unfortunate", _most_unfortunate["manager"],
                            f"{_most_unfortunate['luck_delta']:+.2f} wins vs expected",
                            delta_color="off", delta_arrow="off", border=True,
                        )

                    _median_volatility = float(_cl_df["volatility"].median())
                    _luck_scale = max(1.0, float(_cl_df["luck_delta"].abs().max()))
                    _bubble_sizes = 12 + _cl_df["games"].astype(float).pow(0.5) * 3
                    _fig_consistency = go.Figure(go.Scatter(
                        x=_cl_df["avg_above_league"],
                        y=_cl_df["volatility"],
                        text=_cl_df["manager"],
                        customdata=_cl_df[[
                            "games", "avg_score", "actual_wins", "expected_wins", "luck_delta",
                        ]],
                        mode="markers+text",
                        textposition="top center",
                        marker={
                            "size": _bubble_sizes,
                            "color": _cl_df["luck_delta"],
                            "colorscale": [
                                [0, "#fb7185"],
                                [0.5, "#64748b"],
                                [1, "#34d399"],
                            ],
                            "cmin": -_luck_scale,
                            "cmax": _luck_scale,
                            "colorbar": {"title": "Wins vs expected"},
                            "line": {"color": "rgba(255,255,255,0.7)", "width": 1},
                            "opacity": 0.9,
                        },
                        hovertemplate=(
                            "<b>%{text}</b><br>Points vs league: %{x:+.2f}/week<br>"
                            "Adjusted volatility: %{y:.2f}<br>Avg score: %{customdata[1]:.2f}<br>"
                            "Actual wins: %{customdata[2]:.2f}<br>"
                            "Expected wins: %{customdata[3]:.2f}<br>"
                            "Luck delta: %{customdata[4]:+.2f}<br>Games: %{customdata[0]}<extra></extra>"
                        ),
                    ))
                    _fig_consistency.add_vline(
                        x=0, line_dash="dot", line_color="#94a3b8"
                    )
                    _fig_consistency.add_hline(
                        y=_median_volatility, line_dash="dot", line_color="#94a3b8"
                    )
                    _fig_consistency.update_layout(
                        title="Scoring Quality vs Week-to-Week Volatility",
                        height=520,
                        template="plotly_dark",
                        paper_bgcolor="rgba(0,0,0,0)",
                        plot_bgcolor="rgba(15,23,42,0.36)",
                        margin={"l": 55, "r": 45, "t": 70, "b": 55},
                        showlegend=False,
                        xaxis={
                            "title": "Average Points Above League / Week",
                            "gridcolor": "rgba(148,163,184,0.16)",
                        },
                        yaxis={
                            "title": "Adjusted Scoring Volatility (Lower = Steadier)",
                            "rangemode": "tozero",
                            "gridcolor": "rgba(148,163,184,0.16)",
                        },
                    )
                    _lh_plotly(_fig_consistency)

                    _top_scorer = _cl_df.loc[_cl_df["avg_above_league"].idxmax()]
                    _above_average = _eligible_cl[_eligible_cl["avg_above_league"].gt(0)]
                    _reliable = (
                        _above_average.loc[_above_average["volatility"].idxmin()]
                        if not _above_average.empty else _most_consistent
                    )
                    _top_scorer_style = (
                        "steadier than the league median"
                        if _top_scorer["volatility"] <= _median_volatility
                        else "more volatile than the league median"
                    )
                    _consistency_caveat = (
                        f"**{_most_consistent['manager']}** is the steadiest manager, but their "
                        f"{_most_consistent['avg_above_league']:+.2f} scoring margin shows that consistency "
                        "does not automatically mean strong scoring."
                        if _most_consistent["avg_above_league"] <= 0 else
                        f"**{_most_consistent['manager']}** combines the league's lowest volatility with "
                        f"above-average scoring, a genuinely dependable profile."
                    )
                    st.markdown(
                        f" **What it means:** **{_top_scorer['manager']}** has the strongest weekly scoring "
                        f"edge at {_top_scorer['avg_above_league']:+.2f} points and is {_top_scorer_style}. "
                        f"Among above-average scorers, **{_reliable['manager']}** is the most reliable, meaning "
                        "their advantage is less dependent on occasional spike weeks. "
                        f"{_consistency_caveat} The lower-right area is the ideal combination: strong and steady."
                    )

                    _luck_sorted = _cl_df.sort_values("luck_delta", ascending=True).copy()
                    _luck_text = [
                        f"{actual:.1f} actual / {expected:.1f} expected"
                        for actual, expected in zip(
                            _luck_sorted["actual_wins"], _luck_sorted["expected_wins"]
                        )
                    ]
                    _fig_luck = go.Figure(go.Bar(
                        x=_luck_sorted["luck_delta"],
                        y=_luck_sorted["manager"],
                        orientation="h",
                        text=_luck_text,
                        textposition="outside",
                        cliponaxis=False,
                        marker={
                            "color": [
                                "#34d399" if value >= 0 else "#fb7185"
                                for value in _luck_sorted["luck_delta"]
                            ],
                            "line": {"color": "rgba(255,255,255,0.45)", "width": 1},
                        },
                        customdata=_luck_sorted[[
                            "actual_win_pct", "expected_win_pct", "below_avg_wins",
                            "above_avg_losses", "games",
                        ]],
                        hovertemplate=(
                            "<b>%{y}</b><br>Wins vs expected: %{x:+.2f}<br>"
                            "Actual win rate: %{customdata[0]:.1f}%<br>"
                            "All-play expected rate: %{customdata[1]:.1f}%<br>"
                            "Below-average wins: %{customdata[2]}<br>"
                            "Above-average losses: %{customdata[3]}<br>"
                            "Games: %{customdata[4]}<extra></extra>"
                        ),
                    ))
                    _luck_span = max(1.0, float(_luck_sorted["luck_delta"].abs().max()) * 1.3)
                    _fig_luck.add_vline(x=0, line_color="#94a3b8", line_width=1)
                    _fig_luck.update_layout(
                        title="Schedule Luck: Actual Wins Minus All-Play Expected Wins",
                        height=max(430, 40 * len(_luck_sorted) + 120),
                        template="plotly_dark",
                        paper_bgcolor="rgba(0,0,0,0)",
                        plot_bgcolor="rgba(15,23,42,0.36)",
                        margin={"l": 25, "r": 165, "t": 70, "b": 55},
                        showlegend=False,
                        xaxis={
                            "title": "Actual Wins − Expected Wins",
                            "range": [-_luck_span, _luck_span],
                            "gridcolor": "rgba(148,163,184,0.16)",
                        },
                        yaxis={"title": "", "automargin": True},
                    )
                    _lh_plotly(_fig_luck)

                    _most_aligned = _cl_df.loc[_cl_df["luck_delta"].abs().idxmin()]
                    _fortunate_meaning = (
                        f"won roughly {_most_fortunate['luck_delta']:.1f} more games than their weekly scores "
                        "would predict, so favorable opponent timing boosted the record"
                        if _most_fortunate["luck_delta"] > 0 else
                        "finished almost exactly where their weekly scoring predicted"
                    )
                    _unfortunate_gap = abs(float(_most_unfortunate["luck_delta"]))
                    _unfortunate_meaning = (
                        f"won roughly {_unfortunate_gap:.1f} fewer games than expected, so their record "
                        "understates the quality of their weekly scores"
                        if _most_unfortunate["luck_delta"] < 0 else
                        "also finished close to expectation"
                    )
                    st.markdown(
                        f" **What it means:** **{_most_fortunate['manager']}** {_fortunate_meaning}. "
                        f"**{_most_unfortunate['manager']}** {_unfortunate_meaning}. "
                        f"**{_most_aligned['manager']}** is closest to zero, meaning their actual record most "
                        "closely matches what an all-play schedule would expect. Luck here describes matchup "
                        "timing—not whether the wins count or whether roster decisions were good."
                    )

                    _cl_details = _cl_df.rename(columns={
                        "manager": "Manager",
                        "games": "Games",
                        "avg_score": "Avg Score",
                        "avg_above_league": "Pts vs League / Wk",
                        "volatility": "Adjusted Volatility",
                        "actual_wins": "Actual Wins",
                        "expected_wins": "Expected Wins",
                        "luck_delta": "Wins vs Expected",
                        "actual_win_pct": "Actual Win %",
                        "expected_win_pct": "Expected Win %",
                        "below_avg_wins": "Below-Avg Wins",
                        "above_avg_losses": "Above-Avg Losses",
                    })
                    with st.expander("View complete consistency and luck metrics"):
                        st.dataframe(
                            _cl_details,
                            hide_index=True,
                            width="stretch",
                            column_config={
                                "Avg Score": st.column_config.NumberColumn(
                                    "Avg Score", format="%.2f"
                                ),
                                "Pts vs League / Wk": st.column_config.NumberColumn(
                                    "Pts vs League / Wk", format="%+.2f"
                                ),
                                "Adjusted Volatility": st.column_config.NumberColumn(
                                    "Adjusted Volatility", format="%.2f",
                                    help="Standard deviation of weekly points relative to that week's league average",
                                ),
                                "Actual Wins": st.column_config.NumberColumn(
                                    "Actual Wins", format="%.2f"
                                ),
                                "Expected Wins": st.column_config.NumberColumn(
                                    "Expected Wins", format="%.2f"
                                ),
                                "Wins vs Expected": st.column_config.NumberColumn(
                                    "Wins vs Expected", format="%+.2f"
                                ),
                                "Actual Win %": st.column_config.NumberColumn(
                                    "Actual Win %", format="%.1f%%"
                                ),
                                "Expected Win %": st.column_config.NumberColumn(
                                    "Expected Win %", format="%.1f%%"
                                ),
                            },
                        )

            # ── Sub-tab F: Score Trends ───────────────────────────────────────
            with _lhF:
                import plotly.graph_objects as go

                from fantasy import league_intelligence as _league_intel

                _trend_scope = (
                    _season_filter if _season_filter != "All Time" else "all seasons"
                )
                st.subheader("Score Trends")
                st.caption(
                    f"Regular season only · {_trend_scope}. Raw scoring shows the league environment; "
                    "adjusted scoring shows how far each manager finished above or below that same week's field."
                )

                _score_context = _league_intel.weekly_score_context_frame(_filt_records)
                if _score_context.empty:
                    st.info("No completed regular-season scores are available for this selection.")
                elif _season_filter == "All Time":
                    _season_environment = (
                        _score_context.groupby("season", as_index=False)
                        .agg(
                            average=("score", "mean"),
                            median=("score", "median"),
                            q1=("score", lambda values: values.quantile(0.25)),
                            q3=("score", lambda values: values.quantile(0.75)),
                            completed_weeks=("week", "nunique"),
                        )
                        .sort_values("season")
                    )
                    _manager_seasons = (
                        _score_context.groupby(["manager", "season"], as_index=False)
                        .agg(
                            avg_score=("score", "mean"),
                            avg_adjusted=("adjusted_score", "mean"),
                            games=("week", "nunique"),
                        )
                        .sort_values(["manager", "season"])
                    )
                    _manager_summary = (
                        _manager_seasons.groupby("manager", as_index=False)
                        .agg(
                            seasons=("season", "nunique"),
                            above_average_seasons=(
                                "avg_adjusted", lambda values: int(values.gt(0).sum())
                            ),
                            career_adjusted=("avg_adjusted", "mean"),
                        )
                        .sort_values(
                            ["above_average_seasons", "career_adjusted"],
                            ascending=[False, False],
                        )
                    )
                    _changes = []
                    for _trend_manager, _trend_group in _manager_seasons.groupby("manager"):
                        _trend_group = _trend_group.sort_values("season")
                        if len(_trend_group) >= 2:
                            _changes.append({
                                "manager": _trend_manager,
                                "first_season": _trend_group.iloc[0]["season"],
                                "latest_season": _trend_group.iloc[-1]["season"],
                                "change": float(
                                    _trend_group.iloc[-1]["avg_adjusted"]
                                    - _trend_group.iloc[0]["avg_adjusted"]
                                ),
                            })
                    _changes_df = pd.DataFrame(_changes)

                    _league_average = float(_score_context["score"].mean())
                    _highest_season = _season_environment.loc[
                        _season_environment["average"].idxmax()
                    ]
                    _steady_leader = _manager_summary.iloc[0]
                    _riser = (
                        _changes_df.loc[_changes_df["change"].idxmax()]
                        if not _changes_df.empty else None
                    )

                    _t1, _t2, _t3, _t4 = st.columns(4)
                    with _t1:
                        st.metric(
                            "League Average", f"{_league_average:.2f} pts",
                            f"{int(_score_context.groupby(['season', 'week']).ngroups)} completed weeks",
                            delta_color="off", delta_arrow="off", border=True,
                        )
                    with _t2:
                        st.metric(
                            "Highest-Scoring Season", str(_highest_season["season"]),
                            f"{_highest_season['average']:.2f} pts / team-week",
                            delta_color="off", delta_arrow="off", border=True,
                        )
                    with _t3:
                        if _riser is not None and _riser["change"] > 0:
                            st.metric(
                                "Biggest Riser", _riser["manager"],
                                f"{_riser['change']:+.2f} pts / week",
                                delta_color="off", delta_arrow="off", border=True,
                            )
                        else:
                            _best_change = (
                                f"{_riser['change']:+.2f} best change"
                                if _riser is not None else "Needs 2 seasons"
                            )
                            st.metric(
                                "Biggest Riser", "No positive riser",
                                _best_change,
                                delta_color="off", delta_arrow="off", border=True,
                            )
                    with _t4:
                        st.metric(
                            "Most Often Above Average", _steady_leader["manager"],
                            f"{int(_steady_leader['above_average_seasons'])} of "
                            f"{int(_steady_leader['seasons'])} seasons",
                            delta_color="off", delta_arrow="off", border=True,
                        )

                    _fig_environment = go.Figure()
                    _fig_environment.add_trace(go.Scatter(
                        x=_season_environment["season"],
                        y=_season_environment["q1"],
                        mode="lines",
                        line={"color": "rgba(96,165,250,0)"},
                        hoverinfo="skip",
                        showlegend=False,
                    ))
                    _fig_environment.add_trace(go.Scatter(
                        x=_season_environment["season"],
                        y=_season_environment["q3"],
                        mode="lines",
                        name="Middle 50% of scores",
                        fill="tonexty",
                        fillcolor="rgba(96,165,250,0.16)",
                        line={"color": "rgba(96,165,250,0)"},
                        hovertemplate="%{x}<br>75th percentile: %{y:.2f}<extra></extra>",
                    ))
                    _fig_environment.add_trace(go.Scatter(
                        x=_season_environment["season"],
                        y=_season_environment["average"],
                        name="League average",
                        mode="lines+markers",
                        line={"color": "#34d399", "width": 3},
                        marker={"size": 8},
                        hovertemplate="%{x}<br>Average: %{y:.2f}<extra></extra>",
                    ))
                    _fig_environment.add_trace(go.Scatter(
                        x=_season_environment["season"],
                        y=_season_environment["median"],
                        name="League median",
                        mode="lines+markers",
                        line={"color": "#fbbf24", "width": 2, "dash": "dot"},
                        marker={"size": 7},
                        hovertemplate="%{x}<br>Median: %{y:.2f}<extra></extra>",
                    ))
                    _fig_environment.update_layout(
                        title="League Scoring Environment by Season",
                        height=470,
                        template="plotly_dark",
                        paper_bgcolor="rgba(0,0,0,0)",
                        plot_bgcolor="rgba(15,23,42,0.36)",
                        margin={"l": 55, "r": 30, "t": 70, "b": 55},
                        hovermode="x unified",
                        legend={"orientation": "h", "y": 1.12, "x": 0},
                        xaxis={"title": "Season", "gridcolor": "rgba(148,163,184,0.12)"},
                        yaxis={
                            "title": "Points per Team-Week",
                            "gridcolor": "rgba(148,163,184,0.16)",
                        },
                    )
                    _lh_plotly(_fig_environment)

                    _first_environment = _season_environment.iloc[0]
                    _latest_environment = _season_environment.iloc[-1]
                    _lowest_season = _season_environment.loc[
                        _season_environment["average"].idxmin()
                    ]
                    _environment_change = float(
                        _latest_environment["average"] - _first_environment["average"]
                    )
                    if len(_season_environment) == 1:
                        _environment_meaning = (
                            "Only one completed season is available, so this establishes a baseline rather "
                            "than a scoring trend."
                        )
                    elif abs(_environment_change) < 1:
                        _environment_meaning = (
                            "The first and latest seasons are within one point, so the scoring baseline has "
                            "been broadly stable."
                        )
                    elif _environment_change > 0:
                        _environment_meaning = (
                            f"The latest season runs {_environment_change:.2f} points higher than the first, "
                            "so recent raw scores are easier to inflate without necessarily indicating better management."
                        )
                    else:
                        _environment_meaning = (
                            f"The latest season runs {abs(_environment_change):.2f} points lower than the first, "
                            "so recent raw totals understate performance relative to the earlier scoring environment."
                        )
                    st.markdown(
                        f" **What it means:** **{_highest_season['season']}** was the most generous scoring "
                        f"environment at {_highest_season['average']:.2f} points per team-week; "
                        f"**{_lowest_season['season']}** was the lowest at {_lowest_season['average']:.2f}. "
                        f"{_environment_meaning} The shaded band shows the middle half of weekly scores, so "
                        "a wider band means outcomes were more spread out—not simply higher."
                    )

                    _heatmap = _manager_seasons.pivot(
                        index="manager", columns="season", values="avg_adjusted"
                    )
                    _raw_heatmap = _manager_seasons.pivot(
                        index="manager", columns="season", values="avg_score"
                    ).reindex(index=_heatmap.index, columns=_heatmap.columns)
                    _latest_column = _heatmap.columns[-1]
                    _row_order = (
                        _heatmap.assign(
                            _latest=_heatmap[_latest_column],
                            _career=_heatmap.mean(axis=1),
                        )
                        .sort_values(["_latest", "_career"], ascending=False, na_position="last")
                        .index
                    )
                    _heatmap = _heatmap.reindex(_row_order)
                    _raw_heatmap = _raw_heatmap.reindex(_row_order)
                    _heat_limit = max(1.0, float(_heatmap.abs().max().max()))
                    _heat_text = _heatmap.map(
                        lambda value: "" if pd.isna(value) else f"{value:+.1f}"
                    )
                    _fig_heatmap = go.Figure(go.Heatmap(
                        z=_heatmap.values,
                        x=[str(value) for value in _heatmap.columns],
                        y=_heatmap.index.tolist(),
                        text=_heat_text.values,
                        texttemplate="%{text}",
                        customdata=_raw_heatmap.values,
                        colorscale=[
                            [0, "#be123c"],
                            [0.5, "#334155"],
                            [1, "#059669"],
                        ],
                        zmin=-_heat_limit,
                        zmax=_heat_limit,
                        zmid=0,
                        xgap=2,
                        ygap=2,
                        colorbar={"title": "Pts vs league / week"},
                        hovertemplate=(
                            "<b>%{y}</b> · %{x}<br>Points vs league: %{z:+.2f}/week<br>"
                            "Raw average: %{customdata:.2f}<extra></extra>"
                        ),
                    ))
                    _fig_heatmap.update_layout(
                        title="Manager Performance After Adjusting for Each Week's Scoring Level",
                        height=max(470, 36 * len(_heatmap) + 130),
                        template="plotly_dark",
                        paper_bgcolor="rgba(0,0,0,0)",
                        plot_bgcolor="rgba(15,23,42,0.36)",
                        margin={"l": 25, "r": 80, "t": 70, "b": 50},
                        xaxis={"title": "Season", "side": "top"},
                        yaxis={"title": "", "automargin": True, "autorange": "reversed"},
                    )
                    _lh_plotly(_fig_heatmap)

                    _latest_scores = _manager_seasons[
                        _manager_seasons["season"].eq(_latest_column)
                    ]
                    _latest_leader = _latest_scores.loc[
                        _latest_scores["avg_adjusted"].idxmax()
                    ]
                    if _riser is None:
                        _riser_meaning = (
                            "There is not yet enough multi-season manager history to identify improvement."
                        )
                    elif _riser["change"] > 0:
                        _riser_meaning = (
                            f"**{_riser['manager']}** is the biggest riser, improving "
                            f"{_riser['change']:.2f} adjusted points per week from "
                            f"{_riser['first_season']} to {_riser['latest_season']}; that is real peer-relative "
                            "improvement rather than a league-wide scoring bump."
                        )
                    else:
                        _riser_meaning = (
                            "No returning manager improved their first-to-latest adjusted average; the least "
                            f"negative change belongs to **{_riser['manager']}** at {_riser['change']:+.2f}."
                        )
                    st.markdown(
                        f" **What it means:** In **{_latest_column}**, **{_latest_leader['manager']}** led the "
                        f"league by scoring {_latest_leader['avg_adjusted']:+.2f} points per week versus the "
                        "same-week field. "
                        f"{_riser_meaning} **{_steady_leader['manager']}** posted an above-average season "
                        f"{int(_steady_leader['above_average_seasons'])} time(s), the most in this history. "
                        "Green cells represent a repeatable weekly edge; red cells mean raw points were below "
                        "what that season's exact weeks required."
                    )

                    _season_details = _season_environment.rename(columns={
                        "season": "Season", "average": "League Average",
                        "median": "League Median", "q1": "25th Percentile",
                        "q3": "75th Percentile", "completed_weeks": "Completed Weeks",
                    })
                    _manager_details = _manager_seasons.rename(columns={
                        "manager": "Manager", "season": "Season",
                        "avg_score": "Raw Avg Score", "avg_adjusted": "Pts vs League / Wk",
                        "games": "Games",
                    })
                    with st.expander("View complete score trend data"):
                        st.markdown("**League scoring environment**")
                        st.dataframe(_season_details, hide_index=True, width="stretch")
                        st.markdown("**Manager-season performance**")
                        st.dataframe(
                            _manager_details.sort_values(
                                ["Season", "Pts vs League / Wk"], ascending=[False, False]
                            ),
                            hide_index=True,
                            width="stretch",
                            column_config={
                                "Raw Avg Score": st.column_config.NumberColumn(format="%.2f"),
                                "Pts vs League / Wk": st.column_config.NumberColumn(format="%+.2f"),
                            },
                        )
                else:
                    _weekly_environment = (
                        _score_context.groupby("week", as_index=False)
                        .agg(
                            average=("score", "mean"),
                            median=("score", "median"),
                            q1=("score", lambda values: values.quantile(0.25)),
                            q3=("score", lambda values: values.quantile(0.75)),
                            teams=("manager", "nunique"),
                        )
                        .sort_values("week")
                    )
                    _manager_totals = (
                        _score_context.groupby("manager", as_index=False)
                        .agg(
                            avg_score=("score", "mean"),
                            total_adjusted=("adjusted_score", "sum"),
                            avg_adjusted=("adjusted_score", "mean"),
                            games=("week", "nunique"),
                        )
                        .sort_values("total_adjusted", ascending=False)
                    )
                    _last_four_weeks = sorted(_score_context["week"].unique())[-4:]
                    _recent_form = (
                        _score_context[_score_context["week"].isin(_last_four_weeks)]
                        .groupby("manager", as_index=False)
                        .agg(avg_adjusted=("adjusted_score", "mean"), games=("week", "nunique"))
                        .sort_values("avg_adjusted", ascending=False)
                    )
                    _league_average = float(_score_context["score"].mean())
                    _highest_week = _weekly_environment.loc[
                        _weekly_environment["average"].idxmax()
                    ]
                    _season_leader = _manager_totals.iloc[0]
                    _recent_leader = _recent_form.iloc[0]

                    _t1, _t2, _t3, _t4 = st.columns(4)
                    with _t1:
                        st.metric(
                            "League Average", f"{_league_average:.2f} pts",
                            f"{len(_weekly_environment)} completed weeks",
                            delta_color="off", delta_arrow="off", border=True,
                        )
                    with _t2:
                        st.metric(
                            "Highest-Scoring Week", f"Week {int(_highest_week['week'])}",
                            f"{_highest_week['average']:.2f} pts / team",
                            delta_color="off", delta_arrow="off", border=True,
                        )
                    with _t3:
                        st.metric(
                            "Adjusted Scoring Leader", _season_leader["manager"],
                            f"{_season_leader['total_adjusted']:+.2f} cumulative pts",
                            delta_color="off", delta_arrow="off", border=True,
                        )
                    with _t4:
                        st.metric(
                            "Best Recent Form", _recent_leader["manager"],
                            f"{_recent_leader['avg_adjusted']:+.2f} / week · last "
                            f"{len(_last_four_weeks)}",
                            delta_color="off", delta_arrow="off", border=True,
                        )

                    _fig_weekly_environment = go.Figure()
                    _fig_weekly_environment.add_trace(go.Scatter(
                        x=_weekly_environment["week"],
                        y=_weekly_environment["q1"],
                        mode="lines",
                        line={"color": "rgba(96,165,250,0)"},
                        hoverinfo="skip",
                        showlegend=False,
                    ))
                    _fig_weekly_environment.add_trace(go.Scatter(
                        x=_weekly_environment["week"],
                        y=_weekly_environment["q3"],
                        mode="lines",
                        name="Middle 50% of scores",
                        fill="tonexty",
                        fillcolor="rgba(96,165,250,0.16)",
                        line={"color": "rgba(96,165,250,0)"},
                        hovertemplate="Week %{x}<br>75th percentile: %{y:.2f}<extra></extra>",
                    ))
                    _fig_weekly_environment.add_trace(go.Scatter(
                        x=_weekly_environment["week"],
                        y=_weekly_environment["average"],
                        mode="lines+markers",
                        name="League average",
                        line={"color": "#34d399", "width": 3},
                        marker={"size": 7},
                        hovertemplate="Week %{x}<br>Average: %{y:.2f}<extra></extra>",
                    ))
                    _fig_weekly_environment.add_trace(go.Scatter(
                        x=_weekly_environment["week"],
                        y=_weekly_environment["median"],
                        mode="lines+markers",
                        name="League median",
                        line={"color": "#fbbf24", "width": 2, "dash": "dot"},
                        marker={"size": 6},
                        hovertemplate="Week %{x}<br>Median: %{y:.2f}<extra></extra>",
                    ))
                    _fig_weekly_environment.update_layout(
                        title=f"{_season_filter} Weekly Scoring Environment",
                        height=470,
                        template="plotly_dark",
                        paper_bgcolor="rgba(0,0,0,0)",
                        plot_bgcolor="rgba(15,23,42,0.36)",
                        margin={"l": 55, "r": 30, "t": 70, "b": 55},
                        hovermode="x unified",
                        legend={"orientation": "h", "y": 1.12, "x": 0},
                        xaxis={
                            "title": "Week", "dtick": 1,
                            "gridcolor": "rgba(148,163,184,0.12)",
                        },
                        yaxis={
                            "title": "Points per Team",
                            "gridcolor": "rgba(148,163,184,0.16)",
                        },
                    )
                    _lh_plotly(_fig_weekly_environment)

                    _lowest_week = _weekly_environment.loc[
                        _weekly_environment["average"].idxmin()
                    ]
                    _week_gap = float(_highest_week["average"] - _lowest_week["average"])
                    _latest_week = _weekly_environment.iloc[-1]
                    _latest_spread = float(_latest_week["q3"] - _latest_week["q1"])
                    if len(_weekly_environment) == 1:
                        _weekly_meaning = (
                            "One completed week establishes the baseline; it is too early to call the season "
                            "high- or low-scoring."
                        )
                    else:
                        _weekly_meaning = (
                            f"The {_week_gap:.2f}-point gap between the highest and lowest league averages "
                            "shows how much opponent-independent scoring conditions changed week to week."
                        )
                    st.markdown(
                        f" **What it means:** Week **{int(_highest_week['week'])}** produced the most scoring "
                        f"at {_highest_week['average']:.2f} points per team; Week "
                        f"**{int(_lowest_week['week'])}** produced the least at {_lowest_week['average']:.2f}. "
                        f"{_weekly_meaning} In the latest completed week, the middle half of teams were spread "
                        f"across {_latest_spread:.2f} points, which describes competitive separation rather "
                        "than schedule luck."
                    )

                    _cumulative = _score_context.sort_values(["manager", "week"]).copy()
                    _cumulative["cumulative_adjusted"] = _cumulative.groupby(
                        "manager"
                    )["adjusted_score"].cumsum()
                    _preferred_trend_manager = st.session_state.get("lh_manager")
                    if _preferred_trend_manager not in set(_cumulative["manager"]):
                        _preferred_trend_manager = _season_leader["manager"]
                    _trend_colors = [
                        "#60a5fa", "#fbbf24", "#a78bfa", "#fb7185", "#22d3ee",
                        "#f97316", "#c084fc", "#2dd4bf", "#f472b6", "#94a3b8",
                    ]
                    _fig_cumulative = go.Figure()
                    for _trend_index, (_trend_manager, _manager_weeks) in enumerate(
                        _cumulative.groupby("manager", sort=True)
                    ):
                        _is_highlighted = _trend_manager == _preferred_trend_manager
                        _fig_cumulative.add_trace(go.Scatter(
                            x=_manager_weeks["week"],
                            y=_manager_weeks["cumulative_adjusted"],
                            name=_trend_manager,
                            mode="lines+markers",
                            line={
                                "color": "#34d399" if _is_highlighted else _trend_colors[
                                    _trend_index % len(_trend_colors)
                                ],
                                "width": 4 if _is_highlighted else 2,
                            },
                            marker={"size": 7 if _is_highlighted else 5},
                            opacity=1 if _is_highlighted else 0.72,
                            customdata=_manager_weeks[["score", "league_average", "adjusted_score"]],
                            hovertemplate=(
                                "<b>%{fullData.name}</b> · Week %{x}<br>"
                                "Cumulative vs league: %{y:+.2f}<br>Score: %{customdata[0]:.2f}<br>"
                                "League average: %{customdata[1]:.2f}<br>Weekly edge: "
                                "%{customdata[2]:+.2f}<extra></extra>"
                            ),
                        ))
                    _fig_cumulative.add_hline(
                        y=0, line_dash="dot", line_color="#94a3b8", line_width=1
                    )
                    _fig_cumulative.update_layout(
                        title="Cumulative Points Above or Below the Weekly League Average",
                        height=520,
                        template="plotly_dark",
                        paper_bgcolor="rgba(0,0,0,0)",
                        plot_bgcolor="rgba(15,23,42,0.36)",
                        margin={"l": 55, "r": 30, "t": 70, "b": 55},
                        hovermode="closest",
                        legend={"title": "Manager"},
                        xaxis={
                            "title": "Week", "dtick": 1,
                            "gridcolor": "rgba(148,163,184,0.12)",
                        },
                        yaxis={
                            "title": "Cumulative Points vs League",
                            "gridcolor": "rgba(148,163,184,0.16)",
                        },
                    )
                    _lh_plotly(_fig_cumulative)

                    _runner_up = _manager_totals.iloc[1] if len(_manager_totals) > 1 else None
                    _lead_context = (
                        f", {_season_leader['total_adjusted'] - _runner_up['total_adjusted']:.2f} points "
                        f"ahead of **{_runner_up['manager']}**"
                        if _runner_up is not None else ""
                    )
                    _highlight_row = _manager_totals[
                        _manager_totals["manager"].eq(_preferred_trend_manager)
                    ].iloc[0]
                    _highlight_meaning = (
                        f"The highlighted **{_preferred_trend_manager}** line sits at "
                        f"{_highlight_row['total_adjusted']:+.2f}; positive means sustained scoring above "
                        "the weekly field, while negative means the manager has been chasing that baseline."
                    )
                    st.markdown(
                        f" **What it means:** **{_season_leader['manager']}** has accumulated "
                        f"{_season_leader['total_adjusted']:+.2f} points versus the weekly league average"
                        f"{_lead_context}. **{_recent_leader['manager']}** has the best recent form at "
                        f"{_recent_leader['avg_adjusted']:+.2f} points per week across the last "
                        f"{len(_last_four_weeks)} completed week(s), so the current momentum leader can differ "
                        f"from the full-season leader. {_highlight_meaning}"
                    )

                    _weekly_details = _weekly_environment.rename(columns={
                        "week": "Week", "average": "League Average",
                        "median": "League Median", "q1": "25th Percentile",
                        "q3": "75th Percentile", "teams": "Teams",
                    })
                    _manager_details = _manager_totals.rename(columns={
                        "manager": "Manager", "avg_score": "Raw Avg Score",
                        "total_adjusted": "Cumulative Pts vs League",
                        "avg_adjusted": "Pts vs League / Wk", "games": "Games",
                    })
                    with st.expander("View complete score trend data"):
                        st.markdown("**Weekly scoring environment**")
                        st.dataframe(_weekly_details, hide_index=True, width="stretch")
                        st.markdown("**Manager scoring trends**")
                        st.dataframe(
                            _manager_details,
                            hide_index=True,
                            width="stretch",
                            column_config={
                                "Raw Avg Score": st.column_config.NumberColumn(format="%.2f"),
                                "Cumulative Pts vs League": st.column_config.NumberColumn(format="%+.2f"),
                                "Pts vs League / Wk": st.column_config.NumberColumn(format="%+.2f"),
                            },
                        )

            # ── Sub-tab G: Draft & Roster Insights ────────────────────────────
            with _lhG:
                # Kept in a separate renderer so this already-large history page remains
                # navigable and the analytics stay independently testable.
                from league_insights_view import render as _render_league_insights

                _render_league_insights(
                    _lh, _season_filter, _fetch_player_directory,
                    _fetch_season_transactions,
                )



# ══════════════════════════════════════════════════════════════════════════════
# TAB 7: HELP & GUIDE
# ══════════════════════════════════════════════════════════════════════════════
