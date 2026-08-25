"""Yahoo league adapter for the League History page.

Yahoo does not expose a stable public fantasy SDK here. Its own fantasy site
reads JSON from the unofficial endpoint below, the same class as ESPN's kona
feed and the Draft Board Yahoo ADP pull. Public responses are cached;
credentialed private responses deliberately are not.
"""
from __future__ import annotations

import concurrent.futures as cf
import re
from collections.abc import Mapping, Sequence

import requests
import streamlit as st


YAHOO_MIN_SEASON = 2018
YAHOO_MAX_HISTORY_SEASONS = 10
_YAHOO_BASE = "https://pub-api-ro.fantasysports.yahoo.com/fantasy/v2"
_YAHOO_GET_CACHE_ENTRIES = 256
_YAHOO_SEASON_CACHE_ENTRIES = 80
_WEEK_FETCH_WORKERS = 6
_BENCH_SLOTS = {"BN", "IR", "NA", "TA"}
_CORE_POSITIONS = {"QB", "RB", "WR", "TE", "K", "DEF", "DST", "D/ST"}
_HEADERS = {
    "User-Agent": "Mozilla/5.0 (JoSchoAnalytics League History Yahoo)",
    "Accept": "*/*",
    "Origin": "https://football.fantasysports.yahoo.com",
}

# NFL game keys measured against /league/{key}.l.101 metadata, 2018-2026.
_NFL_GAME_KEYS = {
    2018: "380",
    2019: "390",
    2020: "399",
    2021: "406",
    2022: "414",
    2023: "423",
    2024: "449",
    2025: "461",
    2026: "470",
}
_SEASON_BY_GAME = {key: year for year, key in _NFL_GAME_KEYS.items()}

_LEAGUE_KEY_RE = re.compile(r"(?:^|[^\d])(\d{3,4})\.l\.(\d{1,10})(?:[^\d]|$)")
_URL_ID_RE = re.compile(r"/f1/(\d{1,10})(?:[/?#]|$)", re.I)
_URL_SEASON_RE = re.compile(r"fantasysports\.yahoo\.com/(\d{4})/f1/", re.I)


class YahooLeagueError(RuntimeError):
    """A user-facing Yahoo access or payload error."""

    def __init__(self, message: str, *, status_code: int | None = None):
        super().__init__(message)
        self.status_code = status_code


def parse_league_ref(raw_league_id: str) -> tuple[str, str | None, int | None]:
    """Return (league_id, game_key hint, season hint) from an ID, key, or URL."""
    text = str(raw_league_id or "").strip()
    keyed = _LEAGUE_KEY_RE.search(text)
    if keyed:
        game_key, league_id = keyed.group(1), keyed.group(2)
        return league_id, game_key, _SEASON_BY_GAME.get(game_key)
    url_id = _URL_ID_RE.search(text)
    if url_id:
        season_match = _URL_SEASON_RE.search(text)
        season = int(season_match.group(1)) if season_match else None
        return url_id.group(1), None, season
    return text, None, None


def league_id_error(raw_league_id: str) -> str | None:
    league_id, _, _ = parse_league_ref(raw_league_id)
    if not league_id:
        return "Enter your Yahoo league ID to load your league history."
    if not league_id.isdigit():
        return (
            "Yahoo league IDs contain digits only. Copy the number after /f1/ "
            "in your Yahoo league URL."
        )
    if len(league_id) > 10:
        return (
            "That does not look like a Yahoo league ID. Copy the number after "
            "/f1/ in your Yahoo league URL."
        )
    return None


def season_error(raw_season, current_year: int) -> str | None:
    try:
        season = int(raw_season)
    except (TypeError, ValueError):
        return "Enter the four-digit Yahoo season from your league URL."
    if not YAHOO_MIN_SEASON <= season <= int(current_year):
        return (
            f"Yahoo League History currently supports seasons "
            f"{YAHOO_MIN_SEASON}-{current_year}."
        )
    return None


def game_key_for_season(season: int) -> str:
    key = _NFL_GAME_KEYS.get(int(season))
    if not key:
        raise YahooLeagueError(
            f"Yahoo League History does not have an NFL game key for {season}."
        )
    return key


def league_key(league_id: str, season: int, game_key: str | None = None) -> str:
    return f"{game_key or game_key_for_season(season)}.l.{str(league_id).strip()}"


def private_credentials_error(yahoo_y: str, yahoo_t: str) -> str | None:
    if not str(yahoo_y or "").strip() or not str(yahoo_t or "").strip():
        return "Private Yahoo leagues need both the Y and T cookie values."
    return None


def _cookie_value(raw_value: str, cookie_name: str) -> str:
    value = str(raw_value or "").strip().strip('"').strip("'")
    prefix, separator, remainder = value.partition("=")
    if separator and prefix.strip().casefold() == cookie_name.casefold():
        value = remainder.strip().strip('"').strip("'")
    return value


def _private_cookies(yahoo_y: str, yahoo_t: str) -> dict[str, str]:
    return {
        "Y": _cookie_value(yahoo_y, "Y"),
        "T": _cookie_value(yahoo_t, "T"),
    }


def _count_entries(node: Mapping | None) -> list:
    if not isinstance(node, dict):
        return []
    rows = []
    for key, value in node.items():
        if key == "count":
            continue
        if str(key).isdigit():
            rows.append((int(key), value))
    rows.sort()
    return [value for _, value in rows]


def _collapse_singletons(items: Sequence | None) -> dict:
    out: dict = {}
    for item in items or []:
        if isinstance(item, dict):
            out.update(item)
        elif isinstance(item, list):
            nested = _collapse_singletons(item)
            for key, value in nested.items():
                if key not in out or out[key] in (None, "", [], {}):
                    out[key] = value
    return out


def _league_parts(payload: Mapping | None) -> tuple[dict, dict]:
    league = (payload or {}).get("fantasy_content", {}).get("league")
    if isinstance(league, dict):
        meta = {
            key: value for key, value in league.items()
            if key not in {
                "settings", "standings", "scoreboard", "draft_results", "teams",
            }
        }
        extras = {
            key: league.get(key)
            for key in ("settings", "standings", "scoreboard", "draft_results", "teams")
            if key in league
        }
        return meta, extras
    if not isinstance(league, list) or not league:
        return {}, {}
    meta = league[0] if isinstance(league[0], dict) else {}
    extras: dict = {}
    for part in league[1:]:
        if isinstance(part, dict):
            extras.update(part)
    return meta if isinstance(meta, dict) else {}, extras


def _settings_map(extras: Mapping) -> dict:
    settings = extras.get("settings")
    if isinstance(settings, list) and settings and isinstance(settings[0], dict):
        return settings[0]
    return settings if isinstance(settings, dict) else {}


def _parse_team_node(node) -> tuple[dict, dict]:
    team = node.get("team") if isinstance(node, dict) else node
    parts = team if isinstance(team, list) else [team]
    meta_parts: list = []
    extras: dict = {}
    extra_keys = {
        "team_standings", "team_points", "team_projected_points",
        "roster", "win_probability",
    }
    for part in parts:
        if isinstance(part, list):
            meta_parts.extend(part)
        elif isinstance(part, dict):
            if extra_keys.intersection(part):
                extras.update(part)
            else:
                meta_parts.append(part)
    return _collapse_singletons(meta_parts), extras


def _parse_player_node(node) -> tuple[dict, dict]:
    player = node.get("player") if isinstance(node, dict) else node
    parts = player if isinstance(player, list) else [player]
    meta_parts: list = []
    extras: dict = {}
    extra_keys = {"selected_position", "player_stats", "player_points"}
    for part in parts:
        if isinstance(part, list):
            meta_parts.extend(part)
        elif isinstance(part, dict):
            if extra_keys.intersection(part):
                extras.update(part)
            else:
                meta_parts.append(part)
    return _collapse_singletons(meta_parts), extras


def _id_from_key(key: str, kind: str) -> str:
    text = str(key or "")
    marker = f".{kind}."
    if marker in text:
        return text.rsplit(marker, 1)[-1]
    return text.rsplit(".", 1)[-1] if "." in text else text


def _manager_identity(meta: Mapping) -> tuple[str, str]:
    managers = meta.get("managers")
    rows: list = []
    if isinstance(managers, list):
        rows = managers
    elif isinstance(managers, dict):
        rows = _count_entries(managers) or (
            [managers] if "manager" in managers else []
        )
    guid = ""
    nick = ""
    for row in rows:
        manager = row.get("manager") if isinstance(row, dict) else None
        if isinstance(manager, list):
            manager = _collapse_singletons(manager)
        if not isinstance(manager, dict):
            continue
        guid = str(manager.get("guid") or manager.get("manager_id") or "").strip()
        nick = str(manager.get("nickname") or "").strip()
        if guid or nick:
            break
    team_id = str(meta.get("team_id") or _id_from_key(str(meta.get("team_key") or ""), "t"))
    owner_id = guid or (f"yahoo-team-{team_id}" if team_id else "")
    username = nick or str(meta.get("name") or "").strip() or (
        f"Manager {team_id}" if team_id else "—"
    )
    return owner_id, username


def _player_names(name_node) -> tuple[str, str, str]:
    if isinstance(name_node, dict):
        return (
            str(name_node.get("full") or "").strip(),
            str(name_node.get("first") or "").strip(),
            str(name_node.get("last") or "").strip(),
        )
    text = str(name_node or "").strip()
    if not text:
        return "", "", ""
    bits = text.split()
    first = bits[0] if bits else ""
    last = " ".join(bits[1:]) if len(bits) > 1 else ""
    return text, first, last


def _position_label(selected) -> str:
    node = selected
    if isinstance(node, dict) and "selected_position" in node:
        node = node.get("selected_position")
    if isinstance(node, list):
        node = _collapse_singletons(node)
    if isinstance(node, dict):
        return str(node.get("position") or "").strip().upper()
    return str(node or "").strip().upper()


def _points_total(node) -> float:
    if isinstance(node, dict):
        raw = node.get("total")
    else:
        raw = node
    try:
        return round(float(raw or 0), 2)
    except (TypeError, ValueError):
        return 0.0


def _truthy(value) -> bool:
    return str(value or "").strip().lower() in {"1", "true", "yes"}


def _parse_renew(raw) -> tuple[str, str] | None:
    text = str(raw or "").strip()
    if "_" not in text:
        return None
    game, league_id = text.split("_", 1)
    if not game.isdigit() or not league_id.isdigit():
        return None
    return game, league_id


def _request_json(url: str, *, cookies: dict[str, str] | None = None) -> dict:
    try:
        response = requests.get(
            url,
            headers=_HEADERS,
            cookies=cookies,
            timeout=20,
        )
    except requests.RequestException as exc:
        raise YahooLeagueError("Yahoo did not respond. Try again in a moment.") from exc
    if response.status_code == 401:
        message = (
            "Yahoo denied access. Recopy the Y and T values from the same "
            "signed-in Yahoo browser session, or ask the commissioner to make "
            "the league publicly viewable."
            if cookies else
            "Yahoo denied access. Select Private and provide your Yahoo session "
            "cookies, or ask the commissioner to make the league publicly viewable."
        )
        raise YahooLeagueError(message, status_code=401)
    if response.status_code == 404:
        raise YahooLeagueError(
            "Yahoo did not return a league for that ID and season.",
            status_code=404,
        )
    if response.status_code != 200:
        raise YahooLeagueError(
            f"Yahoo returned HTTP {response.status_code}. Try again in a moment.",
            status_code=response.status_code,
        )
    try:
        payload = response.json()
    except ValueError as exc:
        raise YahooLeagueError("Yahoo returned an unreadable league response.") from exc
    if not isinstance(payload, dict):
        raise YahooLeagueError("Yahoo returned an unexpected league response.")
    if payload.get("error"):
        description = str((payload.get("error") or {}).get("description") or "")
        if "logged in" in description.lower():
            raise YahooLeagueError(
                "Yahoo denied access. Select Private and provide your Yahoo session "
                "cookies, or ask the commissioner to make the league publicly viewable.",
                status_code=401,
            )
        raise YahooLeagueError(
            "Yahoo did not return a league for that ID and season.",
            status_code=400,
        )
    return payload


def _league_url(key: str, suffix: str = "") -> str:
    return f"{_YAHOO_BASE}/league/{key}{suffix}?format=json"


@st.cache_data(ttl=3600, max_entries=_YAHOO_GET_CACHE_ENTRIES)
def _yahoo_get(url: str) -> dict:
    return _request_json(url)


def _yahoo_get_private(url: str, yahoo_y: str, yahoo_t: str) -> dict:
    return _request_json(url, cookies=_private_cookies(yahoo_y, yahoo_t))


def _get(
    url: str,
    *,
    cookies: dict[str, str] | None = None,
    yahoo_y: str = "",
    yahoo_t: str = "",
) -> dict:
    if cookies or yahoo_y or yahoo_t:
        return _yahoo_get_private(url, yahoo_y, yahoo_t)
    return _yahoo_get(url)


def _season_bundle(key: str, **auth) -> tuple[dict, dict]:
    payload = _get(
        _league_url(key, ";out=metadata,settings,standings,draftresults"),
        **auth,
    )
    return _league_parts(payload)


def _history_chain_from_meta(meta: Mapping, start_season: int) -> list[dict]:
    league_id = str(meta.get("league_id") or "").strip()
    season = str(meta.get("season") or start_season)
    name = str(meta.get("name") or "League")
    game_key = str(meta.get("league_key") or "").split(".l.", 1)[0] or game_key_for_season(
        int(season),
    )
    return [{
        "league_id": league_id,
        "season": season,
        "name": name,
        "game_key": game_key,
        "league_key": str(meta.get("league_key") or f"{game_key}.l.{league_id}"),
        "renew": str(meta.get("renew") or ""),
    }]


@st.cache_data(ttl=3600, max_entries=24)
def history_chain(
    league_id: str,
    start_season: int,
    game_key: str | None = None,
) -> list[dict]:
    key = league_key(league_id, int(start_season), game_key)
    meta, _extras = _season_bundle(key)
    if not meta.get("league_id"):
        raise YahooLeagueError(
            "Yahoo did not return a league for that ID and season.",
        )
    return _walk_renew(meta, int(start_season))


def history_chain_private(
    league_id: str,
    start_season: int,
    yahoo_y: str,
    yahoo_t: str,
    game_key: str | None = None,
) -> list[dict]:
    key = league_key(league_id, int(start_season), game_key)
    meta, _extras = _season_bundle(key, yahoo_y=yahoo_y, yahoo_t=yahoo_t)
    if not meta.get("league_id"):
        raise YahooLeagueError(
            "Yahoo did not return a league for that ID and season.",
        )
    return _walk_renew(meta, int(start_season), yahoo_y=yahoo_y, yahoo_t=yahoo_t)


def _walk_renew(start_meta: Mapping, start_season: int, **auth) -> list[dict]:
    chain = _history_chain_from_meta(start_meta, start_season)
    seen = {chain[0]["league_key"]}
    renew = _parse_renew(start_meta.get("renew"))
    while renew and len(chain) < YAHOO_MAX_HISTORY_SEASONS:
        prev_game, prev_id = renew
        prev_key = f"{prev_game}.l.{prev_id}"
        if prev_key in seen:
            break
        try:
            meta, _extras = _season_bundle(prev_key, **auth)
        except YahooLeagueError:
            break
        if not meta.get("league_id"):
            break
        row = _history_chain_from_meta(meta, int(meta.get("season") or 0))[0]
        chain.append(row)
        seen.add(row["league_key"])
        renew = _parse_renew(meta.get("renew"))
    return chain


def _fetch_week_scoreboard(key: str, week: int, **auth) -> dict | None:
    try:
        payload = _get(_league_url(key, f"/scoreboard;week={int(week)}"), **auth)
    except YahooLeagueError:
        return None
    _meta, extras = _league_parts(payload)
    return extras.get("scoreboard")


def _fetch_week_rosters(key: str, week: int, **auth) -> dict | None:
    try:
        payload = _get(
            _league_url(
                key,
                f"/teams/roster;week={int(week)}/players/stats;type=week;week={int(week)}",
            ),
            **auth,
        )
    except YahooLeagueError:
        return None
    _meta, extras = _league_parts(payload)
    return extras.get("teams")


def _fetch_weekly(
    key: str,
    first_week: int,
    final_week: int,
    **auth,
) -> tuple[dict[int, dict], dict[int, dict]]:
    weeks = range(max(1, int(first_week)), min(18, int(final_week)) + 1)

    def fetch_week(week: int) -> tuple[int, dict | None, dict | None]:
        return week, _fetch_week_scoreboard(key, week, **auth), _fetch_week_rosters(
            key, week, **auth,
        )

    scoreboards: dict[int, dict] = {}
    rosters: dict[int, dict] = {}
    with cf.ThreadPoolExecutor(max_workers=_WEEK_FETCH_WORKERS) as pool:
        for week, scoreboard, roster in pool.map(fetch_week, weeks):
            if isinstance(scoreboard, dict):
                scoreboards[week] = scoreboard
            if isinstance(roster, dict):
                rosters[week] = roster
    return scoreboards, rosters


def _remember_player(player_directory: dict, meta: Mapping) -> str:
    player_id = str(
        meta.get("player_id") or _id_from_key(str(meta.get("player_key") or ""), "p")
    ).strip()
    if not player_id:
        return ""
    full, first, last = _player_names(meta.get("name"))
    position = str(
        meta.get("primary_position") or meta.get("display_position") or ""
    ).split(",")[0].strip().upper()
    if position == "DEF":
        position = "D/ST"
    player_directory[player_id] = {
        "full_name": full,
        "first_name": first,
        "last_name": last,
        "position": position if position in _CORE_POSITIONS or position == "D/ST" else position,
    }
    return player_id


def _stat_modifier(settings: Mapping, kind: str) -> float:
    categories = ((settings.get("stat_categories") or {}).get("stats") or [])
    id_by_name = {}
    for row in categories:
        stat = row.get("stat") if isinstance(row, dict) else None
        if not isinstance(stat, dict):
            continue
        label = str(stat.get("name") or stat.get("display_name") or "").strip().lower()
        try:
            stat_id = int(stat.get("stat_id"))
        except (TypeError, ValueError):
            continue
        id_by_name[label] = stat_id
    wanted = None
    if kind == "rec":
        wanted = id_by_name.get("receptions") or id_by_name.get("rec")
    elif kind == "pass_td":
        for label, stat_id in id_by_name.items():
            if "passing" in label and "touchdown" in label:
                wanted = stat_id
                break
    if wanted is None:
        return 0.0
    for row in ((settings.get("stat_modifiers") or {}).get("stats") or []):
        stat = row.get("stat") if isinstance(row, dict) else None
        if not isinstance(stat, dict):
            continue
        try:
            if int(stat.get("stat_id")) != wanted:
                continue
            return float(stat.get("value") or 0)
        except (TypeError, ValueError):
            continue
    return 0.0


def _roster_positions(settings: Mapping) -> list[str]:
    rows = settings.get("roster_positions") or []
    labels = []
    for row in rows:
        slot = row.get("roster_position") if isinstance(row, dict) else None
        if not isinstance(slot, dict):
            continue
        position = str(slot.get("position") or "").strip().upper()
        if position == "DEF":
            position = "D/ST"
        try:
            count = int(slot.get("count") or 0)
        except (TypeError, ValueError):
            count = 0
        labels.extend([position] * max(0, count))
    return labels


def _matchup_sides(matchup: Mapping) -> list:
    if "teams" in matchup:
        return _count_entries(matchup.get("teams") or {})
    sides = []
    nested = matchup.get("0")
    if isinstance(nested, dict) and "teams" in nested:
        sides.extend(_count_entries(nested.get("teams") or {}))
    return sides


def normalize_season(
    league_id: str,
    season: int,
    meta: Mapping,
    extras: Mapping,
    scoreboards: Mapping[int, Mapping] | None = None,
    weekly_rosters: Mapping[int, Mapping] | None = None,
) -> tuple[dict, dict]:
    """Normalize one Yahoo season to the League History payload."""
    scoreboards = scoreboards or {}
    weekly_rosters = weekly_rosters or {}
    settings = _settings_map(extras)
    standings_wrap = extras.get("standings")
    standings_teams = {}
    if isinstance(standings_wrap, list) and standings_wrap:
        standings_teams = (standings_wrap[0] or {}).get("teams") or {}
    elif isinstance(standings_wrap, dict):
        standings_teams = standings_wrap.get("teams") or standings_wrap

    standings = []
    owner_by_team: dict[str, str] = {}
    draft_slot_by_team: dict[str, int] = {}
    for node in _count_entries(standings_teams):
        team_meta, team_extras = _parse_team_node(node)
        team_id = str(
            team_meta.get("team_id")
            or _id_from_key(str(team_meta.get("team_key") or ""), "t")
        )
        owner_id, username = _manager_identity(team_meta)
        owner_by_team[team_id] = owner_id
        try:
            draft_slot_by_team[team_id] = int(team_meta.get("draft_position") or 0)
        except (TypeError, ValueError):
            draft_slot_by_team[team_id] = 0
        record = (team_extras.get("team_standings") or {})
        totals = record.get("outcome_totals") or {}
        try:
            rank = int(record.get("rank") or 0) or None
        except (TypeError, ValueError):
            rank = None
        standings.append({
            "roster_id": int(team_id) if team_id.isdigit() else team_id,
            "owner_id": owner_id,
            "username": username,
            "team_name": str(team_meta.get("name") or ""),
            "wins": int(totals.get("wins") or 0),
            "losses": int(totals.get("losses") or 0),
            "fpts": round(float(record.get("points_for") or 0), 2),
            "fpts_against": round(float(record.get("points_against") or 0), 2),
            "playoff_finish": rank,
        })
    standings.sort(key=lambda row: (
        row["playoff_finish"] or 999,
        -row["wins"],
        -row["fpts"],
    ))

    by_finish = {
        int(row["playoff_finish"]): row
        for row in standings
        if row.get("playoff_finish") is not None
    }
    empty_manager = {"username": "?", "team_name": ""}
    champion = by_finish.get(1, empty_manager)
    runner_up = by_finish.get(2, empty_manager)
    last_rank = max(by_finish, default=0)
    toilet_rows = [
        by_finish[rank]
        for rank in (last_rank - 1, last_rank)
        if rank in by_finish
    ]
    toilet_champions = (
        [{
            "username": by_finish[last_rank]["username"],
            "team_name": by_finish[last_rank].get("team_name", ""),
        }]
        if last_rank else []
    )
    toilet_finalists = sorted({row["username"] for row in toilet_rows})

    matchups = []
    matchup_by_team_week: dict[tuple[str, int], str] = {}
    for week, scoreboard in sorted(scoreboards.items()):
        wrapper = scoreboard.get("0") if isinstance(scoreboard, dict) else None
        matchup_map = (wrapper or {}).get("matchups") if isinstance(wrapper, dict) else scoreboard
        for matchup_node in _count_entries(matchup_map if isinstance(matchup_map, dict) else {}):
            matchup = matchup_node.get("matchup") if isinstance(matchup_node, dict) else matchup_node
            if not isinstance(matchup, dict):
                continue
            sides = [_parse_team_node(side) for side in _matchup_sides(matchup)]
            if len(sides) != 2:
                continue
            (meta_a, extra_a), (meta_b, extra_b) = sides
            rid_a = str(meta_a.get("team_id") or _id_from_key(str(meta_a.get("team_key") or ""), "t"))
            rid_b = str(meta_b.get("team_id") or _id_from_key(str(meta_b.get("team_key") or ""), "t"))
            score_a = _points_total(extra_a.get("team_points"))
            score_b = _points_total(extra_b.get("team_points"))
            if score_a == 0 and score_b == 0:
                continue
            try:
                week_id = int(matchup.get("week") or week)
            except (TypeError, ValueError):
                week_id = int(week)
            matchup_id = f"{week_id}:{rid_a}:{rid_b}"
            matchup_by_team_week[(rid_a, week_id)] = matchup_id
            matchup_by_team_week[(rid_b, week_id)] = matchup_id
            matchups.append({
                "season": str(season),
                "week": week_id,
                "is_playoff": _truthy(matchup.get("is_playoffs")),
                "rid_a": rid_a,
                "score_a": score_a,
                "rid_b": rid_b,
                "score_b": score_b,
            })

    player_directory: dict = {}
    roster_entries = []
    for week, teams_node in sorted(weekly_rosters.items()):
        for node in _count_entries(teams_node):
            team_meta, team_extras = _parse_team_node(node)
            team_id = str(
                team_meta.get("team_id")
                or _id_from_key(str(team_meta.get("team_key") or ""), "t")
            )
            roster = team_extras.get("roster") or {}
            players_node = {}
            if isinstance(roster, dict):
                inner = roster.get("0") if "0" in roster else roster
                if isinstance(inner, dict):
                    players_node = inner.get("players") or {}
            player_ids = []
            starters = []
            player_points = {}
            for player_node in _count_entries(players_node if isinstance(players_node, dict) else {}):
                player_meta, player_extras = _parse_player_node(player_node)
                player_id = _remember_player(player_directory, player_meta)
                if not player_id:
                    continue
                player_ids.append(player_id)
                slot = _position_label(player_extras.get("selected_position"))
                if slot not in _BENCH_SLOTS:
                    starters.append(player_id)
                player_points[player_id] = _points_total(player_extras.get("player_points"))
            roster_entries.append({
                "season": str(season),
                "week": int(week),
                "roster_id": int(team_id) if team_id.isdigit() else team_id,
                "matchup_id": matchup_by_team_week.get((team_id, int(week))),
                "players": player_ids,
                "starters": starters,
                "players_points": player_points,
            })

    draft_map = extras.get("draft_results") or {}
    draft_picks = []
    for pick_node in _count_entries(draft_map if isinstance(draft_map, dict) else {}):
        pick = pick_node.get("draft_result") if isinstance(pick_node, dict) else pick_node
        if not isinstance(pick, dict):
            continue
        team_id = _id_from_key(str(pick.get("team_key") or ""), "t")
        player_id = _id_from_key(str(pick.get("player_key") or ""), "p")
        player = player_directory.get(player_id, {})
        try:
            pick_no = int(pick.get("pick") or 0)
            round_no = int(pick.get("round") or 0)
        except (TypeError, ValueError):
            continue
        n_teams = max(1, len(standings) or int(meta.get("num_teams") or 1))
        pick_in_round = ((pick_no - 1) % n_teams) + 1 if pick_no else 0
        draft_picks.append({
            "pick_no": pick_no,
            "round": round_no,
            "pick_in_round": pick_in_round,
            "draft_slot": draft_slot_by_team.get(team_id, 0),
            "roster_id": int(team_id) if team_id.isdigit() else team_id,
            "picked_by": owner_by_team.get(team_id, ""),
            "player_id": player_id,
            "metadata": {
                "player_id": player_id,
                "full_name": player.get("full_name", ""),
                "first_name": player.get("first_name", ""),
                "last_name": player.get("last_name", ""),
                "position": player.get("position", ""),
            },
        })

    uses_faab = _truthy(settings.get("uses_faab"))
    final_week = int(meta.get("end_week") or 18)
    latest_week = int(meta.get("current_week") or 0)
    is_complete = bool(meta.get("is_finished")) or (
        latest_week >= final_week and bool(by_finish)
    )
    payload = {
        "league_id": str(league_id),
        "draft_id": f"yahoo:{league_id}:{season}",
        "status": "complete" if is_complete else "in_season",
        "champion": {
            "username": champion.get("username", "?"),
            "team_name": champion.get("team_name", ""),
        },
        "runner_up": {
            "username": runner_up.get("username", "?"),
            "team_name": runner_up.get("team_name", ""),
        },
        "toilet_champion": (
            toilet_champions[0] if toilet_champions else empty_manager
        ),
        "toilet_champions": toilet_champions,
        "toilet_finalists": toilet_finalists,
        "toilet_bracket": toilet_finalists,
        "standings": standings,
        "matchups": matchups,
        "draft_picks": draft_picks,
        "roster_entries": roster_entries,
        "league_settings": {
            "total_rosters": len(standings) or int(meta.get("num_teams") or 0),
            "roster_positions": _roster_positions(settings),
            "scoring_settings": {
                "rec": _stat_modifier(settings, "rec"),
                "pass_td": _stat_modifier(settings, "pass_td"),
            },
            "waiver_type": 2 if uses_faab else 0,
            "waiver_budget": None,
        },
    }
    return payload, player_directory


def _fetch_one(league_id: str, season: int, game_key: str | None = None, **auth):
    key = league_key(league_id, int(season), game_key)
    meta, extras = _season_bundle(key, **auth)
    if not meta.get("league_id"):
        raise YahooLeagueError(
            "Yahoo did not return a league for that ID and season.",
        )
    first_week = int(meta.get("start_week") or 1)
    final_week = int(meta.get("end_week") or 18)
    scoreboards, weekly_rosters = _fetch_weekly(key, first_week, final_week, **auth)
    payload, player_directory = normalize_season(
        str(meta.get("league_id") or league_id),
        int(meta.get("season") or season),
        meta,
        extras,
        scoreboards,
        weekly_rosters,
    )
    return str(meta.get("season") or season), payload, player_directory


@st.cache_data(ttl=3600, max_entries=_YAHOO_SEASON_CACHE_ENTRIES)
def fetch_one_season(
    league_id: str,
    season: int,
    game_key: str | None = None,
) -> tuple[str, dict, dict]:
    return _fetch_one(league_id, season, game_key)


def fetch_one_season_private(
    league_id: str,
    season: int,
    yahoo_y: str,
    yahoo_t: str,
    game_key: str | None = None,
) -> tuple[str, dict, dict]:
    return _fetch_one(
        league_id, season, game_key, yahoo_y=yahoo_y, yahoo_t=yahoo_t,
    )


def fetch_history(
    league_id: str,
    start_season: int,
    max_seasons: int | None = None,
    game_key: str | None = None,
) -> dict:
    chain = history_chain(str(league_id), int(start_season), game_key)
    if max_seasons is not None:
        chain = chain[:max(1, int(max_seasons))]
    league_name = chain[0]["name"] if chain else "League"
    seasons = {}
    player_directory = {}
    for item in chain:
        try:
            year, payload, players = fetch_one_season(
                item["league_id"], int(item["season"]), item.get("game_key"),
            )
        except YahooLeagueError:
            continue
        seasons[year] = payload
        player_directory.update(players)
    return {
        "league_name": league_name or "League",
        "seasons": seasons,
        "player_directory": player_directory,
        "provider": "Yahoo",
    }


def fetch_history_private(
    league_id: str,
    start_season: int,
    yahoo_y: str,
    yahoo_t: str,
    max_seasons: int | None = None,
    game_key: str | None = None,
) -> dict:
    chain = history_chain_private(
        league_id, start_season, yahoo_y, yahoo_t, game_key,
    )
    if max_seasons is not None:
        chain = chain[:max(1, int(max_seasons))]
    league_name = chain[0]["name"] if chain else "League"
    seasons = {}
    player_directory = {}
    for item in chain:
        try:
            year, payload, players = fetch_one_season_private(
                item["league_id"], int(item["season"]), yahoo_y, yahoo_t,
                item.get("game_key"),
            )
        except YahooLeagueError:
            continue
        seasons[year] = payload
        player_directory.update(players)
    return {
        "league_name": league_name or "League",
        "seasons": seasons,
        "player_directory": player_directory,
        "provider": "Yahoo",
    }
