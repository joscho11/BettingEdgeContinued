"""ESPN league adapter for the League History page.

ESPN does not publish a supported fantasy API, but its own fantasy site uses the
read-only JSON endpoint below.  This module keeps that provider-specific shape
outside the page and normalizes it to the existing League History payload. Public
responses are cached; credentialed private responses deliberately are not.
"""
from __future__ import annotations

import concurrent.futures as cf
import json
from collections.abc import Mapping

import requests
import streamlit as st


ESPN_MIN_SEASON = 2018
ESPN_MAX_HISTORY_SEASONS = 10
_ESPN_BASE = "https://lm-api-reads.fantasy.espn.com/apis/v3/games/ffl"
_ESPN_GET_CACHE_ENTRIES = 256
_ESPN_SEASON_CACHE_ENTRIES = 80
_WEEK_FETCH_WORKERS = 6

_LINEUP_SLOT_NAMES = {
    0: "QB",
    1: "TQB",
    2: "RB",
    3: "RB/WR",
    4: "WR",
    5: "WR/TE",
    6: "TE",
    7: "SUPER_FLEX",
    16: "D/ST",
    17: "K",
    20: "BN",
    21: "IR",
    23: "FLEX",
}
_NON_STARTER_SLOTS = {20, 21}
_CORE_POSITIONS = {"QB", "RB", "WR", "TE", "D/ST", "K"}


class EspnLeagueError(RuntimeError):
    """A user-facing ESPN access or payload error."""

    def __init__(self, message: str, *, status_code: int | None = None):
        super().__init__(message)
        self.status_code = status_code


def league_id_error(raw_league_id: str) -> str | None:
    league_id = raw_league_id.strip()
    if not league_id:
        return "Enter your ESPN league ID to load your league history."
    if not league_id.isdigit():
        return "ESPN league IDs contain digits only."
    if len(league_id) > 20:
        return "That does not look like an ESPN league ID. Copy the leagueId value from your ESPN URL."
    return None


def season_error(raw_season, current_year: int) -> str | None:
    try:
        season = int(raw_season)
    except (TypeError, ValueError):
        return "Enter the four-digit ESPN season from your league URL."
    if not ESPN_MIN_SEASON <= season <= int(current_year):
        return f"ESPN League History currently supports seasons {ESPN_MIN_SEASON}-{current_year}."
    return None


def _league_url(league_id: str, season: int) -> str:
    return (
        f"{_ESPN_BASE}/seasons/{int(season)}/segments/0/leagues/"
        f"{str(league_id).strip()}"
    )


def private_credentials_error(espn_s2: str, swid: str) -> str | None:
    if not str(swid or "").strip() or not str(espn_s2 or "").strip():
        return "Private ESPN leagues need both the SWID and espn_s2 cookie values."
    return None


def _cookie_value(raw_value: str, cookie_name: str) -> str:
    """Accept either a raw cookie value or ``name=value`` copied from devtools."""
    value = str(raw_value or "").strip().strip('"').strip("'")
    prefix, separator, remainder = value.partition("=")
    if separator and prefix.strip().casefold() == cookie_name.casefold():
        value = remainder.strip().strip('"').strip("'")
    return value


def _private_cookies(espn_s2: str, swid: str) -> dict[str, str]:
    return {
        "espn_s2": _cookie_value(espn_s2, "espn_s2"),
        "SWID": _cookie_value(swid, "SWID"),
    }


def _request_league(
    league_id: str,
    season: int,
    views: tuple[str, ...],
    scoring_period: int | None = None,
    *,
    cookies: dict[str, str] | None = None,
) -> dict:
    params: list[tuple[str, str | int]] = [("view", view) for view in views]
    if scoring_period is not None:
        params.append(("scoringPeriodId", int(scoring_period)))
    try:
        response = requests.get(
            _league_url(league_id, season),
            params=params,
            cookies=cookies,
            timeout=20,
        )
    except requests.RequestException as exc:
        raise EspnLeagueError("ESPN did not respond. Try again in a moment.") from exc
    if response.status_code == 401:
        message = (
            "ESPN denied access. Recopy the SWID and espn_s2 values from the same "
            "signed-in ESPN browser session."
            if cookies else
            "ESPN denied access. Select Private and provide your ESPN session cookies."
        )
        raise EspnLeagueError(message, status_code=401)
    if response.status_code == 404:
        raise EspnLeagueError(
            "ESPN did not return a league for that ID and season.",
            status_code=404,
        )
    if response.status_code != 200:
        raise EspnLeagueError(
            f"ESPN returned HTTP {response.status_code}. Try again in a moment.",
            status_code=response.status_code,
        )
    try:
        payload = response.json()
    except ValueError as exc:
        raise EspnLeagueError("ESPN returned an unreadable league response.") from exc
    if isinstance(payload, list):
        payload = payload[0] if payload else {}
    if not isinstance(payload, dict):
        raise EspnLeagueError("ESPN returned an unexpected league response.")
    return payload


@st.cache_data(ttl=3600, max_entries=_ESPN_GET_CACHE_ENTRIES)
def _espn_get(
    league_id: str,
    season: int,
    views: tuple[str, ...],
    scoring_period: int | None = None,
) -> dict:
    return _request_league(league_id, season, views, scoring_period)


def _espn_get_private(
    league_id: str,
    season: int,
    views: tuple[str, ...],
    espn_s2: str,
    swid: str,
    scoring_period: int | None = None,
) -> dict:
    """Fetch private league data without a shared or credential-keyed cache."""
    return _request_league(
        league_id,
        season,
        views,
        scoring_period,
        cookies=_private_cookies(espn_s2, swid),
    )


@st.cache_data(ttl=3600, max_entries=24)
def history_chain(league_id: str, start_season: int) -> list[dict]:
    """Return the requested ESPN season followed by its advertised prior seasons."""
    summary = _espn_get(str(league_id), int(start_season), ("mSettings",))
    return _history_chain_from_summary(str(league_id), int(start_season), summary)


def history_chain_private(
    league_id: str,
    start_season: int,
    espn_s2: str,
    swid: str,
) -> list[dict]:
    """Credentialed history lookup; never enters Streamlit's shared cache."""
    summary = _espn_get_private(
        str(league_id), int(start_season), ("mSettings",), espn_s2, swid,
    )
    return _history_chain_from_summary(str(league_id), int(start_season), summary)


def _history_chain_from_summary(
    league_id: str,
    start_season: int,
    summary: Mapping,
) -> list[dict]:
    settings = summary.get("settings") or {}
    status = summary.get("status") or {}
    season_id = int(summary.get("seasonId") or start_season)
    previous = []
    for raw_year in status.get("previousSeasons") or []:
        try:
            year = int(raw_year)
        except (TypeError, ValueError):
            continue
        if ESPN_MIN_SEASON <= year < season_id:
            previous.append(year)
    years = [season_id, *sorted(set(previous), reverse=True)]
    years = years[:ESPN_MAX_HISTORY_SEASONS]
    league_name = str(settings.get("name") or "League")
    return [
        {"league_id": str(league_id), "season": str(year), "name": league_name}
        for year in years
    ]


def _fetch_weekly_rosters(
    league_id: str,
    season: int,
    first_week: int,
    final_week: int,
) -> dict[int, dict]:
    weeks = range(max(1, int(first_week)), min(18, int(final_week)) + 1)

    def fetch_week(week: int) -> tuple[int, dict | None]:
        try:
            return week, _espn_get(league_id, season, ("mRoster",), week)
        except EspnLeagueError:
            return week, None

    with cf.ThreadPoolExecutor(max_workers=_WEEK_FETCH_WORKERS) as pool:
        return {
            week: payload
            for week, payload in pool.map(fetch_week, weeks)
            if isinstance(payload, dict)
        }


def _fetch_weekly_rosters_private(
    league_id: str,
    season: int,
    first_week: int,
    final_week: int,
    espn_s2: str,
    swid: str,
) -> dict[int, dict]:
    weeks = range(max(1, int(first_week)), min(18, int(final_week)) + 1)

    def fetch_week(week: int) -> tuple[int, dict | None]:
        try:
            payload = _espn_get_private(
                league_id, season, ("mRoster",), espn_s2, swid, week,
            )
            return week, payload
        except EspnLeagueError:
            return week, None

    with cf.ThreadPoolExecutor(max_workers=_WEEK_FETCH_WORKERS) as pool:
        return {
            week: payload
            for week, payload in pool.map(fetch_week, weeks)
            if isinstance(payload, dict)
        }


@st.cache_data(ttl=86400, max_entries=_ESPN_SEASON_CACHE_ENTRIES)
def _fetch_drafted_players(season: int, player_ids: tuple[int, ...]) -> list[dict]:
    """Fill names for players dropped before the first weekly roster snapshot."""
    return _request_drafted_players(season, player_ids)


def _fetch_drafted_players_private(
    season: int,
    player_ids: tuple[int, ...],
) -> list[dict]:
    """Private-league fallback that keeps draft IDs out of shared cache keys."""
    return _request_drafted_players(season, player_ids)


def _request_drafted_players(season: int, player_ids: tuple[int, ...]) -> list[dict]:
    if not player_ids:
        return []
    url = f"{_ESPN_BASE}/seasons/{int(season)}/players"
    headers = {
        "x-fantasy-filter": json.dumps({"filterIds": {"value": list(player_ids)}}),
    }
    try:
        response = requests.get(
            url,
            params={"view": "players_wl"},
            headers=headers,
            timeout=20,
        )
        response.raise_for_status()
        payload = response.json()
    except (requests.RequestException, ValueError):
        return []
    return [row for row in payload if isinstance(row, dict)] if isinstance(payload, list) else []


def _member_name(member: Mapping | None, owner_id: str) -> str:
    member = member or {}
    display = str(member.get("displayName") or "").strip()
    if display:
        return display
    full_name = " ".join(
        value for value in (
            str(member.get("firstName") or "").strip(),
            str(member.get("lastName") or "").strip(),
        ) if value
    )
    return full_name or (f"Manager {owner_id[-4:]}" if owner_id else "—")


def _player_position(player: Mapping | None) -> str:
    player = player or {}
    for raw_slot in player.get("eligibleSlots") or []:
        try:
            label = _LINEUP_SLOT_NAMES.get(int(raw_slot), "")
        except (TypeError, ValueError):
            continue
        if label in _CORE_POSITIONS:
            return label
    return ""


def _remember_player(player_directory: dict, player: Mapping | None) -> None:
    player = player or {}
    player_id = str(player.get("id") or "").strip()
    if not player_id:
        return
    player_directory[player_id] = {
        "full_name": str(player.get("fullName") or "").strip(),
        "first_name": str(player.get("firstName") or "").strip(),
        "last_name": str(player.get("lastName") or "").strip(),
        "position": _player_position(player),
    }


def _actual_points(player: Mapping | None, week: int) -> float:
    player = player or {}
    for stat in player.get("stats") or []:
        if not isinstance(stat, Mapping):
            continue
        if stat.get("scoringPeriodId") != week or stat.get("statSourceId") != 0:
            continue
        try:
            return round(float(stat.get("appliedTotal") or 0), 2)
        except (TypeError, ValueError):
            return 0.0
    return 0.0


def _owner_id(team: Mapping) -> str:
    primary = str(team.get("primaryOwner") or "").strip()
    if primary:
        return primary
    owners = team.get("owners") or []
    return str(owners[0]).strip() if owners else ""


def _finish(team: Mapping) -> int | None:
    for key in ("rankFinal", "rankCalculatedFinal"):
        try:
            value = int(team.get(key) or 0)
        except (TypeError, ValueError):
            value = 0
        if value > 0:
            return value
    return None


def _scoring_value(scoring_items: list, stat_ids: set[int]) -> float:
    for item in scoring_items:
        if not isinstance(item, Mapping):
            continue
        try:
            stat_id = int(item.get("statId"))
        except (TypeError, ValueError):
            continue
        if stat_id in stat_ids:
            try:
                return float(item.get("points") or 0)
            except (TypeError, ValueError):
                return 0.0
    return 0.0


def normalize_season(
    league_id: str,
    season: int,
    data: Mapping,
    weekly_rosters: Mapping[int, Mapping] | None = None,
    drafted_players: list[Mapping] | None = None,
) -> tuple[dict, dict]:
    """Normalize one raw ESPN season and return (season payload, player directory)."""
    weekly_rosters = weekly_rosters or {}
    settings = data.get("settings") or {}
    schedule_settings = settings.get("scheduleSettings") or {}
    status = data.get("status") or {}
    teams = [team for team in data.get("teams") or [] if isinstance(team, Mapping)]
    members = {
        str(member.get("id") or ""): member
        for member in data.get("members") or []
        if isinstance(member, Mapping)
    }
    team_by_id = {str(team.get("id") or ""): team for team in teams}
    owner_by_team = {team_id: _owner_id(team) for team_id, team in team_by_id.items()}

    standings = []
    for team_id, team in team_by_id.items():
        owner_id = owner_by_team.get(team_id, "")
        record = (team.get("record") or {}).get("overall") or {}
        standings.append({
            "roster_id": team.get("id"),
            "owner_id": owner_id,
            "username": _member_name(members.get(owner_id), owner_id),
            "team_name": str(team.get("name") or team.get("abbrev") or ""),
            "wins": int(record.get("wins") or 0),
            "losses": int(record.get("losses") or 0),
            "fpts": round(float(record.get("pointsFor") or 0), 2),
            "fpts_against": round(float(record.get("pointsAgainst") or 0), 2),
            "playoff_finish": _finish(team),
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

    regular_matchup_count = int(schedule_settings.get("matchupPeriodCount") or 14)
    matchups = []
    matchup_by_team_week: dict[tuple[str, int], str] = {}
    for matchup in data.get("schedule") or []:
        if not isinstance(matchup, Mapping):
            continue
        home = matchup.get("home") or {}
        away = matchup.get("away") or {}
        home_id = str(home.get("teamId") or "")
        away_id = str(away.get("teamId") or "")
        if not home_id or not away_id:
            continue
        home_scores = home.get("pointsByScoringPeriod") or {}
        away_scores = away.get("pointsByScoringPeriod") or {}
        weeks = sorted(set(home_scores) & set(away_scores), key=lambda value: int(value))
        matchup_period = int(matchup.get("matchupPeriodId") or 0)
        matchup_id = str(matchup.get("id") or f"{matchup_period}:{home_id}:{away_id}")
        for raw_week in weeks:
            try:
                week = int(raw_week)
                home_score = float(home_scores[raw_week] or 0)
                away_score = float(away_scores[raw_week] or 0)
            except (TypeError, ValueError):
                continue
            if home_score == 0 and away_score == 0:
                continue
            matchup_by_team_week[(home_id, week)] = matchup_id
            matchup_by_team_week[(away_id, week)] = matchup_id
            matchups.append({
                "season": str(season),
                "week": week,
                "is_playoff": matchup_period > regular_matchup_count,
                "rid_a": home_id,
                "score_a": round(home_score, 2),
                "rid_b": away_id,
                "score_b": round(away_score, 2),
            })

    player_directory: dict = {}
    for player in drafted_players or []:
        _remember_player(player_directory, player)
    roster_entries = []
    for week, week_payload in sorted(weekly_rosters.items()):
        for team in week_payload.get("teams") or []:
            if not isinstance(team, Mapping):
                continue
            team_id = str(team.get("id") or "")
            player_ids = []
            starters = []
            player_points = {}
            for entry in (team.get("roster") or {}).get("entries") or []:
                if not isinstance(entry, Mapping):
                    continue
                player = (entry.get("playerPoolEntry") or {}).get("player") or {}
                player_id = str(entry.get("playerId") or player.get("id") or "")
                if not player_id:
                    continue
                _remember_player(player_directory, player)
                player_ids.append(player_id)
                try:
                    lineup_slot = int(entry.get("lineupSlotId"))
                except (TypeError, ValueError):
                    lineup_slot = 20
                if lineup_slot not in _NON_STARTER_SLOTS:
                    starters.append(player_id)
                player_points[player_id] = _actual_points(player, int(week))
            roster_entries.append({
                "season": str(season),
                "week": int(week),
                "roster_id": team.get("id"),
                "matchup_id": matchup_by_team_week.get((team_id, int(week))),
                "players": player_ids,
                "starters": starters,
                "players_points": player_points,
            })

    picks_raw = (data.get("draftDetail") or {}).get("picks") or []
    first_round_slots = {
        str(pick.get("teamId") or ""): int(pick.get("roundPickNumber") or 0)
        for pick in picks_raw
        if isinstance(pick, Mapping) and int(pick.get("roundId") or 0) == 1
    }
    draft_picks = []
    for pick in picks_raw:
        if not isinstance(pick, Mapping):
            continue
        player_id = str(pick.get("playerId") or "")
        team_id = str(pick.get("teamId") or "")
        player = player_directory.get(player_id, {})
        draft_picks.append({
            "pick_no": int(pick.get("overallPickNumber") or 0),
            "round": int(pick.get("roundId") or 0),
            "pick_in_round": int(pick.get("roundPickNumber") or 0),
            "draft_slot": first_round_slots.get(team_id, 0),
            "roster_id": pick.get("teamId"),
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

    roster_settings = settings.get("rosterSettings") or {}
    roster_positions = []
    for raw_slot, raw_count in (roster_settings.get("lineupSlotCounts") or {}).items():
        try:
            slot = int(raw_slot)
            count = int(raw_count or 0)
        except (TypeError, ValueError):
            continue
        roster_positions.extend([_LINEUP_SLOT_NAMES.get(slot, str(slot))] * max(0, count))
    scoring_items = (settings.get("scoringSettings") or {}).get("scoringItems") or []
    acquisition = settings.get("acquisitionSettings") or {}
    uses_budget = bool(acquisition.get("isUsingAcquisitionBudget"))

    final_week = int(status.get("finalScoringPeriod") or 18)
    latest_week = int(status.get("latestScoringPeriod") or 0)
    is_complete = latest_week >= final_week and bool(by_finish)
    payload = {
        "league_id": str(league_id),
        "draft_id": f"espn:{league_id}:{season}",
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
            "total_rosters": len(teams),
            "roster_positions": roster_positions,
            "scoring_settings": {
                "rec": _scoring_value(scoring_items, {41, 53}),
                "pass_td": _scoring_value(scoring_items, {4}),
            },
            "waiver_type": 2 if uses_budget else 0,
            "waiver_budget": acquisition.get("acquisitionBudget"),
        },
    }
    return payload, player_directory


@st.cache_data(ttl=3600, max_entries=_ESPN_SEASON_CACHE_ENTRIES)
def fetch_one_season(league_id: str, season: int) -> tuple[str, dict, dict]:
    data = _espn_get(
        str(league_id),
        int(season),
        ("mTeam", "mRoster", "mMatchup", "mSettings", "mStandings", "mDraftDetail"),
    )
    status = data.get("status") or {}
    first_week = int(status.get("firstScoringPeriod") or 1)
    final_week = int(status.get("finalScoringPeriod") or 18)
    weekly_rosters = _fetch_weekly_rosters(
        str(league_id), int(season), first_week, final_week,
    )
    draft_player_ids = []
    for pick in (data.get("draftDetail") or {}).get("picks") or []:
        try:
            player_id = int(pick.get("playerId") or 0)
        except (AttributeError, TypeError, ValueError):
            continue
        if player_id > 0:
            draft_player_ids.append(player_id)
    drafted_players = _fetch_drafted_players(
        int(season), tuple(sorted(set(draft_player_ids))),
    )
    payload, player_directory = normalize_season(
        str(league_id), int(season), data, weekly_rosters, drafted_players,
    )
    return str(season), payload, player_directory


def fetch_one_season_private(
    league_id: str,
    season: int,
    espn_s2: str,
    swid: str,
) -> tuple[str, dict, dict]:
    """Fetch one private season without caching its league or roster payloads."""
    data = _espn_get_private(
        str(league_id),
        int(season),
        ("mTeam", "mRoster", "mMatchup", "mSettings", "mStandings", "mDraftDetail"),
        espn_s2,
        swid,
    )
    status = data.get("status") or {}
    first_week = int(status.get("firstScoringPeriod") or 1)
    final_week = int(status.get("finalScoringPeriod") or 18)
    weekly_rosters = _fetch_weekly_rosters_private(
        str(league_id), int(season), first_week, final_week, espn_s2, swid,
    )
    draft_player_ids = []
    for pick in (data.get("draftDetail") or {}).get("picks") or []:
        try:
            player_id = int(pick.get("playerId") or 0)
        except (AttributeError, TypeError, ValueError):
            continue
        if player_id > 0:
            draft_player_ids.append(player_id)
    drafted_players = _fetch_drafted_players_private(
        int(season), tuple(sorted(set(draft_player_ids))),
    )
    payload, player_directory = normalize_season(
        str(league_id), int(season), data, weekly_rosters, drafted_players,
    )
    return str(season), payload, player_directory


def fetch_history(
    league_id: str,
    start_season: int,
    max_seasons: int | None = None,
) -> dict:
    chain = history_chain(str(league_id), int(start_season))
    if max_seasons is not None:
        chain = chain[:max(1, int(max_seasons))]
    league_name = chain[0]["name"] if chain else "League"
    seasons = {}
    player_directory = {}
    for item in chain:
        try:
            year, payload, players = fetch_one_season(
                item["league_id"], int(item["season"]),
            )
        except EspnLeagueError:
            continue
        seasons[year] = payload
        player_directory.update(players)
    return {
        "league_name": league_name or "League",
        "seasons": seasons,
        "player_directory": player_directory,
        "provider": "ESPN",
    }


def fetch_history_private(
    league_id: str,
    start_season: int,
    espn_s2: str,
    swid: str,
    max_seasons: int | None = None,
) -> dict:
    """Compose private history in caller/session memory only."""
    chain = history_chain_private(league_id, start_season, espn_s2, swid)
    if max_seasons is not None:
        chain = chain[:max(1, int(max_seasons))]
    league_name = chain[0]["name"] if chain else "League"
    seasons = {}
    player_directory = {}
    for item in chain:
        try:
            year, payload, players = fetch_one_season_private(
                item["league_id"], int(item["season"]), espn_s2, swid,
            )
        except EspnLeagueError:
            continue
        seasons[year] = payload
        player_directory.update(players)
    return {
        "league_name": league_name or "League",
        "seasons": seasons,
        "player_directory": player_directory,
        "provider": "ESPN",
    }
