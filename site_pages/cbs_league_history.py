"""CBS league adapter for the League History page.

CBS Commissioner leagues do not offer an ESPN-style public-view feed. The
commissioner site exposes the unofficial JSON routes below when the request
carries a signed-in access token. This module keeps that provider-specific
shape outside the page and normalizes it to the existing League History
payload. Credentialed responses deliberately are never shared-cached.
"""
from __future__ import annotations

import concurrent.futures as cf
import re
from collections.abc import Mapping

import requests


CBS_MIN_SEASON = 2018
CBS_MAX_HISTORY_SEASONS = 10
_WEEK_FETCH_WORKERS = 6
_BENCH_STATUS = {"BN", "RS", "R", "I", "IR", "NA", "TA"}
_CORE_POSITIONS = {"QB", "RB", "WR", "TE", "K", "DST", "D/ST", "DEF"}
_REC_ABBR = {"REC", "RECS", "RECEPTION", "RECEPTIONS"}
_PASS_TD_ABBR = {"PTD", "TD-P", "TDP", "PASS_TD", "P-TD", "PYTD", "PASSING_TD"}
_URL_RE = re.compile(
    r"(?:https?://)?([a-z0-9][a-z0-9_-]{0,39})\.football\.cbssports\.com",
    re.I,
)
_ID_RE = re.compile(r"^[a-z0-9][a-z0-9_-]{1,39}$", re.I)
_RESERVED_IDS = {"www", "api", "fantasy", "help", "login"}
_HEADERS = {
    "User-Agent": "Mozilla/5.0 (JoSchoAnalytics League History CBS)",
    "Accept": "*/*",
}


class CbsLeagueError(RuntimeError):
    """A user-facing CBS access or payload error."""

    def __init__(self, message: str, *, status_code: int | None = None):
        super().__init__(message)
        self.status_code = status_code


def parse_league_ref(raw_league_id: str) -> str:
    """Return the CBS league slug from an ID or commissioner URL."""
    text = str(raw_league_id or "").strip()
    matched = _URL_RE.search(text)
    if matched:
        return matched.group(1).lower()
    return text.strip().strip("/").split("/")[0].lower()


def league_id_error(raw_league_id: str) -> str | None:
    league_id = parse_league_ref(raw_league_id)
    if not league_id:
        return "Enter your CBS league ID to load your league history."
    if league_id in _RESERVED_IDS:
        return (
            "That does not look like a CBS league ID. Copy the subdomain from "
            "https://{ID}.football.cbssports.com."
        )
    if not _ID_RE.match(league_id):
        return (
            "CBS league IDs are the subdomain of your league URL, such as cbshelp "
            "in https://cbshelp.football.cbssports.com."
        )
    return None


def season_error(raw_season, current_year: int) -> str | None:
    try:
        season = int(raw_season)
    except (TypeError, ValueError):
        return "Enter the four-digit CBS season from your league site."
    if not CBS_MIN_SEASON <= season <= int(current_year):
        return (
            f"CBS League History currently supports seasons "
            f"{CBS_MIN_SEASON}-{current_year}."
        )
    return None


def private_credentials_error(access_token: str) -> str | None:
    if not str(access_token or "").strip():
        return "Private CBS leagues need the signed-in access token."
    return None


def _token_value(raw_value: str) -> str:
    value = str(raw_value or "").strip().strip('"').strip("'")
    prefix, separator, remainder = value.partition("=")
    if separator and prefix.strip().casefold() in {"authorization", "token", "access_token"}:
        value = remainder.strip().strip('"').strip("'")
    if value.casefold().startswith("bearer "):
        value = value[7:].strip()
    return value


def _as_list(node) -> list:
    if node is None:
        return []
    if isinstance(node, list):
        return [item for item in node if item is not None]
    if isinstance(node, dict):
        numbered = [
            (int(key), value)
            for key, value in node.items()
            if str(key).isdigit()
        ]
        if numbered:
            numbered.sort()
            return [value for _, value in numbered]
        return [node]
    return []


def _body(payload: Mapping | None) -> dict:
    if not isinstance(payload, dict):
        return {}
    body = payload.get("body")
    return body if isinstance(body, dict) else payload


def _int(value, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _float(value, default: float = 0.0) -> float:
    try:
        return round(float(value or 0), 2)
    except (TypeError, ValueError):
        return default


def _period_id(period: Mapping) -> int:
    raw = period.get("id")
    parsed = _int(raw, 0)
    if parsed:
        return parsed
    digits = "".join(ch for ch in str(period.get("label") or "") if ch.isdigit())
    return int(digits) if digits else 0


def _team_id(node: Mapping | None) -> str:
    node = node or {}
    return str(node.get("id") or node.get("team_id") or "").strip()


def _roster_id(team_id: str):
    return int(team_id) if team_id.isdigit() else team_id


def _api_url(
    league_id: str,
    route: str,
    season: int,
    extra: Mapping[str, object] | None = None,
) -> str:
    params = [
        "version=3.0",
        "response_format=json",
        "sport=football",
        f"league_id={league_id}",
        f"year={int(season)}",
    ]
    for key, value in (extra or {}).items():
        params.append(f"{key}={value}")
    return (
        f"https://{league_id}.football.cbssports.com/api{route}?"
        + "&".join(params)
    )


def _request_json(url: str, token: str) -> dict:
    try:
        response = requests.get(
            url,
            headers={**_HEADERS, "Authorization": token},
            timeout=20,
        )
    except requests.RequestException as exc:
        raise CbsLeagueError("CBS did not respond. Try again in a moment.") from exc
    if response.status_code == 401:
        raise CbsLeagueError(
            "CBS denied access. Recopy the access token from the same "
            "signed-in CBS browser session.",
            status_code=401,
        )
    if response.status_code == 404:
        raise CbsLeagueError(
            "CBS did not return a league for that ID and season.",
            status_code=404,
        )
    if response.status_code != 200:
        raise CbsLeagueError(
            f"CBS returned HTTP {response.status_code}. Try again in a moment.",
            status_code=response.status_code,
        )
    content_type = str(response.headers.get("content-type") or "").lower()
    text = response.text or ""
    if "html" in content_type or text.lstrip()[:15].lower().startswith(("<!doctype", "<html")):
        raise CbsLeagueError(
            "CBS denied access. Recopy the access token from the same "
            "signed-in CBS browser session.",
            status_code=401,
        )
    try:
        payload = response.json()
    except ValueError as exc:
        raise CbsLeagueError("CBS returned an unreadable league response.") from exc
    if not isinstance(payload, dict):
        raise CbsLeagueError("CBS returned an unexpected league response.")
    status = payload.get("statusCode")
    if status not in (None, 200, "200"):
        message = str(payload.get("statusMessage") or "")
        if "auth" in message.lower() or "token" in message.lower() or "login" in message.lower():
            raise CbsLeagueError(
                "CBS denied access. Recopy the access token from the same "
                "signed-in CBS browser session.",
                status_code=401,
            )
        raise CbsLeagueError(
            "CBS did not return a league for that ID and season.",
            status_code=_int(status, 400),
        )
    return payload


def _cbs_get_private(
    league_id: str,
    season: int,
    route: str,
    token: str,
    extra: Mapping[str, object] | None = None,
) -> dict:
    """Fetch one CBS resource without a shared or credential-keyed cache."""
    return _request_json(
        _api_url(str(league_id).strip(), route, int(season), extra),
        _token_value(token),
    )


def _details(payload: Mapping | None) -> dict:
    body = _body(payload)
    details = body.get("league_details") or body.get("details") or body
    return details if isinstance(details, dict) else {}


def history_chain_private(
    league_id: str,
    start_season: int,
    access_token: str,
) -> list[dict]:
    """Walk prior years for the same CBS slug. Credentialed; never cached."""
    slug = parse_league_ref(league_id)
    chain: list[dict] = []
    fingerprints: set[tuple] = set()
    for year in range(int(start_season), CBS_MIN_SEASON - 1, -1):
        if len(chain) >= CBS_MAX_HISTORY_SEASONS:
            break
        try:
            details = _details(_cbs_get_private(
                slug, year, "/league/details", access_token,
            ))
        except CbsLeagueError:
            break
        name = str(details.get("name") or "").strip()
        if not name and not details.get("num_teams"):
            break
        fingerprint = (
            name,
            str(details.get("current_period") or ""),
            str(details.get("season_status") or ""),
            str(details.get("num_teams") or ""),
        )
        if fingerprint in fingerprints:
            break
        fingerprints.add(fingerprint)
        chain.append({
            "league_id": slug,
            "season": str(year),
            "name": name or "League",
        })
    return chain


def _fetch_weekly_rosters_private(
    league_id: str,
    season: int,
    first_week: int,
    final_week: int,
    access_token: str,
) -> dict[int, dict]:
    weeks = range(max(1, int(first_week)), min(18, int(final_week)) + 1)

    def fetch_week(week: int) -> tuple[int, dict | None]:
        try:
            payload = _cbs_get_private(
                league_id,
                season,
                "/league/rosters",
                access_token,
                {"team_id": "all", "period": week},
            )
            return week, payload
        except CbsLeagueError:
            return week, None

    with cf.ThreadPoolExecutor(max_workers=_WEEK_FETCH_WORKERS) as pool:
        return {
            week: payload
            for week, payload in pool.map(fetch_week, weeks)
            if isinstance(payload, dict)
        }


def _owner_identity(team: Mapping) -> tuple[str, str]:
    owners = team.get("owners") or team.get("managers") or []
    owner = None
    for row in _as_list(owners):
        if isinstance(row, dict):
            owner = row
            break
    team_id = _team_id(team)
    if not isinstance(owner, dict):
        name = str(team.get("name") or "").strip()
        return (
            f"cbs-team-{team_id}" if team_id else "",
            name or (f"Manager {team_id}" if team_id else "—"),
        )
    owner_id = str(owner.get("id") or owner.get("user_id") or "").strip()
    username = str(owner.get("name") or owner.get("fullname") or "").strip()
    return (
        owner_id or (f"cbs-team-{team_id}" if team_id else ""),
        username or str(team.get("name") or "").strip() or (
            f"Manager {team_id}" if team_id else "—"
        ),
    )


def _position_label(raw) -> str:
    text = str(raw or "").strip().upper()
    if text in {"DEF", "D/ST", "DST"}:
        return "D/ST"
    return text


def _remember_player(player_directory: dict, player: Mapping | None) -> str:
    player = player or {}
    nested = player.get("player") if isinstance(player.get("player"), dict) else player
    player_id = str(nested.get("id") or player.get("id") or "").strip()
    if not player_id:
        return ""
    first = str(nested.get("firstname") or nested.get("first_name") or "").strip()
    last = str(nested.get("lastname") or nested.get("last_name") or "").strip()
    full = str(nested.get("fullname") or nested.get("full_name") or "").strip()
    if not full:
        full = " ".join(part for part in (first, last) if part)
    position = _position_label(
        nested.get("position")
        or nested.get("eligible_positions_display")
        or nested.get("primary_position")
    )
    if position not in _CORE_POSITIONS and position != "D/ST":
        eligible = nested.get("eligible_positions") or []
        for item in _as_list(eligible):
            label = _position_label(item if not isinstance(item, dict) else item.get("abbr"))
            if label in _CORE_POSITIONS or label == "D/ST":
                position = label
                break
    player_directory[player_id] = {
        "full_name": full,
        "first_name": first,
        "last_name": last,
        "position": position,
    }
    return player_id


def _stat_points(rules: Mapping | None, abbrs: set[str]) -> float:
    found = 0.0

    def walk(node) -> None:
        nonlocal found
        if found:
            return
        if isinstance(node, dict):
            abbr = str(node.get("abbr") or node.get("name") or "").upper()
            abbr = abbr.replace(" ", "_")
            if abbr in abbrs:
                for key in ("points", "value", "pts"):
                    if key in node:
                        found = _float(node.get(key))
                        return
            for value in node.values():
                walk(value)
        elif isinstance(node, list):
            for item in node:
                walk(item)

    walk(rules)
    return found


def _roster_positions(rules: Mapping | None) -> list[str]:
    roster = (rules or {}).get("roster") or {}
    positions: list[str] = []
    for row in _as_list(roster.get("positions")):
        if not isinstance(row, dict):
            continue
        label = _position_label(row.get("abbr") or row.get("position"))
        count = _int(row.get("max_active") or row.get("count") or 0)
        if label:
            positions.extend([label] * max(0, count))
    for row in _as_list(roster.get("statuses")):
        if not isinstance(row, dict):
            continue
        description = str(row.get("description") or "").lower()
        count = _int(row.get("max") or 0)
        if "reserve" in description or "bench" in description:
            positions.extend(["BN"] * max(0, count))
        elif "injur" in description:
            positions.extend(["IR"] * max(0, count))
    return positions


def _standings_teams(standings_payload: Mapping | None) -> list[dict]:
    body = _body(standings_payload)
    overall = body.get("overall_standings") or body.get("standings") or body
    if not isinstance(overall, dict):
        return []
    if overall.get("divisions"):
        teams = []
        for division in _as_list(overall.get("divisions")):
            if isinstance(division, dict):
                teams.extend(_as_list(division.get("teams")))
        return [row for row in teams if isinstance(row, dict)]
    return [row for row in _as_list(overall.get("teams")) if isinstance(row, dict)]


def _schedule_periods(schedule_payload: Mapping | None) -> list[dict]:
    body = _body(schedule_payload)
    schedule = body.get("schedule") or body
    if not isinstance(schedule, dict):
        return []
    return [row for row in _as_list(schedule.get("periods")) if isinstance(row, dict)]


def _draft_picks_raw(draft_payload: Mapping | None) -> list[dict]:
    body = _body(draft_payload)
    results = body.get("draft_results") or body.get("draft") or body
    if not isinstance(results, dict):
        return []
    return [row for row in _as_list(results.get("picks")) if isinstance(row, dict)]


def _weekly_scoring_index(scoring_payload: Mapping | None, player_directory: dict) -> dict[tuple[str, int], float]:
    body = _body(scoring_payload)
    weekly = body.get("weekly_scoring") or body
    players = weekly.get("players") if isinstance(weekly, dict) else []
    index: dict[tuple[str, int], float] = {}
    for player in _as_list(players):
        if not isinstance(player, dict):
            continue
        nested = player.get("player") if isinstance(player.get("player"), dict) else player
        player_id = _remember_player(player_directory, nested)
        if not player_id:
            continue
        for period in _as_list(player.get("periods")):
            if not isinstance(period, dict):
                continue
            week = _int(period.get("period") or period.get("week"))
            if week:
                index[(player_id, week)] = _float(period.get("score") or period.get("points"))
    return index


def _roster_teams(week_payload: Mapping | None) -> list[dict]:
    body = _body(week_payload)
    rosters = body.get("rosters") or body
    if isinstance(rosters, dict):
        return [row for row in _as_list(rosters.get("teams")) if isinstance(row, dict)]
    return [row for row in _as_list(rosters) if isinstance(row, dict)]


def normalize_season(
    league_id: str,
    season: int,
    details: Mapping,
    rules: Mapping,
    teams: list[Mapping],
    standings_teams: list[Mapping],
    schedule_periods: list[Mapping],
    draft_picks_raw: list[Mapping],
    weekly_rosters: Mapping[int, Mapping] | None = None,
    weekly_scoring: Mapping | None = None,
) -> tuple[dict, dict]:
    """Normalize one CBS season to the League History payload."""
    weekly_rosters = weekly_rosters or {}
    team_by_id = {
        _team_id(team): team
        for team in teams
        if isinstance(team, Mapping) and _team_id(team)
    }
    owner_by_team = {
        team_id: _owner_identity(team)[0]
        for team_id, team in team_by_id.items()
    }
    standings_by_id = {
        _team_id(row): row
        for row in standings_teams
        if isinstance(row, Mapping) and _team_id(row)
    }

    standings = []
    for team_id, team in team_by_id.items():
        row = standings_by_id.get(team_id, {})
        owner_id, username = _owner_identity(team)
        standings.append({
            "roster_id": _roster_id(team_id),
            "owner_id": owner_id,
            "username": username,
            "team_name": str(team.get("name") or ""),
            "wins": _int(row.get("wins")),
            "losses": _int(row.get("losses")),
            "fpts": _float(row.get("points_scored") or row.get("points")),
            "fpts_against": _float(row.get("points_against")),
            "playoff_finish": _int(row.get("order") or row.get("rank") or 0) or None,
        })
    standings.sort(key=lambda row: (
        row["playoff_finish"] or 999,
        -row["wins"],
        -row["fpts"],
    ))

    matchups = []
    matchup_by_team_week: dict[tuple[str, int], str] = {}
    champ_home = None
    champ_away = None
    for period in schedule_periods:
        week = _period_id(period)
        if not week:
            continue
        period_type = str(period.get("type") or "").lower()
        period_is_playoff = "playoff" in period_type
        for matchup in _as_list(period.get("matchups")):
            if not isinstance(matchup, dict):
                continue
            home = matchup.get("home_team") or {}
            away = matchup.get("away_team") or {}
            rid_a = _team_id(home)
            rid_b = _team_id(away)
            if not rid_a or not rid_b:
                continue
            score_a = _float(home.get("points"))
            score_b = _float(away.get("points"))
            if score_a == 0 and score_b == 0:
                continue
            matchup_type = str(matchup.get("type") or "").lower()
            is_playoff = period_is_playoff or matchup_type in {"playoff", "consolation"}
            matchup_id = str(matchup.get("id") or f"{week}:{rid_a}:{rid_b}")
            matchup_by_team_week[(rid_a, week)] = matchup_id
            matchup_by_team_week[(rid_b, week)] = matchup_id
            matchups.append({
                "season": str(season),
                "week": week,
                "is_playoff": bool(is_playoff),
                "rid_a": rid_a,
                "score_a": score_a,
                "rid_b": rid_b,
                "score_b": score_b,
            })
            if _int(matchup.get("championship")) == 1:
                champ_home, champ_away = home, away

    if champ_home and champ_away:
        winner = champ_home if str(champ_home.get("result") or "").upper() == "W" else champ_away
        loser = champ_away if winner is champ_home else champ_home
        winner_id, loser_id = _team_id(winner), _team_id(loser)
        for row in standings:
            if str(row["roster_id"]) == winner_id:
                row["playoff_finish"] = 1
            elif str(row["roster_id"]) == loser_id:
                row["playoff_finish"] = 2

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

    player_directory: dict = {}
    scoring_index = _weekly_scoring_index(weekly_scoring, player_directory)
    roster_entries = []
    for week, week_payload in sorted(weekly_rosters.items()):
        for team in _roster_teams(week_payload):
            team_id = _team_id(team)
            player_ids = []
            starters = []
            player_points = {}
            for player in _as_list(team.get("players")):
                if not isinstance(player, dict):
                    continue
                player_id = _remember_player(player_directory, player)
                if not player_id:
                    continue
                player_ids.append(player_id)
                status = str(player.get("roster_status") or "").upper()
                slot = _position_label(player.get("roster_pos") or player.get("roster_position"))
                if status not in _BENCH_STATUS and slot not in _BENCH_STATUS:
                    starters.append(player_id)
                player_points[player_id] = scoring_index.get(
                    (player_id, int(week)),
                    _float(player.get("points")),
                )
            roster_entries.append({
                "season": str(season),
                "week": int(week),
                "roster_id": _roster_id(team_id),
                "matchup_id": matchup_by_team_week.get((team_id, int(week))),
                "players": player_ids,
                "starters": starters,
                "players_points": player_points,
            })

    first_round_slots = {
        _team_id(pick.get("team") if isinstance(pick.get("team"), dict) else {}):
        _int(pick.get("round_pick"))
        for pick in draft_picks_raw
        if _int(pick.get("round")) == 1
    }
    draft_picks = []
    for pick in draft_picks_raw:
        player_node = pick.get("player") if isinstance(pick.get("player"), dict) else {}
        team_node = pick.get("team") if isinstance(pick.get("team"), dict) else {}
        player_id = _remember_player(player_directory, player_node)
        team_id = _team_id(team_node)
        player = player_directory.get(player_id, {})
        pick_no = _int(pick.get("overall_pick") or pick.get("pick"))
        round_no = _int(pick.get("round"))
        pick_in_round = _int(pick.get("round_pick"))
        if not pick_no:
            continue
        draft_picks.append({
            "pick_no": pick_no,
            "round": round_no,
            "pick_in_round": pick_in_round,
            "draft_slot": first_round_slots.get(team_id, 0),
            "roster_id": _roster_id(team_id) if team_id else team_id,
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

    faab_budget = _int(
        ((rules.get("transactions") or {}).get("add_drop_faab_starting_budget") or {}).get("value")
        if isinstance(rules.get("transactions"), dict) else 0
    )
    if not faab_budget:
        faab_budget = _int(
            ((rules.get("transactions") or {}).get("faab_budget") or {}).get("value")
            if isinstance(rules.get("transactions"), dict) else 0
        )
    season_status = str(details.get("season_status") or "").lower()
    is_complete = season_status == "postseason" or bool(champ_home)
    payload = {
        "league_id": str(league_id),
        "draft_id": f"cbs:{league_id}:{season}",
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
            "total_rosters": len(standings) or _int(details.get("num_teams")),
            "roster_positions": _roster_positions(rules),
            "scoring_settings": {
                "rec": _stat_points(rules, _REC_ABBR),
                "pass_td": _stat_points(rules, _PASS_TD_ABBR),
            },
            "waiver_type": 2 if faab_budget else 0,
            "waiver_budget": faab_budget or None,
        },
    }
    return payload, player_directory


def _teams_from_payload(payload: Mapping | None) -> list[dict]:
    body = _body(payload)
    teams = body.get("teams") or []
    return [row for row in _as_list(teams) if isinstance(row, dict)]


def _rules_from_payload(payload: Mapping | None) -> dict:
    body = _body(payload)
    rules = body.get("rules")
    return rules if isinstance(rules, dict) else {}


def fetch_one_season_private(
    league_id: str,
    season: int,
    access_token: str,
) -> tuple[str, dict, dict]:
    slug = parse_league_ref(league_id)
    details_payload = _cbs_get_private(slug, season, "/league/details", access_token)
    details = _details(details_payload)
    if not details.get("name") and not details.get("num_teams"):
        raise CbsLeagueError(
            "CBS did not return a league for that ID and season.",
        )
    rules_payload = _cbs_get_private(slug, season, "/league/rules", access_token)
    teams_payload = _cbs_get_private(slug, season, "/league/teams", access_token)
    try:
        schedule_payload = _cbs_get_private(
            slug, season, "/league/schedules", access_token, {"period": "all"},
        )
    except CbsLeagueError:
        schedule_payload = {}
    try:
        standings_payload = _cbs_get_private(
            slug, season, "/league/standings/overall", access_token,
            {"period": _int(details.get("current_period") or details.get("regular_season_periods") or 1)},
        )
    except CbsLeagueError:
        standings_payload = {}
    try:
        draft_payload = _cbs_get_private(
            slug, season, "/league/draft/results", access_token,
        )
    except CbsLeagueError:
        draft_payload = {}
    first_week = 1
    final_week = min(
        18,
        max(
            _int(details.get("regular_season_periods") or 14)
            + _int(details.get("playoff_periods") or 0),
            _int(details.get("current_period") or 1),
        ),
    )
    weekly_rosters = _fetch_weekly_rosters_private(
        slug, int(season), first_week, final_week, access_token,
    )
    try:
        scoring_payload = _cbs_get_private(
            slug, season, "/league/fantasy-points/weekly-scoring", access_token,
            {"team_type": "roster", "team_id": "all"},
        )
    except CbsLeagueError:
        scoring_payload = {}
    payload, player_directory = normalize_season(
        slug,
        int(season),
        details,
        _rules_from_payload(rules_payload),
        _teams_from_payload(teams_payload),
        _standings_teams(standings_payload),
        _schedule_periods(schedule_payload),
        _draft_picks_raw(draft_payload),
        weekly_rosters,
        scoring_payload,
    )
    return str(season), payload, player_directory


def fetch_history_private(
    league_id: str,
    start_season: int,
    access_token: str,
    max_seasons: int | None = None,
) -> dict:
    """Compose private history in caller/session memory only."""
    chain = history_chain_private(league_id, start_season, access_token)
    if max_seasons is not None:
        chain = chain[:max(1, int(max_seasons))]
    league_name = chain[0]["name"] if chain else "League"
    seasons = {}
    player_directory = {}
    for item in chain:
        try:
            year, payload, players = fetch_one_season_private(
                item["league_id"], int(item["season"]), access_token,
            )
        except CbsLeagueError:
            continue
        seasons[year] = payload
        player_directory.update(players)
    return {
        "league_name": league_name or "League",
        "seasons": seasons,
        "player_directory": player_directory,
        "provider": "CBS",
    }
