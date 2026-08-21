"""Versioned contract for one public matchup detail record.

The contract deliberately separates a value from its provenance. A section can be
unavailable without making the whole page unavailable, but an unavailable section
must say why. That keeps the historical demo honest and gives 2026 producers a clear
shape for feature contributions and frozen context.
"""
from __future__ import annotations

import re
from typing import Mapping

MATCHUP_SCHEMA_VERSION = 1
DEMO_SEASON = 2025
DEMO_WEEKS = frozenset(range(10, 17))

NFLVERSE_INJURY_URL = (
    "https://github.com/nflverse/nflverse-data/releases/tag/injuries"
)
METEOSTAT_HOURLY_URL = "https://dev.meteostat.net/data/timeseries/hourly"

_TEAM_RE = re.compile(r"^[A-Z0-9]{2,4}$")
_GAME_ID_RE = re.compile(r"^(?P<season>\d{4})_(?P<week>\d{1,2})_(?P<away>[A-Z0-9]{2,4})_(?P<home>[A-Z0-9]{2,4})$")


class MatchupContractError(ValueError):
    """A matchup artifact or route violates the public contract."""


def matchup_slug(season: int, week: int, away_team: str, home_team: str) -> str:
    """Return the native Streamlit-safe flat route for one game."""
    away, home = str(away_team).upper(), str(home_team).upper()
    if not _TEAM_RE.fullmatch(away) or not _TEAM_RE.fullmatch(home):
        raise MatchupContractError(f"invalid team tokens: {away_team!r}, {home_team!r}")
    season_i, week_i = int(season), int(week)
    if season_i < 2000 or not 1 <= week_i <= 22:
        raise MatchupContractError(f"invalid season/week: {season_i}, {week_i}")
    return f"matchup-{season_i}-week-{week_i}-{away.lower()}-{home.lower()}"


def parse_game_id(game_id: str) -> tuple[int, int, str, str]:
    match = _GAME_ID_RE.fullmatch(str(game_id).strip())
    if not match:
        raise MatchupContractError(f"invalid game_id {game_id!r}")
    return (
        int(match.group("season")),
        int(match.group("week")),
        match.group("away"),
        match.group("home"),
    )


def validate_matchup_detail(detail: Mapping[str, object]) -> None:
    """Fail closed on the stable fields every detail page needs."""
    required = {"schema_version", "game", "release", "prediction", "status", "model", "context", "history", "result", "social"}
    missing = sorted(required - set(detail))
    if missing:
        raise MatchupContractError(f"matchup detail missing sections: {', '.join(missing)}")
    if detail.get("schema_version") != MATCHUP_SCHEMA_VERSION:
        raise MatchupContractError(
            f"unsupported matchup schema {detail.get('schema_version')!r}"
        )
    game = detail.get("game")
    if not isinstance(game, Mapping):
        raise MatchupContractError("game section must be an object")
    for key in ("game_id", "season", "week", "home_team", "away_team", "slug"):
        if game.get(key) in (None, ""):
            raise MatchupContractError(f"game.{key} is required")
    season, week, away, home = parse_game_id(str(game["game_id"]))
    expected = matchup_slug(season, week, away, home)
    if str(game["slug"]) != expected:
        raise MatchupContractError(f"game.slug {game['slug']!r} != {expected!r}")
    if (int(game["season"]), int(game["week"])) != (season, week):
        raise MatchupContractError("game season/week disagree with game_id")
    if (str(game["away_team"]), str(game["home_team"])) != (away, home):
        raise MatchupContractError("game teams disagree with game_id")

    prediction = detail.get("prediction")
    if not isinstance(prediction, Mapping):
        raise MatchupContractError("prediction section must be an object")
    for key in ("projected_margin", "market_spread", "model_edge", "recommendation"):
        if prediction.get(key) is None:
            raise MatchupContractError(f"prediction.{key} is required")

    status = detail.get("status")
    if not isinstance(status, Mapping) or status.get("label") not in {
        "HIGH", "MEDIUM", "PASS"
    }:
        raise MatchupContractError("status.label must be HIGH, MEDIUM, or PASS")


def is_demo_week(season: int, week: int) -> bool:
    return int(season) == DEMO_SEASON and int(week) in DEMO_WEEKS
