"""Hash-verified release catalog for hidden matchup routes."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from publishing.manifest import load_manifest, published_builds, resolve_build_artifact

from .contract import is_demo_week, matchup_slug, parse_game_id


@dataclass(frozen=True)
class MatchupRoute:
    game_id: str
    season: int
    week: int
    away_team: str
    home_team: str
    gameday: str
    slug: str
    title: str
    build_id: str


@dataclass(frozen=True)
class ReleasedGame:
    route: MatchupRoute
    row: dict
    build: dict


def _latest_prediction_builds(root: str | Path) -> list[dict]:
    manifest = load_manifest(root)
    latest: dict[tuple[int, int], dict] = {}
    for build in published_builds("predictions", manifest=manifest, root=root):
        season, week = int(build["season"]), int(build["week"])
        if not (is_demo_week(season, week) or season >= 2026):
            continue
        latest[(season, week)] = build
    return [latest[key] for key in sorted(latest)]


def load_released_games(root: str | Path) -> list[ReleasedGame]:
    """Return one route per game from the newest valid build for each week."""
    games: dict[str, ReleasedGame] = {}
    for build in _latest_prediction_builds(root):
        artifact = resolve_build_artifact(build, root=root, prefer_graded=True)
        if artifact is None:
            continue
        frame = pd.read_parquet(artifact) if artifact.suffix.lower() in {".parquet", ".pq"} else pd.read_csv(artifact)
        for row in frame.to_dict(orient="records"):
            game_id = str(row.get("game_id") or "")
            try:
                season, week, away, home = parse_game_id(game_id)
            except ValueError:
                continue
            if (season, week) != (int(build["season"]), int(build["week"])):
                continue
            if str(row.get("away_team")) != away or str(row.get("home_team")) != home:
                continue
            slug = matchup_slug(season, week, away, home)
            route = MatchupRoute(
                game_id=game_id,
                season=season,
                week=week,
                away_team=away,
                home_team=home,
                gameday=str(row.get("gameday") or ""),
                slug=slug,
                title=f"{away} at {home} · {season} W{week}",
                build_id=str(build.get("build_id") or ""),
            )
            games[game_id] = ReleasedGame(route=route, row=row, build=build)
    return sorted(
        games.values(),
        key=lambda game: (
            -game.route.season,
            game.route.week,
            game.route.gameday,
            game.route.away_team,
        ),
    )


def load_matchup_routes(root: str | Path) -> list[MatchupRoute]:
    return [game.route for game in load_released_games(root)]


def find_released_game(root: str | Path, game_id: str) -> ReleasedGame | None:
    return next(
        (game for game in load_released_games(root) if game.route.game_id == str(game_id)),
        None,
    )
