"""Assemble one provenance-aware matchup view model."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from betting.live_2026 import row_display_high, row_high_dropped
from publishing.contract import sha256_file
from publishing.paths import resolve_site_path

from .catalog import find_released_game
from .contract import (
    MATCHUP_SCHEMA_VERSION,
    METEOSTAT_HOURLY_URL,
    NFLVERSE_INJURY_URL,
    is_demo_week,
    validate_matchup_detail,
)


class MatchupNotFound(LookupError):
    pass


def _number(value: Any) -> float | None:
    try:
        if value is None or pd.isna(value):
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _text(value: Any) -> str | None:
    try:
        if value is None or pd.isna(value):
            return None
    except (TypeError, ValueError):
        return None
    clean = str(value).strip()
    return clean if clean and clean.lower() != "nan" else None


def _primary(row: dict, preferred: str, fallback: str) -> float | None:
    value = _number(row.get(preferred))
    return value if value is not None else _number(row.get(fallback))


def _load_weather(root: Path, game_id: str) -> dict:
    path = root / "betting" / "nfl_weather_2014_2025.csv"
    if not path.is_file():
        return {
            "availability": "unavailable",
            "reason": "The archived weather file is unavailable.",
            "source": None,
        }
    frame = pd.read_csv(path, dtype={"game_id": "string"})
    match = frame[frame["game_id"].astype(str).eq(game_id)]
    if match.empty:
        return {
            "availability": "unavailable",
            "reason": "No station observation was archived for this game.",
            "source": {
                "name": "Meteostat hourly observations",
                "url": METEOSTAT_HOURLY_URL,
                "timing": "Observed at kickoff; not a frozen pregame forecast",
            },
        }
    row = match.iloc[-1].to_dict()
    return {
        "availability": "available",
        "temperature_f": _number(row.get("temp_f")),
        "wind_mph": _number(row.get("wind_mph")),
        "precipitation_in": _number(row.get("precip_in")),
        "humidity_pct": _number(row.get("humidity_pct")),
        "station_name": _text(row.get("station_name")),
        "kickoff_utc": _text(row.get("kickoff_utc")),
        "source": {
            "name": "Meteostat hourly observations",
            "url": METEOSTAT_HOURLY_URL,
            "timing": "Observed at kickoff; not a frozen pregame forecast",
        },
    }


def _load_injury_archive(root: Path) -> dict:
    path = root / "data" / "matchups" / "injuries_2025_weeks10_16.json"
    if not path.is_file():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return {}
    return payload if payload.get("schema_version") == 1 else {}


def _load_demo_model_archive(root: Path, game_id: str) -> tuple[dict, dict]:
    path = root / "data" / "matchups" / "model_2025_weeks10_16.json"
    if not path.is_file():
        return {}, {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return {}, {}
    if payload.get("schema_version") != 1 or not isinstance(payload.get("games"), dict):
        return {}, {}
    record = payload["games"].get(game_id)
    return (record if isinstance(record, dict) else {}), payload


def _injury_context(root: Path, season: int, week: int, away: str, home: str) -> dict:
    payload = _load_injury_archive(root)
    teams = payload.get("teams") if isinstance(payload, dict) else None
    source = {
        "name": "nflverse weekly injury reports",
        "url": NFLVERSE_INJURY_URL,
        "timing": "Archived final weekly report; snapshot time unavailable",
    }
    if not isinstance(teams, dict):
        unavailable = {"availability": "unavailable", "players": [], "reason": "The weekly injury archive is unavailable."}
        return {"source": source, "away": dict(unavailable), "home": dict(unavailable)}

    def one(team: str) -> dict:
        entry = teams.get(f"{season}_{week:02d}_{team}")
        if not isinstance(entry, dict):
            return {
                "team": team,
                "availability": "unavailable",
                "players": [],
                "reason": "No weekly report was archived for this team.",
            }
        return {
            "team": team,
            "availability": "available",
            "players": list(entry.get("players") or []),
            "counts": dict(entry.get("counts") or {}),
            "reason": None,
        }

    return {"source": source, "away": one(away), "home": one(home)}


def _model_outputs(row: dict) -> list[dict]:
    definitions = (
        ("Ensemble", "ens_predicted_margin", "ens_model_edge", "ens_recommendation"),
        ("XGBoost", "predicted_margin", "model_edge", "recommendation"),
        ("Ridge", "ridge_predicted_margin", "ridge_model_edge", "ridge_recommendation"),
        ("LightGBM", "lgbm_predicted_margin", "lgbm_model_edge", "lgbm_recommendation"),
    )
    outputs = []
    for label, margin, edge, recommendation in definitions:
        margin_value = _number(row.get(margin))
        if margin_value is None:
            continue
        outputs.append(
            {
                "model": label,
                "projected_margin": margin_value,
                "edge": _number(row.get(edge)),
                "recommendation": _text(row.get(recommendation)),
            }
        )
    return outputs


def _load_release_enrichment(root: Path, build: dict, game_id: str) -> dict:
    """Read an optional future per-game artifact only after verifying its hash."""
    descriptor = build.get("matchups")
    if not isinstance(descriptor, dict):
        return {}
    relative, expected = descriptor.get("artifact"), descriptor.get("sha256")
    if not relative or not expected:
        return {}
    try:
        path = resolve_site_path(str(relative), root)
    except (TypeError, ValueError):
        return {}
    if not path.is_file():
        return {}
    digest = sha256_file(path)
    if digest != str(expected):
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return {}
    games = payload.get("games") if isinstance(payload, dict) else None
    found = games.get(game_id) if isinstance(games, dict) else None
    return found if isinstance(found, dict) else {}


def _merge_enrichment(detail: dict, enrichment: dict) -> None:
    """Merge only explicitly supported, provenance-bearing future fields."""
    model = enrichment.get("model")
    if isinstance(model, dict):
        for key in ("inputs", "drivers", "explanation_method"):
            if key in model:
                detail["model"][key] = model[key]
        if detail["model"].get("drivers"):
            detail["model"]["driver_availability"] = "available"
    context = enrichment.get("context")
    if isinstance(context, dict):
        for key in ("injuries", "weather"):
            value = context.get(key)
            if isinstance(value, dict) and value.get("source"):
                detail["context"][key] = value
    social = enrichment.get("social")
    if isinstance(social, dict):
        detail["social"].update(social)


def load_matchup_detail(root: str | Path, game_id: str) -> dict:
    base = Path(root).resolve()
    released = find_released_game(base, game_id)
    if released is None:
        raise MatchupNotFound(str(game_id))
    route, row, build = released.route, released.row, released.build
    projected = _primary(row, "ens_predicted_margin", "predicted_margin")
    edge = _primary(row, "ens_model_edge", "model_edge")
    spread = _number(row.get("tuesday_spread_line"))
    if spread is None:
        spread = _number(row.get("spread_line"))
    recommendation = _text(row.get("ens_recommendation")) or _text(row.get("recommendation"))
    if projected is None or edge is None or spread is None or recommendation is None:
        raise MatchupNotFound(f"{game_id}: released row lacks prediction fields")

    legacy = is_demo_week(route.season, route.week)
    tier = (_text(row.get("consensus_tier")) or "PASS").upper()
    if route.season >= 2026:
        tier = "HIGH" if row_display_high(row) else "PASS"
    if tier not in {"HIGH", "MEDIUM", "PASS"}:
        tier = "PASS"
    freeze_at = None if legacy or _text(row.get("mode")) == "backfill" else _text(row.get("logged_at"))
    archive_logged_at = _text(row.get("logged_at")) if legacy else None
    dropped = bool(route.season >= 2026 and row_high_dropped(row))
    demo_model, demo_model_meta = _load_demo_model_archive(base, route.game_id) if legacy else ({}, {})

    actual_margin = _number(row.get("actual_margin"))
    home_score, away_score = _number(row.get("home_score")), _number(row.get("away_score"))
    correct = _number(row.get("ens_model_correct"))
    if correct is None:
        correct = _number(row.get("model_correct"))
    result_status = "pending"
    ats_result = None
    if actual_margin is not None:
        result_status = "final"
        if abs(actual_margin - spread) < 1e-9:
            ats_result = "PUSH"
        elif correct is not None:
            ats_result = "WIN" if int(correct) == 1 else "LOSS"

    events = []
    if freeze_at:
        events.append({"kind": "pick", "timestamp": freeze_at, "label": "Pick frozen", "detail": recommendation})
    else:
        events.append(
            {
                "kind": "archive",
                "timestamp": str(build.get("produced_at") or archive_logged_at or "Unavailable"),
                "label": "Historical prediction backfill",
                "detail": "No authentic pregame freeze timestamp was retained.",
            }
        )
    pick_line = _number(row.get("pick_line"))
    closing_line = _number(row.get("closing_line"))
    if pick_line is not None:
        events.append({"kind": "line", "timestamp": freeze_at, "label": "Pick line", "detail": f"{pick_line:+.1f}"})
    if closing_line is not None:
        events.append({"kind": "line", "timestamp": None, "label": "Closing line", "detail": f"{closing_line:+.1f}"})
    if result_status == "final":
        score = f"{route.away_team} {int(away_score)} · {route.home_team} {int(home_score)}" if away_score is not None and home_score is not None else f"Home margin {actual_margin:+.1f}"
        events.append({"kind": "result", "timestamp": route.gameday, "label": "Final", "detail": score})

    detail = {
        "schema_version": MATCHUP_SCHEMA_VERSION,
        "game": {
            "game_id": route.game_id,
            "season": route.season,
            "week": route.week,
            "away_team": route.away_team,
            "home_team": route.home_team,
            "gameday": route.gameday,
            "gametime": _text(row.get("gametime")),
            "slug": route.slug,
            "historical_demo": legacy,
        },
        "release": {
            "build_id": route.build_id,
            "model_version": str(build.get("model_version") or "Unavailable"),
            "produced_at": str(build.get("produced_at") or "Unavailable"),
            "published_at": str(build.get("published_at") or "Unavailable"),
            "status": str(build.get("status") or "Published"),
        },
        "prediction": {
            "projected_margin": projected,
            "market_spread": spread,
            "model_edge": edge,
            "recommendation": recommendation,
            "recommended_team": route.home_team if edge > 0 else route.away_team if edge < 0 else None,
        },
        "status": {
            "label": tier,
            "is_high": tier == "HIGH",
            "high_dropped": dropped,
            "freeze_at": freeze_at,
            "archive_logged_at": archive_logged_at,
            "mode": _text(row.get("mode")),
            "freeze_note": (
                "This row was backfilled after the games; it is not evidence of a pregame freeze."
                if legacy
                else "Frozen timestamp from the validated prediction release."
            ),
        },
        "model": {
            "version": str(build.get("model_version") or "Unavailable"),
            "outputs": _model_outputs(row),
            "inputs": list(demo_model.get("inputs") or []),
            "drivers": list(demo_model.get("drivers") or []),
            "explanation_method": demo_model.get("explanation_method"),
            "base_value": demo_model.get("base_value"),
            "reconstructed_margin": demo_model.get("reconstructed_margin"),
            "driver_availability": (
                "available" if demo_model.get("drivers")
                else "unavailable" if legacy
                else "awaiting_sidecar"
            ),
            "driver_note": (
                (
                    f"Post-hoc reconstruction from the exact generating code and model blobs at "
                    f"{str(demo_model_meta.get('source_commit') or 'the archived release')[:12]}; "
                    "all 105 rounded margins reproduced."
                )
                if legacy and demo_model.get("drivers")
                else "The archived release could not be reproduced from its retained code and model artifacts."
                if legacy
                else "This release did not include a hash-verified per-game explanation sidecar."
            ),
        },
        "context": {
            "injuries": _injury_context(base, route.season, route.week, route.away_team, route.home_team),
            "weather": _load_weather(base, route.game_id),
        },
        "history": {
            "events": events,
            "line_history_available": pick_line is not None or closing_line is not None,
            "note": (
                "Only one historical prediction snapshot was retained; chronological pick and line movement were not archived."
                if legacy and pick_line is None and closing_line is None
                else None
            ),
        },
        "result": {
            "status": result_status,
            "home_score": home_score,
            "away_score": away_score,
            "actual_margin": actual_margin,
            "ats_result": ats_result,
        },
        "social": {
            "title": f"{route.away_team} at {route.home_team} · JoScho Analytics",
            "description": f"{route.season} Week {route.week}: {recommendation}, {tier}, model edge {edge:+.1f} points.",
            "card_width": 1200,
            "card_height": 630,
        },
    }
    _merge_enrichment(detail, _load_release_enrichment(base, build, route.game_id))
    validate_matchup_detail(detail)
    return detail
