"""Validation for optional hash-bound per-game enrichment artifacts."""
from __future__ import annotations

import json
import math
from pathlib import Path

from .contract import MATCHUP_SCHEMA_VERSION


def read_matchup_artifact(path: str | Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def validate_matchup_artifact(
    path: str | Path,
    *,
    expected_game_ids: set[str],
    season: int,
    week: int,
) -> tuple[list[str], dict]:
    """Return errors plus lightweight checks for one candidate sidecar."""
    errors: list[str] = []
    checks: dict[str, object] = {}
    source = Path(path)
    if not source.is_file():
        return [f"matchup artifact does not exist: {source}"], checks
    try:
        payload = read_matchup_artifact(source)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        return [f"matchup artifact could not be read: {exc}"], checks
    if not isinstance(payload, dict):
        return ["matchup artifact root must be an object"], checks
    if payload.get("schema_version") != MATCHUP_SCHEMA_VERSION:
        errors.append(
            f"unsupported matchup schema_version {payload.get('schema_version')!r}; "
            f"want {MATCHUP_SCHEMA_VERSION}"
        )
    try:
        if int(payload.get("season")) != int(season) or int(payload.get("week")) != int(week):
            errors.append("matchup artifact season/week disagree with prediction metadata")
    except (TypeError, ValueError):
        errors.append("matchup artifact season/week must be integers")
    games = payload.get("games")
    if not isinstance(games, dict):
        return errors + ["matchup artifact games must be an object keyed by game_id"], checks
    actual_ids = set(map(str, games))
    if actual_ids != expected_game_ids:
        errors.append(
            "matchup artifact game coverage mismatch; "
            f"missing={sorted(expected_game_ids-actual_ids)[:8]} "
            f"extra={sorted(actual_ids-expected_game_ids)[:8]}"
        )
    checks["matchup_games"] = len(actual_ids)

    for game_id in sorted(actual_ids & expected_game_ids):
        game = games.get(game_id)
        if not isinstance(game, dict):
            errors.append(f"{game_id}: matchup entry must be an object")
            continue
        model = game.get("model")
        if not isinstance(model, dict):
            errors.append(f"{game_id}: model section is required")
        else:
            inputs = model.get("inputs")
            drivers = model.get("drivers")
            method = str(model.get("explanation_method") or "").strip()
            if not isinstance(inputs, list) or not inputs:
                errors.append(f"{game_id}: model.inputs must be a nonempty list")
            if not isinstance(drivers, list) or not drivers:
                errors.append(f"{game_id}: model.drivers must be a nonempty list")
            if not method:
                errors.append(f"{game_id}: model.explanation_method is required")
            for label, values in (("inputs", inputs), ("drivers", drivers)):
                if not isinstance(values, list):
                    continue
                names = []
                for item in values:
                    if not isinstance(item, dict) or not str(item.get("feature") or "").strip():
                        errors.append(f"{game_id}: model.{label} entries require feature names")
                        continue
                    names.append(str(item["feature"]))
                    if "value" not in item:
                        errors.append(f"{game_id}: model.{label}.{item['feature']} lacks value")
                    if label == "drivers":
                        try:
                            contribution = float(item["contribution"])
                            if not math.isfinite(contribution):
                                raise ValueError
                        except (KeyError, TypeError, ValueError):
                            errors.append(
                                f"{game_id}: driver {item.get('feature')} needs a finite contribution"
                            )
                if len(names) != len(set(names)):
                    errors.append(f"{game_id}: duplicate model.{label} feature names")

        context = game.get("context")
        if not isinstance(context, dict):
            errors.append(f"{game_id}: context section is required")
        else:
            for name in ("injuries", "weather"):
                section = context.get(name)
                source_meta = section.get("source") if isinstance(section, dict) else None
                if not isinstance(source_meta, dict) or not all(
                    str(source_meta.get(key) or "").strip() for key in ("name", "url", "timing")
                ):
                    errors.append(
                        f"{game_id}: context.{name}.source requires name, url, and timing"
                    )
        social = game.get("social")
        if not isinstance(social, dict) or not all(
            str(social.get(key) or "").strip() for key in ("title", "description")
        ):
            errors.append(f"{game_id}: social title and description are required")
    return errors, checks
