"""Shared Draft Board freeze date. No other seasonal settings live here."""
import os
from datetime import date


# Kickoff-week default: the scheduled pre-draft ADP refresh pauses on or after this date.
SEASON_START = date(2026, 9, 9)


def board_refresh_season_start() -> date:
    """Return the environment override when present, else the shared kickoff default."""
    raw = os.environ.get("BOARD_REFRESH_SEASON_START")
    return date.fromisoformat(raw) if raw else SEASON_START


def app_today() -> date:
    """Clock used by date-dependent page copy.

    Visual tests freeze this with APP_TODAY=YYYY-MM-DD so the preseason banner
    does not flip the day the regular season starts. Production leaves it unset.
    """
    raw = os.environ.get("APP_TODAY")
    return date.fromisoformat(raw) if raw else date.today()
