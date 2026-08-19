"""Language fence for the Season Totals page.

While this page is a projection, not a card, it may not name a side, a stake,
or a payout. Tests import the banned set from here so the list cannot drift
from a retyped copy in the page suite.

This is a display guard only. Model code lives in seasonal_totals_v2_beta.
"""
from __future__ import annotations

import re

BANNED = frozenset({
    "bet", "bets", "betting", "edge", "edges", "lock", "locks", "value", "play", "plays",
    "pick", "picks", "side", "sides", "ev", "kelly", "roi", "profit", "profitable",
    "confidence", "tier", "tiers", "vig", "juice", "price", "prices", "odds", "payout",
    "stake", "wager", "recommendation", "breakeven",
})

_SPLIT = re.compile(r"[^a-z0-9]+")


def tokens(s) -> set:
    """Split on non-alphanumerics so games_played yields {games, played}, not {play}."""
    return {t for t in _SPLIT.split(str(s).lower()) if t}
