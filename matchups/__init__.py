"""Shareable matchup-detail contracts and release-backed data access."""

from .catalog import MatchupRoute, load_matchup_routes
from .detail import MatchupNotFound, load_matchup_detail

__all__ = [
    "MatchupNotFound",
    "MatchupRoute",
    "load_matchup_detail",
    "load_matchup_routes",
]
