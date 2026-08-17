"""Pure transforms for the Sleeper League History intelligence views.

This module deliberately contains no Streamlit or network code.  The page owns
fetching/caching; these helpers turn the returned payload into auditable frames
for draft-room and manager-level roster analysis.
"""
from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from functools import lru_cache
from itertools import combinations

import pandas as pd


CORE_POSITIONS = ("QB", "RB", "WR", "TE")
DEFAULT_MIN_ROSTER_WEEKS = 4
SLEEPER_FAAB_WAIVER_TYPE = 2
FAAB_PAID_MIN = 5
PICKUP_LANE_GAP = 1.0
TRADE_CHART_LIMIT = 8
BUST_CHART_LIMIT = 8
INSIGHT_WINDOWS = ("Last season", "Last 3 seasons", "All available seasons")
DEFAULT_INSIGHT_WINDOW = "Last 3 seasons"
RIVALRY_WEEK_MODES = (
    "Classic Rivalries",
    "Maximum Drama",
    "Fresh Blood",
)
RIVALRY_SCORE_EXPLAIN = {
    "Classic Rivalries": (
        "Each 0-100 rivalry score is historical fit, not a prediction: Classic "
        "Rivalries mostly rewards long series, even records, and playoff meetings, "
        "and this slate is the pairing that maximizes that total for the whole league."
    ),
    "Maximum Drama": (
        "Each 0-100 rivalry score is historical fit, not a prediction: Maximum "
        "Drama mostly rewards close games, even series, playoff meetings, and lead "
        "changes, and this slate is the pairing that maximizes that total for the whole league."
    ),
    "Fresh Blood": (
        "Each 0-100 rivalry score is historical fit, not a prediction: Fresh Blood "
        "mostly rewards pairs who rarely meet and have similar career win rates, "
        "and this slate is the pairing that maximizes that total for the whole league."
    ),
}


def _player_name(metadata: Mapping | None, player_id: str) -> str:
    metadata = metadata or {}
    full_name = str(metadata.get("full_name") or "").strip()
    if full_name:
        return full_name
    first = str(metadata.get("first_name") or "").strip()
    last = str(metadata.get("last_name") or "").strip()
    return f"{first} {last}".strip() or player_id


def manager_identity_map(seasons: Mapping) -> dict[str, str]:
    """Return the most recent display name for each stable Sleeper user ID."""
    identities: dict[str, str] = {}
    for season in sorted(seasons, key=lambda value: str(value)):
        for row in seasons[season].get("standings", []):
            user_id = str(row.get("owner_id") or "").strip()
            username = str(row.get("username") or "").strip()
            if user_id and username and username not in {"?", "—"}:
                identities[user_id] = username
    return identities


def manager_display_labels(identities: Mapping[str, str]) -> dict[str, str]:
    """Return unique public labels while keeping stable Sleeper IDs internal.

    Sleeper display names can change and are not guaranteed to be unique.  The
    rivalry views aggregate on ``user_id`` first, then use the latest name.  A
    short ID suffix is shown only when two active identities currently share a
    display name.
    """
    clean = {
        str(user_id): str(name or "").strip()
        for user_id, name in identities.items()
        if str(user_id).strip() and str(name or "").strip()
    }
    counts: dict[str, int] = {}
    for name in clean.values():
        counts[name] = counts.get(name, 0) + 1
    return {
        user_id: (
            name if counts[name] == 1
            else f"{name} ({user_id[-4:]})"
        )
        for user_id, name in clean.items()
    }


def _clean_manager_name(name) -> str:
    value = str(name or "").strip()
    return "" if value in {"?", "—"} else value


def _playoff_finish_rank(finish) -> int | None:
    if finish is None or finish == "":
        return None
    try:
        return int(finish)
    except (TypeError, ValueError):
        return None


def _manager_stats_seed() -> dict:
    return {
        "titles": 0,
        "runner_ups": 0,
        "toilet_titles": 0,
        "toilet_appearances": 0,
        "seasons": 0,
        "wins": 0,
        "losses": 0,
        "best_finish": None,
        "fpts": 0.0,
    }


def bracket_roster_ids(bracket: Sequence[Mapping] | None) -> set[str]:
    """Every roster id that appeared in a Sleeper winners or losers bracket."""
    ids: set[str] = set()
    for match in bracket or []:
        if not isinstance(match, dict):
            continue
        for key in ("t1", "t2", "w", "l"):
            value = match.get(key)
            if value is None or value == "":
                continue
            text = str(value).strip()
            if text and text not in {"None", "?"}:
                ids.add(text)
    return ids


def bracket_placement_ranks(bracket: Sequence[Mapping] | None) -> dict[str, int]:
    """Map roster id to place. Winner of a p-game gets p, loser gets p+1."""
    ranks: dict[str, int] = {}
    for match in bracket or []:
        if not isinstance(match, dict):
            continue
        place, winner, loser = match.get("p"), match.get("w"), match.get("l")
        if place in (None, "") or winner is None or loser is None:
            continue
        try:
            place = int(place)
        except (TypeError, ValueError):
            continue
        winner, loser = str(winner), str(loser)
        current = ranks.get(winner)
        if current is None or place < current:
            ranks[winner] = place
        current = ranks.get(loser)
        loser_place = place + 1
        if current is None or loser_place < current:
            ranks[loser] = loser_place
    return ranks


def last_place_roster_ids(bracket: Sequence[Mapping] | None) -> list[str]:
    """Roster ids with the highest assigned place. Not the toilet-bowl title."""
    ranks = bracket_placement_ranks(bracket)
    if not ranks:
        return []
    worst = max(ranks.values())
    return sorted(rid for rid, place in ranks.items() if place == worst)


def bracket_title_roster_ids(bracket: Sequence[Mapping] | None) -> list[str]:
    """Winner of each p=1 game. Same rule as a Sleeper championship game."""
    winners: list[str] = []
    seen: set[str] = set()
    for match in bracket or []:
        if not isinstance(match, dict):
            continue
        place, winner = match.get("p"), match.get("w")
        if winner is None or winner == "":
            continue
        try:
            if int(place) != 1:
                continue
        except (TypeError, ValueError):
            continue
        rid = str(winner)
        if rid in seen:
            continue
        seen.add(rid)
        winners.append(rid)
    return winners


def _season_sort_key(season) -> tuple:
    text = str(season)
    try:
        return (0, int(text))
    except (TypeError, ValueError):
        return (1, text)


def active_playoff_streaks(seasons: Mapping) -> dict[str, int]:
    """Consecutive playoff seasons ending at the latest decided postseason.

    A season counts as decided when at least one manager reached the playoffs.
    An in-progress year with no bracket yet is ignored, so it does not reset a
    live streak. Missing a later decided season, or missing that year's
    playoffs, resets the count to 0.
    """
    if not seasons:
        return {}
    ordered = sorted(seasons, key=_season_sort_key)
    makers: dict[str, set[str]] = {}
    managers: set[str] = set()
    for season in ordered:
        data = seasons.get(season) or {}
        made: set[str] = set()
        for standing in data.get("standings") or []:
            manager = _clean_manager_name(standing.get("username"))
            if not manager:
                continue
            managers.add(manager)
            if _playoff_finish_rank(standing.get("playoff_finish")) is not None:
                made.add(manager)
        for role in ("champion", "runner_up"):
            manager = _clean_manager_name((data.get(role) or {}).get("username"))
            if manager:
                managers.add(manager)
                made.add(manager)
        makers[str(season)] = made
    decided = [season for season in ordered if makers[str(season)]]
    streaks = {name: 0 for name in managers}
    for name in managers:
        count = 0
        for season in reversed(decided):
            if name in makers[str(season)]:
                count += 1
            else:
                break
        streaks[name] = count
    return streaks


def manager_leaderboard_frame(
    seasons: Mapping,
    game_records: list[Mapping],
) -> pd.DataFrame:
    """Build fair cross-season manager records and era-adjusted scoring metrics.

    Win/loss records come from Sleeper's regular-season standings.  Scoring is
    normalized within each league-week, so ``avg_above_league`` compares a
    manager with the opponents playing under the same settings and scoring era.
    """
    columns = [
        "manager", "titles", "finals", "toilet_titles", "toilet_appearances",
        "seasons", "wins", "losses",
        "win_pct", "best_finish", "avg_score", "avg_above_league",
        "total_points", "games", "active_playoff_streak",
    ]
    stats: dict[str, dict] = {}
    streaks = active_playoff_streaks(seasons)

    def _manager(name) -> str:
        return _clean_manager_name(name)

    for season_data in seasons.values():
        for standing in season_data.get("standings", []) or []:
            manager = _manager(standing.get("username"))
            if not manager:
                continue
            row = stats.setdefault(manager, _manager_stats_seed())
            row["seasons"] += 1
            row["wins"] += int(standing.get("wins") or 0)
            row["losses"] += int(standing.get("losses") or 0)
            try:
                row["fpts"] += float(standing.get("fpts") or 0)
            except (TypeError, ValueError):
                pass
            finish = standing.get("playoff_finish")
            try:
                finish = int(finish) if finish is not None else None
            except (TypeError, ValueError):
                finish = None
            if finish is not None:
                current = row["best_finish"]
                row["best_finish"] = finish if current is None else min(current, finish)

        champion = _manager((season_data.get("champion") or {}).get("username"))
        runner_up = _manager((season_data.get("runner_up") or {}).get("username"))
        if champion:
            stats.setdefault(champion, _manager_stats_seed())["titles"] += 1
        if runner_up:
            stats.setdefault(runner_up, _manager_stats_seed())["runner_ups"] += 1
        toilet_rows = list(season_data.get("toilet_champions") or [])
        if not toilet_rows:
            single = season_data.get("toilet_champion") or {}
            if single:
                toilet_rows = [single]
        for toilet_row in toilet_rows:
            toilet = _manager(toilet_row.get("username"))
            if toilet:
                stats.setdefault(toilet, _manager_stats_seed())["toilet_titles"] += 1
        seen_toilet = set()
        for name in season_data.get("toilet_bracket") or []:
            manager = _manager(name)
            if not manager or manager in seen_toilet:
                continue
            seen_toilet.add(manager)
            stats.setdefault(manager, _manager_stats_seed())["toilet_appearances"] += 1

    weekly_scores: dict[tuple[str, int], list[float]] = {}
    regular_records: list[tuple[str, str, int, float]] = []
    for record in game_records:
        if bool(record.get("is_playoff")):
            continue
        manager = _manager(record.get("username"))
        if not manager:
            continue
        try:
            season = str(record.get("season"))
            week = int(record.get("week") or 0)
            score = float(record.get("score"))
        except (TypeError, ValueError):
            continue
        weekly_scores.setdefault((season, week), []).append(score)
        regular_records.append((manager, season, week, score))

    weekly_means = {
        key: sum(scores) / len(scores)
        for key, scores in weekly_scores.items()
        if scores
    }
    manager_scores: dict[str, list[float]] = {}
    manager_adjusted: dict[str, list[float]] = {}
    for manager, season, week, score in regular_records:
        manager_scores.setdefault(manager, []).append(score)
        manager_adjusted.setdefault(manager, []).append(
            score - weekly_means[(season, week)]
        )

    rows = []
    for manager, values in stats.items():
        wins = int(values["wins"])
        losses = int(values["losses"])
        decisions = wins + losses
        scores = manager_scores.get(manager, [])
        adjusted = manager_adjusted.get(manager, [])
        rows.append({
            "manager": manager,
            "titles": int(values["titles"]),
            "finals": int(values["titles"] + values["runner_ups"]),
            "toilet_titles": int(values["toilet_titles"]),
            "toilet_appearances": int(values["toilet_appearances"]),
            "seasons": int(values["seasons"]),
            "wins": wins,
            "losses": losses,
            "win_pct": round(wins / decisions * 100, 1) if decisions else None,
            "best_finish": values["best_finish"],
            "avg_score": round(sum(scores) / len(scores), 2) if scores else None,
            "avg_above_league": (
                round(sum(adjusted) / len(adjusted), 2) if adjusted else None
            ),
            "total_points": (
                round(sum(scores), 1) if scores else (
                    round(float(values.get("fpts") or 0), 1)
                    if float(values.get("fpts") or 0)
                    else None
                )
            ),
            "games": len(scores),
            "active_playoff_streak": int(streaks.get(manager, 0)),
        })
    return pd.DataFrame(rows, columns=columns)


def tied_leaders(
    frame: pd.DataFrame,
    column: str,
    *,
    name_col: str = "manager",
    min_value: float | None = None,
    ascending: bool = False,
) -> tuple[list[str], float | None]:
    """Return every manager tied at the top of ``column``, sorted by name.

    A secondary stat must not break the tie. Title count 2 and 2 is a shared
    headline even if one manager has a better win rate. ``ascending=True``
    picks the lowest value instead (lowest points per game).
    """
    if frame is None or frame.empty or column not in frame.columns:
        return [], None
    if name_col not in frame.columns:
        return [], None
    values = pd.to_numeric(frame[column], errors="coerce")
    work = frame.loc[values.notna()].copy()
    work["_rank"] = values.loc[work.index]
    if min_value is not None:
        work = work.loc[work["_rank"].ge(min_value)]
    if work.empty:
        return [], None
    top = work["_rank"].min() if ascending else work["_rank"].max()
    names = sorted({str(name) for name in work.loc[work["_rank"].eq(top), name_col]})
    return names, float(top)


def format_tied_names(names: Sequence[str], *, shown: int = 2) -> str:
    """Compact a tied-leader list for a scorecard."""
    labels = [str(name).strip() for name in names if str(name).strip()]
    if not labels:
        return ""
    if len(labels) == 1:
        return labels[0]
    if len(labels) == 2:
        return f"{labels[0]} & {labels[1]}"
    extra = len(labels) - shown
    return f"{', '.join(labels[:shown])} +{extra}"


def format_name_list(names: Sequence[str]) -> str:
    """Full tied-leader list for a caption."""
    labels = [str(name).strip() for name in names if str(name).strip()]
    if len(labels) <= 2:
        return format_tied_names(labels)
    return f"{', '.join(labels[:-1])}, and {labels[-1]}"


def scorecard_headline(names: Sequence[str], *, flip_at: int | None = None) -> str:
    """Name-first until ``flip_at`` managers, then an N-way tie label."""
    labels = [str(name).strip() for name in names if str(name).strip()]
    if flip_at and len(labels) >= flip_at:
        return f"{len(labels)}-way tie"
    return format_tied_names(labels)


def matchup_record_frame(
    game_records: list[Mapping],
    min_valid_score: float = 5.0,
) -> pd.DataFrame:
    """Return one auditable row per played matchup with all-play win context.

    ``all_play_win_pct`` asks how often the actual winner would have beaten every
    other team in that same league-week.  It is a stronger luck signal than
    simply finding the lowest raw score that happened to win.
    """
    columns = [
        "season", "week", "is_playoff", "team_a", "team_b", "is_tie",
        "winner", "loser",
        "winner_score", "loser_score", "margin", "combined",
        "all_play_wins", "all_play_ties", "all_play_opponents",
        "all_play_win_pct",
    ]
    weekly_scores: dict[tuple[str, int], dict[str, float]] = {}
    normalized: list[dict] = []

    for record in game_records:
        manager = str(record.get("username") or "").strip()
        opponent = str(record.get("opp") or "").strip()
        if not manager or not opponent or "?" in {manager, opponent}:
            continue
        try:
            season = str(record.get("season"))
            week = int(record.get("week") or 0)
            score = float(record.get("score"))
            opponent_score = float(record.get("opp_score"))
        except (TypeError, ValueError):
            continue
        if score <= min_valid_score:
            continue
        weekly_scores.setdefault((season, week), {})[manager] = score
        normalized.append({
            "season": season,
            "week": week,
            "is_playoff": bool(record.get("is_playoff")),
            "manager": manager,
            "opponent": opponent,
            "score": score,
            "opponent_score": opponent_score,
        })

    rows = []
    seen: set[tuple[str, int, tuple[str, str]]] = set()
    for record in normalized:
        if record["opponent_score"] <= min_valid_score:
            continue
        pair = tuple(sorted((record["manager"], record["opponent"])))
        key = (record["season"], record["week"], pair)
        if key in seen:
            continue
        seen.add(key)

        score = record["score"]
        opponent_score = record["opponent_score"]
        if score > opponent_score:
            winner, loser = record["manager"], record["opponent"]
            winner_score, loser_score = score, opponent_score
        elif opponent_score > score:
            winner, loser = record["opponent"], record["manager"]
            winner_score, loser_score = opponent_score, score
        else:
            winner = loser = "Tie"
            winner_score = loser_score = score

        all_play_wins = None
        all_play_ties = None
        all_play_opponents = None
        all_play_win_pct = None
        if winner != "Tie":
            field = weekly_scores.get((record["season"], record["week"]), {})
            comparison_scores = [
                value for manager, value in field.items() if manager != winner
            ]
            if comparison_scores:
                all_play_wins = sum(winner_score > value for value in comparison_scores)
                all_play_ties = sum(winner_score == value for value in comparison_scores)
                all_play_opponents = len(comparison_scores)
                all_play_win_pct = round(
                    (all_play_wins + 0.5 * all_play_ties)
                    / all_play_opponents * 100,
                    1,
                )

        rows.append({
            "season": record["season"],
            "week": record["week"],
            "is_playoff": record["is_playoff"],
            "team_a": record["manager"],
            "team_b": record["opponent"],
            "is_tie": winner == "Tie",
            "winner": winner,
            "loser": loser,
            "winner_score": round(winner_score, 2),
            "loser_score": round(loser_score, 2),
            "margin": round(abs(winner_score - loser_score), 2),
            "combined": round(winner_score + loser_score, 2),
            "all_play_wins": all_play_wins,
            "all_play_ties": all_play_ties,
            "all_play_opponents": all_play_opponents,
            "all_play_win_pct": all_play_win_pct,
        })
    return pd.DataFrame(rows, columns=columns).sort_values(
        ["season", "week", "winner"], ignore_index=True
    )


def matchup_index_for_record(
    matchups: pd.DataFrame,
    record: Mapping | None,
) -> int | None:
    """Return the matchup row that owns a Hall of Fame scorecard game."""
    if matchups is None or matchups.empty or not record:
        return None
    try:
        season = str(record.get("season"))
        week = int(record.get("week") or 0)
    except (TypeError, ValueError):
        return None
    names = []
    for key in ("username", "opp", "winner", "loser", "team_a", "team_b"):
        value = str(record.get(key) or "").strip()
        if value and value not in {"?", "—", "Tie"}:
            names.append(value)
    if not names:
        return None
    hit = matchups.index[
        matchups["season"].astype(str).eq(season)
        & pd.to_numeric(matchups["week"], errors="coerce").eq(week)
        & (matchups["team_a"].isin(names) | matchups["team_b"].isin(names))
    ]
    if len(hit) == 0:
        return None
    return int(hit[0])


def scorecard_highlight_labels(
    matchups: pd.DataFrame,
    records: Sequence[tuple[Mapping | None, str]],
) -> dict[int, list[str]]:
    """Map matchup indices to scorecard labels, joining when one game holds several."""
    labels: dict[int, list[str]] = {}
    for record, label in records:
        index = matchup_index_for_record(matchups, record)
        if index is None:
            continue
        bucket = labels.setdefault(index, [])
        if label not in bucket:
            bucket.append(label)
    return labels


def _format_matchup_points(value) -> str:
    number = float(value)
    if abs(number - round(number)) < 1e-9:
        return str(int(round(number)))
    return f"{number:.2f}".rstrip("0").rstrip(".")


def hall_of_fame_delta(record: Mapping | None) -> str | None:
    """Scorecard subtext: winner def. loser · season Wk week (winner - loser)."""
    if not record:
        return None
    try:
        season = record.get("season")
        week = int(record.get("week") or 0)
    except (TypeError, ValueError):
        return None
    if season is None:
        return None

    is_tie = bool(record.get("is_tie"))
    winner = str(record.get("winner") or "").strip()
    if winner == "Tie":
        is_tie = True

    if is_tie:
        left = str(record.get("team_a") or record.get("username") or "").strip()
        right = str(record.get("team_b") or record.get("opp") or "").strip()
        left_score = record.get("winner_score", record.get("score"))
        right_score = record.get("loser_score", record.get("opp_score"))
        if not left or not right or left_score is None or right_score is None:
            return None
        return (
            f"{left} tied {right} · {season} Wk {week} "
            f"({_format_matchup_points(left_score)} - {_format_matchup_points(right_score)})"
        )

    if winner and str(record.get("loser") or "").strip():
        loser = str(record.get("loser")).strip()
        winner_score = record.get("winner_score")
        loser_score = record.get("loser_score")
    else:
        manager = str(record.get("username") or "").strip()
        opponent = str(record.get("opp") or "").strip()
        try:
            score = float(record.get("score"))
            opponent_score = float(record.get("opp_score"))
        except (TypeError, ValueError):
            return None
        if not manager or not opponent:
            return None
        if score > opponent_score:
            winner, loser = manager, opponent
            winner_score, loser_score = score, opponent_score
        elif opponent_score > score:
            winner, loser = opponent, manager
            winner_score, loser_score = opponent_score, score
        else:
            return (
                f"{manager} tied {opponent} · {season} Wk {week} "
                f"({_format_matchup_points(score)} - {_format_matchup_points(opponent_score)})"
            )

    if winner_score is None or loser_score is None:
        return None
    return (
        f"{winner} def. {loser} · {season} Wk {week} "
        f"({_format_matchup_points(winner_score)} - {_format_matchup_points(loser_score)})"
    )


def hall_of_fame_era_caption(
    highest_score: Mapping | None,
    played_records: Sequence[Mapping],
) -> str | None:
    """One-line scoring-era context for the Hall of Fame high score."""
    if not highest_score:
        return None
    season = str(highest_score.get("season") or "").strip()
    try:
        high = float(highest_score.get("score"))
    except (TypeError, ValueError):
        return None
    scores = []
    for record in played_records:
        if str(record.get("season") or "").strip() != season:
            continue
        try:
            scores.append(float(record.get("score")))
        except (TypeError, ValueError):
            continue
    if not scores:
        return None
    average = sum(scores) / len(scores)
    avg_text = (
        str(int(round(average)))
        if abs(average - round(average)) < 1e-9
        else f"{average:.1f}"
    )
    return (
        f"The {_format_matchup_points(high)} high in {season} came in a year "
        f"whose league average was {avg_text}."
    )


def rivalry_summary_frame(matchups: pd.DataFrame) -> pd.DataFrame:
    """Summarize every manager pairing from normalized matchup records."""
    columns = [
        "manager_a", "manager_b", "games", "manager_a_wins", "manager_b_wins",
        "ties", "manager_a_avg_score", "manager_b_avg_score", "avg_point_diff",
        "playoff_meetings", "current_streak_manager", "current_streak",
        "closest_margin", "largest_margin",
    ]
    if matchups.empty:
        return pd.DataFrame(columns=columns)

    work = matchups.copy()
    work["pair"] = work.apply(
        lambda row: tuple(sorted((str(row["team_a"]), str(row["team_b"])))),
        axis=1,
    )
    work["season_sort"] = pd.to_numeric(work["season"], errors="coerce").fillna(0)
    rows = []
    for (manager_a, manager_b), games in work.groupby("pair", sort=True):
        games = games.sort_values(["season_sort", "week"]).copy()
        manager_a_scores = []
        manager_b_scores = []
        results = []
        for _, game in games.iterrows():
            if bool(game["is_tie"]):
                score_a = score_b = float(game["winner_score"])
                result = "Tie"
            elif game["winner"] == manager_a:
                score_a = float(game["winner_score"])
                score_b = float(game["loser_score"])
                result = manager_a
            else:
                score_a = float(game["loser_score"])
                score_b = float(game["winner_score"])
                result = manager_b
            manager_a_scores.append(score_a)
            manager_b_scores.append(score_b)
            results.append(result)

        last_result = results[-1]
        streak = 0
        for result in reversed(results):
            if result != last_result:
                break
            streak += 1
        rows.append({
            "manager_a": manager_a,
            "manager_b": manager_b,
            "games": len(games),
            "manager_a_wins": sum(result == manager_a for result in results),
            "manager_b_wins": sum(result == manager_b for result in results),
            "ties": sum(result == "Tie" for result in results),
            "manager_a_avg_score": round(sum(manager_a_scores) / len(games), 2),
            "manager_b_avg_score": round(sum(manager_b_scores) / len(games), 2),
            "avg_point_diff": round(
                (sum(manager_a_scores) - sum(manager_b_scores)) / len(games), 2
            ),
            "playoff_meetings": int(games["is_playoff"].sum()),
            "current_streak_manager": last_result,
            "current_streak": streak,
            "closest_margin": round(float(games["margin"].min()), 2),
            "largest_margin": round(float(games["margin"].max()), 2),
        })
    return pd.DataFrame(rows, columns=columns).sort_values(
        ["games", "manager_a", "manager_b"],
        ascending=[False, True, True],
        ignore_index=True,
    )


def rivalry_pair_score_frame(
    matchups: pd.DataFrame,
    active_managers: list[str] | tuple[str, ...],
    mode: str = "Classic Rivalries",
) -> pd.DataFrame:
    """Score every possible active-manager pairing for a rivalry-week slate.

    The score is descriptive and league-relative, not a learned probability.
    Sample-size-sensitive inputs use conservative priors and familiarity
    saturates, so one close game cannot outrank an established series merely by
    producing a perfect small-sample win split.
    """
    if mode not in RIVALRY_WEEK_MODES:
        raise ValueError(f"Unknown rivalry-week mode: {mode}")

    columns = [
        "manager_a", "manager_b", "rivalry_score", "games",
        "manager_a_wins", "manager_b_wins", "ties", "avg_margin",
        "close_games", "playoff_meetings", "latest_season",
        "current_streak_manager", "current_streak", "manager_a_win_pct",
        "manager_b_win_pct", "reason",
    ]
    managers = sorted({
        str(manager).strip()
        for manager in active_managers
        if str(manager).strip()
    })
    if len(managers) < 2:
        return pd.DataFrame(columns=columns)

    required = {
        "season", "week", "team_a", "team_b", "is_tie", "winner",
        "margin", "is_playoff",
    }
    if matchups.empty:
        history_work = pd.DataFrame(columns=sorted(required))
    else:
        missing = required.difference(matchups.columns)
        if missing:
            raise ValueError(f"Matchups missing rivalry fields: {sorted(missing)}")
        history_work = matchups.copy()
        history_work["team_a"] = history_work["team_a"].astype(str)
        history_work["team_b"] = history_work["team_b"].astype(str)
    work = history_work[
        history_work["team_a"].isin(managers)
        & history_work["team_b"].isin(managers)
    ].copy()

    if history_work.empty:
        latest_season_value = None
        league_margin = 15.0
    else:
        history_work["season_sort"] = pd.to_numeric(
            history_work["season"], errors="coerce"
        )
        work["season_sort"] = pd.to_numeric(work["season"], errors="coerce")
        latest_numeric = history_work["season_sort"].dropna()
        latest_season_value = (
            float(latest_numeric.max()) if not latest_numeric.empty else None
        )
        margins = pd.to_numeric(history_work["margin"], errors="coerce").dropna()
        league_margin = max(float(margins.median()), 1.0) if not margins.empty else 15.0

    # Overall result quality provides the secondary signal for Fresh Blood.
    manager_points = {manager: 1.0 for manager in managers}
    manager_games = {manager: 2.0 for manager in managers}
    for _, game in history_work.iterrows():
        team_a, team_b = str(game["team_a"]), str(game["team_b"])
        if team_a in manager_games:
            manager_games[team_a] += 1
        if team_b in manager_games:
            manager_games[team_b] += 1
        if bool(game["is_tie"]):
            if team_a in manager_points:
                manager_points[team_a] += 0.5
            if team_b in manager_points:
                manager_points[team_b] += 0.5
        else:
            winner = str(game["winner"])
            if winner in manager_points:
                manager_points[winner] += 1.0
    manager_strength = {
        manager: manager_points[manager] / manager_games[manager]
        for manager in managers
    }

    rows = []
    for manager_a, manager_b in combinations(managers, 2):
        if work.empty:
            games = work.copy()
        else:
            games = work[
                ((work["team_a"] == manager_a) & (work["team_b"] == manager_b))
                | ((work["team_a"] == manager_b) & (work["team_b"] == manager_a))
            ].sort_values(["season_sort", "week"], na_position="first")

        meetings = len(games)
        wins_a = int((games["winner"] == manager_a).sum()) if meetings else 0
        wins_b = int((games["winner"] == manager_b).sum()) if meetings else 0
        ties = int(games["is_tie"].sum()) if meetings else 0
        playoff_meetings = int(games["is_playoff"].sum()) if meetings else 0
        margins = pd.to_numeric(games.get("margin"), errors="coerce").dropna()
        avg_margin = float(margins.mean()) if not margins.empty else None
        close_games = int((margins <= 10).sum()) if not margins.empty else 0

        # Two pseudo-games split evenly around .500 shrink tiny series toward
        # uncertainty instead of declaring a one-game 1-0 result uncompetitive.
        win_share_a = (wins_a + 0.5 * ties + 1.0) / (meetings + 2.0)
        balance = max(0.0, 1.0 - 2.0 * abs(win_share_a - 0.5))
        familiarity = 1.0 - math.exp(-meetings / 4.0)
        novelty = math.exp(-meetings / 2.5)
        closeness = (
            1.0 / (1.0 + avg_margin / league_margin)
            if avg_margin is not None else 0.5
        )
        stakes = 1.0 - math.exp(-playoff_meetings / 1.25)
        strength_similarity = max(
            0.0,
            1.0 - abs(manager_strength[manager_a] - manager_strength[manager_b]),
        )

        result_sequence = [
            str(winner) for winner in games.get("winner", pd.Series(dtype=str)).tolist()
            if str(winner) in {manager_a, manager_b}
        ]
        switches = sum(
            current != previous
            for previous, current in zip(result_sequence, result_sequence[1:])
        )
        back_and_forth = (
            switches / (len(result_sequence) - 1)
            if len(result_sequence) > 1 else 0.5
        )

        current_streak_manager = None
        current_streak = 0
        if result_sequence:
            current_streak_manager = result_sequence[-1]
            for winner in reversed(result_sequence):
                if winner != current_streak_manager:
                    break
                current_streak += 1

        latest_season = None
        recency = 0.0
        if meetings:
            season_values = pd.to_numeric(games["season"], errors="coerce").dropna()
            if not season_values.empty:
                latest_season = int(season_values.max())
                if latest_season_value is not None:
                    years_ago = max(0.0, latest_season_value - latest_season)
                    recency = math.exp(-math.log(2.0) * years_ago / 2.0)
            else:
                latest_season = str(games.iloc[-1]["season"])
                recency = 1.0

        if mode == "Classic Rivalries":
            raw_score = (
                0.30 * familiarity + 0.20 * balance + 0.15 * closeness
                + 0.20 * stakes + 0.10 * recency + 0.05 * back_and_forth
            )
        elif mode == "Maximum Drama":
            raw_score = (
                0.05 * familiarity + 0.20 * balance + 0.30 * closeness
                + 0.20 * stakes + 0.10 * recency + 0.15 * back_and_forth
            )
        else:  # Fresh Blood
            raw_score = (
                0.45 * novelty + 0.35 * strength_similarity
                + 0.10 * balance + 0.10 * closeness
            )

        strength_a = manager_strength[manager_a] * 100
        strength_b = manager_strength[manager_b] * 100
        if meetings == 0:
            reason = (
                "First recorded meeting · historical win rates "
                f"{strength_a:.0f}% and {strength_b:.0f}%"
            )
        elif mode == "Fresh Blood":
            meeting_word = "meeting" if meetings == 1 else "meetings"
            reason = (
                f"Only {meetings} prior {meeting_word} · historical win rates "
                f"{strength_a:.0f}% and {strength_b:.0f}%"
            )
        else:
            details = [f"{meetings} meetings"]
            if playoff_meetings:
                playoff_word = "playoff meeting" if playoff_meetings == 1 else "playoff meetings"
                details.append(f"{playoff_meetings} {playoff_word}")
            if abs(wins_a - wins_b) <= 1:
                details.append(f"series {wins_a}-{wins_b}")
            if avg_margin is not None:
                details.append(f"{avg_margin:.1f}-point average margin")
            if current_streak >= 2:
                details.append(f"{current_streak_manager} has won {current_streak} straight")
            reason = " · ".join(details[:4])

        rows.append({
            "manager_a": manager_a,
            "manager_b": manager_b,
            "rivalry_score": round(max(0.0, min(raw_score, 1.0)) * 100, 1),
            "games": meetings,
            "manager_a_wins": wins_a,
            "manager_b_wins": wins_b,
            "ties": ties,
            "avg_margin": round(avg_margin, 2) if avg_margin is not None else None,
            "close_games": close_games,
            "playoff_meetings": playoff_meetings,
            "latest_season": latest_season,
            "current_streak_manager": current_streak_manager,
            "current_streak": current_streak,
            "manager_a_win_pct": round(strength_a, 1),
            "manager_b_win_pct": round(strength_b, 1),
            "reason": reason,
        })
    return pd.DataFrame(rows, columns=columns).sort_values(
        ["rivalry_score", "manager_a", "manager_b"],
        ascending=[False, True, True],
        ignore_index=True,
    )


def rivalry_week_slate_frame(
    pair_scores: pd.DataFrame,
    locked_pairs: list[tuple[str, str]] | tuple[tuple[str, str], ...] = (),
    avoided_pairs: list[tuple[str, str]] | tuple[tuple[str, str], ...] = (),
) -> pd.DataFrame:
    """Return the maximum-total-score disjoint slate, honoring valid locks.

    This is an exact global optimizer.  It deliberately does not greedily take
    the best remaining edge, which can strand the rest of the league with a
    much weaker slate.  ``avoided_pairs`` powers deterministic alternatives;
    those edges remain a last resort when the constraints leave no other slate.
    """
    if pair_scores.empty:
        result = pair_scores.copy()
        result["locked"] = pd.Series(dtype=bool)
        return result

    required = {"manager_a", "manager_b", "rivalry_score"}
    missing = required.difference(pair_scores.columns)
    if missing:
        raise ValueError(f"Pair scores missing slate fields: {sorted(missing)}")

    scores = pair_scores.copy()
    scores["pair"] = scores.apply(
        lambda row: tuple(sorted((str(row["manager_a"]), str(row["manager_b"])))),
        axis=1,
    )
    row_lookup = {row["pair"]: row for _, row in scores.iterrows()}
    managers = sorted({manager for pair in row_lookup for manager in pair})
    locks = []
    used: set[str] = set()
    for raw_pair in locked_pairs:
        pair = tuple(sorted(map(str, raw_pair)))
        if len(pair) != 2 or pair not in row_lookup:
            continue
        if pair[0] in used or pair[1] in used:
            raise ValueError("Locked rivalry matchups cannot share a manager")
        locks.append(pair)
        used.update(pair)
    avoided = {tuple(sorted(map(str, pair))) for pair in avoided_pairs}

    remaining = tuple(manager for manager in managers if manager not in used)
    open_slot = "\x00OPEN_RIVALRY_SLOT"
    if len(remaining) % 2:
        remaining = tuple(sorted(remaining + (open_slot,)))

    def _edge_value(a: str, b: str) -> float:
        if open_slot in {a, b}:
            return 0.0
        pair = tuple(sorted((a, b)))
        value = float(row_lookup[pair]["rivalry_score"])
        return value - (1000.0 if pair in avoided else 0.0)

    @lru_cache(maxsize=None)
    def _solve(nodes: tuple[str, ...]) -> tuple[float, tuple[tuple[str, str], ...]]:
        if not nodes:
            return 0.0, ()
        first = nodes[0]
        best_value = -math.inf
        best_pairs: tuple[tuple[str, str], ...] | None = None
        for index in range(1, len(nodes)):
            opponent = nodes[index]
            rest = nodes[1:index] + nodes[index + 1:]
            subtotal, subpairs = _solve(rest)
            pair = tuple(sorted((first, opponent)))
            candidate_pairs = tuple(sorted((pair,) + subpairs))
            candidate_value = _edge_value(first, opponent) + subtotal
            if (
                candidate_value > best_value
                or (
                    math.isclose(candidate_value, best_value)
                    and (best_pairs is None or candidate_pairs < best_pairs)
                )
            ):
                best_value = candidate_value
                best_pairs = candidate_pairs
        return best_value, best_pairs or ()

    _, optimized_pairs = _solve(remaining)
    selected_rows = []
    for pair in locks:
        row = row_lookup[pair].drop(labels=["pair"]).to_dict()
        row["locked"] = True
        selected_rows.append(row)
    for pair in optimized_pairs:
        if open_slot in pair:
            manager = pair[1] if pair[0] == open_slot else pair[0]
            row = {column: None for column in pair_scores.columns}
            row.update({
                "manager_a": manager,
                "manager_b": None,
                "reason": "Open slot: the league has an odd number of active managers",
                "locked": False,
            })
        else:
            row = row_lookup[pair].drop(labels=["pair"]).to_dict()
            row["locked"] = False
        selected_rows.append(row)

    result = pd.DataFrame(selected_rows)
    if result.empty:
        return result
    result["_open"] = result["manager_b"].isna()
    result["_score"] = pd.to_numeric(result["rivalry_score"], errors="coerce").fillna(-1)
    return result.sort_values(
        ["_open", "locked", "_score", "manager_a"],
        ascending=[True, False, False, True],
        ignore_index=True,
    ).drop(columns=["_open", "_score"])


def weekly_score_context_frame(
    game_records: list[Mapping],
    include_playoffs: bool = False,
    min_valid_score: float = 5.0,
) -> pd.DataFrame:
    """Return one manager-week per row with same-week league scoring context."""
    columns = [
        "season", "week", "manager", "opponent", "score", "opponent_score",
        "league_average", "league_median", "adjusted_score", "result",
    ]
    records = []
    for record in game_records:
        if bool(record.get("is_playoff")) and not include_playoffs:
            continue
        manager = str(record.get("username") or "").strip()
        opponent = str(record.get("opp") or "").strip()
        if not manager or manager in {"?", "—"}:
            continue
        try:
            season = str(record.get("season"))
            week = int(record.get("week") or 0)
            score = float(record.get("score"))
            opponent_score = float(record.get("opp_score"))
        except (TypeError, ValueError):
            continue
        if score <= min_valid_score or opponent_score <= min_valid_score:
            continue
        records.append({
            "season": season,
            "week": week,
            "manager": manager,
            "opponent": opponent,
            "score": score,
            "opponent_score": opponent_score,
        })
    if not records:
        return pd.DataFrame(columns=columns)

    work = pd.DataFrame(records).drop_duplicates(
        ["season", "week", "manager"], keep="first"
    )
    weekly = work.groupby(["season", "week"])["score"]
    work["league_average"] = weekly.transform("mean")
    work["league_median"] = weekly.transform("median")
    work["adjusted_score"] = work["score"] - work["league_average"]
    work["result"] = work.apply(
        lambda row: (
            "W" if row["score"] > row["opponent_score"]
            else "L" if row["score"] < row["opponent_score"]
            else "T"
        ),
        axis=1,
    )
    return work[columns].sort_values(
        ["season", "week", "manager"], ignore_index=True
    )


def manager_performance_frame(
    game_records: list[Mapping],
    include_playoffs: bool = False,
    min_valid_score: float = 5.0,
) -> pd.DataFrame:
    """Build regular-season manager performance with weekly league context."""
    columns = [
        "manager", "games", "wins", "losses", "ties", "win_pct",
        "avg_score", "avg_above_league", "std_dev", "lucky_wins",
        "unlucky_losses",
    ]
    records = []
    for record in game_records:
        if bool(record.get("is_playoff")) and not include_playoffs:
            continue
        manager = str(record.get("username") or "").strip()
        if not manager or manager in {"?", "—"}:
            continue
        try:
            season = str(record.get("season"))
            week = int(record.get("week") or 0)
            score = float(record.get("score"))
            opponent_score = float(record.get("opp_score"))
        except (TypeError, ValueError):
            continue
        if score <= min_valid_score or opponent_score <= min_valid_score:
            continue
        records.append({
            "season": season,
            "week": week,
            "manager": manager,
            "score": score,
            "opponent_score": opponent_score,
        })
    if not records:
        return pd.DataFrame(columns=columns)

    work = pd.DataFrame(records).drop_duplicates(
        ["season", "week", "manager"], keep="first"
    )
    weekly_means = work.groupby(["season", "week"])["score"].mean().to_dict()
    work["league_average"] = work.apply(
        lambda row: weekly_means[(row["season"], row["week"])], axis=1
    )
    work["adjusted_score"] = work["score"] - work["league_average"]
    work["result"] = work.apply(
        lambda row: (
            "W" if row["score"] > row["opponent_score"]
            else "L" if row["score"] < row["opponent_score"]
            else "T"
        ),
        axis=1,
    )

    rows = []
    for manager, games in work.groupby("manager", sort=True):
        wins = int(games["result"].eq("W").sum())
        losses = int(games["result"].eq("L").sum())
        ties = int(games["result"].eq("T").sum())
        game_count = len(games)
        rows.append({
            "manager": manager,
            "games": game_count,
            "wins": wins,
            "losses": losses,
            "ties": ties,
            "win_pct": round((wins + 0.5 * ties) / game_count * 100, 1),
            "avg_score": round(float(games["score"].mean()), 2),
            "avg_above_league": round(float(games["adjusted_score"].mean()), 2),
            "std_dev": round(float(games["score"].std()), 2) if game_count > 1 else 0.0,
            "lucky_wins": int(
                (games["result"].eq("W") & games["adjusted_score"].lt(0)).sum()
            ),
            "unlucky_losses": int(
                (games["result"].eq("L") & games["adjusted_score"].gt(0)).sum()
            ),
        })
    return pd.DataFrame(rows, columns=columns).sort_values(
        ["avg_above_league", "win_pct"],
        ascending=[False, False],
        ignore_index=True,
    )


def consistency_luck_frame(
    game_records: list[Mapping],
    min_valid_score: float = 5.0,
) -> pd.DataFrame:
    """Measure adjusted scoring volatility and schedule luck via all-play wins."""
    columns = [
        "manager", "games", "avg_score", "avg_above_league", "volatility",
        "actual_wins", "expected_wins", "luck_delta", "actual_win_pct",
        "expected_win_pct", "below_avg_wins", "above_avg_losses",
    ]
    records = []
    for record in game_records:
        if bool(record.get("is_playoff")):
            continue
        manager = str(record.get("username") or "").strip()
        if not manager or manager in {"?", "—"}:
            continue
        try:
            season = str(record.get("season"))
            week = int(record.get("week") or 0)
            score = float(record.get("score"))
            opponent_score = float(record.get("opp_score"))
        except (TypeError, ValueError):
            continue
        if score <= min_valid_score or opponent_score <= min_valid_score:
            continue
        records.append({
            "season": season,
            "week": week,
            "manager": manager,
            "score": score,
            "opponent_score": opponent_score,
        })
    if not records:
        return pd.DataFrame(columns=columns)

    work = pd.DataFrame(records).drop_duplicates(
        ["season", "week", "manager"], keep="first"
    )
    weekly_fields = {
        key: group.set_index("manager")["score"].to_dict()
        for key, group in work.groupby(["season", "week"])
    }

    expected_wins = []
    league_averages = []
    results = []
    for _, row in work.iterrows():
        field = weekly_fields[(row["season"], row["week"])]
        league_average = sum(field.values()) / len(field)
        comparisons = [
            score for manager, score in field.items() if manager != row["manager"]
        ]
        if comparisons:
            wins = sum(row["score"] > score for score in comparisons)
            ties = sum(row["score"] == score for score in comparisons)
            expected_win = (wins + 0.5 * ties) / len(comparisons)
        else:
            expected_win = 0.5
        result = (
            "W" if row["score"] > row["opponent_score"]
            else "L" if row["score"] < row["opponent_score"]
            else "T"
        )
        expected_wins.append(expected_win)
        league_averages.append(league_average)
        results.append(result)
    work["expected_win"] = expected_wins
    work["league_average"] = league_averages
    work["adjusted_score"] = work["score"] - work["league_average"]
    work["result"] = results
    work["actual_win_value"] = work["result"].map({"W": 1.0, "L": 0.0, "T": 0.5})

    rows = []
    for manager, games in work.groupby("manager", sort=True):
        game_count = len(games)
        actual_wins = float(games["actual_win_value"].sum())
        all_play_expected = float(games["expected_win"].sum())
        rows.append({
            "manager": manager,
            "games": game_count,
            "avg_score": round(float(games["score"].mean()), 2),
            "avg_above_league": round(float(games["adjusted_score"].mean()), 2),
            "volatility": (
                round(float(games["adjusted_score"].std()), 2)
                if game_count > 1 else 0.0
            ),
            "actual_wins": round(actual_wins, 2),
            "expected_wins": round(all_play_expected, 2),
            "luck_delta": round(actual_wins - all_play_expected, 2),
            "actual_win_pct": round(actual_wins / game_count * 100, 1),
            "expected_win_pct": round(all_play_expected / game_count * 100, 1),
            "below_avg_wins": int(
                (games["result"].eq("W") & games["adjusted_score"].lt(0)).sum()
            ),
            "above_avg_losses": int(
                (games["result"].eq("L") & games["adjusted_score"].gt(0)).sum()
            ),
        })
    return pd.DataFrame(rows, columns=columns).sort_values(
        ["avg_above_league", "luck_delta"],
        ascending=[False, False],
        ignore_index=True,
    )


def draft_pick_frame(seasons: Mapping) -> pd.DataFrame:
    """Normalize Sleeper draft-pick payloads across league seasons."""
    rows: list[dict] = []
    for season, season_data in seasons.items():
        draft_id = str(season_data.get("draft_id") or "")
        for pick in season_data.get("draft_picks", []) or []:
            if not isinstance(pick, Mapping):
                continue
            metadata = pick.get("metadata") or {}
            player_id = str(pick.get("player_id") or metadata.get("player_id") or "")
            try:
                pick_no = int(pick.get("pick_no") or 0)
                round_no = int(pick.get("round") or 0)
            except (TypeError, ValueError):
                continue
            if not player_id or pick_no <= 0 or round_no <= 0:
                continue
            rows.append({
                "season": str(season),
                "draft_id": draft_id,
                "pick_no": pick_no,
                "round": round_no,
                "pick_in_round": int(pick.get("pick_in_round") or 0),
                "draft_slot": int(pick.get("draft_slot") or 0),
                "roster_id": str(pick.get("roster_id") or ""),
                "user_id": str(pick.get("picked_by") or ""),
                "player_id": player_id,
                "player_name": _player_name(metadata, player_id),
                "position": str(metadata.get("position") or "").upper(),
            })
    columns = [
        "season", "draft_id", "pick_no", "round", "pick_in_round",
        "draft_slot", "roster_id", "user_id", "player_id", "player_name",
        "position",
    ]
    return pd.DataFrame(rows, columns=columns)


def manager_season_frame(picks: pd.DataFrame) -> pd.DataFrame:
    """One row per manager-season with interpretable draft construction fields."""
    columns = [
        "season", "user_id", "draft_slot", "first_qb_round", "first_te_round",
        "qb_count", "te_count", "rb_first_four", "wr_first_four",
    ]
    if picks.empty:
        return pd.DataFrame(columns=columns)

    rows: list[dict] = []
    for (season, user_id), group in picks.groupby(["season", "user_id"], sort=True):
        group = group.sort_values("pick_no")
        qbs = group[group["position"].eq("QB")]
        tes = group[group["position"].eq("TE")]
        first_four = group[group["round"].le(4)]
        nonzero_slots = group.loc[group["draft_slot"].gt(0), "draft_slot"]
        rows.append({
            "season": str(season),
            "user_id": str(user_id),
            "draft_slot": int(nonzero_slots.iloc[0]) if not nonzero_slots.empty else None,
            "first_qb_round": int(qbs.iloc[0]["round"]) if not qbs.empty else None,
            "first_te_round": int(tes.iloc[0]["round"]) if not tes.empty else None,
            "qb_count": int(len(qbs)),
            "te_count": int(len(tes)),
            "rb_first_four": int(first_four["position"].eq("RB").sum()),
            "wr_first_four": int(first_four["position"].eq("WR").sum()),
        })
    return pd.DataFrame(rows, columns=columns)


def position_round_matrix(picks: pd.DataFrame) -> pd.DataFrame:
    """Average number of players selected per draft at each position/round."""
    core = picks[picks["position"].isin(CORE_POSITIONS)].copy()
    if core.empty:
        return pd.DataFrame(index=CORE_POSITIONS)
    drafts = max(int(core["draft_id"].replace("", pd.NA).nunique()), 1)
    counts = core.groupby(["position", "round"]).size().unstack(fill_value=0)
    counts = counts.reindex(CORE_POSITIONS, fill_value=0)
    return (counts / drafts).round(2)


def cumulative_position_frame(picks: pd.DataFrame) -> pd.DataFrame:
    """Cumulative QB/TE counts by season and round."""
    core = picks[picks["position"].isin(("QB", "TE"))].copy()
    if core.empty:
        return pd.DataFrame(columns=["season", "round", "position", "cumulative"])
    max_round = int(picks["round"].max())
    rows: list[dict] = []
    for season in sorted(core["season"].unique()):
        season_rows = core[core["season"].eq(season)]
        for position in ("QB", "TE"):
            position_rows = season_rows[season_rows["position"].eq(position)]
            for round_no in range(1, max_round + 1):
                rows.append({
                    "season": str(season),
                    "round": round_no,
                    "position": position,
                    "cumulative": int(position_rows["round"].le(round_no).sum()),
                })
    return pd.DataFrame(rows)


def roster_construction_frame(manager_seasons: pd.DataFrame) -> pd.DataFrame:
    """Season-level rate of teams spending multiple picks at QB or TE."""
    columns = ["season", "teams", "avg_qb", "avg_te", "qb2_plus_rate", "te2_plus_rate"]
    if manager_seasons.empty:
        return pd.DataFrame(columns=columns)
    rows = []
    for season, group in manager_seasons.groupby("season", sort=True):
        teams = len(group)
        rows.append({
            "season": str(season),
            "teams": teams,
            "avg_qb": round(float(group["qb_count"].mean()), 2),
            "avg_te": round(float(group["te_count"].mean()), 2),
            "qb2_plus_rate": round(float(group["qb_count"].ge(2).mean()), 4),
            "te2_plus_rate": round(float(group["te_count"].ge(2).mean()), 4),
        })
    return pd.DataFrame(rows, columns=columns)


def first_pick_timing_frame(picks: pd.DataFrame) -> pd.DataFrame:
    """First QB and TE selected in each draft season."""
    rows = []
    for season, season_rows in picks.groupby("season", sort=True):
        for position in ("QB", "TE"):
            selected = season_rows[season_rows["position"].eq(position)].sort_values("pick_no")
            if selected.empty:
                continue
            first = selected.iloc[0]
            rows.append({
                "season": str(season),
                "position": position,
                "pick_no": int(first["pick_no"]),
                "round": int(first["round"]),
                "player_name": first["player_name"],
            })
    return pd.DataFrame(rows)


def max_position_run(picks: pd.DataFrame, window_size: int = 12) -> dict | None:
    """Find the densest same-position rolling pick window across all drafts."""
    best: dict | None = None
    for season, season_rows in picks.groupby("season", sort=True):
        season_rows = season_rows.sort_values("pick_no")
        if season_rows.empty:
            continue
        final_pick = int(season_rows["pick_no"].max())
        for position in CORE_POSITIONS:
            for start in range(1, max(final_pick - window_size + 2, 2)):
                stop = start + window_size - 1
                count = int(season_rows[
                    season_rows["pick_no"].between(start, stop)
                    & season_rows["position"].eq(position)
                ].shape[0])
                candidate = {
                    "season": str(season), "position": position,
                    "count": count, "start_pick": start, "end_pick": stop,
                }
                if best is None or count > best["count"]:
                    best = candidate
    return best


def draft_insights(
    picks: pd.DataFrame,
    manager_seasons: pd.DataFrame,
    selected_user_id: str | None = None,
) -> list[dict]:
    """Generate deterministic, evidence-bearing interpretations of a draft room."""
    if picks.empty:
        return []
    draft_count = int(picks["draft_id"].replace("", pd.NA).nunique())
    confidence = "Emerging" if draft_count >= 3 else "Limited"
    insights: list[dict] = []

    first_three = picks[picks["round"].le(3)]
    if not first_three.empty:
        skill_share = float(first_three["position"].isin(("RB", "WR")).mean())
        evidence = f"{len(first_three)} picks across {draft_count} drafts"
        insights.append({
            "title": "The room builds around RB/WR first",
            "finding": f"{skill_share:.0%} of picks in Rounds 1-3 were RBs or WRs.",
            "meaning": (
                "Early QB or TE usually requires a deliberate exception; the room normally "
                "leaves those positions alone while the first skill-position tiers disappear."
            ),
            "evidence": evidence,
            "confidence": confidence,
            "bullet": (
                f"{skill_share:.0%} of Rounds 1-3 were RB/WR, so early QB/TE is the exception "
                f"({evidence})."
            ),
        })

    construction = roster_construction_frame(manager_seasons)
    if not construction.empty:
        qb2 = float(construction["qb2_plus_rate"].mean())
        te2 = float(construction["te2_plus_rate"].mean())
        extras = float((construction["avg_qb"] - 1).mean() + (construction["avg_te"] - 1).mean())
        evidence = f"{int(construction['teams'].sum())} team-drafts"
        insights.append({
            "title": "Opponents spend bench capital on onesie positions",
            "finding": (
                f"{qb2:.0%} of teams drafted 2+ QBs and {te2:.0%} drafted 2+ TEs; "
                f"that is {extras:.2f} extra QB/TE picks per team."
            ),
            "meaning": (
                "A one-QB, one-TE build can turn roughly one additional bench spot into an "
                "RB/WR upside bet, provided you are willing to stream bye-week replacements."
            ),
            "evidence": evidence,
            "confidence": confidence,
            "bullet": (
                f"{qb2:.0%} of teams took 2+ QBs and {te2:.0%} took 2+ TEs "
                f"({extras:.2f} extra onesie picks per team, {evidence})."
            ),
        })

    run = max_position_run(picks)
    if run and run["count"] >= 6:
        evidence = f"Best 12-pick window across {draft_count} drafts"
        insights.append({
            "title": f"Do not buy the back of a {run['position']} run",
            "finding": (
                f"The sharpest run was {run['count']} {run['position']} picks from "
                f"#{run['start_pick']}-#{run['end_pick']} in {run['season']}."
            ),
            "meaning": (
                "Once the run is underway, the untouched position usually offers the cleaner "
                "tier. Use nearby managers to anticipate a run, but avoid chasing it after the value is gone."
            ),
            "evidence": evidence,
            "confidence": confidence,
            "bullet": (
                f"The sharpest run was {run['count']} {run['position']}s from "
                f"#{run['start_pick']}-#{run['end_pick']} in {run['season']}; do not buy the back of it "
                f"({evidence})."
            ),
        })

    if selected_user_id:
        selected = manager_seasons[manager_seasons["user_id"].eq(str(selected_user_id))]
        if not selected.empty:
            manager_rb_share = float(selected["rb_first_four"].sum() / (4 * len(selected)))
            league_rb_share = float(manager_seasons["rb_first_four"].sum() / (4 * len(manager_seasons)))
            delta = manager_rb_share - league_rb_share
            direction = "more" if delta >= 0 else "fewer"
            evidence = f"{len(selected)} of your drafts"
            manager_confidence = "Emerging" if len(selected) >= 3 else "Limited"
            insights.append({
                "title": "Your early-round fingerprint",
                "finding": (
                    f"You used {manager_rb_share:.0%} of your first four picks on RB, "
                    f"versus {league_rb_share:.0%} for the room."
                ),
                "meaning": (
                    f"You take {abs(delta):.0%} {direction} RBs than the league in that range. "
                    "That is a useful bias check when a WR tier is falling; it is not proof that the build is wrong."
                ),
                "evidence": evidence,
                "confidence": manager_confidence,
                "bullet": (
                    f"You used {manager_rb_share:.0%} of your first four picks on RB versus "
                    f"{league_rb_share:.0%} for the room ({evidence})."
                ),
            })
    return insights[:5]


def player_week_frame(seasons: Mapping, player_directory: Mapping | None = None) -> pd.DataFrame:
    """Expand compact weekly roster snapshots to one row per player-week."""
    player_directory = player_directory or {}
    draft_names: dict[tuple[str, str], tuple[str, str]] = {}
    for row in draft_pick_frame(seasons).itertuples(index=False):
        draft_names[(str(row.season), str(row.player_id))] = (row.player_name, row.position)

    rows: list[dict] = []
    for season, season_data in seasons.items():
        season = str(season)
        roster_users = {
            str(row.get("roster_id") or ""): str(row.get("owner_id") or "")
            for row in season_data.get("standings", [])
        }
        for entry in season_data.get("roster_entries", []) or []:
            roster_id = str(entry.get("roster_id") or "")
            user_id = roster_users.get(roster_id, "")
            if not user_id:
                continue
            starters = {str(value) for value in entry.get("starters", []) or []}
            points = entry.get("players_points") or {}
            active_matchup = entry.get("matchup_id") is not None
            for raw_player_id in entry.get("players", []) or []:
                player_id = str(raw_player_id)
                directory_row = player_directory.get(player_id) or {}
                fallback_name, fallback_position = draft_names.get(
                    (season, player_id), (player_id, "")
                )
                name = _player_name(directory_row, fallback_name)
                if name == fallback_name and fallback_name != player_id:
                    name = fallback_name
                position = str(directory_row.get("position") or fallback_position or "").upper()
                try:
                    player_points = float(points.get(player_id) or 0)
                except (TypeError, ValueError):
                    player_points = 0.0
                rows.append({
                    "season": season,
                    "week": int(entry.get("week") or 0),
                    "user_id": user_id,
                    "roster_id": roster_id,
                    "player_id": player_id,
                    "player_name": name,
                    "position": position,
                    "is_starter": player_id in starters,
                    "active_matchup": active_matchup,
                    "points": player_points,
                })
    columns = [
        "season", "week", "user_id", "roster_id", "player_id", "player_name",
        "position", "is_starter", "active_matchup", "points",
    ]
    frame = pd.DataFrame(rows, columns=columns)
    if not frame.empty:
        frame = frame.drop_duplicates(["season", "week", "user_id", "player_id"])
    return frame


def player_season_summary(
    player_weeks: pd.DataFrame,
    min_roster_weeks: int = DEFAULT_MIN_ROSTER_WEEKS,
) -> pd.DataFrame:
    """Summarize production, filtering each player-season by roster tenure."""
    columns = [
        "season", "user_id", "player_id", "player_name", "position",
        "roster_weeks", "starts", "lineup_points", "roster_points", "bench_points",
        "points_per_start",
    ]
    if player_weeks.empty:
        return pd.DataFrame(columns=columns)
    work = player_weeks.copy()
    work["start_count"] = work["is_starter"] & work["active_matchup"]
    work["lineup_value"] = work["points"].where(work["start_count"], 0.0)
    work["roster_value"] = work["points"].where(work["active_matchup"], 0.0)
    grouped = work.groupby(
        ["season", "user_id", "player_id", "player_name", "position"],
        as_index=False,
    ).agg(
        roster_weeks=("week", "nunique"),
        starts=("start_count", "sum"),
        lineup_points=("lineup_value", "sum"),
        roster_points=("roster_value", "sum"),
    )
    grouped = grouped[grouped["roster_weeks"].ge(int(min_roster_weeks))].copy()
    grouped["bench_points"] = grouped["roster_points"] - grouped["lineup_points"]
    grouped["points_per_start"] = grouped["lineup_points"].div(
        grouped["starts"].where(grouped["starts"].gt(0))
    ).fillna(0)
    for column in ("lineup_points", "roster_points", "bench_points", "points_per_start"):
        grouped[column] = grouped[column].round(2)
    return grouped[columns]


def player_career_summary(player_seasons: pd.DataFrame) -> pd.DataFrame:
    """Combine qualifying player-seasons while keeping season membership visible."""
    columns = [
        "user_id", "player_id", "player_name", "position", "season_count", "seasons",
        "roster_weeks", "starts", "lineup_points", "roster_points", "bench_points",
    ]
    if player_seasons.empty:
        return pd.DataFrame(columns=columns)
    rows = []
    for (user_id, player_id, player_name, position), group in player_seasons.groupby(
        ["user_id", "player_id", "player_name", "position"], sort=False
    ):
        seasons = sorted(group["season"].astype(str).unique())
        rows.append({
            "user_id": user_id,
            "player_id": player_id,
            "player_name": player_name,
            "position": position,
            "season_count": len(seasons),
            "seasons": ", ".join(seasons),
            "roster_weeks": int(group["roster_weeks"].sum()),
            "starts": int(group["starts"].sum()),
            "lineup_points": round(float(group["lineup_points"].sum()), 2),
            "roster_points": round(float(group["roster_points"].sum()), 2),
            "bench_points": round(float(group["bench_points"].sum()), 2),
        })
    return pd.DataFrame(rows, columns=columns).sort_values("lineup_points", ascending=False)


def eligible_player_weeks(
    player_weeks: pd.DataFrame,
    player_seasons: pd.DataFrame,
) -> pd.DataFrame:
    """Restrict weekly rows to the player-seasons that passed the tenure filter."""
    if player_weeks.empty or player_seasons.empty:
        return player_weeks.iloc[0:0].copy()
    keys = player_seasons[["season", "user_id", "player_id"]].drop_duplicates()
    return player_weeks.merge(keys, on=["season", "user_id", "player_id"], how="inner")


def value_frame(player_seasons: pd.DataFrame, picks: pd.DataFrame) -> pd.DataFrame:
    """Attach draft cost; non-matches remain honestly labeled in-season additions."""
    columns = list(player_seasons.columns) + ["source", "round", "pick_no"]
    if player_seasons.empty:
        return pd.DataFrame(columns=columns)
    draft_cost = picks[[
        "season", "user_id", "player_id", "round", "pick_no",
    ]].drop_duplicates(["season", "user_id", "player_id"])
    values = player_seasons.merge(
        draft_cost, on=["season", "user_id", "player_id"], how="left"
    )
    values["source"] = values["round"].notna().map({True: "Drafted", False: "In-season addition"})
    return values


def select_insight_seasons(
    all_seasons: list[str],
    completed_draft_seasons: list[str],
    window: str,
) -> list[str]:
    """Choose which seasons feed Draft & Roster Insights."""
    if window == INSIGHT_WINDOWS[0]:
        return list(completed_draft_seasons[-1:])
    if window == INSIGHT_WINDOWS[1]:
        return list(completed_draft_seasons[-3:])
    return list(all_seasons)


def league_uses_faab(seasons: Mapping) -> bool:
    """Sleeper stores FAAB leagues as waiver_type 2."""
    for data in seasons.values():
        settings = data.get("league_settings") or {}
        try:
            waiver_type = int(settings.get("waiver_type") or 0)
        except (TypeError, ValueError):
            waiver_type = 0
        if waiver_type == SLEEPER_FAAB_WAIVER_TYPE:
            return True
    return False


def split_faab_waiver_frames(
    values: pd.DataFrame,
    paid_min: int = FAAB_PAID_MIN,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split 4-week-qualified waiver claims for the cheap-claim bar.

    The $5+ scatter does not use this helper. It is built from every completed
    bid, including players later dropped. Free-agent adds stay off both charts.
    """
    columns = list(values.columns) if not values.empty else [
        "season", "player_id", "user_id", "source", "faab", "lineup_points",
    ]
    empty = pd.DataFrame(columns=columns)
    if values.empty:
        return empty.copy(), empty.copy()
    work = values[values["source"].eq("Waiver")].copy()
    if work.empty:
        return empty.copy(), empty.copy()
    work["faab"] = pd.to_numeric(work["faab"], errors="coerce").fillna(0)
    paid_min = max(int(paid_min), 1)
    return work[work["faab"] < paid_min].copy(), work[work["faab"] >= paid_min].copy()


PAID_WAIVER_COLUMNS = [
    "season", "user_id", "player_id", "player_name", "position",
    "faab", "acq_week", "lineup_points", "starts", "source",
]


def _manager_lineup_totals(player_weeks: pd.DataFrame, user_id: str) -> pd.DataFrame:
    columns = [
        "season", "player_id", "player_name", "position", "lineup_points", "starts",
    ]
    if player_weeks is None or player_weeks.empty or not user_id:
        return pd.DataFrame(columns=columns)
    work = player_weeks[player_weeks["user_id"].astype(str).eq(str(user_id))].copy()
    if work.empty:
        return pd.DataFrame(columns=columns)
    starter = work["is_starter"] & work["active_matchup"]
    work = work.assign(
        lineup_value=work["points"].where(starter, 0.0),
        start_count=starter.astype(int),
    )
    grouped = work.groupby(["season", "player_id"], as_index=False).agg(
        player_name=("player_name", "last"),
        position=("position", "last"),
        lineup_points=("lineup_value", "sum"),
        starts=("start_count", "sum"),
    )
    grouped["lineup_points"] = grouped["lineup_points"].round(2)
    return grouped[columns]


def paid_waiver_claim_frame(
    transactions_by_season: Mapping[str, list],
    roster_owner: Mapping[tuple[str, str], str],
    player_weeks: pd.DataFrame,
    selected_user_id: str,
    paid_min: int = FAAB_PAID_MIN,
) -> pd.DataFrame:
    """Every completed $paid_min+ waiver for this manager. No roster-week floor."""
    empty = pd.DataFrame(columns=PAID_WAIVER_COLUMNS)
    if not transactions_by_season or not selected_user_id:
        return empty
    frames = [
        transaction_adds_frame(transactions, season)
        for season, transactions in transactions_by_season.items()
    ]
    if not frames:
        return empty
    adds = pd.concat(frames, ignore_index=True)
    if adds.empty:
        return empty
    adds["user_id"] = [
        str(roster_owner.get((str(season), str(roster_id)), "") or "")
        for season, roster_id in zip(adds["season"], adds["roster_id"])
    ]
    paid_min = max(int(paid_min), 1)
    adds["faab"] = pd.to_numeric(adds["faab"], errors="coerce")
    work = adds[
        adds["user_id"].eq(str(selected_user_id))
        & adds["source"].eq("Waiver")
        & adds["faab"].ge(paid_min)
    ].copy()
    if work.empty:
        return empty
    work = work.sort_values("faab", ascending=False).drop_duplicates(
        ["season", "user_id", "player_id"], keep="first",
    )
    totals = _manager_lineup_totals(player_weeks, selected_user_id)
    merged = work.merge(totals, on=["season", "player_id"], how="left")
    names = _player_name_lookup(player_weeks)
    merged["player_name"] = merged["player_name"].fillna("").replace("", pd.NA)
    merged["player_name"] = [
        str(name) if pd.notna(name) and str(name).strip()
        else names.get(str(player_id), str(player_id))
        for name, player_id in zip(merged["player_name"], merged["player_id"])
    ]
    merged["position"] = merged["position"].fillna("").astype(str)
    merged["lineup_points"] = pd.to_numeric(
        merged["lineup_points"], errors="coerce",
    ).fillna(0.0)
    merged["starts"] = pd.to_numeric(merged["starts"], errors="coerce").fillna(0).astype(int)
    merged["acq_week"] = merged["week"]
    merged["source"] = "Waiver"
    return merged[PAID_WAIVER_COLUMNS].reset_index(drop=True)


def split_paid_production_frames(
    paid: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Keep zero-point $5+ bids off the scatter so they are not stacked on y=0."""
    columns = list(paid.columns) if paid is not None and not paid.empty else list(PAID_WAIVER_COLUMNS)
    empty = pd.DataFrame(columns=columns)
    if paid is None or paid.empty:
        return empty.copy(), empty.copy()
    points = pd.to_numeric(paid["lineup_points"], errors="coerce").fillna(0.0)
    return paid.loc[points.gt(0)].copy(), paid.loc[points.le(0)].copy()


def compact_name_list(names: str, max_names: int = 2) -> str:
    """Keep the first names of a trade package readable on a bar."""
    raw = str(names or "").strip()
    if not raw or raw == "(none)":
        return "(none)"
    parts = [part.strip() for part in raw.split(",") if part.strip()]
    if len(parts) <= max_names:
        return ", ".join(parts)
    extra = len(parts) - max_names
    return f"{', '.join(parts[:max_names])} +{extra}"


def select_trade_chart_rows(
    trades: pd.DataFrame,
    limit: int = TRADE_CHART_LIMIT,
) -> pd.DataFrame:
    """Show every trade when they fit; otherwise the most lopsided |net|."""
    if trades is None or trades.empty:
        return trades if trades is not None else pd.DataFrame()
    limit = max(int(limit), 1)
    if len(trades) <= limit:
        return trades.sort_values(["season", "week"]).reset_index(drop=True)
    work = trades.copy()
    work["_abs_net"] = work["net"].abs()
    shown = work.sort_values(
        ["_abs_net", "season", "week"], ascending=[False, True, True],
    ).head(limit)
    return (
        shown.drop(columns=["_abs_net"])
        .sort_values("net")
        .reset_index(drop=True)
    )


def trade_opponent_labels(trades: pd.DataFrame) -> list[str]:
    """Other manager plus week, so repeat counterparties stay unique."""
    if trades is None or trades.empty:
        return []
    labels: list[str] = []
    for opponent, season, week in zip(
        trades["opponent"], trades["season"], trades["week"],
    ):
        name = str(opponent or "").strip() or "Unknown"
        labels.append(f"{name} · {season} W{week}")
    return labels


def production_chart_frame(values: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """Place undrafted-by-this-manager players in a Pickup lane past the last round."""
    if values.empty:
        empty = values.copy()
        empty["chart_x"] = pd.Series(dtype=float)
        empty["lane"] = pd.Series(dtype=str)
        return empty, 14
    drafted_rounds = values.loc[values["source"].eq("Drafted"), "round"].dropna()
    max_round = int(drafted_rounds.max()) if not drafted_rounds.empty else 14
    out = values.copy()
    pickup_x = float(max_round + PICKUP_LANE_GAP)
    out["lane"] = out["source"].where(out["source"].eq("Drafted"), "Pickup")
    out["chart_x"] = out["round"].where(out["source"].eq("Drafted"), pickup_x).astype(float)
    return out, max_round


def transaction_adds_frame(transactions: list, season: str) -> pd.DataFrame:
    """One row per completed add: waiver, free agent, or trade."""
    columns = ["season", "player_id", "roster_id", "source", "faab", "week"]
    rows: list[dict] = []
    source_map = {"waiver": "Waiver", "free_agent": "Free agent", "trade": "Trade"}
    for txn in transactions or []:
        if not isinstance(txn, dict) or txn.get("status") != "complete":
            continue
        kind = str(txn.get("type") or "")
        source = source_map.get(kind)
        if source is None:
            continue
        adds = txn.get("adds") or {}
        if not isinstance(adds, dict) or not adds:
            continue
        settings = txn.get("settings") or {}
        faab = None
        if kind == "waiver":
            try:
                faab = int(settings.get("waiver_bid") or 0)
            except (TypeError, ValueError):
                faab = 0
        elif kind == "free_agent":
            faab = 0
        week = txn.get("leg")
        try:
            week = int(week or 0)
        except (TypeError, ValueError):
            week = 0
        for player_id, roster_id in adds.items():
            rows.append({
                "season": str(season),
                "player_id": str(player_id),
                "roster_id": str(roster_id),
                "source": source,
                "faab": faab,
                "week": week,
            })
    return pd.DataFrame(rows, columns=columns)


def first_acquisition_frame(
    transactions_by_season: Mapping[str, list],
    roster_owner: Mapping[tuple[str, str], str],
) -> pd.DataFrame:
    """Keep the first completed add per player-season-manager."""
    columns = ["season", "player_id", "user_id", "roster_id", "source", "faab", "week"]
    frames = [
        transaction_adds_frame(transactions, season)
        for season, transactions in transactions_by_season.items()
    ]
    if not frames:
        return pd.DataFrame(columns=columns)
    adds = pd.concat(frames, ignore_index=True)
    if adds.empty:
        return pd.DataFrame(columns=columns)
    adds["user_id"] = [
        str(roster_owner.get((str(season), str(roster_id)), "") or "")
        for season, roster_id in zip(adds["season"], adds["roster_id"])
    ]
    adds = adds[adds["user_id"].ne("")]
    if adds.empty:
        return pd.DataFrame(columns=columns)
    adds = adds.sort_values(["season", "user_id", "player_id", "week"])
    return adds.drop_duplicates(["season", "user_id", "player_id"], keep="first")[columns]


def attach_acquisitions(
    values: pd.DataFrame,
    acquisitions: pd.DataFrame,
) -> pd.DataFrame:
    """Replace generic in-season labels when transaction history is available."""
    if values.empty:
        return values.copy()
    out = values.copy()
    if acquisitions is None or acquisitions.empty:
        return out
    detail = acquisitions[[
        "season", "user_id", "player_id", "source", "faab", "week",
    ]].rename(columns={"source": "acq_source", "week": "acq_week"})
    out = out.merge(detail, on=["season", "user_id", "player_id"], how="left")
    unlabeled = out["source"].eq("In-season addition") & out["acq_source"].notna()
    out.loc[unlabeled, "source"] = out.loc[unlabeled, "acq_source"]
    if "faab" not in out.columns:
        out["faab"] = pd.NA
    if "acq_week" not in out.columns:
        out["acq_week"] = pd.NA
    return out.drop(columns=["acq_source"], errors="ignore")


TRADE_OUTCOME_COLUMNS = [
    "season", "week", "transaction_id", "user_id",
    "got_points", "gave_points", "net",
    "got_names", "gave_names", "extra", "opponent", "label",
    "player_only",
]


def _txn_week(txn: Mapping) -> int:
    try:
        return int(txn.get("leg") or 0)
    except (TypeError, ValueError):
        return 0


def _roster_id_for_user(
    roster_owner: Mapping[tuple[str, str], str],
    season: str,
    user_id: str,
) -> str:
    season = str(season)
    user_id = str(user_id)
    for (row_season, roster_id), owner_id in roster_owner.items():
        if str(row_season) == season and str(owner_id) == user_id:
            return str(roster_id)
    return ""


def _player_name_lookup(player_weeks: pd.DataFrame) -> dict[str, str]:
    names: dict[str, str] = {}
    if player_weeks is None or player_weeks.empty:
        return names
    for player_id, group in player_weeks.groupby("player_id", sort=False):
        key = str(player_id)
        for raw in group["player_name"]:
            label = str(raw or "").strip()
            if label and label != key:
                names[key] = label
                break
        names.setdefault(key, key)
    return names


def _join_player_names(player_ids: list[str], names: Mapping[str, str]) -> str:
    if not player_ids:
        return "(none)"
    return ", ".join(names.get(player_id, player_id) for player_id in player_ids)


def _lineup_points_after(
    player_weeks: pd.DataFrame,
    *,
    season: str,
    after_week: int,
    user_id: str,
    player_ids: list[str],
) -> float:
    if player_weeks is None or player_weeks.empty or not player_ids or not user_id:
        return 0.0
    mask = (
        player_weeks["season"].astype(str).eq(str(season))
        & player_weeks["week"].gt(int(after_week))
        & player_weeks["user_id"].astype(str).eq(str(user_id))
        & player_weeks["player_id"].astype(str).isin(player_ids)
        & player_weeks["is_starter"]
        & player_weeks["active_matchup"]
    )
    return round(float(player_weeks.loc[mask, "points"].sum()), 2)


def _trade_extras(txn: Mapping, selected_roster: str) -> list[str]:
    extras: list[str] = []
    selected_roster = str(selected_roster)
    for pick in txn.get("draft_picks") or []:
        if not isinstance(pick, dict):
            continue
        season = str(pick.get("season") or "").strip()
        try:
            rnd = int(pick.get("round") or 0)
        except (TypeError, ValueError):
            rnd = 0
        label = f"{season} R{rnd}".strip() if season else f"R{rnd}"
        new_owner = str(pick.get("owner_id") or "")
        old_owner = str(pick.get("previous_owner_id") or "")
        if new_owner == selected_roster:
            extras.append(f"got {label}")
        elif old_owner == selected_roster:
            extras.append(f"sent {label}")
    for item in txn.get("waiver_budget") or []:
        if not isinstance(item, dict):
            continue
        try:
            amount = int(item.get("amount") or 0)
        except (TypeError, ValueError):
            amount = 0
        sender = str(item.get("sender") or "")
        receiver = str(item.get("receiver") or "")
        if receiver == selected_roster:
            extras.append(f"got ${amount} FAAB")
        elif sender == selected_roster:
            extras.append(f"sent ${amount} FAAB")
    return extras


def trade_outcome_frame(
    transactions_by_season: Mapping[str, list],
    roster_owner: Mapping[tuple[str, str], str],
    player_weeks: pd.DataFrame,
    selected_user_id: str,
    identity_map: Mapping[str, str] | None = None,
) -> pd.DataFrame:
    """Score each completed trade as got-vs-gave starting-lineup points.

    Got is what received players scored for this manager after the trade week.
    Gave is what sent players scored for their new manager after that week.
    Picks and FAAB are listed, not converted to points. The four-week roster
    filter does not apply, so late-season trades still appear.
    """
    identity_map = identity_map or {}
    names = _player_name_lookup(player_weeks)
    rows: list[dict] = []
    selected_user_id = str(selected_user_id)
    for season, transactions in (transactions_by_season or {}).items():
        season = str(season)
        selected_roster = _roster_id_for_user(
            roster_owner, season, selected_user_id,
        )
        if not selected_roster:
            continue
        for txn in transactions or []:
            if not isinstance(txn, dict):
                continue
            if txn.get("status") != "complete":
                continue
            if str(txn.get("type") or "") != "trade":
                continue
            adds = txn.get("adds") or {}
            drops = txn.get("drops") or {}
            if not isinstance(adds, dict):
                adds = {}
            if not isinstance(drops, dict):
                drops = {}
            involved = {
                str(roster_id) for roster_id in (txn.get("roster_ids") or [])
                if roster_id is not None
            }
            involved |= {str(roster_id) for roster_id in adds.values()}
            involved |= {str(roster_id) for roster_id in drops.values()}
            if selected_roster not in involved:
                continue
            got_ids = [
                str(player_id) for player_id, roster_id in adds.items()
                if str(roster_id) == selected_roster
            ]
            gave_ids = [
                str(player_id) for player_id, roster_id in drops.items()
                if str(roster_id) == selected_roster
            ]
            if not got_ids and not gave_ids:
                continue
            extras = _trade_extras(txn, selected_roster)
            week = _txn_week(txn)
            got_points = _lineup_points_after(
                player_weeks,
                season=season,
                after_week=week,
                user_id=selected_user_id,
                player_ids=got_ids,
            )
            add_rosters = {
                str(player_id): str(roster_id) for player_id, roster_id in adds.items()
            }
            gave_points = 0.0
            for player_id in gave_ids:
                to_roster = add_rosters.get(player_id, "")
                to_user = str(roster_owner.get((season, to_roster), "") or "")
                gave_points += _lineup_points_after(
                    player_weeks,
                    season=season,
                    after_week=week,
                    user_id=to_user,
                    player_ids=[player_id],
                )
            gave_points = round(gave_points, 2)
            other_users: list[str] = []
            for roster_id in involved:
                owner_id = str(roster_owner.get((season, str(roster_id)), "") or "")
                if owner_id and owner_id != selected_user_id and owner_id not in other_users:
                    other_users.append(owner_id)
            if len(other_users) == 1:
                opponent = str(identity_map.get(other_users[0], other_users[0]))
            elif len(other_users) > 1:
                opponent = "multiple managers"
            else:
                opponent = ""
            label = f"{season} W{week}"
            if opponent:
                label = f"{label} vs {opponent}"
            txn_id = str(
                txn.get("transaction_id") or f"{season}-{week}-{selected_roster}"
            )
            rows.append({
                "season": season,
                "week": week,
                "transaction_id": txn_id,
                "user_id": selected_user_id,
                "got_points": got_points,
                "gave_points": gave_points,
                "net": round(got_points - gave_points, 2),
                "got_names": _join_player_names(got_ids, names),
                "gave_names": _join_player_names(gave_ids, names),
                "extra": "; ".join(extras),
                "opponent": opponent,
                "label": label,
                "player_only": not extras,
            })
    if not rows:
        return pd.DataFrame(columns=TRADE_OUTCOME_COLUMNS)
    return (
        pd.DataFrame(rows, columns=TRADE_OUTCOME_COLUMNS)
        .sort_values(["season", "week", "transaction_id"])
        .reset_index(drop=True)
    )
