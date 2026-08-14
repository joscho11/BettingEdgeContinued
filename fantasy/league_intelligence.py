"""Pure transforms for the Sleeper League History intelligence views.

This module deliberately contains no Streamlit or network code.  The page owns
fetching/caching; these helpers turn the returned payload into auditable frames
for draft-room and manager-level roster analysis.
"""
from __future__ import annotations

from collections.abc import Mapping

import pandas as pd


CORE_POSITIONS = ("QB", "RB", "WR", "TE")
DEFAULT_MIN_ROSTER_WEEKS = 4


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
        insights.append({
            "title": "The room builds around RB/WR first",
            "finding": f"{skill_share:.0%} of picks in Rounds 1–3 were RBs or WRs.",
            "meaning": (
                "Early QB or TE usually requires a deliberate exception; the room normally "
                "leaves those positions alone while the first skill-position tiers disappear."
            ),
            "evidence": f"{len(first_three)} picks across {draft_count} drafts",
            "confidence": confidence,
        })

    construction = roster_construction_frame(manager_seasons)
    if not construction.empty:
        qb2 = float(construction["qb2_plus_rate"].mean())
        te2 = float(construction["te2_plus_rate"].mean())
        extras = float((construction["avg_qb"] - 1).mean() + (construction["avg_te"] - 1).mean())
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
            "evidence": f"{int(construction['teams'].sum())} team-drafts",
            "confidence": confidence,
        })

    run = max_position_run(picks)
    if run and run["count"] >= 6:
        insights.append({
            "title": f"Do not buy the back of a {run['position']} run",
            "finding": (
                f"The sharpest run was {run['count']} {run['position']} picks from "
                f"#{run['start_pick']}–#{run['end_pick']} in {run['season']}."
            ),
            "meaning": (
                "Once the run is underway, the untouched position usually offers the cleaner "
                "tier. Use nearby managers to anticipate a run, but avoid chasing it after the value is gone."
            ),
            "evidence": f"Best {12}-pick window across {draft_count} drafts",
            "confidence": confidence,
        })

    if selected_user_id:
        selected = manager_seasons[manager_seasons["user_id"].eq(str(selected_user_id))]
        if not selected.empty:
            manager_rb_share = float(selected["rb_first_four"].sum() / (4 * len(selected)))
            league_rb_share = float(manager_seasons["rb_first_four"].sum() / (4 * len(manager_seasons)))
            delta = manager_rb_share - league_rb_share
            direction = "more" if delta >= 0 else "fewer"
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
                "evidence": f"{len(selected)} of your drafts",
                "confidence": "Emerging" if len(selected) >= 3 else "Limited",
            })
    return insights


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
