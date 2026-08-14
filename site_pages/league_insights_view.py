"""Chart-first Draft & Roster Insights view for the League History page."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from fantasy import league_intelligence as li


_HERE = Path(__file__).resolve().parents[1]
_BENCHMARK_PATH = _HERE / "fantasy" / "league_intelligence_benchmarks.json"
_POSITION_COLORS = {
    "QB": "#3D95CE",
    "RB": "#00c853",
    "WR": "#a66cff",
    "TE": "#ffb000",
    "": "#8a93a0",
}


def _dark_layout(fig: go.Figure, *, height: int, title: str | None = None) -> go.Figure:
    fig.update_layout(
        title=title,
        height=height,
        margin=dict(l=35, r=20, t=55 if title else 25, b=45),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#f2f5f7"),
        legend_title_text="",
        hoverlabel=dict(bgcolor="#17202b", font_color="#ffffff"),
    )
    fig.update_xaxes(gridcolor="rgba(255,255,255,.08)", zerolinecolor="rgba(255,255,255,.15)")
    fig.update_yaxes(gridcolor="rgba(255,255,255,.08)", zerolinecolor="rgba(255,255,255,.15)")
    return fig


def _sorted_seasons(seasons: dict) -> list[str]:
    def _key(value):
        try:
            return int(value)
        except (TypeError, ValueError):
            return str(value)

    return sorted((str(value) for value in seasons), key=_key)


def _scope_history(history: dict, season_filter: str) -> tuple[dict, str]:
    seasons = history.get("seasons", {})
    if season_filter != "All Time":
        keep = [str(season_filter)]
        label = str(season_filter)
    else:
        completed = [
            season for season in _sorted_seasons(seasons)
            if seasons[season].get("draft_picks")
        ]
        options = ["Last 3 completed seasons", "All available seasons"]
        window = st.radio(
            "Insight window", options, horizontal=True, key="lh_insight_window",
            help="Three seasons is the default so current behavior is not overwhelmed by old league eras.",
        )
        keep = completed[-3:] if window == options[0] else _sorted_seasons(seasons)
        label = "–".join(keep) if keep else "No completed drafts"
    scoped = {season: seasons[season] for season in keep if season in seasons}
    return {"league_name": history.get("league_name"), "seasons": scoped}, label


def _manager_control(scoped: dict) -> tuple[str | None, str]:
    identities = li.manager_identity_map(scoped["seasons"])
    if not identities:
        return None, "Manager"
    labels: dict[str, str] = {}
    for user_id, username in identities.items():
        label = username
        if label in labels:
            label = f"{username} · {user_id[-4:]}"
        labels[label] = user_id
    ordered = sorted(labels, key=str.casefold)
    selected_label = st.selectbox("Manager profile", ordered, key="lh_insight_manager")
    return labels[selected_label], selected_label.split(" · ")[0]


def _load_benchmarks() -> dict:
    try:
        return json.loads(_BENCHMARK_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def _format_matches_benchmark(scoped: dict) -> bool:
    usable = [data for data in scoped["seasons"].values() if data.get("draft_picks")]
    if not usable:
        return False
    for data in usable:
        settings = data.get("league_settings") or {}
        scoring = settings.get("scoring_settings") or {}
        rounds = max(
            [int(pick.get("round") or 0) for pick in data.get("draft_picks", [])] or [0]
        )
        if not (
            int(settings.get("total_rosters") or 0) == 12
            and rounds == 14
            and float(scoring.get("rec") or 0) == 0.5
            and float(scoring.get("pass_td") or 0) == 4.0
            and "SUPER_FLEX" not in (settings.get("roster_positions") or [])
        ):
            return False
    return True


def _render_insight_cards(insights: list[dict]) -> None:
    for insight in insights:
        st.info(
            f"**{insight['title']}**  \n"
            f"{insight['finding']}  \n\n"
            f"{insight['meaning']}  \n\n"
            f"*{insight['confidence']} evidence · {insight['evidence']}*"
        )


def _render_draft_room(scoped: dict, selected_user_id: str | None, manager_name: str) -> None:
    picks = li.draft_pick_frame(scoped["seasons"])
    if picks.empty:
        st.info("No completed Sleeper draft boards are linked to this history window.")
        return

    manager_seasons = li.manager_season_frame(picks)
    drafts = int(picks["draft_id"].replace("", pd.NA).nunique())
    firsts = li.first_pick_timing_frame(picks)
    first_qb = firsts[firsts["position"].eq("QB")]["round"].median()
    first_te = firsts[firsts["position"].eq("TE")]["round"].median()
    construction = li.roster_construction_frame(manager_seasons)
    extra_onesies = (
        float((construction["avg_qb"] - 1).mean() + (construction["avg_te"] - 1).mean())
        if not construction.empty else 0
    )

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Drafts analyzed", drafts)
    m2.metric("Median first QB", f"Round {first_qb:g}" if pd.notna(first_qb) else "—")
    m3.metric("Median first TE", f"Round {first_te:g}" if pd.notna(first_te) else "—")
    m4.metric("Extra QB/TE picks", f"{extra_onesies:.2f} / team")

    matrix = li.position_round_matrix(picks)
    if not matrix.empty:
        heat = go.Figure(go.Heatmap(
            z=matrix.values,
            x=[int(value) for value in matrix.columns],
            y=list(matrix.index),
            text=[[f"{value:.1f}" for value in row] for row in matrix.values],
            texttemplate="%{text}",
            colorscale=[
                [0.0, "#101821"], [0.35, "#173a4c"],
                [0.7, "#087a64"], [1.0, "#00c853"],
            ],
            colorbar=dict(title="Picks<br>per draft"),
            hovertemplate="%{y} · Round %{x}<br>%{z:.2f} picks/draft<extra></extra>",
        ))
        _dark_layout(heat, height=350, title="Where the room spends each round")
        heat.update_xaxes(title="Round", dtick=1)
        heat.update_yaxes(title="")
        st.plotly_chart(heat, width="stretch")
        peak_position, peak_round = matrix.stack().idxmax()
        peak_value = float(matrix.loc[peak_position, peak_round])
        st.caption(
            f"The most concentrated cell is {peak_position} in Round {int(peak_round)} "
            f"({peak_value:.1f} selections per draft). Dark cells show where waiting is usually safe; "
            "bright cells show where a tier can disappear between turns."
        )

    cumulative = li.cumulative_position_frame(picks)
    if not cumulative.empty:
        left, right = st.columns(2)
        for container, position, color in (
            (left, "QB", _POSITION_COLORS["QB"]),
            (right, "TE", _POSITION_COLORS["TE"]),
        ):
            with container:
                fig = go.Figure()
                position_data = cumulative[cumulative["position"].eq(position)]
                for season, season_data in position_data.groupby("season", sort=True):
                    fig.add_trace(go.Scatter(
                        x=season_data["round"], y=season_data["cumulative"],
                        mode="lines+markers", name=str(season),
                        line=dict(width=3),
                        hovertemplate=f"{season} · Round %{{x}}<br>{position}s drafted: %{{y}}<extra></extra>",
                    ))
                _dark_layout(fig, height=360, title=f"{position} draft timing")
                fig.update_xaxes(title="Round", dtick=1)
                fig.update_yaxes(title=f"Cumulative {position}s", rangemode="tozero")
                st.plotly_chart(fig, width="stretch")
                st.caption(
                    "A steep section is the run to anticipate. A flat section means that position "
                    "barely moved, so chasing the previous run would have sacrificed value elsewhere."
                )

    if not construction.empty:
        tax = go.Figure()
        tax.add_trace(go.Bar(
            x=construction["season"], y=construction["avg_qb"], name="QB per team",
            marker_color=_POSITION_COLORS["QB"],
        ))
        tax.add_trace(go.Bar(
            x=construction["season"], y=construction["avg_te"], name="TE per team",
            marker_color=_POSITION_COLORS["TE"],
        ))
        tax.add_hline(y=1, line_dash="dash", line_color="#f2f5f7",
                      annotation_text="One starter", annotation_position="top left")
        _dark_layout(tax, height=380, title="The backup QB/TE tax")
        tax.update_layout(barmode="group")
        tax.update_yaxes(title="Players drafted per team", rangemode="tozero")
        st.plotly_chart(tax, width="stretch")
        st.caption(
            f"The room spends {extra_onesies:.2f} picks per team beyond one QB and one TE. "
            "A lean onesie build converts that draft capital into an extra RB/WR lottery ticket; "
            "the tradeoff is accepting a waiver decision during byes or injuries."
        )

    profile = manager_seasons[manager_seasons["user_id"].eq(str(selected_user_id))]
    if not profile.empty:
        profile = profile.sort_values("season")
        tendency = go.Figure()
        tendency.add_trace(go.Scatter(
            x=profile["season"], y=profile["first_qb_round"],
            name="First QB", mode="lines+markers", marker_color=_POSITION_COLORS["QB"],
        ))
        tendency.add_trace(go.Scatter(
            x=profile["season"], y=profile["first_te_round"],
            name="First TE", mode="lines+markers", marker_color=_POSITION_COLORS["TE"],
        ))
        _dark_layout(tendency, height=350, title=f"{manager_name}'s QB/TE timing")
        tendency.update_yaxes(title="Round selected", autorange="reversed", dtick=1)
        tendency.update_xaxes(title="Season")
        st.plotly_chart(tendency, width="stretch")
        qb_range = profile["first_qb_round"].dropna()
        if len(qb_range) >= 3 and qb_range.max() - qb_range.min() <= 1:
            interpretation = (
                f"{manager_name} has taken the first QB within a one-round band in every observed draft. "
                "That is actionable when this manager sits between two of your turns."
            )
        else:
            interpretation = (
                f"{manager_name}'s timing moves with the board, so the chart is a bias check rather than "
                "a dependable prediction of one exact round."
            )
        st.caption(interpretation)

    benchmarks = _load_benchmarks()
    if _format_matches_benchmark(scoped):
        market = benchmarks.get("first_pick_market_medians", {})
        compare_rows = []
        for row in firsts.itertuples(index=False):
            baseline = market.get(str(row.season), {}).get(row.position)
            if baseline is None:
                continue
            compare_rows.append({
                "season": str(row.season), "position": row.position,
                "gap": float(row.pick_no) - float(baseline),
                "league_pick": int(row.pick_no), "market_pick": float(baseline),
                "player": row.player_name,
            })
        if compare_rows:
            compare = pd.DataFrame(compare_rows)
            market_fig = go.Figure()
            for position in ("QB", "TE"):
                part = compare[compare["position"].eq(position)]
                market_fig.add_trace(go.Bar(
                    x=part["season"], y=part["gap"], name=position,
                    marker_color=_POSITION_COLORS[position],
                    customdata=part[["league_pick", "market_pick", "player"]],
                    hovertemplate=(
                        "%{x} " + position + "<br>League first: #%{customdata[0]} (%{customdata[2]})"
                        "<br>Comparable median: #%{customdata[1]}<br>Difference: %{y:+.1f}<extra></extra>"
                    ),
                ))
            market_fig.add_hline(y=0, line_color="#f2f5f7", line_width=1)
            _dark_layout(market_fig, height=360, title="Does this room delay the first QB or TE?")
            market_fig.update_layout(barmode="group")
            market_fig.update_yaxes(title="Picks later than comparable Sleeper median")
            st.plotly_chart(market_fig, width="stretch")
            st.caption(
                "Positive bars are real discounts relative to comparable 12-team, half-PPR, four-point-passing-TD "
                "drafts. A single positive season is not a promise that the same player tier will fall again."
            )

        timing = benchmarks.get("timing_study", {})
        if timing:
            st.subheader("General timing evidence")
            q_col, t_col = st.columns(2)
            q = timing.get("QB", {})
            t = timing.get("TE", {})
            q_col.info(
                f"**QB: Round {q.get('supported_round_window', '—')}**  \n"
                f"Evidence grade: {str(q.get('evidence_grade', 'unknown')).title()}. "
                "This supports a late-QB default, not blindly passing a large tier discount."
            )
            t_col.info(
                f"**TE: Rounds {t.get('supported_round_window', '—')}**  \n"
                f"Evidence grade: {str(t.get('evidence_grade', 'unknown')).title()}. "
                "This is the stronger default when the elite TE tier does not fall."
            )

    st.subheader("What the evidence suggests")
    _render_insight_cards(li.draft_insights(picks, manager_seasons, selected_user_id))


def _manager_player_data(scoped: dict, selected_user_id: str, player_directory_loader):
    with st.spinner("Matching Sleeper player names…"):
        player_directory = player_directory_loader() or {}
    weeks = li.player_week_frame(scoped["seasons"], player_directory)
    weeks = weeks[weeks["user_id"].eq(str(selected_user_id))].copy()
    seasons = li.player_season_summary(weeks, min_roster_weeks=4)
    career = li.player_career_summary(seasons)
    eligible_weeks = li.eligible_player_weeks(weeks, seasons)
    return weeks, seasons, career, eligible_weeks


def _render_player_facts(
    career: pd.DataFrame,
    eligible_weeks: pd.DataFrame,
    manager_name: str,
) -> None:
    if career.empty or eligible_weeks.empty:
        return
    top = career.iloc[0]
    active_starts = eligible_weeks[
        eligible_weeks["active_matchup"] & eligible_weeks["is_starter"]
    ]
    bench = eligible_weeks[
        eligible_weeks["active_matchup"] & ~eligible_weeks["is_starter"]
    ]
    best_start = active_starts.loc[active_starts["points"].idxmax()] if not active_starts.empty else None
    bench_high = bench.loc[bench["points"].idxmax()] if not bench.empty else None

    weekly_winners = []
    for _, group in active_starts.groupby(["season", "week"]):
        if not group.empty:
            weekly_winners.append(group.loc[group["points"].idxmax(), "player_name"])
    weekly_mvp = pd.Series(weekly_winners).value_counts().index[0] if weekly_winners else "—"
    weekly_mvp_count = int(pd.Series(weekly_winners).value_counts().iloc[0]) if weekly_winners else 0

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Scoring king", top["player_name"], f"{top['lineup_points']:.1f} lineup pts")
    if best_start is not None:
        c2.metric("Best start", best_start["player_name"], f"{best_start['points']:.1f} pts")
        c2.caption(f"{best_start['season']} Week {int(best_start['week'])}")
    if bench_high is not None:
        c3.metric("Biggest bench regret", bench_high["player_name"], f"{bench_high['points']:.1f} pts")
        c3.caption(f"{bench_high['season']} Week {int(bench_high['week'])}")
    c4.metric("Most weekly MVPs", weekly_mvp, f"Led {manager_name} {weekly_mvp_count} times")


def _render_my_team(scoped: dict, selected_user_id: str | None, manager_name: str, player_directory_loader) -> None:
    if not selected_user_id:
        st.info("Choose a manager to load player history.")
        return
    _, player_seasons, career, eligible_weeks = _manager_player_data(
        scoped, selected_user_id, player_directory_loader
    )
    if career.empty:
        st.info("No player-season passed the four-week roster filter in this window.")
        return

    st.caption(
        "A player-season qualifies after four rostered weeks, with separate stints combined. "
        "Scoring counts only weeks containing a real matchup; the main ranking uses points that entered the lineup."
    )
    _render_player_facts(career, eligible_weeks, manager_name)

    top = career.head(12).sort_values("lineup_points")
    scorer = go.Figure(go.Bar(
        x=top["lineup_points"], y=top["player_name"], orientation="h",
        marker_color=[_POSITION_COLORS.get(position, "#8a93a0") for position in top["position"]],
        customdata=top[["position", "starts", "seasons"]],
        hovertemplate=(
            "%{y} · %{customdata[0]}<br>Lineup points: %{x:.1f}<br>Starts: %{customdata[1]}"
            "<br>Seasons: %{customdata[2]}<extra></extra>"
        ),
    ))
    _dark_layout(scorer, height=500, title=f"Who scored the most for {manager_name}?")
    scorer.update_xaxes(title="Points in the starting lineup")
    scorer.update_yaxes(title="")
    st.plotly_chart(scorer, width="stretch")
    leader = career.iloc[0]
    st.caption(
        f"{leader['player_name']} leads because {leader['lineup_points']:.1f} points actually entered "
        f"the lineup across {leader['season_count']} qualifying season(s). Bench production is intentionally "
        "not allowed to inflate this ranking."
    )

    leaders = (
        player_seasons.sort_values(["season", "lineup_points"], ascending=[True, False])
        .groupby("season", as_index=False).first()
    )
    season_fig = go.Figure(go.Bar(
        x=leaders["season"], y=leaders["lineup_points"],
        marker_color=[_POSITION_COLORS.get(position, "#8a93a0") for position in leaders["position"]],
        text=leaders["player_name"], textposition="outside",
        customdata=leaders[["player_name", "position"]],
        hovertemplate="%{x}: %{customdata[0]} (%{customdata[1]})<br>%{y:.1f} lineup pts<extra></extra>",
    ))
    _dark_layout(season_fig, height=390, title="The player who carried each season")
    season_fig.update_yaxes(title="Starting-lineup points", rangemode="tozero")
    st.plotly_chart(season_fig, width="stretch")
    if len(leaders) > 1:
        high = leaders.loc[leaders["lineup_points"].idxmax()]
        st.caption(
            f"The strongest individual season was {high['player_name']} in {high['season']} "
            f"with {high['lineup_points']:.1f} lineup points. This identifies the season's engine, "
            "not merely the player who stayed rostered longest."
        )

    position_points = (
        player_seasons.groupby("position", as_index=False)["lineup_points"].sum()
        .sort_values("lineup_points", ascending=False)
    )
    position_points = position_points[position_points["lineup_points"].gt(0)]
    if not position_points.empty:
        donut = go.Figure(go.Pie(
            labels=position_points["position"], values=position_points["lineup_points"],
            hole=.58,
            marker_colors=[_POSITION_COLORS.get(position, "#8a93a0") for position in position_points["position"]],
            textinfo="label+percent",
            hovertemplate="%{label}: %{value:.1f} lineup pts (%{percent})<extra></extra>",
        ))
        _dark_layout(donut, height=390, title="Where the qualifying lineup points came from")
        st.plotly_chart(donut, width="stretch")
        lead_position = position_points.iloc[0]
        share = lead_position["lineup_points"] / position_points["lineup_points"].sum()
        st.caption(
            f"{lead_position['position']} supplied the largest share at {share:.1%}. Because the four-week "
            "filter excludes short stays, use this as a roster-construction fingerprint rather than a complete "
            "accounting of every emergency start."
        )


def _render_values(scoped: dict, selected_user_id: str | None, manager_name: str, player_directory_loader) -> None:
    if not selected_user_id:
        st.info("Choose a manager to load acquisition value.")
        return
    _, player_seasons, _, _ = _manager_player_data(scoped, selected_user_id, player_directory_loader)
    picks = li.draft_pick_frame(scoped["seasons"])
    picks = picks[picks["user_id"].eq(str(selected_user_id))]
    values = li.value_frame(player_seasons, picks)
    if values.empty:
        st.info("No qualifying player-seasons are available for this manager.")
        return

    st.caption(
        "Drafted players are matched to their actual round. Everyone else is labeled an in-season addition "
        "until the transaction-history phase separates waivers, free agents and trades. No arbitrary combined "
        "value score is used."
    )

    drafted = values[values["source"].eq("Drafted")].copy()
    if not drafted.empty:
        draft_fig = go.Figure()
        for season, group in drafted.groupby("season", sort=True):
            draft_fig.add_trace(go.Scatter(
                x=group["round"], y=group["lineup_points"], mode="markers+text",
                name=str(season), text=group["player_name"], textposition="top center",
                marker=dict(size=11, opacity=.85),
                customdata=group[["player_name", "position", "starts", "pick_no"]],
                hovertemplate=(
                    "%{customdata[0]} · %{customdata[1]}<br>Round %{x:.0f}, pick #%{customdata[3]:.0f}"
                    "<br>Lineup points: %{y:.1f}<br>Starts: %{customdata[2]}<extra></extra>"
                ),
            ))
        _dark_layout(draft_fig, height=520, title=f"{manager_name}'s draft cost versus realized production")
        draft_fig.update_xaxes(title="Draft round", dtick=1, autorange="reversed")
        draft_fig.update_yaxes(title="Starting-lineup points", rangemode="tozero")
        st.plotly_chart(draft_fig, width="stretch")
        late = drafted[drafted["round"].ge(6)].sort_values("lineup_points", ascending=False)
        if not late.empty:
            hit = late.iloc[0]
            st.caption(
                f"The clearest late-round return is {hit['player_name']}: Round {int(hit['round'])}, "
                f"{hit['lineup_points']:.1f} lineup points in {hit['season']}. The upper-right portion of the "
                "chart is where meaningful production arrived after the expensive rounds were gone."
            )

    additions = values[values["source"].eq("In-season addition")].sort_values(
        "lineup_points", ascending=False
    ).head(10)
    if not additions.empty:
        additions = additions.sort_values("lineup_points")
        add_fig = go.Figure(go.Bar(
            x=additions["lineup_points"], y=additions["player_name"], orientation="h",
            marker_color=[_POSITION_COLORS.get(position, "#8a93a0") for position in additions["position"]],
            customdata=additions[["season", "position", "starts"]],
            hovertemplate=(
                "%{y} · %{customdata[1]}<br>%{customdata[0]}<br>Lineup points: %{x:.1f}"
                "<br>Starts: %{customdata[2]}<extra></extra>"
            ),
        ))
        _dark_layout(add_fig, height=440, title="Production acquired after the draft")
        add_fig.update_xaxes(title="Starting-lineup points")
        add_fig.update_yaxes(title="")
        st.plotly_chart(add_fig, width="stretch")
        best = additions.iloc[-1]
        st.caption(
            f"{best['player_name']} produced the most post-draft lineup value in this window at "
            f"{best['lineup_points']:.1f} points. The next phase will determine whether that acquisition "
            "was a waiver claim, free-agent add or trade before calling it a specific type of steal."
        )


def render(history: dict, season_filter: str, player_directory_loader) -> None:
    st.subheader("Draft & Roster Insights")
    st.caption(
        "League-specific tendencies and manager history from Sleeper's completed drafts and weekly roster snapshots. "
        "Three drafts can reveal repeated habits, but every recommendation below keeps its sample visible."
    )
    scoped, scope_label = _scope_history(history, season_filter)
    selected_user_id, manager_name = _manager_control(scoped)
    st.caption(f"Analysis window: {scope_label}")

    view = st.radio(
        "Insight view", ["Draft Room", "My Team", "Best Values"],
        horizontal=True, key="lh_insight_view", label_visibility="collapsed",
    )
    if view == "Draft Room":
        _render_draft_room(scoped, selected_user_id, manager_name)
    elif view == "My Team":
        _render_my_team(scoped, selected_user_id, manager_name, player_directory_loader)
    else:
        _render_values(scoped, selected_user_id, manager_name, player_directory_loader)
