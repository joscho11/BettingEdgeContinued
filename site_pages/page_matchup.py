"""Release-backed matchup detail page."""
from __future__ import annotations

import html
from pathlib import Path

import pandas as pd
import streamlit as st

import dashboard_chrome as chrome
import nav_registry
from matchups.detail import MatchupNotFound, load_matchup_detail
from matchups.social import render_social_card

_ROOT = Path(__file__).resolve().parents[1]


def _margin_label(home: str, away: str, margin: float) -> str:
    if margin > 0:
        return f"{home} by {abs(margin):.1f}"
    if margin < 0:
        return f"{away} by {abs(margin):.1f}"
    return "Even"


def _market_label(home: str, away: str, spread: float) -> str:
    if spread > 0:
        return f"{home} -{abs(spread):.1f}"
    if spread < 0:
        return f"{away} -{abs(spread):.1f}"
    return "Pick'em"


def _fmt_number(value, suffix="") -> str:
    return "—" if value is None else f"{float(value):.1f}{suffix}"


def _render_driver_cards(drivers: list[dict]) -> None:
    cards = []
    for driver in drivers:
        feature = html.escape(str(driver.get("feature") or "Unknown feature"))
        value = html.escape(str(driver.get("value") if driver.get("value") is not None else "—"))
        direction = html.escape(str(driver.get("direction") or "neutral"))
        contribution = float(driver.get("contribution") or 0.0)
        source = driver.get("source")
        source_note = f" · {html.escape(str(source))}" if source else ""
        cards.append(
            '<div class="jsa-driver-card">'
            f'<div><code>{feature}</code><span>Input {value}{source_note}</span></div>'
            f'<div class="jsa-driver-impact"><strong>{contribution:+.2f}</strong>'
            f'<span>toward {direction}</span></div></div>'
        )
    st.html(
        """
        <style>
        .jsa-driver-grid{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:.55rem}
        .jsa-driver-card{display:flex;justify-content:space-between;align-items:center;gap:1rem;
          min-width:0;padding:.75rem .85rem;border:1px solid rgba(148,163,184,.23);
          border-radius:.75rem;background:rgba(18,24,33,.7)}
        .jsa-driver-card>div:first-child{display:flex;flex-direction:column;min-width:0;gap:.18rem}
        .jsa-driver-card code{overflow-wrap:anywhere;color:#e7ecf3;font-size:.82rem}
        .jsa-driver-card span{color:#93a0b1;font-size:.75rem}
        .jsa-driver-impact{display:flex;flex:0 0 auto;flex-direction:column;text-align:right}
        .jsa-driver-impact strong{color:#35d08a;font-variant-numeric:tabular-nums}
        @media(max-width:700px){.jsa-driver-grid{grid-template-columns:1fr}}
        </style>
        <div class="jsa-driver-grid">
        """
        + "".join(cards)
        + "</div>"
    )


def _render_injury_team(entry: dict) -> None:
    team = str(entry.get("team") or "Team")
    st.markdown(f"#### {team}")
    if entry.get("availability") != "available":
        st.caption(str(entry.get("reason") or "No archived report."))
        return
    counts = entry.get("counts") or {}
    with st.container(horizontal=True, vertical_alignment="center"):
        st.badge(f"{int(counts.get('Out', 0))} out", color="red")
        st.badge(f"{int(counts.get('Doubtful', 0))} doubtful", color="orange")
        st.badge(f"{int(counts.get('Questionable', 0))} questionable", color="blue")
    players = entry.get("players") or []
    if not players:
        st.caption("No Out, Doubtful, or Questionable designation in the archived report.")
        return
    for player in players:
        status = str(player.get("status") or "Listed")
        color = {"Out": "red", "Doubtful": "orange", "Questionable": "blue"}.get(status, "gray")
        injury = str(player.get("injury") or "Not listed")
        position = str(player.get("position") or "")
        practice = player.get("practice_status")
        practice_note = f" · {practice}" if practice else ""
        st.markdown(
            f":{color}-badge[{html.escape(status)}] **{html.escape(str(player.get('player') or 'Unknown'))}** "
            f"{html.escape(position)} · {html.escape(injury)}{html.escape(practice_note)}"
        )


def _render_weather(weather: dict) -> None:
    st.markdown("### Weather context")
    source = weather.get("source") or {}
    if weather.get("availability") != "available":
        st.caption(str(weather.get("reason") or "No archived weather context."))
    else:
        with st.container(horizontal=True, key="jsa-metric-even-matchup-weather"):
            st.metric("Temperature", _fmt_number(weather.get("temperature_f"), "°F"), border=True)
            st.metric("Wind", _fmt_number(weather.get("wind_mph"), " mph"), border=True)
            st.metric("Precipitation", _fmt_number(weather.get("precipitation_in"), " in"), border=True)
            st.metric("Humidity", _fmt_number(weather.get("humidity_pct"), "%"), border=True)
        station = weather.get("station_name") or "Station unavailable"
        kickoff = weather.get("kickoff_utc") or "time unavailable"
        st.caption(f"{station} · kickoff observation {kickoff}")
    if source:
        st.markdown(f"Source: [{source.get('name', 'Weather source')}]({source.get('url')})")
        st.caption(str(source.get("timing") or ""))


def _render_model(detail: dict) -> None:
    model = detail["model"]
    st.subheader("Model view")
    outputs = pd.DataFrame(model.get("outputs") or [])
    if not outputs.empty:
        outputs = outputs.rename(
            columns={
                "model": "Model",
                "projected_margin": "Home margin",
                "edge": "Edge",
                "recommendation": "Pick",
            }
        )
        st.dataframe(
            outputs,
            hide_index=True,
            width="stretch",
            column_config={
                "Home margin": st.column_config.NumberColumn(format="%.1f"),
                "Edge": st.column_config.NumberColumn(format="%+.1f"),
            },
        )

    drivers = model.get("drivers") or []
    if drivers:
        st.markdown("### Most influential drivers")
        _render_driver_cards(drivers)
        if model.get("explanation_method"):
            st.caption(f"Explanation method: {model['explanation_method']}")
        if model.get("driver_note"):
            st.caption(str(model["driver_note"]))
        inputs = model.get("inputs") or []
        if inputs:
            with st.expander(f"All {len(inputs)} model inputs"):
                input_frame = pd.DataFrame(inputs).rename(
                    columns={"feature": "Feature", "value": "Value"}
                )
                st.dataframe(input_frame, hide_index=True, width="stretch")
    else:
        with st.container(border=True):
            st.markdown("### Per-game inputs and drivers")
            st.caption(str(model.get("driver_note") or "No explanation artifact is available."))
            st.markdown(
                "The page does not substitute global feature importance for a per-game explanation. "
                "Future releases can populate this section with a hash-verified detail sidecar."
            )


def _render_history(detail: dict) -> None:
    st.subheader("Pick history and result")
    for event in detail["history"].get("events") or []:
        timestamp = event.get("timestamp") or "Time unavailable"
        with st.container(border=True):
            st.markdown(f"**{event.get('label', 'Event')}** · {timestamp}")
            st.caption(str(event.get("detail") or ""))
    note = detail["history"].get("note")
    if note:
        st.caption(str(note))


def _share_url(slug: str) -> str:
    return f"{chrome.CANONICAL_URL.rstrip('/')}/{slug}"


def render(game_id: str) -> None:
    try:
        detail = load_matchup_detail(_ROOT, game_id)
    except MatchupNotFound:
        st.error("This matchup is not part of a validated public release.")
        page = nav_registry.PAGES.get("weekly-predictions")
        if page is not None:
            st.page_link(page, label="Back to weekly predictions", icon=":material/arrow_back:")
        return

    game, prediction, status, result = (
        detail["game"], detail["prediction"], detail["status"], detail["result"]
    )
    st.set_page_config(
        page_title=f"{game['away_team']} at {game['home_team']} | JoScho Analytics",
        page_icon="🏈",
    )
    back = nav_registry.PAGES.get("weekly-predictions")
    if back is not None:
        st.page_link(
            back,
            label="Weekly predictions",
            icon=":material/arrow_back:",
            width="content",
        )

    st.title(f"{game['away_team']} at {game['home_team']}")
    with st.container(horizontal=True, vertical_alignment="center"):
        st.badge(f"{game['season']} Week {game['week']}", color="gray")
        st.badge(
            status["label"],
            color={"HIGH": "green", "MEDIUM": "orange", "PASS": "red"}[status["label"]],
            icon=":material/check_circle:" if status["label"] == "HIGH" else ":material/remove_circle:",
        )
        st.badge(detail["release"]["status"], color="green", icon=":material/publish:")
    date_line = str(game.get("gameday") or "Date unavailable")
    if game.get("gametime"):
        date_line += f" · {game['gametime']}"
    st.caption(f"{date_line} · build {detail['release']['build_id']}")

    if game.get("historical_demo"):
        st.info(
            "**2025 historical demo.** The projection and result are real archived values, "
            "but this row was backfilled after the game. It does not have an authentic "
            "pregame freeze timestamp or line history. Model inputs and local contributions "
            "are a verified post-hoc reconstruction from the archived generator and models."
        )
    elif status.get("high_dropped"):
        st.warning("The Tuesday pick was HIGH, but the current line moved the edge below three points.")

    with st.container(horizontal=True, key="jsa-metric-even-matchup"):
        st.metric(
            "Projected margin",
            _margin_label(game["home_team"], game["away_team"], prediction["projected_margin"]),
            border=True,
        )
        st.metric(
            "Market spread",
            _market_label(game["home_team"], game["away_team"], prediction["market_spread"]),
            border=True,
        )
        st.metric("Model edge", f"{abs(prediction['model_edge']):.1f} points", border=True)
        st.metric(
            "ATS result",
            result.get("ats_result") or ("Pending" if result.get("status") != "final" else "Push"),
            border=True,
        )

    st.subheader("The call")
    call_col, release_col = st.columns([1.35, 1])
    with call_col.container(border=True, height="stretch"):
        st.markdown(f"## {prediction['recommendation']}")
        st.markdown(
            f"The model projects **{_margin_label(game['home_team'], game['away_team'], prediction['projected_margin'])}** "
            f"against a market of **{_market_label(game['home_team'], game['away_team'], prediction['market_spread'])}**."
        )
        st.caption(f"Home-margin convention · edge {prediction['model_edge']:+.1f} points")
    with release_col.container(border=True, height="stretch"):
        st.markdown("### Freeze and release")
        if status.get("freeze_at"):
            st.markdown(f"**Frozen:** {status['freeze_at']}")
        else:
            st.markdown("**Frozen:** Not archived")
        st.caption(str(status.get("freeze_note") or ""))
        st.markdown(f"**Model:** `{detail['model']['version']}`")
        st.caption(f"Published {detail['release']['published_at']}")

    if result.get("status") == "final":
        with st.container(border=True):
            away_score, home_score = result.get("away_score"), result.get("home_score")
            if away_score is not None and home_score is not None:
                st.markdown(
                    f"### Final · {game['away_team']} {int(away_score)}, {game['home_team']} {int(home_score)}"
                )
            st.markdown(f"ATS outcome: **{result.get('ats_result') or 'PUSH'}**")

    _render_model(detail)

    st.subheader("Injury context")
    injuries = detail["context"]["injuries"]
    injury_cols = st.columns(2)
    with injury_cols[0].container(border=True, height="stretch"):
        _render_injury_team(injuries["away"])
    with injury_cols[1].container(border=True, height="stretch"):
        _render_injury_team(injuries["home"])
    source = injuries.get("source") or {}
    st.markdown(f"Source: [{source.get('name', 'Injury source')}]({source.get('url')})")
    st.caption(str(source.get("timing") or ""))

    with st.container(border=True):
        _render_weather(detail["context"]["weather"])

    _render_history(detail)

    st.subheader("Share preview")
    card = render_social_card(detail)
    st.image(card, width="stretch")
    share_url = _share_url(game["slug"])
    st.caption("The URL is shareable now. The downloadable 1200×630 card is ready for social posts.")
    st.code(share_url, language=None)
    with st.container(horizontal=True):
        st.link_button("Open share link", share_url, icon=":material/open_in_new:")
        st.download_button(
            "Download social card",
            data=card,
            file_name=f"{game['slug']}.png",
            mime="image/png",
            icon=":material/download:",
        )
    st.caption(
        "Automatic route-specific social-network unfurls still require an HTML proxy or "
        "static wrapper because Streamlit serves generic initial page metadata."
    )

    st.caption("Model output is informational. Sports betting involves risk.")
