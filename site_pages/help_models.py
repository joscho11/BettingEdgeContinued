"""Help-page model rundowns. Streamlit render only. Data lives in model_explanations."""
from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
_BETTING = _ROOT / "betting"
if str(_BETTING) not in sys.path:
    sys.path.insert(0, str(_BETTING))

import model_explanations as me
from live_2026 import LIVE_HIGH_ATS, LIVE_HIGH_N, LIVE_HIGH_WILSON_LOWER, LIVE_HIGH_WINS

BREAKEVEN = 52.4
ACCENT = "#8abcf5"


def _bar(labels, values, *, y_title, text=None, colors=None, hline=None, hline_text=None,
         height=280):
    import plotly.graph_objects as go

    fig = go.Figure(go.Bar(
        x=list(labels),
        y=list(values),
        text=list(text) if text is not None else None,
        textposition="outside",
        marker_color=colors or ACCENT,
        hovertemplate="%{x}<br>%{y}<extra></extra>",
    ))
    if hline is not None:
        fig.add_hline(
            y=hline, line_dash="dash", line_color="#888",
            annotation_text=hline_text or "", annotation_position="right",
            annotation_font=dict(size=11, color="#9aa4b2"),
        )
    ymax = max(list(values) + ([hline] if hline is not None else [0]))
    fig.update_layout(
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font_color="white",
        yaxis=dict(
            title=y_title,
            gridcolor="#2d3748",
            range=[0, ymax * 1.18 if ymax else 1],
        ),
        xaxis=dict(gridcolor="#2d3748"),
        showlegend=False,
        height=height,
        margin=dict(t=24, b=20, r=112 if hline is not None else 24, l=8),
    )
    st.plotly_chart(fig, width="stretch")


def _cards(models):
    if not models:
        return
    for model in models:
        st.markdown(me.chart_html(model), unsafe_allow_html=True)


def render_rundowns():
    st.markdown(me.CHART_CSS, unsafe_allow_html=True)
    st.subheader("How the models work")
    st.caption(
        "Plain-language rundown for every number this site currently publishes. "
        "The bars show which inputs the model leans on, or how it scored on held-out seasons. "
        "They are not a promise that any one game or player will hit."
    )

    _spread_2026()
    _spread_2025_demo()
    _season_totals()
    _draft_board()
    _weekly_fantasy()
    _totals_demo()
    _rookie_board()


def _spread_2026():
    with st.expander("How the 2026 spread model works"):
        st.markdown(f"""
Each week the model guesses the **margin leftover versus the Tuesday 9:00 ET spread**.
It is 75% a tree model (XGBoost) and 25% a linear model (Ridge). Both see the same
55 inputs. The Tuesday line is one of those inputs. The rest are how the two teams
have been playing, who is available, quarterback and coaching changes, rest, weather,
and venue.

**Every game still gets a pick.** **HIGH** (green) is the only highlighted slice: the
model disagrees with the Tuesday line by 3 or more points, and the live line still
does. If the line moves and that gap falls under 3, HIGH is dropped. A later line
cannot create HIGH. There is no medium tier. The last regular-season week is skipped
for HIGH. Totals are not on the 2026 week page.

**The live claim** is that HIGH slice: **{LIVE_HIGH_WINS}/{LIVE_HIGH_N} = {LIVE_HIGH_ATS * 100:.2f}%**
ATS, one-sided 95% Wilson lower bound **{LIVE_HIGH_WILSON_LOWER * 100:.2f}%**, walk-forward
2021-2025. That interval is above 52.4%. Betting every game does not clear and is not
the claim. No 2026 games are graded yet.

Picks lock Tuesday 9:00 ET. Matchups for weeks 1-18 are on Weekly Predictions now.
        """)
        rows = me.spread_high_season_rows()
        _bar(
            [r["season"] for r in rows],
            [r["pct"] for r in rows],
            y_title="HIGH ATS %",
            text=[r["record"] for r in rows],
            colors=["#00c853" if r["pct"] >= BREAKEVEN else "#ff5252" for r in rows],
            hline=BREAKEVEN,
            hline_text="Break even (52.4%)",
        )
        st.caption(
            "HIGH cover rate by season on the 2021-2025 walk-forward book. 2021 carries "
            "the Wilson interval. 2022 is a coin flip. This does not prove 2026 will look "
            "like any one of those years."
        )


def _spread_2025_demo():
    with st.expander("How the 2025 demo spread worked"):
        st.markdown("""
Weeks 10 through the end of 2025 on this site are a **frozen walkthrough** of the old
three-model consensus, not the 2026 Tuesday model.

The old system blended 75% XGBoost with 25% Ridge for the predicted margin, then let
XGBoost, Ridge, and LightGBM vote on the side. HIGH meant all three agreed and the
edge was 3 or more points. MEDIUM meant they agreed with a 1-point edge. PASS meant
they disagreed or the edge was tiny.

Those weeks still show HIGH / MED / PASS badges and a Min Edge slider. They will not
be restated as the live book.
        """)
        card = me.card_by_id("spread_xgb")
        if card:
            st.caption("What the old XGBoost piece leaned on (mean absolute Tree SHAP, 2014-2024 training games).")
            _cards([card])
            st.caption(
                "The Tuesday line is the largest bar. That is expected. The model is scoring "
                "leftover versus a line that already prices most of the game. These shares "
                "are not a ranking of accuracy."
            )


def _season_totals():
    with st.expander("How Season Totals are built"):
        st.markdown("""
Every team gets a **projected regular-season win total** (ties count as half a win).
The 32 projections always sum to 272 scheduled games.

The model is a linear fit on last year's passing and special-teams efficiency, leftover
opportunity, quarterback status, coaching change, rest, schedule, and **the posted win
total itself**. 2026 numbers are then recentered so the league still sums to 272.

**HIGH** (the check mark) fires when the projection is at least 1 full win off the
posted number. Every other team still shows a projection. HIGH is the only certified
pick on that page.

The all-team sheet does **not** beat the posted win total on average (held-out MAE
2.26 wins versus 2.21 for the posted number). HIGH's one-sided 95% Wilson lower bound
sits under the 52.4% bar. Backtested, not live-validated. 2026 reserve counts are
withheld because camp rosters do not match post-cutdown history.
        """)
        ladder = [
            ("Repeat last year", 2.7955),
            ("Retired Monte Carlo", 2.3650),
            ("This model", 2.2578),
            ("Posted win total", 2.2088),
        ]
        _bar(
            [name for name, _ in ladder],
            [val for _, val in ladder],
            y_title="Average miss (wins)",
            text=[f"{val:.2f}" for _, val in ladder],
            colors=["#8abcf5", "#8abcf5", "#d3ad63", "#00c853"],
        )
        st.caption(
            "Average miss per team on 352 held-out team-seasons (2015-2025). Lower is "
            "better. The posted number is still the tightest sheet. This chart is the "
            "all-teams projection, not the HIGH slice."
        )
        imp = me.season_totals_importance()
        _cards([{
            "label": "Season Totals · what the fit leans on",
            "method": "absolute ridge coefficient share",
            "n": 384,
            "features": imp,
        }])
        st.caption(
            "The posted win total is the largest term. The other bars are the nudges: "
            "true home games, a missing starter QB, and last year's efficiency. Signs "
            "matter on the page (unavailable QB pulls the projection down). This is not "
            "a causal ranking."
        )


def _draft_board():
    ev = me.DRAFT_BOARD_EVAL
    with st.expander("How Model Proj is built"):
        st.markdown(f"""
**Model Proj** is the published season-total half-PPR number for the 180-player board
(24 QB, 60 RB, 72 WR, 24 TE). It is a machine-learning forecast from last year's
production, draft capital, age, and leftover opportunity on the new roster. **ADP and
the two Talent Scores are not inputs.**

Points and positional ranks stay frozen until the dated early-September public-information
snapshot. Sleeper ADP and Sleeper Proj still refresh daily; those live values rewrite
the market ranks and both gap columns, not Model Proj.

**How it scored historically.** On 2021-2025 Model Proj pairwise **{ev['model_pairwise']:.4f}**
versus ADP **{ev['adp_pairwise']:.4f}**, MAE **{ev['model_mae']:.2f}** versus **{ev['adp_mae']:.2f}**,
and beat ADP ordering in {ev['seasons_beat_adp']} seasons (it lost {ev['lost_season']}).
It is **not live-validated**. The first live test is the 2026 season.
        """)
        _bar(
            ["Model Proj", "ADP"],
            [ev["model_mae"], ev["adp_mae"]],
            y_title="Average miss (half-PPR points)",
            text=[f"{ev['model_mae']:.1f}", f"{ev['adp_mae']:.1f}"],
            colors=[ACCENT, "#888888"],
        )
        st.caption(
            "Average miss versus actual season-total half-PPR on the 180-player universe, "
            "2021-2025. Lower is better. Pairwise ordering (how often the higher-ranked "
            "player scored more) is the other score on the Draft Board page. This does "
            "not prove any one 2026 row is right."
        )


def _weekly_fantasy():
    with st.expander("How weekly fantasy projections are built"):
        st.markdown("""
**2026 Week 1 is not on the site yet.** That page opens there on purpose. Rankings
land once the live weekly file is published.

**What you can read today** is the **2025 demo** (weeks 10-17). Those files came from
four per-position XGBoost models trained on 2020-2024, with 2025 held out. Scoring is
half-PPR: 0.5 per reception, yards and touchdowns as usual. Demo weeks also carry extra
stat columns (pass/rush/rec yards, receptions) from eight smaller models. Those extras
will not appear on a 2026 live week unless that file has them.

The 2026 live weekly model is a single points model across positions. It looks at recent
usage, expected fantasy points, opponent, and the implied team total. It is not the
2025 demo files.
        """)
        points = me.weekly_point_cards()
        if points:
            st.caption("2025 demo: which inputs the four fantasy-point models leaned on (XGBoost gain).")
            _cards(points)
            st.caption(
                "Recent snaps and recent fantasy points dominate every position. That is "
                "the honest story: last week's role is most of next week's projection. "
                "Top five bars do not sum to 100%."
            )


def _totals_demo():
    with st.expander("How the Over/Under model works (2025 demo, experimental)"):
        st.markdown("""
**Status: experimental. Tracking only. Do not bet these.** 2026 Weekly Predictions
does not show totals.

The 2025 demo runs a second pair of models (XGBoost and Ridge) on whether the combined
score lands under the Vegas total. A card only flags **UNDER** when both models agree.
There are no OVER bets. Recreational money shades totals high, which is the only reason
an UNDER-only rule was worth testing.

Walk-forward CV (2020-2025, n=575) hit **55.7%**. Live 2025 weeks 10-17 (n=46) hit
**52.2%**, which is break-even inside a wide interval. The amber dashed badge is there
on purpose.
        """)
        cards = [c for c in (me.card_by_id("totals_xgboost"), me.card_by_id("totals_ridge")) if c]
        _cards(cards)
        st.caption(
            "XGBoost uses gain. Ridge uses absolute standardized coefficients. Do not "
            "read either as proof the live 2025 sample has an edge."
        )


def _rookie_board():
    auc = me.ROOKIE_HIT_AUC
    with st.expander("How the Rookie Board numbers are built"):
        st.markdown(f"""
Two different numbers sit on the Rookie Board.

**Hit %** is the share of historical players with a similar profile who had at least
one startable season in their first three years (top-24 RB/WR, top-12 QB/TE, season-total
half-PPR). Three columns: draft capital only, college production and testing only, and
both. They land close together. On the {auc['holdout']} hold-out, full-model AUC was
**{auc['full']}** versus **{auc['draft_only']}** from draft slot alone. College added no
measured edge beyond where he was picked. Backtested, not live-validated. First live
test: end of 2026.

**Season-total projections** (RB, WR, TE) come from the rookie arms of the older
season-total models. They see draft slot, age, vacated opportunity, and college
production. Rookie QBs have no projection: a rookie QB season hinges on whether he
starts, which those features cannot see.
        """)
        _bar(
            ["Draft slot only", "Full model"],
            [auc["draft_only"], auc["full"]],
            y_title="Hit-probability AUC",
            text=[f"{auc['draft_only']:.3f}", f"{auc['full']:.3f}"],
            colors=["#888888", ACCENT],
            height=260,
        )
        st.caption(
            "AUC on the 2019-2023 hold-out classes. 0.50 is a coin flip. The two bars "
            "are almost the same height: draft capital is doing the work. This does not "
            "say college production is meaningless. It says it is already priced into "
            "draft slot at this sample."
        )
        rook = me.rookie_projection_cards()
        if rook:
            st.caption("Rookie season-total projections: top inputs (mean absolute Tree SHAP).")
            _cards(rook)
