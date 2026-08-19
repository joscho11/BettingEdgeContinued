"""Season Win Totals page.

READ-ONLY over files published from seasonal_totals_v2_beta:

    futures/published/season_totals_2026.csv
    futures/published/evidence.json

No model runs here. pandas and streamlit only.

The page is a 32-team projection table. Certified picks are HIGH only.
Copy is checked against futures/language_fence.py.
"""
import json
from pathlib import Path

import pandas as pd
import streamlit as st

_HERE = Path(__file__).resolve().parents[1]
_CSV = _HERE / "futures" / "published" / "season_totals_2026.csv"
_EVIDENCE = _HERE / "futures" / "published" / "evidence.json"

ORIENTATION = ("Model-built NFL projections, run in the open: the numbers, the honest "
               "backtest, and the code on my GitHub.")
PURPOSE = ("My pre-season projection of how many games each team wins, shown next to "
           "the posted win total. Certified picks are HIGH only.")
HONEST_HEADLINE = (
    "**This model does not beat the posted win total.** Across eleven held-out seasons "
    "it missed each team's actual win count by a little more than that posted number did. "
    "Every team still gets a projection. The only certified picks are HIGH."
)
PROJ_HELP = (
    "Projected regular-season wins. Ties count as half a win. The 32 projections always "
    "sum to 272, the number of games on the schedule, so no team can be raised without "
    "another coming down."
)
POSTED_HELP = (
    "The posted regular-season win total. Also an input to the ridge fit."
)
VS_HELP = (
    "Projection minus the posted win total. Sign is the over or under call."
)
CERT_HELP = (
    "Yes means HIGH: the projection disagrees with the posted number by 1 or more wins. "
    "Those are the only certified picks."
)
METHOD = """
**How the number is produced.** Each team gets fifteen structural inputs that exist
before Week 1: prior passing efficiency on offense and defense, special teams,
luck versus Pythagorean expectation, the listed starter, whether that starter is
returning, a rookie flag, unavailability, vacated target and rush share, opening
reserve-list count, a coaching change, true home games, rest, and the strength of
the coming schedule. A ridge model is trained on those fifteen plus the posted
win total, then 2026 is recentered so the 32 projections sum to 272.

**Certified picks.** Every team is projected. A pick is certified only when the
projection is at least 1 win off the posted number (HIGH). Direction is the sign
of vs posted.

**What it does not know.** 2026 opening reserve-list counts are withheld: the
public snapshot is still a 90-man camp roster, and history is post-cutdown.
Quarterback identity for a few clubs is a fallback off the opening roster, not a
week-1 depth chart.

**Honesty.** Eleven seasons. Closer than last year's wins and closer than the
retired Monte Carlo on this site. Not closer than the posted win total. The HIGH
book on this page does not clear the weekly HIGH bar.
"""


def _rg_color(ratio: float) -> str:
    ratio = max(0.0, min(1.0, float(ratio)))
    r = int(round(255 * (1 - ratio)))
    g = int(round(82 + 118 * ratio))
    return f"rgb({r},{g},82)"


@st.cache_data(ttl=3600)
def _load() -> pd.DataFrame:
    return pd.read_csv(_CSV) if _CSV.exists() else pd.DataFrame()


@st.cache_data(ttl=3600)
def _read_json(path_str: str) -> dict:
    p = Path(path_str)
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError, UnicodeDecodeError):
        return {}


def _style(view: pd.DataFrame):
    wins = pd.to_numeric(view["Proj Wins"], errors="coerce")
    lo, hi = float(wins.min()), float(wins.max())
    span = hi - lo
    certified = view["Certified"].astype(str) if "Certified" in view.columns else None

    def _apply(df: pd.DataFrame) -> pd.DataFrame:
        styles = pd.DataFrame("", index=df.index, columns=df.columns)
        if "Proj Wins" in df.columns and span > 0:
            col = df.columns.get_loc("Proj Wins")
            for row, w in enumerate(wins.to_numpy()):
                if not pd.isna(w):
                    styles.iat[row, col] = (f"color: {_rg_color((w - lo) / span)}; "
                                            f"font-weight: 700; font-size: 15px")
        if certified is not None and "Certified" in df.columns:
            col = df.columns.get_loc("Certified")
            for row, flag in enumerate(certified.to_numpy()):
                if flag == "Yes":
                    styles.iat[row, col] = "font-weight: 700; color: rgb(82,200,82)"
        return styles

    return _apply


def _render_evidence(ev: dict) -> None:
    if not ev:
        return
    st.subheader("How good is this, honestly")
    if ev.get("ladder"):
        st.markdown(
            "**On accuracy.** Average miss per team on held-out seasons, in wins. "
            "Lower is better. Row counts differ: the retired Monte Carlo skipped 2023."
        )
        ladder = pd.DataFrame(ev["ladder"]).rename(
            columns={"name": "Approach", "mae": "Average miss (wins)", "n": "n"}
        )
        st.dataframe(
            ladder,
            hide_index=True,
            width="stretch",
            column_config={
                "Average miss (wins)": st.column_config.NumberColumn(format="%.4f"),
                "n": st.column_config.NumberColumn(format="%d"),
            },
        )
    with st.container(key="jsa-metric-even-evidence"):
        c1, c2, c3 = st.columns(3)
        c1.metric(
            "Average miss, this model",
            f"{ev['mae_model']:.4f} wins",
            help=(
                f"{ev['n']} team-seasons, {ev['n_seasons']} seasons (2015-2025). "
                f"The posted win total scored {ev['mae_market']:.4f} on the same rows, "
                f"a gap of {ev['mae_model'] - ev['mae_market']:+.4f} wins."
            ),
        )
        c2.metric(
            "Average miss, posted number",
            f"{ev['mae_market']:.4f} wins",
            help=f"The null. Same {ev['n']} team-seasons.",
        )
        c3.metric(
            "Beats the posted number?",
            "No",
            help=(
                f"Closer than the posted number in {ev['seasons_model_better']} of "
                f"{ev['seasons_total']} seasons. That is not enough to claim a beat."
            ),
        )
    high = ev.get("ou_high") or {}
    if high.get("n"):
        pct = 100.0 * float(high["ats"])
        gap = float(high["gap"])
        n_fwd = int(high.get("forward_n") or 0)
        clears = bool(high.get("clears_breakeven"))
        bar = (
            "That HIGH book clears the weekly HIGH bar."
            if clears
            else "That HIGH book does not clear the weekly HIGH bar."
        )
        st.markdown(
            f"**Certified picks** are HIGH only: the projection disagrees with the "
            f"posted number by {gap:.0f} or more wins. Held-out record: "
            f"**{high['wins']}/{high['n']}**, {pct:.2f}% correct. {bar} "
            f"This year has **{n_fwd}** of those calls. Every other team is still "
            "projected. Those rows are not certified picks."
        )


def render():
    st.title("Season Win Totals")
    st.caption(ORIENTATION)
    st.markdown(f"**{PURPOSE}**")

    df = _load()
    if df.empty:
        st.info(
            "Season win-total projections are not published yet. From "
            "seasonal_totals_v2_beta run `python src/publish_site.py`."
        )
        return

    ev = _read_json(str(_EVIDENCE))
    season = int(df["season"].iloc[0])
    label = str(df["claim"].iloc[0])

    st.warning(HONEST_HEADLINE)

    certified = (
        df["certified"].fillna("").astype(str)
        if "certified" in df.columns
        else pd.Series([""] * len(df))
    )
    view = pd.DataFrame({
        "Team": df["team"],
        "Proj Wins": pd.to_numeric(df["proj_wins"], errors="coerce"),
        "Posted": pd.to_numeric(df["posted"], errors="coerce"),
        "vs posted": pd.to_numeric(df["vs_posted"], errors="coerce"),
        "Certified": certified.replace({"nan": ""}),
    })
    view = view.sort_values("Proj Wins", ascending=False).reset_index(drop=True)
    view.insert(0, "#", range(1, len(view) + 1))

    phone_view = view[["#", "Team", "Proj Wins", "Posted", "Certified"]]
    totals_cfg = {
        "#": st.column_config.NumberColumn(
            format="%d", width=50, pinned=True,
            help="Row number in this table as currently sorted.",
        ),
        "Proj Wins": st.column_config.NumberColumn(format="%.1f", help=PROJ_HELP),
        "Posted": st.column_config.NumberColumn(format="%.1f", help=POSTED_HELP),
        "vs posted": st.column_config.NumberColumn(format="%.1f", help=VS_HELP),
        "Certified": st.column_config.TextColumn(help=CERT_HELP),
    }
    totals_height = min(720, 60 + 35 * len(view))
    with st.container(key="jsa-table-desktop-season-totals"):
        st.dataframe(
            view.style.apply(_style(view), axis=None),
            hide_index=True, width="stretch", height=totals_height,
            column_config=totals_cfg,
        )
    with st.container(key="jsa-table-phone-season-totals"):
        st.dataframe(
            phone_view.style.apply(_style(view), axis=None),
            hide_index=True, width="stretch", height=totals_height,
            column_config={
                name: spec for name, spec in totals_cfg.items() if name in phone_view.columns
            },
        )

    n_cert = int((view["Certified"] == "Yes").sum())
    st.caption(
        f"{season} regular season · {len(view)} teams · projections sum to "
        f"{view['Proj Wins'].sum():.0f} wins, the exact number of games scheduled. "
        f"{n_cert} certified HIGH picks. Color on the projection column encodes "
        "magnitude only."
    )

    _render_evidence(ev)

    with st.expander("How this is built, and what it does not know"):
        st.markdown(METHOD)

    stamp = str(df["generated_at"].iloc[0])[:19].replace("T", " ")
    st.caption(
        f"**{label}** · line_in · generated {stamp} UTC. Built in "
        "seasonal_totals_v2_beta. The old Monte Carlo pipeline is archived."
    )
