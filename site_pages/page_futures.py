"""Season Win Totals page.

READ-ONLY over files published from seasonal_totals_v2_beta:

    futures/published/season_totals_2026.csv
    futures/published/evidence.json

No model runs here. pandas and streamlit only.

The page is a projection table plus an accuracy ladder. Copy is checked
against futures/language_fence.py.
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
           "the posted win total.")
HONEST_HEADLINE = (
    "**This model does not beat the posted win total.** Across eleven held-out seasons "
    "it missed each team's actual win count by a little more than that posted number did. "
    "It is on this page because it is the closest mix I have, not because it is sharper "
    "than the number the books opened."
)
PROJ_HELP = (
    "Projected regular-season wins. Ties count as half a win. The 32 projections always "
    "sum to 272, the number of games on the schedule, so no team can be raised without "
    "another coming down."
)
POSTED_HELP = (
    "The posted regular-season win total used as the intercept of the projection."
)
VS_HELP = (
    "Projection minus the posted win total. Zero means they match."
)
METHOD = """
**How the number is produced.** Each team gets fifteen structural inputs that exist
before Week 1: prior passing efficiency on offense and defense, special teams,
luck versus Pythagorean expectation, the listed starter, whether that starter is
returning, a rookie flag, unavailability, vacated target and rush share, opening
reserve-list count, a coaching change, true home games, rest, and the strength of
the coming schedule. A ridge model is trained on leftover versus the posted win
total, then mixed back in with a weight chosen on the last inner season. The
posted number is the intercept. It is not a column in the fit.

**What 2026 did.** That inner mix weight came back at zero, so this year's
projection sits on the posted number for every team.

**What it does not know.** 2026 opening reserve-list counts are withheld: the
public snapshot is still a 90-man camp roster, and history is post-cutdown.
Quarterback identity for a few clubs is a fallback off the opening roster, not a
week-1 depth chart.

**Honesty.** Eleven seasons. The mix was closer than last year's
wins and closer than the retired Monte Carlo on this site. It was not closer than
the posted win total.
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

    def _apply(df: pd.DataFrame) -> pd.DataFrame:
        styles = pd.DataFrame("", index=df.index, columns=df.columns)
        if "Proj Wins" in df.columns and span > 0:
            col = df.columns.get_loc("Proj Wins")
            for row, w in enumerate(wins.to_numpy()):
                if not pd.isna(w):
                    styles.iat[row, col] = (f"color: {_rg_color((w - lo) / span)}; "
                                            f"font-weight: 700; font-size: 15px")
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
        st.markdown(
            f"**High confidence bets** are leftover-mix calls where the projection "
            f"disagrees with the posted number by {gap:.0f} or more wins, in the "
            f"direction of the mix. Held-out record: **{high['wins']}/{high['n']}**, "
            f"{pct:.2f}% correct. This year's mix weight is 0, so 2026 has none "
            "of those calls."
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

    view = pd.DataFrame({
        "Team": df["team"],
        "Proj Wins": pd.to_numeric(df["proj_wins"], errors="coerce"),
        "Posted": pd.to_numeric(df["posted"], errors="coerce"),
        "vs posted": pd.to_numeric(df["vs_posted"], errors="coerce"),
    })
    view = view.sort_values("Proj Wins", ascending=False).reset_index(drop=True)
    view.insert(0, "#", range(1, len(view) + 1))

    phone_view = view[["#", "Team", "Proj Wins", "Posted"]]
    totals_cfg = {
        "#": st.column_config.NumberColumn(
            format="%d", width=50, pinned=True,
            help="Row number in this table as currently sorted.",
        ),
        "Proj Wins": st.column_config.NumberColumn(format="%.1f", help=PROJ_HELP),
        "Posted": st.column_config.NumberColumn(format="%.1f", help=POSTED_HELP),
        "vs posted": st.column_config.NumberColumn(format="%.1f", help=VS_HELP),
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

    lam = float(df["lambda"].iloc[0]) if "lambda" in df.columns else float("nan")
    st.caption(
        f"{season} regular season · {len(view)} teams · projections sum to "
        f"{view['Proj Wins'].sum():.0f} wins, the exact number of games scheduled. "
        f"This year's mix weight is {lam:.2f}, so vs posted is the recenter only. "
        "Color on the projection column encodes magnitude only."
    )

    _render_evidence(ev)

    with st.expander("How this is built, and what it does not know"):
        st.markdown(METHOD)

    stamp = str(df["generated_at"].iloc[0])[:19].replace("T", " ")
    st.caption(
        f"**{label}** · leftover mix · generated {stamp} UTC. Built in "
        "seasonal_totals_v2_beta. The old Monte Carlo pipeline is archived."
    )
