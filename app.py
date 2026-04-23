import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime as dt

st.set_page_config(
    page_title="BettingEdge | NFL Predictions",
    page_icon="🏈",
    layout="wide"
)

@st.cache_data(ttl=3600)
def load_tracker():
    df = pd.read_csv('predictions_tracker.csv')
    df['season'] = df['season'].astype(int)
    df['week']   = df['week'].astype(int)
    return df

df = load_tracker()

# ── Sidebar ───────────────────────────────────────────────────────────────────
st.sidebar.image(
    "https://upload.wikimedia.org/wikipedia/en/a/a2/National_Football_League_logo.svg",
    width=80
)
st.sidebar.title("BettingEdge")
st.sidebar.caption("XGBoost ATS Predictor")
st.sidebar.divider()

seasons = sorted(df['season'].unique(), reverse=True)
season  = st.sidebar.selectbox("Season", seasons, key="season_select")

weeks   = sorted(df[df['season'] == season]['week'].unique(), reverse=True)
week    = st.sidebar.selectbox("Week", weeks, key="week_select")

edge_threshold = st.sidebar.slider(
    "Min Edge (pts)",
    min_value=0.0,
    max_value=5.0,
    value=1.0,
    step=0.5,
    key="edge_slider",
    help="Only show games where model disagrees with spread by at least this many points"
)

# ── Offseason banner ──────────────────────────────────────────────────────────
now = dt.now()
season_active = (now.month >= 9) or (now.month <= 2)

if not season_active:
    st.info(
        "🏈 **NFL Offseason** — The 2025 season has concluded. "
        "Predictions will return when the 2026 season kicks off in September. "
        "Browse past predictions using the sidebar."
    )

# ── Filter to selected week ───────────────────────────────────────────────────
week_df    = df[(df['season'] == season) & (df['week'] == week)].copy()
results_in = week_df['actual_margin'].notna().any()

# ── Header ────────────────────────────────────────────────────────────────────
st.title(f"🏈 Week {week} Predictions — {season} Season")

if results_in:
    correct = int(week_df['model_correct'].sum())
    total   = len(week_df)
    st.success(
        f"Results are in — Week {week} ATS record: "
        f"**{correct}-{total - correct}** ({correct/total*100:.0f}%)"
    )
else:
    st.info("Games not yet played. Check back after the week's results are in.")

# ── Freshness indicator ───────────────────────────────────────────────────────
if not week_df.empty and 'mode' in week_df.columns:
    mode      = week_df['mode'].iloc[-1]
    logged_at = week_df['logged_at'].iloc[-1]
    mode_labels = {
        'monday':   ('🟡', 'Early Lines',       'Updated Monday with initial lines'),
        'thursday': ('🟠', 'Injury Reports In', 'Updated Thursday with injury data'),
        'sunday':   ('🟢', 'Final Predictions', 'Final update — games starting soon'),
        'backfill': ('🔵', 'Backfilled',        'Historical predictions'),
    }
    icon, label, desc = mode_labels.get(mode, ('⚪', 'Manual Run', ''))
    st.caption(f"{icon} **{label}** — {desc} · Last updated: {logged_at}")

# ── Summary metrics ───────────────────────────────────────────────────────────
st.divider()
col1, col2, col3, col4 = st.columns(4)

# Apply edge filter — this is what gets displayed in cards
filtered_df = week_df[week_df['model_edge'].abs() >= edge_threshold].copy()
hidden_count = len(week_df) - len(filtered_df)

col1.metric("Total Games",  len(week_df))
col2.metric("Showing",      len(filtered_df),
            help=f"Games with |edge| ≥ {edge_threshold} pts")
col3.metric("Avg Edge",     f"{week_df['model_edge'].abs().mean():.1f} pts")

if results_in and len(filtered_df) > 0:
    sc = int(filtered_df['model_correct'].sum())
    col4.metric("ATS Record",
                f"{sc}/{len(filtered_df)} ({sc/len(filtered_df)*100:.0f}%)")
else:
    col4.metric("ATS Record", "Pending")

st.divider()

# ── Game cards ────────────────────────────────────────────────────────────────
st.subheader("Game Predictions")

if week_df.empty:
    st.warning("No predictions found for this week.")

elif filtered_df.empty:
    st.warning(
        f"No games meet the current edge threshold of ±{edge_threshold} pts. "
        f"Lower the slider to see all {len(week_df)} games."
    )

else:
    if hidden_count > 0:
        st.caption(
            f"Showing {len(filtered_df)} of {len(week_df)} games "
            f"— {hidden_count} filtered out (edge < {edge_threshold} pts). "
            f"Lower the slider to see all games."
        )

    filtered_df = filtered_df.sort_values('model_edge', key=abs, ascending=False)

    for _, row in filtered_df.iterrows():
        home      = row['home_team']
        away      = row['away_team']
        spread    = row['spread_line']   # negative = home favored, positive = away favored
        predicted = row['predicted_margin']
        edge      = row['model_edge']

        # spread_line is home-team-relative in your data.
        # For display: favored team is negative, underdog is positive.
        # If spread < 0, home is favored → home gets the negative number as-is,
        #                                   away gets the positive (flipped)
        # If spread > 0, away is favored → away gets the positive as-is,
        #                                   home gets the negative (flipped)
        # Either way: home_spread = spread, away_spread = -spread
        # But we want FAVORED shown first (negative on top), so we sort the rows.
        home_spread    = spread
        away_spread    = -spread

        # Same logic for predicted and edge — keep home-relative,
        # display will handle positive/negative correctly per team
        home_predicted = predicted
        away_predicted = -predicted
        home_edge_val  = edge
        away_edge_val  = -edge

        # Format for display
        def fmt(val):
            return f"{val:+.1f}"

        # Who is favored? (negative spread = favored)
        home_is_favored = spread < 0
        favored_team    = home if home_is_favored else away
        underdog_team   = away if home_is_favored else home

        # Display order: favored on top, underdog on bottom
        if home_is_favored:
            top_team       = home
            bot_team       = away
            top_spread     = fmt(home_spread)
            bot_spread     = fmt(away_spread)
            top_predicted  = fmt(home_predicted)
            bot_predicted  = fmt(away_predicted)
            top_edge_val   = home_edge_val
            bot_edge_val   = away_edge_val
        else:
            top_team       = away
            bot_team       = home
            top_spread     = fmt(away_spread)
            bot_spread     = fmt(home_spread)
            top_predicted  = fmt(away_predicted)
            bot_predicted  = fmt(home_predicted)
            top_edge_val   = away_edge_val
            bot_edge_val   = home_edge_val

        top_edge  = fmt(top_edge_val)
        bot_edge  = fmt(bot_edge_val)

        # Who does the model recommend?
        # Positive home_edge = model thinks home covers = home is the bet
        if edge > 0:
            rec_team  = home
            rec_color = "#00c853"
        elif edge < 0:
            rec_team  = away
            rec_color = "#2979ff"
        else:
            rec_team  = None
            rec_color = "#888888"

        # Results
        results_available = results_in and pd.notna(row['actual_margin'])
        correct           = (row['model_correct'] == 1) if results_in else False
        actual            = row['actual_margin'] if results_available else None

        if results_available:
            # actual_margin is home-relative (positive = home won by that amount)
            home_actual_val = actual
            away_actual_val = -actual
            if home_is_favored:
                top_actual = f"{home_actual_val:+.1f}"
                bot_actual = f"{away_actual_val:+.1f}"
            else:
                top_actual = f"{away_actual_val:+.1f}"
                bot_actual = f"{home_actual_val:+.1f}"
        else:
            top_actual = "—"
            bot_actual = "—"

        # Status
        status_icon  = "🟢" if results_in and correct else ("🔴" if results_in else "⭐")
        result_label = ("✅ Correct" if correct else "❌ Wrong") if results_in else ""

        def stat_box(val, highlight=False):
            bg    = "#1e3a2a" if highlight and rec_color == "#00c853" else ("#1e2a3a4a" if highlight else "#1e2a3a")
            color = rec_color if highlight else "white"
            return (
                f"<div style='text-align:center;background:{bg};border-radius:6px;"
                f"padding:5px 0;font-size:14px;font-weight:600;color:{color}'>{val}</div>"
            )

        top_highlight = rec_team == top_team
        bot_highlight = rec_team == bot_team
        top_weight    = "700" if top_highlight else "400"
        bot_weight    = "700" if bot_highlight else "400"
        top_color     = "white" if top_highlight else "#ccc"
        bot_color     = "white" if bot_highlight else "#ccc"

        with st.container():

            # ── Header ───────────────────────────────────────────────────
            st.markdown(
                f"<div style='font-size:13px;color:#888;margin-bottom:6px'>"
                f"{status_icon}&nbsp;&nbsp;"
                f"<b style='color:#ccc'>{away} @ {home}</b>"
                f"&nbsp;&nbsp;·&nbsp;&nbsp;{row['gameday']}"
                f"{'&nbsp;&nbsp;·&nbsp;&nbsp;' + result_label if result_label else ''}"
                f"</div>",
                unsafe_allow_html=True
            )

            # ── Column headers ────────────────────────────────────────────
            if results_available:
                h0, h1, h2, h3, h4, h5 = st.columns([2.5, 1.2, 1.2, 1.2, 1.2, 1.5])
                h4.markdown("<div style='text-align:center;font-size:11px;color:#aaa;letter-spacing:1px'>ACTUAL</div>", unsafe_allow_html=True)
            else:
                h0, h1, h2, h3, h5 = st.columns([2.5, 1.2, 1.2, 1.2, 1.5])

            h1.markdown("<div style='text-align:center;font-size:11px;color:#aaa;letter-spacing:1px'>SPREAD</div>",    unsafe_allow_html=True)
            h2.markdown("<div style='text-align:center;font-size:11px;color:#aaa;letter-spacing:1px'>PREDICTED</div>", unsafe_allow_html=True)
            h3.markdown("<div style='text-align:center;font-size:11px;color:#aaa;letter-spacing:1px'>EDGE</div>",      unsafe_allow_html=True)

            # ── Top row (favored team) ────────────────────────────────────
            if results_available:
                a0, a1, a2, a3, a4, a5 = st.columns([2.5, 1.2, 1.2, 1.2, 1.2, 1.5])
                a4.markdown(stat_box(top_actual), unsafe_allow_html=True)
            else:
                a0, a1, a2, a3, a5 = st.columns([2.5, 1.2, 1.2, 1.2, 1.5])

            a0.markdown(
                f"<div style='font-weight:{top_weight};font-size:15px;color:{top_color};padding-top:4px'>{top_team}</div>",
                unsafe_allow_html=True
            )
            a1.markdown(stat_box(top_spread),              unsafe_allow_html=True)
            a2.markdown(stat_box(top_predicted),           unsafe_allow_html=True)
            a3.markdown(stat_box(top_edge, top_highlight), unsafe_allow_html=True)

            if top_highlight:
                a5.markdown(
                    f"<div style='background:{rec_color}22;border-left:3px solid {rec_color};"
                    f"padding:5px 8px;border-radius:4px;font-size:12px;font-weight:700;"
                    f"color:{rec_color};text-align:center'>BET<br>{top_team}</div>",
                    unsafe_allow_html=True
                )

            st.markdown("<div style='height:5px'></div>", unsafe_allow_html=True)

            # ── Bottom row (underdog) ─────────────────────────────────────
            if results_available:
                b0, b1, b2, b3, b4, b5 = st.columns([2.5, 1.2, 1.2, 1.2, 1.2, 1.5])
                b4.markdown(stat_box(bot_actual), unsafe_allow_html=True)
            else:
                b0, b1, b2, b3, b5 = st.columns([2.5, 1.2, 1.2, 1.2, 1.5])

            b0.markdown(
                f"<div style='font-weight:{bot_weight};font-size:15px;color:{bot_color};padding-top:4px'>{bot_team}</div>",
                unsafe_allow_html=True
            )
            b1.markdown(stat_box(bot_spread),              unsafe_allow_html=True)
            b2.markdown(stat_box(bot_predicted),           unsafe_allow_html=True)
            b3.markdown(stat_box(bot_edge, bot_highlight), unsafe_allow_html=True)

            if bot_highlight:
                b5.markdown(
                    f"<div style='background:{rec_color}22;border-left:3px solid {rec_color};"
                    f"padding:5px 8px;border-radius:4px;font-size:12px;font-weight:700;"
                    f"color:{rec_color};text-align:center'>BET<br>{bot_team}</div>",
                    unsafe_allow_html=True
                )

            st.divider()
