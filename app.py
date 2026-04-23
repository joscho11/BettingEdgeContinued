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
        spread    = row['spread_line']
        predicted = row['predicted_margin']
        edge      = row['model_edge']

        # Flip to per-team perspective
        home_spread    = f"{spread:+.1f}"
        away_spread    = f"{-spread:+.1f}"
        home_predicted = f"{predicted:+.1f}"
        away_predicted = f"{-predicted:+.1f}"
        home_edge      = f"{edge:+.1f}"
        away_edge      = f"{-edge:+.1f}"

        # Who does the model recommend?
        if edge > 0:
            rec_team  = home
            rec_color = "#00c853"
        elif edge < 0:
            rec_team  = away
            rec_color = "#2979ff"
        else:
            rec_team  = None
            rec_color = "#888888"

        # Results (safe — always defined)
        results_available = results_in and pd.notna(row['actual_margin'])
        correct           = (row['model_correct'] == 1) if results_in else False
        actual            = row['actual_margin'] if results_available else None
        home_actual       = f"{actual:+.1f}"  if results_available else "—"
        away_actual       = f"{-actual:+.1f}" if results_available else "—"

        # Status icon and label
        status_icon  = "🟢" if results_in and correct else ("🔴" if results_in else ("⭐" if abs(edge) >= edge_threshold else "⚪"))
        result_label = ("✅ Correct" if correct else "❌ Wrong") if results_in else ""

        # Highlight vars
        away_highlight = rec_team == away
        home_highlight = rec_team == home
        away_weight    = "700" if away_highlight else "400"
        home_weight    = "700" if home_highlight else "400"
        away_color     = "white" if away_highlight else "#ccc"
        home_color     = "white" if home_highlight else "#ccc"

        with st.container():

            # ── Game date / status header ─────────────────────────────────
            st.markdown(
                f"<div style='font-size:13px;color:#888;margin-bottom:6px'>"
                f"{status_icon}&nbsp;&nbsp;<b style='color:#ccc'>{away} @ {home}</b>"
                f"&nbsp;&nbsp;·&nbsp;&nbsp;{row['gameday']}"
                f"{'&nbsp;&nbsp;·&nbsp;&nbsp;' + result_label if result_label else ''}"
                f"</div>",
                unsafe_allow_html=True
            )

            # ── Column headers ────────────────────────────────────────────
            num_cols = 5 if results_available else 4
            if results_available:
                h0, h1, h2, h3, h4, h5 = st.columns([2.5, 1.2, 1.2, 1.2, 1.2, 1.5])
                h4.markdown("<div style='text-align:center;font-size:11px;color:#aaa;letter-spacing:1px'>ACTUAL</div>", unsafe_allow_html=True)
            else:
                h0, h1, h2, h3, h5 = st.columns([2.5, 1.2, 1.2, 1.2, 1.5])

            h1.markdown("<div style='text-align:center;font-size:11px;color:#aaa;letter-spacing:1px'>SPREAD</div>",    unsafe_allow_html=True)
            h2.markdown("<div style='text-align:center;font-size:11px;color:#aaa;letter-spacing:1px'>PREDICTED</div>", unsafe_allow_html=True)
            h3.markdown("<div style='text-align:center;font-size:11px;color:#aaa;letter-spacing:1px'>EDGE</div>",      unsafe_allow_html=True)

            def stat_box(val, highlight=False):
                bg    = "#1e3a2a" if highlight else "#1e2a3a"
                color = "#00c853" if highlight and rec_color == "#00c853" else ("#2979ff" if highlight else "white")
                return (
                    f"<div style='text-align:center;background:{bg};border-radius:6px;"
                    f"padding:5px 0;font-size:14px;font-weight:600;color:{color}'>{val}</div>"
                )

            # ── Away team row ─────────────────────────────────────────────
            if results_available:
                a0, a1, a2, a3, a4, a5 = st.columns([2.5, 1.2, 1.2, 1.2, 1.2, 1.5])
                a4.markdown(stat_box(away_actual), unsafe_allow_html=True)
            else:
                a0, a1, a2, a3, a5 = st.columns([2.5, 1.2, 1.2, 1.2, 1.5])

            a0.markdown(
                f"<div style='font-weight:{away_weight};font-size:15px;color:{away_color};padding-top:4px'>{away}</div>",
                unsafe_allow_html=True
            )
            a1.markdown(stat_box(away_spread),    unsafe_allow_html=True)
            a2.markdown(stat_box(away_predicted), unsafe_allow_html=True)
            a3.markdown(stat_box(away_edge, highlight=away_highlight), unsafe_allow_html=True)

            if away_highlight:
                a5.markdown(
                    f"<div style='background:{rec_color}22;border-left:3px solid {rec_color};"
                    f"padding:5px 8px;border-radius:4px;font-size:12px;font-weight:700;"
                    f"color:{rec_color};text-align:center'>BET<br>{away}</div>",
                    unsafe_allow_html=True
                )

            st.markdown("<div style='height:5px'></div>", unsafe_allow_html=True)

            # ── Home team row ─────────────────────────────────────────────
            if results_available:
                b0, b1, b2, b3, b4, b5 = st.columns([2.5, 1.2, 1.2, 1.2, 1.2, 1.5])
                b4.markdown(stat_box(home_actual), unsafe_allow_html=True)
            else:
                b0, b1, b2, b3, b5 = st.columns([2.5, 1.2, 1.2, 1.2, 1.5])

            b0.markdown(
                f"<div style='font-weight:{home_weight};font-size:15px;color:{home_color};padding-top:4px'>{home}</div>",
                unsafe_allow_html=True
            )
            b1.markdown(stat_box(home_spread),    unsafe_allow_html=True)
            b2.markdown(stat_box(home_predicted), unsafe_allow_html=True)
            b3.markdown(stat_box(home_edge, highlight=home_highlight), unsafe_allow_html=True)

            if home_highlight:
                b5.markdown(
                    f"<div style='background:{rec_color}22;border-left:3px solid {rec_color};"
                    f"padding:5px 8px;border-radius:4px;font-size:12px;font-weight:700;"
                    f"color:{rec_color};text-align:center'>BET<br>{home}</div>",
                    unsafe_allow_html=True
                )

            st.divider()
