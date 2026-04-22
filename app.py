import streamlit as st
import pandas as pd
import plotly.express as px
from datetime import datetime as dt
import plotly.graph_objects as go

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="BettingEdge | NFL Predictions",
    page_icon="🏈",
    layout="wide"
)

# ── Load data ─────────────────────────────────────────────────────────────────
@st.cache_data(ttl=3600)   # cache for 1 hour so it doesn't reload on every click
def load_tracker():
    return pd.read_csv('predictions_tracker.csv')

df = load_tracker()

# ── Sidebar ───────────────────────────────────────────────────────────────────
st.sidebar.image("https://upload.wikimedia.org/wikipedia/en/a/a2/National_Football_League_logo.svg", width=80)
st.sidebar.title("BettingEdge")
st.sidebar.caption("XGBoost ATS Predictor")

seasons  = sorted(df['season'].unique(), reverse=True)
season   = st.sidebar.selectbox("Season", seasons)

weeks    = sorted(df[df['season'] == season]['week'].unique(), reverse=True)
week     = st.sidebar.selectbox("Week", weeks)

st.sidebar.divider()
edge_threshold = st.sidebar.slider(
    "Min Edge (pts)",
    min_value=0.0,
    max_value=5.0,
    value=1.0,
    step=0.5,
    help="Only show games where model disagrees with spread by this many points"
)

# ── Filter to selected week ───────────────────────────────────────────────────
week_df = df[
    (df['season'] == season) &
    (df['week']   == week)
].copy()

# Results available yet?
results_in = week_df['actual_margin'].notna().any()

# ── Header ────────────────────────────────────────────────────────────────────
# ── Determine what to show ────────────────────────────────────────────────────
df = load_tracker()

# Most recent week that has any data
latest_season = int(df['season'].max())
latest_week   = int(df[df['season'] == latest_season]['week'].max())

# Is the season currently active?
# NFL regular season runs Sept - Jan
now = dt.now()
season_active = (now.month >= 9) or (now.month <= 2)

# ── Sidebar ───────────────────────────────────────────────────────────────────
'''st.sidebar.title("🏈 BettingEdge")

seasons = sorted(df['season'].unique(), reverse=True)
season  = st.sidebar.selectbox("Season", seasons)
weeks   = sorted(df[df['season'] == season]['week'].unique(), reverse=True)
week    = st.sidebar.selectbox("Week", weeks)

edge_threshold = st.sidebar.slider("Min Edge (pts)", 0.0, 5.0, 1.0, 0.5)'''

# ── Offseason banner ──────────────────────────────────────────────────────────
if not season_active:
    st.info(
        "🏈 **NFL Offseason** — The 2025 season has concluded. "
        "Predictions will return when the 2026 season kicks off in September. "
        "Browse past predictions using the sidebar."
    )

# ── Game cards ────────────────────────────────────────────────────────────────
st.subheader("Game Predictions")

# Sort by edge strength
week_df = week_df.sort_values('model_edge', key=abs, ascending=False)

for _, row in week_df.iterrows():
    edge_abs   = abs(row['model_edge'])
    is_flagged = edge_abs >= edge_threshold

    # Card color
    if results_in:
        if row['model_correct'] == 1:
            border = "🟢"
        else:
            border = "🔴"
    elif is_flagged:
        border = "⭐"
    else:
        border = "⚪"

    # Recommendation label
    if row['model_edge'] > 0:
        rec = f"BET HOME ({row['home_team']})"
        rec_color = "#00c853"
    elif row['model_edge'] < 0:
        rec = f"BET AWAY ({row['away_team']})"
        rec_color = "#2979ff"
    else:
        rec = "PASS"
        rec_color = "#888"

    with st.container():
        c1, c2, c3, c4, c5 = st.columns([2, 1.5, 1.5, 1.5, 2])

        c1.markdown(f"**{border} {row['away_team']} @ {row['home_team']}**")
        c1.caption(str(row['gameday']))

        spread_str = f"{row['spread_line']:+.1f}" if pd.notna(row['spread_line']) else "N/A"
        c2.metric("Spread",     spread_str)
        c3.metric("Predicted",  f"{row['predicted_margin']:+.1f}")
        c4.metric("Model Edge", f"{row['model_edge']:+.1f}")

        if results_in and pd.notna(row['actual_margin']):
            c5.metric("Actual",
                      f"{row['actual_margin']:+.1f}",
                      delta="✅ Correct" if row['model_correct'] == 1 else "❌ Wrong",
                      delta_color="normal" if row['model_correct'] == 1 else "inverse")
        else:
            c5.markdown(
                f"<div style='background-color:{rec_color}22; "
                f"border-left: 4px solid {rec_color}; "
                f"padding: 8px; border-radius: 4px; margin-top: 8px;'>"
                f"<b style='color:{rec_color}'>{rec}</b></div>",
                unsafe_allow_html=True
            )

        st.divider()

        # Show freshness indicator
if not week_df.empty and 'mode' in week_df.columns:
    mode      = week_df['mode'].iloc[-1]
    logged_at = week_df['logged_at'].iloc[-1]

    mode_labels = {
        'monday':   ('🟡', 'Early Lines',          'Updated Monday with initial lines'),
        'thursday': ('🟠', 'Injury Reports In',    'Updated Thursday with injury data'),
        'sunday':   ('🟢', 'Final Predictions',    'Final update — games starting soon'),
    }

    icon, label, desc = mode_labels.get(mode, ('⚪', 'Unknown', ''))

    st.info(f"{icon} **{label}** — {desc}  \n"
            f"Last updated: {logged_at}")