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
        if row['model_edge'] > 0:
            rec       = f"BET HOME ({row['home_team']})"
            rec_color = "#00c853"
        elif row['model_edge'] < 0:
            rec       = f"BET AWAY ({row['away_team']})"
            rec_color = "#2979ff"
        else:
            rec       = "PASS"
            rec_color = "#888888"

        if results_in:
            border = "🟢" if row['model_correct'] == 1 else "🔴"
        else:
            border = "⭐"

        with st.container():
            c1, c2, c3, c4, c5 = st.columns([2, 1.5, 1.5, 1.5, 2])

            c1.markdown(f"**{border} {row['away_team']} @ {row['home_team']}**")
            c1.caption(str(row['gameday']))

            spread_str = f"{row['spread_line']:+.1f}" if pd.notna(row['spread_line']) else "N/A"
            c2.metric("Spread",     spread_str)
            c3.metric("Predicted",  f"{row['predicted_margin']:+.1f}")
            c4.metric("Model Edge", f"{row['model_edge']:+.1f}")

            if results_in and pd.notna(row['actual_margin']):
                c5.metric(
                    "Actual",
                    f"{row['actual_margin']:+.1f}",
                    delta="✅ Correct" if row['model_correct'] == 1 else "❌ Wrong",
                    delta_color="normal" if row['model_correct'] == 1 else "inverse"
                )
            else:
                c5.markdown(
                    f"<div style='background-color:{rec_color}22;"
                    f"border-left:4px solid {rec_color};"
                    f"padding:8px;border-radius:4px;margin-top:8px;'>"
                    f"<b style='color:{rec_color}'>{rec}</b></div>",
                    unsafe_allow_html=True
                )

            st.divider()