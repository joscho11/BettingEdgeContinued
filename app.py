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
st.sidebar.markdown("""
    <div style='text-align: center; padding: 10px;'>
        <span style='font-size: 48px;'>🏈</span>
        <h2 style='color: #013369; margin: 0;'>Betting Edge</h2>
        <p style='color: #D50A0A; font-size: 12px; margin: 0;'>NFL Analytics</p>
    </div>
""", unsafe_allow_html=True)
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
        "🏈 **NFL Offseason**: The 2025 season has concluded. "
        "Predictions will return when the 2026 season kicks off in September. "
        "Browse past predictions using the sidebar."
    )

# ── Filter to selected week ───────────────────────────────────────────────────
week_df    = df[(df['season'] == season) & (df['week'] == week)].copy()
results_in = week_df['actual_margin'].notna().any()

# ── Header ────────────────────────────────────────────────────────────────────
st.title(f"🏈 Week {week} Predictions: {season} Season")

if results_in:
    correct = int(week_df['model_correct'].sum())
    total   = len(week_df)
    st.success(
        f"Results are in! Week {week} ATS record: "
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

        def fmt(val):
            return f"{val:+.1f}"

        # spread_line is home-relative: negative = home favored, positive = away favored
        home_is_favored = spread > 0

        # Always show favored team on top with negative spread, underdog on bottom with positive
        if home_is_favored:
            top_team      = home
            bot_team      = away
            top_spread    = fmt(-spread)   # home favored = positive spread in your data, flip to negative
            bot_spread    = fmt(spread)    # away gets positive
            top_predicted = fmt(predicted)
            bot_predicted = fmt(-predicted)
        else:
            top_team      = away
            bot_team      = home
            top_spread    = fmt(spread)    # -13.5 → DEN gets negative ✅
            bot_spread    = fmt(-spread)   # +13.5 → KC gets positive ✅
            top_predicted = fmt(-predicted)
            bot_predicted = fmt(predicted)

        # Who does the model recommend?
        # edge > 0 means model thinks home covers
        if edge > 0:
            rec_team  = home
            rec_color = "#00c853"
        elif edge < 0:
            rec_team  = away
            rec_color = "#2979ff"
        else:
            rec_team  = None
            rec_color = "#888888"

        top_is_rec = rec_team == top_team
        bot_is_rec = rec_team == bot_team

        # Results
        results_available = results_in and pd.notna(row['actual_margin'])
        correct           = (row['model_correct'] == 1) if results_in else False
        actual            = row['actual_margin'] if results_available else None

        if results_available:
            home_score = row.get('home_score', None)
            away_score = row.get('away_score', None)
            has_scores = pd.notna(home_score) and pd.notna(away_score)
            if has_scores:
                top_score = f"{int(home_score)}" if home_is_favored else f"{int(away_score)}"
                bot_score = f"{int(away_score)}" if home_is_favored else f"{int(home_score)}"
            else:
                # Fall back to margin from each team's perspective
                top_score = fmt(actual if home_is_favored else -actual)
                bot_score = fmt(-actual if home_is_favored else actual)
        else:
            top_score = "—"
            bot_score = "—"

        status_icon  = "🟢" if results_in and correct else ("🔴" if results_in else "⭐")
        result_label = ("✅ WIN" if correct else "❌ LOSS") if results_in else ""

        # Styling helpers
        def name_style(is_rec):
            weight = "700" if is_rec else "400"
            color  = "white" if is_rec else "#aaa"
            return weight, color

        def stat_box(val, is_rec=False, is_result=False):
            if is_result and results_available:
                bg    = "#1a2a1a" if correct else "#2a1a1a"
                color = "#00c853" if correct else "#ff5252"
            elif is_rec:
                bg    = "#1e3a2a" if rec_color == "#00c853" else "#1a2040"
                color = rec_color
            else:
                bg    = "#1e2a3a"
                color = "white"
            return (
                f"<div style='text-align:center;background:{bg};border-radius:6px;"
                f"padding:6px 0;font-size:14px;font-weight:600;color:{color};"
                f"height:32px;line-height:20px'>{val}</div>"
            )

        def bet_box(team):
            return (
                f"<div style='background:{rec_color}22;border-left:3px solid {rec_color};"
                f"border-radius:4px;padding:0 8px;font-size:12px;font-weight:700;"
                f"color:{rec_color};text-align:center;height:32px;line-height:32px'>"
                f"BET {team}</div>"
            )

        def empty_box():
            return "<div style='height:32px'></div>"

        with st.container():

            # ── Header ───────────────────────────────────────────────────
            st.markdown(
                f"<div style='font-size:13px;color:#888;margin-bottom:6px'>"
                f"{status_icon}&nbsp;&nbsp;"
                f"<b style='color:#ccc'>{away} @ {home}</b>"
                f"&nbsp;&nbsp;·&nbsp;&nbsp;{row['gameday']}"
                f"{'&nbsp;&nbsp;·&nbsp;&nbsp;<b>' + result_label + '</b>' if result_label else ''}"
                f"</div>",
                unsafe_allow_html=True
            )

            # ── Column headers ────────────────────────────────────────────
            if results_available:
                h0, h1, h2, h3, h4 = st.columns([2.2, 1.2, 1.2, 1.2, 1.8])
                h3.markdown("<div style='text-align:center;font-size:11px;color:#aaa;letter-spacing:1px'>SCORE</div>", unsafe_allow_html=True)
            else:
                h0, h1, h2, h4 = st.columns([2.2, 1.2, 1.2, 1.8])

            h1.markdown("<div style='text-align:center;font-size:11px;color:#aaa;letter-spacing:1px'>SPREAD</div>",    unsafe_allow_html=True)
            h2.markdown("<div style='text-align:center;font-size:11px;color:#aaa;letter-spacing:1px'>PREDICTED</div>", unsafe_allow_html=True)

            # ── Top row (favored) ─────────────────────────────────────────
            if results_available:
                a0, a1, a2, a3, a4 = st.columns([2.2, 1.2, 1.2, 1.2, 1.8])
                a3.markdown(stat_box(top_score, is_result=True), unsafe_allow_html=True)
            else:
                a0, a1, a2, a4 = st.columns([2.2, 1.2, 1.2, 1.8])

            top_w, top_c = name_style(top_is_rec)
            a0.markdown(
                f"<div style='font-weight:{top_w};font-size:15px;color:{top_c};"
                f"padding-top:6px;height:32px'>{top_team}</div>",
                unsafe_allow_html=True
            )
            a1.markdown(stat_box(top_spread),                      unsafe_allow_html=True)
            a2.markdown(stat_box(top_predicted, is_rec=top_is_rec), unsafe_allow_html=True)
            a4.markdown(bet_box(top_team) if top_is_rec else empty_box(), unsafe_allow_html=True)

            st.markdown("<div style='height:6px'></div>", unsafe_allow_html=True)

            # ── Bottom row (underdog) ─────────────────────────────────────
            if results_available:
                b0, b1, b2, b3, b4 = st.columns([2.2, 1.2, 1.2, 1.2, 1.8])
                b3.markdown(stat_box(bot_score, is_result=True), unsafe_allow_html=True)
            else:
                b0, b1, b2, b4 = st.columns([2.2, 1.2, 1.2, 1.8])

            bot_w, bot_c = name_style(bot_is_rec)
            b0.markdown(
                f"<div style='font-weight:{bot_w};font-size:15px;color:{bot_c};"
                f"padding-top:6px;height:32px'>{bot_team}</div>",
                unsafe_allow_html=True
            )
            b1.markdown(stat_box(bot_spread),                      unsafe_allow_html=True)
            b2.markdown(stat_box(bot_predicted, is_rec=bot_is_rec), unsafe_allow_html=True)
            b4.markdown(bet_box(bot_team) if bot_is_rec else empty_box(), unsafe_allow_html=True)

            st.divider()
