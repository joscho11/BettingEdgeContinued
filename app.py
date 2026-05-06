import streamlit as st
import pandas as pd
import json
import os
from datetime import datetime as dt
import glob
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(
    page_title="BettingEdge | NFL Predictions",
    page_icon="🏈",
    layout="wide"
)

st.markdown("""
    <style>
    details {
        border: none !important;
        box-shadow: none !important;
    }
    details summary {
        font-size: 11px !important;
        color: #aaa !important;
        background-color: #2d3748 !important;
        border-radius: 6px !important;
        padding: 4px 10px !important;
        border: 1px solid #4a5568 !important;
        width: fit-content !important;
    }
    details summary:hover {
        color: white !important;
        background-color: #3d4f66 !important;
        border-color: #6b8aad !important;
        cursor: pointer !important;
    }
    details[open] summary {
        border-radius: 6px 6px 0 0 !important;
    }
    details > div {
        background-color: #1a2332 !important;
        border: 1px solid #4a5568 !important;
        border-top: none !important;
        border-radius: 0 0 6px 6px !important;
        padding: 10px !important;
    }
    .st-expander {
        border: none !important;
        box-shadow: none !important;
    }
    [data-testid="stExpander"] {
        border: none !important;
        box-shadow: none !important;
    }
    [data-testid="stExpanderDetails"] {
        border: none !important;
    }
    </style>
""", unsafe_allow_html=True)

@st.cache_data(ttl=3600)
def load_tracker():
    df = pd.read_csv('predictions_tracker.csv')
    df['season'] = df['season'].astype(int)
    df['week']   = df['week'].astype(int)
    return df

df = load_tracker()

def load_agent_analysis(week: int, season: int) -> dict:
    cache_file = f"agent_analysis_{season}_week{week}.json"
    if os.path.exists(cache_file):
        with open(cache_file, 'r') as f:
            return json.load(f)
    return None

def get_confidence(home, away, game_analysis):
    key  = f"{home}_{away}"
    text = game_analysis.get(key, '')
    if '🟢' in text:
        return 'HIGH'
    elif '🟡' in text:
        return 'MEDIUM'
    elif '🔴' in text or 'SKIP' in text.upper():
        return 'SKIP'
    else:
        return 'NO_ANALYSIS'

# ── Sidebar ───────────────────────────────────────────────────────────────────
st.sidebar.markdown("""
    <div style='text-align: center; padding: 10px;'>
        <div style='display: inline-block; background: #013369; color: white;
                    border-radius: 50%; width: 60px; height: 60px;
                    line-height: 60px; font-size: 24px; font-weight: bold;'>
            JS
        </div>
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
now           = dt.now()
season_active = (now.month >= 9) or (now.month <= 2)

if not season_active:
    st.info(
        "🏈 **NFL Offseason**: The 2025 season has concluded. Look at WEEK 10 for demo agent analysis. "
        "Predictions will return when the 2026 season kicks off in September. "
        "Browse past predictions using the sidebar."
    )

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab1, tab2 = st.tabs(["🏈 Weekly Predictions", "📈 Season Performance"])

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1: WEEKLY PREDICTIONS
# ══════════════════════════════════════════════════════════════════════════════
with tab1:

    week_df    = df[(df['season'] == season) & (df['week'] == week)].copy()
    results_in = week_df['actual_margin'].notna().any()

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

    st.divider()
    col1, col2, col3, col4 = st.columns(4)

    filtered_df  = week_df[week_df['model_edge'].abs() >= edge_threshold].copy()
    hidden_count = len(week_df) - len(filtered_df)

    col1.metric("Total Games", len(week_df))
    col2.metric("Showing",     len(filtered_df),
                help=f"Games with |edge| ≥ {edge_threshold} pts")
    col3.metric("Avg Edge",    f"{week_df['model_edge'].abs().mean():.1f} pts")

    if results_in and len(filtered_df) > 0:
        sc = int(filtered_df['model_correct'].sum())
        col4.metric("ATS Record", f"{sc}/{len(filtered_df)} ({sc/len(filtered_df)*100:.0f}%)")
    else:
        col4.metric("ATS Record", "Pending")

    st.divider()

    cached        = load_agent_analysis(week, season)
    game_analysis = cached.get('game_analysis', {}) if cached else {}

    st.markdown("""
        <div style='display:flex;gap:16px;align-items:center;margin-bottom:12px;flex-wrap:wrap;'>
            <span style='font-size:11px;color:#888;letter-spacing:1px;text-transform:uppercase;'>Agent Confidence:</span>
            <span style='font-size:12px;background:#1a3a1a;border:1px solid #00c853;
                        border-radius:4px;padding:2px 8px;color:#00c853;'>🟢 High</span>
            <span style='font-size:12px;background:#3a3a1a;border:1px solid #ffd600;
                        border-radius:4px;padding:2px 8px;color:#ffd600;'>🟡 Medium</span>
            <span style='font-size:12px;background:#3a1a1a;border:1px solid #ff5252;
                        border-radius:4px;padding:2px 8px;color:#ff5252;'>🔴 Skip</span>
        </div>
    """, unsafe_allow_html=True)

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

            home_is_favored = spread > 0

            if home_is_favored:
                top_team      = home
                bot_team      = away
                top_spread    = fmt(-spread)
                bot_spread    = fmt(spread)
                top_predicted = fmt(predicted)
                bot_predicted = fmt(-predicted)
            else:
                top_team      = away
                bot_team      = home
                top_spread    = fmt(spread)
                bot_spread    = fmt(-spread)
                top_predicted = fmt(-predicted)
                bot_predicted = fmt(predicted)

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
                    top_score = fmt(actual if home_is_favored else -actual)
                    bot_score = fmt(-actual if home_is_favored else actual)
            else:
                top_score = "—"
                bot_score = "—"

            result_label = ("✅ WIN" if correct else "❌ LOSS") if results_in else ""

            def name_style(is_rec):
                weight = "700" if is_rec else "400"
                color  = "white" if is_rec else "#aaa"
                return weight, color

            def stat_box(val, is_rec=False, is_result=False):
                bg    = "#1e2a3a"
                color = "white"
                return (
                    f"<div style='text-align:center;background:{bg};border-radius:6px;"
                    f"padding:6px 0;font-size:14px;font-weight:600;color:{color};"
                    f"height:32px;line-height:20px'>{val}</div>"
                )

            def bet_box(team):
                return (
                    f"<div style='background:#1e2a3a;border-left:3px solid #4a6080;"
                    f"border-radius:4px;padding:0 8px;font-size:12px;font-weight:700;"
                    f"color:white;text-align:center;height:32px;line-height:32px'>"
                    f"BET {team}</div>"
                )

            def empty_box():
                return "<div style='height:32px'></div>"

            with st.container():
                st.markdown(
                    f"<div style='font-size:13px;color:#888;margin-bottom:6px'>"
                    f"<b style='color:#ccc'>{away} @ {home}</b>"
                    f"&nbsp;&nbsp;·&nbsp;&nbsp;{row['gameday']}"
                    f"{'&nbsp;&nbsp;·&nbsp;&nbsp;<b>' + result_label + '</b>' if result_label else ''}"
                    f"</div>",
                    unsafe_allow_html=True
                )

                if results_available:
                    h0, h1, h2, h3, h4 = st.columns([2.2, 1.2, 1.2, 1.2, 1.8])
                    h3.markdown("<div style='text-align:center;font-size:11px;color:#aaa;letter-spacing:1px'>SCORE</div>", unsafe_allow_html=True)
                else:
                    h0, h1, h2, h4 = st.columns([2.2, 1.2, 1.2, 1.8])

                h1.markdown("<div style='text-align:center;font-size:11px;color:#aaa;letter-spacing:1px'>SPREAD</div>",    unsafe_allow_html=True)
                h2.markdown("<div style='text-align:center;font-size:11px;color:#aaa;letter-spacing:1px'>PREDICTED</div>", unsafe_allow_html=True)

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
                a1.markdown(stat_box(top_spread),                       unsafe_allow_html=True)
                a2.markdown(stat_box(top_predicted, is_rec=top_is_rec), unsafe_allow_html=True)
                a4.markdown(bet_box(top_team) if top_is_rec else empty_box(), unsafe_allow_html=True)

                st.markdown("<div style='height:6px'></div>", unsafe_allow_html=True)

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
                b1.markdown(stat_box(bot_spread),                       unsafe_allow_html=True)
                b2.markdown(stat_box(bot_predicted, is_rec=bot_is_rec), unsafe_allow_html=True)
                b4.markdown(bet_box(bot_team) if bot_is_rec else empty_box(), unsafe_allow_html=True)

                game_key  = f"{home}_{away}"
                game_text = game_analysis.get(game_key, None)

                # Determine confidence color
                if game_text:
                    if '🟢' in game_text:
                        btn_color = "#00c853"
                        btn_bg    = "#1a3a1a"
                        btn_label = "🟢 Matchup Analysis"
                    elif '🟡' in game_text:
                        btn_color = "#ffd600"
                        btn_bg    = "#3a3a1a"
                        btn_label = "🟡 Matchup Analysis"
                    elif '🔴' in game_text or 'SKIP' in game_text.upper():
                        btn_color = "#ff5252"
                        btn_bg    = "#3a1a1a"
                        btn_label = "🔴 Matchup Analysis"
                    else:
                        btn_color = "#ff5252"
                        btn_bg    = "#3a1a1a"
                        btn_label = "🔴 Matchup Analysis"
                else:
                    btn_color = "#aaaaaa"
                    btn_bg    = "#1e1e1e"
                    btn_label = "⚪ Matchup Analysis"

                col_btn, _ = st.columns([1, 3])
                with col_btn:
                    with st.expander(btn_label):
                        if game_text:
                            st.markdown(game_text)
                        else:
                            st.caption("No analysis yet. Run the notebook to generate.")

                st.divider()

    # ── Agent vs Model Evaluation ─────────────────────────────────────────────
    if cached and game_analysis:
        st.divider()
        st.subheader(f"📊 Week {week}: Agent vs Model")

        def get_confidence_local(home, away):
            key  = f"{home}_{away}"
            text = game_analysis.get(key, '')
            if '🟢' in text:
                return 'HIGH'
            elif '🟡' in text:
                return 'MEDIUM'
            elif '🔴' in text or 'SKIP' in text.upper():
                return 'SKIP'
            else:
                return 'NO_ANALYSIS'

        week_df_eval = week_df.copy()
        week_df_eval['agent_confidence'] = week_df_eval.apply(
            lambda r: get_confidence_local(r['home_team'], r['away_team']), axis=1
        )

        if results_in:
            model_correct = int(week_df_eval['model_correct'].sum())
            model_total   = len(week_df_eval)
            model_pct     = round(model_correct / model_total * 100, 1)

            high_df      = week_df_eval[week_df_eval['agent_confidence'] == 'HIGH']
            high_correct = int(high_df['model_correct'].sum())
            high_total   = len(high_df)
            high_pct     = round(high_correct / high_total * 100, 1) if high_total > 0 else 0

            med_df      = week_df_eval[week_df_eval['agent_confidence'] == 'MEDIUM']
            med_correct = int(med_df['model_correct'].sum())
            med_total   = len(med_df)
            med_pct     = round(med_correct / med_total * 100, 1) if med_total > 0 else 0

            bet_df      = week_df_eval[week_df_eval['agent_confidence'].isin(['HIGH', 'MEDIUM'])]
            bet_correct = int(bet_df['model_correct'].sum())
            bet_total   = len(bet_df)
            bet_pct     = round(bet_correct / bet_total * 100, 1) if bet_total > 0 else 0

            skip_df      = week_df_eval[week_df_eval['agent_confidence'] == 'SKIP']
            skip_correct = int(skip_df['model_correct'].sum())
            skip_total   = len(skip_df)
            skip_pct     = round(skip_correct / skip_total * 100, 1) if skip_total > 0 else 0

            c1, c2, c3, c4 = st.columns(4)
            c1.metric("📈 Model (all games)",  f"{model_correct}/{model_total}", f"{model_pct}%")
            c2.metric("🟢 Agent HIGH only",    f"{high_correct}/{high_total}",   f"{high_pct}%")
            c3.metric("🟡 Agent HIGH+MED",     f"{bet_correct}/{bet_total}",     f"{bet_pct}%")
            c4.metric("🔴 Skipped games",      f"{skip_correct}/{skip_total}",   f"{skip_pct}%",
                      help="Lower % here = agent correctly avoided bad bets")

            st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)

            if skip_total > 0:
                # Correct comparison: agent bet picks vs model all-in
                if bet_pct > model_pct:
                    improvement = round(bet_pct - model_pct, 1)
                    st.success(
                        f"✅ Betting only agent HIGH+MED picks improved accuracy by **{improvement}%** — "
                        f"agent picks went {bet_pct}% ({bet_correct}/{bet_total}) vs model's {model_pct}% ({model_correct}/{model_total}) on all games"
                    )
                elif bet_pct == model_pct:
                    st.info(
                        f"➡️ Agent picks matched model accuracy — both went {model_pct}%"
                    )
                else:
                    decline = round(model_pct - bet_pct, 1)
                    st.warning(
                        f"⚠️ Agent picks underperformed by {decline}% — "
                        f"agent picks went {bet_pct}% ({bet_correct}/{bet_total}) vs model's {model_pct}% ({model_correct}/{model_total}) on all games"
                )
        else:
            st.info("Results not yet available for this week. Check back after games are played.")

# ══════════════════════════════════════════════════════════════════════════════
# TAB 2: SEASON PERFORMANCE
# ══════════════════════════════════════════════════════════════════════════════
with tab2:

    st.title(f"📈 {season} Season Performance")

    # Build season-wide stats from tracker
    season_df = df[
        (df['season'] == season) &
        (df['actual_margin'].notna())
    ].copy()

    if season_df.empty:
        st.warning("No completed games found for this season.")
    else:

        # ── Season summary metrics ────────────────────────────────────
        total_correct = int(season_df['model_correct'].sum())
        total_games   = len(season_df)
        total_pct     = round(total_correct / total_games * 100, 1)

        high_edge_df  = season_df[season_df['model_edge'].abs() >= 3]
        he_correct    = int(high_edge_df['model_correct'].sum())
        he_total      = len(high_edge_df)
        he_pct        = round(he_correct / he_total * 100, 1) if he_total > 0 else 0

        med_edge_df   = season_df[(season_df['model_edge'].abs() >= 1) & (season_df['model_edge'].abs() < 3)]
        me_correct    = int(med_edge_df['model_correct'].sum())
        me_total      = len(med_edge_df)
        me_pct        = round(me_correct / me_total * 100, 1) if me_total > 0 else 0

        low_edge_df   = season_df[season_df['model_edge'].abs() < 1]
        le_correct    = int(low_edge_df['model_correct'].sum())
        le_total      = len(low_edge_df)
        le_pct        = round(le_correct / le_total * 100, 1) if le_total > 0 else 0

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Season ATS",          f"{total_correct}/{total_games}", f"{total_pct}%")
        c2.metric("High Edge (3+ pts)",  f"{he_correct}/{he_total}",       f"{he_pct}%")
        c3.metric("Med Edge (1-3 pts)",  f"{me_correct}/{me_total}",       f"{me_pct}%")
        c4.metric("Low Edge (<1 pt)",    f"{le_correct}/{le_total}",       f"{le_pct}%")

        st.divider()

        # ── Week by week ATS chart ────────────────────────────────────
        weekly = season_df.groupby('week').agg(
            correct=('model_correct', 'sum'),
            total=('model_correct', 'count')
        ).reset_index()
        weekly['pct']      = (weekly['correct'] / weekly['total'] * 100).round(1)
        weekly['record']   = weekly['correct'].astype(str) + '-' + (weekly['total'] - weekly['correct']).astype(str)
        weekly['week_lbl'] = 'Week ' + weekly['week'].astype(str)

        # Cumulative win %
        weekly['cum_correct'] = weekly['correct'].cumsum()
        weekly['cum_total']   = weekly['total'].cumsum()
        weekly['cum_pct']     = (weekly['cum_correct'] / weekly['cum_total'] * 100).round(1)

        st.subheader("Week by Week ATS Record")

        fig_bar = go.Figure()
        fig_bar.add_trace(go.Bar(
            x=weekly['week_lbl'],
            y=weekly['pct'],
            text=weekly['record'],
            textposition='outside',
            marker_color=[
                '#00c853' if p >= 60 else '#ffd600' if p >= 50 else '#ff5252'
                for p in weekly['pct']
            ],
            hovertemplate='%{x}<br>ATS: %{text}<br>Win%%: %{y}%<extra></extra>'
        ))
        fig_bar.add_hline(
            y=50, line_dash="dash", line_color="#888",
            annotation_text="Break even (50%)", annotation_position="right"
        )
        fig_bar.update_layout(
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font_color='white',
            yaxis=dict(range=[0, 100], title='ATS Win %', gridcolor='#2d3748'),
            xaxis=dict(gridcolor='#2d3748'),
            showlegend=False,
            height=350,
            margin=dict(t=20, b=20)
        )
        st.plotly_chart(fig_bar, use_container_width=True)

        st.divider()

        # ── Cumulative win % over season ──────────────────────────────
        st.subheader("Cumulative ATS Win % Over Season")

        fig_line = go.Figure()
        fig_line.add_trace(go.Scatter(
            x=weekly['week_lbl'],
            y=weekly['cum_pct'],
            mode='lines+markers',
            line=dict(color='#2979ff', width=2),
            marker=dict(size=8, color='#2979ff'),
            hovertemplate='%{x}<br>Cumulative Win%%: %{y}%<extra></extra>'
        ))
        fig_line.add_hline(
            y=50, line_dash="dash", line_color="#888",
            annotation_text="Break even (50%)", annotation_position="right"
        )
        fig_line.update_layout(
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font_color='white',
            yaxis=dict(range=[0, 100], title='Cumulative ATS Win %', gridcolor='#2d3748'),
            xaxis=dict(gridcolor='#2d3748'),
            showlegend=False,
            height=350,
            margin=dict(t=20, b=20)
        )
        st.plotly_chart(fig_line, use_container_width=True)

        st.divider()

        # ── High vs low confidence accuracy ──────────────────────────
        st.subheader("Edge Tier Accuracy")

        edge_data = pd.DataFrame([
            {'Tier': 'High Edge (3+ pts)',  'Correct': he_correct, 'Total': he_total, 'Pct': he_pct},
            {'Tier': 'Med Edge (1-3 pts)',  'Correct': me_correct, 'Total': me_total, 'Pct': me_pct},
            {'Tier': 'Low Edge (<1 pt)',    'Correct': le_correct, 'Total': le_total, 'Pct': le_pct},
        ])

        fig_edge = go.Figure()
        fig_edge.add_trace(go.Bar(
            x=edge_data['Tier'],
            y=edge_data['Pct'],
            text=[f"{r['Correct']}/{r['Total']} ({r['Pct']}%)" for _, r in edge_data.iterrows()],
            textposition='outside',
            marker_color=['#00c853', '#ffd600', '#ff5252'],
            hovertemplate='%{x}<br>%{text}<extra></extra>'
        ))
        fig_edge.add_hline(
            y=50, line_dash="dash", line_color="#888",
            annotation_text="Break even", annotation_position="right"
        )
        fig_edge.update_layout(
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font_color='white',
            yaxis=dict(range=[0, 100], title='ATS Win %', gridcolor='#2d3748'),
            xaxis=dict(gridcolor='#2d3748'),
            showlegend=False,
            height=350,
            margin=dict(t=20, b=20)
        )
        st.plotly_chart(fig_edge, use_container_width=True)

        st.divider()

        # ── Best and worst weeks ──────────────────────────────────────
        st.subheader("Best & Worst Weeks")

        col_best, col_worst = st.columns(2)

        with col_best:
            st.markdown("**🏆 Best Weeks**")
            best = weekly.nlargest(3, 'pct')[['week_lbl', 'record', 'pct']]
            best.columns = ['Week', 'Record', 'Win %']
            st.dataframe(best, hide_index=True, use_container_width=True)

        with col_worst:
            st.markdown("**📉 Worst Weeks**")
            worst = weekly.nsmallest(3, 'pct')[['week_lbl', 'record', 'pct']]
            worst.columns = ['Week', 'Record', 'Win %']
            st.dataframe(worst, hide_index=True, use_container_width=True)

        st.divider()

        # ── Full season table ─────────────────────────────────────────
        with st.expander("📋 Full season week by week"):
            table = weekly[['week_lbl', 'record', 'pct', 'cum_pct']].copy()
            table.columns = ['Week', 'Record', 'Win %', 'Cumulative %']
            st.dataframe(table, hide_index=True, use_container_width=True)