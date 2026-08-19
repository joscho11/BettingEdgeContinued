"""Help & Guide page.

FAQ plus a model-rundown section for every prediction product the site currently
publishes. Licensed-adjacent throughout: no CLV / beat-the-close claims, no
public naming of the Draft Board Sleeper mix, aggregate-only Draft Board language.
Live ATS numbers interpolate from live_2026.py. 2025 demo ATS interpolates from
dashboard_data.accuracy_stats.
"""
import streamlit as st

import dashboard_data
import help_models
from dashboard_utils import breakeven_verdict
from live_2026 import LIVE_HIGH_ATS, LIVE_HIGH_N, LIVE_HIGH_WILSON_LOWER, LIVE_HIGH_WINS


def render():
    try:
        df = dashboard_data.load_predictions()
    except FileNotFoundError:
        st.error("predictions_tracker.csv not found. Run the prediction pipeline first.")
        st.stop()
    except Exception as _load_err:
        st.error(f"Failed to load predictions data: {_load_err}")
        st.stop()
    if df.empty:
        st.warning("predictions_tracker.csv has no rows yet. Run the prediction pipeline to populate it.")
        st.stop()
    _demo = df[df["season"] == 2025] if "season" in df.columns else df
    _stats = dashboard_data.accuracy_stats(_demo if not _demo.empty else df)
    _overall_correct = _stats["overall_correct"]
    _overall_total   = _stats["overall_total"]
    _overall_pct     = _stats["overall_pct"]
    _hc_correct      = _stats["hc_correct"]
    _hc_total        = _stats["hc_total"]
    _hc_pct          = _stats["hc_pct"]
    st.title("❓ Help & Guide")
    st.caption("New to the site, or to betting the spread? Start here.")

    st.divider()
    st.subheader("🏈 Betting Basics")

    with st.expander("What is ATS (Against The Spread)?"):
        st.markdown("""
ATS stands for **Against The Spread**. It's the most common way to bet NFL games, and it's what the Weekly Predictions page is built around.

Instead of picking who wins, you bet whether a team wins by more or less than a set number of points. That number is the spread.

**Example:** the Chiefs are favored by 7.5. Bet the Chiefs, they need to win by 8 or more. Bet the Raiders, they need to lose by 7 or fewer, or win outright.

Sportsbooks set the spread to split money, not to predict the final margin. They profit from the juice either way.
        """)

    with st.expander("What is the spread and how does Vegas set it?"):
        st.markdown("""
Oddsmakers set the spread from team strength, injuries, home field, and recent form.

The spread is not a forecast of the actual margin. It is a price meant to attract bets on both sides. If the public piles onto the Chiefs, the line moves to make the other side more attractive.

That movement is where a model can find a gap: the number on the board is a market price, not a true score prediction.
        """)

    with st.expander("What is edge and why does it matter?"):
        st.markdown("""
Edge is the gap between what the model predicts and the posted spread.

If the model has the Chiefs by 10 and the spread is 7.5, that is 2.5 points on the Chiefs. Games under 1 point of disagreement are coin flips in the model's eyes.

On **2026**, every game still shows. HIGH is the green highlight (3 or more points off the Tuesday line, and still 3 off the live line). On the **2025 demo** weeks, use the **Min Edge (pts)** slider to hide the coin flips.
        """)

    with st.expander("What does it mean to cover?"):
        st.markdown("""
Covering means beating the spread.

Chiefs -7.5, win 28-17 (by 11): they covered. Win 24-20 (by 4): they did not.

The other way works too. Raiders +7.5, lose by 4: the Raiders covered even though they lost the game.

The model predicts the margin, then asks which side of the posted number is more likely to cover.
        """)

    with st.expander("How do you actually make money betting?"):
        _hc_line = (f" and **{_hc_pct}%** on high confidence picks "
                    f"({_hc_correct}/{_hc_total})" if _hc_pct is not None else "")
        _be_comment = breakeven_verdict(_overall_pct, _hc_pct)
        st.markdown(f"""
Standard sportsbook odds are about 110 to win 100. You need about **52.4%** of bets to break even. Most casual bettors don't hit that.

The **2026 live book** is Tuesday HIGH: **{LIVE_HIGH_WINS}/{LIVE_HIGH_N} = {LIVE_HIGH_ATS * 100:.2f}%** ATS, one-sided 95% Wilson lower bound **{LIVE_HIGH_WILSON_LOWER * 100:.2f}%**, walk-forward 2021-2025. That interval is above 52.4%. All-bets is not the claim. No 2026 games are graded yet.

The **2025 demo test** on this site (weeks 10-17) is **{_overall_pct}% ATS** ({_overall_correct}/{_overall_total}){_hc_line}. {_be_comment} That walkthrough is the old three-model consensus, not the 2026 live book. Past performance doesn't guarantee anything going forward. There will be bad weeks.

Never bet more than you can afford to lose.
        """)

    with st.expander("What is sharp money vs public money?"):
        st.markdown("""
Public money is casual bettors. Popular teams, primetime, whoever is hot.

Sharp money is professional bettors placing large, calculated bets. When they bet, the line often moves.

**What this site does not know:** I don't have sharp-money or line-movement data. Nothing here tracks where professional money is going. The paragraph above is background on how betting markets work, not an input to the model.
        """)

    st.divider()
    st.subheader("🖥️ How to Use This Website")

    with st.expander("What is live vs demo on this site?"):
        st.markdown("""
**Live 2026 (the current season)**

- **Draft Board:** 180-player Model Proj, frozen until the early-September snapshot. Sleeper ADP and Sleeper Proj refresh daily.
- **Rookie Board:** hit % and RB/WR/TE season-total projections for the 2024-2026 classes.
- **Season Totals (Beta):** 32-team win projections. HIGH is the only certified pick.
- **Weekly Predictions:** 2026 matchups are up. Picks lock Tuesday 9:00 ET. HIGH is the green 3-point Tuesday ticket.
- **Weekly Fantasy:** opens on 2026 Week 1. Those rankings land once that file is published.

**Demo / walkthrough (not the live book)**

- **2025 Weekly Predictions** weeks 10-end: old three-model consensus, HIGH / MED / PASS badges, Min Edge slider.
- **2025 Weekly Fantasy** weeks 10-17: previous per-position XGBoost system, extra prop columns.
- **Over/Under totals:** 2025 demo only, experimental, not on the 2026 week page.

**Not a prediction model**

Talent Scores are descriptive context. League History is your Sleeper league, not a forecast. Film Room is video. DFS is not in the nav yet.
        """)

    with st.expander("How is the site organized?"):
        st.markdown("""
Everything lives in the top navigation bar, grouped into three menus:

- **Fantasy**: Draft Board (the page you land on), Rookie Board, and Weekly Fantasy.
- **Betting**: Weekly Predictions, Track Record, and Season Totals (Beta).
- **More**: Film Room, League History, and this Help & Guide.

The site opens on the **2026 Draft Board** every time. There is no sidebar. Each page carries its own controls (Season and Week, plus a Min Edge slider on the 2025 demo weeks) right at the top.
        """)

    with st.expander("How do I read the game cards?"):
        st.markdown("""
Each card is one matchup.

**SPREAD** is the Vegas line. Negative means that team is favored.

**PREDICTED** is the model's version of the line, sportsbook-style (favorite negative). When the model's number is more extreme than Vegas on a side, that is the edge. Example: Vegas SEA -7, model SEA -11.3. The model likes SEA by 4.3 more points, so it recommends SEA.

**SCORE** fills in after the game. Blank until then.

**BET X** is the recommended side. The bold name is who the model likes.

**2026 live:** every game gets a pick. **HIGH** (green) is a 3+ point disagreement with the Tuesday 9am line, and the live line still 3+. If the line moves and that gap falls under 3, HIGH is dropped. It cannot be created mid-week. No medium tier. No totals on the 2026 week page.

**2025 demo test:** weeks 10 through the end of that season still use HIGH / MED / PASS consensus badges. Those weeks are unchanged.

After results land, each card shows WIN or LOSS based on whether the pick covered.
        """)

    with st.expander("What is the Min Edge filter?"):
        st.markdown("""
On **2026**, every game is shown. HIGH picks are the green highlight. The Min Edge slider is hidden.

On the **2025 demo** weeks, **Min Edge (pts)** at the top of Weekly Predictions controls which games show up. 0.0 (default) is every game. 3.0 is only the high-conviction plays from that old consensus.
        """)

    with st.expander("How often does the site update?"):
        st.markdown("""
**2026 live.** Picks lock Tuesday 9:00 ET off the frozen line. After that, the only change on the week page is dropping HIGH if the live line shrinks the gap under 3 points. A later line cannot promote a game into HIGH. Totals stay off this season's week page.

**2025 demo test** used the older Monday / Thursday / Sunday refresh. Those weeks are frozen as a walkthrough.

During the offseason, 2026 matchups stay on Weekly Predictions with no picks until the Tuesday freeze. The pre-season **Draft Board** refreshes
Sleeper ADP and Sleeper projections daily for its fixed 180-player universe; draft-price
ranks, Sleeper ranks, and both gap columns move with those updates. Model Proj points and
ranks remain frozen until the dated early-September public-information snapshot.
        """)

    with st.expander("What is the Track Record page?"):
        st.markdown("""
Season-long ATS, not one week.

Week-by-week bars, a cumulative line, and a breakdown of HIGH versus the rest. Best and worst weeks, a season table, and (on the 2025 demo) a separate Over/Under section.

2026 Track Record grades HIGH by the Tuesday 3-point rule. There is no medium bucket. 2026 Track Record does not include totals.
        """)

    with st.expander("What is the Season Totals (Beta) page?"):
        st.markdown("""
The Season Totals page leads with the **high-confidence** win record on held-out
seasons (projection at least 1 win off the posted number). Every team still gets
a projection; only those high-confidence rows are highlighted with a check mark.

The footnote states once that the all-team projection does not beat the posted
win total on average, and once that the one-sided 95% Wilson lower bound on
high confidence sits under the 52.4% bar.

How the number is built sits in **How Season Totals are built** below.
        """)

    with st.expander("What is the Weekly Fantasy page?"):
        st.markdown("""
Weekly half-PPR projections for QB, RB, WR, and TE.

The page opens on **2026 Week 1**. Rankings for that week will be here soon. **2025 weeks are a demo** from the previous weekly model.

Search by player. Older demo weeks also show extra columns (team EPA, implied total, health, some stat projections). Actual points fill in after the games.

How those files are built sits in **How weekly fantasy projections are built** below.
        """)

    with st.expander("What is the Draft Board page?"):
        st.markdown("""
A **pre-season comparison table** for 2026, separate from Weekly Fantasy. The universe is fixed: 24 QBs, 60 RBs, 72 WRs, 24 TEs. Each row puts Sleeper draft price and positional rank next to **Model Proj**. Sleeper's current season projection is shown when the player record matches. Each projection also has a gap versus draft-price rank at that position.

**What the gap is.** Position Rank minus that projection's position rank. Positive means the projection ranks him above his draft cost; negative means below. It is arithmetic between two ranks on the same row, never a recommendation.

**How good is the number?** On 2021-2025 Model Proj scored .7101 pairwise versus ADP's .6965, MAE 49.31 versus 51.75, and beat ADP ordering in 5 of 6 seasons (it lost 2020). It is **not live-validated**. The first live test is the 2026 season.

Model Proj values are frozen until the planned dated early-September public-information snapshot. **Sleeper ADP and Sleeper Proj refresh daily; their positional ranks, Sleeper Gap, and Model Gap recalculate from each successful pull.**

The two talent columns are described further down this page. They are descriptive context and feed no other column.

Use **Position**, **Player search**, **Sort by**, and **Order**. No-data rows always sort to the bottom. **Show projection and talent detail** is on by default. On a phone the board shows player, position, ADP, rank, both gaps, and NFL Talent Score. The CSV download always contains all thirteen columns.
        """)

    with st.expander("What is the Rookie Board page?"):
        st.markdown("""
Drafted rookies from the 2024, 2025, and 2026 classes. **Hit probability** is the share of historical players with a similar profile who had at least one startable season in their first three years (top-24 for RB/WR, top-12 for QB/TE, season-total half-PPR). That is a best-of-three-years outcome, not a per-season rate.

Three versions sit side by side: draft capital only, college production and athletic testing only, and both. They land close together. At this sample, college production and testing added no measured edge beyond draft position. Backtested on the 2019-2023 classes, not live-validated.

Beside those sit rookie-year **season-total projections** for RB, WR, and TE, next to Sleeper when Sleeper has one. Rookie QBs show no projection: a rookie QB season hinges on whether he starts, which the features cannot see.

Collapsed lists of **college players who are not in this year's rookie class** use the same College Talent instrument. Most are still in college. They appear on no rookie board and say nothing about whether any of them will be drafted.
        """)

    with st.expander("What is the Film Room page?"):
        st.markdown("""
Short, model-backed video breakdowns. The newest episode is already loaded. Pick another title to swap the player. **📖 Full breakdown** opens the write-up the short couldn't fit.

Some older videos predate later validation work and make calls I wouldn't make today. Those carry a **📼 Archived: why?** pop-out. They stay up, unedited, as part of the record.
        """)

    with st.expander("What is the DFS Optimizer page?"):
        st.markdown("""
**Coming soon.** It is **not in the nav yet**. It ships when it is actually usable, around 2026 Week 1.

When it is live, you upload a DraftKings NFL Classic salary CSV and get the highest-projected legal 9-player lineup under the $50,000 cap, using this site's weekly projections. DST will use DraftKings' season average until there is a team-defense model.
        """)

    with st.expander("What is the League History page?"):
        st.markdown("""
Paste your Sleeper league ID (from `sleeper.com/leagues/{ID}/league`) and hit Load. The page pulls standings, drafts, weekly scores, and a few chart-first views: Draft & Roster Insights, All-Time Leaderboard, Hall of Fame, Rivalries, Report Cards, and Consistency & Luck.

First load is usually a few seconds per season. The same ID is instant for an hour after that. Filter by season or view all-time. Info icons on the cards explain each statistic. This page is your league's history, not a prediction model.
        """)

    with st.expander("What is the LLM agent and what does it do?"):
        st.markdown("""
The agent sat on top of the prediction models using LlamaIndex and Anthropic's Claude API.

**Paused as of August 2026.** The agent's line-movement tool was never connected to
a real market feed. It returned hardcoded example values, and the one cached week it
produced stated those as if they were observed sharp-money and line-movement figures. I've
taken that cached analysis down, disabled the weekly agent run, and the site now refuses to
render any market claim that can't prove where it came from.

The colored High / Medium / Skip buttons are not on the game cards right now. What you see
on **2026** Weekly Predictions is **Tuesday HIGH**: a green highlight when the model
disagrees with the Tuesday 9am line by 3+ points and the live line still does. The
**2025 demo** weeks still show **Model Consensus** (HIGH / MED / PASS).

When an approved agent artifact is present again, High means the model edge is strong and outside signals lined up, Medium means mixed signals, and Skip means pass.
        """)

    st.divider()
    help_models.render_rundowns()

    st.divider()
    st.subheader("🏆 Fantasy Projections")

    with st.expander("How do the fantasy projections work?"):
        st.markdown("""
The Weekly Fantasy page uses a separate system from the betting model.

**2025 demo (weeks 10-17 on this page).** Four per-position XGBoost models trained on 2020-2024, with 2025 held out. Walkthrough only. Not the 2026 live weekly model.

**2026 live, starting Week 1.** One half-PPR points model. Same scoring: 0.5 per reception, yards and touchdowns as usual. The first live week is 2026 Week 1.

The models are rebuilt in the offseason as more data lands.
        """)

    with st.expander("How accurate are the fantasy projections?"):
        st.markdown("""
**2025 demo.** Those weeks were scored against a simple 3-week rolling average, on the previous per-position XGBoost system:

| Position | Model MAE | Baseline MAE | Improvement |
|----------|-----------|--------------|-------------|
| QB | 7.0 pts | 7.5 pts | Better |
| RB | 4.5 pts | 4.6 pts | Better |
| WR | 3.9 pts | 4.1 pts | Better |
| TE | 3.2 pts | 3.5 pts | Better |

MAE is the average number of points the projection was off by. For WR, about 3.9 points. Any individual week can be much higher or lower. Treat the numbers as a ranking tool, not a precise point forecast.

**2026 live.** First live week is Week 1. That run uses the new weekly model, not the 2025 demo files.
        """)

    with st.expander("What are the prop stat columns?"):
        st.markdown("""
These extra stat columns appear on the **2025 demo weeks**. 2026 live weeks show half-PPR points; the extra yard/reception columns only show when that file has them.

Eight separate XGBoost models, trained on the same data with each stat as the target:

| Column | Position | What it predicts |
|--------|----------|-----------------|
| Proj Pass Yds | QB | Passing yards |
| Proj Rush Yds | QB / RB | Rushing yards |
| Proj Rec Yds | RB / WR / TE | Receiving yards |
| Proj Receptions | WR / TE | Number of receptions |

Useful as a rough look at player props. They are **independent** models, so they will not add up to the fantasy-point total. QB passing yards has the highest error (about 70 yards). RB and TE receiving yards are the tightest (about 10-14 yards MAE).
        """)

    with st.expander("What do the column headers mean?"):
        st.markdown("""
**Player:** name and NFL team.

**Opponent:** this week's opponent. `@` is away, `vs` is home.

**Proj Pts:** projected half-PPR points. 0.5 per reception, 1 per 10 rush or receiving yards, 6 per TD.

**Off EPA:** the team's offensive efficiency over the last 4 games, Expected Points Added per play. Higher is better. Demo weeks only.

**EPA Rank:** where that offense ranks among 32 teams this season. Green to red.

**Team Total:** Vegas implied team total, how many points Vegas expects this team to score.

**Health:** injury report. ✅ Healthy · 🟡 Questionable · ⚠️ Doubtful · ❌ Out. Players ruled Out are removed.

**Actual Pts / Actual [stat]:** fills in after the game. Blank means the player did not play.
        """)

    with st.expander("What is Off EPA?"):
        st.markdown("""
**Off EPA** is Offensive Expected Points Added per play, averaged over the team's last 4 games. Demo-week column.

EPA measures how much each play moves the needle toward scoring. A 5-yard gain on 3rd and 4 is worth more than a 5-yard gain on 1st and 10, because it accounts for down, distance, and field position.

- **Positive (e.g. +0.15):** efficient recently
- **Near zero (e.g. +0.01):** average
- **Negative (e.g. -0.12):** struggling

League average hovers near 0. Above +0.10 is strong, below -0.10 is poor. Players on efficient offenses tend to see more opportunities in good game scripts. It is one of the stronger predictors in the 2025 demo models.
        """)

    with st.expander("How often do fantasy projections update?"):
        st.markdown("""
Fantasy projections are generated separately from the weekly betting GitHub Action. That job only papermills the spread and totals notebooks. Weekly fantasy is not on that Tuesday cron yet.

The projection file for each week is saved once and does not change after that. It reflects the injury and depth-chart data available at the time it was run. Actual stats fill in after each game, pulling live from nflreadpy and caching for 1 hour.

If you're looking at a past week, the actuals shown are the real NFL stats for that game.
        """)

    st.divider()
    st.subheader("🧮 NFL Talent Score & College Talent Score")

    with st.expander("What are the two score columns on the Draft Board?"):
        st.markdown("""
The Draft Board carries two context columns I build myself, answering two different questions.

**The NFL Talent Score** is my model-based estimate of what a player does with each opportunity (each carry, route, or throw), separated from his situation where that separation is statistically possible. It is not a summary of his production, and models can be wrong. Volume is excluded by design: how often a player is used tells you about his coach's plans, not his per-play skill. Every position reads its own dedicated build, scored against qualified starters at that position. A player below his position's volume floor is left **blank** rather than quietly placed on another position's scale.

**The College Talent Score** is a college-production read for 2026 rookies at all four positions (QB, RB, WR and TE), each from its own dedicated college build, scaled against past prospects at that position who reached the NFL. It describes what a prospect did in college; it does not claim to predict NFL careers or fantasy outcomes.

**Two limits on the college side, stated plainly.** There is no strength-of-schedule adjustment: production against a weaker opponent counts exactly the same as production against a stronger one, which is why several of the highest college scores belong to small-school players the draft market rated far lower. And the underlying data covers FBS only, so a prospect from a smaller division can never be scored. That blank is by construction, not a missing lookup.

**They are two different scales.** The NFL column ranks NFL players against NFL players; the college column ranks prospects against past prospects. A 90 in one is not a 90 in the other, and neither feeds any other number on this board.
        """)

    with st.expander("How the talent scores are built (and what they don't measure)"):
        st.markdown("""
There are eight builds behind these two columns, one for each of NFL and college at each of the four positions. Each takes a small set of per-opportunity measures I call facets (broken tackles per carry, yards per route run, completion rate versus expectation, and so on), scores every player against his own position in his own season, and then shrinks each measure toward the position average according to how much data sits behind it. A thin sample gets pulled toward the middle rather than being trusted at face value.

That shrinkage has a consequence worth stating: **a facet measured on very few plays contributes far less than its nominal weight suggests.** Contested catches are the clearest case. A tight end sees a handful of them a season, so that facet ends up carrying a few percent of the score no matter what weight I assign it. I measured this before reading any results, and where I tried to compensate by raising the weight, the players the facet exists to reward generally got *worse*, not better. So the weights ship as ratified.

Quarterbacks are the asterisk on the NFL side: one starter per team means a QB's situation cannot be separated from him, so QB scores ship **unadjusted**, a different kind of estimate under the same header. The QB build does now measure performance under pressure, though on a small enough sample that it contributes little. A consequence I have not engineered away: an immobile quarterback cannot score well here, because designed rushing carries a quarter of the composite.

Recent seasons count more, on a decay I chose and wrote down rather than fitted. Scores are clipped to a 50-99 display range (40-99 for college quarterbacks), so **50 is the floor of the display, not a league-average player**. Ranks are a more reliable read than any single number.

These columns are context only. A test found that efficiency measures like these do not predict where the draft market misprices players, so they never combine with the projections, the ranks, or the gap columns anywhere on the board. The college instruments in particular were each measured against NFL outcomes and each came back **dead**. They ship as description of college production, and nothing more.

The full write-up covering every design choice, the admission gates, and where it fails lives in my research notes.
        """)

    st.divider()
    st.subheader("🔧 Behind the Scenes")

    with st.expander("What data does the site use?"):
        st.markdown("""
Play-by-play and schedule data come from nflreadpy, going back far enough to build each model's training window (2018 for the 2026 spread model, 2014 for older demo files).

Injury reports feed availability: Out and Doubtful players reduce a team's talent and lineup features. The All-Pro CSV (1997-2025) is a custom roster-talent file, updated manually each January.

Posted win totals on Season Totals are a sportsbook snapshot with an as-of date on that page. Sleeper ADP and Sleeper projections on the Draft Board refresh daily. The LLM agent is paused (August 2026).
        """)

    with st.expander("Is this financial advice?"):
        st.markdown("""
No. This is a personal data science project. I built it to see whether a machine learning model can find a consistent edge against the spread.

Nothing on this site should be taken as betting or financial advice. Sports betting involves real financial risk. Always bet responsibly.
        """)

    st.markdown("""
        <div style='text-align:center;padding:28px 0 12px 0;border-top:1px solid #2d3748;margin-top:12px'>
            <div style='font-size:11px;color:#444;margin-bottom:10px;letter-spacing:0.3px'>
                Not financial advice. Sports betting involves real risk. Bet responsibly.
            </div>
            <div style='font-size:13px;color:#666'>
                Built by <b style='color:#999'>Joseph Schoenbaum</b>
                &nbsp;·&nbsp;
                <a href='https://github.com/joscho11/JoSchoAnalytics'
                   style='color:#3D95CE;text-decoration:none'>GitHub</a>
                &nbsp;·&nbsp;
                <a href='https://venmo.com/u/JoScho'
                   style='color:#3D95CE;text-decoration:none'>💙 Venmo @JoScho</a>
            </div>
        </div>
    """, unsafe_allow_html=True)
