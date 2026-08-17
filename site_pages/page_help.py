"""Help & Guide page. Originally moved byte-identical from the retired app.py Help tab
(Batch 3d); refreshed 2026-07-14 to match the current multipage site — "tab" → "page"
throughout, the retired-sidebar references replaced (each page carries its own controls at
the top now), a Film Room entry + a site-organization note added, and the offseason /
Draft-Board and agent data-source notes corrected. Licensed-adjacent throughout: the
honest-numbers discipline (no CLV / beat-the-close claims) and the aggregate-only Draft
Board language are preserved unchanged. The live model stats the prose interpolates still
come from dashboard_data.accuracy_stats — the same shared plumbing app.py uses.
"""
import streamlit as st

import dashboard_data
from dashboard_utils import breakeven_verdict


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
    _stats = dashboard_data.accuracy_stats(df)   # 3a shared plumbing (byte-identical values)
    _acc_col         = _stats["acc_col"]
    _completed       = _stats["completed"]
    _overall_correct = _stats["overall_correct"]
    _overall_total   = _stats["overall_total"]
    _overall_pct     = _stats["overall_pct"]
    _hc_correct      = _stats["hc_correct"]
    _hc_total        = _stats["hc_total"]
    _hc_pct          = _stats["hc_pct"]
    st.title("❓ Help & Guide")
    st.caption("New to sports betting or just not sure how this site works? This page covers everything.")

    st.divider()

    # ── Section 1: Betting Basics ─────────────────────────────────────────────
    st.subheader("🏈 Betting Basics")

    with st.expander("What is ATS (Against The Spread)?"):
        st.markdown("""
ATS stands for **Against The Spread**. It's the most common way to bet on NFL games and it's what this whole site is built around.

Instead of just picking who wins, you're betting on whether a team wins by more or less than a set number of points. That number is called the spread.

**Here's a simple example:**

The Chiefs are favored by 7.5 points. If you bet the Chiefs, they need to win by 8 or more for you to win. If you bet the Raiders, they just need to lose by 7 or fewer or win outright. That's it.

Vegas sets the spread to try and split betting money evenly. They don't care who wins the game. They care about getting 50% of bets on each side so they profit from the juice no matter what.
        """)

    with st.expander("What is the spread and how does Vegas set it?"):
        st.markdown("""
The spread is set by oddsmakers at sportsbooks like DraftKings or FanDuel. They factor in team strength, injuries, home field, recent form, and a bunch of other stuff.

The key thing to understand is the spread is not meant to predict the actual final margin. It's meant to generate equal action on both sides. That distinction matters.

If the public loves the Chiefs and piles money on them, Vegas moves the line to make betting the Raiders more attractive. The line is always adjusting based on where money is flowing.

This is actually where edge comes from. If Vegas has to shade a line one way to balance public money, it can create value on the other side.
        """)

    with st.expander("What is edge and why does it matter?"):
        st.markdown("""
Edge is the gap between what the model predicts and what Vegas set as the spread.

If the model thinks the Chiefs will win by 10 but the spread is only 7.5, that's a 2.5 point edge on the Chiefs. The model is saying Vegas underpriced the Chiefs.

The bigger the edge, the more the model disagrees with the market. Games with a small edge (under 1 point) are basically coin flips in the model's eyes. Use the **Min Edge (pts)** slider at the top of the Weekly Predictions page to filter down to only the games where the model has real conviction.

You want to be betting games where the model has conviction, not games where it's a coin flip.
        """)

    with st.expander("What does it mean to cover?"):
        st.markdown("""
Covering just means beating the spread.

If the Chiefs are 7.5 point favorites and win 28 to 17, they won by 11. They covered. If they win 24 to 20, they won by 4. They didn't cover.

It works the other way too. If you bet the Raiders plus 7.5 and they lose by 4, the Raiders covered even though they lost the game.

The model is trying to predict the margin of victory and figure out which side of the spread is more likely to cover.
        """)

    with st.expander("How do you actually make money betting?"):
        # The high-confidence rate comes from the agent-analysis caches, which cover far
        # fewer games than the tracker — print its own denominator so a 6-game figure
        # cannot read like the 117-game one beside it.
        _hc_line = (f" and **{_hc_pct}%** on high confidence picks "
                    f"({_hc_correct}/{_hc_total})" if _hc_pct is not None else "")
        _be_comment = breakeven_verdict(_overall_pct, _hc_pct)
        st.markdown(f"""
Honestly it's really hard and most people lose money. I want to be upfront about that.

Standard sportsbook odds are around 110 to win 100. That means you need to win about 52.4% of your bets just to break even. Most casual bettors don't hit that number.

To be profitable over time you need to consistently win more than 52.4%, bet games where there's real edge instead of gut feeling, and manage your bankroll properly. A common rule is never betting more than 2 to 5% of your total bankroll on a single game.

The model is currently at **{_overall_pct}% ATS** overall ({_overall_correct}/{_overall_total}){_hc_line}. {_be_comment} But I want to be clear that past performance doesn't guarantee anything going forward. There will be bad weeks.

Never bet more than you can afford to lose.
        """)

    with st.expander("What is sharp money vs public money?"):
        st.markdown("""
Public money is casual bettors going with their gut. They tend to bet popular teams, primetime games, and whoever is on a hot streak. They're not doing deep analysis.

Sharp money is professional bettors who are placing large, calculated bets based on models and data. When sharps bet big, the line moves.

Watching line movement can tell you a lot. If the Chiefs open at 7 and move to 7.5, someone is betting the Chiefs heavily. If it's sharp money driving that, it's a signal worth paying attention to.

**To be clear about what this site does and doesn't know:** I don't have sharp-money or
line-movement data. Nothing on this site tracks where professional money is going, and no
number you see here is derived from it. The above is background on how betting markets
work, not a description of an input to my model.
        """)

    st.divider()

    # ── Section 2: How to Use the Website ────────────────────────────────────
    st.subheader("🖥️ How to Use This Website")

    with st.expander("How is the site organized?"):
        st.markdown("""
Everything lives in the top navigation bar, grouped into three menus:

- **Betting**: Weekly Predictions (the page you land on), Track Record, and Season Totals (Beta).
- **Fantasy**: Draft Board, Rookie Board, Weekly Fantasy, and DFS Optimizer.
- **More**: Film Room, League History, and this Help & Guide.

The site opens on **Weekly Predictions** every time. There is no sidebar. Each page carries its own controls (like the Season, Week, and Min Edge pickers) right at the top.
        """)

    with st.expander("How do I read the game cards?"):
        st.markdown("""
Each card shows one matchup for the week. Here's what the columns mean:

**SPREAD** is the Vegas line. A negative number means that team is favored.

**PREDICTED** is the model's version of the line — also shown sportsbook-style (favorite negative, underdog positive). When the model's number is *more* extreme than the Vegas spread on a side, that's where the edge is. Example: Vegas has SEA -7 but the model says SEA -11.3 — the model likes SEA by 4.3 more points than Vegas, so it recommends betting SEA.

**SCORE** shows the final score after the game is played. It's blank until results come in.

**BET X** shows which side the model recommends. The bold team name is who the model likes.

After results are in, each card will show either WIN or LOSS based on whether the model's pick covered.
        """)

    with st.expander("What do the agent confidence colors mean?"):
        st.markdown("""
**Paused as of August 2026.** The colored High / Medium / Skip buttons are not on the game cards right now. The agent's line-movement tool never had a real market feed, so I disabled the weekly agent run and took the cached week down.

What you see on Weekly Predictions is **Model Consensus** (HIGH / MED / PASS): whether the three spread models agree on a side, plus how large the ensemble edge is. That is the live filter.

The expander "What is the LLM agent and what does it do?" has the full pause note. When an approved agent artifact is present again, High means the model edge is strong and outside signals lined up, Medium means mixed signals, and Skip means pass.
        """)

    with st.expander("What is the Min Edge filter?"):
        st.markdown("""
The **Min Edge (pts)** slider at the top of the Weekly Predictions page controls which games show up.

At 0.0 (the default) you see every game. At 1.0 you only see games where the model disagrees with Vegas by at least 1 point. At 3.0 you're only seeing the high conviction plays.

Slide it up to filter down to your highest-confidence plays.
        """)

    with st.expander("How often does the site update?"):
        st.markdown("""
During the season the site runs on an automated schedule through GitHub Actions.

Tuesday morning it fills in the previous week's results and posts initial predictions for the upcoming week using the opening Vegas lines. Thursday night it refreshes those predictions after injury reports drop. Sunday morning it locks in final predictions before kickoff. Then the cycle repeats on Tuesday.

During the offseason the weekly predictions pause. The pre-season **Draft Board** refreshes
Sleeper ADP and Sleeper projections daily for its fixed 180-player universe; draft-price
ranks, Sleeper ranks, and both gap columns move with those updates. Model Proj points and
ranks remain frozen until the dated early-September public-information snapshot.
Weekly predictions spin back up when the season kicks off in September.
        """)

    with st.expander("What is the Track Record page?"):
        st.markdown("""
The Track Record page is where you can see how the model has done across the whole season, not just one week.

It shows a week by week bar chart of ATS win percentage, a cumulative trend line showing how accuracy has moved over time, and a breakdown of how high edge games performed compared to low edge games.

There's also a best and worst weeks section, a full season table, and a separate Over/Under model section showing how the totals picks performed.
        """)

    with st.expander("What is the Season Totals (Beta) page?"):
        st.markdown("""
The Season Totals page is a pre-season projection of how many games each NFL team wins, plus a simulated range around that number.

**It does not beat the archived market consensus.** Over a ten-season backtest it landed slightly further from each team's actual win count than that consensus did. There is no side, no odds, and no confidence tier. Gate C stays shut.

Treat it as a published projection with an honest range, not a betting card. It lives in the Betting menu.
        """)

    with st.expander("What is the Over/Under (Totals) model? (Experimental)"):
        st.markdown("""
**Status: experimental — tracking only, not yet a confident pick.**

In addition to picking sides against the spread, the site runs a separate model for the over/under total. It predicts whether the final combined score will go over or under the Vegas total line. It uses the same underlying features as the spread model plus 14 totals-specific inputs: the Vegas total line, implied team totals, weather (temperature and wind), dome/outdoor status, rolling points scored and allowed by each team over the last 5 games, the league scoring environment over the last 4 weeks, pace (plays per game), and whether it's a division game.

The key finding from development: the edge only shows up on **UNDER picks**, not OVERs. The reason is that recreational bettors tend to bet OVER — everyone loves a shootout — which causes books to shade totals lines slightly high. That creates a systematic edge on the UNDER side that the model is designed to find.

A pick is only flagged as **UNDER** when both the XGBoost and Ridge models independently predict the score will come in below the line. When they disagree, the model passes.

**Where it stands:**
- Walk-forward CV (2020–2025, n=575): **55.7%** hit rate, comfortably above the 52.4% break-even.
- Live 2025 (weeks 10–17, n=46): **52.2%** hit rate, essentially at break-even. The sample is too small to distinguish real edge from CV noise (95% CI is roughly 37–67%).

That's why the badges on the game cards are amber/dashed instead of green — the model says UNDER, but I haven't yet confirmed live that it's actually profitable. I track it through the 2026 season and reassess after a full season of real evidence (~96 picks). **Don't bet these picks; treat them as something to watch.**
        """)

    with st.expander("What is the Weekly Fantasy page?"):
        st.markdown("""
The Weekly Fantasy page shows weekly half-PPR fantasy projections for every active QB, RB, WR, and TE. Each position has its own subtab.

You can filter by team or health status and see projected fantasy points alongside position-specific stat projections (passing yards, rushing yards, receptions, receiving yards). Once the week's games are played, actual stats fill in automatically.

See the Fantasy Projections section below for more detail on how the models work.
        """)

    with st.expander("What is the DFS Optimizer page?"):
        st.markdown("""
The DFS Optimizer page is a DraftKings NFL Classic lineup optimizer. It is **Coming soon** until 2026 Week 1 (September).

The page currently shows that notice, not an upload box. When it is live, you will upload a DraftKings salary CSV and get the highest-projected legal 9-player lineup under the $50,000 salary cap. See the DFS Optimizer section below for how the solver works.
        """)

    with st.expander("What is the Draft Board page?"):
        st.markdown("""
The Draft Board is a **pre-season comparison table** for the 2026 season, separate from the Weekly Fantasy page. It lists the exact 180-player universe published by my independent model: 24 QBs, 60 RBs, 72 WRs and 24 TEs. For each player it puts the current Sleeper draft price and positional rank next to **Model Proj**, which is 75% that independent model and 25% Sleeper's published projection. Sleeper's current season projection is shown when its player record can be matched. Alongside each available projection is the gap between draft-price rank and projected rank at that position.

**What the gap is.** Position Rank minus that projection's position rank. Positive means the projection ranks him above his draft cost; negative means below. It is a plain arithmetic difference between two ranks shown on the same row, descriptive context, never a recommendation about any player.

**How good is the number on the board?** Model Proj is 75% the independent v6 hurdle blend and 25% Sleeper's published projection, then the same affine calibration. On 2021-2025 that mix scored .7101 pairwise versus ADP's .6965, MAE 49.31 versus 51.75, and beat ADP ordering in 5 of 6 seasons (it lost 2020, when Sleeper projections are empty). It is **not live-validated**. The first live test is the 2026 season.

The independent v6 model alone (no Sleeper mix) scored .6892 pairwise versus ADP's .6965, MAE 51.97 versus 51.75, and beat ADP in 2023 only. That independent score is still the research baseline. It is not what the board displays.

The Model Proj values are frozen until the planned dated early-September public-information snapshot. **Sleeper ADP and Sleeper Proj refresh daily; their positional ranks, Sleeper Gap, and Model Gap recalculate from each successful pull.**

The two talent columns are described in their own section further down this page. They are descriptive context on their own scales and feed no other column.

Use the **Position** filter and the **Player search** box to narrow the board, and the **Sort by** and **Order** controls to reorder it — those sort numerically, with no-data rows always at the bottom. **Show projection and talent detail** is on by default; turn it off for a compact nine-column view showing just the price-versus-projection comparison. On a phone the board shows player, position, ADP, and both gaps. The CSV download always contains all thirteen columns.
        """)

    with st.expander("What is the Rookie Board page?"):
        st.markdown("""
The Rookie Board covers drafted rookies from the 2024, 2025 and 2026 classes. For each one it shows a **hit probability** — the share of historical players with a similar profile who had at least one startable season in their first three years (top-24 for RB/WR, top-12 for QB/TE, in season-total half-PPR). It is a best-of-three-years outcome, not a per-season rate.

Three versions of that same number sit side by side: one from **draft capital only** (where he was picked), one from **college production and athletic testing only**, and one from **both**. They land close together, and that is the honest finding — at this sample, college production and testing added no measured edge beyond draft position. Backtested on the 2019–2023 classes, not live-validated.

Beside those sit the rookie-year **season-total projections** for RB, WR and TE, next to Sleeper's projection and the difference. The quarterback rookie arm was built and then held back as too thin — a rookie QB's season hinges mostly on whether he starts, which the features cannot see — so rookie QBs show no projection.

The page also carries four collapsed lists of **college players who are not in this year's rookie class** — same College Talent instrument, same scale. Most are still in college. They appear on no rookie board and the list says nothing about whether or when any of them will be drafted.
        """)

    with st.expander("What is the Film Room page?"):
        st.markdown("""
The Film Room page collects short, model-backed video breakdowns. Open the page and the newest episode is already loaded. Pick another title from the list to swap the player; only one TikTok embed is on the page at a time. Click **📖 Full breakdown** under the player to open the write-up the short couldn't fit.

Some older videos predate my validation work and make calls I wouldn't make today. Those carry a **📼 Archived: why?** pop-out explaining what's changed; they stay up, unedited, as part of the record, and point you to what I publish now.
        """)

    st.divider()

    # ── Section 3: Fantasy Projections ───────────────────────────────────────
    st.subheader("🏆 Fantasy Projections")

    with st.expander("How do the fantasy projections work?"):
        st.markdown("""
The Weekly Fantasy page uses a separate machine learning system from the betting model. There are four XGBoost models — one for each position (QB, RB, WR, TE) — each trained on NFL player stats from 2020 through 2024 with the 2025 season held out as a real-world test.

Each model predicts **half-PPR fantasy points** for the upcoming week based on roughly 80 features, including:

- The player's recent production (3 and 5-game rolling averages for targets, carries, receiving yards, etc.)
- Their team's offensive efficiency (EPA per play, yards per play, red zone rate)
- The opponent's defensive quality (EPA allowed, pass rate faced, red zone defense)
- Vegas implied team total — how many points Vegas expects the team to score
- Injury and availability status for the player and their key teammates
- Depth chart position
- Home/away split, weather, and surface

The models are retrained each offseason as more data becomes available.
        """)

    with st.expander("How accurate are the fantasy projections?"):
        st.markdown("""
The models were evaluated on the full 2025 holdout season against a simple 3-week rolling average baseline:

| Position | Model MAE | Baseline MAE | Improvement |
|----------|-----------|--------------|-------------|
| QB | 7.0 pts | 7.5 pts | ✅ Better |
| RB | 4.5 pts | 4.6 pts | ✅ Better |
| WR | 3.9 pts | 4.1 pts | ✅ Better |
| TE | 3.2 pts | 3.5 pts | ✅ Better |

MAE (Mean Absolute Error) is the average number of points the projection was off by. So for WR, the model was off by about 3.9 points on average. Given the inherent variance in fantasy football, this is a reasonable result — but any individual week can be much higher or lower.

The projections are most useful as a relative ranking tool rather than a precise point forecast. A player projected at 18 points is likely to outscore one projected at 10, but the exact numbers should be treated as estimates.
        """)

    with st.expander("What are the prop stat columns?"):
        st.markdown("""
In addition to projected fantasy points, each position tab shows position-specific stat projections from eight separate XGBoost models:

| Column | Position | What it predicts |
|--------|----------|-----------------|
| Proj Pass Yds | QB | Passing yards |
| Proj Rush Yds | QB / RB | Rushing yards |
| Proj Rec Yds | RB / WR / TE | Receiving yards |
| Proj Receptions | WR / TE | Number of receptions |

These prop stat models were trained on the same data as the main models but with each individual stat as the target. They're useful as a rough reference when looking at player prop bets on sportsbooks (e.g. over/under pass yards, reception totals).

A few things to keep in mind:
- The prop projections are **independent** models — their values won't perfectly add up to the fantasy point total
- QB passing yards has the highest error (~70 yards off on average), so treat it as directional
- RB and TE receiving yards are the most accurate prop models (~10–14 yards MAE)
        """)

    with st.expander("What do the column headers mean?"):
        st.markdown("""
**Player** — Player name and their NFL team.

**Opponent** — This week's opponent. `@` means away game, `vs` means home game.

**Proj Pts** — Projected half-PPR fantasy points. Half-PPR scoring: 0.5 pts per reception, 1 pt per 10 rush or receiving yards, 6 pts per TD.

**Off EPA** — The team's offensive efficiency over the last 4 games, measured in Expected Points Added per play. Higher is better. See "What is Off EPA?" below for a full explanation.

**EPA Rank** — Where the team's offense ranks among all 32 teams this season (1st = best, 32nd = worst). Color-coded green to red.

**Team Total** — Vegas implied team total: how many points Vegas expects this team to score. Higher means more expected scoring opportunity for that team's players.

**Health** — The player's injury status from the NFL injury report: ✅ Healthy · 🟡 Questionable · ⚠️ Doubtful · ❌ Out. Players officially ruled Out are removed from the projections entirely.

**Actual Pts / Actual [stat]** — Once the week's games are played, actual fantasy points and stats fill in automatically. A blank cell means the player did not play (DNP) in that game.
        """)

    with st.expander("What is Off EPA?"):
        st.markdown("""
**Off EPA** stands for Offensive Expected Points Added per play, averaged over the team's last 4 games.

EPA measures how much each play moves the needle toward scoring. A 5-yard gain on 3rd and 4 is worth a lot more EPA than a 5-yard gain on 1st and 10. So EPA per play is a better measure of offensive efficiency than yards or points, because it accounts for down, distance, and field position.

- **Positive (e.g. +0.15)** — the offense has been efficient recently, generating more value per play than expected
- **Near zero (e.g. +0.01)** — average offense
- **Negative (e.g. -0.12)** — the offense has been struggling

League average hovers near 0. Values above +0.10 are strong, below -0.10 are poor.

This matters for fantasy because players on efficient offenses tend to see more opportunities in positive game scripts and convert them at a higher rate. It's one of the stronger predictors in the model for every position.
        """)

    with st.expander("How often do fantasy projections update?"):
        st.markdown("""
Fantasy projections are generated separately from the weekly betting GitHub Action. That job only papermills the spread and totals notebooks. Weekly fantasy is not on that Tuesday cron yet.

The projection file for each week is saved once and does not change after that. It reflects the injury and depth chart data available at the time it was run. Actual stats fill in automatically after each game is played, pulling live from nflreadpy and caching for 1 hour.

If you're looking at a past week, the actuals shown are the real NFL stats for that game.
        """)

    st.divider()

    # ── Section 3b: NFL & College Talent Score (Draft Board columns) ──────────
    st.subheader("🧮 NFL Talent Score & College Talent Score")

    with st.expander("What are the two score columns on the Draft Board?"):
        st.markdown("""
The Draft Board carries two context columns I build myself, answering two different questions.

**The NFL Talent Score** is my model-based estimate of what a player does with each opportunity — each carry, route, or throw — separated from his situation where that separation is statistically possible. It is not a summary of his production, and models can be wrong. Volume is excluded by design: how often a player is used tells you about his coach's plans, not his per-play skill. Every position reads its own dedicated build, scored against qualified starters at that position. A player below his position's volume floor is left **blank** rather than quietly placed on another position's scale.

**The College Talent Score** is a college-production read for 2026 rookies at all four positions — QB, RB, WR and TE — each from its own dedicated college build, scaled against past prospects at that position who reached the NFL. It describes what a prospect did in college; it does not claim to predict NFL careers or fantasy outcomes.

**Two limits on the college side, stated plainly.** There is no strength-of-schedule adjustment: production against a weaker opponent counts exactly the same as production against a stronger one, which is why several of the highest college scores belong to small-school players the draft market rated far lower. And the underlying data covers FBS only, so a prospect from a smaller division can never be scored — that blank is by construction, not a missing lookup.

**They are two different scales.** The NFL column ranks NFL players against NFL players; the college column ranks prospects against past prospects. A 90 in one is not a 90 in the other, and neither feeds any other number on this board.
        """)

    with st.expander("How the talent scores are built (and what they don't measure)"):
        st.markdown("""
There are eight builds behind these two columns — one for each of NFL and college, at each of the four positions. Each takes a small set of per-opportunity measures I call facets (broken tackles per carry, yards per route run, completion rate versus expectation, and so on), scores every player against his own position in his own season, and then shrinks each measure toward the position average according to how much data sits behind it. A thin sample gets pulled toward the middle rather than being trusted at face value.

That shrinkage has a consequence worth stating: **a facet measured on very few plays contributes far less than its nominal weight suggests.** Contested catches are the clearest case — a tight end sees a handful of them a season, so that facet ends up carrying a few percent of the score no matter what weight I assign it. I measured this before reading any results, and where I tried to compensate by raising the weight, the players the facet exists to reward generally got *worse*, not better. So the weights ship as ratified.

Quarterbacks are the asterisk on the NFL side: one starter per team means a QB's situation cannot be separated from him, so QB scores ship **unadjusted** — a different kind of estimate under the same header. The QB build does now measure performance under pressure, though on a small enough sample that it contributes little. A consequence I have not engineered away: an immobile quarterback cannot score well here, because designed rushing carries a quarter of the composite.

Recent seasons count more, on a decay I chose and wrote down rather than fitted. Scores are clipped to a 50–99 display range (40–99 for college quarterbacks), so **50 is the floor of the display, not a league-average player**. Ranks are a more reliable read than any single number.

These columns are context only: a pre-registered test found that efficiency measures like these do not predict where the draft market misprices players, so they never combine with the projections, the ranks, or the gap columns anywhere on the board. The college instruments in particular were each measured against NFL outcomes and each came back **dead** — they ship as description of college production, and nothing more.

The full write-up covering every design choice, the admission gates, and where it fails lives in my research notes.
        """)

    st.divider()

    # ── Section 4: DFS Optimizer ──────────────────────────────────────────────
    st.subheader("🎯 DFS Optimizer")

    with st.expander("What is the DFS Optimizer?"):
        st.markdown("""
**Coming soon** until 2026 Week 1. The page is not live yet. This is how it will work when it is.

It will take this site's weekly fantasy projections and solve for the highest-projected legal lineup under the $50,000 salary cap using an integer linear program. The optimizer fills all 9 roster slots (QB, 2 RB, 3 WR, TE, FLEX, DST) subject to DraftKings' constraints.

The planned weekly workflow:
1. Download your DraftKings salary CSV from any NFL Classic contest lobby
2. Upload it in the DFS Optimizer page
3. The optimizer fuzzy-matches DK player names to my projected points and solves the lineup
4. Lock or exclude specific players and re-run if you want to tweak it
5. Download the finished lineup ready for DraftKings import

DST will use DraftKings' season average until there is a team-defense projection model. That limitation will be listed on the page.
        """)

    with st.expander("How does the optimizer actually work?"):
        st.markdown("""
Under the hood it's an integer linear program (ILP) solved with the PuLP library.

The optimizer treats each player as a binary variable — either in the lineup (1) or out (0) — and maximizes total projected points subject to hard constraints:

- Exactly 1 QB
- At least 2 RBs
- At least 3 WRs
- At least 1 TE
- Exactly 1 DST
- Exactly 9 total players (the FLEX slot is filled implicitly by the solver)
- Total salary ≤ $50,000
- No more than 8 players from the same team

The solver finds the globally optimal combination given those constraints in under a second. It's not greedy — it considers every valid roster combination simultaneously.

Projections are converted to full DraftKings Classic scoring (full PPR, milestone bonuses for 300+ passing yards, 100+ rushing yards, 100+ receiving yards).
        """)

    st.divider()

    # ── Section 5: League History ─────────────────────────────────────────────
    st.subheader("🏅 League History")

    with st.expander("What is the League History page?"):
        st.markdown("""
The League History page pulls your Sleeper fantasy league's historical data and displays it in one place.

Enter your Sleeper league ID (found in your league's URL: `sleeper.com/leagues/{ID}/league`) and select Load. The status panel lists the linked seasons, then names each year while standings, drafts, and weekly scores load. First load is usually about 2-4 seconds per season; the same ID is instant for an hour after that.

You can filter by season or view all-time records across every year your league has existed. It's useful for settling debates about who's actually been the best manager historically versus just the most recent champion.

**Draft & Roster Insights** opens first and adds three chart-first views in this order: My Team (default), Best Values, and Draft Room. On All Time, the insight window is Last season, Last 3 seasons (default), or All available seasons. My Team shows lineup scoring and season facts after four rostered weeks per player-season. Best Values plots drafted players by round. $5+ bids that scored sit on a scatter. Cheap waiver claims ($0-$4) sit beside $5+ bids that scored zero, both as ranked bars so neither piles on the scatter. Free-agent adds stay off the FAAB charts. Cheap waiver charts only include players with four rostered weeks in that season. Trades are graded as got-versus-gave starting-lineup points from the week after the deal. The other manager is the row label. If a manager has many trades, the chart keeps the eight most lopsided results. Draft Room shows positional timing, runs, backup QB/TE usage and manager tendencies.

The **All-Time Leaderboard** replaces the old table-first records view with headline leaders, a win-rate-versus-adjusted-scoring map, and a ranked win-rate chart. Eight cards in two rows of four: Most Titles, Most Finals Appearances, Longest Active Playoff Streak, Best Win %, Most Points, Most Toilet Bowl Titles, Most Toilet Bracket Appearances, and Lowest Scoring Team. Each card has an info icon. One or two count leaders share a card by name. Three or more show as an N-way tie, with the names under the cards. Most Points is the sum of regular-season weekly scores in the window, excluding playoffs, and is not era-adjusted. Most Finals Appearances counts championship games as champ or runner-up. Longest Active Playoff Streak counts consecutive playoff seasons through the latest completed postseason in the window. An in-progress year does not reset it. Toilet Bowl Titles count last place in that season's consolation (losers) bracket, using the same tie rules as championship titles. Toilet Bracket Appearances count seasons in that consolation bracket. Lowest Scoring Team is regular-season points per game. All Time needs more than 2 seasons, so a one-year disaster does not take it. The map still uses weekly scoring versus that week's league average.

The **Hall of Fame** is a chart-first league record book. All eight record cards sit at the top in two rows of four, each with an info icon. A caption under the cards places the high score against that season's league average. Its Chaos Map shows every matchup by combined score and victory margin. Luckiest Win uses all-play probability, the share of teams the winner would have beaten that week, rather than assuming the lowest winning score was automatically the luckiest.

The **Rivalries** tab has three focused views: **Build a Week**, **Explore a Matchup**, and **League Matrix**. Only the selected view is shown. The builder returns the single best full-week slate for current managers in Classic Rivalries, Maximum Drama, or Fresh Blood mode, with an optional history window. Classic Rivalries uses the longest series and playoff history. Maximum Drama favors close, back-and-forth games. Fresh Blood pairs managers who rarely play each other and have similar records. A sentence at the bottom of Build a Week says how that style's 0-100 rivalry score is built. The explorer compares one series by record, average scoring edge, playoff meetings, current streak, and game margins. The matrix shows each row manager's record against every opponent; green favors the row manager and rose favors the column opponent. Build-a-Week cards use green for 70+ fit, yellow for 50-69, and red below 50.

The **Report Cards** tab gives each manager transparent league ranks for head-to-head winning, scoring relative to the weekly league average, and consistency. Consistency is shown as a plus-or-minus point swing around that manager's own weekly average, not as a standard-deviation abbreviation. Smaller is steadier. All-time mode charts weekly scoring versus that season's league average. Win rate sits on hover and in the season table, not as a second axis. A single-season filter switches to a weekly performance chart. The opponent profile shows average scoring margin versus each opponent, with the record and meeting count on each bar. Complete season and opponent details are collapsed below.

The **Consistency & Luck** tab separates scoring quality, week-to-week volatility, and matchup timing. The four cards are Most Consistent, Most Volatile, Most Fortunate, and Most Unfortunate, each with an info icon. The scatter plots scoring versus week-to-week volatility, with steadier managers at the top. Schedule luck is the bar chart below it, not a color on the scatter. Expected wins come from all-play probability: each score is compared with every other team in that same league-week. Actual wins minus expected wins estimates schedule luck, while adjusted volatility measures how steady a manager remained after accounting for the week's scoring environment.
        """)

    st.divider()

    # ── Section 6: Model explanations ────────────────────────────────────────
    st.subheader("🧠 What Drives the Models")
    st.caption("The current 2026 Draft Board uses 75% independent v6 plus 25% Sleeper's published projection.")

    with st.expander("What inputs drive the 2026 Draft Board projections"):
        st.markdown("""
The Draft Board's **Model Proj** starts from the independent v6 pipeline, then mixes in
Sleeper's published projection. It is a season-total half-PPR forecast, not a ranking
copied from ADP or the talent scores.

For each QB, RB, WR, and TE, three systems estimate three pieces:

1. the probability of playing at least five games;
2. expected games played if he clears that threshold; and
3. expected half-PPR points per game if he clears it.

Each system multiplies those pieces into a season-point estimate. One is deterministic
LightGBM, one is fixed-seed ExtraTrees, and one is Ridge. Their raw estimates are blended
equally. That independent raw blend is then mixed 75/25 with Sleeper's published half-PPR
projection. A position-specific affine calibration is fit using only earlier out-of-fold
mixed predictions and results. That last step maps the mixed raw value back to the
historical point scale without using the season being scored.

The hurdle blend uses 132 cutoff-valid, non-outcome inputs drawn from prior production and usage,
play-by-play and PFF-derived performance, injury history, age and draft context, and available
preseason role, roster, coach, and vacated-usage context. **ADP and the two Talent Scores are
not model inputs.** Sleeper's published projection is mixed in at 25% after the hurdle blend.
It is not one of the 132 columns.

The board currently publishes exactly 180 players: 24 QB, 60 RB, 72 WR, and 24 TE. Model Proj
points and positional ranks are frozen for the current snapshot. Separately, the Draft Board
refreshes Sleeper ADP and Sleeper projection points daily; those live values recalculate the
market ranks and both displayed gap columns, but do not change Model Proj.
        """)

    st.divider()

    # ── Section 7: Behind the Scenes ─────────────────────────────────────────
    st.subheader("🔧 Behind the Scenes")

    with st.expander("How does the prediction model work?"):
        st.markdown("""
The site runs two independent prediction systems: one for the **spread** (ATS picks) and one for the **over/under total**.

**Spread model**

Four models trained on over 3,000 NFL games spanning 11 seasons (2014–2024).

The primary model is the **Ensemble (fixed75)** — a fixed-weight blend of 75% XGBoost and 25% Ridge regression. It sets the predicted edge for each game and determines the sort order.

The three direction voters are **XGBoost**, **Ridge**, and **LightGBM** — three independent models that each predict which side of the spread they favor.

Each game is evaluated by all four models. The consensus tier is assigned based on voter agreement plus Ensemble edge size:

- **HIGH** — all three voters agree on direction *and* the Ensemble edge is 3+ points
- **MEDIUM** — all three voters agree on direction *and* the Ensemble edge is 1+ points (but under 3)
- **PASS** — the voters disagree, or they agree but edge is under 1 point

85 features were engineered, then trimmed to the top 35 via a walk-forward ablation study. The main features are rolling EPA, strength of schedule, All-Pro roster quality, injury impact, QB changes, coaching history, and home field advantage.

**Totals model (experimental)**

A separate two-model system (XGBoost + Ridge) trained to predict whether the final combined score will be over or under the Vegas total line. Uses 35 spread features plus 14 totals-specific inputs (total line, implied team totals, weather, dome status, rolling points, league scoring environment, pace, division game flag).

The CV result (2020–2025, 55.7% on 575 picks) suggests a real UNDER-side edge, consistent with the known retail OVER bias. **But live 2025 results so far (52.2% on 46 picks) are at break-even, not yet confirming the CV.** The 2025 sample is too small to tell — I'm tracking through 2026 before treating these as real picks.

All models are retrained each offseason as new data comes in.
        """)

    with st.expander("What is the LLM agent and what does it do?"):
        st.markdown("""
The agent is built on top of the prediction models using LlamaIndex and Anthropic's Claude API.

**Status: paused as of August 2026.** The agent's line-movement tool was never connected to
a real market feed — it returned hardcoded example values, and the one cached week it
produced stated those as if they were observed sharp-money and line-movement figures. I've
taken that cached analysis down, disabled the weekly agent run, and the site now refuses to
render any market claim that can't prove where it came from. The section below describes
what the agent did; it is not currently producing anything.

It called tools for model predictions, injury reports, historical head to head matchups
going back to 2015, and a model confidence analyzer.

Each week it went through every game, called those tools, and reasoned about whether the
model's prediction was backed up by other signals. It never overrode the model — it asked
whether the rest lined up with what the model was saying.

If the model liked a team, they were healthy, and they dominated the matchup historically,
the agent marked it high confidence. If the model liked a team but their star QB was out,
it would tell you to skip it.

The idea is that raw model predictions are a starting point. The agent adds a layer of reasoning to help filter out plays where the edge might just be noise.
        """)

    with st.expander("How accurate is the model?"):
        _best_week = _completed.groupby(['season','week'])[_acc_col].agg(['sum','count'])
        _best_week['pct'] = _best_week['sum'] / _best_week['count']
        _bw = _best_week['pct'].idxmax() if not _best_week.empty else None
        _bw_str = (f"Season {_bw[0]} Week {_bw[1]} was the strongest week so far at "
                   f"{int(_best_week.loc[_bw,'sum'])} out of {int(_best_week.loc[_bw,'count'])} correct. "
                   ) if _bw else ""
        _hc_line2 = (f" and **{_hc_pct}%** on high confidence picks "
                     f"({_hc_correct}/{_hc_total})" if _hc_pct is not None else "")
        _be_comment2 = breakeven_verdict(_overall_pct, _hc_pct)
        st.markdown(f"""
The model has gone **{_overall_pct}% ATS** across {_overall_total} completed games ({_overall_correct} correct){_hc_line2}. {_be_comment2}

{_bw_str}
I want to be honest though. Past performance doesn't guarantee anything going forward. There will be bad weeks. The goal is to track this over multiple seasons and see if the edge holds up.
        """)

    with st.expander("What data does it use?"):
        st.markdown("""
The model pulls play-by-play and schedule data from nflreadpy going back to 1999. Real weekly injury reports (from `nfl.load_injuries()`) feed directly into the feature set — Out and Doubtful players reduce a team's weighted All-Pro score, which is one of the stronger predictors.

The All-Pro data is a custom CSV covering selections from 1997 to 2025. It's used as a proxy for roster talent: players are weighted over a 3-year lookback (4/2/1) so recent selections matter more. This gets updated manually each January.

Injury data is real, pulled live from nflreadpy, and feeds the model's All-Pro injury impact. The LLM agent is paused (August 2026). Its line-movement tool used hardcoded example values, so that analysis is not on the site.
        """)

    with st.expander("Is this financial advice?"):
        st.markdown("""
No. This is a personal data science project. I built it to explore whether a machine learning model can find a consistent edge against the spread.

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
