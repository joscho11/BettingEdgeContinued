# When Should You Draft a QB and TE?

*2018-2025 · 1,422 public 12-team 1QB Sleeper redraft leagues · 17,064 teams · regular-season points vs the rest of that league*

**The one-liner:** Waiting on quarterback is the better historical pattern, not a hard rule. Tight end splits into two good paths: pay for an elite name early, or wait until rounds 10-11. The middle of the TE board is the hole. These are group associations on a format-gated public-league sample, not a draft-day script.

---

## 1. The sample

Source panel: `fantasy_qb_te_draft_timing_study`. Parent pool is the 3,641-league draft-order sample. **1,422** league-seasons pass every primary format gate and produce **17,064** team-seasons.

Gates that drop a league: not 1QB (1,661), not redraft or IDP (1,924), TE premium (1,142), keeper picks (115), incomplete board (75), no required TE (59), short draft (295). Counts overlap. 21 teams never drafted a QB. 33 never drafted a TE. Those rows sit outside the timing windows.

Each team is tagged by the **first** QB it drafted and the **first** TE it drafted, then scored on regular-season points inside its own league. Superflex and TE-premium leagues are out on purpose. Mixing them in would answer a different question.

Season counts: 2018 n=5, then 42 / 75 / 109 / 222 / 259 / 367 / 343. 2018 is too thin to lean on. Year-by-year charts in the video start at 2019.

---

## 2. Quarterback

Unadjusted means from `eda_timing_summary.csv`. Points per week are versus that team's league average.

| Window | n | Pts/wk vs league | Top six |
| --- | ---: | ---: | ---: |
| 1-3 | 3,832 | −0.81 | 48.1% |
| 4-5 | 4,257 | −0.05 | 49.4% |
| 6-7 | 3,494 | +0.50 | 50.9% |
| 8-9 | 2,873 | −0.34 | 49.1% |
| 10-11 | 1,757 | +0.84 | 52.9% |
| 12+ | 830 | +1.19 | 54.6% |

12+ minus 1-3 is **1.9976** points per week, about **28.0** over 14 weeks. The video rounded that to a two-point weekly gap and about 28 over a fantasy regular season.

Slot-and-season-adjusted, the 12+ window is still positive: z **+0.117** (95% CI **+0.057 to +0.180**), n=830. The study grades that **exploratory**.

The late-QB edge is not a one-way street. On 2019-2025 z-scores, QB 12+ is positive in **4 of 7** seasons (2019, 2023, 2024, 2025) and negative in 2020, 2021, and 2022. That is the check the video puts on screen. Taking an elite QB who falls past his price stays on the table.

---

## 3. Tight end

Same unadjusted file.

| Window | n | Pts/wk vs league | Top six |
| --- | ---: | ---: | ---: |
| 1-3 | 3,350 | +0.25 | 50.0% |
| 4-5 | 3,842 | −0.60 | 48.0% |
| 6-7 | 3,878 | −0.31 | 50.1% |
| 8-9 | 3,173 | +0.41 | 51.3% |
| 10-11 | 1,692 | +0.79 | 53.1% |
| 12+ | 1,096 | +0.14 | 48.7% |

Rounds 10-11 minus rounds 4-5 is **1.3873** points per week, about **19.4** over 14 weeks. The video rounded that to 1.39 and about 19.

Adjusted, TE 10-11 is z **+0.078** (95% CI **+0.036 to +0.125**), n=1,692. Graded **moderate**. On 2019-2025, that window is positive in **6 of 7** seasons. The miss is 2022.

**Availability filter (75% of that league's fantasy weeks):** early TE flips ahead of late TE. Available 1-3: n=2,139, z +0.074, top six **52.7%**. Available 10-11: n=1,143, z +0.062, top six **52.3%**. In points per week versus league, available 1-3 is **+0.93** and available 10-11 is **+0.58**. You cannot know who stays healthy on draft day, so the injury-inclusive table stays the headline, and the two-path read (elite early, or 10-11) is the one the video speaks.

---

## 4. What this does not prove

- **A rule you must follow in 2026.** The QB 12+ result is exploratory. It flipped sign in three of seven recent seasons.
- **That an elite QB taken early is a mistake.** The video says the opposite: if he falls past his price, taking him is still viable.
- **That any TE in rounds 10-11 is a smash.** The window is a group mean. The player still has to hit.
- **Superflex, TE-premium, best ball, or dynasty.** Those formats were gated out.
- **Head-to-head wins or playoff outcomes.** Regular-season points versus the rest of that league.
- **A large share of scoring.** Timing-window eta-squared on within-league points z-score is **0.226%** for QB and **0.175%** for TE. Small edges stacked over many leagues, not a season-deciding lever.

---

## 5. Not in the video

A 6x6 QB-by-TE heatmap was built and dropped. 36 cells do not fit a phone frame.

Walk-forward, the training-selected window (not the pooled 12+ / 10-11 headline) was positive in **3 of 6** later seasons at each position. For QB, that picker usually landed on 6-7, not 12+. For TE it bounced between 1-3 and 10-11. That is why the video treats the late-QB preference as general, not as a required round.

---

*Sources: `fantasy_qb_te_draft_timing_study` artifacts `panel_manifest.json`, `eda_timing_summary.csv`, `eda_timing_by_season.csv`, `eda_timing_window_associations.csv`, `availability_timing_summary.csv`, `recommendations.csv`, `validation_forward_seasons.csv`. Posted TikTok id `7674314953565670687`, 2026-08-15. This is football analysis, not betting advice.*
