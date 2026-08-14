# Does draft order actually decide your season?

*2018-2025 · 3,641 public 12-team Sleeper snake leagues · regular-season points only*

**The one-liner:** In this convenience sample, earlier snake seats have a small points-for head start, not a season-deciding one. Slots 2-8 finish top-six on points **51.1%** of the time. Slots 9-12 finish top-six **49.0%**. Slot 4 is the strongest seat. Slot 1 is the weakest until you condition on the first-rounder staying available, and even then the gap stays small. The injury-inclusive result is the headline.

This is a descriptive association. Public-league assignment is not a randomized experiment.

---

## 1. The sample

Locked population: 12-team, completed NFL season, season-long (not best ball), standard snake, public Sleeper. Scoring and roster format may vary. Outcome is regular-season points inside each league. Playoffs stay out.

Retained panel: **3,641** league-seasons, **43,692** team-seasons, 2018-2025, **3,641** observations per draft slot. No 2017 candidate passed validation. Season counts: 2018 n=9, then 134 / 367 / 503 / 526 / 601 / 762 / 739.

---

## 2. Observed seats

Source: `eda_slot_summary.csv`. Rates use fractional credit for exact cutoff ties.

| Slot | Points z-score | Mean finish | Top-six | Top scorer |
| ---: | ---: | ---: | ---: | ---: |
| 1 | -0.102 | 6.83 | 46.1% | 7.05% |
| 2 | -0.006 | 6.48 | 49.8% | 7.90% |
| 3 | +0.039 | 6.36 | 51.5% | 8.83% |
| 4 | +0.056 | 6.29 | 52.8% | 9.38% |
| 5 | +0.030 | 6.39 | 51.3% | 8.42% |
| 6 | +0.030 | 6.40 | 50.9% | 9.11% |
| 7 | +0.015 | 6.43 | 51.3% | 9.02% |
| 8 | +0.025 | 6.42 | 50.3% | 8.33% |
| 9 | -0.017 | 6.55 | 49.2% | 7.18% |
| 10 | -0.045 | 6.64 | 48.8% | 8.91% |
| 11 | -0.007 | 6.53 | 49.1% | 8.11% |
| 12 | -0.019 | 6.58 | 49.0% | 7.76% |

Slots 2-8 mean finish **6.40**. Slots 9-12 mean finish **6.57**. Slot 4 minus slot 1 is **6.74** percentage points in top-six on this 3,641-league panel. Do not mix that figure with the **6.80** point gap on the 3,626-league availability panel below.

League-cluster bootstrap, 2,000 replicates, vs a 50% top-six baseline: slot 4 **+2.80** pp (95% CI **+1.13 to +4.45**), slot 1 **−3.94** pp (95% CI **−5.52 to −2.28**). The slot 4 minus slot 1 top-six gap is negative in 2020 and 2021. It is a tie in 2018 (n=9). In 2024, slot 1 top-six was **37.2%** across 762 leagues.

---

## 3. Pick 1 and first-round availability

Mechanism check, not a replacement for the injury-inclusive primary result. Recovers actual Sleeper first-round picks for **3,626 of 3,641** drafts. A first-rounder is "available" if that player appeared in at least **75%** of that league's fantasy regular-season weeks.

- Slot 1 unavailability is **42.3%**, the highest of any seat. Next-worst is slot 10 at 29.8%.
- After dropping unavailable first-rounders, slot 1 top-six rises from **46.1% to 53.3%**. The slot 4 minus slot 1 gap shrinks from **6.80 to 2.84** percentage points.
- The 2-8 vs 9-12 top-six gap moves from **2.1 to 2.7** pp (2.13 to 2.66 unrounded). Pick 1 joins the early group once the injured first-rounders are out.

Consensus 1.01 in this sample, weeks available in that league's fantasy regular season:

| Year | 1.01 | Weeks | Avail |
| --- | --- | --- | ---: |
| 2021 | CMC | 7 of 14 | 50% |
| 2022 | Jonathan Taylor | 10 of 14 | 71% |
| 2023 | Justin Jefferson | 6 of 14 | 43% |
| 2024 | CMC | 4 of 14 | 29% |
| 2025 | Ja'Marr Chase | 12 of 14 | 86% |

2024 CMC is the example the video names: 4 of 14 fantasy weeks.

---

## 4. What this does not prove

- **A causal effect of draft slot.** Public Sleeper assignment is a convenience sample, not a randomized experiment.
- **Best ball, head-to-head wins, or playoff outcomes.** Regular-season points-for only.
- **That snake is "rigged" or that auction is required.** Auction gives every manager a shot at the same names. Snake does not. Familiarity still explains why people run snake.
- **That pick 1 is a cursed seat in every season.** 2024 is the loud year. 2020 and 2021 already flipped the slot 4 minus slot 1 gap.
- **That filtering injuries makes early seats a large edge.** The remaining 2-8 vs 9-12 gap is 2.7 points. Keep the injury-inclusive 3,641-league result as the real-world headline.

---

## 5. Not in the video

A straight line through seats 1-12 vs within-league points z-score has Pearson r **−0.0009** (n = 43,692). That does not say seats are interchangeable. It says the pattern is not linear: slot 1 is still weak and slot 4 is still the peak.

Snake pairs (that seat's round-1 and round-2 Sleeper picks, scored in nflverse PPR inside the league's fantasy window, 3,624 drafts with complete two-round picks) were computed and kept off the cut. Injury-inclusive combined PPR: 1+24 **352.9**, 4+21 **387.8**, 12+13 **380.3**. Both-available: 1+24 **448.9**, 12+13 **446.0**. Those bars are not a claim the video made.

---

*Sources: `fantasy_draft_order_study` retained panel (3,641 league-seasons), `eda_slot_summary.csv`, `inference_season_sensitivity.csv`, `availability_evaluation.json` (3,626 drafts, 75% week threshold). Regular-season points-for as Sleeper records them. This is football analysis, not betting advice.*
