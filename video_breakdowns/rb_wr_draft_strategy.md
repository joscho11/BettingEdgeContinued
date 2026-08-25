# How many RBs should you draft early?

*2018-2025 · 1,371 public Sleeper 12-team 1QB redraft leagues · 16,452 teams · regular-season points vs the rest of that league*

**The one-liner:** Zero RB and one-RB (WR-heavy) drafts lag a two-RB start. Two or three RBs by round 6 is the pattern that held. Forcing a fourth does not help. Round 1 RB vs WR is a wash, so take the best player. These are adjusted associations on a format-gated public-league sample, not a draft-day script.

---

## 1. The sample

Source panel: `fantasy_rb_wr_draft_strategy_study`. **1,371** league-seasons, **16,452** team-seasons, 2018-2025.

Each team is tagged by how many RBs it had drafted through round 6, then scored on regular-season points inside its own league. Superflex and other gated formats stay out.

Strategy counts at round 6: Zero RB n=254, WR-heavy (1 RB) n=2,229, balanced (2 RB) n=8,666, RB-heavy (3+) n=5,303.

---

## 2. Round 6 RB count

Adjusted league-relative points per week vs balanced (2 RB):

| Build through round 6 | n | vs balanced |
| --- | ---: | ---: |
| Zero RB | 254 | **−1.89** pts/wk |
| WR-heavy (1 RB) | 2,229 | **−0.80** pts/wk |
| Balanced (2 RB) | 8,666 | baseline |
| RB-heavy (3+) | 5,303 | z **+0.026** (95% CI **−0.013 to +0.064**) |

The 3+ vs 2 interval includes zero. Exact 3 vs 2 is z **+0.033** (95% CI **−0.006 to +0.072**). Four-plus is slightly negative. The video's read: get to two RBs by round 6; a third is optional; do not force a fourth.

---

## 3. Round 1 is not an RB-vs-WR rule

First-round RB minus WR: z **−0.024** (95% CI **−0.059 to +0.009**). Adjusted top-six rates: RB **49.50%**, WR **51.18%**. First-round unavailability is similar (RB **28.7%**, WR **27.7%**). The round 1 pick is "draft the best player," not a position mandate.

---

## 4. What this does not prove

- **A rule you must follow in 2026.** Walk-forward, the training-selected build was positive in **3 of 6** next seasons. RB-heavy was selected four times and positive in two.
- **That three RBs beats two.** The interval on that contrast includes zero.
- **That zero RB cannot win a league.** n=254 is the thin tail. The mean is worse, not impossible.
- **Superflex, best ball, or dynasty.** Those formats were gated out.
- **Head-to-head wins or playoff outcomes.** Regular-season points versus the rest of that league.
- **Causation.** Managers who take two RBs early also make other choices. The result is an adjusted association.

---

*Sources: `fantasy_rb_wr_draft_strategy_study` artifacts `panel_manifest.json`, `inference_strategy_intervals.csv`, `inference_strategy_contrasts.csv`, `inference_checkpoint_intervals.csv`, `validation_forward_seasons.csv`, `first_round_inference.csv`, `first_round_availability_by_position.csv`. Posted TikTok id `7674717547266133278`, 2026-08-16. This is football analysis, not betting advice.*
