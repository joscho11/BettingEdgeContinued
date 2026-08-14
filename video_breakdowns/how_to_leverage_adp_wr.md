# How to Leverage ADP: WR Edition

*2026 · two receivers a round apart, and a 5+ disagreement both ways*

**The one-liner:** Zay Flowers and DJ Moore sit about a round apart on Sleeper ADP. Both projections disagree with those prices in the same direction, and both disagreements clear five spots. Flowers is the cheap one. Moore is the expensive one. This write-up carries the efficiency evidence the video compressed, including the counters that cut against each call.

The rule is the one from the series opener: `consensus = sign(model_gap) × min(|model_gap|, |sleeper_gap|)`, so the **weaker** of the two disagreements is what gets reported. This instalment does not publish either projection's rank or point total for a named player. The claim is directional.

---

## 1. The two names

Snapshot capture `2026-08-08T184925Z`. Sign-agreeing rows only. Rounds assume 12 teams.

| Player | Team | ADP | Pos | Consensus | Call |
|---|---|---:|---|---:|---|
| Zay Flowers | BAL | 46 | WR20 | **+7** | market too low |
| DJ Moore | BUF | 54 | WR25 | **−11** | market too high |

Both clear the 5+ bar the historical cell uses. Tre Tucker also cleared it on this snapshot and did not ship in this cut.

Yards per route here are **on-field pass snaps**, not charted routes: nflverse participation, receiver in `offense_players` on a pass play, 250+ snaps, n=87. That denominator runs a few percent large, so these sit below the charted numbers public sites publish. Direction and ordering hold.

---

## 2. Zay Flowers: elite rate, Baltimore volume

| | Value | Rank |
|---|---|---|
| Yards per route | **2.73** | **4th of 87** |
| Routes per game | 26.1 | **44th of 87** |
| Route share | 443 of Baltimore's 466 dropbacks | **95%** |
| Red zone targets | **12** | **49th of 62** (80+ tgt) |

The volume is the offense, not a committee. Baltimore ran 466 dropbacks in 2025, 32nd of 32, 27.4 a game, 48.7% pass rate, also last. League-median dropbacks were 34.8 a game. Same 95% role at that volume is **33 routes a game**. At 2.73 yards per snap that is **1,533 yards** against the **1,211** he actually had. The video rounded those to 1,500 and 1,200.

**The counter:** 12 red zone targets is 10.1% of his own targets, 49th of 62 among receivers with 80+ targets. That is why 1,211 yards produced only 5 touchdowns. A new staff is the hope, not the proof. Alignment with the projections is a directional call, not a claim that the red-zone usage has already changed.

---

## 3. DJ Moore: Buffalo already has a WR1

2025 was the worst of his career on rate. Catch rate 59%, **36th of 44** receivers with 80+ targets (median 63%). Yards per route **1.29, 57th of 87**. Yards after the catch per reception **4.4, 24th of 56** on the panel the video used. He changed teams, so Chicago target counts stay out.

Khalil Shakir is the existing connection. Same 2025 season, head to head:

| | Moore | Shakir |
|---|---|---|
| Catch rate | 59% (36th of 44) | **76% (3rd of 44)** |
| Yards per route | 1.29 (57th of 87) | **2.01 (18th of 87)** |
| YAC per catch | 4.4 (24th of 56) | **7.5 (2nd of 56)** |
| Target depth | **11.6 yds** | 3.7 yds |

Moore wins target depth and nothing else. The video rounded that pair to 12 and 4. Shakir's average target sat 5.5 yards downfield in 2024 and 3.7 in 2025. That is a slot job. Moore will play outside. The comparison is still the right one for "does Buffalo already have a proven top target," and it is the wrong one for "what does the outside job pay."

**Not in the video, the outside-job composite.** Top outside receiver by yards in each game Josh Allen started (season aDOT 9.0+, 5+ routes to be eligible): **2.32 yards per route** since Diggs left, on **20.4 routes a game**. Diggs himself ran 34.1. Moore ran 31.1 in 2025. The 2.32 describes a committee. It is also a max-of-N and inflated by construction: the same procedure on the Diggs era *with* Diggs returns 2.64 against his own 2.39. Shakir led the whole team in receiving yards in 16 of those 32 games.

**Team-lag, also not in the recorded VO.** The veteran feature set still carries prior-season team context, so Moore is projected on Chicago's offense. Buffalo threw 31.9 times a game in 2025. Chicago threw 35.3. That is a lower-volume room even before the role fight.

---

## 4. The record, stated carefully

Since 2021 at receiver, drafted top-180, universe A, complete seasons, when both projections disagreed with ADP in the same direction by 5+ spots, the call landed on the right side **53 of 67 times, 79.1%**, Wilson interval [68, 87]. Ties counted in the denominator.

Shuffling finishes scores about 48%. That permutation null is **all-position**. It was never run at a WR-only 5+ bar, so 48% is a study-wide reference, not a receiver-specific chance rate.

Two things this does not say: it does not say my model beats the market (it does not, at any position, and the filter only fires when my numbers *agree* with Sleeper's projections), and it is backtested, not live validated. 2026 is the first live season. The full board, refreshed daily, is on the Draft Board page.

---

## 5. What this does not prove

- **A group rate is not a per-player probability.** 53 of 67 describes 67 historical WR calls. It is not the chance that Flowers or Moore hits.
- **The 1,533-yard line is a volume translation, not a 2026 projection.** Same 95% role at league-median dropbacks, his 2025 rate held fixed.
- **Yards per route here are on-field snaps.** They are not PFF or other charted-route numbers.
- **The chance benchmark is not a receiver number.** No permutation null was computed for WR-only or for the 5+ bar at one position.
- **Alignment with the projections is directional.** Exact ranks and point totals for these two players are not published here.

---

*Sources: snapshot `2026-08-08T184925Z` joined to frozen `wr_projection_2026.csv`; historical cell from `player_season_results.csv`, `drafted_top180` / universe A / complete / WR / 2021-2025; yards per route from nflverse participation on 2025 regular-season pass plays (n=87, 250+ on-field pass snaps); red-zone targets from nflverse pbp, named receiver inside the 20, 80+ target panel n=62. This is football analysis, not betting advice.*
