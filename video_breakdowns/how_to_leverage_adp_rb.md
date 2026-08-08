# How to Leverage ADP: RB Edition

*2026 · one backfield with two mispriced backs, and the fourteen RBs with a gap*

**The one-liner:** Fourteen running backs clear a three-spot disagreement between the market and both projections, and unlike tight end there is no direction to it: seven say the market is too low, seven say it is too high. The video builds around the strangest pair on the board, **Bucky Irving and Kenny Gainwell, who are about to share the Tampa Bay backfield six rounds apart in price**, with both projections pushing the two prices toward each other.

The rule is the one from the series opener: `consensus = sign(model_gap) × min(|model_gap|, |sleeper_gap|)`, so the **weaker** of the two disagreements is what gets reported.

---

## 1. The fourteen

Snapshot capture `2026-08-06T204952Z`, 70 eligible running backs. Model values joined to the **frozen raw** `rb_projection_2026.csv`, never the analyst display overlay. Talent scores are my own composite indices; a `c` marks a college score shown because the player is below the NFL index's sample floor (not enough NFL carries to grade, which is a limitation of the instrument, not a statement about the player).

| Player | Team | Round (12-tm) | ADP | Sleeper | My model | Call | Talent |
|---|---|---:|---|---|---|---:|---|
| Travis Etienne | NO | 4 | RB19 | RB16 | RB12 | **+3** | 66.9 |
| Bucky Irving | TB | 4 | RB20 | RB23 | RB24 | **−3** | 81.2 |
| D'Andre Swift | CHI | 5 | RB25 | RB19 | RB17 | **+6** | 67.5 |
| RJ Harvey | DEN | 6 | RB28 | RB37 | RB32 | **−4** | 71.9 |
| Jonathon Brooks | CAR | 8 | RB36 | RB42 | RB69 | **−6** | 86.2 c |
| Kenny Gainwell | TB | 10 | RB39 | RB35 | RB27 | **+4** | 68.5 |
| Tyler Allgeier | ARI | 11 | RB41 | RB48 | RB54 | **−7** | 70.2 |
| Brian Robinson | ATL | 12 | RB46 | RB49 | RB57 | **−3** | 65.9 |
| Braelon Allen | NYJ | 15 | RB54 | RB63 | RB64 | **−9** | 62.9 c |
| Tank Bigsby | PHI | 17 | RB58 | RB51 | RB55 | **+3** | 69.6 |
| Kaytron Allen | WAS | 18 | RB61 | RB64 | RB67 | **−3** | 72.9 c |
| Kimani Vidal | LAC | 18 | RB62 | RB57 | RB52 | **+5** | 65.1 |
| Ray Davis | BUF | 19 | RB64 | RB52 | RB56 | **+8** | 77.0 |
| Emanuel Wilson | SEA | 20 | RB67 | RB60 | RB50 | **+7** | 58.6 |

**How the bar changes the board**, over the 70 eligible backs:

| Bar | Qualifying | Market too low | Market too high |
|---|---:|---:|---:|
| 1+ spot | 32 | 17 | 15 |
| 2+ spots | 20 | 10 | 10 |
| **3+ spots** | **14** | **7** | **7** |
| 5+ spots | 7 | 4 | 3 |
| 8+ spots | 2 | 1 | 1 |

That symmetry is the position-level finding. Tight end was five-of-seven in one direction; running back splits evenly at every single bar. A balanced split is what an unbiased error distribution looks like, so the interesting rows here are individual, not positional.

One row the video deliberately does not narrate: **Jonathon Brooks at −6**. His raw model gap is −33, by far the largest on the board, and it is the model pricing two ACL tears and three career games, not an analytics insight. He keeps his row for completeness.

## 2. The Tampa pair

**Bucky Irving, round 4.** In 2024 he led all qualified backs in yards after contact per carry, 3.93, 1st of 46 (per the public PFF player pages; minimum 100 attempts). In 2025 he was last, 2.33, 49th of 49, and his breakaway share (the percent of his rushing yards from runs of 15+) fell from 34.9% to 13.4%, 8th to 44th. His plain yards per carry tell the same story: 5.42 to 3.40 (nflverse), and not one of his ten 2025 games reached his 2024 average. Two independent charting shops agree on the shape: Sportradar's version (via Pro Football Reference) has him 3rd of 46 in 2024 and 43rd of 49 in 2025, so the collapse is robust to who charted it.

The context the video gives in the same breath: he played 10 games, missing weeks 5 through 12 and returning with a Questionable designation, so some or all of that collapse may be the injury rather than the player. That is the honest uncertainty in the row, and it is exactly what the two projections are pricing when they rank him below his round-4 cost.

**Kenny Gainwell, round 10.** The case is efficiency, not just the 73 catches (4th among all RBs in 2025, behind only McCaffrey, Robinson and Gibbs, nflverse). Per route run he gained 1.44 yards, 10th of the 42 backs with 30+ targets, and his yards after contact per attempt was 3.13, 17th of 49. In the video I say plainly that I expect the backfield to feature more Gainwell than people imagine; that is my read, not the board's.

## 3. D'Andre Swift, round 5

The largest positive call on the board at +6. The charting run grade had him 6th of 49 in 2025, and the counter comes from the same source: after contact Swift made 2.93 a carry and Kyle Monangai, behind the same line and the same play calls, made 2.89. Strip the blocking out and the two Chicago backs were nearly the same runner. What separates them is the passing game: Swift's route grade ranked 23rd of the 42 backs with 30+ targets; Monangai's ranked 41st. The passing downs are Swift's, and in the video I side with the projections on him.

## 4. RJ Harvey, round 6

Both projections rank him below his price, and the receiving skill is genuinely there: 1.40 yards per route run, 11th of 42, one slot behind Gainwell. The ground game is the problem. In the same backfield, J.K. Dobbins beat him after contact 3.18 to 2.72 (15th vs 38th) and in breakaway share 33.8% to 24.0% (8th vs 24th), in seven fewer games. Denver then drafted Jonah Coleman, who averaged 3.58 after contact in his final college season, 54th of 165 FBS backs with 100+ attempts. On Coleman the two projections point opposite ways, so he gets no call here, only the roster fact: one more mouth in that room.

## 5. Travis Etienne, the beat that got cut

Etienne's segment was recorded and cut for runtime, so the write-up carries it. Round 4, and both projections rank him above the price at +3. He played all 17 games and led this table in touches with 296. The deeper number is that his improvement came after contact: 2.48 to 3.08 yards after contact per attempt year over year, with missed tackles forced going from 17 to 47. Contact production travels with the player better than scheme stats do, which is why the New Orleans move worries me less than it worries the market. In the video's closing table he carries a checkmark: I side with the projections.

## 6. The record, stated carefully

Since 2021 at running back, when both projections disagreed with ADP in the same direction by 3+ spots, the call landed on the right side **66 of 82 times, 80.5%**, Wilson interval [71, 88]. Shuffling the finishes scores about 48%, so that is the chance benchmark, not 50%. Two things this does not say: it does not say my model beats the market (it does not, at any position, and the whole mechanism only fires when my numbers *agree* with Sleeper's projections), and it is backtested, not live validated; 2026 is the first live season. The full board, refreshed daily, is on the Draft Board page.
