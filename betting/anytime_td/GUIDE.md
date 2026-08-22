# Anytime TDs

Status: checked against the site board and 2025 overlap numbers on 2026-08-22.

The Anytime TDs tab is a 2025 weeks 10-17 demo. It compares our chance a skill
player scores a rushing or receiving touchdown with the sportsbook Yes price.
Passing touchdowns are out. The page is for fun. It is not a proven edge.
Bet responsibly.

Training lives in the private `td_count_model_beta` repo. This public tree only
ships CSV.

## What the board is

Every skill player the books quoted that week, sorted by our P(TD). It is not a
pick list. A short "we like these" card lost on 2025, and a typical quote is
around one in five, so misses will outnumber hits. That is the bet, not a broken
model.

| Column | Meaning |
|---|---|
| Our P(TD) | Our chance of a rushing or receiving TD |
| Book | Median implied Yes from at least 3 US books, two hours before kickoff |
| vs book | Our probability minus the book, in percentage points. Not a bet |
| Our fair | American odds implied by our P(TD) |
| P(2+) | Chance of two or more rushing or receiving TDs |
| Hit | Did they score a rushing or receiving TD? |

On a phone the grid keeps #, Player, Ours, Book, and Hit. Position tabs swipe.

## How it scored in 2025

Product arm: 34 locked usage features plus that week's Sleeper half-PPR
projection. The anytime price is not an input.

| Slice | Priced rows | Result |
|---|---:|---|
| Full 2025 overlap | 5,310 | Books about 0.08% more accurate (0.13985 vs 0.13996 on the season score) |
| Demo weeks 10-17 | 2,524 | Our numbers were closer in 5 of 8 weeks. Books still won the eight-week total |
| Yes-edge cut | 960 | Lost vs the book. Not a betting record |

Week 18 is out (rest and backups). Sleeper's dump has no freeze timestamp.

## Live 2026

Not on the site yet. When a week is added, Joseph pastes Yes prices about three
hours before kickoff. No Odds API pull for live weeks. Do not mix that clock
with the 2025 two-hour, three-book bar.

## Public files

| Path | Role |
|---|---|
| `anytime_td/` | Frozen 2025 week CSVs plus `meta.json` |
| `../site_pages/page_anytime_td.py` | Comparison board |
| `../tests/test_anytime_td.py` | Offline AppTest |
