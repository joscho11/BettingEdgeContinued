# Futures capture targets (locked 2026-08-17, amended same day)

Joseph's list of NFL futures to log for **2026 and later seasons**. This is a capture wishlist, not a model change and not a site claim. Gate C stays shut.

**Dropped 2026-08-17:** Super Bowl appearance (same event as AFC/NFC winner). 1-seed / first-round bye.

Two market types, keep them separate in any snapshot:

- **O/U**: a number plus Over and Under prices (win totals; player season yards/TDs/receptions).
- **Winner**: one name pays (Super Bowl, conference, division, season leaders, awards).
- **Yes/No**: make playoffs and miss playoffs. Log miss as its own side only when the book posts it separately from make.
- **Threshold Yes**: player reaches N+ (DK pass yds/TDs, rec yds). Not a substitute for a two-sided O/U.

## Team

1. Super Bowl winner
3. Regular-season win totals (each of 32 teams, O/U)
4. AFC winner
5. NFC winner
6. Division winners (all 8)
8. Make playoffs (each team, Yes/No)
9. Miss playoffs (each team), only if posted as its own market
40. Team to win most regular-season games
41. Team to have fewest regular-season wins
42. Team to go 17-0 in the regular season
43. Last undefeated team
44. Regular-season sacks (threshold Yes; not the sack-leader winner market)
45. Regular-season interceptions (threshold Yes; not the INT-leader winner market)
46. Most regular-season interceptions thrown (winner; QB INTs, not defensive INT leader)

## Player O/U (each listed player the book posts)

10. Regular-season passing yards
11. Regular-season passing TDs
12. Regular-season receiving yards
13. Regular-season receiving TDs
14. Regular-season rushing yards
15. Regular-season rushing TDs
16. Regular-season receptions

## Player season leaders (winner)

17. Passing yards leader
18. Rushing yards leader
19. Receiving yards leader
20. Passing TDs leader
21. Rushing TDs leader
22. Receiving TDs leader
23. Receptions leader
24. Rookie receiving yards leader
25. Rookie passing yards leader
26. Rookie rushing yards leader
27. QB rushing yards leader
28. TE receiving yards leader
29. RB receiving yards leader
30. Sacks leader
31. INT leader

## Awards (winner)

32. MVP
33. Offensive Player of the Year
34. Defensive Player of the Year
35. Offensive Rookie of the Year
36. Defensive Rookie of the Year
37. Coach of the Year
38. Comeback Player of the Year
39. Protector of the Year

## Coverage vs sources (as of 2026-08-17, after ESPN snap)

ESPN logged to `futures/data/espn/nfl_futures.csv` (832 quotes, DraftKings).
Odds API Super Bowl snap remains multi-book. Canonical 2026 win totals are DraftKings-only in `win_totals_2026_named_books.csv` (08-17 featured). FanDuel pastes stay on disk only as fallback (miss playoffs).

| Item | Odds API | ESPN | On disk now |
|---|---|---|---|
| Super Bowl winner | yes | yes (DK) | yes (Odds API, ESPN, DK paste; FD paste leftover) |
| Team win totals | no | no | yes (DK featured 08-17, 32/32 both sides; canonical file is DK-only) |
| AFC/NFC, divisions | no | yes (DK) | yes (ESPN + DK paste 08-17) |
| Most regular-season wins (40) | no | yes (DK) | yes (ESPN + DK paste 08-17) |
| Fewest regular-season wins (41) | no | no | yes (DK paste 08-17) |
| Team to go 17-0 (42) | no | no | yes (DK paste 08-17) |
| Last undefeated team (43) | no | no | yes (DK paste 08-17) |
| Sacks thresholds (44) | no | no | yes (DK 8/10/12/15/20 Yes plus O/U, 08-17) |
| INT thresholds (45) | no | no | yes (DK player milestones 2+/4+, 08-17) |
| Make playoffs | no | no | yes (DK paste 2026-08-17) |
| Miss playoffs | no | no | yes (FanDuel only; DK did not post a separate miss board) |
| Player O/U 10-16 | no | no | 10-16 O/U plus thresholds where posted; Mahomes pass-TD Under missing |
| Yard leaders 17-19 | no | yes | yes (ESPN + DK paste 08-17 for all three) |
| TD / receptions / sacks / INT leaders | no | no | pass/rush/rec TDs + receptions + sacks yes; defensive INT leader (31) empty; INTs thrown (46) yes |
| Rookie / QB-rush / TE-rec / RB-rec leaders | no | no | no |
| Awards 32-39 | no | yes | yes (ESPN) |

Revisit Covers Sports Odds History when 2026 tables fill. Do not scrape sportsbook apps.
Snapshot: `python futures/snapshot_espn_futures.py`
