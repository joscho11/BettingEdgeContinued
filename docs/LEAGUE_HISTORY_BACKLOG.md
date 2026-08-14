# League History backlog

## Next: acquisition history and player counts

- Add an explicit **Load acquisition history** action. Fetch Sleeper transactions only
  after that action so the normal history load does not add 18 requests per season.
- Split in-season additions into waiver, free-agent and trade acquisitions. Preserve the
  full trade package so a traded player's production is not presented as zero-cost value.
- Add the requested **player counts** view: count how many qualifying seasons each player
  appeared on the selected manager's roster. A player counts at most once per season even
  if he had multiple roster stints, and a player-season qualifies only after at least four
  distinct rostered weeks.
- Make the four-week threshold configurable while retaining four as the public default.
- Show the counts as a horizontal chart first, with a season/acquisition drill-down rather
  than a table-first presentation.
- Add a current-draft threat map after Sleeper assigns draft order: identify the managers
  between the selected user's turns and apply only tendencies supported by at least three
  completed drafts.

## Definition checks

- Roster weeks include a player's separate stints but deduplicate season/week.
- Player scoring excludes weeks with `matchup_id = null`.
- Lineup production and bench production remain separate.
- Manager identity is keyed by Sleeper user ID so display-name changes do not split history.
- Conclusions show the number of drafts or player-seasons behind them.
