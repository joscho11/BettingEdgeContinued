# League History backlog

## Acquisition history

- Best Values loads waiver and trade transactions when that view is opened (cached). There is
  no second Load button. Cheap claims ($0-$4) are a ranked bar and require four rostered
  weeks in that season. $5+ bids that scored sit on a scatter; zero-point $5+ bids are a
  ranked bar so they do not stack at zero. Free-agent adds are not on the FAAB charts.
- Trades are graded as got-versus-gave starting-lineup points from the week after the deal.
  The other manager is the row label. More than eight trades keeps the most lopsided |net|.
  Picks and FAAB stay in the hover, not converted to points.
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
