# League History backlog

## Cloud load crash (2026-08-15)

- Streamlit Cloud can copy a new `page_league_history.py` into a live process
  without restarting, while `league_insights_view` stays pinned. The page then
  called four arguments into a three-argument `render`. That is a TypeError at
  the call site, and Cloud redacts the message. Fix: reload stale `site_pages`
  helpers plus `fantasy.league_intelligence` before the page runs.

## Acquisition history

- Best Values loads waiver and trade transactions when that view is opened (cached). There is
  no second Load button. $5+ bids that scored sit on a full-width scatter (mustard `#C4A35A`).
  Cheap claims ($0-$4) sit beside zero-point $5+ bids, both as ranked bars. Cheap claims
  require four rostered weeks. Free-agent adds are not on the FAAB charts.
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
- Leaderboard scorecards are Most Titles, Most Finals Appearances, Longest Active Playoff Streak, Best Win %, and Most Points, each with
  an info icon. Three or more title, finals, or active-streak leaders render as an N-way tie, with names in
  the caption. Most Points is regular-season weekly scores only, not era-adjusted.
  Active playoff streak counts consecutive playoff seasons through the latest completed
  postseason in the window. An in-progress season does not reset it.
- Hall of Fame record cards each have an info icon. All eight sit at the top in two rows
  of four. Luckiest Win is all-play (share of teams the winner would have beaten that
  week), not the lowest winning score.
- Chaos Map diamonds mark every Hall of Fame scorecard game, not only blowout, closest,
  and highest total. One game that holds several records shares a joined label.
- Hall of Fame no longer has a season scoring-range chart. A caption under the cards
  places the high score against that season's league average. The Score Trends tab is gone.
  Era context is that caption plus Report Cards scoring versus the weekly league average.
- Rivalry Week Builder shows the single best slate for the selected style. No colored
  style chip, no matchup locks, and no Generate another slate. Style definitions sit
  under the Slate style menu.
- Build a Week ends with one sentence on how that style's 0-100 rivalry score is built.
- Report Cards Consistency shows ±pts around that manager's own weekly average, not "SD".
- Season trajectory is scoring vs league only. Win rate is hover plus the season table, not a second axis.
- Opponent bars show record and meeting count. Copy names the sample size instead of calling a scoring margin a matchup identity.
- Consistency & Luck cards each have an (i) definition. Volatility is the swing vs that week's league average. Fortune is actual wins minus all-play expected wins.
- Consistency scatter is scoring vs volatility only, steadier at the top. Luck stays on the bar chart and on hover.
- Score Trends tab removed. Nothing from it moved.
- Report Cards chart caption: All-time is season-by-season scoring vs league. One season switches to weekly.
