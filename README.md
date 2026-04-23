# BettingEdgeContinued
Deployment of model

Week Before Kickoff — Check Your GitHub Actions
Go to your repo → Actions → Weekly NFL Predictions → Run workflow manually with mode = thursday. Watch the logs. If it completes successfully the automation is ready. If it fails you'll see exactly which line errored and we can fix it before the season starts.

The Season Rhythm From Week 1
Monday 9am    → GitHub Actions runs automatically
               Logs week 1 predictions (early lines)
               Updates week 0 results (preseason — nothing to update)

Thursday 9pm  → GitHub Actions refreshes with injury reports

Sunday 7am    → Final predictions locked in

Monday 9am    → Results from Sunday's games filled in automatically
               New week's predictions generated
               Cycle continues all season
The only manual thing you need to do each season is update the All-Pro CSV in January and make sure the automation still runs cleanly before week 1. Everything else is handled by GitHub Actions.