# futures/

Live Season Totals numbers come from `cowork_OS/seasonal_totals_v2_beta`.
Publish with `python src/publish_site.py` in that project. This folder keeps
capture scripts and the published copies the Streamlit page reads.

| Path | Role |
|---|---|
| `published/season_totals_2026.csv` | 2026 line_in sheet the page renders |
| `published/evidence.json` | MAE ladder and claim the page renders |
| `language_fence.py` | Banned-token list for page tests |
| `snapshot_espn_futures.py` | ESPN futures capture |
| `snapshot_odds_api_futures.py` | Odds API outrights capture |
| `parse_dk_futures_paste.py` | DraftKings paste parser |
| `FUTURES_CAPTURE_TARGETS.md` | Capture wishlist |
| `DATA_SOURCE_NOTES.md` | How historical win-total files were built |
| `artifacts/espn_futures_snapshot.json` | Last ESPN capture stamp |
| `artifacts/odds_api_futures_snapshot.json` | Last Odds API capture stamp |

The retired M4-c Monte Carlo pipeline (notebooks, `m4_engine.py`, `tier_lock.py`,
`futures_predictions.csv`) lives in
`archive/legacy-futures-m4c-2026-08-18/`. Moved, not deleted.
