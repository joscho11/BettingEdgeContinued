# Archived M4-c season-totals pipeline (2026-08-18)

Moved, not deleted. Live Season Totals page reads leftover mix published from
`cowork_OS/seasonal_totals_v2_beta` into `JoSchoAnalytics/futures/published/`.

| Tree | Why archived | Live source | Still in `futures/` |
|---|---|---|---|
| `season_team_totals/` (notebooks, `m4_engine.py`, `tier_lock.py`) | Site no longer runs Monte Carlo M4-c | `seasonal_totals_v2_beta/` | Capture scripts, `language_fence.py`, `published/` |
| `01_acquire_win_totals.ipynb`, `build_evidence_artifact.py`, `eval_v2.py`, `acquire_v2_snapshots.py`, `build_v2_features.py` | In-JSA model path | same | ESPN / Odds API snapshots |
| `PREREGISTRATION.md`, `PROPOSED_AMENDMENT_FREE_MARKET_ARCHIVE.md`, `PRESEASON_FEATURE_NOTES.md` | Governed the retired pipeline | FINDINGS.md in the sibling project | `FUTURES_CAPTURE_TARGETS.md`, `DATA_SOURCE_NOTES.md` |
| `futures_predictions.csv` plus M4-c artifact JSONs | Old page inputs | `python src/publish_site.py` | `artifacts/espn_futures_snapshot.json`, `artifacts/odds_api_futures_snapshot.json` |

Do not restore these notebooks onto the live Streamlit path.
