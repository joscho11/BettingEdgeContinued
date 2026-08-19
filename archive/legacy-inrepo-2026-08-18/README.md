# Archived in-repo training (2026-08-18)

Moved, not deleted. Live products are sibling repos. Frozen site artifacts stayed put.

| Tree | Why archived | Live source | Still in JoSchoAnalytics |
|---|---|---|---|
| Spread training (`predict_betting.ipynb`, `model_comparison.ipynb`, `retrain_spread_models.py`, `features.ipynb`) | 2026 website spread is `spread_v3_beta` | `cowork_OS/spread_v3_beta/` | `betting/models/` pkls (2025 demo + Help hashes). Do not overwrite. `live_2026.py`, `features.py`, trackers. |
| Seasonal projection training (`fantasy/projections` harnesses, coaching, research) | Website board pulls `projections_v2` | `cowork_OS/projections_v2/` | `fantasy/projections/results/` published CSV, `fantasy/projections/models/` hashed pkls, ADP overlay under `fantasy/seasonal_projections/` |
| Weekly projection training (`predict_fantasy.ipynb`, `retrain_models.py`, `model.ipynb`, `data_pipeline.ipynb`) | New weekly model is `weekly_projections_v2` | `cowork_OS/weekly_projections_v2/` (not wired yet) | `fantasy/fantasy_projections/*.csv` the page still reads, `fantasy/models/` hashed pkls |

Season-totals M4-c (in-JSA `futures/` notebooks) archived separately to
`archive/legacy-futures-m4c-2026-08-18/`. Live page reads
`seasonal_totals_v2_beta` via `futures/published/`.

Do not restore these notebooks onto the live weekly GHA path.
