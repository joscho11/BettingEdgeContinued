# Independent V2 dashboard import, 2026

The dashboard's 2026 Draft Board reads
`independent_half_ppr_points_2026.csv` as its independent-model source. The
file is a copied, read-only publication artifact from the separate
`projections_v2/independent_model` project; the dashboard does not train,
re-score, calibrate, or alter it.

- Source pipeline: `equal_hurdle_blend_v6`.
- Architecture: equal 1/3 raw blend of deterministic LightGBM, fixed-seed
  ExtraTrees, and Ridge participation-hurdle forecasts, followed by rolling
  affine calibration.
- Feature set: 132 cutoff-valid non-outcome features. ADP is not a model input.
- Published universe: 180 rows (24 QB, 60 RB, 72 WR, and 24 TE).
- Historical 2021-25 benchmark: 51.967235 MAE, .689210 pairwise accuracy,
  66.358707 RMSE. ADP was better on the same panel: 51.754135 MAE,
  .696551 pairwise, 65.689277 RMSE. The model beats ADP ordering in 2023
  only (1 of 6 seasons).
- Source artifact SHA-256:
  `6a4c48f4cc10a4e65abe5d9f3b651a034210e37a8d809bd40ea4537fe6fd7a37`.
- Imported dashboard CSV SHA-256:
  `2887cba971abd35bc3162906b3a057a3ef28b1bbf97f20e9f81615189f8114f7`.

The import was refreshed on 2026-08-12 when v6 replaced v5. V5 remains
immutable in the projections_v2 registry. This does not change the 180-player
universe or the daily Sleeper ADP overlay.

V6 model points and V6 positional ranks are frozen until the planned dated
early-September 2026 public-information snapshot. The dashboard separately refreshes
Sleeper ADP and Sleeper projection points daily over these exact 180 players, then
recomputes live Sleeper ranks plus Sleeper Gap and Model Gap. That future snapshot
must be preserved with its source files, capture time, player-resolution report, and
cutoff before it replaces this dashboard import.
