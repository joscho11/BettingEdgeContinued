# Independent V2 dashboard import — 2026

The dashboard's 2026 Draft Board reads
`independent_half_ppr_points_2026.csv` as its independent-model source. The
file is a copied, read-only publication artifact from the separate
`projections_v2/independent_model` project; the dashboard does not train,
re-score, calibrate, or alter it.

- Source pipeline: `equal_hurdle_blend_v2`.
- Architecture: equal raw blend of deterministic LightGBM and fixed-seed
  ExtraTrees participation-hurdle forecasts, followed by rolling affine
  calibration.
- Feature set: 132 cutoff-valid non-outcome features. ADP is not a model input.
- Published universe: 180 rows — 24 QB, 60 RB, 72 WR, and 24 TE.
- Historical 2021–25 benchmark: 53.767686 MAE, .679571833 pairwise accuracy,
  68.095667 RMSE. ADP was better on the same panel: 51.754135 MAE,
  .696550876 pairwise, 65.689277 RMSE.
- Source artifact SHA-256:
  `e6385740cce8c5f137915aa8245c7c69d68e4260f6291df64cc7535162342595`.
- Imported dashboard CSV SHA-256:
  `4b90bed049ec0278239b88e57ad4a0d3b28f82ecc487a241e458c8e9d0d06e66`.

V2 model points and V2 positional ranks are frozen until the planned dated
early-September 2026 public-information snapshot. The dashboard separately refreshes
Sleeper ADP and Sleeper projection points daily over these exact 180 players, then
recomputes live Sleeper ranks plus Sleeper Gap and Model Gap. That future V2 snapshot
must be preserved with its source files, capture time, player-resolution report, and
cutoff before it replaces this dashboard import.
