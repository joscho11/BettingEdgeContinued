# 2026 Draft Board Model Proj import

The dashboard's 2026 Draft Board reads
`independent_half_ppr_points_2026.csv` as its Model Proj source. The file is a
copied, read-only publication artifact from `projections_v2/independent_model`.
The dashboard does not train, re-score, calibrate, or alter it.

- Published number: 75% independent `equal_hurdle_blend_v6` raw plus 25%
  Sleeper published half-PPR projection, then rolling affine calibration.
- Independent research baseline (unchanged): `equal_hurdle_blend_v6`. Equal 1/3
  raw LightGBM, ExtraTrees, and Ridge hurdle blend. 132 cutoff-valid non-outcome
  features. ADP is not a model input. Sleeper is not one of the 132 columns.
- Frozen v6 2026 board snapshot:
  `projections_v2/independent_model/outputs/independent_half_ppr_points_2026_equal_hurdle_blend_v6.csv`
  SHA-256 `2887cba971abd35bc3162906b3a057a3ef28b1bbf97f20e9f81615189f8114f7`.
- Published universe: 180 rows (24 QB, 60 RB, 72 WR, and 24 TE). 2026 Sleeper
  coverage on that universe is 180/180.
- Historical 2021-25, published mix: MAE 49.312, pairwise .710126, RMSE 63.090,
  top-six 53/120. Beats ADP ordering in 2021-2025 (5 of 6). Loses 2020, when
  Sleeper projections are empty.
- Historical 2021-25, independent v6 alone: MAE 51.967, pairwise .689210,
  RMSE 66.359, top-six 47/120. Beats ADP in 2023 only (1 of 6).
- ADP on the same 2021-25 panel: pairwise .696530.
- Published 2026 board CSV SHA-256:
  `9d7da8f0a28e0ceb98f525fe93d54548d9fc1cd69173aa5c0eb646ada580e761`.
- Imported dashboard CSV SHA-256 (same bytes):
  `9d7da8f0a28e0ceb98f525fe93d54548d9fc1cd69173aa5c0eb646ada580e761`.
- Independent v6 OOF SHA-256 (not the board CSV):
  `6a4c48f4cc10a4e65abe5d9f3b651a034210e37a8d809bd40ea4537fe6fd7a37`.

The import was refreshed on 2026-08-13 when Joseph put the 25% Sleeper mix on
the website and kept v6 as the independent model. This does not change the
180-player universe or the daily Sleeper ADP overlay.

Model Proj points and ranks are frozen until the planned dated early-September
2026 public-information snapshot. The dashboard separately refreshes Sleeper ADP
and Sleeper projection points daily over these exact 180 players, then recomputes
live Sleeper ranks plus Sleeper Gap and Model Gap.
