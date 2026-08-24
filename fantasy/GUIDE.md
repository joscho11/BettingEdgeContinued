# Weekly fantasy projections

Status: checked against the current producer card, site code, and release contract on 2026-08-21.

The current weekly fantasy product is built in `weekly_projections_v2_prod`, a separate private repository. This public repository owns candidate validation, immutable releases, the website, and the frozen 2025 demo.

## Current 2026 system

The production recipe is `l1_wipe_prior_share_ranks`. It uses a gradient-boosted tree with 95 features covering prior usage, opportunity, recent production, team context, opponent context, availability, and position-specific rates. It produces the top 180 player projections for each released week.

Sleeper projections are an evaluation benchmark, not a model feature. The producer can publish without Sleeper data. Benchmark coverage is reported separately so a missing market snapshot cannot change the model inputs.

The producer writes a candidate CSV and SHA-256 sidecar. The public [publishing contract](../publishing/README.md) validates the schema, coverage, source metadata, and sidecar before copying the candidate into `data/releases/builds/`. The site reads the active manifest pointer. Direct writes into `fantasy/fantasy_projections/` are retired for 2026.

No 2026 fantasy week has been graded yet. The manifest remains on the frozen 2025 demo baseline until a valid 2026 candidate is published.

## Locked evaluation

The production card scores a 2025 holdout with training restricted to 2021 through 2024. The comparison below uses the 3,060 top-180 player-weeks that also had Sleeper coverage.

| Metric | JoScho model | Sleeper | Difference |
|---|---:|---:|---:|
| Mean absolute error | 5.003 | 5.188 | Model lower by 0.185 points |
| Within-position, within-week Spearman | 0.397 | 0.402 | Model lower by 0.005 |
| Historical lineup points | 2,016.00 | 2,080.72 | Model lower by 64.72 points |

The model has a modest point-error advantage on the matched sample. It does not beat Sleeper on player ordering or the lineup simulation, so the current evidence does not support a broad superiority claim.

Position results show where the average error improvement comes from:

| Position | Player-weeks | Model MAE | Sleeper MAE | Model rank correlation | Sleeper rank correlation |
|---|---:|---:|---:|---:|---:|
| QB | 408 | 6.373 | 7.081 | 0.294 | 0.266 |
| RB | 1,020 | 4.980 | 4.966 | 0.580 | 0.625 |
| WR | 1,224 | 4.748 | 4.953 | 0.402 | 0.422 |
| TE | 408 | 4.452 | 4.552 | 0.312 | 0.298 |

QB, WR, and TE lower mean absolute error in this holdout. RB does not. Rank correlation is better for QB and TE, and worse for RB and WR. These are historical holdout results, not live 2026 performance.

## Frozen 2025 demo

The public repository keeps Weeks 10 through 17 of the earlier system under `fantasy/fantasy_projections/`. Those CSVs support the demo and reproducibility checks. They are not updated by the 2026 producer. The Weekly Fantasy page hides Out, Doubtful, IR/inactives, and anyone with no box-score row for that week. The frozen files stay byte-identical.

The corrected version-one holdout comparison against a rolling three-game baseline was:

| Position | Model MAE | Rolling-3 MAE | Paired result |
|---|---:|---:|---|
| QB | 6.859 | 7.378 | Difference was distinguishable in the audit |
| RB | 4.489 | 4.578 | No distinguishable difference |
| WR | 4.015 | 4.027 | No distinguishable difference |
| TE | 3.200 | 3.485 | Difference was distinguishable in the audit |

The earlier statement that every position clearly beat the baseline was too strong. Only QB and TE separated from the rolling baseline in that audit.

## Release and grading flow

```text
private producer
    -> candidate CSV + SHA-256 sidecar
    -> public validation and immutable build
    -> active manifest pointer
    -> Weekly Fantasy page
    -> separate result ledger after games finish
```

Published projection files remain immutable. Actual points and errors are written under `data/releases/results/`. This separates what the model knew at release time from what happened later.

## Public repository map

| Path | Role |
|---|---|
| `fantasy_projections/` | Frozen 2025 demo projections |
| `projections/results/` | Published seasonal projection CSVs the Draft Board reads |
| `rookie/board_data/` | Published Rookie Board CSVs |
| `talent/` | NFL and college talent-score CSVs |
| `seasonal_projections/` | Draft Board ADP refresh overlays |
| `../publishing/` | Candidate validation, builds, activation, rollback, and grading |
| `../data/releases/` | Immutable release registry and result ledgers |
| `../site_pages/page_weekly_fantasy.py` | Release-backed weekly fantasy page |

Training for the 2026 weekly producer is in the private `weekly_projections_v2_prod` repository. This public tree does not ship notebooks or serialized models.

## Reproducing the public state

The public repository can verify and render a release. It cannot retrain the private 2026 producer.

```bash
python -m publishing.cli status
python -m publishing.cli validate --artifact path/to/candidate.csv --metadata path/to/candidate.metadata.json
```

Use the build ID and SHA-256 recorded in `data/releases/manifest.json` when tying a displayed week back to a candidate artifact.
