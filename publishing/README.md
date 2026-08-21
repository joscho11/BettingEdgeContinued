# Weekly publication contract

Status: checked against the current CLI, validators, manifest, and grading workflow on 2026-08-21.
At this check, the active public baselines are the 2025 Week 10 demos, Weeks 10-17 are registered,
and no 2026 release has been graded.

This package is the only public write boundary for spread predictions and weekly
fantasy projections. Producer repositories write candidates to their own `outputs/`
directories; the website validates, snapshots, and activates them. No new top-level
folder under `cowork_OS` is required.

## Release lifecycle

1. `spread_v3_prod` or `weekly_projections_v2_prod` writes a weekly CSV plus
   `<artifact>.metadata.json` sidecar.
2. `publishing.cli publish` verifies the schema, season/week, identifiers,
   duplicates, finite values, model version, SHA-256, row and coverage claims,
   timezone-aware production time, and exact NFL schedule coverage.
3. A passing candidate is copied into an immutable build directory under
   `data/releases/builds/`. Only then does the manifest's active pointer move.
4. Weekly Predictions and Weekly Fantasy use that active pointer as their default.
   Track Record stays on 2025 until a 2026 prediction release has at least one final
   graded game.
5. `.github/workflows/grade_releases.yml` polls after game windows. Results are
   written separately under `data/releases/results/`; the released prediction or
   projection snapshot is never edited.

The checked-in manifest also drives the `Published`, `Scheduled`, and
`Awaiting projections` badges. A missing, malformed, or hash-mismatched
artifact fails closed and cannot become the default.

## Publish a producer candidate

Run from the JoSchoAnalytics directory. If `--schedule` is omitted for a live
candidate, the CLI obtains the season schedule through `nflreadpy` before validating.

```powershell
python -m publishing.cli publish `
  --artifact ..\spread_v3_prod\outputs\predictions_2026_week01.csv `
  --metadata ..\spread_v3_prod\outputs\predictions_2026_week01.metadata.json

python -m publishing.cli publish `
  --artifact ..\weekly_projections_v2_prod\independent_model\outputs\projections_2026_week01.csv `
  --metadata ..\weekly_projections_v2_prod\independent_model\outputs\projections_2026_week01.metadata.json
```

A validation failure returns exit code 1 and leaves the active and previous pointers
unchanged. `--no-activate` registers a passing build without moving the default.

### Per-game matchup details

Prediction producers may pass a matchup JSON file when generating the sidecar. The
metadata records its filename in `matchup_artifact_name` and its content hash in
`matchup_artifact_sha256`. The optional artifact is a JSON object with a `games`
object keyed by `game_id`; each game must contain model inputs,
local driver attribution, sourced injury and weather context, and social metadata.

When present, the publisher validates exact game coverage and content, verifies its
SHA-256, copies it into the immutable release build, and records it in the release
manifest. A declared artifact that is missing, incomplete, or hash-mismatched blocks
publication. Live prediction releases without one still publish with a warning so
the existing Week 1 producer is not broken; their matchup pages explicitly mark the
unavailable sections instead of substituting invented data.

```powershell
python -m publishing.cli sidecar `
  --product predictions `
  --artifact ..\spread_v3_prod\outputs\predictions_2026_week01.csv `
  --matchups ..\spread_v3_prod\outputs\matchups_2026_week01.json `
  --season 2026 --week 1 `
  --model-version spread-v3.1 `
  --out ..\spread_v3_prod\outputs\predictions_2026_week01.metadata.json
```

The current historical demo covers the 105 released games in 2025 Weeks 10–16.
It uses the archived final injury reports and observed kickoff-hour weather that
exist in this workspace. Because those prediction releases were backfilled, their
pages label the timestamp as an archive build time rather than a pregame freeze.
The original input matrix was not serialized, but the exact generating notebook and
model blobs remain in git. `scripts/build_matchup_demo_model_context.py` rebuilds all
35 inputs, verifies all 105 rounded ensemble margins against the immutable releases,
and generates local 0.75 XGBoost TreeSHAP + 0.25 Ridge contributions. The pages label
these explanations as post-hoc reconstruction rather than contemporaneous snapshots.

## Status, scheduling, grading, and rollback

```powershell
python -m publishing.cli status
python -m publishing.cli schedule --product fantasy --season 2026 --week 1
python -m publishing.cli grade-published
python -m publishing.cli rollback --product predictions
python -m publishing.cli rollback --product fantasy --build-id <build-id>
```

Rollback only changes the manifest pointer and re-verifies the stored artifact hash.
It does not copy over or delete any build. Regrading unchanged inputs is idempotent,
so the scheduled workflow creates no commit when results have not changed.

## Bootstrap baseline

`python -m publishing.cli bootstrap` registers validated immutable copies of the
2025 Weeks 10–17 demo data, activates Week 10 for both products, schedules the
prediction Week 1 target, and leaves fantasy Week 1 awaiting its candidate. It does
not modify the legacy tracker or fantasy CSVs.
