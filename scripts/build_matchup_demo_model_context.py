"""Reconstruct and verify 2025 W10-W16 per-game model inputs/drivers.

The archived prediction CSV retained model outputs but not the input matrix. The
exact generating notebook and model blobs still exist in git, so this builder loads
them from the pinned release commit, rebuilds every weekly row from the local
nflverse snapshot, and refuses to write unless all retained predictions reproduce.

The generated JSON is runtime-only data for the website. Deployment does not need
git history, the model packages, or the sibling workspace cache.
"""
from __future__ import annotations

import argparse
from io import BytesIO, StringIO
import json
from pathlib import Path
import subprocess
from types import SimpleNamespace

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb


SOURCE_COMMIT = "3666265a9c6e43794e955467b0c00530db00dc99"
WEEKS = tuple(range(10, 17))


class _FrameResult:
    def __init__(self, frame: pd.DataFrame):
        self._frame = frame

    def to_pandas(self) -> pd.DataFrame:
        return self._frame.copy()


def _git_blob(root: Path, path: str) -> bytes:
    return subprocess.check_output(
        ["git", "show", f"{SOURCE_COMMIT}:{path}"],
        cwd=root,
    )


def _load_namespace(root: Path, injuries: pd.DataFrame) -> tuple[dict, dict]:
    notebook = json.loads(_git_blob(root, "betting/predict_betting.ipynb"))
    ensemble = joblib.load(BytesIO(_git_blob(root, "betting/models/ensemble_prod_model.pkl")))
    standalone = joblib.load(BytesIO(_git_blob(root, "betting/models/xgboost_prod_model.pkl")))
    lightgbm = joblib.load(BytesIO(_git_blob(root, "betting/models/lgbm_prod_model.pkl")))

    pipeline = standalone["pipeline"]
    preprocessor = pipeline.named_steps["preprocessor"]
    known_categories = {"roof", "surface"}
    categorical, numeric = None, None
    for _, _, columns in preprocessor.transformers_:
        if set(columns) & known_categories:
            categorical = list(columns)
        else:
            numeric = list(columns)
    if categorical is None or numeric is None:
        raise RuntimeError("historical standalone model feature contract is unreadable")

    def _load_injuries(*, seasons=None, **_):
        frame = injuries
        if seasons is not None:
            frame = frame[pd.to_numeric(frame["season"], errors="coerce").isin(seasons)]
        return _FrameResult(frame)

    loader = SimpleNamespace(load_injuries=_load_injuries)
    namespace = {
        "__name__": "matchup_demo_reconstruction",
        "pd": pd,
        "np": np,
        "nfl": loader,
        "_re": __import__("re"),
        "_ud": __import__("unicodedata"),
        "model_features": categorical + numeric,
        "ens_feat_cols": list(ensemble["feature_cols"]),
        "lgbm_feat_cols": list(lightgbm["feature_cols"]),
    }
    exec("".join(notebook["cells"][22]["source"]), namespace)
    exec("".join(notebook["cells"][28]["source"]), namespace)
    exec("".join(notebook["cells"][31]["source"]), namespace)
    return namespace, {
        "ensemble": ensemble,
        "pipeline": pipeline,
        "lightgbm": lightgbm,
    }


def _allpro(root: Path) -> pd.DataFrame:
    frame = pd.read_csv(
        StringIO(_git_blob(root, "betting/nfl_allpro_1997_2025.csv").decode("utf-8"))
    )
    frame = frame[frame["Team"] != "2TM"].copy()
    frame["Team"] = frame["Team"].replace(
        {
            "STL": "LA", "LAR": "LA", "OAK": "LV", "LVR": "LV",
            "SD": "LAC", "SDG": "LAC", "NWE": "NE", "KAN": "KC",
            "GNB": "GB", "NOR": "NO", "TAM": "TB", "SFO": "SF",
            "ARZ": "ARI", "BLT": "BAL", "CLV": "CLE", "HST": "HOU",
            "JAC": "JAX",
        }
    )
    return frame


def _json_value(value):
    if value is None or pd.isna(value):
        return None
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        return round(float(value), 6)
    return str(value)


def build(root: Path, raw_root: Path) -> dict:
    schedules = pd.read_parquet(raw_root / "nflverse" / "schedules_1999_2025.parquet")
    pbp = pd.read_parquet(raw_root / "nflverse_jsa_cache" / "pbp.parquet")
    injuries = pd.read_parquet(raw_root / "nflverse_jsa_cache" / "injuries.parquet")
    schedules["season"] = pd.to_numeric(schedules["season"], errors="raise").astype(int)
    schedules["week"] = pd.to_numeric(schedules["week"], errors="raise").astype(int)
    full_schedule = schedules[schedules["season"].eq(2025)].copy().reset_index(drop=True)
    coach_history = schedules[schedules["result"].notna()].copy()
    week_margin_lookup = (
        schedules[
            schedules["season"].between(2014, 2022)
            & schedules["game_type"].eq("REG")
            & schedules["result"].notna()
        ]
        .groupby("week")["result"]
        .apply(lambda values: values.abs().mean())
    )
    pbp_run_pass = pbp[
        pbp["play_type"].isin(["run", "pass"])
        & pbp["posteam"].notna()
        & pbp["defteam"].notna()
    ].copy()

    namespace, models = _load_namespace(root, injuries)
    # The historical function imports nflreadpy inside its body. Fetch once and
    # serve the same frame to all seven weekly builds.
    import nflreadpy

    original_ngs = nflreadpy.load_nextgen_stats
    ngs_passing = original_ngs(seasons=[2024], stat_type="passing").to_pandas()

    def _cached_ngs(*, seasons=None, **_):
        frame = ngs_passing
        if seasons is not None:
            frame = frame[pd.to_numeric(frame["season"], errors="coerce").isin(seasons)]
        return _FrameResult(frame)

    nflreadpy.load_nextgen_stats = _cached_ngs
    try:
        features = []
        for week in WEEKS:
            weekly = namespace["build_features"](
                week,
                2025,
                full_schedule,
                pbp_run_pass,
                _allpro(root),
                week_margin_lkp=week_margin_lookup,
                coach_hist_df=coach_history,
            )
            features.append(weekly)
    finally:
        nflreadpy.load_nextgen_stats = original_ngs
    feature_frame = pd.concat(features, ignore_index=True)
    feature_frame = feature_frame[feature_frame["week"].isin(WEEKS)].copy()

    ensemble = models["ensemble"]
    columns = list(ensemble["feature_cols"])
    matrix = namespace["build_numeric_features"](
        feature_frame, columns, ensemble["roof_surface_encoder"]
    )
    scaled = ensemble["scaler"].transform(matrix)
    xgb_prediction = ensemble["xgb_model"].predict(matrix)
    ridge_prediction = ensemble["ridge_model"].predict(scaled)
    blend_prediction = (
        float(ensemble["xgb_weight"]) * xgb_prediction
        + (1.0 - float(ensemble["xgb_weight"])) * ridge_prediction
    )

    released = []
    for path in sorted(
        (root / "data" / "releases" / "builds" / "predictions" / "2025").glob(
            "week*/predictions-2025w*/artifact.csv"
        )
    ):
        frame = pd.read_csv(path)
        released.append(frame[frame["week"].isin(WEEKS)])
    release_frame = pd.concat(released, ignore_index=True).drop_duplicates("game_id", keep="last")
    verification = feature_frame[["game_id"]].copy()
    verification["rebuilt"] = np.round(blend_prediction, 1)
    if verification["game_id"].duplicated().any():
        duplicate_counts = verification["game_id"].value_counts()
        raise RuntimeError(
            "feature reconstruction duplicated games: "
            f"{duplicate_counts[duplicate_counts.gt(1)].to_dict()}"
        )
    verification = verification.merge(
        release_frame[["game_id", "ens_predicted_margin"]], on="game_id", how="left", validate="one_to_one"
    )
    delta = (verification["rebuilt"] - verification["ens_predicted_margin"]).abs()
    if len(verification) != 105 or verification["ens_predicted_margin"].isna().any():
        raise RuntimeError(f"release coverage failed: {len(verification)} reconstructed rows")
    if float(delta.max()) > 1e-6:
        failures = verification.loc[delta.gt(1e-6)].head(10).to_dict(orient="records")
        raise RuntimeError(f"historical predictions do not reproduce: {failures}")

    xgb_contributions = ensemble["xgb_model"].get_booster().predict(
        xgb.DMatrix(matrix, feature_names=columns), pred_contribs=True
    )
    ridge_contributions = scaled * np.asarray(ensemble["ridge_model"].coef_)
    weight = float(ensemble["xgb_weight"])
    combined = weight * xgb_contributions[:, :-1] + (1.0 - weight) * ridge_contributions
    bias = (
        weight * xgb_contributions[:, -1]
        + (1.0 - weight) * float(ensemble["ridge_model"].intercept_)
    )
    reconstructed = bias + combined.sum(axis=1)
    if float(np.max(np.abs(reconstructed - blend_prediction))) > 1e-4:
        raise RuntimeError("local contributions do not add back to the ensemble prediction")

    games = {}
    for index, row in feature_frame.reset_index(drop=True).iterrows():
        ranked = np.argsort(-np.abs(combined[index]))[:8]
        home, away = str(row["home_team"]), str(row["away_team"])
        games[str(row["game_id"])] = {
            "inputs": [
                {"feature": feature, "value": _json_value(row.get(feature))}
                for feature in columns
            ],
            "drivers": [
                {
                    "feature": columns[position],
                    "value": _json_value(row.get(columns[position])),
                    "contribution": round(float(combined[index, position]), 4),
                    "direction": home if combined[index, position] >= 0 else away,
                }
                for position in ranked
            ],
            "explanation_method": "0.75 XGBoost TreeSHAP + 0.25 standardized Ridge additive contribution",
            "base_value": round(float(bias[index]), 4),
            "reconstructed_margin": round(float(reconstructed[index]), 6),
        }
    return {
        "schema_version": 1,
        "season": 2025,
        "weeks": list(WEEKS),
        "source_commit": SOURCE_COMMIT,
        "provenance": (
            "Post-hoc reconstruction from the exact generating notebook and model blobs; "
            "all 105 rounded ensemble margins matched the released archive."
        ),
        "games": games,
    }


def main(argv=None) -> int:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--raw-root",
        type=Path,
        default=root.parent / "workspace" / "nfl" / "raw",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=root / "data" / "matchups" / "model_2025_weeks10_16.json",
    )
    args = parser.parse_args(argv)
    payload = build(root, args.raw_root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"wrote {len(payload['games'])} verified model records to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
