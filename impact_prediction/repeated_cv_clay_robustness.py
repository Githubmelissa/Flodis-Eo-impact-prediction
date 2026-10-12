from pathlib import Path
import warnings

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")


# =====================================================
# PATHS
# =====================================================
EVENT_TABLE_PATH = Path(
    "/home/melissa/flood_project/models_benchmark/impact_prediction/outputs/tabular_clay_pca_fusion_event_table.csv"
)

OUTPUT_DIR = Path(
    "/home/melissa/flood_project/models_benchmark/impact_prediction/outputs"
)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_RESULTS = OUTPUT_DIR / "repeated_cv_clay_displacement_model_results.csv"
OUTPUT_SUMMARY = OUTPUT_DIR / "repeated_cv_clay_displacement_summary.csv"
OUTPUT_PAIRWISE = OUTPUT_DIR / "repeated_cv_clay_displacement_pairwise_comparison.csv"
OUTPUT_PREDS = OUTPUT_DIR / "repeated_cv_clay_displacement_cv_predictions.csv"


# =====================================================
# SETTINGS
# =====================================================
RANDOM_STATE_START = 0
N_SEEDS = 30
N_SPLITS = 5
N_ESTIMATORS = 300

# Start with displacement robustness.
# Later we can add: "total_affected_EMDAT", "total_deaths", "total_damages_000_USD"
TARGETS = [
    "displacements",
]

REFERENCE_CANDIDATE = "worldpop_gpm_srtm"

META_COLS = {
    "GFD_ID",
    "event_id",
    "row_id",
    "ISO3",
    "iso3_list",
    "year",
    "gfd_event_date",
    "during_start",
    "during_end",
    "source_year_file",
    "displacement_records",
    "mortality_damage_records",
}

TARGET_COLS = {
    "displacements",
    "total_deaths",
    "total_affected_EMDAT",
    "total_damages_000_USD",
}


# =====================================================
# METRICS
# =====================================================
def safe_log1p(y):
    y = np.asarray(y, dtype=float)
    y = np.clip(y, 0, None)
    return np.log1p(y)


def safe_expm1(y_log):
    y_log = np.asarray(y_log, dtype=float)
    y_log = np.clip(y_log, 0, 30)
    return np.expm1(y_log)


def rmse(y_true, y_pred):
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def safe_r2(y_true, y_pred):
    try:
        return float(r2_score(y_true, y_pred))
    except Exception:
        return np.nan


def evaluate_predictions(y_true_original, y_pred_log):
    y_true_log = safe_log1p(y_true_original)
    y_pred_original = safe_expm1(y_pred_log)

    return {
        "log_mae": float(mean_absolute_error(y_true_log, y_pred_log)),
        "log_rmse": rmse(y_true_log, y_pred_log),
        "log_r2": safe_r2(y_true_log, y_pred_log),
        "original_mae": float(mean_absolute_error(y_true_original, y_pred_original)),
        "original_rmse": rmse(y_true_original, y_pred_original),
        "original_r2": safe_r2(y_true_original, y_pred_original),
    }


# =====================================================
# FEATURE GROUPS
# =====================================================
def numeric_existing(df, cols):
    return [
        c for c in cols
        if c in df.columns and pd.api.types.is_numeric_dtype(df[c])
    ]


def clean_feature_list(cols):
    forbidden = META_COLS | TARGET_COLS
    return [c for c in cols if c not in forbidden]


def get_feature_groups(df):
    numeric_cols = [
        c for c in df.columns
        if pd.api.types.is_numeric_dtype(df[c])
    ]

    worldpop_cols = [
        c for c in numeric_cols
        if (
            "population" in c.lower()
            or c in ["flood_area_km2"]
            or "flood_area_km2" in c.lower()
        )
    ]

    gpm_cols = [
        c for c in numeric_cols
        if (
            "gpm" in c.lower()
            or "rainfall" in c.lower()
        )
    ]

    srtm_cols = [
        c for c in numeric_cols
        if (
            "elevation" in c.lower()
            or "slope" in c.lower()
        )
    ]

    clay_quality_cols = numeric_existing(
        df,
        [
            "n_total_patches",
            "n_valid_patches",
            "n_skipped_patches",
            "valid_patch_fraction",
        ],
    )

    clay_deep_cols = [
        c for c in numeric_cols
        if (
            c.startswith("clay_before_")
            or c.startswith("clay_after_")
            or c.startswith("clay_delta_")
        )
    ]

    groups = {
        "worldpop": clean_feature_list(worldpop_cols),
        "gpm": clean_feature_list(gpm_cols),
        "srtm": clean_feature_list(srtm_cols),
        "clay_quality": clean_feature_list(clay_quality_cols),
        "clay_deep": clean_feature_list(clay_deep_cols),
    }

    return groups


def build_candidates(groups):
    worldpop = groups["worldpop"]
    gpm = groups["gpm"]
    srtm = groups["srtm"]
    clay_quality = groups["clay_quality"]
    clay_deep = groups["clay_deep"]

    worldpop_gpm = sorted(set(worldpop + gpm))
    worldpop_gpm_srtm = sorted(set(worldpop + gpm + srtm))

    candidates = [
        {
            "candidate": "baseline_median",
            "model": "baseline_median",
            "base_cols": [],
            "clay_deep_cols": [],
            "pca_components": None,
        },
        {
            "candidate": "worldpop_gpm",
            "model": "random_forest",
            "base_cols": worldpop_gpm,
            "clay_deep_cols": [],
            "pca_components": None,
        },
        {
            "candidate": "worldpop_gpm_srtm",
            "model": "random_forest",
            "base_cols": worldpop_gpm_srtm,
            "clay_deep_cols": [],
            "pca_components": None,
        },
        {
            "candidate": "worldpop_gpm_plus_clay_quality",
            "model": "random_forest",
            "base_cols": sorted(set(worldpop_gpm + clay_quality)),
            "clay_deep_cols": [],
            "pca_components": None,
        },
        {
            "candidate": "worldpop_gpm_srtm_plus_clay_quality",
            "model": "random_forest",
            "base_cols": sorted(set(worldpop_gpm_srtm + clay_quality)),
            "clay_deep_cols": [],
            "pca_components": None,
        },
        {
            "candidate": "worldpop_gpm_srtm_plus_clay_pca_4",
            "model": "random_forest",
            "base_cols": worldpop_gpm_srtm,
            "clay_deep_cols": clay_deep,
            "pca_components": 4,
        },
        {
            "candidate": "worldpop_gpm_srtm_plus_clay_quality_pca_4",
            "model": "random_forest",
            "base_cols": sorted(set(worldpop_gpm_srtm + clay_quality)),
            "clay_deep_cols": clay_deep,
            "pca_components": 4,
        },
        {
            "candidate": "worldpop_gpm_srtm_plus_clay_pca_8",
            "model": "random_forest",
            "base_cols": worldpop_gpm_srtm,
            "clay_deep_cols": clay_deep,
            "pca_components": 8,
        },
    ]

    return candidates


# =====================================================
# MODEL PIPELINE
# =====================================================
def make_preprocessor(base_cols, clay_deep_cols, pca_components):
    transformers = []

    if len(base_cols) > 0:
        base_pipe = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
            ]
        )
        transformers.append(("base", base_pipe, base_cols))

    if len(clay_deep_cols) > 0:
        clay_steps = [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]

        if pca_components is not None:
            clay_steps.append(
                ("pca", PCA(n_components=pca_components, random_state=42))
            )

        clay_pipe = Pipeline(steps=clay_steps)
        transformers.append(("clay_deep", clay_pipe, clay_deep_cols))

    if len(transformers) == 0:
        raise ValueError("No feature columns supplied.")

    return ColumnTransformer(transformers=transformers, remainder="drop")


def make_model(candidate, seed):
    if candidate["model"] == "baseline_median":
        return DummyRegressor(strategy="median")

    if candidate["model"] == "random_forest":
        return RandomForestRegressor(
            n_estimators=N_ESTIMATORS,
            random_state=seed,
            min_samples_leaf=2,
            n_jobs=-1,
        )

    raise ValueError(candidate["model"])


def make_pipeline(candidate, seed):
    model = make_model(candidate, seed)

    if candidate["model"] == "baseline_median":
        return model

    preprocessor = make_preprocessor(
        base_cols=candidate["base_cols"],
        clay_deep_cols=candidate["clay_deep_cols"],
        pca_components=candidate["pca_components"],
    )

    return Pipeline(
        steps=[
            ("preprocess", preprocessor),
            ("model", model),
        ]
    )


# =====================================================
# CV
# =====================================================
def run_one_candidate(df, target, candidate, seed):
    labeled = df[df[target].notna()].copy()
    labeled = labeled[labeled[target] >= 0].copy()

    n = len(labeled)

    if n < N_SPLITS:
        return None, []

    y_original = labeled[target].astype(float).values
    y_log = safe_log1p(y_original)

    cv = KFold(n_splits=N_SPLITS, shuffle=True, random_state=seed)

    if candidate["model"] == "baseline_median":
        X = np.zeros((n, 1))
    else:
        feature_cols = list(candidate["base_cols"]) + list(candidate["clay_deep_cols"])

        if len(feature_cols) == 0:
            return None, []

        # PCA cannot have more components than the smallest training fold.
        min_train_size = n - int(np.ceil(n / N_SPLITS))
        if candidate["pca_components"] is not None:
            if candidate["pca_components"] >= min_train_size:
                return None, []

        X = labeled[feature_cols].copy()

    pipe = make_pipeline(candidate, seed)

    try:
        y_pred_log = cross_val_predict(pipe, X, y_log, cv=cv)
    except Exception as e:
        print(
            "SKIP",
            "target=", target,
            "candidate=", candidate["candidate"],
            "seed=", seed,
            "error=", repr(e),
        )
        return None, []

    metrics = evaluate_predictions(y_original, y_pred_log)

    result_row = {
        "target": target,
        "seed": seed,
        "candidate": candidate["candidate"],
        "model": candidate["model"],
        "n_samples": n,
        "n_features_base": len(candidate["base_cols"]),
        "n_features_clay_deep": len(candidate["clay_deep_cols"]),
        "pca_components": np.nan if candidate["pca_components"] is None else candidate["pca_components"],
        **metrics,
    }

    pred_rows = []

    for gfd_id, yt, yp_log in zip(labeled["GFD_ID"], y_original, y_pred_log):
        pred_rows.append(
            {
                "target": target,
                "seed": seed,
                "candidate": candidate["candidate"],
                "model": candidate["model"],
                "GFD_ID": int(gfd_id),
                "y_true": float(yt),
                "y_true_log": float(np.log1p(yt)),
                "y_pred_log": float(yp_log),
                "y_pred": float(safe_expm1([yp_log])[0]),
            }
        )

    return result_row, pred_rows


def build_summary(results_df):
    group_cols = ["target", "candidate", "model"]

    summary = (
        results_df
        .groupby(group_cols)
        .agg(
            n_runs=("seed", "nunique"),
            mean_log_rmse=("log_rmse", "mean"),
            std_log_rmse=("log_rmse", "std"),
            median_log_rmse=("log_rmse", "median"),
            min_log_rmse=("log_rmse", "min"),
            max_log_rmse=("log_rmse", "max"),
            mean_log_r2=("log_r2", "mean"),
            std_log_r2=("log_r2", "std"),
            mean_original_rmse=("original_rmse", "mean"),
            std_original_rmse=("original_rmse", "std"),
        )
        .reset_index()
    )

    winners = results_df.loc[
        results_df.groupby(["target", "seed"])["log_rmse"].idxmin()
    ].copy()

    win_counts = (
        winners
        .groupby(group_cols)
        .size()
        .reset_index(name="win_count")
    )

    summary = summary.merge(win_counts, on=group_cols, how="left")
    summary["win_count"] = summary["win_count"].fillna(0).astype(int)

    summary = summary.sort_values(["target", "mean_log_rmse"])

    return summary


def build_pairwise(results_df):
    rows = []

    for target in sorted(results_df["target"].unique()):
        target_df = results_df[results_df["target"] == target].copy()

        reference = target_df[
            target_df["candidate"] == REFERENCE_CANDIDATE
        ][["seed", "log_rmse"]].rename(columns={"log_rmse": "reference_log_rmse"})

        if len(reference) == 0:
            continue

        for candidate in sorted(target_df["candidate"].unique()):
            if candidate == REFERENCE_CANDIDATE:
                continue

            cand = target_df[target_df["candidate"] == candidate][
                ["seed", "log_rmse"]
            ].rename(columns={"log_rmse": "candidate_log_rmse"})

            merged = reference.merge(cand, on="seed", how="inner")

            if len(merged) == 0:
                continue

            diff = merged["candidate_log_rmse"] - merged["reference_log_rmse"]

            rows.append(
                {
                    "target": target,
                    "reference_candidate": REFERENCE_CANDIDATE,
                    "candidate": candidate,
                    "n_seeds": len(merged),
                    "mean_log_rmse_diff_candidate_minus_reference": float(diff.mean()),
                    "std_log_rmse_diff": float(diff.std()),
                    "median_log_rmse_diff": float(diff.median()),
                    "candidate_better_count": int((diff < 0).sum()),
                    "candidate_better_fraction": float((diff < 0).mean()),
                }
            )

    return pd.DataFrame(rows).sort_values(
        ["target", "mean_log_rmse_diff_candidate_minus_reference"]
    )


# =====================================================
# MAIN
# =====================================================
def main():
    print("=" * 100)
    print("Repeated CV Clay robustness")
    print("Event table:", EVENT_TABLE_PATH)
    print("Seeds:", N_SEEDS)
    print("Splits:", N_SPLITS)
    print("Targets:", TARGETS)

    if not EVENT_TABLE_PATH.exists():
        raise FileNotFoundError(EVENT_TABLE_PATH)

    df = pd.read_csv(EVENT_TABLE_PATH)

    print("Loaded event table:", df.shape)
    print("Unique GFD_ID:", df["GFD_ID"].nunique())

    groups = get_feature_groups(df)

    print()
    print("Feature group sizes:")
    for name, cols in groups.items():
        print(f"  {name}: {len(cols)}")

    candidates = build_candidates(groups)

    print()
    print("Candidates:")
    for c in candidates:
        print(
            " ",
            c["candidate"],
            "| model:",
            c["model"],
            "| base:",
            len(c["base_cols"]),
            "| clay:",
            len(c["clay_deep_cols"]),
            "| pca:",
            c["pca_components"],
        )

    all_results = []
    all_preds = []

    seeds = list(range(RANDOM_STATE_START, RANDOM_STATE_START + N_SEEDS))

    for target in TARGETS:
        if target not in df.columns:
            print("Skipping missing target:", target)
            continue

        print()
        print("=" * 100)
        print("TARGET:", target)
        print("Labels:", int(df[target].notna().sum()), "/", len(df))

        for seed in seeds:
            print()
            print("Seed:", seed)

            for candidate in candidates:
                print("  Running:", candidate["candidate"])

                result_row, pred_rows = run_one_candidate(
                    df=df,
                    target=target,
                    candidate=candidate,
                    seed=seed,
                )

                if result_row is not None:
                    all_results.append(result_row)
                    all_preds.extend(pred_rows)

                # Save continuously
                if len(all_results) > 0:
                    pd.DataFrame(all_results).to_csv(OUTPUT_RESULTS, index=False)

                if len(all_preds) > 0:
                    pd.DataFrame(all_preds).to_csv(OUTPUT_PREDS, index=False)

    results_df = pd.DataFrame(all_results)
    preds_df = pd.DataFrame(all_preds)

    results_df.to_csv(OUTPUT_RESULTS, index=False)
    preds_df.to_csv(OUTPUT_PREDS, index=False)

    summary_df = build_summary(results_df)
    pairwise_df = build_pairwise(results_df)

    summary_df.to_csv(OUTPUT_SUMMARY, index=False)
    pairwise_df.to_csv(OUTPUT_PAIRWISE, index=False)

    print()
    print("=" * 100)
    print("DONE")
    print("Saved raw results:", OUTPUT_RESULTS)
    print("Saved summary:", OUTPUT_SUMMARY)
    print("Saved pairwise:", OUTPUT_PAIRWISE)
    print("Saved predictions:", OUTPUT_PREDS)

    print()
    print("SUMMARY:")
    print(
        summary_df[
            [
                "target",
                "candidate",
                "model",
                "n_runs",
                "mean_log_rmse",
                "std_log_rmse",
                "mean_log_r2",
                "win_count",
            ]
        ].to_string(index=False)
    )

    print()
    print("PAIRWISE VS:", REFERENCE_CANDIDATE)
    print(pairwise_df.to_string(index=False))


if __name__ == "__main__":
    main()