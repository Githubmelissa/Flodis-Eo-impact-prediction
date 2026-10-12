from pathlib import Path
import warnings

import numpy as np
import pandas as pd

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
TABULAR_PATH = Path(
    "/home/melissa/flood_project/models_benchmark/impact_prediction/outputs/tabular_phase1_event_table_FIXED.csv"
)

S1_GEE_PATH = Path(
    "/home/melissa/flood_project/models_benchmark/impact_prediction/outputs/s1_gee_event_features_by_GFD_ID.csv"
)

OUTPUT_DIR = Path(
    "/home/melissa/flood_project/models_benchmark/impact_prediction/outputs"
)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUT_EVENT_TABLE = OUTPUT_DIR / "tabular_s1gee_event_table.csv"
OUT_RESULTS = OUTPUT_DIR / "repeated_cv_s1gee_displacement_model_results.csv"
OUT_SUMMARY = OUTPUT_DIR / "repeated_cv_s1gee_displacement_summary.csv"
OUT_PAIRWISE = OUTPUT_DIR / "repeated_cv_s1gee_displacement_pairwise_comparison.csv"
OUT_PREDS = OUTPUT_DIR / "repeated_cv_s1gee_displacement_cv_predictions.csv"


# =====================================================
# SETTINGS
# =====================================================
TARGET = "displacements"
N_SEEDS = 30
N_SPLITS = 5
N_ESTIMATORS = 300
REFERENCE_CANDIDATE = "worldpop_gpm_srtm"


TARGET_KEYWORDS = [
    "displacement",
    "death",
    "deaths",
    "mortality",
    "affected",
    "emdat",
    "damage",
    "damages",
    "idmc",
]

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
    "s1gee_iso3_list",
    "s1gee_event_date",
    "s1gee_year",
}


# =====================================================
# HELPERS
# =====================================================
def is_target_or_leak(col):
    if col == "GFD_ID":
        return False
    c = str(col).lower()
    return any(k in c for k in TARGET_KEYWORDS)


def rmse(y_true, y_pred):
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def safe_r2(y_true, y_pred):
    try:
        return float(r2_score(y_true, y_pred))
    except Exception:
        return np.nan


def safe_log1p(y):
    return np.log1p(np.clip(np.asarray(y, dtype=float), 0, None))


def safe_expm1(y_log):
    return np.expm1(np.clip(np.asarray(y_log, dtype=float), 0, 30))


def evaluate(y_true_original, y_pred_log):
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


def numeric_cols(df, cols):
    return [
        c for c in cols
        if c in df.columns
        and pd.api.types.is_numeric_dtype(df[c])
        and c not in META_COLS
        and not is_target_or_leak(c)
    ]


def get_feature_groups(df):
    all_numeric = [
        c for c in df.columns
        if pd.api.types.is_numeric_dtype(df[c])
        and c not in META_COLS
        and not is_target_or_leak(c)
    ]

    worldpop = [
        c for c in all_numeric
        if "population" in c.lower()
        or "flood_area_km2" in c.lower()
    ]

    gpm = [
        c for c in all_numeric
        if "gpm" in c.lower()
        or "rainfall" in c.lower()
    ]

    srtm = [
        c for c in all_numeric
        if "elevation" in c.lower()
        or "slope" in c.lower()
    ]

    s1gee = [
        c for c in all_numeric
        if c.startswith("s1gee_")
    ]

    groups = {
        "worldpop": sorted(set(worldpop)),
        "gpm": sorted(set(gpm)),
        "srtm": sorted(set(srtm)),
        "s1gee": sorted(set(s1gee)),
    }

    return groups


def make_rf(seed):
    return RandomForestRegressor(
        n_estimators=N_ESTIMATORS,
        random_state=seed,
        min_samples_leaf=2,
        n_jobs=-1,
    )


def make_pipeline(model_name, seed):
    if model_name == "baseline_median":
        return DummyRegressor(strategy="median")

    if model_name == "random_forest":
        return Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                ("model", make_rf(seed)),
            ]
        )

    raise ValueError(model_name)


def run_candidate(df, candidate, seed):
    labeled = df[df[TARGET].notna()].copy()
    labeled = labeled[labeled[TARGET] >= 0].copy()

    n = len(labeled)
    if n < N_SPLITS:
        return None, []

    y_original = labeled[TARGET].astype(float).values
    y_log = safe_log1p(y_original)

    cv = KFold(n_splits=N_SPLITS, shuffle=True, random_state=seed)

    if candidate["model"] == "baseline_median":
        X = np.zeros((n, 1))
    else:
        feature_cols = candidate["feature_cols"]
        if len(feature_cols) == 0:
            return None, []
        X = labeled[feature_cols].copy()

    model = make_pipeline(candidate["model"], seed)

    try:
        y_pred_log = cross_val_predict(model, X, y_log, cv=cv)
    except Exception as e:
        print("SKIP:", candidate["candidate"], "seed:", seed, "error:", repr(e))
        return None, []

    metrics = evaluate(y_original, y_pred_log)

    row = {
        "target": TARGET,
        "seed": seed,
        "candidate": candidate["candidate"],
        "model": candidate["model"],
        "n_samples": n,
        "n_features": len(candidate["feature_cols"]),
        **metrics,
    }

    pred_rows = []
    for gfd_id, yt, yp_log in zip(labeled["GFD_ID"], y_original, y_pred_log):
        pred_rows.append(
            {
                "target": TARGET,
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

    return row, pred_rows


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

    return summary.sort_values(["target", "mean_log_rmse"])


def build_pairwise(results_df):
    rows = []

    for target in sorted(results_df["target"].unique()):
        target_df = results_df[results_df["target"] == target]

        ref = target_df[target_df["candidate"] == REFERENCE_CANDIDATE][
            ["seed", "log_rmse"]
        ].rename(columns={"log_rmse": "reference_log_rmse"})

        for cand_name in sorted(target_df["candidate"].unique()):
            if cand_name == REFERENCE_CANDIDATE:
                continue

            cand = target_df[target_df["candidate"] == cand_name][
                ["seed", "log_rmse"]
            ].rename(columns={"log_rmse": "candidate_log_rmse"})

            m = ref.merge(cand, on="seed", how="inner")
            if len(m) == 0:
                continue

            diff = m["candidate_log_rmse"] - m["reference_log_rmse"]

            rows.append(
                {
                    "target": target,
                    "reference_candidate": REFERENCE_CANDIDATE,
                    "candidate": cand_name,
                    "n_seeds": len(m),
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
    print("Repeated CV: tabular + S1 GEE event-level features")
    print("Tabular:", TABULAR_PATH)
    print("S1 GEE:", S1_GEE_PATH)

    tab = pd.read_csv(TABULAR_PATH)
    s1 = pd.read_csv(S1_GEE_PATH)

    print("Tabular shape:", tab.shape)
    print("S1 GEE shape:", s1.shape)

    # Remove any accidental target/leak columns from S1 table
    leak_s1 = [c for c in s1.columns if c != "GFD_ID" and is_target_or_leak(c)]
    if leak_s1:
        print("Dropping S1 leakage columns:", leak_s1)
        s1 = s1.drop(columns=leak_s1)

    df = tab.merge(s1, on="GFD_ID", how="inner")
    df.to_csv(OUT_EVENT_TABLE, index=False)

    print("Merged event table:", df.shape)
    print("Unique GFD_ID:", df["GFD_ID"].nunique())
    print("Saved merged table:", OUT_EVENT_TABLE)

    groups = get_feature_groups(df)

    print()
    print("Feature group sizes:")
    for k, v in groups.items():
        print(f"  {k}: {len(v)}")

    worldpop_gpm = sorted(set(groups["worldpop"] + groups["gpm"]))
    worldpop_gpm_srtm = sorted(set(groups["worldpop"] + groups["gpm"] + groups["srtm"]))
    s1gee = groups["s1gee"]

    candidates = [
        {
            "candidate": "baseline_median",
            "model": "baseline_median",
            "feature_cols": [],
        },
        {
            "candidate": "s1gee_only",
            "model": "random_forest",
            "feature_cols": s1gee,
        },
        {
            "candidate": "worldpop_gpm",
            "model": "random_forest",
            "feature_cols": worldpop_gpm,
        },
        {
            "candidate": "worldpop_gpm_srtm",
            "model": "random_forest",
            "feature_cols": worldpop_gpm_srtm,
        },
        {
            "candidate": "worldpop_gpm_plus_s1gee",
            "model": "random_forest",
            "feature_cols": sorted(set(worldpop_gpm + s1gee)),
        },
        {
            "candidate": "worldpop_gpm_srtm_plus_s1gee",
            "model": "random_forest",
            "feature_cols": sorted(set(worldpop_gpm_srtm + s1gee)),
        },
    ]

    print()
    print("Candidates:")
    for c in candidates:
        print(c["candidate"], "|", c["model"], "| n_features:", len(c["feature_cols"]))

    all_results = []
    all_preds = []

    for seed in range(N_SEEDS):
        print()
        print("=" * 100)
        print("Seed:", seed)

        for cand in candidates:
            print("Running:", cand["candidate"])

            row, pred_rows = run_candidate(df, cand, seed)

            if row is not None:
                all_results.append(row)
                all_preds.extend(pred_rows)

            if len(all_results) > 0:
                pd.DataFrame(all_results).to_csv(OUT_RESULTS, index=False)

            if len(all_preds) > 0:
                pd.DataFrame(all_preds).to_csv(OUT_PREDS, index=False)

    results_df = pd.DataFrame(all_results)
    preds_df = pd.DataFrame(all_preds)

    results_df.to_csv(OUT_RESULTS, index=False)
    preds_df.to_csv(OUT_PREDS, index=False)

    summary_df = build_summary(results_df)
    pairwise_df = build_pairwise(results_df)

    summary_df.to_csv(OUT_SUMMARY, index=False)
    pairwise_df.to_csv(OUT_PAIRWISE, index=False)

    print()
    print("=" * 100)
    print("DONE")
    print("Results:", OUT_RESULTS)
    print("Summary:", OUT_SUMMARY)
    print("Pairwise:", OUT_PAIRWISE)
    print("Predictions:", OUT_PREDS)

    print()
    print("SUMMARY")
    print(summary_df.to_string(index=False))

    print()
    print("PAIRWISE VS:", REFERENCE_CANDIDATE)
    print(pairwise_df.to_string(index=False))


if __name__ == "__main__":
    main()