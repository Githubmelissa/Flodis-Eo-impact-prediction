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
TABULAR_PATH = Path(
    "/home/melissa/flood_project/models_benchmark/impact_prediction/outputs/tabular_phase1_event_table_FIXED.csv"
)

S1GEE_PATH = Path(
    "/home/melissa/flood_project/models_benchmark/impact_prediction/outputs/s1_gee_event_features_by_GFD_ID.csv"
)

SNUNET_WEIGHTED_PATH = Path(
    "/home/melissa/flood_project/models_benchmark/impact_prediction/outputs/snunet_flood_weighted_event_deep_features.csv"
)

OUTPUT_DIR = Path(
    "/home/melissa/flood_project/models_benchmark/impact_prediction/outputs"
)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUT_EVENT_TABLE = OUTPUT_DIR / "s1gee_snunet_weighted_displacement_event_table.csv"
OUT_RESULTS = OUTPUT_DIR / "repeated_cv_s1gee_snunet_weighted_displacement_results.csv"
OUT_SUMMARY = OUTPUT_DIR / "repeated_cv_s1gee_snunet_weighted_displacement_summary.csv"
OUT_PAIRWISE = OUTPUT_DIR / "repeated_cv_s1gee_snunet_weighted_displacement_pairwise.csv"
OUT_PREDS = OUTPUT_DIR / "repeated_cv_s1gee_snunet_weighted_displacement_predictions.csv"


# =====================================================
# SETTINGS
# =====================================================
TARGET = "displacements"
N_SEEDS = 30
N_SPLITS = 5
N_ESTIMATORS = 300
REFERENCE_CANDIDATE = "worldpop_gpm_plus_s1gee"

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
    "snunet_fw_weight_mode",
}


# =====================================================
# HELPERS
# =====================================================
def is_leak(col):
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


def log_target(y):
    return np.log1p(np.clip(np.asarray(y, dtype=float), 0, None))


def inv_log_target(y_log):
    return np.expm1(np.clip(np.asarray(y_log, dtype=float), 0, 30))


def evaluate(y_true_original, y_pred_log):
    y_true_log = log_target(y_true_original)
    y_pred_original = inv_log_target(y_pred_log)

    return {
        "log_mae": float(mean_absolute_error(y_true_log, y_pred_log)),
        "log_rmse": rmse(y_true_log, y_pred_log),
        "log_r2": safe_r2(y_true_log, y_pred_log),
        "original_mae": float(mean_absolute_error(y_true_original, y_pred_original)),
        "original_rmse": rmse(y_true_original, y_pred_original),
        "original_r2": safe_r2(y_true_original, y_pred_original),
    }


def get_feature_groups(df):
    numeric = [
        c for c in df.columns
        if pd.api.types.is_numeric_dtype(df[c])
        and c not in META_COLS
        and not is_leak(c)
    ]

    worldpop = [
        c for c in numeric
        if "population" in c.lower()
        or c.lower() == "flood_area_km2"
    ]

    gpm = [
        c for c in numeric
        if "gpm" in c.lower()
        or "rainfall" in c.lower()
    ]

    srtm = [
        c for c in numeric
        if "elevation" in c.lower()
        or "slope" in c.lower()
    ]

    s1gee = [
        c for c in numeric
        if c.startswith("s1gee_")
    ]

    snunet_unweighted = [
        c for c in numeric
        if c.startswith("snunet_deep_unweighted_mean_")
    ]

    snunet_flood_weighted = [
        c for c in numeric
        if c.startswith("snunet_deep_flood_weighted_mean_")
    ]

    snunet_top5 = [
        c for c in numeric
        if c.startswith("snunet_deep_top5_flood_mean_")
    ]

    return {
        "worldpop": sorted(set(worldpop)),
        "gpm": sorted(set(gpm)),
        "srtm": sorted(set(srtm)),
        "s1gee": sorted(set(s1gee)),
        "snunet_unweighted": sorted(set(snunet_unweighted)),
        "snunet_flood_weighted": sorted(set(snunet_flood_weighted)),
        "snunet_top5": sorted(set(snunet_top5)),
    }


def make_rf(seed):
    return RandomForestRegressor(
        n_estimators=N_ESTIMATORS,
        random_state=seed,
        min_samples_leaf=2,
        n_jobs=-1,
    )


def make_model(candidate, seed):
    if candidate["model"] == "baseline_median":
        return DummyRegressor(strategy="median")

    if candidate["model"] == "rf_raw":
        return Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("model", make_rf(seed)),
            ]
        )

    if candidate["model"] == "rf_base_plus_deep_pca":
        base_cols = candidate["base_cols"]
        deep_cols = candidate["deep_cols"]
        n_pca = candidate["n_pca"]

        preprocess = ColumnTransformer(
            transformers=[
                (
                    "base",
                    Pipeline(
                        steps=[
                            ("imputer", SimpleImputer(strategy="median")),
                            ("scaler", StandardScaler()),
                        ]
                    ),
                    base_cols,
                ),
                (
                    "deep_pca",
                    Pipeline(
                        steps=[
                            ("imputer", SimpleImputer(strategy="median")),
                            ("scaler", StandardScaler()),
                            ("pca", PCA(n_components=n_pca, random_state=seed)),
                        ]
                    ),
                    deep_cols,
                ),
            ],
            remainder="drop",
        )

        return Pipeline(
            steps=[
                ("preprocess", preprocess),
                ("model", make_rf(seed)),
            ]
        )

    raise ValueError(candidate["model"])


def run_candidate(df, candidate, seed):
    labeled = df[df[TARGET].notna()].copy()
    labeled = labeled[labeled[TARGET] >= 0].copy()

    n = len(labeled)
    if n < N_SPLITS:
        return None, []

    y_original = labeled[TARGET].astype(float).values
    y_log = log_target(y_original)

    cv = KFold(n_splits=N_SPLITS, shuffle=True, random_state=seed)

    if candidate["model"] == "baseline_median":
        X = np.zeros((n, 1))

    elif candidate["model"] == "rf_raw":
        feature_cols = candidate["feature_cols"]
        X = labeled[feature_cols].copy()

    elif candidate["model"] == "rf_base_plus_deep_pca":
        feature_cols = candidate["base_cols"] + candidate["deep_cols"]
        X = labeled[feature_cols].copy()

    else:
        raise ValueError(candidate["model"])

    model = make_model(candidate, seed)

    try:
        y_pred_log = cross_val_predict(model, X, y_log, cv=cv)
    except Exception as e:
        print("SKIP:", candidate["candidate"], "seed:", seed, "error:", repr(e))
        return None, []

    metrics = evaluate(y_original, y_pred_log)

    if candidate["model"] == "rf_base_plus_deep_pca":
        n_features = len(candidate["base_cols"]) + candidate["n_pca"]
        n_raw_features = len(candidate["base_cols"]) + len(candidate["deep_cols"])
    else:
        n_features = len(candidate.get("feature_cols", []))
        n_raw_features = n_features

    result_row = {
        "target": TARGET,
        "seed": seed,
        "candidate": candidate["candidate"],
        "model": candidate["model"],
        "n_samples": n,
        "n_features_model_input": n_features,
        "n_raw_features": n_raw_features,
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
                "y_pred": float(inv_log_target([yp_log])[0]),
            }
        )

    return result_row, pred_rows


def build_summary(results):
    group_cols = ["target", "candidate", "model"]

    summary = (
        results.groupby(group_cols)
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

    winners = results.loc[
        results.groupby(["target", "seed"])["log_rmse"].idxmin()
    ].copy()

    wins = (
        winners.groupby(group_cols)
        .size()
        .reset_index(name="win_count")
    )

    summary = summary.merge(wins, on=group_cols, how="left")
    summary["win_count"] = summary["win_count"].fillna(0).astype(int)

    return summary.sort_values(["target", "mean_log_rmse"])


def build_pairwise(results):
    rows = []

    for target in sorted(results["target"].unique()):
        sub = results[results["target"] == target]

        ref = sub[sub["candidate"] == REFERENCE_CANDIDATE][
            ["seed", "log_rmse"]
        ].rename(columns={"log_rmse": "reference_log_rmse"})

        for cand in sorted(sub["candidate"].unique()):
            if cand == REFERENCE_CANDIDATE:
                continue

            cdf = sub[sub["candidate"] == cand][
                ["seed", "log_rmse"]
            ].rename(columns={"log_rmse": "candidate_log_rmse"})

            m = ref.merge(cdf, on="seed", how="inner")

            if len(m) == 0:
                continue

            diff = m["candidate_log_rmse"] - m["reference_log_rmse"]

            rows.append(
                {
                    "target": target,
                    "reference_candidate": REFERENCE_CANDIDATE,
                    "candidate": cand,
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


def main():
    print("=" * 100)
    print("Repeated CV: S1 GEE + weighted SNUNet deep features")

    tab = pd.read_csv(TABULAR_PATH)
    s1gee = pd.read_csv(S1GEE_PATH)
    snunet = pd.read_csv(SNUNET_WEIGHTED_PATH)

    print("Tabular shape:", tab.shape)
    print("S1 GEE shape:", s1gee.shape)
    print("SNUNet weighted shape:", snunet.shape)

    # Safety: remove possible leakage from S1/SNUNet tables
    s1gee = s1gee[[c for c in s1gee.columns if c == "GFD_ID" or not is_leak(c)]].copy()
    snunet = snunet[[c for c in snunet.columns if c == "GFD_ID" or not is_leak(c)]].copy()

    # Important: inner merge gives fair comparison on same 80 events
    df = tab.merge(s1gee, on="GFD_ID", how="inner")
    df = df.merge(snunet, on="GFD_ID", how="inner")

    df.to_csv(OUT_EVENT_TABLE, index=False)

    print("Merged event table:", df.shape)
    print("Unique GFD_ID:", df["GFD_ID"].nunique())
    print("Displacement labels:", int(df[TARGET].notna().sum()))
    print("Saved:", OUT_EVENT_TABLE)

    groups = get_feature_groups(df)

    print()
    print("Feature group sizes:")
    for k, v in groups.items():
        print(f"  {k}: {len(v)}")

    worldpop_gpm = sorted(set(groups["worldpop"] + groups["gpm"]))
    worldpop_gpm_srtm = sorted(set(groups["worldpop"] + groups["gpm"] + groups["srtm"]))
    worldpop_gpm_s1gee = sorted(set(worldpop_gpm + groups["s1gee"]))
    worldpop_gpm_srtm_s1gee = sorted(set(worldpop_gpm_srtm + groups["s1gee"]))

    candidates = [
        {
            "candidate": "baseline_median",
            "model": "baseline_median",
            "feature_cols": [],
        },
        {
            "candidate": "worldpop_gpm_srtm",
            "model": "rf_raw",
            "feature_cols": worldpop_gpm_srtm,
        },
        {
            "candidate": "worldpop_gpm_plus_s1gee",
            "model": "rf_raw",
            "feature_cols": worldpop_gpm_s1gee,
        },
        {
            "candidate": "worldpop_gpm_srtm_plus_s1gee",
            "model": "rf_raw",
            "feature_cols": worldpop_gpm_srtm_s1gee,
        },
        {
            "candidate": "worldpop_gpm_s1gee_plus_snunet_unweighted_pca8",
            "model": "rf_base_plus_deep_pca",
            "base_cols": worldpop_gpm_s1gee,
            "deep_cols": groups["snunet_unweighted"],
            "n_pca": 8,
        },
        {
            "candidate": "worldpop_gpm_s1gee_plus_snunet_floodweighted_pca8",
            "model": "rf_base_plus_deep_pca",
            "base_cols": worldpop_gpm_s1gee,
            "deep_cols": groups["snunet_flood_weighted"],
            "n_pca": 8,
        },
        {
            "candidate": "worldpop_gpm_s1gee_plus_snunet_top5_pca8",
            "model": "rf_base_plus_deep_pca",
            "base_cols": worldpop_gpm_s1gee,
            "deep_cols": groups["snunet_top5"],
            "n_pca": 8,
        },
    ]

    print()
    print("Candidates:")
    for c in candidates:
        if c["model"] == "rf_base_plus_deep_pca":
            print(
                c["candidate"],
                "| base:", len(c["base_cols"]),
                "| deep raw:", len(c["deep_cols"]),
                "| PCA:", c["n_pca"],
            )
        else:
            print(
                c["candidate"],
                "| features:", len(c.get("feature_cols", [])),
            )

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

                pd.DataFrame(all_results).to_csv(OUT_RESULTS, index=False)
                pd.DataFrame(all_preds).to_csv(OUT_PREDS, index=False)

    results = pd.DataFrame(all_results)
    preds = pd.DataFrame(all_preds)

    results.to_csv(OUT_RESULTS, index=False)
    preds.to_csv(OUT_PREDS, index=False)

    summary = build_summary(results)
    pairwise = build_pairwise(results)

    summary.to_csv(OUT_SUMMARY, index=False)
    pairwise.to_csv(OUT_PAIRWISE, index=False)

    print()
    print("=" * 100)
    print("DONE")
    print("Results:", OUT_RESULTS)
    print("Summary:", OUT_SUMMARY)
    print("Pairwise:", OUT_PAIRWISE)
    print("Predictions:", OUT_PREDS)

    print()
    print("SUMMARY")
    print(summary.to_string(index=False))

    print()
    print("PAIRWISE VS:", REFERENCE_CANDIDATE)
    print(pairwise.to_string(index=False))


if __name__ == "__main__":
    main()