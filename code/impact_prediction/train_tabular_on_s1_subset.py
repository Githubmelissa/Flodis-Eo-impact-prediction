from pathlib import Path
import warnings

import numpy as np
import pandas as pd

from sklearn.dummy import DummyRegressor
from sklearn.ensemble import (
    RandomForestRegressor,
    ExtraTreesRegressor,
    HistGradientBoostingRegressor,
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import RidgeCV, ElasticNetCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


warnings.filterwarnings("ignore")


# =====================================================
# PATHS
# =====================================================
TABULAR_TABLE_PATH = Path(
    "/home/melissa/flood_project/models_benchmark/impact_prediction/outputs/tabular_phase1_event_table_FIXED.csv"
)

S1_TABLE_PATH = Path(
    "/home/melissa/flood_project/models_benchmark/KuroSiwo/flodis_outputs/merged_prediction_tables/s1_features_with_flodis_targets_all85.csv"
)

OUT_DIR = Path(
    "/home/melissa/flood_project/models_benchmark/impact_prediction/outputs"
)
OUT_DIR.mkdir(parents=True, exist_ok=True)

OUT_SUBSET_TABLE = OUT_DIR / "tabular_phase1_S1_subset_event_table.csv"
OUT_RESULTS = OUT_DIR / "tabular_phase1_S1_subset_model_results.csv"
OUT_CV_PREDS = OUT_DIR / "tabular_phase1_S1_subset_cv_predictions.csv"


TARGETS = [
    "displacements",
    "total_deaths",
    "total_affected_EMDAT",
    "total_damages_000_USD",
]


# =====================================================
# FEATURE SETS
# =====================================================
WORLDPOP_FEATURES = [
    "wp_flood_area_km2_mean",
    "wp_flood_area_km2_max",
    "wp_population_total_sum",
    "wp_population_total_mean",
    "wp_population_mean_mean",
    "wp_population_max_max",
    "n_phase1_rows",
    "n_iso3",
]

GPM_FEATURES = [
    "gpm_count_mean",
    "gpm_rainfall_total_mm_mean",
    "gpm_rainfall_total_mm_max",
    "gpm_rainfall_max_mm_max",
    "gpm_rainfall_mean_mm_mean",
    "gpm_rainfall_days_mean",
    "event_month",
]

SRTM_FEATURES = [
    "srtm_elevation_mean",
    "srtm_elevation_stdDev_mean",
    "srtm_elevation_min",
    "srtm_elevation_max",
    "srtm_slope_mean",
    "srtm_slope_stdDev_mean",
    "srtm_slope_min",
    "srtm_slope_max",
]

FEATURE_SETS = {
    "worldpop": WORLDPOP_FEATURES,
    "gpm": GPM_FEATURES,
    "srtm": SRTM_FEATURES,
    "worldpop_gpm": WORLDPOP_FEATURES + GPM_FEATURES,
    "worldpop_srtm": WORLDPOP_FEATURES + SRTM_FEATURES,
    "gpm_srtm": GPM_FEATURES + SRTM_FEATURES,
    "worldpop_gpm_srtm": WORLDPOP_FEATURES + GPM_FEATURES + SRTM_FEATURES,
}


MODEL_NAMES = [
    "ridge",
    "elasticnet",
    "random_forest",
    "extra_trees",
    "hist_gradient_boosting",
]


# =====================================================
# HELPERS
# =====================================================
def rmse(y_true, y_pred):
    return mean_squared_error(y_true, y_pred) ** 0.5


def safe_expm1(log_values):
    return np.expm1(np.clip(log_values, 0, 30))


def evaluate(y_log, pred_log):
    y_true = np.expm1(y_log)
    y_pred = safe_expm1(pred_log)

    return {
        "log_mae": mean_absolute_error(y_log, pred_log),
        "log_rmse": rmse(y_log, pred_log),
        "log_r2": r2_score(y_log, pred_log),
        "original_mae": mean_absolute_error(y_true, y_pred),
        "original_rmse": rmse(y_true, y_pred),
        "original_r2": r2_score(y_true, y_pred),
    }


def choose_cv(n_rows):
    n_splits = min(5, n_rows)

    return KFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=42,
    )


def make_model(model_name):
    if model_name == "ridge":
        return Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                ("model", RidgeCV(alphas=np.logspace(-4, 4, 25))),
            ]
        )

    if model_name == "elasticnet":
        return Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                (
                    "model",
                    ElasticNetCV(
                        l1_ratio=[0.1, 0.5, 0.9],
                        alphas=np.logspace(-4, 2, 20),
                        cv=3,
                        max_iter=10000,
                        random_state=42,
                    ),
                ),
            ]
        )

    if model_name == "random_forest":
        return Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                (
                    "model",
                    RandomForestRegressor(
                        n_estimators=500,
                        max_depth=4,
                        min_samples_leaf=3,
                        random_state=42,
                        n_jobs=-1,
                    ),
                ),
            ]
        )

    if model_name == "extra_trees":
        return Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                (
                    "model",
                    ExtraTreesRegressor(
                        n_estimators=500,
                        max_depth=4,
                        min_samples_leaf=3,
                        random_state=42,
                        n_jobs=-1,
                    ),
                ),
            ]
        )

    if model_name == "hist_gradient_boosting":
        return Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                (
                    "model",
                    HistGradientBoostingRegressor(
                        max_iter=200,
                        learning_rate=0.05,
                        max_leaf_nodes=5,
                        l2_regularization=1.0,
                        random_state=42,
                    ),
                ),
            ]
        )

    raise ValueError(model_name)


# =====================================================
# LOAD TABLES
# =====================================================
print("=" * 100)
print("Loading fixed tabular event table")
print(TABULAR_TABLE_PATH)

tab = pd.read_csv(TABULAR_TABLE_PATH)
tab["GFD_ID"] = pd.to_numeric(tab["GFD_ID"], errors="coerce")
tab = tab[tab["GFD_ID"].notna()].copy()
tab["GFD_ID"] = tab["GFD_ID"].astype(int)

print("Tabular table shape:", tab.shape)
print("Tabular unique GFD_ID:", tab["GFD_ID"].nunique())


print()
print("=" * 100)
print("Loading S1 event table only to get the S1 GFD_ID subset")
print(S1_TABLE_PATH)

s1 = pd.read_csv(S1_TABLE_PATH, usecols=["GFD_ID"])
s1["GFD_ID"] = pd.to_numeric(s1["GFD_ID"], errors="coerce")
s1 = s1[s1["GFD_ID"].notna()].copy()
s1["GFD_ID"] = s1["GFD_ID"].astype(int)

s1_ids = sorted(s1["GFD_ID"].unique())

print("S1 table rows:", len(s1))
print("S1 unique GFD_ID:", len(s1_ids))


# =====================================================
# FILTER TABULAR TO S1 SUBSET
# =====================================================
df = tab[tab["GFD_ID"].isin(s1_ids)].copy()
df = df.sort_values("GFD_ID").reset_index(drop=True)

missing_tabular_ids = sorted(set(s1_ids) - set(df["GFD_ID"].unique()))
extra_tabular_ids = sorted(set(tab["GFD_ID"].unique()) - set(s1_ids))

print()
print("=" * 100)
print("S1 SUBSET CHECK")
print("Rows after filtering tabular table to S1 events:", df.shape)
print("Unique GFD_ID after filtering:", df["GFD_ID"].nunique())
print("S1 IDs missing from tabular table:", len(missing_tabular_ids))
print("Tabular IDs not used because no S1 features:", len(extra_tabular_ids))

if len(missing_tabular_ids) > 0:
    print("Missing S1 IDs from tabular table:")
    print(missing_tabular_ids)

df.to_csv(OUT_SUBSET_TABLE, index=False)

print()
print("Saved S1-subset tabular table:")
print(OUT_SUBSET_TABLE)


print()
print("=" * 100)
print("Target availability on S1 subset")
for t in TARGETS:
    if t in df.columns:
        print(t, ":", df[t].notna().sum(), "/", len(df))
    else:
        print(t, ": missing")


# =====================================================
# TRAIN / EVALUATE TABULAR ONLY ON S1 SUBSET
# =====================================================
all_results = []
all_cv_preds = []

for target in TARGETS:
    if target not in df.columns:
        continue

    labeled = df[df[target].notna()].copy()
    labeled = labeled[labeled[target] >= 0].copy()

    print()
    print("#" * 100)
    print("TARGET:", target)
    print("Labeled rows on S1 subset:", len(labeled))
    print("#" * 100)

    if len(labeled) < 8:
        print("SKIP: not enough labeled rows")
        continue

    y_raw = labeled[target].astype(float).values
    y_log = np.log1p(y_raw)

    cv = choose_cv(len(labeled))

    # -----------------------------
    # Median baseline
    # -----------------------------
    dummy_X = np.zeros((len(labeled), 1))
    dummy = DummyRegressor(strategy="median")
    pred_log = cross_val_predict(dummy, dummy_X, y_log, cv=cv)

    metrics = evaluate(y_log, pred_log)

    row = {
        "target": target,
        "model": "baseline_median",
        "feature_set": "none",
        "subset": "S1_events_only",
        "n_rows": len(labeled),
        "n_features": 0,
        **metrics,
    }

    all_results.append(row)
    print(row)

    tmp = pd.DataFrame(
        {
            "target": target,
            "model": "baseline_median",
            "feature_set": "none",
            "subset": "S1_events_only",
            "GFD_ID": labeled["GFD_ID"].values,
            "true_value": y_raw,
            "pred_value": safe_expm1(pred_log),
            "true_log1p": y_log,
            "pred_log1p": pred_log,
        }
    )
    all_cv_preds.append(tmp)

    # -----------------------------
    # Real tabular models
    # -----------------------------
    for feature_set_name, feature_cols in FEATURE_SETS.items():
        available_cols = [c for c in feature_cols if c in labeled.columns]
        X = labeled[available_cols].copy()

        # Remove fully missing columns
        X = X[[c for c in X.columns if not X[c].isna().all()]]

        if X.shape[1] == 0:
            continue

        for model_name in MODEL_NAMES:
            print()
            print(
                "Target:", target,
                "| Feature set:", feature_set_name,
                "| Model:", model_name,
            )

            model = make_model(model_name)

            try:
                pred_log = cross_val_predict(model, X, y_log, cv=cv)
                metrics = evaluate(y_log, pred_log)

                row = {
                    "target": target,
                    "model": model_name,
                    "feature_set": feature_set_name,
                    "subset": "S1_events_only",
                    "n_rows": len(labeled),
                    "n_features": X.shape[1],
                    **metrics,
                }

                all_results.append(row)
                print(row)

                tmp = pd.DataFrame(
                    {
                        "target": target,
                        "model": model_name,
                        "feature_set": feature_set_name,
                        "subset": "S1_events_only",
                        "GFD_ID": labeled["GFD_ID"].values,
                        "true_value": y_raw,
                        "pred_value": safe_expm1(pred_log),
                        "true_log1p": y_log,
                        "pred_log1p": pred_log,
                    }
                )
                all_cv_preds.append(tmp)

            except Exception as e:
                print("ERROR:", e)

                row = {
                    "target": target,
                    "model": model_name,
                    "feature_set": feature_set_name,
                    "subset": "S1_events_only",
                    "n_rows": len(labeled),
                    "n_features": X.shape[1],
                    "error": str(e),
                }
                all_results.append(row)


# =====================================================
# SAVE
# =====================================================
results_df = pd.DataFrame(all_results)

if "log_rmse" in results_df.columns:
    results_df = results_df.sort_values(["target", "log_rmse"], na_position="last")

results_df.to_csv(OUT_RESULTS, index=False)

if len(all_cv_preds) > 0:
    cv_preds_df = pd.concat(all_cv_preds, ignore_index=True)
    cv_preds_df.to_csv(OUT_CV_PREDS, index=False)

print()
print("=" * 100)
print("SAVED")
print("S1 subset tabular table:", OUT_SUBSET_TABLE)
print("Results:", OUT_RESULTS)
print("CV predictions:", OUT_CV_PREDS)

print()
print("=" * 100)
print("BEST RESULTS PER TARGET ON S1 SUBSET")
for target in TARGETS:
    sub = results_df[results_df["target"] == target].copy()

    if len(sub) == 0:
        continue

    print()
    print(target)

    if "log_rmse" in sub.columns:
        print(sub.head(10).to_string(index=False))
    else:
        print(sub.to_string(index=False))

print()
print("=" * 100)
print("DONE")