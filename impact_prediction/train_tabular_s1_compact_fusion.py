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

OUT_FUSION_TABLE = OUT_DIR / "tabular_s1_compact_fusion_event_table.csv"
OUT_RESULTS = OUT_DIR / "tabular_s1_compact_fusion_model_results.csv"
OUT_CV_PREDS = OUT_DIR / "tabular_s1_compact_fusion_cv_predictions.csv"


TARGETS = [
    "displacements",
    "total_deaths",
    "total_affected_EMDAT",
    "total_damages_000_USD",
]

TARGET_AND_META_COLS = [
    "displacements",
    "total_deaths",
    "total_affected_EMDAT",
    "total_damages_000_USD",
    "displacement_records",
    "mortality_damage_records",
]


# =====================================================
# TABULAR FEATURE SETS
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

TABULAR_FEATURE_SETS = {
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


def is_numeric(df, col):
    return pd.api.types.is_numeric_dtype(df[col])


def deduplicate_keep_order(cols):
    seen = set()
    out = []
    for c in cols:
        if c not in seen:
            out.append(c)
            seen.add(c)
    return out


# =====================================================
# S1 FEATURE SELECTION
# =====================================================
def select_s1_quality_features(s1):
    quality_keywords = [
        "n_total_patches",
        "n_valid_patches",
        "n_skipped_patches",
        "valid_patch_fraction",
    ]

    cols = []

    for c in s1.columns:
        cl = c.lower()

        if c == "GFD_ID":
            continue

        if not is_numeric(s1, c):
            continue

        if any(k in cl for k in quality_keywords):
            cols.append(c)

    return deduplicate_keep_order(cols)


def select_s1_compact_features(s1):
    """
    Select compact S1 features only.

    We want:
    - quality features
    - probability / class / summary features from SNUNet and FloodViT

    We avoid:
    - target columns
    - debug columns
    - flood_mask columns
    - high-dimensional deep embedding columns
    """

    forbidden_keywords = [
        "displacement",
        "death",
        "affected",
        "damage",
        "record",
        "flood_mask",
        "debug",
    ]

    deep_keywords = [
        "deep",
        "embedding",
        "embed",
        "feature_",
        "features_",
        "feat_",
        "hidden",
        "encoder",
        "encoding",
        "gap",
    ]

    useful_keywords = [
        "n_total_patches",
        "n_valid_patches",
        "n_skipped_patches",
        "valid_patch_fraction",
        "prob",
        "class",
        "pred",
        "logit",
        "entropy",
        "uncert",
        "flood",
        "water",
        "change",
    ]

    cols = []

    for c in s1.columns:
        cl = c.lower()

        if c == "GFD_ID":
            continue

        if not is_numeric(s1, c):
            continue

        if any(k in cl for k in forbidden_keywords):
            continue

        if any(k in cl for k in deep_keywords):
            continue

        if any(k in cl for k in useful_keywords):
            cols.append(c)

    return deduplicate_keep_order(cols)


# =====================================================
# LOAD TABLES
# =====================================================
print("=" * 100)
print("Loading fixed tabular table")
print(TABULAR_TABLE_PATH)

tab = pd.read_csv(TABULAR_TABLE_PATH)
tab["GFD_ID"] = pd.to_numeric(tab["GFD_ID"], errors="coerce")
tab = tab[tab["GFD_ID"].notna()].copy()
tab["GFD_ID"] = tab["GFD_ID"].astype(int)

print("Tabular shape:", tab.shape)
print("Tabular unique GFD_ID:", tab["GFD_ID"].nunique())


print()
print("=" * 100)
print("Loading S1 feature table")
print(S1_TABLE_PATH)

s1 = pd.read_csv(S1_TABLE_PATH)
s1["GFD_ID"] = pd.to_numeric(s1["GFD_ID"], errors="coerce")
s1 = s1[s1["GFD_ID"].notna()].copy()
s1["GFD_ID"] = s1["GFD_ID"].astype(int)

print("S1 shape:", s1.shape)
print("S1 unique GFD_ID:", s1["GFD_ID"].nunique())


# Remove old target columns from S1 table.
# The target source must be the fixed tabular target table.
drop_from_s1 = [c for c in TARGET_AND_META_COLS if c in s1.columns]
s1_features = s1.drop(columns=drop_from_s1)

print()
print("Dropped old target/meta columns from S1:")
print(drop_from_s1)


# =====================================================
# MERGE TABULAR + S1
# =====================================================
df = tab.merge(s1_features, on="GFD_ID", how="inner")
df = df.replace([np.inf, -np.inf], np.nan)
df = df.sort_values("GFD_ID").reset_index(drop=True)

df.to_csv(OUT_FUSION_TABLE, index=False)

print()
print("=" * 100)
print("FUSION TABLE")
print("Fusion shape:", df.shape)
print("Fusion unique GFD_ID:", df["GFD_ID"].nunique())
print("Saved:", OUT_FUSION_TABLE)

print()
print("Target availability:")
for t in TARGETS:
    if t in df.columns:
        print(t, ":", df[t].notna().sum(), "/", len(df))


# =====================================================
# SELECT S1 COMPACT FEATURES
# =====================================================
s1_quality_features = select_s1_quality_features(df)
s1_compact_features = select_s1_compact_features(df)

print()
print("=" * 100)
print("S1 FEATURE SELECTION")
print("S1 quality feature count:", len(s1_quality_features))
print("S1 compact feature count:", len(s1_compact_features))

print()
print("S1 quality features:")
print(s1_quality_features)

print()
print("First 80 S1 compact features:")
print(s1_compact_features[:80])

if len(s1_compact_features) == 0:
    print()
    print("WARNING: No S1 compact features were found.")
    print("The script will only test tabular and S1 quality features.")

if len(s1_compact_features) > 150:
    print()
    print("WARNING: More than 150 S1 compact columns selected.")
    print("Check the printed feature names to make sure deep embeddings were not included.")


# =====================================================
# BUILD FEATURE SETS
# =====================================================
FEATURE_SETS = {}

# Tabular-only feature sets
for name, cols in TABULAR_FEATURE_SETS.items():
    FEATURE_SETS[name] = cols

# S1-only feature sets
if len(s1_quality_features) > 0:
    FEATURE_SETS["s1_quality_only"] = s1_quality_features

if len(s1_compact_features) > 0:
    FEATURE_SETS["s1_compact_only"] = s1_compact_features

# Tabular + S1 quality
for name, cols in TABULAR_FEATURE_SETS.items():
    if len(s1_quality_features) > 0:
        FEATURE_SETS[f"{name}_plus_s1_quality"] = cols + s1_quality_features

# Tabular + S1 compact
for name, cols in TABULAR_FEATURE_SETS.items():
    if len(s1_compact_features) > 0:
        FEATURE_SETS[f"{name}_plus_s1_compact"] = cols + s1_compact_features


print()
print("=" * 100)
print("FEATURE SETS TO TEST")
for k, v in FEATURE_SETS.items():
    print(k, ":", len(v), "features")


# =====================================================
# TRAIN / EVALUATE
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
    print("Labeled rows:", len(labeled))
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
            "GFD_ID": labeled["GFD_ID"].values,
            "true_value": y_raw,
            "pred_value": safe_expm1(pred_log),
            "true_log1p": y_log,
            "pred_log1p": pred_log,
        }
    )
    all_cv_preds.append(tmp)

    # -----------------------------
    # Real feature sets
    # -----------------------------
    for feature_set_name, feature_cols in FEATURE_SETS.items():
        available_cols = [c for c in feature_cols if c in labeled.columns]
        X = labeled[available_cols].copy()

        # Keep numeric columns only
        X = X[[c for c in X.columns if is_numeric(X, c)]]

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
                "| Features:", X.shape[1],
            )

            model = make_model(model_name)

            try:
                pred_log = cross_val_predict(model, X, y_log, cv=cv)
                metrics = evaluate(y_log, pred_log)

                row = {
                    "target": target,
                    "model": model_name,
                    "feature_set": feature_set_name,
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
                    "n_rows": len(labeled),
                    "n_features": X.shape[1],
                    "error": str(e),
                }
                all_results.append(row)


# =====================================================
# SAVE RESULTS
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
print("Fusion table:", OUT_FUSION_TABLE)
print("Results:", OUT_RESULTS)
print("CV predictions:", OUT_CV_PREDS)

print()
print("=" * 100)
print("BEST RESULTS PER TARGET")
for target in TARGETS:
    sub = results_df[results_df["target"] == target].copy()

    if len(sub) == 0:
        continue

    print()
    print(target)

    if "log_rmse" in sub.columns:
        print(sub.head(15).to_string(index=False))
    else:
        print(sub.to_string(index=False))

print()
print("=" * 100)
print("DONE")