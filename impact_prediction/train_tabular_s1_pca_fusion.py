from pathlib import Path
import warnings

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
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

CROMA_TABLE_PATH = Path(
    "/home/melissa/flood_project/models_benchmark/CROMA/flodis_outputs/croma_event_features_valid_only/all_events_croma_event_features_valid_only.csv"
)

OUT_DIR = Path(
    "/home/melissa/flood_project/models_benchmark/impact_prediction/outputs"
)
OUT_DIR.mkdir(parents=True, exist_ok=True)

OUT_FUSION_TABLE = OUT_DIR / "tabular_s1_pca_fusion_event_table.csv"
OUT_RESULTS = OUT_DIR / "tabular_s1_pca_fusion_model_results.csv"
OUT_CV_PREDS = OUT_DIR / "tabular_s1_pca_fusion_cv_predictions.csv"


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
# TABULAR FEATURES
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

BASE_FEATURE_SETS = {
    "worldpop_gpm": WORLDPOP_FEATURES + GPM_FEATURES,
    "worldpop_gpm_srtm": WORLDPOP_FEATURES + GPM_FEATURES + SRTM_FEATURES,
}


MODEL_NAMES = [
    "ridge",
    "elasticnet",
    "random_forest",
    "extra_trees",
    "hist_gradient_boosting",
]

PCA_COMPONENTS_TO_TEST = [8, 16, 32]


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


def min_train_size_for_cv(n_rows, n_splits):
    max_test_size = int(np.ceil(n_rows / n_splits))
    return n_rows - max_test_size


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


def make_model(model_name):
    if model_name == "ridge":
        return RidgeCV(alphas=np.logspace(-4, 4, 25))

    if model_name == "elasticnet":
        return ElasticNetCV(
            l1_ratio=[0.1, 0.5, 0.9],
            alphas=np.logspace(-4, 2, 20),
            cv=3,
            max_iter=10000,
            random_state=42,
        )

    if model_name == "random_forest":
        return RandomForestRegressor(
            n_estimators=500,
            max_depth=4,
            min_samples_leaf=3,
            random_state=42,
            n_jobs=-1,
        )

    if model_name == "extra_trees":
        return ExtraTreesRegressor(
            n_estimators=500,
            max_depth=4,
            min_samples_leaf=3,
            random_state=42,
            n_jobs=-1,
        )

    if model_name == "hist_gradient_boosting":
        return HistGradientBoostingRegressor(
            max_iter=200,
            learning_rate=0.05,
            max_leaf_nodes=5,
            l2_regularization=1.0,
            random_state=42,
        )

    raise ValueError(model_name)


def make_tabular_only_pipeline(model_name, tabular_cols):
    preprocessor = ColumnTransformer(
        transformers=[
            (
                "tabular",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                tabular_cols,
            )
        ],
        remainder="drop",
    )

    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", make_model(model_name)),
        ]
    )


def make_tabular_deep_pca_pipeline(model_name, tabular_cols, deep_cols, n_components):
    """
    IMPORTANT:
    PCA is inside the sklearn Pipeline.
    So during cross-validation, PCA is fitted only on the training fold.
    This avoids data leakage.
    """

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "tabular",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                tabular_cols,
            ),
            (
                "deep_pca",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                        ("pca", PCA(n_components=n_components, random_state=42)),
                    ]
                ),
                deep_cols,
            ),
        ],
        remainder="drop",
    )

    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", make_model(model_name)),
        ]
    )


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


# Remove old target columns from S1 table
drop_from_s1 = [c for c in TARGET_AND_META_COLS if c in s1.columns]
s1_features = s1.drop(columns=drop_from_s1)

print()
print("Dropped target/meta columns from S1:")
print(drop_from_s1)


# =====================================================
# LOAD CROMA IF AVAILABLE
# =====================================================
croma_features = None

if CROMA_TABLE_PATH.exists():
    print()
    print("=" * 100)
    print("Loading CROMA feature table")
    print(CROMA_TABLE_PATH)

    croma = pd.read_csv(CROMA_TABLE_PATH)
    croma["GFD_ID"] = pd.to_numeric(croma["GFD_ID"], errors="coerce")
    croma = croma[croma["GFD_ID"].notna()].copy()
    croma["GFD_ID"] = croma["GFD_ID"].astype(int)

    # Prefix CROMA non-GFD columns to avoid collisions
    rename_map = {}
    for c in croma.columns:
        if c != "GFD_ID":
            if not c.startswith("croma_"):
                rename_map[c] = f"croma_{c}"

    croma = croma.rename(columns=rename_map)

    # Remove old targets if any
    drop_from_croma = [c for c in TARGET_AND_META_COLS if c in croma.columns]
    croma_features = croma.drop(columns=drop_from_croma)

    print("CROMA shape:", croma_features.shape)
    print("CROMA unique GFD_ID:", croma_features["GFD_ID"].nunique())
else:
    print()
    print("=" * 100)
    print("WARNING: CROMA table not found. CROMA PCA features will be skipped.")
    print(CROMA_TABLE_PATH)


# =====================================================
# MERGE TABULAR + S1 + CROMA
# =====================================================
df = tab.merge(s1_features, on="GFD_ID", how="inner")

if croma_features is not None:
    df = df.merge(croma_features, on="GFD_ID", how="left")

df = df.replace([np.inf, -np.inf], np.nan)
df = df.sort_values("GFD_ID").reset_index(drop=True)

df.to_csv(OUT_FUSION_TABLE, index=False)

print()
print("=" * 100)
print("PCA FUSION TABLE")
print("Fusion shape:", df.shape)
print("Fusion unique GFD_ID:", df["GFD_ID"].nunique())
print("Saved:", OUT_FUSION_TABLE)

print()
print("Target availability:")
for t in TARGETS:
    if t in df.columns:
        print(t, ":", df[t].notna().sum(), "/", len(df))


# =====================================================
# SELECT DEEP FEATURE COLUMNS
# =====================================================
snunet_deep_cols = [
    c for c in df.columns
    if c.startswith("snunet_deep_") and is_numeric(df, c)
]

floodvit_deep_cols = [
    c for c in df.columns
    if c.startswith("floodvit_deep_") and is_numeric(df, c)
]

croma_metadata_keywords = [
    "event_id",
    "n_total_patches",
    "n_valid_patches",
    "n_skipped_patches",
    "valid_patch_fraction",
]

croma_deep_cols = []
if croma_features is not None:
    for c in df.columns:
        cl = c.lower()

        if not c.startswith("croma_"):
            continue

        if not is_numeric(df, c):
            continue

        if any(k in cl for k in croma_metadata_keywords):
            continue

        croma_deep_cols.append(c)

snunet_deep_cols = deduplicate_keep_order(snunet_deep_cols)
floodvit_deep_cols = deduplicate_keep_order(floodvit_deep_cols)
croma_deep_cols = deduplicate_keep_order(croma_deep_cols)

print()
print("=" * 100)
print("DEEP FEATURE COUNTS")
print("SNUNet deep:", len(snunet_deep_cols))
print("FloodViT deep:", len(floodvit_deep_cols))
print("CROMA deep:", len(croma_deep_cols))


DEEP_FEATURE_GROUPS = {}

if len(snunet_deep_cols) > 0:
    DEEP_FEATURE_GROUPS["snunet_pca"] = snunet_deep_cols

if len(floodvit_deep_cols) > 0:
    DEEP_FEATURE_GROUPS["floodvit_pca"] = floodvit_deep_cols

if len(croma_deep_cols) > 0:
    DEEP_FEATURE_GROUPS["croma_pca"] = croma_deep_cols

if len(snunet_deep_cols) > 0 and len(floodvit_deep_cols) > 0:
    DEEP_FEATURE_GROUPS["snunet_floodvit_pca"] = snunet_deep_cols + floodvit_deep_cols

if len(floodvit_deep_cols) > 0 and len(croma_deep_cols) > 0:
    DEEP_FEATURE_GROUPS["floodvit_croma_pca"] = floodvit_deep_cols + croma_deep_cols

if len(snunet_deep_cols) > 0 and len(floodvit_deep_cols) > 0 and len(croma_deep_cols) > 0:
    DEEP_FEATURE_GROUPS["all_deep_pca"] = snunet_deep_cols + floodvit_deep_cols + croma_deep_cols


print()
print("Deep groups:")
for k, v in DEEP_FEATURE_GROUPS.items():
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
    n_splits = cv.get_n_splits()
    min_train_n = min_train_size_for_cv(len(labeled), n_splits)

    print("CV splits:", n_splits)
    print("Minimum train fold size:", min_train_n)

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
        "pca_components": 0,
        "n_rows": len(labeled),
        "n_tabular_features": 0,
        "n_deep_features_before_pca": 0,
        "n_features_after_pca_expected": 0,
        **metrics,
    }

    all_results.append(row)
    print(row)

    tmp = pd.DataFrame(
        {
            "target": target,
            "model": "baseline_median",
            "feature_set": "none",
            "pca_components": 0,
            "GFD_ID": labeled["GFD_ID"].values,
            "true_value": y_raw,
            "pred_value": safe_expm1(pred_log),
            "true_log1p": y_log,
            "pred_log1p": pred_log,
        }
    )
    all_cv_preds.append(tmp)

    # -----------------------------
    # Tabular-only baselines
    # -----------------------------
    for base_name, base_cols in BASE_FEATURE_SETS.items():
        tabular_cols = [c for c in base_cols if c in labeled.columns and is_numeric(labeled, c)]
        tabular_cols = [c for c in tabular_cols if not labeled[c].isna().all()]

        if len(tabular_cols) == 0:
            continue

        X = labeled[tabular_cols].copy()

        for model_name in MODEL_NAMES:
            print()
            print(
                "Target:", target,
                "| Feature set:", base_name,
                "| Model:", model_name,
                "| Tabular features:", len(tabular_cols),
            )

            model = make_tabular_only_pipeline(model_name, tabular_cols)

            try:
                pred_log = cross_val_predict(model, X, y_log, cv=cv)
                metrics = evaluate(y_log, pred_log)

                row = {
                    "target": target,
                    "model": model_name,
                    "feature_set": base_name,
                    "pca_components": 0,
                    "n_rows": len(labeled),
                    "n_tabular_features": len(tabular_cols),
                    "n_deep_features_before_pca": 0,
                    "n_features_after_pca_expected": len(tabular_cols),
                    **metrics,
                }

                all_results.append(row)
                print(row)

                tmp = pd.DataFrame(
                    {
                        "target": target,
                        "model": model_name,
                        "feature_set": base_name,
                        "pca_components": 0,
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

                all_results.append(
                    {
                        "target": target,
                        "model": model_name,
                        "feature_set": base_name,
                        "pca_components": 0,
                        "n_rows": len(labeled),
                        "n_tabular_features": len(tabular_cols),
                        "n_deep_features_before_pca": 0,
                        "n_features_after_pca_expected": len(tabular_cols),
                        "error": str(e),
                    }
                )

    # -----------------------------
    # Tabular + PCA deep features
    # -----------------------------
    for base_name, base_cols in BASE_FEATURE_SETS.items():
        tabular_cols = [c for c in base_cols if c in labeled.columns and is_numeric(labeled, c)]
        tabular_cols = [c for c in tabular_cols if not labeled[c].isna().all()]

        if len(tabular_cols) == 0:
            continue

        for deep_group_name, deep_cols_all in DEEP_FEATURE_GROUPS.items():
            deep_cols = [
                c for c in deep_cols_all
                if c in labeled.columns and is_numeric(labeled, c)
            ]
            deep_cols = [c for c in deep_cols if not labeled[c].isna().all()]

            if len(deep_cols) == 0:
                continue

            for n_pca in PCA_COMPONENTS_TO_TEST:
                # PCA n_components must be <= smallest training fold size and <= number of deep features
                if n_pca > min_train_n:
                    print(
                        "Skipping",
                        target,
                        base_name,
                        deep_group_name,
                        "PCA",
                        n_pca,
                        "because n_pca > min_train_n",
                        min_train_n,
                    )
                    continue

                if n_pca > len(deep_cols):
                    continue

                feature_set_name = f"{base_name}_plus_{deep_group_name}_{n_pca}"

                X_cols = tabular_cols + deep_cols
                X = labeled[X_cols].copy()

                for model_name in MODEL_NAMES:
                    print()
                    print(
                        "Target:", target,
                        "| Feature set:", feature_set_name,
                        "| Model:", model_name,
                        "| Tabular:", len(tabular_cols),
                        "| Deep before PCA:", len(deep_cols),
                        "| PCA:", n_pca,
                    )

                    model = make_tabular_deep_pca_pipeline(
                        model_name=model_name,
                        tabular_cols=tabular_cols,
                        deep_cols=deep_cols,
                        n_components=n_pca,
                    )

                    try:
                        pred_log = cross_val_predict(model, X, y_log, cv=cv)
                        metrics = evaluate(y_log, pred_log)

                        row = {
                            "target": target,
                            "model": model_name,
                            "feature_set": feature_set_name,
                            "pca_components": n_pca,
                            "n_rows": len(labeled),
                            "n_tabular_features": len(tabular_cols),
                            "n_deep_features_before_pca": len(deep_cols),
                            "n_features_after_pca_expected": len(tabular_cols) + n_pca,
                            **metrics,
                        }

                        all_results.append(row)
                        print(row)

                        tmp = pd.DataFrame(
                            {
                                "target": target,
                                "model": model_name,
                                "feature_set": feature_set_name,
                                "pca_components": n_pca,
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

                        all_results.append(
                            {
                                "target": target,
                                "model": model_name,
                                "feature_set": feature_set_name,
                                "pca_components": n_pca,
                                "n_rows": len(labeled),
                                "n_tabular_features": len(tabular_cols),
                                "n_deep_features_before_pca": len(deep_cols),
                                "n_features_after_pca_expected": len(tabular_cols) + n_pca,
                                "error": str(e),
                            }
                        )


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
        print(sub.head(20).to_string(index=False))
    else:
        print(sub.to_string(index=False))

print()
print("=" * 100)
print("DONE")