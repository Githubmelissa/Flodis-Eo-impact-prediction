from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.decomposition import PCA
from sklearn.dummy import DummyRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import RidgeCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


# =====================================================
# PATHS
# =====================================================
KURO_DIR = Path("/home/melissa/flood_project/models_benchmark/KuroSiwo")
CROMA_DIR = Path("/home/melissa/flood_project/models_benchmark/CROMA")

BASE_TABLE = (
    KURO_DIR
    / "flodis_outputs/merged_prediction_tables/s1_features_with_flodis_targets_all85.csv"
)

CROMA_TABLE = (
    CROMA_DIR
    / "flodis_outputs/croma_event_features_valid_only/all_events_croma_event_features_valid_only.csv"
)

OUT_MERGED = (
    KURO_DIR
    / "flodis_outputs/merged_prediction_tables/s1_features_with_croma_targets_all85.csv"
)

OUT_DIR = KURO_DIR / "flodis_outputs/prediction_results"
OUT_DIR.mkdir(parents=True, exist_ok=True)

OUT_RESULTS = OUT_DIR / "all_targets_with_croma_model_results.csv"
OUT_CV_PREDS = OUT_DIR / "all_targets_with_croma_cv_predictions.csv"


# =====================================================
# TARGETS
# =====================================================
TARGETS = [
    "displacements",
    "total_deaths",
    "total_affected_EMDAT",
    "total_damages_000_USD",
]

TARGET_AND_RECORD_COLS = [
    "displacements",
    "total_deaths",
    "total_affected_EMDAT",
    "total_damages_000_USD",
    "displacement_records",
    "mortality_damage_records",
]

DROP_KEYWORDS = [
    "flood_mask",
    "debug",
]


# =====================================================
# HELPERS
# =====================================================
def rmse(y_true, y_pred):
    return mean_squared_error(y_true, y_pred) ** 0.5


def safe_expm1(log_values):
    return np.expm1(np.maximum(log_values, 0))


def evaluate_predictions(y_log, pred_log):
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


def make_ridge_pipeline():
    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", RidgeCV(alphas=np.logspace(-4, 4, 25))),
        ]
    )


def make_pca_ridge_pipeline(n_components=10):
    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("pca", PCA(n_components=n_components, random_state=42)),
            ("model", RidgeCV(alphas=np.logspace(-4, 4, 25))),
        ]
    )


def clean_feature_table(df):
    df = df.copy()
    df = df.replace([np.inf, -np.inf], np.nan)
    return df


def drop_all_nan_columns(X):
    keep_cols = [c for c in X.columns if not X[c].isna().all()]
    return X[keep_cols]


def prepare_croma_table(croma):
    croma = croma.copy()

    rename_map = {
        "event_id": "croma_event_id",
        "n_total_patches": "croma_n_total_patches",
        "n_valid_patches": "croma_n_valid_patches",
        "n_skipped_patches": "croma_n_skipped_patches",
        "valid_patch_fraction": "croma_valid_patch_fraction",
    }

    croma = croma.rename(columns=rename_map)
    return croma


def is_numeric_feature(df, col):
    if col == "GFD_ID":
        return False

    if col in TARGET_AND_RECORD_COLS:
        return False

    if any(k in col.lower() for k in DROP_KEYWORDS):
        return False

    return pd.api.types.is_numeric_dtype(df[col])


def get_feature_cols(df, mode):
    cols = []

    for col in df.columns:
        if not is_numeric_feature(df, col):
            continue

        if mode == "quality":
            if (
                "n_valid_patches" in col
                or "valid_patch_fraction" in col
                or "n_skipped_patches" in col
                or "n_total_patches" in col
            ):
                cols.append(col)

        elif mode == "snunet":
            if col.startswith("snunet_"):
                cols.append(col)

        elif mode == "floodvit":
            if col.startswith("floodvit_"):
                cols.append(col)

        elif mode == "croma":
            if col.startswith("croma_"):
                cols.append(col)

        elif mode == "floodvit_croma":
            if col.startswith("floodvit_") or col.startswith("croma_"):
                cols.append(col)

        elif mode == "all_deep":
            if (
                col.startswith("snunet_")
                or col.startswith("floodvit_")
                or col.startswith("croma_")
            ):
                cols.append(col)

    return cols


# =====================================================
# LOAD + MERGE
# =====================================================
print("=" * 100)
print("Loading base table:")
print(BASE_TABLE)

base = pd.read_csv(BASE_TABLE)
base = clean_feature_table(base)

print("Base shape:", base.shape)

print()
print("Loading CROMA table:")
print(CROMA_TABLE)

croma = pd.read_csv(CROMA_TABLE)
croma = prepare_croma_table(croma)
croma = clean_feature_table(croma)

print("CROMA shape:", croma.shape)

df = base.merge(croma, on="GFD_ID", how="left")

df = clean_feature_table(df)
df.to_csv(OUT_MERGED, index=False)

print()
print("Merged shape:", df.shape)
print("Saved merged table:", OUT_MERGED)
print("Unique GFD_ID:", df["GFD_ID"].nunique())


# =====================================================
# TRAIN / EVALUATE
# =====================================================
all_results = []
all_cv_predictions = []

FEATURE_SETS = [
    ("quality", "ridge"),
    ("snunet", "ridge"),
    ("floodvit", "ridge"),
    ("croma", "ridge"),
    ("floodvit_croma", "ridge"),
    ("croma", "pca_ridge"),
    ("floodvit_croma", "pca_ridge"),
    ("all_deep", "pca_ridge"),
]

for target in TARGETS:
    print()
    print("#" * 100)
    print("TARGET:", target)
    print("#" * 100)

    labeled = df[df[target].notna()].copy()

    print("Labeled rows:", len(labeled))

    if len(labeled) < 5:
        print("SKIP: not enough rows.")
        continue

    cv = choose_cv(len(labeled))

    y_raw = labeled[target].astype(float).values
    y_log = np.log1p(y_raw)

    # =================================================
    # Baseline
    # =================================================
    dummy_X = np.zeros((len(labeled), 1))
    dummy = DummyRegressor(strategy="median")
    dummy_pred_log = cross_val_predict(dummy, dummy_X, y_log, cv=cv)

    metrics = evaluate_predictions(y_log, dummy_pred_log)

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

    # =================================================
    # Feature models
    # =================================================
    for feature_set, model_type in FEATURE_SETS:
        print()
        print("Feature set:", feature_set, "| Model:", model_type)

        feature_cols = get_feature_cols(labeled, feature_set)

        X = labeled[feature_cols].copy()
        X = drop_all_nan_columns(X)

        print("Features:", X.shape[1])

        if X.shape[1] == 0:
            print("SKIP: no features.")
            continue

        if model_type == "ridge":
            model = make_ridge_pipeline()
        elif model_type == "pca_ridge":
            model = make_pca_ridge_pipeline(n_components=10)
        else:
            raise ValueError(model_type)

        pred_log = cross_val_predict(model, X, y_log, cv=cv)

        metrics = evaluate_predictions(y_log, pred_log)

        row = {
            "target": target,
            "model": model_type,
            "feature_set": feature_set,
            "n_rows": len(labeled),
            "n_features": X.shape[1],
            **metrics,
        }

        all_results.append(row)
        print(row)

        tmp = pd.DataFrame(
            {
                "target": target,
                "GFD_ID": labeled["GFD_ID"].values,
                "model": model_type,
                "feature_set": feature_set,
                "true_value": y_raw,
                "pred_value": safe_expm1(pred_log),
                "true_log1p": y_log,
                "pred_log1p": pred_log,
            }
        )

        all_cv_predictions.append(tmp)


# =====================================================
# SAVE
# =====================================================
results_df = pd.DataFrame(all_results)
results_df = results_df.sort_values(["target", "log_rmse"])
results_df.to_csv(OUT_RESULTS, index=False)

cv_pred_df = pd.concat(all_cv_predictions, ignore_index=True)
cv_pred_df.to_csv(OUT_CV_PREDS, index=False)

print()
print("=" * 100)
print("SAVED:")
print("Results:", OUT_RESULTS)
print("CV predictions:", OUT_CV_PREDS)

print()
print("MODEL COMPARISON:")
print(results_df.to_string(index=False))

print()
print("=" * 100)
print("DONE")