from pathlib import Path
import warnings

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import RidgeCV
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

CLAY_TABLE_PATH = Path(
    "/home/melissa/flood_project/models_benchmark/Clay/flodis_outputs/clay_event_features_valid_only/all_events_clay_event_features_valid_only.csv"
)

OUTPUT_DIR = Path(
    "/home/melissa/flood_project/models_benchmark/impact_prediction/outputs"
)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_EVENT_TABLE = OUTPUT_DIR / "tabular_clay_pca_fusion_event_table.csv"
OUTPUT_RESULTS = OUTPUT_DIR / "tabular_clay_pca_fusion_model_results.csv"
OUTPUT_CV_PREDS = OUTPUT_DIR / "tabular_clay_pca_fusion_cv_predictions.csv"


# =====================================================
# SETTINGS
# =====================================================
RANDOM_STATE = 42

TARGETS = [
    "displacements",
    "total_deaths",
    "total_affected_EMDAT",
    "total_damages_000_USD",
]

META_COLS = {
    "GFD_ID",
    "event_id",
    "row_id",
    "ISO3",
    "year",
    "gfd_event_date",
    "during_start",
    "during_end",
    "source_year_file",
}

PCA_COMPONENTS = [4, 8, 16, 32]


# =====================================================
# BASIC HELPERS
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


def get_models():
    return {
        "ridge": RidgeCV(alphas=np.logspace(-4, 4, 25)),
        "random_forest": RandomForestRegressor(
            n_estimators=500,
            random_state=RANDOM_STATE,
            min_samples_leaf=2,
            n_jobs=-1,
        ),
        "extra_trees": ExtraTreesRegressor(
            n_estimators=500,
            random_state=RANDOM_STATE,
            min_samples_leaf=2,
            n_jobs=-1,
        ),
    }


def numeric_existing(df, cols):
    out = []
    for c in cols:
        if c in df.columns and pd.api.types.is_numeric_dtype(df[c]):
            out.append(c)
    return out


def get_feature_groups(df):
    all_numeric = [
        c for c in df.columns
        if pd.api.types.is_numeric_dtype(df[c])
    ]

    worldpop_cols = [
        c for c in all_numeric
        if (
            "population" in c.lower()
            or c == "flood_area_km2"
        )
    ]

    gpm_cols = [
        c for c in all_numeric
        if (
            "gpm" in c.lower()
            or "rainfall" in c.lower()
        )
    ]

    srtm_cols = [
        c for c in all_numeric
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
        c for c in all_numeric
        if (
            c.startswith("clay_before_")
            or c.startswith("clay_after_")
            or c.startswith("clay_delta_")
        )
    ]

    # Remove target leakage / target columns from every group
    forbidden = set(TARGETS) | META_COLS

    def clean(cols):
        return [c for c in cols if c not in forbidden]

    groups = {
        "worldpop": clean(worldpop_cols),
        "gpm": clean(gpm_cols),
        "srtm": clean(srtm_cols),
        "clay_quality": clean(clay_quality_cols),
        "clay_deep": clean(clay_deep_cols),
    }

    return groups


def make_preprocessor(base_cols, clay_deep_cols=None, pca_components=None):
    transformers = []

    if len(base_cols) > 0:
        base_pipe = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
            ]
        )
        transformers.append(("base", base_pipe, base_cols))

    if clay_deep_cols is not None and len(clay_deep_cols) > 0:
        if pca_components is None:
            clay_pipe = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler()),
                ]
            )
        else:
            clay_pipe = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler()),
                    ("pca", PCA(n_components=pca_components, random_state=RANDOM_STATE)),
                ]
            )

        transformers.append(("clay_deep", clay_pipe, clay_deep_cols))

    if len(transformers) == 0:
        raise ValueError("No features provided to preprocessor")

    return ColumnTransformer(transformers=transformers, remainder="drop")


def make_pipeline(model, base_cols, clay_deep_cols=None, pca_components=None):
    preprocessor = make_preprocessor(
        base_cols=base_cols,
        clay_deep_cols=clay_deep_cols,
        pca_components=pca_components,
    )

    return Pipeline(
        steps=[
            ("preprocess", preprocessor),
            ("model", model),
        ]
    )


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


def run_cv_for_candidate(df, target, candidate_name, base_cols, clay_deep_cols, pca_components):
    labeled = df[df[target].notna()].copy()

    # Remove negative or invalid labels if any
    labeled = labeled[labeled[target] >= 0].copy()

    n = len(labeled)
    if n < 5:
        return [], []

    n_splits = min(5, n)
    min_train_size = n - int(np.ceil(n / n_splits))

    # PCA must fit inside each training fold
    if pca_components is not None:
        if pca_components >= min_train_size:
            return [], []

    y_original = labeled[target].astype(float).values
    y_log = safe_log1p(y_original)

    cv = KFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)

    result_rows = []
    pred_rows = []

    # Baseline median
    if candidate_name == "baseline_median":
        model = DummyRegressor(strategy="median")
        y_pred_log = cross_val_predict(model, np.zeros((n, 1)), y_log, cv=cv)

        metrics = evaluate_predictions(y_original, y_pred_log)

        result_rows.append(
            {
                "target": target,
                "candidate": candidate_name,
                "model": "baseline_median",
                "n_samples": n,
                "n_features_base": 0,
                "n_features_clay_deep": 0,
                "pca_components": np.nan,
                **metrics,
            }
        )

        for gfd_id, yt, yp_log in zip(labeled["GFD_ID"], y_original, y_pred_log):
            pred_rows.append(
                {
                    "target": target,
                    "candidate": candidate_name,
                    "model": "baseline_median",
                    "GFD_ID": int(gfd_id),
                    "y_true": float(yt),
                    "y_true_log": float(np.log1p(yt)),
                    "y_pred_log": float(yp_log),
                    "y_pred": float(safe_expm1([yp_log])[0]),
                }
            )

        return result_rows, pred_rows

    feature_cols_for_df = list(base_cols)
    if clay_deep_cols is not None:
        feature_cols_for_df += list(clay_deep_cols)

    if len(feature_cols_for_df) == 0:
        return [], []

    X = labeled[feature_cols_for_df].copy()

    for model_name, model in get_models().items():
        pipe = make_pipeline(
            model=model,
            base_cols=base_cols,
            clay_deep_cols=clay_deep_cols,
            pca_components=pca_components,
        )

        try:
            y_pred_log = cross_val_predict(pipe, X, y_log, cv=cv)
        except Exception as e:
            print(
                f"SKIP target={target} candidate={candidate_name} model={model_name}: {repr(e)}"
            )
            continue

        metrics = evaluate_predictions(y_original, y_pred_log)

        result_rows.append(
            {
                "target": target,
                "candidate": candidate_name,
                "model": model_name,
                "n_samples": n,
                "n_features_base": len(base_cols),
                "n_features_clay_deep": 0 if clay_deep_cols is None else len(clay_deep_cols),
                "pca_components": np.nan if pca_components is None else pca_components,
                **metrics,
            }
        )

        for gfd_id, yt, yp_log in zip(labeled["GFD_ID"], y_original, y_pred_log):
            pred_rows.append(
                {
                    "target": target,
                    "candidate": candidate_name,
                    "model": model_name,
                    "GFD_ID": int(gfd_id),
                    "y_true": float(yt),
                    "y_true_log": float(np.log1p(yt)),
                    "y_pred_log": float(yp_log),
                    "y_pred": float(safe_expm1([yp_log])[0]),
                }
            )

    return result_rows, pred_rows


def build_candidates(groups):
    worldpop = groups["worldpop"]
    gpm = groups["gpm"]
    srtm = groups["srtm"]
    clay_quality = groups["clay_quality"]
    clay_deep = groups["clay_deep"]

    worldpop_gpm = sorted(set(worldpop + gpm))
    worldpop_gpm_srtm = sorted(set(worldpop + gpm + srtm))

    candidates = []

    candidates.append(
        {
            "name": "baseline_median",
            "base_cols": [],
            "clay_deep_cols": None,
            "pca": None,
        }
    )

    candidates.append(
        {
            "name": "worldpop_gpm",
            "base_cols": worldpop_gpm,
            "clay_deep_cols": None,
            "pca": None,
        }
    )

    candidates.append(
        {
            "name": "worldpop_gpm_srtm",
            "base_cols": worldpop_gpm_srtm,
            "clay_deep_cols": None,
            "pca": None,
        }
    )

    candidates.append(
        {
            "name": "clay_quality",
            "base_cols": clay_quality,
            "clay_deep_cols": None,
            "pca": None,
        }
    )

    candidates.append(
        {
            "name": "worldpop_gpm_plus_clay_quality",
            "base_cols": sorted(set(worldpop_gpm + clay_quality)),
            "clay_deep_cols": None,
            "pca": None,
        }
    )

    candidates.append(
        {
            "name": "worldpop_gpm_srtm_plus_clay_quality",
            "base_cols": sorted(set(worldpop_gpm_srtm + clay_quality)),
            "clay_deep_cols": None,
            "pca": None,
        }
    )

    for k in PCA_COMPONENTS:
        candidates.append(
            {
                "name": f"clay_pca_{k}",
                "base_cols": [],
                "clay_deep_cols": clay_deep,
                "pca": k,
            }
        )

        candidates.append(
            {
                "name": f"worldpop_gpm_plus_clay_pca_{k}",
                "base_cols": worldpop_gpm,
                "clay_deep_cols": clay_deep,
                "pca": k,
            }
        )

        candidates.append(
            {
                "name": f"worldpop_gpm_srtm_plus_clay_pca_{k}",
                "base_cols": worldpop_gpm_srtm,
                "clay_deep_cols": clay_deep,
                "pca": k,
            }
        )

        candidates.append(
            {
                "name": f"worldpop_gpm_plus_clay_quality_pca_{k}",
                "base_cols": sorted(set(worldpop_gpm + clay_quality)),
                "clay_deep_cols": clay_deep,
                "pca": k,
            }
        )

        candidates.append(
            {
                "name": f"worldpop_gpm_srtm_plus_clay_quality_pca_{k}",
                "base_cols": sorted(set(worldpop_gpm_srtm + clay_quality)),
                "clay_deep_cols": clay_deep,
                "pca": k,
            }
        )

    return candidates


# =====================================================
# MAIN
# =====================================================
def main():
    print("=" * 100)
    print("Tabular + Clay PCA impact prediction")
    print("Tabular:", TABULAR_TABLE_PATH)
    print("Clay:", CLAY_TABLE_PATH)

    if not TABULAR_TABLE_PATH.exists():
        raise FileNotFoundError(TABULAR_TABLE_PATH)

    if not CLAY_TABLE_PATH.exists():
        raise FileNotFoundError(CLAY_TABLE_PATH)

    tabular = pd.read_csv(TABULAR_TABLE_PATH)
    clay = pd.read_csv(CLAY_TABLE_PATH)

    print("Tabular shape:", tabular.shape)
    print("Clay shape:", clay.shape)

    if "GFD_ID" not in tabular.columns:
        raise KeyError("GFD_ID missing from tabular table")

    if "GFD_ID" not in clay.columns:
        raise KeyError("GFD_ID missing from Clay table")

    # Avoid duplicate target columns from Clay if they ever exist
    drop_from_clay = [c for c in TARGETS if c in clay.columns]
    if len(drop_from_clay) > 0:
        clay = clay.drop(columns=drop_from_clay)

    df = tabular.merge(clay, on="GFD_ID", how="inner")

    print("Merged shape:", df.shape)
    print("Unique GFD_ID:", df["GFD_ID"].nunique())

    df.to_csv(OUTPUT_EVENT_TABLE, index=False)
    print("Saved event table:", OUTPUT_EVENT_TABLE)

    groups = get_feature_groups(df)

    print()
    print("Feature group sizes:")
    for name, cols in groups.items():
        print(f"  {name}: {len(cols)}")

    print()
    print("Target availability:")
    for target in TARGETS:
        if target in df.columns:
            print(f"  {target}: {int(df[target].notna().sum())}/{len(df)}")
        else:
            print(f"  {target}: MISSING")

    candidates = build_candidates(groups)

    print()
    print("Candidates:", len(candidates))

    all_results = []
    all_preds = []

    for target in TARGETS:
        if target not in df.columns:
            print("Skipping missing target:", target)
            continue

        print()
        print("=" * 100)
        print("TARGET:", target)

        for cand in candidates:
            name = cand["name"]
            base_cols = cand["base_cols"]
            clay_deep_cols = cand["clay_deep_cols"]
            pca = cand["pca"]

            print("Running:", target, "|", name)

            result_rows, pred_rows = run_cv_for_candidate(
                df=df,
                target=target,
                candidate_name=name,
                base_cols=base_cols,
                clay_deep_cols=clay_deep_cols,
                pca_components=pca,
            )

            all_results.extend(result_rows)
            all_preds.extend(pred_rows)

            pd.DataFrame(all_results).to_csv(OUTPUT_RESULTS, index=False)
            pd.DataFrame(all_preds).to_csv(OUTPUT_CV_PREDS, index=False)

    results_df = pd.DataFrame(all_results)
    preds_df = pd.DataFrame(all_preds)

    if len(results_df) > 0:
        results_df = results_df.sort_values(
            ["target", "log_rmse", "candidate", "model"]
        )
        results_df.to_csv(OUTPUT_RESULTS, index=False)

    if len(preds_df) > 0:
        preds_df.to_csv(OUTPUT_CV_PREDS, index=False)

    print()
    print("=" * 100)
    print("DONE")
    print("Saved event table:", OUTPUT_EVENT_TABLE)
    print("Saved results:", OUTPUT_RESULTS)
    print("Saved CV predictions:", OUTPUT_CV_PREDS)

    if len(results_df) > 0:
        print()
        print("Best model per target:")
        for target in TARGETS:
            sub = results_df[results_df["target"] == target].copy()
            if len(sub) == 0:
                continue

            best = sub.sort_values("log_rmse").iloc[0]
            print()
            print(target)
            print(best[[
                "candidate",
                "model",
                "n_samples",
                "n_features_base",
                "n_features_clay_deep",
                "pca_components",
                "log_mae",
                "log_rmse",
                "log_r2",
                "original_rmse",
                "original_r2",
            ]])


if __name__ == "__main__":
    main()