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
# SETTINGS
# =====================================================
N_SEEDS = 30
SEEDS = list(range(N_SEEDS))

TARGET = "displacements"

OUT_DIR = Path(
    "/home/melissa/flood_project/models_benchmark/impact_prediction/outputs"
)
OUT_DIR.mkdir(parents=True, exist_ok=True)

COMPACT_FUSION_TABLE = OUT_DIR / "tabular_s1_compact_fusion_event_table.csv"
PCA_FUSION_TABLE = OUT_DIR / "tabular_s1_pca_fusion_event_table.csv"

OUT_RESULTS = OUT_DIR / "repeated_cv_displacement_model_results.csv"
OUT_SUMMARY = OUT_DIR / "repeated_cv_displacement_summary.csv"
OUT_PAIRWISE = OUT_DIR / "repeated_cv_displacement_pairwise_comparison.csv"
OUT_PREDS = OUT_DIR / "repeated_cv_displacement_cv_predictions.csv"


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

WORLDPOP_GPM = WORLDPOP_FEATURES + GPM_FEATURES
WORLDPOP_GPM_SRTM = WORLDPOP_FEATURES + GPM_FEATURES + SRTM_FEATURES


# =====================================================
# HELPERS
# =====================================================
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


def make_cv(seed, n_rows):
    return KFold(
        n_splits=min(5, n_rows),
        shuffle=True,
        random_state=seed,
    )


def make_rf(seed):
    return RandomForestRegressor(
        n_estimators=500,
        max_depth=4,
        min_samples_leaf=3,
        random_state=seed,
        n_jobs=-1,
    )


def make_simple_rf_pipeline(seed):
    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("model", make_rf(seed)),
        ]
    )


def make_pca_rf_pipeline(seed, tabular_cols, deep_cols, n_components):
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
                        ("pca", PCA(n_components=n_components, random_state=seed)),
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
            ("model", make_rf(seed)),
        ]
    )


def select_s1_quality_features(df):
    quality_keywords = [
        "n_total_patches",
        "n_valid_patches",
        "n_skipped_patches",
        "valid_patch_fraction",
    ]

    cols = []

    for c in df.columns:
        cl = c.lower()

        if c == "GFD_ID":
            continue

        if not is_numeric(df, c):
            continue

        if any(k in cl for k in quality_keywords):
            cols.append(c)

    return deduplicate_keep_order(cols)


def select_s1_compact_features(df):
    """
    Same compact selection logic used in the previous compact fusion script.
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

    for c in df.columns:
        cl = c.lower()

        if c == "GFD_ID":
            continue

        if not is_numeric(df, c):
            continue

        if any(k in cl for k in forbidden_keywords):
            continue

        if any(k in cl for k in deep_keywords):
            continue

        if any(k in cl for k in useful_keywords):
            cols.append(c)

    return deduplicate_keep_order(cols)


# =====================================================
# LOAD COMPACT FUSION TABLE
# =====================================================
print("=" * 100)
print("Loading compact fusion table")
print(COMPACT_FUSION_TABLE)

df = pd.read_csv(COMPACT_FUSION_TABLE)
df["GFD_ID"] = pd.to_numeric(df["GFD_ID"], errors="coerce")
df = df[df["GFD_ID"].notna()].copy()
df["GFD_ID"] = df["GFD_ID"].astype(int)

df = df.replace([np.inf, -np.inf], np.nan)

labeled = df[df[TARGET].notna()].copy()
labeled = labeled[labeled[TARGET] >= 0].copy()
labeled = labeled.sort_values("GFD_ID").reset_index(drop=True)

print("Fusion table shape:", df.shape)
print("Labeled rows for displacement:", labeled.shape)
print("Unique GFD_ID:", labeled["GFD_ID"].nunique())


# =====================================================
# SELECT FEATURES
# =====================================================
worldpop_gpm_cols = [
    c for c in WORLDPOP_GPM
    if c in labeled.columns and is_numeric(labeled, c)
]

worldpop_gpm_srtm_cols = [
    c for c in WORLDPOP_GPM_SRTM
    if c in labeled.columns and is_numeric(labeled, c)
]

s1_quality_cols = select_s1_quality_features(labeled)
s1_compact_cols = select_s1_compact_features(labeled)

print()
print("=" * 100)
print("Feature counts")
print("WorldPop + GPM:", len(worldpop_gpm_cols))
print("WorldPop + GPM + SRTM:", len(worldpop_gpm_srtm_cols))
print("S1 quality:", len(s1_quality_cols))
print("S1 compact:", len(s1_compact_cols))


# =====================================================
# LOAD PCA FUSION TABLE FOR DEEP PCA CANDIDATE
# =====================================================
pca_labeled = None
snunet_floodvit_deep_cols = []

if PCA_FUSION_TABLE.exists():
    print()
    print("=" * 100)
    print("Loading PCA fusion table")
    print(PCA_FUSION_TABLE)

    pca_df = pd.read_csv(PCA_FUSION_TABLE)
    pca_df["GFD_ID"] = pd.to_numeric(pca_df["GFD_ID"], errors="coerce")
    pca_df = pca_df[pca_df["GFD_ID"].notna()].copy()
    pca_df["GFD_ID"] = pca_df["GFD_ID"].astype(int)
    pca_df = pca_df.replace([np.inf, -np.inf], np.nan)

    pca_labeled = pca_df[pca_df[TARGET].notna()].copy()
    pca_labeled = pca_labeled[pca_labeled[TARGET] >= 0].copy()
    pca_labeled = pca_labeled.sort_values("GFD_ID").reset_index(drop=True)

    snunet_deep_cols = [
        c for c in pca_labeled.columns
        if c.startswith("snunet_deep_") and is_numeric(pca_labeled, c)
    ]

    floodvit_deep_cols = [
        c for c in pca_labeled.columns
        if c.startswith("floodvit_deep_") and is_numeric(pca_labeled, c)
    ]

    snunet_floodvit_deep_cols = deduplicate_keep_order(
        snunet_deep_cols + floodvit_deep_cols
    )

    print("PCA table labeled rows:", pca_labeled.shape)
    print("SNUNet deep:", len(snunet_deep_cols))
    print("FloodViT deep:", len(floodvit_deep_cols))
    print("SNUNet + FloodViT deep:", len(snunet_floodvit_deep_cols))
else:
    print()
    print("=" * 100)
    print("WARNING: PCA fusion table not found. PCA candidate will be skipped.")
    print(PCA_FUSION_TABLE)


# =====================================================
# CANDIDATES
# =====================================================
candidates = []

candidates.append(
    {
        "candidate": "baseline_median",
        "table": "compact",
        "type": "baseline",
        "features": [],
    }
)

candidates.append(
    {
        "candidate": "rf_worldpop_gpm",
        "table": "compact",
        "type": "simple_rf",
        "features": worldpop_gpm_cols,
    }
)

candidates.append(
    {
        "candidate": "rf_worldpop_gpm_srtm",
        "table": "compact",
        "type": "simple_rf",
        "features": worldpop_gpm_srtm_cols,
    }
)

candidates.append(
    {
        "candidate": "rf_worldpop_gpm_plus_s1_quality",
        "table": "compact",
        "type": "simple_rf",
        "features": deduplicate_keep_order(worldpop_gpm_cols + s1_quality_cols),
    }
)

candidates.append(
    {
        "candidate": "rf_worldpop_gpm_srtm_plus_s1_quality",
        "table": "compact",
        "type": "simple_rf",
        "features": deduplicate_keep_order(worldpop_gpm_srtm_cols + s1_quality_cols),
    }
)

candidates.append(
    {
        "candidate": "rf_worldpop_gpm_plus_s1_compact",
        "table": "compact",
        "type": "simple_rf",
        "features": deduplicate_keep_order(worldpop_gpm_cols + s1_compact_cols),
    }
)

candidates.append(
    {
        "candidate": "rf_worldpop_gpm_srtm_plus_s1_compact",
        "table": "compact",
        "type": "simple_rf",
        "features": deduplicate_keep_order(worldpop_gpm_srtm_cols + s1_compact_cols),
    }
)

if pca_labeled is not None and len(snunet_floodvit_deep_cols) > 0:
    pca_tabular_cols = [
        c for c in WORLDPOP_GPM_SRTM
        if c in pca_labeled.columns and is_numeric(pca_labeled, c)
    ]

    candidates.append(
        {
            "candidate": "rf_worldpop_gpm_srtm_plus_snunet_floodvit_pca8",
            "table": "pca",
            "type": "pca_rf",
            "tabular_features": pca_tabular_cols,
            "deep_features": snunet_floodvit_deep_cols,
            "pca_components": 8,
        }
    )


print()
print("=" * 100)
print("Candidates")
for c in candidates:
    if c["type"] == "pca_rf":
        print(
            c["candidate"],
            "| tabular:",
            len(c["tabular_features"]),
            "| deep:",
            len(c["deep_features"]),
            "| PCA:",
            c["pca_components"],
        )
    else:
        print(c["candidate"], "| features:", len(c["features"]))


# =====================================================
# RUN REPEATED CV
# =====================================================
all_results = []
all_preds = []

for seed in SEEDS:
    print()
    print("=" * 100)
    print("SEED:", seed)
    print("=" * 100)

    for cand in candidates:
        candidate_name = cand["candidate"]

        if cand["table"] == "compact":
            data = labeled
        elif cand["table"] == "pca":
            data = pca_labeled
        else:
            raise ValueError(cand["table"])

        y_raw = data[TARGET].astype(float).values
        y_log = np.log1p(y_raw)
        cv = make_cv(seed, len(data))

        print()
        print("Candidate:", candidate_name)

        try:
            if cand["type"] == "baseline":
                X = np.zeros((len(data), 1))
                model = DummyRegressor(strategy="median")

            elif cand["type"] == "simple_rf":
                features = [
                    f for f in cand["features"]
                    if f in data.columns and is_numeric(data, f)
                ]
                features = [f for f in features if not data[f].isna().all()]

                X = data[features].copy()
                model = make_simple_rf_pipeline(seed)

            elif cand["type"] == "pca_rf":
                tabular_features = [
                    f for f in cand["tabular_features"]
                    if f in data.columns and is_numeric(data, f)
                ]
                tabular_features = [
                    f for f in tabular_features
                    if not data[f].isna().all()
                ]

                deep_features = [
                    f for f in cand["deep_features"]
                    if f in data.columns and is_numeric(data, f)
                ]
                deep_features = [
                    f for f in deep_features
                    if not data[f].isna().all()
                ]

                X = data[tabular_features + deep_features].copy()

                model = make_pca_rf_pipeline(
                    seed=seed,
                    tabular_cols=tabular_features,
                    deep_cols=deep_features,
                    n_components=cand["pca_components"],
                )

            else:
                raise ValueError(cand["type"])

            pred_log = cross_val_predict(model, X, y_log, cv=cv)
            metrics = evaluate(y_log, pred_log)

            if cand["type"] == "baseline":
                n_features = 0
                n_tabular = 0
                n_deep = 0
                pca_components = 0
            elif cand["type"] == "simple_rf":
                n_features = X.shape[1]
                n_tabular = X.shape[1]
                n_deep = 0
                pca_components = 0
            else:
                n_features = len(tabular_features) + cand["pca_components"]
                n_tabular = len(tabular_features)
                n_deep = len(deep_features)
                pca_components = cand["pca_components"]

            row = {
                "seed": seed,
                "target": TARGET,
                "candidate": candidate_name,
                "model": "random_forest" if cand["type"] != "baseline" else "baseline_median",
                "n_rows": len(data),
                "n_features_used": n_features,
                "n_tabular_features": n_tabular,
                "n_deep_features_before_pca": n_deep,
                "pca_components": pca_components,
                **metrics,
            }

            all_results.append(row)

            print(
                "log_RMSE:",
                round(metrics["log_rmse"], 4),
                "| log_R2:",
                round(metrics["log_r2"], 4),
            )

            pred_df = pd.DataFrame(
                {
                    "seed": seed,
                    "target": TARGET,
                    "candidate": candidate_name,
                    "GFD_ID": data["GFD_ID"].values,
                    "true_value": y_raw,
                    "pred_value": safe_expm1(pred_log),
                    "true_log1p": y_log,
                    "pred_log1p": pred_log,
                }
            )
            all_preds.append(pred_df)

        except Exception as e:
            print("ERROR:", e)
            all_results.append(
                {
                    "seed": seed,
                    "target": TARGET,
                    "candidate": candidate_name,
                    "model": "error",
                    "error": str(e),
                }
            )


# =====================================================
# SAVE RAW RESULTS
# =====================================================
results_df = pd.DataFrame(all_results)
results_df.to_csv(OUT_RESULTS, index=False)

preds_df = pd.concat(all_preds, ignore_index=True)
preds_df.to_csv(OUT_PREDS, index=False)


# =====================================================
# SUMMARY
# =====================================================
valid = results_df[results_df["log_rmse"].notna()].copy()

summary = (
    valid
    .groupby("candidate")
    .agg(
        n_runs=("seed", "count"),
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

# Win count: how often each candidate is best for a seed
best_each_seed = (
    valid
    .sort_values(["seed", "log_rmse"])
    .groupby("seed")
    .head(1)
)

win_counts = (
    best_each_seed
    .groupby("candidate")
    .size()
    .reset_index(name="win_count")
)

summary = summary.merge(win_counts, on="candidate", how="left")
summary["win_count"] = summary["win_count"].fillna(0).astype(int)
summary = summary.sort_values("mean_log_rmse").reset_index(drop=True)

summary.to_csv(OUT_SUMMARY, index=False)


# =====================================================
# PAIRWISE COMPARISON AGAINST TABULAR BASELINE
# =====================================================
REFERENCE = "rf_worldpop_gpm_srtm"

pairwise_rows = []

pivot = valid.pivot_table(
    index="seed",
    columns="candidate",
    values="log_rmse",
    aggfunc="first",
)

if REFERENCE in pivot.columns:
    for cand in pivot.columns:
        if cand == REFERENCE:
            continue

        tmp = pivot[[REFERENCE, cand]].dropna().copy()

        if len(tmp) == 0:
            continue

        diff = tmp[cand] - tmp[REFERENCE]

        pairwise_rows.append(
            {
                "reference_candidate": REFERENCE,
                "candidate": cand,
                "n_seeds": len(tmp),
                "mean_log_rmse_diff_candidate_minus_reference": diff.mean(),
                "std_log_rmse_diff": diff.std(),
                "median_log_rmse_diff": diff.median(),
                "candidate_better_count": int((diff < 0).sum()),
                "candidate_better_fraction": float((diff < 0).mean()),
            }
        )

pairwise = pd.DataFrame(pairwise_rows)
pairwise = pairwise.sort_values(
    "mean_log_rmse_diff_candidate_minus_reference"
).reset_index(drop=True)

pairwise.to_csv(OUT_PAIRWISE, index=False)


# =====================================================
# PRINT FINAL SUMMARY
# =====================================================
print()
print("=" * 100)
print("SAVED")
print("Raw repeated CV results:", OUT_RESULTS)
print("Summary:", OUT_SUMMARY)
print("Pairwise comparison:", OUT_PAIRWISE)
print("CV predictions:", OUT_PREDS)

print()
print("=" * 100)
print("SUMMARY SORTED BY MEAN LOG_RMSE")
print(summary.to_string(index=False))

print()
print("=" * 100)
print("PAIRWISE VS TABULAR RF WORLDPOP+GPM+SRTM")
print(pairwise.to_string(index=False))

print()
print("=" * 100)
print("DONE")