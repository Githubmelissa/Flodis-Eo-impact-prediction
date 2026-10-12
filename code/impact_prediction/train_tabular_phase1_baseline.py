from pathlib import Path
import ast
import re
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
FEATURE_ROOT = Path("/home/melissa/flood_project/data/FLODIS_TABULAR_PHASE1")
TARGET_ROOT = Path("/home/melissa/flood_project/data")

WORLDPOP_DIR = FEATURE_ROOT / "WorldPop_features"
GPM_DIR = FEATURE_ROOT / "GPM_features"
SRTM_DIR = FEATURE_ROOT / "SRTM_Features"   # IMPORTANT: capital F

DISPLACEMENT_PATH = TARGET_ROOT / "FLODIS_displacement_clean.csv"
MORT_DAMAGE_PATH = TARGET_ROOT / "FLODIS_mortality_damage_GFD_ready.csv"

OUT_DIR = Path("/home/melissa/flood_project/models_benchmark/impact_prediction/outputs")
OUT_DIR.mkdir(parents=True, exist_ok=True)

OUT_TABLE = OUT_DIR / "tabular_phase1_event_table_FIXED.csv"
OUT_RESULTS = OUT_DIR / "tabular_phase1_model_results_FIXED.csv"
OUT_CV_PREDS = OUT_DIR / "tabular_phase1_cv_predictions_FIXED.csv"

YEARS = [2015, 2016, 2017, 2018]

TARGETS = [
    "displacements",
    "total_deaths",
    "total_affected_EMDAT",
    "total_damages_000_USD",
]


# =====================================================
# HELPERS
# =====================================================
def read_yearly_files(folder, prefix):
    dfs = []

    for year in YEARS:
        path = folder / f"{prefix}_phase1_{year}.csv"

        if not path.exists():
            raise FileNotFoundError(f"Missing file: {path}")

        df = pd.read_csv(path)
        df["source_year_file"] = year
        dfs.append(df)

    out = pd.concat(dfs, ignore_index=True)
    out["GFD_ID"] = pd.to_numeric(out["GFD_ID"], errors="coerce").astype("Int64")
    out = out[out["GFD_ID"].notna()].copy()
    out["GFD_ID"] = out["GFD_ID"].astype(int)

    return out


def unique_join(values):
    vals = []

    for v in values:
        if pd.isna(v):
            continue
        vals.append(str(v))

    vals = sorted(set(vals))
    return ",".join(vals)


def parse_gfd_ids(value):
    """
    Handles formats like:
    [4580, 4581, 4583]
    '4580'
    '4580, 4581'
    """
    if pd.isna(value):
        return []

    if isinstance(value, (int, np.integer)):
        return [int(value)]

    if isinstance(value, (float, np.floating)):
        if np.isnan(value):
            return []
        return [int(value)]

    text = str(value).strip()

    if text == "" or text.lower() == "nan":
        return []

    try:
        parsed = ast.literal_eval(text)

        if isinstance(parsed, (list, tuple, set)):
            ids = []
            for x in parsed:
                try:
                    ids.append(int(float(x)))
                except Exception:
                    pass
            return ids

        if isinstance(parsed, (int, float)):
            return [int(parsed)]

    except Exception:
        pass

    ids = re.findall(r"\d+", text)
    return [int(x) for x in ids]


def rmse(y_true, y_pred):
    return mean_squared_error(y_true, y_pred) ** 0.5


def safe_expm1(log_values):
    """
    Convert log1p predictions back to original scale.
    Main comparison should use log metrics.
    """
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
# TARGET BUILDERS
# =====================================================
def build_displacement_targets(path):
    """
    Build clean displacement target table.

    IMPORTANT:
    We do NOT use the displacements column from WorldPop/GPM/SRTM files.
    We use FLODIS_displacement_clean.csv only.
    """

    if not path.exists():
        raise FileNotFoundError(f"Missing displacement target file: {path}")

    disp = pd.read_csv(path)

    print()
    print("=" * 100)
    print("Reading displacement target file")
    print(path)
    print("shape:", disp.shape)
    print("columns:", list(disp.columns))

    if "displacements" not in disp.columns:
        raise ValueError("No 'displacements' column found in displacement target file.")

    rows = []

    if "GFD_ID" in disp.columns:
        tmp = disp[["GFD_ID", "displacements"]].copy()
        tmp["GFD_ID"] = pd.to_numeric(tmp["GFD_ID"], errors="coerce")
        tmp["displacements"] = pd.to_numeric(tmp["displacements"], errors="coerce")
        tmp = tmp[tmp["GFD_ID"].notna()].copy()
        tmp["GFD_ID"] = tmp["GFD_ID"].astype(int)
        rows = tmp.to_dict("records")

    elif "GFD_matches_nr" in disp.columns:
        for _, row in disp.iterrows():
            gfd_ids = parse_gfd_ids(row["GFD_matches_nr"])
            displacement_value = pd.to_numeric(row["displacements"], errors="coerce")

            for gfd_id in gfd_ids:
                rows.append(
                    {
                        "GFD_ID": int(gfd_id),
                        "displacements": displacement_value,
                    }
                )

    else:
        raise ValueError(
            "Displacement file has neither 'GFD_ID' nor 'GFD_matches_nr'. "
            "Cannot merge displacement targets."
        )

    out = pd.DataFrame(rows)

    if len(out) == 0:
        raise ValueError("No displacement target rows were created.")

    out["displacements"] = pd.to_numeric(out["displacements"], errors="coerce")

    out = (
        out
        .groupby("GFD_ID")
        .agg(
            displacements=("displacements", lambda s: s.sum(min_count=1)),
            displacement_records=("displacements", "count"),
        )
        .reset_index()
    )

    print("Exploded displacement target shape:", out.shape)
    print("Unique GFD_ID in displacement target:", out["GFD_ID"].nunique())
    print("Non-null displacements:", out["displacements"].notna().sum())

    return out


def build_mortality_damage_targets(path):
    if not path.exists():
        print("WARNING: mortality/damage file not found:", path)
        return pd.DataFrame(columns=["GFD_ID"])

    md = pd.read_csv(path)

    print()
    print("=" * 100)
    print("Reading mortality/damage target file")
    print(path)
    print("shape:", md.shape)
    print("columns:", list(md.columns))

    if "total_damages_(000_USD)" in md.columns:
        md = md.rename(
            columns={
                "total_damages_(000_USD)": "total_damages_000_USD"
            }
        )

    if "GFD_ID" not in md.columns:
        raise ValueError("Mortality/damage file must contain GFD_ID.")

    md["GFD_ID"] = pd.to_numeric(md["GFD_ID"], errors="coerce")
    md = md[md["GFD_ID"].notna()].copy()
    md["GFD_ID"] = md["GFD_ID"].astype(int)

    target_cols = [
        c for c in [
            "total_deaths",
            "total_affected_EMDAT",
            "total_damages_000_USD",
        ]
        if c in md.columns
    ]

    for c in target_cols:
        md[c] = pd.to_numeric(md[c], errors="coerce")

    agg_dict = {
        c: (c, lambda s: s.sum(min_count=1))
        for c in target_cols
    }

    agg_dict["mortality_damage_records"] = ("GFD_ID", "count")

    out = (
        md
        .groupby("GFD_ID")
        .agg(**agg_dict)
        .reset_index()
    )

    print("Mortality/damage target shape:", out.shape)
    print("Unique GFD_ID in mortality/damage target:", out["GFD_ID"].nunique())

    for c in target_cols:
        print(c, "non-null:", out[c].notna().sum())

    return out


# =====================================================
# LOAD FEATURES
# =====================================================
print("=" * 100)
print("Reading yearly feature files")

worldpop = read_yearly_files(WORLDPOP_DIR, "WorldPop")
gpm = read_yearly_files(GPM_DIR, "GPM")
srtm = read_yearly_files(SRTM_DIR, "SRTM")

print("WorldPop:", worldpop.shape, "unique GFD_ID:", worldpop["GFD_ID"].nunique())
print("GPM:", gpm.shape, "unique GFD_ID:", gpm["GFD_ID"].nunique())
print("SRTM:", srtm.shape, "unique GFD_ID:", srtm["GFD_ID"].nunique())

print()
print("WorldPop columns:", list(worldpop.columns))
print("GPM columns:", list(gpm.columns))
print("SRTM columns:", list(srtm.columns))


# =====================================================
# AGGREGATE FEATURES TO ONE ROW PER GFD_ID
# =====================================================
print()
print("=" * 100)
print("Aggregating FEATURES to one row per GFD_ID")
print("IMPORTANT: displacement is NOT taken from feature files.")

wp = (
    worldpop
    .groupby("GFD_ID")
    .agg(
        year=("year", "first"),
        iso3_list=("ISO3", unique_join),
        n_phase1_rows=("row_id", "count"),
        n_iso3=("ISO3", "nunique"),

        wp_flood_area_km2_mean=("flood_area_km2", "mean"),
        wp_flood_area_km2_max=("flood_area_km2", "max"),
        wp_population_total_sum=("population_total", "sum"),
        wp_population_total_mean=("population_total", "mean"),
        wp_population_mean_mean=("population_mean", "mean"),
        wp_population_max_max=("population_max", "max"),
    )
    .reset_index()
)

gpm_tmp = gpm.copy()

if "gfd_event_date" in gpm_tmp.columns:
    gpm_tmp["gfd_event_date"] = pd.to_datetime(
        gpm_tmp["gfd_event_date"],
        errors="coerce",
    )
    gpm_tmp["event_month"] = gpm_tmp["gfd_event_date"].dt.month
else:
    gpm_tmp["event_month"] = np.nan

gpm_event = (
    gpm_tmp
    .groupby("GFD_ID")
    .agg(
        gpm_count_mean=("gpm_count", "mean"),
        gpm_rainfall_total_mm_mean=("rainfall_total_mm", "mean"),
        gpm_rainfall_total_mm_max=("rainfall_total_mm", "max"),
        gpm_rainfall_max_mm_max=("rainfall_max_mm", "max"),
        gpm_rainfall_mean_mm_mean=("rainfall_mean_mm", "mean"),
        gpm_rainfall_days_mean=("rainfall_days", "mean"),
        event_month=("event_month", "first"),
    )
    .reset_index()
)

srtm_event = (
    srtm
    .groupby("GFD_ID")
    .agg(
        srtm_elevation_mean=("elevation_mean", "mean"),
        srtm_elevation_stdDev_mean=("elevation_stdDev", "mean"),
        srtm_elevation_min=("elevation_min", "min"),
        srtm_elevation_max=("elevation_max", "max"),
        srtm_slope_mean=("slope_mean", "mean"),
        srtm_slope_stdDev_mean=("slope_stdDev", "mean"),
        srtm_slope_min=("slope_min", "min"),
        srtm_slope_max=("slope_max", "max"),
    )
    .reset_index()
)

df = wp.merge(gpm_event, on="GFD_ID", how="left")
df = df.merge(srtm_event, on="GFD_ID", how="left")

print("Feature table shape before targets:", df.shape)
print("Feature table unique GFD_ID:", df["GFD_ID"].nunique())


# =====================================================
# MERGE CLEAN TARGETS
# =====================================================
disp_target = build_displacement_targets(DISPLACEMENT_PATH)
md_target = build_mortality_damage_targets(MORT_DAMAGE_PATH)

df = df.merge(disp_target, on="GFD_ID", how="left")
df = df.merge(md_target, on="GFD_ID", how="left")


# =====================================================
# SAVE EVENT TABLE
# =====================================================
df = df.replace([np.inf, -np.inf], np.nan)
df = df.sort_values("GFD_ID")
df.to_csv(OUT_TABLE, index=False)

print()
print("=" * 100)
print("FINAL EVENT TABLE")
print("Event table shape:", df.shape)
print("Unique GFD_ID:", df["GFD_ID"].nunique())
print("Saved:", OUT_TABLE)

print()
print("Target availability:")
for t in TARGETS:
    if t in df.columns:
        print(t, ":", df[t].notna().sum(), "/", len(df))
    else:
        print(t, ": missing")

print()
print("First rows target check:")
target_check_cols = [
    c for c in [
        "GFD_ID",
        "year",
        "iso3_list",
        "displacements",
        "total_deaths",
        "total_affected_EMDAT",
        "total_damages_000_USD",
    ]
    if c in df.columns
]
print(df[target_check_cols].head(20).to_string(index=False))


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
    # Baseline median
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
    # Real models
    # -----------------------------
    for feature_set_name, feature_cols in FEATURE_SETS.items():
        available_cols = [c for c in feature_cols if c in labeled.columns]
        X = labeled[available_cols].copy()

        X = X[[c for c in X.columns if not X[c].isna().all()]]

        if X.shape[1] == 0:
            continue

        for model_name in MODEL_NAMES:
            print()
            print("Target:", target, "| Feature set:", feature_set_name, "| Model:", model_name)

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
print("Event table:", OUT_TABLE)
print("Results:", OUT_RESULTS)
print("CV predictions:", OUT_CV_PREDS)

print()
print("BEST RESULTS PER TARGET:")
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