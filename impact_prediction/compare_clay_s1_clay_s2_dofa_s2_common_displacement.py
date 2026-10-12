from pathlib import Path
import warnings

import numpy as np
import pandas as pd

from sklearn.decomposition import PCA
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import RepeatedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

warnings.filterwarnings("ignore")


# ============================================================
# PATHS
# ============================================================

# Exact 45-event cohort used in the previous fair comparison
COHORT_FILE = Path(
    "/home/melissa/flood_project/models_benchmark/"
    "impact_prediction/outputs/"
    "clay_s1_s2_common_displacement/"
    "common_event_cohort.csv"
)

CLAY_S1_FILE = Path(
    "/home/melissa/flood_project/models_benchmark/"
    "Clay/flodis_outputs/"
    "clay_event_features_valid_only/"
    "all_events_clay_event_features_valid_only.csv"
)

CLAY_S2_FILE = Path(
    "/home/melissa/flood_project/models_benchmark/"
    "Clay/flodis_outputs/"
    "clay_s2_event_features_valid_only/"
    "all_events_clay_s2_event_features_valid_only.csv"
)

DOFA_S2_FILE = Path(
    "/home/melissa/flood_project/models_benchmark/"
    "DOFA/flodis_outputs/"
    "dofa_s2_event_features_valid_only/"
    "all_events_dofa_s2_event_features_valid_only.csv"
)

OUTPUT_DIR = Path(
    "/home/melissa/flood_project/models_benchmark/"
    "impact_prediction/outputs/"
    "clay_s1_clay_s2_dofa_s2_common_displacement"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

FOLD_FILE = OUTPUT_DIR / "fold_results.csv"
PRED_FILE = OUTPUT_DIR / "cv_predictions.csv"
REPEAT_FILE = OUTPUT_DIR / "repeat_results.csv"
SUMMARY_FILE = OUTPUT_DIR / "comparison_summary.csv"
BEST_FILE = OUTPUT_DIR / "best_per_representation.csv"
EVENT_FILE = OUTPUT_DIR / "common_event_table.csv"


# ============================================================
# SETTINGS
# ============================================================

TARGET = "displacements"

N_SPLITS = 5
N_REPEATS = 6
RANDOM_STATE = 42

PCA_COMPONENTS = [4, 8, 16]


# ============================================================
# HELPERS
# ============================================================

def prepare_ids(df, name):

    df = df.copy()

    if "GFD_ID" not in df.columns:
        raise KeyError(f"GFD_ID missing from {name}")

    df["GFD_ID"] = pd.to_numeric(
        df["GFD_ID"],
        errors="coerce"
    )

    df = df.dropna(subset=["GFD_ID"])

    df["GFD_ID"] = df["GFD_ID"].astype(int)

    if df["GFD_ID"].duplicated().any():
        raise RuntimeError(
            f"Duplicate GFD_ID values found in {name}"
        )

    return df


def log_rmse(y_true, y_pred):

    return float(
        np.sqrt(
            mean_squared_error(
                y_true,
                y_pred
            )
        )
    )


def safe_r2(y_true, y_pred):

    if len(y_true) < 2:
        return np.nan

    try:
        return float(
            r2_score(
                y_true,
                y_pred
            )
        )
    except Exception:
        return np.nan


def get_models():

    return {
        "ridge": Ridge(
            alpha=1.0
        ),

        "random_forest": RandomForestRegressor(
            n_estimators=300,
            min_samples_leaf=2,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),

        "extra_trees": ExtraTreesRegressor(
            n_estimators=300,
            min_samples_leaf=2,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
    }


def preprocess_deep(
    X_train,
    X_test,
    n_components,
):

    imputer = SimpleImputer(
        strategy="median"
    )

    scaler = StandardScaler()

    pca = PCA(
        n_components=n_components,
        svd_solver="randomized",
        random_state=RANDOM_STATE,
    )

    # Fit ONLY on training fold
    X_train = imputer.fit_transform(
        X_train
    )

    X_test = imputer.transform(
        X_test
    )

    X_train = scaler.fit_transform(
        X_train
    )

    X_test = scaler.transform(
        X_test
    )

    X_train = pca.fit_transform(
        X_train
    )

    X_test = pca.transform(
        X_test
    )

    return (
        X_train.astype(np.float32),
        X_test.astype(np.float32),
    )


def detect_clay_s1_features(df):

    cols = []

    for c in df.columns:

        low = c.lower()

        if not pd.api.types.is_numeric_dtype(df[c]):
            continue

        if not low.startswith("clay"):
            continue

        if (
            "before" in low
            or "after" in low
            or "delta" in low
        ):
            cols.append(c)

    return cols


def detect_clay_s2_features(df):

    return [
        c
        for c in df.columns
        if (
            c.lower().startswith("clay_s2_")
            and pd.api.types.is_numeric_dtype(df[c])
            and (
                "before" in c.lower()
                or "after" in c.lower()
                or "delta" in c.lower()
            )
        )
    ]


def detect_dofa_s2_features(df):

    return [
        c
        for c in df.columns
        if (
            c.lower().startswith("dofa_s2_")
            and pd.api.types.is_numeric_dtype(df[c])
        )
    ]


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 110)
print("CLAY-S1 vs CLAY-S2 vs DOFA-S2")
print("FAIR COMMON-EVENT DISPLACEMENT COMPARISON")
print("=" * 110)

cohort = prepare_ids(
    pd.read_csv(COHORT_FILE),
    "cohort"
)

clay_s1 = prepare_ids(
    pd.read_csv(CLAY_S1_FILE),
    "Clay S1"
)

clay_s2 = prepare_ids(
    pd.read_csv(CLAY_S2_FILE),
    "Clay S2"
)

dofa_s2 = prepare_ids(
    pd.read_csv(DOFA_S2_FILE),
    "DOFA S2"
)


# ============================================================
# VALID EMBEDDINGS ONLY
# ============================================================

for name, table in [
    ("Clay S1", clay_s1),
    ("Clay S2", clay_s2),
    ("DOFA S2", dofa_s2),
]:

    if "n_valid_patches" in table.columns:

        before = len(table)

        table.drop(
            table[
                table["n_valid_patches"] <= 0
            ].index,
            inplace=True
        )

        print(
            f"{name}: {len(table)} valid events "
            f"(from {before})"
        )


# ============================================================
# FEATURE COLUMNS
# ============================================================

clay_s1_cols = detect_clay_s1_features(
    clay_s1
)

clay_s2_cols = detect_clay_s2_features(
    clay_s2
)

dofa_s2_cols = detect_dofa_s2_features(
    dofa_s2
)

print()
print("Feature dimensions:")
print(
    "Clay S1:",
    len(clay_s1_cols)
)
print(
    "Clay S2:",
    len(clay_s2_cols)
)
print(
    "DOFA S2:",
    len(dofa_s2_cols)
)


if len(clay_s1_cols) == 0:
    raise RuntimeError(
        "Could not detect Clay S1 deep feature columns."
    )

if len(clay_s2_cols) == 0:
    raise RuntimeError(
        "Could not detect Clay S2 deep feature columns."
    )

if len(dofa_s2_cols) == 0:
    raise RuntimeError(
        "Could not detect DOFA S2 deep feature columns."
    )


# ============================================================
# PREFIX COLUMNS BEFORE MERGING
# Prevent accidental collisions.
# ============================================================

clay_s1_small = clay_s1[
    ["GFD_ID"] + clay_s1_cols
].copy()

clay_s2_small = clay_s2[
    ["GFD_ID"] + clay_s2_cols
].copy()

dofa_s2_small = dofa_s2[
    ["GFD_ID"] + dofa_s2_cols
].copy()


clay_s1_rename = {
    c: f"S1__{c}"
    for c in clay_s1_cols
}

clay_s2_rename = {
    c: f"S2CLAY__{c}"
    for c in clay_s2_cols
}

dofa_s2_rename = {
    c: f"S2DOFA__{c}"
    for c in dofa_s2_cols
}

clay_s1_small = clay_s1_small.rename(
    columns=clay_s1_rename
)

clay_s2_small = clay_s2_small.rename(
    columns=clay_s2_rename
)

dofa_s2_small = dofa_s2_small.rename(
    columns=dofa_s2_rename
)


# ============================================================
# MERGE INTO EXACT OLD COHORT
# ============================================================

df = (
    cohort
    .merge(
        clay_s1_small,
        on="GFD_ID",
        how="inner"
    )
    .merge(
        clay_s2_small,
        on="GFD_ID",
        how="inner"
    )
    .merge(
        dofa_s2_small,
        on="GFD_ID",
        how="inner"
    )
)

df = (
    df
    .sort_values("GFD_ID")
    .reset_index(drop=True)
)


print()
print("=" * 110)
print("COMMON COHORT")
print("=" * 110)

print(
    "Previous cohort:",
    len(cohort)
)

print(
    "Final cohort:",
    len(df)
)


missing = sorted(
    set(cohort["GFD_ID"])
    - set(df["GFD_ID"])
)

if missing:
    print(
        "Events lost:",
        missing
    )
else:
    print(
        "No events lost."
    )


print()
print(
    "GFD IDs:",
    df["GFD_ID"].tolist()
)


if len(df) != len(cohort):

    print()
    print(
        "WARNING: cohort changed from",
        len(cohort),
        "to",
        len(df)
    )


if TARGET not in df.columns:

    raise KeyError(
        f"{TARGET} missing from cohort table"
    )


df.to_csv(
    EVENT_FILE,
    index=False
)


# ============================================================
# FINAL FEATURE NAMES AFTER MERGE
# ============================================================

s1_cols = [
    f"S1__{c}"
    for c in clay_s1_cols
]

s2_clay_cols = [
    f"S2CLAY__{c}"
    for c in clay_s2_cols
]

s2_dofa_cols = [
    f"S2DOFA__{c}"
    for c in dofa_s2_cols
]


# ============================================================
# MATRICES
# ============================================================

X_s1 = (
    df[s1_cols]
    .apply(
        pd.to_numeric,
        errors="coerce"
    )
    .to_numpy(
        dtype=np.float32
    )
)

X_s2_clay = (
    df[s2_clay_cols]
    .apply(
        pd.to_numeric,
        errors="coerce"
    )
    .to_numpy(
        dtype=np.float32
    )
)

X_s2_dofa = (
    df[s2_dofa_cols]
    .apply(
        pd.to_numeric,
        errors="coerce"
    )
    .to_numpy(
        dtype=np.float32
    )
)

y_raw = pd.to_numeric(
    df[TARGET],
    errors="coerce"
).to_numpy(
    dtype=float
)

if not np.isfinite(y_raw).all():

    raise RuntimeError(
        "Non-finite displacement labels "
        "found in common cohort."
    )

y = np.log1p(
    np.clip(
        y_raw,
        0,
        None
    )
)

gfd_ids = df[
    "GFD_ID"
].to_numpy()


print()
print("=" * 110)
print("MATRICES")
print("=" * 110)

print(
    "Events:",
    len(df)
)

print(
    "Clay S1:",
    X_s1.shape
)

print(
    "Clay S2:",
    X_s2_clay.shape
)

print(
    "DOFA S2:",
    X_s2_dofa.shape
)

print(
    "Non-finite Clay S1:",
    int(
        (~np.isfinite(X_s1)).sum()
    )
)

print(
    "Non-finite Clay S2:",
    int(
        (~np.isfinite(X_s2_clay)).sum()
    )
)

print(
    "Non-finite DOFA S2:",
    int(
        (~np.isfinite(X_s2_dofa)).sum()
    )
)


# ============================================================
# EXACT SAME SPLITS
# ============================================================

cv = RepeatedKFold(
    n_splits=N_SPLITS,
    n_repeats=N_REPEATS,
    random_state=RANDOM_STATE,
)

splits = list(
    cv.split(
        np.arange(len(df))
    )
)

print()
print(
    f"CV: {N_SPLITS}-fold x "
    f"{N_REPEATS} repeats"
)

print(
    "Total validation folds:",
    len(splits)
)


# ============================================================
# EXPERIMENT
# ============================================================

fold_rows = []
prediction_rows = []


for split_index, (
    train_idx,
    test_idx
) in enumerate(splits):

    repeat_id = (
        split_index // N_SPLITS
    ) + 1

    fold_id = (
        split_index % N_SPLITS
    ) + 1

    global_fold = (
        split_index + 1
    )

    print(
        f"Repeat {repeat_id}/{N_REPEATS}"
        f" | Fold {fold_id}/{N_SPLITS}"
    )

    y_train = y[train_idx]
    y_test = y[test_idx]

    # ========================================================
    # BASELINE
    # ========================================================

    median_value = np.median(
        y_train
    )

    baseline_pred = np.full(
        len(test_idx),
        median_value
    )

    baseline_rmse = log_rmse(
        y_test,
        baseline_pred
    )

    baseline_r2 = safe_r2(
        y_test,
        baseline_pred
    )

    fold_rows.append({
        "repeat": repeat_id,
        "fold_in_repeat": fold_id,
        "global_fold": global_fold,
        "candidate": "median_baseline",
        "model": "median_baseline",
        "pca_components": np.nan,
        "log_rmse": baseline_rmse,
        "log_r2": baseline_r2,
        "beats_baseline": np.nan,
    })

    for idx, pred in zip(
        test_idx,
        baseline_pred
    ):

        prediction_rows.append({
            "repeat": repeat_id,
            "candidate": "median_baseline",
            "model": "median_baseline",
            "pca_components": np.nan,
            "GFD_ID": int(gfd_ids[idx]),
            "y_true_log": float(y[idx]),
            "y_pred_log": float(pred),
        })


    # ========================================================
    # PCA DIMENSION
    # ========================================================

    for k in PCA_COMPONENTS:

        print(
            "   PCA:",
            k
        )

        # Each representation gets its OWN PCA.
        #
        # This is important:
        # DOFA and Clay are separate feature spaces.

        s1_train, s1_test = (
            preprocess_deep(
                X_s1[train_idx],
                X_s1[test_idx],
                k
            )
        )

        clay_train, clay_test = (
            preprocess_deep(
                X_s2_clay[train_idx],
                X_s2_clay[test_idx],
                k
            )
        )

        dofa_train, dofa_test = (
            preprocess_deep(
                X_s2_dofa[train_idx],
                X_s2_dofa[test_idx],
                k
            )
        )


        # ====================================================
        # REPRESENTATIONS
        # ====================================================

        candidates = {

            # Radar only
            "clay_s1": (
                s1_train,
                s1_test
            ),

            # Optical only
            "clay_s2": (
                clay_train,
                clay_test
            ),

            "dofa_s2": (
                dofa_train,
                dofa_test
            ),

            # Same-model multimodal
            "clay_s1_plus_clay_s2": (
                np.hstack([
                    s1_train,
                    clay_train
                ]),
                np.hstack([
                    s1_test,
                    clay_test
                ])
            ),

            # Cross-model multimodal
            "clay_s1_plus_dofa_s2": (
                np.hstack([
                    s1_train,
                    dofa_train
                ]),
                np.hstack([
                    s1_test,
                    dofa_test
                ])
            ),
        }


        # ====================================================
        # MODELS
        # ====================================================

        for (
            candidate_name,
            (
                X_train_candidate,
                X_test_candidate
            )
        ) in candidates.items():

            for (
                model_name,
                model
            ) in get_models().items():

                model.fit(
                    X_train_candidate,
                    y_train
                )

                pred = model.predict(
                    X_test_candidate
                )

                score_rmse = log_rmse(
                    y_test,
                    pred
                )

                score_r2 = safe_r2(
                    y_test,
                    pred
                )

                fold_rows.append({
                    "repeat": repeat_id,
                    "fold_in_repeat": fold_id,
                    "global_fold": global_fold,
                    "candidate": candidate_name,
                    "model": model_name,
                    "pca_components": k,
                    "log_rmse": score_rmse,
                    "log_r2": score_r2,
                    "beats_baseline":
                        score_rmse < baseline_rmse,
                })

                for idx, value in zip(
                    test_idx,
                    pred
                ):

                    prediction_rows.append({
                        "repeat": repeat_id,
                        "candidate": candidate_name,
                        "model": model_name,
                        "pca_components": k,
                        "GFD_ID": int(
                            gfd_ids[idx]
                        ),
                        "y_true_log": float(
                            y[idx]
                        ),
                        "y_pred_log": float(
                            value
                        ),
                    })


    # Save after every fold
    pd.DataFrame(
        fold_rows
    ).to_csv(
        FOLD_FILE,
        index=False
    )

    pd.DataFrame(
        prediction_rows
    ).to_csv(
        PRED_FILE,
        index=False
    )


# ============================================================
# REPEAT-LEVEL METRICS
# ============================================================

pred_df = pd.DataFrame(
    prediction_rows
)

fold_df = pd.DataFrame(
    fold_rows
)

repeat_rows = []


for keys, group in pred_df.groupby(
    [
        "repeat",
        "candidate",
        "model",
        "pca_components"
    ],
    dropna=False
):

    (
        repeat_id,
        candidate,
        model_name,
        pca_k
    ) = keys

    yt = group[
        "y_true_log"
    ].to_numpy()

    yp = group[
        "y_pred_log"
    ].to_numpy()

    repeat_rows.append({
        "repeat": repeat_id,
        "candidate": candidate,
        "model": model_name,
        "pca_components": pca_k,
        "log_rmse": log_rmse(
            yt,
            yp
        ),
        "log_r2": safe_r2(
            yt,
            yp
        ),
    })


repeat_df = pd.DataFrame(
    repeat_rows
)

repeat_df.to_csv(
    REPEAT_FILE,
    index=False
)


# ============================================================
# SUMMARY
# ============================================================

fold_summary = (
    fold_df
    .groupby(
        [
            "candidate",
            "model",
            "pca_components"
        ],
        dropna=False,
        as_index=False
    )
    .agg(
        mean_fold_log_rmse=(
            "log_rmse",
            "mean"
        ),

        std_fold_log_rmse=(
            "log_rmse",
            "std"
        ),

        mean_fold_log_r2=(
            "log_r2",
            "mean"
        ),

        wins_vs_baseline=(
            "beats_baseline",
            lambda x:
                int(
                    x.fillna(False).sum()
                )
        ),
    )
)


repeat_summary = (
    repeat_df
    .groupby(
        [
            "candidate",
            "model",
            "pca_components"
        ],
        dropna=False,
        as_index=False
    )
    .agg(
        mean_repeat_log_rmse=(
            "log_rmse",
            "mean"
        ),

        std_repeat_log_rmse=(
            "log_rmse",
            "std"
        ),

        mean_repeat_log_r2=(
            "log_r2",
            "mean"
        ),

        std_repeat_log_r2=(
            "log_r2",
            "std"
        ),
    )
)


summary = fold_summary.merge(
    repeat_summary,
    on=[
        "candidate",
        "model",
        "pca_components"
    ],
    how="left"
)

summary = (
    summary
    .sort_values(
        "mean_repeat_log_rmse"
    )
    .reset_index(drop=True)
)

summary.to_csv(
    SUMMARY_FILE,
    index=False
)


# ============================================================
# BEST CONFIGURATION PER REPRESENTATION
# ============================================================

order = [
    "median_baseline",
    "clay_s1",
    "clay_s2",
    "dofa_s2",
    "clay_s1_plus_clay_s2",
    "clay_s1_plus_dofa_s2",
]

best_rows = []


for candidate in order:

    sub = summary[
        summary["candidate"]
        == candidate
    ].copy()

    if len(sub) == 0:
        continue

    best_rows.append(
        sub.sort_values(
            "mean_repeat_log_rmse"
        ).iloc[0]
    )


best = pd.DataFrame(
    best_rows
)

best.to_csv(
    BEST_FILE,
    index=False
)


# ============================================================
# DISPLAY
# ============================================================

print()
print("=" * 120)
print("FINAL FAIR COMPARISON")
print("=" * 120)

print(
    best[
        [
            "candidate",
            "model",
            "pca_components",
            "mean_repeat_log_rmse",
            "std_repeat_log_rmse",
            "mean_repeat_log_r2",
            "std_repeat_log_r2",
            "wins_vs_baseline",
        ]
    ].to_string(
        index=False
    )
)

print()
print("=" * 120)
print("DONE")
print("=" * 120)

print(
    "Saved:",
    BEST_FILE
)
