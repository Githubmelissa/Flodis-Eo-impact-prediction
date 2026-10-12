from pathlib import Path
import warnings

import numpy as np
import pandas as pd

from sklearn.decomposition import PCA
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import RepeatedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")


# =====================================================
# PATHS
# =====================================================

DOFA_FILE = Path(
    "/home/melissa/flood_project/models_benchmark/DOFA/"
    "flodis_outputs/dofa_s2_event_features_valid_only/"
    "all_events_dofa_s2_event_features_valid_only.csv"
)

# We use ONLY the target columns from this existing table
TARGET_TABLE = Path(
    "/home/melissa/flood_project/models_benchmark/"
    "impact_prediction/outputs/clay_s2_all_targets/"
    "clay_s2_all_targets_table.csv"
)

OUTPUT_DIR = Path(
    "/home/melissa/flood_project/models_benchmark/"
    "impact_prediction/outputs/dofa_s2_all_targets"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

MERGED_FILE = OUTPUT_DIR / "dofa_s2_all_targets_table.csv"
FOLD_FILE = OUTPUT_DIR / "dofa_s2_all_targets_fold_results.csv"
SUMMARY_FILE = OUTPUT_DIR / "dofa_s2_all_targets_summary.csv"


# =====================================================
# TARGETS
# =====================================================

TARGETS = [
    "displacements",
    "total_deaths",
    "total_affected_EMDAT",
    "total_damages_(000_USD)",
]


# =====================================================
# CV SETTINGS
# =====================================================

def get_settings(n):

    if n >= 25:

        return (
            5,
            6,
            [4, 8, 16],
        )

    elif n >= 10:

        return (
            3,
            10,
            [2, 4, 8],
        )

    else:

        return (
            2,
            15,
            [1, 2],
        )


# =====================================================
# MODELS
# =====================================================

def make_models(k):

    return {

        "ridge": Pipeline([
            (
                "scale",
                StandardScaler(),
            ),
            (
                "pca",
                PCA(
                    n_components=k,
                    svd_solver="randomized",
                    random_state=42,
                ),
            ),
            (
                "model",
                Ridge(alpha=1.0),
            ),
        ]),

        "random_forest": Pipeline([
            (
                "scale",
                StandardScaler(),
            ),
            (
                "pca",
                PCA(
                    n_components=k,
                    svd_solver="randomized",
                    random_state=42,
                ),
            ),
            (
                "model",
                RandomForestRegressor(
                    n_estimators=300,
                    min_samples_leaf=2,
                    random_state=42,
                    n_jobs=-1,
                ),
            ),
        ]),

        "extra_trees": Pipeline([
            (
                "scale",
                StandardScaler(),
            ),
            (
                "pca",
                PCA(
                    n_components=k,
                    svd_solver="randomized",
                    random_state=42,
                ),
            ),
            (
                "model",
                ExtraTreesRegressor(
                    n_estimators=300,
                    min_samples_leaf=2,
                    random_state=42,
                    n_jobs=-1,
                ),
            ),
        ]),
    }


# =====================================================
# LOAD DATA
# =====================================================

dofa = pd.read_csv(
    DOFA_FILE
)

targets = pd.read_csv(
    TARGET_TABLE
)

dofa["GFD_ID"] = pd.to_numeric(
    dofa["GFD_ID"],
    errors="coerce",
)

targets["GFD_ID"] = pd.to_numeric(
    targets["GFD_ID"],
    errors="coerce",
)

dofa = dofa.dropna(
    subset=["GFD_ID"]
)

targets = targets.dropna(
    subset=["GFD_ID"]
)

dofa["GFD_ID"] = (
    dofa["GFD_ID"]
    .astype(int)
)

targets["GFD_ID"] = (
    targets["GFD_ID"]
    .astype(int)
)


# Keep only actual DOFA embeddings
dofa = dofa[
    dofa["n_valid_patches"] > 0
].copy()


# =====================================================
# DEEP FEATURES
# =====================================================

deep_cols = [
    c
    for c in dofa.columns
    if c.startswith("dofa_s2_")
]

print("=" * 100)
print("DOFA SENTINEL-2 ALL TARGETS")
print("=" * 100)

print(
    "DOFA events with embeddings:",
    len(dofa),
)

print(
    "DOFA deep features:",
    len(deep_cols),
)


# =====================================================
# TARGET TABLE
# =====================================================

available_target_cols = [
    c
    for c in TARGETS
    if c in targets.columns
]

target_small = (
    targets[
        ["GFD_ID"]
        + available_target_cols
    ]
    .drop_duplicates(
        "GFD_ID"
    )
)

df = dofa.merge(
    target_small,
    on="GFD_ID",
    how="left",
)

df = (
    df
    .sort_values("GFD_ID")
    .reset_index(drop=True)
)

df.to_csv(
    MERGED_FILE,
    index=False,
)


# =====================================================
# CHECK
# =====================================================

print()
print("Label availability:")

for target in TARGETS:

    if target not in df.columns:

        print(
            target,
            ": MISSING"
        )

        continue

    n = int(
        pd.to_numeric(
            df[target],
            errors="coerce",
        )
        .notna()
        .sum()
    )

    print(
        target,
        ":",
        n,
    )


# =====================================================
# RUN EXPERIMENTS
# =====================================================

all_rows = []

for target in TARGETS:

    if target not in df.columns:
        continue

    print()
    print("=" * 100)
    print(
        "TARGET:",
        target,
    )
    print("=" * 100)

    sub = df.copy()

    sub[target] = pd.to_numeric(
        sub[target],
        errors="coerce",
    )

    sub = sub[
        sub[target].notna()
    ].copy()

    sub = sub[
        sub[target] >= 0
    ].copy()

    if len(sub) < 4:

        print(
            "Too few events:",
            len(sub),
        )

        continue

    X = (
        sub[deep_cols]
        .apply(
            pd.to_numeric,
            errors="coerce",
        )
        .to_numpy(
            dtype=np.float32
        )
    )

    y_raw = (
        sub[target]
        .to_numpy(
            dtype=np.float64
        )
    )

    # Remove any non-finite rows
    valid = (
        np.isfinite(X).all(axis=1)
        &
        np.isfinite(y_raw)
    )

    X = X[valid]
    y_raw = y_raw[valid]

    y = np.log1p(
        y_raw
    )

    n = len(y)

    (
        n_splits,
        n_repeats,
        pca_values,
    ) = get_settings(n)

    print(
        "Events:",
        n,
    )

    print(
        "X:",
        X.shape,
    )

    print(
        "CV:",
        f"{n_splits}-fold x "
        f"{n_repeats}",
    )

    print(
        "PCA:",
        pca_values,
    )

    cv = RepeatedKFold(
        n_splits=n_splits,
        n_repeats=n_repeats,
        random_state=42,
    )

    # Generate once so every model gets same folds
    splits = list(
        cv.split(X)
    )

    # =================================================
    # MEDIAN BASELINE
    # =================================================

    for fold_id, (
        train_idx,
        test_idx,
    ) in enumerate(
        splits,
        start=1,
    ):

        y_train = y[
            train_idx
        ]

        y_test = y[
            test_idx
        ]

        pred = np.full(
            len(test_idx),
            np.median(y_train),
        )

        fold_rmse = np.sqrt(
            mean_squared_error(
                y_test,
                pred,
            )
        )

        fold_r2 = r2_score(
            y_test,
            pred,
        )

        all_rows.append({

            "target":
                target,

            "n_events":
                n,

            "n_splits":
                n_splits,

            "n_repeats":
                n_repeats,

            "model":
                "median_baseline",

            "pca_components":
                0,

            "fold":
                fold_id,

            "log_rmse":
                fold_rmse,

            "log_r2":
                fold_r2,
        })

    # =================================================
    # DOFA
    # =================================================

    min_train_size = min(
        len(train_idx)
        for train_idx, _
        in splits
    )

    for k in pca_values:

        if k >= min_train_size:

            print(
                "Skipping PCA",
                k,
                "- too large"
            )

            continue

        models = make_models(
            k
        )

        for (
            model_name,
            model,
        ) in models.items():

            print(
                "Running:",
                model_name,
                "PCA",
                k,
            )

            for fold_id, (
                train_idx,
                test_idx,
            ) in enumerate(
                splits,
                start=1,
            ):

                X_train = X[
                    train_idx
                ]

                X_test = X[
                    test_idx
                ]

                y_train = y[
                    train_idx
                ]

                y_test = y[
                    test_idx
                ]

                model.fit(
                    X_train,
                    y_train,
                )

                pred = model.predict(
                    X_test
                )

                fold_rmse = np.sqrt(
                    mean_squared_error(
                        y_test,
                        pred,
                    )
                )

                fold_r2 = r2_score(
                    y_test,
                    pred,
                )

                all_rows.append({

                    "target":
                        target,

                    "n_events":
                        n,

                    "n_splits":
                        n_splits,

                    "n_repeats":
                        n_repeats,

                    "model":
                        model_name,

                    "pca_components":
                        k,

                    "fold":
                        fold_id,

                    "log_rmse":
                        fold_rmse,

                    "log_r2":
                        fold_r2,
                })

    # Save after each target
    pd.DataFrame(
        all_rows
    ).to_csv(
        FOLD_FILE,
        index=False,
    )


# =====================================================
# SUMMARY
# =====================================================

fold_df = pd.DataFrame(
    all_rows
)

summary = (
    fold_df
    .groupby(
        [
            "target",
            "n_events",
            "n_splits",
            "n_repeats",
            "model",
            "pca_components",
        ],
        as_index=False,
    )
    .agg(

        mean_log_rmse=(
            "log_rmse",
            "mean",
        ),

        std_log_rmse=(
            "log_rmse",
            "std",
        ),

        mean_log_r2=(
            "log_r2",
            "mean",
        ),

        std_log_r2=(
            "log_r2",
            "std",
        ),
    )
)

summary = summary.sort_values(
    [
        "target",
        "mean_log_rmse",
    ]
)

summary.to_csv(
    SUMMARY_FILE,
    index=False,
)


# =====================================================
# DISPLAY
# =====================================================

print()
print("=" * 100)
print("FINAL RESULTS")
print("=" * 100)

for target in TARGETS:

    sub = summary[
        summary["target"]
        == target
    ]

    if len(sub) == 0:
        continue

    print()
    print("-" * 100)
    print(target)
    print("-" * 100)

    print(
        sub[
            [
                "n_events",
                "model",
                "pca_components",
                "mean_log_rmse",
                "std_log_rmse",
                "mean_log_r2",
                "std_log_r2",
            ]
        ]
        .to_string(
            index=False
        )
    )

print()
print("Saved:")
print(SUMMARY_FILE)
