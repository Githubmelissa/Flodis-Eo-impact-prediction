from pathlib import Path
import ast
import re
import warnings

import numpy as np
import pandas as pd

from sklearn.base import clone
from sklearn.decomposition import PCA
from sklearn.ensemble import (
    ExtraTreesRegressor,
    RandomForestRegressor,
)
from sklearn.linear_model import Ridge
from sklearn.metrics import (
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import RepeatedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")


# =====================================================
# PATHS
# =====================================================

CLAY_S2_FILE = Path(
    "/home/melissa/flood_project/"
    "models_benchmark/Clay/flodis_outputs/"
    "clay_s2_event_features_valid_only/"
    "all_events_clay_s2_event_features_valid_only.csv"
)

DISPLACEMENT_FILE = Path(
    "/home/melissa/flood_project/data/"
    "FLODIS_displacement_clean.csv"
)

MORTALITY_DAMAGE_FILE = Path(
    "/home/melissa/flood_project/data/"
    "FLODIS_mortality_damage_GFD_ready.csv"
)

OUTPUT_DIR = Path(
    "/home/melissa/flood_project/"
    "models_benchmark/impact_prediction/outputs/"
    "clay_s2_all_targets"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

MERGED_FILE = (
    OUTPUT_DIR /
    "clay_s2_all_targets_table.csv"
)

FOLD_RESULTS_FILE = (
    OUTPUT_DIR /
    "clay_s2_all_targets_fold_results.csv"
)

SUMMARY_FILE = (
    OUTPUT_DIR /
    "clay_s2_all_targets_summary.csv"
)


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
# HELPER: FIRST NON-NULL VALUE
# =====================================================

def first_non_null(series):
    values = series.dropna()

    if len(values) == 0:
        return np.nan

    return values.iloc[0]


# =====================================================
# LOAD CLAY S2 FEATURES
# =====================================================

def load_clay_s2_features():

    df = pd.read_csv(
        CLAY_S2_FILE
    )

    df["GFD_ID"] = pd.to_numeric(
        df["GFD_ID"],
        errors="coerce",
    )

    df = df.dropna(
        subset=["GFD_ID"]
    )

    df["GFD_ID"] = (
        df["GFD_ID"]
        .astype(int)
    )

    # Only events that actually have
    # Sentinel-2 Clay embeddings
    df = df[
        df["n_valid_patches"] > 0
    ].copy()

    deep_cols = [
        c for c in df.columns
        if c.startswith("clay_s2_")
    ]

    print(
        "Clay S2 events with embeddings:",
        len(df),
    )

    print(
        "Clay S2 deep columns:",
        len(deep_cols),
    )

    return df, deep_cols


# =====================================================
# LOAD DISPLACEMENT TARGET
# =====================================================

def load_displacement():

    df = pd.read_csv(
        DISPLACEMENT_FILE
    )

    # Find displacement column
    displacement_col = None

    for candidate in [
        "displacements",
        "displacement",
        "total_displacements",
        "figure",
    ]:

        if candidate in df.columns:
            displacement_col = candidate
            break

    if displacement_col is None:
        raise KeyError(
            "Could not find displacement "
            "column in displacement file."
        )

    if "GFD_matches_nr" not in df.columns:
        raise KeyError(
            "GFD_matches_nr not found."
        )

    rows = []

    for _, row in df.iterrows():

        raw = row["GFD_matches_nr"]

        if pd.isna(raw):
            continue

        # Parse list of GFD IDs
        try:

            gids = ast.literal_eval(
                str(raw)
            )

            if not isinstance(
                gids,
                (list, tuple),
            ):
                gids = [gids]

        except Exception:

            gids = re.findall(
                r"\d+",
                str(raw),
            )

        value = pd.to_numeric(
            row[displacement_col],
            errors="coerce",
        )

        for gid in gids:

            try:

                rows.append({
                    "GFD_ID": int(gid),
                    "displacements": value,
                })

            except Exception:
                continue

    out = pd.DataFrame(
        rows
    )

    if len(out) == 0:
        raise RuntimeError(
            "No displacement labels parsed."
        )

    out = (
        out
        .groupby(
            "GFD_ID",
            as_index=False,
        )
        .agg({
            "displacements":
                first_non_null
        })
    )

    return out


# =====================================================
# LOAD EM-DAT TARGETS
# =====================================================

def load_other_targets():

    df = pd.read_csv(
        MORTALITY_DAMAGE_FILE
    )

    if "GFD_ID" not in df.columns:
        raise KeyError(
            "GFD_ID missing from mortality/"
            "damage target file."
        )

    df["GFD_ID"] = pd.to_numeric(
        df["GFD_ID"],
        errors="coerce",
    )

    df = df.dropna(
        subset=["GFD_ID"]
    )

    df["GFD_ID"] = (
        df["GFD_ID"]
        .astype(int)
    )

    # Handle possible alternative
    # damage column name
    if (
        "total_damages_(000_USD)"
        not in df.columns
    ):

        if (
            "total_damages_000_USD"
            in df.columns
        ):

            df[
                "total_damages_(000_USD)"
            ] = df[
                "total_damages_000_USD"
            ]

    required = [
        "total_deaths",
        "total_affected_EMDAT",
        "total_damages_(000_USD)",
    ]

    available = [
        c for c in required
        if c in df.columns
    ]

    print(
        "Available EM-DAT targets:",
        available,
    )

    subset = df[
        ["GFD_ID"] + available
    ].copy()

    for col in available:

        subset[col] = pd.to_numeric(
            subset[col],
            errors="coerce",
        )

    agg_dict = {
        col: first_non_null
        for col in available
    }

    subset = (
        subset
        .groupby(
            "GFD_ID",
            as_index=False,
        )
        .agg(
            agg_dict
        )
    )

    return subset


# =====================================================
# BUILD COMPLETE TABLE
# =====================================================

def build_dataset():

    features, deep_cols = (
        load_clay_s2_features()
    )

    displacement = (
        load_displacement()
    )

    other_targets = (
        load_other_targets()
    )

    df = features.merge(
        displacement,
        on="GFD_ID",
        how="left",
    )

    df = df.merge(
        other_targets,
        on="GFD_ID",
        how="left",
    )

    df.to_csv(
        MERGED_FILE,
        index=False,
    )

    print()
    print("=" * 100)
    print("LABEL AVAILABILITY")
    print("=" * 100)

    for target in TARGETS:

        if target not in df.columns:

            print(
                target,
                ": column missing"
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
            f"{target}: {n}"
        )

    print()
    print(
        "Saved merged table:",
        MERGED_FILE,
    )

    return df, deep_cols


# =====================================================
# CV SETTINGS
# =====================================================

def get_experiment_settings(
    n_samples,
):
    """
    Adapt CV and PCA size to the
    number of available labelled events.

    Approximately 30 validation folds
    are produced for every target.
    """

    if n_samples >= 25:

        n_splits = 5
        n_repeats = 6

        pca_components = [
            4,
            8,
            16,
        ]

    elif n_samples >= 10:

        n_splits = 3
        n_repeats = 10

        pca_components = [
            2,
            4,
            8,
        ]

    else:

        n_splits = 2
        n_repeats = 15

        pca_components = [
            1,
            2,
        ]

    return (
        n_splits,
        n_repeats,
        pca_components,
    )


# =====================================================
# MODELS
# =====================================================

def make_models(
    n_components,
):

    # StandardScaler and PCA are INSIDE
    # each pipeline, so both are fitted
    # only on the training fold.
    return {

        "ridge": Pipeline([
            (
                "scale",
                StandardScaler(),
            ),
            (
                "pca",
                PCA(
                    n_components=
                        n_components,
                    svd_solver=
                        "randomized",
                    random_state=42,
                ),
            ),
            (
                "model",
                Ridge(
                    alpha=1.0
                ),
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
                    n_components=
                        n_components,
                    svd_solver=
                        "randomized",
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
                    n_components=
                        n_components,
                    svd_solver=
                        "randomized",
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
# RUN ONE TARGET
# =====================================================

def run_target(
    full_df,
    deep_cols,
    target,
):

    print()
    print("=" * 100)
    print(
        "TARGET:",
        target,
    )
    print("=" * 100)

    if target not in full_df.columns:

        print(
            "Target column missing. "
            "Skipping."
        )

        return []

    target_df = (
        full_df.copy()
    )

    target_df[target] = (
        pd.to_numeric(
            target_df[target],
            errors="coerce",
        )
    )

    # Keep labelled events only
    target_df = target_df[
        target_df[target].notna()
    ].copy()

    # Impact values must be >= 0
    target_df = target_df[
        target_df[target] >= 0
    ].copy()

    if len(target_df) < 4:

        print(
            "Too few labelled events:",
            len(target_df),
        )

        return []

    # -----------------------------------------
    # Deep feature matrix
    # -----------------------------------------

    X = (
        target_df[
            deep_cols
        ]
        .apply(
            pd.to_numeric,
            errors="coerce",
        )
        .to_numpy(
            dtype=np.float32
        )
    )

    y_raw = (
        target_df[target]
        .to_numpy(
            dtype=np.float64
        )
    )

    # Remove any event with a
    # non-finite deep feature
    good_rows = (
        np.isfinite(X)
        .all(axis=1)
        &
        np.isfinite(y_raw)
    )

    removed = int(
        (~good_rows).sum()
    )

    if removed > 0:

        print(
            "Removing rows with "
            "non-finite values:",
            removed,
        )

    X = X[
        good_rows
    ]

    y_raw = y_raw[
        good_rows
    ]

    target_df = (
        target_df
        .iloc[
            np.where(
                good_rows
            )[0]
        ]
        .copy()
    )

    n_samples = len(
        target_df
    )

    # Log-transform impact
    y = np.log1p(
        y_raw
    )

    (
        n_splits,
        n_repeats,
        pca_components,
    ) = get_experiment_settings(
        n_samples
    )

    print(
        "Events:",
        n_samples,
    )

    print(
        "X shape:",
        X.shape,
    )

    print(
        "CV:",
        f"{n_splits}-fold x "
        f"{n_repeats} repeats",
    )

    print(
        "Total validation folds:",
        n_splits * n_repeats,
    )

    print(
        "PCA components:",
        pca_components,
    )

    # -----------------------------------------
    # CV
    # -----------------------------------------

    cv = RepeatedKFold(
        n_splits=n_splits,
        n_repeats=n_repeats,
        random_state=42,
    )

    # Generate splits ONCE.
    # Every configuration uses exactly
    # the same train/test folds.
    splits = list(
        cv.split(X)
    )

    results = []

    # =================================================
    # MEDIAN BASELINE
    # =================================================

    print(
        "\nRunning median baseline..."
    )

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

        median_value = np.median(
            y_train
        )

        prediction = np.full(
            shape=len(test_idx),
            fill_value=median_value,
            dtype=np.float64,
        )

        rmse = np.sqrt(
            mean_squared_error(
                y_test,
                prediction,
            )
        )

        r2 = r2_score(
            y_test,
            prediction,
        )

        results.append({
            "target": target,
            "n_events": n_samples,
            "n_splits": n_splits,
            "n_repeats": n_repeats,
            "model":
                "median_baseline",
            "pca_components": 0,
            "fold": fold_id,
            "log_rmse": rmse,
            "log_r2": r2,
        })

    # =================================================
    # CLAY S2 DEEP FEATURES
    # =================================================

    for n_components in (
        pca_components
    ):

        # Extra safety:
        # PCA components must fit in
        # the smallest training set.
        smallest_train_size = (
            n_samples
            -
            int(
                np.ceil(
                    n_samples
                    / n_splits
                )
            )
        )

        if (
            n_components
            >= smallest_train_size
        ):

            print(
                f"Skipping PCA-{n_components}: "
                "too many components for "
                "training-fold size."
            )

            continue

        models = make_models(
            n_components
        )

        for (
            model_name,
            estimator,
        ) in models.items():

            print(
                f"Running "
                f"{model_name} "
                f"PCA-{n_components}"
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

                model = clone(
                    estimator
                )

                model.fit(
                    X_train,
                    y_train,
                )

                prediction = (
                    model.predict(
                        X_test
                    )
                )

                rmse = np.sqrt(
                    mean_squared_error(
                        y_test,
                        prediction,
                    )
                )

                r2 = r2_score(
                    y_test,
                    prediction,
                )

                results.append({
                    "target":
                        target,

                    "n_events":
                        n_samples,

                    "n_splits":
                        n_splits,

                    "n_repeats":
                        n_repeats,

                    "model":
                        model_name,

                    "pca_components":
                        n_components,

                    "fold":
                        fold_id,

                    "log_rmse":
                        rmse,

                    "log_r2":
                        r2,
                })

    return results


# =====================================================
# MAIN
# =====================================================

def main():

    print("=" * 100)
    print(
        "CLAY SENTINEL-2 "
        "ALL-TARGET IMPACT EXPERIMENT"
    )
    print("=" * 100)

    print(
        "Clay features:",
        CLAY_S2_FILE,
    )

    print(
        "Displacement labels:",
        DISPLACEMENT_FILE,
    )

    print(
        "Other impact labels:",
        MORTALITY_DAMAGE_FILE,
    )

    # -----------------------------------------
    # Build complete dataset
    # -----------------------------------------

    full_df, deep_cols = (
        build_dataset()
    )

    print()
    print(
        "Deep predictors only:",
        len(deep_cols),
    )

    print(
        "Quality/coverage variables "
        "are NOT used as predictors."
    )

    all_results = []

    # -----------------------------------------
    # Run all targets
    # -----------------------------------------

    for target in TARGETS:

        target_results = (
            run_target(
                full_df,
                deep_cols,
                target,
            )
        )

        all_results.extend(
            target_results
        )

    if len(all_results) == 0:

        raise RuntimeError(
            "No experiment results "
            "were produced."
        )

    # =================================================
    # SAVE FOLD RESULTS
    # =================================================

    fold_df = pd.DataFrame(
        all_results
    )

    fold_df.to_csv(
        FOLD_RESULTS_FILE,
        index=False,
    )

    # =================================================
    # SUMMARY
    # =================================================

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

    # Sort independently for each target
    summary = (
        summary
        .sort_values(
            [
                "target",
                "mean_log_rmse",
            ]
        )
    )

    summary.to_csv(
        SUMMARY_FILE,
        index=False,
    )

    # =================================================
    # DISPLAY BEST RESULTS
    # =================================================

    print()
    print("=" * 100)
    print("FINAL RESULTS")
    print("=" * 100)

    for target in TARGETS:

        target_summary = (
            summary[
                summary["target"]
                == target
            ]
            .sort_values(
                "mean_log_rmse"
            )
        )

        if len(
            target_summary
        ) == 0:
            continue

        print()
        print("-" * 100)
        print(
            "TARGET:",
            target,
        )
        print("-" * 100)

        print(
            target_summary[
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
    print("=" * 100)
    print("SAVED")
    print("=" * 100)

    print(
        "Merged dataset:"
    )
    print(
        MERGED_FILE
    )

    print()
    print(
        "Fold results:"
    )
    print(
        FOLD_RESULTS_FILE
    )

    print()
    print(
        "Summary:"
    )
    print(
        SUMMARY_FILE
    )


if __name__ == "__main__":
    main()
