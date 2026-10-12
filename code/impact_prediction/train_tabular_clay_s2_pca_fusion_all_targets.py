from pathlib import Path
import warnings

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import RepeatedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")


# =====================================================
# PATHS
# =====================================================

TABULAR_TABLE_PATH = Path(
    "/home/melissa/flood_project/"
    "models_benchmark/impact_prediction/outputs/"
    "tabular_phase1_event_table_FIXED.csv"
)

CLAY_S2_TABLE_PATH = Path(
    "/home/melissa/flood_project/"
    "models_benchmark/Clay/flodis_outputs/"
    "clay_s2_event_features_valid_only/"
    "all_events_clay_s2_event_features_valid_only.csv"
)

OUTPUT_DIR = Path(
    "/home/melissa/flood_project/"
    "models_benchmark/impact_prediction/outputs/"
    "tabular_clay_s2_pca_fusion_all_targets"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

OUTPUT_EVENT_TABLE = (
    OUTPUT_DIR /
    "tabular_clay_s2_fusion_event_table.csv"
)

OUTPUT_FOLD_RESULTS = (
    OUTPUT_DIR /
    "tabular_clay_s2_fusion_fold_results.csv"
)

OUTPUT_SUMMARY = (
    OUTPUT_DIR /
    "tabular_clay_s2_fusion_summary.csv"
)

OUTPUT_BEST = (
    OUTPUT_DIR /
    "tabular_clay_s2_fusion_best_per_target.csv"
)


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

# No Clay quality variables are included as predictors.
# They are only used to determine whether an event
# actually has a valid S2 deep representation.


# =====================================================
# HELPERS
# =====================================================

def safe_log1p(y):
    y = np.asarray(
        y,
        dtype=float,
    )

    y = np.clip(
        y,
        0,
        None,
    )

    return np.log1p(y)


def rmse(
    y_true,
    y_pred,
):
    return float(
        np.sqrt(
            mean_squared_error(
                y_true,
                y_pred,
            )
        )
    )


def safe_r2(
    y_true,
    y_pred,
):
    try:

        if len(y_true) < 2:
            return np.nan

        return float(
            r2_score(
                y_true,
                y_pred,
            )
        )

    except Exception:

        return np.nan


# =====================================================
# MODELS
# =====================================================

def get_models():

    return {

        "ridge": Ridge(
            alpha=1.0,
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


# =====================================================
# FEATURE GROUPS
# =====================================================

def get_feature_groups(df):

    all_numeric = [
        c
        for c in df.columns
        if pd.api.types.is_numeric_dtype(
            df[c]
        )
    ]

    # -----------------------------------------
    # WorldPop
    # Preserve logic from previous experiments
    # -----------------------------------------

    worldpop_cols = [
        c
        for c in all_numeric
        if (
            "population" in c.lower()
            or c == "flood_area_km2"
        )
    ]

    # -----------------------------------------
    # GPM rainfall
    # -----------------------------------------

    gpm_cols = [
        c
        for c in all_numeric
        if (
            "gpm" in c.lower()
            or "rainfall" in c.lower()
        )
    ]

    # -----------------------------------------
    # SRTM
    # -----------------------------------------

    srtm_cols = [
        c
        for c in all_numeric
        if (
            "elevation" in c.lower()
            or "slope" in c.lower()
        )
    ]

    # -----------------------------------------
    # Clay Sentinel-2 DEEP features
    # -----------------------------------------

    clay_s2_deep_cols = [
        c
        for c in all_numeric
        if (
            c.startswith(
                "clay_s2_before_"
            )
            or c.startswith(
                "clay_s2_after_"
            )
            or c.startswith(
                "clay_s2_delta_"
            )
        )
    ]

    # -----------------------------------------
    # Prevent leakage
    # -----------------------------------------

    forbidden = (
        set(TARGETS)
        | META_COLS
    )

    def clean(cols):

        return sorted(
            set(
                c
                for c in cols
                if c not in forbidden
            )
        )

    groups = {

        "worldpop":
            clean(
                worldpop_cols
            ),

        "gpm":
            clean(
                gpm_cols
            ),

        "srtm":
            clean(
                srtm_cols
            ),

        "clay_s2_deep":
            clean(
                clay_s2_deep_cols
            ),
    }

    return groups


# =====================================================
# CV SETTINGS
# =====================================================

def get_cv_settings(
    n_samples,
):
    """
    Adapt CV and PCA dimensionality to target size.

    Each target gets approximately 30 validation folds.

    displacement ~50:
        5 folds x 6 repeats

    affected ~14:
        3 folds x 10 repeats

    deaths / damages:
        2 folds x 15 repeats
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
# PREPROCESSING
# =====================================================

def make_preprocessor(
    base_cols,
    deep_cols=None,
    pca_components=None,
):

    transformers = []

    # -----------------------------------------
    # Contextual branch
    # -----------------------------------------

    if len(base_cols) > 0:

        base_pipe = Pipeline(
            steps=[
                (
                    "imputer",
                    SimpleImputer(
                        strategy="median"
                    ),
                ),
                (
                    "scaler",
                    StandardScaler(),
                ),
            ]
        )

        transformers.append(
            (
                "contextual",
                base_pipe,
                base_cols,
            )
        )

    # -----------------------------------------
    # Clay S2 deep branch
    # -----------------------------------------

    if (
        deep_cols is not None
        and len(deep_cols) > 0
    ):

        deep_steps = [
            (
                "imputer",
                SimpleImputer(
                    strategy="median"
                ),
            ),
            (
                "scaler",
                StandardScaler(),
            ),
        ]

        if pca_components is not None:

            deep_steps.append(
                (
                    "pca",
                    PCA(
                        n_components=
                            pca_components,
                        svd_solver=
                            "randomized",
                        random_state=
                            RANDOM_STATE,
                    ),
                )
            )

        deep_pipe = Pipeline(
            steps=deep_steps
        )

        transformers.append(
            (
                "clay_s2_deep",
                deep_pipe,
                deep_cols,
            )
        )

    if len(transformers) == 0:

        raise ValueError(
            "No features were provided."
        )

    return ColumnTransformer(
        transformers=transformers,
        remainder="drop",
    )


def make_pipeline(
    model,
    base_cols,
    deep_cols=None,
    pca_components=None,
):

    preprocessor = (
        make_preprocessor(
            base_cols=
                base_cols,

            deep_cols=
                deep_cols,

            pca_components=
                pca_components,
        )
    )

    return Pipeline(
        steps=[
            (
                "preprocess",
                preprocessor,
            ),
            (
                "model",
                model,
            ),
        ]
    )


# =====================================================
# BUILD CANDIDATES
# =====================================================

def build_candidates(
    groups,
    pca_components,
):

    worldpop = (
        groups["worldpop"]
    )

    gpm = (
        groups["gpm"]
    )

    srtm = (
        groups["srtm"]
    )

    clay = (
        groups["clay_s2_deep"]
    )

    worldpop_gpm = sorted(
        set(
            worldpop
            + gpm
        )
    )

    worldpop_gpm_srtm = sorted(
        set(
            worldpop
            + gpm
            + srtm
        )
    )

    candidates = []

    # =================================================
    # CONTEXTUAL BASELINES
    # =================================================

    candidates.append({
        "name":
            "worldpop_gpm",

        "base_cols":
            worldpop_gpm,

        "deep_cols":
            None,

        "pca":
            None,
    })

    candidates.append({
        "name":
            "worldpop_gpm_srtm",

        "base_cols":
            worldpop_gpm_srtm,

        "deep_cols":
            None,

        "pca":
            None,
    })

    # =================================================
    # DEEP / FUSION
    # =================================================

    for k in pca_components:

        # -----------------------------------------
        # Clay S2 only
        # -----------------------------------------

        candidates.append({
            "name":
                f"clay_s2_pca_{k}",

            "base_cols":
                [],

            "deep_cols":
                clay,

            "pca":
                k,
        })

        # -----------------------------------------
        # WP + GPM + Clay S2
        # -----------------------------------------

        candidates.append({
            "name":
                f"worldpop_gpm_plus_"
                f"clay_s2_pca_{k}",

            "base_cols":
                worldpop_gpm,

            "deep_cols":
                clay,

            "pca":
                k,
        })

        # -----------------------------------------
        # WP + GPM + SRTM + Clay S2
        # -----------------------------------------

        candidates.append({
            "name":
                f"worldpop_gpm_srtm_plus_"
                f"clay_s2_pca_{k}",

            "base_cols":
                worldpop_gpm_srtm,

            "deep_cols":
                clay,

            "pca":
                k,
        })

    return candidates


# =====================================================
# RUN ONE TARGET
# =====================================================

def run_target(
    df,
    groups,
    target,
):

    print()
    print("=" * 100)
    print(
        "TARGET:",
        target,
    )
    print("=" * 100)

    # -----------------------------------------
    # Keep only labelled events
    # -----------------------------------------

    labeled = (
        df[
            df[target].notna()
        ]
        .copy()
    )

    labeled[target] = (
        pd.to_numeric(
            labeled[target],
            errors="coerce",
        )
    )

    labeled = (
        labeled[
            labeled[target]
            .notna()
        ]
        .copy()
    )

    labeled = (
        labeled[
            labeled[target] >= 0
        ]
        .copy()
    )

    n_samples = len(
        labeled
    )

    if n_samples < 4:

        print(
            "Too few samples:",
            n_samples,
        )

        return []

    # -----------------------------------------
    # Log target
    # -----------------------------------------

    y_original = (
        labeled[target]
        .astype(float)
        .to_numpy()
    )

    y_log = safe_log1p(
        y_original
    )

    # -----------------------------------------
    # CV settings
    # -----------------------------------------

    (
        n_splits,
        n_repeats,
        pca_components,
    ) = get_cv_settings(
        n_samples
    )

    print(
        "Events:",
        n_samples,
    )

    print(
        "CV:",
        f"{n_splits}-fold x "
        f"{n_repeats} repeats",
    )

    print(
        "Validation folds:",
        n_splits
        * n_repeats,
    )

    print(
        "PCA candidates:",
        pca_components,
    )

    # -----------------------------------------
    # Generate SAME splits for every candidate
    # -----------------------------------------

    cv = RepeatedKFold(
        n_splits=
            n_splits,

        n_repeats=
            n_repeats,

        random_state=
            RANDOM_STATE,
    )

    dummy_X = np.zeros(
        (
            n_samples,
            1,
        )
    )

    splits = list(
        cv.split(
            dummy_X
        )
    )

    # -----------------------------------------
    # Baseline first
    # -----------------------------------------

    rows = []

    baseline_rmse = {}

    print(
        "Running median baseline..."
    )

    for fold_id, (
        train_idx,
        test_idx,
    ) in enumerate(
        splits,
        start=1,
    ):

        y_train = (
            y_log[
                train_idx
            ]
        )

        y_test = (
            y_log[
                test_idx
            ]
        )

        median_value = (
            np.median(
                y_train
            )
        )

        pred = np.full(
            len(test_idx),
            median_value,
            dtype=float,
        )

        fold_rmse = rmse(
            y_test,
            pred,
        )

        fold_r2 = safe_r2(
            y_test,
            pred,
        )

        baseline_rmse[
            fold_id
        ] = fold_rmse

        rows.append({

            "target":
                target,

            "n_samples":
                n_samples,

            "n_splits":
                n_splits,

            "n_repeats":
                n_repeats,

            "candidate":
                "baseline_median",

            "model":
                "baseline_median",

            "pca_components":
                np.nan,

            "n_features_base":
                0,

            "n_features_deep":
                0,

            "fold":
                fold_id,

            "log_rmse":
                fold_rmse,

            "log_r2":
                fold_r2,

            "beats_baseline":
                np.nan,
        })

    # -----------------------------------------
    # Candidate configurations
    # -----------------------------------------

    candidates = (
        build_candidates(
            groups,
            pca_components,
        )
    )

    # Smallest training fold
    smallest_train_size = min(
        len(train_idx)
        for train_idx, _
        in splits
    )

    print(
        "Smallest training fold:",
        smallest_train_size,
    )

    # =================================================
    # RUN MODELS
    # =================================================

    for candidate in candidates:

        candidate_name = (
            candidate["name"]
        )

        base_cols = (
            candidate["base_cols"]
        )

        deep_cols = (
            candidate["deep_cols"]
        )

        pca_k = (
            candidate["pca"]
        )

        # PCA safety
        if pca_k is not None:

            if pca_k >= smallest_train_size:

                print(
                    "Skipping",
                    candidate_name,
                    "- PCA too large "
                    "for training fold",
                )

                continue

        # Build DataFrame containing
        # all columns needed by pipeline
        feature_cols = list(
            base_cols
        )

        if deep_cols is not None:

            feature_cols += list(
                deep_cols
            )

        feature_cols = list(
            dict.fromkeys(
                feature_cols
            )
        )

        if len(feature_cols) == 0:
            continue

        X = labeled[
            feature_cols
        ].copy()

        print()
        print(
            "Candidate:",
            candidate_name,
        )

        print(
            "  contextual features:",
            len(base_cols),
        )

        print(
            "  Clay S2 deep features:",
            (
                0
                if deep_cols is None
                else len(deep_cols)
            ),
        )

        for (
            model_name,
            model,
        ) in get_models().items():

            print(
                "   Model:",
                model_name,
            )

            for fold_id, (
                train_idx,
                test_idx,
            ) in enumerate(
                splits,
                start=1,
            ):

                X_train = X.iloc[
                    train_idx
                ]

                X_test = X.iloc[
                    test_idx
                ]

                y_train = y_log[
                    train_idx
                ]

                y_test = y_log[
                    test_idx
                ]

                pipe = make_pipeline(

                    model=model,

                    base_cols=
                        base_cols,

                    deep_cols=
                        deep_cols,

                    pca_components=
                        pca_k,
                )

                try:

                    pipe.fit(
                        X_train,
                        y_train,
                    )

                    prediction = (
                        pipe.predict(
                            X_test
                        )
                    )

                except Exception as e:

                    print(
                        "SKIP:",
                        target,
                        candidate_name,
                        model_name,
                        fold_id,
                        repr(e),
                    )

                    continue

                fold_rmse = rmse(
                    y_test,
                    prediction,
                )

                fold_r2 = safe_r2(
                    y_test,
                    prediction,
                )

                rows.append({

                    "target":
                        target,

                    "n_samples":
                        n_samples,

                    "n_splits":
                        n_splits,

                    "n_repeats":
                        n_repeats,

                    "candidate":
                        candidate_name,

                    "model":
                        model_name,

                    "pca_components":
                        (
                            np.nan
                            if pca_k is None
                            else pca_k
                        ),

                    "n_features_base":
                        len(
                            base_cols
                        ),

                    "n_features_deep":
                        (
                            0
                            if deep_cols is None
                            else len(
                                deep_cols
                            )
                        ),

                    "fold":
                        fold_id,

                    "log_rmse":
                        fold_rmse,

                    "log_r2":
                        fold_r2,

                    "beats_baseline":
                        (
                            fold_rmse
                            <
                            baseline_rmse[
                                fold_id
                            ]
                        ),
                })

    return rows


# =====================================================
# MAIN
# =====================================================

def main():

    print("=" * 100)

    print(
        "TABULAR + CLAY SENTINEL-2 "
        "DEEP FUSION"
    )

    print("=" * 100)

    print(
        "Tabular:"
    )
    print(
        TABULAR_TABLE_PATH
    )

    print()
    print(
        "Clay S2:"
    )
    print(
        CLAY_S2_TABLE_PATH
    )

    # =================================================
    # CHECK FILES
    # =================================================

    if not TABULAR_TABLE_PATH.exists():

        raise FileNotFoundError(
            TABULAR_TABLE_PATH
        )

    if not CLAY_S2_TABLE_PATH.exists():

        raise FileNotFoundError(
            CLAY_S2_TABLE_PATH
        )

    # =================================================
    # LOAD
    # =================================================

    tabular = pd.read_csv(
        TABULAR_TABLE_PATH
    )

    clay = pd.read_csv(
        CLAY_S2_TABLE_PATH
    )

    print()
    print(
        "Tabular shape:",
        tabular.shape,
    )

    print(
        "Clay S2 shape:",
        clay.shape,
    )

    # =================================================
    # BASIC CHECKS
    # =================================================

    if "GFD_ID" not in tabular.columns:

        raise KeyError(
            "GFD_ID missing "
            "from tabular table."
        )

    if "GFD_ID" not in clay.columns:

        raise KeyError(
            "GFD_ID missing "
            "from Clay S2 table."
        )

    # -------------------------------------------------
    # Support alternate damage column spelling
    # -------------------------------------------------

    if (
        "total_damages_000_USD"
        not in tabular.columns
        and
        "total_damages_(000_USD)"
        in tabular.columns
    ):

        tabular = tabular.rename(
            columns={
                "total_damages_(000_USD)":
                    "total_damages_000_USD"
            }
        )

    # -------------------------------------------------
    # Make GFD IDs consistent
    # -------------------------------------------------

    tabular["GFD_ID"] = (
        pd.to_numeric(
            tabular["GFD_ID"],
            errors="coerce",
        )
    )

    clay["GFD_ID"] = (
        pd.to_numeric(
            clay["GFD_ID"],
            errors="coerce",
        )
    )

    tabular = tabular.dropna(
        subset=["GFD_ID"]
    )

    clay = clay.dropna(
        subset=["GFD_ID"]
    )

    tabular["GFD_ID"] = (
        tabular["GFD_ID"]
        .astype(int)
    )

    clay["GFD_ID"] = (
        clay["GFD_ID"]
        .astype(int)
    )

    # =================================================
    # KEEP ONLY EVENTS WITH S2 EMBEDDINGS
    # =================================================

    if (
        "n_valid_patches"
        not in clay.columns
    ):

        raise KeyError(
            "n_valid_patches "
            "missing from Clay S2 table."
        )

    clay = clay[
        clay["n_valid_patches"] > 0
    ].copy()

    print()
    print(
        "Clay S2 events with embeddings:",
        len(clay),
    )

    # =================================================
    # REMOVE ANY TARGETS FROM CLAY TABLE
    # =================================================

    drop_from_clay = [
        c
        for c in TARGETS
        if c in clay.columns
    ]

    if len(drop_from_clay) > 0:

        clay = clay.drop(
            columns=
                drop_from_clay
        )

    # =================================================
    # MERGE
    # =================================================

    df = tabular.merge(
        clay,
        on="GFD_ID",
        how="inner",
        suffixes=(
            "",
            "_clay_s2",
        ),
    )

    print()
    print(
        "Merged shape:",
        df.shape,
    )

    print(
        "Unique events:",
        df[
            "GFD_ID"
        ].nunique(),
    )

    # =================================================
    # SAVE EVENT TABLE
    # =================================================

    df.to_csv(
        OUTPUT_EVENT_TABLE,
        index=False,
    )

    print(
        "Saved event table:",
        OUTPUT_EVENT_TABLE,
    )

    # =================================================
    # FEATURE GROUPS
    # =================================================

    groups = (
        get_feature_groups(
            df
        )
    )

    print()
    print("=" * 100)
    print(
        "FEATURE GROUPS"
    )
    print("=" * 100)

    for (
        group_name,
        cols,
    ) in groups.items():

        print(
            f"{group_name}: "
            f"{len(cols)}"
        )

    if (
        len(
            groups[
                "clay_s2_deep"
            ]
        )
        != 12288
    ):

        print()
        print(
            "WARNING: expected "
            "12288 Clay S2 deep "
            "columns but found:",
            len(
                groups[
                    "clay_s2_deep"
                ]
            ),
        )

    # =================================================
    # TARGET AVAILABILITY
    # =================================================

    print()
    print("=" * 100)
    print(
        "TARGET AVAILABILITY"
    )
    print("=" * 100)

    for target in TARGETS:

        if target not in df.columns:

            print(
                target,
                ": MISSING"
            )

        else:

            print(
                target,
                ":",
                int(
                    df[target]
                    .notna()
                    .sum()
                ),
                "/",
                len(df),
            )

    # =================================================
    # RUN ALL TARGETS
    # =================================================

    all_rows = []

    for target in TARGETS:

        if target not in df.columns:

            print(
                "Skipping missing target:",
                target,
            )

            continue

        target_rows = run_target(
            df=
                df,

            groups=
                groups,

            target=
                target,
        )

        all_rows.extend(
            target_rows
        )

        # Save continuously
        pd.DataFrame(
            all_rows
        ).to_csv(
            OUTPUT_FOLD_RESULTS,
            index=False,
        )

    # =================================================
    # FOLD RESULTS
    # =================================================

    results = pd.DataFrame(
        all_rows
    )

    if len(results) == 0:

        raise RuntimeError(
            "No results produced."
        )

    results.to_csv(
        OUTPUT_FOLD_RESULTS,
        index=False,
    )

    # =================================================
    # SUMMARY
    # =================================================

    summary = (
        results
        .groupby(
            [
                "target",
                "n_samples",
                "n_splits",
                "n_repeats",
                "candidate",
                "model",
                "pca_components",
                "n_features_base",
                "n_features_deep",
            ],
            as_index=False,
            dropna=False,
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

            wins_vs_baseline=(
                "beats_baseline",
                lambda x:
                    int(
                        x.fillna(
                            False
                        ).sum()
                    ),
            ),

            win_rate_vs_baseline=(
                "beats_baseline",
                lambda x:
                    (
                        float(
                            x.dropna()
                            .mean()
                        )
                        if (
                            len(
                                x.dropna()
                            )
                            > 0
                        )
                        else np.nan
                    ),
            ),
        )
    )

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
        OUTPUT_SUMMARY,
        index=False,
    )

    # =================================================
    # BEST PER TARGET
    # =================================================

    best_rows = []

    for target in TARGETS:

        sub = summary[
            summary["target"]
            == target
        ].copy()

        if len(sub) == 0:
            continue

        best = (
            sub
            .sort_values(
                "mean_log_rmse"
            )
            .iloc[0]
        )

        best_rows.append(
            best
        )

    best_df = pd.DataFrame(
        best_rows
    )

    best_df.to_csv(
        OUTPUT_BEST,
        index=False,
    )

    # =================================================
    # DISPLAY
    # =================================================

    print()
    print("=" * 100)
    print(
        "FINAL RESULTS"
    )
    print("=" * 100)

    for target in TARGETS:

        sub = summary[
            summary["target"]
            == target
        ].copy()

        if len(sub) == 0:
            continue

        print()
        print("-" * 100)
        print(
            "TARGET:",
            target,
        )
        print("-" * 100)

        display_cols = [
            "n_samples",
            "candidate",
            "model",
            "pca_components",
            "mean_log_rmse",
            "std_log_rmse",
            "mean_log_r2",
            "std_log_r2",
            "wins_vs_baseline",
            "win_rate_vs_baseline",
        ]

        print(
            sub[
                display_cols
            ]
            .sort_values(
                "mean_log_rmse"
            )
            .to_string(
                index=False
            )
        )

    print()
    print("=" * 100)
    print(
        "BEST MODEL PER TARGET"
    )
    print("=" * 100)

    print(
        best_df[
            [
                "target",
                "n_samples",
                "candidate",
                "model",
                "pca_components",
                "mean_log_rmse",
                "mean_log_r2",
                "wins_vs_baseline",
                "win_rate_vs_baseline",
            ]
        ]
        .to_string(
            index=False
        )
    )

    print()
    print("=" * 100)
    print("DONE")
    print("=" * 100)

    print(
        "Event table:"
    )
    print(
        OUTPUT_EVENT_TABLE
    )

    print()
    print(
        "Fold results:"
    )
    print(
        OUTPUT_FOLD_RESULTS
    )

    print()
    print(
        "Summary:"
    )
    print(
        OUTPUT_SUMMARY
    )

    print()
    print(
        "Best per target:"
    )
    print(
        OUTPUT_BEST
    )


if __name__ == "__main__":
    main()