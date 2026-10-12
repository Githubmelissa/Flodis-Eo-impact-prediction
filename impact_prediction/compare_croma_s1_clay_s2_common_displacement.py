from pathlib import Path
import warnings

import numpy as np
import pandas as pd

from sklearn.decomposition import PCA
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import RepeatedKFold
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")


# =====================================================
# PATHS
# =====================================================

# SAME cohort used in the previous fair Clay S1/S2 experiment
COMMON_COHORT_PATH = Path(
    "/home/melissa/flood_project/models_benchmark/"
    "impact_prediction/outputs/"
    "clay_s1_s2_common_displacement/"
    "common_event_cohort.csv"
)

CROMA_S1_PATH = Path(
    "/home/melissa/flood_project/models_benchmark/"
    "CROMA/flodis_outputs/"
    "croma_event_features_valid_only/"
    "all_events_croma_event_features_valid_only.csv"
)

CLAY_S2_PATH = Path(
    "/home/melissa/flood_project/models_benchmark/"
    "Clay/flodis_outputs/"
    "clay_s2_event_features_valid_only/"
    "all_events_clay_s2_event_features_valid_only.csv"
)

OUTPUT_DIR = Path(
    "/home/melissa/flood_project/models_benchmark/"
    "impact_prediction/outputs/"
    "croma_s1_clay_s2_common_displacement"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

FOLD_RESULTS_FILE = OUTPUT_DIR / "fold_results.csv"
PREDICTIONS_FILE = OUTPUT_DIR / "cv_predictions.csv"
REPEAT_RESULTS_FILE = OUTPUT_DIR / "repeat_results.csv"
SUMMARY_FILE = OUTPUT_DIR / "comparison_summary.csv"
BEST_FILE = OUTPUT_DIR / "best_per_representation.csv"
EVENT_TABLE_FILE = OUTPUT_DIR / "common_event_table.csv"


# =====================================================
# SETTINGS
# =====================================================

TARGET = "displacements"

RANDOM_STATE = 42

N_SPLITS = 5
N_REPEATS = 6

PCA_COMPONENTS = [
    4,
    8,
    16,
]


# =====================================================
# HELPERS
# =====================================================

def rmse(y_true, y_pred):

    return float(
        np.sqrt(
            mean_squared_error(
                y_true,
                y_pred,
            )
        )
    )


def safe_r2(y_true, y_pred):

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


def prepare_gfd_id(df, name):

    df = df.copy()

    if "GFD_ID" not in df.columns:

        raise KeyError(
            f"GFD_ID missing from {name}"
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

    if df["GFD_ID"].duplicated().any():

        duplicates = (
            df.loc[
                df["GFD_ID"].duplicated(),
                "GFD_ID",
            ]
            .tolist()
        )

        raise RuntimeError(
            f"Duplicate GFD_ID in {name}: "
            f"{duplicates}"
        )

    return df


# =====================================================
# MODELS
# =====================================================

def get_models():

    return {

        "ridge": Ridge(
            alpha=1.0,
        ),

        "random_forest":
            RandomForestRegressor(
                n_estimators=300,
                min_samples_leaf=2,
                random_state=RANDOM_STATE,
                n_jobs=-1,
            ),

        "extra_trees":
            ExtraTreesRegressor(
                n_estimators=300,
                min_samples_leaf=2,
                random_state=RANDOM_STATE,
                n_jobs=-1,
            ),
    }


# =====================================================
# PREPROCESSING
# =====================================================

def preprocess_context(
    X_train,
    X_test,
):

    imputer = SimpleImputer(
        strategy="median"
    )

    scaler = StandardScaler()

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

    return (
        X_train.astype(np.float32),
        X_test.astype(np.float32),
    )


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

    # Fit everything ONLY on training fold
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


# =====================================================
# LOAD COMMON DATASET
# =====================================================

def load_common_dataset():

    print("=" * 100)
    print("LOADING SAME COMMON COHORT")
    print("=" * 100)

    cohort = pd.read_csv(
        COMMON_COHORT_PATH
    )

    croma = pd.read_csv(
        CROMA_S1_PATH
    )

    clay_s2 = pd.read_csv(
        CLAY_S2_PATH
    )

    cohort = prepare_gfd_id(
        cohort,
        "common cohort",
    )

    croma = prepare_gfd_id(
        croma,
        "CROMA S1",
    )

    clay_s2 = prepare_gfd_id(
        clay_s2,
        "Clay S2",
    )

    print(
        "Original common cohort:",
        len(cohort),
    )

    print(
        "CROMA table events:",
        len(croma),
    )

    print(
        "Clay S2 table events:",
        len(clay_s2),
    )

    # =================================================
    # VALID EMBEDDINGS ONLY
    # =================================================

    if "n_valid_patches" in croma.columns:

        croma = croma[
            croma["n_valid_patches"] > 0
        ].copy()

    if "n_valid_patches" in clay_s2.columns:

        clay_s2 = clay_s2[
            clay_s2["n_valid_patches"] > 0
        ].copy()

    print()
    print(
        "CROMA events with valid embeddings:",
        len(croma),
    )

    print(
        "Clay S2 events with valid embeddings:",
        len(clay_s2),
    )

    # =================================================
    # IDENTIFY DEEP FEATURES
    # =================================================

    croma_deep_cols = [
        c
        for c in croma.columns
        if (
            c.startswith("croma_")
            and pd.api.types.is_numeric_dtype(
                croma[c]
            )
        )
    ]

    clay_s2_deep_cols = [
        c
        for c in clay_s2.columns
        if (
            (
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
            and pd.api.types.is_numeric_dtype(
                clay_s2[c]
            )
        )
    ]

    print()
    print(
        "CROMA S1 deep features:",
        len(croma_deep_cols),
    )

    print(
        "Clay S2 deep features:",
        len(clay_s2_deep_cols),
    )

    if len(croma_deep_cols) == 0:

        raise RuntimeError(
            "No CROMA deep feature "
            "columns detected."
        )

    if len(clay_s2_deep_cols) == 0:

        raise RuntimeError(
            "No Clay S2 deep feature "
            "columns detected."
        )

    # =================================================
    # RENAME TO AVOID ANY COLUMN COLLISIONS
    # =================================================

    croma_small = croma[
        ["GFD_ID"]
        + croma_deep_cols
    ].copy()

    clay_small = clay_s2[
        ["GFD_ID"]
        + clay_s2_deep_cols
    ].copy()

    # =================================================
    # MERGE INTO EXACT PREVIOUS COHORT
    # =================================================

    df = (
        cohort
        .merge(
            croma_small,
            on="GFD_ID",
            how="inner",
        )
        .merge(
            clay_small,
            on="GFD_ID",
            how="inner",
        )
    )

    df = (
        df
        .sort_values(
            "GFD_ID"
        )
        .reset_index(
            drop=True
        )
    )

    print()
    print("=" * 100)
    print("FINAL COMMON COHORT")
    print("=" * 100)

    print(
        "Previous Clay common cohort:",
        len(cohort),
    )

    print(
        "CROMA-S1 + Clay-S2 common events:",
        len(df),
    )

    missing_croma = sorted(
        set(
            cohort["GFD_ID"]
        )
        - set(
            croma["GFD_ID"]
        )
    )

    if len(missing_croma) > 0:

        print(
            "Previous cohort events "
            "missing CROMA:",
            missing_croma,
        )

    print()
    print("GFD IDs:")
    print(
        df["GFD_ID"]
        .tolist()
    )

    # =================================================
    # CONTEXTUAL COLUMNS
    #
    # common_event_cohort.csv already contains
    # contextual features from previous fair experiment
    # =================================================

    contextual_cols = [
        c
        for c in cohort.columns
        if c not in [
            "GFD_ID",
            TARGET,
        ]
        and pd.api.types.is_numeric_dtype(
            cohort[c]
        )
    ]

    print()
    print(
        "Contextual features:",
        len(contextual_cols),
    )

    # =================================================
    # NUMERIC CHECKS
    # =================================================

    croma_matrix = (
        df[croma_deep_cols]
        .apply(
            pd.to_numeric,
            errors="coerce",
        )
        .to_numpy()
    )

    clay_matrix = (
        df[clay_s2_deep_cols]
        .apply(
            pd.to_numeric,
            errors="coerce",
        )
        .to_numpy()
    )

    print(
        "Non-finite CROMA values:",
        int(
            (~np.isfinite(croma_matrix))
            .sum()
        ),
    )

    print(
        "Non-finite Clay S2 values:",
        int(
            (~np.isfinite(clay_matrix))
            .sum()
        ),
    )

    df.to_csv(
        EVENT_TABLE_FILE,
        index=False,
    )

    print()
    print(
        "Saved common table:",
        EVENT_TABLE_FILE,
    )

    return (
        df,
        contextual_cols,
        croma_deep_cols,
        clay_s2_deep_cols,
    )


# =====================================================
# MAIN
# =====================================================

def main():

    print("=" * 110)

    print(
        "CROMA-S1 + CLAY-S2 "
        "CROSS-MODEL FUSION"
    )

    print("=" * 110)

    (
        df,
        contextual_cols,
        croma_cols,
        clay_s2_cols,
    ) = load_common_dataset()

    n = len(df)

    if n < 20:

        raise RuntimeError(
            f"Only {n} common events found. "
            "Stopping because cohort became "
            "too small."
        )

    # =================================================
    # MATRICES
    # =================================================

    X_context = (
        df[contextual_cols]
        .apply(
            pd.to_numeric,
            errors="coerce",
        )
        .to_numpy(
            dtype=np.float32
        )
    )

    X_croma = (
        df[croma_cols]
        .apply(
            pd.to_numeric,
            errors="coerce",
        )
        .to_numpy(
            dtype=np.float32
        )
    )

    X_clay_s2 = (
        df[clay_s2_cols]
        .apply(
            pd.to_numeric,
            errors="coerce",
        )
        .to_numpy(
            dtype=np.float32
        )
    )

    y_original = (
        pd.to_numeric(
            df[TARGET],
            errors="coerce",
        )
        .to_numpy(
            dtype=float
        )
    )

    y = safe_log1p(
        y_original
    )

    gfd_ids = (
        df["GFD_ID"]
        .astype(int)
        .to_numpy()
    )

    print()
    print("=" * 100)
    print("MATRICES")
    print("=" * 100)

    print(
        "Events:",
        n,
    )

    print(
        "Contextual:",
        X_context.shape,
    )

    print(
        "CROMA S1:",
        X_croma.shape,
    )

    print(
        "Clay S2:",
        X_clay_s2.shape,
    )

    # =================================================
    # EXACT SAME SPLITS FOR ALL CONFIGURATIONS
    # =================================================

    cv = RepeatedKFold(
        n_splits=N_SPLITS,
        n_repeats=N_REPEATS,
        random_state=RANDOM_STATE,
    )

    splits = list(
        cv.split(
            np.arange(n)
        )
    )

    print()
    print(
        "CV:",
        f"{N_SPLITS}-fold x "
        f"{N_REPEATS} repeats"
    )

    print(
        "Validation folds:",
        len(splits),
    )

    fold_rows = []
    prediction_rows = []

    # =================================================
    # CV LOOP
    # =================================================

    for split_index, (
        train_idx,
        test_idx,
    ) in enumerate(
        splits
    ):

        repeat_id = (
            split_index // N_SPLITS
        ) + 1

        fold_in_repeat = (
            split_index % N_SPLITS
        ) + 1

        global_fold = (
            split_index + 1
        )

        print(
            f"Repeat {repeat_id}/{N_REPEATS}"
            f" | Fold "
            f"{fold_in_repeat}/{N_SPLITS}"
        )

        y_train = y[
            train_idx
        ]

        y_test = y[
            test_idx
        ]

        # =================================================
        # MEDIAN BASELINE
        # =================================================

        median_value = np.median(
            y_train
        )

        baseline_pred = np.full(
            len(test_idx),
            median_value,
            dtype=float,
        )

        baseline_rmse = rmse(
            y_test,
            baseline_pred,
        )

        baseline_r2 = safe_r2(
            y_test,
            baseline_pred,
        )

        fold_rows.append({

            "repeat":
                repeat_id,

            "fold_in_repeat":
                fold_in_repeat,

            "global_fold":
                global_fold,

            "candidate":
                "baseline_median",

            "model":
                "baseline_median",

            "pca_components":
                np.nan,

            "log_rmse":
                baseline_rmse,

            "log_r2":
                baseline_r2,

            "beats_baseline":
                np.nan,
        })

        for idx, pred in zip(
            test_idx,
            baseline_pred,
        ):

            prediction_rows.append({

                "repeat":
                    repeat_id,

                "candidate":
                    "baseline_median",

                "model":
                    "baseline_median",

                "pca_components":
                    np.nan,

                "GFD_ID":
                    int(
                        gfd_ids[idx]
                    ),

                "y_true_log":
                    float(
                        y[idx]
                    ),

                "y_pred_log":
                    float(
                        pred
                    ),
            })

        # =================================================
        # CONTEXTUAL FEATURES
        # =================================================

        (
            context_train,
            context_test,
        ) = preprocess_context(

            X_context[
                train_idx
            ],

            X_context[
                test_idx
            ],
        )

        # Contextual only
        for (
            model_name,
            model,
        ) in get_models().items():

            model.fit(
                context_train,
                y_train,
            )

            pred = model.predict(
                context_test
            )

            score_rmse = rmse(
                y_test,
                pred,
            )

            score_r2 = safe_r2(
                y_test,
                pred,
            )

            fold_rows.append({

                "repeat":
                    repeat_id,

                "fold_in_repeat":
                    fold_in_repeat,

                "global_fold":
                    global_fold,

                "candidate":
                    "contextual",

                "model":
                    model_name,

                "pca_components":
                    np.nan,

                "log_rmse":
                    score_rmse,

                "log_r2":
                    score_r2,

                "beats_baseline":
                    score_rmse
                    < baseline_rmse,
            })

            for idx, value in zip(
                test_idx,
                pred,
            ):

                prediction_rows.append({

                    "repeat":
                        repeat_id,

                    "candidate":
                        "contextual",

                    "model":
                        model_name,

                    "pca_components":
                        np.nan,

                    "GFD_ID":
                        int(
                            gfd_ids[idx]
                        ),

                    "y_true_log":
                        float(
                            y[idx]
                        ),

                    "y_pred_log":
                        float(
                            value
                        ),
                })

        # =================================================
        # DEEP REPRESENTATIONS
        # =================================================

        for k in PCA_COMPONENTS:

            print(
                "   PCA:",
                k,
            )

            # -----------------------------------------
            # CROMA S1
            # -----------------------------------------

            (
                croma_train,
                croma_test,
            ) = preprocess_deep(

                X_croma[
                    train_idx
                ],

                X_croma[
                    test_idx
                ],

                k,
            )

            # -----------------------------------------
            # Clay S2
            # -----------------------------------------

            (
                clay_train,
                clay_test,
            ) = preprocess_deep(

                X_clay_s2[
                    train_idx
                ],

                X_clay_s2[
                    test_idx
                ],

                k,
            )

            # =================================================
            # CANDIDATES
            # =================================================

            candidate_matrices = {

                "croma_s1":
                    (
                        croma_train,
                        croma_test,
                    ),

                "clay_s2":
                    (
                        clay_train,
                        clay_test,
                    ),

                "croma_s1_plus_clay_s2":
                    (
                        np.hstack([
                            croma_train,
                            clay_train,
                        ]),

                        np.hstack([
                            croma_test,
                            clay_test,
                        ]),
                    ),

                "contextual_plus_croma_s1":
                    (
                        np.hstack([
                            context_train,
                            croma_train,
                        ]),

                        np.hstack([
                            context_test,
                            croma_test,
                        ]),
                    ),

                "contextual_plus_clay_s2":
                    (
                        np.hstack([
                            context_train,
                            clay_train,
                        ]),

                        np.hstack([
                            context_test,
                            clay_test,
                        ]),
                    ),

                "contextual_plus_croma_s1_plus_clay_s2":
                    (
                        np.hstack([
                            context_train,
                            croma_train,
                            clay_train,
                        ]),

                        np.hstack([
                            context_test,
                            croma_test,
                            clay_test,
                        ]),
                    ),
            }

            # =================================================
            # MODELS
            # =================================================

            for (
                candidate_name,
                (
                    X_train_candidate,
                    X_test_candidate,
                ),
            ) in candidate_matrices.items():

                for (
                    model_name,
                    model,
                ) in get_models().items():

                    model.fit(
                        X_train_candidate,
                        y_train,
                    )

                    pred = model.predict(
                        X_test_candidate
                    )

                    score_rmse = rmse(
                        y_test,
                        pred,
                    )

                    score_r2 = safe_r2(
                        y_test,
                        pred,
                    )

                    fold_rows.append({

                        "repeat":
                            repeat_id,

                        "fold_in_repeat":
                            fold_in_repeat,

                        "global_fold":
                            global_fold,

                        "candidate":
                            candidate_name,

                        "model":
                            model_name,

                        "pca_components":
                            k,

                        "log_rmse":
                            score_rmse,

                        "log_r2":
                            score_r2,

                        "beats_baseline":
                            score_rmse
                            < baseline_rmse,
                    })

                    for idx, value in zip(
                        test_idx,
                        pred,
                    ):

                        prediction_rows.append({

                            "repeat":
                                repeat_id,

                            "candidate":
                                candidate_name,

                            "model":
                                model_name,

                            "pca_components":
                                k,

                            "GFD_ID":
                                int(
                                    gfd_ids[idx]
                                ),

                            "y_true_log":
                                float(
                                    y[idx]
                                ),

                            "y_pred_log":
                                float(
                                    value
                                ),
                        })

        # Save after each fold
        pd.DataFrame(
            fold_rows
        ).to_csv(
            FOLD_RESULTS_FILE,
            index=False,
        )

        pd.DataFrame(
            prediction_rows
        ).to_csv(
            PREDICTIONS_FILE,
            index=False,
        )

    # =====================================================
    # FINAL TABLES
    # =====================================================

    fold_df = pd.DataFrame(
        fold_rows
    )

    pred_df = pd.DataFrame(
        prediction_rows
    )

    # =====================================================
    # REPEAT-LEVEL METRICS
    # =====================================================

    repeat_rows = []

    for keys, group in pred_df.groupby(
        [
            "repeat",
            "candidate",
            "model",
            "pca_components",
        ],
        dropna=False,
    ):

        (
            repeat_id,
            candidate,
            model_name,
            pca_k,
        ) = keys

        y_true = (
            group[
                "y_true_log"
            ]
            .to_numpy()
        )

        y_pred = (
            group[
                "y_pred_log"
            ]
            .to_numpy()
        )

        repeat_rows.append({

            "repeat":
                repeat_id,

            "candidate":
                candidate,

            "model":
                model_name,

            "pca_components":
                pca_k,

            "log_rmse":
                rmse(
                    y_true,
                    y_pred,
                ),

            "log_r2":
                safe_r2(
                    y_true,
                    y_pred,
                ),
        })

    repeat_df = pd.DataFrame(
        repeat_rows
    )

    repeat_df.to_csv(
        REPEAT_RESULTS_FILE,
        index=False,
    )

    # =====================================================
    # SUMMARIES
    # =====================================================

    fold_summary = (
        fold_df
        .groupby(
            [
                "candidate",
                "model",
                "pca_components",
            ],
            as_index=False,
            dropna=False,
        )
        .agg(

            mean_fold_log_rmse=(
                "log_rmse",
                "mean",
            ),

            std_fold_log_rmse=(
                "log_rmse",
                "std",
            ),

            mean_fold_log_r2=(
                "log_r2",
                "mean",
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
                        if len(
                            x.dropna()
                        ) > 0
                        else np.nan
                    ),
            ),
        )
    )

    repeat_summary = (
        repeat_df
        .groupby(
            [
                "candidate",
                "model",
                "pca_components",
            ],
            as_index=False,
            dropna=False,
        )
        .agg(

            mean_repeat_log_rmse=(
                "log_rmse",
                "mean",
            ),

            std_repeat_log_rmse=(
                "log_rmse",
                "std",
            ),

            mean_repeat_log_r2=(
                "log_r2",
                "mean",
            ),

            std_repeat_log_r2=(
                "log_r2",
                "std",
            ),
        )
    )

    summary = fold_summary.merge(
        repeat_summary,
        on=[
            "candidate",
            "model",
            "pca_components",
        ],
        how="left",
    )

    summary = (
        summary
        .sort_values(
            "mean_repeat_log_rmse"
        )
        .reset_index(
            drop=True
        )
    )

    summary.to_csv(
        SUMMARY_FILE,
        index=False,
    )

    # =====================================================
    # BEST PER REPRESENTATION
    # =====================================================

    representation_order = [

        "baseline_median",

        "contextual",

        "croma_s1",

        "clay_s2",

        "croma_s1_plus_clay_s2",

        "contextual_plus_croma_s1",

        "contextual_plus_clay_s2",

        "contextual_plus_croma_s1_plus_clay_s2",
    ]

    best_rows = []

    for candidate in representation_order:

        sub = summary[
            summary["candidate"]
            == candidate
        ].copy()

        if len(sub) == 0:
            continue

        best_rows.append(
            sub
            .sort_values(
                "mean_repeat_log_rmse"
            )
            .iloc[0]
        )

    best_df = pd.DataFrame(
        best_rows
    )

    best_df.to_csv(
        BEST_FILE,
        index=False,
    )

    # =====================================================
    # DISPLAY
    # =====================================================

    display_cols = [

        "candidate",

        "model",

        "pca_components",

        "mean_repeat_log_rmse",

        "std_repeat_log_rmse",

        "mean_repeat_log_r2",

        "std_repeat_log_r2",

        "wins_vs_baseline",

        "win_rate_vs_baseline",
    ]

    print()
    print("=" * 120)

    print(
        "BEST CONFIGURATION "
        "PER REPRESENTATION"
    )

    print("=" * 120)

    print(
        best_df[
            display_cols
        ]
        .to_string(
            index=False
        )
    )

    print()
    print("=" * 120)
    print("DONE")
    print("=" * 120)

    print(
        "Best table:"
    )

    print(
        BEST_FILE
    )


if __name__ == "__main__":
    main()
