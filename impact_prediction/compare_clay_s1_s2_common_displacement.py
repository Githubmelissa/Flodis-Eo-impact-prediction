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

TABULAR_PATH = Path(
    "/home/melissa/flood_project/"
    "models_benchmark/impact_prediction/outputs/"
    "tabular_phase1_event_table_FIXED.csv"
)

CLAY_S1_PATH = Path(
    "/home/melissa/flood_project/"
    "models_benchmark/Clay/flodis_outputs/"
    "clay_event_features_valid_only/"
    "all_events_clay_event_features_valid_only.csv"
)

CLAY_S2_PATH = Path(
    "/home/melissa/flood_project/"
    "models_benchmark/Clay/flodis_outputs/"
    "clay_s2_event_features_valid_only/"
    "all_events_clay_s2_event_features_valid_only.csv"
)

OUTPUT_DIR = Path(
    "/home/melissa/flood_project/"
    "models_benchmark/impact_prediction/outputs/"
    "clay_s1_s2_common_displacement"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

COMMON_EVENTS_FILE = (
    OUTPUT_DIR /
    "common_event_cohort.csv"
)

FOLD_RESULTS_FILE = (
    OUTPUT_DIR /
    "fold_results.csv"
)

PREDICTIONS_FILE = (
    OUTPUT_DIR /
    "cv_predictions.csv"
)

REPEAT_RESULTS_FILE = (
    OUTPUT_DIR /
    "repeat_results.csv"
)

SUMMARY_FILE = (
    OUTPUT_DIR /
    "comparison_summary.csv"
)


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

EXPECTED_COMMON_EVENTS = 45


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

TARGET_COLS = {
    "displacements",
    "total_deaths",
    "total_affected_EMDAT",
    "total_damages_000_USD",
    "total_damages_(000_USD)",
}


# =====================================================
# BASIC HELPERS
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

    if "GFD_ID" not in df.columns:
        raise KeyError(
            f"GFD_ID missing from {name}"
        )

    df = df.copy()

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

    duplicated = (
        df["GFD_ID"]
        .duplicated()
    )

    if duplicated.any():

        duplicate_ids = (
            df.loc[
                duplicated,
                "GFD_ID"
            ]
            .tolist()
        )

        raise RuntimeError(
            f"Duplicate GFD_ID values "
            f"in {name}: {duplicate_ids}"
        )

    return df


# =====================================================
# FEATURE GROUPS
# =====================================================

def get_contextual_columns(tabular):

    numeric_cols = [
        c
        for c in tabular.columns
        if pd.api.types.is_numeric_dtype(
            tabular[c]
        )
    ]

    worldpop = [
        c
        for c in numeric_cols
        if (
            "population" in c.lower()
            or c == "flood_area_km2"
        )
    ]

    gpm = [
        c
        for c in numeric_cols
        if (
            "gpm" in c.lower()
            or "rainfall" in c.lower()
        )
    ]

    srtm = [
        c
        for c in numeric_cols
        if (
            "elevation" in c.lower()
            or "slope" in c.lower()
        )
    ]

    forbidden = (
        TARGET_COLS
        | META_COLS
    )

    contextual = sorted(
        set(
            worldpop
            + gpm
            + srtm
        )
        - forbidden
    )

    return (
        sorted(set(worldpop) - forbidden),
        sorted(set(gpm) - forbidden),
        sorted(set(srtm) - forbidden),
        contextual,
    )


def get_s1_deep_columns(df):

    return sorted([
        c
        for c in df.columns
        if (
            c.startswith("clay_before_")
            or c.startswith("clay_after_")
            or c.startswith("clay_delta_")
        )
    ])


def get_s2_deep_columns(df):

    return sorted([
        c
        for c in df.columns
        if (
            c.startswith("clay_s2_before_")
            or c.startswith("clay_s2_after_")
            or c.startswith("clay_s2_delta_")
        )
    ])


# =====================================================
# PREPROCESSING
# Everything is FIT ON TRAINING FOLD ONLY
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

    # -----------------------------------------
    # Imputation
    # -----------------------------------------

    X_train = imputer.fit_transform(
        X_train
    )

    X_test = imputer.transform(
        X_test
    )

    # -----------------------------------------
    # Scaling
    # -----------------------------------------

    X_train = scaler.fit_transform(
        X_train
    )

    X_test = scaler.transform(
        X_test
    )

    # -----------------------------------------
    # PCA
    # IMPORTANT:
    # fitted ONLY on training fold
    # -----------------------------------------

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
# BUILD COMMON EVENT COHORT
# =====================================================

def build_common_dataset():

    print("=" * 100)
    print("LOADING DATA")
    print("=" * 100)

    tabular = pd.read_csv(
        TABULAR_PATH
    )

    s1 = pd.read_csv(
        CLAY_S1_PATH
    )

    s2 = pd.read_csv(
        CLAY_S2_PATH
    )

    tabular = prepare_gfd_id(
        tabular,
        "tabular",
    )

    s1 = prepare_gfd_id(
        s1,
        "Clay S1",
    )

    s2 = prepare_gfd_id(
        s2,
        "Clay S2",
    )

    print(
        "Tabular events:",
        len(tabular),
    )

    print(
        "Clay S1 events:",
        len(s1),
    )

    print(
        "Clay S2 events:",
        len(s2),
    )

    # =================================================
    # TARGET
    # =================================================

    if TARGET not in tabular.columns:

        raise KeyError(
            f"{TARGET} missing "
            "from tabular table."
        )

    tabular[TARGET] = (
        pd.to_numeric(
            tabular[TARGET],
            errors="coerce",
        )
    )

    # =================================================
    # CONTEXTUAL FEATURES
    # =================================================

    (
        worldpop_cols,
        gpm_cols,
        srtm_cols,
        contextual_cols,
    ) = get_contextual_columns(
        tabular
    )

    print()
    print("Contextual groups:")

    print(
        "  WorldPop:",
        len(worldpop_cols),
    )

    print(
        "  GPM:",
        len(gpm_cols),
    )

    print(
        "  SRTM:",
        len(srtm_cols),
    )

    print(
        "  Full contextual:",
        len(contextual_cols),
    )

    # =================================================
    # S1 DEEP
    # =================================================

    if "n_valid_patches" not in s1.columns:
        raise KeyError(
            "n_valid_patches missing "
            "from Clay S1 table"
        )

    s1 = s1[
        s1["n_valid_patches"] > 0
    ].copy()

    s1_deep_cols = (
        get_s1_deep_columns(
            s1
        )
    )

    print()
    print(
        "S1 events with embeddings:",
        len(s1),
    )

    print(
        "S1 deep features:",
        len(s1_deep_cols),
    )

    # =================================================
    # S2 DEEP
    # =================================================

    if "n_valid_patches" not in s2.columns:
        raise KeyError(
            "n_valid_patches missing "
            "from Clay S2 table"
        )

    s2 = s2[
        s2["n_valid_patches"] > 0
    ].copy()

    s2_deep_cols = (
        get_s2_deep_columns(
            s2
        )
    )

    print(
        "S2 events with embeddings:",
        len(s2),
    )

    print(
        "S2 deep features:",
        len(s2_deep_cols),
    )

    if len(s1_deep_cols) == 0:
        raise RuntimeError(
            "No S1 deep features found."
        )

    if len(s2_deep_cols) == 0:
        raise RuntimeError(
            "No S2 deep features found."
        )

    # Usually both should be 12,288
    if len(s1_deep_cols) != 12288:

        print(
            "WARNING: expected 12288 "
            "S1 Clay deep features, found:",
            len(s1_deep_cols),
        )

    if len(s2_deep_cols) != 12288:

        print(
            "WARNING: expected 12288 "
            "S2 Clay deep features, found:",
            len(s2_deep_cols),
        )

    # =================================================
    # REDUCE TABLES BEFORE MERGING
    # =================================================

    tab_keep = (
        ["GFD_ID", TARGET]
        + contextual_cols
    )

    tabular_small = (
        tabular[
            list(
                dict.fromkeys(
                    tab_keep
                )
            )
        ]
        .copy()
    )

    s1_small = s1[
        ["GFD_ID"]
        + s1_deep_cols
    ].copy()

    s2_small = s2[
        ["GFD_ID"]
        + s2_deep_cols
    ].copy()

    # =================================================
    # MERGE ALL MODALITIES FIRST
    # =================================================

    df = (
        tabular_small
        .merge(
            s1_small,
            on="GFD_ID",
            how="inner",
        )
        .merge(
            s2_small,
            on="GFD_ID",
            how="inner",
        )
    )

    # Keep displacement-labelled events
    df = df[
        df[TARGET].notna()
    ].copy()

    df = df[
        df[TARGET] >= 0
    ].copy()

    # VERY IMPORTANT:
    # fixed row order before generating CV splits
    df = (
        df
        .sort_values(
            "GFD_ID"
        )
        .reset_index(
            drop=True
        )
    )

    # =================================================
    # NUMERIC CHECK
    # =================================================

    s1_matrix = (
        df[s1_deep_cols]
        .apply(
            pd.to_numeric,
            errors="coerce",
        )
        .to_numpy()
    )

    s2_matrix = (
        df[s2_deep_cols]
        .apply(
            pd.to_numeric,
            errors="coerce",
        )
        .to_numpy()
    )

    print()
    print("=" * 100)
    print("COMMON COHORT")
    print("=" * 100)

    print(
        "Common S1 + S2 + displacement events:",
        len(df),
    )

    print(
        "Expected:",
        EXPECTED_COMMON_EVENTS,
    )

    print(
        "Non-finite S1 deep values:",
        int(
            (~np.isfinite(s1_matrix))
            .sum()
        ),
    )

    print(
        "Non-finite S2 deep values:",
        int(
            (~np.isfinite(s2_matrix))
            .sum()
        ),
    )

    print()
    print(
        "GFD IDs:"
    )

    print(
        df["GFD_ID"]
        .tolist()
    )

    if len(df) != EXPECTED_COMMON_EVENTS:

        print()
        print(
            "WARNING:"
        )

        print(
            f"Expected {EXPECTED_COMMON_EVENTS} "
            f"common events but found {len(df)}."
        )

    # =================================================
    # SAVE COMMON COHORT
    # =================================================

    cohort_output = df[
        ["GFD_ID", TARGET]
        + contextual_cols
    ].copy()

    cohort_output.to_csv(
        COMMON_EVENTS_FILE,
        index=False,
    )

    return (
        df,
        contextual_cols,
        s1_deep_cols,
        s2_deep_cols,
    )


# =====================================================
# MAIN EXPERIMENT
# =====================================================

def main():

    print("=" * 100)
    print(
        "FAIR CLAY S1 vs S2 vs S1+S2 "
        "DISPLACEMENT COMPARISON"
    )
    print("=" * 100)

    (
        df,
        contextual_cols,
        s1_cols,
        s2_cols,
    ) = build_common_dataset()

    n = len(df)

    if n < 10:
        raise RuntimeError(
            "Too few common events."
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

    X_s1 = (
        df[s1_cols]
        .apply(
            pd.to_numeric,
            errors="coerce",
        )
        .to_numpy(
            dtype=np.float32
        )
    )

    X_s2 = (
        df[s2_cols]
        .apply(
            pd.to_numeric,
            errors="coerce",
        )
        .to_numpy(
            dtype=np.float32
        )
    )

    y_original = (
        df[TARGET]
        .astype(float)
        .to_numpy()
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
    print("FINAL MATRICES")
    print("=" * 100)

    print(
        "Events:",
        n,
    )

    print(
        "Context:",
        X_context.shape,
    )

    print(
        "Clay S1:",
        X_s1.shape,
    )

    print(
        "Clay S2:",
        X_s2.shape,
    )

    # =================================================
    # SAME SPLITS FOR EVERYTHING
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
        "Total validation folds:",
        len(splits),
    )

    fold_rows = []
    prediction_rows = []

    # =================================================
    # LOOP THROUGH EXACT SAME SPLITS
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

        print()
        print(
            f"Repeat {repeat_id}/{N_REPEATS} "
            f"| Fold {fold_in_repeat}/{N_SPLITS}"
        )

        y_train = y[
            train_idx
        ]

        y_test = y[
            test_idx
        ]

        # =============================================
        # MEDIAN BASELINE
        # =============================================

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

        # =============================================
        # CONTEXTUAL PREPROCESSING
        # =============================================

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

        # =============================================
        # CONTEXTUAL BASELINE MODEL
        # =============================================

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
                    "contextual_wp_gpm_srtm",

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

                    "fold_in_repeat":
                        fold_in_repeat,

                    "global_fold":
                        global_fold,

                    "candidate":
                        "contextual_wp_gpm_srtm",

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

        # =============================================
        # DEEP REPRESENTATIONS
        # =============================================

        for k in PCA_COMPONENTS:

            print(
                "  PCA:",
                k,
            )

            # -----------------------------------------
            # S1 PCA
            # -----------------------------------------

            (
                s1_train,
                s1_test,
            ) = preprocess_deep(

                X_s1[
                    train_idx
                ],

                X_s1[
                    test_idx
                ],

                n_components=k,
            )

            # -----------------------------------------
            # S2 PCA
            # -----------------------------------------

            (
                s2_train,
                s2_test,
            ) = preprocess_deep(

                X_s2[
                    train_idx
                ],

                X_s2[
                    test_idx
                ],

                n_components=k,
            )

            # =========================================
            # BUILD FAIR CANDIDATE MATRICES
            # =========================================

            candidate_matrices = {

                # -------------------------------------
                # DEEP ONLY
                # -------------------------------------

                "clay_s1":
                    (
                        s1_train,
                        s1_test,
                    ),

                "clay_s2":
                    (
                        s2_train,
                        s2_test,
                    ),

                "clay_s1_s2":
                    (
                        np.hstack([
                            s1_train,
                            s2_train,
                        ]),

                        np.hstack([
                            s1_test,
                            s2_test,
                        ]),
                    ),

                # -------------------------------------
                # CONTEXT + DEEP
                # -------------------------------------

                "contextual_plus_clay_s1":
                    (
                        np.hstack([
                            context_train,
                            s1_train,
                        ]),

                        np.hstack([
                            context_test,
                            s1_test,
                        ]),
                    ),

                "contextual_plus_clay_s2":
                    (
                        np.hstack([
                            context_train,
                            s2_train,
                        ]),

                        np.hstack([
                            context_test,
                            s2_test,
                        ]),
                    ),

                "contextual_plus_clay_s1_s2":
                    (
                        np.hstack([
                            context_train,
                            s1_train,
                            s2_train,
                        ]),

                        np.hstack([
                            context_test,
                            s1_test,
                            s2_test,
                        ]),
                    ),
            }

            # =========================================
            # MODELS
            # =========================================

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

        # ---------------------------------------------
        # SAVE AFTER EVERY FOLD
        # ---------------------------------------------

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

    # =================================================
    # FINAL FOLD RESULTS
    # =================================================

    fold_df = pd.DataFrame(
        fold_rows
    )

    pred_df = pd.DataFrame(
        prediction_rows
    )

    fold_df.to_csv(
        FOLD_RESULTS_FILE,
        index=False,
    )

    pred_df.to_csv(
        PREDICTIONS_FILE,
        index=False,
    )

    # =================================================
    # REPEAT-LEVEL POOLED METRICS
    #
    # Each repeat contains predictions for ALL
    # common events.
    #
    # This R2 is more meaningful than averaging
    # R2 across tiny individual folds.
    # =================================================

    repeat_rows = []

    repeat_group_cols = [
        "repeat",
        "candidate",
        "model",
        "pca_components",
    ]

    for keys, group in pred_df.groupby(
        repeat_group_cols,
        dropna=False,
    ):

        (
            repeat_id,
            candidate,
            model_name,
            pca_k,
        ) = keys

        y_true = (
            group["y_true_log"]
            .to_numpy()
        )

        y_pred = (
            group["y_pred_log"]
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

    # =================================================
    # FOLD SUMMARY
    # =================================================

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
                        x.fillna(False)
                        .sum()
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

    # =================================================
    # REPEAT SUMMARY
    # =================================================

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

    # =================================================
    # COMPLETE SUMMARY
    # =================================================

    summary = fold_summary.merge(
        repeat_summary,
        on=[
            "candidate",
            "model",
            "pca_components",
        ],
        how="left",
    )

    # Repeat-level RMSE is the primary sorting metric
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

    # =================================================
    # DISPLAY
    # =================================================

    print()
    print("=" * 120)
    print("FINAL FAIR COMPARISON")
    print("=" * 120)

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

    print(
        summary[
            display_cols
        ]
        .head(30)
        .to_string(
            index=False
        )
    )

    print()
    print("=" * 120)
    print("BEST CONFIGURATION FOR EACH REPRESENTATION")
    print("=" * 120)

    representation_order = [
        "baseline_median",
        "contextual_wp_gpm_srtm",
        "clay_s1",
        "clay_s2",
        "clay_s1_s2",
        "contextual_plus_clay_s1",
        "contextual_plus_clay_s2",
        "contextual_plus_clay_s1_s2",
    ]

    best_rows = []

    for candidate in representation_order:

        sub = summary[
            summary["candidate"]
            == candidate
        ].copy()

        if len(sub) == 0:
            continue

        best = (
            sub
            .sort_values(
                "mean_repeat_log_rmse"
            )
            .iloc[0]
        )

        best_rows.append(
            best
        )

    best_df = pd.DataFrame(
        best_rows
    )

    print(
        best_df[
            display_cols
        ]
        .to_string(
            index=False
        )
    )

    BEST_FILE = (
        OUTPUT_DIR /
        "best_per_representation.csv"
    )

    best_df.to_csv(
        BEST_FILE,
        index=False,
    )

    print()
    print("=" * 120)
    print("DONE")
    print("=" * 120)

    print(
        "Common cohort:"
    )
    print(
        COMMON_EVENTS_FILE
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
        "Predictions:"
    )
    print(
        PREDICTIONS_FILE
    )

    print()
    print(
        "Repeat results:"
    )
    print(
        REPEAT_RESULTS_FILE
    )

    print()
    print(
        "Summary:"
    )
    print(
        SUMMARY_FILE
    )

    print()
    print(
        "Best per representation:"
    )
    print(
        BEST_FILE
    )


if __name__ == "__main__":
    main()
