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


# ============================================================
# PATHS
# ============================================================

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

SMUNET_S2_FILE = Path(
    "/home/melissa/flood_project/models_benchmark/"
    "UDL4FL/flodis_outputs/"
    "smunet_s2_event_features_valid_only/"
    "all_events_smunet_s2_event_features_valid_only.csv"
)

OUTPUT_DIR = Path(
    "/home/melissa/flood_project/models_benchmark/"
    "impact_prediction/outputs/"
    "s2_foundation_vs_smunet_common_displacement"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SUMMARY_FILE = OUTPUT_DIR / "comparison_summary.csv"
BEST_FILE = OUTPUT_DIR / "best_per_representation.csv"
REPEAT_FILE = OUTPUT_DIR / "repeat_results.csv"
PRED_FILE = OUTPUT_DIR / "cv_predictions.csv"


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
            f"Duplicate GFD_ID in {name}"
        )

    return df


def rmse(y_true, y_pred):

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


# ============================================================
# LOAD TABLES
# ============================================================

print("=" * 110)
print("S2 FOUNDATION MODELS vs FLOOD-SPECIFIC sm_unet_s2")
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

smunet_s2 = prepare_ids(
    pd.read_csv(SMUNET_S2_FILE),
    "sm_unet_s2"
)


# ============================================================
# KEEP VALID EVENTS
# ============================================================

for name, table in [
    ("Clay S1", clay_s1),
    ("Clay S2", clay_s2),
    ("DOFA S2", dofa_s2),
    ("sm_unet_s2", smunet_s2),
]:

    if "n_valid_patches" in table.columns:

        table.drop(
            table[
                table["n_valid_patches"] <= 0
            ].index,
            inplace=True
        )

        print(
            name,
            "valid events:",
            len(table)
        )


# ============================================================
# FEATURE COLUMNS
# ============================================================

# Clay S1
clay_s1_cols = [
    c for c in clay_s1.columns
    if (
        c.lower().startswith("clay")
        and not c.lower().startswith("clay_s2")
        and (
            "before" in c.lower()
            or "after" in c.lower()
            or "delta" in c.lower()
        )
        and pd.api.types.is_numeric_dtype(
            clay_s1[c]
        )
    )
]

# Clay S2
clay_s2_cols = [
    c for c in clay_s2.columns
    if (
        c.lower().startswith("clay_s2_")
        and (
            "before" in c.lower()
            or "after" in c.lower()
            or "delta" in c.lower()
        )
        and pd.api.types.is_numeric_dtype(
            clay_s2[c]
        )
    )
]

# DOFA S2
dofa_s2_cols = [
    c for c in dofa_s2.columns
    if (
        c.lower().startswith("dofa_s2_")
        and pd.api.types.is_numeric_dtype(
            dofa_s2[c]
        )
    )
]

# sm_unet_s2
smunet_s2_cols = [
    c for c in smunet_s2.columns
    if (
        c.lower().startswith("smunet_s2_")
        and pd.api.types.is_numeric_dtype(
            smunet_s2[c]
        )
    )
]


print()
print("Feature dimensions:")
print("Clay S1:", len(clay_s1_cols))
print("Clay S2:", len(clay_s2_cols))
print("DOFA S2:", len(dofa_s2_cols))
print("sm_unet_s2:", len(smunet_s2_cols))


for name, cols in [
    ("Clay S1", clay_s1_cols),
    ("Clay S2", clay_s2_cols),
    ("DOFA S2", dofa_s2_cols),
    ("sm_unet_s2", smunet_s2_cols),
]:

    if len(cols) == 0:
        raise RuntimeError(
            f"No deep features detected for {name}"
        )


# ============================================================
# RENAME BEFORE MERGE
# ============================================================

def make_small(
    df,
    cols,
    prefix,
):

    small = df[
        ["GFD_ID"] + cols
    ].copy()

    rename = {
        c: f"{prefix}__{c}"
        for c in cols
    }

    return (
        small.rename(columns=rename),
        [rename[c] for c in cols],
    )


clay_s1_small, clay_s1_final = make_small(
    clay_s1,
    clay_s1_cols,
    "S1CLAY"
)

clay_s2_small, clay_s2_final = make_small(
    clay_s2,
    clay_s2_cols,
    "S2CLAY"
)

dofa_s2_small, dofa_s2_final = make_small(
    dofa_s2,
    dofa_s2_cols,
    "S2DOFA"
)

smunet_s2_small, smunet_s2_final = make_small(
    smunet_s2,
    smunet_s2_cols,
    "S2SMUNET"
)


# ============================================================
# EXACT SAME COMMON COHORT
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
    .merge(
        smunet_s2_small,
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
    "Original common cohort:",
    len(cohort)
)

print(
    "Final common cohort:",
    len(df)
)

missing = sorted(
    set(cohort["GFD_ID"])
    - set(df["GFD_ID"])
)

if missing:
    print("Lost events:", missing)
else:
    print("No events lost.")


if TARGET not in df.columns:
    raise KeyError(
        f"{TARGET} missing"
    )


# ============================================================
# MATRICES
# ============================================================

def matrix(cols):

    return (
        df[cols]
        .apply(
            pd.to_numeric,
            errors="coerce"
        )
        .to_numpy(
            dtype=np.float32
        )
    )


X_s1 = matrix(
    clay_s1_final
)

X_clay_s2 = matrix(
    clay_s2_final
)

X_dofa_s2 = matrix(
    dofa_s2_final
)

X_smunet_s2 = matrix(
    smunet_s2_final
)


y_raw = pd.to_numeric(
    df[TARGET],
    errors="coerce"
).to_numpy(dtype=float)

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
print("Matrices:")
print("Clay S1:", X_s1.shape)
print("Clay S2:", X_clay_s2.shape)
print("DOFA S2:", X_dofa_s2.shape)
print("sm_unet_s2:", X_smunet_s2.shape)


# ============================================================
# SAME REPEATED CV SPLITS
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


# ============================================================
# RUN
# ============================================================

prediction_rows = []


for split_index, (
    train_idx,
    test_idx,
) in enumerate(splits):

    repeat_id = (
        split_index // N_SPLITS
    ) + 1

    fold_id = (
        split_index % N_SPLITS
    ) + 1

    print(
        f"Repeat {repeat_id}/{N_REPEATS}"
        f" | Fold {fold_id}/{N_SPLITS}"
    )

    y_train = y[
        train_idx
    ]

    y_test = y[
        test_idx
    ]


    # --------------------------------------------------------
    # BASELINE
    # --------------------------------------------------------

    baseline_pred = np.full(
        len(test_idx),
        np.median(y_train)
    )

    for idx, pred in zip(
        test_idx,
        baseline_pred
    ):

        prediction_rows.append({
            "repeat": repeat_id,
            "fold": fold_id,
            "candidate":
                "median_baseline",
            "model":
                "median_baseline",
            "pca_components":
                np.nan,
            "GFD_ID":
                int(gfd_ids[idx]),
            "y_true_log":
                float(y[idx]),
            "y_pred_log":
                float(pred),
        })


    # --------------------------------------------------------
    # PCA
    # --------------------------------------------------------

    for k in PCA_COMPONENTS:

        s1_train, s1_test = (
            preprocess_deep(
                X_s1[train_idx],
                X_s1[test_idx],
                k
            )
        )

        clay_train, clay_test = (
            preprocess_deep(
                X_clay_s2[train_idx],
                X_clay_s2[test_idx],
                k
            )
        )

        dofa_train, dofa_test = (
            preprocess_deep(
                X_dofa_s2[train_idx],
                X_dofa_s2[test_idx],
                k
            )
        )

        smunet_train, smunet_test = (
            preprocess_deep(
                X_smunet_s2[train_idx],
                X_smunet_s2[test_idx],
                k
            )
        )


        candidates = {

            "clay_s1": (
                s1_train,
                s1_test
            ),

            "clay_s2": (
                clay_train,
                clay_test
            ),

            "dofa_s2": (
                dofa_train,
                dofa_test
            ),

            "smunet_s2": (
                smunet_train,
                smunet_test
            ),

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

            "clay_s1_plus_smunet_s2": (
                np.hstack([
                    s1_train,
                    smunet_train
                ]),
                np.hstack([
                    s1_test,
                    smunet_test
                ])
            ),
        }


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

                for idx, value in zip(
                    test_idx,
                    pred
                ):

                    prediction_rows.append({
                        "repeat": repeat_id,
                        "fold": fold_id,
                        "candidate":
                            candidate_name,
                        "model":
                            model_name,
                        "pca_components":
                            k,
                        "GFD_ID":
                            int(gfd_ids[idx]),
                        "y_true_log":
                            float(y[idx]),
                        "y_pred_log":
                            float(value),
                    })


# ============================================================
# SAVE PREDICTIONS
# ============================================================

pred_df = pd.DataFrame(
    prediction_rows
)

pred_df.to_csv(
    PRED_FILE,
    index=False
)


# ============================================================
# REPEAT METRICS
# ============================================================

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

    repeat_id, candidate, model_name, pca_k = keys

    yt = group[
        "y_true_log"
    ].to_numpy()

    yp = group[
        "y_pred_log"
    ].to_numpy()

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
                yt,
                yp
            ),

        "log_r2":
            safe_r2(
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

summary = (
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
# BEST PER REPRESENTATION
# ============================================================

order = [
    "median_baseline",
    "clay_s1",
    "clay_s2",
    "dofa_s2",
    "smunet_s2",
    "clay_s1_plus_clay_s2",
    "clay_s1_plus_dofa_s2",
    "clay_s1_plus_smunet_s2",
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
        ]
    ].to_string(
        index=False
    )
)

print()
print("Saved:")
print(BEST_FILE)
