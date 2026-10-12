from pathlib import Path
import re
import numpy as np
import pandas as pd


# =====================================================
# PATHS
# =====================================================
INPUT_DIR = Path("/home/melissa/flood_project/data/S1_GEE_features")

OUTPUT_DIR = Path("/home/melissa/flood_project/models_benchmark/impact_prediction/outputs")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

RAW_COMBINED_OUT = OUTPUT_DIR / "s1_gee_raw_combined.csv"
EVENT_FEATURES_OUT = OUTPUT_DIR / "s1_gee_event_features_by_GFD_ID.csv"


# =====================================================
# SETTINGS
# =====================================================
# Anything related to labels/targets must be removed from input features.
# This avoids leakage from columns like:
# displacements, no_affected_EMDAT, total_deaths, damages, etc.
TARGET_OR_LEAK_KEYWORDS = [
    "displacement",
    "displacements",
    "death",
    "deaths",
    "mortality",
    "affected",
    "emdat",
    "damage",
    "damages",
    "idmc",
]

META_COLS = {
    "row_id",
    "GFD_ID",
    "ISO3",
    "iso3",
    "year",
    "gfd_event_date",
    "during_start",
    "during_end",
    "source_file",
}


# =====================================================
# UTILS
# =====================================================
def parse_gfd_id(x):
    """
    Robustly parse GFD_ID.
    Works if GFD_ID is 4211 or if it appears inside a string.
    """
    if pd.isna(x):
        return np.nan

    s = str(x).strip()

    try:
        return int(float(s))
    except Exception:
        pass

    m = re.search(r"\d+", s)
    if m:
        return int(m.group(0))

    return np.nan


def clean_column_names(df):
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    return df


def is_leak_column(col):
    """
    Detect target/leakage columns by keyword.
    GFD_ID is allowed even though it contains ID-like text.
    """
    if col == "GFD_ID":
        return False

    c = str(col).lower()

    return any(k in c for k in TARGET_OR_LEAK_KEYWORDS)


# =====================================================
# READ ALL FILES
# =====================================================
def read_all_s1_files():
    files = sorted(INPUT_DIR.rglob("S1_phase1_*.csv"))

    print("=" * 100)
    print("Searching in:", INPUT_DIR)
    print("Number of S1 GEE files found:", len(files))

    if len(files) == 0:
        raise FileNotFoundError(f"No S1_phase1_*.csv files found under {INPUT_DIR}")

    dfs = []

    for f in files:
        print("Reading:", f)

        df = pd.read_csv(f)
        df = clean_column_names(df)

        df["source_file"] = f.name

        if "GFD_ID" not in df.columns:
            raise ValueError(f"Missing GFD_ID column in {f}")

        df["GFD_ID"] = df["GFD_ID"].apply(parse_gfd_id)
        df = df[df["GFD_ID"].notna()].copy()
        df["GFD_ID"] = df["GFD_ID"].astype(int)

        dfs.append(df)

    raw = pd.concat(dfs, ignore_index=True)

    print()
    print("Raw combined shape:", raw.shape)
    print("Unique GFD_ID:", raw["GFD_ID"].nunique())
    print("Columns:")
    print(raw.columns.tolist())

    raw.to_csv(RAW_COMBINED_OUT, index=False)
    print()
    print("Saved raw combined:", RAW_COMBINED_OUT)

    return raw


# =====================================================
# BUILD EVENT-LEVEL FEATURES
# =====================================================
def build_event_features(raw):
    raw = raw.copy()

    # -------------------------------------------------
    # Drop target/leakage columns
    # -------------------------------------------------
    leak_cols_present = [
        c for c in raw.columns
        if c != "GFD_ID" and is_leak_column(c)
    ]

    if leak_cols_present:
        print()
        print("Dropping target/leak columns:")
        for c in leak_cols_present:
            print(" ", c)

        raw = raw.drop(columns=leak_cols_present)

    # -------------------------------------------------
    # Select numeric S1 GEE features
    # -------------------------------------------------
    numeric_cols = [
        c for c in raw.columns
        if pd.api.types.is_numeric_dtype(raw[c])
        and c not in META_COLS
        and not is_leak_column(c)
    ]

    print()
    print("Numeric S1 GEE feature columns kept:")
    for c in numeric_cols:
        print(" ", c)

    if len(numeric_cols) == 0:
        raise ValueError("No numeric S1 GEE feature columns found after leakage removal.")

    # -------------------------------------------------
    # Metadata: one row per GFD_ID
    # -------------------------------------------------
    event_meta = raw.groupby("GFD_ID").agg(
        s1gee_n_rows=("GFD_ID", "size"),
    ).reset_index()

    if "ISO3" in raw.columns:
        iso_meta = (
            raw.groupby("GFD_ID")["ISO3"]
            .apply(lambda x: ",".join(sorted(set(str(v) for v in x.dropna()))))
            .reset_index(name="s1gee_iso3_list")
        )

        iso_count = (
            raw.groupby("GFD_ID")["ISO3"]
            .apply(lambda x: len(set(str(v) for v in x.dropna())))
            .reset_index(name="s1gee_n_iso3")
        )

        event_meta = event_meta.merge(iso_meta, on="GFD_ID", how="left")
        event_meta = event_meta.merge(iso_count, on="GFD_ID", how="left")

    if "year" in raw.columns:
        year_meta = (
            raw.groupby("GFD_ID")["year"]
            .agg(lambda x: x.dropna().iloc[0] if len(x.dropna()) > 0 else np.nan)
            .reset_index(name="s1gee_year")
        )
        event_meta = event_meta.merge(year_meta, on="GFD_ID", how="left")

    if "gfd_event_date" in raw.columns:
        date_meta = (
            raw.groupby("GFD_ID")["gfd_event_date"]
            .agg(lambda x: x.dropna().iloc[0] if len(x.dropna()) > 0 else np.nan)
            .reset_index(name="s1gee_event_date")
        )
        event_meta = event_meta.merge(date_meta, on="GFD_ID", how="left")

    # -------------------------------------------------
    # Aggregate numeric features to event level
    # -------------------------------------------------
    agg_dict = {}

    for c in numeric_cols:
        agg_dict[c] = ["mean", "std", "min", "max"]

    features = raw.groupby("GFD_ID").agg(agg_dict)

    features.columns = [
        f"s1gee_{col}_{stat}"
        for col, stat in features.columns
    ]

    features = features.reset_index()

    # Events with only one row have NaN std; set to 0.
    std_cols = [c for c in features.columns if c.endswith("_std")]
    features[std_cols] = features[std_cols].fillna(0.0)

    out = event_meta.merge(features, on="GFD_ID", how="left")

    # -------------------------------------------------
    # Final leakage check
    # -------------------------------------------------
    bad_cols = [
        c for c in out.columns
        if c != "GFD_ID" and is_leak_column(c)
    ]

    if bad_cols:
        print()
        print("ERROR: Leakage columns still present:")
        for c in bad_cols:
            print(" ", c)
        raise ValueError("Target leakage detected in output table.")

    out = out.sort_values("GFD_ID").reset_index(drop=True)

    out.to_csv(EVENT_FEATURES_OUT, index=False)

    print()
    print("=" * 100)
    print("Final S1 GEE event feature table")
    print("Shape:", out.shape)
    print("Unique GFD_ID:", out["GFD_ID"].nunique())
    print("Saved:", EVENT_FEATURES_OUT)
    print()
    print(out.head())

    return out


# =====================================================
# MAIN
# =====================================================
def main():
    raw = read_all_s1_files()
    event_features = build_event_features(raw)

    print()
    print("=" * 100)
    print("DONE")
    print("Output file:")
    print(EVENT_FEATURES_OUT)


if __name__ == "__main__":
    main()