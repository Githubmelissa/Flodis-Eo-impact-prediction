from pathlib import Path
import re
import numpy as np
import pandas as pd
from pandas.errors import EmptyDataError


# =====================================================
# PATHS
# =====================================================
SNUNET_PATCH_DIR = Path(
    "/home/melissa/flood_project/models_benchmark/KuroSiwo/flodis_outputs/snunet_event_features_valid_only"
)

WEIGHTS_PATH = Path(
    "/home/melissa/flood_project/models_benchmark/impact_prediction/outputs/s1_patch_flood_weights.csv"
)

OUTPUT_DIR = Path(
    "/home/melissa/flood_project/models_benchmark/impact_prediction/outputs"
)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUT_PATCH_TABLE = OUTPUT_DIR / "snunet_patch_deep_features_with_flood_weights.csv"
OUT_EVENT_TABLE = OUTPUT_DIR / "snunet_flood_weighted_event_deep_features.csv"


EPS = 1e-8


def parse_event_id_from_name(name):
    m = re.search(r"event[_-]?(\d+)", str(name))
    if m:
        return int(m.group(1))
    return None


def normalize_patch_id(x):
    if pd.isna(x):
        return None

    s = str(x).strip()

    m = re.search(r"patch[_-]?(\d+)", s)
    if m:
        return f"patch_{int(m.group(1)):03d}"

    m = re.search(r"\d+", s)
    if m:
        return f"patch_{int(m.group(0)):03d}"

    return s


def weighted_mean(values, weights):
    values = np.asarray(values, dtype=float)
    weights = np.asarray(weights, dtype=float)

    mask = np.isfinite(values) & np.isfinite(weights)

    if mask.sum() == 0:
        return np.nan

    values = values[mask]
    weights = weights[mask]

    wsum = weights.sum()

    if wsum <= EPS:
        return float(np.nanmean(values))

    return float(np.sum(values * weights) / wsum)


def topk_flood_mean(g, col, k=5):
    tmp = g.sort_values("flood_fraction", ascending=False).head(k)

    if len(tmp) == 0:
        return np.nan

    return float(np.nanmean(tmp[col].values.astype(float)))


def read_all_snunet_patch_features():
    files = sorted(SNUNET_PATCH_DIR.glob("event_*_snunet_patch_features.csv"))

    print("=" * 100)
    print("SNUNet patch feature dir:", SNUNET_PATCH_DIR)
    print("Number of files:", len(files))

    if len(files) == 0:
        raise FileNotFoundError("No SNUNet patch feature files found.")

    dfs = []

    for f in files:
        event_id = parse_event_id_from_name(f.name)

        try:
            df = pd.read_csv(f)
        except EmptyDataError:
            print("SKIP empty file:", f)
            continue
        except Exception as e:
            print("SKIP unreadable file:", f, repr(e))
            continue

        if df.empty:
            print("SKIP empty dataframe:", f)
            continue

        df.columns = [str(c).strip() for c in df.columns]

        if "GFD_ID" not in df.columns:
            df["GFD_ID"] = event_id

        df["GFD_ID"] = df["GFD_ID"].fillna(event_id).astype(int)

        if "patch_id" not in df.columns:
            raise ValueError(f"patch_id column missing in {f}")

        df["patch_id"] = df["patch_id"].apply(normalize_patch_id)
        df["source_snunet_file"] = f.name

        dfs.append(df)

    out = pd.concat(dfs, ignore_index=True)

    print("Combined patch table shape:", out.shape)
    print("Unique events:", out["GFD_ID"].nunique())
    print("Total patch rows:", len(out))

    return out


def merge_with_flood_weights(patch_df):
    weights = pd.read_csv(WEIGHTS_PATH)
    weights.columns = [str(c).strip() for c in weights.columns]

    weights["GFD_ID"] = weights["GFD_ID"].astype(int)
    weights["patch_id"] = weights["patch_id"].apply(normalize_patch_id)

    keep_cols = [
        "GFD_ID",
        "patch_id",
        "valid_fraction",
        "flood_fraction",
        "flood_weight",
        "is_valid_patch",
        "is_flooded_patch",
    ]

    weights = weights[[c for c in keep_cols if c in weights.columns]]

    merged = patch_df.merge(
        weights,
        on=["GFD_ID", "patch_id"],
        how="left",
    )

    merged["valid_fraction"] = merged["valid_fraction"].fillna(1.0)
    merged["flood_fraction"] = merged["flood_fraction"].fillna(0.0)
    merged["flood_weight"] = merged["flood_weight"].fillna(0.0)

    merged.to_csv(OUT_PATCH_TABLE, index=False)

    print()
    print("=" * 100)
    print("Merged patch table saved:", OUT_PATCH_TABLE)
    print("Merged shape:", merged.shape)
    print("Mean flood fraction:", merged["flood_fraction"].mean())
    print("Max flood fraction:", merged["flood_fraction"].max())
    print("Rows with flood weight > 0:", int((merged["flood_weight"] > 0).sum()))

    return merged


def aggregate_to_event(merged):
    deep_cols = [
        c for c in merged.columns
        if c.startswith("snunet_deep_")
        and pd.api.types.is_numeric_dtype(merged[c])
    ]

    print()
    print("=" * 100)
    print("Number of SNUNet deep columns:", len(deep_cols))
    print(deep_cols[:10])

    if len(deep_cols) == 0:
        raise ValueError("No snunet_deep_* columns found.")

    rows = []

    for gfd_id, g in merged.groupby("GFD_ID"):
        row = {
            "GFD_ID": int(gfd_id),
            "snunet_fw_n_patches": len(g),
            "snunet_fw_mean_flood_fraction": float(g["flood_fraction"].mean()),
            "snunet_fw_max_flood_fraction": float(g["flood_fraction"].max()),
            "snunet_fw_sum_flood_weight": float(g["flood_weight"].sum()),
            "snunet_fw_n_weighted_patches": int((g["flood_weight"] > 0).sum()),
        }

        w_flood = g["flood_weight"].values.astype(float)
        w_valid = g["valid_fraction"].values.astype(float)

        if np.nansum(w_flood) <= EPS:
            weights_used = w_valid
            row["snunet_fw_weight_mode"] = "valid_fraction_fallback"
        else:
            weights_used = w_flood
            row["snunet_fw_weight_mode"] = "flood_weight"

        for col in deep_cols:
            vals = g[col].values.astype(float)

            row[f"snunet_deep_unweighted_mean_{col[-3:]}"] = float(np.nanmean(vals))
            row[f"snunet_deep_flood_weighted_mean_{col[-3:]}"] = weighted_mean(vals, weights_used)
            row[f"snunet_deep_top5_flood_mean_{col[-3:]}"] = topk_flood_mean(g, col, k=5)

        rows.append(row)

    out = pd.DataFrame(rows).sort_values("GFD_ID").reset_index(drop=True)
    out.to_csv(OUT_EVENT_TABLE, index=False)

    print()
    print("=" * 100)
    print("DONE")
    print("Saved event table:", OUT_EVENT_TABLE)
    print("Shape:", out.shape)
    print("Unique events:", out["GFD_ID"].nunique())
    print(out.head())

    return out


def main():
    patch_df = read_all_snunet_patch_features()
    merged = merge_with_flood_weights(patch_df)
    aggregate_to_event(merged)


if __name__ == "__main__":
    main()