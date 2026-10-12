from pathlib import Path
import re
import numpy as np
import pandas as pd
import rasterio


# =====================================================
# PATHS
# =====================================================
S1_PATCH_ROOT = Path("/home/melissa/flood_project/S1_PATCHES_CLEAN")

OUTPUT_DIR = Path("/home/melissa/flood_project/models_benchmark/impact_prediction/outputs")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUT_CSV = OUTPUT_DIR / "s1_patch_flood_weights.csv"


# =====================================================
# SETTINGS
# =====================================================
MIN_VALID_FRACTION = 0.80
FLOOD_THRESHOLD = 0.5


def parse_event_id(event_folder_name):
    """
    event_4294 -> 4294
    """
    m = re.search(r"\d+", event_folder_name)
    if m:
        return int(m.group(0))
    return None


def compute_patch_stats(stack_path):
    """
    S1_stack.tif bands:
    1 before_VV
    2 before_VH
    3 after_VV
    4 after_VH
    5 delta_VV
    6 delta_VH
    7 flood_mask
    """

    with rasterio.open(stack_path) as src:
        arr = src.read()

    if arr.shape[0] < 7:
        raise ValueError(f"Expected at least 7 bands, got {arr.shape[0]} for {stack_path}")

    s1_bands = arr[:4].astype("float32")
    flood_mask = arr[6].astype("float32")

    # valid pixels: finite in before/after VV/VH
    valid = np.all(np.isfinite(s1_bands), axis=0)

    n_pixels = valid.size
    n_valid = int(valid.sum())
    valid_fraction = n_valid / n_pixels if n_pixels > 0 else 0.0

    if n_valid == 0:
        flood_fraction = 0.0
        flood_pixel_count = 0
    else:
        flood_valid = flood_mask[valid]
        flood_valid = flood_valid[np.isfinite(flood_valid)]

        if len(flood_valid) == 0:
            flood_fraction = 0.0
            flood_pixel_count = 0
        else:
            flooded = flood_valid > FLOOD_THRESHOLD
            flood_pixel_count = int(flooded.sum())
            flood_fraction = float(flooded.mean())

    # simple weight for later aggregation
    # patches with poor validity or no flood get low weight
    flood_weight = flood_fraction * valid_fraction

    return {
        "n_pixels": n_pixels,
        "n_valid_pixels": n_valid,
        "valid_fraction": valid_fraction,
        "flood_pixel_count": flood_pixel_count,
        "flood_fraction": flood_fraction,
        "flood_weight": flood_weight,
        "is_valid_patch": valid_fraction >= MIN_VALID_FRACTION,
        "is_flooded_patch": flood_fraction > 0.0,
    }


def main():
    rows = []

    event_dirs = sorted(S1_PATCH_ROOT.glob("event_*"))

    print("=" * 100)
    print("S1 patch root:", S1_PATCH_ROOT)
    print("Number of event folders:", len(event_dirs))

    for event_dir in event_dirs:
        event_id = parse_event_id(event_dir.name)

        if event_id is None:
            print("Skipping folder with no event id:", event_dir)
            continue

        patch_dirs = sorted(event_dir.glob("patch_*"))

        for patch_dir in patch_dirs:
            stack_path = patch_dir / "S1_stack.tif"

            if not stack_path.exists():
                continue

            try:
                stats = compute_patch_stats(stack_path)

                rows.append({
                    "GFD_ID": event_id,
                    "event_id": event_id,
                    "event_folder": event_dir.name,
                    "patch_id": patch_dir.name,
                    "stack_path": str(stack_path),
                    **stats,
                })

            except Exception as e:
                rows.append({
                    "GFD_ID": event_id,
                    "event_id": event_id,
                    "event_folder": event_dir.name,
                    "patch_id": patch_dir.name,
                    "stack_path": str(stack_path),
                    "error": repr(e),
                })

                print("ERROR:", stack_path, repr(e))

    df = pd.DataFrame(rows)

    df.to_csv(OUT_CSV, index=False)

    print()
    print("=" * 100)
    print("DONE")
    print("Saved:", OUT_CSV)
    print("Shape:", df.shape)

    if len(df) > 0:
        print("Unique events:", df["GFD_ID"].nunique())
        print("Total patches:", len(df))

        if "is_valid_patch" in df.columns:
            print("Valid patches:", int(df["is_valid_patch"].sum()))

        if "is_flooded_patch" in df.columns:
            print("Flooded patches:", int(df["is_flooded_patch"].sum()))

        print()
        print("Example rows:")
        print(df.head(20))

        print()
        print("Event summary:")
        summary = (
            df.groupby("GFD_ID")
            .agg(
                n_patches=("patch_id", "count"),
                mean_valid_fraction=("valid_fraction", "mean"),
                mean_flood_fraction=("flood_fraction", "mean"),
                max_flood_fraction=("flood_fraction", "max"),
                sum_flood_weight=("flood_weight", "sum"),
                n_flooded_patches=("is_flooded_patch", "sum"),
                n_valid_patches=("is_valid_patch", "sum"),
            )
            .reset_index()
            .sort_values("GFD_ID")
        )

        print(summary.head(30))


if __name__ == "__main__":
    main()