from pathlib import Path
import warnings

import numpy as np
import pandas as pd
import rasterio
import torch
import torch.nn.functional as F
from tqdm import tqdm

from udl4fl.model_setup import load_model_by_name

warnings.filterwarnings("ignore")


# ============================================================
# PATHS
# ============================================================

PATCH_MANIFEST = Path(
    "/home/melissa/flood_project/data/"
    "s2_patch_inventory.csv"
)

WEIGHTS_FOLDER = (
    "/home/melissa/flood_project/"
    "models_benchmark/UDL4FL/cloudsen12_models"
)

OUTPUT_DIR = Path(
    "/home/melissa/flood_project/"
    "models_benchmark/UDL4FL/flodis_outputs/"
    "smunet_s2_event_features_valid_only"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

OUTPUT_CSV = (
    OUTPUT_DIR /
    "all_events_smunet_s2_event_features_valid_only.csv"
)


# ============================================================
# SETTINGS
# ============================================================

DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

IMAGE_SIZE = 256

MIN_VALID_FRACTION = 0.80

BATCH_SIZE = 8


# ============================================================
# EXACT TRAINING NORMALIZATION FROM sm_unet_s2.json
#
# Order:
# B2, B3, B4, B8, B11, B12
# ============================================================

MEAN = torch.tensor(
    [
        3758.07470703125,
        3238.08251953125,
        3418.9013671875,
        3981.962646484375,
        2391.6611328125,
        1790.324951171875,
    ],
    dtype=torch.float32,
).view(1, 6, 1, 1)


STD = torch.tensor(
    [
        2794.095703125,
        2549.494140625,
        2811.781005859375,
        2500.478759765625,
        1500.025390625,
        1241.9818115234375,
    ],
    dtype=torch.float32,
).view(1, 6, 1, 1)


# ============================================================
# HELPERS
# ============================================================

def fill_invalid_per_band(
    arr,
    valid_mask,
):

    arr = arr.astype(
        np.float32
    ).copy()

    valid_mask = (
        valid_mask > 0
    )

    for band_idx in range(
        arr.shape[0]
    ):

        band = arr[
            band_idx
        ]

        valid = (
            valid_mask
            &
            np.isfinite(band)
        )

        if valid.sum() == 0:

            band[:] = 0.0

        else:

            median_value = np.median(
                band[valid]
            )

            bad = (
                (~valid_mask)
                |
                (~np.isfinite(band))
            )

            band[bad] = median_value

        arr[
            band_idx
        ] = band

    return arr


def resize_image(
    image,
):

    x = torch.from_numpy(
        image.astype(
            np.float32
        )
    ).unsqueeze(0)

    x = F.interpolate(
        x,
        size=(
            IMAGE_SIZE,
            IMAGE_SIZE,
        ),
        mode="bilinear",
        align_corners=False,
    )

    return x.squeeze(0)


def read_before_after(
    path,
):

    with rasterio.open(
        path
    ) as src:

        if src.count != 15:

            raise ValueError(
                f"Expected 15 bands, "
                f"found {src.count}: {path}"
            )

        before = src.read(
            [1, 2, 3, 4, 5, 6]
        ).astype(
            np.float32
        )

        after = src.read(
            [7, 8, 9, 10, 11, 12]
        ).astype(
            np.float32
        )

        before_valid = src.read(
            13
        )

        after_valid = src.read(
            14
        )

    before = fill_invalid_per_band(
        before,
        before_valid,
    )

    after = fill_invalid_per_band(
        after,
        after_valid,
    )

    before = resize_image(
        before
    )

    after = resize_image(
        after
    )

    return (
        before,
        after,
    )


# ============================================================
# MANIFEST
# ============================================================

def load_manifest():

    df = pd.read_csv(
        PATCH_MANIFEST
    )

    if df["read_ok"].dtype != bool:

        df["read_ok"] = (
            df["read_ok"]
            .astype(str)
            .str.lower()
            .eq("true")
        )

    if (
        df["spectral_all_zero"].dtype
        != bool
    ):

        df["spectral_all_zero"] = (
            df["spectral_all_zero"]
            .astype(str)
            .str.lower()
            .eq("true")
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

    df["usable"] = (
        df["read_ok"]
        &
        (~df["spectral_all_zero"])
        &
        (
            df[
                "common_valid_fraction"
            ]
            >= MIN_VALID_FRACTION
        )
    )

    return df


# ============================================================
# MODEL
# ============================================================

def load_model():

    print(
        "Loading pretrained sm_unet_s2..."
    )

    model = load_model_by_name(
        "sm_unet_s2",
        weights_folder=
            WEIGHTS_FOLDER,
        device=DEVICE,
    )

    model.eval()

    return model


# ============================================================
# FEATURE EXTRACTION
# ============================================================

def extract_embeddings(
    model,
    batch,
):

    batch = batch.to(
        DEVICE
    )

    mean = MEAN.to(
        DEVICE
    )

    std = STD.to(
        DEVICE
    )

    # Normalize using exact checkpoint statistics
    batch = (
        batch - mean
    ) / std

    with torch.no_grad():

        feature_maps = (
            model.network.encoder(
                batch
            )
        )

        deepest = (
            feature_maps[-1]
        )

        embeddings = (
            F.adaptive_avg_pool2d(
                deepest,
                output_size=1,
            )
            .flatten(1)
        )

    return (
        embeddings
        .detach()
        .cpu()
        .numpy()
        .astype(np.float32)
    )


# ============================================================
# EVENT AGGREGATION
# ============================================================

def aggregate_embeddings(
    prefix,
    embeddings,
):

    output = {}

    stats = {
        "mean":
            embeddings.mean(
                axis=0
            ),

        "std":
            embeddings.std(
                axis=0
            ),

        "min":
            embeddings.min(
                axis=0
            ),

        "max":
            embeddings.max(
                axis=0
            ),
    }

    dim = embeddings.shape[1]

    for stat_name, values in (
        stats.items()
    ):

        for i in range(dim):

            output[
                f"smunet_s2_{prefix}_"
                f"{i:04d}_event_"
                f"{stat_name}"
            ] = float(
                values[i]
            )

    return output


# ============================================================
# PROCESS EVENT
# ============================================================

def process_event(
    gfd_id,
    event_df,
    model,
):

    total_patches = len(
        event_df
    )

    valid_df = (
        event_df[
            event_df["usable"]
        ]
        .copy()
    )

    valid_paths = [
        Path(p)
        for p in valid_df[
            "path"
        ].tolist()
    ]

    row = {
        "GFD_ID":
            int(gfd_id),

        "event_id":
            f"event_{int(gfd_id)}",

        "n_total_patches":
            total_patches,

        "n_valid_patches":
            len(valid_paths),

        "n_skipped_patches":
            total_patches
            - len(valid_paths),

        "valid_patch_fraction":
            (
                len(valid_paths)
                / total_patches
                if total_patches > 0
                else 0.0
            ),
    }

    if len(valid_paths) == 0:

        print(
            f"event_{gfd_id}: "
            "zero valid patches"
        )

        return row


    before_all = []
    after_all = []

    extracted_count = 0


    for start in range(
        0,
        len(valid_paths),
        BATCH_SIZE,
    ):

        batch_paths = (
            valid_paths[
                start:
                start + BATCH_SIZE
            ]
        )

        before_list = []
        after_list = []

        for path in batch_paths:

            try:

                before, after = (
                    read_before_after(
                        path
                    )
                )

                before_list.append(
                    before
                )

                after_list.append(
                    after
                )

            except Exception as e:

                print(
                    "Skipping:",
                    path,
                    repr(e),
                )


        if len(before_list) == 0:
            continue


        before_batch = torch.stack(
            before_list,
            dim=0,
        )

        after_batch = torch.stack(
            after_list,
            dim=0,
        )


        before_embeddings = (
            extract_embeddings(
                model,
                before_batch,
            )
        )

        after_embeddings = (
            extract_embeddings(
                model,
                after_batch,
            )
        )


        before_all.append(
            before_embeddings
        )

        after_all.append(
            after_embeddings
        )

        extracted_count += (
            before_embeddings.shape[0]
        )


    if len(before_all) == 0:

        print(
            f"event_{gfd_id}: "
            "no embeddings extracted"
        )

        row[
            "n_extracted_patches"
        ] = 0

        return row


    before_all = np.concatenate(
        before_all,
        axis=0,
    )

    after_all = np.concatenate(
        after_all,
        axis=0,
    )


    delta_all = (
        after_all
        - before_all
    )


    row[
        "n_extracted_patches"
    ] = extracted_count


    row.update(
        aggregate_embeddings(
            "before",
            before_all,
        )
    )

    row.update(
        aggregate_embeddings(
            "after",
            after_all,
        )
    )

    row.update(
        aggregate_embeddings(
            "delta",
            delta_all,
        )
    )


    print(
        f"event_{gfd_id}",
        "valid:",
        len(valid_paths),
        "extracted:",
        extracted_count,
        "embedding:",
        before_all.shape,
    )

    return row


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 100)

    print(
        "sm_unet_s2 SENTINEL-2 "
        "ALL-EVENT FEATURE EXTRACTION"
    )

    print("=" * 100)

    print(
        "Device:",
        DEVICE
    )

    print(
        "Input size:",
        IMAGE_SIZE
    )

    print(
        "Minimum valid fraction:",
        MIN_VALID_FRACTION
    )

    print(
        "Batch size:",
        BATCH_SIZE
    )


    manifest = (
        load_manifest()
    )


    print()
    print(
        "Total patches:",
        len(manifest)
    )

    print(
        "Usable patches:",
        int(
            manifest[
                "usable"
            ].sum()
        )
    )

    print(
        "Events:",
        manifest[
            "GFD_ID"
        ].nunique()
    )

    print(
        "Events with valid patches:",
        manifest.loc[
            manifest["usable"],
            "GFD_ID"
        ].nunique()
    )


    model = load_model()


    event_ids = sorted(
        manifest[
            "GFD_ID"
        ].unique()
    )


    rows = []


    for gfd_id in tqdm(
        event_ids,
        desc="sm_unet_s2 events",
    ):

        event_df = (
            manifest[
                manifest["GFD_ID"]
                == gfd_id
            ]
        )


        row = process_event(
            gfd_id,
            event_df,
            model,
        )


        rows.append(
            row
        )


        # Save progressively
        pd.DataFrame(
            rows
        ).sort_values(
            "GFD_ID"
        ).to_csv(
            OUTPUT_CSV,
            index=False,
        )


    result = (
        pd.DataFrame(
            rows
        )
        .sort_values(
            "GFD_ID"
        )
        .reset_index(
            drop=True
        )
    )


    result.to_csv(
        OUTPUT_CSV,
        index=False,
    )


    deep_cols = [
        c
        for c in result.columns
        if c.startswith(
            "smunet_s2_"
        )
    ]


    embedded = (
        result[
            "n_extracted_patches"
        ]
        .fillna(0)
        > 0
    )


    print()
    print("=" * 100)
    print("DONE")
    print("=" * 100)

    print(
        "Saved:",
        OUTPUT_CSV
    )

    print(
        "Shape:",
        result.shape
    )

    print(
        "Events:",
        len(result)
    )

    print(
        "Events with embeddings:",
        int(
            embedded.sum()
        )
    )

    print(
        "Deep feature columns:",
        len(deep_cols)
    )


if __name__ == "__main__":
    main()
