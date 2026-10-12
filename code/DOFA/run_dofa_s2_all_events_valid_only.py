from pathlib import Path
import os
import warnings

# =====================================================
# CACHE
# =====================================================
os.environ.setdefault(
    "TORCH_HOME",
    "/dev/shm/melissa/cache/torch"
)
os.environ.setdefault(
    "HF_HOME",
    "/dev/shm/melissa/cache/huggingface"
)
os.environ.setdefault(
    "TMPDIR",
    "/dev/shm/melissa/tmp"
)

for p in [
    os.environ["TORCH_HOME"],
    os.environ["HF_HOME"],
    os.environ["TMPDIR"],
]:
    os.makedirs(p, exist_ok=True)

import numpy as np
import pandas as pd
import rasterio
import torch
import torch.nn.functional as F
from tqdm import tqdm

warnings.filterwarnings("ignore")


# =====================================================
# PATHS
# =====================================================

PATCH_MANIFEST = Path(
    "/home/melissa/flood_project/data/"
    "s2_patch_inventory.csv"
)

OUTPUT_DIR = Path(
    "/home/melissa/flood_project/"
    "models_benchmark/DOFA/flodis_outputs/"
    "dofa_s2_event_features_valid_only"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

OUTPUT_CSV = (
    OUTPUT_DIR /
    "all_events_dofa_s2_event_features_valid_only.csv"
)


# =====================================================
# SETTINGS
# =====================================================

DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

IMAGE_SIZE = 224

MIN_VALID_FRACTION = 0.80

BATCH_SIZE = 8


# =====================================================
# SENTINEL-2 BANDS
# =====================================================

BAND_NAMES = [
    "B2",
    "B3",
    "B4",
    "B8",
    "B11",
    "B12",
]

WAVELENGTHS = [
    0.490,
    0.560,
    0.665,
    0.842,
    1.610,
    2.190,
]


# =====================================================
# DOFA SENTINEL-2 NORMALIZATION
# =====================================================

S2_MEAN = torch.tensor(
    [
        126.63977424,  # B2 blue
        114.81779093,  # B3 green
        114.10997390,  # B4 red
        101.43563300,  # B8 NIR
        72.32804172,   # B11 SWIR1
        56.66528851,   # B12 SWIR2
    ],
    dtype=torch.float32,
).view(6, 1, 1)

S2_STD = torch.tensor(
    [
        67.42465279,
        69.96844919,
        77.84352553,
        60.29744676,
        47.88519516,
        42.55886798,
    ],
    dtype=torch.float32,
).view(6, 1, 1)


# =====================================================
# DATA HELPERS
# =====================================================

def fill_invalid_per_band(
    arr,
    valid_mask,
):

    arr = arr.astype(
        "float32"
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

            median_value = (
                np.median(
                    band[valid]
                )
            )

            bad = (
                (~valid_mask)
                |
                (~np.isfinite(band))
            )

            band[bad] = (
                median_value
            )

        arr[
            band_idx
        ] = band

    return arr


def preprocess_image(
    image,
):

    # FLODIS S2:
    # reflectance stored as 0...10000
    image = np.clip(
        image,
        0.0,
        10000.0,
    )

    # Match DOFA S2 scale:
    # 0...10000 -> 0...255
    image = (
        image
        / 10000.0
        * 255.0
    )

    image = torch.from_numpy(
        image.astype(
            np.float32
        )
    )

    image = (
        F.interpolate(
            image.unsqueeze(0),
            size=(
                IMAGE_SIZE,
                IMAGE_SIZE,
            ),
            mode="bilinear",
            align_corners=False,
        )
        .squeeze(0)
    )

    image = (
        image - S2_MEAN
    ) / S2_STD

    return image


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
            "float32"
        )

        after = src.read(
            [7, 8, 9, 10, 11, 12]
        ).astype(
            "float32"
        )

        before_valid = (
            src.read(13)
        )

        after_valid = (
            src.read(14)
        )

    before = fill_invalid_per_band(
        before,
        before_valid,
    )

    after = fill_invalid_per_band(
        after,
        after_valid,
    )

    before = preprocess_image(
        before
    )

    after = preprocess_image(
        after
    )

    return (
        before,
        after,
    )


# =====================================================
# MANIFEST
# =====================================================

def load_manifest():

    df = pd.read_csv(
        PATCH_MANIFEST
    )

    # Ensure booleans
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

        df[
            "spectral_all_zero"
        ] = (
            df[
                "spectral_all_zero"
            ]
            .astype(str)
            .str.lower()
            .eq("true")
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

    df["GFD_ID"] = (
        pd.to_numeric(
            df["GFD_ID"],
            errors="coerce",
        )
    )

    df = df.dropna(
        subset=["GFD_ID"]
    )

    df["GFD_ID"] = (
        df["GFD_ID"]
        .astype(int)
    )

    return df


# =====================================================
# DOFA
# =====================================================

def load_dofa():

    print(
        "Loading pretrained DOFA..."
    )

    model = torch.hub.load(
        "zhu-xlab/DOFA",
        "vit_base_dofa",
        pretrained=True,
        trust_repo=True,
    )

    model = model.to(
        DEVICE
    )

    model.eval()

    return model


def extract_embeddings(
    model,
    batch,
):

    batch = batch.to(
        DEVICE
    )

    with torch.no_grad():

        embedding = (
            model.forward_features(
                batch,
                wave_list=
                    WAVELENGTHS,
            )
        )

    if (
        embedding.ndim != 2
    ):

        embedding = (
            embedding.flatten(
                start_dim=1
            )
        )

    return (
        embedding
        .detach()
        .cpu()
        .numpy()
    )


# =====================================================
# EVENT AGGREGATION
# =====================================================

def aggregate_embeddings(
    prefix,
    embeddings,
):

    output = {}

    if embeddings.shape[0] == 0:
        return output

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
                f"dofa_s2_{prefix}_"
                f"{i:04d}_event_"
                f"{stat_name}"
            ] = float(
                values[i]
            )

    return output


# =====================================================
# PROCESS ONE EVENT
# =====================================================

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

        Path(path)

        for path in
        valid_df[
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
            f"WARNING event_{gfd_id}: "
            "zero valid S2 patches"
        )

        return row

    before_embeddings = []
    after_embeddings = []

    # =================================================
    # BATCH PATCHES
    # =================================================

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
                    "Skipping bad patch:",
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

        before_emb = (
            extract_embeddings(
                model,
                before_batch,
            )
        )

        after_emb = (
            extract_embeddings(
                model,
                after_batch,
            )
        )

        before_embeddings.append(
            before_emb
        )

        after_embeddings.append(
            after_emb
        )

    # =================================================
    # CHECK
    # =================================================

    if len(
        before_embeddings
    ) == 0:

        print(
            f"WARNING event_{gfd_id}: "
            "no embeddings extracted"
        )

        return row

    before_embeddings = (
        np.concatenate(
            before_embeddings,
            axis=0,
        )
    )

    after_embeddings = (
        np.concatenate(
            after_embeddings,
            axis=0,
        )
    )

    # Temporal change representation
    delta_embeddings = (
        after_embeddings
        - before_embeddings
    )

    # =================================================
    # EVENT REPRESENTATION
    # =================================================

    row.update(
        aggregate_embeddings(
            "before",
            before_embeddings,
        )
    )

    row.update(
        aggregate_embeddings(
            "after",
            after_embeddings,
        )
    )

    row.update(
        aggregate_embeddings(
            "delta",
            delta_embeddings,
        )
    )

    print(
        f"event_{gfd_id}",
        "total:",
        total_patches,
        "valid:",
        len(valid_paths),
        "embedding:",
        before_embeddings.shape,
    )

    return row


# =====================================================
# MAIN
# =====================================================

def main():

    print("=" * 100)

    print(
        "DOFA SENTINEL-2 "
        "ALL-EVENT FEATURE EXTRACTION"
    )

    print("=" * 100)

    print(
        "Device:",
        DEVICE,
    )

    print(
        "Manifest:",
        PATCH_MANIFEST,
    )

    print(
        "Output:",
        OUTPUT_CSV,
    )

    print(
        "Bands:",
        BAND_NAMES,
    )

    print(
        "Wavelengths:",
        WAVELENGTHS,
    )

    print(
        "Minimum common-valid fraction:",
        MIN_VALID_FRACTION,
    )

    print(
        "Batch size:",
        BATCH_SIZE,
    )

    # =================================================
    # CHECK FILE
    # =================================================

    if not PATCH_MANIFEST.exists():

        raise FileNotFoundError(
            PATCH_MANIFEST
        )

    # =================================================
    # MANIFEST
    # =================================================

    manifest = load_manifest()

    print()
    print(
        "Manifest patches:",
        len(manifest),
    )

    print(
        "Usable patches:",
        int(
            manifest[
                "usable"
            ].sum()
        ),
    )

    print(
        "Events:",
        manifest[
            "GFD_ID"
        ].nunique(),
    )

    print(
        "Events with >=1 usable patch:",
        manifest.loc[
            manifest["usable"],
            "GFD_ID",
        ].nunique(),
    )

    # =================================================
    # MODEL
    # =================================================

    model = load_dofa()

    # =================================================
    # EVENTS
    # =================================================

    event_ids = sorted(
        manifest[
            "GFD_ID"
        ]
        .unique()
    )

    rows = []

    print()
    print(
        "Starting extraction..."
    )

    for gfd_id in tqdm(
        event_ids,
        desc="DOFA S2 events",
    ):

        event_df = (
            manifest[
                manifest["GFD_ID"]
                == gfd_id
            ]
        )

        row = process_event(
            gfd_id=
                gfd_id,

            event_df=
                event_df,

            model=
                model,
        )

        rows.append(
            row
        )

        # Save after every event
        partial = (
            pd.DataFrame(
                rows
            )
            .sort_values(
                "GFD_ID"
            )
        )

        partial.to_csv(
            OUTPUT_CSV,
            index=False,
        )

    # =================================================
    # FINAL SAVE
    # =================================================

    df = (
        pd.DataFrame(
            rows
        )
        .sort_values(
            "GFD_ID"
        )
    )

    df.to_csv(
        OUTPUT_CSV,
        index=False,
    )

    deep_cols = [
        c
        for c in df.columns
        if c.startswith(
            "dofa_s2_"
        )
    ]

    print()
    print("=" * 100)
    print("DONE")
    print("=" * 100)

    print(
        "Saved:",
        OUTPUT_CSV,
    )

    print(
        "Shape:",
        df.shape,
    )

    print(
        "Events:",
        df[
            "GFD_ID"
        ].nunique(),
    )

    print(
        "Events with embeddings:",
        int(
            (
                df[
                    "n_valid_patches"
                ]
                > 0
            ).sum()
        ),
    )

    print(
        "Deep feature columns:",
        len(
            deep_cols
        ),
    )


if __name__ == "__main__":
    main()
