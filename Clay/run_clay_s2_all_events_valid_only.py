from pathlib import Path
import os
import sys
import warnings

# =====================================================
# CACHE SETTINGS
# Important because /home is full
# =====================================================
os.environ.setdefault("HF_HOME", "/dev/shm/melissa/cache/huggingface")
os.environ.setdefault("HF_HUB_CACHE", "/dev/shm/melissa/cache/huggingface/hub")
os.environ.setdefault("HUGGINGFACE_HUB_CACHE", "/dev/shm/melissa/cache/huggingface/hub")
os.environ.setdefault("HF_XET_CACHE", "/dev/shm/melissa/cache/huggingface/xet")
os.environ.setdefault("TORCH_HOME", "/dev/shm/melissa/cache/torch")
os.environ.setdefault("XDG_CACHE_HOME", "/dev/shm/melissa/cache")
os.environ.setdefault("TMPDIR", "/dev/shm/melissa/tmp")

for p in [
    os.environ["HF_HOME"],
    os.environ["HF_HUB_CACHE"],
    os.environ["HF_XET_CACHE"],
    os.environ["TORCH_HOME"],
    os.environ["TMPDIR"],
]:
    os.makedirs(p, exist_ok=True)

# Use real Clay repo cloned in /dev/shm
sys.path.insert(0, "/dev/shm/melissa/Clay_model")

import numpy as np
import pandas as pd
import rasterio
import torch
import torch.nn.functional as F
import yaml
from tqdm import tqdm

warnings.filterwarnings("ignore")


# =====================================================
# PATHS
# =====================================================
CLAY_ROOT = Path("/dev/shm/melissa/Clay_model")
CHECKPOINT_PATH = Path("/dev/shm/melissa/clay_checkpoints/clay-v1.5.ckpt")
METADATA_PATH = CLAY_ROOT / "configs" / "metadata.yaml"

S2_ROOT = Path("/home/melissa/flood_project/S2_PATCHES_CLEAN")

PATCH_MANIFEST = Path(
    "/home/melissa/flood_project/data/s2_patch_inventory.csv"
)

OUTPUT_DIR = Path(
    "/home/melissa/flood_project/"
    "models_benchmark/Clay/flodis_outputs/"
    "clay_s2_event_features_valid_only"
)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_CSV = (
    OUTPUT_DIR /
    "all_events_clay_s2_event_features_valid_only.csv"
)


# =====================================================
# SETTINGS
# =====================================================
SENSOR_NAME = "sentinel-2-l2a"

# Sentinel-2 bands actually exported
# B2, B3, B4, B8, B11, B12
CLAY_BANDS = [
    "blue",      # B2
    "green",     # B3
    "red",       # B4
    "nir",       # B8
    "swir16",    # B11
    "swir22",    # B12
]

CHIP_SIZE = 256
MIN_VALID_FRACTION = 0.80
BATCH_SIZE = 4

# IMPORTANT:
# True  -> test ONE patch only
# False -> extract features for ALL events
SANITY_TEST_ONLY = False

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# =====================================================
# DATA HELPERS
# =====================================================
def fill_invalid_per_band(arr, valid_mask):
    """
    Fill invalid/cloud-masked pixels using the median of
    the valid pixels of each spectral band.
    """

    arr = arr.astype("float32").copy()
    valid_mask = valid_mask > 0

    for b in range(arr.shape[0]):
        band = arr[b]

        valid = valid_mask & np.isfinite(band)

        if valid.sum() == 0:
            band[:] = 0.0
        else:
            median_value = np.median(band[valid])

            invalid = (
                (~valid_mask) |
                (~np.isfinite(band))
            )

            band[invalid] = median_value

        arr[b] = band

    return arr


def read_s2_before_after(path):
    """
    Sentinel-2 TIFF layout:

    Bands 1-6 = BEFORE
        1  B2   blue
        2  B3   green
        3  B4   red
        4  B8   nir
        5  B11  swir16
        6  B12  swir22

    Bands 7-12 = AFTER
        7  B2
        8  B3
        9  B4
        10 B8
        11 B11
        12 B12

    Band 13 = before_valid_mask
    Band 14 = after_valid_mask
    Band 15 = flood_mask
    """

    with rasterio.open(path) as src:

        if src.count != 15:
            raise ValueError(
                f"Expected 15 bands, found {src.count}: {path}"
            )

        before = src.read(
            [1, 2, 3, 4, 5, 6]
        ).astype("float32")

        after = src.read(
            [7, 8, 9, 10, 11, 12]
        ).astype("float32")

        before_valid = src.read(13)
        after_valid = src.read(14)

    # Replace invalid pixels using valid-pixel median
    before = fill_invalid_per_band(
        before,
        before_valid,
    )

    after = fill_invalid_per_band(
        after,
        after_valid,
    )

    # Convert to Torch tensors
    before = torch.from_numpy(before).unsqueeze(0)
    after = torch.from_numpy(after).unsqueeze(0)

    # Clay input size
    before = F.interpolate(
        before,
        size=(CHIP_SIZE, CHIP_SIZE),
        mode="bilinear",
        align_corners=False,
    ).squeeze(0)

    after = F.interpolate(
        after,
        size=(CHIP_SIZE, CHIP_SIZE),
        mode="bilinear",
        align_corners=False,
    ).squeeze(0)

    return before, after


# =====================================================
# CLAY HELPERS
# =====================================================
def load_metadata():
    with open(METADATA_PATH, "r") as f:
        metadata = yaml.safe_load(f)

    if SENSOR_NAME not in metadata:
        print("Available sensors:")
        print(list(metadata.keys()))

        raise KeyError(
            f"{SENSOR_NAME} not found in metadata.yaml"
        )

    return metadata


def get_metadata_tensors(metadata):
    """
    Select Clay metadata only for the 6 Sentinel-2
    bands available in our FLODIS exports.
    """

    sensor_meta = metadata[SENSOR_NAME]
    bands_meta = sensor_meta["bands"]

    print("Clay sensor:", SENSOR_NAME)
    print("Selected bands:", CLAY_BANDS)
    print("GSD:", sensor_meta.get("gsd"))

    wavelengths = []
    means = []
    stds = []

    for band in CLAY_BANDS:

        wavelengths.append(
            float(
                bands_meta["wavelength"][band]
            )
        )

        means.append(
            float(
                bands_meta["mean"][band]
            )
        )

        stds.append(
            float(
                bands_meta["std"][band]
            )
        )

    wavelengths = torch.tensor(
        wavelengths,
        dtype=torch.float32,
        device=DEVICE,
    )

    means = torch.tensor(
        means,
        dtype=torch.float32,
        device=DEVICE,
    ).view(1, -1, 1, 1)

    stds = torch.tensor(
        stds,
        dtype=torch.float32,
        device=DEVICE,
    ).view(1, -1, 1, 1)

    print(
        "Wavelengths:",
        wavelengths.detach().cpu().numpy(),
    )

    print(
        "Means:",
        means.flatten().detach().cpu().numpy(),
    )

    print(
        "Stds:",
        stds.flatten().detach().cpu().numpy(),
    )

    return wavelengths, means, stds


def load_clay_model():
    from claymodel.module import ClayMAEModule

    print("Loading Clay checkpoint:")
    print(CHECKPOINT_PATH)

    old_cwd = os.getcwd()
    os.chdir(str(CLAY_ROOT))

    try:
        model = ClayMAEModule.load_from_checkpoint(
            str(CHECKPOINT_PATH),
            metadata_path=str(METADATA_PATH),
            map_location=DEVICE,
        )
    finally:
        os.chdir(old_cwd)

    model = model.to(DEVICE)
    model.eval()

    if (
        hasattr(model, "model")
        and hasattr(model.model, "encoder")
    ):
        model.model.encoder.mask_ratio = 0.0
        model.model.encoder.shuffle = False

    return model


def get_clay_encoder(model):

    if hasattr(model, "encoder"):
        return model.encoder

    if (
        hasattr(model, "model")
        and hasattr(model.model, "encoder")
    ):
        return model.model.encoder

    raise AttributeError(
        "Could not find Clay encoder"
    )


def extract_batch_embeddings(
    model,
    batch,
    wavelengths,
    means,
    stds,
):
    """
    Run Clay encoder and average spatial tokens
    to obtain one embedding per image.
    """

    encoder = get_clay_encoder(model)

    if hasattr(encoder, "mask_ratio"):
        encoder.mask_ratio = 0.0

    if hasattr(encoder, "shuffle"):
        encoder.shuffle = False

    batch = batch.to(DEVICE)

    # IMPORTANT:
    # Do NOT divide Sentinel-2 values by 10000 here.
    # Clay metadata means/stds are also in scaled
    # Sentinel-2 reflectance units (~1000-3000).
    batch = (batch - means) / stds

    batch_size = batch.shape[0]

    datacube = {
        "pixels": batch,

        "time": torch.zeros(
            batch_size,
            4,
            dtype=torch.float32,
            device=DEVICE,
        ),

        "latlon": torch.zeros(
            batch_size,
            4,
            dtype=torch.float32,
            device=DEVICE,
        ),

        "gsd": torch.tensor(
            10.0,
            dtype=torch.float32,
            device=DEVICE,
        ),

        "waves": wavelengths.to(DEVICE),
    }

    with torch.no_grad():
        out = encoder(datacube)

    if isinstance(out, (tuple, list)):
        encoded = out[0]

    elif torch.is_tensor(out):
        encoded = out

    else:
        raise RuntimeError(
            f"Unexpected encoder output type: {type(out)}"
        )

    # Clay usually returns:
    # [B, number_of_tokens, 1024]
    #
    # Token 0 = CLS token
    # Remaining tokens = spatial tokens
    if encoded.ndim == 3:

        if encoded.shape[1] > 1:
            embedding = (
                encoded[:, 1:, :]
                .mean(dim=1)
            )
        else:
            embedding = encoded.mean(dim=1)

    elif encoded.ndim == 2:
        embedding = encoded

    else:
        embedding = encoded.flatten(
            start_dim=1
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
    """
    Aggregate patch embeddings into an event-level
    representation using mean/std/min/max.

    embeddings:
        [number_of_patches, embedding_dimension]
    """

    output = {}

    if embeddings.shape[0] == 0:
        return output

    stats = {
        "mean": embeddings.mean(axis=0),
        "std": embeddings.std(axis=0),
        "min": embeddings.min(axis=0),
        "max": embeddings.max(axis=0),
    }

    dim = embeddings.shape[1]

    for stat_name, values in stats.items():

        for i in range(dim):

            output[
                f"clay_s2_{prefix}_"
                f"{i:04d}_event_{stat_name}"
            ] = float(values[i])

    return output


# =====================================================
# PATCH MANIFEST
# =====================================================
def load_patch_manifest():
    """
    Load Sentinel-2 quality information generated
    during the previous validation step.
    """

    df = pd.read_csv(PATCH_MANIFEST)

    # Make sure boolean columns are interpreted correctly
    if df["read_ok"].dtype != bool:
        df["read_ok"] = (
            df["read_ok"]
            .astype(str)
            .str.lower()
            .eq("true")
        )

    if df["spectral_all_zero"].dtype != bool:
        df["spectral_all_zero"] = (
            df["spectral_all_zero"]
            .astype(str)
            .str.lower()
            .eq("true")
        )

    # Usable for THIS Clay experiment:
    # - readable
    # - not completely empty
    # - at least 80% common valid pixels
    df["usable"] = (
        df["read_ok"]
        &
        (~df["spectral_all_zero"])
        &
        (
            df["common_valid_fraction"]
            >= MIN_VALID_FRACTION
        )
    )

    return df


# =====================================================
# EVENT PROCESSING
# =====================================================
def process_event(
    gfd_id,
    event_df,
    model,
    wavelengths,
    means,
    stds,
):

    total_patches = len(event_df)

    valid_df = (
        event_df[
            event_df["usable"]
        ]
        .copy()
    )

    valid_paths = [
        Path(path)
        for path in valid_df["path"].tolist()
    ]

    row = {
        "GFD_ID": int(gfd_id),
        "event_id": f"event_{int(gfd_id)}",

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

    for start in range(
        0,
        len(valid_paths),
        BATCH_SIZE,
    ):

        batch_paths = valid_paths[
            start:start + BATCH_SIZE
        ]

        before_list = []
        after_list = []

        for path in batch_paths:

            try:
                before, after = (
                    read_s2_before_after(path)
                )

                before_list.append(before)
                after_list.append(after)

            except Exception as e:

                print(
                    f"Skipping bad S2 patch "
                    f"{path}: {repr(e)}"
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

        before_emb = extract_batch_embeddings(
            model,
            before_batch,
            wavelengths,
            means,
            stds,
        )

        after_emb = extract_batch_embeddings(
            model,
            after_batch,
            wavelengths,
            means,
            stds,
        )

        before_embeddings.append(
            before_emb
        )

        after_embeddings.append(
            after_emb
        )

    if len(before_embeddings) == 0:

        print(
            f"WARNING event_{gfd_id}: "
            "valid paths existed but "
            "no embeddings were extracted"
        )

        return row

    before_embeddings = np.concatenate(
        before_embeddings,
        axis=0,
    )

    after_embeddings = np.concatenate(
        after_embeddings,
        axis=0,
    )

    # Temporal deep representation
    delta_embeddings = (
        after_embeddings
        - before_embeddings
    )

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
# ONE-PATCH SANITY TEST
# =====================================================
def run_sanity_test(
    manifest,
    model,
    wavelengths,
    means,
    stds,
):
    """
    Test one good Sentinel-2 patch before processing
    the entire dataset.
    """

    usable = manifest[
        manifest["usable"]
    ].copy()

    if len(usable) == 0:
        raise RuntimeError(
            "No usable Sentinel-2 patches found."
        )

    test_row = usable.iloc[0]

    test_path = Path(
        test_row["path"]
    )

    print()
    print("=" * 100)
    print("ONE-PATCH SANITY TEST")
    print("=" * 100)

    print("GFD_ID:", test_row["GFD_ID"])
    print("Patch:", test_row["patch_id"])
    print(
        "Common valid fraction:",
        test_row["common_valid_fraction"],
    )

    print("Path:")
    print(test_path)

    before, after = (
        read_s2_before_after(
            test_path
        )
    )

    print()
    print(
        "Before tensor:",
        before.shape,
    )

    print(
        "After tensor:",
        after.shape,
    )

    print(
        "Before range:",
        float(before.min()),
        float(before.max()),
    )

    print(
        "After range:",
        float(after.min()),
        float(after.max()),
    )

    # Test BEFORE
    before_embedding = (
        extract_batch_embeddings(
            model,
            before.unsqueeze(0),
            wavelengths,
            means,
            stds,
        )
    )

    # Test AFTER as well
    after_embedding = (
        extract_batch_embeddings(
            model,
            after.unsqueeze(0),
            wavelengths,
            means,
            stds,
        )
    )

    print()
    print(
        "Clay BEFORE embedding shape:",
        before_embedding.shape,
    )

    print(
        "Clay AFTER embedding shape:",
        after_embedding.shape,
    )

    delta_embedding = (
        after_embedding
        - before_embedding
    )

    print(
        "Clay DELTA embedding shape:",
        delta_embedding.shape,
    )

    print()
    print("=" * 100)
    print("SANITY TEST PASSED")
    print("=" * 100)


# =====================================================
# MAIN
# =====================================================
def main():

    print("=" * 100)
    print(
        "Clay Sentinel-2 feature extraction"
    )

    print("Device:", DEVICE)
    print("S2 root:", S2_ROOT)
    print("Patch manifest:", PATCH_MANIFEST)
    print("Clay repo:", CLAY_ROOT)
    print("Checkpoint:", CHECKPOINT_PATH)
    print("Output:", OUTPUT_CSV)
    print("Batch size:", BATCH_SIZE)

    print(
        "Min common valid fraction:",
        MIN_VALID_FRACTION,
    )

    print(
        "Sanity test only:",
        SANITY_TEST_ONLY,
    )

    print("=" * 100)

    # -------------------------------------------------
    # Check required files/folders
    # -------------------------------------------------
    if not S2_ROOT.exists():
        raise FileNotFoundError(
            S2_ROOT
        )

    if not PATCH_MANIFEST.exists():
        raise FileNotFoundError(
            PATCH_MANIFEST
        )

    if not CLAY_ROOT.exists():
        raise FileNotFoundError(
            CLAY_ROOT
        )

    if not CHECKPOINT_PATH.exists():
        raise FileNotFoundError(
            CHECKPOINT_PATH
        )

    if not METADATA_PATH.exists():
        raise FileNotFoundError(
            METADATA_PATH
        )

    # -------------------------------------------------
    # Load Clay metadata
    # -------------------------------------------------
    metadata = load_metadata()

    wavelengths, means, stds = (
        get_metadata_tensors(
            metadata
        )
    )

    # -------------------------------------------------
    # Load Clay model
    # -------------------------------------------------
    model = load_clay_model()

    # -------------------------------------------------
    # Load S2 patch-quality manifest
    # -------------------------------------------------
    manifest = load_patch_manifest()

    print()
    print(
        "Manifest patches:",
        len(manifest),
    )

    print(
        "Usable patches:",
        int(manifest["usable"].sum()),
    )

    print(
        "Events in manifest:",
        manifest["GFD_ID"].nunique(),
    )

    print(
        "Events with >=1 usable patch:",
        manifest.loc[
            manifest["usable"],
            "GFD_ID",
        ].nunique(),
    )

    # =================================================
    # SANITY TEST
    # =================================================
    if SANITY_TEST_ONLY:

        run_sanity_test(
            manifest,
            model,
            wavelengths,
            means,
            stds,
        )

        return

    # =================================================
    # FULL EXTRACTION
    # =================================================
    event_ids = sorted(
        manifest[
            "GFD_ID"
        ]
        .astype(int)
        .unique()
    )

    print()
    print(
        "Starting full extraction."
    )

    print(
        "Events:",
        len(event_ids),
    )

    rows = []

    for gfd_id in tqdm(
        event_ids,
        desc="S2 events",
    ):

        event_df = manifest[
            manifest["GFD_ID"]
            == gfd_id
        ]

        row = process_event(
            gfd_id,
            event_df,
            model,
            wavelengths,
            means,
            stds,
        )

        rows.append(row)

        # Save progress after every event
        partial_df = (
            pd.DataFrame(rows)
            .sort_values("GFD_ID")
        )

        partial_df.to_csv(
            OUTPUT_CSV,
            index=False,
        )

    # Final save
    df = (
        pd.DataFrame(rows)
        .sort_values("GFD_ID")
    )

    df.to_csv(
        OUTPUT_CSV,
        index=False,
    )

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
        df["GFD_ID"].nunique(),
    )

    print(
        "Events with S2 embeddings:",
        int(
            (
                df["n_valid_patches"] > 0
            ).sum()
        ),
    )

    print(
        "Deep feature columns:",
        len(
            [
                c for c in df.columns
                if c.startswith("clay_s2_")
            ]
        ),
    )


if __name__ == "__main__":
    main()