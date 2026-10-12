from pathlib import Path
import os
import warnings

# =====================================================
# CACHE
# Keep downloads out of /home
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
import rasterio
import torch
import torch.nn.functional as F

warnings.filterwarnings("ignore")


# =====================================================
# SETTINGS
# =====================================================

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available()
    else "cpu"
)

TEST_PATCH = Path(
    "/home/melissa/flood_project/"
    "S2_PATCHES_CLEAN/"
    "event_4294/patch_005.tif"
)

IMAGE_SIZE = 224


# =====================================================
# OUR SENTINEL-2 BANDS
#
# TIFF:
# B2 B3 B4 B8 B11 B12
#
# DOFA only needs the corresponding wavelengths.
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
    0.490,   # B2 blue
    0.560,   # B3 green
    0.665,   # B4 red
    0.842,   # B8 NIR
    1.610,   # B11 SWIR1
    2.190,   # B12 SWIR2
]


# =====================================================
# DOFA SENTINEL-2 NORMALIZATION
#
# Official DOFA Sentinel-2 demo uses this band order:
#
# red, green, blue,
# rededge1, rededge2, rededge3,
# nir, swir1, swir2
#
# We select only the statistics corresponding to
# our 6 available bands and reorder them to:
#
# blue, green, red, nir, swir1, swir2
# =====================================================

S2_MEAN = torch.tensor(
    [
        126.63977424,  # blue
        114.81779093,  # green
        114.10997390,  # red
        101.43563300,  # nir
        72.32804172,   # swir1
        56.66528851,   # swir2
    ],
    dtype=torch.float32,
).view(6, 1, 1)

S2_STD = torch.tensor(
    [
        67.42465279,   # blue
        69.96844919,   # green
        77.84352553,   # red
        60.29744676,   # nir
        47.88519516,   # swir1
        42.55886798,   # swir2
    ],
    dtype=torch.float32,
).view(6, 1, 1)


# =====================================================
# HELPERS
# =====================================================

def fill_invalid_per_band(
    arr,
    valid_mask,
):
    """
    Fill cloud/invalid pixels with the median
    of valid pixels independently per band.
    """

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
    """
    FLODIS Sentinel-2 is stored as reflectance * 10000.

    Convert reflectance to an 8-bit-like 0-255 range
    to match the scale used by the official DOFA
    Sentinel-2 demo, then apply DOFA S2 statistics.
    """

    image = np.clip(
        image,
        0.0,
        10000.0,
    )

    # reflectance:
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

    # Resize deterministically to DOFA input size
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

    # Sentinel-2 normalization
    image = (
        image - S2_MEAN
    ) / S2_STD

    return image


def read_before_after(
    path,
):
    """
    Our S2 TIFF:

     1 B2 before
     2 B3 before
     3 B4 before
     4 B8 before
     5 B11 before
     6 B12 before

     7 B2 after
     8 B3 after
     9 B4 after
    10 B8 after
    11 B11 after
    12 B12 after

    13 before_valid_mask
    14 after_valid_mask
    15 flood_mask
    """

    with rasterio.open(
        path
    ) as src:

        if src.count != 15:

            raise ValueError(
                f"Expected 15 bands, "
                f"found {src.count}"
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
# MAIN
# =====================================================

def main():

    print("=" * 100)
    print(
        "DOFA ONE-PATCH SENTINEL-2 TEST"
    )
    print("=" * 100)

    print(
        "Device:",
        DEVICE,
    )

    print(
        "Patch:",
        TEST_PATCH,
    )

    print(
        "Bands:",
        BAND_NAMES,
    )

    print(
        "Wavelengths:",
        WAVELENGTHS,
    )

    if not TEST_PATCH.exists():

        raise FileNotFoundError(
            TEST_PATCH
        )

    # =================================================
    # CHECK TIMM
    # =================================================

    try:

        import timm

        print(
            "timm version:",
            timm.__version__,
        )

    except Exception as e:

        print()
        print(
            "TIMM ERROR:"
        )

        print(
            repr(e)
        )

        print()
        print(
            "Stop here and send me "
            "this error."
        )

        return

    # =================================================
    # LOAD IMAGE
    # =================================================

    before, after = (
        read_before_after(
            TEST_PATCH
        )
    )

    print()
    print(
        "Before:",
        before.shape,
    )

    print(
        "After:",
        after.shape,
    )

    print(
        "Before normalized range:",
        float(
            before.min()
        ),
        float(
            before.max()
        ),
    )

    print(
        "After normalized range:",
        float(
            after.min()
        ),
        float(
            after.max()
        ),
    )

    # =================================================
    # LOAD PRETRAINED DOFA
    # =================================================

    print()
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

    # =================================================
    # EXTRACT DEEP FEATURES
    # =================================================

    before_batch = (
        before
        .unsqueeze(0)
        .to(DEVICE)
    )

    after_batch = (
        after
        .unsqueeze(0)
        .to(DEVICE)
    )

    with torch.no_grad():

        before_embedding = (
            model.forward_features(
                before_batch,
                wave_list=WAVELENGTHS,
            )
        )

        after_embedding = (
            model.forward_features(
                after_batch,
                wave_list=WAVELENGTHS,
            )
        )

    delta_embedding = (
        after_embedding
        - before_embedding
    )

    # =================================================
    # RESULTS
    # =================================================

    print()
    print("=" * 100)
    print("DOFA OUTPUT")
    print("=" * 100)

    print(
        "Before embedding:",
        tuple(
            before_embedding.shape
        ),
    )

    print(
        "After embedding:",
        tuple(
            after_embedding.shape
        ),
    )

    print(
        "Delta embedding:",
        tuple(
            delta_embedding.shape
        ),
    )

    print()
    print(
        "Before embedding mean/std:",
        float(
            before_embedding.mean()
        ),
        float(
            before_embedding.std()
        ),
    )

    print(
        "After embedding mean/std:",
        float(
            after_embedding.mean()
        ),
        float(
            after_embedding.std()
        ),
    )

    print(
        "Delta embedding mean/std:",
        float(
            delta_embedding.mean()
        ),
        float(
            delta_embedding.std()
        ),
    )

    finite = (
        torch.isfinite(
            before_embedding
        ).all()
        and
        torch.isfinite(
            after_embedding
        ).all()
    )

    print()
    print(
        "Finite embeddings:",
        bool(finite),
    )

    if (
        before_embedding.shape
        == (1, 768)
        and
        after_embedding.shape
        == (1, 768)
        and finite
    ):

        print()
        print("=" * 100)
        print(
            "SANITY TEST PASSED"
        )
        print("=" * 100)

    else:

        print()
        print("=" * 100)
        print(
            "UNEXPECTED OUTPUT - "
            "SEND ME THE RESULT"
        )
        print("=" * 100)


if __name__ == "__main__":
    main()
