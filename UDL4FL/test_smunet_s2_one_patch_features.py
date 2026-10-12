from pathlib import Path

import numpy as np
import rasterio
import torch
import torch.nn.functional as F

from udl4fl.model_setup import load_model_by_name


# ============================================================
# SETTINGS
# ============================================================

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

PATCH = Path(
    "/home/melissa/flood_project/"
    "S2_PATCHES_CLEAN/event_4294/patch_005.tif"
)

WEIGHTS_FOLDER = (
    "/home/melissa/flood_project/"
    "models_benchmark/UDL4FL/cloudsen12_models"
)

IMAGE_SIZE = 256


# ============================================================
# EXACT NORMALIZATION FROM sm_unet_s2.json
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

def fill_invalid(arr, mask):

    arr = arr.astype(np.float32).copy()

    mask = mask > 0

    for b in range(arr.shape[0]):

        band = arr[b]

        valid = (
            mask
            & np.isfinite(band)
        )

        if valid.sum() == 0:
            band[:] = 0
        else:
            med = np.median(
                band[valid]
            )

            bad = (
                (~mask)
                | (~np.isfinite(band))
            )

            band[bad] = med

        arr[b] = band

    return arr


def resize(arr):

    x = torch.from_numpy(
        arr.astype(np.float32)
    ).unsqueeze(0)

    x = F.interpolate(
        x,
        size=(IMAGE_SIZE, IMAGE_SIZE),
        mode="bilinear",
        align_corners=False,
    )

    return x


def read_patch(path):

    with rasterio.open(path) as src:

        print(
            "Original raster size:",
            src.height,
            "x",
            src.width,
        )

        print(
            "Bands:",
            src.count,
        )

        before = src.read(
            [1, 2, 3, 4, 5, 6]
        ).astype(np.float32)

        after = src.read(
            [7, 8, 9, 10, 11, 12]
        ).astype(np.float32)

        before_mask = src.read(13)
        after_mask = src.read(14)

    before = fill_invalid(
        before,
        before_mask
    )

    after = fill_invalid(
        after,
        after_mask
    )

    before = resize(before)
    after = resize(after)

    return before, after


# ============================================================
# MAIN
# ============================================================

print("=" * 90)
print("sm_unet_s2 ONE PATCH FEATURE TEST")
print("=" * 90)

print("Device:", DEVICE)
print("Patch:", PATCH)


# ------------------------------------------------------------
# Load official pretrained checkpoint
# ------------------------------------------------------------

model = load_model_by_name(
    "sm_unet_s2",
    weights_folder=WEIGHTS_FOLDER,
    device=DEVICE,
)

model.eval()


# ------------------------------------------------------------
# Read image
# ------------------------------------------------------------

before_raw, after_raw = read_patch(
    PATCH
)

print()
print(
    "Before raw:",
    before_raw.shape,
    float(before_raw.min()),
    float(before_raw.max()),
)

print(
    "After raw:",
    after_raw.shape,
    float(after_raw.min()),
    float(after_raw.max()),
)


# ------------------------------------------------------------
# Normalize exactly as checkpoint expects
# ------------------------------------------------------------

mean = MEAN.to(DEVICE)
std = STD.to(DEVICE)

before = before_raw.to(DEVICE)
after = after_raw.to(DEVICE)

before_norm = (
    before - mean
) / std

after_norm = (
    after - mean
) / std


print()
print(
    "Before normalized range:",
    float(before_norm.min()),
    float(before_norm.max()),
)

print(
    "After normalized range:",
    float(after_norm.min()),
    float(after_norm.max()),
)


# ------------------------------------------------------------
# Encoder features
# ------------------------------------------------------------

with torch.no_grad():

    before_features = (
        model.network.encoder(
            before_norm
        )
    )

    after_features = (
        model.network.encoder(
            after_norm
        )
    )


print()
print("=" * 90)
print("ENCODER FEATURE MAPS")
print("=" * 90)

for i, feat in enumerate(
    before_features
):

    print(
        f"Stage {i}:",
        tuple(feat.shape)
    )


# ------------------------------------------------------------
# Deepest encoder representation
# ------------------------------------------------------------

before_deep = before_features[-1]
after_deep = after_features[-1]

before_embedding = (
    F.adaptive_avg_pool2d(
        before_deep,
        output_size=1
    )
    .flatten(1)
)

after_embedding = (
    F.adaptive_avg_pool2d(
        after_deep,
        output_size=1
    )
    .flatten(1)
)

delta_embedding = (
    after_embedding
    - before_embedding
)


print()
print("=" * 90)
print("DEEP EMBEDDINGS")
print("=" * 90)

print(
    "Deep feature map:",
    tuple(before_deep.shape)
)

print(
    "Before embedding:",
    tuple(before_embedding.shape)
)

print(
    "After embedding:",
    tuple(after_embedding.shape)
)

print(
    "Delta embedding:",
    tuple(delta_embedding.shape)
)

print(
    "Finite:",
    bool(
        torch.isfinite(
            before_embedding
        ).all()
        and
        torch.isfinite(
            after_embedding
        ).all()
    )
)


# ------------------------------------------------------------
# Also verify full flood segmentation
# model.predict() takes RAW values and performs its own
# normalization because normalized_data=False.
# ------------------------------------------------------------

with torch.no_grad():

    pred_before, prob_before = (
        model.predict(
            before_raw.to(DEVICE)
        )
    )

    pred_after, prob_after = (
        model.predict(
            after_raw.to(DEVICE)
        )
    )


print()
print("=" * 90)
print("SEGMENTATION CHECK")
print("=" * 90)

print(
    "Prediction shape:",
    tuple(pred_after.shape)
)

print(
    "Probability shape:",
    tuple(prob_after.shape)
)

print(
    "Before classes:",
    torch.unique(
        pred_before
    ).cpu().tolist()
)

print(
    "After classes:",
    torch.unique(
        pred_after
    ).cpu().tolist()
)

# Class 1 = water according to model JSON:
# ["land", "water"]

before_water_fraction = (
    pred_before.float()
    .mean()
    .item()
)

after_water_fraction = (
    pred_after.float()
    .mean()
    .item()
)

print(
    "Predicted water fraction before:",
    before_water_fraction
)

print(
    "Predicted water fraction after:",
    after_water_fraction
)

print()
print("=" * 90)
print("TEST COMPLETE")
print("=" * 90)
