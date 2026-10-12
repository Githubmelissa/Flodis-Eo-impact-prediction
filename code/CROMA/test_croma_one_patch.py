from pathlib import Path
import sys

import numpy as np
import rasterio
import torch
import torch.nn.functional as F

# Import CROMA
THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS_DIR))

from use_croma import PretrainedCROMA


# =====================================================
# PATHS
# =====================================================
CROMA_DIR = Path("/home/melissa/flood_project/models_benchmark/CROMA")
WEIGHTS_PATH = CROMA_DIR / "checkpoints/CROMA_base.pt"

PATCH_PATH = Path(
    "/home/melissa/flood_project/S1_PATCHES_CLEAN/event_4328/patch_000/S1_stack.tif"
)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
IMAGE_SIZE = 120


# =====================================================
# FUNCTIONS
# =====================================================
def normalize_sar_pair(x):
    """
    x shape: (2, H, W)
    Normalize VV and VH separately to [0, 1].
    """
    x = x.astype(np.float32)
    out = np.zeros_like(x, dtype=np.float32)

    for c in range(x.shape[0]):
        band = x[c]
        finite = np.isfinite(band)

        if finite.sum() == 0:
            out[c] = 0.0
            continue

        vals = band[finite]
        mean = vals.mean()
        std = vals.std()

        if std < 1e-6:
            norm = np.zeros_like(band, dtype=np.float32)
        else:
            min_value = mean - 2 * std
            max_value = mean + 2 * std
            norm = (band - min_value) / (max_value - min_value)
            norm = np.clip(norm, 0.0, 1.0)

        norm[~finite] = 0.0
        out[c] = norm

    return out


def read_before_after(stack_path):
    """
    Our S1_stack.tif bands:
    1 before_VV
    2 before_VH
    3 after_VV
    4 after_VH
    5 delta_VV
    6 delta_VH
    7 flood_mask
    """
    with rasterio.open(stack_path) as src:
        arr = src.read().astype(np.float32)

    before = arr[[0, 1], :, :]
    after = arr[[2, 3], :, :]

    before = normalize_sar_pair(before)
    after = normalize_sar_pair(after)

    before_t = torch.from_numpy(before).unsqueeze(0)
    after_t = torch.from_numpy(after).unsqueeze(0)

    before_t = F.interpolate(
        before_t,
        size=(IMAGE_SIZE, IMAGE_SIZE),
        mode="bilinear",
        align_corners=False,
    )

    after_t = F.interpolate(
        after_t,
        size=(IMAGE_SIZE, IMAGE_SIZE),
        mode="bilinear",
        align_corners=False,
    )

    return before_t, after_t


# =====================================================
# MAIN
# =====================================================
print("=" * 80)
print("Device:", DEVICE)
print("Patch:", PATCH_PATH)
print("Weights:", WEIGHTS_PATH)

assert PATCH_PATH.exists(), f"Patch not found: {PATCH_PATH}"
assert WEIGHTS_PATH.exists(), f"Weights not found: {WEIGHTS_PATH}"

before, after = read_before_after(PATCH_PATH)

print("Before tensor:", before.shape, before.min().item(), before.max().item())
print("After tensor:", after.shape, after.min().item(), after.max().item())

model = PretrainedCROMA(
    pretrained_path=str(WEIGHTS_PATH),
    size="base",
    modality="SAR",
    image_resolution=IMAGE_SIZE,
)

model = model.to(DEVICE)
model.eval()

before = before.to(DEVICE)
after = after.to(DEVICE)

with torch.no_grad():
    out_before = model(SAR_images=before)
    out_after = model(SAR_images=after)

before_emb = out_before["SAR_GAP"]
after_emb = out_after["SAR_GAP"]
delta_emb = after_emb - before_emb

print()
print("CROMA output keys:", out_before.keys())
print("Before embedding:", before_emb.shape)
print("After embedding:", after_emb.shape)
print("Delta embedding:", delta_emb.shape)

print()
print("Before embedding mean/std:", before_emb.mean().item(), before_emb.std().item())
print("After embedding mean/std:", after_emb.mean().item(), after_emb.std().item())
print("Delta embedding mean/std:", delta_emb.mean().item(), delta_emb.std().item())

print()
print("SUCCESS: CROMA one-patch test worked.")