from pathlib import Path
import sys
import re

import numpy as np
import pandas as pd
import rasterio
import torch
import torch.nn.functional as F
from tqdm import tqdm

THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS_DIR))

from use_croma import PretrainedCROMA


# =====================================================
# PATHS
# =====================================================
CROMA_DIR = Path("/home/melissa/flood_project/models_benchmark/CROMA")
WEIGHTS_PATH = CROMA_DIR / "checkpoints/CROMA_base.pt"

PATCH_ROOT = Path("/home/melissa/flood_project/S1_PATCHES_CLEAN")

OUT_DIR = CROMA_DIR / "flodis_outputs/croma_event_features_valid_only"
OUT_DIR.mkdir(parents=True, exist_ok=True)

OUT_ALL = OUT_DIR / "all_events_croma_event_features_valid_only.csv"


# =====================================================
# SETTINGS
# =====================================================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
IMAGE_SIZE = 120
BATCH_SIZE = 16

VALID_FINITE_THRESHOLD = 0.80


# =====================================================
# HELPERS
# =====================================================
def extract_gfd_id(event_dir):
    m = re.search(r"event_(\d+)", event_dir.name)
    if m is None:
        return None
    return int(m.group(1))


def check_patch_valid(stack_path):
    """
    Check only bands 1-4:
    before_VV, before_VH, after_VV, after_VH
    """
    try:
        with rasterio.open(stack_path) as src:
            arr = src.read([1, 2, 3, 4]).astype(np.float32)

        finite_fracs = []
        for i in range(4):
            band = arr[i]
            finite_fracs.append(np.isfinite(band).mean())

        return min(finite_fracs) >= VALID_FINITE_THRESHOLD

    except Exception:
        return False


def normalize_sar_pair(x):
    """
    x shape: (2, H, W)
    Normalize VV/VH separately to [0, 1].
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

    return before_t.squeeze(0), after_t.squeeze(0)


def aggregate_features(patch_features):
    """
    patch_features shape: (n_valid_patches, feature_dim)
    """
    agg = {}

    agg["mean"] = np.mean(patch_features, axis=0)
    agg["std"] = np.std(patch_features, axis=0)
    agg["min"] = np.min(patch_features, axis=0)
    agg["max"] = np.max(patch_features, axis=0)

    return agg


def process_event(event_dir, model):
    gfd_id = extract_gfd_id(event_dir)

    patch_dirs = sorted([p for p in event_dir.glob("patch_*") if p.is_dir()])
    stack_paths = [p / "S1_stack.tif" for p in patch_dirs]
    stack_paths = [p for p in stack_paths if p.exists()]

    n_total = len(stack_paths)

    valid_paths = []
    for p in stack_paths:
        if check_patch_valid(p):
            valid_paths.append(p)

    n_valid = len(valid_paths)
    n_skipped = n_total - n_valid

    row = {
        "GFD_ID": gfd_id,
        "event_id": event_dir.name,
        "n_total_patches": n_total,
        "n_valid_patches": n_valid,
        "n_skipped_patches": n_skipped,
        "valid_patch_fraction": n_valid / n_total if n_total > 0 else 0.0,
    }

    if n_valid == 0:
        return row

    all_patch_features = []

    for start in range(0, n_valid, BATCH_SIZE):
        batch_paths = valid_paths[start:start + BATCH_SIZE]

        before_list = []
        after_list = []

        for stack_path in batch_paths:
            before_t, after_t = read_before_after(stack_path)
            before_list.append(before_t)
            after_list.append(after_t)

        before_batch = torch.stack(before_list, dim=0).to(DEVICE)
        after_batch = torch.stack(after_list, dim=0).to(DEVICE)

        with torch.no_grad():
            before_out = model(SAR_images=before_batch)
            after_out = model(SAR_images=after_batch)

        before_emb = before_out["SAR_GAP"]
        after_emb = after_out["SAR_GAP"]
        delta_emb = after_emb - before_emb

        # feature = before + after + delta
        features = torch.cat([before_emb, after_emb, delta_emb], dim=1)
        features = features.detach().cpu().numpy()

        all_patch_features.append(features)

    all_patch_features = np.concatenate(all_patch_features, axis=0)

    agg = aggregate_features(all_patch_features)

    for stat_name, values in agg.items():
        for i, v in enumerate(values):
            row[f"croma_{stat_name}_{i:04d}"] = float(v)

    return row


# =====================================================
# MAIN
# =====================================================
print("=" * 100)
print("Running CROMA feature extraction")
print("Device:", DEVICE)
print("Patch root:", PATCH_ROOT)
print("Weights:", WEIGHTS_PATH)
print("Output:", OUT_ALL)
print("=" * 100)

assert PATCH_ROOT.exists(), f"Patch root not found: {PATCH_ROOT}"
assert WEIGHTS_PATH.exists(), f"Weights not found: {WEIGHTS_PATH}"

print("Loading CROMA...")
model = PretrainedCROMA(
    pretrained_path=str(WEIGHTS_PATH),
    size="base",
    modality="SAR",
    image_resolution=IMAGE_SIZE,
)

model = model.to(DEVICE)
model.eval()

event_dirs = sorted([p for p in PATCH_ROOT.glob("event_*") if p.is_dir()])
print("Number of events:", len(event_dirs))

rows = []

for event_dir in tqdm(event_dirs):
    try:
        row = process_event(event_dir, model)
        rows.append(row)

        event_out = OUT_DIR / f"{event_dir.name}_croma_features.csv"
        pd.DataFrame([row]).to_csv(event_out, index=False)

    except Exception as e:
        print("ERROR in", event_dir.name, ":", e)
        rows.append(
            {
                "GFD_ID": extract_gfd_id(event_dir),
                "event_id": event_dir.name,
                "error": str(e),
            }
        )

df = pd.DataFrame(rows)
df = df.sort_values("GFD_ID")
df.to_csv(OUT_ALL, index=False)

print()
print("=" * 100)
print("DONE")
print("Saved:", OUT_ALL)
print("Shape:", df.shape)
print("Events:", len(df))
print("Nonzero valid events:", (df["n_valid_patches"] > 0).sum())
print("=" * 100)