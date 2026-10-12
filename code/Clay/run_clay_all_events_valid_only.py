from pathlib import Path
import os
import sys
import warnings
from collections import defaultdict

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

S1_ROOT = Path("/home/melissa/flood_project/S1_PATCHES_CLEAN")

OUTPUT_DIR = Path(
    "/home/melissa/flood_project/models_benchmark/Clay/flodis_outputs/clay_event_features_valid_only"
)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_CSV = OUTPUT_DIR / "all_events_clay_event_features_valid_only.csv"

SENSOR_NAME = "sentinel-1-rtc"
CHIP_SIZE = 256
MIN_VALID_FRACTION = 0.80
BATCH_SIZE = 4

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# =====================================================
# DATA HELPERS
# =====================================================
def fill_nan_per_band(arr):
    arr = arr.astype("float32").copy()

    for b in range(arr.shape[0]):
        band = arr[b]
        finite = np.isfinite(band)

        if finite.sum() == 0:
            band[:] = 0.0
        else:
            med = np.nanmedian(band[finite])
            band[~finite] = med

        arr[b] = band

    return arr


def get_patch_valid_fraction(path):
    with rasterio.open(path) as src:
        arr = src.read([1, 2, 3, 4]).astype("float32")

    finite_all = np.isfinite(arr).all(axis=0)
    return float(finite_all.mean())


def read_patch_before_after(path):
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

    with rasterio.open(path) as src:
        before = src.read([1, 2]).astype("float32")
        after = src.read([3, 4]).astype("float32")

    before = fill_nan_per_band(before)
    after = fill_nan_per_band(after)

    before = torch.from_numpy(before).unsqueeze(0)
    after = torch.from_numpy(after).unsqueeze(0)

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


def list_events():
    event_dirs = sorted([p for p in S1_ROOT.glob("event_*") if p.is_dir()])
    return event_dirs


def get_gfd_id(event_dir):
    return int(event_dir.name.replace("event_", ""))


def get_patch_paths(event_dir):
    return sorted(event_dir.glob("patch_*/S1_stack.tif"))


# =====================================================
# CLAY HELPERS
# =====================================================
def load_metadata():
    with open(METADATA_PATH, "r") as f:
        metadata = yaml.safe_load(f)

    if SENSOR_NAME not in metadata:
        print("Available sensors in metadata.yaml:")
        print(list(metadata.keys()))
        raise KeyError(f"{SENSOR_NAME} not found in metadata.yaml")

    return metadata


def get_metadata_tensors(metadata):
    sensor_meta = metadata[SENSOR_NAME]
    band_order = sensor_meta["band_order"]
    bands_meta = sensor_meta["bands"]

    print("Clay sensor:", SENSOR_NAME)
    print("Band order:", band_order)
    print("GSD:", sensor_meta.get("gsd"))

    wavelengths = []
    means = []
    stds = []

    for band in band_order:
        wavelengths.append(float(bands_meta["wavelength"][band]))
        means.append(float(bands_meta["mean"][band]))
        stds.append(float(bands_meta["std"][band]))

    wavelengths = torch.tensor(wavelengths, dtype=torch.float32, device=DEVICE)
    means = torch.tensor(means, dtype=torch.float32, device=DEVICE).view(1, -1, 1, 1)
    stds = torch.tensor(stds, dtype=torch.float32, device=DEVICE).view(1, -1, 1, 1)

    print("wavelengths:", wavelengths.detach().cpu().numpy())
    print("means:", means.flatten().detach().cpu().numpy())
    print("stds:", stds.flatten().detach().cpu().numpy())

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

    if hasattr(model, "model") and hasattr(model.model, "encoder"):
        model.model.encoder.mask_ratio = 0.0
        model.model.encoder.shuffle = False

    return model


def get_clay_encoder(model):
    if hasattr(model, "encoder"):
        return model.encoder

    if hasattr(model, "model") and hasattr(model.model, "encoder"):
        return model.model.encoder

    raise AttributeError("Could not find Clay encoder")


def extract_batch_embeddings(model, batch, wavelengths, means, stds):
    encoder = get_clay_encoder(model)

    if hasattr(encoder, "mask_ratio"):
        encoder.mask_ratio = 0.0
    if hasattr(encoder, "shuffle"):
        encoder.shuffle = False

    batch = batch.to(DEVICE)
    batch = (batch - means) / stds

    bsz = batch.shape[0]

    datacube = {
        "pixels": batch,
        "time": torch.zeros(bsz, 4, dtype=torch.float32, device=DEVICE),
        "latlon": torch.zeros(bsz, 4, dtype=torch.float32, device=DEVICE),
        "gsd": torch.tensor(10.0, dtype=torch.float32, device=DEVICE),
        "waves": wavelengths.to(DEVICE),
    }

    with torch.no_grad():
        out = encoder(datacube)

    if isinstance(out, (tuple, list)):
        encoded = out[0]
    elif torch.is_tensor(out):
        encoded = out
    else:
        raise RuntimeError(f"Unexpected encoder output type: {type(out)}")

    # encoded usually [B, 1025, 1024]
    # token 0 = CLS, tokens 1: = spatial patch tokens
    if encoded.ndim == 3:
        if encoded.shape[1] > 1:
            emb = encoded[:, 1:, :].mean(dim=1)
        else:
            emb = encoded.mean(dim=1)
    elif encoded.ndim == 2:
        emb = encoded
    else:
        emb = encoded.flatten(start_dim=1)

    return emb.detach().cpu().numpy()


def aggregate_embeddings(prefix, embeddings):
    """
    embeddings shape: [n_patches, dim]
    returns mean/std/min/max event features
    """

    out = {}

    if embeddings.shape[0] == 0:
        return out

    stats = {
        "mean": embeddings.mean(axis=0),
        "std": embeddings.std(axis=0),
        "min": embeddings.min(axis=0),
        "max": embeddings.max(axis=0),
    }

    dim = embeddings.shape[1]

    for stat_name, values in stats.items():
        for i in range(dim):
            out[f"clay_{prefix}_{i:04d}_event_{stat_name}"] = float(values[i])

    return out


# =====================================================
# EVENT PROCESSING
# =====================================================
def process_event(event_dir, model, wavelengths, means, stds):
    gfd_id = get_gfd_id(event_dir)
    patch_paths = get_patch_paths(event_dir)

    total_patches = len(patch_paths)
    valid_paths = []

    for path in patch_paths:
        try:
            valid_frac = get_patch_valid_fraction(path)
        except Exception:
            continue

        if valid_frac >= MIN_VALID_FRACTION:
            valid_paths.append(path)

    row = {
        "GFD_ID": gfd_id,
        "event_id": event_dir.name,
        "n_total_patches": total_patches,
        "n_valid_patches": len(valid_paths),
        "n_skipped_patches": total_patches - len(valid_paths),
        "valid_patch_fraction": len(valid_paths) / total_patches if total_patches > 0 else 0.0,
    }

    if len(valid_paths) == 0:
        print(f"WARNING: {event_dir.name} has zero valid patches")
        return row

    before_embeddings = []
    after_embeddings = []

    for start in range(0, len(valid_paths), BATCH_SIZE):
        batch_paths = valid_paths[start : start + BATCH_SIZE]

        before_list = []
        after_list = []

        for path in batch_paths:
            try:
                before, after = read_patch_before_after(path)
                before_list.append(before)
                after_list.append(after)
            except Exception as e:
                print(f"Skipping bad patch {path}: {repr(e)}")

        if len(before_list) == 0:
            continue

        before_batch = torch.stack(before_list, dim=0)
        after_batch = torch.stack(after_list, dim=0)

        before_emb = extract_batch_embeddings(model, before_batch, wavelengths, means, stds)
        after_emb = extract_batch_embeddings(model, after_batch, wavelengths, means, stds)

        before_embeddings.append(before_emb)
        after_embeddings.append(after_emb)

    if len(before_embeddings) == 0:
        print(f"WARNING: {event_dir.name} had valid paths but no embeddings")
        return row

    before_embeddings = np.concatenate(before_embeddings, axis=0)
    after_embeddings = np.concatenate(after_embeddings, axis=0)
    delta_embeddings = after_embeddings - before_embeddings

    row.update(aggregate_embeddings("before", before_embeddings))
    row.update(aggregate_embeddings("after", after_embeddings))
    row.update(aggregate_embeddings("delta", delta_embeddings))

    print(
        event_dir.name,
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
    print("Clay all-events Sentinel-1 feature extraction")
    print("Device:", DEVICE)
    print("S1 root:", S1_ROOT)
    print("Clay repo:", CLAY_ROOT)
    print("Checkpoint:", CHECKPOINT_PATH)
    print("Output:", OUTPUT_CSV)
    print("Batch size:", BATCH_SIZE)
    print("Min valid fraction:", MIN_VALID_FRACTION)

    if not S1_ROOT.exists():
        raise FileNotFoundError(S1_ROOT)

    if not CLAY_ROOT.exists():
        raise FileNotFoundError(CLAY_ROOT)

    if not CHECKPOINT_PATH.exists():
        raise FileNotFoundError(CHECKPOINT_PATH)

    if not METADATA_PATH.exists():
        raise FileNotFoundError(METADATA_PATH)

    metadata = load_metadata()
    wavelengths, means, stds = get_metadata_tensors(metadata)

    model = load_clay_model()

    event_dirs = list_events()
    print("Found events:", len(event_dirs))

    rows = []

    for event_dir in tqdm(event_dirs, desc="Events"):
        row = process_event(event_dir, model, wavelengths, means, stds)
        rows.append(row)

        # Save after every event, so progress is not lost
        df_partial = pd.DataFrame(rows).sort_values("GFD_ID")
        df_partial.to_csv(OUTPUT_CSV, index=False)

    df = pd.DataFrame(rows).sort_values("GFD_ID")
    df.to_csv(OUTPUT_CSV, index=False)

    print()
    print("=" * 100)
    print("DONE")
    print("Saved:", OUTPUT_CSV)
    print("Shape:", df.shape)
    print("Events:", df["GFD_ID"].nunique())
    print("Nonzero valid events:", int((df["n_valid_patches"] > 0).sum()))


if __name__ == "__main__":
    main()