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

# Use the real Clay repo cloned in /dev/shm
sys.path.insert(0, "/dev/shm/melissa/Clay_model")

import numpy as np
import rasterio
import torch
import torch.nn.functional as F
import yaml

warnings.filterwarnings("ignore")


# =====================================================
# PATHS
# =====================================================
CLAY_ROOT = Path("/dev/shm/melissa/Clay_model")
CHECKPOINT_PATH = Path("/dev/shm/melissa/clay_checkpoints/clay-v1.5.ckpt")
METADATA_PATH = CLAY_ROOT / "configs" / "metadata.yaml"

PATCH_PATH = Path(
    "/home/melissa/flood_project/S1_PATCHES_CLEAN/event_4211/patch_018/S1_stack.tif"
)

SENSOR_NAME = "sentinel-1-rtc"
CHIP_SIZE = 256

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# =====================================================
# HELPERS
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


def read_s1_pair(path, mode):
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
        if mode == "before":
            arr = src.read([1, 2]).astype("float32")
        elif mode == "after":
            arr = src.read([3, 4]).astype("float32")
        else:
            raise ValueError(mode)

    print(mode, "raw shape:", arr.shape)
    print(mode, "finite fraction:", float(np.isfinite(arr).mean()))

    arr = fill_nan_per_band(arr)

    x = torch.from_numpy(arr).unsqueeze(0).to(DEVICE)

    x = F.interpolate(
        x,
        size=(CHIP_SIZE, CHIP_SIZE),
        mode="bilinear",
        align_corners=False,
    )

    return x


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
        wavelength_value = float(bands_meta["wavelength"][band])

        wavelengths.append(wavelength_value)
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

    # Clay checkpoint internally expects configs/metadata.yaml relative to repo root.
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
        print("Using model.encoder")
        return model.encoder

    if hasattr(model, "model") and hasattr(model.model, "encoder"):
        print("Using model.model.encoder")
        return model.model.encoder

    print("Available model attributes:")
    print([a for a in dir(model) if not a.startswith("_")][:100])
    raise AttributeError("Could not find Clay encoder")


def extract_embedding(model, chip, wavelengths, label):
    batch_size = chip.shape[0]
    encoder = get_clay_encoder(model)

    if hasattr(encoder, "mask_ratio"):
        encoder.mask_ratio = 0.0
    if hasattr(encoder, "shuffle"):
        encoder.shuffle = False

    datacube = {
        "pixels": chip,  # [B, C, H, W]
        "time": torch.zeros(batch_size, 4, dtype=torch.float32, device=DEVICE),
        "latlon": torch.zeros(batch_size, 4, dtype=torch.float32, device=DEVICE),
        "gsd": torch.tensor(10.0, dtype=torch.float32, device=DEVICE),
        "waves": wavelengths.to(DEVICE),  # [C]
    }

    with torch.no_grad():
        out = encoder(datacube)

    if isinstance(out, (tuple, list)):
        encoded = out[0]
    elif torch.is_tensor(out):
        encoded = out
    else:
        raise RuntimeError(f"Unexpected encoder output type: {type(out)}")

    print(label, "encoded token shape:", tuple(encoded.shape))

    # IMPORTANT:
    # encoded shape is usually [B, 1025, 1024]
    # token 0 = CLS token
    # tokens 1: = spatial patch tokens
    # We use mean spatial tokens because CLS gave identical before/after embeddings.
    if encoded.ndim == 3:
        if encoded.shape[1] > 1:
            emb = encoded[:, 1:, :].mean(dim=1)
        else:
            emb = encoded.mean(dim=1)
    elif encoded.ndim == 2:
        emb = encoded
    else:
        emb = encoded.flatten(start_dim=1)

    print(label, "final embedding shape:", tuple(emb.shape))

    return emb.detach().cpu().numpy()


# =====================================================
# MAIN
# =====================================================
def main():
    print("=" * 100)
    print("Clay one-patch Sentinel-1 test")
    print("Device:", DEVICE)
    print("Clay repo:", CLAY_ROOT)
    print("Metadata:", METADATA_PATH)
    print("Checkpoint:", CHECKPOINT_PATH)
    print("Patch:", PATCH_PATH)

    if not CLAY_ROOT.exists():
        raise FileNotFoundError(CLAY_ROOT)

    if not CHECKPOINT_PATH.exists():
        raise FileNotFoundError(CHECKPOINT_PATH)

    if not METADATA_PATH.exists():
        raise FileNotFoundError(METADATA_PATH)

    if not PATCH_PATH.exists():
        raise FileNotFoundError(PATCH_PATH)

    metadata = load_metadata()
    wavelengths, means, stds = get_metadata_tensors(metadata)

    model = load_clay_model()

    before = read_s1_pair(PATCH_PATH, "before")
    after = read_s1_pair(PATCH_PATH, "after")

    before_norm = (before - means) / stds
    after_norm = (after - means) / stds

    print("before norm min/max:", float(before_norm.min()), float(before_norm.max()))
    print("after norm min/max:", float(after_norm.min()), float(after_norm.max()))

    emb_before = extract_embedding(model, before_norm, wavelengths, "before")
    emb_after = extract_embedding(model, after_norm, wavelengths, "after")
    emb_delta = emb_after - emb_before

    print()
    print("=" * 100)
    print("SUCCESS")
    print("before embedding:", emb_before.shape)
    print("after embedding:", emb_after.shape)
    print("delta embedding:", emb_delta.shape)
    print("before mean/std:", float(emb_before.mean()), float(emb_before.std()))
    print("after mean/std:", float(emb_after.mean()), float(emb_after.std()))
    print("delta mean/std:", float(emb_delta.mean()), float(emb_delta.std()))

    if float(emb_delta.std()) == 0.0:
        print()
        print("WARNING: delta std is still 0.0. Before/after embeddings are identical.")
    else:
        print()
        print("OK: before/after Clay embeddings are different.")


if __name__ == "__main__":
    main()