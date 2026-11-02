"""Shared constants and loaders for the EuroSAT training data and Kaggle test set.

All arrays returned by this module use the Kaggle test-set layout:
12 bands, channels first (N, 12, 64, 64), uint16, in TEST_BANDS order.

The first call to ``load_train()`` / ``load_test()`` reads the raw files and
writes a single .npy cache to data/cache/, later calls load from that cache.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
EUROSAT_MS_DIR = PROJECT_ROOT / "data" / "eurosat" / "ms" / "EuroSAT_MS"
KAGGLE_DIR = PROJECT_ROOT / "data" / "kaggle"
TEST_DIR = KAGGLE_DIR / "testset" / "testset"
CACHE_DIR = PROJECT_ROOT / "data" / "cache"
SPLITS_DIR = PROJECT_ROOT / "artifacts" / "splits"

CLASSES = [
    "AnnualCrop",
    "Forest",
    "HerbaceousVegetation",
    "Highway",
    "Industrial",
    "Pasture",
    "PermanentCrop",
    "Residential",
    "River",
    "SeaLake",
]
CLASS_TO_ID = {name: i for i, name in enumerate(CLASSES)}

# Band order of the EuroSAT GeoTIFFs (Level-1C, 13 bands).
TRAIN_BANDS = ["B01", "B02", "B03", "B04", "B05", "B06", "B07",
               "B08", "B09", "B10", "B11", "B12", "B8A"]
# Band order of the Kaggle .npy files (Level-2A, 12 bands, no B10).
TEST_BANDS = ["B01", "B02", "B03", "B04", "B05", "B06", "B07",
              "B08", "B8A", "B09", "B11", "B12"]
# Index into TRAIN_BANDS for each band in TEST_BANDS: drops B10, moves B8A.
TRAIN_TO_TEST_IDX = [TRAIN_BANDS.index(b) for b in TEST_BANDS]

# Central wavelengths in nm (Sentinel-2A MSI), handy for spectral plots.
WAVELENGTH_NM = {
    "B01": 443, "B02": 490, "B03": 560, "B04": 665, "B05": 705, "B06": 740,
    "B07": 783, "B08": 842, "B8A": 865, "B09": 945, "B10": 1375,
    "B11": 1610, "B12": 2190,
}


def band_index(name: str) -> int:
    """Return the channel index of a band in the 12-band test layout."""
    return TEST_BANDS.index(name)


def read_train_tif(path: Path) -> np.ndarray:
    """Read one EuroSAT GeoTIFF and return it as (12, 64, 64) in test band order."""
    import rasterio

    with rasterio.open(path) as src:
        return src.read()[TRAIN_TO_TEST_IDX]


def read_test_npy(path: Path) -> np.ndarray:
    """Read one Kaggle test file (64, 64, 12) and return it as (12, 64, 64)."""
    return np.load(path).transpose(2, 0, 1)


def _patch_centers(paths: list[Path]) -> tuple[np.ndarray, np.ndarray]:
    """Return longitude/latitude of each GeoTIFF's centre (WGS84)."""
    import rasterio
    from rasterio.warp import transform

    lon, lat = np.empty(len(paths)), np.empty(len(paths))
    for i, path in enumerate(paths):
        with rasterio.open(path) as src:
            x, y = src.xy(src.height // 2, src.width // 2)
            xs, ys = transform(src.crs, "EPSG:4326", [x], [y])
        lon[i], lat[i] = xs[0], ys[0]
    return lon, lat


def train_metadata() -> pd.DataFrame:
    """Return one row per training image: path, label, label_id, lon, lat.

    Row order matches train.csv and the cached array from ``load_train()``.
    """
    meta_path = CACHE_DIR / "train_meta.csv"
    if meta_path.exists():
        return pd.read_csv(meta_path)

    meta = pd.read_csv(KAGGLE_DIR / "train.csv")
    meta["label_id"] = meta["label"].map(CLASS_TO_ID)
    print("Reading patch coordinates (one-time)...")
    meta["lon"], meta["lat"] = _patch_centers([EUROSAT_MS_DIR / p for p in meta["path"]])
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    meta.to_csv(meta_path, index=False)
    return meta


def load_train(mmap: bool = True) -> tuple[np.ndarray, pd.DataFrame]:
    """Return all training images (27000, 12, 64, 64) and their metadata."""
    from tqdm import tqdm

    meta = train_metadata()
    cache = CACHE_DIR / "train_12b.npy"
    if not cache.exists():
        images = np.lib.format.open_memmap(
            cache, mode="w+", dtype=np.uint16, shape=(len(meta), 12, 64, 64)
        )
        for i, path in enumerate(tqdm(meta["path"], desc="Caching train images")):
            images[i] = read_train_tif(EUROSAT_MS_DIR / path)
        images.flush()
        del images
    return np.load(cache, mmap_mode="r" if mmap else None), meta


def load_test(mmap: bool = True) -> tuple[np.ndarray, pd.DataFrame]:
    """Return all Kaggle test images (N, 12, 64, 64) ordered like test.csv."""
    from tqdm import tqdm

    ids = pd.read_csv(KAGGLE_DIR / "test.csv")
    cache = CACHE_DIR / "test_12b.npy"
    if not cache.exists():
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        images = np.lib.format.open_memmap(
            cache, mode="w+", dtype=np.uint16, shape=(len(ids), 12, 64, 64)
        )
        for i, test_id in enumerate(tqdm(ids["test_id"], desc="Caching test images")):
            images[i] = read_test_npy(TEST_DIR / f"test_{test_id}.npy")
        images.flush()
        del images
    return np.load(cache, mmap_mode="r" if mmap else None), ids


def load_train_l2a(mmap: bool = True, tag: str = "") -> tuple[np.ndarray, np.ndarray]:
    """Level-2A versions of the training images (scripts/download_l2a.py) and which ones exist.

    Returns (images (27000, 12, 64, 64) uint16 in TEST_BANDS order, ok (27000,) bool). Rows with
    ok = False have no clear Level-2A image and must not be used. ``tag`` picks a separate set
    downloaded with ``--tag`` (e.g. "summer" for E17b).
    """
    name = f"train_l2a_{tag}" if tag else "train_l2a"
    images = CACHE_DIR / f"{name}_12b.npy"
    status = CACHE_DIR / f"{name}_status.csv"
    if not images.exists() or not status.exists():
        raise FileNotFoundError(f"no {name} images yet: run scripts/download_l2a.py" + (f" --tag {tag}" if tag else ""))
    return np.load(images, mmap_mode="r" if mmap else None), pd.read_csv(status)["ok"].values.astype(bool)


def load_split(name: str = "split_seed42") -> pd.DataFrame:
    """Return the saved train/val/test split (one row per training image)."""
    return pd.read_csv(SPLITS_DIR / f"{name}.csv")
