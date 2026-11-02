"""Topological features of images (persistent homology), for notebooks/data/07_topological_features.

Each image is turned into a few single-channel "height maps" (NIR, visible brightness, NDVI, edge
strength). Every channel is rank-normalised per image, so its values are the pixel's rank in [0, 1].
Persistent homology of a rank-normalised single band does not change under any increasing
brightness/contrast change of that band (such as haze removal or histogram matching): only the
order of the pixels matters.

For each channel, the sublevel filtration (dark regions grow first) and the superlevel filtration
(bright regions grow first) are computed with gudhi's cubical complexes, in dimension 0
(connected regions) and 1 (holes/loops), and each diagram is summarised by a fixed-length vector.
"""

from __future__ import annotations

import gudhi
import numpy as np
from joblib import Parallel, delayed
from scipy import ndimage
from scipy.stats import rankdata

from src.data import band_index

# (channel, direction): "sub" = sublevel, "super" = superlevel (computed on 1 - x)
FILTRATIONS = [("nir", "sub"), ("nir", "super"), ("vis", "sub"), ("vis", "super"),
               ("ndvi", "sub"), ("ndvi", "super"), ("edges", "super")]
DIMS = (0, 1)
BETTI_GRID = np.linspace(0.05, 0.95, 8)
PERSISTENCE_LEVELS = (0.02, 0.1, 0.25)
SUMMARY_NAMES = ([f"n_pers>{t}" for t in PERSISTENCE_LEVELS]
                 + ["total_pers", "max_pers", "entropy", "mean_birth", "mean_mid"]
                 + [f"betti@{t:.2f}" for t in BETTI_GRID])


def rank_normalise(channel: np.ndarray) -> np.ndarray:
    """Pixel ranks scaled to [0, 1] (ties get their average rank)."""
    r = rankdata(channel.ravel(), method="average").reshape(channel.shape)
    return (r - 1) / max(r.size - 1, 1)


def channels(image: np.ndarray) -> dict[str, np.ndarray]:
    """The four height maps of one (12, 64, 64) image, each rank-normalised."""
    x = image.astype(np.float32)
    nir, red = x[band_index("B08")], x[band_index("B04")]
    vis = x[[band_index(b) for b in ("B02", "B03", "B04")]].mean(0)
    ndvi = (nir - red) / (nir + red + 1)
    vis_n = rank_normalise(vis)
    edges = np.hypot(ndimage.sobel(vis_n, 0), ndimage.sobel(vis_n, 1))
    return {"nir": rank_normalise(nir), "vis": vis_n, "ndvi": rank_normalise(ndvi),
            "edges": rank_normalise(edges)}


def diagram(height: np.ndarray, dim: int) -> np.ndarray:
    """Finite (birth, death) pairs of the sublevel filtration of ``height`` in dimension ``dim``."""
    cc = gudhi.CubicalComplex(top_dimensional_cells=height)
    cc.compute_persistence()
    d = np.asarray(cc.persistence_intervals_in_dimension(dim)).reshape(-1, 2)
    return d[np.isfinite(d[:, 1])]


def summarise(d: np.ndarray) -> np.ndarray:
    """Fixed-length summary of one persistence diagram (see SUMMARY_NAMES)."""
    if len(d) == 0:
        return np.zeros(len(SUMMARY_NAMES), dtype=np.float32)
    birth, death = d[:, 0], d[:, 1]
    pers = death - birth
    total = pers.sum()
    p = pers[pers > 0] / total if total > 0 else np.array([1.0])
    betti = [((birth <= t) & (death > t)).sum() for t in BETTI_GRID]
    return np.array([*(np.sum(pers > t) for t in PERSISTENCE_LEVELS), total, pers.max(),
                     -(p * np.log(p)).sum(), birth.mean(), ((birth + death) / 2).mean(), *betti],
                    dtype=np.float32)


def image_features(image: np.ndarray) -> np.ndarray:
    ch = channels(image)
    out = []
    for name, direction in FILTRATIONS:
        h = ch[name] if direction == "sub" else 1 - ch[name]
        for dim in DIMS:
            out.append(summarise(diagram(h, dim)))
    return np.concatenate(out)


def feature_names() -> list[str]:
    return [f"{c}_{d}_H{k}_{s}" for c, d in FILTRATIONS for k in DIMS for s in SUMMARY_NAMES]


def topological_features(images: np.ndarray, n_jobs: int = -1, chunk: int = 256) -> np.ndarray:
    """Features of all images (N, 12, 64, 64) -> (N, n_features), computed in parallel."""
    def run(block):
        return np.stack([image_features(img) for img in block])
    blocks = [np.asarray(images[i:i + chunk]) for i in range(0, len(images), chunk)]
    return np.concatenate(Parallel(n_jobs=n_jobs)(delayed(run)(b) for b in blocks))


def spectral_features(images: np.ndarray) -> np.ndarray:
    """Baseline without geometry: per band mean, std and 10/50/90th percentile (12 x 5 values)."""
    x = np.asarray(images, dtype=np.float32).reshape(len(images), images.shape[1], -1)
    q = np.percentile(x, [10, 50, 90], axis=2).transpose(1, 2, 0)
    return np.concatenate([x.mean(2), x.std(2), q.reshape(len(x), -1)], axis=1)
