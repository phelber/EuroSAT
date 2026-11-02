"""Spectral indices as extra input channels (E13).

The network gets, next to the 12 bands, a few normalised differences that directly answer
"is this pixel water, vegetation or built-up/bare?":

    NDVI  = (B08 - B04) / (B08 + B04)   green vegetation
    NDWI  = (B03 - B08) / (B03 + B08)   open water (McFeeters)
    MNDWI = (B03 - B11) / (B03 + B11)   water, also separates it from built-up areas
    NDBI  = (B11 - B08) / (B11 + B08)   built-up and bare soil
    NDRE  = (B8A - B05) / (B8A + B05)   red-edge, vegetation type/vigour (crops vs pasture vs forest)

The labels stay the 10 EuroSAT classes (Kaggle only scores those); the indices only help the
model to *see* water/vegetation/built-up. They are ratios, so a common brightness factor
cancels; an additive haze offset does not, which is why they are computed after ``hist_split``.

``SpectralIndexLayer`` sits in front of the network (like ``LowFreqEqualizer``). It receives the
normalised bands, undoes the normalisation, computes the indices, standardises them with stored
training statistics and appends them as channels. No learnable parameters.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn

INDEX_BANDS = {
    "NDVI": ("B08", "B04"),
    "NDWI": ("B03", "B08"),
    "MNDWI": ("B03", "B11"),
    "NDBI": ("B11", "B08"),
    "NDRE": ("B8A", "B05"),
}
DEFAULT_INDICES = ["NDVI", "NDWI", "MNDWI", "NDBI", "NDRE"]
EPS = 1.0  # in reflectance x 10,000 units; avoids 0/0 on dark pixels


def _pairs(bands: list[str], indices: list[str]) -> list[tuple[int, int]]:
    missing = {b for name in indices for b in INDEX_BANDS[name]} - set(bands)
    if missing:
        raise ValueError(f"indices {indices} need bands {sorted(missing)}, which are not in the input")
    return [(bands.index(a), bands.index(b)) for a, b in (INDEX_BANDS[n] for n in indices)]


def compute_indices_np(images: np.ndarray, bands: list[str], indices: list[str] = DEFAULT_INDICES) -> np.ndarray:
    """Raw images (N, C, H, W) -> indices (N, len(indices), H, W) in [-1, 1], float32."""
    out = np.empty((len(images), len(indices), *images.shape[2:]), dtype=np.float32)
    for k, (i, j) in enumerate(_pairs(bands, indices)):
        a, b = images[:, i].astype(np.float32), images[:, j].astype(np.float32)
        out[:, k] = np.clip((a - b) / np.maximum(a + b, EPS), -1, 1)
    return out


def index_stats(images: np.ndarray, bands: list[str], indices: list[str] = DEFAULT_INDICES,
                chunk: int = 2048) -> tuple[list[float], list[float]]:
    """Per-index mean and std over raw training images, in chunks."""
    total = np.zeros(len(indices))
    total_sq = np.zeros(len(indices))
    n = 0
    for start in range(0, len(images), chunk):
        idx = compute_indices_np(np.asarray(images[start:start + chunk]), bands, indices).astype(np.float64)
        total += idx.sum(axis=(0, 2, 3))
        total_sq += (idx ** 2).sum(axis=(0, 2, 3))
        n += idx.shape[0] * idx.shape[2] * idx.shape[3]
    mean = total / n
    std = np.sqrt(np.maximum(total_sq / n - mean ** 2, 1e-8))
    return mean.round(5).tolist(), std.round(5).tolist()


class SpectralIndexLayer(nn.Module):
    """Normalised bands (N, C, H, W) -> bands + standardised indices (N, C + len(indices), H, W)."""

    def __init__(self, bands: list[str], band_mean: list[float], band_std: list[float],
                 index_mean: list[float], index_std: list[float], indices: list[str] = DEFAULT_INDICES):
        super().__init__()
        self.bands, self.indices = list(bands), list(indices)
        pairs = _pairs(self.bands, self.indices)
        self.register_buffer("a_idx", torch.tensor([p[0] for p in pairs]))
        self.register_buffer("b_idx", torch.tensor([p[1] for p in pairs]))
        self.register_buffer("band_mean", torch.tensor(band_mean, dtype=torch.float32).view(1, -1, 1, 1))
        self.register_buffer("band_std", torch.tensor(band_std, dtype=torch.float32).view(1, -1, 1, 1))
        self.register_buffer("index_mean", torch.tensor(index_mean, dtype=torch.float32).view(1, -1, 1, 1))
        self.register_buffer("index_std", torch.tensor(index_std, dtype=torch.float32).view(1, -1, 1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raw = x * self.band_std + self.band_mean
        a, b = raw[:, self.a_idx], raw[:, self.b_idx]
        idx = ((a - b) / (a + b).clamp_min(EPS)).clamp(-1, 1)
        return torch.cat([x, (idx - self.index_mean) / self.index_std], dim=1)

    def config(self) -> dict:
        return {"type": "spectral_indices", "bands": self.bands, "indices": self.indices,
                "band_mean": self.band_mean.flatten().tolist(), "band_std": self.band_std.flatten().tolist(),
                "index_mean": self.index_mean.flatten().tolist(), "index_std": self.index_std.flatten().tolist()}
