"""Band-wise corrections that make EuroSAT (Level-1C) look like the Kaggle test set (Level-2A).

All matchers work on (N, 12, H, W) arrays in TEST_BANDS order and map training values to
test-like values, band by band. Fit them on the *training split* and the unlabelled Kaggle
test images, then apply them to every EuroSAT image (train, val, internal test).
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from src.data import band_index

WATER_NIR_THRESHOLD = 500  # images with mean B08 below this are treated as water


def land_mask(images: np.ndarray) -> np.ndarray:
    """True for images that are not water-like (mean NIR >= WATER_NIR_THRESHOLD)."""
    nir = images[:, band_index("B08")].mean(axis=(1, 2))
    return nir >= WATER_NIR_THRESHOLD


def sample_pixels(images: np.ndarray, n_images: int = 4000, pixel_step: int = 4,
                  seed: int = 0) -> np.ndarray:
    """Random subset of pixels per band, shape (12, n). Keeps quantile fitting fast."""
    rng = np.random.default_rng(seed)
    idx = np.sort(rng.choice(len(images), min(n_images, len(images)), replace=False))
    x = np.asarray(images[idx], dtype=np.float32)[:, :, ::pixel_step, ::pixel_step]
    return x.transpose(1, 0, 2, 3).reshape(x.shape[1], -1)


@dataclass
class HistogramMatcher:
    """Per-band quantile mapping: a training value at quantile q becomes the test value at q.

    Beyond the fitted range (outside the 0.5th-99.5th percentile) the mapping continues in a
    straight line with the curve's average slope, so values that are rare in the fitting data
    (e.g. water when fitted on land only) are still mapped sensibly instead of being clamped
    or pushed below zero by a steep tail.
    """

    source_q: list[list[float]]  # per band: quantiles of the training data
    target_q: list[list[float]]  # per band: quantiles of the test data
    name: str = "histogram"

    @classmethod
    def fit(cls, source: np.ndarray, target: np.ndarray, n_quantiles: int = 199,
            name: str = "histogram") -> "HistogramMatcher":
        """``source`` and ``target`` are (N, 12, H, W) image arrays."""
        q = np.linspace(0.005, 0.995, n_quantiles)
        src, tgt = sample_pixels(source), sample_pixels(target)
        source_q = np.quantile(src, q, axis=1).T
        target_q = np.quantile(tgt, q, axis=1).T
        # np.interp needs strictly increasing x values
        source_q = np.maximum.accumulate(source_q + np.arange(n_quantiles) * 1e-6, axis=1)
        return cls(source_q.tolist(), target_q.tolist(), name)

    def apply_band(self, x: np.ndarray, b: int) -> np.ndarray:
        sq, tq = np.asarray(self.source_q[b]), np.asarray(self.target_q[b])
        y = np.interp(x, sq, tq)
        slope = (tq[-1] - tq[0]) / max(sq[-1] - sq[0], 1e-6)
        y = np.where(x < sq[0], tq[0] + (x - sq[0]) * slope, y)
        y = np.where(x > sq[-1], tq[-1] + (x - sq[-1]) * slope, y)
        return y

    def __call__(self, images: np.ndarray, chunk: int = 2048) -> np.ndarray:
        """Map (N, 12, H, W) images; returns uint16."""
        out = np.empty(images.shape, dtype=np.uint16)
        for start in range(0, len(images), chunk):
            x = np.asarray(images[start:start + chunk], dtype=np.float32)
            for b in range(x.shape[1]):
                x[:, b] = self.apply_band(x[:, b], b)
            out[start:start + chunk] = np.clip(np.rint(x), 0, 65535)
        return out

    def inverse(self) -> "HistogramMatcher":
        """The mapping from test-like values back to training-like values."""
        tq = np.maximum.accumulate(np.asarray(self.target_q) + np.arange(len(self.target_q[0])) * 1e-6, axis=1)
        return HistogramMatcher(tq.tolist(), self.source_q, self.name + "_inverse")

    def save(self, path: Path) -> None:
        Path(path).write_text(json.dumps({"type": "histogram", **asdict(self)}))


@dataclass
class LinearPercentileMatcher:
    """Per-band y = a*x + b that maps the training 2nd/98th percentile to the test ones.

    This is the harmonisation used by last year's team M.A.R.S (reference method).
    """

    a: list[float]
    b: list[float]
    name: str = "linear_p2_p98"

    @classmethod
    def fit(cls, source: np.ndarray, target: np.ndarray, name: str = "linear_p2_p98"):
        src, tgt = sample_pixels(source), sample_pixels(target)
        s2, s98 = np.percentile(src, [2, 98], axis=1)
        t2, t98 = np.percentile(tgt, [2, 98], axis=1)
        a = (t98 - t2) / (s98 - s2)
        return cls(a.tolist(), (t2 - a * s2).tolist(), name)

    def __call__(self, images: np.ndarray, chunk: int = 2048) -> np.ndarray:
        a = np.asarray(self.a, dtype=np.float32).reshape(1, -1, 1, 1)
        b = np.asarray(self.b, dtype=np.float32).reshape(1, -1, 1, 1)
        out = np.empty(images.shape, dtype=np.uint16)
        for start in range(0, len(images), chunk):
            x = np.asarray(images[start:start + chunk], dtype=np.float32)
            out[start:start + chunk] = np.clip(np.rint(a * x + b), 0, 65535)
        return out

    def save(self, path: Path) -> None:
        Path(path).write_text(json.dumps({"type": "linear", **asdict(self)}))


def load_matcher(path: Path):
    """Load a matcher saved with ``.save()``."""
    data = json.loads(Path(path).read_text())
    kind = data.pop("type")
    if kind == "split":
        return SplitMatcher(HistogramMatcher(**data["land"]), HistogramMatcher(**data["water"]), data["name"])
    return HistogramMatcher(**data) if kind == "histogram" else LinearPercentileMatcher(**data)


@dataclass
class SplitMatcher:
    """Separate histogram matching for land and water images.

    Each image is assigned to land or water by its own mean NIR (``land_mask``), which works
    for training images without using labels. Only training images are transformed, so the
    Kaggle test images are never touched.
    """

    land: HistogramMatcher
    water: HistogramMatcher
    name: str = "hist_split"

    @classmethod
    def fit(cls, source: np.ndarray, target: np.ndarray, name: str = "hist_split") -> "SplitMatcher":
        ls, lt = land_mask(source), land_mask(target)
        return cls(HistogramMatcher.fit(source[ls], target[lt], name="land"),
                   HistogramMatcher.fit(source[~ls], target[~lt], name="water"), name)

    def __call__(self, images: np.ndarray, chunk: int = 2048) -> np.ndarray:
        images = np.asarray(images)
        is_land = land_mask(images)
        out = np.empty(images.shape, dtype=np.uint16)
        if is_land.any():
            out[is_land] = self.land(images[is_land], chunk)
        if (~is_land).any():
            out[~is_land] = self.water(images[~is_land], chunk)
        return out

    def save(self, path: Path) -> None:
        Path(path).write_text(json.dumps({"type": "split", "name": self.name,
                                          "land": asdict(self.land), "water": asdict(self.water)}))
