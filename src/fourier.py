"""Fourier-domain methods against the remaining train/test difference.

After ``hist_split``, brightness and colour of training and test images match, but their
spatial frequency content does not (``05_fourier_analysis``): the 10 m bands differ in the large
structures (low frequencies, haze), the 20 m bands in medium detail (different upsampling).

- A · ``FourierDomainAugment``: training augmentation, mixes the low-frequency amplitudes of a
  training image with those of a random Kaggle image (Fourier Domain Adaptation, Yang & Soatto 2020).
- B · ``SpectralMatcher``: preprocessing, rescales each training image's spectrum per band so its
  average amplitude profile matches the Kaggle images (like histogram matching, but per frequency).
- C · ``LowFreqEqualizer``: a fixed first layer of the model, brings the low-frequency energy of
  every image (training and test) to the same level per band.

All work per band on (N, C, 64, 64) arrays. Amplitudes are scaled, phases (the image content,
edges and shapes) are kept.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn

from src.preprocessing import land_mask

SIZE = 64


def radius_grid(size: int = SIZE) -> np.ndarray:
    """Spatial frequency (cycles per pixel) of every FFT coefficient, shape (size, size)."""
    f = np.fft.fftfreq(size)
    return np.sqrt(f[:, None] ** 2 + f[None, :] ** 2)


RADIUS_EDGES = np.linspace(0, 0.71, 25)  # 24 rings from 0 to the highest frequency


def radial_amplitude(images: np.ndarray, edges: np.ndarray = RADIUS_EDGES, chunk: int = 1024) -> np.ndarray:
    """Mean amplitude per band and frequency ring, shape (C, len(edges) - 1).

    Each image's mean is removed first, so only spatial structure counts.
    """
    r = radius_grid(images.shape[-1])
    ring = np.digitize(r, edges[1:-1])  # ring index of every coefficient
    power = np.zeros((images.shape[1], len(edges) - 1))
    for start in range(0, len(images), chunk):
        x = np.asarray(images[start:start + chunk], dtype=np.float32)
        x = x - x.mean(axis=(2, 3), keepdims=True)
        p = (np.abs(np.fft.fft2(x)) ** 2).sum(axis=0)  # (C, H, W)
        for k in range(power.shape[1]):
            power[:, k] += p[:, ring == k].mean(axis=1)
    return np.sqrt(power / len(images))


# ----------------------------------------------------------------------------
# B · Spectral matching (preprocessing of training images)
# ----------------------------------------------------------------------------

@dataclass
class SpectralMatcher:
    """Per band and frequency ring: multiply amplitudes by (test amplitude / train amplitude).

    Fitted separately for land and water images (like ``hist_split``); each image uses the curve
    of its own type. The mean of every image (zero frequency) is not changed.
    """

    land_ratio: list[list[float]]
    water_ratio: list[list[float]]
    edges: list[float]
    max_factor: float = 2.0

    @classmethod
    def fit(cls, source: np.ndarray, target: np.ndarray, max_factor: float = 2.0) -> "SpectralMatcher":
        ls, lt = land_mask(source), land_mask(target)
        land = radial_amplitude(target[lt]) / radial_amplitude(source[ls])
        water = radial_amplitude(target[~lt]) / radial_amplitude(source[~ls])
        clip = lambda a: np.clip(a, 1 / max_factor, max_factor).tolist()
        return cls(clip(land), clip(water), RADIUS_EDGES.tolist(), max_factor)

    def _gain(self, ratio: list[list[float]], size: int) -> np.ndarray:
        """(C, size, size) multiplication factors, interpolated between ring centres."""
        edges = np.asarray(self.edges)
        centres = (edges[:-1] + edges[1:]) / 2
        r = radius_grid(size)
        gain = np.stack([np.interp(r, centres, band) for band in np.asarray(ratio)])
        gain[:, 0, 0] = 1.0  # keep the image mean
        return gain.astype(np.float32)

    def __call__(self, images: np.ndarray, chunk: int = 1024) -> np.ndarray:
        images = np.asarray(images)
        is_land = land_mask(images)
        gains = {True: self._gain(self.land_ratio, images.shape[-1]),
                 False: self._gain(self.water_ratio, images.shape[-1])}
        out = np.empty(images.shape, dtype=np.uint16)
        for start in range(0, len(images), chunk):
            x = np.asarray(images[start:start + chunk], dtype=np.float32)
            g = np.where(is_land[start:start + chunk, None, None, None], gains[True], gains[False])
            y = np.fft.ifft2(np.fft.fft2(x) * g).real
            out[start:start + chunk] = np.clip(np.rint(y), 0, 65535)
        return out

    def save(self, path: Path) -> None:
        Path(path).write_text(json.dumps(asdict(self)))

    @classmethod
    def load(cls, path: Path) -> "SpectralMatcher":
        return cls(**json.loads(Path(path).read_text()))


# ----------------------------------------------------------------------------
# C · Low-frequency equaliser (fixed first layer of the model)
# ----------------------------------------------------------------------------

def low_freq_mask(size: int = SIZE, radius: float = 0.05) -> np.ndarray:
    """True for the low frequencies 0 < r <= radius (the zero frequency, i.e. the mean, excluded)."""
    r = radius_grid(size)
    return (r > 0) & (r <= radius)


def low_freq_energy(images: np.ndarray, radius: float = 0.05) -> np.ndarray:
    """RMS amplitude of the low frequencies per image and band, shape (N, C)."""
    mask = low_freq_mask(images.shape[-1], radius)
    x = np.asarray(images, dtype=np.float32)
    return np.sqrt((np.abs(np.fft.fft2(x))[..., mask] ** 2).mean(axis=-1))


class LowFreqEqualizer(nn.Module):
    """Scales the low frequencies of every image so their energy equals a fixed per-band level.

    Large-scale contrast (which haze reduces) becomes the same for every image, in training and
    at test time; finer detail and the image mean are untouched. ``template`` is the median
    low-frequency energy of the (normalised) training images. No learnable parameters.
    """

    def __init__(self, template: list[float], radius: float = 0.05, max_factor: float = 3.0, size: int = SIZE):
        super().__init__()
        self.radius = radius
        self.max_factor = max_factor
        self.register_buffer("template", torch.tensor(template, dtype=torch.float32).view(1, -1, 1, 1))
        self.register_buffer("mask", torch.from_numpy(low_freq_mask(size, radius)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        f = torch.fft.fft2(x)
        energy = torch.sqrt((f.abs()[..., self.mask] ** 2).mean(dim=-1) + 1e-8)  # (N, C)
        scale = (self.template.view(1, -1) / energy).clamp(1 / self.max_factor, self.max_factor)
        gain = torch.where(self.mask, scale[..., None, None], torch.ones_like(scale)[..., None, None])
        return torch.fft.ifft2(f * gain).real

    def config(self) -> dict:
        return {"template": self.template.flatten().tolist(), "radius": self.radius, "max_factor": self.max_factor}


def equalize_low_freq_np(images: np.ndarray, template: np.ndarray, radius: float = 0.05,
                         max_factor: float = 3.0) -> np.ndarray:
    """NumPy version of LowFreqEqualizer, for measurements on raw images (float32 output)."""
    mask = low_freq_mask(images.shape[-1], radius)
    x = np.asarray(images, dtype=np.float32)
    f = np.fft.fft2(x)
    energy = np.sqrt((np.abs(f[..., mask]) ** 2).mean(axis=-1)) + 1e-6
    scale = np.clip(np.asarray(template)[None, :] / energy, 1 / max_factor, max_factor)
    gain = np.where(mask, scale[..., None, None], 1.0)
    return np.fft.ifft2(f * gain).real.astype(np.float32)


# ----------------------------------------------------------------------------
# A · Fourier Domain Adaptation (training augmentation)
# ----------------------------------------------------------------------------

class FourierDomainAugment:
    """FDA: blend a training image's low-frequency amplitudes with those of a random Kaggle image.

    Applied to raw values before normalisation. Each training image gets a partner of the same
    type (land or water, by mean NIR), so water statistics are not transferred to land images.
    With probability ``p``; blend weight λ ~ U(0, max_blend); the image mean and the phases are kept.
    """

    def __init__(self, target_images: torch.Tensor, nir_index: int, radius: float = 0.05,
                 p: float = 0.5, max_blend: float = 1.0, water_nir_threshold: float = 500):
        self.target = target_images
        self.nir_index = nir_index
        self.radius, self.p, self.max_blend = radius, p, max_blend
        self.threshold = water_nir_threshold
        target_land = target_images[:, nir_index].float().mean(dim=(1, 2)) >= water_nir_threshold
        self.land_idx = torch.nonzero(target_land).flatten()
        self.water_idx = torch.nonzero(~target_land).flatten()
        self.mask = torch.from_numpy(low_freq_mask(target_images.shape[-1], radius)).to(target_images.device)

    def config(self) -> dict:
        return {"method": "fda", "radius": self.radius, "p": self.p, "max_blend": self.max_blend}

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        n, dev = x.shape[0], x.device
        is_land = x[:, self.nir_index].mean(dim=(1, 2)) >= self.threshold
        pick_land = self.land_idx[torch.randint(len(self.land_idx), (n,), device=dev)]
        pick_water = self.water_idx[torch.randint(len(self.water_idx), (n,), device=dev)]
        partner = self.target[torch.where(is_land, pick_land, pick_water)].float()

        fx, ft = torch.fft.fft2(x), torch.fft.fft2(partner)
        amp_x, phase = fx.abs(), torch.angle(fx)
        lam = torch.rand(n, 1, 1, 1, device=dev) * self.max_blend
        lam = lam * (torch.rand(n, 1, 1, 1, device=dev) < self.p).float()
        amp = torch.where(self.mask, (1 - lam) * amp_x + lam * ft.abs(), amp_x)
        out = torch.fft.ifft2(torch.polar(amp, phase)).real
        return out.clamp_min(0)
