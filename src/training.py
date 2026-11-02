"""Training, evaluation, prediction and saving runs.

Images are kept on the GPU as int16 for the whole run (values never exceed 32,767),
and augmentation runs on the GPU too. This keeps the GPU busy without DataLoader
worker processes.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn

from src.data import CLASSES, PROJECT_ROOT
from src.utils import synchronize

RUNS_DIR = PROJECT_ROOT / "runs"


# ----------------------------------------------------------------------------
# Data on the device
# ----------------------------------------------------------------------------

def to_device_images(images: np.ndarray, device: torch.device) -> torch.Tensor:
    """(N, C, 64, 64) uint16 array -> int16 tensor on ``device``."""
    clipped = np.minimum(images, np.iinfo(np.int16).max).astype(np.int16)
    return torch.from_numpy(clipped).to(device)


@dataclass
class Normalizer:
    """Per-band standardisation: (x - mean) / std."""

    mean: list[float]
    std: list[float]

    @classmethod
    def fit(cls, images: np.ndarray, chunk: int = 2048) -> "Normalizer":
        """Compute per-band mean and std of a (N, C, H, W) array, in chunks."""
        total = np.zeros(images.shape[1])
        total_sq = np.zeros(images.shape[1])
        for start in range(0, len(images), chunk):
            x = np.asarray(images[start:start + chunk], dtype=np.float64)
            total += x.sum(axis=(0, 2, 3))
            total_sq += (x ** 2).sum(axis=(0, 2, 3))
        n = len(images) * images.shape[2] * images.shape[3]
        mean = total / n
        std = np.sqrt(total_sq / n - mean ** 2)
        return cls(mean=mean.tolist(), std=std.tolist())

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        mean = torch.tensor(self.mean, device=x.device).view(1, -1, 1, 1)
        std = torch.tensor(self.std, device=x.device).view(1, -1, 1, 1)
        return (x.float() - mean) / std


def augment(x: torch.Tensor) -> torch.Tensor:
    """Random 90° rotation and horizontal/vertical flip, independently per image."""
    n = x.shape[0]
    k = torch.randint(0, 4, (n,), device=x.device)
    out = x.clone()
    for r in range(1, 4):
        mask = k == r
        if mask.any():
            out[mask] = torch.rot90(x[mask], r, dims=(2, 3))
    flip_h = (torch.rand(n, device=x.device) < 0.5).view(-1, 1, 1, 1)
    flip_v = (torch.rand(n, device=x.device) < 0.5).view(-1, 1, 1, 1)
    out = torch.where(flip_h, out.flip(3), out)
    out = torch.where(flip_v, out.flip(2), out)
    return out


@dataclass
class PhotometricAugment:
    """Random per-image contrast and per-band brightness changes, applied to raw values.

    For every image (with probability ``p``):
    - contrast: deviations from the image's own mean (per band) are scaled by
      c = c_image * c_band, with c_image ~ U(contrast) shared by all bands and
      c_band ~ U(band_contrast) per band;
    - brightness: each band is multiplied by g ~ U(brightness).
    Values are clipped at 0. Rotations/flips are applied separately by ``augment``.
    """

    contrast: tuple[float, float] = (0.9, 1.5)
    band_contrast: tuple[float, float] = (0.9, 1.15)
    brightness: tuple[float, float] = (0.93, 1.07)
    p: float = 0.8

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        n, c = x.shape[:2]
        dev = x.device
        u = lambda lo_hi, *shape: lo_hi[0] + (lo_hi[1] - lo_hi[0]) * torch.rand(*shape, device=dev)
        active = (torch.rand(n, 1, 1, 1, device=dev) < self.p).float()
        contrast = u(self.contrast, n, 1, 1, 1) * u(self.band_contrast, n, c, 1, 1)
        brightness = u(self.brightness, n, c, 1, 1)
        contrast = 1 + active * (contrast - 1)
        brightness = 1 + active * (brightness - 1)
        mean = x.mean(dim=(2, 3), keepdim=True)
        out = (mean + (x - mean) * contrast) * brightness
        return out.clamp_min(0)


# ----------------------------------------------------------------------------
# Training and prediction
# ----------------------------------------------------------------------------

def _forward(model, x, extra):
    """model(x), or model(x, extra) for models with per-image side features."""
    return model(x) if extra is None else model(x, extra)


def train_one_epoch(model, images, labels, normalizer, optimizer, scheduler, criterion,
                    batch_size, use_augment=True, photometric=None, raw_augment=None, extra=None,
                    after_step=None, weights=None):
    """One pass over the training data. Returns (mean loss, accuracy).

    ``raw_augment`` (e.g. FourierDomainAugment) and then ``photometric`` (e.g. PhotometricAugment)
    are applied to the raw values before normalisation. ``extra``: optional per-image features
    (N, F) passed to the model next to the images (e.g. topological features).
    ``after_step(model)`` is called after every optimizer step (used for the weight average).
    ``labels``: class indices (N,) or probability targets (N, 10) (soft pseudo-labels). With
    ``weights`` (N,), the loss is a weighted mean; ``criterion`` must then use ``reduction="none"``.
    """
    model.train()
    order = torch.randperm(len(images), device=images.device)
    loss_sum, correct = 0.0, 0
    for start in range(0, len(order), batch_size):
        idx = order[start:start + batch_size]
        x = images[idx].float()
        if raw_augment is not None:
            x = raw_augment(x)
        if photometric is not None:
            x = photometric(x)
        x = normalizer(x)
        if use_augment:
            x = augment(x)
        y = labels[idx]
        logits = _forward(model, x, None if extra is None else extra[idx])
        if weights is None:
            loss = criterion(logits, y)
        else:
            w = weights[idx]
            loss = (criterion(logits, y) * w).sum() / w.sum()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        if scheduler is not None:
            scheduler.step()
        if after_step is not None:
            after_step(model)
        loss_sum += loss.item() * len(idx)
        correct += (logits.argmax(1) == (y if y.ndim == 1 else y.argmax(1))).sum().item()
    return loss_sum / len(images), correct / len(images)


@torch.no_grad()
def predict_proba(model, images, normalizer, batch_size=512, tta=False, extra=None):
    """Class probabilities (N, 10) as a NumPy array.

    With ``tta=True`` the probabilities are averaged over the 8 rotations/flips of each image.
    """
    model.eval()
    out = []
    for start in range(0, len(images), batch_size):
        x = normalizer(images[start:start + batch_size])
        e = None if extra is None else extra[start:start + batch_size]
        if tta:
            views = [torch.rot90(v, r, dims=(2, 3)) for v in (x, x.flip(3)) for r in range(4)]
            p = torch.stack([_forward(model, v, e).softmax(1) for v in views]).mean(0)
        else:
            p = _forward(model, x, e).softmax(1)
        out.append(p.cpu())
    return torch.cat(out).numpy()


@torch.no_grad()
def predict_views(model, images, normalizer, batch_size=512, extra=None):
    """Class probabilities for each of the 8 rotations/flips separately: shape (8, N, 10).

    Averaging over the first axis gives the same result as ``predict_proba(..., tta=True)``.
    """
    model.eval()
    out = []
    for start in range(0, len(images), batch_size):
        x = normalizer(images[start:start + batch_size])
        e = None if extra is None else extra[start:start + batch_size]
        views = [torch.rot90(v, r, dims=(2, 3)) for v in (x, x.flip(3)) for r in range(4)]
        out.append(torch.stack([_forward(model, v, e).softmax(1) for v in views]).cpu())
    return torch.cat(out, dim=1).numpy()


@torch.no_grad()
def adapt_batchnorm(model, images, normalizer, batch_size=256, extra=None):
    """AdaBN: re-estimate every BatchNorm layer's mean/variance on ``images``.

    All learned weights stay unchanged; only the running statistics that say what "normal"
    activations look like are recomputed, as an equal-weight average over all batches.
    Dropout and stochastic depth stay switched off. Changes ``model`` in place.
    """
    model.eval()
    bn_layers = [m for m in model.modules() if isinstance(m, nn.modules.batchnorm._BatchNorm)]
    for bn in bn_layers:
        bn.reset_running_stats()
        bn.momentum = None  # cumulative average instead of exponential moving average
        bn.train()
    order = torch.randperm(len(images), device=images.device)
    for start in range(0, len(order), batch_size):
        idx = order[start:start + batch_size]
        _forward(model, normalizer(images[idx]), None if extra is None else extra[idx])
    model.eval()
    return model


@dataclass
class TrainConfig:
    epochs: int = 20
    batch_size: int = 128
    lr: float = 1e-3
    weight_decay: float = 1e-4
    label_smoothing: float = 0.1
    patience: int = 5          # stop after this many epochs without a better val accuracy
    augment: bool = True
    photometric: dict | None = None  # PhotometricAugment settings, None = off
    extra: dict = field(default_factory=dict)


def fit(model, train_images, train_labels, val_images, val_labels, normalizer,
        cfg: TrainConfig, monitor=None, raw_augment=None, train_extra=None, val_extra=None,
        snapshots: dict | None = None, snapshot_epochs=(), ema_decay: float | None = None,
        train_weights=None):
    """Train with AdamW + one-cycle schedule and early stopping on val accuracy.

    ``monitor(model) -> dict`` is called after every epoch; its values are added to the
    history (used to watch the Kaggle test set without labels). ``raw_augment`` is an extra
    augmentation on raw values (e.g. FourierDomainAugment), see ``train_one_epoch``.
    Returns (history DataFrame, best state_dict).

    Optional, for weight-averaging experiments: pass a dict as ``snapshots`` and it is filled with
    CPU state_dicts: ``"epoch_<k>"`` after every epoch in ``snapshot_epochs``, and with ``ema_decay``
    set, ``"ema"``: an exponential moving average of the weights, updated after every step
    (BatchNorm statistics are copied, not averaged; re-measure them before use).

    ``train_labels`` may be probability targets (N, 10) instead of class indices; ``train_weights``
    (N,) weights each image's loss (used for soft pseudo-labels, E09).
    """
    device = train_images.device
    criterion = nn.CrossEntropyLoss(label_smoothing=cfg.label_smoothing,
                                    reduction="mean" if train_weights is None else "none")
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    steps_per_epoch = -(-len(train_images) // cfg.batch_size)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=cfg.lr, total_steps=cfg.epochs * steps_per_epoch, pct_start=0.15
    )
    val_labels_np = val_labels.cpu().numpy()
    photometric = PhotometricAugment(**cfg.photometric) if cfg.photometric else None

    history, best_acc, best_state, since_best = [], -1.0, None, 0
    after_step, ema = None, None
    if ema_decay is not None:
        ema = {k: v.detach().clone().float() for k, v in model.state_dict().items()}

        @torch.no_grad()
        def after_step(m):
            for k, v in m.state_dict().items():
                if v.dtype.is_floating_point and not k.endswith(("running_mean", "running_var")):
                    ema[k].mul_(ema_decay).add_(v.detach(), alpha=1 - ema_decay)
                else:
                    ema[k].copy_(v)
    for epoch in range(1, cfg.epochs + 1):
        t0 = time.time()
        train_loss, train_acc = train_one_epoch(
            model, train_images, train_labels, normalizer, optimizer, scheduler, criterion,
            cfg.batch_size, cfg.augment, photometric, raw_augment, train_extra, after_step, train_weights,
        )
        val_prob = predict_proba(model, val_images, normalizer, extra=val_extra)
        val_loss = float(nn.functional.nll_loss(
            torch.log(torch.from_numpy(val_prob).clamp_min(1e-8)), torch.from_numpy(val_labels_np)))
        val_acc = float((val_prob.argmax(1) == val_labels_np).mean())
        row = {"epoch": epoch, "train_loss": train_loss, "train_acc": train_acc,
               "val_loss": val_loss, "val_acc": val_acc, "lr": scheduler.get_last_lr()[0]}
        if monitor is not None:
            row.update(monitor(model))
        synchronize(device)
        row["seconds"] = time.time() - t0
        history.append(row)

        if snapshots is not None and epoch in snapshot_epochs:
            snapshots[f"epoch_{epoch}"] = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        improved = val_acc > best_acc
        if improved:
            best_acc, since_best = val_acc, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            since_best += 1
        extra = "  ".join(f"{k} {v:.3f}" for k, v in row.items()
                          if k not in {"epoch", "train_loss", "train_acc", "val_loss", "val_acc", "lr", "seconds"}
                          and isinstance(v, float))
        print(f"epoch {epoch:2d}  train loss {train_loss:.3f} acc {train_acc:.3f}  "
              f"val loss {val_loss:.3f} acc {val_acc:.3f}{' *' if improved else '  '}  "
              f"{extra}  ({row['seconds']:.0f}s)")
        if since_best >= cfg.patience:
            print(f"Early stop: no improvement for {cfg.patience} epochs.")
            break

    if snapshots is not None and ema is not None:
        snapshots["ema"] = {k: v.cpu().to(best_state[k].dtype) for k, v in ema.items()}
    model.load_state_dict(best_state)
    return pd.DataFrame(history), best_state


# ----------------------------------------------------------------------------
# Saving and loading runs
# ----------------------------------------------------------------------------

def new_run_dir(name: str) -> Path:
    """runs/<YYYY-MM-DD_HHMM>_<name>/"""
    run_dir = RUNS_DIR / f"{datetime.now():%Y-%m-%d_%H%M}_{name}"
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def save_run(run_dir: Path, state_dict: dict, config: dict, history: pd.DataFrame,
             metrics: dict) -> None:
    """Save weights, config, training history and metrics of one run."""
    torch.save(state_dict, run_dir / "model.pt")
    (run_dir / "config.json").write_text(json.dumps(config, indent=2))
    history.to_csv(run_dir / "history.csv", index=False)
    (run_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))


def load_run(run_dir: Path, device: torch.device):
    """Rebuild the model of a saved run. Returns (model, normalizer, config)."""
    from src.models import build_model

    config = json.loads((Path(run_dir) / "config.json").read_text())
    model = build_model(config["model"], bands=config["bands"], pretrained=False,
                        input_layer=config.get("input_layer"), **config.get("model_kwargs", {}))
    model.load_state_dict(torch.load(Path(run_dir) / "model.pt", map_location="cpu"))
    normalizer = Normalizer(**config["normalizer"])
    return model.to(device).eval(), normalizer, config


def write_submission(probabilities: np.ndarray, test_ids: pd.DataFrame, path: Path) -> pd.DataFrame:
    """Write a Kaggle submission CSV (test_id, label) and return it."""
    submission = pd.DataFrame({
        "test_id": test_ids["test_id"].values,
        "label": [CLASSES[i] for i in probabilities.argmax(1)],
    })
    submission.to_csv(path, index=False)
    return submission
