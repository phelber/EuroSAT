"""Device selection, seeding and small helpers shared by all notebooks."""

from __future__ import annotations

import os
import random

import numpy as np
import torch


def get_device(prefer: str | None = None) -> torch.device:
    """Return the best available device: CUDA, then Apple GPU (MPS), then CPU.

    ``prefer`` ("cuda", "mps" or "cpu") forces a device, e.g. for debugging on CPU.
    """
    if prefer is not None:
        return torch.device(prefer)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def describe_device(device: torch.device) -> str:
    """Human-readable name of the device, for logs."""
    if device.type == "cuda":
        return f"CUDA: {torch.cuda.get_device_name(device)}"
    if device.type == "mps":
        return "Apple GPU (MPS)"
    return f"CPU ({torch.get_num_threads()} threads)"


def seed_everything(seed: int) -> None:
    """Seed Python, NumPy and PyTorch.

    Results on GPUs are still not bit-for-bit reproducible, but runs become comparable.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def synchronize(device: torch.device) -> None:
    """Wait for queued GPU work to finish (for accurate timing)."""
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps":
        torch.mps.synchronize()


def count_parameters(model: torch.nn.Module) -> int:
    """Total number of parameters, as checked for the challenge tracks."""
    return sum(p.numel() for p in model.parameters())
