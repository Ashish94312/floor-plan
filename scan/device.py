"""Torch device selection and global seeding."""

from __future__ import annotations

import os
import random

import numpy as np

SEED = 0


def select_device(preference: str | None = None):
    """CUDA if available, then MPS, then CPU. `preference` forces one."""
    import torch

    if preference:
        return torch.device(preference)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def seed_everything(seed: int = SEED) -> np.random.Generator:
    """Seed torch / numpy / random and return the RNG to pass explicitly to RANSAC etc."""
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
    return np.random.default_rng(seed)


def go_offline() -> None:
    """After weights are fetched the pipeline never touches the network."""
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    # Ops missing on MPS fall back to CPU instead of crashing.
    os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
