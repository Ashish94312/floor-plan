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


class PeakMemory:
    """Samples the accelerator's allocated memory every 50 ms inside a `with` block (MPS / CUDA)."""

    def __init__(self, device):
        import threading

        self.device = device
        self.peak_gb = 0.0
        self._stop = threading.Event()
        self._t = threading.Thread(target=self._run, daemon=True)

    def _now(self) -> float:
        import torch

        if self.device.type == "mps":
            return torch.mps.driver_allocated_memory() / 1024**3
        if self.device.type == "cuda":
            return torch.cuda.memory_allocated() / 1024**3
        return 0.0

    def _run(self):
        import time

        while not self._stop.is_set():
            self.peak_gb = max(self.peak_gb, self._now())
            time.sleep(0.05)

    def __enter__(self):
        self._t.start()
        return self

    def __exit__(self, *exc):
        self._stop.set()
        self._t.join()
        self.peak_gb = round(max(self.peak_gb, self._now()), 2)
