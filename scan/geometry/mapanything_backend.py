"""MapAnything backbone (D17): loading, fully offline.

MapAnything's DINOv2 encoder calls torch.hub.load("facebookresearch/dinov2", ...), which pings
GitHub on every load and downloads the repo on first use. We redirect that call to the copy
vendored in weights/dinov2-code at a pinned commit (scan-fetch-weights), so a run makes no
network calls and always builds the encoder from the code we benchmarked.
"""

from __future__ import annotations

import contextlib
from collections.abc import Iterator

import torch

from scan.weights import code_path, model_path

DINOV2_HUB_REPO = "facebookresearch/dinov2"


@contextlib.contextmanager
def local_dinov2_hub() -> Iterator[None]:
    """Route torch.hub.load("facebookresearch/dinov2", ...) to the vendored local copy."""
    original = torch.hub.load
    local_dir = str(code_path("dinov2-code"))

    def load(repo_or_dir, model, *args, **kwargs):
        if repo_or_dir == DINOV2_HUB_REPO:
            kwargs.pop("force_reload", None)
            kwargs.pop("trust_repo", None)
            kwargs.pop("skip_validation", None)
            kwargs["source"] = "local"
            return original(local_dir, model, *args, **kwargs)
        return original(repo_or_dir, model, *args, **kwargs)

    torch.hub.load = load
    try:
        yield
    finally:
        torch.hub.load = original


def load_model(device: torch.device):
    """MapAnything (Apache-2.0 weights) from weights/, in eval mode on `device`."""
    from mapanything.models import MapAnything

    with local_dinov2_hub():
        model = MapAnything.from_pretrained(str(model_path("mapanything-apache")))
    return model.to(device).eval()
