"""MapAnything backbone (D17): loading, fully offline.

MapAnything's DINOv2 encoder calls torch.hub.load("facebookresearch/dinov2", ...), which pings
GitHub on every load and downloads the repo on first use. We redirect that call to the copy
vendored in weights/dinov2-code at a pinned commit (scan-fetch-weights), so a run makes no
network calls and always builds the encoder from the code we benchmarked.
"""

from __future__ import annotations

import contextlib
from collections.abc import Iterator

import numpy as np
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


# ------------------------------------------------------------------------------- inference

MAPANYTHING_COMMIT = "3d10cf7a3016fc0f9bb13a071ee66c47b10be0d9"  # pinned in pyproject.toml


def _cache_key(frames, cfg) -> str:
    from scan.cache import cache_key
    from scan.weights import CODE, MODELS

    g = cfg["geometry"]
    return cache_key(
        kind="mapanything",
        images=[f.meta.sha256 for f in frames],
        K=[np.round(f.K, 4).tolist() for f in frames],  # covers focal, working size, orientation
        working_max_side=cfg["ingest"]["working_max_side"],
        weights=MODELS["mapanything-apache"].revision,
        code=MAPANYTHING_COMMIT,
        dinov2=CODE["dinov2-code"].commit,
        f_scale=g["f_scale"],
        amp_dtype=g["amp_dtype"],
        memory_efficient=g["memory_efficient"],
    )


def predict(frames, cfg, device, get_model, use_cache: bool = True) -> tuple[dict[str, np.ndarray], dict]:
    """Run MapAnything on `frames` (one room, or all rooms for a joint run).

    Returns stacked per-view arrays at the model's working resolution (S views, H x W):
      pts (S,H,W,3) world points [m]   depth (S,H,W) z-depth [m]   conf (S,H,W)   mask (S,H,W) bool
      T_wc (S,4,4) camera-to-world     K_model (S,3,3) model's focal   K_exif (S,3,3) EXIF focal
      rgb (S,H,W,3) uint8              names (S,) file names           rooms (S,) room per view
    and an info dict (cache hit/miss, timings).
    """
    from scan import cache

    key = _cache_key(frames, cfg)
    if use_cache and (hit := cache.load(cfg, "mapanything", key)) is not None:
        # labels are not in the key (content is): take them from the frames, so a renamed room folder still matches
        hit["names"] = np.array([f.image_path.name for f in frames])
        hit["rooms"] = np.array([f.room_hint for f in frames])
        return hit, {"cache": "hit", "key": key, "infer_s": 0.0}

    import time

    from mapanything.utils.image import preprocess_inputs
    from PIL import Image

    g = cfg["geometry"]
    fs = float(g["f_scale"])
    views = []
    for f in frames:
        K = f.K.copy()
        K[0, 0] *= fs
        K[1, 1] *= fs
        views.append({"img": Image.fromarray(f.rgb), "intrinsics": torch.tensor(K, dtype=torch.float32)})
    views = preprocess_inputs(views)  # -> 518 px long side, intrinsics rescaled to match
    K_exif = np.stack([v["intrinsics"][0].numpy().astype(np.float64) for v in views])
    K_exif[:, 0, 0] /= fs
    K_exif[:, 1, 1] /= fs

    from scan.device import PeakMemory

    model = get_model()
    torch.manual_seed(0)
    t0 = time.perf_counter()
    with PeakMemory(device) as mem, torch.inference_mode():
        preds = model.infer(
            views,
            memory_efficient_inference=g["memory_efficient"],
            use_amp=True,
            amp_dtype=g["amp_dtype"],
            apply_mask=True,
            mask_edges=True,
            apply_confidence_mask=False,
        )
    if device.type == "mps":
        torch.mps.synchronize()
    infer_s = time.perf_counter() - t0

    def stack(k, fn=lambda x: x):
        return np.stack([fn(p[k][0].float().cpu().numpy()) for p in preds])

    out = {
        "pts": stack("pts3d").astype(np.float32),
        "depth": stack("depth_z", lambda x: x[..., 0]).astype(np.float32),
        "conf": stack("conf").astype(np.float32),
        "mask": np.stack([p["mask"][0, ..., 0].cpu().numpy().astype(bool) for p in preds]),
        "T_wc": stack("camera_poses").astype(np.float64),
        "K_model": stack("intrinsics").astype(np.float64),
        "K_exif": K_exif,
        "rgb": stack("img_no_norm", lambda x: (np.clip(x, 0, 1) * 255).round().astype(np.uint8)),
        "names": np.array([f.image_path.name for f in frames]),
        "rooms": np.array([f.room_hint for f in frames]),
    }
    if use_cache:
        cache.save(cfg, "mapanything", key, out)
    return out, {"cache": "miss", "key": key, "infer_s": round(infer_s, 2), "views": len(frames), "peak_accel_gb": mem.peak_gb}
