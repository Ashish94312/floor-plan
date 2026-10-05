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


# ------------------------------------------------------------------------------- portrait turning
# geometry.rotate_portrait: portrait frames go into the model turned 90 deg clockwise (np.rot90(img, -1))
# and every output is turned back, so the rest of the pipeline never sees the difference (E5: portrait
# input made a backbone misjudge the focal ~26%; MapAnything puts video frames at ~22 mm vs ~30, E22b).
# Pixel centres: u_old = v_new, v_old = H - u_new; camera axes: x_old = y_new, y_old = -x_new.
M_TURN = np.array([[0.0, 1.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])  # p_old = M_TURN @ p_new


def turn_K(K: np.ndarray, H: float) -> np.ndarray:
    """Intrinsics of an image of height H turned 90 deg clockwise."""
    return np.array([[K[1, 1], 0.0, H - K[1, 2]], [0.0, K[0, 0], K[0, 2]], [0.0, 0.0, 1.0]])


def unturn_K(K: np.ndarray, W_turned: float) -> np.ndarray:
    """Inverse of turn_K; W_turned = width of the turned image (= the original height)."""
    return np.array([[K[1, 1], 0.0, K[1, 2]], [0.0, K[0, 0], W_turned - K[0, 2]], [0.0, 0.0, 1.0]])


def unturn_R(R: np.ndarray) -> np.ndarray:
    """Camera-to-world rotation of the original (portrait) camera from the turned one."""
    return R @ M_TURN.T


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
        **({"rotate_portrait": True} if _turns(frames, cfg) else {}),  # old keys unchanged when off
        **({"max_aspect": g["max_aspect"]} if g.get("max_aspect") else {}),
        **({"pad_scale": g["pad_scale"]} if g.get("pad_scale") else {}),
    )


def pad_frame(img: np.ndarray, K: np.ndarray, pad_scale: float | None) -> tuple[np.ndarray, np.ndarray, tuple[int, int]]:
    """Centre the image on a black canvas pad_scale times its size (E22n). MapAnything assumes a field of view
    of its own (~72 deg on the long side) and takes a focal to match; on 32 mm video frames (61 deg) that
    focal is ~20% short and depth is squeezed. On a canvas whose field of view at the TRUE focal is what the
    model expects, the content keeps its true size. Returns (canvas, K on the canvas, content offset x, y)."""
    if not pad_scale or pad_scale <= 1:
        return img, K, (0, 0)
    H, W = img.shape[:2]
    Hc, Wc = round(H * pad_scale / 2) * 2, round(W * pad_scale / 2) * 2
    x0, y0 = (Wc - W) // 2, (Hc - H) // 2
    canvas = np.zeros((Hc, Wc, 3), img.dtype)
    canvas[y0:y0 + H, x0:x0 + W] = img
    K = K.copy()
    K[0, 2] += x0
    K[1, 2] += y0
    return canvas, K, (x0, y0)


def content_box(K_in: np.ndarray, K_out: np.ndarray, off: tuple[int, int], hw: tuple[int, int]) -> tuple[int, int, int, int]:
    """Pixel box (x0, y0, x1, y1) of the padded frame's content at the model's resolution. K_in: canvas K at
    working resolution, K_out: the same after preprocess_inputs (scale + centre crop); hw: content H, W."""
    s = K_out[0, 0] / K_in[0, 0]
    ox, oy = s * K_in[0, 2] - K_out[0, 2], s * K_in[1, 2] - K_out[1, 2]
    xa, ya = int(np.ceil(s * off[0] - ox)) + 1, int(np.ceil(s * off[1] - oy)) + 1  # 1 px in from the black edge
    xb, yb = int(np.floor(s * (off[0] + hw[1]) - ox)) - 1, int(np.floor(s * (off[1] + hw[0]) - oy)) - 1
    return xa, ya, xb, yb


def crop_aspect(img: np.ndarray, K: np.ndarray, max_aspect: float | None) -> tuple[np.ndarray, np.ndarray]:
    """Centre-crop the long side to short side x max_aspect (E22n: 9:16 video frames -> 3:4 like photos)."""
    if not max_aspect:
        return img, K
    H, W = img.shape[:2]
    K = K.copy()
    if H > W * max_aspect:
        y0 = (H - round(W * max_aspect)) // 2
        img, K[1, 2] = img[y0:y0 + round(W * max_aspect)], K[1, 2] - y0
    elif W > H * max_aspect:
        x0 = (W - round(H * max_aspect)) // 2
        img, K[0, 2] = img[:, x0:x0 + round(H * max_aspect)], K[0, 2] - x0
    return np.ascontiguousarray(img), K


def _turns(frames, cfg) -> list[bool]:
    """Per frame: is it turned before the model? Empty list when the switch is off."""
    if not cfg["geometry"].get("rotate_portrait"):
        return []
    t = [f.rgb.shape[0] > f.rgb.shape[1] for f in frames]
    return t if any(t) else []


def _infer(model, views, g, device) -> tuple[list, float, float]:
    import time

    from scan.device import PeakMemory

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
    return preds, time.perf_counter() - t0, mem.peak_gb


def refit_with_depth(pred: dict[str, np.ndarray], cfg, device, get_model, scale: float = 1.0, use_cache: bool = True
                     ) -> tuple[dict[str, np.ndarray], dict]:
    """Second model pass for video (E22n): the model re-solves poses (and refines depth) given each view's
    depth stretched to the true focal (cloud.depth_focal_fix) and the true intrinsics as inputs. Pass 1
    keeps its own focal (~23-27 mm vs 32 true) and squeezes depth along the line of sight; stretching depth
    alone leaves the poses in the squeezed world (E22m). Inputs at the model's working resolution (pass 1's
    image), masked pixels as depth 0 (= no input)."""
    from scan import cache
    from scan.cache import cache_key
    from scan.geometry.cloud import depth_focal_fix

    fixed, r = depth_focal_fix(pred, scale)
    key = cache_key(kind="mapanything_refit", pass1=hashlib_of(pred), scale=scale, code=MAPANYTHING_COMMIT)
    if use_cache and (hit := cache.load(cfg, "mapanything", key)) is not None:
        hit["names"], hit["rooms"] = pred["names"], pred["rooms"]
        return hit, {"cache": "hit", "key": key, "infer_s": 0.0, "depth_focal_ratio": _pct(r)}

    from mapanything.utils.image import preprocess_inputs
    from PIL import Image

    g = cfg["geometry"]
    views = []
    for i in range(len(r)):
        d = np.where(pred["mask"][i], fixed["depth"][i], 0.0).astype(np.float32)
        views.append({"img": Image.fromarray(pred["rgb"][i]), "intrinsics": torch.tensor(fixed["K_exif"][i], dtype=torch.float32),
                      "depth_z": torch.tensor(d), "is_metric_scale": torch.tensor([True])})
    views = preprocess_inputs(views)  # same size in and out (pass 1's working resolution)
    preds, infer_s, peak = _infer(get_model(), views, g, device)

    def stack(k, fn=lambda x: x):
        return np.stack([fn(p[k][0].float().cpu().numpy()) for p in preds])

    out = {
        "pts": stack("pts3d").astype(np.float32),
        "depth": stack("depth_z", lambda x: x[..., 0]).astype(np.float32),
        "conf": stack("conf").astype(np.float32),
        "mask": np.stack([p["mask"][0, ..., 0].cpu().numpy().astype(bool) for p in preds]) & pred["mask"],
        "T_wc": stack("camera_poses").astype(np.float64),
        "K_model": stack("intrinsics").astype(np.float64),
        "K_exif": fixed["K_exif"],
        "rgb": pred["rgb"],
        "names": pred["names"],
        "rooms": pred["rooms"],
    }
    if use_cache:
        cache.save(cfg, "mapanything", key, out)
    return out, {"cache": "miss", "key": key, "infer_s": round(infer_s, 2), "peak_accel_gb": peak, "depth_focal_ratio": _pct(r)}


def hashlib_of(pred: dict[str, np.ndarray]) -> str:
    import hashlib

    h = hashlib.sha256()
    for k in ("depth", "T_wc", "K_model", "K_exif"):
        h.update(np.ascontiguousarray(pred[k]).tobytes())
    return h.hexdigest()


def _pct(r: np.ndarray) -> list[float]:
    return [round(float(x), 3) for x in np.percentile(r, [10, 50, 90])]


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

    from mapanything.utils.image import preprocess_inputs
    from PIL import Image

    g = cfg["geometry"]
    fs = float(g["f_scale"])
    turned = _turns(frames, cfg) or [False] * len(frames)
    views, pads = [], []
    for f, t in zip(frames, turned):
        img, K = crop_aspect(f.rgb, f.K, g.get("max_aspect"))
        hw = img.shape[:2]
        img, K, off = pad_frame(img, K, g.get("pad_scale"))
        pads.append((K.copy(), off, hw))
        if t:
            K, img = turn_K(K, img.shape[0]), np.ascontiguousarray(np.rot90(img, -1))
        K[0, 0] *= fs
        K[1, 1] *= fs
        views.append({"img": Image.fromarray(img), "intrinsics": torch.tensor(K, dtype=torch.float32)})
    views = preprocess_inputs(views)  # -> 518 px long side, intrinsics rescaled to match
    K_exif = np.stack([v["intrinsics"][0].numpy().astype(np.float64) for v in views])
    K_exif[:, 0, 0] /= fs
    K_exif[:, 1, 1] /= fs
    for i, (v, t) in enumerate(zip(views, turned)):
        if t:
            K_exif[i] = unturn_K(K_exif[i], v["img"].shape[-1])

    preds, infer_s, peak = _infer(get_model(), views, g, device)

    def back(a, t):  # per-pixel output of a turned view -> original orientation
        return np.ascontiguousarray(np.rot90(a, 1, axes=(0, 1))) if t else a

    def stack(k, fn=lambda x: x, pixels=True):
        return np.stack([back(fn(p[k][0].float().cpu().numpy()), t and pixels) for p, t in zip(preds, turned)])

    T_wc = stack("camera_poses", pixels=False).astype(np.float64)
    K_model = stack("intrinsics", pixels=False).astype(np.float64)
    for i, (p, t) in enumerate(zip(preds, turned)):
        if t:
            T_wc[i, :3, :3] = unturn_R(T_wc[i, :3, :3])
            K_model[i] = unturn_K(K_model[i], p["pts3d"].shape[2])
    out = {
        "pts": stack("pts3d").astype(np.float32),
        "depth": stack("depth_z", lambda x: x[..., 0]).astype(np.float32),
        "conf": stack("conf").astype(np.float32),
        "mask": np.stack([back(p["mask"][0, ..., 0].cpu().numpy().astype(bool), t) for p, t in zip(preds, turned)]),
        "T_wc": T_wc,
        "K_model": K_model,
        "K_exif": K_exif,
        "rgb": stack("img_no_norm", lambda x: (np.clip(x, 0, 1) * 255).round().astype(np.uint8)),
        "names": np.array([f.image_path.name for f in frames]),
        "rooms": np.array([f.room_hint for f in frames]),
    }
    if g.get("pad_scale") and g["pad_scale"] > 1:  # back to the content: every view has the same box (same frame size)
        xa, ya, xb, yb = content_box(pads[0][0], K_exif[0], pads[0][1], pads[0][2])  # both without f_scale
        for k in ("pts", "depth", "conf", "mask", "rgb"):
            out[k] = np.ascontiguousarray(out[k][:, ya:yb, xa:xb])
        for k in ("K_model", "K_exif"):
            out[k] = out[k].copy()
            out[k][:, 0, 2] -= xa
            out[k][:, 1, 2] -= ya
    if use_cache:
        cache.save(cfg, "mapanything", key, out)
    return out, {"cache": "miss", "key": key, "infer_s": round(infer_s, 2), "views": len(frames), "peak_accel_gb": peak}
