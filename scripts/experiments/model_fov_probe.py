"""E22s. How does MapAnything's own focal respond to how a video frame is presented? (no tape involved)

The target is physics: the true focal is known (vanishing points / self-calibration), so the best presentation
is the one where the model's focal comes out closest to it. Same keyframes, shown as: as-is, centre-cropped to
4:3, padded to 4:3 (black bands, content kept). Prints the model's focal (35 mm eq. of the ORIGINAL frame) and
the median depth per presentation.

Usage: uv run python scripts/experiments/model_fov_probe.py captures/home01_video_e hall [k]
"""

import math
import sys
from pathlib import Path

import numpy as np
import torch

from scan.config import load_config
from scan.device import select_device
from scan.geometry.mapanything_backend import load_model
from scan.io.photos import intrinsics
from scan.io.video import VIDEO_EXT, decode, keyframes

cap, room = Path(sys.argv[1]), sys.argv[2]
k = int(sys.argv[3]) if len(sys.argv) > 3 else 12
f35 = float(sys.argv[4]) if len(sys.argv) > 4 else 33.1
cfg = load_config()
vc = cfg["video"]
clip = next(p for p in sorted((cap / "video" / room).iterdir()) if p.suffix.lower() in VIDEO_EXT)
frames, ts, _ = decode(clip, vc["decode_fps"], cfg["ingest"]["working_max_side"])
picks, _ = keyframes(frames, k, vc["min_sharpness"], min_gap=round(vc["min_keyframe_gap_s"] * vc["decode_fps"]))
imgs = [frames[i] for i in picks]
H, W = imgs[0].shape[:2]
diag = cfg["ingest"]["intrinsics"]["film_diagonal_mm"]
K0 = intrinsics(f35, W, H, diag)


def crop43(img, K):
    w = round(img.shape[0] * 4 / 3) if img.shape[1] > img.shape[0] else img.shape[1]
    h = img.shape[0] if img.shape[1] > img.shape[0] else round(img.shape[1] * 4 / 3)
    x0, y0 = (img.shape[1] - w) // 2, (img.shape[0] - h) // 2
    K = K.copy()
    K[0, 2] -= x0
    K[1, 2] -= y0
    return np.ascontiguousarray(img[y0:y0 + h, x0:x0 + w]), K


def pad43(img, K):
    h, w = img.shape[:2]
    Hc, Wc = (round(w * 3 / 4), w) if w > h else (h, round(h * 3 / 4))
    Hc, Wc = max(Hc, h), max(Wc, w)
    y0, x0 = (Hc - h) // 2, (Wc - w) // 2
    c = np.zeros((Hc, Wc, 3), img.dtype)
    c[y0:y0 + h, x0:x0 + w] = img
    K = K.copy()
    K[0, 2] += x0
    K[1, 2] += y0
    return c, K


from mapanything.utils.image import preprocess_inputs  # noqa: E402
from PIL import Image  # noqa: E402

dev = select_device(None)
model = load_model(dev)
print(f"{cap.name} {room}: {len(imgs)} keyframes {W}x{H}, true focal {f35} mm (35 mm eq.)")
for name, fn in (("as-is 16:9", lambda i, K: (i, K)), ("crop 4:3", crop43), ("pad 4:3", pad43)):
    views, Ks = [], []
    for im in imgs:
        i2, K2 = fn(im, K0)
        views.append({"img": Image.fromarray(i2), "intrinsics": torch.tensor(K2, dtype=torch.float32)})
        Ks.append(K2)
    views = preprocess_inputs(views)
    Kin = np.stack([v["intrinsics"][0].numpy() for v in views])  # before infer: it moves inputs to the device
    size = f"{views[0]['img'].shape[-1]}x{views[0]['img'].shape[-2]}"
    with torch.inference_mode():
        preds = model.infer(views, memory_efficient_inference=True, use_amp=True, amp_dtype="fp16", apply_mask=True,
                            mask_edges=True, apply_confidence_mask=False)
    Km = np.stack([p["intrinsics"][0].float().cpu().numpy() for p in preds])
    ratio = Km[:, 0, 0] / Kin[:, 0, 0]  # model focal / true focal, same pixels
    depth = np.concatenate([p["depth_z"][0, ..., 0].float().cpu().numpy()[p["mask"][0, ..., 0].cpu().numpy()] for p in preds])
    print(f"  {name:11s} input {size}  model focal "
          f"{' / '.join(f'{f35 * x:.1f}' for x in np.percentile(ratio, [10, 50, 90]))} mm (p10/50/90)  "
          f"median depth {np.median(depth):.2f} m")
