"""E5. Portrait experiments: (A) as-is, (B) rotate to landscape. Report VGGT focal vs EXIF and
geometry ratios that don't depend on metric scale.

Usage: uv run python scripts/experiments/portrait_rotation.py [photos_dir]
"""
import sys
from pathlib import Path
import numpy as np, torch
from PIL import Image
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import spike_phase0 as sp
from vggt.utils.geometry import unproject_depth_map_to_point_map

dev = sp.select_device(); sp.seed_everything()
paths = sp.list_images(Path(sys.argv[1] if len(sys.argv) > 1 else "captures/home01_photo_portrait/photos/bedroom1"), 8)
imgs = [sp.load_rgb(p) for p in paths]

def analyse(tag, ims, down_row):
    x = sp.vggt_tensor(ims)
    v = sp.run_vggt(x, dev, "half", 2)
    H, W = x.shape[-2:]
    f_exif = 26 / 43.27 * np.hypot(W, H)
    K = v["intrinsic"]
    E = v["extrinsic"]
    cent = -np.einsum("sji,sj->si", E[:, :, :3], E[:, :, 3])
    d = np.linalg.norm(cent[:, None] - cent[None], axis=-1)
    # gravity from the camera axis that points image-down in the ORIGINAL photo
    up = -(E[:, down_row[0], :3] * down_row[1]).mean(0); up /= np.linalg.norm(up)
    pts = unproject_depth_map_to_point_map(v["depth"][..., None], E, K)
    P = pts[v["depth_conf"] > np.percentile(v["depth_conf"], 50)]
    h = P @ up
    hist, edges = np.histogram(h, bins=np.arange(h.min(), h.max(), 0.01))
    ceil = edges[np.argmax(np.where(edges[:-1] > 0.2 * h.max(), hist, 0))]
    # horizontal extent via Manhattan rotation search on upper band
    e1 = np.cross(up, [1.0, 0, 0]); e1 /= np.linalg.norm(e1); e2 = np.cross(up, e1)
    band = P[(h > 0.3 * ceil) & (h < 0.9 * ceil)]
    q = np.stack([band @ e1, band @ e2], 1)
    best = max(((sum(float((np.histogram(r[:, k], bins=np.arange(r[:, k].min(), r[:, k].max() + 0.01, 0.01))[0] ** 2).sum()) for k in (0, 1)), a)
                for a in np.deg2rad(np.arange(0, 90, 0.5))
                for r in [q @ np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])]))
    a = best[1]; r = q @ np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
    ext = [float(np.percentile(r[:, k], 99) - np.percentile(r[:, k], 1)) for k in (0, 1)]
    print(f"[{tag}] {W}x{H}  fx {K[:,0,0].round(0)}  fy {K[:,1,1].round(0)}  EXIF f {f_exif:.0f}")
    print(f"   ceiling-above-cam0 {ceil:.3f}  horiz extents (p1-p99) {np.round(ext,3)}  max cam dist {d.max():.3f}")
    print(f"   shape ratios: extents/ceil-above-cam = {np.round(np.array(ext)/ceil,2)}   tape: 239/144={239/144:.2f} 291.5/144={291.5/144:.2f} (if phone at 1.35 m)")
    return v

print("A: portrait as-is")
analyse("portrait", imgs, (1, 1))           # image-down = camera +y
print("B: rotated 90deg CCW to landscape")
rot = [im.transpose(Image.Transpose.ROTATE_90) for im in imgs]
analyse("rot90", rot, (0, 1))               # original down -> camera +x after CCW rotation
