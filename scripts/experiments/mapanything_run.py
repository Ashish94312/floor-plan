"""MapAnything (Apache) on one room's photos, with/without EXIF intrinsics. Saves metric points.

E6-E10 ran this in a separate .venv-mapanything (same mapanything commit). Since the switch (D17)
it runs in the main env, fully offline (DINOv2 code from weights/dinov2-code):
  PYTORCH_ENABLE_MPS_FALLBACK=1 uv run python scripts/experiments/mapanything_run.py \
      <photos_dir>[,<photos_dir2>,...] <out_dir> k|nok [EXCLUDED.HEIC,...]
Several comma-separated room folders run as ONE joint reconstruction; each image's room
(folder name) is saved in `rooms`.
"""
import sys, time, threading, json
from pathlib import Path
import numpy as np, torch
from PIL import Image, ImageOps
from pillow_heif import register_heif_opener
from mapanything.utils.image import preprocess_inputs

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scan.device import go_offline  # noqa: E402
from scan.geometry.mapanything_backend import load_model  # noqa: E402

go_offline()
register_heif_opener()

photo_dirs, out, use_k = [Path(d) for d in sys.argv[1].split(",")], Path(sys.argv[2]), sys.argv[3] == "k"
out.mkdir(parents=True, exist_ok=True)
dev = torch.device("mps")
torch.manual_seed(0)
exclude = set(sys.argv[4].split(",")) if len(sys.argv) > 4 else set()
paths = [p for d in photo_dirs for p in sorted(d.iterdir())
         if p.suffix.lower() in {".heic", ".jpg", ".jpeg", ".png"} and p.name not in exclude]
views = []
for p in paths:
    im = ImageOps.exif_transpose(Image.open(p)).convert("RGB")
    W, H = im.size
    v = {"img": im}
    if use_k:
        f = 26 / 43.27 * np.hypot(W, H)   # EXIF 35mm-equivalent focal -> pixels
        v["intrinsics"] = torch.tensor([[f, 0, W / 2], [0, f, H / 2], [0, 0, 1]], dtype=torch.float32)
    views.append(v)
views = preprocess_inputs(views)

peak = [0]; stop = threading.Event()
def mon():
    while not stop.is_set():
        peak[0] = max(peak[0], torch.mps.driver_allocated_memory()); time.sleep(0.05)
t0 = time.perf_counter()
model = load_model(dev)
t_load = time.perf_counter() - t0
threading.Thread(target=mon, daemon=True).start()
t0 = time.perf_counter()
with torch.inference_mode():
    preds = model.infer(views, memory_efficient_inference=True, use_amp=True, amp_dtype="fp16",
                        apply_mask=True, mask_edges=True, apply_confidence_mask=False)
torch.mps.synchronize(); t_inf = time.perf_counter() - t0; stop.set()

pts = np.stack([p["pts3d"][0].float().cpu().numpy() for p in preds])          # (S,H,W,3) metric, world
mask = np.stack([p["mask"][0, ..., 0].cpu().numpy() if "mask" in p else np.ones(pts.shape[1:3], bool) for p in preds]).astype(bool)
conf = np.stack([p["conf"][0].float().cpu().numpy() for p in preds]) if "conf" in preds[0] else np.ones(pts.shape[:3])
c2w = np.stack([p["camera_poses"][0].float().cpu().numpy() for p in preds])   # (S,4,4)
K = np.stack([p["intrinsics"][0].float().cpu().numpy() for p in preds])
w2c = np.linalg.inv(c2w)[:, :3, :]
cols = np.stack([p["img_no_norm"][0].float().cpu().numpy() for p in preds])
np.savez_compressed(out / "ma_raw.npz", pts=pts, mask=mask, conf=conf, extrinsic=w2c, intrinsic=K, cols=cols,
                    names=np.array([p.name for p in paths]), rooms=np.array([p.parent.name for p in paths]))
print(json.dumps({"n_images": len(paths), "intrinsics_in": use_k, "hw": list(pts.shape[1:3]), "load_s": round(t_load, 1), "infer_s": round(t_inf, 1),
                  "peak_mps_gb": round(peak[0] / 2**30, 2), "fx_out": K[:, 0, 0].round(1).tolist(), "fy_out": K[:, 1, 1].round(1).tolist(),
                  "keys": sorted(preds[0].keys())}))
