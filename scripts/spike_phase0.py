"""Phase 0 spike (PLAN.md): one measured room's photos -> metric point cloud.

LEGACY (VGGT + Depth Anything): needs the `vggt` package, removed at the MapAnything switch (D17).
To re-run E1-E7: `git checkout 31dbc88 && uv sync && uv run scan-fetch-weights --optional`.

Answers O1/O2 with numbers:
  * VGGT-1B time + peak memory on this machine, fp32 vs fp16
  * Depth Anything V2 Metric scale estimate (s, sigma_log_s)
  * Room width / length / ceiling height from the scaled cloud vs the tape measure
  * OWLv2 detections (door, window, mirror, water stain, crack) look sane

Usage:
  uv run python scripts/spike_phase0.py captures/spike/photos/bedroom \
      --tape-short 312 --tape-long 405 --tape-height 274       # cm, optional
"""

from __future__ import annotations

import argparse
import gc
import json
import threading
import time
from pathlib import Path

import numpy as np
import psutil
import torch
from PIL import Image, ImageOps
from pillow_heif import register_heif_opener

from scan.device import go_offline, seed_everything, select_device
from scan.weights import model_path

register_heif_opener()
go_offline()

IMG_EXT = {".jpg", ".jpeg", ".png", ".heic", ".heif"}
VGGT_WIDTH = 518
PATCH = 14
OWL_QUERIES = ["a door", "a doorway", "a window", "a mirror", "a water stain on a wall", "a crack in a wall"]


# --------------------------------------------------------------------------- monitoring


class MemoryMonitor:
    """Samples process RSS, MPS driver memory and system available memory every 50 ms."""

    def __init__(self, device: torch.device):
        self.device = device
        self.proc = psutil.Process()
        self.peak_rss = 0
        self.peak_mps = 0
        self.min_avail = psutil.virtual_memory().available
        self.swap_start = psutil.swap_memory().used
        self.peak_swap = self.swap_start
        self._stop = threading.Event()
        self._t = threading.Thread(target=self._run, daemon=True)

    def _run(self):
        while not self._stop.is_set():
            self.peak_rss = max(self.peak_rss, self.proc.memory_info().rss)
            if self.device.type == "mps":
                self.peak_mps = max(self.peak_mps, torch.mps.driver_allocated_memory())
            self.min_avail = min(self.min_avail, psutil.virtual_memory().available)
            self.peak_swap = max(self.peak_swap, psutil.swap_memory().used)
            time.sleep(0.05)

    def __enter__(self):
        self._t.start()
        return self

    def __exit__(self, *exc):
        self._stop.set()
        self._t.join()

    def summary(self) -> dict:
        gb = 1024**3
        return {
            "peak_rss_gb": round(self.peak_rss / gb, 2),
            "peak_mps_driver_gb": round(self.peak_mps / gb, 2),
            "min_system_available_gb": round(self.min_avail / gb, 2),
            "swap_growth_gb": round((self.peak_swap - self.swap_start) / gb, 2),
        }


def sync(device: torch.device) -> None:
    if device.type == "mps":
        torch.mps.synchronize()
    elif device.type == "cuda":
        torch.cuda.synchronize()


def free(device: torch.device) -> None:
    gc.collect()
    if device.type == "mps":
        torch.mps.empty_cache()
    elif device.type == "cuda":
        torch.cuda.empty_cache()


# --------------------------------------------------------------------------- images


def list_images(folder: Path, max_images: int, exclude: tuple[str, ...] = ()) -> list[Path]:
    paths = sorted(p for p in folder.iterdir() if p.suffix.lower() in IMG_EXT and p.name not in exclude)
    if len(paths) < 2:
        raise SystemExit(f"Need at least 2 photos in {folder}, found {len(paths)}")
    return paths[:max_images]


def load_rgb(path: Path) -> Image.Image:
    """EXIF-rotated RGB. (VGGT's own loader skips exif_transpose.)"""
    return ImageOps.exif_transpose(Image.open(path)).convert("RGB")


def vggt_resize(img: Image.Image) -> Image.Image:
    """Long side -> 518, short side -> nearest multiple of 14. No crop, so portrait photos
    keep floor and ceiling (VGGT's crop mode would cut a portrait frame to a square)."""
    w, h = img.size
    k = VGGT_WIDTH / max(w, h)
    nw = VGGT_WIDTH if w >= h else round(w * k / PATCH) * PATCH
    nh = VGGT_WIDTH if h > w else round(h * k / PATCH) * PATCH
    return img.resize((nw, nh), Image.Resampling.BICUBIC)


def vggt_tensor(imgs: list[Image.Image]) -> torch.Tensor:
    out = [torch.from_numpy(np.asarray(vggt_resize(im), dtype=np.float32) / 255.0).permute(2, 0, 1) for im in imgs]
    shapes = {tuple(t.shape) for t in out}
    if len(shapes) != 1:
        raise SystemExit(f"Mixed image shapes after resize {shapes}; don't mix portrait and landscape.")
    return torch.stack(out)


# --------------------------------------------------------------------------- models


def mem_gb(device: torch.device) -> float:
    return round(torch.mps.driver_allocated_memory() / 1024**3, 2) if device.type == "mps" else 0.0


def run_vggt(images: torch.Tensor, device: torch.device, precision: str, dpt_chunk: int) -> dict:
    """Aggregator -> camera head -> depth head, run by hand so we can:
    skip the point head (we unproject depth + cameras), chunk the DPT head over frames,
    and keep the 0.9B-param aggregator in fp16 while the heads stay fp32 ('half')."""
    from safetensors.torch import load_file
    from vggt.models.vggt import VGGT
    from vggt.utils.pose_enc import pose_encoding_to_extri_intri

    t0 = time.perf_counter()
    model = VGGT(enable_track=False, enable_point=False)
    state = load_file(str(model_path("vggt-1b") / "model.safetensors"), device="cpu")
    missing, _unexpected = model.load_state_dict(state, strict=False)
    if missing:
        raise RuntimeError(f"VGGT weights missing keys: {missing[:5]}")
    del state
    if precision == "half":
        model.aggregator.half()
    model = model.to(device).eval()
    sync(device)
    t_load = time.perf_counter() - t0
    stages = {"after_load_gb": mem_gb(device)}

    x = images.to(device)
    t0 = time.perf_counter()
    with torch.inference_mode():
        autocast = precision == "fp16" and device.type != "cpu"
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=autocast):
            xin = x.half() if precision == "half" else x
            tokens, ps_idx = model.aggregator(xin[None])
            sync(device)
            stages["after_aggregator_gb"] = mem_gb(device)
            if precision == "half":
                tokens = [t.float() if t is not None else None for t in tokens]
            pose_enc = model.camera_head(tokens)[-1]
            depth, depth_conf = model.depth_head(
                tokens, images=x[None], patch_start_idx=ps_idx, frames_chunk_size=dpt_chunk
            )
        sync(device)
        stages["after_heads_gb"] = mem_gb(device)
        extr, intr = pose_encoding_to_extri_intri(pose_enc.float(), x.shape[-2:])
    t_inf = time.perf_counter() - t0

    out = {
        "extrinsic": extr[0].float().cpu().numpy(),  # (S,3,4) cam-from-world, OpenCV
        "intrinsic": intr[0].float().cpu().numpy(),  # (S,3,3)
        "depth": depth[0, ..., 0].float().cpu().numpy(),  # (S,H,W)
        "depth_conf": depth_conf[0].float().cpu().numpy(),  # (S,H,W)
        "t_load_s": t_load,
        "t_infer_s": t_inf,
        "stages": stages,
    }
    del model, tokens, depth, depth_conf, x
    free(device)
    return out


def run_metric_depth(
    imgs: list[Image.Image], hw: tuple[int, int], device: torch.device, rotate: bool
) -> tuple[np.ndarray, float]:
    """Depth Anything V2 Metric Indoor (Small), in metres, resized to VGGT resolution.
    Runs on the upright photo; if VGGT got a rotated copy, the depth map is rotated to match."""
    from transformers import AutoImageProcessor, AutoModelForDepthEstimation

    path = model_path("dav2-metric-indoor-small")
    proc = AutoImageProcessor.from_pretrained(path)
    model = AutoModelForDepthEstimation.from_pretrained(path).to(device).eval()
    t0 = time.perf_counter()
    depths = []
    with torch.inference_mode():
        for img in imgs:
            inputs = proc(images=vggt_resize(img), return_tensors="pt").to(device)  # same framing as VGGT
            pred = model(**inputs).predicted_depth  # (1,h',w')
            size = (hw[1], hw[0]) if rotate else hw
            d = torch.nn.functional.interpolate(pred[:, None], size=size, mode="bilinear", align_corners=False)
            d = d[0, 0].float().cpu().numpy()
            depths.append(np.ascontiguousarray(np.rot90(d)) if rotate else d)  # rot90 = CCW, like PIL ROTATE_90
    sync(device)
    t = time.perf_counter() - t0
    del model
    free(device)
    return np.stack(depths), t


def run_owl(imgs: list[Image.Image], names: list[str], out_dir: Path, device: torch.device, thr: float) -> tuple[dict, float]:
    from PIL import ImageDraw
    from transformers import Owlv2ForObjectDetection, Owlv2Processor

    path = model_path("owlv2-base-ensemble")
    proc = Owlv2Processor.from_pretrained(path)
    model = Owlv2ForObjectDetection.from_pretrained(path).to(device).eval()
    out_dir.mkdir(parents=True, exist_ok=True)
    results = {}
    t0 = time.perf_counter()
    with torch.inference_mode():
        for img, name in zip(imgs, names):
            small = img.copy()
            small.thumbnail((960, 960))
            inputs = proc(text=[OWL_QUERIES], images=small, return_tensors="pt").to(device)
            outputs = model(**inputs)
            side = max(small.size)  # OWLv2 pads to a square; boxes live in that frame
            det = proc.post_process_grounded_object_detection(
                outputs, threshold=thr, target_sizes=[(side, side)], text_labels=[OWL_QUERIES]
            )[0]
            draw = ImageDraw.Draw(small)
            dets = []
            for box, score, label in zip(det["boxes"], det["scores"], det["text_labels"]):
                b = [round(v, 1) for v in box.tolist()]
                dets.append({"label": label, "score": round(float(score), 3), "box": b})
                draw.rectangle(b, outline="red", width=3)
                draw.text((b[0] + 4, b[1] + 4), f"{label} {score:.2f}", fill="red")
            small.save(out_dir / f"{Path(name).stem}.jpg", quality=85)
            results[name] = dets
    sync(device)
    t = time.perf_counter() - t0
    del model
    free(device)
    return results, t


# --------------------------------------------------------------------------- geometry


def scale_from_metric(d_vggt, conf, d_metric) -> dict:
    """Per-image median ratio metric/VGGT on confident pixels -> (s, sigma_log_s)."""
    per_img = []
    for dv, c, dm in zip(d_vggt, conf, d_metric):
        m = (c > np.percentile(c, 50)) & (dv > 1e-4) & (dm > 0.2) & (dm < 10.0)
        per_img.append(float(np.median(dm[m] / dv[m])))
    logs = np.log(per_img)
    s = float(np.exp(np.median(logs)))
    mad = 1.4826 * float(np.median(np.abs(logs - np.median(logs))))
    return {
        "s": s,
        "sigma_log_s_between_images": mad,
        "sigma_log_s_of_median": mad / np.sqrt(len(logs)),
        "per_image": [round(v, 4) for v in per_img],
    }


def unproject(depth, extr, intr, conf, keep_pct: float):
    from vggt.utils.geometry import unproject_depth_map_to_point_map

    pts = unproject_depth_map_to_point_map(depth[..., None], extr, intr)  # (S,H,W,3)
    thr = np.percentile(conf, 100 - keep_pct)
    mask = (conf >= thr) & (depth > 1e-4)
    return pts[mask], mask


def fit_plane_normal(p: np.ndarray) -> np.ndarray:
    c = p - p.mean(0)
    return np.linalg.svd(c, full_matrices=False)[2][-1]


def peak_extremes(x: np.ndarray, bin_m: float = 0.01, rel: float = 0.2) -> tuple[float, float]:
    """Outermost histogram peaks (walls) along one axis."""
    from scipy.ndimage import gaussian_filter1d
    from scipy.signal import find_peaks

    lo, hi = np.percentile(x, [0.2, 99.8])
    edges = np.arange(lo - 0.1, hi + 0.1 + bin_m, bin_m)
    h, _ = np.histogram(x, edges)
    hs = gaussian_filter1d(h.astype(float), 2)
    peaks, _ = find_peaks(hs, height=rel * hs.max())
    if len(peaks) < 2:
        return float(lo), float(hi)
    centres = (edges[:-1] + edges[1:]) / 2
    return float(centres[peaks[0]]), float(centres[peaks[-1]])


def measure_room(pts: np.ndarray, extr: np.ndarray, rng: np.random.Generator, down_axis: int = 1) -> dict:
    """Gravity from camera 'down' axes, refined by floor/ceiling planes; Manhattan walls."""
    if len(pts) > 400_000:
        pts = pts[rng.choice(len(pts), 400_000, replace=False)]

    # Camera +y is image-down (OpenCV). In world coords that's row 1 of R_cam_from_world.
    # Rotated portrait input: photo-down became image-right, i.e. camera +x (row 0).
    down = extr[:, down_axis, :3].mean(0)
    up = -down / np.linalg.norm(down)

    for _ in range(2):  # refine 'up' from floor + ceiling planes
        h = pts @ up
        floor_h, ceil_h = peak_extremes(h, bin_m=0.01, rel=0.15)
        band = 0.03
        nf = fit_plane_normal(pts[np.abs(h - floor_h) < band])
        nc = fit_plane_normal(pts[np.abs(h - ceil_h) < band])
        nf = nf if nf @ up > 0 else -nf
        nc = nc if nc @ up > 0 else -nc
        up = (nf + nc) / np.linalg.norm(nf + nc)

    h = pts @ up
    floor_h, ceil_h = peak_extremes(h, bin_m=0.01, rel=0.15)
    height = ceil_h - floor_h

    # Horizontal basis; upper wall band avoids most furniture.
    e1 = np.cross(up, [1.0, 0, 0] if abs(up[0]) < 0.9 else [0, 1.0, 0])
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(up, e1)
    sel = (h > floor_h + 0.5 * height) & (h < ceil_h - 0.10)
    q = np.stack([pts[sel] @ e1, pts[sel] @ e2], 1)

    # Manhattan angle: rotation that makes x/y histograms sharpest.
    best = (-1.0, 0.0)
    for deg in np.arange(0, 90, 0.25):
        a = np.deg2rad(deg)
        r = q @ np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
        sc = sum(float((np.histogram(r[:, k], bins=np.arange(r[:, k].min(), r[:, k].max() + 0.02, 0.02))[0] ** 2).sum()) for k in (0, 1))
        if sc > best[0]:
            best = (sc, deg)
    a = np.deg2rad(best[1])
    rot = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
    r = q @ rot
    x0, x1 = peak_extremes(r[:, 0])
    y0, y1 = peak_extremes(r[:, 1])
    dims = sorted([x1 - x0, y1 - y0])
    return {
        "ceiling_height_m": height,
        "short_side_m": dims[0],
        "long_side_m": dims[1],
        "floor_area_m2": dims[0] * dims[1],
        "manhattan_deg": best[1],
        "_plot": {"r": r, "x": (x0, x1), "y": (y0, y1), "h": h, "fh": floor_h, "ch": ceil_h},
    }


def save_ply(path: Path, pts: np.ndarray, cols: np.ndarray) -> None:
    import open3d as o3d

    pc = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pts.astype(np.float64)))
    pc.colors = o3d.utility.Vector3dVector(cols.astype(np.float64))
    o3d.io.write_point_cloud(str(path), pc)


def save_plots(out: Path, room: dict) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    p = room["_plot"]
    fig, ax = plt.subplots(1, 2, figsize=(13, 6))
    r = p["r"][:: max(1, len(p["r"]) // 60000)]
    ax[0].scatter(r[:, 0], r[:, 1], s=0.3, c="0.3")
    for v in p["x"]:
        ax[0].axvline(v, c="r")
    for v in p["y"]:
        ax[0].axhline(v, c="r")
    ax[0].set_aspect("equal")
    ax[0].set_title(f"Top-down (upper wall band) {room['short_side_m']:.2f} x {room['long_side_m']:.2f} m")
    ax[1].hist(p["h"], bins=300, color="0.4")
    ax[1].axvline(p["fh"], c="r")
    ax[1].axvline(p["ch"], c="r")
    ax[1].set_title(f"Heights along gravity  ceiling {room['ceiling_height_m']:.3f} m")
    fig.tight_layout()
    fig.savefig(out / "spike_room.png", dpi=120)
    plt.close(fig)


# --------------------------------------------------------------------------- main


def pct_err(est_m: float, tape_cm: float | None):
    return None if tape_cm is None else round(100 * (est_m * 100 - tape_cm) / tape_cm, 2)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("photos", type=Path, help="Folder with one room's photos")
    ap.add_argument("--out", type=Path, default=None, help="Default: <photos>/../../out_spike/<room>")
    ap.add_argument("--max-images", type=int, default=8)
    ap.add_argument("--exclude", nargs="*", default=[], help="File names to leave out (e.g. one odd-orientation shot)")
    ap.add_argument("--precision", choices=["fp32", "fp16", "half"], default="half",
                    help="fp16 = autocast; half = fp16 aggregator weights, fp32 heads")
    ap.add_argument("--dpt-chunk", type=int, default=2, help="Frames per DPT depth-head pass")
    ap.add_argument("--device", default=None)
    ap.add_argument("--keep-pct", type=float, default=50, help="Keep top N%% of points by VGGT confidence")
    ap.add_argument("--owl-threshold", type=float, default=0.15)
    ap.add_argument("--skip-owl", action="store_true")
    ap.add_argument("--tape-short", type=float, help="Tape: shorter wall-to-wall distance, cm")
    ap.add_argument("--tape-long", type=float, help="Tape: longer wall-to-wall distance, cm")
    ap.add_argument("--tape-height", type=float, help="Tape: ceiling height, cm")
    args = ap.parse_args()

    rng = seed_everything()
    device = select_device(args.device)
    paths = list_images(args.photos, args.max_images, tuple(args.exclude))
    out = args.out or args.photos.parent.parent / "out_spike" / f"{args.photos.name}_{args.precision}"
    out.mkdir(parents=True, exist_ok=True)
    print(f"device={device} precision={args.precision} images={len(paths)} -> {out}")

    imgs = [load_rgb(p) for p in paths]
    # VGGT mis-predicts horizontal FoV on portrait frames (fx ~26% low vs EXIF on iPhone 13),
    # stretching the room sideways. Rotated to landscape it predicts fx == fy within 8% of EXIF.
    portrait = imgs[0].height > imgs[0].width
    x = vggt_tensor([im.transpose(Image.Transpose.ROTATE_90) for im in imgs] if portrait else imgs)
    hw = tuple(x.shape[-2:])
    report: dict = {"device": str(device), "precision": args.precision, "n_images": len(paths), "vggt_hw": hw,
                    "portrait_rotated": portrait}

    t_total = time.perf_counter()
    with MemoryMonitor(device) as mon:
        v = run_vggt(x, device, args.precision, args.dpt_chunk)
    report["vggt"] = {"load_s": round(v["t_load_s"], 2), "infer_s": round(v["t_infer_s"], 2), **mon.summary(), "stages": v["stages"]}
    print(f"VGGT   load {v['t_load_s']:.1f}s  infer {v['t_infer_s']:.1f}s  {mon.summary()}  {v['stages']}")

    with MemoryMonitor(device) as mon:
        d_metric, t_md = run_metric_depth(imgs, hw, device, rotate=portrait)
    report["metric_depth"] = {"infer_s": round(t_md, 2), **mon.summary()}
    sc = scale_from_metric(v["depth"], v["depth_conf"], d_metric)
    report["scale"] = sc
    print(f"DAv2   {t_md:.1f}s  s={sc['s']:.4f}  sigma_log_s(between imgs)={sc['sigma_log_s_between_images']:.3f}")

    # Scale the whole VGGT reconstruction: depth and camera translations share one scale.
    s = sc["s"]
    extr = v["extrinsic"].copy()
    extr[:, :, 3] *= s
    pts, mask = unproject(v["depth"] * s, extr, v["intrinsic"], v["depth_conf"], args.keep_pct)
    cols = x.permute(0, 2, 3, 1).numpy()[mask]
    room = measure_room(pts, extr, rng, down_axis=0 if portrait else 1)
    report["room"] = {
        k: round(val, 4) for k, val in room.items() if not k.startswith("_")
    }
    report["vs_tape_pct"] = {
        "short_side": pct_err(room["short_side_m"], args.tape_short),
        "long_side": pct_err(room["long_side_m"], args.tape_long),
        "ceiling_height": pct_err(room["ceiling_height_m"], args.tape_height),
    }
    np.savez_compressed(out / "raw.npz", extrinsic=v["extrinsic"], intrinsic=v["intrinsic"], depth=v["depth"],
                        depth_conf=v["depth_conf"], depth_metric=d_metric, scale=s, names=np.array([p.name for p in paths]))
    sub = rng.choice(len(pts), min(len(pts), 500_000), replace=False)
    save_ply(out / "cloud.ply", pts[sub], cols[sub])
    save_plots(out, room)
    print(
        f"ROOM   {room['short_side_m']:.3f} x {room['long_side_m']:.3f} m, "
        f"ceiling {room['ceiling_height_m']:.3f} m, area {room['floor_area_m2']:.2f} m2  "
        f"vs tape %: {report['vs_tape_pct']}"
    )

    if not args.skip_owl:
        with MemoryMonitor(device) as mon:
            dets, t_owl = run_owl(imgs, [p.name for p in paths], out / "owl", device, args.owl_threshold)
        report["owl"] = {"infer_s": round(t_owl, 2), **mon.summary(), "detections": dets}
        n = sum(len(d) for d in dets.values())
        print(f"OWLv2  {t_owl:.1f}s  {n} detections -> {out / 'owl'}")

    report["total_s"] = round(time.perf_counter() - t_total, 2)
    (out / "spike_report.json").write_text(json.dumps(report, indent=2))
    print(f"TOTAL  {report['total_s']}s  report -> {out / 'spike_report.json'}")


if __name__ == "__main__":
    main()
