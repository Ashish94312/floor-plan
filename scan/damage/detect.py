"""C8 Damage (ARCHITECTURE §10.10) + detector-based closed doors / windows / mirrors.

OWLv2 runs once per photo on the 1024 px ingest copy (results cached). Each box is lifted into the
aligned 3D frame: its centre pixel's depth gives the surface it lies on (nearest wall within 25 cm,
or floor / ceiling); its corners are cast as rays onto that surface's plane, giving metric width,
height and position (from_left = distance from the wall's start, which is the left end seen from
inside; from_floor = height). Detections of the same thing in several photos are merged.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class SurfaceBox:
    cls: str
    surface_id: str  # <room>-W<n> | <room>-F | <room>-C
    u0: float  # wall: along the wall from its start; floor/ceiling: x
    u1: float
    v0: float  # wall: height from the floor; floor/ceiling: y
    v1: float
    score: float
    photo: str


# ----------------------------------------------------------------------------------- detector


def _cache_key(frames, cfg):
    from scan.cache import cache_key
    from scan.weights import MODELS

    d = cfg["detector"]
    return cache_key(kind="owlv2", images=[f.meta.sha256 for f in frames], weights=MODELS["owlv2-base-ensemble"].revision,
                     classes=d["classes"], side=d["image_side"], nms=d["nms_iou"],
                     working_max_side=cfg["ingest"]["working_max_side"])


def detect(frames, cfg, device, use_cache=True) -> dict[str, list[dict]]:
    """photo name -> [{cls, score, box (x0, y0, x1, y1) in the frame's rgb pixels}]."""
    import json

    from scan import cache

    key = _cache_key(frames, cfg)
    if use_cache and (hit := cache.load(cfg, "owlv2", key)) is not None:
        return json.loads(str(hit["json"]))
    import torch
    from PIL import Image
    from torchvision.ops import nms
    from transformers import Owlv2ForObjectDetection, Owlv2Processor

    from scan.weights import model_path

    d = cfg["detector"]
    path = model_path("owlv2-base-ensemble")
    proc = Owlv2Processor.from_pretrained(path)
    model = Owlv2ForObjectDetection.from_pretrained(path).to(device).eval()
    labels, owner = [], []
    for cls, c in d["classes"].items():
        for q in c["queries"]:
            labels.append(q)
            owner.append(cls)
    out: dict[str, list[dict]] = {}
    with torch.inference_mode():
        for f in frames:
            img = Image.fromarray(f.rgb)
            side = max(img.size)  # OWLv2 pads to a square (top-left anchored)
            inputs = proc(text=[labels], images=img, return_tensors="pt").to(device)
            res = proc.post_process_grounded_object_detection(
                model(**inputs), threshold=min(c["threshold"] for c in d["classes"].values()),
                target_sizes=[(side, side)], text_labels=[labels])[0]
            dets = []
            for cls in d["classes"]:
                idx = [k for k, lab in enumerate(res["labels"].tolist()) if owner[lab] == cls
                       and float(res["scores"][k]) >= d["classes"][cls]["threshold"]]
                if not idx:
                    continue
                boxes, scores = res["boxes"][idx].float().cpu(), res["scores"][idx].float().cpu()
                for k in nms(boxes, scores, d["nms_iou"]).tolist():
                    dets.append({"cls": cls, "score": round(float(scores[k]), 4),
                                 "box": [round(float(v), 1) for v in boxes[k].tolist()]})
            out[f.image_path.name] = dets
    if use_cache:
        cache.save(cfg, "owlv2", key, {"json": np.array(json.dumps(out))})
    return out


# ----------------------------------------------------------------------------------- lifting


def _frame_to_view(frame_hw, view_hw):
    """MapAnything resizes by max(scale) and centre-crops (verified to 1 px, WORKLOG E18)."""
    H, W = frame_hw
    Hd, Wd = view_hw
    s = max(Wd / W, Hd / H)
    return s, (W * s - Wd) / 2, (H * s - Hd) / 2


def _ray(K, T, px, py):
    """World ray (origin, unit direction) through pixel (px, py) of a view."""
    d_cam = np.array([(px - K[0, 2]) / K[0, 0], (py - K[1, 2]) / K[1, 1], 1.0])
    d = T[:3, :3] @ d_cam
    return T[:3, 3], d / np.linalg.norm(d)


def lift(det, frame_hw, i, views, layout, planes, dcfg) -> SurfaceBox | None:
    """Box -> surface + metric extent on it."""
    K, T, depth = views["K_exif"][i], views["T_wc"][i], views["depth"][i]
    s, ox, oy = _frame_to_view(frame_hw, depth.shape)
    x0, y0, x1, y1 = det["box"]
    cx, cy = (x0 + x1) / 2 * s - ox, (y0 + y1) / 2 * s - oy
    ix, iy = int(np.clip(cx, 0, depth.shape[1] - 1)), int(np.clip(cy, 0, depth.shape[0] - 1))
    o, r = _ray(K, T, cx, cy)
    zc = depth[iy, ix]
    if not np.isfinite(zc) or zc <= 0:
        return None
    p = o + r * zc / (T[:3, :3].T @ r)[2]  # depth is z-depth: scale the unit ray to that z
    room = layout.walls[0].wall_id.rsplit("-W", 1)[0]
    # nearest surface
    best = (dcfg["max_surface_dist_m"], None)
    for w in layout.walls:
        a, b = np.array(w.start), np.array(w.end)
        L = np.linalg.norm(b - a)
        u_dir = (b - a) / L
        rel = p[:2] - a
        u, off = float(rel @ u_dir), abs(float(rel @ np.array([-u_dir[1], u_dir[0]])))
        if -0.1 < u < L + 0.1 and off < best[0]:
            best = (off, ("wall", w, a, u_dir))
    ceil = planes.ceiling_z
    if abs(p[2] - planes.floor_z) < best[0]:
        best = (abs(p[2] - planes.floor_z), ("floor", None, None, None))
    if ceil is not None and abs(p[2] - ceil) < best[0]:
        best = (abs(p[2] - ceil), ("ceiling", None, None, None))
    if best[1] is None:
        return None
    kind, w, a, u_dir = best[1]
    corners = [(x0, y0), (x1, y0), (x0, y1), (x1, y1)]
    uv = []
    for X, Y in corners:
        o, r = _ray(K, T, X * s - ox, Y * s - oy)
        if kind == "wall":
            n = np.array([-u_dir[1], u_dir[0], 0.0])
            denom = r @ n
            if abs(denom) < 1e-6:
                return None
            q = o + r * (((np.append(a, 0) - o) @ n) / denom)
            uv.append(((q[:2] - a) @ u_dir, q[2]))
        else:
            zplane = planes.floor_z if kind == "floor" else ceil
            if abs(r[2]) < 1e-6:
                return None
            q = o + r * ((zplane - o[2]) / r[2])
            uv.append((q[0], q[1]))
    uv = np.array(uv)
    sid = w.wall_id if kind == "wall" else f"{room}-{'F' if kind == 'floor' else 'C'}"
    return SurfaceBox(det["cls"], sid, float(uv[:, 0].min()), float(uv[:, 0].max()), float(uv[:, 1].min()),
                      float(uv[:, 1].max()), det["score"], "")


def _iou(a: SurfaceBox, b: SurfaceBox) -> float:
    iw = max(0.0, min(a.u1, b.u1) - max(a.u0, b.u0))
    ih = max(0.0, min(a.v1, b.v1) - max(a.v0, b.v0))
    inter = iw * ih
    union = (a.u1 - a.u0) * (a.v1 - a.v0) + (b.u1 - b.u0) * (b.v1 - b.v0) - inter
    return inter / union if union > 0 else 0.0


def merge(boxes: list[SurfaceBox], iou: float) -> list[list[SurfaceBox]]:
    """Group boxes of one class on one surface that overlap (same object seen from several photos)."""
    groups: list[list[SurfaceBox]] = []
    for b in sorted(boxes, key=lambda b: -b.score):
        for g in groups:
            if g[0].cls == b.cls and g[0].surface_id == b.surface_id and _iou(g[0], b) > iou:
                g.append(b)
                break
        else:
            groups.append([b])
    return groups


# ----------------------------------------------------------------------------------- per capture


@dataclass
class DamageSeg:
    damage_id: str
    surface_id: str
    cls: str
    width_m: float
    height_m: float
    area_m2: float
    from_left_m: float
    from_floor_m: float
    score: float
    photos: int


def surface_boxes(cap, clouds, cfg, device, use_cache=True) -> dict[str, list[SurfaceBox]]:
    """Detector boxes lifted onto each room's surfaces, tagged with their photo."""
    frames = [f for fs in cap.rooms.values() for f in fs]
    dets = detect(frames, cfg, device, use_cache)
    out: dict[str, list[SurfaceBox]] = {}
    for r, c in clouds.items():
        names = [str(n) for n in c.views["names"]]
        boxes = []
        for f in cap.rooms[r]:
            if f.image_path.name not in names:  # dropped as misplaced (check_views)
                continue
            i = names.index(f.image_path.name)
            for d in dets.get(f.image_path.name, []):
                sb = lift(d, f.rgb.shape[:2], i, c.views, c.layout, c.planes, cfg["damage"])
                if sb is not None:
                    sb.photo = f.image_path.name
                    boxes.append(sb)
        out[r] = boxes
    return out


def _consensus(group):
    """Median extent of one object's boxes (clamped to the surface: offsets >= 0, heights >= 0) and the
    number of distinct photos."""
    a = np.array([[b.u0, b.u1, b.v0, b.v1] for b in group])
    u0, u1, v0, v1 = np.median(a, 0)
    if "-W" in group[0].surface_id:  # wall: u along the wall from its start, v = height
        u0, v0 = max(u0, 0.0), max(v0, 0.0)
    return float(u0), float(u1), float(v0), float(v1), max(b.score for b in group), len({b.photo for b in group})


def damage_regions(boxes: dict[str, list[SurfaceBox]], cfg) -> dict[str, list[DamageSeg]]:
    dc = cfg["damage"]
    out = {}
    for r, bs in boxes.items():
        regs = []
        dmg = [b for b in bs if b.cls in dc["fill"] and _surface_kind(b.surface_id) in dc["surfaces"]]
        for g in merge(dmg, dc["merge_iou"]):
            u0, u1, v0, v1, score, n = _consensus(g)
            if n < dc["min_photos"]:
                continue
            w, h = u1 - u0, v1 - v0
            regs.append(DamageSeg(f"{r}-D{len(regs) + 1}", g[0].surface_id, g[0].cls, round(w, 3), round(h, 3),
                                  round(w * h * dc["fill"][g[0].cls], 4), round(u0, 3), round(v0, 3), round(score, 3), n))
        out[r] = regs
    return out


def detector_openings(boxes: dict[str, list[SurfaceBox]], openings: dict, cfg) -> list[str]:
    """Closed doors / covered windows the depth voting can't see: add detector doors/windows seen in
    >= min_photos photos where no opening exists yet. Mirrors cancel see-through openings they cover.
    Returns log lines."""
    from scan.layout.openings import OpeningSeg

    dc = cfg["damage"]
    log = []
    for r, bs in boxes.items():
        ops = openings[r]
        for g in merge([b for b in bs if b.cls == "mirror" and b.surface_id.rsplit("-", 1)[1].startswith("W")], 0.3):
            u0, u1, v0, v1, _, n = _consensus(g)
            for o in list(ops):
                if o.wall_id == g[0].surface_id and min(u1, o.offset_m + o.width_m) - max(u0, o.offset_m) > 0.5 * o.width_m:
                    ops.remove(o)
                    log.append(f"{o.opening_id} removed: a mirror covers it (phantom opening)")
        for g in merge([b for b in bs if b.cls in ("door", "window")], 0.3):
            u0, u1, v0, v1, _, n = _consensus(g)
            if n < dc["min_photos"] or "-W" not in g[0].surface_id:
                continue
            w, h = u1 - u0, v1 - v0
            if g[0].cls == "door" and (v0 > 0.35 or h < 1.6 or not 0.5 < w < 1.4):
                continue  # a door reaches the floor and is door-sized
            if any(o.wall_id == g[0].surface_id and min(u1, o.offset_m + o.width_m) - max(u0, o.offset_m) > 0.3 * w for o in ops):
                continue  # the geometric opening already covers it (and measures it better)
            kind = g[0].cls
            ops.append(OpeningSeg(f"{r}-O{len(ops) + 1}", kind, g[0].surface_id, round(u0, 3), round(w, 3),
                                  round(v1 if kind == "door" else h, 3), None if kind == "door" else round(v0, 3), n,
                                  notes=["detector (closed door / covered window): box extent, wider interval"]))
            log.append(f"{r}-O{len(ops)}: {kind} from the detector on {g[0].surface_id}, {w:.2f} x {h:.2f} m ({n} photos)")
    return log


def _surface_kind(sid: str) -> str:
    tail = sid.rsplit("-", 1)[1]
    return "wall" if tail.startswith("W") else ("floor" if tail == "F" else "ceiling")
