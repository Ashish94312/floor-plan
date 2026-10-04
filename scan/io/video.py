"""Video tier input (ARCHITECTURE §10.1): clips -> sharp, well-spread keyframes as Frames.

Per-room clips (video/<room>/*.mov) map onto the photo pipeline: each room's clip becomes a handful
of keyframes (one sharpest frame per time slice, so they cover the clip and avoid motion blur).
Video files carry no focal length, and stabilisation crops the view, so the focal is estimated from
vanishing points over the keyframes (median), falling back to a nominal value with a warning.
A single continuous walkthrough needs room segmentation (C4) and is not supported yet.
"""

from __future__ import annotations

import hashlib
import json
import math
import subprocess
from pathlib import Path

import cv2
import numpy as np

from scan.errors import InputError
from scan.io.photos import intrinsics
from scan.types import Frame, PhotoMeta

VIDEO_EXT = {".mov", ".mp4", ".m4v"}


def probe(path: Path) -> dict:
    out = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
                          "stream=width,height:stream_side_data=rotation:format=duration:format_tags=com.apple.quicktime.model",
                          "-of", "json", str(path)], capture_output=True, text=True, check=False)
    if out.returncode != 0:
        raise InputError(f"{path}: cannot read video ({out.stderr.strip()[:200]})")
    d = json.loads(out.stdout)
    st = d["streams"][0]
    rot = next((sd.get("rotation", 0) for sd in st.get("side_data_list", []) if "rotation" in sd), 0)
    W, H = (st["height"], st["width"]) if abs(int(rot)) % 180 == 90 else (st["width"], st["height"])
    return {"W": W, "H": H, "duration": float(d["format"]["duration"]),
            "model": d["format"].get("tags", {}).get("com.apple.quicktime.model")}


def decode(path: Path, fps: float, max_side: int) -> tuple[np.ndarray, np.ndarray, dict]:
    """All frames at `fps`, upright (ffmpeg applies the rotation), long side = max_side."""
    info = probe(path)
    k = max_side / max(info["W"], info["H"])
    w, h = int(round(info["W"] * k / 2) * 2), int(round(info["H"] * k / 2) * 2)
    p = subprocess.run(["ffmpeg", "-v", "error", "-i", str(path), "-vf", f"fps={fps},scale={w}:{h}",
                        "-f", "rawvideo", "-pix_fmt", "rgb24", "-"], capture_output=True, check=False)
    if p.returncode != 0 or not p.stdout:
        raise InputError(f"{path}: video decode failed ({p.stderr.decode()[:200]})")
    frames = np.frombuffer(p.stdout, np.uint8).reshape(-1, h, w, 3)
    return frames, np.arange(len(frames)) / fps, info


def keyframes(frames: np.ndarray, k: int, min_sharpness: float, min_gap: int = 0) -> list[int]:
    """One sharpest frame per equal time slice (coverage + no motion blur), at least `min_gap` frames
    after the previous pick (neighbouring slices could otherwise pick near-duplicates at their border)."""
    sharp = np.array([cv2.Laplacian(cv2.cvtColor(f, cv2.COLOR_RGB2GRAY), cv2.CV_64F).var() for f in frames])
    picks = []
    for sl in np.array_split(np.arange(len(frames)), k):
        if picks:
            sl = sl[sl >= picks[-1] + min_gap]
        if len(sl) and sharp[sl].max() >= min_sharpness:
            picks.append(int(sl[np.argmax(sharp[sl])]))
    return picks, sharp


def link_features(frames: np.ndarray, sharp: np.ndarray, idx, side: int, min_sharpness: float) -> list:
    """SIFT features of candidate frames (indices `idx`, sharp ones only) at `side` px long side:
    [(frame index, keypoints xy, descriptors)]."""
    sift = cv2.SIFT_create(nfeatures=1500)
    out = []
    for i in idx:
        if sharp[i] < min_sharpness:
            continue
        g = cv2.cvtColor(frames[i], cv2.COLOR_RGB2GRAY)
        k = side / max(g.shape)
        g = cv2.resize(g, None, fx=k, fy=k, interpolation=cv2.INTER_AREA)
        kp, d = sift.detectAndCompute(g, None)
        if d is not None and len(kp) >= 50:
            out.append((int(i), np.float32([p.pt for p in kp]), d))
    return out


def link_pairs(feats_a: list, feats_b: list, min_inliers: int, ratio: float = 0.8, shortlist: int = 3
               ) -> list[tuple[int, int, int]]:
    """Frame pairs (inliers, frame in A, frame in B) where two clips see the same thing: a doorway, or
    the room beyond it. Two stages: (1) each A frame's descriptors are looked up in all B frames at once
    and the B frames with the most nearest neighbours are shortlisted (no ratio test here: neighbouring
    B frames hold the same points, so the 1st and 2nd neighbour are often one point twice); (2) each
    shortlisted pair is matched on its own (ratio test) and verified with a RANSAC fundamental matrix,
    so plain white walls or marble that merely look alike do not count. Sorted best first."""
    if not feats_a or not feats_b:
        return []
    index = cv2.FlannBasedMatcher({"algorithm": 1, "trees": 4}, {"checks": 32})
    index.add([d for _, _, d in feats_b])
    index.train()
    bf = cv2.BFMatcher(cv2.NORM_L2)
    out = []
    for fa, pa, da in feats_a:
        votes = np.bincount([m[0].imgIdx for m in index.knnMatch(da, k=1) if m], minlength=len(feats_b))
        best = (0, None)
        for jb in np.argsort(votes)[::-1][:shortlist]:
            _, pb, db = feats_b[jb]
            good = [m for m, n in (x for x in bf.knnMatch(da, db, k=2) if len(x) == 2) if m.distance < ratio * n.distance]
            if len(good) < min_inliers:
                continue
            _F, mask = cv2.findFundamentalMat(pa[[m.queryIdx for m in good]], pb[[m.trainIdx for m in good]],
                                              cv2.FM_RANSAC, 2.0, 0.999)
            inl = int(mask.sum()) if mask is not None else 0
            if inl > best[0]:
                best = (inl, int(jb))
        if best[0] >= min_inliers:
            out.append((best[0], fa, feats_b[best[1]][0]))
    return sorted(out, reverse=True)


def ingest_video(cap, cfg: dict) -> None:
    vc = cfg["video"]
    vdir = cap.root / "video"
    loose = [f for f in vdir.iterdir() if f.suffix.lower() in VIDEO_EXT]
    if loose:
        raise InputError(f"{vdir}: a single walkthrough clip ({loose[0].name}) needs room segmentation, not supported yet. "
                         "Record one clip per room into video/<room>/ instead.")
    room_dirs = sorted(d for d in vdir.iterdir() if d.is_dir() and not d.name.startswith("."))
    clips = {d.name: sorted(f for f in d.iterdir() if f.suffix.lower() in VIDEO_EXT) for d in room_dirs}
    clips = {r: c for r, c in clips.items() if c}
    if not clips:
        raise InputError(f"{vdir}: no clips. Expected video/<room>/*.mov")
    per_room = vc["keyframes_per_room"]
    from scan.geometry.vanishing import focal_px

    linking = vc["link_frames_per_pair"] > 0 and len(clips) > 1
    raw, fs = {}, []
    lcand: dict[str, list] = {}  # room -> [(frame, name, sharpness)] link candidates
    lfeat: dict[str, list] = {}  # room -> [(index into lcand[room], keypoints, descriptors)]
    for room, files in clips.items():
        fr_list, names, sharps, flags = [], [], [], []
        lcand[room], lfeat[room] = [], []
        for f in files:
            frames, ts, info = decode(f, vc["decode_fps"], cfg["ingest"]["working_max_side"])
            picks, sharp = keyframes(frames, max(1, per_room // len(files)), vc["min_sharpness"],
                                     min_gap=round(vc["min_keyframe_gap_s"] * vc["decode_fps"]))
            for i in np.argsort(sharp)[::-1][: vc["vp_frames_per_clip"]]:  # focal: many sharp frames, not just keyframes
                f_px = focal_px(frames[i], seed=int(i))
                if f_px is not None:
                    fs.append(f_px / math.hypot(*frames[i].shape[:2]))  # focal per unit diagonal
            digest = hashlib.sha256(f.read_bytes()).hexdigest()
            for i in picks:
                fr_list.append(frames[i])
                names.append((f, ts[i], digest, info))
                sharps.append(sharp[i])
                flags.append(False)
            if linking:
                step = max(1, round(vc["decode_fps"] / vc["link_fps"]))
                for i, kp, d in link_features(frames, sharp, range(0, len(frames), step), vc["link_side_px"],
                                              vc["min_sharpness"]):
                    lfeat[room].append((len(lcand[room]), kp, d))
                    lcand[room].append((frames[i], (f, ts[i], digest, info), sharp[i]))
        raw[room] = (fr_list, names, sharps, flags)
    links = _add_links(raw, lcand, lfeat, vc) if linking else []
    del lcand, lfeat
    # focal from vanishing points, pooled over all clips (same phone + mode)
    diag_mm = cfg["ingest"]["intrinsics"]["film_diagonal_mm"]
    nominal = vc["default_f35_mm"] / diag_mm
    plausible = [x for x in fs if 0.6 * nominal <= x <= 1.6 * nominal]
    if vc.get("force_f35_mm"):
        f_rel, source = vc["force_f35_mm"] / diag_mm, "default"
    elif len(plausible) >= vc["min_vp_frames"]:
        f_rel, source = float(np.median(plausible)), "vanishing_points"
    else:
        f_rel, source = nominal, "default"
        cap.warnings.append(f"video focal: only {len(plausible)} frames gave vanishing points; assuming "
                            f"{vc['default_f35_mm']} mm (35 mm eq.), wider intervals")
    f35 = f_rel * diag_mm
    for room, (fr_list, names, sharps, flags) in raw.items():
        frames_out = []
        order = sorted(range(len(names)), key=lambda k: (str(names[k][0]), names[k][1]))  # time order (links appended)
        for fr, (f, t, digest, info), sh, lk in ((fr_list[k], names[k], sharps[k], flags[k]) for k in order):
            H, W = fr.shape[:2]
            K = intrinsics(f35, W, H, diag_mm)
            meta = PhotoMeta(sha256=hashlib.sha256(f"{digest}@{t:.3f}".encode()).hexdigest(), full_size=(W, H),
                             orientation="portrait" if H > W else "landscape", device=info["model"], lens=None,
                             f35_mm=round(f35, 2), intrinsics_source=source, captured_at=None, sharpness=float(sh))
            frames_out.append(Frame(image_path=f.parent / f"{f.stem}_t{t:06.2f}", K=K, rgb=np.ascontiguousarray(fr),
                                    room_hint=room, timestamp=float(t), meta=meta, link=lk))
        if sum(not fr.link for fr in frames_out) < cfg["ingest"]["min_images_per_room"]:
            raise InputError(f"{vdir / room}: only {len(frames_out)} sharp keyframes; record slower / steadier.")
        cap.rooms[room] = frames_out
    cap.devices = sorted({i[3]["model"] for _, n, _, _ in raw.values() for i in n if i[3]["model"]})
    cap.warnings.append(f"video: {sum(len(v) for v in cap.rooms.values())} keyframes from {sum(len(c) for c in clips.values())} "
                        f"clips; focal {f35:.1f} mm (35 mm eq.) from {source} ({len(plausible)} frames)")
    if linking:
        cap.warnings.append("video links (both rooms see the same thing): " + (
            "; ".join(f"{a} {ta:.1f} s <-> {b} {tb:.1f} s ({n} matches)" for a, ta, b, tb, n in links) or "none found"))


def _add_links(raw: dict, lcand: dict, lfeat: dict, vc: dict) -> list[tuple]:
    """For each pair of rooms, the best `link_frames_per_pair` frame pairs that see the same thing join
    both rooms' keyframes, flagged as links: they tie the rooms together in a joint model run, but they
    look INTO the other room, so they are not used to measure or outline either room (E22l).
    Returns [(room a, time a, room b, time b, matches)]."""
    import itertools

    links = []
    for a, b in itertools.combinations(raw, 2):
        chosen: list[tuple[float, float]] = []
        for n, ia, ib in link_pairs(lfeat[a], lfeat[b], vc["link_min_inliers"]):
            ta, tb = lcand[a][ia][1][1], lcand[b][ib][1][1]
            if any(abs(ta - x) < vc["link_spread_s"] and abs(tb - y) < vc["link_spread_s"] for x, y in chosen):
                continue  # a neighbour of a link already taken
            chosen.append((ta, tb))
            links.append((a, round(float(ta), 2), b, round(float(tb), 2), n))
            for room, i in ((a, ia), (b, ib)):
                fr, name, sh = lcand[room][i]
                if all(n_[0] != name[0] or abs(n_[1] - name[1]) >= vc["min_keyframe_gap_s"] for n_ in raw[room][1]):
                    raw[room][0].append(fr)
                    raw[room][1].append(name)
                    raw[room][2].append(sh)
                    raw[room][3].append(True)
            if len(chosen) == vc["link_frames_per_pair"]:
                break
    return links
