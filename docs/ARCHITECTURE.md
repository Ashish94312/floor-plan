# Architecture — HLD and LLD

| | |
|---|---|
| **System** | `scan`: iPhone capture → measured whole-property floor plan + damage assessment |
| **Status** | v1 design. Phase 0 spike done: **O1–O3 decided** (backbone = MapAnything + EXIF intrinsics, D17; precision/memory D13; Open3D D14). Evidence in [WORKLOG.md](WORKLOG.md) E1–E10. |
| **Related** | [REQUIREMENTS.md](../REQUIREMENTS.md) · [PLAN.md](PLAN.md) · [DECISIONS.md](DECISIONS.md) · [CAPTURE_PROTOCOL.md](CAPTURE_PROTOCOL.md) |

---

## Contents

**Part A — High-Level Design (HLD)**
1. [Goals and constraints](#1-goals-and-constraints)
2. [Design principles](#2-design-principles)
3. [System context](#3-system-context)
4. [Component overview](#4-component-overview)
5. [Data flow per tier](#5-data-flow-per-tier)
6. [Technology stack](#6-technology-stack)
7. [Runtime and deployment](#7-runtime-and-deployment)

**Part B — Low-Level Design (LLD)**
8. [Repository layout](#8-repository-layout)
9. [Data model](#9-data-model)
10. [Stage-by-stage design](#10-stage-by-stage-design)
11. [Uncertainty model](#11-uncertainty-model)
12. [CLI, configuration, caching, determinism](#12-cli-configuration-caching-determinism)
13. [Evaluation harness](#13-evaluation-harness)
14. [Error handling](#14-error-handling)
15. [Performance budget](#15-performance-budget)
16. [Testing strategy](#16-testing-strategy)

**Part C — Audit**
17. [Requirement traceability](#17-requirement-traceability)
18. [Risks and known failure modes](#18-risks-and-known-failure-modes)
19. [Open design decisions](#19-open-design-decisions)

---

# Part A — High-Level Design

## 1. Goals and constraints

**Goal:** one command turns any of three capture tiers into the same output contract: per-room plans, a stitched plan, damage, concealed-damage flags, scope items, an interval on every number, JSON, and a rendered plan.

| Constraint | Effect on design |
|---|---|
| 48-hour build, single developer | Reuse pretrained models. Simple, explainable geometry. No custom training. |
| Dev hardware: MacBook M4, 16 GB RAM, about 43 GB free disk after weights | Models must fit in memory on Apple MPS or CPU. Required weights about 5.5 GB (MapAnything 4.9 GB + OWLv2 0.6 GB + DINOv2 code). |
| Dev phone: iPhone 13 (no LiDAR) | Photo and video developed on real data. LiDAR developed on synthetic data only. |
| Walk-in uses reviewers' iPhone 15+ | Read camera parameters from each file. Never hardcode a phone model. |
| Must run without our servers | Everything runs locally. Weights fetched by script. |
| Reproducible and repeatable | Deterministic seeds, sorted inputs, replayable cache. |
| Defended live with tools closed | Prefer simple, classical geometry over clever black boxes wherever accuracy allows. |

## 2. Design principles

1. **One core, thin adapters.** Each tier only answers *"where are the surfaces, where was the camera, and how sure am I?"* Everything after that is shared code.
2. **Every number is a measurement.** A measurement is always `value + lo + hi`, never a bare float. Uncertainty flows through the pipeline instead of being added at the end.
3. **Scale is a first-class random variable.** For photo and video, metric scale is the dominant error, so it is estimated explicitly as `(s, σ_log s)` and propagated to every length.
4. **Manhattan world by default.** Indoor walls are mostly at 90°. This assumption simplifies layout, stitching and drift correction. Non-Manhattan rooms are a known, documented failure mode.
5. **Models are swappable behind interfaces.** Geometry backends, the depth model and the detector each sit behind a small interface, so a swap is one file.
6. **Fail loud, degrade gracefully.** Bad input gives a clear error. A weak stage gives wider intervals plus a warning, never a silent wrong answer.
7. **Deterministic.** The same input gives the same JSON, byte for byte where possible.

## 3. System context

```
                 ┌───────────────────────────── CAPTURE ─────────────────────────────┐
  Operator ───►  │ iPhone + CAPTURE_PROTOCOL.md                                      │
                 │  Tier 1 Camera app → photos/<room>/*.heic|jpg  (2–8 per room)     │
                 │  Tier 2 Camera app → video/walkthrough.mov                        │
                 │  Tier 3 Stray Scanner (Pro) → lidar/<recording>/                  │
                 └──────────────────────────────┬────────────────────────────────────┘
                                                │ AirDrop → captures/<name>/
                                                ▼
                 ┌──────────────────────── scan (local, Mac/Linux) ──────────────────┐
  weights/ ────► │  ingest → geometry → align → segment → layout → openings          │
  (fetched by    │  → stitch (+drift correction) → damage → rules → intervals        │
   script)       │  → export JSON + render                                           │
                 └──────────────────────────────┬────────────────────────────────────┘
                                                ▼
                         captures/<name>/out/  result.json · plan.png · rooms/*.png
                                                │
  data/ground_truth/<property>.yaml ───────────►│ scan-eval
                                                ▼
                         reports/  gates · repeatability · calibration · head-to-head
```

## 4. Component overview

| # | Component | Responsibility | Photo | Video | LiDAR |
|---|---|---|:-:|:-:|:-:|
| C1 | **Ingest** | Detect the tier, validate input, load images or frames, read intrinsics | ✅ | ✅ | ✅ |
| C2 | **Geometry adapter** | Produce a metric point cloud, camera poses and the scale distribution | Multi-view model + metric depth | Same, chunked over keyframes | Back-project LiDAR depth with ARKit poses |
| C3 | **Alignment** | Gravity (up axis) and Manhattan axes | ✅ | ✅ | ✅ |
| C4 | **Room segmentation** | Split one global map into rooms | Not needed (one folder per room) | ✅ | ✅ |
| C5 | **Room layout** | Wall polygon, ceiling height, floor area | ✅ | ✅ | ✅ |
| C6 | **Openings** | Detect doors and windows, measure width and height, attach to a wall | ✅ | ✅ | ✅ |
| C7 | **Stitching + drift correction** | Place rooms in one frame, adjacency, no overlap, global optimisation | Door matching | Pose-graph loop closure | Pose-graph loop closure (ICP) |
| C8 | **Damage** | Detect damage, classify it, assign it to a surface, metric extent | ✅ | ✅ | ✅ |
| C9 | **Rules** | Concealed-damage flags (with rule ID) and scope line items | ✅ | ✅ | ✅ |
| C10 | **Uncertainty** | Intervals on every measurement, per-tier calibration | ✅ | ✅ | ✅ |
| C11 | **Export + render** | JSON (validated against our published schema), PNG/SVG plans | ✅ | ✅ | ✅ |
| C12 | **Eval** | Ground-truth matching, gates, reports | ✅ | ✅ | ✅ |
| C13 | **Synthetic generator** | Fake rooms in Stray Scanner format with exact ground truth | — | — | ✅ (tests) |

## 5. Data flow per tier

Only the left side differs per tier. From **Alignment** onward, the code is identical.

```
TIER 1 PHOTO
 photos/<room>/*   ─► C1 EXIF K per image ─► C2 per-room multi-view reconstruction
                                              + metric-depth scale (s, σ)
                                              (one reconstruction PER ROOM FOLDER)
                                                       │
TIER 2 VIDEO                                           │
 walkthrough.mov  ─► C1 keyframes (sharp, ~2 fps) ─► C2 chunked reconstruction
                                              + chunk alignment + scale (s, σ)
                                              (one GLOBAL map) ─► C4 segment rooms
                                                       │
TIER 3 LIDAR                                           │
 stray/<rec>/     ─► C1 depth + conf + odometry + K ─► C2 depth back-projection
                                              (one GLOBAL map, metric, σ small) ─► C4
                                                       │
                                                       ▼
            ┌──────────────── SHARED CORE (per room) ────────────────┐
            │ C3 align → C5 layout → C6 openings → C8 damage          │
            └───────────────────────────┬─────────────────────────────┘
                                        ▼
            C7 stitch:  photo = door matching + global least squares
                        video/LiDAR = poses already global; loop closure + Manhattan snap
                                        ▼
            C9 rules → C10 intervals → C11 JSON + render
```

## 6. Technology stack

| Concern | Choice | Licence | Why |
|---|---|---|---|
| Language / env | Python 3.11, `uv` + lockfile | — | Fast reproducible installs on a clean machine |
| Deep learning | PyTorch (MPS / CUDA / CPU) | BSD | Runs on Apple Silicon and Linux |
| Multi-view geometry | **MapAnything** (`facebook/map-anything-apache`), given the EXIF focal length as intrinsics. Its DINOv2 encoder code is vendored at a pinned commit (`weights/dinov2-code`) so loading is offline | Apache-2.0 | Feed-forward metric pose + depth from 2–N unposed images, and it accepts known intrinsics. Classical SfM fails on few, textureless indoor photos. VGGT-1B + DAv2 was rejected: scale off by +53 to +57% on the measured bedroom, and the licence is non-commercial (D15, D17, E7) |
| Metric scale | From MapAnything directly (metric output). Spread calibrated from benchmark residuals, plus a door-height anchor | — | Depth Anything V2 Metric was dropped as the scale source: biased 1.4–1.55× in this home (E4–E7) |
| Open-vocabulary detector | **OWLv2** (`google/owlv2-base-patch16-ensemble`) via `transformers` | Apache-2.0 | One model for doors, windows, mirrors and both damage classes. No training. |
| Geometry ops | NumPy, SciPy, Open3D (planes, ICP, pose graph) | MIT / BSD | Standard, well documented |
| 2D polygons | Shapely | BSD | Area, union, overlap checks |
| Image I/O | Pillow + pillow-heif, OpenCV, ffmpeg | Various | HEIC from iPhone, video decoding |
| Room segmentation | scikit-image | BSD | Connected components and watershed |
| Schema | Pydantic v2 → exported JSON Schema | MIT | One source of truth for the published schema |
| CLI | Typer | MIT | `scan`, `scan-eval`, `scan-bench` |
| Render | Matplotlib | PSF | PNG + SVG floor plans |

All third-party models and their licences are listed in the README (a constraint requires disclosure).

## 7. Runtime and deployment

- **Install:** `uv sync` then `uv run scan-fetch-weights` (downloads into `weights/`, checks SHA256).
- **Run:** `uv run scan captures/<name>`, auto-detecting the tier.
- **Device:** picks CUDA if available, then MPS, then CPU. CPU is supported but slower.
- **Offline:** after the weights are fetched, there are no network calls (`HF_HUB_OFFLINE=1` is set at runtime).
- **Cache:** model outputs are cached in `.cache/` keyed by input hash and model ID. `--no-cache` forces the live path.

---

# Part B — Low-Level Design

## 8. Repository layout

```
scan/
  cli.py                  # entry points: scan, scan-eval, scan-bench, scan-fetch-weights, scan-schema
  pipeline.py             # orchestrates stages, records timing, handles partial failure
  config.py               # loads config/*.yaml, merges CLI overrides
  schema.py               # Pydantic output models = the published JSON schema
  types.py                # internal dataclasses (Frame, RoomCloud, ...)
  device.py               # torch device selection, seeding
  cache.py                # content-addressed cache for model outputs
  io/
    ingest.py             # tier detection + validation
    photos.py             # HEIC/JPEG load, EXIF → intrinsics, orientation
    video.py              # ffmpeg decode, keyframe selection
    stray.py              # Stray Scanner reader
  geometry/
    backend.py            # GeometryBackend protocol
    mapanything_backend.py# backbone (D17): offline load, EXIF intrinsics, metric pose + depth
    scale.py              # scale uncertainty (s, σ_log s): calibration residuals + door-height anchor
    lidar.py              # depth back-projection + fusion
    chunks.py             # video chunk alignment (Sim3 / Umeyama)
    align.py              # gravity + Manhattan axes
    planes.py             # RANSAC plane fitting helpers
  layout/
    segment.py            # global map → rooms (video/LiDAR)
    room.py               # wall polygon, ceiling, floor area
    openings.py           # detection, 3D lifting, wall association, merging
  stitch/
    doors.py              # door-pair matching across rooms (photo tier)
    place.py              # placement + global least squares (drift correction)
    loop.py               # loop closure + pose graph (video/LiDAR)
  damage/
    detect.py             # OWLv2 → boxes → surface assignment → metric extent
    rules.py              # concealed-damage + scope rule engine
  uncertainty/
    intervals.py          # build Measurement objects from error sources
    calibrate.py          # fit per-tier inflation factor k
  render/
    plan.py               # per-room + stitched floor plans
  eval/
    gt.py                 # load ground-truth YAML
    match.py              # room/wall/opening/damage matching
    gates.py              # G1–G5, tier tolerances, calibration
    report.py             # markdown tables
  synth/
    rooms.py              # synthetic rooms → Stray Scanner format + GT YAML
config/
  default.yaml            # thresholds and parameters
  rules.yaml              # concealed-damage + scope rules
  calibration.yaml        # per-tier k (written by scan-bench)
schema/
  scan_output.schema.json # generated from scan/schema.py
scripts/
tests/
docs/
data/ground_truth/
captures/                 # gitignored
```

## 9. Data model

### 9.1 Internal types (`scan/types.py`)

```python
@dataclass
class Frame:
    image_path: Path
    K: np.ndarray                 # 3x3 intrinsics at the working resolution
    T_wc: np.ndarray | None       # 4x4 camera-to-world (None until geometry)
    depth: np.ndarray | None      # HxW metres (None for photo/video before geometry)
    depth_conf: np.ndarray | None # HxW in [0,1]
    room_hint: str | None         # photo tier: folder name; else None
    timestamp: float | None       # video/LiDAR

@dataclass
class Scale:
    s: float                      # multiply backbone units by s → metres
    sigma_log: float              # std of log(s); 0 for LiDAR (sensor noise handled separately)

@dataclass
class RoomCloud:
    room_id: str
    points: np.ndarray            # N x 3, metres, gravity-aligned (z up), Manhattan-aligned (x,y)
    colors: np.ndarray            # N x 3
    frames: list[Frame]
    scale: Scale
    T_room_world: np.ndarray      # 4x4 room-local → property frame (identity until stitched)
```

### 9.2 Output schema (`scan/schema.py`, the published schema)

```python
class Measurement(BaseModel):
    value: float
    lo: float
    hi: float
    unit: Literal["m", "m2", "count"]

class Wall(BaseModel):
    wall_id: str                  # "<room>-W<n>", W1 = wall with main door, then clockwise from above
    start: tuple[float, float]    # room-local metres
    end: tuple[float, float]
    length: Measurement

class Opening(BaseModel):
    opening_id: str               # "<room>-O<n>"
    type: Literal["door", "window", "opening"]
    wall_id: str
    offset_along_wall: Measurement
    width: Measurement
    height: Measurement
    connects_to: str | None       # room_id, filled by stitching
    views: int                    # number of frames supporting it

class Room(BaseModel):
    room_id: str
    label: str | None
    polygon: list[tuple[float, float]]
    walls: list[Wall]
    ceiling_height: Measurement
    floor_area: Measurement
    openings: list[Opening]
    status: Literal["ok", "partial", "failed"]
    warnings: list[str]

class DamageRegion(BaseModel):
    damage_id: str
    surface_id: str               # "<room>-W<n>" | "<room>-F" | "<room>-C"
    cls: Literal["water_stain", "crack"]
    width: Measurement
    height: Measurement
    area: Measurement
    position_on_surface: tuple[float, float]   # (from_left, from_floor) metres
    detection_score: float

class ConcealedFlag(BaseModel):
    flag_id: str
    surface_id: str
    rule_id: str
    rule_text: str
    evidence: list[str]           # damage_ids

class ScopeItem(BaseModel):
    item_id: str
    surface_id: str
    action: str
    quantity: Measurement
    rule_id: str
    source: list[str]

class StitchedPlan(BaseModel):
    room_poses: dict[str, tuple[float, float, float]]   # room_id → (x, y, theta_deg)
    adjacency: list[tuple[str, str, str]]               # (room_a, room_b, via_opening_id)
    footprint_area: Measurement                         # sum of room floor areas (net internal)
    overlaps: list[tuple[str, str, float]]              # (room_a, room_b, overlap_m2); must be empty
    drift_correction: bool
    drift_method: str

class ScanResult(BaseModel):
    schema_version: str
    capture_id: str
    tier: Literal["photo", "video", "lidar"]
    devices: list[str]            # from EXIF / metadata
    interval_level: float         # 0.90
    rooms: list[Room]
    stitched_plan: StitchedPlan
    damage: list[DamageRegion]
    concealed_damage_flags: list[ConcealedFlag]
    scope_items: list[ScopeItem]
    timing_s: dict[str, float]
    warnings: list[str]
    software: dict[str, str]      # git commit, model IDs, config hash
```

`uv run scan-schema` writes `schema/scan_output.schema.json`. That file **is** the published schema.

## 10. Stage-by-stage design

Each stage lists its input, output, algorithm, key parameters and failure behaviour.

### 10.1 C1 Ingest (`io/`)

| | |
|---|---|
| **Input** | `captures/<name>/` |
| **Tier detection** | `lidar/[<rec>/]odometry.csv`, or a bare Stray Scanner export (`odometry.csv` at the capture root, as in the assignment samples) → LiDAR. `video/*.mov|mp4` → video. `photos/<room>/` → photo. Override with `--tier`. If several are present, error unless `--tier` is given. |
| **Photo** | Load HEIC/JPEG with EXIF orientation applied. HEIC decode runs in threads (it dominates, about 280 ms per 12 MP photo). Intrinsics: `f_px = FocalLengthIn35mmFilm × image_diagonal_px / 43.27`. The 35 mm equivalent is defined on the film **diagonal**; a width-based formula would be 4% short on 4:3 (D18). No `FocalLengthIn35mmFilm` → 26 mm-equivalent default with a warning and wider σ. Focal outside 22–30 mm, or digital zoom → warning (not the 1× lens). Keep an in-memory copy at long side 1024 px with K scaled to match. |
| **Video** | `ffmpeg` decode at 4 fps. Score sharpness by Laplacian variance. Keep the sharpest frame in each 0.5 s window, giving about 2 fps of keyframes. Intrinsics from metadata if present, else the 26 mm-equivalent default plus a backbone estimate. |
| **LiDAR** | See 10.4. |
| **Validation** | Room folder with fewer than 2 valid images → error naming the folder. More than 8 → keep the 8 sharpest (Laplacian variance) and warn. Unreadable file → skip and warn. Non-image file → ignore and warn. Same photo twice in one room → drop and warn. **Same photo in two rooms → error** (room ambiguous, D19). Mixed portrait/landscape in a room → keep the majority, warn (D19). Room names other than `[a-z0-9_]` → warn. Video under 5 s → error. |

### 10.2 C2 Geometry — photo tier (`geometry/`)

1. **Backbone (MapAnything, D17):** run with **EXIF intrinsics** (focal from `FocalLengthIn35mmFilm` on the film diagonal, D18; principal point at the centre). Output per photo: metric z-depth `D_i`, confidence `C_i`, validity mask, camera-to-world pose `T_i`, the model's own intrinsics.
   - **Joint by default (`geometry.joint: auto`, D20):** all rooms in **one** run when the capture has ≤ 16 photos. Rooms then share one metric scale and are already placed relative to each other (E9, E15). Above 16 photos (memory: 14 photos = 8.1 GB), each room runs separately with a warning and wider intervals. `--joint` / `--per-room` force either.
   - **EXIF rays (`geometry.rays: exif`, D20):** points = `D_i` × EXIF ray, placed with `T_i`. The model's own focal is about 12% too long (E12, verified with vanishing points), which compressed ceilings 3–12%. Joint + EXIF: ceilings −0.1% / +0.6%, walls 5 of 6 within 1.5% (E15).
2. **Metric scale uncertainty:** `s = 1` (already metric). Per-room runs show per-room offsets of about ±5% (E7, E8, E14). A joint run shares one scale. `σ_log` comes from leave-one-room-out calibration residuals (C9), with floor `σ_floor_photo = 0.05`.
3. **Door-height anchor (optional):** if a door is detected with both its top and bottom visible, its height `h` gives `log s_door ~ N(log(2.03 / h), 0.05²)`. Fuse by inverse-variance weighting in log space.
4. **Point cloud** (`geometry/cloud.py`): world points rebuilt from depth with EXIF rays (`geometry.rays: exif`, default) or MapAnything's own points (`model`). Keep pixels in the model's validity mask with confidence above the room's 30th percentile. Merge views and voxel-downsample to 2 cm (Open3D, averaged points and colours). Outputs are cached by content (D21). Each room's per-view depth, pose and intrinsics stay attached for later stages (openings, damage).

### 10.3 C2 Geometry — video tier

1. Keyframes from 10.1 are split into **chunks of 24 frames with 4 frames of overlap** (fits in 16 GB; tuned in the spike).
2. Run the backbone per chunk. Align chunk *k* to chunk *k−1* with a Sim(3) Umeyama fit on the overlapping frames' camera centres and depth points.
3. Metric scale from the backbone, reconciled across chunks by the Sim(3) alignment. Uncertainty as in photo step 2, floor `σ_floor_video = 0.02`.
4. **Loop closure (C7, drift correction):** the protocol ends the walk at the start view. Match the last chunk against the first chunk (backbone run on 4 start + 4 end frames), giving a loop constraint. Optimise the chunk Sim(3) pose graph (Open3D) so the error is spread along the loop instead of piling up at the end.
5. Fuse into one global cloud, then pass to C3 and C4.

### 10.4 C2 Geometry — LiDAR tier (Stray Scanner)

Stray Scanner export format (verify against a real export when a Pro device is available):

| File | Content |
|---|---|
| `rgb.mp4` | Colour video (1920×1440) |
| `depth/NNNNNN.png` | uint16 depth in mm (256×192) |
| `confidence/NNNNNN.png` | uint8 confidence: 0 low, 1 medium, 2 high |
| `odometry.csv` | `timestamp, frame, x, y, z, qx, qy, qz, qw` (ARKit camera-to-world) |
| `camera_matrix.csv` | 3×3 K for the RGB resolution |

1. Take every 3rd frame. Back-project depth pixels with confidence 2, using K scaled to 256×192, through the ARKit pose.
2. **Drift correction:** ICP between start and end keyframe clouds gives a loop constraint. Optimise the pose graph over keyframe poses. Then Manhattan snap (10.5).
3. Scale `s = 1`, `σ_log = 0`. Sensor noise enters via σ_geom (Section 11).

### 10.5 C3 Alignment (`geometry/align.py`)

1. **Normals:** Open3D, 10 cm neighbourhood.
2. **Rough up:** minus the mean of the cameras' image-down axes (photo/video), or ARKit gravity (LiDAR).
3. **Floor and ceiling (consensus planes, D22):**
   - On points whose normal is within 25° of up, build a height histogram (1 cm bins).
   - **Floor** = the lowest peak holding ≥ 15% of the biggest. This skips the bed and table tops.
   - **Ceiling** = the highest such peak ≥ 1.8 m above the floor.
   - Each plane: direction from the ±3 cm core around the peak; height = median of the whole layer (floor −5..+12 cm, ceiling mirrored), so views that disagree by a few cm are averaged, not cherry-picked.
   - **Up** = floor normal, averaged with the ceiling normal if they agree within 3°. Flipped if the cameras would end up below the floor.
4. **Manhattan axes:**
   - For points with |n·up| < 0.2, take each normal's azimuth folded mod 90° and build a 0.5° histogram (circularly smoothed).
   - Refine the peak with a circular mean of 4φ over normals within ±5°.
   - Rotate so walls lie along x / y. Report the support: the share of wall normals within 5° of an axis.
5. **One transform per reconstruction frame.** A joint run's rooms share it, so their placement is kept. Points and camera poses move into the aligned frame (floor z = 0).
6. **Per room** (`room_planes`): floor and ceiling layers re-found in the aligned frame. Ceiling height = median ceiling z − median floor z. No ceiling found → `status = partial` + warning (ceiling not visible).

### 10.6 C4 Room segmentation — video/LiDAR (`layout/segment.py`)

1. Build a 2D occupancy grid (5 cm cells) on the floor plane:
   - **Wall cells:** points with z between 0.3 m and ceiling − 0.2 m.
   - **Free cells:** the floor points plus the camera trajectory.
2. **Cut at doors:** opening detections (C6, run globally) give door segments. Draw each as a wall line across the free space.
3. Connected components of free space give rooms. Components under 1.5 m² are merged into their neighbour.
4. **Fallback:** if no doors are detected, run a watershed on the distance transform of free space.
5. Each room's points are cropped by its region (dilated 30 cm) to form a `RoomCloud`.

### 10.7 C5 Room layout (`layout/room.py`, D23: free-space carving)

1. **Lines of sight:** per photo, up to 25k pixels (mask + confidence filtered), each a ray from its camera to a surface. Rays are drawn from above onto a 2 cm grid, stopping 6 cm short of the surface. Cell value = number of photos that saw through it.
2. **Room free space:**
   - 10 cm gap-fill (rays fan out near far walls).
   - **Ownership in a joint frame:** a cell belongs to the room whose photos saw through it most, so lines of sight through doorways don't leak into the neighbour.
   - Per-room frames: cut necks narrower than 70 cm.
   - Keep the component with the most floor points (plus cameras).
   - Rectangular open 0.6 m (removes doorway stubs) and close 1.0 m.
3. **Rectilinear outline:** contour → snap edges to x or y → merge collinear runs → drop jogs shorter than 15 cm → **fill inward corner notches < 0.6 m²** (both edges < 1 m: corner furniture or unseen corners), with a warning.
4. **Refine:** each wall's position = median of wall points (vertical surfaces from 0.3 m to 15 cm under the ceiling) found from 10 cm inside to 35 cm outside the outline edge. Carved space stops at or before a wall, so the true wall is outside. The median gives consensus over views that disagree by a few cm. Corners come from the refined lines. Per wall, the support (point count) and spread (robust std, which feeds σ_fit) are recorded.
5. **Walls:** polygon edges, clockwise seen from above. W1 is **provisional** (the south-most edge, west end first) until step 1.7 re-indexes it by the main door (D11). Eval takes the best cyclic shift meanwhile (disclosed).
6. **Ceiling height:** from C3 (median ceiling z − median floor z of the room).
7. **Floor area:** shoelace area of the polygon.
8. **Fallback:** too few wall points, no free region, or fewer than 4 edges → axis-aligned 2–98% bounding rectangle of the points above 1 m, `status = partial`, warning.

### 10.8 C6 Openings (`layout/openings.py`)

1. **Detect:** OWLv2 on each frame with the queries `door`, `doorway`, `open door`, `window`, `mirror`. Score threshold 0.15 (tuned on the benchmark).
2. **Lift to 3D:** sample depth along the box's left and right edges at mid-height and its top and bottom edges. Back-project, then project onto the nearest wall polygon edge within 30 cm.
3. **Measure:** width = extent along the wall. Height = top z − bottom z (bottom = 0 for doors). For windows, also record the sill.
4. **Geometric confirmation:** an open doorway has points *behind* the wall plane inside the box (you can see through it). A window has glass or few points.
5. **Merge across views:** cluster by (wall, offset along wall) within 25 cm and take the median width. `views` = cluster size.
6. **Phantom suppression:**
   - Drop any opening overlapping a `mirror` detection with IoU > 0.3.
   - Video/LiDAR: require at least 2 views.
   - Photo: accept 1 view, but its interval is wider.
7. **Width refinement:** in the depth image, find the jamb edges as the strongest depth discontinuities near the box's vertical edges (search ±8 px). Use them if found.

### 10.9 C7 Stitching and drift correction (`stitch/`)

**Photo tier (rooms are reconstructed separately):**

1. **Candidate door pairs:** every door in room A × every door in room B, for A ≠ B.
2. **Pair score:**
   - Width agreement: `exp(−(w_a − w_b)² / (2(σ_a² + σ_b²)))`
   - Visual evidence: SIFT + RANSAC inlier count between A's photo facing that door and all of B's photos, normalised. The protocol's doorway photo shows the next room, which makes this strong.
3. **Assignment:** greedy maximum-weight matching (each door used at most once), keeping edges that connect the graph. Edges below a threshold are left unmatched (exterior doors, closets).
4. **Placement:** BFS from the largest room. For matched doors (A, B):
   - **Rotation:** B's door-wall inward normal = − A's door-wall inward normal, a multiple of 90° under Manhattan.
   - **Translation:** door centres coincide, offset by wall thickness `t = 0.12 m` along the normal.
5. **Drift correction (global least squares):** unknowns are each room's (x, y). The rotations are fixed from step 4. Residuals: every matched door pair's centre offset, plus a soft penalty on overlap area. Solve with `scipy.optimize.least_squares`. This spreads chain error instead of letting it accumulate along the BFS order.
6. **Overlap check:** Shapely intersection of the room polygons. Any overlap above 0.05 m² → retry with the next-best door assignment. Still overlapping → report it in `overlaps` with a warning (gate G5 fails visibly, not silently).
7. **Ablation:** `--no-drift-correction` skips step 5 and uses BFS placement only.

**Video/LiDAR (poses are already global):** drift correction = loop closure (10.3 step 4 / 10.4 step 2) + Manhattan snap. Room placement comes from the global map. Adjacency = rooms sharing a door segment from C4. `--no-drift-correction` uses raw chained or ARKit poses.

**Footprint:** the sum of room floor areas (net internal area, matching what a tape measures), plus the outline polygon for rendering. See DECISIONS.

### 10.10 C8 Damage (`damage/detect.py`)

1. OWLv2 queries: `water stain on wall`, `brown stain`, `crack in wall`, `crack`. Map them to the classes `water_stain` and `crack`. Threshold tuned on the staged room.
2. **Surface assignment:** back-project the box centre and assign it to the nearest plane (wall edge, floor or ceiling).
3. **Metric extent:** project the box corners onto that surface plane to get width and height in metres. `area ≈ width × height × fill` (fill 0.6 for stains, about 0.1 for cracks).
4. Merge across views on the same surface (IoU on surface coordinates > 0.3).
5. **Known limitation:** box-based extent overestimates area. The interval reflects this, and SAM refinement is on the cut list.

### 10.11 C9 Rules (`damage/rules.py`, `config/rules.yaml`)

The rules are data, not code, so they can be read and defended.

```yaml
concealed_damage_rules:
  - id: STAIN_NEAR_CEILING
    when: {class: water_stain, top_within_m_of_ceiling: 0.30}
    text: "Water stain within 30 cm of ceiling: possible leak above ceiling or inside wall cavity"
  - id: STAIN_NEAR_FLOOR
    when: {class: water_stain, bottom_within_m_of_floor: 0.20}
    text: "Water stain within 20 cm of floor: possible plumbing leak or rising damp behind finish"
  - id: STAIN_ON_CEILING
    when: {class: water_stain, surface: ceiling}
    text: "Ceiling stain: possible leak from floor above or roof"
  - id: LONG_CRACK
    when: {class: crack, length_gt_m: 0.40}
    text: "Crack longer than 40 cm: possible structural movement behind finish"

scope_rules:
  - when: {class: water_stain}
    items:
      - {action: "Moisture-meter check behind surface", quantity: 1, unit: count}
      - {action: "Stain-block prime and repaint full surface", quantity: surface_area, unit: m2}
  - when: {class: crack}
    items:
      - {action: "Rake out, fill and sand crack", quantity: damage_length, unit: m}
      - {action: "Repaint full surface", quantity: surface_area, unit: m2}
```

`surface_area` = wall length × ceiling height − opening areas on that wall. It carries the combined interval.

### 10.12 C11 Render (`render/plan.py`)

- **Per room:** the polygon with wall lengths printed at mid-edge as `3.62 m ±0.22`, openings drawn as gaps with door swing arcs or window double lines, damage drawn as hatched marks on the walls, and the ceiling height and floor area in the room centre.
- **Stitched:** all rooms placed, room labels and areas, matched doors drawn as connections, any overlaps in red.
- Output: `out/plan.png` (stitched), `out/plan.svg`, `out/rooms/<room>.png`, `out/result.json` (validated against the schema on write). Diagnostics (`cloud.ply`, `align.png`, `layout.png`, `geometry.json`) go to `out/debug/` (D24).

## 11. Uncertainty model

For a length-like measurement `m` (wall length, opening width, ceiling height):

```
σ_m² = (m · σ_log s)²      # scale error: dominant for photo/video, 0 for LiDAR
     + σ_geom²             # local geometry noise per tier (LiDAR ≈ 1 cm, video ≈ 2 cm, photo ≈ 3 cm)
     + σ_fit²              # wall/opening fit residual from inlier spread (robust std / √n)
     + σ_extra²            # penalties: one view only, fallback layout, occlusion, missing EXIF

interval = m ± k_tier · z₀.₉₀ · σ_m        (z₀.₉₀ = 1.645)
```

- **Areas:** propagate with `σ_A/A ≈ 2 σ_log s` for scale, plus the edge-fit terms.
- **Calibration** (`uncertainty/calibrate.py`): on the benchmark, find the smallest `k_tier` so that empirical coverage ≥ 90%. Fit with **leave-one-room-out** so no room grades its own interval. Store the result in `config/calibration.yaml`. Before the benchmark, `k_tier = 1.5` (conservative).
- **Reported:** coverage per tier and per measurement type, plus mean interval width. Wide-and-honest beats narrow-and-wrong ("confident garbage" caps the score).

## 12. CLI, configuration, caching, determinism

```
uv run scan <capture_dir> [--tier photo|video|lidar] [--out DIR]
                          [--no-drift-correction] [--no-cache] [--device cpu|mps|cuda]
uv run scan-eval <out_dir> --gt data/ground_truth/<property>.yaml
uv run scan-bench [--suite bench.yaml]       # runs every capture + eval → reports/
uv run scan-fetch-weights                    # downloads + verifies weights
uv run scan-schema                           # writes schema/scan_output.schema.json
```

- **Config:** `config/default.yaml` holds every threshold quoted in Section 10. CLI flags override it. The resolved config hash goes into `result.json → software`.
- **Cache:** key = SHA256(input file bytes + model ID + model params + working resolution). Value = `.npz` with the model outputs. Replaying the cache gives identical numbers, and `--no-cache` must reproduce them within float tolerance (tested).
- **Determinism:**
  - `torch.manual_seed(0)`, `np.random.default_rng(0)` passed explicitly to every RANSAC
  - Inputs sorted by filename
  - `torch.use_deterministic_algorithms(True, warn_only=True)`
  - No time-dependent logic

## 13. Evaluation harness

**Ground truth:** [data/ground_truth/TEMPLATE.yaml](../data/ground_truth/TEMPLATE.yaml).

**Matching** (`eval/match.py`):
- **Rooms:**
  - Photo tier: by folder name = GT room ID.
  - Video/LiDAR: Hungarian assignment on `|area_pred − area_gt| + λ·|perimeter_pred − perimeter_gt|`.
- **Walls:** both use the same rule (W1 = main-door wall, clockwise). If the predicted main door is ambiguous, take the cyclic shift that minimises total error and **report this choice** (slightly optimistic, disclosed).
- **Openings:** Hungarian on (wall, offset). Unmatched GT = missed. Unmatched prediction = phantom.

**Gates** (`eval/gates.py`). Formulas follow REQUIREMENTS §8.4:

| Gate | Computation |
|---|---|
| G1 | `correct / (n_gt + n_phantom)` where correct = matched with \|Δw\| ≤ 2 cm. Pass ≥ 0.85. |
| G2 | Per room: \|mean(pred) − gt\| ≤ 1.5 cm. Repeated rooms: max − min ≤ 1 cm. Label each failure as `biased` or `unrepeatable`. |
| G3 | Per wall pair across two captures: \|L1 − L2\| ≤ max(1 cm, 0.5% · L_gt). |
| G4 | Footprint error and stitched plan with drift correction on vs off (two runs, side-by-side image + table). |
| G5 | Photo tier: adjacency graph == GT graph, overlaps empty, \|footprint error\| ≤ 8%. |
| Tier | Wall length relative error: photo ≤ 8%, video ≤ 3%. |
| Calibration | Coverage per tier and measurement type. |

**Reports:** `reports/benchmark.md` (all tables), `reports/repeatability.md`, `reports/head_to_head.md`, `reports/timing.md`.

## 14. Error handling

| Situation | Behaviour | Exit code |
|---|---|---|
| Invalid input layout or unreadable capture | Clear message naming the file or folder, no output | 2 |
| Weights missing | "Run `uv run scan-fetch-weights`" | 3 |
| One room fails layout | Room present with `status: failed`, a reason, null measurements. The rest continues. | 0 (warning) |
| Ceiling not visible | `status: partial`, ceiling from prior (2.5 m ± 0.3) with a very wide interval | 0 |
| Stitching cannot place a room | Room listed unplaced, with a warning. Footprint interval widened. | 0 |
| Out of memory on the backbone | Retry automatically with a smaller chunk or resolution, logged | 0 |
| Unexpected exception | Traceback to `out/error.log`, partial `result.json` with `status: failed` | 1 |

## 15. Performance budget

Targets on an M4 with MPS (to be measured in the spike):

| Stage | Photo (5 rooms × 6 imgs) | Video (5 min) | LiDAR (5 min) |
|---|---|---|---|
| Ingest | < 5 s | < 30 s | < 20 s |
| Backbone | < 2 min | < 5 min | — |
| Metric depth | < 30 s | < 1 min | — |
| Detector (openings + damage) | < 1 min | < 2 min | < 2 min |
| Layout + stitch + rules + render | < 15 s | < 30 s | < 30 s |
| **Total** | **< 4 min** | **< 9 min** | **< 3 min** |

Memory ceiling: 12 GB peak (leaves headroom on 16 GB).

## 16. Testing strategy

| Level | What | Where |
|---|---|---|
| Unit | Plane fit, Manhattan angle, rectilinear polygon, shoelace area, Sim(3) Umeyama, interval maths, rule engine | `tests/unit/` |
| Synthetic end-to-end | `synth/rooms.py` produces 3 rooms + a hall with exact GT, LiDAR noise and injected drift. The LiDAR tier must pass G3 and G4, and drift on must beat drift off. | `tests/synth/` |
| Real golden | One real photo capture. The JSON must validate against the schema and match the stored golden within tolerance. | `tests/golden/` |
| Determinism | Run the same capture twice → identical JSON. Cache vs live → equal within 1e-6. | `tests/test_determinism.py` |
| Fresh machine | Clone into a new folder → `uv sync` → fetch weights → run, timed (< 15 min) | Manual, documented in README |

---

# Part C — Audit

## 17. Requirement traceability

This table seeds the compliance matrix (deliverable 1).

| Requirement | Component / file | Artifact | Planned status |
|---|---|---|---|
| Three tiers, same contract | `io/ingest.py`, `pipeline.py`, `schema.py` | `out/result.json` per tier | Photo ✅ · Video ✅ · LiDAR ⚠️ synthetic only |
| Per-room plan (walls, ceiling, area, openings) | `layout/room.py`, `layout/openings.py` | `rooms` in JSON, `rooms/*.png` | ✅ |
| Stitched plan, adjacency, no overlap | `stitch/*` | `stitched_plan`, `plan.png` | ✅ |
| Photo-tier whole-property stitch (G5) | `stitch/doors.py`, `stitch/place.py` | `reports/benchmark.md#g5` | ✅ |
| Drift handling + ablation (G4) | `stitch/place.py`, `stitch/loop.py`, `--no-drift-correction` | `reports/benchmark.md#g4` | ✅ |
| Damage: class + metric extent | `damage/detect.py` | `damage` in JSON | ✅ (box-level extent) |
| Concealed flags + rule fired | `damage/rules.py`, `config/rules.yaml` | `concealed_damage_flags` | ✅ |
| Scope items keyed to surfaces | `damage/rules.py` | `scope_items` | ✅ |
| Interval on every measurement | `uncertainty/intervals.py`, `schema.Measurement` | Every number in JSON | ✅ |
| Calibration scored per tier | `uncertainty/calibrate.py`, `eval/gates.py` | `reports/benchmark.md#calibration` | ✅ |
| One command per capture | `cli.py` | README | ✅ |
| Published JSON schema | `schema.py` → `schema/scan_output.schema.json` | schema file | ✅ |
| Rendered plan | `render/plan.py` | `plan.png`, `plan.svg` | ✅ |
| Opening gate (G1) | `layout/openings.py`, `eval/gates.py` | report | Measured (likely fail at photo tier) |
| Ceiling gate (G2) + bias/variance label | `layout/room.py`, `eval/gates.py` | report | Measured |
| Repeatability (G3) | determinism + `eval/gates.py` | `reports/repeatability.md` | Measured |
| Photo ±8%, video ±3% | `eval/gates.py` | report | Measured |
| Head-to-head | `eval/report.py` + magicplan export | `reports/head_to_head.md` | ⚠️ vs video tier (no LiDAR device) |
| Device matrix | `docs/DEVICE_MATRIX.md` | doc | ✅ (iPhone 13 deviation disclosed) |
| Capture route | `docs/CAPTURE_PROTOCOL.md` | doc | ✅ (LiDAR section untested on a device) |
| Weights by script | `scan-fetch-weights` | `weights/` | ✅ |
| Runs without own infra | offline flag | README | ✅ |
| Reproduction bundle + deterministic cache | `cache.py`, `scan-bench` | bundle | ✅ |
| Mirrors, glass, wet-look, low light | `openings.py` mirror suppression, confidence filtering, protocol | tech report section | ⚠️ partial handling, documented |
| Fix loop | `docs/FIX_DECLARATION.md`, git tags `fix-before` / `fix-after` | `reports/fix_loop.md` | Planned (Phase 5) |
| Technical report ≤ 6 pages | `docs/TECHNICAL_REPORT.md` | PDF | Planned (Phase 6) |

## 18. Risks and known failure modes

| # | Risk / failure mode | Likelihood | Impact | Mitigation |
|---|---|---|---|---|
| R1 | Backbone too slow or too big for 16 GB on MPS | ~~Medium~~ Resolved | High | MapAnything: 14 photos in 60 s at 8.1 GB peak (E9). Load takes about 40 s |
| R2 | Photo-tier scale error > 8% | High | High | Metric-depth median plus door prior. σ_log sized honestly so intervals cover. |
| R3 | Photo stitching mis-pairs doors | Medium | High | Doorway photo in the protocol. Width + visual score. Overlap check with retry. |
| R4 | Openings: 2 cm gate unreachable from photos | High | Medium | Jamb-edge refinement. Report honestly. Likely fix-loop target. |
| R5 | Mirrors create phantom openings or rooms | Medium | Medium | Mirror detection suppression. Protocol says avoid mirrors. |
| R6 | Glass windows have no depth | Medium | Low | Windows detected from RGB. Wall plane fitted from surrounding points. |
| R7 | Non-Manhattan rooms (angled or curved walls) | Low (dev home) | Medium | Documented failure mode. Fallback rectangle + wide interval. |
| R8 | Video room segmentation fails without detected doors | Medium | High | Watershed fallback. Trajectory-based split. |
| R9 | LiDAR path untested on real data | Certain | Medium | Synthetic tests. Disclosed. Stray format verified from documentation only. |
| R10 | Disk full | ~~High~~ Low | High | 43 GB free after weights |
| R11 | Run out of time before the fix loop | Medium | Very high | Feature freeze at H31 ([PLAN.md](PLAN.md)). |
| R12 | Developer can't defend the code live | Medium | High | DECISIONS.md. Simple algorithms. Review each module on completion. |

## 19. Open design decisions

| # | Decision | Options | Decided by |
|---|---|---|---|
| O1 | Geometry backbone | **Decided: MapAnything + EXIF intrinsics** | D17 (E4–E7) |
| O2 | Working resolution and video chunk size | 518 px long side. Chunk size to re-measure for MapAnything (14 frames = 8.1 GB) | D13, E9. Finalised in Tier 2 |
| O3 | Open3D vs pure NumPy/SciPy | **Decided: Open3D** | D14 |
| O4 | Detector thresholds | 0.15 (start) | Tuned on the benchmark (disclosed) |
| O5 | Wall thickness for stitching | 0.12 m fixed / estimated | Fixed for v1. Fix-loop candidate. |
