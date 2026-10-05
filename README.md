# scan: iPhone capture → measured floor plan

This tool turns an iPhone capture of a home into a measured floor plan. It takes photos, a walkthrough video or a LiDAR scan, and produces:

- a per-room plan: walls, ceiling height, floor area, doors and windows
- one stitched plan of the whole property
- damage regions, concealed-damage flags and scope line items

Every number comes with a 90% interval. Output is JSON (with a published schema) plus a rendered plan.

**One command per capture. Runs locally, with no calls to our own servers.**

Contents: [Quick start](#quick-start) · [Inputs](#inputs-what-a-capture-folder-looks-like) · [Run every capture](#run-every-capture-in-this-repo) · [How each tier works](#how-each-tier-builds-the-plan) · [Testing](#testing) · [Where each tier stands](#where-each-tier-stands) · [Time and memory](#time-and-memory) · [Data](#data) · [Models](#models-and-data-used) · [Repository](#repository) · [Troubleshooting](#troubleshooting)

## Quick start

Target time: under 15 minutes on a clean machine.

**Needs:**
- macOS on Apple Silicon (tested: M4, 16 GB) or Linux with CUDA
- `ffmpeg` / `ffprobe` on the PATH (video tier)
- about 6 GB of disk for the model weights
- 16 GB of memory (a 20-photo joint run peaks at about 12 GB)

```bash
git clone <this repo> scan && cd scan
curl -LsSf https://astral.sh/uv/install.sh | sh   # only if `uv` is missing
uv sync                                           # Python 3.11 env, pinned by uv.lock (dev group: pytest, ruff, pycolmap)
uv run scan-fetch-weights                         # ~5.2 GB, SHA256-checked (models below)
uv run scan captures/<capture>                    # one command per capture
```

Results go to `<capture>/out/` (or `--out <dir>`):

| File | What it is |
|---|---|
| `result.json` | The full output contract. Schema: [schema/scan_output.schema.json](schema/scan_output.schema.json) (`uv run scan-schema` regenerates it) |
| `plan.png`, `plan.svg` | The stitched plan, with dimensions and intervals |
| `rooms/*.png` | One plan per room |
| `debug/` | Diagnostics: point cloud per room (`cloud.ply`), alignment and layout plots, `geometry.json`. Not part of the contract. |

After `scan-fetch-weights` the pipeline never touches the network, so it runs offline.

### `scan` options

| Option | Effect |
|---|---|
| `--tier photo\|video\|lidar` | Pick the tier when a folder holds more than one (normally detected from the layout) |
| `--out <dir>` | Output folder (default `<capture>/out`) |
| `--joint` / `--per-room` | Photo tier: one MapAnything run over all rooms (shared scale; ≤ 20 photos fit in 16 GB), or one run per room. Default: joint when it fits |
| `--no-drift-correction` | Ablation: stitch with poses as they are (no wall snapping) |
| `--no-cache` | Ignore the model-output cache in `.cache/` and run the models live |
| `--device cpu\|mps\|cuda` | Force a device (default: best available) |

## Inputs: what a capture folder looks like

How to film each tier is a one-page protocol: [docs/CAPTURE_PROTOCOL.md](docs/CAPTURE_PROTOCOL.md) (v2: one video clip per room; LiDAR as one walk, tilting up at every doorway). The tier is detected from the folder layout:

| Tier | Layout | Notes |
|---|---|---|
| Photo | `<capture>/photos/<room>/*.HEIC` | One folder per room, 2–8 photos (more: the 8 sharpest are kept), one lens, landscape. Intrinsics from EXIF |
| Video | `<capture>/video/<room>/*.MOV` | One clip per room. Keyframes are picked per clip; the focal is measured from vanishing points (video files carry none) |
| LiDAR | a Stray Scanner export: `odometry.csv`, `depth/`, `confidence/`, `rgb.mp4`, `camera_matrix.csv` | One walk through the whole home (rooms are split automatically): the bare export as the capture folder, or `<capture>/lidar/<export>`. Or one recording per room: `<capture>/lidar/<room>/<export>` |

- Room folder names: lowercase letters, digits and `_` (for example `bedroom1`, `kitchen`).
- Optional `<capture>/hints.yaml`: layout facts the person filming knows, not measurements. Today one key: `same_line: [[kitchen, hall]]` (these rooms' outer walls continue each other). Example: `captures/home01_video_e/hints.yaml` (in the data bundle).
- If the input is unusable, the run stops with exit code 2 and names the folder or file that is the problem.

## Run every capture in this repo

The captures are too large for git. They ship as a separate data bundle, unpacked into `captures/`; the three assignment LiDAR samples sit at the repo root (see [Data](#data)). Run captures **one at a time** (each run loads its models; two at once can exhaust 16 GB).

```bash
# Photo tier (MapAnything): the same three rooms of home01, taped
uv run scan captures/home01_photo_a                 # portrait photos
uv run scan captures/home01_photo_b                 # landscape photos, as the protocol says

# Video tier (MapAnything on keyframes): the same three rooms
uv run scan captures/home01_video_d                 # portrait clips
uv run scan captures/home01_video_e                 # landscape clips, as the protocol says (uses its hints.yaml)

# LiDAR tier (phone depth + ARKit poses; no learned geometry): assignment samples, one walk each
uv run scan 1a8384c3f6 --out captures/lidar_samples/1a8384c3f6/out
uv run scan c00a170fe1 --out captures/lidar_samples/c00a170fe1/out
uv run scan c7d28f72c6 --out captures/lidar_samples/c7d28f72c6/out
```

The LiDAR samples are run with `--out` so their results land under `captures/` with everything else, instead of inside the sample folders.

**Every capture folder**, with its command and where the result goes. Every runnable one was checked to ingest (2026-10-05):

| Capture | Tier, rooms | Command | Final output | Notes |
|---|---|---|---|---|
| `home01_photo_a` | photo, bedroom1 + hall + kitchen | `uv run scan captures/home01_photo_a` | `out/` | portrait photos; benchmark |
| `home01_photo_b` | photo, same 3 rooms | `uv run scan captures/home01_photo_b` | `out/` | landscape (protocol); benchmark |
| `home01_video_d` | video, same 3 rooms | `uv run scan captures/home01_video_d` | `out/` | portrait clips; benchmark |
| `home01_video_e` | video, same 3 rooms | `uv run scan captures/home01_video_e` | `out/` | landscape clips (protocol), `hints.yaml`; **the final video result**; benchmark |
| `1a8384c3f6` (repo root) | LiDAR, one walk | `uv run scan 1a8384c3f6 --out captures/lidar_samples/1a8384c3f6/out` | `captures/lidar_samples/1a8384c3f6/out/` | assignment sample; split into 2 rooms |
| `c00a170fe1` (repo root) | LiDAR, one walk | `uv run scan c00a170fe1 --out captures/lidar_samples/c00a170fe1/out` | `captures/lidar_samples/c00a170fe1/out/` | assignment sample; 3 rooms |
| `c7d28f72c6` (repo root) | LiDAR, one walk | `uv run scan c7d28f72c6 --out captures/lidar_samples/c7d28f72c6/out` | `captures/lidar_samples/c7d28f72c6/out/` | assignment sample; 6 rooms |
| `lidar_samples/<rec>/photo_from_lidar` | photo, made from a LiDAR walk (E26) | `uv run scan captures/lidar_samples/<rec>/photo_from_lidar --per-room` | its `out/` | MapAnything test against LiDAR; `photo_from_lidar_joint` with `--joint` (E26b) |
| `home01_video_a`, `home01_video_b` | video, 3 rooms | `uv run scan captures/home01_video_a` (same for `_b`) | `out/` | earlier clips (portrait, older protocol); experiments only |
| `home01_video_c` | video, bedroom1 | `uv run scan captures/home01_video_c` | `out/` | earlier clip; experiments only |
| `home01_video_b_hall`, `home01_video_d_hall` | video, hall | `uv run scan captures/home01_video_d_hall` (same for `_b_hall`) | `out/` | one-room subsets for experiments |
| `home01_video_d_bh` | video, bedroom1 + hall | `uv run scan captures/home01_video_d_bh` | `out/` | two-room subset for experiments |
| `vggt_examples` | photo, 2 rooms (no EXIF focal) | `uv run scan captures/vggt_examples` | `out/` | Phase 0 spike images; default focal with a warning |
| `home01_photo_portrait` | — | not runnable | — | its photos were lost in a data incident (WORKLOG 13:38, 2026-10-04); thumbnails only |
| `magicplan` | — | not a capture | — | the consumer app's exports, for the head-to-head |

The video captures take 10–50 s to ingest (keyframes + vanishing points) before the model runs.

## How each tier builds the plan

| Step | Photo | Video | LiDAR |
|---|---|---|---|
| Ingest | HEIC/JPEG, EXIF-upright, EXIF focal | Sharp keyframes per clip, focal from vanishing points, 16:9 frames padded to 4:3 | Up to 120 frames per recording; depth (high confidence, ≤ 5 m), ARKit pose and intrinsics per frame |
| 3D geometry | MapAnything: poses + metric depth | MapAnything | **The phone's LiDAR depth through the ARKit poses** (no model) |
| Scale | Focal vote + calibration from taped rooms | Door-height anchor (doors 2.05 m) | Metric already (sensor) |
| Floor / up | Floor and ceiling planes, Manhattan walls | same | same, with gravity from ARKit |
| Rooms | One folder per room | One clip per room | Bare walk: split at doorways (necks ≤ 1.0 m, wall seen above door height, detector doors); corridors the walk crossed become `passageN` spaces |
| Room outline | Free-space carving from lines of sight → square-cornered polygon, walls on the measured wall points | same | same, each room from its own segment; overlaps between rooms resolved |
| Stitching | Joint run shares one frame; otherwise door matching | Link frames, then doors; `hints.yaml` | One ARKit frame per recording |
| Openings | Depth seen through a wall + OWLv2 detector | same | same; frames detected upright |
| Damage / scope | OWLv2 boxes lifted onto surfaces; rules in `config/rules.yaml` | same | same |

Every threshold lives in [config/default.yaml](config/default.yaml), each with the experiment it came from. Design: [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md); decisions: [docs/DECISIONS.md](docs/DECISIONS.md).

## Testing

### 1. Unit tests and lint (no weights or captures needed)

```bash
uv run pytest -q                 # ~120 tests, ~70 s
uv run ruff check scan tests     # lint
```

| Test file(s) | Covers |
|---|---|
| `test_ingest.py`, `test_video.py`, `test_lidar.py` | Tier detection, photo EXIF, video keyframes, the Stray Scanner reader (synthetic box rooms from `scan/synth/stray.py`), upright detection |
| `test_geometry.py`, `test_align.py`, `test_scale_vote.py` | Back-projection, focal effects, floor/Manhattan alignment, the photo scale vote |
| `test_layout.py`, `test_layout_cameras.py`, `test_segment.py` | Free-space carving, room outlines, splitting one LiDAR walk into rooms (doors, lintels, detector doorways) |
| `test_openings.py`, `test_damage.py` | Doors/windows from depth votes, detector openings, damage regions |
| `test_stitch_doors.py`, `test_stitch_links.py`, `test_snap.py`, `test_joint_scale.py`, `test_door_scale.py` | Placing rooms, wall snapping, declared wall lines, shared scale |
| `test_output.py`, `test_eval.py`, `test_calibrate.py` | Output contract, scoring against the tape, interval calibration |
| `tests/test_determinism.py` (+ one test in `test_geometry.py`) | A cached real run replays identically. **Skipped** unless `captures/home01_photo_a` and its cached model outputs are present |

### 2. Accuracy against the tape (photo and video)

```bash
uv run scan-eval captures/home01_photo_a/out --gt data/ground_truth/home01.yaml   # one capture -> eval.json + table
uv run scan-bench                 # runs every capture in config/bench.yaml, then scores them
uv run scan-bench --skip-run      # scores the existing out/ folders without re-running
```

`scan-bench` writes [reports/benchmark.md](reports/benchmark.md): the gates per capture and tier, repeatability (same rooms filmed twice), the drift-correction ablation, the head-to-head against magicplan and timing. The laser ground truth is [data/ground_truth/home01.yaml](data/ground_truth/home01.yaml); the magicplan numbers are [data/head_to_head/magicplan.yaml](data/head_to_head/magicplan.yaml).

### 3. Interval calibration

```bash
uv run scan-calibrate captures/*/out/eval.json      # fits interval widths -> config/calibration.yaml
scripts/calibrate_photo_vote.sh                      # photo size level + widths from scratch (runs the captures)
```

### 4. Determinism

```bash
uv run python scripts/check_determinism.py captures/home01_photo_a <report_dir>   # cached twice + live vs cached
```

### 5. LiDAR samples (no tape: LiDAR is the reference)

The assignment's LiDAR samples have no tape measurements. They are checked two ways:

```bash
# a) look at the room split (free space, room cores, rooms, upright frames per room)
uv run python scripts/experiments/e25_segment.py c7d28f72c6 captures/lidar_samples/c7d28f72c6/experiments/segment.png

# b) MapAnything vs LiDAR on the same walk: the walk as a photo capture (8 frames per room, ARKit focal as EXIF),
#    the photo tier on it, then per-room errors against the LiDAR plan
uv run python scripts/experiments/e26_lidar_to_photos.py c7d28f72c6 captures/lidar_samples/c7d28f72c6/photo_from_lidar
uv run scan captures/lidar_samples/c7d28f72c6/photo_from_lidar --per-room
uv run python scripts/experiments/e26_compare.py c00a170fe1 c7d28f72c6 1a8384c3f6
# joint variant: e26_lidar_to_photos.py <rec> .../photo_from_lidar_joint <n per room, total <= 20>, then
# `uv run scan .../photo_from_lidar_joint --joint` and e26_compare.py --ma=photo_from_lidar_joint ...
```

Run the LiDAR plans (above) first: the comparison reads `captures/lidar_samples/<rec>/out/result.json`.

```bash
# c) is a plan clean? overlaps between rooms, separate pieces, each room's gap to its nearest neighbour
uv run python scripts/experiments/e27_plan_check.py captures/lidar_samples/*/out/result.json
```

### 6. Experiments and scripts

Every experiment (E1–E27) is logged in [docs/WORKLOG.md](docs/WORKLOG.md) with its question, setup, command, result and impact. Its outputs are in `captures/<capture>/experiments/`.

**[scripts/README.md](scripts/README.md) lists every script** (tools and the experiment scripts) with its worklog entry, what it does and the exact command. It is generated from the scripts' own headers: `uv run python scripts/make_scripts_index.py`.

## Where each tier stands

| Tier | Status |
|---|---|
| Photo | Works end to end, including the stitched multi-room plan. home01_photo_a: 12/14 walls within ±8%, mean error 2.9%, footprint −2.7%. Details in [reports/benchmark.md](reports/benchmark.md). |
| Video | Runs end to end. Room sizes are less reliable (about ±10%; home01_video_e footprint −4.3%). Intervals are calibrated against the tape. |
| LiDAR | Runs end to end on the three assignment samples (E25, E27). Geometry is the phone's own (floor fit 1.3–1.7 cm). Rooms come from splitting the walk at doorways, corridors become passages; every plan is one connected piece with no overlaps: `c7d28f72c6` 6 rooms + 2 passages (ceilings filmed), `c00a170fe1` 3 rooms, `1a8384c3f6` only 2 (its walk never filmed above 2.3 m, so few doorways show). No taped LiDAR capture yet, so its intervals are not calibrated. |

MapAnything on the same LiDAR walks is far from the LiDAR plans: mean room-area error 80% with one run per room (E26) and 176% with one joint run over all rooms (E26b). Walk-through frames are close to walls and narrow, which it misplaces. The photo tier is meant for the protocol's corner photos.

## Time and memory

Measured on an M4 with 16 GB, one capture at a time:

| Run | Wall time | Peak memory |
|---|---|---|
| LiDAR, one walk (cached detector) | 20–75 s | 1.8–2.6 GB |
| LiDAR, first run (detector live) | 2–3 min | ~2.5 GB |
| Photo, one MapAnything run per room (8 photos per room) | 1.5–3.5 min | 5.8–8.6 GB |
| Photo, joint run (≤ 20 photos) | ~1.5 min inference + 30 s model load | ~9 GB accelerator (MPS), 6–8 GB process |

Model outputs are cached in `.cache/`, keyed by the inputs and settings. A cached re-run skips the models and gives byte-identical plans.

## Data

| Folder | What it is | In the benchmark |
|---|---|---|
| `captures/home01_photo_a` | Photos, portrait, bedroom1 + hall + kitchen | yes |
| `captures/home01_photo_b` | Photos, landscape (protocol), same rooms | yes |
| `captures/home01_video_d` | Video, portrait clips, same rooms | yes |
| `captures/home01_video_e` | Video, landscape clips (protocol), same rooms; `hints.yaml` | yes (final video result) |
| `captures/magicplan` | The consumer app's exports of home01 (head-to-head) | as the comparison |
| `1a8384c3f6`, `c00a170fe1`, `c7d28f72c6` (repo root) | Assignment LiDAR samples: Stray Scanner walks through multi-room homes | no (no tape); plans in `captures/lidar_samples/` |
| `captures/home01_video_a`, `_b`, `_c`, `_b_hall`, `_d_bh`, `_d_hall`, `home01_photo_portrait`, `vggt_examples` | Earlier captures and the Phase 0 spike | no (experiments only) |

`captures/`, the LiDAR samples, `weights/` and `.cache/` are gitignored.

## Models and data used

All models are fetched by `scan-fetch-weights`, pinned to a revision and checked by SHA256.

| Model | Use | Licence |
|---|---|---|
| `facebook/map-anything-apache` (code `facebookresearch/map-anything@3d10cf7`) | Camera poses and metric depth from photos and video frames | Apache-2.0 |
| DINOv2 code (`facebookresearch/dinov2@7764ea0`) | Backbone code needed by MapAnything | Apache-2.0 |
| `google/owlv2-base-patch16-ensemble` | Doors, windows, mirrors and damage, from text queries | Apache-2.0 |

The LiDAR tier uses no geometry model; OWLv2 is its only model. VGGT-1B and Depth Anything V2 were tried in the first spike and dropped. They are optional (`--optional`) and the pipeline doesn't use them.

## Repository

| Path | Contents |
|---|---|
| `scan/` | The pipeline: `io/` ingest (photos, video, LiDAR), `geometry/`, `layout/` (rooms, openings, LiDAR room split), `stitch/`, `damage/`, `uncertainty/`, `render/`, `eval/`, `synth/` (synthetic LiDAR rooms for tests) |
| `config/` | `default.yaml` (all settings), `calibration.yaml` (fitted by `scan-calibrate`), `rules.yaml` (concealed-damage rules), `bench.yaml` |
| `data/` | Laser ground truth, and the magicplan numbers for the head-to-head |
| `docs/` | [REQUIREMENTS](REQUIREMENTS.md), [ARCHITECTURE](docs/ARCHITECTURE.md), [CAPTURE_PROTOCOL](docs/CAPTURE_PROTOCOL.md), [DECISIONS](docs/DECISIONS.md), [WORKLOG](docs/WORKLOG.md) (every experiment), [FIX_DECLARATION](docs/FIX_DECLARATION.md), [COMPLIANCE_MATRIX](docs/COMPLIANCE_MATRIX.md), [TECHNICAL_REPORT](docs/TECHNICAL_REPORT.md) |
| `reports/` | [benchmark.md](reports/benchmark.md), [fix_loop.md](reports/fix_loop.md) |
| `scripts/` | Tools and `experiments/` (the scripts behind every worklog entry); index with commands: [scripts/README.md](scripts/README.md) |
| `tests/` | `uv run pytest -q` |

## Troubleshooting

| Symptom | Fix |
|---|---|
| Run slows down or stalls, machine swapping | Run one capture at a time. For photos, `--per-room` keeps each MapAnything run small |
| `No capture found in …` (exit 2) | The folder layout doesn't match a tier: see [Inputs](#inputs-what-a-capture-folder-looks-like) |
| `… contains several tiers` | Pass `--tier photo\|video\|lidar` |
| Model files missing | `uv run scan-fetch-weights` |
| A changed setting doesn't seem to apply | Settings are part of the cache key; to force the models anyway, `--no-cache` |
| `cv2` has no `__version__` after changing packages | `uv sync --reinstall-package opencv-python-headless` |
