# Work Log

Running record of **everything** done on this project: setup, data, captures, measurements, incidents, experiments, and the decisions that came out of them. The final technical report is written from this file.

- **Timeline:** one line per piece of work, newest last. Link experiments (`E<n>`) and decisions (`D<n>`, in [DECISIONS.md](DECISIONS.md)).
- **Experiments:** full entries below the timeline: Question · Setup (data, code, command, commit) · Results · What affected it · Change made · Impact.
- **Outputs:** `captures/<capture>/experiments/E<n>_<tag>/` (gitignored, shipped in the data bundle). Scripts live in `scripts/` or `scripts/experiments/`, never only in a temp folder.
- **Code state:** from the first commit on, give the commit hash for each experiment.

## Timeline

### 2026-10-04

| Time | What | Result / link |
|---|---|---|
| 09:25–10:03 | Requirements extracted from the brief. Architecture, plan, capture protocol, ground-truth template written. Decisions D1–D12 | `REQUIREMENTS.md`, `docs/` |
| ~10:30 | Disk cleanup | 59 GB free (target 40 GB) |
| 11:22 | Phase 0 scaffold: uv env (Python 3.11), package skeleton per ARCHITECTURE §8, CLI stubs, `scan-fetch-weights` with pinned HF revisions + SHA256 | `uv run scan --help` works. MPS confirmed. Open3D OK on arm64 (D14) |
| 11:27 | Model licences checked: VGGT-1B is CC-BY-NC-4.0, the rest Apache-2.0 | D15 |
| 11:30 | Weights fetched and verified: VGGT-1B, DAv2 Metric Indoor Small, OWLv2 base, MapAnything (Apache) | 10 GB in `weights/` |
| 11:30–12:10 | VGGT memory, precision and frame scaling on the M4. OWLv2 sanity check | E1, E2, E3 → D13 |
| ~12:30 | Ground truth measured with a laser: bedroom1 + hall. Water stain staged on bedroom W2 | `data/ground_truth/home01.yaml` |
| 12:56–12:59 | First capture: bedroom1 + hall, 4 photos each, **portrait**, not taken from corners | `home01_photo_portrait` |
| 13:00 | Capture check: `IMG_4876` duplicated into both folders by AirDrop (" 2" suffix). Wrong copy removed by hash. All photos portrait | — |
| 13:05–13:20 | VGGT fails on the real room. Diagnosed: portrait field-of-view error + DAv2 scale bias. Rotation fix. MapAnything + EXIF tested | E4, E5, E6 → D16, D17 (tentative) |
| 13:20 | Portrait set renamed `home01_photo_portrait`. Reshoot requested | — |
| 13:23–13:26 | Reshoot: bedroom1, 8 photos (7 portrait, 1 landscape), corners + door + mid-wall | `home01_photo_a` |
| 13:30 | **O1 decided: MapAnything + EXIF intrinsics** (+2.5% / +1.2% / −3.0% against the tape). VGGT + DAv2 +53 to +57% | E7 → D17 |
| ~13:35 | Experiment scripts moved from the temp folder into `scripts/experiments/`. Outputs archived under `captures/*/experiments/` | — |
| ~13:38 | **Data incident:** `home01_photo_portrait/photos` (IMG_4873–4880), `home01_photo_b/` and the E5 raw output deleted during a manual cleanup. Not recoverable from git (`captures/` is gitignored, no commits yet). Originals still on the iPhone. E4–E6 numbers kept in this log | Lesson: back up `captures/` separately from git |
| 13:38 | Assignment-provided **real LiDAR samples** (Stray Scanner exports) added at the repo root, gitignored: `1a8384c3f6` (289 MB), `c00a170fe1` (93 MB), `c7d28f72c6` (530 MB). Each has `rgb.mp4`, `depth/`, `confidence/`, `odometry.csv`, `imu.csv`, `camera_matrix.csv` | Tier 3 can be tested on real data, not only synthetic. D3 to be revisited |
| 13:50 | Log widened from experiments-only to everything (this file). First git commits | — |
| 13:45–13:47 | Hall captured: 7 portrait photos (`IMG_4890`–`4897`, no `4893`), 1× lens, no duplicates. Openings seen: bedroom doorway, carved wooden door, sticker door + window, open kitchen doorway | `home01_photo_a/photos/hall` |
| ~14:05 | MapAnything + EXIF on the hall. Walls −4.9% / −5.7%, ceiling −9.0% (all low: scale about 5–9% small here, against +1 to +2.5% in the bedroom) | E8 |
| ~14:10 | Compared against the requirement gates: walls pass photo-tier ±8% in both rooms. **Ceiling G2 (≤ 1.5 cm) is far off** (8 cm bedroom, 25 cm hall) | Next priorities set below |
| ~14:20 | Agreed order: E9 (joint run) → switch to MapAnything + offline setup → Tier 1 proper | — |
| ~14:25 | **Joint bedroom + hall run (14 photos): the rooms register through the doorway.** Shared wall 10–15 cm apart (a believable wall thickness), and both rooms on one scale | E9 |
| ~14:30 | User: the hall is **L-shaped**. The kitchen opening near W4 takes up the 361 → 230 difference. Confirmed by the E9 cloud (about 1.3 m extra width at the W1 end) | Hall ground truth needs restructuring |
| ~14:35 | Ceilings 6–8% low in every run. Cause found: MapAnything's output focal is 1.118× the EXIF focal. Rebuilding points with EXIF rays brings the joint ceilings to +0.6% / +1.7%, but doubles walls more | E10 |
| ~14:45 | **Switched the main env to MapAnything.** Removed `vggt` and `opencv-python`; added `mapanything@3d10cf7` (brings `opencv-python-headless==4.10.0.84`, numpy 2.4.6). VGGT + DAv2 marked legacy (`--optional`) in the weights registry | D17 |
| ~14:45 | Gotcha: uninstalling `opencv-python` in place deleted `cv2` files shared with the headless build (`cv2.__version__` missing). Fixed with `uv sync --reinstall-package opencv-python-headless`. A fresh clone won't hit this | — |
| ~14:50 | **Offline:** DINOv2 encoder code vendored at `facebookresearch/dinov2@7764ea0` into `weights/dinov2-code` by `scan-fetch-weights` (verified by `hubconf.py` SHA256). `scan/geometry/mapanything_backend.py` routes `torch.hub.load("facebookresearch/dinov2")` to it, so no network call at load. It's the same code E6–E10 used (only `__pycache__` differs) | — |
| ~14:55 | Offline run with network blocked and an empty torch cache: works, and is **bit-identical to E7** | E11 |
| ~14:55 | ARCHITECTURE updated (stack, layout, §10.2 photo geometry, risks, O1–O3 decided). Spike script marked legacy (re-run E1–E7 at commit `31dbc88`). Temp `.venv-mapanything` removed | — |
| ~15:05 | **Tier 1 started.** Step 1.1 ingest: `scan/io/ingest.py` (tier detection incl. bare Stray Scanner exports, validation), `scan/io/photos.py` (EXIF-upright decode, diagonal-based intrinsics, sharpness), `scan/types.py`, `scan/config.py` + `config/default.yaml`, `scan/errors.py` (exit codes 2/3). `uv run scan` prints the capture summary | D18, D19 |
| ~15:10 | Found while writing ingest: ARCHITECTURE's f35 → pixels formula used the film width (4% short on 4:3). Switched to the diagonal, which E6–E11 already used | D18 |
| ~15:15 | Ingest on `home01_photo_a`: bedroom1 7 + hall 7 photos, portrait, f35 = 26. `IMG_4887` (landscape) dropped automatically, as done by hand in E7. Bad folders give clear errors with exit code 2 (empty `photo_b`, the old portrait folder, a LiDAR sample detected as LiDAR) | — |
| ~15:20 | Speed: HEIC decode is 283 of 355 ms per photo. Threaded decode → 185 ms per photo (15 photos in 2.78 s, about 5.6 s for 30 against the guessed 5 s budget). 19 unit tests pass. Lint clean in `scan/` and `tests/` | Step 1.1 exit check ✅ |
| ~15:20 | Moved the portrait-capture thumbnails out of `photos/` to `captures/home01_photo_portrait/thumbnails_640px/`, so the folder isn't read as a room | — |

### Open data issues

- **Hall:** W1 (361) and W3 (230) can't both be right for a rectangle. Photo 4879 shows a recess. Every straight section of wall needs measuring.
- **Hall:** missing from the YAML but visible in the photos: sticker door, the window next to it, the open kitchen doorway (`IMG_4894`). The bathroom door is commented out.
- **Hall shape (L-shaped, per user + E9):** W1 (shared with the bedroom) is 361. The far wall W3 is 230. The extra 131 cm at the W1 end is the kitchen opening near W4. As a polygon that is about 6 walls: W1 → a short end wall at the kitchen side (about 0.9 m in E9) → a step wall back (about 1.3 m) → the long wall → W3 → the other long wall. The step segments need measuring, and the YAML walls renumbered clockwise.
- **Bedroom:** W2 (289) and W4 (294) differ by 5 cm. Re-measure, or confirm it's real.
- **Stain D1:** converted from raw notes, assuming 50 cm is the horizontal width and 85 cm runs to the stain's near edge. **Not visible in any photo yet.**
- **Captures still needed:** bedroom round 2 (`home01_photo_b`, repeatability G2/G3).

---

## Experiments

### Environment (E1–E7)

| | |
|---|---|
| Machine | MacBook M4, 16 GB unified memory, macOS (Darwin 25.5), MPS |
| Python env | uv, Python 3.11.8, torch 2.14.1, transformers 5.18.0, numpy 1.26.4, open3d 0.20.0 |
| VGGT | code `facebookresearch/vggt@a288dd0`, weights `facebook/VGGT-1B@860abec` |
| MapAnything | code `facebookresearch/map-anything@3d10cf7` in a separate env (see `scripts/experiments/mapanything_run.py`), weights `facebook/map-anything-apache@00f9c24` |
| Metric depth | `depth-anything/Depth-Anything-V2-Metric-Indoor-Small-hf@8078d68` |
| Detector | `google/owlv2-base-patch16-ensemble@cfd3195` |
| Code state | uncommitted (before the first commit) |
| Phone | iPhone 13, 1× lens (5.1 mm, 26 mm equivalent), HEIC 4032×3024 |

### Environment (E11 onward)

Same machine. Main env after the switch: torch 2.14.1, numpy 2.4.6, `opencv-python-headless` 4.10.0.84, open3d 0.20.0, transformers 5.18.0, `mapanything@3d10cf7`. DINOv2 code `facebookresearch/dinov2@7764ea0` vendored in `weights/dinov2-code`. No `vggt`.

### Ground truth used

`data/ground_truth/home01.yaml`, laser measurements.

- **bedroom1:** W1 = W3 = 239 cm, W2 = 289 cm, W4 = 294 cm (averaged to 291.5), ceiling 279 / 279.5 cm.
- **Tape targets for the spike:** short side 2.39 m, long side 2.915 m, ceiling 2.7925 m.

### Summary

| ID | Question | Result | Outcome |
|---|---|---|---|
| E1 | Does VGGT-1B fit and run fast on a 16 GB M4? | Stock fp32: 212 s, 17.2 GB, +15 GB swap. Hand-run fp16 aggregator: **28 s, 6.8 GB**, same geometry | D13 |
| E2 | How does VGGT scale with frame count? | 8 → 28 s, 16 → 54 s, 24 → 100 s. Memory flat at 6.8 GB | D13 (keyframe budget for video) |
| E3 | Is OWLv2 sane out of the box? | Doors and doorways right. False mirror / water stain hits at 0.15–0.32 | O4 still open |
| E4 | First real room: VGGT + DAv2 vs tape | Failed: 0.14 × 4.26 m, ceiling 3.55 m | Found E5 and the scale problem |
| E5 | Does rotating portrait photos fix VGGT? | Focal fixed (fx = fy ≈ 422 against EXIF 390). Scale still about 1.4× | D16 |
| E6 | MapAnything, with and without EXIF focal | With focal: walls −1.5% / −0.4%, ceiling +5.7%. Without: ceiling +11.8% | D17 (tentative) |
| E7 | O1 decider on a protocol capture | VGGT + DAv2 **+53 to +57%**. MapAnything + focal **+2.5% / +1.2% / −3.0%** | **D17: MapAnything** |
| E8 | MapAnything on a second room (hall) | Walls −4.9% / −5.7% (pass ±8%), ceiling −9.0%. Scale differs per room | Joint multi-room run and scale anchors are next (E9) |
| E9 | One joint run over bedroom + hall | Rooms register through the doorway, shared scale. Bedroom −0.8% / +0.5%. Hall about +1 to +4% (read from the plot; the heuristic failed). Ceilings −6.2% / −8.3% | Photo-tier stitching may come from the joint run (G5) |
| E10 | Rebuild points with the EXIF focal instead of the model's rays | Joint ceilings **+0.6% / +1.7%** (from −6 / −8%). Walls more doubled (up to about 15 cm). Per-room runs mixed | Ceiling fix found. Pose/ray consistency still needed |
| E11 | MapAnything in the main env, fully offline | Runs with network blocked and an empty torch cache. **Bit-identical to E7** (max point difference 0.0 m) | Switch verified. Offline requirement met for the backbone |

### What affects the results (so far)

| Factor | Effect seen | Experiment | Handling |
|---|---|---|---|
| Precision and memory | fp32 VGGT overflows 16 GB, swaps, and runs 7.5× slower | E1 | fp16 aggregator, skip point head, chunk depth head |
| Frame count | VGGT time grows faster than linear | E2 | Keyframe budget for video |
| Photo orientation | Portrait makes VGGT guess fx 26% low, stretching the room sideways | E4, E5 | Rotate to landscape (D16). MapAnything + EXIF doesn't care |
| Known focal length (EXIF) | Halves MapAnything's ceiling error (+11.8% → +5.7%) | E6 | Always pass EXIF intrinsics |
| Metric scale source | DAv2 metric scale biased 1.4–1.55× in this home | E4–E7 | Dropped DAv2 as the scale source (D17) |
| Shots not taken from corners | The wall behind the photographer is never seen | E4, E5 | Protocol: back in the corner |
| Bed covering the floor | Bed top picked as the floor, ceiling off by up to 28% | E4, E6 | Protocol shots show the floor (E7 fine). Layout: lowest plane with support |
| Built-in wardrobe | Wardrobe front picked as a wall (−20%) | E6 | Layout: outermost plane with support |
| View through a door | Points from the next room land in this room's cloud | E4, E6 | Layout: clip to the room polygon |
| Peak-picking heuristic | Strongest-peak picking fails. Same result at threshold 0.2 and 0.08 once the photos are good | E4, E7 | Rewrite in `layout/room.py` |
| View alignment | Walls doubled by about 8 cm | E7 | Fix-loop candidate (G3, 1 cm) |
| Per-room scale | MapAnything metric scale +1 to +2.5% in the bedroom, −5 to −9% in the hall when run separately | E7, E8 | **Joint run shares one scale** (E9) |
| Model focal ≠ EXIF focal | MapAnything outputs a focal 1.118× the EXIF focal, which compresses vertical extents. Ceilings come out 6–8% low | E6–E10 | EXIF rays fix ceilings (E10) but break pose consistency. Needs a consistent fix |
| Sparse far walls | Heuristic misses walls seen only from far away, or picks things outside the room (hall long side 1.47 / 6.69 m) | E9, E10 | Layout must use per-room polygons, not global percentile peaks |
| File handling | AirDrop renamed a duplicate `IMG_4876 2.HEIC` into the wrong folder. Capture folders vanished from disk (13:3x) | E4 | Hash-check on ingest. Keep originals on the phone |
| Determinism | Re-running E4 gave identical numbers | E4 | ✓ |

---

### E1. VGGT memory and precision on the M4

- **Question:** can VGGT-1B run within the 12 GB memory ceiling and 60 s target?
- **Setup:**
  - Data: VGGT repo example `room`, 8 images (4:3, resized to 518×392).
  - Command: `scripts/spike_phase0.py --precision fp32|fp16|half --dpt-chunk 2`.
  - Outputs: `captures/vggt_examples/experiments/E1_*`.
- **Results:**

  | Mode | Load | Infer | Peak MPS | Swap growth | Min free RAM | Room (w × l, ceiling) | Scale s |
  |---|---|---|---|---|---|---|---|
  | fp32, stock forward (camera + depth + point heads) | 21.6 s | 211.9 s | 17.18 GB | +15.0 GB | 0.10 GB | 4.747 × 5.350, 2.480 | 2.7248 |
  | fp16 autocast | 18.8 s | 177.6 s | 17.16 GB | +9.9 GB | 0.08 GB | 4.748 × 5.350, 2.490 | 2.7256 |
  | **half**: fp16 aggregator, fp32 heads, no point head, depth head 2 frames/pass | 20.7 s | **28.2 s** | **6.79 GB** | +1.5 GB | 1.46 GB | 4.747 × 5.350, 2.560 | 2.7222 |

  Per-stage memory in half mode: 2.73 GB after load, 3.73 GB after the aggregator, 6.79 GB after the heads.
- **What affected it:**
  - Autocast keeps the fp32 weights (5 GB) resident.
  - VGGT's heads disable autocast only for CUDA (`torch.cuda.amp.autocast(enabled=False)`), so on MPS fp16 reaches the heads too.
  - Running all frames through the full-resolution DPT heads at once is the memory peak.
  - The 2.48 vs 2.56 m ceiling difference is heuristic noise: the example photos barely show the ceiling. It is not a precision effect.
- **Bug found:** the aggregator returns `None` for layers the heads don't use, which broke the fp32 cast. Fixed.
- **Change:** `run_vggt` runs aggregator → camera head → depth head by hand (`scripts/spike_phase0.py`).
- **Impact:** D13. Without this, VGGT is unusable on the dev machine.

### E2. VGGT frame-count scaling

- **Setup:**
  - Data: VGGT example `kitchen`, first 16 and first 24 images.
  - Mode: half.
  - Outputs: `captures/vggt_examples/experiments/E2_*`.
- **Results:**

  | Frames | Load | Infer | Peak MPS | After aggregator | Swap growth |
  |---|---|---|---|---|---|
  | 8 (E1) | 20.7 s | 28.2 s | 6.79 GB | 3.73 GB | +1.5 GB |
  | 16 | 18.4 s | 54.1 s | 6.79 GB | 3.73 GB | +0.8 GB |
  | 24 | 18.6 s | 100.3 s | 6.81 GB | 4.73 GB | +0.2 GB |

- **What affected it:** global attention grows faster than linear with frame count. The peak is set by the chunked depth head, so memory stays flat.
- **Impact:**
  - Photo tier: about 25 s per room, which fits the budget.
  - Video tier: 600 keyframes would not fit, so the keyframe budget must be about 0.5 fps or coverage-based (Tier 2 step 2.1).
  - These timings must be re-measured for MapAnything.

### E3. OWLv2 sanity check

- **Setup:**
  - Data: VGGT example `room`, 8 images.
  - Queries: door, doorway, window, mirror, water stain on a wall, crack in a wall.
  - Threshold 0.15. Images resized to ≤960 px.
  - Boxes mapped through the padded square frame (`target_sizes=(side, side)`).
  - Outputs: `captures/vggt_examples/experiments/E1_room_fp16_autocast/owl/`.
- **Results:** 29 detections in 13.0 s.
  - Correct: door 0.36, doorway 0.28.
  - False positives: "mirror" 0.53 on a white panel; "water stain" 0.16–0.32 on plain walls.
  - Box positions line up with objects, so the coordinate mapping is right.
- **Impact:** needs per-class thresholds and NMS (O4, tuned on the benchmark).

### E4. First real capture: VGGT + DAv2 against the tape

- **Setup:**
  - Data: `home01_photo_portrait/bedroom1`, 4 portrait photos (`IMG_4873`–`4876`).
  - Capture faults: not taken from corners, bed covering most of the floor.
  - Mode: half. Frames resized so the long side is 518 (392×518), no crop.
  - Command: `scripts/spike_phase0.py <dir> --tape-short 239 --tape-long 291.5 --tape-height 279.25`.
- **Results:** short side 0.140 m (−94%), long side 4.260 m (+46%), ceiling 3.550 m (+27%). s = 3.41, between-image σ_log s = 0.065. Infer 17.1 s.
- **Diagnostics (from saved raw outputs):**
  - VGGT focal: fx = 279–304 px, fy = 370–402 px. **EXIF focal is 390 px.** fx is 26% low, which stretches the room sideways.
  - Largest distance between cameras: 4.77 m. The room diagonal is only 3.77 m, so the scale or shape is wrong.
  - The ceiling sits 2.06 m above camera 0. If the scale were right, the phone would have been held at 0.73 m, so the scale is about 1.44× too big.
  - DAv2 median depth in image 0 is 3.61 m, in a room 2.9 m deep. DAv2 overestimates.
  - The measurement heuristic found two peaks on the same wall (giving 0.14 m) and took the bed top as the floor.
- **Other findings:**
  - `IMG_4876` arrived in both folders (AirDrop's " 2" suffix); identical SHA256. The wrong copy was removed.
  - Re-running gave identical numbers, so the pipeline is deterministic.
- **Raw outputs:** lost; the capture folder disappeared from disk around 13:3x. The numbers above come from the run logs. Originals are on the phone as `IMG_4873`–`4880`. Thumbnails are in `captures/home01_photo_portrait/thumbnails_640px/`.

### E5. Rotating portrait photos to landscape for VGGT

- **Question:** is the fx error caused by portrait framing?
- **Setup:**
  - Data: same 4 photos as E4.
  - `scripts/experiments/portrait_rotation.py`: compares as-is vs rotated 90° anticlockwise. Gravity is read from camera +x after rotation.
  - Then the spike was run with rotation built in.
- **Results:**

  | Input | fx (px) | fy (px) | EXIF | Extents / ceiling-above-camera |
  |---|---|---|---|---|
  | Portrait 392×518 | 279–304 | 370–402 | 390 | 2.35, 1.86 (lopsided) |
  | Rotated 518×392 | 421–426 | 421–427 | 390 | 1.86, 1.87 |

  Spike with rotation: 2.610 × 3.616 m, ceiling 3.170 m (+9.2%, +24.0%, +13.5%). s = 3.28, σ = 0.082. Infer 13.0 s.
  - One wall (behind the cameras) was never photographed, so the long side is not valid.
  - Measured to the wall behind the wardrobe (−2.95), the width is 3.38 m (+41%). That agrees with the 1.44× scale estimate from E4.
- **Change:** rotation built into the spike (D16). Depth Anything runs on the upright photo and its depth map is rotated to match.
- **Impact:** geometry fixed. Scale is still wrong, so the problem is the scale source.

### E6. MapAnything with and without EXIF focal

- **Setup:**
  - Data: same 4 photos.
  - `scripts/experiments/mapanything_run.py <dir> <out> k|nok`: fp16 AMP, memory-efficient heads (mini-batch of 1 on MPS).
  - Measurement: `scripts/experiments/measure_cloud.py`.
  - Outputs: `captures/home01_photo_portrait/experiments/E6_*`.
- **Results:**

  | Mode | Load | Infer | Peak MPS | Focal out (px) | Short (auto) | Long (auto) | Ceiling (auto) | Floor-to-top (p99.7) |
  |---|---|---|---|---|---|---|---|---|
  | EXIF focal given | 43.9 s | 19.5 s | 6.12 GB | 362–452 | 1.900 (−20.5%)* | 2.870 (−1.5%) | 2.010 (−28.0%)** | 2.953 (+5.7%) |
  | No focal | 37.2 s | 14.1 s | 6.12 GB | 308–470 | 2.550 (+6.7%) | 2.920 (+0.2%) | 3.100 (+11.0%) | 3.122 (+11.8%) |

  \* The heuristic picked the wardrobe front. Measured to the wall behind the wardrobe: 2.40 m (−0.4%).
  \** The heuristic picked the bed top as the floor.
- **Findings:**
  - The EXIF focal helps: the ceiling-height error halves.
  - The output focal (362–452) doesn't equal the input focal (389). Not yet understood; logged as an open question.
  - **On load, MapAnything downloads DINOv2 code from GitHub through torch hub.** This breaks the offline requirement, so it must be pre-cached.
- **Impact:** D17 (tentative).

### E7. Backbone decider (O1) on a protocol capture

- **Setup:**
  - Data: `home01_photo_a/bedroom1`, 8 photos taken to protocol (corners + door + mid-wall). Floor visible.
  - 7 portrait photos used; `IMG_4887` (landscape) excluded because it doesn't match the others' shape.
  - VGGT: `scripts/spike_phase0.py <dir> --exclude IMG_4887.HEIC --tape-short 239 --tape-long 291.5 --tape-height 279.25`.
  - MapAnything: `scripts/experiments/mapanything_run.py <dir> <out> k IMG_4887.HEIC`.
  - Both measured with `measure_cloud.py` at peak thresholds 0.2 and 0.08.
  - Outputs: `captures/home01_photo_a/experiments/E7_*`.
- **Results:**

  | Backbone | Load | Infer | Peak MPS | Short | Long | Ceiling |
  |---|---|---|---|---|---|---|
  | VGGT (rotated) + DAv2 scale (s = 3.88, σ = 0.072) | 19.4 s | 24.7 s | 7.79 GB | 3.650 (+52.7%) | 4.570 (+56.8%) | 4.300 (+54.0%) |
  | **MapAnything + EXIF focal** | 33.0 s | 28.6 s | 7.13 GB | **2.450 (+2.5%)** | **2.950 (+1.2%)** | **2.710 (−3.0%)** |

  - Both peak thresholds give identical numbers.
  - The MapAnything plot shows a sharp floor (at −1.41, phone about 1.41 m high) and a sharp ceiling. It skipped the wardrobe front and found the wall behind it.
  - Walls are doubled by about 8 cm.
  - OWLv2 on this capture: 80 detections in 10.1 s, not yet reviewed.
- **What affected it:**
  - VGGT's three errors are equal, so the shape is right and the DAv2 scale is about 1.55× too big.
  - Good capture (floor visible, corners covered) fixed the heuristic failures seen in E4 and E6.
- **Impact:** **D17: O1 = MapAnything + EXIF intrinsics.** Passes the Phase 0 exit check (within 10%).

---

### E8. MapAnything on the hall

- **Question:** does the E7 accuracy carry over to a second, non-rectangular room?
- **Setup:**
  - Data: `home01_photo_a/photos/hall`, 7 portrait photos.
  - Env: `.venv-mapanything` (in the repo, gitignored).
  - Commands:
    - `PYTORCH_ENABLE_MPS_FALLBACK=1 .venv-mapanything/bin/python scripts/experiments/mapanything_run.py captures/home01_photo_a/photos/hall captures/home01_photo_a/experiments/E8_hall_mapanything_k k`
    - `uv run python scripts/experiments/measure_cloud.py <out>/ma_raw.npz 0.08 361 370 280.25`
  - Code at commit `5e99225`, plus a `measure_cloud.py` change to take tape values as arguments.
- **Results:** load 39.3 s, infer 23.7 s, peak MPS 7.13 GB. Focal out 395–449 px (input 389).

  | | Tape | MapAnything | Error |
  |---|---|---|---|
  | Long side (W2 / W4) | 3.70 m | 3.520 m | −4.9% |
  | Short side | W3 2.30 / W1 3.61 | 2.170 m | −5.7% against W3 |
  | Ceiling | 2.8025 m | 2.550 m | −9.0% |

  - The plot shows a clean box: straight long walls, complete end walls, and a 15 cm step at one end (the recess?).
  - A sparse patch of points comes from seeing through a doorway.
- **What affected it:**
  - All three errors are low, so this is a **scale error, not a shape error**.
  - MapAnything's metric scale is −5 to −9% here against +1 to +2.5% in the bedroom. Per-room scale is not consistent.
- **Impact:**
  - Walls pass the photo-tier ±8% in both rooms. The ceiling misses G2 (≤ 1.5 cm) by a wide margin.
  - Next: run all rooms jointly so they share one scale, and try scale anchors. The doors are measured at 211 cm in both rooms.
  - The 361 vs 230 hall conflict still needs measuring.

### E9. Joint bedroom + hall reconstruction

- **Question:** in one run, do the rooms share one scale, and do the door shots register them to each other (stitching for free)?
- **Setup:**
  - `scripts/experiments/mapanything_run.py captures/home01_photo_a/photos/bedroom1,captures/home01_photo_a/photos/hall <out> k IMG_4887.HEIC`. The runner now accepts several room folders and saves each image's room.
  - Analysis: `scripts/experiments/joint_rooms.py <out>/ma_raw.npz bedroom1=239,291.5,279.25 hall=230,370,280.25`.
  - 14 photos (7 + 7). Code at commit `8fe2940`, plus the runner and analysis changes.
  - Outputs: `captures/home01_photo_a/experiments/E9_joint_bedroom_hall/`.
- **Results:** load 37.1 s, infer 59.6 s, peak MPS 8.14 GB. Focal out 389–458 px (mean 436) against EXIF 390.

  | Room | Short | Long | Ceiling |
  |---|---|---|---|
  | bedroom1 (auto) | 2.370 (−0.8%) | 2.930 (+0.5%) | 2.620 (−6.2%) |
  | hall (auto) | 1.470 (−36.1%) | 2.280 (−38.4%) | 2.570 (−8.3%) |
  | hall (read from plot) | ~2.30 at W3 (≈ 0%) | ~3.65 (W2/W4 370, −1.4%) | — |

  - The hall's auto numbers are a **heuristic failure**: the far wall (W3) is seen only from far away and didn't clear the peak threshold.
  - The joint top-down shows a clean bedroom rectangle and the hall attached across the shared wall. The hall wall is 10–15 cm from the bedroom's inner wall face, a believable interior wall thickness.
  - The L-shape shows: about 1.3 m extra width at the W1 end (the kitchen opening, confirmed by the user).
  - Hall photos that look through the doorway land on the bedroom's far wall in the right place.
- **What affected it:** shared doorway views tie the rooms together. Most hall photos were taken from the W1 end, so the cameras sit in a line there.
- **Impact:**
  - Joint reconstruction gives **one shared scale** and **the room-to-room placement**. That is a candidate for photo-tier stitching (G5) instead of door matching + least squares (ARCHITECTURE §10). Needs more rooms to trust.
  - The measuring heuristic has to become per-room polygon fitting.

### E10. Rebuild points with the EXIF focal (ceiling fix)

- **Question:** are the low ceilings (−6 to −8%) caused by MapAnything's output focal being 1.118× the EXIF focal?
- **Setup:**
  - `scripts/experiments/reproject_exif.py <ma_raw.npz> <out>`: keeps the predicted depth and poses, swaps the rays for EXIF-intrinsics rays.
  - Applied to E9 (joint), E7 (bedroom alone) and E8 (hall alone). Re-measured with `joint_rooms.py` / `measure_cloud.py`.
  - No model re-run.
- **Results:**

  | Run | Short | Long | Ceiling |
  |---|---|---|---|
  | Joint, bedroom1 | 2.420 (+1.3%) | 3.010 (+3.3%) | **2.810 (+0.6%, +1.75 cm)** |
  | Joint, hall | 2.320 (+0.9%) | 6.685 (heuristic picked points outside the room) | **2.850 (+1.7%, +4.75 cm)** |
  | Joint, all points | — | — | 2.820 |
  | Bedroom alone (E7 + EXIF rays) | 2.710 (+13.4%) | 3.090 (+6.0%) | 2.890 (+3.5%) |
  | Hall alone (E8 + EXIF rays) | 2.220 (−3.5%) | 3.530 (−4.6%) | 2.710 (−3.3%) |

  - Joint top-down: hall walls coherent (about 3.75 × 3.75 including the kitchen opening, W3 about 2.30).
  - **Bedroom walls doubled or tripled, spread up to about 15 cm.** The predicted poses are consistent with the predicted rays, not with the EXIF rays.
- **What affected it:** the hypothesis holds. The too-long model focal compresses the vertical extents in each camera. Swapping only the rays creates a pose/ray mismatch.
- **Impact:**
  - The ceiling bias is explained and fixable.
  - The right fix keeps rays and poses consistent: for example, refine the poses with EXIF intrinsics fixed, or work out why MapAnything doesn't honour the input intrinsics (E6 open question).
  - G2 (≤ 1.5 cm) is now within reach in the bedroom (+1.75 cm) but not yet in the hall (+4.75 cm).

### E11. MapAnything switch: offline run in the main env

- **Question:**
  - Does the main env run MapAnything with no network access?
  - Does it reproduce the separate-env results (E7)?
- **Setup:**
  - Command: `HTTP(S)_PROXY=http://127.0.0.1:9 ALL_PROXY=… TORCH_HOME=<empty dir> uv run --offline python scripts/experiments/mapanything_run.py captures/home01_photo_a/photos/bedroom1 captures/home01_photo_a/experiments/E11_offline_main_env k IMG_4887.HEIC`.
  - Then `measure_cloud.py` at 0.08, and a point-by-point comparison with E7's `ma_raw.npz`.
- **Results:**
  - Ran fine: load 42.2 s, infer 25.8 s, peak 7.13 GB. The torch cache dir stayed empty.
  - 2.450 / 2.950 / 2.710 m, the same as E7.
  - **Max point difference vs E7: 0.0 m; masks 100% equal.**
- **What affected it:** nothing. Same mapanything commit, same DINOv2 code, same weights, same inputs, deterministic on MPS.
- **Impact:**
  - Switch verified.
  - The backbone meets the offline requirement (NFR-21).
  - E7's numbers stay valid for the new env.

## Open questions / next experiments

- **Repeatability:** bedroom1 again (`home01_photo_b`). Walls must agree within 1 cm (G3). The 8 cm doubled walls from E7 are the main risk.
- **4 vs 8 photos per room:** what is the minimum that still passes?
- **Output focal vs input focal** in MapAnything (E6): is the input being used as a soft hint only?
- **MapAnything load time:** 33–44 s. Investigate caching or keeping the model loaded across rooms.
- **Hall:** not a rectangle (W1 361 vs W3 230). This tests layout beyond 4 walls.
- **Stitching via joint reconstruction (after E9):** does it hold with more rooms, and with rooms whose doorway views are weak? Compare against door matching + least squares on G5.
- **Pose/ray consistency (after E10):** why does MapAnything not honour the input intrinsics? Options: check the `preprocess_inputs` intrinsics handling, or a small pose refinement with EXIF intrinsics fixed.
- **Ceiling G2 (≤ 1.5 cm):** current error 8–25 cm. Probably the hardest gate for the photo tier. Report honestly with intervals if it can't be met.
- **Mixed orientation within a room:** `IMG_4887` was excluded in E7; the pipeline must handle it.
- **E7 OWLv2 detections:** review, then start tuning O4.
