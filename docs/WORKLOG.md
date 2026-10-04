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

### Open data issues

- **Hall:** W1 (361) and W3 (230) can't both be right for a rectangle. Photo 4879 shows a recess. Every straight section of wall needs measuring.
- **Hall:** sticker door not measured. Kitchen and bathroom doors commented out in the YAML.
- **Bedroom:** W2 (289) and W4 (294) differ by 5 cm. Re-measure, or confirm it's real.
- **Stain D1:** converted from raw notes, assuming 50 cm is the horizontal width and 85 cm runs to the stain's near edge. **Not visible in any photo yet.**
- **Captures still needed:** hall (`home01_photo_a`), bedroom round 2 (`home01_photo_b`).

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
- **Raw outputs:** lost; the capture folder disappeared from disk around 13:3x. The numbers above come from the run logs. Originals are on the phone as `IMG_4873`–`4880`. Thumbnails are in `captures/home01_photo_portrait/photos/thumbnails_640px/`.

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

## Open questions / next experiments

- **Repeatability:** bedroom1 again (`home01_photo_b`). Walls must agree within 1 cm (G3). The 8 cm doubled walls from E7 are the main risk.
- **4 vs 8 photos per room:** what is the minimum that still passes?
- **Output focal vs input focal** in MapAnything (E6): is the input being used as a soft hint only?
- **Offline:** cache DINOv2 code in `scan-fetch-weights`, then verify a run with the network off.
- **MapAnything load time:** 33–44 s. Investigate caching or keeping the model loaded across rooms.
- **Hall:** not a rectangle (W1 361 vs W3 230). This tests layout beyond 4 walls.
- **Mixed orientation within a room:** `IMG_4887` was excluded in E7; the pipeline must handle it.
- **E7 OWLv2 detections:** review, then start tuning O4.
