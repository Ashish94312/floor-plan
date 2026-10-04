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
| ~14:55 | ARCHITECTURE updated (stack, layout, §10.2 photo geometry, risks, O1–O3 decided). Spike script marked legacy (re-run E1–E7 at commit `1d4eca8`). Temp `.venv-mapanything` removed | — |
| ~15:05 | **Tier 1 started.** Step 1.1 ingest: `scan/io/ingest.py` (tier detection incl. bare Stray Scanner exports, validation), `scan/io/photos.py` (EXIF-upright decode, diagonal-based intrinsics, sharpness), `scan/types.py`, `scan/config.py` + `config/default.yaml`, `scan/errors.py` (exit codes 2/3). `uv run scan` prints the capture summary | D18, D19 |
| ~15:10 | Found while writing ingest: ARCHITECTURE's f35 → pixels formula used the film width (4% short on 4:3). Switched to the diagonal, which E6–E11 already used | D18 |
| ~15:15 | Ingest on `home01_photo_a`: bedroom1 7 + hall 7 photos, portrait, f35 = 26. `IMG_4887` (landscape) dropped automatically, as done by hand in E7. Bad folders give clear errors with exit code 2 (empty `photo_b`, the old portrait folder, a LiDAR sample detected as LiDAR) | — |
| ~15:20 | Speed: HEIC decode is 283 of 355 ms per photo. Threaded decode → 185 ms per photo (15 photos in 2.78 s, about 5.6 s for 30 against the guessed 5 s budget). 19 unit tests pass. Lint clean in `scan/` and `tests/` | Step 1.1 exit check ✅ |
| ~15:20 | Moved the portrait-capture thumbnails out of `photos/` to `captures/home01_photo_portrait/thumbnails_640px/`, so the folder isn't read as a room | — |
| ~15:30 | **Step 1.2 started.** First, settle which focal is physically right before building on it | — |
| ~15:35 | Checked our side: MapAnything's preprocessing passes the EXIF focal through correctly (769 px at 1024 → 392.6 px at 392×518). The ~436 px output is the model's own bias, not our bug | E12 |
| ~15:40 | **Vanishing-point check (no learned model):** on oblique corner views, the focal from geometry is 0.94–1.08× EXIF. **EXIF is right; MapAnything's focal is ~12% too long**, which is why ceilings come out low. Head-on views can't be used for this check | E12 |
| ~15:50 | Pre-compensating (feeding f × 0.893): the output focal follows the input (median 379 px). Ceilings +1.3% / +3.1%, but per-view focals spread 329–427 px and ceiling points from different views disagree by > 10 cm. Not clean | E13 |
| ~15:55 | **Decision: park the ceiling fix as the fix-loop candidate (G2).** Root cause proven (E10, E12, E13), fix planned (EXIF rays + ICP pose re-fit). Default stays MapAnything's own consistent rays. Both are config switches | D20 |
| ~16:00 | Built step 1.2: `predict()` in `scan/geometry/mapanything_backend.py` (cache-aware, model loaded lazily once), `scan/geometry/cloud.py` (model/EXIF rays, confidence filter, 2 cm voxel), `scan/cache.py` (content-addressed `.npz`, atomic write), `scan/pipeline.py` (ingest → geometry → `out/rooms/<room>/cloud.ply` + `out/geometry.json`), `--joint` flag | — |
| ~16:05 | First run, home01_photo_a: bedroom1 142,923 points, hall 133,180. 1 min 45 s live (load 35 s, about 32 s inference per room). **Cached re-run: 7 s, byte-identical `.ply` files.** Sizes match the experiments: bedroom 2.390 (0.0%) / 2.940 (+0.9%) / ceiling −3.0%; hall ≈ E8 | Step 1.2 exit check ✅ (cloud in metres) |
| ~16:10 | 26 unit tests pass (projection maths, the E12 focal-compression effect, confidence filter + voxel, cache round-trip, real-capture cache replay). Lint clean | — |
| ~16:15 | User: **fix the ceiling now** instead of parking it for the fix loop. D20 reversed | — |
| ~16:20 | Fix attempt B, EXIF rays + pairwise ICP + pose graph: **failed.** Most view pairs overlap 5–30%, ICP slides along flat walls (corrections of 1–2 m and 40–150°), the graph disconnects. ICP code moved to `scripts/experiments/` as a negative result | E14 |
| ~16:25 | Found along the way: per room, EXIF rays don't misalign views (gap between views 2.2–3.3 cm, same as model rays), but each room's own scale is off by about ±5% (bedroom big, hall small) | E14 |
| ~16:35 | **Joint run + EXIF rays wins:** ceilings −0.1% / +0.6%, walls 5 of 6 within 1.5%, mean abs error 1.75% (joint + model rays 3.55%). Bedroom short side +7.9% (open door leaf + wardrobe recess, inside ±8%). View gaps lower with EXIF rays | E15 → D20 |
| ~16:40 | New defaults: `geometry.joint: auto` (joint if ≤ 16 photos, else per room with a warning), `geometry.rays: exif`, `sigma_log_floor_photo: 0.05`. CLI `--joint/--per-room`. `plan_groups()` + unit test. 27 tests pass. Pipeline default on home01_photo_a: hall +1.3 / +0.3 / ceiling +0.3%, bedroom +7.5 / +1.9 / ceiling −1.9% (rough spike measurement; the real wall fit is step 1.4) | Step 1.2 complete |
| ~16:50 | **Step 1.3 alignment** (`scan/geometry/align.py`). Normals (10 cm). Rough up from the camera axes. Floor = lowest strong horizontal peak, ceiling = highest one ≥ 1.8 m above it. Up = floor normal, averaged with the ceiling normal when they agree within 3°. Manhattan angle from wall-normal azimuths folded mod 90°. One transform per reconstruction frame (joint → the whole house shares it); points and camera poses move into the aligned frame. `out/rooms/<room>/align.png` debug plots | — |
| ~16:55 | First run: tilt 1.1°, walls rotated −24.5°, Manhattan support 60%, floor RMS 1.2 cm. Ceilings from plane fits: hall 2.813 (+1.05 cm), bedroom 2.861 (+6.9 cm). The bedroom height histogram shows **two floor peaks**, 0.0 and +0.09 | — |
| ~17:00 | Per-view check: each view's own floor z (bedroom 0.024–0.075, hall −0.054–+0.030) shows **views disagree by 5–8 cm on surface heights** (doubling, as with walls), not low objects. Single-view floor/ceiling detection is unreliable (often picks the bed or the ceiling), so per-view heights aren't usable directly | — |
| ~17:05 | Floor/ceiling became **consensus planes**: the layer around the peak (floor −5..+12 cm, ceiling mirrored), offset at the median, direction from the dominant ±3 cm core. The first version took the direction from the whole layer: the synthetic test showed that a partially doubled floor tilts it (1.5–4 cm error), so it was fixed. Heights in the aligned frame = median z per layer | D22 |
| ~17:10 | Result (estimator chosen on principle, not by matching the tape): **hall ceiling 2.812 (+0.97 cm), bedroom 2.759 (−3.3 cm)**. One variant gave the bedroom +0.1 cm, which shows the bedroom's floor doubling (~9 cm) makes ±3–4 cm the honest uncertainty there. Final alignment: tilt 0.56°, floor–ceiling angle 1.01°, walls −24.49°, support 60% | Step 1.3 exit check ✅ (floor at z = 0, walls on x/y). G2 not claimed: needs `photo_b` repeatability, and the doubling remains |
| ~17:15 | Tests: synthetic box room (bed top at 0.55 m, half the floor doubled +6 cm, rotated 30°, tilted 3°) → floor 0 ± 2 cm, ceiling 2.70 ± 1.5 cm, walls axis-aligned, cameras end up above the floor, missing ceiling → `partial` + warning. A test bug (wall points spaced by fraction, not metres, so normals were undefined) was fixed. 30 tests pass | — |
| ~17:25 | **Step 1.4 room layout, v1 (ARCHITECTURE design):** high wall band (ceiling −0.65..−0.15 m) → 2 cm wall grid → close gaps → flood-fill enclosed region → rectilinear outline → median wall refine. Hall: a 3.711 × 2.339 rectangle (L-notch missed). **Bedroom fell back to a bounding box** | — |
| ~17:30 | Why v1 failed in the bedroom: in the high band, the boundary at y ≈ 1.78 is a **ceiling beam (bulkhead)**; the wall is behind it at y ≈ 2.35, visible only below the beam (wardrobe niche). The wall around the door was never seen (the camera stood in the doorway), leaving a 0.6–0.9 m corner hole that gap-closing couldn't bridge | — |
| ~17:40 | **v2: free-space carving.** Each observed pixel is a line of sight from its camera that passed through empty space. Drawing all of them from above gives the room's free space, which furniture, beams and door gaps can't hide. First run: right shapes, but the hall region leaked into the bedroom through the bedroom door (hall photos seeing through it), plus doorway stubs and a 0.85 × 0.5 m unseen corner block | D23 |
| ~17:50 | Added: **ownership** (joint frame: each free cell belongs to the room whose own photos saw through it most, which removes the through-door leak), rectangular open 0.6 m (removes doorway stubs), **corner-notch filling** (inward steps < 0.6 m², both edges < 1 m: corner furniture or unseen corners). The hall's real L survives (its notch is an outward 1.6 m² bump) | D23 |
| ~18:00 | Ray-cast synthetic tests (rectangle, L-shape, wall order) caught two real bugs. (1) The confidence filter `conf > 30th percentile` drops **everything** when confidence is uniform → `>=` (here and in `build_cloud`). (2) Lines of sight fan out near far walls → speckled free space that the neck-cut eroded (room 48 cm short). Fixed with a 10 cm gap-fill on free space and an **asymmetric wall search** (10 cm inside, 35 cm outside: carved space stops at or before a wall, never past it) | D23 |
| ~18:10 | **Step 1.4 result** on home01_photo_a: **bedroom 4 walls** 2.926 × 2.416 (tape 2.89–2.94 × 2.39: −0.5..+1.2%, +1.1%), area 7.07 m² (+1.5%), 0.43 m² corner notch filled with a warning. **Hall: L-shape, 6 walls**: W1 3.726 (370: +0.7%), W2 2.326 (230: +1.1%), W6 3.735 (shared wall, 361: +3.5%), notch walls 2.772 / 1.410 / 0.954 (not yet measured by the user). All within ±3.5% (photo tier ±8%). Layout takes 1 s. 34 tests pass | Step 1.4 exit check ✅ |
| ~18:20 | **Step 1.5 output contract.** `scan/schema.py` (Pydantic: every number a `Measurement` value/lo/hi/unit, validator lo ≤ value ≤ hi, `extra=forbid`, stable surface ids, all keys present at every tier) → `uv run scan-schema` writes `schema/scan_output.schema.json` (committed, test-checked against the code) | D24 |
| ~18:25 | First interval model `scan/uncertainty/intervals.py` (ARCHITECTURE §11): σ² = (m·σ_log)² + σ_geom² + σ_fit² (+ extra). σ_fit from each wall's measured spread (doubling) via the two bounding walls; ceiling from the floor/ceiling layer RMS; area = scale term + edge terms; footprint adds scale terms linearly in a joint frame (shared scale). k = 1.5 until calibration → intervals about ±13% (honest-wide; observed errors 1–3%) | D24 |
| ~18:30 | `scan/stitch/plan.py`: joint run = rooms already placed (`joint_reconstruction`); adjacency = parallel walls ≤ 35 cm apart overlapping ≥ 50 cm → **found bedroom1-W2 ↔ hall-W6 (the door wall) automatically**; overlaps via shapely (0); separate frames → `unplaced` + warning (step 1.8). `scan/render/plan.py`: magicplan-style plans (12 cm dark walls, `3.62 m ±0.22` labels, room name/area/ceiling, 1 m scale bar; shared-wall labels inside their own room) | — |
| ~18:35 | **MILESTONE A ✅**: `uv run scan captures/home01_photo_a` → `out/result.json` (validated on write), `out/plan.png`, `out/plan.svg`, `out/rooms/<room>.png`; diagnostics moved to `out/debug/` (cloud.ply, align.png, layout.png, geometry.json). Footprint 17.08 m² ±4.35. Whole run 9 s cached. 40 tests pass | PLAN 1.5 |
| ~18:45 | User marked the hall's north part on the plan: **the kitchen lies beyond plan walls W3/W4** (north of the main hall box, west of the notch); the notch is the passage along the bedroom wall and the kitchen opening is in the notch's west wall (plan W4, ~1.41 m; matches photo 4894). Hall ground truth rewritten as a **6-wall L** in the user's numbering (W1 = bedroom-door wall, clockwise): W1 361, W2 370, W3 230 kept; W4 / W5 / W6 (north segment, kitchen-side notch wall, notch top) `null` until measured; kitchen opening O2 on W5 (size to measure). The old "W4 = 369.5" was measured across the notch mouth → kept as a check `W4 + W6 = 369.5`: plan gives 2.772 + 0.954 = 3.726 (+0.8%) | `data/ground_truth/home01.yaml` |
| ~18:55 | User: the kitchen spans from plan W4 (~1.41 m deep) west to the line of the hall's west wall, so the flat is a full rectangle (hall + passage notch + kitchen in the north-west corner). Added `kitchen` to the ground truth as `captured: false` with **all values null**; plan values appear only as measuring guides in comments. **Ground truth is never copied from the pipeline's plan**: that would grade the pipeline against itself | `home01.yaml` |
| ~19:05 | Kitchen photographed: 3 photos (`IMG_4898`–`4900`), portrait, 1× lens, no duplicates. Thin: the protocol asks for 4–8, and `4900` is a plain wall | `photos/kitchen/` |
| ~19:10 | Added a peak-memory recorder to the backbone step (`PeakMemory`, logged per run). **3-room joint run, 17 photos: 9.14 GB peak, 87 s.** First plan: kitchen 6 odd walls, adjacency hall↔kitchen via hall-W4 (the kitchen-opening wall) ✓, **1 overlap** | E16 |
| ~19:15 | Kitchen walls in two inconsistent sets. Per-photo check of wall-normal angle against the house axes: all bedroom/hall photos 1.5–4.6°; kitchen `4899` 2.1°, `4898` 7.2°, **`4900` 19.0° → misplaced by the backbone** | E16 |
| ~19:20 | Added `check_views`: photos > 8° off the house axes are dropped and the room is rebuilt (never below 2 photos; otherwise all are kept and the room is marked partial). `max_joint_views` 16 → 20 (measured). Result: `4900` dropped, kitchen 4 walls 3.12 m², ceiling 2.52 m (suspicious), **0 overlaps**, adjacency bedroom↔hall and hall↔kitchen | E16 |
| ~19:25 | User: no gap between kitchen and hall (one continuous wall); kitchen W3 (north) is on the same line as hall W5 (passage end); kitchen W2 (west) on the same line as hall W2. So the kitchen is as deep as hall W4. → next: structural wall snapping | — |
| ~19:25 | Work log: open data issues reviewed. 5 resolved items moved to a "Resolved" table with their resolution; kitchen `captured: true` in the ground truth | — |
| ~19:35 | **Structural wall snapping** (`scan/stitch/snap.py`, D25): wall faces linked as one physical wall (shared partition / collinear outer wall), the group line set by the best-seen face, polygons rebuilt. Anchor rule went mean → weighted median → **max support**: the mean and the median let short, poorly-seen faces drag the well-seen bedroom wall (+8.4 / +10.1 cm). The hall-W1 tape check (one data point) informed the choice and the benchmark must confirm it | D25 |
| ~19:40 | Result: kitchen–hall gap closed (kitchen W1 −24 cm), kitchen north on the building's north line (−24 cm), kitchen west on the hall's west line (+5 cm), hall passage end −10 cm. **Hall W1 3.615 (tape 361: +0.1%, was +2.9%)**, hall W2/W3 +0.4/+0.7%, W4+W6 check +0.5%, bedroom +0.2/+1.6%. Ablation (`--no-drift-correction`): footprint 20.30 against 20.06 m², 0 overlaps both | G4 ablation available |
| ~19:45 | User asked whether this is general or tuned to the flat. Answer: general rules (no flat-specific values), thresholds from one home, one tape check informed the anchor rule. Recorded in D25. 43 tests pass (3 synthetic snapping tests added) | — |
| ~19:50 | User: the kitchen is **open** to the passage (no wall on plan W4), and the ground truth should match the plan | — |
| ~19:55 | Wall coverage per edge measured (share of length with vertical-surface points): walls 0.76–1.0; hall-W4 0.42, kitchen-W4 0.27 (the open side); kitchen-W1 0.04 but its partner hall-W3 1.00 (a real wall the kitchen's 2 photos never looked at). Rule: open only if both rooms' faces are low (D26) | D26 |
| ~20:00 | `classify_open_walls` + `merge_open_boundaries` (open faces meet at their midline, no wall thickness), `Wall.kind` in the schema (regenerated), renderer draws open sides as one dashed line without a wall band. Plan: kitchen open to the passage, partition kept. W4 + W6 check 2.68 + 1.04 = 3.72 (+0.7%) | — |
| ~20:05 | Ground truth in line with the plan: hall W5 and kitchen W1 `kind: open` (replacing the opening entries), plan ↔ ground-truth wall-number tables for hall and kitchen, kitchen notes updated (IMG_4900 dropped, ceiling 2.52 to verify). 46 tests pass (3 open-boundary tests) | `home01.yaml` |
| ~20:05 | User: doors (e.g. the bedroom door) come in a later step (1.7) | — |
| ~20:10 | User: use connected rooms to get heights (kitchen ↔ hall ↔ bedroom). Height histograms: the kitchen ceiling peak (2.76) matches hall/bedroom, but **the kitchen floor was never seen** (lowest peaks 0.20 / 0.29 m: plinth or shelves behind the counters) → 2.52 m. Fix: joint-frame rooms use the shared floor (z = 0, fitted from all rooms) when their own floor is > 8 cm off; ceiling mismatches > 10 cm are warned, not overwritten. **Kitchen ceiling 2.52 → 2.741 m** (bedroom 2.769, hall 2.803). 48 tests pass | D27 |
| ~20:20 | User: remove the Claude co-author line from all commits; keep commit messages short. History rewritten with `filter-branch` (20 commits; code identical to backup branch `backup-before-trailer-removal`); commit hashes referenced in docs updated | — |
| ~20:35 | **Step 1.6 eval harness:** `scan/eval/{gt,match,gates,report}.py`, `uv run scan-eval <out> --gt <yaml>` → `eval.md` + `eval.json`. Walls matched by rotation (open/wall kind + measured lengths; shift reported). Ground truth: open walls carry `connects_to` | — |
| ~20:35 | **home01_photo_a scored:** walls **7/7 within ±8%** (mean 0.89%, max 1.63%); W4+W6 check +0.5%; ceilings hall +0.1 cm ✅ G2, bedroom −2.4 cm ❌ G2; bedroom area +1.9%; adjacency equals the tape (bedroom–hall, hall–kitchen); 0 overlaps; G5 pass (footprint n/a until all walls are measured). **Calibration: 100% coverage at ±15% mean half-width** → intervals far too wide (step 1.10). 51 tests pass | Step 1.6 exit check ✅ |
| ~20:45 | **Step 1.7 openings.** E17: face maps show doorway holes under lintels, but see-through points are mostly missing (the confidence filter drops far pixels) → **visibility voting on raw depth** (D28). Votes per wall: bedroom door a clean see-through rectangle; hall W6 two openings; hall's open side red as expected | E17 |
| ~20:55 | Door width 0.72 against tape 0.90. Column profile: votes flip through→wall at u = 0.15 / 0.86 and wall-face points start at 0.88 → the clear opening really is ~0.72. Systematic −18/−19 cm on both doors → likely definition (frame outer vs clear opening). **User asked to re-measure** (inside faces of the frame) | — |
| ~21:00 | Noise filter for windows (≥ 0.5 × 0.5 m, ≥ 2 photos), `connects_to` by the opening's position on the shared wall, **W1 = main-door wall** (D11) → plan numbering = ground-truth numbering in all rooms. Plan draws door gaps + swing arcs (shared door once), windows as double lines. Output: openings with intervals; adjacency `via_opening` | D28 |
| ~21:05 | G1 in `scan-eval`: 0/5 (doors −18 cm, window missed, front door scored as a phantom on W1 while the ground truth says W3). Synthetic ray-cast room with holes: **door 0.90 exact, window 0.90 × 1.20 sill 0.90 exact**. The synthetic room revealed a leak risk (high-confidence through-door rays carve outside) → test uses realistic distance-falling confidence; risk logged. 53 tests pass | Step 1.7 (geometric part) ✅ |
| ~21:10 | User: the front door is on the same wall as the bedroom door → ground truth O4 moved W3 → W1 (matches the pipeline's hall-O2) | `home01.yaml` |
| ~21:20 | **Step 1.9 damage + detector.** E18: OWLv2 (8 queries, per-class thresholds, NMS) on 17 photos in 35 s (cached after). Frame → depth mapping verified (scale + centre crop, 1 px). Lifted boxes: window on bedroom W3 in 4 photos ✓, closed sticker door on hall W3 in 3 ✓, wardrobe as a "door" (1 photo), dozens of stain/crack false positives on marble floors and the kitchen splashback | E18 |
| ~21:35 | Rules: ≥ 2 distinct photos per object, damage on walls/ceilings only, detector doors/windows only where voting found none (after W1), mirrors cancel see-through openings; rules engine + `config/rules.yaml`; damage/flags/scope in the output with intervals; damage drawn on plans; damage scoring in `scan-eval` | D29 |
| ~21:40 | home01: window added (0.69 × 1.06; tape 0.90 × 1.21 → −21 cm), sticker door added (hall W3, 0.93 × 1.93; not yet in the tape file → counted as a phantom), **0 damage** (stain D1 missed: in no photo). G1 0/5. 56 tests pass | Step 1.9 ✅ |
| ~21:55 | **Step 1.10 calibration** (D30): split-conformal k per tier_mode × type, small-sample guards (n < 9 → max × 2; k ≥ 0.5), leave-one-room-out coverage; `scan-calibrate` writes `config/calibration.yaml`; results record the k used. Bug found: per-room measurements were built from the uncalibrated config → fixed | D30 |
| ~22:00 | home01 calibrated: wall/ceiling/area k = 0.5 (floor; n too small), LOO coverage 100%. **Interval mean half-width ±15% → ±5.0%, coverage still 100% (10/10)**. 61 tests pass | Step 1.10 ✅ |
| ~22:10 | **Step 1.8 per-room stitching** (D31): door pairs (width ±15 cm) → 90°-multiple rotation + shift → **visibility score** (through-door depth must land in the other room) → spanning tree → snapping/openings on the rooms sharing the main frame. Config keys first landed under the wrong YAML section (fixed) | D31 |
| ~22:20 | home01 `--per-room`: bedroom ↔ hall matched via the bedroom door (score 0.88, front door rejected), hall rotated 180°; kitchen unplaced (open side, no door); 0 overlaps; walls 7/7 within ±8% but bedroom +5.5–5.7% (per-room scale; joint is +0.2–1.6%) | — |
| ~22:30 | Synthetic stitching test exposed a **carving leak through doorways in per-room runs** (protocol door shot looks straight through; confident through-door depth extends the outline). Test isolated with true outlines: real door beats a same-width decoy, rotation exact, shift within 8 cm. Leak logged as an open risk with a fix direction. 62 tests pass | Step 1.8 ✅ (tree; loop closure + open-side matching not done) |
| ~22:40 | **Step 1.11 determinism.** Cached run twice → `result.json` identical (except timing) but `plan.svg` differed: matplotlib writes the date + random element ids → fixed (no date, fixed hash salt, no PNG software tag) → all output files byte-identical | — |
| ~22:45 | **Live (`--no-cache`) vs cached: bit-identical**, max \|diff\| 0.0 over 210 numbers, same structure. Live 198 s, cached 9–12 s. `scripts/check_determinism.py` + `tests/test_determinism.py` (two cached runs, identical result and files). Outline helper tests added (staircase snap, notch fill vs real L). 65 tests pass | E19 |
| ~22:50 | **TIER 1 COMPLETE** (tag `tier1`). home01_photo_a: walls 7/7 within ±8% (mean 0.89%, max 1.63%), hall ceiling +0.1 cm ✅ / bedroom −2.4 cm ❌ G2, G5 pass (adjacency = tape, 0 overlaps), calibration 100% at ±5.0%, G1 0/5 (door widths −18 cm: tape definition to check; window/closed door from the detector with box extents), damage 0/1 (stain in no photo), G3 n/a (no repeat capture yet) | `tier1` |
| ~23:00 | User added stain photos (`photos/stain/`, 2 shots) and 3 **per-room video clips** (`home01_video_a/video/{bedroom,hall,kitchen}/`, 1080p HEVC portrait, 39.6 / 36.5 / 18.6 s) instead of one walkthrough. Stain photos moved into `bedroom1/` (a `stain` folder would be read as a room). The stain is a real damp patch on the wall behind the bed head | — |
| ~23:05 | **E20:** both stain photos were on the 0.5× ultra-wide lens with digital zoom (16 / 20 mm eq.). Joint run with them: **everything shrank ~4.6%** (walls −3.5 to −5.5%, ceilings −13 / −14 cm, coverage 100% → 30%), plus new phantom openings and 2 marble false-positive stains (2 photos each); stain still not detected. → **Photos not on the 1× lens are excluded** (`ingest.reject_non_main_lens`). Rerun = cache hit, numbers restored exactly | E20 |
| ~23:15 | **Tier 2 step 2.1 video ingest** (`scan/io/video.py`): per-room clips `video/<room>/*.mov` (single walkthrough → clear error: needs room segmentation). ffmpeg decode at 4 fps with rotation applied (158 frames in 4.3 s); keyframes = sharpest frame per time slice (budget 20 ÷ rooms = 6 per room); sharpness varies 1–467, so selection matters | — |
| ~23:20 | **Video focal from vanishing points** (`scan/geometry/vanishing.py`, E12 method as a library): keyframes alone gave 2 valid estimates → use the 20 sharpest frames per clip → 9 plausible estimates, tightly clustered 30.7–32.9 mm, **median 31.9 mm** (photo 26 mm: 16:9 + stabilisation crop). The nominal 28 mm would have been 14% off. Fallback now 32 mm | E21 |
| ~23:30 | **First video run (home01_video_a) is poor:** 18 keyframes joint, bedroom 2.82 m² (tape ~7.0), hall 3.03 (~10), footprint 9.7 (~20), 1 overlap, bedroom not connected; 5 kitchen marble "stains" (neighbouring frames confirm the same false positive). Cause: **coverage** (5–6 narrow-FOV frames per room see one corner; no ceiling found) + poor registration (bedroom cameras placed outside the room). Next: 12–16 keyframes per room in per-room runs + door stitching, or overlapping chunks (ARCHITECTURE §10.3); damage needs frames from distinct viewpoints, not neighbouring video frames | E21 |
| ~20:40 | **E22 keyframes per room** (`scripts/experiments/video_keyframes.py`, config `video.keyframes_per_room`, default 12; mapanything cache hit now takes room labels from the frames so a renamed folder still matches). **video_a:** k6 joint = E21; k8 / k12 per-room + door stitching: bedroom 8.8 m² (tape 6.97, leaks into the hall through the doorway), hall 9.3–9.8 (~10), footprint 22 (~20), ceilings 1.93–2.31 m (tape 2.79). **video_b** (re-filmed, 44.7 / 45.7 / 27.6 s; kitchen only 8 sharp keyframes; focal 32.9 mm from 7 VP frames), k12: bedroom 3.96 m² (−43%): short walls −7.9%, long walls −38%, so the outline is cut short along the long axis; door merged into a 1.98 m opening; ceilings 2.16 / 1.94 / 1.89 m (−63 / −86 cm); rooms not connected (adjacency []); 6 kitchen marble phantom stains. Walls 0/4 within ±3%, mean 23.1% | E22 |
| ~20:45 | **E22 k=20** (video_b): no better (bedroom −35%, hall 12 walls, ceilings 1.91–2.10 m). More frames do not fix it. Contact sheets (`out_k12/debug/<room>_keyframes.jpg`): bedroom keyframes are narrow close-ups of blank white wall and wardrobe doors; ceiling and floor rarely in view. Hall frames look fine yet hall ceiling 1.94 m with a clean L-shape → points to scale/focal, not coverage | E22 |
| ~20:50 | **E22b focal test** (`scripts/experiments/video_focal.py`, hall only, `video.force_f35_mm`): f35 26 / 28.3 / 32.9 mm → ceiling 2.38 / 2.21 / 1.94 m (tape 2.80), ceiling ÷ longest wall 0.850 / 0.787 / 0.680 (tape 0.757 → f ≈ 29.6 mm; VP 32.9 ~10% long). Longest wall stays ~2.8 m (tape 3.70) at every focal. **MapAnything ignores the given video focal**: its own output focal is 21.8 / 22.1 / 22.7 mm and median depth 1.68 / 1.71 / 1.78 m whatever we pass. EXIF rays (≈ 33 mm) then disagree with its poses by ~45% (photos: 12%, E12) | E22b |
| ~20:55 | **E22c model rays for video** (`video.rays: model` overrides `geometry.rays` for the video tier; photos keep EXIF rays). Cache hits, no new inference. video_b k12: ceilings 2.16 / 1.94 → **2.83 / 2.91 m** (tape 2.79 / 2.80), bedroom short walls −7.9 → −3.5%, long walls −38 → −22%, area −43 → −25%. video_a k12: ceilings 2.31 / 2.27 → 2.86 / 2.96 m; bedroom still leaks through the doorway (+37%). Rooms still not connected in either capture | E22c |
| ~20:55 | User re-filmed the bedroom (`home01_video_b/video/bedroom1/IMG_4913.MOV`, 91.8 s; sharpness median 62, 34% of frames < 30). Old `IMG_4909` still in the folder. Step-by-step plan (user): bedroom alone first, then hall and kitchen. Bedroom-only capture `captures/home01_video_b_bedroom1/` (symlink) | — |
| ~21:00 | User: go step by step, bedroom first. New clip moved to its own capture `captures/home01_video_c/video/bedroom1/IMG_4913.MOV` (video_b back to its 3 original clips). Another session was running video_b sweeps in parallel (2 models on 16 GB); stopped on the user's request | — |
| ~21:10 | **E22d bedroom alone (video_c, model rays):** k12 area 5.68 m² (−18.5%), ceiling 2.877 m (+8.5 cm); k20 5.82 m² (−16.5%), 2.924 m (+13.1 cm). Clean single walls and sharp floor/ceiling peaks (filmed tilting up to the ceiling). Still 6 walls: **the door wall (W1) is never in view** (all cameras stand on the door side; keyframe sheet `keyframes_bedroom1.jpg`), so the outline stops where the views stop. Scale alone is ~+3–8% (window wall 2.53–2.59 vs 2.39, ceiling +3–5%), like photo per-room runs (E7/E8). Keyframes 69.0 / 69.2 s were near-duplicates → `video.min_keyframe_gap_s: 1.0` | E22d |
| ~21:20 | User re-filmed the bedroom for the protocol (`home01_video_d/video/bedroom1/IMG_4916.MOV`, 131 s, **60 fps**: sharpness median 208, 7% blurry, best so far; first put in `home01_photo_d`, renamed). k12 / k20: area 7.45 / 7.62 m² (+7 / +9%), ceiling 3.06 / 3.00 m, **8 walls**: frames filmed from the hall through the doorway (0–24 s) carve hall floor into the bedroom (door 0.9 m > neck cut 0.70 m) | E22e |
| ~21:25 | **E22e ceiling rule (negative result).** Per-view split (k12): view 11 looking up at the fan puts 70 k ceiling points at 2.7–2.9 m; window-wall views 2, 4–6 put a second copy at 2.95–3.15 m; the rule "highest peak ≥ 1.8 m above the floor" took the copy. Tried "highest peak holding ≥ 50% of the strongest ceiling-zone peak": photo_a kitchen 2.741 → **2.640 m** (kitchen peaks 2.113 / 2.643 / 2.763 m with strength 584 / 454 / 206: the real ceiling is the weak one above loft and cabinet undersides), trimmed video_d bedroom 2.876 → 2.721 m and long walls −18 → −26%. **Reverted.** The check also caught a crash (`if array` with 2+ peaks) | E22e |
| ~21:30 | User: drop the hall part. Lossless trim (`ffmpeg -ss 26 -to 126 -c copy`, rotation and model tag kept) → `video/bedroom1/IMG_4916_room.MOV` (100 s); original moved to `home01_video_d/original/` (outside `video/`, not read). Untrimmed outputs kept as `out_k12_full`, `out_k20_full`. Trimmed k12: **4 walls**, short walls 2.376 (−0.6%), long walls 2.373 (−18 / −19%), ceiling 2.876 m (+8.4 cm), area 5.64 m² (−19%). The door wall is behind every camera (never filmed); the outline stopped in front of the cameras (cameras at x 0–0.3 outside the polygon at x 0.35) | E22f |
| ~21:40 | **E22f cameras inside the room** (`layout._contain_cameras`, `layout.camera_wall_margin_m: 0.20`, `min_wall_support: 20`): a wall with < 20 supporting points is pushed out until every camera in its span is 0.20 m inside, with a warning (lower bound, weak-wall interval). Synthetic room (west wall never filmed, cameras at x 0.3): west wall 0.515 → 0.10 m (true 0.0). **Trimmed bedroom k12: walls −0.6 / +1.2 / −0.6 / −0.6% (4/4 within ±3%), area 6.95 m² (−0.3%)**, ceiling 2.876 m (+8.4 cm, G2 fail). The long walls come from the camera rule (a bound that landed on the truth here), intervals ±0.46 m. Also committed the other session's `_drop_short_edges` separately (eb9e008); it was in the tree for every run since ~21:00 | E22f |
| ~21:50 | **E22g hall alone** (`home01_video_d/video/hall/IMG_4919.MOV`, 134 s 60 fps, all filmed inside the hall, no trim; run via symlink capture `home01_video_d_hall`). k12 / k20: ceiling 2.842 / 2.851 m (+3.9 / +4.8 cm), area 8.50 / 8.86 m², 10 walls. Wall points show the right shape (L with the passage ~1.0 m wide, bedroom-door wall continuous into the passage) but **every wall ~11% short**: main block 3.25 × 2.05 vs 3.70 × 2.30, bedroom-door wall 3.25 vs 3.61. Heights right, lengths short: consistent with the model's video focal (~22 mm vs ~30 true) compressing depth along the line of sight; the small bedroom is mostly seen side-on and escapes it | E22g |
| ~22:00 | **E22h bedroom + hall together** (video_d). Joint k10 (20 views): bedroom −4.7 / 0.0 / −4.7 / −1.7%, area −5.4%; hall still short (long side −12.6%, short −9.0%, W4+W6 check −3.7%), passage lost (6 walls), ceilings +12 / +9 cm, **1 overlap**, not adjacent. Per-room k12 + door stitching: bedroom 4/4 within ±1.2% (= alone), hall = alone, **bedroom unplaced** (no door pair matched), 2 phantom stains on the passage end. A shared reconstruction does not fix the hall's scale | E22h |
| ~22:15 | **E22i portrait frames turned 90° before the model** (`geometry.rotate_portrait`, outputs turned back: pixel maps rot90, R_wc · M_TURNᵀ, K swapped; round-trip test exact). video_d k12 per-room: model focal unchanged (bedroom 23.7 → 23.3 mm, hall 26.8 → 23.5 mm; given 32.2) but **metric scale collapses** (median depth 2.04 → 1.24 m, 2.16 → 1.14 m): ceilings 2.03 / 2.02 m (−77 cm), bedroom long walls +30%. The model's size prior expects upright rooms. **Negative result; switch kept, off.** Also: the model's video focal differs per room (bedroom 23.7, hall 26.8 mm), so a fixed per-capture focal correction would not fix both | E22i |
| ~22:20 | User added the kitchen (`home01_video_d/video/kitchen/IMG_4920.MOV`, 94.9 s 60 fps, sharpness median 74, 33% < 30; narrow galley, many close blank-wall frames; 34–46 s look out into the hall). Plan (user): join bedroom + hall first, then the kitchen, then one plan from all videos | — |
| ~22:30 | **E22j door joining (bedroom + hall, `home01_video_d_bh` symlinks).** Openings, door matching and damage lifting projected with `K_exif` while video points use the model's rays (`rays: model`): fixed with `cloud.ray_K` (photos unchanged: rays exif). → rooms joined for the first time (adjacency bedroom1–hall, 0 overlaps). The never-filmed bedroom door wall read as one 2.38 m "door" → openings skipped on walls with < `min_wall_support` points (photo_a openings/walls identical; test). k12: bedroom −0.6 / +3.1 / −0.6 / +1.4%, area +1.6%, joined **rotated 90°**: its only detected door is on a long wall (tape: short wall W1), the real door wall was never filmed from inside. k20: some frames glimpse the door wall, the camera rule no longer applies, long walls −25%, not joined | E22j |
| ~22:40 | **E22k all three rooms (video_d).** Per-room k12 + doors: bedroom + hall joined as in E22j, kitchen 6 walls 4.52 m², ceiling 2.883 m, **unplaced** (no door, open side; per-room open-side matching not implemented); 2 phantom stains on the hall passage end (kitchen marble seen through the opening). **Joint, 8 per room, `max_joint_views` 24** (video frames are 16:9: ~25% fewer tokens; peak 9.15 GB): **arrangement right** (bedroom along the hall's east wall, kitchen north-west) but sizes worse: bedroom 2.60 × 2.31 (−10 / −3.5%), hall 3.26 × 2.16, passage lost, kitchen 1.90 × 1.20, ceilings +14 / +24 cm, 1 overlap, 0.4 m kitchen–hall gap. Plan: joint run for placement, per-room runs for shape | E22k |
| ~21:00 | **Zero-length wall crash** (video_b k24, EXIF rays): `_refine` moves each wall line on its own, so the two parallel lines around a short jog met → 0 m wall → NaN direction in openings / damage → negative damage width → `Measurement` validation error. Fix: `_drop_short_edges` after `_refine` (edges < `layout.min_edge_m` dropped, neighbours merged at the better-supported line); photo results unaffected (shortest wall ~1 m). Test added, 71 pass. **E22c sharpness supply** (`scripts/experiments/video_sharpness.py`, `captures/e22c_sharpness.log`): sharp keyframes available at k = 8…40 — video_b bedroom/hall 30–40, video_a 20–30, video_b kitchen caps at ~17 (median sharpness 27). Second session's k16 / k20_model runs died without output while the other session's runs started (two GPU runs on 16 GB); queue `scripts/experiments/e22_sweep.sh` stopped | — |

### Open data issues

- **Hall measurements:** W4 (north segment), W5 (open side to the kitchen), W6 (passage end), sticker door (which wall + size), the window beside it (size + sill).
- **Door widths: re-measure** the clear opening, door open, laser between the **inside faces of the frame** (bedroom door, front door). The pipeline measures 0.72 / 0.70 against tape 0.90 / 0.89, a consistent −18 cm that suggests the tape went to the frame's outer edges (D28).
- **Kitchen measurements:** its 4 walls, ceiling (2 spots) and the opening from the hall.
- **Kitchen photos (retake):** only 3 taken; `IMG_4900` (a plain wall) was placed 19° off the house axes and dropped automatically (E16), so the kitchen rests on 2 photos, and its floor wasn't seen (ceiling now measured from the shared floor, D27: 2.741 m). Retake 4–8: corners with the floor in frame + one shot through the opening into the hall + one from the hall into the kitchen.
- **Bedroom:** W2 (289) and W4 (294) differ by 5 cm. Re-measure, or confirm it's real.
- **Stain D1:** a real damp patch on the wall behind the bed head (user photos 18:04, but taken on the 0.5× lens with zoom → excluded, E20). Re-photograph it **on the 1× lens from 2+ corners**, ideally as part of the bedroom round 2 (`home01_photo_b`).
- **Sticker door (hall W3):** the detector finds it (0.93 × 1.93). Add it to the ground truth with a measured size.
- **Captures still needed:** bedroom round 2 (`home01_photo_b`, repeatability G2/G3).

### Resolved data issues

| Issue | Resolution | When |
|---|---|---|
| Hall W1 (361) and W3 (230) contradict a rectangle | The hall is **L-shaped**: a passage along the bedroom wall and the kitchen in the north-west corner. The reconstruction showed it (E9, step 1.4) and the user confirmed it. Ground truth rewritten as 6 walls; the old 369.5 kept as the check W4 + W6 | 18:45 |
| Kitchen doorway missing from the hall's ground truth | Added as O2 (type opening) on hall W5, size to measure. Sticker door and window moved into the open "Hall measurements" item | 18:45 |
| Kitchen not captured; 3 rooms might exceed the 16-photo joint limit | Kitchen photographed (3 photos). 3-room joint run measured: **17 photos = 9.14 GB peak** (about 0.35 GB per photo) → `max_joint_views` raised to 20 (about 10.2 GB, under the 12 GB ceiling) | 19:10 |
| Duplicate `IMG_4876` in two room folders (AirDrop) | Wrong copy removed. Ingest now rejects a photo found in two rooms (D19) | 13:00 / 15:10 |
| Hall wall count unknown (4 or more?) | 6 walls (L-shape), see the first row | 18:45 |

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
| E12 | Which focal is physically right? (vanishing points) | Oblique views: focal from geometry is 0.94–1.08× EXIF. **EXIF right, MapAnything ~12% long** | Ceiling bias explained → fix-loop candidate (D20) |
| E13 | Pre-compensate by feeding f × 0.893 | Output follows input (median 379 px). Ceilings +1.3% / +3.1%, but per-view focals inconsistent (ceilings disagree > 10 cm) | Not adopted |
| E14 | EXIF rays + pairwise ICP pose re-fit | **Failed:** low overlap, ICP slides along walls, graph disconnected; poses unchanged. Per-room EXIF rays: ceilings +4.2 / −4.0%, walls +12.6 / +5.7 / −3.0 / −4.9% | ICP dropped. Per-room scale ±5% is the real limit |
| E15 | Ray source × per-room vs joint | **Joint + EXIF: ceilings −0.1% / +0.6%, walls 5 of 6 within 1.5%, mean abs 1.75%** (joint + model 3.55%, per-room EXIF 5.7%, per-room model 4.8%) | **D20: joint + EXIF is the default** |
| E16 | 3-room joint run + photo consistency | 17 photos = 9.14 GB (fits). One kitchen photo misplaced 19° → auto-dropped. Kitchen weak (2 photos) | `max_joint_views` 20; `check_views` in the pipeline |
| E17 | Openings: wall face / see-through maps → visibility voting | Bedroom door from both rooms, hall second door, open side; synthetic door/window exact | D28 |
| E18 | OWLv2 on 17 photos, lifted to surfaces | Window (4 photos) + closed sticker door (3) found; marble floor/splashback → many stain/crack false positives; wardrobe as a "door" (1 photo) | D29: ≥ 2 photos, walls/ceilings only |
| E19 | Determinism: cached twice + live vs cached | All files byte-identical (after the SVG date fix); live vs cached max \|diff\| 0.0 over 210 numbers | FR-RUN-05 / NFR-07 met |
| E20 | 2 zoomed ultra-wide photos added to the joint run | Uniform −4.6% scale shift, coverage 30% | Non-1× photos excluded at ingest |
| E21 | First video run: 6 keyframes per room, focal from vanishing points | Focal 31.9 mm (9 frames, tight); reconstruction poor: rooms 40% of true area, bedroom unconnected | More frames per room (per-room runs / chunks) |
| E22 | Keyframes per room (6 / 8 / 12), per-room runs + door stitching, two captures (video_a, video_b) | Not stable across captures: bedroom +27% (video_a, doorway leak) vs −43% (video_b, outline cut short); ceilings 0.5–0.9 m low in every run; rooms never connected in video_b | Per-room video at k ≤ 12 does not work yet |
| E22b | Video focal: hall at f35 26 / 28.3 / 32.9 mm | Shape ratio says ~29.6 mm (VP 32.9). MapAnything outputs ~22 mm whatever focal it is given; depth barely changes | Do not mix EXIF/VP rays with the model's poses for video |
| E22c | `rays: model` for the video tier | Ceilings −63 / −86 cm → +4 / +10 cm (video_b), +7 / +16 cm (video_a). Bedroom long side still −22% (video_b) / doorway leak (video_a) | `video.rays: model` default |
| E22d | Re-filmed bedroom alone (video_c), k12 / k20 | Clean walls, ceiling +8.5 / +13 cm; area −18.5 / −16.5% because the door wall was never filmed; per-room scale +3–8% | Film the door wall; keyframes ≥ 1 s apart |
| E22e | Ceiling = highest *strong* peak (≥ 50% of the strongest) | Fixes a weak doubled copy above the ceiling but drops the real kitchen ceiling (weaker than loft/cabinet undersides): 2.741 → 2.640 m | Reverted; capture trim removed the copy's main source |
| E22f | Trim frames filmed from outside the room + unseen walls placed 0.20 m behind the cameras | Bedroom (video_d, k12) 8 → 4 walls, all within ±1.2%, area −0.3%; ceiling +8.4 cm | `layout.camera_wall_margin_m`; protocol: film each room from inside it |
| E22g | Hall alone, video_d, k12 / k20 | Shape right, all walls ~11% short, ceiling +1.4% | Depth compression along the line of sight (model video focal ~22 mm) |
| E22h | Bedroom + hall: joint k10 vs per-room k12 + doors | Joint: hall still −9 to −13%, bedroom −5%, 1 overlap. Per-room: bedroom ±1.2%, hall −11%, door match failed | Neither stitches yet |
| E22i | Portrait video frames turned to landscape for the model | Focal unchanged (~23 mm), metric scale halves: ceilings −27%, walls ±30% | Keep portrait (`geometry.rotate_portrait: false`) |
| E22j | Door joining bedroom + hall (video_d) with ray-consistent intrinsics, no openings on unseen walls | Joined (first time), 0 overlaps; bedroom rotated 90° because its own door was never filmed from inside | Protocol: film each door from inside its room; one-sided matching (through-door views) as a later fix |
| E22k | Three rooms: per-room k12 + doors vs joint 3 × 8 (24 views) | Per-room: kitchen unplaced. Joint: right arrangement, worse sizes (−3 to −14%), ceilings +14 / +24 cm | Next: per-room shapes placed by the joint run |

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
| Model focal ≠ EXIF focal | MapAnything outputs a focal ~1.12× EXIF, and vanishing points prove EXIF right (E12). Compresses vertical extents, so ceilings come out 3–12% low | E6–E15 | **Fixed:** EXIF rays + joint run (D20) |
| Per-room metric scale | Each separate run guesses absolute size off by about ±5% (bedroom big, hall small) | E7, E8, E14 | Joint run shares one scale (E15). Fallback per-room runs get wider intervals |
| Open doors, built-in wardrobes | An open door leaf and a wardrobe recess confuse the room's extent (bedroom short side +7.5–7.9%) | E15 | Layout step 1.4 must pick the room boundary, not the outermost points |
| Ceiling beams / bulkheads | In a high band the beam face looks like the wall (bedroom: beam at 1.78, wall at 2.35) | Step 1.4 v1 | Free-space carving uses lines of sight at all heights (D23) |
| Lines of sight through doorways | The hall's free space leaked into the bedroom | Step 1.4 v2 | Ownership in the joint frame (D23); neck cut for per-room runs |
| Corner furniture / unseen corners | A 0.43 m² corner block made the bedroom 6-sided | Step 1.4 v2 | Inward corner notches < 0.6 m² filled, with a warning (D23) |
| Ray density near far walls | Speckled free space eroded by morphology (synthetic room 48 cm short) | Step 1.4 tests | 10 cm gap-fill + asymmetric (outward) wall search (D23) |
| **View-to-view doubling** | Views disagree by 5–8 cm on where a surface is (bedroom floor copies 9 cm apart, right wall 15 cm). Directly limits ceiling accuracy (bedroom ±3–4 cm) and walls | E7, E15, step 1.3 | Consensus planes (D22) take the median of all copies. **Real fix still open**: likely the fix-loop target (G2 / G3) |
| Input resolution to backbone | 1024 px ingest copies instead of full-resolution photos: bedroom short side 2.450 → 2.390 m (+2.5% → 0.0%) | E11 vs step 1.2 | Keep 1024 px |
| Sparse far walls | Heuristic misses walls seen only from far away, or picks things outside the room (hall long side 1.47 / 6.69 m) | E9, E10 | Layout must use per-room polygons, not global percentile peaks |
| File handling | AirDrop renamed a duplicate `IMG_4876 2.HEIC` into the wrong folder. Capture folders vanished from disk (13:3x) | E4 | Hash-check on ingest. Keep originals on the phone |
| Video frames vs model focal | MapAnything predicts its own focal for video frames (~22 mm) and ignores the given one (VP ~32 mm). Re-projecting with the given focal compresses heights 25–30% and breaks view consistency (slanted, doubled walls) | E22b, E22c | Video uses the model's rays (`video.rays: model`) |
| Video framing | Portrait 1× video close to walls: narrow (37° horizontal) views of blank white walls; ceiling rarely in view. Registration fails and more frames (8 → 20) do not help | E21, E22 | Protocol: film from corners towards the opposite corner, tilt up to the ceiling and down to the floor |
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
  - Code at commit `e143866`, plus a `measure_cloud.py` change to take tape values as arguments.
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
  - 14 photos (7 + 7). Code at commit `bfb1a8c`, plus the runner and analysis changes.
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

### E12. Which focal is right? Vanishing points as an independent referee

- **Question:** MapAnything outputs a focal ~1.12× the EXIF focal we give it (E6–E11). Which one matches the real camera?
- **Setup:**
  1. Check our input path: run `preprocess_inputs` on a 1024 px ingest frame and print its intrinsics.
  2. `scripts/experiments/vanishing_focal.py <photos_dir> <E9 ma_raw.npz> <out>`:
     - LSD line segments (OpenCV).
     - Drop near-vertical lines.
     - Two RANSAC passes find the two horizontal vanishing points v1, v2 (length-weighted support, 1.5° tolerance, least-squares refine).
     - Focal from perpendicular directions: **f² = −(v1 − c)·(v2 − c)**.
     - Compared per photo with EXIF and with MapAnything's output (E9).
  - Outputs: `captures/home01_photo_a/experiments/E12_vanishing_focal/<room>/*_vp.jpg`, showing the lines assigned to v1 (red) and v2 (blue).
- **Results:**
  - **Preprocessing is correct.** 769.1 px at 768×1024 becomes 392.6 px at 392×518, matching EXIF (390.3 px from the formula).
  - Vanishing points (focal ratio to EXIF), on views with two clear wall directions:

    | Photo | VP / EXIF | Model / EXIF |
    |---|---|---|
    | hall 4890 | 1.029 | 1.147 |
    | hall 4891 | 1.017 | 1.158 |
    | hall 4892 | 0.937 | 1.088 |
    | hall 4897 | 1.080 | 1.134 |
    | bedroom 4882 | 1.014 | 1.160 |
    | bedroom 4884 | 0.947 | 1.148 |

    Hall median 1.03 over 7 photos.
  - **Method failures:** bedroom 4881 / 4888 / 4889 gave 1.7–3.2 and 4883 / 4885 gave no answer. These are head-on views of a wall: one vanishing point is near infinity, so the f² product is ill-conditioned, and the "second VP" was clutter (bedsheets, bag).
- **What affected it:** only oblique views (two-point perspective) carry focal information. Clutter lines create false vanishing points.
- **Impact:**
  - **The EXIF focal is right to about ±5%. MapAnything's output focal is biased about 12% long.**
  - That bias compresses vertical extents in each camera, so ceilings come out 6–10% low.
  - Rebuilding with EXIF rays is physically correct (as E10's ceilings showed), but needs the poses re-fitted.

### E13. Pre-compensate the model's focal bias

- **Question:** if MapAnything's output focal tracks its input, does feeding f_EXIF × 0.893 (1 / 1.12) make it output the right focal with consistent depth and poses?
- **Setup:**
  - `scripts/experiments/mapanything_run.py <bedroom1>,<hall> <out> k IMG_4887.HEIC --f-scale 0.8929` (new `--f-scale` option).
  - Measured with `joint_rooms.py`.
  - Outputs: `captures/home01_photo_a/experiments/E13_joint_fscale0.89/`.
- **Results:**
  - Fed 348.5 px; output focal median 379.3 px (329–427). Gain about 1.09 (was 1.12 at 390 in).
  - Ceilings: bedroom 2.830 (+1.3%), hall 2.890 (+3.1%), joint 2.830 (were −6% / −8% in E9).
  - Walls: bedroom short 1.910 (−20%, wardrobe front picked), long 2.810 (−3.6%). Hall short 2.230 (−3.0%); long side is a heuristic failure.
  - **Top-down: room interiors filled with ceiling points inside the upper wall band.** Ceiling heights disagree by more than 10 cm between views.
- **What affected it:** the output focal follows the input only on average. The per-view focal spread (329–427) gives each view a different vertical scale.
- **Impact:** not adopted. It fixes the average ceiling but not consistency. The planned fix is EXIF rays + ICP pose re-fit (fix B), parked for the fix loop (D20).

### E14. EXIF rays + ICP pose re-fit (fix attempt B): negative result

- **Question:**
  - Can pairwise ICP re-align views built with EXIF rays, removing E10's wall doubling?
  - And so give correct ceilings with consistent walls?
- **Setup:**
  - `scripts/experiments/icp_refine.py`:
    - Per view: EXIF-ray cloud in the camera frame, 2 cm voxel, normals.
    - Every pair: coarse-to-fine point-to-plane ICP with a Tukey loss (40 → 15 → 5 cm matching radius), starting from MapAnything's relative pose.
    - Edges where fitness ≥ 0.30, then Open3D pose-graph global optimisation.
  - `scripts/experiments/icp_eval.py captures/home01_photo_a/out <plots>` compares model / exif / exif_icp on the cached step 1.2 runs.
  - Metrics: inter-view residual (median distance from each view's points to the nearest point of the other views, over points within 20 cm) and size against tape.
- **Results:**
  - Pose graph **not connected**: bedroom 5 / 21 pairs, hall 10 / 21 pass fitness 0.30. Open3D refused to optimise, so the poses were unchanged (shift 0.0 m). Deterministic on rerun.
  - Pair diagnostics: overlap at 5 cm mostly 0.00–0.30. ICP corrections of 1–2 m / 40–150° on low-overlap pairs, up to 169 m (hall 2-5). Even good pairs ask for 10–50 cm.

  | Room | Mode | Residual | Short | Long | Ceiling |
  |---|---|---|---|---|---|
  | bedroom | model | 2.9 cm | +2.1% | +0.9% | −3.3% |
  | bedroom | exif | 3.3 cm | +12.6% | +5.7% | +4.2% |
  | hall | model | 2.9 cm | −5.7% | −4.9% | −11.9% |
  | hall | exif | 2.2 cm | −3.0% | −4.9% | −4.0% |

- **What affected it:** corner photos of flat walls overlap little, and point-to-plane ICP on planar scenes can slide. The per-room metric scale differs between rooms (bedroom about +5%, hall about −4% with EXIF rays).
- **Impact:**
  - ICP dropped; the code is kept only as an experiment.
  - Insight: the ceiling error has **two parts**: the focal bias (fixed by EXIF rays) and the per-room scale (about ±5%, needs a shared scale or an anchor). That led to E15.

### E15. Ray source × per-room vs joint run

- **Question:** does a joint run (one shared scale) plus EXIF rays fix both parts of the ceiling error, without doubled walls?
- **Setup:**
  - `uv run scan captures/home01_photo_a --joint --out captures/home01_photo_a/out_joint` (fills the cache).
  - `scripts/experiments/rays_eval.py captures/home01_photo_a/out captures/home01_photo_a/out_joint --plots captures/home01_photo_a/experiments/E15_rays_joint`: residual + spike measurement per room, for model and EXIF rays.
- **Results:**

  | Run | Room | Rays | Residual | Short | Long | Ceiling |
  |---|---|---|---|---|---|---|
  | per-room | bedroom1 | model | 2.9 cm | +2.1% | +0.9% | −3.3% |
  | per-room | bedroom1 | exif | 3.3 cm | +12.6% | +5.7% | +4.2% |
  | per-room | hall | model | 2.9 cm | −5.7% | −4.9% | −11.9% |
  | per-room | hall | exif | 2.2 cm | −3.0% | −4.9% | −4.0% |
  | joint | bedroom1 | model | 2.9 cm | −1.3% | −2.9% | −6.2% |
  | joint | bedroom1 | **exif** | **2.6 cm** | +7.9% | **−1.5%** | **−0.1%** |
  | joint | hall | model | 4.8 cm | −0.4% | −2.2% | −8.3% |
  | joint | hall | **exif** | **3.8 cm** | **+0.4%** | **−0.0%** | **+0.6%** |

  Mean absolute error over the 6 numbers:
  - joint + EXIF **1.75%**
  - joint + model 3.55%
  - per-room + model 4.8%
  - per-room + EXIF 5.7%

  Bedroom (joint + EXIF) plot: left wall −0.23, wall behind the wardrobe 2.33 (2.56 m against tape 2.39). The line at x ≈ 1.78 near the door is the **open door leaf**; the wardrobe front is about 1.75.
- **What affected it:**
  - The joint run shares one metric scale (it cancels the opposite per-room scale errors of E14).
  - EXIF rays remove the focal bias.
  - The open door leaf and the wardrobe recess affect the bedroom's short side.
- **Impact:** **new default (D20): `joint: auto` + `rays: exif`.** Ceiling G2 is within reach: bedroom −0.3 cm (pass), hall +1.7 cm (0.2 cm over), pending the real layout fit (step 1.4).

### E16. Three rooms in one joint run; detecting misplaced photos

- **Question:**
  - Does a 3-room joint run (17 photos) fit in memory?
  - Does the kitchen join the plan correctly?
- **Setup:**
  - `uv run scan captures/home01_photo_a --joint` with a new `PeakMemory` sampler (accelerator memory every 50 ms) around inference.
  - Per-photo check: in the aligned frame, the median angle between each photo's wall normals and the nearest house axis.
- **Results:**
  - Memory: 9.14 GB peak, 87 s inference (14 photos was 8.1 GB, so about 0.35 GB per photo).
  - First plan: kitchen 6 walls, 4.63 m²; hall↔kitchen adjacency through hall-W4 (the kitchen-opening wall, correct); 1 overlap (kitchen edge 10 cm into the passage).
  - Off-axis angle per photo: bedroom 1.5–4.3°, hall 2.3–4.6°, kitchen 4899 2.1°, 4898 7.2°, **4900 19.0°**.
  - After dropping photos > 8°: kitchen from 2 photos, 4 walls, 3.12 m², ceiling 2.52 m; 0 overlaps.
- **What affected it:** a plain, low-texture wall photo (`4900`) gave the backbone too little to place it, so it came out rotated about 19°. With only 3 photos in the room, nothing outvoted it.
- **Impact:**
  - `max_joint_views` = 20 (measured).
  - `check_views` drops misplaced photos automatically (this protects the walk-in).
  - Kitchen needs a retake (4–8 photos).
  - The ceiling of 2.52 m needs verifying.

## Open questions / next experiments

- **Carving leak through doorways (per-room runs, D31):** remove carved regions reached only through a detected doorway; test with the synthetic two-room house (`tests/unit/test_stitch_doors.py` scene) using carved, not true, outlines.
- **Per-room stitching gaps:** least-squares loop closure; matching open sides (an open kitchen has no door).

- **Repeatability:** bedroom1 again (`home01_photo_b`). Walls must agree within 1 cm (G3). The 8 cm doubled walls from E7 are the main risk.
- **4 vs 8 photos per room:** what is the minimum that still passes?
- **Output focal vs input focal** in MapAnything (E6): is the input being used as a soft hint only?
- **MapAnything load time:** 33–44 s. Investigate caching or keeping the model loaded across rooms.
- **Hall:** not a rectangle (W1 361 vs W3 230). This tests layout beyond 4 walls.
- **Stitching via joint reconstruction (after E9):** does it hold with more rooms, and with rooms whose doorway views are weak? Compare against door matching + least squares on G5.
- **Joint runs for larger homes:** a 5-room capture (30–40 photos) exceeds `max_joint_views`. Options: a room plus its doorway neighbours, overlapping groups merged by shared views, or lower resolution. Measure memory first.
- **Fix loop:** pick the worst gate from the benchmark (Phase 4). The ceiling focal bias is fixed (D20). Prime candidate now: **view-to-view doubling** (5–8 cm surface disagreement between views) limiting G2 (bedroom ceiling −3.3 cm) and G3.
- **Ceiling G2 (≤ 1.5 cm):** current error 8–25 cm. Probably the hardest gate for the photo tier. Report honestly with intervals if it can't be met.
- **Mixed orientation within a room:** `IMG_4887` was excluded in E7; the pipeline must handle it.
- **E7 OWLv2 detections:** review, then start tuning O4.
