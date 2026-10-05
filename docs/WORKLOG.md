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
| ~22:55 | **E22l diagnosis (user: "why is the video plan so different?").** `scripts/experiments/joint_overview.py` (both runs in one frame, same orientation): (1) everything ~11–12% small in the joint video run; (2) hall passage lost: from the hall the passage mouth reads as a 0.94 m "window" (furniture hides its lower part), and 5 of 6 kitchen frames were filmed standing in the passage; (3) kitchen walls slanted (poor registration), 1.90 m instead of ~2.68, 0.4 m off the hall; (4) bedroom: `show_opening.py` shows its "door" (0.95 m tall, 1 frame) is the gap behind the open door leaf; the real door wall is seen in one frame only (68 s); (5) ceilings high | E22l |
| ~23:05 | **E22l link frames** (user: find the frames that connect the rooms). `video.link_pairs`: SIFT on 1 fps candidates, per frame a FLANN shortlist over the other room's frames, then pairwise ratio test + RANSAC fundamental matrix (first version used one ratio test over all frames: neighbouring frames hold the same points, so almost everything failed). video_d: bedroom↔hall 46 pairs (bedroom 89 s looking out the door ↔ hall 11 s; bedroom 37 s ↔ hall 106–113 s looking in), hall↔kitchen 28 (fridge/TV corner), bedroom↔kitchen 0 (they only see each other's door from opposite sides). `links.jpg`. Joint 3 × 7 + 4 links (29 views, 9.17 GB): with link views used for layout the kitchen spills into the hall (3 overlaps, false bedroom–kitchen adjacency, bedroom −13%); with link views only in the model run (`Frame.link`) bedroom −15 / −32%, hall smaller, only hall–kitchen adjacent. ~7 frames per room is too little coverage for a joint run; links are correct but cannot fix it | E22l |
| ~21:00 | **Zero-length wall crash** (video_b k24, EXIF rays): `_refine` moves each wall line on its own, so the two parallel lines around a short jog met → 0 m wall → NaN direction in openings / damage → negative damage width → `Measurement` validation error. Fix: `_drop_short_edges` after `_refine` (edges < `layout.min_edge_m` dropped, neighbours merged at the better-supported line); photo results unaffected (shortest wall ~1 m). Test added, 71 pass. **E22c sharpness supply** (`scripts/experiments/video_sharpness.py`, `captures/e22c_sharpness.log`): sharp keyframes available at k = 8…40 — video_b bedroom/hall 30–40, video_a 20–30, video_b kitchen caps at ~17 (median sharpness 27). Second session's k16 / k20_model runs died without output while the other session's runs started (two GPU runs on 16 GB); queue `scripts/experiments/e22_sweep.sh` stopped | — |
| ~23:45 | **E22m depth focal fix** (`cloud.depth_focal_fix`, `geometry.depth_focal_fix` / `depth_focal_scale`; other session): per view depth × f_true / f_model, points rebuilt with the true focal. video_d_bh joint k12 (28 views): scale 1.0 / 0.92 / 0.85 (32.2 / 29.6 / 27.4 mm) → bedroom 9.50 / 8.03 / 7.10 m² (tape 6.97), hall 14.04 / 11.64 / 9.84. Per-view ratio spread p10–p90 0.97–1.48: views stretched by different amounts while poses stay in the squeezed world; carving keeps the farthest wall → overshoot. QuickTime metadata `camera.focal_length.35mm_equivalent` = 27 mm (`scan/io/quicktime.py`); fed as the video focal: bedroom 5.94 (fix off) / 7.88 (fix on) | E22m |
| 00:10 (10-05) | **E22n self-calibration** (`scripts/experiments/video_selfcal.py`): focal where E = KᵀFK has two equal singular values, over 48–102 frame pairs per clip with parallax → **hall 32.25, bedroom 32.0, kitchen 31.75 mm** (vanishing points 31.9). QuickTime's 27 mm is the lens' nominal focal, not the 16:9 + stabilisation-cropped frame (16:9 crop alone: 26 → 28.3 mm). True video focal ≈ 32 mm | E22n |
| 00:20 | **E22n link placement** (`scan/stitch/links.py`): per-room runs keep their link frames (`RoomCloud.run_views`); SIFT matches between the two link frames → depth in each room's run → 3D–3D pairs → RANSAC rotation + shift in x/y, rotation snapped to 90°. video_d k12 per-room, 32 mm: **all 3 rooms placed, arrangement right** (bedroom east of the hall, kitchen north-west) without any door detection. bedroom↔hall 49/101 points, rms 9 cm, dz +2 cm, scale hall/bedroom 0.95; hall↔kitchen 77/79, rms 10 cm, dz −6 cm, scale kitchen/hall 0.89. Shapes: bedroom long walls −9 / −11% (extra link views changed its run; E22j ±1.2%), hall 10 walls, kitchen 6 walls, 2 overlaps. Bug on the way: `stitch()` reused `via` (opening id) → method None → crash; renamed, test added | E22n |
| 00:30 | **E22n model pass 2 with depth input** (`geometry.depth_focal_fix: refit`, `mapanything_backend.refit_with_depth`): stretched depth + 32 mm intrinsics fed back to MapAnything. It follows the depth (×1.17–1.23) but moves its focal only part way (bedroom 24.0 → 27.2, hall 28.3 → 29.9 mm) → long depth × short rays: ceilings +55 / +72 cm, bedroom +122%. **Negative** | E22n |
| 00:50 | **E22n frames cropped to 3:4 before the model** (`geometry.max_aspect: 1.333`): model focal falls to 16–19 mm (in full-frame terms; uncropped 24–28), ceilings 2.30 / none / 2.35 m. The model assumes a roughly fixed field of view whatever it is shown, it does not measure it. **Negative** | E22n |
| 01:10 | **E22n plan diagnosis** (user: bedroom W2 should line up with the kitchen's north wall; extra walls inside the hall). `scripts/experiments/plan_diagnosis.py` (wall points coloured by max height, cameras, outer wall lines), `wall_evidence.py` (wall pixels tinted in the frames). (1) bedroom-W2 y +1.905 vs kitchen-W6 +1.574 vs hall passage end +1.471: the hall is ~10% short (east side 3.25 vs 3.61 m) so its north end and the kitchen hung off it fall 0.33–0.43 m south of the bedroom's (correct) north wall; snapping joins collinear walls only within 0.25 m. (2) hall east bump (W4–W7, 0.96 × 0.63 m, W5/W6 zero points) = free space carved through the open bedroom door (0.9 m > 0.70 m neck cut); 0.27 m step W3/W7 = the one straight east wall reconstructed at two positions (north part seen only at grazing angles by the door jamb); passage west side closed through wall-end piers although it is open to the kitchen, and the kitchen outline swallows the passage (kitchen frames look out into it) | E22n |
| ~07:00 | User: old chat stopped, continue here; asked whether the order of filming walls matters (no: keyframes = sharpest frame per equal time slice, the model takes a room's frames as one set; what matters is time per area, facing walls, every wall from inside, doorways, and orientation: protocol says landscape, clips are portrait). **Video focal source reverted to vanishing points** (QuickTime 27 mm reported only, as the lens' nominal focal) | — |
| ~07:15 | **E22n padded canvas** (`geometry.pad_scale: 1.25`: frame centred on a black canvas, outputs cut back to the frame, `content_box` tested against MapAnything's preprocessing). The model's focal moves to the truth for the first time: median 31.3 / 34.0 / 31.1 mm (bedroom / hall / kitchen; unpadded 24–28). Sizes overshoot instead: bedroom bbox 3.51 × 2.68 (tape 2.89 × 2.39), hall north-south 3.84 (3.61), ceilings +13 / +20 / +20 cm; hall main block 3.71 × 2.25 (3.70 × 2.30), kitchen 2.68. Content is smaller on the canvas → the model's size prior scales the scene up. Lead, not adopted (the pad that balances both would be fitted to this flat) | E22n |
| ~07:35 | **E22n doorway leaks.** (a) `stitch.relayout_after_placement` (default on): after link placement the rooms share a frame, layout again with ownership (a cell goes to the room whose own frames saw through it most). video_d k12: hall 10 → **4 walls** (3.57 × 2.08, tape main block 3.70 × 2.30), east bump gone, **overlaps 2 → 0**; the passage goes to the kitchen (its frames were filmed standing in it: kitchen + passage strip 3.33 × 1.15); bedroom door side −0.14 m (2.62 → 2.51: hall frames saw that strip through the door more than the bedroom's own, whose door wall was never filmed from inside). (b) `layout.lintel_min_z_m` (default off): wall points above door height cut the free space (the lintel closes a door, an open side stays open). Synthetic two-room house: leak 0.3+ → < 0.05 m², areas right (test). Real: hall bump gone but the **passage is cut off too** (a beam across the passage mouth looks like a lintel) and the bedroom shrinks 0.45 m. Giving barrier cells back to the room re-opened the synthetic leak; reverted | E22n |
| ~08:30 | User shared generic advice (use the real iPhone K downstream; scale from a known distance). Checked against our runs: real K × model depth = E22b (ceilings 1.94 m), ratio-corrected depth = E22m (overshoot). New direction taken from it: stop using the model's camera poses too | — |
| ~08:45 | **E22o camera re-solve, v1** (`scan/geometry/repose.py`, `video.repose`): depth × f_true / f_model, PnP with the true K between keyframe pairs (SIFT at working resolution), rotation averaging, robust position least squares, model offsets × measured stretch as weak prior. Probe on the cached hall run: PnP baselines **1.30×** the model's (11 pairs > 0.3 m), rotations within 0.2–7°. Real video_d: baseline stretch 1.31 / 1.25 / 1.16 (bedroom / hall / kitchen), but sizes overshoot (hall 3.67–3.96 × 2.48, bedroom 3.95 × 2.61, ceilings +5–9 cm; output folder later reused by v2). Per-frame check: the depth scale each frame's matches imply does **not** follow f_true / f_model (correlation −0.45 hall, −0.67 bedroom, model rays): the per-frame focal correction behind E22m and v1 is wrong | E22o |
| ~09:05 | **E22o v2**: with the TRUE rays a frame's model depth is the true shape up to one scale per frame (exact if the model kept image-plane sizes, E22m's assumption, and still the right parametrisation if not). Per pair 3D–3D similarity (RANSAC Umeyama; works without baseline) → per-frame log scale, rotations, positions by least squares; absolute scale per room = ceiling height of the model's own reconstruction. Synthetic (frames squeezed 0.80–0.90, spacing 0.77): cameras within 3 cm, stretch 1.297 (true 1.30), test. Real video_d k12: hall **L-shape with passage, 6 walls like the tape** (west 2.29 vs 2.30, passage end 1.02 vs 1.04, east 3.72 vs 3.61, south 3.27 vs 3.70), ceiling 2.82; bedroom 3.18 × 2.68 (+8 to +12%), ceiling 3.17; kitchen 4.58 × 1.44. Pair residuals median 2–4 cm. The anchor is the weak step: pipeline ceiling ≠ anchor ceiling (bedroom 3.17 vs 2.92) because per-frame scale noise spreads the ceiling layer and the highest-peak rule picks the top. Off by default | E22o |
| ~09:40 | User: fix the size anchor, **derived, not trial and error**. Derivation: geometry (true K + matches) fixes each frame's shape, pose and RELATIVE scale; the room's overall scale k is unobservable from geometry. The only metric source in a plain video is the model's size prior, and a size prior fixes Z/f (H = Z h / f), not Z: so frame i votes for scale f_true / f_model,i; with the relative scales every frame votes for k; median, spread = the room's scale uncertainty (→ `Scale.sigma_log`). Ceiling becomes a check, not an input. Synthetic: votes agree exactly (test). Real, run once: bedroom 13 votes, spread ±5.7%, walls +10 / +15%, ceiling 3.10; hall 16 votes ±21%, walls −6…+6% with the tape's 6-wall layout, ceiling 2.97; kitchen ±17%, ceiling 2.79. **Reading:** the bedroom's frames agree with each other on a scale ~12% too big: the model's size prior itself is biased on these frames, and nothing inside a plain video can see that (no IMU/LiDAR). Remaining principled levers: one scale for the whole flat (link frames make rooms' relative scales geometric), and an external reference length; the prior's systematic error to be measured across captures (calibration, D30), not tuned per capture | E22o |
| ~10:30 | **E22p one scale for the flat** (`scan/stitch/scale.py`, `stitch.flat_scale`, runs only with `video.repose`). Derivation: link frames see the same points from two rooms, so the rooms' size ratio is measured (3D similarity of the shared points, RANSAC, tolerance ∝ camera distance; the old diagnostic used rigid-fit inliers, biased to 1); per-room corrections satisfy every ratio (weighted least squares), one level per linked group from the frames' votes. Tests: ratios hold exactly, unlinked room keeps its votes, a room measured 10% small reads s = 1/0.9. Real runs, each once: (1) level = median over all frames → hall ×1.106, kitchen ×0.794, spread 0.23 (`out_k12_32_True_flat_v1`). (2) Hypothesis "link frames unreached by pairs carry a default scale" → rule added (ratios only from frames tied in by pairs); **falsified**: unreached frames were bedroom 86.5 s, kitchen 80.75 / 93.75 s, not link frames; result identical. (3) Diagnosis: per-frame scales smooth within rooms (hall 1.07–1.19, kitchen 1.22–1.44, link frames inside), bedroom–hall ratio 1.079 agrees with the ceiling ratio 1.10 and the tape (bedroom ~+12%), hall–kitchen ratio 0.718 (75/79 inliers): the kitchen's frames vote ~40% off the others. Inconsistency found in the derivation: frames of one room share its bias, so the ROOM is the unit of evidence → level = median over rooms of room-median votes (`out_k12_32_True_flat_framevotes` = run before this change). (4) Final: bedroom ×1.00 (middle room sets the level), hall ×1.079, kitchen ×0.775, spread ±24%; link ratios 1.00 after; bedroom +10 / +17%, hall 3.83 × 4.05 (L lost), kitchen 0.79 × 2.50. As derived: rooms consistent with each other, absolute size = the model's size sense of the level-setting room (+12% here). Absolute size needs a reference from outside the video | E22p |
| ~11:00 | User chose option 2: calibrate the model's size error across taped captures (works on any data, not fitted to one flat). **E22q** (D32): `calibrate.size_rows / fit_size_bias`, `scan-calibrate --bias-only` (merges into `config/calibration.yaml`), pipeline applies `scale_bias` for re-solved video (`uncertainty.apply_scale_bias`), interval mode `video_per_room_repose`. Calibration runs (production settings: VP focal, repose on, flat scale off, bias off), k12: video_a / b / c / d. Bug found on the way: the 3D-similarity refit could lose all inliers (near-collinear matches) → NaN → `SVD did not converge`; refit now keeps the last supported model (test). Per room: a bedroom ceiling −3%, walls −8 / −61% (outline lost, not scale); a hall ceiling +16% (7/12 frames tied in); b bedroom / hall ceilings +7 / +6% (walls not scorable); c bedroom walls −1…−5%, ceiling +16% (spurious ceiling layer); d bedroom walls +9…+15%, ceiling +9%; d hall walls −20 / −7 / +5%, ceiling +6%. Rule (from the median's robustness): a room enters with ≥ 3 measurements. Fit: **×1.017, spread 0.062 (log) from 4 rooms / 2 physical**. Leave-one-physical-room-out: −7.6 → −6.6%, −2.3 → −1.2%, +9.4 → +10.6%, −1.1 → +1.2%. **No systematic bias to remove; the error is per room, ±6% (1σ), content-dependent.** The calibration's value is the honest interval, not a correction | E22q |
| ~11:30 | **New capture `home01_video_e`** (user re-filmed per the protocol): one clip per room, **landscape** 1920×1080 60 fps (all earlier clips portrait): bedroom1 `IMG_4924.MOV` 124 s, hall `IMG_4925.MOV` 160 s, kitchen `IMG_4927.MOV` 75 s. Sharpness median 162 / 190 / 115 (video_d 189 / 159 / 74), 40 sharp keyframes available per room (`captures/home01_video_e/sharpness.log`). Runs: A default, B repose + flat scale + size calibration, C repose for calibration | — |
| ~11:20 | **Ground truth:** hall W6 (passage width) = 89 cm, laser. Derived, each marked `DERIVED` with its formula in the YAML: hall W5 = W1 − W3 = 131 (right-angled L), hall W4 = 369.5 − W6 = 280.5, kitchen W2 = hall W4 (shared partition), kitchen W4 = W2 (rectangle). Plan guide had W6 ~1.04 / W4 ~2.68 (pipeline off 15 / 12.5 cm). Still to measure: kitchen W3 (→ W1), kitchen ceiling ×2, hall sticker door + window | `data/ground_truth/home01.yaml` |
| ~11:40 | **Ground truth:** kitchen W1 (open side, kitchen face) = 117 cm, laser; kitchen W3 = W1 (rectangle, DERIVED). Check: hall W5 131 − 117 = 14 cm = partition thickness (half-brick + plaster), consistent. Plan guide 1.18 m. Still to measure: kitchen ceiling ×2 | `data/ground_truth/home01.yaml` |
| ~11:45 | **Ground truth:** kitchen ceiling 279 cm, laser, one spot. Pipeline (photo, shared floor D27) 2.741 m → −4.9 cm, fails G2 (≤ 1.5 cm). All walls of all 3 rooms now filled; open: hall sticker door + window (optional, openings only) | `data/ground_truth/home01.yaml` |
| ~11:50 | **New capture `home01_photo_b`** (repeat of all 3 rooms, 1× lens, 11:35–11:41): bedroom1 12 files (one AirDrop duplicate `IMG_4938 2`), hall 11, kitchen 4; bedroom + hall **landscape**, kitchen + hall `IMG_4939` portrait; bedroom photos digital zoom 1.035×. Ingest kept the sharpest 8 per room → 20 views, one joint run (237 s, peak 12.8 GB). **Result: everything ~12.5% small** (bedroom walls −9.4…−11.1%, ceilings 2.435 / 2.451 vs 2.79 / 2.80, coverage 17% of 12 intervals = confident garbage); `photo_a` re-run on current code + new GT: walls 13/14 within ±8%, coverage 95%. **Cause (from the cache, `K_model / K_exif` per view):** MapAnything reads the focal 1.13× EXIF on portrait (`photo_a`, median) and 0.96× on landscape (`photo_b`, 0.95–0.97); size ratio b/a 0.866 ≈ focal ratio 0.850: the model's metric depth follows its own focal reading (E22o: a size prior fixes Z/f). Scale ÷ (f_model/f_exif): a 0.895, b 0.911. Second bug: kitchen portrait photos in a landscape joint batch → f_model/f_exif 0.75 (≈ 392/518), ceiling not found: minority-orientation views are cropped to the batch shape. Zoom 1.035× not applied to K (3.5%, would enlarge, not the cause). Protocol says landscape, so the walk-in path is the failing one | `captures/home01_photo_b/out/eval.md` |
| ~12:02 | **Fix loop: declaration** (`docs/FIX_DECLARATION.md`), tagged `fix-before` before any pipeline change. Gate: photo walls ±8% + calibrated intervals on `photo_b` (2/8, coverage 2/10). Fix: one global scale vote k = median over rooms of median log(f_exif / f_model) + calibrated level (fit_size_bias, photo_joint). Prediction from `scripts/experiments/orientation_scale_predict.py` (k a 0.910, b 1.044; c ×1.073, sigma_log 0.030): `photo_b` 6/8 walls, ≥ 8/10 covered; kitchen short sides ~+9% stay failing (crop defect, separate fix) | `docs/FIX_DECLARATION.md` |
| ~12:17 | **Fix loop: shipped** (`a7692f6`, tag `fix-after`). Vote + level applied once to the raw run (k × c before alignment; a first version applied the level after alignment and lost the bedroom's shared floor). Calibration CLI bug found: a normal `scan-calibrate` also refitted the size bias on runs already carrying it (factor → 1.006); now `--bias-only` only. Level ×1.0821 (sigma_log 0.038). **photo_b:** ceilings −35.8 / −35.1 → −4.1 / −2.9 cm, coverage 2/12 → 8/8, but walls 2/4 scored: at true scale the bedroom's doubled north wall (~0.7 m apart) leaves a 0.6 m² NW step = `notch_max_area_m2`, kept → 6-wall L (unscorable); kitchen short sides +11.9%. **photo_a:** 12/14 walls, 19/19 covered, ceilings −6.6 / −4.2 / −9.8 cm. Repeat ceilings a vs b: 33.4 → 2.5 cm (bedroom), 35.2 → 1.4 cm (hall). Gate fail → fail; scale predictions held, wall count did not. Threshold not touched (tuning against the tape) | `reports/fix_loop.md` |
| ~16:30 | **Parked, not in the code** (`scripts/experiments/parked_zoom_pad.patch`): (1) digital zoom in K (vanishing points: zoomed photos 1.06 × EXIF vs unzoomed 1.015, and the 35 mm tag reads 26 at 1.035×, so it leaves the zoom out); (2) a portrait room in a landscape joint run padded instead of centre-cropped (content mask on the padding, detector cache key). Recalibrated with both: level ×1.034, spread 0.088 (was 0.038); `photo_b` mean wall error 8 → 15%, `photo_a` 2.85 → 6.7%. The joint model output itself moved (vote-only: bedroom −6.2 → +2.7%, kitchen −0.7 → +11.6%), more than the K change explains. Ablation (zoom only / padding only) not run: scoring items still uncovered come first | — |
| ~17:06 | **Benchmark** (`scan-bench`, `config/bench.yaml` → `reports/benchmark.md`; hall area stated in the ground truth as an L: 9.68 m², footprint 19.93 m²). photo_a: 12/14 walls ±8%, mean 2.9%, coverage 100%, footprint −2.7%, G5 pass; photo_b: 2/4 scorable (bedroom 6 walls, hall 8), footprint +2.8%; video_d / video_e: 4/8 and 0/4 within ±3%, coverage 79 / 80%, footprint −15 / −18%. G1 0% everywhere; G2 photo biased low (−3…−10 cm), video unrepeatable. G3 photo 0/16, video 1/11. G4 on/off: −2.7/−1.4% (a), +2.8/+5.6% (b), −18.3/−14.7% (video_e). **Head-to-head vs magicplan 2026.38.0** (iPhone 13, camera scan; `data/head_to_head/magicplan.yaml`): photo_a 7/8 (88%), headline photo_b 4/8 (50%), video_e 3/8. magicplan: bedroom ceiling 2.14 (tape 2.79), area 4.83 (6.97), hall drawn as a rectangle. **Lens check** now relative to the capture's own lens (modal focal, ±10%), not a fixed 22–30 mm range: a Pro iPhone with 1× set to 35 mm would have lost every photo. **LiDAR reader** (`scan/io/lidar.py`, Stray Scanner; synthetic box rooms `scan/synth/stray.py`): points land on the walls within 3 mm, extents within 1 cm (tests); pipeline wiring next. README, COMPLIANCE_MATRIX, DEVICE_MATRIX written | `reports/benchmark.md` |
| ~11:55 | **E22r landscape capture (video_e), k12.** A default: bedroom 8.46 m² (6 walls, ceiling 2.55), hall 5.52 (ceiling 2.85), kitchen 2.88 (ceiling 2.01); B repose + flat scale + calibration: bedroom 12.61 (10 walls), hall 11.0, ceilings 3.19 / none / 3.66; C (repose, calibration mode) bedroom 15.41. Diagnosis: (1) **the model's focal on landscape 16:9 frames is far worse than on portrait**: median 14.6 / 18.8 / 20.7 mm (bedroom / hall / kitchen, VP 33.1) vs portrait video_d 24.0 / 28.3 / 25.0 mm (VP 31.9): it assumes ~100° across a 59° view, depth squeezed to about half; the re-solve cannot recover a squeeze that large (the per-frame shape assumption breaks). Frames upright (checked). (2) Links found are correct (`E22r_check/sheet.jpg`) but **the kitchen clip's first ~22 s are filmed from the passage / hall** (0–10 s entrance, 15–21 s looking back into the bedroom door, `kitchen_timeline.jpg`): hall footage labelled kitchen. Not added to the size calibration: orientation changes the model's error, so landscape is a different population | E22r |
| ~12:20 | User: make the landscape footage work (no re-film). **E22s model-FOV probe** (`scripts/experiments/model_fov_probe.py`, target = the known true focal, no tape): same 12 keyframes per room shown as-is / centre-cropped to 4:3 / padded to 4:3 (black bands, all content, K shifted). Model focal median (p10–p90), true 33.1 mm: landscape hall 20.3 (15–26) / 24.8 / **29.4 (29.1–29.6)**; bedroom 14.6 (14–27) / 13.1 / **28.2 (26.5–28.8)**; kitchen 21.4 (16–29) / 21.8 / **30.2 (29.5–31.3)**; portrait video_d hall (true 31.9) 26.6 (23–32) / 18.1 / **32.7 (25–34)**. Padding to 4:3 brings the model's focal to 85–100% of the truth and removes most per-frame scatter; cropping does not (it changes what is in view). Reading: trained mostly on 4:3 images, the model misreads a 16:9 frame's field of view; on a 4:3 canvas every content pixel keeps its true ray. `geometry.pad_to_aspect` / `video.pad_to_aspect` (outputs cut back to the frame; test against MapAnything's preprocessing, both orientations). User also shared generic floor-plan advice (camera trajectory, K under resize, wall-line fitting, Manhattan snapping, corners from intersections): all already in the pipeline | E22s |
| ~10:42 | **Incident (E22s):** first full pad-to-4:3 run crashed in floor fitting (empty cloud). Cause: preprocessing crops the 4:3 canvas slightly, so the content box started left of the image (negative index) and the slice wrapped to 1 px wide. The test compared box numbers, not the sliced array. Fix: `content_box` clips to the model's output size; the test now checks the box lies inside the image and the slice has that shape. The 4 broken cache entries were deleted (the cache key doesn't cover code). Rerun on video_e (landscape) and video_d (portrait) | E22s |
| ~10:55 | **E22s full runs, pad to 4:3, no repose** (k12, `video.pad_to_aspect=1.3333`). Model focal / true: landscape hall 0.88, kitchen 0.91, p10–p90 within ±0.03 (as-is 0.57 / 0.62, p10–p90 0.45–0.87). Depth grows in step (×1.46 / ×1.55): the focal–depth coupling. **Landscape video_e:** bedroom now scores (as-is: 0 walls matched): 4/4 walls, −3.4 / +6.5 / −3.4 / +4.7%, area +2.0% (as-is +21%); all 3 rooms placed by links; hall a 3.27×2.14 rectangle (L-shape lost), 2 overlaps. Ceiling: bedroom 1.93 m (tape 2.79), hall and kitchen none. Cause is geometry, not the model: a 16:9 landscape frame at the true 33 mm sees ±18° vertically, so with the lens ~1.4 m below the ceiling a level camera sees the ceiling only beyond ~4.3 m, more than these rooms. Points above 2.4 m are 1–3% of the cloud. The highest horizontal peak with ≥15% of the biggest is then a furniture top (bedroom), or nothing. The as-is ceilings came from the model's too-wide view bending the top rows upward. Missing ceiling does not change wall evidence (`wall_points` falls back to the 98th percentile height). **Portrait video_d:** bedroom −0.3 / −4.2 / −0.3 / −5.8%, area −5.3%, ceiling +23 cm; hall 6 walls but wrong shape (W1 −72%, W4+W6 +27%); mean 14.2% over 8 walls vs 11.2% over 7 for the repose run (E22q settings). Reading: padding fixes the model's view of the frame (focal, bedroom shape in landscape), not the hall shape. Next: pad + repose on both | E22s |
| ~11:05 | **E22s pad + repose** (E22q settings: repose on, flat scale off, bias off; + pad 4:3). Repose's frame-to-frame scales ≈ 1, so its room scale k = median f_true/f_model: landscape 1.19 / 1.14 / 1.10 (bedroom / hall / kitchen), portrait 1.37 / 0.99 / 1.11. Bedroom walls vs tape: landscape −12.8 / +12.6 / −12.8 / +10.6% (area −2.6%; one axis squashed, the other stretched, worse than pad alone, ±5%), portrait +12.9 / +21.0 / +12.9 / +18.9% (area +35%; repose without pad +9…+15%). Hall: 4 walls in both (no L). Reading: on padded frames the model's own geometry (its depth with its own rays) is the closest to tape in both orientations, bedroom ≤6.5% per wall. Re-solving cameras on the true rays and scaling by f_true/f_model overshoots: that vote assumes the model's angular size (Z/f) is right, and these runs say it isn't where the model's focal differs from the truth. So the two corrections don't stack; pad without repose is the better video setting on this evidence (2 captures, 1 flat: not yet a default). Hall shape still wrong in every variant | E22s |
| ~11:30 | **E22t hall diagnosis** (video_e, pad 4:3; `plan_diagnosis.py`, `wall_evidence.py`, new `passage_ownership.py`; outputs `captures/home01_video_e/experiments/E22t_hall/`). Plan ↔ tape: hall south = TV wall (GT W2 3.70), east = bedroom door + front door (GT W1), west = sticker door + curtained window (GT W3 2.30), north = sofa partition (GT W4). (1) Bedroom north wall vs kitchen north wall: y 1.820 vs 1.841, collinear within 2 cm (user's first issue, fixed in this run). (2) The 'extra boundary' in the hall is the hall's north line drawn across the passage mouth: the hall outline stops at the partition line (y 0.50) over its full width, only 0.70 m detected as open (passage ~1.04). The passage went to the kitchen's outline (kitchen 3.82 wide = kitchen 2.68 + passage 1.04). Hypothesis 'the kitchen clip's first 22 s, filmed from the hall, claim the passage' FALSIFIED: views seeing ≥20% of the passage: hall 1 of 11 (0.31 views/cell), kitchen 6 of 11, of which 4 stand inside the kitchen looking east through it (83–98%). Ownership (most views) follows the footage correctly. Kitchen ↔ passage has no wall, so the split is a naming convention (tape: passage = hall), not geometry; trimming or parameters cannot change it. (3) Hall own size: E-W 3.27 vs 3.70 (−12%), N-S 2.14 vs 2.30 (−7%), while the kitchen spans the same E-W 3.82 vs 3.72 (+3%) and the hall–kitchen link puts both rooms on one scale (0.995, measured at 2.2–2.9 m). So the hall's west wall (3.3 m from the east cameras) is placed short: a range-dependent depth error inside the hall run, not a room-scale error. West lines disagree by 0.55 m, beyond what the snap joins. (4) Bedroom west (door) wall: 4 support points, never filmed from inside the bedroom; its position comes from free space only, hence the ~0.3 m overlap with the hall | E22t |
| ~12:00 | User: no re-film, the video holds what's needed; try three ways in order (1 coverage audit, 2 coverage-based keyframes, 3 floor-line measurement), move on when one fails. **E22u step 1** (`scripts/experiments/coverage_audit.py`, `captures/home01_video_e/experiments/E22u_coverage/`). Every decoded frame (4 fps: 498 / 641 / 300) is localised in the stitched plan frame by PnP (SIFT to the room run's views, their 3D points, true K). Tracked 45 / 56 / 49% (plain walls have no features). Check on keyframes, PnP vs model run: position median 2 cm, heading 0.1–0.2°. Implausible poses (outside the plan + 0.5 m, or not 0.3–2.5 m up) dropped: 2 / 4 / 3. Visibility: wall samples (10 cm, at 0.3 / 0.9 / 1.5 m) and passage floor cells projected, occluded by the stitched cloud's depth buffer; ≥30% of the samples = 'sees'. Frames seeing it, all tracked / keyframes: passage from the hall clip 22 / 1, from the kitchen clip 18 / 3; hall north line (W1) 8 / 0; bedroom door wall (W2) from the bedroom clip 28 / 1; bedroom south (W1) 20 / 0; hall west (W4) 70 / 5. Bedroom clip headings only east / west (none north or south). Verdict: the video holds the views the time-sliced keyframes miss; over all frames the hall clip saw the passage more than the kitchen clip did (keyframes flipped it, so ownership went to the kitchen). → step 2. Hook: `video.extra_keyframes` (frame names added to the time slices, `io.video.extra_picks`, test) | E22u |
| ~12:00 | **E22u step 2, frame choice** (same script; per frame also the 10 cm plan cells of the surfaces it sees in front, and sharpness). Greedy: start from what the run's views see; add the tracked frame with the most unseen cells (sharpness breaks ties), ≥ min_keyframe_gap_s from every chosen frame, until the run holds geometry.max_joint_views (20, memory). A tracked frame has ≥25 PnP inliers to the run, so it overlaps it (no overlap threshold). The keyframes already see 86 / 82 / 81% of the cells all tracked frames see (bedroom 1029 of 1203, hall 1292 of 1570, kitchen 997 of 1225). Picks add +4…+55 cells each: bedroom 7 (mostly from the east end looking west at the door wall), hall 6 (t075 sees the hall north line and the bedroom side), kitchen 6 (t068.5 sees the passage; t005.5 and t017 stand in the hall). Note: the objective counts a cell once, while ownership counts views per cell, so the passage (already seen by 1 hall keyframe) earned the hall no extra frame. Run: `video_keyframes.py … video.extra_keyframes=[…] --tag=pad43_cov` | E22u |
| ~12:15 | **E22u step 2 result** (`out_k12_pad43_cov`, 20 views per room). Ground truth now has the kitchen (commit 074f32f: 1.17 × 2.805, ceiling 2.79; passage W6 0.89). Bedroom −4.6 / +11.3 / −4.6 / +9.4% (pad alone −3.4 / +6.5%), area +5.3%; hall still 4 walls, a rectangle; kitchen still includes the passage (3.78 vs kitchen 2.805, kitchen + passage 3.695); kitchen short side +2.7%; hall ceiling now found (2.64 m, extra frames looked up). Coverage frames do not fix the hall or the passage → step 3 | E22u |
| ~12:20 | **E22v step 3 premise** (`scripts/experiments/floor_range.py`): on floor pixels of each run view, model depth vs geometric depth (true-focal ray meets z = 0 from the camera height), median per distance bin. Hall 0.948 / 0.943 / 0.943 / 0.964 / 0.987 / 0.994 over 1–4 m; kitchen 0.94–0.96; bedroom 0.95 → 0.88 at 3 m. No range compression in the hall (the ~5% offset is the model's focal, 0.89 of the truth, bending its rays). FALSIFIED: the hall's far wall is not short from range. The hall run is uniformly ~10% small: walls 0.88 / 0.93 of tape, and the hall run sees an east-wall door (bedroom or front door, both 2.10 m) 1.90 m tall (0.90), while the bedroom run sees its door 2.10 m (1.00); kitchen 1.02 / 1.03. A uniform per-room scale error is invisible from inside the room: the floor line takes its scale from the same camera height. Cross-room ratios disagree: links say kitchen = bedroom × 1.07 and hall = kitchen × 1.005, the door says hall = bedroom × 0.905, a 17% loop error. The hall–kitchen link's kitchen-side frames (7 s, 16 s) are kitchen-clip frames standing in the hall | E22v |
| ~12:35 | **E22w option A: cross-room size ratios from all keyframe pairs** (`scripts/experiments/cross_ratios.py`, `captures/home01_video_e/experiments/E22w_cross/`). Every own keyframe of room A × every own keyframe of room B (SIFT + F-RANSAC), 3D in both runs, RANSAC similarity A ~ s B (tolerance 5% of range). Shared points are all far: 4.4–6.4 m from the cameras (seen through doorways). Bedroom ~ kitchen pooled 0.842 (22 frame pairs, 0.77–0.91); hall ~ kitchen pooled 1.069, two clusters by range: 0.95 / 0.98 at 2.7–2.8 m, 1.11–1.20 at 4.5–6.2 m; bedroom ~ hall: one pair at 6.4 m, 0.70 (pooled 0.63). Loop s(b,h) s(h,k) / s(b,k) = 0.80. FAILED: the runs disagree in a range-dependent way beyond ~4 m, and every point two rooms share is that far, so shared points can't put the rooms on one scale here (also explains the link-scale contradictions in E22v) | E22w |
| ~12:45 | **E22x option C: passage ownership from every tracked frame** (`scripts/experiments/ownership_all_frames.py` on E22u's per-frame camera + seen cells; same carving and argmax rule as the layout). Passage owned, keyframes only: hall 23% / kitchen 77%; all tracked frames: hall 63% / kitchen 37% (hall proper 94% hall, kitchen proper 92% kitchen, both ways). But the flip is clip length: hall 358 tracked frames vs kitchen 145. Per clip frame the kitchen sees the passage more (0.22 vs 0.13 frames/cell), and the kitchen filmer stood in the passage for 44 tracked frames (28–40 s), the hall filmer for 1. NOT ADOPTED: no counting rule here has a physical reason to prefer one; the passage is open to both and its room is a naming convention (tape: hall). Picking the rule that matches the tape would be fitting to the answer. What is physical: no wall points → no wall line; the false boundary across the passage mouth should be drawn open whoever owns the passage | E22x |
| ~12:55 | **E22y: is there a false wall on the hall's north line?** (`scripts/experiments/wall_profile.py`, wall points per 10 cm along a plan wall). hall-W1 (pad run): points over 0–2.5 m (the partition), none over 2.5–3.2 m, then the corner; the openings module's opening hall-O1 spans 2.48–3.18 m, exactly the empty stretch. No invented wall in this run: walls are drawn only where wall points are (classify_open_walls + openings already do it). Partition / mouth in hall units 2.48 / 0.70 m → ÷0.90 (hall scale, E22v) 2.76 / 0.78 vs tape 2.805 / 0.89. The 'extra boundary' the user saw came from earlier runs (portrait / repose) | E22y |
| ~13:10 | **E22z: door height as each room's size anchor** (user OK with general knowledge). Check first (`scripts/experiments/door_top.py`, open / wall votes per 5 cm row over the opening's width): hall door on the east wall (bedroom or front door) see-through to ~1.93 m, then wall votes 35 per row from 1.98 to 2.18 m. The lintel was seen, so 1.90–1.95 m is measured: 0.90–0.93 of 2.10, confirming the hall scale (E22v). The passage mouths (hall-O1, kitchen-O1) have no votes above them (top = where the view ended); the bedroom's 'door' (0.80 × 0.75) has no votes at all (junk). New `scan/stitch/door_scale.py`: floor-reaching openings whose lintel was seen (≥2 voted rows in the 0.20 m above the top, every one mostly wall); room factor = prior / median height, prior `stitch.door_height_prior_m` (null = off; 2.05 ± 0.05: doors 2.0–2.1 m, not this flat's tape), combined with the room's own size spread in log space (inverse variance), then the room is re-laid out before placement. Tests: lintel seen vs view ended vs single row; weighting | E22z |
| ~13:30 | **E22z result** (`out_k12_pad43_door`, `stitch.door_height_prior_m=2.05`). First run: no room got a door, because the per-row lintel rule rejected the row straddling the top (open 11 / wall 5); fixed to a band-wide wall share (test). Second run: the bedroom's 1.60 m 'door' passed and scaled the bedroom ×1.164 (walls +13 / +7%); added the standard gate: a height more than 3σ (room and door spreads combined) from the prior is not a door (test). Final: hall door 1.90 m → ×1.051; hall 3.41 × 2.29 vs tape 3.70 × 2.30 (−7.8 / −0.7%, was −12 / −7%); bedroom unchanged (−3.4 / +6.5 / −3.4 / +4.7%, area +2.0%); kitchen short side +4.0%, long side 3.82 = kitchen + passage (3.695, +3.4%; scored against the kitchen alone, 2.805: the passage-label convention, E22x). Kept off by default: one door in one room of one capture is the evidence so far | E22z |
| ~13:35 | **Session handoff (E22s–E22z).** Best video_e (landscape) plan: `video.pad_to_aspect=1.3333 stitch.door_height_prior_m=2.05` → `captures/home01_video_e/out_k12_pad43_door/plan.png`. Works: pad to 4:3 (model focal 0.88–0.91 of the truth, bedroom right); rooms placed by links; bedroom/kitchen north walls collinear (2 cm); walls drawn only where wall points are (no false stub, E22y); door anchor corrects the hall partly. Falsified / not adopted: pad + repose (E22s), coverage-chosen extra keyframes (E22u: the video has the views, but they don't fix hall or passage), range compression (E22v: model vs floor geometry flat to 4 m), cross-room ratios from shared points (E22w: all shared points 4.4–6.4 m away, loop 0.80), all-frame ownership (E22x: clip-length effect). Open: hall still −8% E-W; passage label is a convention (tape: hall), needs a declared rule; landscape never sees the ceiling (tilt up a few seconds per room, or film portrait); bedroom door wall barely filmed. Not committed. The full suite has 1 failure, `test_home01_layout_against_tape` (photo tier, bedroom −4.7 / −6.3%), from the other session's uncommitted photo-scale work, not this session's video-only / off-by-default changes | handoff |
| ~14:30 | **E23a: user's plan review (video_e)** — bedroom overlaps the hall, its door on the wrong wall, a window on kitchen W3 that doesn't exist. Evidence (`opening_evidence.py`, `detector_boxes.py`, `E23a_openings/`): (1) the bedroom clip starts in the doorway, the layout keeps every camera inside its room, so the barely-filmed west wall (4 points) was pushed 0.3 m into the hall; snap_walls only joined opposite faces with a gap, never crossed ones. (2) The drawn bedroom door was a 0.75 m see-through gap on the south wall; the real door was seen only from the hall (hall-O2 0.72 × 2.0) and never put on the bedroom's face. (3) The W3 'window' is the sink window's REFLECTION in the glossy marble tiles of the next wall: detector boxes at the corner, same height (1.12–2.08 vs 1.14–2.08), solid tile depth. Then a detector 'door' appeared: a white wall panel beside the doorway (scores 0.34 / 0.26). Fixes, each with a test: snap treats crossed opposite faces as one partition, and an unsupported face may be off by snap_tolerance + camera_wall_margin_m (how far the layout pushes an unseen wall); see-through 'doors' whose top was seen below `openings.min_door_height_m` (1.6, the detector's existing bound, now one setting) are dropped; a door is copied onto a partition face too poorly seen to vote; window pairs at a shared corner at the same height = window + reflection: the one without open cells goes, and when depth can't tell (neither seen through: grilles and glass read as wall under the video's model rays, 46 / 70 noise votes, 0 open cells) the dimmer detection goes (a reflection loses light: scores 0.45 vs 0.365, real higher in all 4 photos); a detector door whose partition was well seen from the other room with no opening there is dropped (front doors have no filmed room behind them). Result (`out_k12_pad43_door_e23a`): overlaps 2 → 0; bedroom one door on the hall wall (0.63, clipped at the bedroom's corner; tape 0.90) + its window; kitchen only the sink window. Bedroom now 2.77 × 2.31 (−4.0 / −5.7 / −3.4%): its west wall took the hall's partition line, so the hall's ~8% short scale shows. Left: kitchen water-stain phantoms on marble (E18), ceilings in landscape | E23a |
| ~15:00 | **E23b: one scale for all rooms (joint run for size).** What was missing (E22s–E23a): every room's size comes from its own model run, ~6–10% apart; all after-the-fact repairs failed or were partial (E22v–E22z). `scan/stitch/joint_scale.py` (`stitch.joint_scale`, off by default): after alignment, one extra model run over the rooms' link frames plus evenly spaced keyframes of every room, up to geometry.max_joint_views (20); for each room, per shared frame the median log ratio of the two depth maps over pixels valid in both (same frame, same pixels, every range: not the far shared points of E22w); room factor = median over frames, spread/√n added to the room's size spread; rooms scaled before layout. With the joint scale, the door anchor pools every room's measured doors into one flat-wide factor (rooms stay on one scale). Tests: frame choice, factor. Runs: joint alone, joint + door 2.05 | E23b |
| ~14:45 | **E23b result.** Joint run (30 views, all rooms): per-room factors bedroom ×1.016, hall ×0.998, kitchen ×0.935 (per-frame spread ±2–4%). The joint run keeps the hall as small next to the bedroom as the per-room runs do: the hall's ~10% is NOT a per-run scale error but how the model sees the hall in this video (portrait joint run E22k too: hall 3.26 × 2.16). Joint only: bedroom −5.2 / −9.5%, hall 3.22 × 2.18, kitchen short −2.9%, mean 11.4%. Joint + pooled doors (bedroom 1.65 + hall 1.90 → ×1.052, first try lost the door flag to zsh not splitting `$args`): bedroom 2.39 × 2.76, hall 3.39 × 2.29, mean 10.4%. Not the better route; kept off by default | E23b |
| ~15:00 | **E23c: declared same wall line** (user: kitchen + passage span the hall's width; tape 2.805 + 0.89 = 3.695 vs 3.70). `captures/<capture>/hints.yaml` `same_line: [[kitchen, hall]]` = layout facts the filmer knows, not measurements; snap_walls joins those rooms' end-to-end outer walls on the same side up to `stitch.same_line_max_offset_m` (1.0) at the MEAN of the rooms' estimates (each is off by its room's size error; wall points don't say which room's size is right, so best-seen is not the rule here). The 0.25 m automatic tolerance is unchanged (a 0.4 m jog can be real). Test. Result (pad + door 2.05 + hint, `out_k12_pad43_door_e23c`): hall W3 = kitchen W3 = 3.52 (was 3.41 / 3.82; tape 3.70, −4.9%), hall depth 2.29 (−0.6%), bedroom 2.31 × 2.87 (−3.4 / −0.7 / −3.4 / −2.4%), kitchen short +4%, overlaps 0, mean 8.6% (the kitchen long wall's +25% is the passage-label convention). Best plan so far. User asked to use the hall's own W3 for both: that gives 3.41 (−8%); the mean is closer. A known taped length could anchor the flat's scale if the user enters one (that wall then leaves scoring) | E23c |
| ~15:20 | **E23c fix: the declared join runs last.** First version moved the kitchen's west wall 0.3 m during snapping, before openings; the kitchen's points stayed where they were, so the real wall read as 'seen through' behind the moved line (> 0.15 m margin): phantom windows (kitchen W4, a resized sink window) and the bedroom's detector door came back (it now had an 'opening' behind it). Now: `snap.join_declared` after openings, detector and damage, which are measured on walls where the rooms' points are, and `surface_anchors` / `reanchor` carry each opening and wall-damage to the moved walls by plan position (test). Result: hall W3 = kitchen W3 = 3.615 (mean of 3.41 / 3.82; tape 3.70, −2.3%), hall depth 2.29 (−0.6%), bedroom 2.31 × 2.77 (−3.4 / −4.0 / −5.7%), kitchen short +4%; openings: bedroom 1 door (hall wall) + window, hall passage + door, kitchen passage + sink window; overlaps 0. Left: kitchen marble 'water stains' (E18), ceilings in landscape | E23c |
| ~15:40 | **E23d: door width.** User: the drawn door size is wrong (bedroom door drawn 0.63 m, tape 0.90 jamb to jamb, TEMPLATE: inside of the frame). Two causes. (1) Copying the hall's door onto the bedroom's face CUT it at the bedroom's corner (the rooms disagree by 9 cm on where that corner is): now the copy keeps its width and shifts inside the wall; the width is a local measurement, its place along the wall carries the placement error (test). Bedroom door 0.63 → 0.72. (2) The hall's own votes put the doorway at 0.72–0.73 m: see-through from u 0.35 to 1.08 m with solid wall on both sides (sharp edges, not a jamb/oblique-view effect). The hall's geometry is short there, like its door height (1.90 vs 2.10): left for the new-approach test (E24) | E23d |

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
| E22l | Link frames (rooms seeing the same thing) added to a 3-room joint run | Links found and correct (46 / 28 / 0 pairs); joint 3 × 7 still too little coverage per room: −13 to −32% | Next: joint run for placement only, per-room runs for shape |
| E22m | Per-view depth × f_true / f_model (joint, 3 focal values; QuickTime 27 mm) | Overshoots at 32 / 29.6 mm (+36 / +15% bedroom area); 27 mm closest by chance; per-view spread 0.97–1.48 | Not adopted: poses stay in the squeezed world |
| E22n | Focal self-calibration; link placement; model pass 2 with depth; 3:4 crop | Focal 32 mm (3 clips, = VP); links place all 3 rooms right (rms ~10 cm); pass 2 and crop make it worse | Links adopted (`stitch.link_placement`); model focal cannot be steered; remaining errors = hall −10% scale, doorway leak, wall doubling |
| E22o | Camera re-solve with the true focal (PnP v1; 3D-3D per-frame scales v2; frame votes for the room scale) | Model squeezes camera spacing 1.16–1.31×; per-frame focal correction is wrong (corr −0.45/−0.67); v2: frames consistent (2–4 cm), hall L-shape recovered; room size from the model's size sense | `geometry.repose`, off by default |
| E22p | One scale for the flat (link size ratios + room votes) | Rooms agree with each other (ratios 1.00 after); the level inherits one room's size error (+12% here) | `stitch.flat_scale` (with repose) |
| E22q | Size bias calibrated across taped captures (D32) | ×1.017 ± 6.2% from 4 rooms / 2 physical; LOO: no gain | Video size error is per-room random ±6%, not a bias |

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
| Model focal on video frames | MapAnything assumes a roughly fixed field of view: own focal 24–28 mm vs 32 true (self-cal + VP), per view 21–33 mm; ignores given intrinsics, follows given depth but not given rays, and a 3:4 crop drops it to 16–19 mm. Lengths along the line of sight squeeze, heights stay right: hall −10% | E22b, E22m, E22n | Placement from link frames; scale still open (room-level fit to shared / collinear walls) |
| Video orientation | MapAnything's focal guess: portrait 9:16 ~75–85% of the true focal, landscape 16:9 ~45–60% (same rooms, phone, focal). Landscape squeezes depth about twice as much | E22i, E22r | Film video in portrait for this model; calibrate per orientation |
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

### E22n. Video: focal truth, rooms placed by link frames, and why the plan still disagrees

- **Question:**
  - What is the true focal of the video frames (vanishing points 32 mm vs QuickTime 27 mm)?
  - Can the rooms be joined from what both videos see (link frames), without detecting doors?
  - Can the model be made to use the true focal (depth input, crop)?
- **Setup:**
  - `scripts/experiments/video_selfcal.py` on the three home01_video_d clips (4 fps, frame pairs 0.5–1.5 s apart, SIFT + RANSAC F, pairs a homography explains skipped, focal swept 18–42 mm).
  - `scan/stitch/links.py` + `RoomCloud.run_views` + `Capture.links`; `stitch.link_*` in the config; placement tree in `stitch/doors.place_rooms` takes link edges first, doors second, and repeats passes until no room is added.
  - `video_keyframes.py captures/home01_video_d 12 video.force_f35_mm=32` (→ `out_k12_32`), `+ geometry.depth_focal_fix=refit` (→ `out_k12_32_refit`), `+ geometry.max_aspect=1.333` (→ `out_k12_32_1.333`).
  - Diagnosis: `plan_diagnosis.py`, `wall_evidence.py` → `captures/home01_video_d/experiments/E22n_diagnosis/`.
- **Results:**
  - Self-calibration: hall 32.25, bedroom 32.0, kitchen 31.75 mm (per-pair p25–p75 ~29.5–34.5). QuickTime 27 mm = nominal lens focal.
  - Link placement: 3/3 rooms placed, rotations snapped from −1.5° / +1.9°, rms 9 / 10 cm, height offsets +2 / −6 cm. Relative scale from the links: hall 0.95 × bedroom, kitchen 0.89 × hall.
  - Pass 2 with depth: model focal 24.0 → 27.2 (bedroom), 28.3 → 29.9 mm (hall); ceilings +55 / +72 cm.
  - 3:4 crop: model focal 16–19 mm; ceilings −45 to −50 cm or not found.
  - Plan vs the user's check: bedroom north wall 0.33 m north of the kitchen's; the hall's east bump, 0.27 m step and closed passage side are not real.
- **What affected it:**
  - The model's focal prior. Each room is squeezed by its own amount (hall ~10%); placement is right where the rooms meet, so the error shows at the far ends (north line).
  - Free-space carving through an open 0.9 m door (wider than the 0.70 m neck cut).
  - Grazing-angle views of the hall's east wall (doubling, 0.27 m).
  - Kitchen frames filmed looking out into the passage carve the passage into the kitchen.
- **Change made:** link placement on by default for per-room video runs; `apply_placement` also moves `run_views`; refit and crop kept as switches, off.
- **Impact:** first video plan with all three rooms in the right arrangement. Next: (a) room-level fit after placement: room shifts (and per-room horizontal scale) so shared partitions and collinear outer walls agree, video tolerance ~0.4 m; (b) clip carving beyond well-supported wall lines (doorway leaks); (c) room ownership of overlapping free space after placement.

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
