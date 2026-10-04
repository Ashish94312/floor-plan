# Build Plan — Tier by Tier (48 hours)

| | |
|---|---|
| **Deadline** | 48 hours from start (2026-10-04) |
| **Order** | Phase 0 setup → **Tier 1 Photo** → **Tier 2 Video** → **Tier 3 LiDAR** → benchmark → fix loop → reports |
| **Hard rule** | **Feature freeze at H31.** Whatever is unfinished gets cut. The fix loop (25% of the score) is never cut. |
| **Design** | [ARCHITECTURE.md](ARCHITECTURE.md) |

---

## Timeline at a glance

```
H0 ──H2 ────────────── H15 ── H21 ──────── H27 ──── H31 ─H33 ──────── H39 ───────── H45 ── H48
 │P0│   TIER 1 PHOTO    │SLEEP│  TIER 2    │ TIER 3 │BENCH│  FIX LOOP   │ REPORTS +   │BUFFER│
 │  │ (core is built    │     │  VIDEO     │ LiDAR  │+H2H │ (25%)       │ PACKAGING   │      │
 │  │  here, reused     │     │            │        │     │             │             │      │
 │  │  by tiers 2 & 3)  │     │            │        │  ▲ FEATURE FREEZE │             │      │
```

**Why Tier 1 takes the longest:** the shared core (layout, openings, stitching, damage, rules, intervals, JSON, render, eval) is built during Tier 1. Tiers 2 and 3 only add a new adapter in front of it.

---

## Parallel track — your physical work (do first, about 3 hours)

- [ ] Free disk space to about 40 GB
- [ ] Get a tape measure
- [ ] Stage 2 damage types in one furnished room (tea-stained paper = water stain, drawn line on masking tape = crack)
- [ ] Measure every room → `data/ground_truth/home01.yaml` (copy of TEMPLATE.yaml)
- [ ] Photograph a hand-drawn layout sketch
- [ ] Capture **photos** of all rooms + hallway, following CAPTURE_PROTOCOL.md → `captures/home01_photo_a/`
- [ ] Capture **one room twice** (photos) → `captures/home01_photo_b/` (that room only)
- [ ] Capture the **video** walkthrough → `captures/home01_video_a/`
- [ ] Capture the **video** walkthrough again → `captures/home01_video_b/` (repeatability)
- [ ] Scan 2 rooms with **magicplan** and export or screenshot the dimensions → `captures/magicplan/`

---

## Phase 0 — Setup and spike (H0–H2)

| Task | Output |
|---|---|
| `pyproject.toml`, `uv` env, package skeleton, CLI stubs | `uv run scan --help` works |
| Install PyTorch and confirm MPS | Device check prints `mps` |
| **Spike:** run VGGT (and MapAnything if time) on 4–8 iPhone 13 photos of one measured room | Time, peak memory, point cloud |
| **Spike:** Depth Anything V2 Metric scale on the same photos | Scale estimate vs tape-measured wall |
| **Spike:** OWLv2 on the same photos (door, window, stain, crack) | Detections look sane |
| Decide O1–O3 (ARCHITECTURE §19), record in DECISIONS.md | Decision logged |

**Exit:** one room's photos → metric point cloud in under 60 s on the M4, with scale within about 10% of the tape measure.

---

## Tier 1 — Photo (H2–H15)

| Step | Task | Files | Exit check |
|---|---|---|---|
| 1.1 | Ingest: folders, HEIC/JPEG, EXIF intrinsics, validation | `io/ingest.py`, `io/photos.py` | Bad folder → clear error |
| 1.2 | Geometry: backbone + metric scale `(s, σ)` + point cloud | `geometry/vggt_backend.py`, `metric_depth.py`, `scale.py` | Cloud in metres |
| 1.3 | Alignment: floor, gravity, Manhattan, ceiling | `geometry/align.py`, `planes.py` | Floor at z=0, walls axis-aligned |
| 1.4 | Room layout: polygon, walls, ceiling height, floor area | `layout/room.py` | Numbers printed for one room |
| 1.5 | Schema + JSON export + render | `schema.py`, `render/plan.py`, `pipeline.py` | **MILESTONE A:** `uv run scan captures/home01_photo_a` → JSON + per-room PNGs |
| 1.6 | Eval harness: GT loader, matching, wall/ceiling/area errors | `eval/*` | `scan-eval` prints an error table |
| 1.7 | Openings: detect, lift, measure, merge, mirror suppression | `layout/openings.py` | G1 number reported |
| 1.8 | Stitching: door matching, placement, global LS, overlap check, ablation flag | `stitch/doors.py`, `stitch/place.py` | **MILESTONE B:** stitched `plan.png`; G4 and G5 reported |
| 1.9 | Damage + rules + scope | `damage/*`, `config/rules.yaml` | Staged damage found and flagged |
| 1.10 | Intervals + calibration | `uncertainty/*` | Every number has `lo`/`hi`; coverage reported |
| 1.11 | Determinism test + unit tests for the geometry helpers | `tests/` | Same JSON twice |

**Tier 1 exit:** full output contract from photo folders, stitched plan, all gates computed. **Commit and tag `tier1`.**

---

## Tier 2 — Video (H21–H27)

| Step | Task | Files | Exit check |
|---|---|---|---|
| 2.1 | Frame decode + keyframe selection (sharpness) | `io/video.py` | About 2 fps of sharp keyframes |
| 2.2 | Chunked reconstruction + Sim(3) chunk alignment | `geometry/chunks.py` | One global cloud |
| 2.3 | Loop closure + pose graph (drift correction) + Manhattan snap | `stitch/loop.py` | Drift on/off ablation image |
| 2.4 | Room segmentation (door cuts, watershed fallback) | `layout/segment.py` | Correct room count on home01 |
| 2.5 | Reuse the core (layout → render) | — | Full contract from video |
| 2.6 | Eval at the video tier + repeatability (video_a vs video_b) | — | ±3% wall gate reported |

**Tier 2 exit:** `uv run scan captures/home01_video_a` → full contract. **Commit and tag `tier2`.**

---

## Tier 3 — LiDAR (H27–H31)

| Step | Task | Files | Exit check |
|---|---|---|---|
| 3.1 | Synthetic generator: 3 rooms + hall in Stray format, noise, injected drift, GT YAML | `synth/rooms.py` | Files written |
| 3.2 | Stray Scanner reader | `io/stray.py` | Reads the synthetic capture |
| 3.3 | Depth back-projection + fusion | `geometry/lidar.py` | Global cloud |
| 3.4 | ICP loop closure + pose graph | `stitch/loop.py` | Drift on beats drift off on synthetic data |
| 3.5 | Reuse the core | — | Full contract from LiDAR |
| 3.6 | Synthetic end-to-end test | `tests/synth/` | Passes in CI-style run |

**Tier 3 exit:** LiDAR path runs end to end on synthetic data and is ready for a real Stray Scanner export at the walk-in. **Commit and tag `tier3`.**

### ⛔ FEATURE FREEZE — H31

---

## Phase 4 — Benchmark + head-to-head (H31–H33)

- [ ] `scan-bench` runs every capture and its eval and writes `reports/benchmark.md`
- [ ] Calibrate `k_tier` (leave-one-room-out) → `config/calibration.yaml`, then re-run
- [ ] Head-to-head table: magicplan vs our video tier on 2 rooms → `reports/head_to_head.md`
- [ ] Timing table → `reports/timing.md`

## Phase 5 — Fix loop (H33–H39)

1. [ ] Pick the **single worst gate** from `reports/benchmark.md`
2. [ ] Write `docs/FIX_DECLARATION.md` (failing number, root-cause hypothesis + evidence, planned fix, predicted number)
3. [ ] **Commit the declaration and tag `fix-before` BEFORE touching any code**
4. [ ] Implement the fix, re-run the benchmark, tag `fix-after`
5. [ ] `reports/fix_loop.md`: before vs after, `git diff fix-before..fix-after`, and why it reached (or fell short of) the prediction

## Phase 6 — Reports and packaging (H39–H45)

- [ ] `docs/TECHNICAL_REPORT.md` (≤ 6 pages): architecture, tiers + device matrix, drift, error budget, calibration, fix-loop story, failure modes
- [ ] `docs/DEVICE_MATRIX.md`
- [ ] `docs/COMPLIANCE_MATRIX.md` (start from ARCHITECTURE §17)
- [ ] README: install → fetch weights → run in under 15 min, model licences and disclosures
- [ ] Reproduction bundle: raw captures + GT + magicplan export (zip or external link, since they're too big for git)
- [ ] **Fresh-clone test**, timed

## Buffer (H45–H48)

Things will break. Don't plan work here.

---

## Cut list (if behind, drop in this order)

1. SAM mask refinement for damage (keep box extent)
2. MapAnything comparison (keep one backbone)
3. Non-rectangular room refinement (keep the rectilinear fallback)
4. Door-height scale prior (keep metric depth only)
5. Video watershed fallback (keep door-cut segmentation)

**Never cut:** the fix loop, the compliance matrix, honest intervals, all three tiers runnable, git history.

## Commit cadence

Commit at every row marked with an exit check (roughly 1–2 hours apart), using messages that say *what* and *why*. Tag `tier1`, `tier2`, `tier3`, `fix-before`, `fix-after`.
