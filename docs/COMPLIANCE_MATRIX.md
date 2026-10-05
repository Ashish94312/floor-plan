# Compliance Matrix

Each row is one requirement: where it lives in the code, the artifact that proves it, and its status. Status values:
- ✅ Met
- ⚠️ Partial (with the number)
- ❌ Not met
- ➖ N/A (with the reason)

Requirement IDs refer to [REQUIREMENTS.md](../REQUIREMENTS.md). Numbers come from [reports/benchmark.md](../reports/benchmark.md) (`uv run scan-bench`).

## Capture route

| ID | Requirement | File path | Artifact | Status |
|---|---|---|---|---|
| FR-CAP-01/03 | One capture route: Route 2, a one-page stock-app protocol | `docs/CAPTURE_PROTOCOL.md` | protocol | ✅ Camera app (photo, video), Stray Scanner (LiDAR) |
| FR-CAP-04 | The route covers all three tiers | `docs/CAPTURE_PROTOCOL.md` | protocol | ✅ |
| FR-CAP-05 | Device matrix | `docs/DEVICE_MATRIX.md` | doc | ✅ (tested on an iPhone 13, disclosed) |
| NFR-16 | A non-engineer can follow the protocol | — | — | ❌ not yet tested with a non-engineer |

## Input and output

| ID | Requirement | File path | Artifact | Status |
|---|---|---|---|---|
| FR-IN-01 | Photo tier: one folder per room, 2–8 stills | `scan/io/ingest.py`, `scan/io/photos.py` | `tests/unit/test_ingest.py` | ✅ duplicates dropped, more than 8 → sharpest 8, mixed orientation → majority, non-1× lens excluded |
| FR-IN-02 | Video tier: handheld walkthrough | `scan/io/video.py` | `tests/unit/test_video.py` | ✅ keyframes by sharpness, focal from vanishing points |
| FR-IN-03 | LiDAR tier: depth, poses, intrinsics | `scan/io/lidar.py`, `scan/layout/segment.py`, `scan/synth/stray.py` | `tests/unit/test_lidar.py`, `tests/unit/test_segment.py` | ✅ runs end to end on the three assignment Stray Scanner samples (E25); ⚠️ no tape-measured LiDAR capture, so accuracy and intervals are not calibrated |
| FR-IN-04 | Clear errors on unusable input | `scan/errors.py`, `scan/io/ingest.py` | `tests/unit/test_ingest.py` | ✅ |
| FR-IN-05 | Tier auto-detected | `scan/io/ingest.py` | `tests/unit/test_ingest.py::test_tier_detection` | ✅ |
| FR-RUN-01, NFR-03 | One command per capture | `scan/cli.py` (`scan <capture>`) | README | ✅ |
| FR-RUN-02 | JSON conforms to the published schema | `scan/schema.py` | `schema/scan_output.schema.json` | ✅ written by `scan-schema` |
| FR-RUN-03 | Rendered plan, per room and stitched | `scan/render/plan.py` | `out/plan.png`, `plan.svg`, `rooms/*.png` | ✅ |
| FR-RUN-04 | Per-stage timing | `scan/pipeline.py` | `timing_s` in `result.json`; benchmark timing table | ✅ |
| FR-RUN-05, NFR-07 | The live path runs; the cache replays deterministically | `scan/cache.py`, `--no-cache` | `tests/test_determinism.py`; WORKLOG E19 | ✅ live vs cached: max difference 0.0 over 210 numbers |

## Geometry, stitching, drift

| ID | Requirement | File path | Artifact | Status |
|---|---|---|---|---|
| FR-GEO-01 | Per-room walls, ceiling, area, openings | `scan/layout/room.py`, `scan/layout/openings.py` | `rooms` in `result.json` | ✅ |
| FR-GEO-02, G1 | Opening widths ≤ 2 cm on ≥ 85% (misses and phantoms count) | `scan/layout/openings.py`, `scan/eval/gates.py` | benchmark: gates | ❌ 0% at every capture: doors read 0.70–0.72 m against a tape of 0.89–0.90 m; detector phantoms |
| FR-GEO-03 | Metric scale at every tier | `scan/geometry/cloud.py` (photo focal vote), `scan/uncertainty/calibrate.py` | `reports/fix_loop.md` | ✅ photo: focal vote + calibrated level |
| FR-GEO-04, FR-DMG-05 | An interval on every measurement | `scan/uncertainty/intervals.py`, `scan/schema.py` (`Measurement`) | every number in `result.json` | ✅ |
| G2, FR-EVAL-06 | Ceiling ≤ 1.5 cm, spread ≤ 1 cm; bias or variance stated | `scan/layout/room.py` | benchmark: gates, repeatability | ❌ photo: −2.9 … −9.8 cm, **biased low**, nearly repeatable (spread 1.4–2.5 cm). Video: **unrepeatable** (spread up to 88 cm) |
| G3, NFR-13 | Two captures agree within max(1 cm, 0.5%) per wall | `scan/eval/bench.py` | benchmark: repeatability | ❌ photo 0/16 (landscape bedroom and hall outlines not scorable; kitchen 15–18 cm apart), video 2/15. Same input → same output holds (E19) |
| FR-STI-01/02/03 | Stitched plan, correct adjacency, no overlaps | `scan/stitch/` | `plan.png`; benchmark G5 column | ✅ photo (both captures). ❌ video: 0 overlaps (final config), but one extra bedroom–kitchen adjacency |
| FR-STI-04, G5 | Photo tier stitches per-room folders, footprint ±8% | `scan/pipeline.py` (joint run), `scan/stitch/doors.py` | benchmark: gates | ✅ footprint −2.7% (photo_a), +2.8% (photo_b), adjacency right, 0 overlaps |
| FR-DRI-01/02, G4 | Drift correction, toggleable, ablation on/off | `scan/stitch/snap.py`, `scan/stitch/links.py`, `--no-drift-correction` | benchmark: G4 table | ✅ method and ablation reported: on vs off footprint −2.7 / −1.4% (photo_a), +2.8 / +5.6% (photo_b), −4.3 / −3.6% (video_e). It helps only on photo_b |
| Accuracy | Photo ±8%, video ±3%, LiDAR (Round 1 gates) | `scan/eval/gates.py` | benchmark: gates | ⚠️ photo 12/14 (photo_a), 2/4 scorable (photo_b). ❌ video 2/14 (video_d) and 0/8 (video_e) within ±3%. LiDAR not measured (no device) |
| NFR-08, FR-EVAL-05 | Calibrated intervals at every tier; no confident garbage | `scan/uncertainty/calibrate.py`, `config/calibration.yaml` | benchmark: coverage column | ✅ photo 100%. ⚠️ video 70–75% (target 90%) |

## Damage and scope

| ID | Requirement | File path | Artifact | Status |
|---|---|---|---|---|
| FR-DMG-01 | Damage regions with a class and metric extent | `scan/damage/detect.py` (OWLv2, lifted onto surfaces) | `damage` in `result.json` | ⚠️ mechanism works; the staged stain was missed (0/1); video shows 7–10 phantoms |
| FR-DMG-02, B2 | Two damage classes staged in one furnished room | `data/ground_truth/home01.yaml` | — | ❌ one class staged (a real water stain); crack not staged |
| FR-DMG-03, NFR-14 | Concealed-damage flags state the rule that fired | `scan/damage/rules.py`, `config/rules.yaml` | `concealed_damage_flags` (`rule_id`, `rule_text`) | ✅ mechanism (`tests/unit/test_damage.py`); fires only when damage is found |
| FR-DMG-04 | Scope items keyed to surfaces | `scan/damage/rules.py` | `scope_items` | ✅ mechanism, as above |

## Benchmark, head-to-head, fix loop

| ID | Requirement | File path | Artifact | Status |
|---|---|---|---|---|
| B1 | Multi-room capture: 3+ rooms plus a connector | `captures/home01_*` | bedroom1, hall (the connector), kitchen | ✅ |
| B3 | The same rooms at all three tiers | `captures/home01_photo_*`, `home01_video_*` | — | ⚠️ photo and video only; no LiDAR device |
| B4 | One room captured twice at the same tier | `home01_photo_a`/`_b`, `home01_video_d`/`_e` | benchmark: repeatability | ✅ all three rooms, twice at each tier |
| B5 | Laser ground truth on everything | `data/ground_truth/home01.yaml` | YAML (derived values carry their formula) | ✅ walls, ceilings, doors, window, stain; hall sticker door and window not measured |
| FR-EVAL-02 | Every gate scored automatically | `scan/eval/gates.py`, `scan/eval/bench.py` | `reports/benchmark.md` | ✅ photo, video |
| FR-EVAL-04 | Head-to-head vs a consumer app, ≥ 70% beat or tie | `scan/eval/bench.py`, `data/head_to_head/magicplan.yaml` | benchmark: head-to-head | ⚠️ vs magicplan 2026.38.0: photo_a 7/8 (88%) pass; the headline photo_b 4/8 (50%) fails; video_e 3/8. Photo tier, not LiDAR (no device) |
| Fix loop | Declaration, shipped fix, regenerable before and after, readable diff | `docs/FIX_DECLARATION.md`, tags `fix-before` / `fix-after` | `reports/fix_loop.md` | ⚠️ root cause shown, scale fixed (photo_b ceilings −36 → −3/−4 cm, coverage 17% → 100%); declared gate still failing (2/4 walls), explained |

## Delivery

| ID | Requirement | File path | Artifact | Status |
|---|---|---|---|---|
| NFR-02 | README → running in < 15 min on a clean machine | `README.md` | — | ⚠️ written; timed fresh-clone run still to do |
| NFR-04, NFR-21 | No calls to our own infrastructure; runs offline | `scan/device.py` (`go_offline`) | WORKLOG E11: offline run, empty torch cache, bit-identical | ✅ |
| NFR-05 | Weights fetched by script | `scan/weights.py`, `scan-fetch-weights` | pinned revisions + SHA256 | ✅ |
| NFR-06 | Every number regenerable from raw inputs | `scan-bench`, `scripts/calibrate_photo_vote.sh` | `reports/benchmark.md` | ✅ (raw captures in the data bundle) |
| NFR-09 | Mirrors, glass, wet-look, low light | `scan/damage/detect.py` (mirror removes phantom opening), `scan/pipeline.py` (`check_views` drops misplaced photos), confidence filtering | technical report | ⚠️ mirrors handled; glass, wet-look, low light only by protocol and confidence filtering |
| NFR-17 | Incremental git history | git | `git log` | ✅ |
| NFR-19 | Technical report ≤ 6 pages | `docs/TECHNICAL_REPORT.md` | — | ⏳ |
| NFR-20 | Pinned dependencies | `uv.lock` | — | ✅ |
