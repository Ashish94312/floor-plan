# Technical Report: iPhone Capture → Measured Floor Plan

Every number here can be regenerated:
- [reports/benchmark.md](../reports/benchmark.md) (`uv run scan-bench`)
- [reports/fix_loop.md](../reports/fix_loop.md)

The decisions (D1–D32) are in [DECISIONS.md](DECISIONS.md), and every experiment (E1–E25) is in [WORKLOG.md](WORKLOG.md).

## 1. What was built

One command per capture (`scan <capture>`) turns an iPhone capture into:
- a per-room plan: walls, ceiling height, floor area, doors and windows
- one stitched whole-property plan
- damage regions, concealed-damage flags with the rule that fired, and scope items keyed to surfaces

Every number has a 90% interval. Output is JSON (published schema) plus PNG/SVG plans. Everything runs locally: two Apache-2.0 models are fetched by script and pinned by revision and SHA256, and nothing calls our own infrastructure.

The capture route is Route 2: the stock Camera app for photos and video, Stray Scanner for LiDAR, with a one-page protocol.

| Tier | State on the benchmark (home01: bedroom, L-shaped hall, kitchen; laser ground truth) |
|---|---|
| Photo | Works end to end, including the stitch. Portrait capture: 12/14 walls within ±8%, mean 2.9%, footprint −2.7%. Landscape capture: footprint +2.8%, but two room outlines are not scorable (§7). Intervals contain the tape value 100% of the time. |
| Video | Runs end to end. Final config (§3): no overlaps, footprint −4.3%. Walls: mean 10.3%, 0/8 within the ±3% target. Intervals contain the tape value 75% of the time (§7). |
| LiDAR | Runs end to end on the three assignment Stray Scanner samples (E25): phone depth through ARKit poses, the walk split into rooms at doorways (2, 3 and 6 rooms). No Pro iPhone was available, so there is no tape-measured LiDAR result. On the same walks MapAnything is far off (mean room-area error 80%, E26). |

## 2. Architecture

There is one shared core with thin tier adapters (D5). The adapters differ only in where depth and camera poses come from:

| Stage | Photo | Video | LiDAR |
|---|---|---|---|
| Ingest | HEIC/JPEG, EXIF intrinsics, duplicate and lens checks (D18, D19) | Keyframes by sharpness, focal from vanishing points | Stray Scanner export: ARKit depth, confidence, poses |
| Depth and poses | MapAnything, one **joint** run over all rooms, EXIF rays (D17, D20) | MapAnything per room, model rays, frames padded to 4:3 (E22s) | From the phone, no model |

The shared core then runs in this order:

1. **Alignment.** Gravity from the floor, Manhattan rotation, floor at z = 0. Floor and ceiling are consensus planes (D22).
2. **Layout.** Free-space carving: lines of sight from each camera to its depth mark empty floor, and walls are the boundary of that region (D23). Open sides between rooms are found from wall coverage (D26).
3. **Openings.** Visibility voting on raw depth: a wall stretch you can see through is an opening (D28). OWLv2 adds closed doors, windows and mirrors, but only when seen in 2 or more photos (D29).
4. **Stitching.** A joint run shares one frame. Otherwise rooms are placed by link frames (views two rooms share) or by door matching (D31).
5. **Drift correction** (§4).
6. **Damage and rules.** YAML rules (D7).
7. **Intervals.** Calibrated per mode and measurement type (D30).
8. **Output.** JSON and render.

## 3. Metric scale: the hard part of photo and video

A photo carries no depth, so absolute size comes from the model's sense of how big things are. Two findings shaped the design:

- **Known intrinsics matter.** MapAnything reads the photo focal about 12% long. Rebuilding points from EXIF rays, in one joint run over all rooms, took ceiling errors from −6/−8% to +0.6/−0.1% (E10, E15, D20). The joint run also gives every room the same scale.
- **A size prior fixes Z/f, not Z** (E22o). An object of size H spanning h pixels sits at Z = H·f/h, so the model's depth follows the focal it believes. That belief depends on orientation: about 1.13× EXIF for portrait and 0.96× for landscape. This was the fix loop (§8). Each view votes log(f_EXIF / f_model); the run is scaled by the median over rooms; a single level (the model's absolute size sense, ×1.082) is calibrated on taped captures.
- **Video.** Each room is a separate model run. Its size error is per room (±6%) and not a systematic bias (E22q). Two settings make up the final video config (user's choice, after reviewing the plans):
  - padding frames to 4:3, so the model reads the focal within about 10% (E22s)
  - a door-height anchor: a door whose top was seen fixes its room's size (E22z)

## 4. Drift handling (G4)

**Photo** (D25).
1. All rooms come from one joint reconstruction, so they share one frame. There is no chain of per-room poses to drift.
2. **Structural wall snapping** removes what remains: the two faces of a shared partition become one 0.12 m wall, and collinear outer walls are joined into one line, with the best-seen face setting the line.

**Video.** Each room is its own run. Rooms are placed by link frames (views that see into the neighbouring room, E22n) and then snapped the same way. Lines the person filming knows to be one wall can be declared in `hints.yaml` (E23c).

`--no-drift-correction` turns snapping off. The ablation, stitched footprint against the 19.93 m² tape:

| Capture | On | Off |
|---|---|---|
| photo_a | −2.7% | −1.4% |
| photo_b | +2.8% | +5.6% |
| video_e | −4.3% | −3.6% |

Snapping helps when rooms disagree (photo_b). On photo_a, which is already consistent, it costs 1.3%. The footprint is the sum of net room areas (D10), so snapping moves it only through the shared walls.

## 5. Error budget

| Source | Size | Evidence | Handled by |
|---|---|---|---|
| Model size sense (one level per mode) | photo ±3.8% (1σ, log) after the vote; video ±6% per room | `config/calibration.yaml`, E22q | Calibrated level; its σ goes into every interval |
| Orientation-dependent focal reading | 13% between portrait and landscape | fix loop | Focal vote (§3) |
| View-to-view surface doubling | 5–8 cm typical; 0.7 m on photo_b's bedroom north wall | E7, E15, fix-loop post-mortem | Consensus planes; **open** |
| Wall position from carving | 2–3 cm (`sigma_geom_m`) | layout tests | In the interval |
| Ceiling | photo −3 to −10 cm (biased low); video large | benchmark G2 | Reported as bias (photo) and as variance (video) |
| Door width | −18 cm systematic (0.72 m read vs 0.90 m tape) | benchmark G1 | **Open** (§7) |

## 6. Calibration analysis

- **Interval model.** For a length L: σ² = (L·σ_log)² + σ_position². The level's spread goes into σ_log. A split-conformal factor k per mode and measurement type is fitted on taped residuals (D30). Sample guards apply: with n < 10 we use max ratio × 2, floored at 0.5.
- **Coverage:** photo 100% (target 90%); video 70–75%. Video intervals are too narrow: its per-room size error (±6%, E22q) is larger than the door anchor's spread suggests.
- **The fix loop removed confident garbage on the protocol capture.** photo_b went from 17% coverage, with narrow intervals 12% off, to 100%.
- **Caveat.** Calibration rests on two captures of one flat (5 rooms, 3 physical). The leave-one-room-out errors of the level are −3.0% to +8.3%. The walk-in on an unseen flat is the real test, which is why σ_log is kept at or above 5% (`sigma_log_floor_photo`) and widened by the level's own spread.

## 7. Known failure modes (all seen on the benchmark)

1. **Doors are read 18 cm narrow (G1 is 0%).** Visibility voting sees the clear gap between solid wall on both sides: 0.72 m. The tape went jamb to jamb: 0.90 m. Either the depth model softens the jamb, or the tape definition differs; this is unresolved. Detector phantoms also count against G1. A tile reflection read as a window was removed by mirror suppression, but some remain.
2. **Doubled walls, so room outlines can't be scored.** On the landscape capture, the views disagree on the bedroom's north wall by about 0.7 m. Carving makes a 6-wall L, and the hall comes out with 8 walls. The corner-fill threshold (0.6 m²) decides between the two copies; we did not tune it against the tape.
3. **Ceilings biased low on photo** by 3–10 cm. The rooms agree with each other (1.4–2.5 cm apart across captures), so this is bias, not noise.
4. **Video room sizes.** The hall comes out about 10% small even in a joint run (E23b), which is how the model sees that clip. Ceilings are not repeatable across captures. Landscape clips rarely see the ceiling.
5. **Damage.** The one staged stain (a real damp patch) was missed. Video frames give 7–10 detector phantoms. The two-class requirement is unmet: only a water stain was staged.
6. **Mixed orientation.** A room shot in portrait inside a landscape capture is centre-cropped by the model's batching, which loses its floor and ceiling. A padding fix was built but parked because it made results worse (`scripts/experiments/parked_zoom_pad.patch`).
7. **Hard surfaces.**
   - Mirrors: suppressed as openings when the detector finds them.
   - Glass, wet-look surfaces, low light: handled only by the protocol (lights on, no pointing into windows) and by confidence filtering, which drops the model's least confident 30% of pixels. Not measured.
8. **Scope.** Single storey, Manhattan-world rooms (D9). Angled walls fall back to a rectangle with a wide interval.

## 8. Fix loop

**Declared before any code change** (tag `fix-before`):
- Gate: the photo tier on the landscape capture: 2/8 walls within ±8%, 17% coverage.
- Root cause: model depth follows its orientation-dependent focal reading. Evidence: per-view f_model/f_EXIF of 1.13 (portrait) against 0.96 (landscape), and a size ratio of 0.866 against a focal ratio of 0.850.
- Fix: the focal vote plus a calibrated level.

**After** (tag `fix-after`):
- Scale factors exactly as predicted (k = 0.910 / 1.044).
- photo_b ceilings from −35 cm to −4 / −3 cm; coverage from 17% to 100%.
- The two captures' ceilings now agree within 1.4–2.5 cm (they were 33–35 cm apart).

**What did not hold.** At the correct scale, the bedroom's doubled north wall (failure 2) turns the outline into an L that can't be scored. The gate stayed failing (2/4) and the wall-count prediction was wrong. The post-mortem is in [reports/fix_loop.md](../reports/fix_loop.md).

## 9. Head-to-head

We compared against magicplan 2026.38.0 on the same iPhone 13 (camera scan, free tier), over 8 shared dimensions in 2 rooms. The brief asks for our LiDAR tier against the app; with no LiDAR device, our photo and video tiers stand in.

| Our capture | Beat or tie (pass ≥ 70%) |
|---|---|
| photo_a | 7/8 (88%) |
| photo_b (headline, chosen before the run) | 4/8 (50%): the unscorable outlines count as losses |
| video_e | 3/8 (38%) |

magicplan's bedroom ceiling read 2.14 m against 2.79 m on the tape, and it drew the L-shaped hall as a rectangle.

## 10. How we got here: experiments, changes and reasons

The full record, with commands and output folders, is in [WORKLOG.md](WORKLOG.md). The key steps, in order:

| Experiment | What it showed | What we changed | Why |
|---|---|---|---|
| E1–E7: backbone spike | VGGT + Depth Anything metric scale: +53 to +57% size error. MapAnything with EXIF focal: +2.5 / +1.2 / −3.0% | MapAnything + EXIF intrinsics (D17) | Only option within the photo tier's ±8% |
| E9, E15: per-room vs joint runs | Separate runs differ ±5% in scale. One joint run shares the scale: 5 of 6 walls within 1.5% | Photo default: one joint run over all rooms (D20) | Rooms get one scale, and stitching comes free |
| E10, E12: EXIF rays; vanishing points as referee | The model reads the focal about 12% long, which compresses heights. Ceilings −6/−8% → +0.6/−0.1% with EXIF rays | Points rebuilt from EXIF rays (D20) | Vanishing points confirmed EXIF is right |
| E14: ICP pose refit | Failed: low overlap, slides along walls | Dropped | — |
| E16: misplaced photos | One photo placed 19° off the house axes | Auto-drop views off the axes (`check_views`) | Protects a walk-in run from one bad photo |
| E17–E18: openings | Detector boxes are imprecise. OWLv2 calls marble "stains" | Openings by visibility voting (D28); detector kept only when 2+ photos agree (D29) | Geometry, not boxes, measures widths |
| E19: determinism | Cached twice, and live vs cached: identical | Content-keyed cache (D21) | The brief allows a cache only if it replays deterministically |
| E20: mixed lenses | Two 0.5× photos shrank the flat by 4.6% | Exclude photos not on the capture's main lens (later made relative to the capture: Pro iPhones' 1× can be 28/35 mm) | Joint scale needs one lens |
| E21–E22d: first video | Portrait clips near walls: views too narrow, rooms 40% too small | Protocol: film from the corners, landscape, see floor and ceiling | Capture fix, not code |
| E22b–c: video focal | The model ignores a given focal on video frames | Video uses the model's own rays | EXIF rays broke multi-view consistency |
| E22n: link frames | Views that see into the next room place rooms correctly (rms ~10 cm) | Link placement for per-room video runs | No door detection needed |
| E22o–q: video scale | The model's size error is per room (±6%), not a constant bias | Calibrated intervals instead of a correction (D32) | Honest intervals |
| E22s: pad frames to 4:3 | The model then reads the focal within about 10% | Kept (final video config) | Bedroom size right |
| E22u–x | Extra keyframes, floor-line ranging, cross-room ratios, all-frame ownership | Not adopted (falsified or no gain) | Evidence did not support them |
| E22z: door-height anchor | Hall −12% → −8% where a door top was seen | Kept for video (final config), gated at 3σ | A general prior (door ≈ 2.05 m), not tape |
| E23a: plan review | Crossed partitions, a door on the wrong wall, a window that was a reflection | Fixes in snapping, shared doors, mirror rule | User's review of the drawn plan |
| E23b: joint scale for video | The hall stays about 10% small even in one joint run | Off by default | Not the better route |
| E23c: declared same wall line | Kitchen + passage span the hall's width | `hints.yaml` `same_line` | A layout fact the person filming knows |
| Fix loop (§8) | Photo scale follows the orientation-dependent focal | Focal vote + calibrated level | Removed the photo tier's confident garbage |
| Zoom and orientation padding | Each was principled, but together they made photo_b worse (mean 8 → 15%) | Parked as a patch | Cover the scoring items first |
| Final video config | Padding + door anchor: 0 overlaps, footprint −4.3% (off: 2 overlaps, −18.3%) | `video.pad_to_aspect: 1.3333`, `video.door_height_prior_m: 2.05` | Chosen by the user after reviewing the plans |

## 11. Assumptions where the brief is open

- Intervals at 90%.
- G3 uses the looser max(1 cm, 0.5%).
- The G1 denominator counts phantoms.
- Ceiling = the dominant ceiling plane above the dominant floor plane.
- Floor area = net internal area.
- Single storey.
- The walk-in venue is offline.
- Photos are used as RGB plus EXIF only.
