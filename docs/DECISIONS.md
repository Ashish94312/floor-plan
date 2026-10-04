# Design Decisions Log

Every decision here must be defensible live at the defense, without tools. Each entry: the decision, why, and what it costs.

## D1. Capture route: Route 2 (stock apps + one-page protocol)
- **Why:** No Apple Developer account and no LiDAR device available to test an iOS app. App Store apps install in minutes. Capture route is 5% of the score; pipeline accuracy, the fix loop and the walk-in test are 70%.
- **Cost:** Less control over capture (no live guidance or quality checks on the phone).
- **Protocol:** [CAPTURE_PROTOCOL.md](CAPTURE_PROTOCOL.md)

## D2. Development device: iPhone 13 (no LiDAR)
- **Why:** It is the only device available within the 48-hour deadline.
- **Deviation from spec:** The spec asks for photo and video tiers from an iPhone 15 or newer. Benchmark photo and video captures were taken on an iPhone 13. This is disclosed in the device matrix and compliance matrix.
- **Mitigation:** The pipeline reads camera intrinsics per image from EXIF and never hardcodes a phone model, so iPhone 15+ captures at the walk-in use the same code path.

## D3. LiDAR tier: implemented against the Stray Scanner export format, validated on synthetic rooms only
- **Why:** No Pro device is available. The walk-in may still choose LiDAR, so the code path must exist and run.
- **Cost:** No real-capture accuracy numbers at the LiDAR tier. Reported honestly as "Partial".

## D4. Head-to-head: magicplan on iPhone 13 vs our pipeline
- **Why:** The spec asks for the LiDAR tier. That's impossible without a Pro device. magicplan runs without LiDAR, so this is the closest honest comparison available.
- **Deviation:** Disclosed. The comparison is made against our video tier.

## D5. Architecture: one shared core with thin tier adapters
- **Why:** The output contract is identical across tiers. Each adapter only produces frames, depth, poses, intrinsics and their uncertainty. Layout, stitching, damage, rules, intervals and output are shared, so all tiers stay consistent and share fixes.

## D6. Drift handling: plane-anchored (Manhattan-world) correction, switchable for ablation
- **Why:** Most indoor walls are axis-aligned. Snapping room orientations to shared dominant axes removes accumulated rotation drift cheaply and robustly. "Poses as-is" is an automatic fail.
- **Ablation:** A `--no-drift-correction` flag produces the stitched footprint without it.

## D7. Concealed damage and scope: explicit YAML rules
- **Why:** The spec requires stating "the rule that fired". Readable rules are auditable and easy to defend.

## D8. Uncertainty: per-tier error model calibrated on benchmark residuals (leave-one-room-out)
- **Why:** Confident garbage caps the total score. Intervals widen with tier. Calibrating with leave-one-room-out avoids grading the intervals on the same data used to fit them.

## D9. Manhattan-world assumption for layout and stitching
- **Why:** Most rooms have walls at 90°. The assumption makes wall fitting, stitching rotation (multiples of 90°) and drift correction simple and robust.
- **Cost:** Angled or curved walls are approximated. Documented as a known failure mode, with a fallback rectangle and wider intervals.

## D10. Footprint = sum of room floor areas (net internal area)
- **Why:** This is what a tape measure can verify. The PDF doesn't define "footprint". The outline polygon is still rendered.

## D11. Wall numbering: W1 = main-door wall, then clockwise from above
- **Why:** The same rule is used in the ground-truth sheet and the pipeline, so walls can be matched for scoring. If the main door is ambiguous, eval takes the best cyclic shift and discloses it.

## D12. Single open-vocabulary detector (OWLv2) for openings, mirrors and damage
- **Why:** One model download, no training, readable text queries. Mirror detection suppresses phantom openings.
- **Cost:** Box-level extent overestimates damage area. Intervals reflect this.

## D13. VGGT runs with an fp16 aggregator, fp32 heads, no point head, depth head chunked 2 frames at a time (spike, O2)
- **Why:** Stock fp32 VGGT on 8 images on the M4 (16 GB) peaked at 17 GB of MPS memory, grew swap by 15 GB and took 212 s. fp16 autocast barely helped (17 GB, 178 s) because the fp32 weights stay resident. Running the stages by hand fixes it: the 0.9B-parameter aggregator in fp16, camera and depth heads in fp32, point head skipped (we unproject depth + cameras instead), depth head 2 frames per pass.
- **Measured (VGGT example scenes, 518×392):**

  | Frames | Inference | Peak MPS | Swap growth |
  |---|---|---|---|
  | 8 | 28 s | 6.8 GB | 1.5 GB |
  | 16 | 54 s | 6.8 GB | 0.8 GB |
  | 24 | 100 s | 6.8 GB | 0.2 GB |

  Weight load takes about 20 s on top of that. Room dimensions match fp32 to the millimetre and scale to 0.1%.
- **Cost:** Time grows faster than linear with frame count. A photo room (6–8 images) costs about 25 s, so 5 rooms fit the 2-minute backbone budget. A 5-minute video at 2 fps (600 keyframes) does **not** fit the 5-minute budget at 24-frame chunks. Tier 2 needs a keyframe budget (about 0.5 fps, or coverage-based selection). This is decided in Tier 2 step 2.1.

## D14. Open3D kept (O3)
- **Why:** Open3D 0.20.0 installs from a wheel on arm64 / Python 3.11 with numpy 1.26, and plane segmentation works. No pure-NumPy fallback needed.

## D15. Model licences (disclosed in README)
- VGGT-1B: **CC-BY-NC-4.0** (non-commercial). The commercial variant is gated behind manual approval, so it can't be relied on within 48 h. Acceptable for a case-study evaluation, and disclosed.
- MapAnything (Apache-2.0 build), Depth Anything V2 Metric Indoor Small, OWLv2 base: Apache-2.0.
- If O1 picks MapAnything, the whole stack is Apache-2.0. That is a point in its favour if accuracy is comparable.
- All weights are pinned to a Hugging Face commit + SHA256 in `scan/weights.py`.

## D16. Portrait photos are rotated to landscape before VGGT
- **Why:** On the first real capture (iPhone 13, portrait, 392×518), VGGT predicted fx ≈ 290 px against fy ≈ 383 px and an EXIF focal of 390 px. The predicted horizontal field of view was 26% too wide, so the room came out stretched sideways. With the same photos rotated 90°, it predicts fx = fy ≈ 422 px (within 8% of EXIF). Depth Anything still sees the upright photo, and its depth map is rotated to match. Gravity is read from camera +x instead of +y.
- **Also:** frames are resized so the long side is 518 px, with no centre-crop. VGGT's stock crop mode would cut a portrait frame to a square and lose the floor and ceiling.

## D17. O1 decided: MapAnything (Apache) with EXIF intrinsics replaces VGGT-1B + Depth Anything V2 metric
- **Evidence:** bedroom1, 4 portrait photos (home01_photo_a), against the tape measure (239 × 291.5 cm, ceiling 279.25 cm):

  | Backbone | Walls | Ceiling | Infer / load / peak MPS |
  |---|---|---|---|
  | VGGT-1B (rotated) + DAv2 metric scale | ~+40% (wall behind the wardrobe at 3.38 m against 2.39 m; one wall missing) | +14% to +45% | 13 s / 18 s / 6.8 GB |
  | MapAnything, no intrinsics | +0.2% long, +6.7% short | +11.8% | 14 s / 37 s / 6.1 GB |
  | **MapAnything + EXIF intrinsics** | **−1.5% long, −0.4% short** (measured to the wall behind the wardrobe) | **+5.7%** | 19.5 s / 44 s / 6.1 GB |

- **Why it wins:** VGGT can't take the known focal length, and the DAv2 metric scale was biased about 1.4× on this room. MapAnything accepts the focal length from EXIF and outputs metres directly. The whole stack then becomes Apache-2.0 (see D15).
- **Confirmed** on the reshoot (home01_photo_a/bedroom1: 7 portrait photos following the protocol, measured automatically, results the same at peak thresholds 0.2 and 0.08):

  | Backbone | Short side | Long side | Ceiling | Infer / peak MPS |
  |---|---|---|---|---|
  | VGGT-1B (rotated) + DAv2 metric | 3.650 m (+52.7%) | 4.570 m (+56.8%) | 4.300 m (+54.0%) | 25 s / 7.8 GB |
  | **MapAnything + EXIF intrinsics** | **2.450 m (+2.5%)** | **2.950 m (+1.2%)** | **2.710 m (−3.0%)** | 29 s / 7.1 GB |

  VGGT gets the shape right (all three errors are about equal) but the DAv2 metric scale is about 1.55× too big. MapAnything passes the Phase 0 exit check (within 10% of the tape). Remaining error: walls doubled by about 8 cm where views don't align perfectly. That is a fix-loop candidate for the repeatability gate (G3).
- **Protocol note:** both bedroom captures came back portrait, though the protocol says landscape. Portrait keeps floor and ceiling in frame, and MapAnything + EXIF handles it. Proposal: allow either orientation, but one orientation per room.
- **Costs to fix if adopted:**
  - At load, MapAnything fetches DINOv2 code from GitHub through torch hub. This breaks the offline requirement, so it must be pre-cached by `scan-fetch-weights`.
  - Load takes about 40 s.
  - Its `opencv-python-headless` pin clashes with VGGT's `opencv-python`. Dropping VGGT removes the clash.
- **Spike lessons for the layout stage:** built-in wardrobes create a false wall in front of the real one. A bed covering the floor creates a false floor peak. Pick the outermost plane with real support, not the strongest peak.

## D18. Intrinsics from EXIF use the film diagonal: `f_px = f35 × diagonal_px / 43.27`
- **Why:** The "35 mm-equivalent focal length" is defined by matching field of view on the 36 × 24 mm film **diagonal** (43.27 mm). On iPhone 13 this gives 3028 px for 4032×3024, which matches the physical lens (5.1 mm × crop factor 5.1). The width-based formula first written in ARCHITECTURE (`f35 / 36 × long_side`) gives 2912 px, **4% short**. That error lands directly in lengths and ceiling height. E6–E11 already used the diagonal formula.
- **Cost:** EXIF stores f35 as a whole number (26), so f is only known to about ±2%. If calibration shows a bias, the fix is a per-device correction learned from the benchmark.

## D19. Ingest policy for ambiguous photo sets
- **Same photo in two room folders → error, not a guess.** AirDrop's " 2" copies put `IMG_4876` in both bedroom and hall (E4). The code can't know which room is right, and a wrong guess silently corrupts layout and stitching. The message lists the pairs so the user can fix it in seconds. A duplicate within one room is harmless and is just dropped.
- **Mixed portrait/landscape in one room → keep the majority, drop the rest with a warning.** One batch needs one image shape. Rotating the odd one out would change its gravity direction. This is the same choice made by hand in E7 (`IMG_4887`).
- **More than 8 photos → keep the 8 sharpest**, in filename order.

## D20. Ceiling fix: joint run + EXIF rays by default (reverses "park for the fix loop")
- **History:**
  - D20 first parked the ceiling fix for the fix loop. The user chose to fix it now (2026-10-04, ~16:15).
  - The fix loop will pick its worst gate from the benchmark later.
- **Root cause (E12):** MapAnything's output focal is about 12% longer than the true focal. Vanishing points confirm the EXIF focal. A too-long focal compresses vertical extents, so ceilings came out 3–12% low.
- **Tried:**
  - EXIF rays with per-room runs (E14/E15): ceilings +4.2% / −4.0%, unbiased but each room's own scale is off by about ±5%. Bedroom walls inflated (short side +12.6%).
  - EXIF rays + pairwise ICP pose re-fit (E14): **failed**. Sparse corner views overlap 5–30%, so ICP slides along flat walls (corrections of metres and tens of degrees) and the pose graph disconnects. Moved to `scripts/experiments/icp_refine.py` as a recorded negative result.
  - Pre-compensating the input focal (E13): ceilings inconsistent between views.
- **Decision:** `geometry.joint: auto` (one joint run when ≤ `max_joint_views` = 16 photos, else per room with a warning) and `geometry.rays: exif`.
- **Evidence (E15, home01_photo_a, against tape):**

  | Config | Bedroom short | Bedroom long | Bedroom ceiling | Hall short | Hall long | Hall ceiling | Mean abs |
  |---|---|---|---|---|---|---|---|
  | joint + EXIF | +7.9% | −1.5% | **−0.1%** | +0.4% | −0.0% | **+0.6%** | **1.75%** |
  | joint + model rays | — | — | — | — | — | — | 3.55% |

  - Gaps between views are also lower with EXIF rays (bedroom 2.6 cm against 2.9; hall 3.8 against 4.8). EXIF rays are more self-consistent, not less.
  - Why joint helps: one metric scale shared by all rooms averages out the per-room scale error, and doorway views tie the rooms together.
- **Costs:**
  - Joint memory grows with photo count (14 photos = 8.1 GB). Larger homes fall back to per-room runs, which have no shared scale and get wider intervals.
  - The bedroom short side stays +7.5–7.9% (open door leaf + wardrobe recess). It's inside the photo tier's ±8% but is the weakest number.
  - Only one home so far. To be confirmed by the benchmark and repeat captures.

## D21. Model outputs are cached by content
- **Why:**
  - A live run costs about 35 s to load plus about 32 s per room. Development and the benchmark re-run the same captures many times.
  - The cache key covers everything that changes the output: image SHA256s, intrinsics, working size, weights revision, MapAnything commit, DINOv2 commit, `f_scale`, AMP dtype and memory mode. A stale hit is impossible without a key collision.
- **Check:** a cached re-run (7 s) wrote `.ply` files byte-identical to the live run. `--no-cache` forces the live path (FR-RUN-05).

## D22. Floor and ceiling are consensus planes, not the lowest/highest copy
- **Why:**
  - Views disagree by 5–8 cm on a surface's height (per-view floor z in the bedroom ranged 0.024–0.075; the histogram shows two floor copies 9 cm apart).
  - Taking the lowest floor peak (to skip the bed) also picked the *bottom copy*, which biased the bedroom ceiling +6.9 cm.
- **How:**
  - Find the lowest/highest strong horizontal peak as before; the bed, 50 cm up, stays excluded.
  - Take the whole layer (floor −5..+12 cm, ceiling mirrored), so every view's copy is in.
  - Height = **median** z of the layer.
  - Direction = least-squares normal of the dominant copy only (±3 cm). The synthetic test showed a partial second copy tilts a whole-layer fit.
- **Chosen on principle:** the median of all observations is the consensus estimate. The variant that matched the bedroom tape best (+0.1 cm) was **not** picked for that reason. The final numbers are bedroom −3.3 cm and hall +0.97 cm.
- **Cost:** with real doubling, the answer can only be as good as the copies agree. Bedroom uncertainty is about ±3–4 cm. The fix for the doubling itself is open (fix-loop candidate).

## D23. Room layout by free-space carving, not a wall-height band
- **Why:**
  - The planned layout (ARCHITECTURE v1: a high band of wall points → grid → enclosed region) assumes the high band shows the real walls. In home01 it shows a **ceiling beam** 0.57 m in front of the bedroom wall, and it misses the wall around the door the camera stood in. The bedroom fell back to a bounding box.
  - Lines of sight are better evidence. Every observed pixel proves the line from its camera to that surface was empty. Furniture, beams and door gaps can't hide the room, because rays pass over the bed, under the beam, and the doorway camera carves the room itself.
- **How (`layout/room.py`):**
  1. Per photo, up to 25k lines of sight drawn from above onto a 2 cm grid, stopping 6 cm short of each surface.
  2. 10 cm gap-fill, because rays fan out near far walls.
  3. **Joint frame: ownership.** A free cell belongs to the room whose own photos saw through it most, which removes lines of sight through doorways into the next room. Per-room frames instead cut necks narrower than 70 cm.
  4. Component with the most floor points (plus cameras).
  5. Rectangular open 0.6 m (removes doorway stubs) and close 1.0 m.
  6. Rectilinear outline (snap to x/y, merge runs, drop jogs < 15 cm).
  7. **Fill inward corner notches < 0.6 m² with both edges < 1 m** (corner furniture or unseen corners; floor plans and the tape treat them as room). Warned.
  8. Each wall = median of wall points (vertical surfaces, 0.3 m up to the ceiling) from **10 cm inside to 35 cm outside** the outline. Carved space stops at or before a wall, so the wall is outside. The median gives consensus over doubled copies.
- **Evidence:**
  - home01: bedroom 4 walls within −0.5..+1.2% / +1.1%; hall L-shape, 6 walls, measured walls within +0.7..+3.5%.
  - Synthetic ray-cast rooms (rectangle and L-shape) exact to 3 cm.
- **Costs and limits:**
  - Corner-notch filling can't tell a built-in wardrobe from a real wall jog under 0.6 m². A real small jog would be squared off (warned).
  - W1 is provisional (south-most wall) until doors are found in step 1.7.
  - Thresholds were set on one home plus synthetic rooms; the benchmark must confirm them.
