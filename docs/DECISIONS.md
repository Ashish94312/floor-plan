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

## D24. Output contract, interval model v1, and stitched plan from the joint frame
- **Schema** (`scan/schema.py`, the published `schema/scan_output.schema.json`):
  - Every number is a `Measurement` (value, lo, hi, unit) with a validator that lo ≤ value ≤ hi. Unknown keys are rejected.
  - Stable surface ids: `<room>-W<n>`, `-F` (floor), `-C` (ceiling).
  - Keys produced by later steps (openings, damage, flags, scope) are present as empty lists, so the shape never changes between tiers or versions (REQUIREMENTS §7.2 key rules).
- **Interval model v1:** ARCHITECTURE §11 with σ_fit taken from each wall's **measured spread** rather than the standard error of the median. Views disagree by a few cm, so the spread is the honest position uncertainty. k_tier = 1.5 until calibration (step 1.10), which gives intervals about ±13% while observed errors are 1–3%. That's wide on purpose: the requirements penalise confidently-wrong outputs harder than wide ones.
- **Footprint interval:** in a joint frame all rooms share one metric scale, so their scale terms add linearly (fully correlated). With separate runs they add in quadrature.
- **Stitching (photo tier, joint):**
  - The joint reconstruction *is* the placement (poses = identity in the shared frame).
  - Adjacency = parallel walls ≤ 35 cm apart (wall thickness + error) overlapping ≥ 50 cm. `via_opening` is filled once doors are detected (step 1.7).
  - Separate runs → `unplaced`, until door-based stitching (step 1.8).
- **Output layout:** deliverables at `out/` (`result.json`, `plan.png`, `plan.svg`, `rooms/<room>.png`); diagnostics in `out/debug/`.

## D25. Photo-tier drift correction = joint frame + structural wall snapping
- **Why:**
  - In a joint run every room is placed, but each room's walls are fitted separately. The same physical wall can show up as two faces with a gap between them (kitchen ↔ hall, 20–24 cm), or two rooms' outer walls a few cm off one line.
  - The user confirmed what the building really is: one partition between kitchen and hall, one north wall line, one west wall line.
- **Rule:**
  - Every wall line is one face of a physical wall whose centre is t/2 behind it (t = 0.12 m, O5).
  - Faces are linked when their centre estimates agree within 25 cm, and either they face each other with overlapping spans (a shared partition) or they face the same way side by side, within 50 cm along the line (one outer wall).
  - **The best-seen face (most supporting points) sets each group's line**; the other faces snap to it. Polygons, lengths and areas are rebuilt.
- **How the anchor rule was chosen (be transparent):**
  - A support-weighted **mean** dragged the well-seen bedroom north wall +8.4 cm toward short, poorly-seen faces.
  - A weighted **median** was decided by a hair (bedroom 11,597 of 23,653 points) and picked the hall's 0.98 m passage-end wall, which the tape shows is over-extended. Hall W1 tape 361 means the passage end sits at 2.31; it was at 2.43, while the bedroom wall was at 2.29.
  - **Max support** keeps the best-seen wall. Result: hall W1 3.615 (tape 361, +0.1%, was +2.9%); bedroom 2.922 / 2.429 (+0.2% / +1.6%); every measured wall within 1.6%.
  - This choice was informed by **one tape check** and must be validated on the benchmark.
- **General, but thresholds from one home:**
  - No flat-specific values are in the code: the same rules apply to any capture.
  - Risk: a real 10–25 cm step between collinear outer walls would be wrongly flattened.
  - **Ablation:** `--no-drift-correction` turns snapping off (G4). On home01: footprint 20.30 m² off against 20.06 m² on; 0 overlaps both ways; the kitchen–hall gap is present only with snapping off.

## D26. Open boundaries between rooms (open kitchen, archway) are detected from wall coverage
- **Why:** the layout outputs closed outlines, so every edge was drawn as a wall. home01's kitchen is **open** to the hall's passage (user). Its edge there has no physical wall, so drawing one is wrong in the plan, in wall counts and in scope items (no wall to repaint).
- **Rule:**
  - **Coverage** of an edge = share of its length (10 cm bins) with ≥ 5 vertical-surface points within 12 cm of the line.
  - An edge is `open` only if its coverage is < 50% **and** the other room's face on the same boundary is also < 50%: both rooms saw through it.
  - Measured on home01: real walls 0.76–1.0. Open side: hall 0.42, kitchen 0.27. The kitchen ↔ hall partition: kitchen face 0.04 (its 2 photos never looked at it) but hall face 1.00, so it stays a wall.
  - A low-coverage outer wall with no room behind it is *poorly seen*, not open.
- **Geometry:** an open boundary has no thickness. Both rooms' faces move to their midline (`merge_open_boundaries`) and the polygons are rebuilt, so the rooms meet at one dashed line.
- **Schema:** `Wall.kind: wall | open` (default `wall`). Ground truth uses the same field (hall W5, kitchen W1 open).
- **Limits:**
  - Partly open edges (a wide opening in a short wall) are all-or-nothing for now. Openings inside walls (doors, windows) come in step 1.7.
  - The 50% threshold was set on one home.

## D27. Joint-run rooms share one floor; ceiling mismatches are warned, not overwritten
- **Why:** user: connected rooms let you work out each other's heights. The data showed the kitchen's ceiling was right (2.76 m, matching hall 2.80 / bedroom 2.83 peaks), but its 2 photos never saw the floor. The counters hide it, and the lowest surface found was 0.20–0.29 m up. That made the kitchen 2.52 m tall.
- **Rule:**
  - In a joint frame the floor z = 0 is fitted from **all** rooms' floor points (step 1.3).
  - A room whose own floor estimate is > 8 cm away from it didn't see its floor, so its ceiling height is measured from the shared floor, with a warning.
  - Ceilings are **not** copied between rooms, because kitchens and bathrooms can genuinely be lower. A room > 10 cm off its frame-mates' median gets a warning instead.
- **Result:** kitchen ceiling 2.52 → **2.741 m** (bedroom 2.769, hall 2.803). No mismatch warnings.
- **Limit:** assumes one floor level per joint run. A step down between rooms of more than 8 cm (e.g. a sunken bathroom) would be flattened. The warning says so.

## D28. Openings by visibility voting on raw depth, not detector boxes
- **Why:**
  - G1 asks for widths within 2 cm, and detector boxes are loose.
  - Point-cloud "see-through" evidence is mostly missing: the confidence filter drops far pixels, and through-door views are far.
  - A wall hole alone can't be told apart from furniture hiding the wall.
- **How (`layout/openings.py`):**
  - Every 2 × 5 cm cell on a wall is projected into every photo, and that photo's **raw** depth votes: wall (depth ≈ wall distance ±15 cm), through (beyond: an opening), or nothing (in front: furniture).
  - A cell is open if ≥ 60% of its votes say through.
  - Open regions → width from jamb to jamb: extend outward through columns with no wall evidence, stopping at the first column voting wall.
  - Type: from the floor with a lintel → door; no lintel → opening; wall below → window (≥ 0.5 × 0.5 m, ≥ 2 photos).
  - `connects_to` = the room whose wall face spans the opening's centre.
  - **W1 = main-door wall** (D11): the widest door out of the captured rooms, else the widest door, else an open side. Plan and ground-truth numbering now agree in every home01 room (eval rotation shift 0).
- **Evidence:**
  - Synthetic ray-cast room: door 0.90 m exact, window 0.90 × 1.20, sill 0.90 exact.
  - home01: bedroom door found from both rooms (0.72 × 2.00), plus a second hall door on W1 (probably the front door; the ground truth says W3, to confirm).
  - **Both doors read 18–19 cm narrower than the tape, consistently.** Evidence on both sides of the edges agrees (votes flip at u = 0.15 / 0.86; wall-face points start at 0.88). So it's likely a definition difference (tape = frame outer edges vs clear opening). The user is to re-measure the inside faces of the frame.
- **Limits:**
  - Closed doors and curtained or glass windows vote wall: the bedroom window was missed. They need a detector (OWLv2), still to do.
  - Mirrors would vote "through" (phantom).
  - A high-confidence view through an exterior opening can leak the free-space carving outside (seen in the synthetic room with uniform confidence). Mitigated on real data by the confidence filter. Planned fix: re-carve clipped at the first layout's walls.

## D29. Damage, closed doors and windows from OWLv2, kept only when seen in 2+ photos
- **Why (E18):** at threshold 0.2, OWLv2 on 17 photos fires dozens of water stains and cracks, mostly on **marble floors** and the marble kitchen splashback (veins look like stains and cracks), and once calls the wardrobe a "door". Real objects (the bedroom window in 4 photos, the closed sticker door in 3) are consistent across photos; false positives mostly appear once.
- **Rule:**
  - One OWLv2 pass per photo (1024 px, cached). Each box is lifted onto its surface: centre depth → nearest wall within 25 cm, or floor/ceiling; corners → rays onto that plane → metric extent.
  - The 1024 px → depth-map mapping (scale max(Wd/W, Hd/H) + centre crop) was verified to 1 px against MapAnything's own image.
  - Boxes on the same surface overlapping > 30% are merged. **Keep only groups seen in ≥ 2 different photos.**
  - **Damage on walls and ceilings only** for now (floors excluded: marble).
  - Detector doors/windows are added only where the depth voting found no opening, and only after W1 is fixed (a closed bathroom door must not become the "main door"). Doors must reach the floor and be door-sized.
  - Mirrors remove see-through openings they cover.
  - Rules (`config/rules.yaml`) → concealed flags + scope items, each naming its rule.
- **home01:**
  - Window added on bedroom W3 (0.69 × 1.06, 4 photos; tape 0.90 × 1.21, sill 0.93: the box hugs the glass).
  - Sticker door added on hall W3 (0.93 × 1.93, 3 photos).
  - **0 damage** (no wall/ceiling damage in 2+ photos; the staged stain is in no photo).
- **Limits:**
  - Box extents are loose (intervals widened).
  - Single-photo damage is dropped (could miss damage seen once).
  - Floor damage is not supported on patterned floors.
  - Thresholds were tuned on one home (O4).

## D30. Interval calibration: split-conformal k per mode and measurement type, with small-sample guards
- **Why:** with k = 1.5, intervals were about ±15% wide while observed errors were 0.1–1.6% (coverage 100%). They were honest but uninformative.
- **Rule (`uncertainty/calibrate.py`, `uv run scan-calibrate <eval.json>…` → `config/calibration.yaml`):**
  - For each scored measurement: r = |error| / half-width used, converted to units of z·σ by multiplying by the k used.
  - New k = the ⌈(n+1)·q⌉-th smallest value. This is split-conformal: coverage ≥ q if future rooms resemble these.
  - Fitted separately per **tier_mode** (`photo_joint` vs `photo_per_room`: separate runs have no shared scale and bigger errors) and per **type** (wall, ceiling, area).
  - **Guards:** if n is too small for that order statistic, use max r × 2. Never go below k = 0.5.
  - Leave-one-room-out coverage is reported.
- **Excluded:** opening widths stay at the default k while their −18 cm looks like a tape-definition issue (D28). Their coverage is still reported (currently missed).
- **home01:**
  - n = 7 walls / 2 ceilings / 1 area: all below the conformal minimum (9), so every k = 0.5 (floor).
  - Max ratio at k = 1.5: walls 0.115, ceilings 0.066, so the floor leaves a 3× margin.
  - LOO coverage 100% (7/7, 2/2).
  - **Mean half-width ±15% → ±5.0%, coverage still 100%.**
  - The result records the k used (`software.interval_k`, `interval_mode`); eval and calibration read it from there.
- **Limit:** one home. The benchmark (Phase 4) must refit with more captures, and the floor should come down only when n ≥ 9 per type.

## D31. Separately reconstructed rooms are placed by visibility-scored door matching
- **Why:**
  - Homes above `max_joint_views` (20 photos) run room by room, each in its own frame. Every room is already levelled and squared (step 1.3), so placing B in A's frame is a 90° multiple plus a shift. A door pair fixes both: B's door wall faces A's, and the centres sit one wall thickness apart.
  - Rooms often have several doors of similar width (home01 hall: bedroom 0.72, front 0.70). Width alone can't choose.
- **Rule (`stitch/doors.py`):**
  - Candidate pairs: widths within 15 cm; outlines must not overlap.
  - **Score = visibility.** Raw depth seen *through* A's door (rays crossing the door segment, landing > 25 cm beyond it) should fall inside B's outline (+25 cm), and vice versa. Accept ≥ 0.5.
  - Spanning tree from the room with the most photos, best scores first.
  - Placed rooms then go through the same snapping / open-side / door steps as a joint run, restricted to the rooms sharing the main frame.
  - Rooms with no door match stay `unplaced` (warning, G5 fails).
- **Evidence:**
  - Synthetic: real door chosen over a same-width decoy; rotation exact, shift within 8 cm.
  - home01 `--per-room`: bedroom ↔ hall matched through the bedroom door (score 0.88, not the front door); 0 overlaps; walls 7/7 within ±8%.
  - But accuracy is worse than joint: bedroom +5.5–5.7%, from per-room scale, which confirms D20.
  - The kitchen stays unplaced: it opens onto the hall without a door, so there's nothing to match.
- **Not done yet:**
  - Least-squares loop closure (rooms connected by more than a tree), so the per-room drift correction is currently tree + wall snapping.
  - Matching open sides (no door).
- **Known risk (found here):** in per-room runs, free-space carving can **leak through doorways**. The protocol's door shot looks straight through the door, and if that depth is confident, the room's outline extends into the next room (seen in the synthetic room: wrong walls, door widths 2.2 m). Joint runs are protected by ownership; real home01 data was protected by low through-door confidence.
  - Fix direction: drop carved regions reached only through a detected doorway. A plain larger neck cut would erase narrow room parts such as the hall's 0.98 m passage.

## D32. Video size: calibrated across taped captures, not fitted per capture (E22o–E22q)
- **Why:** after the camera re-solve (geometry.repose) a video room's shape and its frames' relative scales come from geometry, but its overall size can only come from the model's size sense (frame votes f_true / f_model). That carries an error geometry cannot see. Fitting it to one capture's tape would be overfitting; the user asked for a method that holds when data and rooms change.
- **Model:** b = log(tape / plan) = mu + e_room. After the re-solve a room's error is one isotropic scale, so walls and ceilings measure the same b. Measurements of a room share its e, and captures of one physical room share content, so: room = median of its measurements (only rooms with >= 3); mu = median over physical rooms; tau = RMS of rooms around mu.
- **Rule (`uncertainty/calibrate.py`, `scan-calibrate --bias-only <eval.json>…`):** learnt only for `*_repose` modes (without the re-solve the error is a squeeze along the line of sight, not a scale); merged into `config/calibration.yaml` as `scale_bias`; the pipeline multiplies re-solved video by the factor and sets the room's scale uncertainty to at least tau. Interval widths (D30) are refitted afterwards on corrected runs.
- **home01 (video_a–d):** x1.017, tau 0.062, 4 rooms / 2 physical. Leave-one-physical-room-out: corrections change errors by about 1 point either way.
- **Conclusion:** no systematic bias worth removing; video room size is good to about +-6% (1 sigma) per room, content-dependent. Below the +-3% video wall tolerance only with better footage or an external reference. More taped flats refine mu and tau with no code change.
