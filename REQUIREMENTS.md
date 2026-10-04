# Applied AI Engineer Case Study — Requirements Specification

> **Source:** `Applied_AI_Case_Study.pdf` (Aug 2026, 5 pages + cover)
> **Purpose:** One place to see *what* must be built, *what* the output must look like, *how* it is judged, and *what can go wrong*.

**How to read this document**

- Items tagged **[PDF]** are stated directly in the case study. Treat them as hard requirements.
- Items tagged **[Derived]** are inferred from the PDF or are standard engineering practice needed to meet a [PDF] requirement. They are recommendations, not stated rules.
- Section 18 lists everything the PDF leaves unclear. Resolve those before building.

---

## Table of Contents

1. [Summary — What We Are Building](#1-summary--what-we-are-building)
2. [Glossary](#2-glossary)
3. [Scoring — What Matters Most](#3-scoring--what-matters-most)
4. [System Overview](#4-system-overview)
5. [Inputs — Capture Route, Tiers, Devices](#5-inputs--capture-route-tiers-devices)
6. [Functional Requirements](#6-functional-requirements)
7. [Output Contract — What the Output Must Be](#7-output-contract--what-the-output-must-be)
8. [Acceptance Gates (Pass/Fail Numbers)](#8-acceptance-gates-passfail-numbers)
9. [Non-Functional Requirements](#9-non-functional-requirements)
10. [Edge Cases](#10-edge-cases)
11. [Benchmark Dataset Requirements](#11-benchmark-dataset-requirements)
12. [Head-to-Head vs a Consumer App](#12-head-to-head-vs-a-consumer-app)
13. [The Fix Loop](#13-the-fix-loop)
14. [Process Evidence](#14-process-evidence)
15. [Deliverables Checklist](#15-deliverables-checklist)
16. [The Walk-In Test](#16-the-walk-in-test)
17. [Constraints](#17-constraints)
18. [Open Questions and Ambiguities](#18-open-questions-and-ambiguities)
19. [Definition of Done](#19-definition-of-done)

---

## 1. Summary — What We Are Building

An end-to-end system that turns an **iPhone capture of a property** into a **measured, whole-property floor plan with damage assessment**.

| | |
|---|---|
| **Input** | Phone capture at one of three tiers: **Photos** (2–8 per room), **Video** (handheld walkthrough), or **LiDAR** (depth + poses + intrinsics) |
| **Processing** | One command per capture. Runs locally, without calling our own servers. |
| **Output** | Per-room dimensioned plans, one stitched whole-property plan, damage regions, concealed-damage flags, scope line items, a confidence interval on every number, JSON + a rendered plan image |
| **Proof** | A self-built benchmark with laser/tape ground truth, a head-to-head against a consumer app (e.g. Polycam, magicplan), a shipped fix with before/after runs, and a live "walk-in test" on an unseen space |

**The single most important idea:** every tier, even plain photos, must produce the **same kind of output** (including the stitched multi-room plan). Thinner input is allowed to be less accurate, but the confidence intervals must **widen honestly**. Being confidently wrong is penalised harder than being uncertain.

---

## 2. Glossary

| Term | Meaning |
|---|---|
| **Capture route** | How the data gets off the phone: either your own iOS app (Route 1) or a written protocol for an off-the-shelf app (Route 2). |
| **Tier** | Level of sensor data: Photo (lowest), Video, LiDAR (highest). |
| **Per-room plan** | A 2D floor plan of one room with wall lengths, ceiling height, floor area and openings. |
| **Stitched plan** | All rooms placed together in one whole-property floor plan, correctly connected. This is the "product surface". |
| **Adjacency** | Which rooms connect to which, and through which opening. |
| **Opening** | A door, window, or other gap in a wall. Its width is scored. |
| **Missed opening** | A real opening the system did not detect. |
| **Phantom opening** | An opening the system reports that does not exist (e.g. a mirror read as a doorway). |
| **Connector** | A hallway or passage joining rooms. |
| **Confidence interval (CI)** | A range `[lo, hi]` around each measurement that should contain the true value. |
| **Calibration** | Whether the intervals are honest: a 90% interval should contain the true value about 90% of the time. |
| **Confident garbage** | A narrow interval around a wrong value. Caps your total score. |
| **Drift** | Pose error that builds up as you walk through several rooms, which bends or misplaces the stitched plan. |
| **Loop closure / pose graph** | Common ways to correct drift by recognising previously seen places and redistributing error. |
| **Ablation** | Running the pipeline with a component on and off to show its effect. |
| **Repeatability** | Two captures of the same room at the same tier give the same result. |
| **Bias vs. unrepeatable** | Bias = consistently off by the same amount. Unrepeatable = results scatter between captures. Both fail. |
| **Damage region** | An area on a surface (wall, floor, ceiling) with a damage class and a real-world size. |
| **Concealed-damage flag** | An inferred, non-visible problem (e.g. likely leak behind a wall), reported with the rule that triggered it. |
| **Scope line item** | A repair work item (e.g. "patch drywall, 1.2 m²") tied to a specific surface. |
| **Round 1** | An earlier round of this case study. Its contract and gates still apply but are **not included in this PDF**. |
| **Defense** | The live review where you explain every design decision with tools closed. |
| **Walk-in test** | At the defense, the reviewers capture an unseen space with their own iPhone and your pipeline runs on it live. |

---

## 3. Scoring — What Matters Most

| Weight | Component | What it is designed to catch |
|---:|---|---|
| **30%** | **Walk-in test** — cold run on their capture, scored against their laser measurements | Systems that only work on the author's own data |
| **25%** | **Fix loop delta** | Shipping a diagnosis instead of an actual repair |
| **15%** | **Verified benchmark accuracy** across all three tiers | Claims that fall apart when someone reproduces them |
| **10%** | **Compliance matrix coverage** | Building a different product from the one specified |
| **10%** | **Head-to-head** vs an incumbent app (magicplan, Polycam, etc.) | Avoiding benchmarks |
| **5%** | **Capture route quality** — install time, protocol clarity, non-engineer experience | Pipelines with no path into a real user's hands |
| **5%** | **Process evidence** (git history) | Unauditable single-commit repos |

**Takeaway [Derived]:** 55% of the score (walk-in + fix loop) depends on the system generalising to unseen spaces and on actually fixing a measured problem. Robustness and honest uncertainty matter more than peak accuracy on your own data.

---

## 4. System Overview

```
 ┌───────────────────────── CAPTURE (on iPhone 15+) ─────────────────────────┐
 │  Route 1: own iOS app (ARKit/RoomPlan/LiDAR/IMU)                          │
 │     OR                                                                    │
 │  Route 2: off-the-shelf app + one-page protocol for a non-engineer        │
 │                                                                           │
 │  Tier 1: Photos (2–8 / room, one folder per room, no depth, no poses)     │
 │  Tier 2: Video  (handheld walkthrough clip)                               │
 │  Tier 3: LiDAR  (depth + poses + intrinsics, Pro devices only)            │
 └───────────────────────────────────┬───────────────────────────────────────┘
                                     │ files handed to pipeline
                                     ▼
 ┌──────────────────────── PIPELINE (one command per capture) ───────────────┐
 │  Ingest & validate → Reconstruct geometry (metric scale)                  │
 │  → Per-room layout (walls, ceiling, floor, openings)                      │
 │  → Multi-room stitching + DRIFT CORRECTION (must be switchable on/off)    │
 │  → Damage detection (class + metric extent, per surface)                  │
 │  → Concealed-damage rules → Scope line items                              │
 │  → Uncertainty estimation (interval on every number, widens with tier)    │
 └───────────────────────────────────┬───────────────────────────────────────┘
                                     ▼
 ┌──────────────────────────────── OUTPUT ───────────────────────────────────┐
 │  JSON (to the published schema)  +  rendered floor plan (per room +       │
 │  stitched whole-property plan)                                            │
 └───────────────────────────────────────────────────────────────────────────┘
```

---

## 5. Inputs — Capture Route, Tiers, Devices

### 5.1 Capture route (choose exactly one) [PDF]

No captures are provided. You own the problem from the phone's sensors onward.

| | Route 1: Own iOS capture app | Route 2: Stock capture protocol |
|---|---|---|
| **What** | Your own app using ARKit, RoomPlan, raw LiDAR depth, camera, IMU, or anything else in the SDK | An off-the-shelf tool from the App Store (LiDAR logging app, native Camera, etc.) plus a written protocol |
| **Ship as** | TestFlight build or dev build | A one-page protocol |
| **Hard limit** | Installs on the reviewers' device in **under 10 minutes** | Must be followable **literally** by a non-engineer |
| **Protocol must cover** | — | What to install · how to walk · how long · what to avoid · how to hand files to the pipeline |
| **Risk** | Signing / provisioning / install friction | Ambiguous wording. "If the page is ambiguous, the capture you get reflects that." |

### 5.2 Input tiers (all three mandatory) [PDF]

| Tier | Input | Device | Has depth? | Has poses? | Notes |
|---|---|---|:-:|:-:|---|
| **1. Photos** | **2 to 8** stills per room, **one folder per room** | Any iPhone 15 or newer | No | No | "The floor: any picture in, results out." Must still produce the **stitched whole-property plan**. |
| **2. Video** | One handheld walkthrough clip | Any iPhone 15 or newer | No | No (must be estimated) | |
| **3. LiDAR** | Depth, poses, intrinsics | Pro-class iPhones | Yes | Yes | |

All tiers produce the **same output contract** (Section 7). Only the interval widths differ.

### 5.3 Device matrix [PDF]

You must submit a matrix stating which tier runs on which hardware and the accuracy each tier **honestly** delivers. Suggested template [Derived]:

| Tier | Minimum device | Devices tested | Sensors / data used | Wall length error (measured) | Ceiling height error | Opening width error | Footprint error | Interval coverage (calibration) | Known limitations |
|---|---|---|---|---|---|---|---|---|---|
| Photo | iPhone 15 | | RGB + EXIF | | | | | | |
| Video | iPhone 15 | | RGB video (+ IMU if available) | | | | | | |
| LiDAR | iPhone 15 Pro | | Depth + ARKit poses + intrinsics | | | | | | |

---

## 6. Functional Requirements

### 6.1 Capture

| ID | Requirement | Source |
|---|---|---|
| FR-CAP-01 | Provide exactly one capture route: a Route 1 iOS app or a Route 2 one-page protocol. | [PDF] |
| FR-CAP-02 | Route 1: the app is installable as a TestFlight or dev build on the reviewers' device in under 10 minutes. | [PDF] |
| FR-CAP-03 | Route 2: name the specific off-the-shelf tool and write a one-page protocol covering install, walking pattern, duration, things to avoid, and file hand-off. | [PDF] |
| FR-CAP-04 | The capture route must support all three tiers (Photo, Video, LiDAR). | [PDF] |
| FR-CAP-05 | Submit a device matrix (tier → hardware → honest accuracy). | [PDF] |

### 6.2 Input handling

| ID | Requirement | Source |
|---|---|---|
| FR-IN-01 | Photo tier accepts a set of folders, one folder per room, each with 2–8 stills from any iPhone 15+. No depth, no poses. | [PDF] |
| FR-IN-02 | Video tier accepts one handheld walkthrough clip from any iPhone 15+. | [PDF] |
| FR-IN-03 | LiDAR tier accepts depth maps, camera poses and intrinsics from Pro-class devices. | [PDF] |
| FR-IN-04 | Validate input before processing and fail with a clear, actionable message on unusable input (empty folder, wrong format, corrupt file). | [Derived] |
| FR-IN-05 | Auto-detect the tier from the input layout, or accept it as an explicit argument. | [Derived] |

### 6.3 Geometry and per-room plan

| ID | Requirement | Source |
|---|---|---|
| FR-GEO-01 | Produce a dimensioned per-room plan with **walls** (lengths), **ceiling height**, **floor area**, and **openings**. | [PDF] |
| FR-GEO-02 | Detect openings and measure their widths. Both missed and phantom openings count as failures. | [PDF] |
| FR-GEO-03 | Recover metric scale at every tier, including photos with no depth. | [PDF] (implied by gates) |
| FR-GEO-04 | Attach a confidence interval to **every** measurement. | [PDF] |

### 6.4 Multi-room stitching and drift

| ID | Requirement | Source |
|---|---|---|
| FR-STI-01 | Produce one stitched whole-property plan with every room **placed, connected and dimensioned**. | [PDF] |
| FR-STI-02 | Adjacency (which room connects to which) must be correct. | [PDF] |
| FR-STI-03 | Rooms must not overlap in the stitched plan. | [PDF] |
| FR-STI-04 | The photo tier must stitch too: per-room photo folders → one stitched plan. A photo path that only handles single rooms fails. | [PDF] |
| FR-STI-05 | The plan should look like something a homeowner would recognise from Polycam or magicplan. | [PDF] |
| FR-DRI-01 | Implement an explicit drift-correction method for multi-room captures (loop closure, pose graph, plane-anchored correction, or similar). Using poses as-is is an automatic fail. | [PDF] |
| FR-DRI-02 | Drift correction must be toggleable so an ablation can show the stitched footprint with it **on** and **off**. | [PDF] |

### 6.5 Damage and scope

| ID | Requirement | Source |
|---|---|---|
| FR-DMG-01 | Detect per-surface damage regions with a **damage class** and **metric extent** (real-world size). | [PDF] |
| FR-DMG-02 | Support at least the two damage classes staged in the benchmark (classes are your choice, e.g. water stain, crack, hole, mould). | [PDF] + [Derived] |
| FR-DMG-03 | Emit concealed-damage flags, each stating **the rule that fired**. | [PDF] |
| FR-DMG-04 | Emit scope line items **keyed to surfaces** (each item references a specific wall/floor/ceiling). | [PDF] |
| FR-DMG-05 | Damage extent and scope quantities carry confidence intervals like every other measurement. | [PDF] ("every measurement") |

### 6.6 Execution and output

| ID | Requirement | Source |
|---|---|---|
| FR-RUN-01 | **One command per capture** produces the full output. | [PDF] |
| FR-RUN-02 | Output JSON conforms to **the published schema**. | [PDF] |
| FR-RUN-03 | Output a **rendered plan** (image or vector) for both per-room and stitched plans. | [PDF] |
| FR-RUN-04 | Record per-stage timing so the benchmark report can include it. | [PDF] (timing is a report item) |
| FR-RUN-05 | A live (non-cached) path must run end to end. Cached model outputs are allowed only alongside it. | [PDF] |

### 6.7 Evaluation tooling

| ID | Requirement | Source |
|---|---|---|
| FR-EVAL-01 | Build a benchmark set meeting the composition in Section 11. | [PDF] |
| FR-EVAL-02 | Score every gate (Section 8) at all three tiers automatically from raw inputs and ground truth. | [PDF] + [Derived] |
| FR-EVAL-03 | Produce a repeatability table. | [PDF] |
| FR-EVAL-04 | Produce a head-to-head table against one consumer app (Section 12). | [PDF] |
| FR-EVAL-05 | Measure interval calibration (coverage) at every tier. | [PDF] |
| FR-EVAL-06 | Diagnose ceiling-height errors as **biased** or **unrepeatable** and state which in the report. | [PDF] |

---

## 7. Output Contract — What the Output Must Be

### 7.1 Required contents per capture [PDF]

Every capture, at every tier, must produce **all** of the following:

| # | Output | Must include |
|---|---|---|
| 1 | **Per-room plan** | Walls (dimensioned), ceiling height, floor area, openings |
| 2 | **Stitched multi-room plan** | Every room placed, connected, dimensioned; correct adjacency; no overlaps |
| 3 | **Damage regions** | Per surface, with class and metric extent |
| 4 | **Concealed-damage flags** | Each with the rule that fired |
| 5 | **Scope line items** | Each keyed to a surface |
| 6 | **Confidence intervals** | On **every** measurement |
| 7 | **JSON** | Conforming to the published schema |
| 8 | **Rendered plan** | Human-readable floor plan |

### 7.2 Illustrative JSON shape [Derived]

> **Not the official schema.** The PDF refers to "the published schema", which is probably the Round 1 schema and is not included here. If that schema exists, it takes priority. This sketch only shows the kind of information each field must carry.

```json
{
  "schema_version": "1.0",
  "capture_id": "house01_photo_run1",
  "tier": "photo",
  "device": { "model": "iPhone 15", "os": "iOS 18.x" },
  "units": "m",
  "interval_level": 0.90,
  "drift_correction": { "method": "pose_graph_loop_closure", "enabled": true },

  "rooms": [
    {
      "room_id": "R1",
      "label": "Bedroom",
      "ceiling_height": { "value": 2.41, "lo": 2.33, "hi": 2.49 },
      "floor_area_m2":  { "value": 12.6, "lo": 11.6, "hi": 13.6 },
      "walls": [
        {
          "wall_id": "R1-W1",
          "length": { "value": 3.62, "lo": 3.40, "hi": 3.84 },
          "start": [0.00, 0.00],
          "end":   [3.62, 0.00]
        }
      ],
      "openings": [
        {
          "opening_id": "R1-O1",
          "type": "door",
          "wall_id": "R1-W2",
          "width":  { "value": 0.82, "lo": 0.78, "hi": 0.86 },
          "height": { "value": 2.03, "lo": 1.97, "hi": 2.09 },
          "connects_to": "H1"
        }
      ]
    }
  ],

  "stitched_plan": {
    "room_poses": [ { "room_id": "R1", "x": 0.0, "y": 0.0, "theta_deg": 0.0 } ],
    "adjacency":  [ { "from": "R1", "to": "H1", "via_opening": "R1-O1" } ],
    "footprint_area_m2": { "value": 68.4, "lo": 63.0, "hi": 73.9 },
    "overlaps": []
  },

  "damage": [
    {
      "damage_id": "D1",
      "surface_id": "R1-W3",
      "class": "water_stain",
      "area_m2": { "value": 0.42, "lo": 0.35, "hi": 0.50 },
      "extent_on_surface": { "u_min": 1.10, "u_max": 1.85, "v_min": 1.60, "v_max": 2.20 },
      "detection_confidence": 0.87
    }
  ],

  "concealed_damage_flags": [
    {
      "flag_id": "F1",
      "surface_id": "R1-W3",
      "rule_id": "STAIN_NEAR_CEILING_JUNCTION",
      "rule_text": "Water stain within 0.3 m of the ceiling junction suggests a leak above or behind the surface",
      "evidence": ["D1"]
    }
  ],

  "scope_items": [
    {
      "item_id": "S1",
      "surface_id": "R1-W3",
      "action": "Remove and replace drywall, seal, prime and paint",
      "quantity": { "value": 0.60, "lo": 0.50, "hi": 0.72 },
      "unit": "m2",
      "source": ["D1", "F1"]
    }
  ],

  "timing_s": { "ingest": 1.2, "reconstruction": 48.0, "layout": 6.5, "stitch": 3.1, "damage": 9.4, "total": 68.2 }
}
```

**Key rules [Derived from PDF]:**

- Every numeric measurement is an object with `value`, `lo`, `hi`, never a bare number.
- Every surface has a stable ID so damage, flags and scope items can reference it.
- `tier` and `interval_level` are always present so calibration can be scored.
- The same keys appear at every tier. A photo-tier output is never missing the stitched plan.

### 7.3 Rendered plan expectations

- One image per room plus one stitched whole-property image. [Derived]
- Walls, openings (doors/windows), room labels and dimensions are drawn. [PDF + Derived]
- Recognisable as a consumer floor plan (Polycam / magicplan style). [PDF]
- Damage regions and flags shown on, or alongside, the affected surfaces. [Derived]

---

## 8. Acceptance Gates (Pass/Fail Numbers)

> **Round 1 gates also apply** but are **not listed in this PDF**. Get the Round 1 document. See Section 18.

### 8.1 New gates added this round [PDF]

| # | Metric | Gate (must achieve) | Notes |
|---|---|---|---|
| G1 | **Opening widths** | Width error **≤ 2 cm** on **≥ 85%** of openings | Detection is scored too. A missed opening and a phantom opening each count as a miss. |
| G2 | **Ceiling height** | Error **≤ 1.5 cm** per room. If a room is captured more than once, the **spread across captures ≤ 1 cm**. | Repeatable-but-biased fails. Unrepeatable fails. The report must say which one you have. |
| G3 | **Repeatability** | Two captures of the same room at the same tier agree **within 1 cm or 0.5% per wall** | "Same room in, same plan out." |
| G4 | **Drift accountability** | The report states the drift-handling method, **plus an ablation** of the stitched footprint with it on and off | "Poses used as-is" = **automatic fail**. |
| G5 | **Photo-tier whole-property stitch** | Per-room photo folders → one stitched plan, correct adjacency, **no overlaps**, footprint **within ±8%**, calibrated intervals | Single-room-only photo path = **fail**. |

### 8.2 Per-tier accuracy targets [PDF]

| Tier | Wall length tolerance | Calibration |
|---|---|---|
| Photo | **±8%** with calibrated intervals | Scored |
| Video | **±3%** | Scored |
| LiDAR | Round 1 gates (not in this PDF) | Scored |

**Score cap:** "Confident garbage on thin input caps your total score." Narrow intervals around wrong values on photo/video input limit the whole submission.

### 8.3 Other pass conditions [PDF]

| Item | Pass condition |
|---|---|
| Head-to-head (LiDAR tier, 2 rooms) | Beat or tie the consumer app on **≥ 70%** of shared dimensions |
| Route 1 install | **< 10 min** on the reviewers' device |
| Repo setup | README → running on a fresh capture in **< 15 min** on a clean machine |
| Technical report | **≤ 6 pages** |

### 8.4 How to compute the gates [Derived, confirm against Round 1 rules]

| Gate | Suggested computation |
|---|---|
| G1 | `pass_rate = (# GT openings detected with |width error| ≤ 2 cm) / (# GT openings + # phantom openings)`, must be ≥ 0.85 |
| G2 | Per room: `|mean(pred) − truth| ≤ 1.5 cm` **and**, for repeated rooms, `max(pred) − min(pred) ≤ 1 cm` |
| G3 | Per wall pair: `|L1 − L2| ≤ max(1 cm, 0.5% × L_truth)` (the "or" is read as the looser of the two; confirm) |
| G5 | `|footprint_pred − footprint_truth| / footprint_truth ≤ 8%`, adjacency graph equals the true graph, overlap area = 0 |
| Calibration | Coverage = fraction of measurements whose `[lo, hi]` contains ground truth. It should be close to `interval_level` at every tier. |

---

## 9. Non-Functional Requirements

| ID | Category | Requirement | Source |
|---|---|---|---|
| NFR-01 | Install time | Route 1 app installs on the reviewers' device in < 10 minutes | [PDF] |
| NFR-02 | Setup time | README to running on a fresh capture in < 15 minutes on a clean machine | [PDF] |
| NFR-03 | Simplicity | One command per capture | [PDF] |
| NFR-04 | Self-contained | Runs without calling **your own** infrastructure. Third-party pretrained models, datasets or APIs are allowed **with disclosure**. | [PDF] |
| NFR-05 | Artifacts | Weights and large binaries are fetched by script or volume, not committed to the repo | [PDF] |
| NFR-06 | Reproducibility | Every reported number can be regenerated from raw inputs on the reviewers' machine | [PDF] |
| NFR-07 | Determinism | Cached model outputs are allowed only if the cache replays **deterministically** and the live path also runs | [PDF] |
| NFR-08 | Honest uncertainty | Intervals widen as sensor data thins. Calibration is scored at every tier. | [PDF] |
| NFR-09 | Robustness | Handles mirrors, glass, wet-look surfaces and low light, and the submission documents how | [PDF] |
| NFR-10 | Generalisation | Runs cold on an unseen space captured by someone else on their own iPhone 15+ | [PDF] |
| NFR-11 | Hardware | Photo and video on any iPhone 15+. LiDAR on Pro-class devices. | [PDF] |
| NFR-12 | Capture method | Handheld consumer capture only. No tripods, rigs or special hardware. | [PDF] |
| NFR-13 | Repeatability | Same room in → same plan out (G3). Fix random seeds and avoid non-deterministic ops where possible. | [PDF] + [Derived] |
| NFR-14 | Explainability | Concealed-damage flags state the rule that fired | [PDF] |
| NFR-15 | Performance | Timing is reported. The walk-in test runs live while reviewers laser-measure the room, so runtime should be minutes, not hours. | [PDF] + [Derived] |
| NFR-16 | Usability | The capture protocol and app work for a non-engineer | [PDF] |
| NFR-17 | Auditability | Incremental git history that could belong to the person who built it | [PDF] |
| NFR-18 | Ownership | Every design decision can be defended live with tools closed | [PDF] |
| NFR-19 | Documentation | Technical report ≤ 6 pages | [PDF] |
| NFR-20 | Portability | Pin dependencies (lockfile or container) so a clean machine reproduces results | [Derived] |
| NFR-21 | Offline readiness | Prefer a fully local path for the walk-in test, since venue network access is not guaranteed | [Derived] |

---

## 10. Edge Cases

Each row gives the situation and the **expected behaviour**. The general rule across all of them: **degrade gracefully, widen intervals, never be confidently wrong, and never crash silently.**

### 10.1 Input and file handling

| Edge case | Expected behaviour |
|---|---|
| Room folder with **1 photo** (below minimum) | Reject with a clear message, or process with very wide intervals and a warning. Never crash. |
| Room folder with **more than 8 photos** | Accept (use all, or select the best 8) and log it. |
| **Empty** room folder | Clear validation error naming the folder. |
| Non-image or **corrupt** files in a folder | Skip with a warning and continue if ≥ 2 valid photos remain. |
| **HEIC** vs JPEG vs ProRAW | All supported. |
| **EXIF stripped** (no focal length) | Fall back to estimated intrinsics and widen intervals. |
| Photos from **different iPhones** in one capture | Per-image intrinsics, not one shared camera. |
| **0.5× ultrawide** or 2×/3× telephoto lens | Handle the different intrinsics and distortion per image. |
| Portrait/landscape **orientation** mixed | Respect the EXIF orientation flag. |
| Arbitrary **folder names** ("kitchen", "room 2", "IMG_folder") | No reliance on names for adjacency. Use names only as labels. |
| Video **too short** / very long | Short: warn and widen intervals. Long: subsample frames and stay within the time budget. |
| Video **orientation change** mid-clip | Handle the rotation metadata. |
| LiDAR export from a **different app version** | Validate the expected files and give a clear error if the format differs. |

### 10.2 Photo tier (no depth, no poses)

| Edge case | Expected behaviour |
|---|---|
| **No absolute scale** in monocular photos | Recover scale from priors or learned depth (standard door heights, ceiling heights, known objects) and widen intervals to reflect the scale uncertainty. |
| Photos with **little overlap** within a room | Still produce a layout with wider intervals, or flag low coverage. |
| **Corners or walls never photographed** | Infer from the room's shape assumptions, mark low confidence, widen intervals. |
| **Stitching rooms with no shared poses** | Infer adjacency from shared openings, doorway views into the next room, and matching door dimensions. |
| Two rooms with **identical-looking doors** | Do not mis-pair them. Use dimensions, wall position and visual context. Lower confidence if ambiguous. |
| **Connector / hallway** as its own folder | Treated as a room and placed in the stitched plan. |
| Stitch produces **overlapping rooms** | Must be resolved. Overlap is a gate failure (G5). |

### 10.3 Video tier

| Edge case | Expected behaviour |
|---|---|
| **Motion blur** / fast walking | Drop blurry frames and widen intervals if coverage is lost. |
| **Auto-exposure** jumps between rooms | Robust features and matching. |
| **Rolling shutter**, video stabilisation, variable frame rate | Account for changed effective intrinsics. Don't assume a fixed frame rate. |
| **HDR / Dolby Vision** encoding | Decode correctly and tone-map before processing. |
| Walkthrough **revisits** a room | Use the revisit for loop closure (drift correction). |
| **Tracking loss** mid-clip | Re-localise, or split into segments and re-join. Never output a silently broken plan. |

### 10.4 LiDAR tier

| Edge case | Expected behaviour |
|---|---|
| Room larger than the **LiDAR range** (about 5 m) | Fuse multiple viewpoints, fall back to visual geometry for far walls, widen intervals. |
| **Dark / absorbent** surfaces with sparse depth | Fill from neighbouring planes and widen intervals locally. |
| **ARKit pose drift** across many rooms | Drift correction required (G4). Poses as-is fail. |
| ARKit **relocalisation jump** | Detect discontinuities and do not trust the jump blindly. |
| Reviewer's iPhone is **not Pro** but LiDAR tier is chosen | Clear message that LiDAR needs a Pro device. The device matrix states this up front. |

### 10.5 Environment (explicitly required by the PDF)

| Edge case | Risk | Expected behaviour |
|---|---|---|
| **Mirrors** | Phantom rooms, phantom openings, walls placed too far away | Detect mirror regions, suppress reflected geometry, do not report phantom openings. |
| **Glass** (windows, glass doors, shower screens) | LiDAR/depth passes through, so walls or openings are missed | Infer the wall plane from frames and edges and still report the opening. |
| **Wet-look / glossy surfaces** (tile, polished floors) | Specular reflections, noisy depth, false damage | Robust plane fitting. Don't classify reflections or highlights as damage. |
| **Low light** | Noise, blur, failed matching | Widen intervals and warn. The protocol should tell users to turn lights on. |

### 10.6 Room geometry

| Edge case | Expected behaviour |
|---|---|
| **Non-rectangular** rooms (L-shape, angled walls) | Support polygon layouts, not just boxes. |
| **Curved** walls | Approximate with segments and report the approximation. |
| **Sloped / vaulted** ceilings, beams, soffits | Report how ceiling height was defined (e.g. min/max/main plane) and keep it consistent across captures. |
| **Furnished** room, furniture hiding walls and corners | Estimate occluded walls and widen intervals. Don't treat furniture as walls. |
| **Open-plan** spaces with no door between areas | Consistent rule for splitting rooms. Represent the open boundary as an opening/adjacency. |
| **Closets, alcoves, built-ins** | Consistent rule on whether they count as floor area. Document it. |
| **Stairs / multiple floors** | Not explicitly covered by the PDF. At minimum detect and warn. See open questions. |

### 10.7 Openings

| Edge case | Expected behaviour |
|---|---|
| Door **open vs closed** | Detect the opening either way and measure the frame width, not the leaf. |
| **Sliding / pocket / double** doors | Width = full opening. |
| **Archway** with no door | Counts as an opening. |
| **Windows** | Detected as openings (type = window). |
| **Mirrors, large art, TVs, dark wardrobes** | Must not become phantom openings (each phantom counts as a miss in G1). |
| Opening **partly hidden** by furniture | Detect it, with a wider interval on width. |

### 10.8 Damage

| Edge case | Expected behaviour |
|---|---|
| Damage **spanning two surfaces** (wall–ceiling corner) | Split into per-surface regions, each keyed to its surface. |
| **Shadows, dirt, wall art, wallpaper patterns** | Not classified as damage. |
| **Small damage** below image resolution | Report with low confidence or not at all. Don't invent extent. |
| Damage **partly occluded** | Report the visible extent with a wider interval. |
| **Two damage classes in one room** (staged benchmark) | Both detected and correctly classified. |
| No damage present | Empty damage list. No false flags. |
| Concealed-damage rule fires on weak evidence | Flag still lists the rule and the evidence, so a human can judge. |

### 10.9 Repeatability and determinism

| Edge case | Expected behaviour |
|---|---|
| Same room captured twice from **different start points or walk directions** | Wall lengths agree within G3. |
| Same room captured under **different lighting** | Still within G3. |
| **Same input run twice** | Bit-identical, or numerically identical, output. Fix seeds. |
| Cached vs live run | Same numbers, within numerical tolerance. |

### 10.10 Walk-in test (operational)

| Edge case | Expected behaviour |
|---|---|
| Space is **unlike anything in your benchmark** | Pipeline still runs. Intervals reflect the uncertainty. |
| Reviewers follow the protocol **literally**, including ambiguous lines | Protocol is written so there is only one way to read it. |
| Reviewers pick **any** of the three tiers on the day | All three tiers ready and tested end to end. |
| **No internet** at the venue | Local path works. Weights pre-fetched. |
| Newer **iOS version** than you tested | Capture route still works. Avoid fragile private APIs. |
| Pipeline **crashes** mid-run | Clear error and partial output rather than nothing. Ideally never happens (test cold runs repeatedly). |

---

## 11. Benchmark Dataset Requirements

You build the benchmark yourself. Its composition is fixed so it cannot be flattered. [PDF]

| # | Requirement | Purpose |
|---|---|---|
| B1 | **One multi-room capture:** 3+ rooms **plus a connector** (hallway/passage) | Stitching, adjacency, drift |
| B2 | **One furnished room with staged damage** spanning **two damage classes** | Damage detection, occlusion |
| B3 | **The same rooms captured at all three tiers**, including the multi-room set. At the photo tier it arrives as per-room folders and must still stitch. | Cross-tier comparison, G5 |
| B4 | **At least one room captured twice at the same tier** | Repeatability (G2 spread, G3) |
| B5 | **Laser or tape ground truth on everything**. Submit raw sensor data and measurements. | Verifiable accuracy |

**Ground truth to record per room [Derived]:** every wall length, ceiling height (at several points), floor area (or the dimensions to compute it), every opening width and height, damage region extents, and the true adjacency graph.

**Benchmark report must include [PDF]:** gates at all three tiers, repeatability table, head-to-head table, timing.

---

## 12. Head-to-Head vs a Consumer App

| Item | Requirement [PDF] |
|---|---|
| Rooms | 2 of your benchmark rooms |
| Your side | Your pipeline at the **LiDAR** tier |
| Their side | One consumer scanning app of your choice (e.g. Polycam, magicplan). The free tier is fine. |
| Disclose | App name and version. Submit its export. |
| Format | **One table**: your error vs theirs, dimension by dimension |
| Pass | Beat or tie on **≥ 70%** of shared dimensions |
| Not accepted | Cost as an excuse ("free tiers exist") |

Suggested table [Derived]:

| Room | Dimension | Ground truth | Ours | Our error | App | App error | Result (win/tie/loss) |
|---|---|---|---|---|---|---|---|

---

## 13. The Fix Loop

**Worth 25% of the total score.** [PDF]

### 13.1 Steps

1. **Write a one-page fix declaration** covering:
   1. The **single worst-performing gate** in your own benchmark, with the failing number.
   2. Your **root-cause hypothesis** and the evidence for it.
   3. The **fix** you intend to ship and the **number you predict** after it.
2. **Ship the fix.**
3. **Submit:** the before run and the after run (**both regenerable by the reviewers**) plus a **readable diff**.

### 13.2 Scoring

| Outcome | Marks |
|---|---|
| Correct root cause + shipped fix + gate moves **fail → pass** | **Full** |
| Correct root cause + shipped fix + meaningful movement **short of the gate**, and the report explains why | **Majority** |
| Prediction **badly wrong** in either direction | Marks for an honest post-mortem only. **None for the prediction.** |
| Analysis with **no shipped fix** | **Zero**, regardless of quality |
| Fix with **no regenerable before/after** | **Zero** |

**Practical note [Derived]:** tag the "before" commit in git so the before run can be regenerated exactly. Keep the fix in a focused commit or PR so the diff is readable.

---

## 14. Process Evidence

| Rule [PDF] | Meaning |
|---|---|
| Commit as you work. The history is read. | Incremental, meaningful commits. |
| A repo that appears fully formed in 1–2 commits at the deadline scores **zero** on this part | The narrative report is then also read "with proportional suspicion". |
| Commits per day are not counted | They check the history *could belong to the person who built it*. |
| AI coding tools are **allowed and expected** | The defense settles it: every design decision is defended live, **tools closed**. |

---

## 15. Deliverables Checklist

| # | Deliverable | Contents [PDF] | Done |
|---|---|---|:-:|
| 1 | **Compliance matrix** | requirement → file path → artifact → status | ☐ |
| 2 | **Capture route** | TestFlight/dev build **or** the one-page stock-capture protocol, **plus the device matrix** | ☐ |
| 3 | **Repo** | README → running on a fresh capture in < 15 min on a clean machine. One command per capture. | ☐ |
| 4 | **Reproduction bundle** | Everything needed to regenerate every reported number from raw inputs. Deterministic cache allowed only if the live path also runs. | ☐ |
| 5 | **Benchmark report** | Gates at all three tiers, repeatability table, head-to-head table, timing | ☐ |
| 6 | **Fix loop bundle** | Declaration, before run, after run, readable diff | ☐ |
| 7 | **Technical report (≤ 6 pages)** | Architecture, tier design and device matrix, drift handling, error budget, calibration analysis, fix loop story, known failure modes | ☐ |
| 8 | **Raw benchmark data** | Sensor logs, ground truth, app exports | ☐ |

### 15.1 Compliance matrix template [Derived]

| Req ID | Requirement | File path | Artifact | Status |
|---|---|---|---|---|
| FR-STI-04 | Photo tier stitches per-room folders into one plan | `pipeline/stitch/photo_stitch.py` | `bench/house01_photo/plan_stitched.png` | ✅ Met |
| G1 | Opening widths ≤ 2 cm on ≥ 85% | `eval/gates/openings.py` | `reports/benchmark.md#g1` | ⚠️ Partial (81%) |

Suggested status values: ✅ Met · ⚠️ Partial (with number) · ❌ Not met · ➖ N/A (with reason).

---

## 16. The Walk-In Test

**30% of the total score — the largest single item.** [PDF]

| Aspect | Detail |
|---|---|
| Space | One you have **never seen** |
| Device | **Their own** iPhone 15 or newer |
| Tier | **They choose on the day.** All three must be ready. |
| Capture | They follow **your capture route exactly** (literally, for Route 2) |
| Run | Your pipeline runs **cold, in front of them** |
| Scoring | They laser-measure the space **while the pipeline runs** and score your output on the spot |

**Readiness checklist [Derived]:**

- ☐ Fresh-machine install rehearsed end to end, timed
- ☐ Weights pre-fetched. Offline run verified.
- ☐ Each tier tested on a space not in the benchmark
- ☐ Someone else (a non-engineer) has followed the protocol and the result was processed successfully
- ☐ Runtime per tier known and acceptable
- ☐ Failure messages are clear if input is bad

---

## 17. Constraints

| Constraint [PDF] |
|---|
| Handheld consumer capture only |
| Any pretrained model, dataset or API is allowed **with disclosure** |
| Everything runs **without calling your own infrastructure** (same as Round 1) |
| Weights and large binaries are fetched by script or volume |
| Real properties contain **mirrors, glass, wet-look surfaces and low light**. Your submission must cover them. |
| Technical report max **6 pages** |

---

## 18. Open Questions and Ambiguities

Resolve these before building, or state your assumption in the technical report.

| # | Question | Why it matters | Suggested assumption if unanswered |
|---|---|---|---|
| Q1 | **What are the Round 1 gates?** The PDF says they "apply" but does not list them. | LiDAR-tier wall-length tolerance, floor area tolerance and others are undefined here. | Get the Round 1 document. |
| Q2 | **What is "the published schema"?** | The JSON must conform to it. | Use the Round 1 schema if it exists. Otherwise publish your own JSON Schema in the repo. |
| Q3 | **Which damage classes** are in scope? | Detection model and scope items depend on it. | Pick two common, visible classes (e.g. water stain + crack/hole) and document them. |
| Q4 | **What are the concealed-damage rules?** | Flags must state "the rule that fired". | Define a small, documented rule set (e.g. ceiling stain under a wet room → possible leak above). |
| Q5 | **Scope line item format** (units, costing, Xactimate-style codes?) | Defines output shape. | Action + surface ID + quantity + unit, no pricing. |
| Q6 | **Confidence level** of intervals (80%? 90%? 95%?) | Calibration is scored against it. | 90%, stated in the JSON (`interval_level`). |
| Q7 | **Repeatability "1 cm or 0.5%"**: the looser or the stricter of the two? | Changes pass/fail on long walls. | Looser: `max(1 cm, 0.5% × length)`. |
| Q8 | **Opening-score denominator** with phantoms | Changes the G1 rate. | `correct / (true openings + phantoms)`. |
| Q9 | **Ceiling height definition** for sloped or uneven ceilings | Affects G2 and repeatability. | Dominant ceiling plane height above the dominant floor plane. Document it. |
| Q10 | **Multi-storey properties** in scope? | "Whole-property" could imply stairs and floors. | Single storey. Detect stairs and warn. |
| Q11 | **Time budget** for the live walk-in run | Must finish "while they measure". | Target a few minutes per room on a laptop. Report actual timing. |
| Q12 | **Machine spec** of the reviewers' "clean machine" (OS, GPU or not?) | Affects model choice and the 15-minute setup. | Assume a laptop, possibly without a CUDA GPU. Support CPU/Apple Silicon. |
| Q13 | **Network access** at the defense? | Third-party APIs allowed, but risky live. | Assume offline. Keep a local path. |
| Q14 | **When** is the fix declaration submitted (before the fix, or with the final submission)? | "Acknowledge" may mean it is time-stamped before the fix. | Commit it to the repo before starting the fix. Git history proves ordering. |
| Q15 | **Photo tier: "any iPhone 15 or newer"** includes non-Pro models. Can photos contain anything beyond pixels + EXIF? | Rules out depth from Portrait mode, etc. | Use RGB + EXIF only. |
| Q16 | **Floor area**: include closets, alcoves, under built-ins? | Affects the floor area value. | Include walkable floor inside the wall polygon. Document it. |

---

## 19. Definition of Done

The submission is complete when **all** of the following are true:

**Product**
- ☐ All three tiers (Photo, Video, LiDAR) run end to end with **one command per capture**
- ☐ Every tier outputs the **full contract**: per-room plans, stitched plan, damage, concealed flags, scope items, intervals on everything, JSON, rendered plan
- ☐ Photo tier **stitches** per-room folders into one whole-property plan with no overlaps
- ☐ Drift correction implemented, **toggleable**, with an on/off ablation
- ☐ Mirrors, glass, wet-look surfaces and low light handled and documented

**Accuracy and honesty**
- ☐ G1–G5 measured at all applicable tiers, plus Round 1 gates
- ☐ Photo wall lengths within ±8%, video within ±3%, with calibrated intervals
- ☐ Calibration (interval coverage) reported per tier. No confident garbage.
- ☐ Ceiling-height error diagnosed as bias or variance
- ☐ Head-to-head: beat or tie on ≥ 70% of shared dimensions vs a named app and version

**Fix loop**
- ☐ Declaration written (worst gate, root cause plus evidence, fix, predicted number)
- ☐ Fix shipped. Before and after runs both regenerable. Readable diff.

**Reproducibility and delivery**
- ☐ Fresh-machine setup < 15 min, verified
- ☐ Route 1 install < 10 min, **or** Route 2 one-page protocol tested by a non-engineer
- ☐ Weights fetched by script. Live path runs. Cache replays deterministically.
- ☐ All 8 deliverables present (Section 15)
- ☐ Technical report ≤ 6 pages
- ☐ Raw data, ground truth and app exports submitted

**Process**
- ☐ Git history shows incremental work
- ☐ You can explain and defend every design decision without tools
