# Fix Loop: Photo Scale Follows the Model's Orientation-Dependent Focal

**Declaration:** [docs/FIX_DECLARATION.md](../docs/FIX_DECLARATION.md), tag `fix-before` (commit `0bf54e2`).
**Fix:** tag `fix-after` (commit `a7692f6`). The diff: `git diff fix-before fix-after -- scan config tests scripts`.

## What shipped

- **Scale vote.** Each photo-tier model run is scaled once, by k × c, before alignment.
  - k = median over rooms of each room's median log(f_exif / f_model,i).
  - Points, depth and camera positions are scaled together, as one similarity.
  - Code: `scan/geometry/cloud.py`, `focal_scale_vote` and `scale_pred`.
- **Level c.** Fitted by `scan-calibrate --bias-only` on vote-only runs. Mode `photo_joint_vote`: ×1.0821, sigma_log 0.038, from 5 rooms (3 physical).
- **Intervals.** The level's sigma is added to the per-room scale uncertainty (hypot). Interval widths were then refitted for the new mode.
- **Calibration CLI bug, found on the way.** A normal `scan-calibrate` run also refitted the size bias on runs that already carried it, which pulled the factor to about 1. It now fits interval widths only, and the bias only with `--bias-only`.
- **Regenerate the calibration:** `scripts/calibrate_photo_vote.sh`.

## Before and after

`home01_photo_b`, the protocol capture (landscape):

| | Before | After |
|---|---|---|
| Walls within ±8% | 2 / 8 | **2 / 4 scored** (bedroom outline has 6 walls, so its 4 walls can't be scored) |
| Intervals containing the tape value (walls, ceilings, areas) | 2 / 12 (17%) | **8 / 8 (100%)** |
| Bedroom ceiling | 2.435 m (−35.8 cm) | **2.751 m (−4.1 cm)** |
| Hall ceiling | 2.451 m (−35.1 cm) | **2.774 m (−2.9 cm)** |
| Kitchen long walls | −11.0% | +4.0% |
| Kitchen short walls | −2.4% | +11.9% |

`home01_photo_a` (portrait, same rooms):

| | Before | After |
|---|---|---|
| Walls within ±8% | 13 / 14 | 12 / 14 |
| Intervals containing the tape value | 18 / 19 | 19 / 19 |
| Mean wall error | 2.8% | 2.85% |
| Ceilings | −2.4 / +0.1 / −4.9 cm | −6.6 / −4.2 / −9.8 cm |

Repeatability, same room across the two captures:

| | Before | After |
|---|---|---|
| Bedroom ceiling, a vs b | 33.4 cm apart | **2.5 cm apart** |
| Hall ceiling, a vs b | 35.2 cm apart | **1.4 cm apart** |

## Prediction vs result

| Predicted | Result | Held? |
|---|---|---|
| k = 0.910 (a), 1.044 (b) | 0.910, 1.044 | ✅ exact |
| Level c ≈ ×1.07 | ×1.082 | ✅ |
| `photo_b` ceilings about −2% | −1.5% / −1.0% | ✅ |
| `photo_b` coverage ≥ 8 / 10 (walls and ceilings) | 8 / 8 (walls, ceilings, areas) | ✅ |
| `photo_b` walls 6 / 8 within ±8% | 2 / 4 scored | ❌ |
| Kitchen short walls about +9%, failing | +11.9%, failing | ✅ (direction and failure) |
| `photo_a` 13 / 14 walls, ≥ 13 / 17 covered | 12 / 14, 19 / 19 | ≈ (one wall at 8.2%) |

**Gate outcome:** fail → fail. The scale, ceiling and calibration predictions held. The wall-count prediction did not.

## Post-mortem: why the bedroom walls became unscorable

The prediction scaled the **old** outline linearly. But the old outline had been laid out on geometry that was 12.5% too small, and the layout's thresholds are metric. Now they act on true-scale geometry:

1. In `photo_b`, the bedroom's north wall is **doubled about 0.7 m apart**. The views disagree on where the wall is: some put it at y ≈ 1.8 m and others at y ≈ 2.5 m (`out/debug/bedroom1/layout.png`).
2. Carving meets the two copies in a step. At the true scale the north-west corner step is 0.83 × 0.71 m ≈ 0.6 m², which is the `notch_max_area_m2` threshold. At the old scale it was about 0.46 m², so it was filled. Now it is kept, and the bedroom becomes a 6-wall L.
3. Neither 4-wall answer is right anyway. The inner line gives a north–south length of about 2.0 m and the outer line about 2.7 m, against 2.39 m on the tape.

So the scale fix did what it claimed. The remaining failure is the view-to-view doubling that was already on the open list (WORKLOG, "View-to-view doubling"). The threshold was **not** changed: that would be tuning against the tape.

**Cost on `photo_a`:**
- The single level is a compromise between the two captures, so `photo_a` moves from about +1% to about −1.5%.
- Its ceilings get worse by about 4 cm.
- One wall (8.2%) crosses the ±8% line.

## Regenerating both runs

```
git checkout fix-before      # before; then fix-after for the after run
uv run scan captures/home01_photo_b [--no-cache]
uv run scan-eval captures/home01_photo_b/out --gt data/ground_truth/home01.yaml
```

Do the same for `home01_photo_a`. The `fix-after` calibration is committed in `config/calibration.yaml`. `scripts/calibrate_photo_vote.sh` refits it from scratch.
