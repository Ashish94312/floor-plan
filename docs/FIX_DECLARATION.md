# Fix Declaration

Committed and tagged `fix-before` **before** any pipeline code changed. Date: 2026-10-05.

## 1. The gate and the failing number

**Photo tier: wall lengths within ±8%, with calibrated intervals**, on `home01_photo_b`, the capture shot as the protocol says (landscape, 1× lens).

| Capture | Walls within ±8% | Intervals that contain the tape value (target 90%) |
|---|---|---|
| `home01_photo_b` (landscape) | **2 / 8** (bedroom −9.4 … −11.1%, kitchen long side −11.0%) | **2 / 10 = 20%** |
| `home01_photo_a` (portrait, same rooms) | 13 / 14 | 16 / 17 |

Ceilings fail with the walls: 2.435 / 2.451 m against the tape's 2.79 / 2.80 m.

**Why this gate.** G1 openings is lower on paper (0 / 5). But its tape definition is still being checked: the frame's inside or outside faces, D28. This gate is the one the walk-in hits, because reviewers follow the protocol. A 20% coverage is "confident garbage", which caps the total score.

## 2. Root cause hypothesis and evidence

**Hypothesis.** MapAnything's metric depth follows its own reading of the focal length, and that reading depends on photo orientation. We rebuild points from the true (EXIF) rays but keep the model's depth. So the whole reconstruction is scaled by f_model / f_true, a factor that changes with orientation.

Why it should hold: a size prior fixes Z / f, not Z. An object of size H that spans h pixels gives Z = H f / h. If the model's f is 13% long, its Z is 13% long (E22o).

**Evidence**, from the cached model outputs (`K_model / K_exif` per view; `scripts/experiments/orientation_scale_predict.py`):

| | Portrait (`photo_a`) | Landscape (`photo_b`) |
|---|---|---|
| f_model / f_exif, median | 1.13 | 0.96 (0.95–0.97 over 16 views) |
| Plan size / tape size | ≈ 1.01 | ≈ 0.88 |

- The ratio between the captures matches: the size ratio b / a is 0.866 and the focal ratio b / a is 0.850.
- Divided by its own focal factor, each capture's scale is 0.895 (a) and 0.911 (b), which agree within 2%.
- `photo_a` was right because the two errors cancelled: the model's size sense is about 10% small, and the portrait focal reading is 13% long.

## 3. The fix

1. **Scale vote** (photo tier). Every view votes for the reconstruction's scale with log(f_exif / f_model,i).
   - k = median over rooms of each room's median vote. A room is the unit of evidence, because its views share its bias (E22p).
   - The whole reconstruction (points and camera positions together) is multiplied by k. This is one similarity transform, so multi-view consistency is kept.
   - Per-view correction is **not** used, because E22o showed it fails.
2. **Level.** The model's absolute size sense is a constant that geometry cannot see. It is calibrated across the taped captures with the existing `fit_size_bias` (leave one physical room out, as in D32), for mode `photo_joint`.
3. **Intervals.** The level's spread (sigma_log) is added to the scale uncertainty, so intervals widen by what the calibration does not know.

## 4. Predicted numbers after the fix

Predicted by applying k and c to the current outputs.

| | Before | Predicted after |
|---|---|---|
| `photo_b` walls within ±8% | 2 / 8 | **6 / 8**: bedroom 4 / 4 (within ±1.5%), kitchen long sides within ±1% |
| `photo_b` interval coverage | 2 / 10 | **≥ 8 / 10** |
| `photo_b` ceilings | −12.8 / −12.5% | about −2% (still fails G2's 1.5 cm) |
| `photo_a` | 13 / 14 walls, 16 / 17 covered | 13 / 14 walls, ≥ 13 / 17 covered |
| Bedroom repeat (a vs b), per wall | 23–33 cm apart | **≤ 8 cm** (G3's 1 cm still out of reach) |

**Inputs to the prediction:**
- k = 0.910 for `photo_a` and 1.044 for `photo_b`.
- Level c ≈ ×1.07 (sigma_log 0.03; 5 rooms, 3 physical).
- Leave-one-room-out: the `photo_b` bedroom ends at +2.4% and `photo_a` rooms at −0.2 … −3.0%.

**Out of scope, so it will still fail.** The kitchen's short walls are predicted at about +9%. That is a separate defect: views whose orientation differs from the joint batch are centre-cropped, so the floor and ceiling are cut away. It was found in the same run and gets its own fix after this one.

## 5. Regenerating the before and after runs

```
git checkout fix-before      # then again with fix-after
uv run scan captures/home01_photo_b --no-cache
uv run scan-eval captures/home01_photo_b/out --gt data/ground_truth/home01.yaml
```

Do the same for `home01_photo_a`. `--no-cache` forces the live model path. Without it, the cached model outputs replay deterministically (E19).
