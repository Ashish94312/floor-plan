# scan: iPhone capture → measured floor plan

This tool turns an iPhone capture of a home into a measured floor plan. It takes photos, a walkthrough video or a LiDAR scan, and produces:

- a per-room plan: walls, ceiling height, floor area, doors and windows
- one stitched plan of the whole property
- damage regions, concealed-damage flags and scope line items

Every number comes with a 90% interval. Output is JSON (with a published schema) plus a rendered plan.

**One command per capture. Runs locally, with no calls to our own servers.**

## Quick start

Target time: under 15 minutes on a clean machine.

**Needs:**
- macOS on Apple Silicon (tested: M4, 16 GB) or Linux with CUDA
- about 6 GB of disk for the model weights
- 16 GB of memory (a 20-photo run peaks at about 12 GB)

```bash
git clone <this repo> scan && cd scan
curl -LsSf https://astral.sh/uv/install.sh | sh   # only if `uv` is missing
uv sync                                           # Python 3.11 env, pinned by uv.lock
uv run scan-fetch-weights                         # ~5.2 GB, SHA256-checked (models below)
uv run scan captures/<capture>                    # one command per capture
```

Results go to `captures/<capture>/out/`:

| File | What it is |
|---|---|
| `result.json` | The full output contract. Schema: [schema/scan_output.schema.json](schema/scan_output.schema.json) |
| `plan.png`, `plan.svg` | The stitched plan, with dimensions and intervals |
| `rooms/*.png` | One plan per room |
| `debug/` | Diagnostics (point clouds, alignment, layout). Not part of the contract. |

After `scan-fetch-weights` the pipeline never touches the network, so it runs offline.

## Capturing

The capture route is Route 2: the stock Camera app for photos and video, and the free Stray Scanner app for LiDAR. It's written up as a one-page protocol in [docs/CAPTURE_PROTOCOL.md](docs/CAPTURE_PROTOCOL.md). The tier is detected from the folder layout:

| Tier | Layout |
|---|---|
| Photo | `<capture>/photos/<room>/*.HEIC`: one folder per room, 2–8 photos, 1× lens, landscape |
| Video | `<capture>/video/*.MOV` |
| LiDAR | a Stray Scanner export: `odometry.csv`, `depth/`, `confidence/`, `rgb.mp4`, `camera_matrix.csv` |

Room folder names are lowercase with no spaces, for example `bedroom1` or `kitchen`. If the input is unusable, the run stops and says which folder or file is the problem.

## Scoring against a tape

```bash
uv run scan-eval captures/<capture>/out --gt data/ground_truth/home01.yaml   # one capture
uv run scan-bench                                                            # every capture in config/bench.yaml
```

`scan-bench` writes [reports/benchmark.md](reports/benchmark.md). It covers:
- the gates for every capture and tier
- repeatability
- the drift-correction ablation
- the head-to-head against magicplan
- timing

The ground truth (laser) is in [data/ground_truth/](data/ground_truth/).

The raw captures and the magicplan export are too large for git. They ship as a separate data bundle, unpacked into `captures/`.

## Where each tier stands

| Tier | Status |
|---|---|
| Photo | Works end to end, including the stitched multi-room plan. Accuracy is in [reports/benchmark.md](reports/benchmark.md). |
| Video | Runs end to end. Room sizes are less reliable (about ±10%). Intervals are calibrated against the tape. |
| LiDAR | Reads Stray Scanner exports; see the benchmark for status. |

## Models and data used

All models are fetched by `scan-fetch-weights`, pinned to a revision and checked by SHA256.

| Model | Use | Licence |
|---|---|---|
| `facebook/map-anything-apache` (code `facebookresearch/map-anything@3d10cf7`) | Camera poses and metric depth from photos and video frames | Apache-2.0 |
| DINOv2 code (`facebookresearch/dinov2@7764ea0`) | Backbone code needed by MapAnything | Apache-2.0 |
| `google/owlv2-base-patch16-ensemble` | Doors, windows, mirrors and damage, from text queries | Apache-2.0 |

VGGT-1B and Depth Anything V2 were tried in the first spike and dropped. They are optional (`--optional`) and the pipeline doesn't use them.

## Repository

| Path | Contents |
|---|---|
| `scan/` | The pipeline: `io/` ingest, `geometry/`, `layout/`, `stitch/`, `damage/`, `uncertainty/`, `render/`, `eval/` |
| `config/` | `default.yaml` (all settings), `calibration.yaml` (fitted by `scan-calibrate`), `rules.yaml` (concealed-damage rules), `bench.yaml` |
| `data/` | Laser ground truth, and the magicplan numbers for the head-to-head |
| `docs/` | [REQUIREMENTS](REQUIREMENTS.md), [ARCHITECTURE](docs/ARCHITECTURE.md), [DECISIONS](docs/DECISIONS.md), [WORKLOG](docs/WORKLOG.md) (every experiment), [FIX_DECLARATION](docs/FIX_DECLARATION.md), [COMPLIANCE_MATRIX](docs/COMPLIANCE_MATRIX.md) |
| `reports/` | [benchmark.md](reports/benchmark.md), [fix_loop.md](reports/fix_loop.md) |
| `scripts/` | `calibrate_photo_vote.sh`; `experiments/` holds the scripts behind every worklog entry |
| `tests/` | `uv run pytest -q` |
