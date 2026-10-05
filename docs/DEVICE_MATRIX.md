# Device Matrix

Each row is one tier: the hardware it needs, what it was tested on, and how accurate it is. The accuracy figures come from [reports/benchmark.md](../reports/benchmark.md), scored against a laser on home01 (3 rooms: bedroom, L-shaped hall, kitchen).

| Tier | Hardware | Capture app | Tested on | Walls | Ceilings | Intervals contain the tape value |
|---|---|---|---|---|---|---|
| Photo | Any iPhone 15 or newer; 1× lens, HEIC/JPEG with EXIF | Camera (Photo) | iPhone 13, 1× lens (26 mm eq.), 2 captures | Portrait capture: mean 2.9%, 12/14 within ±8%. Landscape capture: mean 6.7%, but the bedroom and hall outlines are not scorable (doubled walls) | −3 to −10 cm (biased low) | 100% (target 90%) |
| Video | Any iPhone 15 or newer; 1× lens | Camera (Video) | iPhone 13, 1080p, 2 captures | Landscape (final config): mean 10.3%, footprint −4.3%. Portrait: mean 20.9%. Not within the ±3% target | −14 to −86 cm (unrepeatable) | 70–75% |
| LiDAR | Pro iPhone (12 Pro or newer) | Stray Scanner (free) | Assignment sample recordings and synthetic rooms; no tape-measured LiDAR capture | not measured | not measured | not measured |

## Notes

- **Phone.** The benchmark phone is an iPhone 13, older than the iPhone 15+ in the brief. The pipeline reads the focal length from each photo's EXIF data, so it doesn't depend on one model. An iPhone 15's 24 mm main camera is handled the same way.
- **1× lens settings.** Pro iPhones can set the 1× lens to 28 or 35 mm. A capture taken entirely on one setting is kept. Photos on a different lens from the rest of the capture (for example a 0.5× shot) are excluded, with a warning.
- **Orientation.** The protocol asks for landscape. Model scale depends on orientation, and the scale vote corrects for that (`reports/fix_loop.md`). Photos in the minority orientation within a room are dropped.
- **Run machine.** The pipeline was run on a MacBook with an M4 and 16 GB of memory.
  - A 20-photo joint run peaks at about 12.8 GB. A cold photo run of 3 rooms takes about 4 minutes, of which the model takes about 2.
  - Video spends about 50–55 s per 3-room capture decoding frames and choosing keyframes (hardware decode).
  - Cached re-runs take seconds.
