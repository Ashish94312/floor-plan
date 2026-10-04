# Capture Protocol (v1)

One page. Follow it exactly, top to bottom. Allow about 15 minutes per tier for a 4-room home.

## Before every capture

1. Turn on **every light** in every room. Open curtains.
2. Open **all interior doors** fully and leave them open.
3. Do not move furniture between captures.
4. Keep people and pets out of the shot.
5. Use the **1×** lens only. Do not zoom, and do not tap 0.5×, 2× or 3×.
6. Hold the phone **sideways (landscape)** at chest height.

## Tier 1 — Photos (any iPhone, built-in Camera app, PHOTO mode)

Do this room by room, including hallways.

1. **Corners.** Stand in each corner with your back to it. Take one photo of the opposite side of the room. Floor and ceiling must both be visible. Take one photo per corner.
2. **Doorways.** For each doorway leading out of the room, stand one step inside the room, face the doorway and take one photo through it, showing the next room.
3. A room needs **at least 2 and at most 8** photos. If you reach 8, stop.
4. **Hand-off.** In the Photos app, select that room's photos, tap **Share**, then **Options**, switch **All Photos Data** on, and AirDrop them to the Mac. On the Mac, move them into `captures/<capture-name>/photos/<room-name>/`. Room names are lowercase with no spaces, for example `kitchen`, `bedroom1`, `hall`.

## Tier 2 — Video (any iPhone, built-in Camera app, VIDEO mode)

1. Stand in the doorway of the first room and press record.
2. Walk at **half your normal speed**. A quarter turn should take about 5 seconds.
3. In each room, walk once around the edge. Point the phone across the room so the line where the floor meets the wall and the line where the ceiling meets the wall are both visible.
4. Go through every doorway into every room. Pass back through the hallway between rooms.
5. **Finish where you started**, pointing at the same view you recorded first. Then stop.
6. Aim for about 1 minute per room, 10 minutes at most.
7. **Hand-off.** AirDrop the video to the Mac. Save it as `captures/<capture-name>/video/walkthrough.mov`.

## Tier 3 — LiDAR (iPhone Pro only, free App Store app "Stray Scanner")

1. Install **Stray Scanner** from the App Store and open it.
2. Tap **record**. Walk exactly as described in Tier 2, steps 2–5. Tap **stop**.
3. **Hand-off.** Open the Files app, then On My iPhone, then Stray Scanner. Long-press the newest folder, tap **Share**, and AirDrop it to the Mac. Put the folder in `captures/<capture-name>/lidar/`.

## Avoid (all tiers)

- Do not stand directly in front of a mirror, and do not point at a mirror for more than a moment.
- Do not point straight into a bright window.
- No fast turns, running or shaking. Keep fingers off the lens.
- Do not switch lights off or close doors partway through a capture.

## Run

```
uv run scan captures/<capture-name>
```

Results appear in `captures/<capture-name>/out/`.
