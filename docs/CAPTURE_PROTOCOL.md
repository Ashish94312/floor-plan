# Capture Protocol (v2)

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

**One clip per room**, including hallways. One continuous clip through the whole home is not supported.

1. Step **inside** the room and press record. Film every part of the room from inside it.
2. Walk at **half your normal speed**. A quarter turn should take about 5 seconds.
3. Film from the corners towards the opposite corner. Tilt up until the ceiling line is in view, then down to the floor line, on every wall.
4. Film each doorway **from inside this room**, with a moment looking through it into the next room.
5. Aim for 1–2 minutes per room. Then stop.
6. **Hand-off.** AirDrop the clips to the Mac. Put each room's clip in its own folder: `captures/<capture-name>/video/<room-name>/<clip>.MOV`, with the same room names as for photos.

## Tier 3 — LiDAR (iPhone Pro only, free App Store app "Stray Scanner")

**One recording through the whole home.** The pipeline splits it into rooms at the doorways.

1. Install **Stray Scanner** from the App Store and open it.
2. Tap **record** in the first room. Walk at **half your normal speed** through every room and hallway: in each room, film every wall, and tilt up to the ceiling and down to the floor.
3. **At every doorway, tilt up so the wall above the door and the ceiling are in view** before walking through. The wall above a door is what separates two rooms in the plan; a walk that never films above about 2 m comes out as one big room.
4. Tap **stop**.
5. **Hand-off.** Open the Files app, then On My iPhone, then Stray Scanner. Long-press the newest folder, tap **Share**, and AirDrop it to the Mac. Put the folder in `captures/<capture-name>/lidar/`. To film one recording per room instead, put each in `captures/<capture-name>/lidar/<room-name>/`.

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
