"""Video: a room's size from its doors (E22z).

Each room's model run has its own size error (per room ~6%, D32; the home01_video_e hall run is ~10% small, E22v).
Nothing inside one room shows a uniform size error, and in landscape video the shared ceiling height (the same in
every room) is out of view (E22s). Doors are in view, and door heights are an architectural standard: 2.0-2.1 m
(prior stitch.door_height_prior_m +- door_height_sigma_m, general knowledge, not this flat's tape).

A door's height counts only when it was measured: the opening reaches the floor and the wall ABOVE its top was seen
(wall votes in the band just above it). Otherwise the top is where the view ended (a landscape frame sees +-18 deg
vertically, E22z). The room's size factor from its doors, prior / measured height, is combined with the room's own
size uncertainty in log space (inverse variance): a room whose size is already well known moves less.
"""

from __future__ import annotations

import numpy as np

from scan.layout.openings import extract, wall_votes


def lintel_seen(open_v: np.ndarray, wall_v: np.ndarray, cols: slice, top_m: float, cfg: dict) -> bool:
    """Was the wall above an opening's top seen? In the band above the top (vote grid rows = cell_z_m) at least 2 rows
    hold votes and, over the band, wall votes dominate. Over the band, not per row: the reported top is a percentile of
    the open cells, so the first row above it straddles the edge and still holds some see-through votes."""
    oc, sc = cfg["openings"], cfg["stitch"]
    dz = oc["cell_z_m"]
    k0 = int(np.ceil(top_m / dz - 1e-6))
    k1 = min(open_v.shape[0], k0 + round(sc["door_lintel_band_m"] / dz))
    o, w = open_v[k0:k1, cols].sum(1), wall_v[k0:k1, cols].sum(1)
    voted = (o + w) > 0
    return bool(voted.sum() >= 2 and w.sum() >= oc["min_votes"] and w.sum() / (o.sum() + w.sum()) >= oc["min_open_ratio"])


def measured_door_heights(cloud, layout, cfg: dict) -> list[float]:
    """Heights of the room's floor-reaching openings whose lintel was seen (the room's current scale)."""
    oc, lc = cfg["openings"], cfg["layout"]
    ceil = cloud.planes.ceiling_z or float(np.percentile(cloud.points[:, 2], 98))
    out = []
    for w in layout.walls:
        if w.kind != "wall" or w.support < lc["min_wall_support"]:
            continue
        open_v, wall_v, L = wall_votes(w, cloud.views, ceil, cfg)
        for o in extract(w, open_v, wall_v, L, ceil, cfg):
            if o["sill"] is not None:  # window
                continue
            c0 = int(o["u0"] / oc["cell_u_m"])
            cols = slice(c0, max(c0 + 1, int((o["u0"] + o["width"]) / oc["cell_u_m"])))
            if lintel_seen(open_v, wall_v, cols, o["height"], cfg):
                out.append(float(o["height"]))
    return out


def door_scale(heights: list[float], sigma_room: float, cfg: dict) -> tuple[float, float] | None:
    """(size factor, its log uncertainty) from measured door heights and the room's own log uncertainty. A height that
    disagrees with the prior by more than 3 sigma of the two spreads combined is not a door (or not all of one)."""
    sc = cfg["stitch"]
    prior, sp = sc["door_height_prior_m"], sc["door_height_sigma_m"]

    def s_of(h):  # prior spread + the vote grid's step
        return float(np.hypot(sp / prior, cfg["openings"]["cell_z_m"] / h))

    heights = [h for h in heights if abs(np.log(prior / h)) <= 3 * np.hypot(sigma_room, s_of(h))]
    if not heights:
        return None
    h = float(np.median(heights))
    s_door = s_of(h)
    w = sigma_room**2 / (sigma_room**2 + s_door**2)
    return float(np.exp(w * np.log(prior / h))), float(1.0 / np.sqrt(1 / sigma_room**2 + 1 / s_door**2))
