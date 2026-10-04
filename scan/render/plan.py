"""C11 Render (ARCHITECTURE §10.12): human-readable floor plans from the ScanResult.

Per room: walls as a dark band, each wall dimensioned "3.62 m ±0.22" outside its midpoint, room name,
area and ceiling height (with intervals) in the middle. Stitched: every placed room, shared walls,
overlaps in red. Openings and damage are drawn once later steps produce them.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from shapely.geometry import Polygon

WALL_T = 0.12  # drawn wall thickness [m]
FILLS = ["#f4efe6", "#e6eef4", "#eef4e6", "#f4e6ee", "#efe6f4", "#e6f4f1"]


def _pm(m) -> str:
    return f"±{max(m.hi - m.value, m.value - m.lo):.2f}"


def _draw_room(ax, room, fill: str, show_dims: bool = True, inside: set[str] = frozenset()) -> None:
    """inside: wall_ids whose dimension goes inside the room (shared walls in the stitched plan)."""
    P = np.array(room.polygon)
    poly = Polygon(P)
    ax.fill(P[:, 0], P[:, 1], color=fill, zorder=1)
    band = poly.buffer(WALL_T, join_style="mitre").difference(poly)
    for w in room.walls:  # open boundaries: cut the wall band away, draw a dashed line instead
        if w.kind != "open":
            continue
        a, b = np.array(w.start), np.array(w.end)
        d = (b - a) / (np.linalg.norm(b - a) + 1e-12)
        out = np.array([-d[1], d[0]])
        cut = Polygon([a - out * 0.01, b - out * 0.01, b + out * (WALL_T + 0.02), a + out * (WALL_T + 0.02)])
        band = band.difference(cut)
        ax.plot([a[0], b[0]], [a[1], b[1]], ls=(0, (4, 3)), color="#777", lw=1.2, zorder=3)
    walls = {w.wall_id: w for w in room.walls}
    for o in room.openings:  # doors / openings: gap in the wall band; windows: drawn over the band below
        if o.type == "window" or o.wall_id not in walls:
            continue
        w = walls[o.wall_id]
        a, b = np.array(w.start), np.array(w.end)
        d = (b - a) / (np.linalg.norm(b - a) + 1e-12)
        out = np.array([-d[1], d[0]])
        p0, p1 = a + d * o.offset_along_wall.value, a + d * (o.offset_along_wall.value + o.width.value)
        band = band.difference(Polygon([p0 - out * 0.02, p1 - out * 0.02, p1 + out * (WALL_T + 0.02), p0 + out * (WALL_T + 0.02)]))
        if o.type == "door" and not (o.connects_to and o.connects_to < room.room_id):  # shared door: draw once
            r_ = o.width.value
            ang0 = np.degrees(np.arctan2(*(d)[::-1]))
            ax.plot([p0[0], p0[0] - out[0] * r_], [p0[1], p0[1] - out[1] * r_], color="#555", lw=1, zorder=3)
            th = np.radians(np.linspace(ang0, ang0 - 90, 30))
            ax.plot(p0[0] + r_ * np.cos(th), p0[1] + r_ * np.sin(th), color="#999", lw=0.8, ls=":", zorder=3)
    for g in getattr(band, "geoms", [band]):
        x, y = g.exterior.xy
        ax.fill(x, y, color="#2f2f2f", zorder=2)
        for hole in g.interiors:
            hx, hy = hole.xy
            ax.fill(hx, hy, color=fill, zorder=2)
    for o in room.openings:
        if o.type != "window" or o.wall_id not in walls:
            continue
        w = walls[o.wall_id]
        a, b = np.array(w.start), np.array(w.end)
        d = (b - a) / (np.linalg.norm(b - a) + 1e-12)
        out = np.array([-d[1], d[0]])
        p0, p1 = a + d * o.offset_along_wall.value, a + d * (o.offset_along_wall.value + o.width.value)
        ax.fill(*np.array([p0, p1, p1 + out * WALL_T, p0 + out * WALL_T]).T, color="white", zorder=3)
        for f in (0.3, 0.7):
            q0, q1 = p0 + out * WALL_T * f, p1 + out * WALL_T * f
            ax.plot([q0[0], q1[0]], [q0[1], q1[1]], color="#333", lw=0.8, zorder=3)
    if show_dims:
        for w in room.walls:
            a, b = np.array(w.start), np.array(w.end)
            d = b - a
            out = np.array([-d[1], d[0]]) / (np.linalg.norm(d) + 1e-12)  # polygon is clockwise -> left = outside
            mid = (a + b) / 2 + (-out * 0.28 if w.wall_id in inside else out * (WALL_T + 0.22))
            vertical = abs(d[0]) < abs(d[1])
            tag = " (open)" if w.kind == "open" else ""
            ax.text(*mid, f"{w.wall_id.split('-')[-1]}{tag}  {w.length.value:.2f} m {_pm(w.length)}", fontsize=7.5,
                    ha="center", va="center", rotation=90 if vertical else 0, color="#333", zorder=4)
    c = poly.representative_point()
    lines = [room.room_id, f"{room.floor_area.value:.2f} m² {_pm(room.floor_area)}"]
    if room.ceiling_height is not None:
        lines.append(f"ceiling {room.ceiling_height.value:.2f} m {_pm(room.ceiling_height)}")
    if room.status != "ok":
        lines.append(f"[{room.status}]")
    ax.text(c.x, c.y, "\n".join(lines), ha="center", va="center", fontsize=9, color="#111", zorder=4,
            linespacing=1.4, fontweight="bold" if room.status == "ok" else "normal")


def _scale_bar(ax, xmin, ymin) -> None:
    ax.plot([xmin, xmin + 1.0], [ymin, ymin], color="k", lw=2)
    ax.text(xmin + 0.5, ymin - 0.12, "1 m", ha="center", va="top", fontsize=8)


def _frame(ax, polys, margin=0.9):
    allp = np.vstack(polys)
    lo, hi = allp.min(0) - margin, allp.max(0) + margin
    ax.set_xlim(lo[0], hi[0])
    ax.set_ylim(lo[1], hi[1])
    ax.set_aspect("equal")
    ax.axis("off")
    _scale_bar(ax, lo[0] + 0.15, lo[1] + 0.35)


def _draw_damage(ax, rooms, damage) -> None:
    """Damage drawn as a red bar on the inside of its wall, labelled with its id and class."""
    walls = {w.wall_id: w for r in rooms for w in r.walls}
    for d in damage:
        w = walls.get(d.surface_id)
        if w is None:
            continue
        a, b = np.array(w.start), np.array(w.end)
        u = (b - a) / (np.linalg.norm(b - a) + 1e-12)
        inward = -np.array([-u[1], u[0]])
        p0 = a + u * d.position_on_surface[0] + inward * 0.06
        p1 = p0 + u * d.width.value
        ax.plot([p0[0], p1[0]], [p0[1], p1[1]], color="#c0392b", lw=4, solid_capstyle="butt", zorder=5)
        mid = (p0 + p1) / 2 + inward * 0.18
        ax.text(*mid, f"{d.damage_id.rsplit('-', 1)[1]} {d.cls.replace('_', ' ')}", color="#c0392b", fontsize=7,
                ha="center", va="center", zorder=5)


def render_room(room, path: Path, damage=()) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 7))
    _draw_room(ax, room, FILLS[0])
    _draw_damage(ax, [room], [d for d in damage if d.surface_id.startswith(room.room_id + "-")])
    _frame(ax, [np.array(room.polygon)])
    ax.set_title(f"{room.room_id} ({room.layout_method})", fontsize=10)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def render_stitched(result, path_png: Path, path_svg: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    sp = result.stitched_plan
    placed = {p.room_id for p in sp.room_poses if p.placed}
    rooms = [r for r in result.rooms if r.room_id in placed]
    fig, ax = plt.subplots(figsize=(10, 8))
    if rooms:
        from scan.stitch.plan import all_shared_walls

        shared = all_shared_walls({r.room_id: r for r in rooms}, 0.35, 0.5)
        for i, r in enumerate(rooms):
            _draw_room(ax, r, FILLS[i % len(FILLS)], show_dims=True, inside=shared)
        _draw_damage(ax, rooms, result.damage)
        by_id = {r.room_id: r for r in rooms}
        for o in sp.overlaps:
            inter = Polygon(by_id[o.room_a].polygon).intersection(Polygon(by_id[o.room_b].polygon))
            for g in getattr(inter, "geoms", [inter]):
                if g.geom_type == "Polygon":
                    x, y = g.exterior.xy
                    ax.fill(x, y, color="red", alpha=0.5, zorder=5)
        _frame(ax, [np.array(r.polygon) for r in rooms])
    else:
        ax.text(0.5, 0.5, "no rooms placed", ha="center", transform=ax.transAxes)
        ax.axis("off")
    fp = sp.footprint_area
    adj = ", ".join(f"{a.room_a}–{a.room_b}" for a in sp.adjacency) or "none"
    unplaced = [p.room_id for p in sp.room_poses if not p.placed]
    ax.set_title(
        f"{result.capture_id}  ({result.tier} tier, {sp.placement_method})\n"
        f"footprint {fp.value:.2f} m² {_pm(fp)}   adjacency: {adj}   overlaps: {len(sp.overlaps)}"
        + (f"   unplaced: {', '.join(unplaced)}" if unplaced else ""),
        fontsize=10,
    )
    fig.tight_layout()
    path_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path_png, dpi=150)
    fig.savefig(path_svg)
    plt.close(fig)
