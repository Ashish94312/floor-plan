"""Ground truth loader (data/ground_truth/<property>.yaml). Centimetres in, metres out; null = not measured."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml


@dataclass
class GTWall:
    wall_id: str
    length_m: float | None
    kind: str = "wall"
    connects_to: str | None = None


@dataclass
class GTOpening:
    opening_id: str
    type: str
    wall_id: str
    width_m: float | None
    height_m: float | None


@dataclass
class GTRoom:
    room_id: str
    captured: bool
    ceiling_m: float | None
    walls: list[GTWall]
    checks: list[tuple[list[str], float]] = field(default_factory=list)  # (wall ids, summed length m)
    connects_to: set[str] = field(default_factory=set)  # via doors/openings/open walls
    openings: list[GTOpening] = field(default_factory=list)

    def area_m2(self) -> float | None:
        """Rectangle rooms with all 4 walls measured: mean of opposite walls multiplied."""
        L = [w.length_m for w in self.walls]
        if len(L) != 4 or any(v is None for v in L):
            return None
        return (L[0] + L[2]) / 2 * (L[1] + L[3]) / 2


def _m(cm):
    return None if cm is None else float(cm) / 100


def load_gt(path: Path) -> dict[str, GTRoom]:
    d = yaml.safe_load(Path(path).read_text())
    rooms = {}
    for r in d["rooms"]:
        ceil = [v for v in (r.get("ceiling_height_cm") or []) if v is not None]
        walls = [GTWall(w["id"], _m(w.get("length_cm")), w.get("kind", "wall"), w.get("connects_to")) for w in r["walls"]]
        checks = [(c["sum"], _m(c["length_cm"])) for c in r.get("checks", [])]
        conn = {o["connects_to"] for o in r.get("openings", []) if o.get("connects_to")}
        conn |= {w.connects_to for w in walls if w.connects_to}
        ops = [GTOpening(o["id"], o["type"], o["wall"], _m(o.get("width_cm")), _m(o.get("height_cm")))
               for o in r.get("openings", [])]
        rooms[r["id"]] = GTRoom(r["id"], r.get("captured", True), sum(ceil) / 100 / len(ceil) if ceil else None,
                                walls, checks, conn, ops)
    return rooms


def gt_adjacency(rooms: dict[str, GTRoom]) -> set[frozenset]:
    captured = {r for r, g in rooms.items() if g.captured}
    return {frozenset((r, c)) for r, g in rooms.items() if r in captured for c in g.connects_to if c in captured}
