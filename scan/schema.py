"""The published output schema (ARCHITECTURE §9.2, REQUIREMENTS §7.2).

Every measurement is a Measurement (value + lo + hi + unit), never a bare number. Every surface has a
stable id ("<room>-W<n>", "<room>-F" floor, "<room>-C" ceiling) so damage, flags and scope items can
reference it. The same keys appear at every tier: fields a step hasn't produced yet are empty lists,
never missing. `uv run scan-schema` writes schema/scan_output.schema.json from these models.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

SCHEMA_VERSION = "1.0"


class _Model(BaseModel):
    model_config = ConfigDict(extra="forbid")


class Measurement(_Model):
    value: float
    lo: float
    hi: float
    unit: Literal["m", "m2", "count", "deg"]

    @model_validator(mode="after")
    def _ordered(self):
        if not (self.lo <= self.value <= self.hi):
            raise ValueError(f"interval must contain the value: lo={self.lo} value={self.value} hi={self.hi}")
        return self


class Wall(_Model):
    wall_id: str = Field(description="<room>-W<n>; W1 = wall with the main door, then clockwise seen from above")
    start: tuple[float, float] = Field(description="room-local metres (aligned frame, x/y along the walls)")
    end: tuple[float, float]
    length: Measurement


class Opening(_Model):
    opening_id: str = Field(description="<room>-O<n>")
    type: Literal["door", "window", "opening"]
    wall_id: str
    offset_along_wall: Measurement
    width: Measurement
    height: Measurement
    sill_height: Measurement | None = None
    connects_to: str | None = Field(default=None, description="room_id on the other side (filled by stitching)")
    views: int = Field(description="number of photos/frames supporting the detection")


class Room(_Model):
    room_id: str
    label: str | None = None
    status: Literal["ok", "partial", "failed"]
    polygon: list[tuple[float, float]] = Field(description="floor outline, clockwise seen from above, metres")
    walls: list[Wall]
    ceiling_height: Measurement | None = Field(description="null only if the ceiling was not visible (status partial)")
    floor_area: Measurement
    openings: list[Opening]
    layout_method: str
    warnings: list[str]


class DamageRegion(_Model):
    damage_id: str
    surface_id: str = Field(description='"<room>-W<n>" | "<room>-F" | "<room>-C"')
    cls: Literal["water_stain", "crack"] = Field(alias="class")
    width: Measurement
    height: Measurement
    area: Measurement
    position_on_surface: tuple[float, float] = Field(description="(from_left, from_floor) metres")
    detection_score: float

    model_config = ConfigDict(extra="forbid", populate_by_name=True)


class ConcealedFlag(_Model):
    flag_id: str
    surface_id: str
    rule_id: str
    rule_text: str
    evidence: list[str] = Field(description="damage_ids that triggered the rule")


class ScopeItem(_Model):
    item_id: str
    surface_id: str
    action: str
    quantity: Measurement
    rule_id: str
    source: list[str]


class RoomPose(_Model):
    room_id: str
    x: float
    y: float
    theta_deg: float
    placed: bool = Field(description="false if the room could not be placed in the property frame")


class Adjacency(_Model):
    room_a: str
    room_b: str
    via_opening: str | None = Field(description="opening_id, once openings are detected")
    shared_wall: tuple[str, str] = Field(description="(wall_id in room_a, wall_id in room_b)")


class Overlap(_Model):
    room_a: str
    room_b: str
    area_m2: float


class DriftCorrection(_Model):
    enabled: bool
    method: str


class StitchedPlan(_Model):
    placement_method: str
    room_poses: list[RoomPose]
    adjacency: list[Adjacency]
    footprint_area: Measurement = Field(description="sum of room floor areas (net internal area, D10)")
    overlaps: list[Overlap] = Field(description="must be empty (G5)")
    drift_correction: DriftCorrection


class ScanResult(_Model):
    schema_version: str = SCHEMA_VERSION
    capture_id: str
    tier: Literal["photo", "video", "lidar"]
    devices: list[str]
    units: Literal["m"] = "m"
    interval_level: float = Field(description="nominal coverage of every [lo, hi] interval")
    rooms: list[Room]
    stitched_plan: StitchedPlan
    damage: list[DamageRegion]
    concealed_damage_flags: list[ConcealedFlag]
    scope_items: list[ScopeItem]
    timing_s: dict[str, float]
    warnings: list[str]
    software: dict[str, str] = Field(description="git commit, model ids, config hash")


def json_schema() -> dict:
    s = ScanResult.model_json_schema(by_alias=True)
    s["$id"] = f"https://example.invalid/scan_output/{SCHEMA_VERSION}"
    s["title"] = "ScanResult"
    return s
