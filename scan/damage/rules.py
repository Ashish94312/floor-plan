"""C9 Rules (ARCHITECTURE §10.11): concealed-damage flags and scope items from config/rules.yaml."""

from __future__ import annotations

from pathlib import Path

import yaml

RULES = Path(__file__).resolve().parents[2] / "config" / "rules.yaml"


def _fires(when: dict, d, surface_kind: str, ceiling_z: float | None) -> bool:
    if when.get("class") and when["class"] != d.cls:
        return False
    if "surface" in when and when["surface"] != surface_kind:
        return False
    if "top_within_m_of_ceiling" in when and (
        surface_kind != "wall" or ceiling_z is None or ceiling_z - (d.from_floor_m + d.height_m) > when["top_within_m_of_ceiling"]
    ):
        return False
    if "bottom_within_m_of_floor" in when and (surface_kind != "wall" or d.from_floor_m > when["bottom_within_m_of_floor"]):
        return False
    return not ("length_gt_m" in when and max(d.width_m, d.height_m) <= when["length_gt_m"])


def apply_rules(damage: list, surface_area: dict[str, float], ceiling_z: float | None, prefix: str = "", path: Path = RULES):
    """-> (flags, scope) as plain dicts; every item names the rule that produced it."""
    rules = yaml.safe_load(path.read_text())
    flags, scope = [], []
    for d in damage:
        kind = "wall" if "-W" in d.surface_id else ("ceiling" if d.surface_id.endswith("-C") else "floor")
        for rule in rules["concealed_damage_rules"]:
            if _fires(rule["when"], d, kind, ceiling_z):
                flags.append({"flag_id": f"{prefix}F{len(flags) + 1}", "surface_id": d.surface_id, "rule_id": rule["id"],
                              "rule_text": rule["text"], "evidence": [d.damage_id]})
        for k, rule in enumerate(rules["scope_rules"]):
            if rule["when"]["class"] != d.cls:
                continue
            for item in rule["items"]:
                q = item["quantity"]
                qty = {"surface_area": surface_area.get(d.surface_id, 0.0),
                       "damage_length": max(d.width_m, d.height_m)}.get(q, q)
                scope.append({"item_id": f"{prefix}S{len(scope) + 1}", "surface_id": d.surface_id, "action": item["action"],
                              "quantity": float(qty), "unit": item["unit"], "rule_id": f"SCOPE_{d.cls.upper()}_{k}",
                              "source": [d.damage_id] + [f["flag_id"] for f in flags if d.damage_id in f["evidence"]]})
    return flags, scope
