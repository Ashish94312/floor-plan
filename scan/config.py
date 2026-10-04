"""Load config/default.yaml, merge overrides, hash the resolved config for result.json."""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from typing import Any

import yaml

DEFAULT_PATH = Path(__file__).resolve().parents[1] / "config" / "default.yaml"


def _merge(base: dict, over: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in over.items():
        out[k] = _merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
    return out


def load_config(path: Path | None = None, overrides: dict[str, Any] | None = None) -> dict[str, Any]:
    cfg = yaml.safe_load((path or DEFAULT_PATH).read_text())
    return _merge(cfg, overrides or {})


def config_hash(cfg: dict[str, Any]) -> str:
    return hashlib.sha256(json.dumps(cfg, sort_keys=True).encode()).hexdigest()[:12]
