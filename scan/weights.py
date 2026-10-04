"""Pinned model weights: registry, download, SHA256 verification.

Every model is pinned to a Hugging Face commit and every large file to its SHA256,
so a fresh clone fetches exactly the bytes we benchmarked with.
"""

from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass, field
from pathlib import Path

WEIGHTS_DIR = Path(os.environ.get("SCAN_WEIGHTS_DIR", "weights"))


@dataclass(frozen=True)
class ModelSpec:
    name: str  # local folder under weights/
    repo_id: str
    revision: str  # HF commit SHA
    licence: str
    files: tuple[str, ...]
    sha256: dict[str, str] = field(default_factory=dict)  # large files only
    optional: bool = False

    @property
    def path(self) -> Path:
        return WEIGHTS_DIR / self.name


MODELS: dict[str, ModelSpec] = {
    m.name: m
    for m in [
        ModelSpec(
            name="vggt-1b",
            repo_id="facebook/VGGT-1B",
            revision="860abec7937da0a4c03c41d3c269c366e82abdf9",
            licence="CC-BY-NC-4.0",
            files=("config.json", "model.safetensors"),
            sha256={
                "model.safetensors": "f164acf60724910d8fe1578bb499d800850c7bb0948db7555c413f9fbe60467e"
            },
        ),
        ModelSpec(
            name="dav2-metric-indoor-small",
            repo_id="depth-anything/Depth-Anything-V2-Metric-Indoor-Small-hf",
            revision="8078d68a9c75a972131914f6afd0c1723be0da7f",
            licence="Apache-2.0",
            files=("config.json", "model.safetensors", "preprocessor_config.json"),
            sha256={
                "model.safetensors": "e990eb82fbf11b05b7813261196a2b841bdcf5a05f64396724a8987fa90504a3"
            },
        ),
        ModelSpec(
            name="owlv2-base-ensemble",
            repo_id="google/owlv2-base-patch16-ensemble",
            revision="cfd3195ba4ea9592eec887ded089f4c08eff231d",
            licence="Apache-2.0",
            files=(
                "config.json",
                "model.safetensors",
                "preprocessor_config.json",
                "added_tokens.json",
                "merges.txt",
                "special_tokens_map.json",
                "tokenizer_config.json",
                "vocab.json",
            ),
            sha256={
                "model.safetensors": "e1e130b9e404cf91a75ad45644c1da9d7fa5284085eecc864266a6923efb99e7"
            },
        ),
        ModelSpec(
            name="mapanything-apache",
            repo_id="facebook/map-anything-apache",
            revision="00f9c245bbcb60522d1ed7f9e9d88462c6e3f38a",
            licence="Apache-2.0",
            files=("config.json", "model.safetensors"),
            sha256={
                "model.safetensors": "fa06c0fdccefc5048e072c85935d5789b1e36b307f3859033c17f9dcb9fd5201"
            },
            optional=True,  # O1 alternative backbone; only needed for the spike comparison
        ),
    ]
}


def sha256_file(path: Path, chunk: int = 16 * 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while block := f.read(chunk):
            h.update(block)
    return h.hexdigest()


def _verified_marker(path: Path) -> Path:
    return path.with_name(path.name + ".sha256")


def is_verified(path: Path, expected: str) -> bool:
    """True if a previous run already hashed this exact file (same size + mtime)."""
    marker = _verified_marker(path)
    if not (path.exists() and marker.exists()):
        return False
    st = path.stat()
    return marker.read_text().strip() == f"{expected} {st.st_size} {int(st.st_mtime)}"


def verify(path: Path, expected: str) -> None:
    if is_verified(path, expected):
        return
    actual = sha256_file(path)
    if actual != expected:
        raise RuntimeError(
            f"SHA256 mismatch for {path}\n  expected {expected}\n  got      {actual}\n"
            f"Delete the file and re-run scan-fetch-weights."
        )
    st = path.stat()
    _verified_marker(path).write_text(f"{expected} {st.st_size} {int(st.st_mtime)}\n")


def fetch(spec: ModelSpec, log=print) -> Path:
    from huggingface_hub import hf_hub_download

    spec.path.mkdir(parents=True, exist_ok=True)
    for filename in spec.files:
        target = spec.path / filename
        expected = spec.sha256.get(filename)
        if target.exists() and (expected is None or is_verified(target, expected)):
            log(f"  [skip] {spec.name}/{filename}")
            continue
        log(f"  [get ] {spec.name}/{filename}")
        hf_hub_download(
            repo_id=spec.repo_id,
            filename=filename,
            revision=spec.revision,
            local_dir=spec.path,
        )
        if expected:
            log(f"  [hash] {spec.name}/{filename}")
            verify(target, expected)
    return spec.path


def model_path(name: str) -> Path:
    """Local path of a fetched model. Fails with a clear message if it is missing."""
    spec = MODELS[name]
    missing = [f for f in spec.files if not (spec.path / f).exists()]
    if missing:
        raise FileNotFoundError(
            f"Weights for '{name}' missing in {spec.path} ({', '.join(missing)}). "
            f"Run: uv run scan-fetch-weights"
        )
    return spec.path
