"""Entry points: scan, scan-eval, scan-bench, scan-fetch-weights, scan-schema."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import typer


def _not_yet(step: str) -> None:
    typer.secho(f"Not implemented yet ({step}).", fg=typer.colors.YELLOW, err=True)
    raise typer.Exit(code=2)


def scan(
    capture_dir: Path = typer.Argument(..., exists=True, file_okay=False, help="Capture folder"),
    tier: Optional[str] = typer.Option(None, help="photo | video | lidar (auto-detected if omitted)"),
    out: Optional[Path] = typer.Option(None, help="Output dir (default: <capture_dir>/out)"),
    no_drift_correction: bool = typer.Option(False, "--no-drift-correction", help="Ablation: stitch with poses as-is"),
    no_cache: bool = typer.Option(False, "--no-cache", help="Force the live model path"),
    device: Optional[str] = typer.Option(None, help="cpu | mps | cuda"),
) -> None:
    """Turn a capture folder into plans, damage, scope, intervals, JSON and renders."""
    _not_yet("Tier 1 step 1.5")


def evaluate(
    out_dir: Path = typer.Argument(..., exists=True, file_okay=False),
    gt: Path = typer.Option(..., exists=True, dir_okay=False, help="Ground-truth YAML"),
) -> None:
    """Score a scan output against tape-measured ground truth."""
    _not_yet("Tier 1 step 1.6")


def bench(suite: Optional[Path] = typer.Option(None, help="bench.yaml")) -> None:
    """Run every capture + eval and write reports/."""
    _not_yet("Phase 4")


def fetch_weights(
    optional: bool = typer.Option(False, "--optional", help="Also fetch optional models (MapAnything)"),
    only: Optional[list[str]] = typer.Option(None, help="Fetch only these model names"),
) -> None:
    """Download all model weights into weights/ and verify SHA256."""
    from scan.weights import MODELS, WEIGHTS_DIR, fetch

    specs = [m for m in MODELS.values() if (optional or not m.optional)]
    if only:
        specs = [MODELS[n] for n in only]
    typer.echo(f"Fetching {len(specs)} model(s) into {WEIGHTS_DIR}/")
    for spec in specs:
        typer.echo(f"{spec.name}  ({spec.repo_id}@{spec.revision[:8]}, {spec.licence})")
        fetch(spec, log=typer.echo)
    typer.secho("All weights present and verified.", fg=typer.colors.GREEN)


def schema() -> None:
    """Write schema/scan_output.schema.json from scan/schema.py."""
    _not_yet("Tier 1 step 1.5")


def scan_main() -> None:
    typer.run(scan)


def eval_main() -> None:
    typer.run(evaluate)


def bench_main() -> None:
    typer.run(bench)


def fetch_weights_main() -> None:
    typer.run(fetch_weights)


def schema_main() -> None:
    typer.run(schema)
