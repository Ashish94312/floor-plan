"""Entry points: scan, scan-eval, scan-bench, scan-fetch-weights, scan-schema."""

from __future__ import annotations

from pathlib import Path

import typer

from scan.errors import ScanError


def _not_yet(step: str) -> None:
    typer.secho(f"Not implemented yet ({step}).", fg=typer.colors.YELLOW, err=True)
    raise typer.Exit(code=2)


def scan(
    capture_dir: Path = typer.Argument(..., exists=True, file_okay=False, help="Capture folder"),
    tier: str | None = typer.Option(None, help="photo | video | lidar (auto-detected if omitted)"),
    out: Path | None = typer.Option(None, help="Output dir (default: <capture_dir>/out)"),
    no_drift_correction: bool = typer.Option(False, "--no-drift-correction", help="Ablation: stitch with poses as-is"),
    no_cache: bool = typer.Option(False, "--no-cache", help="Force the live model path"),
    joint: bool = typer.Option(False, "--joint", help="Photo tier: reconstruct all rooms in one run (E9)"),
    device: str | None = typer.Option(None, help="cpu | mps | cuda"),
) -> None:
    """Turn a capture folder into plans, damage, scope, intervals, JSON and renders."""
    from scan.config import load_config
    from scan.pipeline import run

    cfg = load_config(overrides={"geometry": {"joint": True}} if joint else None)
    try:
        cap, _clouds, summary = run(capture_dir, cfg, tier, out, use_cache=not no_cache, device=device, log=typer.echo)
    except ScanError as e:
        typer.secho(f"Error: {e}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=e.exit_code) from None
    except NotImplementedError as e:
        _not_yet(f"{e} (see PLAN.md)")
    _print_capture(cap)
    g = summary["geometry"]
    typer.echo(f"Geometry: {g['backbone']}, joint={g['joint']}, rays={g['rays']}, runs: "
               + ", ".join(f"{k}={v['cache']}" for k, v in g["runs"].items()))
    for r, info in summary["rooms"].items():
        typer.echo(f"  {r:12s} {info['points']:>7d} points, extent {info['extent_m']} m -> {info['ply']}")
    typer.echo(f"Timing: {summary['timing_s']}")
    typer.secho(f"Wrote {(out or capture_dir / 'out') / 'geometry.json'}. Alignment is Tier 1 step 1.3 (next).",
                fg=typer.colors.YELLOW)


def _print_capture(cap) -> None:
    typer.echo(f"Capture {cap.capture_id}: tier={cap.tier}, devices={', '.join(cap.devices) or 'unknown'}")
    for room, frames in cap.rooms.items():
        o = frames[0].meta.orientation
        f35 = sorted({fr.meta.f35_mm for fr in frames})
        typer.echo(f"  {room:12s} {len(frames)} photos, {o}, f35={'/'.join(f'{x:g}' for x in f35)} mm")
    for w in cap.warnings:
        typer.secho(f"  warning: {w}", fg=typer.colors.YELLOW)


def evaluate(
    out_dir: Path = typer.Argument(..., exists=True, file_okay=False),
    gt: Path = typer.Option(..., exists=True, dir_okay=False, help="Ground-truth YAML"),
) -> None:
    """Score a scan output against tape-measured ground truth."""
    _not_yet("Tier 1 step 1.6")


def bench(suite: Path | None = typer.Option(None, help="bench.yaml")) -> None:
    """Run every capture + eval and write reports/."""
    _not_yet("Phase 4")


def fetch_weights(
    optional: bool = typer.Option(False, "--optional", help="Also fetch legacy spike models (VGGT, DAv2)"),
    only: list[str] | None = typer.Option(None, help="Fetch only these model names"),
) -> None:
    """Download all model weights into weights/ and verify SHA256."""
    from scan.weights import CODE, MODELS, WEIGHTS_DIR, fetch, fetch_code

    specs = [m for m in MODELS.values() if (optional or not m.optional)]
    if only:
        specs = [MODELS[n] for n in only]
    typer.echo(f"Fetching {len(specs)} model(s) into {WEIGHTS_DIR}/")
    for spec in specs:
        typer.echo(f"{spec.name}  ({spec.repo_id}@{spec.revision[:8]}, {spec.licence})")
        fetch(spec, log=typer.echo)
    if not only:
        for code in CODE.values():
            typer.echo(f"{code.name}  ({code.repo}@{code.commit[:8]}, {code.licence})")
            fetch_code(code, log=typer.echo)
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
