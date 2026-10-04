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
    joint: bool | None = typer.Option(
        None, "--joint/--per-room", help="Photo tier: one joint run over all rooms, or one run per room (default: auto)"
    ),
    device: str | None = typer.Option(None, help="cpu | mps | cuda"),
) -> None:
    """Turn a capture folder into plans, damage, scope, intervals, JSON and renders."""
    from scan.config import load_config
    from scan.pipeline import run

    overrides: dict = {}
    if joint is not None:
        overrides["geometry"] = {"joint": joint}
    if no_drift_correction:
        overrides["stitch"] = {"snap_walls": False}
    cfg = load_config(overrides=overrides)
    try:
        cap, _clouds, summary = run(capture_dir, cfg, tier, out, use_cache=not no_cache, device=device, log=typer.echo)
    except ScanError as e:
        typer.secho(f"Error: {e}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=e.exit_code) from None
    except NotImplementedError as e:
        _not_yet(f"{e} (see PLAN.md)")
    _print_capture(cap)
    res = summary["result"]
    g = summary["geometry"]
    typer.echo(f"Geometry: {g['backbone']}, joint={g['joint']}, rays={g['rays']}, runs: "
               + ", ".join(f"{k}={v['cache']}" for k, v in g["runs"].items()))

    def pm(m):
        return f"{m.value:.3f} [{m.lo:.3f}, {m.hi:.3f}]"

    for room in res.rooms:
        ceil = pm(room.ceiling_height) if room.ceiling_height else "not visible"
        typer.echo(f"  {room.room_id:10s} {room.status:7s} ceiling {ceil} m, area {pm(room.floor_area)} m2")
        for w in room.walls:
            typer.echo(f"      {w.wall_id:14s} {pm(w.length)} m")
    sp = res.stitched_plan
    typer.echo(f"Stitched: {sp.placement_method}, footprint {pm(sp.footprint_area)} m2, adjacency "
               + (", ".join(f"{a.room_a}-{a.room_b} via {a.shared_wall}" for a in sp.adjacency) or "none")
               + f", overlaps {len(sp.overlaps)}")
    out_dir = out or capture_dir / "out"
    typer.echo(f"Timing: {summary['timing_s']}")
    typer.secho(f"Wrote {out_dir / 'result.json'}, plan.png, plan.svg, rooms/*.png (diagnostics in debug/)",
                fg=typer.colors.GREEN)


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
    import json

    from scan.config import load_config
    from scan.eval.gates import evaluate as score
    from scan.eval.gt import load_gt
    from scan.eval.report import markdown
    from scan.schema import ScanResult

    res_path = out_dir / "result.json"
    if not res_path.exists():
        typer.secho(f"Error: {res_path} not found. Run: uv run scan <capture>", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=2)
    ev = score(ScanResult.model_validate_json(res_path.read_text()), load_gt(gt), load_config())
    md = markdown(ev)
    (out_dir / "eval.json").write_text(json.dumps(ev, indent=2, default=str))
    (out_dir / "eval.md").write_text(md)
    typer.echo(md)
    typer.secho(f"Wrote {out_dir / 'eval.md'} and eval.json", fg=typer.colors.GREEN)


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
    import json

    from scan.schema import json_schema

    out = Path(__file__).resolve().parents[1] / "schema" / "scan_output.schema.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(json_schema(), indent=2) + "\n")
    typer.echo(f"Wrote {out}")


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
