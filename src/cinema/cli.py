"""`cinema` command-line interface."""
from __future__ import annotations

import json
import logging
import os
import platform
import shutil
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path

import typer

from .config import load_config

app = typer.Typer(help="Autonomous local AI cinematic shorts factory (Apple Silicon).", no_args_is_help=True)


def _cfg():
    return load_config()


def _sigterm_as_interrupt() -> None:
    """SIGTERM (abort) unwinds like Ctrl-C so the resource manager's finally-blocks reap GPU workers."""
    import signal

    def handler(signum, frame):
        raise KeyboardInterrupt
    signal.signal(signal.SIGTERM, handler)


def _console_log() -> None:
    root = logging.getLogger("cinema")
    if not any(isinstance(h, logging.StreamHandler) and not hasattr(h, "baseFilename") for h in root.handlers):
        h = logging.StreamHandler(sys.stdout)
        h.setFormatter(logging.Formatter("%(asctime)s %(message)s", "%H:%M:%S"))
        root.addHandler(h)
    root.setLevel(logging.INFO)


def _echo(obj) -> None:
    typer.echo(json.dumps(obj, indent=2, default=str))


# --------------------------------------------------------------------------- readiness
@app.command()
def doctor() -> None:
    """Verify system readiness (tools, models, memory, disk) for autonomous production."""
    from .model_registry import autonomous_readiness, refresh
    from .resource_manager import disk_free_gib, memory_snapshot
    cfg = _cfg()
    mem = memory_snapshot()
    entries = refresh(cfg)
    problems = autonomous_readiness(cfg)
    try:
        import httpx
        ollama = httpx.get(f"{cfg.ollama_url}/api/version", timeout=5).json().get("version")
    except Exception:
        ollama = None
    _echo({"machine": platform.machine(), "macos": platform.mac_ver()[0],
           "chip": subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True, text=True).stdout.strip(),
           "memory": mem, "disk_free_gib": round(disk_free_gib(cfg.root), 1),
           "disk_reserve_gib": cfg.memory.min_free_gib_disk, "ollama_version": ollama,
           "ffmpeg": shutil.which("ffmpeg"), "mlxgen": shutil.which("mlxgen"),
           "models": {e.role: f"{e.repo} [{e.status}]" for e in entries},
           "ready_for_autonomous_run": not problems, "blocking_problems": problems})
    raise typer.Exit(0 if not problems else 1)


@app.command()
def setup(models: bool = typer.Option(False, "--models", help="Download missing approved models (~51 GB worst case)"),
          yes: bool = typer.Option(False, "--yes", help="Skip the confirmation prompt")) -> None:
    """Install/configure permitted dependencies and (optionally) download approved models."""
    from .model_registry import refresh
    cfg = _cfg()
    for d in ("models/licenses", "projects", "outputs", "logs", "state", "benchmarks"):
        (cfg.root / d).mkdir(parents=True, exist_ok=True)
    if not (cfg.root / "config.yaml").exists():
        shutil.copy(cfg.root / "config.example.yaml", cfg.root / "config.yaml")
    if not shutil.which("mlxgen"):
        subprocess.run(["uv", "tool", "install", "mlx-gen"], check=True)
    missing = [e for e in refresh(cfg) if e.production and e.status in ("missing", "incomplete")]
    typer.echo("Missing: " + (", ".join(f"{e.role}={e.repo}" for e in missing) or "none"))
    if models and missing:
        if not yes and not typer.confirm(f"Download {len(missing)} model(s)? (100 GiB free-space floor enforced)"):
            raise typer.Exit(1)
        free = shutil.disk_usage(cfg.root).free / 1024 ** 3
        if free < cfg.memory.min_free_gib_disk + 60:
            typer.echo(f"Refusing: {free:.0f} GiB free leaves less than the reserve after downloads")
            raise typer.Exit(1)
        for e in missing:
            if e.runtime == "ollama":
                subprocess.run(["ollama", "pull", e.repo], check=True)
            elif e.role == "tts":
                base = "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/"
                for f in ("kokoro-v1.0.onnx", "voices-v1.0.bin"):
                    subprocess.run(["curl", "-L", "-o", str(cfg.root / "models" / f), base + f], check=True)
            elif "/" in e.repo:
                subprocess.run(["uvx", "--from", "huggingface_hub", "hf", "download", e.repo], check=True)
    _fetch_licenses(cfg)
    typer.echo("Setup complete. Run `uv run cinema doctor`.")


def _fetch_licenses(cfg) -> None:
    import httpx
    out = cfg.root / "models" / "licenses"
    out.mkdir(parents=True, exist_ok=True)
    urls = {
        "stable-audio-3-community-license.md":
            "https://huggingface.co/stabilityai/stable-audio-3-optimized/raw/main/LICENSE.md",
        "qwen-image-2.1-turbo-README.md": "https://huggingface.co/Qwen/Qwen-Image-2.1-Turbo/raw/main/README.md",
        "flux2-klein-4b-README.md": "https://huggingface.co/black-forest-labs/FLUX.2-klein-4B/raw/main/README.md",
        "wan2.2-ti2v-5b-LICENSE.txt": "https://huggingface.co/Wan-AI/Wan2.2-TI2V-5B/raw/main/LICENSE.txt",
        "qwen3-vl-32b-README.md": "https://huggingface.co/mlx-community/Qwen3-VL-32B-Instruct-4bit/raw/main/README.md",
        "kokoro-82m-README.md": "https://huggingface.co/hexgrad/Kokoro-82M/raw/main/README.md",
    }
    for name, url in urls.items():
        p = out / name
        if p.exists():
            continue
        try:
            r = httpx.get(url, timeout=30, follow_redirects=True)
            if r.status_code == 200:
                p.write_text(r.text)
        except httpx.HTTPError:
            pass


@app.command()
def models() -> None:
    """Installed models, licenses, runtime compatibility and recorded benchmarks."""
    from .model_registry import refresh
    for e in refresh(_cfg()):
        d = asdict(e)
        typer.echo(f"\n[{e.role}] {e.repo}\n  status={e.status} revision={e.revision} size_gb={e.size_gb} "
                   f"runtime={e.runtime}\n  license={e.license} | commercial: {e.commercial_use}"
                   f"\n  production={e.production} required={e.required}  {e.notes}")
        if d["benchmark"]:
            typer.echo("  benchmark: " + json.dumps(d["benchmark"])[:600])


# --------------------------------------------------------------------------- benchmarks / tests
@app.command()
def benchmark(stage: str = typer.Option(..., help="story | image | image-ab | video | validators | memory"),
              preset: str = typer.Option(None, help="video preset to benchmark"),
              writer: str = typer.Option(None, help="override writer model (story stage)"),
              writer_think: bool = typer.Option(None, "--writer-think/--no-writer-think"),
              job: str = typer.Option(None, help="job whose approved refs image-ab reuses")) -> None:
    """Benchmark a stage on this Mac with fixed prompts; results go to benchmarks/results.json."""
    from . import benchmarks
    _console_log()
    fn = {"story": benchmarks.story, "image": benchmarks.image, "video": benchmarks.video,
          "validators": benchmarks.validators, "memory": benchmarks.memory,
          "image-ab": benchmarks.image_ab}.get(stage)
    if fn is None:
        raise typer.BadParameter("stage must be story, image, video, validators or memory")
    cfg = _cfg()
    if writer:
        cfg.models.writer = writer
    if writer_think is not None:
        cfg.models.writer_think = writer_think
    if stage == "image-ab":
        if not job:
            raise typer.BadParameter("--job JOB_ID required for image-ab")
        _echo(fn(cfg, job))
        return
    _echo(fn(cfg, preset) if stage == "video" else fn(cfg))


@app.command("test-video")
def test_video(preset: str = typer.Option(None, help="preset name (default: config default)")) -> None:
    """Generate one short REAL moving clip (FLUX keyframe -> Wan I2V) and run technical checks."""
    from . import benchmarks
    _console_log()
    _echo(benchmarks.test_video(_cfg(), preset))


@app.command("test-pipeline")
def test_pipeline(duration: int = typer.Option(12, help="seconds (small multi-shot test)"),
                  genre: str = "mystery", preset: str = typer.Option(None)) -> None:
    """Small end-to-end autonomous test (3 shots by default) through every gate."""
    from .orchestrator import Orchestrator
    _console_log()
    orch = Orchestrator(_cfg())
    job = orch.create_job(genre, duration, preset, test=True)
    typer.echo(f"test job {job}")
    status = orch.run(job)
    typer.echo(f"{job}: {status}")
    raise typer.Exit(0 if status == "COMPLETE" else 2)


# --------------------------------------------------------------------------- production
@app.command()
def run(autonomous: bool = typer.Option(False, "--autonomous", help="required: no prompts, no downloads"),
        duration: int = typer.Option(45, min=30, max=60), genre: str = typer.Option("random"),
        topic: str = typer.Option(None, help='free-text creative brief, e.g. "future romance on Mars"'),
        preset: str = typer.Option(None, help="quality preset (see config.yaml)"),
        style: str = typer.Option(None, help="override visual style"),
        seed: int = typer.Option(None), detach: bool = typer.Option(False, help="run in background")) -> None:
    """Produce a new Short end-to-end without human interaction."""
    from .orchestrator import Orchestrator
    if not autonomous:
        typer.echo("Pass --autonomous to confirm a zero-touch production run.")
        raise typer.Exit(2)
    _sigterm_as_interrupt()
    orch = Orchestrator(_cfg())
    job = orch.create_job(genre, duration, preset, style, seed, brief=topic)
    if detach:
        _detach(["resume-internal", job])
        typer.echo(f"started {job} in background; `uv run cinema status {job}`")
        return
    _console_log()
    typer.echo(f"job {job}")
    status = orch.run(job)
    typer.echo(f"{job}: {status}")
    raise typer.Exit(0 if status == "COMPLETE" else 2)


@app.command("resume-internal", hidden=True)
def resume_internal(job_id: str) -> None:
    from .orchestrator import Orchestrator
    _sigterm_as_interrupt()
    _console_log()
    Orchestrator(_cfg()).run(job_id)


def _detach(args: list[str]) -> None:
    cfg = _cfg()
    log = cfg.root / "logs" / f"detached-{args[-1]}.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    subprocess.Popen([sys.executable, "-m", "cinema.cli", *args], stdout=log.open("a"), stderr=subprocess.STDOUT,
                     stdin=subprocess.DEVNULL, start_new_session=True, cwd=cfg.root,
                     env={**os.environ, "PYTHONPATH": str(cfg.root / "src")})


@app.command()
def resume(job_id: str, detach: bool = typer.Option(False)) -> None:
    """Resume an interrupted or BLOCKED job, reusing every validated asset."""
    from .orchestrator import Orchestrator
    if detach:
        _detach(["resume-internal", job_id])
        typer.echo(f"resuming {job_id} in background")
        return
    _sigterm_as_interrupt()
    _console_log()
    status = Orchestrator(_cfg()).resume(job_id)
    typer.echo(f"{job_id}: {status}")
    raise typer.Exit(0 if status == "COMPLETE" else 2)


@app.command()
def pause(job_id: str) -> None:
    """Pause after the current model step finishes (nothing is lost); continue with `cinema resume JOB_ID`."""
    cfg = _cfg()
    (cfg.root / "projects" / job_id / "PAUSE").write_text("pause requested\n")
    typer.echo(f"{job_id}: will pause after the current step (shot/image/validation) finishes.")


@app.command()
def status(job_id: str, events: int = 15) -> None:
    """Stage progress, asset states and recent events for a job."""
    from .storage.database import DB
    cfg = _cfg()
    db = DB(cfg.db_path)
    job = db.job(job_id)
    if not job:
        raise typer.BadParameter(f"unknown job {job_id}")
    assets = db.assets(job_id)
    by: dict[str, dict[str, int]] = {}
    for a in assets:
        by.setdefault(a["kind"], {}).setdefault(a["status"], 0)
        by[a["kind"]][a["status"]] += 1
    _echo({"job": {k: job[k] for k in ("id", "status", "genre", "duration", "preset", "created_at", "updated_at",
                                       "error", "resume_count")},
           "assets": by,
           "attention": [{k: a[k] for k in ("asset_id", "status", "attempts", "feedback")} for a in assets
                         if a["status"] in ("RETRY_PENDING", "BLOCKED", "FAILED")],
           "recent_events": [f"{e['created_at']} {e['level']}: {e['message']}" for e in db.events(job_id, events)],
           "output": str(cfg.root / "outputs" / job_id)})


@app.command()
def jobs() -> None:
    """All productions."""
    from .storage.database import DB
    cfg = _cfg()
    for j in DB(cfg.db_path).jobs():
        typer.echo(f"{j['id']}  {j['status']:<18} {j['genre']:<22} {j['duration']}s  {j['preset']:<9} "
                   f"{(j['error'] or '')[:80]}")


@app.command()
def cleanup(dry_run: bool = typer.Option(True, "--dry-run/--execute")) -> None:
    """Show (or delete with --execute) safe cleanup candidates: rejected candidates and scratch work files
    of COMPLETE/FAILED jobs. Approved assets, finals and reports are never touched."""
    from .storage.database import DB
    cfg = _cfg()
    db = DB(cfg.db_path)
    cands, total = [], 0
    for j in db.jobs():
        if j["status"] not in ("COMPLETE", "FAILED"):
            continue
        for sub in ("work", "temp"):
            d = cfg.root / "projects" / j["id"] / sub
            if d.exists():
                size = sum(p.stat().st_size for p in d.rglob("*") if p.is_file())
                cands.append((d, size))
                total += size
    for d, size in cands:
        typer.echo(f"{'DELETE' if not dry_run else 'would delete'} {d}  ({size / 1e9:.2f} GB)")
        if not dry_run:
            shutil.rmtree(d)
    typer.echo(f"total {'freed' if not dry_run else 'reclaimable'}: {total / 1e9:.2f} GB")


@app.command("publish-auth")
def publish_auth(platform: str = typer.Argument(..., help="youtube")) -> None:
    """One-time interactive sign-in for publishing (opens the browser)."""
    if platform != "youtube":
        raise typer.BadParameter("only youtube needs browser sign-in; Instagram uses secrets/instagram.json")
    from .publish.youtube import authorize
    authorize(_cfg())
    typer.echo("YouTube authorized (secrets/youtube_token.json).")


@app.command()
def bot() -> None:
    """Run the Telegram control bot (start / status / abort / publish; sends finished videos)."""
    from .telegram_bot import main
    main()


if __name__ == "__main__":
    app()
