"""Model adapters. Each heavyweight call goes through ResourceManager (lock, preflight, isolated
process, measured exit). Swapping a model means changing config + one adapter, not the pipeline."""
from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

from .config import Config, Preset
from .resource_manager import ResourceManager, WorkerResult
from .storage.files import atomic_json

MLXGEN = shutil.which("mlxgen") or str(Path.home() / ".local/bin/mlxgen")


def batch_worker(rm: ResourceManager, cfg: Config, role: str, module: str, model: str, options: dict,
                 items: list[dict], workdir: Path, job_id: str | None, timeout: float | None = None) -> dict:
    """Run cinema.workers.<module> on items; returns {item_id: result}. Resumable per item."""
    workdir.mkdir(parents=True, exist_ok=True)
    req, resp = workdir / f"{role}_request.json", workdir / f"{role}_response.json"
    resp.unlink(missing_ok=True)
    atomic_json(req, {"options": options, "items": items})
    rm.run_worker(role, model, [sys.executable, "-m", f"cinema.workers.{module}", str(req), str(resp)],
                  workdir / f"{role}.log", job_id, timeout=timeout, cwd=cfg.root,
                  env={"PYTHONPATH": str(cfg.root / "src"), "HF_HUB_OFFLINE": "1", "TOKENIZERS_PARALLELISM": "false"})
    data = json.loads(resp.read_text()) if resp.exists() else {}
    return data.get("results", {})


# ------------------------------------------------------------------ MLX-Gen image (FLUX.2 Klein)

def flux_image(rm: ResourceManager, cfg: Config, prompt: str, out: Path, width: int, height: int,
               seed: int, steps: int, log: Path, job_id: str | None,
               references: list[Path] | None = None) -> WorkerResult:
    if cfg.image_backend == "cloudflare":
        from .cloudflare import AllAccountsExhausted, flux2_dev
        try:
            meta = flux2_dev(cfg, prompt, out, width, height, seed, references)
            return WorkerResult("image", 0, meta["seconds"], 0.0, None, {}, {}, True,
                                extra={"backend": "cloudflare", **meta})
        except AllAccountsExhausted as exc:
            import logging
            logging.getLogger("cinema").warning("Cloudflare unavailable (%s); using local FLUX.2 klein 4B", exc)
        except RuntimeError as exc:
            from .resource_manager import WorkerFailed
            raise WorkerFailed(f"cloudflare image failed: {exc}") from exc
    return _flux_local(rm, cfg, prompt, out, width, height, seed, steps, log, job_id, references)


def _flux_local(rm: ResourceManager, cfg: Config, prompt: str, out: Path, width: int, height: int,
                seed: int, steps: int, log: Path, job_id: str | None,
                references: list[Path] | None = None) -> WorkerResult:
    cmd = [MLXGEN, "generate", "--model", cfg.models.image, "--prompt", prompt, "--width", str(width),
           "--height", str(height), "--steps", str(steps), "--guidance", "1.0", "--seed", str(seed),
           "--metadata", "--output", str(out)]
    for r in references or []:
        cmd += ["--image", str(r)]
    if references:
        # Measured: FLUX.2 rejects --i2i-mode edit, and rejects multi-reference with a single image (the
        # automatic route handles one reference). Without exact-resize the canvas follows the first
        # reference's aspect (3:4 portrait -> 544x736), not the requested 9:16 size.
        if len(references) > 1:
            cmd += ["--i2i-mode", "multi-reference"]
        cmd += ["--canvas-policy", "exact-resize"]
    return rm.run_worker("image", cfg.models.image, cmd, log, job_id, timeout=1800,
                         env={"HF_HUB_OFFLINE": "1"})


# ------------------------------------------------------------------ MLX-Gen video (Wan2.2 TI2V-5B)

def wan_frames(duration: float, preset: Preset) -> int:
    """Smallest 4n+1 frame count covering the duration (plus a small trim margin), capped."""
    need = int(duration * preset.fps + 0.999) + 2
    n = 4 * ((need - 1 + 3) // 4) + 1
    return min(n, preset.max_frames)


def wan_video(rm: ResourceManager, cfg: Config, prompt: str, negative: str, image: Path, out: Path,
              preset: Preset, frames: int, seed: int, log: Path, job_id: str | None,
              steps: int | None = None, guidance: float | None = None) -> WorkerResult:
    cmd = [MLXGEN, "generate", "--model", cfg.models.video, "--prompt", prompt, "--negative-prompt", negative,
           "--image", str(image), "--width", str(preset.width), "--height", str(preset.height),
           "--frames", str(frames), "--fps", str(preset.fps), "--steps", str(steps or preset.steps),
           "--guidance", str(guidance or preset.guidance), "--flow-shift", str(preset.flow_shift),
           "--seed", str(seed), "--metadata", "--json-events", "--canvas-policy", "exact-resize",
           "--output", str(out)]
    if preset.low_ram:
        cmd.append("--low-ram")
    return rm.run_worker("video", cfg.models.video, cmd, log, job_id, timeout=3 * 3600,
                         env={"HF_HUB_OFFLINE": "1"}, disk_gib=3)
