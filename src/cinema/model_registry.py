"""Model registry: repository, revision, license, size, runtime, path, download + validation status.

Licenses were read from the official model cards (see models/licenses/ after `cinema setup`).
Autonomous runs only use models that are already present; `autonomous_readiness` blocks otherwise.
"""
from __future__ import annotations

import importlib.util
import json
import shutil
from dataclasses import asdict, dataclass, field
from pathlib import Path

import httpx

from .config import Config
from .storage.files import atomic_json

HF = Path.home() / ".cache" / "huggingface" / "hub"


@dataclass
class Entry:
    role: str
    repo: str
    runtime: str
    license: str
    commercial_use: str
    source: str
    notes: str = ""
    required: bool = True
    production: bool = True
    revision: str | None = None
    size_gb: float | None = None
    path: str | None = None
    status: str = "unknown"
    benchmark: dict = field(default_factory=dict)


def catalog(cfg: Config) -> list[Entry]:
    m = cfg.models
    return [
        Entry("writer", m.writer, "ollama-cloud" if cfg.is_cloud(m.writer) else "ollama", "Apache-2.0",
              "permitted (model); Ollama Cloud free-tier terms apply", "https://ollama.com/library/gemma4",
              "story, shot list, rewrites, edit director"),
        Entry("critic", m.critic, "ollama-cloud" if cfg.is_cloud(m.critic) else "ollama", "Apache-2.0",
              "permitted (model); Ollama Cloud free-tier terms apply", "https://ollama.com/library/gpt-oss",
              "independent script critic + final narrative critic (different model family from writer)"),
        Entry("vlm", m.vlm, "mlx-vlm", "Apache-2.0", "permitted", f"https://huggingface.co/{m.vlm}",
              "independent image/video/final visual validator"),
        Entry("image", m.image, "mlx-gen (FLUX.2)", "Apache-2.0", "permitted", f"https://huggingface.co/{m.image}",
              "FLUX.2 [klein] 4B (Apache-2.0; the 9B variant is non-commercial and NOT used)"),
        Entry("video", m.video, "mlx-gen (Wan2.2)", "Apache-2.0", "permitted", f"https://huggingface.co/{m.video}",
              "Wan2.2 TI2V-5B, 8-bit MLX package; runtime promotes sensitive paths to BF16"),
        Entry("whisper", m.whisper, "mlx-whisper", "MIT", "permitted", f"https://huggingface.co/{m.whisper}",
              "independent speech transcription"),
        Entry("tts", "hexgrad/Kokoro-82M (v1.0 ONNX via kokoro-onnx)", "onnxruntime", "Apache-2.0 (model), MIT (kokoro-onnx)",
              "permitted", "https://github.com/thewh1teagle/kokoro-onnx", "narration and voices",
              path=str(cfg.resolve(m.kokoro_model))),
        Entry("clap", m.clap, "transformers (CPU)", "Apache-2.0", "permitted", f"https://huggingface.co/{m.clap}",
              "sound-effect relevance (supporting evidence only)", required=False),
        Entry("sfx", "stabilityai/stable-audio-3-optimized (sm-sfx, sm-music, MLX)", "sa3_mlx",
              "Stability AI Community License", "permitted for individuals/orgs under USD 1M annual revenue; "
              "enterprise license above that", "https://huggingface.co/stabilityai/stable-audio-3-optimized",
              "ambience, sound effects, instrumental score", required=False, path=str(Path(m.sa3_dir).expanduser())),
        Entry("image-cloud", cfg.cloudflare.model, "cloudflare-workers-ai",
              "FLUX Non-Commercial License (weights); outputs usable commercially per model card",
              "outputs: personal, scientific and commercial use per BFL model card; Cloudflare/BFL ToS apply",
              "https://developers.cloudflare.com/workers-ai/models/flux-2-dev/",
              "optional backend (image_backend: cloudflare), rotating account ring",
              required=cfg.image_backend == "cloudflare", production=cfg.image_backend == "cloudflare"),
        Entry("experimental-image", "Qwen/Qwen-Image-2.1-Turbo", "not installed", "Qwen Research License",
              "NOT permitted for commercial use without a separate license", "https://huggingface.co/Qwen/Qwen-Image-2.1-Turbo",
              "excluded from production; no verified MLX route for the Turbo checkpoint; 32.5 GB",
              required=False, production=False),
        Entry("experimental-video", "ByteDance Bernini-R 1.3B (via mlx-gen)", "not installed", "unverified",
              "unverified", "https://github.com/lpalbou/mlx-gen/blob/main/docs/bernini.md",
              "challenger not benchmarked; not downloaded", required=False, production=False),
    ]


def _hf_status(repo: str) -> tuple[str, str | None, float | None, str | None]:
    d = HF / ("models--" + repo.replace("/", "--"))
    if not d.exists():
        return "missing", None, None, None
    if list((d / "blobs").glob("*.incomplete")):
        return "downloading", None, None, str(d)
    ref = d / "refs" / "main"
    rev = ref.read_text().strip() if ref.exists() else None
    size = sum(p.stat().st_size for p in (d / "blobs").iterdir() if p.is_file()) / 1e9
    snap = d / "snapshots" / rev if rev else None
    return ("present" if snap and snap.exists() else "incomplete"), rev, round(size, 2), str(snap or d)


def _ollama_status(cfg: Config, name: str) -> tuple[str, str | None, float | None]:
    try:
        tags = httpx.get(f"{cfg.ollama_url}/api/tags", timeout=10).json().get("models", [])
    except httpx.HTTPError:
        return "ollama-offline", None, None
    for t in tags:
        if t["name"] == name or t["name"] == f"{name}:latest":
            return "present", t.get("digest", "")[:12], round(t.get("size", 0) / 1e9, 2)
    return "missing", None, None


def refresh(cfg: Config) -> list[Entry]:
    entries = catalog(cfg)
    bench = {}
    bpath = cfg.root / "benchmarks" / "results.json"
    if bpath.exists():
        bench = json.loads(bpath.read_text())
    for e in entries:
        if e.runtime == "ollama-cloud":
            try:
                from .cloud import list_models
                e.status = "present" if e.repo in list_models(cfg) else "missing"
                e.revision = "ollama.com (cloud)"
            except Exception as exc:
                e.status = f"cloud-unreachable ({type(exc).__name__})"
        elif e.runtime == "cloudflare-workers-ai":
            f = cfg.resolve(cfg.cloudflare.keys_file)
            n = len(json.loads(f.read_text()).get("accounts", [])) if f.exists() else 0
            e.status = f"present ({n} accounts)" if n else "missing (no accounts file)"
            if cfg.image_backend == "cloudflare" and n:
                e.status = "present"
            e.revision = "workers-ai (cloud)"
        elif e.runtime == "ollama":
            e.status, e.revision, e.size_gb = _ollama_status(cfg, e.repo)
        elif e.role == "tts":
            p1, p2 = cfg.resolve(cfg.models.kokoro_model), cfg.resolve(cfg.models.kokoro_voices)
            e.status = "present" if p1.exists() and p2.exists() else "missing"
            e.size_gb = round((p1.stat().st_size + p2.stat().st_size) / 1e9, 2) if e.status == "present" else None
            e.revision = "kokoro-v1.0"
        elif e.role == "sfx":
            p = Path(e.path or "")
            e.status = "present" if (p / ".venv" / "bin" / "python").exists() and (p / "models").exists() else "missing"
            e.revision = "sa3-optimized (local clone)"
        elif e.production:
            e.status, e.revision, e.size_gb, e.path = _hf_status(e.repo)
        else:
            e.status = "not installed"
        e.benchmark = bench.get(e.role, {})
    atomic_json(cfg.root / "models" / "registry.json", [asdict(e) for e in entries])
    return entries


def autonomous_readiness(cfg: Config, rm=None) -> list[str]:
    """Everything an autonomous run needs must already be local. Returns blocking problems."""
    problems = []
    for tool in ("ffmpeg", "ffprobe", "caffeinate"):
        if not shutil.which(tool):
            problems.append(f"{tool} not found")
    if not (shutil.which("mlxgen") or (Path.home() / ".local/bin/mlxgen").exists()):
        problems.append("mlxgen not installed (uv tool install mlx-gen)")
    for mod in ("mlx_vlm", "mlx_whisper", "cv2", "kokoro_onnx", "Vision", "psutil"):
        if importlib.util.find_spec(mod) is None:
            problems.append(f"python module {mod} missing (uv sync)")
    for e in refresh(cfg):
        if e.required and e.production and e.status != "present":
            problems.append(f"{e.role} model {e.repo}: {e.status}")
    free = shutil.disk_usage(cfg.root).free / 1024 ** 3
    if free < cfg.memory.min_free_gib_disk + 20:
        problems.append(f"only {free:.0f} GiB free; need {cfg.memory.min_free_gib_disk} GiB reserve + 20 GiB working space")
    return problems
