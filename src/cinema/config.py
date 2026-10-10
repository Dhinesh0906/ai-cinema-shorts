"""Runtime configuration loaded from config.yaml (falls back to config.example.yaml)."""
from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import BaseModel, Field, model_validator

PROJECT_ROOT = Path(__file__).resolve().parents[2]


class Preset(BaseModel):
    """Native generation settings. Export is always 1080x1920; native size is recorded separately."""
    width: int = Field(ge=256)
    height: int = Field(ge=256)
    fps: int = Field(ge=8, le=30)
    max_frames: int = Field(ge=17, le=121)
    steps: int = Field(ge=4)
    guidance: float = 5.0
    flow_shift: float = 3.0
    low_ram: bool = False
    image_steps: int = Field(default=4, ge=1)
    portrait_width: int = 768
    portrait_height: int = 1024

    @model_validator(mode="after")
    def _shape(self) -> "Preset":
        if self.width >= self.height:
            raise ValueError("preset must be vertical (width < height)")
        if self.width % 32 or self.height % 32:
            raise ValueError("Wan TI2V-5B needs 32-pixel multiples")
        if (self.max_frames - 1) % 4:
            raise ValueError("max_frames must be 4n+1 for the Wan temporal VAE")
        return self


class Models(BaseModel):
    writer: str = "gemma4:31b"
    # Measured: Ollama 0.30.10 only enforces the JSON-schema `format` for qwen3.5 when thinking is on.
    writer_think: bool = False
    critic: str = "gpt-oss:120b"
    critic_think: str = "medium"
    vlm: str = "mlx-community/Qwen3-VL-32B-Instruct-4bit"
    image: str = "AbstractFramework/flux.2-klein-4b-4bit"
    video: str = "AbstractFramework/wan2.2-ti2v-5b-diffusers-8bit"
    whisper: str = "mlx-community/whisper-large-v3-turbo"
    clap: str = "laion/clap-htsat-unfused"
    kokoro_model: str = "models/kokoro-v1.0.onnx"
    kokoro_voices: str = "models/voices-v1.0.bin"
    sa3_dir: str = "~/Desktop/stable-audio-3/optimized/mlx"


class Cloud(BaseModel):
    """Ollama Cloud (ollama.com) for LLMs listed in `models`; everything else stays local."""
    models: list[str] = ["gemma4:31b", "gpt-oss:120b"]
    keys_file: str = "secrets/ollama_api_keys.json"
    cycle_pause_s: float = 30.0          # pause after every key was rate limited, then start over
    max_error_seconds: float = 1800.0    # network/5xx errors (not rate limits) give up after this


class Cloudflare(BaseModel):
    """FLUX.2 [dev] on Cloudflare Workers AI (used when image_backend == "cloudflare")."""
    model: str = "@cf/black-forest-labs/flux-2-dev"
    keys_file: str = "secrets/cloudflare_accounts.json"
    steps: int = 25
    send_seed: bool = True
    ref_max_side: int = 504          # API requires input images smaller than 512x512
    timeout_s: float = 300
    on_all_exhausted: str = "wait"   # wait: reset ring and keep cycling | local: use local klein 4B
    cycle_pause_s: float = 900       # daily quotas reset at 00:00 UTC; don't hammer exhausted accounts
    max_server_errors: int = 3       # 5xx responses before handing the image to local klein 4B


class OpenRouterVideo(BaseModel):
    """Opportunistic cloud video (video_backend: openrouter); falls back to local Wan 2.2."""
    model: str = "alibaba/wan-2.6:free"
    keys_file: str = "secrets/openrouter_keys.json"
    resolution: str = "720p"
    durations: list[int] = [5, 10]
    poll_s: float = 15
    timeout_s: float = 1200


class Retries(BaseModel):
    max_attempts: int = Field(default=3, ge=1, le=10)
    story_rewrites: int = Field(default=3, ge=1, le=10)
    final_repairs: int = Field(default=2, ge=0, le=5)


class Thresholds(BaseModel):
    # VLM checklist: fraction of non-mandatory items that must be "yes".
    image_optional_ratio: float = 0.6
    video_optional_ratio: float = 0.5
    # Technical video gates.
    min_mean_flow: float = 0.15          # px/frame at 256px analysis width; below = frozen
    max_freeze_fraction: float = 0.35
    max_black_fraction: float = 0.2
    max_temporal_spike: float = 6.0      # max/median adjacent-frame difference
    max_feature_drift: float = 0.0      # 0 disables; calibrated by `cinema benchmark --stage validators`
    # Speech.
    max_wer: float = 0.15
    max_peak_dbfs: float = -0.5
    min_mean_db: float = -35.0
    # Final mix.
    final_max_wer: float = 0.25
    target_lufs: float = -14.0


class MemoryPolicy(BaseModel):
    min_free_gib_disk: float = Field(default=100, ge=100)
    launch_min_available_gib: float = 8.0   # headroom required on top of the model estimate
    recovery_timeout_s: int = 180
    worker_timeout_s: int = 3 * 3600
    sample_interval_s: float = 0.5


class Config(BaseModel):
    root: Path = PROJECT_ROOT
    ollama_url: str = "http://127.0.0.1:11434"
    models: Models = Models()
    cloud: Cloud = Cloud()
    image_backend: str = "local"     # local (FLUX.2 klein 4B, MLX) | cloudflare (FLUX.2 dev)
    cloudflare: Cloudflare = Cloudflare()
    channel_name: str = "Flicker Velle"
    publish_privacy: str = "public"  # YouTube privacyStatus (unaudited API projects are forced private)
    publish_platforms: list[str] = ["youtube"]  # auto-posted; Instagram is posted by hand (caption in preview)
    video_backend: str = "local"     # local (Wan2.2 TI2V-5B MLX) | openrouter (wan-2.6:free, local fallback)
    openrouter: OpenRouterVideo = OpenRouterVideo()
    retries: Retries = Retries()
    thresholds: Thresholds = Thresholds()
    memory: MemoryPolicy = MemoryPolicy()
    default_preset: str = "balanced"
    presets: dict[str, Preset]
    shot_min_seconds: float = 2.5
    shot_max_seconds: float = 5.0
    narration_wps: float = 2.6
    burn_captions: bool = True
    generate_music: bool = True
    generate_sfx: bool = True
    image_candidates: int = Field(default=2, ge=1, le=4)
    experimental: bool = False
    visual_style: str = ("photorealistic cinematic film still, anamorphic 35mm, natural skin texture, "
                         "motivated practical lighting, subtle film grain")

    @model_validator(mode="after")
    def _preset(self) -> "Config":
        if self.default_preset not in self.presets:
            raise ValueError(f"default_preset {self.default_preset!r} not in presets")
        return self

    def path(self, *parts: str) -> Path:
        return self.root.joinpath(*parts)

    @property
    def db_path(self) -> Path:
        return self.root / "state" / "cinema.sqlite3"

    def is_cloud(self, model: str) -> bool:
        return model in self.cloud.models

    def resolve(self, p: str) -> Path:
        q = Path(p).expanduser()
        return q if q.is_absolute() else self.root / q


def load_config(path: Path | None = None) -> Config:
    path = path or PROJECT_ROOT / "config.yaml"
    if not path.exists():
        path = PROJECT_ROOT / "config.example.yaml"
    data = yaml.safe_load(path.read_text()) or {}
    cfg = Config.model_validate(data)
    if not cfg.root.is_absolute():
        cfg.root = (path.parent / cfg.root).resolve()
    return cfg
