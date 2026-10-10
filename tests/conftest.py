import subprocess
from pathlib import Path

import pytest

from cinema.config import load_config


@pytest.fixture
def cfg(tmp_path):
    c = load_config()
    c = c.model_copy(update={"root": tmp_path})
    c.memory.recovery_timeout_s = 5
    c.memory.launch_min_available_gib = 0.5
    c.memory.min_free_gib_disk = 100
    (tmp_path / "state").mkdir()
    return c


def make_clip(path: Path, kind: str = "moving", seconds: float = 3.0, size: str = "480x832", fps: int = 16) -> Path:
    src = {"moving": f"testsrc2=s={size}:r={fps}:d={seconds}",
           "still": f"color=c=gray:s={size}:r={fps}:d={seconds}",
           "black": f"color=c=black:s={size}:r={fps}:d={seconds}"}[kind]
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i", src, "-pix_fmt", "yuv420p", str(path)],
                   check=True)
    return path


def make_tone(path: Path, seconds: float = 1.5, freq: int = 440, volume: float = 0.3) -> Path:
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i",
                    f"sine=frequency={freq}:duration={seconds}:sample_rate=24000", "-af", f"volume={volume}",
                    str(path)], check=True)
    return path
