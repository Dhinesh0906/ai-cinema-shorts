"""Stable Audio 3 (pure MLX) sound-effect / ambience / music worker.

Runs the official sa3_mlx script from its own virtualenv once per item, inside this worker's
process group (so the resource manager's exit checks cover it). stdin is closed so the script
can never stop at an interactive prompt.

item: {"id", "prompt", "negative"?, "seconds", "kind": "sfx"|"ambience"|"music", "seed", "out"}
"""
from __future__ import annotations

import subprocess
import time
from pathlib import Path

from .common import run_batch

DIT = {"sfx": ("sm-sfx", "same-s"), "ambience": ("sm-sfx", "same-s"), "music": ("sm-music", "same-s")}


def factory(opts: dict):
    sa3 = Path(opts["sa3_dir"]).expanduser()
    py = sa3 / ".venv" / "bin" / "python"
    script = sa3 / "scripts" / "sa3_mlx.py"
    if not py.exists() or not script.exists():
        raise FileNotFoundError(f"Stable Audio 3 MLX not installed at {sa3}")

    def handle(item: dict) -> dict:
        dit, dec = DIT[item["kind"]]
        cmd = [str(py), str(script), "--prompt", item["prompt"], "--dit", dit, "--decoder", dec,
               "--seconds", f"{float(item['seconds']):.2f}", "--seed", str(int(item["seed"])),
               "--steps", str(int(item.get("steps", 8))), "--out", item["out"]]
        if item.get("negative"):
            cmd += ["--negative-prompt", item["negative"], "--cfg", str(item.get("cfg", 3.0))]
        t0 = time.monotonic()
        r = subprocess.run(cmd, cwd=sa3, stdin=subprocess.DEVNULL, capture_output=True, text=True, timeout=900)
        if r.returncode or not Path(item["out"]).exists():
            raise RuntimeError(f"sa3 failed ({r.returncode}): {r.stderr[-800:] or r.stdout[-800:]}")
        return {"path": item["out"], "seconds": round(time.monotonic() - t0, 2), "dit": dit, "decoder": dec}

    return handle


if __name__ == "__main__":
    run_batch(factory)
