"""Vision-language validator worker (MLX-VLM, Qwen3-VL). Loads once, answers a batch, exits.

item: {"id", "prompt", "images": [paths], "max_tokens"?}
-> {"text", "seconds", "prompt_tokens"?, "peak_memory_gb"?}
"""
from __future__ import annotations

import tempfile
import time
from pathlib import Path

from PIL import Image

from .common import run_batch


def _prep(paths: list[str], max_side: int, tmp: Path) -> list[str]:
    out = []
    for i, p in enumerate(paths):
        im = Image.open(p).convert("RGB")
        scale = max_side / max(im.size)
        if scale < 1:
            im = im.resize((max(28, int(im.width * scale)), max(28, int(im.height * scale))), Image.LANCZOS)
        q = tmp / f"{i:02d}.png"
        im.save(q)
        out.append(str(q))
    return out


def factory(opts: dict):
    import mlx.core as mx
    from mlx_vlm import generate, load
    from mlx_vlm.prompt_utils import apply_chat_template

    model, processor = load(opts["model"])
    config = model.config
    max_side = int(opts.get("max_side", 672))

    def handle(item: dict) -> dict:
        t0 = time.monotonic()
        with tempfile.TemporaryDirectory() as td:
            images = _prep(item["images"], int(item.get("max_side", max_side)), Path(td))
            prompt = apply_chat_template(processor, config, item["prompt"], num_images=len(images))
            res = generate(model, processor, prompt, images, max_tokens=int(item.get("max_tokens", 1500)),
                           temperature=0.0, verbose=False)
        text = getattr(res, "text", res)
        return {"text": text, "seconds": round(time.monotonic() - t0, 2),
                "prompt_tokens": getattr(res, "prompt_tokens", None),
                "generation_tps": round(getattr(res, "generation_tps", 0) or 0, 1),
                "peak_memory_gb": round(mx.get_peak_memory() / 1e9, 2)}

    return handle


if __name__ == "__main__":
    run_batch(factory)
