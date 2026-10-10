"""CLAP audio-text similarity worker (laion/clap-htsat-unfused via transformers, CPU).

Supporting evidence only: a sound passes relevance when its own prompt ranks above unrelated
distractor descriptions. item: {"id", "audio", "texts": [target, *distractors]} -> {"scores": [...]}
"""
from __future__ import annotations

from .common import run_batch


def factory(opts: dict):
    import subprocess

    import numpy as np
    import torch
    from transformers import ClapModel, ClapProcessor

    model = ClapModel.from_pretrained(opts["model"]).eval()
    proc = ClapProcessor.from_pretrained(opts["model"])

    def handle(item: dict) -> dict:
        raw = subprocess.run(["ffmpeg", "-v", "error", "-i", item["audio"], "-ac", "1", "-ar", "48000", "-f", "f32le", "-"],
                             capture_output=True, check=True).stdout
        audio = np.frombuffer(raw, np.float32)
        inputs = proc(text=item["texts"], audio=[audio], sampling_rate=48000, return_tensors="pt", padding=True)
        with torch.no_grad():
            out = model(**inputs)
        scores = out.logits_per_audio[0].softmax(-1).tolist()
        return {"scores": [round(s, 4) for s in scores]}

    return handle


if __name__ == "__main__":
    run_batch(factory)
