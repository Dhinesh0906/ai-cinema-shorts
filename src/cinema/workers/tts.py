"""Kokoro-82M (kokoro-onnx) speech worker.

item: {"id", "text", "voice", "speed", "out"} -> {"path", "seconds", "sample_rate"}
"""
from __future__ import annotations

from .common import run_batch


def factory(opts: dict):
    import soundfile as sf
    from kokoro_onnx import Kokoro

    kokoro = Kokoro(opts["model"], opts["voices"])

    def handle(item: dict) -> dict:
        samples, sr = kokoro.create(item["text"], voice=item["voice"], speed=float(item.get("speed", 1.0)),
                                    lang="en-us")
        sf.write(item["out"], samples, sr, subtype="PCM_16")
        return {"path": item["out"], "seconds": round(len(samples) / sr, 3), "sample_rate": sr}

    return handle


if __name__ == "__main__":
    run_batch(factory)
