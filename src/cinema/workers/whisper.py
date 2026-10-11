"""Speech validator worker (MLX Whisper large-v3-turbo).

item: {"id", "audio"} -> {"text", "words": [{"word","start","end","probability"}], "language"}
"""
from __future__ import annotations

from .common import run_batch


def factory(opts: dict):
    import mlx_whisper

    repo = opts["model"]

    def handle(item: dict) -> dict:
        r = mlx_whisper.transcribe(item["audio"], path_or_hf_repo=repo, language="en", word_timestamps=True,
                                   condition_on_previous_text=False, temperature=0.0,
                                   initial_prompt=item.get("prompt") or None)
        words = [{"word": w["word"].strip(), "start": round(w["start"], 3), "end": round(w["end"], 3),
                  "probability": round(w.get("probability", 0.0), 3)}
                 for seg in r.get("segments", []) for w in seg.get("words", [])]
        return {"text": r.get("text", "").strip(), "words": words, "language": r.get("language")}

    return handle


if __name__ == "__main__":
    run_batch(factory)
