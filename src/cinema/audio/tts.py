"""Speech line planning for Kokoro: one file per narration line / dialogue line, stable voices."""
from __future__ import annotations

import re
from dataclasses import dataclass

from ..schemas import Screenplay


@dataclass
class Line:
    asset_id: str
    shot_id: str
    kind: str           # narration | dialogue
    speaker: str        # "narrator" or character id
    voice: str
    text: str
    window: float       # seconds available inside the shot


def speakable(text: str) -> str:
    text = text.replace("…", "...").replace("—", ", ").replace("–", ", ")
    text = re.sub(r"\s+", " ", text).strip()
    return text


def plan_lines(sp: Screenplay, lead: float = 0.3, tail: float = 0.2) -> list[Line]:
    lines: list[Line] = []
    for s in sp.shots:
        window = max(0.8, s.duration - lead - tail)
        if s.narration.strip():
            lines.append(Line(f"narr_{s.id}", s.id, "narration", "narrator", sp.story.narrator_voice_id,
                              speakable(s.narration), window))
        for i, d in enumerate(s.dialogue, 1):
            lines.append(Line(f"dlg_{s.id}_{i}", s.id, "dialogue", d.speaker_id, sp.character(d.speaker_id).voice_id,
                              speakable(d.text), window))
    return lines


def speed_for(seconds_at_1x: float, window: float, max_speed: float = 1.15) -> float | None:
    """Speed needed to fit the window; None if even max_speed doesn't fit (needs a shorter line)."""
    if seconds_at_1x <= window:
        return 1.0
    need = seconds_at_1x / window
    return round(need + 0.02, 2) if need <= max_speed else None
