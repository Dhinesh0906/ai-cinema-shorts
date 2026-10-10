"""Sound design plan: per-location ambience beds, per-shot effects, one instrumental score.
Generated locally by Stable Audio 3 (Stability AI Community License); relevance is checked with CLAP
against distractor descriptions (supporting evidence only)."""
from __future__ import annotations

import hashlib
from dataclasses import dataclass

from ..schemas import Screenplay

DISTRACTORS = ["upbeat pop song with vocals", "crowd cheering in a stadium", "birds chirping on a sunny morning",
               "car engine revving", "people talking in a cafe", "electronic dance beat", "dog barking",
               "silence"]


@dataclass
class SoundTask:
    asset_id: str
    kind: str            # sfx | ambience | music
    prompt: str
    seconds: float
    seed: int
    shot_id: str | None = None
    location_id: str | None = None


def _seed(text: str, base: int) -> int:
    return (int(hashlib.sha1(text.encode()).hexdigest()[:8], 16) + base) % (2 ** 31)


def plan_sounds(sp: Screenplay, base_seed: int, music: bool, sfx: bool) -> list[SoundTask]:
    tasks: list[SoundTask] = []
    if sfx:
        for loc in sp.story.locations:
            secs = sum(s.duration for s in sp.shots if s.location_id == loc.id)
            if secs <= 0:
                continue
            tasks.append(SoundTask(f"amb_{loc.id}", "ambience",
                                   f"{loc.ambience_sound}, continuous ambient background, no music, no voices",
                                   min(round(secs + 1.0, 2), 60.0), _seed(loc.id, base_seed), location_id=loc.id))
        for s in sp.shots:
            for i, fx in enumerate(s.sfx[:2], 1):
                tasks.append(SoundTask(f"sfx_{s.id}_{i}", "sfx", f"{fx}, isolated sound effect, no music",
                                       min(4.0, s.duration), _seed(s.id + fx, base_seed), shot_id=s.id))
    if music:
        tasks.append(SoundTask("music", "music",
                               f"{sp.story.music_mood}, cinematic instrumental film score, no vocals",
                               float(sp.target_seconds) + 1.0, _seed("music", base_seed)))
    return tasks


def distractors_for(task: SoundTask) -> list[str]:
    return [d for d in DISTRACTORS if d.split()[0] not in task.prompt.lower()][:6]
