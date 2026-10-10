"""AI editing director: the writer LLM chooses in/out points, transitions, grade and mix levels from the
ACTUAL approved media durations. Its plan is validated deterministically; invalid plans get the errors
fed back; after retries a documented deterministic fallback plan is used (and labelled as such)."""
from __future__ import annotations

import json

from ..config import Config
from ..llm import LLMError, chat_json
from ..schemas import EditPlan, Screenplay
from .edl import EDLError, ShotMedia, SoundMedia, SpeechMedia, default_plan, resolve

DIRECTOR_SYSTEM = """You are a professional film editor cutting a vertical cinematic short from APPROVED AI-generated
shots. Principles: strong opening, purposeful cuts, rhythm that follows the drama, minimal transitions (mostly
hard cuts; a crossfade or dip only where time passes or mood shifts), clear narration over a quiet bed, no
over-processing. You may trim the start/end of a clip (generated clips often settle in the first ~0.3s), but
keep every shot, in screenplay order."""


def direct(cfg: Config, job_id: str, sp: Screenplay, media: dict[str, ShotMedia], speech: list[SpeechMedia],
           sounds: list[SoundMedia], session: dict, seed: int, min_total: float, max_total: float, note: str = ""):
    shots = [{"shot_id": s.id, "beat": s.beat, "planned_seconds": s.duration,
              "clip_seconds": round(media[s.id].clip_duration, 3), "action": s.action,
              "narration_seconds": round(sum(x.duration for x in speech if x.shot_id == s.id), 2)}
             for s in sp.shots]
    user = f"""Story: {sp.story.title} — {sp.story.logline}. Genre: {sp.story.genre}. Palette: {sp.story.color_palette}.
Approved shots (in order): {json.dumps(shots)}
Constraints:
- include every shot in order; 0 <= in_point < out_point <= clip_seconds; each segment >= 1.5s
- each segment should be at least narration_seconds + 0.5 long so its narration fits
- total edited duration (sum of segments minus crossfade/dip overlaps) must be between {min_total:.1f} and
  {max_total:.1f} seconds; aim for {sp.target_seconds}s
- transition_in for the first shot: cut or fade_from_black; transition_seconds 0 for cut, 0.2-0.8 otherwise
- choose one grade that suits the genre; music quieter than ambience under narration (-24..-14 dB typical)
Return JSON only."""
    if note:
        user += f"\nQuality control rejected the previous cut: {note}. Address this."
    feedback = ""
    for attempt in range(3):
        try:
            plan = chat_json(cfg, cfg.models.writer, DIRECTOR_SYSTEM, user + feedback, EditPlan, session=session,
                             temperature=0.4, seed=seed + attempt)
            return resolve(job_id, plan, sp, media, speech, sounds, "ai_director", min_total, max_total), None
        except (EDLError, LLMError) as exc:
            feedback = f"\nYour previous plan was rejected: {exc}. Fix it."
            last = str(exc)
    plan = default_plan(sp, media, sp.target_seconds)
    return resolve(job_id, plan, sp, media, speech, sounds, "deterministic_fallback", min_total, max_total), last
