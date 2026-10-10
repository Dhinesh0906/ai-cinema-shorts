"""AUTONOMOUS EDITING: AI director plans the cut on actual media -> validated EDL -> FFmpeg render.
Also writes the YouTube metadata (title/description/tags with synthetic-content disclosure)."""
from __future__ import annotations

import json
from pathlib import Path

from pydantic import BaseModel, Field

from ..editing.director import direct
from ..editing.edl import ShotMedia, SoundMedia, SpeechMedia
from ..editing.ffmpeg_renderer import render, write_srt
from ..llm import LLMError, chat_json
from ..schemas import EDL, Decision, Screenplay
from ..state_machine import AssetState, JobState
from ..storage.files import atomic_json, read_json
from ..video.motion_analysis import stream_info
from .context import Ctx


class PublishMeta(BaseModel):
    title: str = Field(max_length=100)
    description: str
    tags: list[str] = Field(min_length=3, max_length=15)


def duration_bounds(target: int, test: bool = False) -> tuple[float, float]:
    """Production films must land in 30-60 s; the small `test-pipeline` job is exempt from that floor."""
    if test:
        return target - 3.0, target + 3.0
    return max(30.0, target - 3.0), min(60.0, target + 3.0)


def gather(ctx: Ctx, sp: Screenplay):
    media = {}
    for s in sp.shots:
        a = ctx.asset(f"clip:{s.id}")
        media[s.id] = ShotMedia(s.id, a["path"], stream_info(Path(a["path"]))["duration"])
    speech = []
    for a in ctx.db.assets(ctx.job_id, "speech"):
        if a["status"] != AssetState.PASSED:
            continue
        m = a["meta"]
        speech.append(SpeechMedia(a["asset_id"].split(":", 1)[1], m["shot_id"], m["kind"], a["path"],
                                  m["duration"], m.get("words", []), m["text"]))
    order = {s.id: i for i, s in enumerate(sp.shots)}
    speech.sort(key=lambda x: (order[x.shot_id], x.kind != "narration", x.asset_id))
    sounds = []
    for kind in ("sfx", "ambience", "music"):
        for a in ctx.db.assets(ctx.job_id, kind):
            if a["status"] == AssetState.PASSED:
                m = a["meta"]
                sounds.append(SoundMedia(a["asset_id"].split(":", 1)[1], kind, a["path"], m["duration"],
                                         m.get("shot_id"), m.get("location_id")))
    return media, speech, sounds


def run(ctx: Ctx, sp: Screenplay, feedback: str = "") -> Path:
    cfg = ctx.cfg
    deps = [f"clip:{s.id}" for s in sp.shots] + [a["asset_id"] for a in ctx.db.assets(ctx.job_id, "speech")]
    ctx.ensure("edl", "edl", deps)
    ctx.ensure("final", "final", ["edl"])
    final = ctx.dirs.edit / "final.mp4"
    if ctx.passed("final"):
        return final
    ctx.state(JobState.EDITING)
    media, speech, sounds = gather(ctx, sp)
    lo, hi = duration_bounds(sp.target_seconds, bool(ctx.job['params'].get('test')))
    edl_path = ctx.dirs.edit / "edit_decision_list.json"
    if ctx.passed("edl") and edl_path.exists():
        edl = EDL.model_validate(read_json(edl_path))
    else:
        ctx.to("edl", AssetState.GENERATING, attempts_inc=True)
        with ctx.rm.ollama_session("writer", cfg.models.writer, ctx.job_id) as sess:
            edl, err = direct(cfg, ctx.job_id, sp, media, speech, sounds, sess, ctx.seed, lo, hi, note=feedback)
            if feedback and "too loud" in feedback:
                for a in edl.audio:  # QA found narration masked: deterministic -4 dB on beds
                    if a.kind in ("music", "ambience", "sfx"):
                        a.gain_db -= 4
            try:
                meta = chat_json(cfg, cfg.models.writer, "You write YouTube Shorts metadata for original fiction.",
                                 f"Story: {sp.story.model_dump_json()}\nWrite a gripping title (<70 chars, no "
                                 f"clickbait lies), a 2-3 sentence description, and 5-10 tags. Return JSON.",
                                 PublishMeta, session=sess, temperature=0.6)
            except LLMError:
                meta = PublishMeta(title=sp.story.title, description=sp.story.logline,
                                   tags=[sp.story.genre, "short film", "cinematic", "original story"])
        if err:
            ctx.warn(f"AI director plan invalid after retries; deterministic fallback EDL used: {err}")
        ctx.decide(Decision(asset_id="edl", stage="edit", criterion="EDL validation", evaluator="edl.resolve",
                            passed=True, evidence=f"source={edl.source}; duration={edl.duration}s; "
                            f"{len(edl.video)} shots; {len(edl.audio)} audio events; fallback_reason={err}"))
        atomic_json(edl_path, edl)
        atomic_json(ctx.dirs.edit / "publish_meta.json", meta)
        ctx.approve("edl", edl_path, edl_path, meta={"source": edl.source, "duration": edl.duration})
    ctx.to("final", AssetState.GENERATING, attempts_inc=True)
    write_srt(edl, ctx.dirs.edit / "subtitles.srt")
    tmp = ctx.dirs.edit / "final.render.mp4"
    render(edl, tmp, ctx.dirs.edit / "render_work", ctx.dirs.logs / "render.log", cfg.burn_captions,
           cfg.thresholds.target_lufs)
    tmp.replace(final)
    ctx.to("final", AssetState.VALIDATING, path=str(final))
    ctx.info(f"rendered {final} ({edl.duration}s, EDL source {edl.source})")
    return final
