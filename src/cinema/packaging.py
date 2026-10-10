"""outputs/JOB_ID deliverable package and honest failure reports."""
from __future__ import annotations

import json
import shutil
from dataclasses import asdict
from pathlib import Path

from .editing.ffmpeg_renderer import thumbnail
from .model_registry import refresh
from .pipeline.context import Ctx
from .schemas import EDL, Screenplay
from .state_machine import JobState
from .storage.files import atomic_json, read_json, sha256
from .video.motion_analysis import stream_info

DISCLOSURE = ("This is an original fictional short film. Visuals, voices, sound and music were generated with AI "
              "models running locally; any resemblance to real persons is coincidental. "
              "Altered or synthetic content: yes (YouTube 'altered or synthetic content' disclosure should be "
              "enabled on upload).")


def _common(ctx: Ctx, out: Path, sp: Screenplay) -> None:
    atomic_json(out / "story.json", sp.story)
    atomic_json(out / "screenplay.json", sp)
    atomic_json(out / "characters.json", [
        {**c.model_dump(), "reference_image": _path(ctx, f"char_ref:{c.id}")} for c in sp.story.characters])
    atomic_json(out / "locations.json", [
        {**loc.model_dump(), "reference_image": _path(ctx, f"loc_ref:{loc.id}")} for loc in sp.story.locations])
    atomic_json(out / "shot_plan.json", [
        {**s.model_dump(), "keyframe": _path(ctx, f"keyframe:{s.id}"), "clip": _path(ctx, f"clip:{s.id}"),
         "start": sp.starts()[s.id]} for s in sp.shots])
    entries = refresh(ctx.cfg)
    used_roles = {"writer", "critic", "vlm", "image", "video", "whisper", "tts"}
    if ctx.db.assets(ctx.job_id, "music") or ctx.db.assets(ctx.job_id, "sfx") or ctx.db.assets(ctx.job_id, "ambience"):
        used_roles |= {"sfx", "clap"}
    used = [asdict(e) for e in entries if e.role in used_roles]
    atomic_json(out / "model_manifest.json", used)
    atomic_json(out / "license_manifest.json", {
        "models": [{k: e[k] for k in ("role", "repo", "license", "commercial_use", "source")} for e in used],
        "software": [{"name": "mlx-gen", "license": "MIT"}, {"name": "FFmpeg", "license": "LGPL/GPL (Homebrew build)"},
                     {"name": "Ollama", "license": "MIT"}, {"name": "kokoro-onnx", "license": "MIT"},
                     {"name": "mlx-vlm", "license": "MIT"}, {"name": "mlx-whisper", "license": "MIT"},
                     {"name": "Apple Vision framework", "license": "macOS system framework"}],
        "fonts": [{"name": "Avenir Next (macOS system font)", "use": "burned-in captions",
                   "note": "Apple system font; verify font licensing terms for commercial distribution"}],
        "third_party_audio": "none downloaded; all sound generated locally by Stable Audio 3 when present",
        "experimental_models_used": False})
    logs = out / "logs"
    logs.mkdir(exist_ok=True)
    for f in ctx.dirs.logs.glob("*.log"):
        shutil.copy2(f, logs / f.name)
    vr = out / "validation_reports"
    if vr.exists():
        shutil.rmtree(vr)
    shutil.copytree(ctx.dirs.reports, vr)
    atomic_json(out / "decisions.json", ctx.db.decisions(ctx.job_id))
    atomic_json(out / "worker_runs.json", ctx.db.worker_runs(ctx.job_id))


def _path(ctx: Ctx, aid: str) -> str | None:
    try:
        return ctx.asset(aid)["path"]
    except KeyError:
        return None


def package(ctx: Ctx, sp: Screenplay, report: dict, failed: bool = False) -> Path:
    out = ctx.cfg.root / "outputs" / ctx.job_id
    out.mkdir(parents=True, exist_ok=True)
    _common(ctx, out, sp)
    edl = EDL.model_validate(read_json(ctx.dirs.edit / "edit_decision_list.json"))
    atomic_json(out / "edit_decision_list.json", edl)
    atomic_json(out / "quality_report.json", report)
    final_src = ctx.dirs.edit / "final.mp4"
    name = "REJECTED_draft.mp4" if failed else "final.mp4"
    shutil.copy2(final_src, out / name)
    srt = ctx.dirs.edit / "subtitles.srt"
    if srt.exists():
        shutil.copy2(srt, out / "subtitles.srt")
    if failed:
        return out
    from .story.history import record
    record(ctx.cfg, ctx.job_id, sp)
    hook = next((v for v in edl.video if v["shot_id"] == sp.shots[0].id), edl.video[0])
    thumbnail(out / "final.mp4", min(edl.duration * 0.15, (hook["out"] - hook["in"]) * 0.6), out / "thumbnail.png")
    meta_p = ctx.dirs.edit / "publish_meta.json"
    meta = read_json(meta_p) if meta_p.exists() else {"title": sp.story.title, "description": sp.story.logline,
                                                       "tags": [sp.story.genre]}
    (out / "youtube_title.txt").write_text(meta["title"].strip() + "\n")
    (out / "youtube_description.txt").write_text(meta["description"].strip() + "\n\n" + DISCLOSURE + "\n\n#Shorts\n")
    atomic_json(out / "youtube_tags.json", meta["tags"])
    clips = [ctx.asset(f"clip:{s.id}") for s in sp.shots]
    info = stream_info(out / "final.mp4")
    atomic_json(out / "render_metadata.json", {
        "native_generation": {"model": ctx.cfg.models.video, "width": ctx.preset.width, "height": ctx.preset.height,
                              "fps": ctx.preset.fps, "steps": ctx.preset.steps, "preset": ctx.job["preset"]},
        "export": {"width": info["width"], "height": info["height"], "fps": info["fps"], "codec": info["codec"],
                   "audio_codec": info["audio_codec"], "duration": info["duration"],
                   "upscaled": True, "upscale_method": "lanczos (no detail added)"},
        "clips": [{"id": c["asset_id"], **{k: v for k, v in c["meta"].items() if k.startswith("attempt")
                                           or k in ("approved_attempt", "semantic_score")}} for c in clips],
        "sha256": sha256(out / "final.mp4")})
    runs = ctx.db.worker_runs(ctx.job_id)
    atomic_json(out / "production_summary.json", {
        "job_id": ctx.job_id, "status": "COMPLETE", "title": sp.story.title, "genre": sp.story.genre,
        "duration_requested": ctx.target, "duration_final": info["duration"],
        "project_dir": str(ctx.dirs.root), "final_video": str(out / "final.mp4"),
        "shots": len(sp.shots), "attempts": {a["asset_id"]: a["attempts"] for a in ctx.db.assets(ctx.job_id)},
        "worker_runs": len(runs), "total_worker_seconds": round(sum(r["seconds"] or 0 for r in runs), 1),
        "peak_footprint_gib_by_role": _peaks(runs),
        "edl_source": edl.source, "limitations": report.get("limitations", []),
        "subjective_quality": "not proven by automated checks; human review recommended before publishing",
        "published": False})
    return out


def _peaks(runs: list[dict]) -> dict:
    out: dict[str, float] = {}
    for r in runs:
        v = r["peak_footprint_gib"] or r["peak_rss_gib"] or 0
        out[r["worker"]] = max(out.get(r["worker"], 0), v)
    return out


def failure_report(ctx: Ctx, status: JobState, reason: str) -> Path:
    out = ctx.cfg.root / "outputs" / ctx.job_id
    out.mkdir(parents=True, exist_ok=True)
    assets = ctx.db.assets(ctx.job_id)
    bad = [a for a in assets if a["status"] in ("BLOCKED", "FAILED", "RETRY_PENDING")]
    rep = {"job_id": ctx.job_id, "status": str(status), "reason": reason,
           "resumable": status == JobState.BLOCKED,
           "resume_command": f"uv run cinema resume {ctx.job_id}" if status == JobState.BLOCKED else None,
           "problem_assets": [{k: a[k] for k in ("asset_id", "status", "attempts", "feedback")} for a in bad],
           "asset_summary": {s: sum(a["status"] == s for a in assets) for s in {a["status"] for a in assets}},
           "recent_events": ctx.db.events(ctx.job_id, 40),
           "note": "No final film is claimed. Approved intermediate assets are kept for resume."}
    atomic_json(out / "failure_report.json", rep)
    try:
        _common(ctx, out, ctx.screenplay())
    except Exception:
        pass
    return out
