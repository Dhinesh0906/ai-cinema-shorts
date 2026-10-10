"""MOVING VIDEO GENERATION (Wan2.2 TI2V-5B first-frame I2V) -> technical + temporal + semantic validation
-> regenerate only failed shots with adjusted seed/prompt/parameters. No static fallback exists."""
from __future__ import annotations

import json
import zlib
from pathlib import Path

from ..model_workers import batch_worker, wan_frames, wan_video
from ..quality import vlm
from ..resource_manager import WorkerFailed
from ..schemas import Decision, Screenplay
from ..state_machine import AssetState, JobState
from ..storage.files import read_json
from ..video.generator import motion_prompt, negative_prompt
from ..video.motion_analysis import analyze, extract_frames, sample_times
from . import best_of
from .context import Ctx, StageBlocked

N_FRAMES = 6
_CLOUD_DOWN: dict[str, bool] = {}


def _params(ctx: Ctx, attempt: int) -> dict:
    """Escalate generation effort on retries (documented, bounded)."""
    p = ctx.preset
    return {"steps": p.steps + (5 if attempt >= 3 else 0), "guidance": p.guidance + (1.0 if attempt >= 3 else 0.0)}


def run(ctx: Ctx, sp: Screenplay) -> None:
    cfg = ctx.cfg
    for s in sp.shots:
        ctx.ensure(f"clip:{s.id}", "clip", [f"keyframe:{s.id}"])
    pending = [s for s in sp.shots if not ctx.passed(f"clip:{s.id}")]
    while pending:
        ctx.state(JobState.GENERATING_VIDEO)
        made: dict[str, Path] = {}
        for s in pending:
            aid = f"clip:{s.id}"
            a = ctx.asset(aid)
            prev = a["meta"].get(f"attempt{a['attempts']}", {})
            if a["status"] == AssetState.VALIDATING and prev.get("output") and Path(prev["output"]).exists():
                made[s.id] = Path(prev["output"])  # rendered before a pause/crash: validate, don't re-render
                ctx.info(f"{s.id}: reusing already-rendered attempt {a['attempts']}")
                continue
            attempt = a["attempts"] + 1
            if attempt > cfg.retries.max_attempts:  # retries used up: ship the best validated attempt
                best = best_of.pick(ctx, best_of.clip_candidates(ctx, s.id), [f"keyframe:{s.id}"])
                if best is None:
                    ctx.to(aid, AssetState.BLOCKED, feedback=f"{best_of.NO_USABLE}: {a['feedback']}"[:1500])
                    raise StageBlocked(f"{aid} failed video validation {cfg.retries.max_attempts} times and no "
                                       f"attempt is usable: {a['feedback']}")
                note = best_of.summary(best, cfg.retries.max_attempts)
                ctx.approve(aid, best["path"], ctx.dirs.clips / f"{s.id}.mp4",
                            meta={"best_of": True, "semantic_score": best["score"], "source": best["report"]})
                ctx.decide(Decision(asset_id=aid, stage="video", criterion="asset approved", evaluator="controller",
                                    passed=True, score=best["score"], attempt=a["attempts"], evidence=note,
                                    action="best-of fallback"))
                ctx.warn(f"{s.id} {note}")
                continue
            ctx.to(aid, AssetState.GENERATING, attempts_inc=True)
            correction = (a["feedback"] or "")[:500] if attempt > 1 else ""
            prompt = motion_prompt(sp, s, ctx.style, _correction_text(correction))
            frames = wan_frames(s.duration, ctx.preset)
            seed = (ctx.seed + zlib.crc32(aid.encode()) % 100_000 + attempt * 7919) % 2**31
            prm = _params(ctx, attempt)
            out = ctx.dirs.work / f"{s.id}_a{attempt}.mp4"
            keyframe = Path(ctx.asset(f"keyframe:{s.id}")["path"])
            ctx.info(f"generating {s.id} attempt {attempt}: {frames} frames @ {ctx.preset.width}x{ctx.preset.height}")
            cloud_meta = None
            if cfg.video_backend == "openrouter" and not _CLOUD_DOWN.get(ctx.job_id):
                from ..openrouter_video import CloudVideoUnavailable, generate as cloud_video
                try:
                    cloud_meta = cloud_video(cfg, prompt, negative_prompt(), keyframe, out, s.duration, seed)
                    ctx.info(f"{s.id} attempt {attempt} rendered by {cfg.openrouter.model} in {cloud_meta['seconds']}s")
                except CloudVideoUnavailable as exc:
                    _CLOUD_DOWN[ctx.job_id] = True  # don't re-ask for every shot of this run
                    ctx.info(f"cloud video unavailable ({str(exc)[:120]}); rendering locally with Wan 2.2 (expected fallback)")
            try:
                if cloud_meta is None:
                    r = wan_video(ctx.rm, cfg, prompt, negative_prompt(), keyframe, out, ctx.preset, frames, seed,
                                  ctx.dirs.logs / "video.log", ctx.job_id, steps=prm["steps"], guidance=prm["guidance"])
                else:
                    from ..resource_manager import WorkerResult
                    r = WorkerResult("video", 0, cloud_meta["seconds"], 0.0, None, {}, {}, True, extra=cloud_meta)
            except WorkerFailed as exc:
                ctx.warn(f"{s.id} generation failed: {exc}")
                ctx.to(aid, AssetState.RETRY_PENDING, feedback=f"generation crashed: {exc}")
                continue
            md_path = out.with_suffix(".metadata.json")
            md = read_json(md_path) if md_path.exists() else {}
            ctx.db.set_asset(ctx.job_id, aid, meta={f"attempt{attempt}": {
                "model": (cloud_meta or {}).get("model", cfg.models.video), "revision": md.get("base_model"),
                "backend": "openrouter" if cloud_meta else "local", "prompt": prompt,
                "negative_prompt": negative_prompt(), "input_reference": str(keyframe), "seed": seed,
                "resolution": [ctx.preset.width, ctx.preset.height], "frames": frames, "fps": ctx.preset.fps,
                "steps": prm["steps"], "guidance": prm["guidance"], "flow_shift": ctx.preset.flow_shift,
                "seconds": r.seconds, "peak_footprint_gib": r.peak_footprint_gib, "peak_rss_gib": r.peak_rss_gib,
                "mlx_peak_gib": round((md.get("runtime_memory", {}).get("mlx_peak_memory_bytes") or 0) / 2**30, 2),
                "output": str(out)}})
            made[s.id] = out
            ctx.to(aid, AssetState.VALIDATING)

        # ---------------- technical + temporal measurements (no model)
        ctx.state(JobState.VALIDATING_VIDEO)
        tech_ok: dict[str, dict] = {}
        nxt = []
        for s in pending:
            aid = f"clip:{s.id}"
            if s.id not in made:
                nxt.append(s)
                continue
            attempt = ctx.asset(aid)["attempts"]
            keyframe = Path(ctx.asset(f"keyframe:{s.id}")["path"])
            m = analyze(made[s.id], expected_duration=min(s.duration, ctx.preset.max_frames / ctx.preset.fps) - 0.1,
                        reference=keyframe, thresholds=cfg.thresholds)
            ctx.save_report(f"{s.id}_a{attempt}_technical", m)
            ctx.decide(Decision(asset_id=aid, stage="video", criterion="technical checks", evaluator="ffmpeg+opencv",
                                passed=m["passed"], evidence=json.dumps({k: m.get(k) for k in (
                                    "mean_flow", "freeze_fraction", "black_fraction", "temporal_spike",
                                    "saturation_jump", "duration", "problems")}), attempt=attempt))
            if m["passed"]:
                tech_ok[s.id] = m
            else:
                fb = "technical: " + "; ".join(m["problems"])
                ctx.to(aid, AssetState.RETRY_PENDING, feedback=fb)
                ctx.warn(f"{s.id} failed technical checks: {fb}")
                nxt.append(s)

        # ---------------- semantic + continuity (VLM, one load for the batch)
        items, cl_by = [], {}
        for s in pending:
            if s.id not in tech_ok:
                continue
            frames = extract_frames(made[s.id], sample_times(tech_ok[s.id]["duration"], N_FRAMES),
                                    ctx.dirs.work / f"frames_{made[s.id].stem}")
            refs, roles = [], []
            for cid in s.character_ids:
                refs.append(ctx.asset(f"char_ref:{cid}")["path"])
                roles.append(f"REFERENCE portrait of {sp.character(cid).name}")
            refs.append(ctx.asset(f"keyframe:{s.id}")["path"])
            roles.append("REFERENCE approved first frame of this shot")
            times = sample_times(tech_ok[s.id]["duration"], N_FRAMES)
            roles += [f"VIDEO FRAME {i+1} of {N_FRAMES} at t={t:.2f}s (chronological)" for i, t in enumerate(times)]
            cl = vlm.video_checklist(sp, s)
            cl_by[s.id] = cl
            ctxt = (f"Shot {s.id} of '{sp.story.title}'. Script action: {s.action}. Camera: {s.camera_motion}. "
                    f"Judge temporal events by comparing EARLY frames with LATE frames; one frame alone cannot "
                    f"show an action.")
            items.append({"id": s.id, "images": refs + [str(f) for f in frames],
                          "prompt": vlm.build_prompt(cl, ctxt, roles), "max_tokens": 1800, "max_side": 672})
        results = batch_worker(ctx.rm, cfg, "vlm", "vlm", cfg.models.vlm, {"model": cfg.models.vlm, "max_side": 672},
                               items, ctx.dirs.work / "vlm_video", ctx.job_id) if items else {}
        for s in pending:
            if s.id not in tech_ok:
                continue
            aid = f"clip:{s.id}"
            attempt = ctx.asset(aid)["attempts"]
            res = results.get(s.id, {"error": "no result"})
            cl = cl_by[s.id]
            if "error" in res:
                ctx.to(aid, AssetState.RETRY_PENDING, feedback=f"validator error: {res['error']}")
                nxt.append(s)
                continue
            try:
                verdict = vlm.parse_verdict(res["text"], cl)
            except ValueError as exc:
                ctx.to(aid, AssetState.RETRY_PENDING, feedback=f"unparseable validator output: {exc}")
                nxt.append(s)
                continue
            sc = vlm.score(verdict, cl, cfg.thresholds.video_optional_ratio)
            ctx.save_report(f"{s.id}_a{attempt}_semantic", {
                "verdict": verdict.model_dump(), "passed": sc.passed, "score": sc.score,
                "mandatory_failed": sc.mandatory_failed, "optional_ratio": sc.optional_ratio,
                "validator": cfg.models.vlm, "technical": tech_ok[s.id], "raw": res["text"][:4000],
                "limitations": "Semantic judgement from 6 ordered frames; fast or subtle motion between "
                               "samples is not observed. Identity judged visually, not proven."})
            for chk, v in zip(cl.checks, verdict.checks):
                ctx.decide(Decision(asset_id=aid, stage="video", criterion=chk.text, evaluator=cfg.models.vlm,
                                    passed=v.verdict == "yes", mandatory=chk.mandatory, evidence=v.evidence,
                                    attempt=attempt))
            if sc.passed:
                dest = ctx.dirs.clips / f"{s.id}.mp4"
                ctx.approve(aid, made[s.id], dest, meta={"approved_attempt": attempt, "semantic_score": sc.score,
                                                         "technical": {k: tech_ok[s.id].get(k) for k in (
                                                             "mean_flow", "duration", "width", "height", "fps")}})
                ctx.info(f"{s.id} approved (score {sc.score})")
            else:
                fb = "; ".join(sc.mandatory_failed)[:900] + (f" | fix: {verdict.correction}" if verdict.correction else "")
                ctx.to(aid, AssetState.RETRY_PENDING, feedback=fb)
                ctx.warn(f"{s.id} rejected (attempt {attempt}): {fb[:200]}")
                nxt.append(s)
        pending = [s for s in nxt if not ctx.passed(f"clip:{s.id}")]  # best-of approvals are done


def _correction_text(feedback: str) -> str:
    if not feedback:
        return ""
    if "|" in feedback and "fix:" in feedback:
        return feedback.split("fix:", 1)[1].strip()[:300]
    if feedback.startswith("technical"):
        return "smooth stable motion, no flicker, no abrupt changes, keep the scene coherent"
    return feedback[:300]
