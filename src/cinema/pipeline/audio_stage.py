"""KOKORO SPEECH -> WHISPER + loudness validation -> regenerate only failed lines.
STABLE AUDIO 3 ambience/sfx/score -> technical + CLAP relevance (sound design is optional: assets that
cannot pass are left out of the mix and reported, never faked)."""
from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel

from ..audio import analysis
from ..audio.effects import distractors_for, plan_sounds
from ..audio.speech_validator import compare
from ..audio.tts import Line, plan_lines, speakable, speed_for
from ..llm import chat_json
from ..model_workers import batch_worker
from ..schemas import Decision, Screenplay
from ..state_machine import AssetState, JobState
from ..storage.files import atomic_json, sha256
from .context import Ctx, StageBlocked


class Shortened(BaseModel):
    text: str


def _tts(ctx: Ctx, jobs: list[tuple[Line, float, Path]]) -> dict:
    cfg = ctx.cfg
    items = [{"id": ln.asset_id, "text": ln.text, "voice": ln.voice, "speed": spd, "out": str(out)}
             for ln, spd, out in jobs]
    return batch_worker(ctx.rm, cfg, "tts", "tts", "kokoro-82m-v1.0-onnx",
                        {"model": str(cfg.resolve(cfg.models.kokoro_model)),
                         "voices": str(cfg.resolve(cfg.models.kokoro_voices))},
                        items, ctx.dirs.work / "tts", ctx.job_id)


def _synth_fit(ctx: Ctx, jobs: list[tuple[Line, float, Path]]):
    """TTS a batch; lines slightly over their window are re-synthesized faster (<= 1.15x).
    Returns (results, jobs with final speeds, lines that cannot fit at any allowed speed)."""
    res = _tts(ctx, jobs)
    refit, too_long = [], []
    for ln, spd, out in jobs:
        r = res.get(ln.asset_id, {})
        if "error" in r or not out.exists() or r["seconds"] <= ln.window:
            continue
        need = speed_for(r["seconds"] / spd, ln.window)
        (too_long.append(ln) if need is None else refit.append((ln, need, out)))
    if refit:
        res |= _tts(ctx, refit)
        for ln, spd, _ in refit:
            ctx.db.set_asset(ctx.job_id, f"speech:{ln.asset_id}", meta={"speed": spd})
        speeds = {ln.asset_id: spd for ln, spd, _ in refit}
        jobs = [(ln, speeds.get(ln.asset_id, spd), out) for ln, spd, out in jobs]
    return res, jobs, too_long


def _shorten(ctx: Ctx, sp: Screenplay, lines: list[Line], seconds: dict[str, float]) -> Screenplay:
    """Writer adapts lines that cannot fit their shot even at max speed (meaning kept; logged)."""
    cfg = ctx.cfg
    with ctx.rm.ollama_session("writer", cfg.models.writer, ctx.job_id) as sess:
        for ln in lines:
            max_words = max(3, int(len(ln.text.split()) * ln.window / max(seconds[ln.asset_id], 0.1) * 0.95))
            r = chat_json(cfg, cfg.models.writer, "You tighten film narration without losing meaning.",
                          f"Rewrite this line in at most {max_words} words, same meaning and tone, no new facts:\n"
                          f"{ln.text}\nReturn JSON {{\"text\": ...}}", Shortened, session=sess, temperature=0.3)
            new = speakable(r.text)
            ctx.decide(Decision(asset_id=ln.asset_id, stage="audio", criterion="speech fits shot timing",
                                evaluator=cfg.models.writer, passed=False, evidence=f"{seconds[ln.asset_id]:.2f}s > "
                                f"{ln.window:.2f}s window", action=f"line adapted: {ln.text!r} -> {new!r}"))
            shot = sp.shot(ln.shot_id)
            if ln.kind == "narration":
                shot.narration = new
            else:
                idx = int(ln.asset_id.rsplit("_", 1)[1]) - 1
                shot.dialogue[idx].text = new
    path = ctx.dirs.script / "screenplay.json"
    atomic_json(path, sp)
    ctx.db.set_asset(ctx.job_id, "screenplay", sha256=sha256(path), meta={"narration_adapted": True})
    return sp


def speech(ctx: Ctx, sp: Screenplay) -> Screenplay:
    cfg, t = ctx.cfg, ctx.cfg.thresholds
    for ln in plan_lines(sp):
        ctx.ensure(f"speech:{ln.asset_id}", "speech", ["screenplay"])
    shortened_once: set[str] = set()
    while True:
        lines = {ln.asset_id: ln for ln in plan_lines(sp)}
        pending = [ln for ln in lines.values() if not ctx.passed(f"speech:{ln.asset_id}")]
        if not pending:
            return sp
        ctx.state(JobState.GENERATING_AUDIO)
        jobs = []
        for ln in pending:
            aid = f"speech:{ln.asset_id}"
            a = ctx.asset(aid)
            attempt = a["attempts"] + 1
            if attempt > cfg.retries.max_attempts:
                ctx.to(aid, AssetState.BLOCKED)
                raise StageBlocked(f"{aid} failed speech validation {cfg.retries.max_attempts} times: {a['feedback']}")
            ctx.to(aid, AssetState.GENERATING, attempts_inc=True)
            spd = a["meta"].get("speed", 1.0) * (0.94 if attempt == 3 else 1.0)  # slower, clearer on last try
            jobs.append((ln, round(spd, 3), ctx.dirs.work / f"{ln.asset_id}_a{attempt}.wav"))
        res, jobs, too_long = _synth_fit(ctx, jobs)
        if too_long:  # cannot fit even at max speed: writer adapts those lines once, then re-synthesize
            if {ln.asset_id for ln in too_long} & shortened_once:
                raise StageBlocked("speech lines still do not fit their shots after adaptation: "
                                   + ", ".join(sorted(ln.asset_id for ln in too_long)))
            shortened_once |= {ln.asset_id for ln in too_long}
            sp = _shorten(ctx, sp, too_long, {ln.asset_id: res[ln.asset_id]["seconds"] for ln in too_long})
            fresh = {ln.asset_id: ln for ln in plan_lines(sp)}
            ids = {ln.asset_id for ln in too_long}
            jobs = [(fresh[ln.asset_id], spd, out) for ln, spd, out in jobs]
            res2, redone, still = _synth_fit(ctx, [j for j in jobs if j[0].asset_id in ids])
            if still:
                raise StageBlocked("adapted speech lines still exceed their shot windows: "
                                   + ", ".join(ln.asset_id for ln in still))
            res |= res2
            redone_by = {j[0].asset_id: j for j in redone}
            jobs = [redone_by.get(j[0].asset_id, j) for j in jobs]

        # ---------------- independent Whisper transcription
        ctx.state(JobState.VALIDATING_AUDIO)
        names = [c.name for c in sp.story.characters] + [loc.name for loc in sp.story.locations]
        for ln, _, _ in jobs:
            ctx.to(f"speech:{ln.asset_id}", AssetState.VALIDATING)
        wres = batch_worker(ctx.rm, cfg, "whisper", "whisper", cfg.models.whisper, {"model": cfg.models.whisper},
                            [{"id": ln.asset_id, "audio": str(out), "prompt": ", ".join(names) + "."}
                             for ln, _, out in jobs if out.exists()],
                            ctx.dirs.work / "whisper", ctx.job_id)
        for ln, spd, out in jobs:
            aid = f"speech:{ln.asset_id}"
            attempt = ctx.asset(aid)["attempts"]
            problems = []
            if not out.exists() or "error" in res.get(ln.asset_id, {}):
                problems.append(f"TTS failed: {res.get(ln.asset_id, {}).get('error')}")
                w, cmp, m = {}, {}, {}
            else:
                w = wres.get(ln.asset_id, {})
                if "error" in w or not w:
                    problems.append(f"whisper failed: {w.get('error')}")
                    cmp = {}
                else:
                    cmp = compare(ln.text, w["text"], names)
                    if cmp["wer"] > t.max_wer:
                        problems.append(f"WER {cmp['wer']:.2f} > {t.max_wer}")
                    if cmp["missing_content_words"]:
                        problems.append(f"missing words {cmp['missing_content_words']}")
                m = analysis.measure(out)
                lead, trail = analysis.leading_trailing_silence(out)
                m |= {"leading_silence": lead, "trailing_silence": trail}
                if m["peak_db"] is not None and m["peak_db"] > t.max_peak_dbfs:
                    problems.append(f"peak {m['peak_db']} dBFS (clipping risk)")
                if m["rms_db"] is not None and m["rms_db"] < t.min_mean_db:
                    problems.append(f"too quiet ({m['rms_db']} dB RMS)")
                inner = [s for s in m["silences"] if s["start"] > 0.05 and s["end"] < m["duration"] - 0.05
                         and s["duration"] > 0.9]
                if inner:
                    problems.append(f"unexpected gap(s) {inner}")
                if m["duration"] > ln.window + 0.05:
                    problems.append(f"duration {m['duration']:.2f}s exceeds shot window {ln.window:.2f}s")
            ctx.save_report(f"{ln.asset_id}_a{attempt}", {"expected": ln.text, "transcript": w.get("text"),
                                                         "comparison": cmp, "measurements": m, "speed": spd,
                                                         "voice": ln.voice, "problems": problems,
                                                         "validator": cfg.models.whisper})
            ctx.decide(Decision(asset_id=aid, stage="audio", criterion="speech matches script & technical checks",
                                evaluator=cfg.models.whisper, passed=not problems, score=cmp.get("wer"),
                                evidence="; ".join(problems) or f"transcript: {w.get('text', '')[:300]}",
                                attempt=attempt, action="" if not problems else "regenerate line"))
            if problems:
                ctx.to(aid, AssetState.RETRY_PENDING, feedback="; ".join(problems)[:1000])
                ctx.warn(f"{aid} rejected: {problems}")
            else:
                dest = ctx.dirs.voice / f"{ln.asset_id}.wav"
                ctx.approve(aid, out, dest, meta={"text": ln.text, "voice": ln.voice, "speed": spd,
                                                  "duration": m["duration"], "shot_id": ln.shot_id, "kind": ln.kind,
                                                  "words": w.get("words", []), "wer": cmp["wer"]})


def sounds(ctx: Ctx, sp: Screenplay) -> None:
    cfg = ctx.cfg
    tasks = plan_sounds(sp, ctx.seed, cfg.generate_music, cfg.generate_sfx)
    if not tasks:
        return
    sa3 = Path(cfg.models.sa3_dir).expanduser()
    if not (sa3 / ".venv" / "bin" / "python").exists():
        ctx.warn(f"Stable Audio 3 not installed at {sa3}; film will have narration only")
        return
    for tk in tasks:
        ctx.ensure(f"sound:{tk.asset_id}", tk.kind, ["screenplay"])
    for _ in range(cfg.retries.max_attempts):
        pending = []
        for tk in tasks:
            aid = f"sound:{tk.asset_id}"
            a = ctx.asset(aid)
            if ctx.passed(aid) or a["status"] in (AssetState.FAILED, AssetState.BLOCKED):
                continue
            if a["attempts"] >= cfg.retries.max_attempts:
                ctx.to(aid, AssetState.FAILED, feedback=a["feedback"])
                continue
            pending.append(tk)
        if not pending:
            break
        ctx.state(JobState.GENERATING_AUDIO)
        items = []
        for tk in pending:
            aid = f"sound:{tk.asset_id}"
            att = ctx.asset(aid)["attempts"] + 1
            ctx.to(aid, AssetState.GENERATING, attempts_inc=True)
            items.append({"id": tk.asset_id, "prompt": tk.prompt, "seconds": tk.seconds, "kind": tk.kind,
                          "seed": tk.seed + att * 131, "negative": "speech, vocals, talking, distortion, clipping",
                          "out": str(ctx.dirs.work / f"{tk.asset_id}_a{att}.wav")})
        gen = batch_worker(ctx.rm, cfg, "sfx", "sfx", "stable-audio-3-small (sm-sfx/sm-music)",
                           {"sa3_dir": cfg.models.sa3_dir}, items, ctx.dirs.work / "sfx", ctx.job_id)
        ctx.state(JobState.VALIDATING_AUDIO)
        ok_items, tech = [], {}
        for it, tk in zip(items, pending):
            aid = f"sound:{tk.asset_id}"
            ctx.to(aid, AssetState.VALIDATING)
            r = gen.get(tk.asset_id, {})
            problems = []
            if "error" in r or not Path(it["out"]).exists():
                problems.append(f"generation failed: {r.get('error')}")
            else:
                m = analysis.measure(Path(it["out"]), silence_db=-60, min_silence=1.0)
                tech[tk.asset_id] = m
                if m["rms_db"] is None or m["rms_db"] < -55:
                    problems.append("silent output")
                if m["peak_db"] is not None and m["peak_db"] > -0.1:
                    problems.append("clipping")
                if m["duration"] < tk.seconds * 0.8:
                    problems.append(f"too short ({m['duration']}s)")
            if problems:
                ctx.to(aid, AssetState.RETRY_PENDING, feedback="; ".join(problems))
            else:
                ok_items.append((it, tk))
        clap = batch_worker(ctx.rm, cfg, "clap", "clap", cfg.models.clap, {"model": cfg.models.clap},
                            [{"id": tk.asset_id, "audio": it["out"], "texts": [tk.prompt.split(",")[0], *distractors_for(tk)]}
                             for it, tk in ok_items], ctx.dirs.work / "clap", ctx.job_id) if ok_items else {}
        for it, tk in ok_items:
            aid = f"sound:{tk.asset_id}"
            c = clap.get(tk.asset_id, {})
            scores = c.get("scores")
            relevant = bool(scores) and scores[0] == max(scores)
            ctx.decide(Decision(asset_id=aid, stage="audio", criterion="CLAP relevance (target ranks first)",
                                evaluator=cfg.models.clap, passed=relevant, mandatory=False,
                                score=scores[0] if scores else None, evidence=str(c)[:500],
                                attempt=ctx.asset(aid)["attempts"]))
            ctx.save_report(f"{tk.asset_id}_a{ctx.asset(aid)['attempts']}", {"prompt": tk.prompt, "clap": c,
                            "technical": tech.get(tk.asset_id), "relevant": relevant,
                            "note": "CLAP is supporting evidence, not a judge of quality"})
            if relevant or "error" in c:  # CLAP unavailable -> technical pass only, recorded as such
                dest = (ctx.dirs.music if tk.kind == "music" else ctx.dirs.sfx) / f"{tk.asset_id}.wav"
                ctx.approve(aid, Path(it["out"]), dest, meta={"kind": tk.kind, "prompt": tk.prompt,
                            "duration": tech[tk.asset_id]["duration"], "shot_id": tk.shot_id,
                            "location_id": tk.location_id, "clap_checked": "error" not in c,
                            "license": "Stability AI Community License (generated output)"})
            else:
                ctx.to(aid, AssetState.RETRY_PENDING, feedback=f"CLAP ranked a distractor above the prompt: {scores}")


def _finish_sounds(ctx: Ctx) -> None:
    for kind in ("sfx", "ambience", "music"):
        for a in ctx.db.assets(ctx.job_id, kind):
            if a["status"] == AssetState.RETRY_PENDING:
                ctx.to(a["asset_id"], AssetState.FAILED, feedback=a["feedback"])
                ctx.warn(f"{a['asset_id']} left out of the mix after {a['attempts']} attempts: {a['feedback']}")


def run(ctx: Ctx, sp: Screenplay) -> Screenplay:
    sp = speech(ctx, sp)
    sounds(ctx, sp)
    _finish_sounds(ctx)
    return sp
