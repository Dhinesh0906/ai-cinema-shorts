"""COMPLETE FILM VALIDATION. Independent of per-asset checks. Each failure is attributed to its origin
(EDITING / AUDIO / VIDEO / IMAGE / SCRIPT) so the controller rebuilds only the affected artifacts."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from ..audio import analysis
from ..audio.speech_validator import compare, norm_words
from ..image.identity import feature_distance
from ..model_workers import batch_worker
from ..quality import vlm
from ..schemas import EDL, Decision, Screenplay
from ..state_machine import JobState
from ..storage.files import read_json
from ..story.validator import mandatory_failures
from ..video.motion_analysis import analyze, extract_frames, stream_info
from .context import Ctx
from .edit_stage import duration_bounds


@dataclass
class Repair:
    origin: str          # EDITING | AUDIO | VIDEO | IMAGE | SCRIPT
    asset_id: str
    reason: str


def _check(ctx: Ctx, results: list, name: str, passed: bool, evidence, mandatory: bool = True,
           repair: Repair | None = None, evaluator: str = "deterministic") -> None:
    results.append({"check": name, "passed": passed, "mandatory": mandatory, "evidence": evidence,
                    "repair": repair.__dict__ if (repair and not passed) else None, "evaluator": evaluator})
    ctx.decide(Decision(asset_id="final", stage="final", criterion=name, evaluator=evaluator, passed=passed,
                        mandatory=mandatory, evidence=json.dumps(evidence, default=str)[:3000],
                        action=f"{repair.origin} -> rebuild {repair.asset_id}" if (repair and not passed) else ""))


def run(ctx: Ctx, sp: Screenplay, final: Path) -> dict:
    cfg = ctx.cfg
    ctx.state(JobState.FINAL_VALIDATION)
    edl = EDL.model_validate(read_json(ctx.dirs.edit / "edit_decision_list.json"))
    R: list = []
    edit = Repair("EDITING", "edl", "")

    # 1. technical video/audio container checks
    info = stream_info(final)
    lo, hi = duration_bounds(sp.target_seconds, bool(ctx.job['params'].get('test')))
    _check(ctx, R, "container: H.264 + AAC, 1080x1920", info["codec"] == "h264" and info["audio_codec"] == "aac"
           and (info["width"], info["height"]) == (1080, 1920), info, repair=Repair("EDITING", "edl", "bad container"))
    _check(ctx, R, f"duration within {lo}-{hi}s", lo - 0.1 <= info["duration"] <= hi + 0.1,
           {"duration": info["duration"]}, repair=Repair("EDITING", "edl", "duration out of bounds"))
    _check(ctx, R, "audio/video stream durations match", info["audio_duration"] is not None
           and abs(info["audio_duration"] - info["duration"]) < 0.25,
           {"video": info["duration"], "audio": info["audio_duration"]}, repair=Repair("EDITING", "edl", "a/v length"))
    tech = analyze(final, expected_duration=lo, thresholds=cfg.thresholds)
    _check(ctx, R, "decodes cleanly, not black/frozen, real motion", not [p for p in tech["problems"]
           if "temporal spike" not in p], {k: tech.get(k) for k in ("problems", "mean_flow", "freeze_fraction",
           "black_fraction")}, repair=Repair("EDITING", "edl", "; ".join(tech["problems"])))
    # per-segment freeze/black attribution -> the shot itself
    for v in edl.video:
        seg = analyze(Path(v["path"]), thresholds=cfg.thresholds)
        bad = [p for p in seg["problems"] if "frozen" in p or "black" in p or "static" in p]
        _check(ctx, R, f"segment {v['shot_id']} has real motion", not bad, bad,
               repair=Repair("VIDEO", f"clip:{v['shot_id']}", "; ".join(bad)))

    # 2. audio technical
    m = analysis.measure(final)
    _check(ctx, R, "loudness near target", m["lufs"] is not None and abs(m["lufs"] - cfg.thresholds.target_lufs) <= 2.0,
           {"lufs": m["lufs"]}, repair=Repair("EDITING", "edl", "loudness"))
    _check(ctx, R, "no clipping (true peak <= -1 dBFS)", m["true_peak_dbfs"] is None or m["true_peak_dbfs"] <= -0.9,
           {"true_peak": m["true_peak_dbfs"]}, repair=Repair("EDITING", "edl", "clipping"))
    long_sil = [s for s in m["silences"] if s["duration"] > 2.5]
    _check(ctx, R, "no unexpected long silences", not long_sil, long_sil, mandatory=False,
           repair=Repair("EDITING", "edl", "silence"))

    # 3. shot sequence: final frame at each segment middle best matches its own approved clip
    work = ctx.dirs.work / "final_check"
    seq_ok, seq_ev = True, []
    mids = []
    for v in edl.video:
        seg_len = v["out"] - v["in"]
        t_final = v["timeline_start"] + seg_len / 2
        mids.append(t_final)
    final_frames = extract_frames(final, mids, work, "final_")
    clip_frames = [extract_frames(Path(v["path"]), [v["in"] + (v["out"] - v["in"]) / 2], work, f"clip{i:02d}_")[0]
                   for i, v in enumerate(edl.video)]
    for i, ff in enumerate(final_frames):
        d = [feature_distance(ff, cf) for cf in clip_frames]
        best = min(range(len(d)), key=d.__getitem__)
        seq_ev.append({"segment": edl.video[i]["shot_id"], "best_match": edl.video[best]["shot_id"], "dist": d[i]})
        seq_ok &= best == i
    _check(ctx, R, "all shots present in screenplay order", seq_ok, seq_ev, repair=Repair("EDITING", "edl", "order"),
           evaluator="apple-vision featureprint")

    # 4. independent transcription of the final mix
    wres = batch_worker(ctx.rm, cfg, "whisper", "whisper", cfg.models.whisper, {"model": cfg.models.whisper},
                        [{"id": "final", "audio": str(final)}], ctx.dirs.work / "whisper_final", ctx.job_id)
    w = wres.get("final", {})
    expected = " ".join(a_text(ctx, e.asset_id) for e in edl.audio if e.kind in ("narration", "dialogue"))
    if "error" in w or not w:
        _check(ctx, R, "final narration transcribes", False, w, repair=edit)
    else:
        cmp = compare(expected, w["text"])
        masked = cmp["wer"] > cfg.thresholds.final_max_wer
        _check(ctx, R, "narration in final mix matches approved script", not masked,
               {"wer": cmp["wer"], "missing": cmp["missing_content_words"][:30]},
               repair=Repair("EDITING", "edl", "narration masked by music/ambience: too loud"), evaluator=cfg.models.whisper)
        # 5. sync: each speech event's first words are heard within 0.75 s of its EDL start
        sync_ev, sync_ok = [], True
        words = w.get("words", [])
        for e in [x for x in edl.audio if x.kind in ("narration", "dialogue")]:
            first = norm_words(a_text(ctx, e.asset_id))[:2]
            hit = _find(words, first, e.start)
            ok = hit is not None and abs(hit - e.start) <= 0.75 + 0.6
            sync_ev.append({"line": e.asset_id, "edl_start": e.start, "heard_at": hit})
            sync_ok &= ok
        _check(ctx, R, "speech heard at planned times (A/V sync)", sync_ok, sync_ev, repair=edit,
               evaluator=cfg.models.whisper)
        # 6. captions cover the speech they display
        cap_bad = []
        for c in edl.captions:
            cw = norm_words(c.text)[:1]
            hit = _find(words, cw, c.start)
            if hit is None or not (c.start - 0.6 <= hit <= c.end + 0.3):
                cap_bad.append({"caption": c.text, "start": c.start, "heard_at": hit})
        _check(ctx, R, "captions aligned with speech", len(cap_bad) <= max(1, len(edl.captions) // 10), cap_bad,
               repair=edit)

    # 7. semantic film inspection: storyboard of the final film
    roles, imgs = [], []
    for c in sp.story.characters:
        imgs.append(ctx.asset(f"char_ref:{c.id}")["path"])
        roles.append(f"REFERENCE portrait of {c.name}")
    # 3 frames per shot (early / middle / late): one mid frame cannot show an action or a beat.
    for i, v in enumerate(edl.video):
        seg = v["out"] - v["in"]
        times = [v["timeline_start"] + seg * f for f in (0.15, 0.5, 0.85)]
        for j, f in enumerate(extract_frames(final, times, work, f"story_s{i+1:02d}_")):
            imgs.append(str(f))
            roles.append(f"SHOT {i+1} ({v['shot_id']}) frame {'abc'[j]} of 3 ({['early', 'middle', 'late'][j]})")
    cl = vlm.Checklist()
    for i, s in enumerate(sp.shots):
        cl.add(f"Shot {i+1} frames a-c (early to late) plausibly show: {s.action}", False, "beat")
    for c in sp.story.characters:
        shots = [i + 1 for i, s in enumerate(sp.shots) if c.id in s.character_ids]
        if shots:
            cl.add(f"{c.name} looks like the same person (face, hair, costume) in shots {shots}", True,
                   "continuity")
    cl.add("No frame shows severe corruption, melting faces or garbage imagery", True, "artifact")
    vres = batch_worker(ctx.rm, cfg, "vlm", "vlm", cfg.models.vlm, {"model": cfg.models.vlm, "max_side": 448},
                        [{"id": "film", "images": imgs, "prompt": vlm.build_prompt(
                            cl, f"Final cut of '{sp.story.title}': {sp.story.logline}", roles), "max_tokens": 2000}],
                        ctx.dirs.work / "vlm_final", ctx.job_id)
    vr = vres.get("film", {})
    if "error" in vr or not vr:
        _check(ctx, R, "semantic film inspection ran", False, vr, mandatory=False, evaluator=cfg.models.vlm)
        verdict = None
    else:
        verdict = vlm.parse_verdict(vr["text"], cl)
        sc = vlm.score(verdict, cl, cfg.thresholds.video_optional_ratio)
        for chk, v in zip(cl.checks, verdict.checks):
            rep = None
            if chk.tag == "continuity":
                bad_shot = _worst_identity_shot(ctx, sp, chk.text, final_frames)
                rep = Repair("IMAGE", f"keyframe:{bad_shot}", v.evidence) if bad_shot else None
            elif chk.tag == "beat":
                idx = int(chk.text.split()[1]) - 1
                rep = Repair("VIDEO", f"clip:{sp.shots[idx].id}", v.evidence)
            _check(ctx, R, chk.text, v.verdict == "yes", v.evidence, mandatory=chk.mandatory, repair=rep,
                   evaluator=cfg.models.vlm)
        _check(ctx, R, "story beats represented (VLM ratio)", sc.optional_ratio >= cfg.thresholds.video_optional_ratio,
               {"ratio": sc.optional_ratio}, mandatory=False, evaluator=cfg.models.vlm)

    # 8. narrative consistency by the separate reasoning critic
    from ..story.validator import CRITIC_SYSTEM
    from ..llm import chat_json
    from ..schemas import CriticReport
    with ctx.rm.ollama_session("critic", cfg.models.critic, ctx.job_id) as sess:
        user = (f"Approved screenplay: {sp.model_dump_json()}\nWhat the finished film actually contains:\n"
                f"- transcript of the final audio: {w.get('text', '')!r}\n"
                f"- visual inspector description of the frames: {verdict.description if verdict else 'unavailable'}\n"
                f"- per-shot video inspection (6 ordered frames per shot, before editing): {_shot_reports(ctx, sp)}\n"
                "Note: visual evidence is SAMPLED frames; a brief moment can fall between samples. Treat a beat as "
                "missing only if no evidence supports it, and prefer the per-shot inspection for actions.\n"
                "Check: (1) narrative consistency of the finished film, (2) every required story beat is present, "
                "(3) narration agrees with the visuals. Only report problems that the evidence shows. "
                "affected = shot ids or ['story']. Return JSON.")
        try:
            rep = chat_json(cfg, cfg.models.critic, CRITIC_SYSTEM, user, CriticReport, session=sess,
                            think=cfg.models.critic_think, temperature=0.2)
            for f in rep.findings:
                origin = Repair("SCRIPT", "screenplay", f.evidence) if "story" in f.affected else (
                    Repair("VIDEO", f"clip:{f.affected[0]}", f.evidence) if f.affected else None)
                _check(ctx, R, f"critic: {f.criterion}", f.result == "pass" or f.severity != "critical",
                       {"severity": f.severity, "evidence": f.evidence}, mandatory=f.severity == "critical",
                       repair=origin, evaluator=cfg.models.critic)
        except Exception as exc:
            _check(ctx, R, "narrative critic ran", False, str(exc), mandatory=False, evaluator=cfg.models.critic)

    mandatory_failed = [r for r in R if r["mandatory"] and not r["passed"]]
    report = {"passed": not mandatory_failed, "checks": R, "mandatory_failed": mandatory_failed,
              "final": str(final), "stream": info, "audio": m, "technical": {k: tech.get(k) for k in (
                  "mean_flow", "freeze_fraction", "black_fraction", "temporal_spike")},
              "limitations": [
                  "Automated checks cannot prove cinematic or artistic quality.",
                  "Identity continuity is judged by a VLM on sampled frames plus Apple Vision feature prints; "
                  "neither proves identical identity.",
                  "There is no lip sync; all speech is narration or off-screen voice.",
                  "Native generation resolution is lower than the 1080x1920 export (lanczos upscale)."]}
    ctx.save_report("final_validation", report)
    return report


def _worst_identity_shot(ctx: Ctx, sp: Screenplay, check_text: str, frames: list[Path]) -> str | None:
    """Shot whose film frame is furthest (Apple Vision face feature print) from the character reference."""
    from ..image.identity import face_distance
    c = next((c for c in sp.story.characters if check_text.startswith(c.name)), None)
    if c is None:
        return None
    ref = Path(ctx.asset(f"char_ref:{c.id}")["path"])
    scored = []
    for i, s in enumerate(sp.shots):
        if c.id in s.character_ids and i < len(frames):
            d = face_distance(ref, frames[i])
            scored.append((d if d is not None else 99.0, s.id))
    return max(scored)[1] if scored else None


def _shot_reports(ctx: Ctx, sp: Screenplay) -> str:
    out = {}
    for s in sp.shots:
        a = ctx.asset(f"clip:{s.id}")
        att = a["meta"].get("approved_attempt", a["attempts"])
        p = ctx.dirs.reports / f"{s.id}_a{att}_semantic.json"
        if p.exists():
            out[s.id] = read_json(p)["verdict"]["description"][:600]
    return json.dumps(out)


def a_text(ctx: Ctx, asset_id: str) -> str:
    try:
        return ctx.asset(f"speech:{asset_id}")["meta"]["text"]
    except KeyError:
        return ""


def _find(words: list[dict], target: list[str], near: float) -> float | None:
    """Start time of the occurrence of target word(s) closest to `near`."""
    if not target:
        return None
    toks = [(norm_words(w["word"]) or [""])[0] for w in words]
    best = None
    for i in range(len(toks)):
        if toks[i] == target[0] and (len(target) < 2 or i + 1 >= len(toks) or toks[i + 1] == target[1]):
            t = words[i]["start"]
            if best is None or abs(t - near) < abs(best - near):
                best = t
    return best
