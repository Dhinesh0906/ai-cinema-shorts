"""Benchmarks and adversarial validator tests on THIS machine. Every number here is measured; results
append to benchmarks/results.json (role -> list of runs) and feed `cinema models`."""
from __future__ import annotations

import json
import shutil
import subprocess
import threading
import time
from datetime import datetime
from pathlib import Path

from .audio.analysis import measure
from .audio.speech_validator import compare
from .config import Config
from .fixtures import broken, good
from .image.generator import character_prompt, location_prompt
from .model_workers import batch_worker, flux_image, wan_frames, wan_video
from .quality import vlm
from .resource_manager import ResourceManager, memory_snapshot
from .storage.database import DB
from .storage.files import atomic_json
from .story.generator import write_shots, write_story
from .story.validator import critic_review, deterministic_checks, mandatory_failures
from .video.generator import negative_prompt
from .video.motion_analysis import analyze, extract_frames, sample_times

FIX_PROMPT_VIDEO = ("The lighthouse keeper slowly raises the brass lantern toward the window and leans closer, "
                    "squinting at something outside; rain streams down the glass, lantern flame flickers, storm "
                    "clouds move, slow cinematic push-in, photorealistic film")
FIX_PROMPT_KEY = ("Cinematic film still, vertical frame. A weathered lighthouse keeper in his 60s with a grey beard, "
                  "navy wool peacoat and knit cap stands at a rain-streaked window inside a dim lighthouse lamp room "
                  "at night, holding a brass oil lantern, warm lantern light on his face, cold blue storm outside, "
                  "shallow depth of field, 35mm film grain, photorealistic")


def _env(cfg: Config):
    db = DB(cfg.db_path)
    return db, ResourceManager(cfg, db)


def _save(cfg: Config, role: str, result: dict) -> dict:
    p = cfg.root / "benchmarks" / "results.json"
    data = json.loads(p.read_text()) if p.exists() else {}
    result = {"when": datetime.now().isoformat(timespec="seconds"), **result}
    data.setdefault(role, [])
    if isinstance(data[role], dict):
        data[role] = [data[role]]
    data[role].append(result)
    atomic_json(p, data)
    return result


def _dir(cfg: Config, name: str) -> Path:
    d = cfg.root / "benchmarks" / name
    d.mkdir(parents=True, exist_ok=True)
    return d


# ------------------------------------------------------------------------------ story
def story(cfg: Config) -> dict:
    db, rm = _env(cfg)
    d = _dir(cfg, "story")
    t0 = time.monotonic()
    with rm.ollama_session("writer", cfg.models.writer, None) as s:
        st = write_story(cfg, "mystery", 45, [], s, 1234)
        t1 = time.monotonic()
        plan = write_shots(cfg, st, 45, s, 1235)
        tok = s.get("eval_tokens")
    t2 = time.monotonic()
    from .schemas import Screenplay
    from .pipeline.script_stage import normalize
    sp = Screenplay(story_id="bench", target_seconds=45, story=st, shots=plan.shots)

    class _C:  # minimal ctx for normalize
        def __init__(self, cfg): self.cfg = cfg
    sp = normalize(_C(cfg), sp)
    det = deterministic_checks(cfg, sp)
    atomic_json(d / "story_45s.json", sp)
    with rm.ollama_session("critic", cfg.models.critic, None) as s:
        rep = critic_review(cfg, sp, s, 99)
    t3 = time.monotonic()
    return _save(cfg, "writer", {"model": cfg.models.writer, "story_seconds": round(t1 - t0, 1),
                 "shots_seconds": round(t2 - t1, 1), "eval_tokens": tok, "shots": len(sp.shots),
                 "deterministic_failures": [f.criterion for f in mandatory_failures(det)],
                 "critic_model": cfg.models.critic, "critic_seconds": round(t3 - t2, 1),
                 "critic_overall_pass": rep.overall_pass,
                 "critic_mandatory": [f.criterion for f in mandatory_failures(rep.findings)],
                 "runs": [r for r in db.worker_runs() if r["worker"] in ("writer", "critic")][-2:]})


# ------------------------------------------------------------------------------ image
def image(cfg: Config) -> dict:
    db, rm = _env(cfg)
    d = _dir(cfg, "image")
    p = cfg.presets[cfg.default_preset]
    g = good()
    tasks = [("portrait", character_prompt(g, "char_elias", cfg.visual_style), p.portrait_width, p.portrait_height, []),
             ("location", location_prompt(g, "loc_lamp_room", cfg.visual_style), p.width, p.height, []),
             ("keyframe_text", FIX_PROMPT_KEY, p.width, p.height, [])]
    out = []
    for name, prompt, w, h, refs in tasks:
        r = flux_image(rm, cfg, prompt, d / f"{name}.png", w, h, 77, p.image_steps, d / "image.log", None, refs)
        out.append({"name": name, "seconds": r.seconds, "peak_footprint_gib": r.peak_footprint_gib, "size": [w, h]})
    # multi-reference keyframe conditioned on the portrait + location plate
    r = flux_image(rm, cfg, "Cinematic film still, vertical 9:16. The man from reference image 1 stands in the room "
                   "from reference image 2, raising his brass lantern toward the rain-streaked window. "
                   + cfg.visual_style, d / "keyframe_multiref.png", p.width, p.height, 78, p.image_steps,
                   d / "image.log", None, [d / "portrait.png", d / "location.png"])
    out.append({"name": "keyframe_multiref", "seconds": r.seconds, "peak_footprint_gib": r.peak_footprint_gib})
    return _save(cfg, "image", {"model": cfg.models.image, "steps": p.image_steps, "images": out,
                                "dir": str(d)})


# ------------------------------------------------------------------------------ video
def _keyframe(cfg: Config, rm, d: Path, preset) -> Path:
    kf = d / f"keeper_{preset.width}x{preset.height}.png"
    if not kf.exists():
        flux_image(rm, cfg, FIX_PROMPT_KEY, kf, preset.width, preset.height, 7, preset.image_steps, d / "image.log", None)
    return kf


def video(cfg: Config, preset_name: str | None = None) -> dict:
    db, rm = _env(cfg)
    d = _dir(cfg, "video")
    name = preset_name or cfg.default_preset
    p = cfg.presets[name]
    kf = _keyframe(cfg, rm, d, p)
    out = d / f"bench_{name}_{p.width}x{p.height}_{p.max_frames}f_{p.steps}s.mp4"
    r = wan_video(rm, cfg, FIX_PROMPT_VIDEO, negative_prompt(), kf, out, p, p.max_frames, 42, d / "video.log", None)
    tech = analyze(out, reference=kf, thresholds=cfg.thresholds)
    md = json.loads(out.with_suffix(".metadata.json").read_text()) if out.with_suffix(".metadata.json").exists() else {}
    return _save(cfg, "video", {"model": cfg.models.video, "preset": name, "width": p.width, "height": p.height,
                                "frames": p.max_frames, "fps": p.fps, "steps": p.steps, "low_ram": p.low_ram,
                                "wall_seconds": r.seconds, "generation_seconds": md.get("generation_time_seconds"),
                                "peak_footprint_gib": r.peak_footprint_gib, "peak_rss_gib": r.peak_rss_gib,
                                "technical": {k: tech.get(k) for k in ("passed", "problems", "mean_flow",
                                                                       "temporal_spike", "feature_drift")},
                                "output": str(out)})


def test_video(cfg: Config, preset_name: str | None) -> dict:
    """Real moving clip, short and cheap enough for a smoke test (49 frames)."""
    db, rm = _env(cfg)
    d = _dir(cfg, "test_video")
    p = cfg.presets[preset_name or cfg.default_preset]
    kf = _keyframe(cfg, rm, d, p)
    out = d / f"test_{datetime.now():%H%M%S}.mp4"
    r = wan_video(rm, cfg, FIX_PROMPT_VIDEO, negative_prompt(), kf, out, p, 49, 42, d / "video.log", None)
    tech = analyze(out, reference=kf, thresholds=cfg.thresholds)
    return {"output": str(out), "seconds": r.seconds, "peak_footprint_gib": r.peak_footprint_gib,
            "technical_passed": tech["passed"], "problems": tech["problems"], "mean_flow": tech.get("mean_flow"),
            "note": "technical pass only shows real decodable motion; semantic validation runs in the pipeline"}


# ------------------------------------------------------------------------------ validators (adversarial)
def validators(cfg: Config) -> dict:
    db, rm = _env(cfg)
    d = _dir(cfg, "validators")
    res = {"script": _script_suite(cfg, rm), "image": _image_suite(cfg, rm, d), "video": _video_suite(cfg, rm, d),
           "audio": _audio_suite(cfg, rm, d)}
    summary = {}
    for k, cases in res.items():
        bad = [c for c in cases if c["expected"] == "reject"]
        goodc = [c for c in cases if c["expected"] == "accept"]
        summary[k] = {"cases": len(cases),
                      "false_accept": sum(c["got"] == "accept" for c in bad), "bad_cases": len(bad),
                      "false_reject": sum(c["got"] == "reject" for c in goodc), "good_cases": len(goodc)}
    return _save(cfg, "validators", {"summary": summary, "cases": res,
                                     "note": "small hand-built test set; rates are indicative, not statistically strong"})


def _script_suite(cfg: Config, rm) -> list[dict]:
    cases = []
    g = good()
    det = mandatory_failures(deterministic_checks(cfg, g))
    with rm.ollama_session("critic", cfg.models.critic, None) as s:
        verdicts = []
        for run in range(2):  # consistency on the known-good script
            rep = critic_review(cfg, g, s, 500 + run)
            verdicts.append([f.criterion for f in mandatory_failures(rep.findings)])
        cases.append({"name": "known_good", "expected": "accept", "deterministic": [f.criterion for f in det],
                      "critic_runs": verdicts,
                      "got": "reject" if det or any(verdicts) else "accept",
                      "critic_consistent": (not verdicts[0]) == (not verdicts[1])})
        for name, (sp, detector, desc) in broken().items():
            det = mandatory_failures(deterministic_checks(cfg, sp))
            crit = []
            if not det:
                rep = critic_review(cfg, sp, s, 600)
                crit = mandatory_failures(rep.findings)
            caught_by = "deterministic" if det else ("critic" if crit else None)
            cases.append({"name": name, "expected": "reject", "description": desc, "expected_detector": detector,
                          "caught_by": caught_by, "got": "reject" if caught_by else "accept",
                          "findings": [f"{f.criterion}: {f.evidence[:160]}" for f in (det or crit)][:5],
                          "actionable_fix": [f.recommended_correction[:160] for f in (det or crit)][:3]})
    return cases


def _image_suite(cfg: Config, rm, d: Path) -> list[dict]:
    g = good()
    p = cfg.presets[cfg.default_preset]
    base = character_prompt(g, "char_elias", cfg.visual_style)
    variants = {
        "correct": (base, "accept"),
        "wrong_clothing": (base.replace("navy wool peacoat with brass buttons", "bright red rain jacket")
                                .replace("charcoal knit cap", "yellow baseball cap"), "reject"),
        "wrong_person": (base.replace("A 64-year-old man", "A 25-year-old woman")
                             .replace("full white beard, grey hair under cap", "long black hair"), "reject"),
        "two_people": (base.replace("Cinematic character reference portrait",
                                    "Two different men standing side by side, cinematic portrait"), "reject"),
    }
    cl = vlm.image_checklist(g, None, character_id="char_elias")
    items, imgs = [], {}
    for name, (prompt, _) in variants.items():
        out = d / f"img_{name}.png"
        if not out.exists():
            flux_image(rm, cfg, prompt, out, p.portrait_width, p.portrait_height, 31, p.image_steps, d / "image.log", None)
        imgs[name] = out
        items.append({"id": name, "images": [str(out)], "prompt": vlm.build_prompt(cl, "Character reference portrait "
                      "for Elias Marr.", ["GENERATED image under inspection"])})
    # wrong location: location plate checked against the lamp-room spec
    lcl = vlm.image_checklist(g, None, location_id="loc_lamp_room")
    beach = d / "img_wrong_location.png"
    if not beach.exists():
        flux_image(rm, cfg, "Cinematic establishing shot of an empty sunny tropical beach with palm trees, no people",
                   beach, p.width, p.height, 32, p.image_steps, d / "image.log", None)
    lamp = d / "img_right_location.png"
    if not lamp.exists():
        flux_image(rm, cfg, location_prompt(g, "loc_lamp_room", cfg.visual_style), lamp, p.width, p.height, 33,
                   p.image_steps, d / "image.log", None)
    for name, path in (("wrong_location", beach), ("right_location", lamp)):
        items.append({"id": name, "images": [str(path)], "prompt": vlm.build_prompt(lcl, "Location plate.",
                      ["GENERATED image under inspection"])})
    res = batch_worker(rm, cfg, "vlm", "vlm", cfg.models.vlm, {"model": cfg.models.vlm, "max_side": 768}, items,
                       d / "vlm_images", None)
    expected = {**{k: v[1] for k, v in variants.items()}, "wrong_location": "reject", "right_location": "accept"}
    cases = []
    for name, exp in expected.items():
        r = res.get(name, {})
        c = lcl if "location" in name else cl
        if "error" in r or not r:
            cases.append({"name": name, "expected": exp, "got": "error", "error": r.get("error")})
            continue
        v = vlm.parse_verdict(r["text"], c)
        sc = vlm.score(v, c, cfg.thresholds.image_optional_ratio)
        cases.append({"name": name, "expected": exp, "got": "accept" if sc.passed else "reject",
                      "mandatory_failed": sc.mandatory_failed[:4], "seconds": r.get("seconds"),
                      "peak_memory_gb": r.get("peak_memory_gb")})
    return cases


def _video_suite(cfg: Config, rm, d: Path) -> list[dict]:
    g = good()
    vdir = cfg.root / "benchmarks" / "video"
    good_clip = next(iter(sorted(vdir.glob("keeper_i2v_480x832_81f_25s*.mp4"))), None)
    collapsed = vdir / "smoke_i2v_256x448_17f.mp4"
    kf = vdir / "keeper_480x832.png"
    cases = []
    # synthetic technical failures
    frozen = d / "frozen.mp4"
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-loop", "1", "-i", str(kf), "-t", "5", "-r", "16",
                    "-pix_fmt", "yuv420p", str(frozen)], check=True)
    black = d / "black.mp4"
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i", "color=black:s=480x832:r=16:d=5",
                    "-pix_fmt", "yuv420p", str(black)], check=True)
    broken_f = d / "broken.mp4"
    if good_clip:
        data = good_clip.read_bytes()
        broken_f.write_bytes(data[: len(data) // 3])
    for name, path, exp in (("frozen_video", frozen, "reject"), ("black_video", black, "reject"),
                            ("broken_file", broken_f, "reject"), ("collapsed_generation", collapsed, "reject"),
                            ("good_clip_technical", good_clip, "accept")):
        if path is None or not Path(path).exists():
            cases.append({"name": name, "expected": exp, "got": "error", "error": "fixture missing"})
            continue
        m = analyze(Path(path), thresholds=cfg.thresholds)
        cases.append({"name": name, "layer": "technical", "expected": exp, "got": "accept" if m["passed"] else "reject",
                      "problems": m.get("problems")})
    # semantic: same real clip judged against the right action and a wrong action
    if good_clip:
        right = g.shots[0].model_copy(update={"action": "the keeper raises his brass lantern toward the window",
                                              "expected_action_sequence": ["he raises the lantern", "he leans toward the window"]})
        wrong = right.model_copy(update={"action": "the keeper drops the lantern on the floor and runs out of the room",
                                         "expected_action_sequence": ["the lantern falls to the floor",
                                                                      "he runs out of the room"]})
        frames = extract_frames(good_clip, sample_times(5.0, 6), d / "frames_good")
        cframes = extract_frames(collapsed, sample_times(1.0, 6), d / "frames_collapsed") if collapsed.exists() else []
        items = []
        for name, shot, fr in (("right_action", right, frames), ("wrong_action", wrong, frames),
                               ("collapsed_semantic", right, cframes)):
            if not fr:
                continue
            cl = vlm.video_checklist(g, shot)
            roles = ["REFERENCE approved first frame"] + [f"VIDEO FRAME {i+1} (chronological)" for i in range(len(fr))]
            items.append({"id": name, "images": [str(kf)] + [str(f) for f in fr],
                          "prompt": vlm.build_prompt(cl, f"Script action: {shot.action}", roles), "max_side": 512,
                          "_cl": shot})
        res = batch_worker(rm, cfg, "vlm", "vlm", cfg.models.vlm, {"model": cfg.models.vlm, "max_side": 512},
                           [{k: v for k, v in it.items() if k != "_cl"} for it in items], d / "vlm_video", None)
        for it in items:
            r = res.get(it["id"], {})
            exp = "accept" if it["id"] == "right_action" else "reject"
            if "error" in r or not r:
                cases.append({"name": it["id"], "expected": exp, "got": "error", "error": r.get("error")})
                continue
            cl = vlm.video_checklist(g, it["_cl"])
            v = vlm.parse_verdict(r["text"], cl)
            sc = vlm.score(v, cl, cfg.thresholds.video_optional_ratio)
            cases.append({"name": it["id"], "layer": "semantic", "expected": exp,
                          "got": "accept" if sc.passed else "reject", "mandatory_failed": sc.mandatory_failed[:4],
                          "description": v.description[:300], "seconds": r.get("seconds")})
    return cases


def _audio_suite(cfg: Config, rm, d: Path) -> list[dict]:
    text = "Forty years on this rock, and tonight the sea answered back."
    tts = batch_worker(rm, cfg, "tts", "tts", "kokoro", {"model": str(cfg.resolve(cfg.models.kokoro_model)),
                       "voices": str(cfg.resolve(cfg.models.kokoro_voices))},
                       [{"id": "line", "text": text, "voice": "bm_lewis", "speed": 1.0, "out": str(d / "line.wav")},
                        {"id": "line_missing", "text": "Forty years on this rock, and tonight.", "voice": "bm_lewis",
                         "speed": 1.0, "out": str(d / "line_missing.wav")}], d / "tts", None)
    clipped = d / "line_clipped.wav"
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(d / "line.wav"), "-af", "volume=24dB,alimiter=limit=1:level=false",
                    str(clipped)], check=True)
    gap = d / "line_gap.wav"
    # build a mid-line gap: first half + 1.5 s silence + second half
    dur = measure(d / "line.wav")["duration"]
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(d / "line.wav"), "-filter_complex",
                    f"[0:a]atrim=0:{dur/2},asetpts=PTS-STARTPTS[a];aevalsrc=0:d=1.5:s=24000[s];"
                    f"[0:a]atrim={dur/2},asetpts=PTS-STARTPTS[b];[a][s][b]concat=n=3:v=0:a=1", str(gap)], check=True)
    files = {"correct": (d / "line.wav", "accept"), "missing_words": (d / "line_missing.wav", "reject"),
             "clipped": (clipped, "reject"), "mid_line_gap": (gap, "reject")}
    wres = batch_worker(rm, cfg, "whisper", "whisper", cfg.models.whisper, {"model": cfg.models.whisper},
                        [{"id": k, "audio": str(v[0])} for k, v in files.items()], d / "whisper", None)
    t = cfg.thresholds
    cases = []
    for k, (path, exp) in files.items():
        w = wres.get(k, {})
        cmp = compare(text, w.get("text", ""))
        m = measure(path)
        probs = []
        if cmp["wer"] > t.max_wer or cmp["missing_content_words"]:
            probs.append(f"wer {cmp['wer']} missing {cmp['missing_content_words']}")
        if m["peak_db"] is not None and m["peak_db"] > t.max_peak_dbfs:
            probs.append(f"peak {m['peak_db']}")
        inner = [s for s in m["silences"] if s["start"] > 0.05 and s["end"] < m["duration"] - 0.05 and s["duration"] > 0.9]
        if inner:
            probs.append(f"gap {inner}")
        cases.append({"name": k, "expected": exp, "got": "reject" if probs else "accept", "problems": probs,
                      "transcript": w.get("text")})
    return cases


# ------------------------------------------------------------------------------ memory isolation
def memory(cfg: Config) -> dict:
    """Prove one heavy worker at a time, clean exits, Ollama unload, and memory recovery."""
    db, rm = _env(cfg)
    d = _dir(cfg, "memory")
    start_id = max([r["id"] for r in db.worker_runs()] or [0])
    snaps = {"start": memory_snapshot()}
    p = cfg.presets[cfg.default_preset]
    errors = []

    def img(i):
        try:
            flux_image(rm2(), cfg, FIX_PROMPT_KEY, d / f"concurrent_{i}.png", p.width, p.height, 100 + i,
                       p.image_steps, d / "image.log", None)
        except Exception as exc:
            errors.append(str(exc))

    def rm2():  # separate manager instance = separate process-level contender for the file lock
        return ResourceManager(cfg, DB(cfg.db_path))

    threads = [threading.Thread(target=img, args=(i,)) for i in range(2)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    snaps["after_concurrent_images"] = memory_snapshot()
    # Ollama: load during session, verify listed, then verify unloaded
    with rm.ollama_session("writer", cfg.models.writer, None) as s:
        import httpx
        httpx.post(f"{cfg.ollama_url}/api/generate", json={"model": cfg.models.writer, "prompt": "Say OK.",
                   "stream": False, "keep_alive": "5m", "think": False, "options": {"num_predict": 5}}, timeout=600)
        loaded_during = rm.ollama_loaded()
        snaps["ollama_loaded"] = memory_snapshot()
    loaded_after = rm.ollama_loaded()
    snaps["after_ollama"] = memory_snapshot()
    runs = [r for r in db.worker_runs() if r["id"] > start_id]
    spans = sorted((r["started_at"], r["ended_at"], r["worker"]) for r in runs)
    overlaps = [(a, b) for a, b in zip(spans, spans[1:]) if b[0] < a[1]]
    leftover = subprocess.run(["pgrep", "-fl", "mlxgen|cinema.workers"], capture_output=True, text=True).stdout.strip()
    return _save(cfg, "memory", {
        "worker_runs": runs, "overlapping_heavy_runs": overlaps, "all_exited_clean": all(r["exited_clean"] for r in runs),
        "ollama_loaded_during_session": loaded_during, "ollama_loaded_after_session": loaded_after,
        "leftover_worker_processes": leftover or None, "snapshots": snaps, "errors": errors,
        "verdict": "PASS" if not overlaps and not loaded_after and not leftover and all(r["exited_clean"] for r in runs)
        and not errors else "FAIL"})


# ------------------------------------------------------------------------------ image backend A/B
def image_ab(cfg: Config, job_id: str) -> dict:
    """Same keyframes (same refs, prompts, seeds) from local klein 4B vs Cloudflare FLUX.2 dev,
    judged by the same VLM checklist + Apple Vision face distance to the character portrait."""
    from .image.generator import keyframe_prompt, keyframe_refs
    from .image.identity import face_distance
    from .pipeline.context import Ctx
    db, rm = _env(cfg)
    ctx = Ctx(cfg, db, rm, job_id)
    sp = ctx.screenplay()
    d = _dir(cfg, f"image_ab/{job_id}")
    p = ctx.preset
    chars = {c.id: Path(ctx.asset(f"char_ref:{c.id}")["path"]) for c in sp.story.characters}
    locs = {loc.id: Path(ctx.asset(f"loc_ref:{loc.id}")["path"]) for loc in sp.story.locations}
    items, gen = [], []
    for backend in ("local", "cloudflare"):
        cfg.image_backend = backend
        for s in sp.shots:
            refs, roles = keyframe_refs(sp, s, chars, locs)
            out = d / f"{backend}_{s.id}.png"
            prompt = keyframe_prompt(sp, s, ctx.style, roles)
            r = flux_image(rm, cfg, prompt, out, p.width, p.height, 4242, p.image_steps, d / "image.log", None, refs)
            gen.append({"backend": backend, "shot": s.id, "seconds": r.seconds, "extra": r.extra})
            cl = vlm.image_checklist(sp, s)
            vroles = [ro for ro in roles if ro.startswith("character")] + ["GENERATED image under inspection"]
            items.append({"id": f"{backend}:{s.id}", "images": [str(x) for x, ro in zip(refs, roles)
                          if ro.startswith("character")] + [str(out)],
                          "prompt": vlm.build_prompt(cl, f"First frame of {s.id}: {s.action}", vroles)})
    cfg.image_backend = "local"
    res = batch_worker(rm, cfg, "vlm", "vlm", cfg.models.vlm, {"model": cfg.models.vlm, "max_side": 768}, items,
                       d / "vlm", None)
    rows = []
    for it in items:
        backend, sid = it["id"].split(":")
        s = sp.shot(sid)
        cl = vlm.image_checklist(sp, s)
        r = res.get(it["id"], {})
        sc = vlm.score(vlm.parse_verdict(r["text"], cl), cl, cfg.thresholds.image_optional_ratio) if "text" in r else None
        fd = face_distance(chars[s.character_ids[0]], d / f"{backend}_{sid}.png") if s.character_ids else None
        rows.append({"backend": backend, "shot": sid, "passed": sc.passed if sc else None,
                     "score": sc.score if sc else None, "mandatory_failed": sc.mandatory_failed[:3] if sc else r,
                     "face_distance_to_ref": fd})
    summary = {b: {"passed": sum(1 for r in rows if r["backend"] == b and r["passed"]),
                   "mean_score": round(sum(r["score"] or 0 for r in rows if r["backend"] == b) /
                                       max(1, sum(1 for r in rows if r["backend"] == b)), 3),
                   "mean_seconds": round(sum(g["seconds"] for g in gen if g["backend"] == b) /
                                         max(1, sum(1 for g in gen if g["backend"] == b)), 1)}
               for b in ("local", "cloudflare")}
    return _save(cfg, "image_ab", {"job": job_id, "summary": summary, "rows": rows, "generation": gen,
                                   "dir": str(d)})
