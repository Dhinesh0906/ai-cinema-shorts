"""VLM verdict parsing/scoring, and retry limits of a full stage with mocked model adapters."""
import json

import pytest
from PIL import Image

from cinema.fixtures import good
from cinema.quality import vlm


def _verdict(cl, answers):
    return json.dumps({"description": "x", "checks": [{"item": c.text, "verdict": a, "evidence": "e"}
                                                      for c, a in zip(cl.checks, answers)],
                       "defects": [], "correction": "make the coat navy"})


def test_parse_and_score_strict():
    cl = vlm.image_checklist(good(), None, character_id="char_elias")
    n = len(cl.checks)
    v = vlm.parse_verdict("Sure! ```json\n" + _verdict(cl, ["yes"] * n) + "\n```", cl)
    assert vlm.score(v, cl, 0.6).passed
    # a missing answer counts as unclear -> mandatory failure
    v = vlm.parse_verdict(_verdict(cl, ["yes"] * (n - 1)), cl)
    sc = vlm.score(v, cl, 0.6)
    assert not sc.passed and sc.mandatory_failed
    # wrong costume is a hard fail even if everything else is yes
    answers = ["yes"] * n
    idx = next(i for i, c in enumerate(cl.checks) if c.tag == "costume")
    answers[idx] = "no"
    assert not vlm.score(vlm.parse_verdict(_verdict(cl, answers), cl), cl, 0.6).passed


def test_video_checklist_has_temporal_and_artifact_gates():
    sp = good()
    cl = vlm.video_checklist(sp, sp.shots[0])
    tags = {c.tag for c in cl.checks if c.mandatory}
    assert {"action", "continuity", "artifact", "location"} <= tags


@pytest.mark.parametrize("vlm_text, outcome", [
    (json.dumps({"description": "wrong", "checks": [], "defects": [], "correction": "fix"}), "best_of"),
    ("I cannot judge this image.", "blocked"),
])
def test_image_stage_after_max_attempts(cfg, monkeypatch, tmp_path, vlm_text, outcome):
    """Validator rejects everything -> each asset tried exactly max_attempts times, then the best validated
    candidate ships; if no candidate was ever validated (unparseable verdicts) the job BLOCKS."""
    from cinema.orchestrator import Orchestrator
    from cinema.pipeline import image_stage
    from cinema.pipeline.context import Ctx, StageBlocked
    from cinema.storage.files import atomic_json

    calls = {"gen": 0, "vlm": 0}

    def fake_flux(rm, cfg_, prompt, out, w, h, seed, steps, log, job_id, refs=None):
        calls["gen"] += 1
        Image.new("RGB", (w, h), (seed % 255, 0, 0)).save(out)

        class R:
            seconds, peak_footprint_gib, extra = 0.1, 0.1, {}
        return R()

    def fake_batch(rm, cfg_, role, module, model, options, items, workdir, job_id, timeout=None):
        calls["vlm"] += 1
        return {it["id"]: {"text": vlm_text} for it in items}

    monkeypatch.setattr(image_stage, "flux_image", fake_flux)
    monkeypatch.setattr(image_stage, "batch_worker", fake_batch)
    orch = Orchestrator(cfg)
    job = orch.create_job("mystery", 30, "balanced", seed=1)
    from cinema.state_machine import JobState
    orch.db.set_job_status(job, JobState.PREFLIGHT)
    ctx = Ctx(cfg, orch.db, orch.rm, job)
    sp = good()
    atomic_json(ctx.dirs.script / "screenplay.json", sp)
    if outcome == "blocked":
        with pytest.raises(StageBlocked):
            image_stage.run(ctx, sp)
        a = orch.db.asset(job, "char_ref:char_elias")
        assert a["status"] == "BLOCKED" and a["feedback"].startswith("no usable attempt")
        assert calls["vlm"] == cfg.retries.max_attempts  # one validator load per round, not per image
    else:
        image_stage.run(ctx, sp)
        a = orch.db.asset(job, "char_ref:char_elias")
        assert a["status"] == "PASSED" and a["meta"]["best_of"] is True
    assert a["attempts"] == cfg.retries.max_attempts


def test_best_of_ranks_fewest_mandatory_failures_then_score(cfg, tmp_path):
    import os
    from cinema.orchestrator import Orchestrator
    from cinema.pipeline import best_of
    from cinema.pipeline.context import Ctx
    orch = Orchestrator(cfg)
    ctx = Ctx(cfg, orch.db, orch.rm, orch.create_job("mystery", 30, "balanced", seed=1))
    kf = tmp_path / "kf.png"
    kf.write_bytes(b"k")
    orch.db.ensure_asset(ctx.job_id, "keyframe:shot_01", "keyframe", [])
    orch.db.set_asset(ctx.job_id, "keyframe:shot_01", path=str(kf))
    reports = {1: (0.9, ["a", "b"]), 2: (0.4, ["a"]), 3: (0.8, ["c"])}   # 3 wins: 1 failure, higher score
    for i, (score, failed) in reports.items():
        (ctx.dirs.work / f"shot_01_a{i}.mp4").write_bytes(b"v")
        (ctx.dirs.reports / f"shot_01_a{i}_semantic.json").write_text(
            json.dumps({"score": score, "mandatory_failed": failed}))
    best = best_of.pick(ctx, best_of.clip_candidates(ctx, "shot_01"), ["keyframe:shot_01"])
    assert best["path"].name == "shot_01_a3.mp4"
    stale = os.path.getmtime(kf) - 100                                    # rendered before the current keyframe
    os.utime(ctx.dirs.work / "shot_01_a3.mp4", (stale, stale))
    assert best_of.pick(ctx, best_of.clip_candidates(ctx, "shot_01"), ["keyframe:shot_01"])["path"].name == "shot_01_a2.mp4"
    (ctx.dirs.work / "shot_01_a2.mp4").unlink()                           # crashed render: no file
    (ctx.dirs.reports / "shot_01_a1_semantic.json").unlink()              # technical failure: no semantic report
    assert best_of.pick(ctx, best_of.clip_candidates(ctx, "shot_01"), ["keyframe:shot_01"]) is None


def test_flux_reference_modes(cfg, monkeypatch, tmp_path):
    """FLUX.2: multi-reference only with 2+ images; always exact canvas when conditioning."""
    from cinema import model_workers as mw
    seen = []
    monkeypatch.setattr(mw.ResourceManager, "run_worker", lambda self, *a, **k: seen.append(a[2]))
    rm = mw.ResourceManager.__new__(mw.ResourceManager)
    for refs in ([tmp_path / "a.png"], [tmp_path / "a.png", tmp_path / "b.png"]):
        mw._flux_local(rm, cfg, "p", tmp_path / "o.png", 480, 832, 1, 4, tmp_path / "l", None, refs)
    one, two = seen
    assert "multi-reference" not in one and "exact-resize" in one
    assert "multi-reference" in two and "exact-resize" in two


def test_resume_after_blocked_keeps_exhausted_visuals_for_best_of(cfg, monkeypatch):
    from cinema.orchestrator import Orchestrator
    from cinema.state_machine import AssetState, JobState
    orch = Orchestrator(cfg)
    job = orch.create_job("mystery", 30, "balanced", seed=1)
    db = orch.db
    for aid, fb in (("clip:shot_01", "hand in frame"), ("clip:shot_02", "no usable attempt: all crashed"),
                    ("clip:shot_03", "costume")):
        db.ensure_asset(job, aid, "clip", [])
        n = cfg.retries.max_attempts if aid != "clip:shot_03" else 1
        for _ in range(n):
            db.set_asset(job, aid, attempts_inc=True)
        db.set_asset(job, aid, AssetState.GENERATING)
        db.set_asset(job, aid, AssetState.VALIDATING)
        db.set_asset(job, aid, AssetState.RETRY_PENDING, feedback=fb)
    db.set_job_status(job, JobState.BLOCKED)
    monkeypatch.setattr(orch, "run", lambda job_id: None)
    orch.resume(job)
    att = {a["asset_id"]: a["attempts"] for a in db.assets(job)}
    assert att["clip:shot_01"] == cfg.retries.max_attempts   # exhausted + usable -> best-of, no extra tries
    assert att["clip:shot_02"] == 0                          # nothing usable -> fresh tries
    assert att["clip:shot_03"] == 0                          # still had budget -> unchanged behaviour


def test_video_stage_ships_best_of_after_max_attempts(cfg, monkeypatch):
    """Every clip rejected 3 times -> best attempt approved, stage returns (no BLOCK, no endless loop)."""
    from cinema.orchestrator import Orchestrator
    from cinema.pipeline import video_stage
    from cinema.pipeline.context import Ctx
    from cinema.state_machine import AssetState, JobState
    from cinema.storage.files import atomic_json
    renders = []

    def fake_wan(rm, cfg_, prompt, neg, keyframe, out, preset, frames, seed, log, job_id, steps, guidance):
        out.write_bytes(b"v")
        renders.append(out.name)

        class R:
            seconds, peak_footprint_gib, peak_rss_gib = 1.0, 0.1, 0.1
        return R()

    cfg.video_backend = "local"
    monkeypatch.setattr(video_stage, "wan_video", fake_wan)
    monkeypatch.setattr(video_stage, "analyze", lambda *a, **k: {"passed": True, "duration": 4.0, "problems": []})
    monkeypatch.setattr(video_stage, "extract_frames", lambda *a, **k: [])
    monkeypatch.setattr(video_stage, "batch_worker", lambda *a, **k: {it["id"]: {"text": json.dumps(
        {"description": "x", "checks": [], "defects": [], "correction": "fix"})} for it in a[6]})
    orch = Orchestrator(cfg)
    job = orch.create_job("mystery", 30, "balanced", seed=1)
    orch.db.set_job_status(job, JobState.PREFLIGHT)
    ctx = Ctx(cfg, orch.db, orch.rm, job)
    sp = good()
    sp.shots = sp.shots[:2]
    atomic_json(ctx.dirs.script / "screenplay.json", sp)
    refs = [f"keyframe:{s.id}" for s in sp.shots] + [f"char_ref:{c.id}" for c in sp.story.characters]
    for aid in refs:
        img = ctx.dirs.sub("references") / f"{aid.replace(':', '_')}.png"
        img.write_bytes(b"k")
        orch.db.ensure_asset(job, aid, aid.split(":")[0], [])
        orch.db.set_asset(job, aid, path=str(img))
        for st in (AssetState.GENERATING, AssetState.VALIDATING, AssetState.PASSED):
            orch.db.set_asset(job, aid, st)
    video_stage.run(ctx, sp)
    assert len(renders) == 2 * cfg.retries.max_attempts
    for s in sp.shots:
        a = orch.db.asset(job, f"clip:{s.id}")
        assert a["status"] == "PASSED" and a["meta"]["best_of"] and (ctx.dirs.clips / f"{s.id}.mp4").exists()
