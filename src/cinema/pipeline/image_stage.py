"""IMAGE AND CHARACTER GENERATION -> INDEPENDENT IMAGE VALIDATION -> targeted regeneration.

Round structure (bounded): generate candidates for every pending asset with FLUX (one isolated process
per image, nothing else loaded in between) -> VLM worker validates the whole batch in one load ->
best passing candidate is approved -> only failures regenerate with targeted corrections.
"""
from __future__ import annotations

import zlib
from dataclasses import dataclass
from pathlib import Path

from PIL import Image

from ..image import identity
from ..image.generator import character_prompt, keyframe_prompt, keyframe_refs, location_prompt
from ..model_workers import batch_worker, flux_image
from ..quality import vlm
from ..resource_manager import WorkerFailed
from ..schemas import Decision, Screenplay
from ..state_machine import AssetState, JobState
from . import best_of
from .context import Ctx, StageBlocked


@dataclass
class Spec:
    aid: str
    kind: str
    dest: Path
    width: int
    height: int
    checklist: vlm.Checklist
    context: str
    deps: list[str]
    shot_id: str | None = None
    char_id: str | None = None
    loc_id: str | None = None


def _specs(ctx: Ctx, sp: Screenplay, which: str) -> list[Spec]:
    p, d = ctx.preset, ctx.dirs
    out = []
    if which == "refs":
        for c in sp.story.characters:
            out.append(Spec(f"char_ref:{c.id}", "char_ref", d.characters / f"{c.id}.png", p.portrait_width,
                            p.portrait_height, vlm.image_checklist(sp, None, character_id=c.id),
                            f"Character reference portrait for {c.name} ({c.role}) in '{sp.story.title}'.",
                            ["screenplay"], char_id=c.id))
        for loc in sp.story.locations:
            out.append(Spec(f"loc_ref:{loc.id}", "loc_ref", d.locations / f"{loc.id}.png", p.width, p.height,
                            vlm.image_checklist(sp, None, location_id=loc.id),
                            f"Empty location plate: {loc.name}.", ["screenplay"], loc_id=loc.id))
    else:
        for s in sp.shots:
            deps = [f"char_ref:{c}" for c in s.character_ids] + [f"loc_ref:{s.location_id}"]
            ctxt = (f"First frame of {s.id} ({s.beat}) in '{sp.story.title}'. Action that will follow: {s.action}. "
                    f"Framing: {s.framing}.")
            out.append(Spec(f"keyframe:{s.id}", "keyframe", d.keyframes / f"{s.id}.png", p.width, p.height,
                            vlm.image_checklist(sp, s), ctxt, ["screenplay", *deps], shot_id=s.id))
    return out


def _prompt(ctx: Ctx, sp: Screenplay, spec: Spec, correction: str, refs_roles: list[str]) -> str:
    if spec.kind == "char_ref":
        return character_prompt(sp, spec.char_id, ctx.style, correction)
    if spec.kind == "loc_ref":
        return location_prompt(sp, spec.loc_id, ctx.style, correction)
    return keyframe_prompt(sp, sp.shot(spec.shot_id), ctx.style, refs_roles, correction)


def _refs(ctx: Ctx, sp: Screenplay, spec: Spec) -> tuple[list[Path], list[str]]:
    if spec.kind != "keyframe":
        return [], []
    chars = {c.id: Path(ctx.asset(f"char_ref:{c.id}")["path"]) for c in sp.story.characters}
    locs = {loc.id: Path(ctx.asset(f"loc_ref:{loc.id}")["path"]) for loc in sp.story.locations}
    return keyframe_refs(sp, sp.shot(spec.shot_id), chars, locs)


def _round(ctx: Ctx, sp: Screenplay, specs: list[Spec], label: str) -> None:
    cfg = ctx.cfg
    for s in specs:
        ctx.ensure(s.aid, s.kind, s.deps)
    pending = [s for s in specs if not ctx.passed(s.aid)]
    while pending:
        # ---------------- generate candidates (image model only)
        ctx.state(JobState.GENERATING_IMAGES)
        cands: dict[str, list[Path]] = {}
        for s in pending:
            a = ctx.asset(s.aid)
            attempt = a["attempts"] + 1
            if attempt > cfg.retries.max_attempts:  # retries used up: ship the best validated candidate
                best = best_of.pick(ctx, best_of.image_candidates(ctx, s.aid), s.deps)
                if best is None:
                    ctx.to(s.aid, AssetState.BLOCKED, feedback=f"{best_of.NO_USABLE}: {a['feedback']}"[:1500])
                    raise StageBlocked(f"{s.aid} failed image validation {cfg.retries.max_attempts} times and no "
                                       f"candidate is usable: {a['feedback']}")
                note = best_of.summary(best, cfg.retries.max_attempts)
                ctx.approve(s.aid, best["path"], s.dest, meta={"best_of": True, "score": best["score"],
                                                               "source_candidate": best["path"].name})
                ctx.decide(Decision(asset_id=s.aid, stage="image", criterion="asset approved", evaluator="controller",
                                    passed=True, score=best["score"], attempt=a["attempts"], evidence=note,
                                    action="best-of fallback"))
                ctx.warn(f"{s.aid} {note}")
                continue
            ctx.to(s.aid, AssetState.GENERATING, attempts_inc=True)
            refs, roles = _refs(ctx, sp, s)
            correction = (a["feedback"] or "") if attempt > 1 else ""
            prompt = _prompt(ctx, sp, s, correction[:600], roles)
            cands[s.aid] = []
            for k in range(cfg.image_candidates):
                seed = (ctx.seed + zlib.crc32(s.aid.encode()) % 100_000 + attempt * 1000 + k) % 2**31
                out = ctx.dirs.work / f"{s.aid.replace(':', '_')}_a{attempt}_c{k}.png"
                try:
                    r = flux_image(ctx.rm, cfg, prompt, out, s.width, s.height, seed, ctx.preset.image_steps,
                                   ctx.dirs.logs / "image.log", ctx.job_id, refs)
                    cands[s.aid].append(out)
                    ctx.db.set_asset(ctx.job_id, s.aid, meta={f"cand{k}": {"seed": seed, "prompt": prompt,
                                     "refs": [str(x) for x in refs], "seconds": r.seconds,
                                     "backend": r.extra.get("backend", "local"),
                                     "model": r.extra.get("model", cfg.models.image),
                                     "peak_footprint_gib": r.peak_footprint_gib}})
                except WorkerFailed as exc:
                    ctx.warn(f"{s.aid} candidate {k} generation failed: {exc}")
            ctx.to(s.aid, AssetState.VALIDATING)
        # ---------------- validate whole batch (VLM only)
        ctx.state(JobState.VALIDATING_IMAGES)
        items, meta = [], {}
        for s in pending:
            refs, roles = _refs(ctx, sp, s)
            if s.kind == "keyframe":  # only character refs go to the validator
                pairs = [(r, ro) for r, ro in zip(refs, roles) if ro.startswith("character")]
            else:
                pairs = []
            for k, c in enumerate(cands.get(s.aid, [])):
                iid = f"{s.aid}#{k}"
                size = Image.open(c).size
                if size != (s.width, s.height):  # deterministic gate before any VLM time is spent
                    ctx.decide(Decision(asset_id=s.aid, stage="image", criterion="exact canvas size",
                                        evaluator="deterministic", passed=False, attempt=ctx.asset(s.aid)["attempts"],
                                        evidence=f"{size} != {(s.width, s.height)}"))
                    continue
                roles_v = [ro for _, ro in pairs] + ["GENERATED image under inspection"]
                items.append({"id": iid, "images": [str(r) for r, _ in pairs] + [str(c)],
                              "prompt": vlm.build_prompt(s.checklist, s.context, roles_v), "max_tokens": 1400})
                meta[iid] = (s, c)
        results = batch_worker(ctx.rm, cfg, "vlm", "vlm", cfg.models.vlm, {"model": cfg.models.vlm, "max_side": 768},
                               items, ctx.dirs.work / f"vlm_{label}", ctx.job_id) if items else {}
        nxt = []
        for s in pending:
            if ctx.passed(s.aid):  # approved by the best-of fallback this round
                continue
            attempt = ctx.asset(s.aid)["attempts"]
            best, best_score, notes = None, -1.0, []
            for k, c in enumerate(cands.get(s.aid, [])):
                iid = f"{s.aid}#{k}"
                res = results.get(iid, {"error": "no result"})
                if "error" in res:
                    notes.append(f"validator error: {res['error']}")
                    continue
                try:
                    verdict = vlm.parse_verdict(res["text"], s.checklist)
                except ValueError as exc:
                    notes.append(f"unparseable validator output: {exc}")
                    continue
                sc = vlm.score(verdict, s.checklist, cfg.thresholds.image_optional_ratio)
                evidence = _vision_evidence(ctx, sp, s, c)
                ctx.save_report(f"{s.aid.replace(':', '_')}_a{attempt}_c{k}",
                                {"checklist": [c_.__dict__ for c_ in s.checklist.checks], "verdict": verdict.model_dump(),
                                 "passed": sc.passed, "score": sc.score, "mandatory_failed": sc.mandatory_failed,
                                 "optional_ratio": sc.optional_ratio, "apple_vision": evidence,
                                 "validator": cfg.models.vlm, "raw": res["text"][:4000]})
                for chk, v in zip(s.checklist.checks, verdict.checks):
                    ctx.decide(Decision(asset_id=s.aid, stage="image", criterion=chk.text, evaluator=cfg.models.vlm,
                                        passed=v.verdict == "yes", mandatory=chk.mandatory, evidence=v.evidence,
                                        attempt=attempt, action=f"candidate {k}"))
                if evidence.get("subject_count_mismatch"):
                    ctx.decide(Decision(asset_id=s.aid, stage="image", criterion="Apple Vision person count",
                                        evaluator="apple-vision", passed=False, mandatory=False,
                                        evidence=str(evidence), attempt=attempt))
                if sc.passed and sc.score > best_score:
                    best, best_score = c, sc.score
                elif not sc.passed:
                    notes.append("; ".join(sc.mandatory_failed)[:500] + (f" | fix: {verdict.correction}" if verdict.correction else ""))
            if best is not None:
                ctx.approve(s.aid, best, s.dest, meta={"score": best_score, "validator": cfg.models.vlm,
                                                       "attempt": attempt, "source_candidate": best.name})
                ctx.decide(Decision(asset_id=s.aid, stage="image", criterion="asset approved", evaluator="controller",
                                    passed=True, score=best_score, attempt=attempt, action="approved best candidate"))
            else:
                fb = " || ".join(notes)[:1500] or "no candidate passed"
                ctx.to(s.aid, AssetState.RETRY_PENDING, feedback=fb)
                ctx.decide(Decision(asset_id=s.aid, stage="image", criterion="asset approved", evaluator="controller",
                                    passed=False, attempt=attempt, evidence=fb, action="regenerate with correction"))
                ctx.warn(f"{s.aid} rejected (attempt {attempt}): {fb[:200]}")
                nxt.append(s)
        pending = nxt


def _vision_evidence(ctx: Ctx, sp: Screenplay, s: Spec, img: Path) -> dict:
    """Apple Vision supporting evidence (never a gate on its own)."""
    try:
        people = identity.detect_people(img)
    except Exception as exc:  # pragma: no cover - framework hiccup
        return {"error": str(exc)}
    expected = 1 if s.kind == "char_ref" else 0 if s.kind == "loc_ref" else sp.shot(s.shot_id).subject_count
    ev = {"faces": people["face_count"], "humans": people["human_count"], "expected_people": expected,
          "subject_count_mismatch": max(people["face_count"], people["human_count"]) != expected
          and not (expected > 0 and people["human_count"] == 0 and people["face_count"] == 0)}
    if s.kind == "keyframe" and len(sp.shot(s.shot_id).character_ids) == 1:
        cid = sp.shot(s.shot_id).character_ids[0]
        try:
            ev["face_distance_to_ref"] = identity.face_distance(Path(ctx.asset(f"char_ref:{cid}")["path"]), img)
        except Exception as exc:  # pragma: no cover
            ev["face_distance_error"] = str(exc)
    return ev


def run(ctx: Ctx, sp: Screenplay) -> None:
    _round(ctx, sp, _specs(ctx, sp, "refs"), "refs")
    ctx.info("character and location references approved")
    _round(ctx, sp, _specs(ctx, sp, "keyframes"), "keyframes")
    ctx.info("all shot keyframes approved")
