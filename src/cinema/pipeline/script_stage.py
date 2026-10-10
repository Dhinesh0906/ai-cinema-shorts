"""STORY GENERATION -> deterministic checks -> INDEPENDENT critic -> targeted rewrite -> APPROVED."""
from __future__ import annotations

import json
import uuid

from ..schemas import Decision, Finding, Screenplay
from ..state_machine import AssetState, JobState
from ..storage.files import atomic_json, read_json
from ..story.generator import revise, write_shots, write_story
from ..story.validator import critic_review, deterministic_checks, mandatory_failures
from .context import Ctx, StageBlocked


def normalize(ctx: Ctx, sp: Screenplay) -> Screenplay:
    """Deterministic repairs that never need an LLM: sequential ids and exact duration total."""
    st = sp.story.model_copy(deep=True)
    st.narrator_voice_id = st.narrator_voice_id.strip().lower()
    for c in st.characters:
        c.voice_id = c.voice_id.strip().lower()
        c.id = c.id.strip()
    for loc in st.locations:
        loc.id = loc.id.strip()
    sp = sp.model_copy(update={"story": st})
    shots = []
    for i, s in enumerate(sp.shots, 1):
        shots.append(s.model_copy(update={"id": f"shot_{i:02d}", "location_id": s.location_id.strip(),
                                          "character_ids": [c.strip() for c in s.character_ids]}))
    lo, hi = ctx.cfg.shot_min_seconds, ctx.cfg.shot_max_seconds
    durs = [min(hi, max(lo, s.duration)) for s in shots]
    for _ in range(20):  # scale toward target while respecting per-shot bounds
        total = sum(durs)
        if abs(total - sp.target_seconds) < 0.05:
            break
        free = [i for i, d in enumerate(durs) if (lo < d < hi) or (total < sp.target_seconds and d <= lo)
                or (total > sp.target_seconds and d >= hi)] or list(range(len(durs)))
        delta = (sp.target_seconds - total) / len(free)
        for i in free:
            durs[i] = min(hi, max(lo, durs[i] + delta))
    durs = [round(d, 1) for d in durs]
    if durs:
        durs[-1] = round(durs[-1] + sp.target_seconds - sum(durs), 2)
    shots = [s.model_copy(update={"duration": d}) for s, d in zip(shots, durs)]
    return sp.model_copy(update={"shots": shots})


def _avoid(ctx: Ctx) -> list[str]:
    """Finished-film memory (state/story_history.jsonl) plus other jobs' current scripts."""
    from ..story.history import avoid_list
    out = avoid_list(ctx.cfg)
    seen = {o.split(" — ")[0] for o in out}
    for j in ctx.db.jobs():
        p = ctx.cfg.root / "projects" / j["id"] / "script" / "screenplay.json"
        if j["id"] != ctx.job_id and p.exists():
            try:
                st = read_json(p)["story"]
                if st["title"] not in seen:
                    out.append(f"{st['title']} — {st['logline']} | twist: {st.get('twist', '')}")
            except (KeyError, ValueError):
                pass
    return out


def _log_findings(ctx: Ctx, findings: list[Finding], evaluator: str, attempt: int) -> None:
    for f in findings:
        ctx.decide(Decision(asset_id="screenplay", stage="script", criterion=f.criterion, evaluator=evaluator,
                            passed=f.result == "pass", mandatory=f.severity in ("critical", "major"),
                            score=f.confidence, evidence=f.evidence,
                            action=f.recommended_correction if f.result == "fail" else "", attempt=attempt))


def run(ctx: Ctx) -> Screenplay:
    ctx.ensure("screenplay", "script", [])
    if ctx.passed("screenplay"):
        return ctx.screenplay()
    cfg, d = ctx.cfg, ctx.dirs
    draft_path = d.script / "draft.json"
    sp: Screenplay | None = Screenplay.model_validate(read_json(draft_path)) if draft_path.exists() else None
    findings: list[Finding] = []
    if (d.script / "pending_findings.json").exists():
        findings = [Finding.model_validate(f) for f in read_json(d.script / "pending_findings.json")]
    a = ctx.asset("screenplay")
    attempt = a["attempts"]
    while attempt < cfg.retries.story_rewrites:
        attempt += 1
        ctx.state(JobState.GENERATING_SCRIPT)
        ctx.to("screenplay", AssetState.GENERATING, attempts_inc=True)
        seed = ctx.seed + attempt * 101
        with ctx.rm.ollama_session("writer", cfg.models.writer, ctx.job_id) as sess:
            if sp is None:
                genre = ctx.job["genre"]
                story = write_story(cfg, genre, ctx.target, _avoid(ctx), sess, seed,
                                    brief=ctx.job["params"].get("brief"))
                ctx.info(f"story drafted: {story.title}", genre=genre)
                plan = write_shots(cfg, story, ctx.target, sess, seed + 1)
                sp = Screenplay(story_id=ctx.job_id, target_seconds=ctx.target, story=story, shots=plan.shots)
            elif findings:
                ctx.info(f"rewriting script for {len(findings)} findings (attempt {attempt})")
                sp = revise(cfg, sp, findings, sess, seed)
            sp = normalize(ctx, sp)
            det = deterministic_checks(cfg, sp)
            fixes = 0
            while mandatory_failures(det) and fixes < 2:  # cheap in-session repair before using a critic pass
                fixes += 1
                sp = normalize(ctx, revise(cfg, sp, mandatory_failures(det), sess, seed + 7 * fixes))
                det = deterministic_checks(cfg, sp)
        atomic_json(draft_path, sp)
        atomic_json(d.script / f"draft_attempt{attempt}.json", sp)
        _log_findings(ctx, det, "deterministic", attempt)
        if mandatory_failures(det):
            findings = mandatory_failures(det)
            atomic_json(d.script / "pending_findings.json", findings)
            ctx.save_report(f"script_attempt{attempt}", {"deterministic": det, "critic": None})
            ctx.to("screenplay", AssetState.RETRY_PENDING, feedback="; ".join(f.evidence for f in findings)[:2000])
            continue

        ctx.state(JobState.VALIDATING_SCRIPT)
        ctx.to("screenplay", AssetState.VALIDATING)
        with ctx.rm.ollama_session("critic", cfg.models.critic, ctx.job_id) as sess:
            report = critic_review(cfg, sp, sess, seed)
        _log_findings(ctx, report.findings, cfg.models.critic, attempt)
        mand = mandatory_failures(report.findings)
        consistent = report.overall_pass == (not mand)
        ctx.save_report(f"script_attempt{attempt}", {"deterministic": det, "critic": report.model_dump(),
                                                      "critic_self_consistent": consistent})
        if not consistent:
            ctx.warn("critic overall_pass disagrees with its own findings; findings take precedence")
        if not mand:
            sp = sp.model_copy(update={"story_id": ctx.job_id})
            atomic_json(d.script / "screenplay.json", sp)
            (d.script / "pending_findings.json").unlink(missing_ok=True)
            ctx.approve("screenplay", d.script / "screenplay.json", d.script / "screenplay.json",
                        meta={"title": sp.story.title, "attempts": attempt, "critic": cfg.models.critic})
            ctx.info(f"script approved: {sp.story.title}")
            return sp
        findings = mand
        atomic_json(d.script / "pending_findings.json", findings)
        ctx.to("screenplay", AssetState.RETRY_PENDING,
               feedback=json.dumps([f.criterion + ": " + f.evidence for f in mand])[:2000])
    ctx.set("screenplay", AssetState.BLOCKED)
    raise StageBlocked(f"script failed mandatory checks after {attempt} attempts: "
                       + "; ".join(f"{f.criterion}: {f.evidence}" for f in findings)[:1500])


def new_story_id() -> str:
    return uuid.uuid4().hex[:8]
