"""Script validation: deterministic checks first, then an independent reasoning critic (GPT-OSS 20B)."""
from __future__ import annotations

import re

from ..config import Config
from ..llm import chat_json
from ..schemas import KOKORO_VOICES, CriticReport, Finding, Screenplay

BEAT_ORDER = ["hook", "beginning", "conflict", "escalation", "climax", "ending"]
ID_RE = re.compile(r"^[a-z][a-z0-9_]*$")


def _f(criterion: str, evidence: str, fix: str, affected: list[str], severity: str = "critical") -> Finding:
    return Finding(criterion=criterion, result="fail", severity=severity, evidence=evidence,
                   recommended_correction=fix, confidence=1.0, correction_required=severity != "minor",
                   affected=affected or ["story"])


def words(text: str) -> int:
    return len(re.findall(r"[A-Za-z0-9']+", text))


def deterministic_checks(cfg: Config, sp: Screenplay) -> list[Finding]:
    """Schema-level facts an LLM must never be trusted to check. Returns failures only."""
    out: list[Finding] = []
    st = sp.story
    chars = {c.id for c in st.characters}
    locs = {loc.id for loc in st.locations}
    shot_ids = [s.id for s in sp.shots]

    for kind, ids in (("character", [c.id for c in st.characters]), ("location", [x.id for x in st.locations]),
                      ("shot", shot_ids)):
        dupes = sorted({i for i in ids if ids.count(i) > 1})
        if dupes:
            out.append(_f(f"unique {kind} ids", f"duplicate ids {dupes}", "make ids unique", dupes))
        bad = [i for i in ids if not ID_RE.match(i)]
        if bad:
            out.append(_f(f"{kind} id format", f"invalid ids {bad}", "use snake_case ids", bad, "major"))

    total = round(sum(s.duration for s in sp.shots), 3)
    if abs(total - sp.target_seconds) > 0.5:
        out.append(_f("duration total", f"shots sum to {total}s, target {sp.target_seconds}s",
                      f"adjust durations to sum to {sp.target_seconds}", shot_ids))
    lo, hi = cfg.shot_min_seconds, cfg.shot_max_seconds
    for s in sp.shots:
        if not lo - 1e-6 <= s.duration <= hi + 1e-6:
            out.append(_f("shot duration range", f"{s.id} is {s.duration}s (allowed {lo}-{hi}s)",
                          f"set duration within {lo}-{hi}s", [s.id]))
    expected = [f"shot_{i:02d}" for i in range(1, len(sp.shots) + 1)]
    if shot_ids != expected:
        out.append(_f("timeline ordering", f"shot ids {shot_ids} are not sequential", f"use {expected}", shot_ids, "major"))
    beats = [BEAT_ORDER.index(s.beat) for s in sp.shots]
    if beats != sorted(beats):
        out.append(_f("beat ordering", f"beats out of order: {[s.beat for s in sp.shots]}",
                      "order shots hook -> ending", shot_ids))
    if sp.shots and sp.shots[0].beat != "hook":
        out.append(_f("opening hook", "first shot is not the hook", "make shot_01 the hook", [sp.shots[0].id]))
    from .generator import beat_plan
    plan = beat_plan(len(sp.shots))
    for beat in ("hook", "climax", "ending"):
        if beat not in {s.beat for s in sp.shots}:
            out.append(_f("required beats", f"no shot covers the {beat}",
                          f"use beats {dict(zip(shot_ids, plan))}", shot_ids))

    for s in sp.shots:
        missing = sorted(set(s.character_ids) - chars)
        if missing:
            out.append(_f("character ids", f"{s.id} references unknown characters {missing}",
                          f"use only {sorted(chars)}", [s.id]))
        if s.location_id not in locs:
            out.append(_f("location ids", f"{s.id} references unknown location {s.location_id}",
                          f"use one of {sorted(locs)}", [s.id]))
        if s.subject_count < len(set(s.character_ids)):
            out.append(_f("subject count", f"{s.id} lists {len(set(s.character_ids))} characters but "
                          f"subject_count={s.subject_count}", "make subject_count match visible people", [s.id], "major"))
        for d in s.dialogue:
            if d.speaker_id not in chars:
                out.append(_f("speaker ids", f"{s.id} dialogue speaker {d.speaker_id} unknown",
                              "use a story character id", [s.id]))
            if not d.text.strip():
                out.append(_f("empty dialogue", f"{s.id} has an empty dialogue line", "remove it", [s.id], "major"))
        spoken = words(s.narration) + sum(words(d.text) for d in s.dialogue)
        limit = cfg.narration_wps * s.duration + 1
        if spoken > limit:
            out.append(_f("speech fits shot", f"{s.id} has {spoken} spoken words for {s.duration}s (max {limit:.0f})",
                          "shorten narration/dialogue", [s.id]))
        for name, val in (("action", s.action), ("image_prompt", s.image_prompt), ("motion_prompt", s.motion_prompt),
                          ("framing", s.framing), ("camera_motion", s.camera_motion), ("lighting", s.lighting)):
            if len(val.strip()) < 3:
                out.append(_f("required fields", f"{s.id}.{name} is empty", f"fill {name}", [s.id]))
        if not s.must_show or any(not m.strip() for m in s.must_show):
            out.append(_f("validation requirements", f"{s.id} has empty must_show", "list visual facts", [s.id], "major"))

    used_chars = {c for s in sp.shots for c in s.character_ids} | {d.speaker_id for s in sp.shots for d in s.dialogue}
    for c in st.characters:
        if c.id not in used_chars:
            out.append(_f("unused character", f"{c.id} never appears or speaks", "use or remove the character",
                          [c.id], "major"))
        if c.voice_id not in KOKORO_VOICES:
            out.append(_f("voice id", f"{c.id} voice {c.voice_id} is not a Kokoro voice",
                          f"choose from {sorted(KOKORO_VOICES)}", [c.id], "major"))
    if st.narrator_voice_id not in KOKORO_VOICES:
        out.append(_f("voice id", f"narrator voice {st.narrator_voice_id} invalid", "choose a Kokoro voice",
                      ["story"], "major"))
    for loc in st.locations:
        if loc.id not in {s.location_id for s in sp.shots}:
            out.append(_f("unused location", f"{loc.id} never used", "use or remove it", [loc.id], "minor"))
    total_words = sum(words(s.narration) + sum(words(d.text) for d in s.dialogue) for s in sp.shots)
    if total_words < 0.8 * sp.target_seconds:
        out.append(_f("narration coverage", f"only {total_words} spoken words for {sp.target_seconds}s",
                      "add narration so the story is understandable", ["story"], "major"))
    return out


CRITIC_SYSTEM = """You are a strict, independent script editor reviewing a screenplay for a vertical AI-generated
cinematic short. You did NOT write it. Your job is to find real problems, not to praise. Approve only when the
script is genuinely coherent and filmable. Do not invent problems either: every failure must cite concrete
evidence from the script (quote it or name the shot id)."""

CRITERIA = [
    ("story coherence", "the plot makes sense from hook to ending; events follow causally"),
    ("plot contradictions", "no facts, times, objects, or character states contradict each other"),
    ("character continuity", "characters' look/costume/props and knowledge stay consistent across shots"),
    ("character motivation", "what characters do is motivated"),
    ("missing information", "a viewer can understand what happened from visuals + narration alone"),
    ("pacing and escalation", "tension rises; no shots that repeat the same beat"),
    ("hook strength", "the first shot grabs attention immediately"),
    ("ending", "the ending is satisfying, surprising or haunting and pays off the setup"),
    ("narration and dialogue quality", "lines are evocative, not expository filler; no on-screen lip-synced talk"),
    ("visual feasibility", "each shot is ONE simple visible action a 5-second image-to-video model can render; "
                           "no crowds, fights, fine hand work, or on-screen text"),
    ("narration-visual agreement", "each shot's narration matches what that shot shows"),
]


def critic_review(cfg: Config, sp: Screenplay, session: dict, seed: int) -> CriticReport:
    criteria = "\n".join(f"{i+1}. {name}: {desc}" for i, (name, desc) in enumerate(CRITERIA))
    user = f"""Screenplay JSON:
{sp.model_dump_json()}

Evaluate EVERY criterion below and output one finding per criterion (result pass or fail):
{criteria}

For each finding: severity critical (the film would not make sense or cannot be generated), major
(clearly hurts the film), or minor (polish). correction_required=true only for critical/major failures.
affected = the specific shot ids (e.g. ["shot_03"]) or ["story"].
overall_pass = true only if no critical or major failure exists. Return JSON only."""
    return chat_json(cfg, cfg.models.critic, CRITIC_SYSTEM, user, CriticReport, session=session,
                     think=cfg.models.critic_think, temperature=0.2, seed=seed, num_predict=20000)


def mandatory_failures(findings: list[Finding]) -> list[Finding]:
    return [f for f in findings if f.result == "fail" and f.severity in ("critical", "major")]
