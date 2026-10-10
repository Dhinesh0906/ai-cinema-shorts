"""Checklist-based visual inspection with an independent VLM (Qwen3-VL via MLX-VLM).

The VLM must describe what it sees BEFORE judging, answer every checklist item yes/no/unclear with
evidence, and is told explicitly to answer "unclear" rather than assume. Pass/fail is computed here
from mandatory items — the VLM never returns a bare "approved".
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field

from pydantic import ValidationError

from ..schemas import CheckItem, Screenplay, Shot, VisualVerdict


@dataclass
class Check:
    text: str
    mandatory: bool
    tag: str  # subject_count | character | costume | location | prop | lighting | framing | action | ...


@dataclass
class Checklist:
    checks: list[Check] = field(default_factory=list)

    def add(self, text: str, mandatory: bool, tag: str) -> None:
        self.checks.append(Check(text, mandatory, tag))


INSPECTOR = """You are a strict, independent visual QA inspector for a film production. You did not create these
images. Judge ONLY what is actually visible. If something cannot be clearly seen, answer "unclear" — never
assume. Do not be lenient: a wrong costume color, a missing prop, a different hairstyle, an extra person, or
the wrong setting is a "no"."""


def char_line(sp: Screenplay, cid: str) -> str:
    c = sp.character(cid)
    return (f"{c.name}: {c.age}-year-old {c.gender_presentation}, {c.ethnicity_and_skin}; hair: {c.hair}; "
            f"costume: {c.costume}")


def image_checklist(sp: Screenplay, shot: Shot | None, *, character_id: str | None = None,
                    location_id: str | None = None) -> Checklist:
    cl = Checklist()
    if character_id:  # character reference portrait
        c = sp.character(character_id)
        cl.add("Exactly one person is visible", True, "subject_count")
        cl.add(f"The person looks about {c.age} years old and presents as {c.gender_presentation}", True, "character")
        cl.add(f"Skin/ethnicity matches: {c.ethnicity_and_skin}", False, "character")
        cl.add(f"Hair matches: {c.hair}", True, "character")
        cl.add(f"Costume matches: {c.costume}", True, "costume")
        if c.accessories and c.accessories.lower() not in ("none", "n/a", ""):
            cl.add(f"Accessories visible: {c.accessories}", False, "prop")
        cl.add(f"Face matches: {c.face}", False, "character")
    elif location_id:  # location plate
        loc = sp.location(location_id)
        cl.add("No people are visible", True, "subject_count")
        cl.add(f"The setting is clearly: {loc.description}", True, "location")
        cl.add(f"Lighting matches: {loc.lighting}", False, "lighting")
        for p in loc.key_props[:4]:
            cl.add(f"Visible prop: {p}", False, "prop")
    else:
        assert shot is not None
        loc = sp.location(shot.location_id)
        n = shot.subject_count
        cl.add(f"Exactly {n} {'person is' if n == 1 else 'people are'} visible" if n else "No people are visible",
               True, "subject_count")
        for cid in shot.character_ids:
            c = sp.character(cid)
            cl.add(f"{c.name} is present and matches the reference portrait (same face shape, hair: {c.hair})",
                   True, "character")
            cl.add(f"{c.name} wears: {c.costume}", True, "costume")
        cl.add(f"The setting is: {loc.name} — {loc.description}", True, "location")
        for m in shot.must_show:
            cl.add(m, False, "must_show")
        for p in shot.props[:3]:
            cl.add(f"Visible prop: {p}", False, "prop")
        cl.add(f"Framing is roughly: {shot.framing}", False, "framing")
        cl.add(f"Lighting/mood matches: {shot.lighting}", False, "lighting")
    cl.add("No overlaid captions, subtitles, watermark or logo (writing on in-world objects such as books or "
           "signs is acceptable)", True, "artifact")
    cl.add("No severe anatomical defects (extra/missing limbs or fingers, melted or duplicated faces)", True,
           "artifact")
    return cl


def video_checklist(sp: Screenplay, shot: Shot) -> Checklist:
    loc = sp.location(shot.location_id)
    cl = Checklist()
    cl.add(f"Across the frames the shot depicts: {shot.action}", True, "action")
    for i, ev in enumerate(shot.expected_action_sequence, 1):
        cl.add(f"Event {i} visibly happens: {ev}", False, "action")
    if len(shot.expected_action_sequence) > 1:
        cl.add("The events happen in the listed order (compare early vs late frames)", False, "temporal")
    for cid in shot.character_ids:
        c = sp.character(cid)
        cl.add(f"{c.name} keeps the same face, hair and costume in every frame (no identity change)", True,
               "continuity")
        cl.add(f"{c.name} wears: {c.costume}", True, "costume")
    n = shot.subject_count
    cl.add(f"The number of visible people stays {n} (nobody appears or vanishes)" if n else
           "No people appear", True, "subject_count")
    if shot.character_ids:
        who = "; ".join(f"{sp.character(c).name} wears {sp.character(c).costume}" for c in shot.character_ids)
        cl.add(f"Every visible arm, hand, sleeve or body part (including at the frame edges and in the "
               f"foreground) belongs to a listed character and matches their costume ({who}); a limb or "
               f"sleeve in any other color counts as NO", True, "stray_limb")
    cl.add(f"The setting stays the same place as the approved first frame ({loc.name}); no scene change", True,
           "location")
    cl.add(f"Camera behaviour roughly matches: {shot.camera_motion}", False, "camera")
    if shot.environment_motion.strip():
        cl.add(f"Environmental motion visible: {shot.environment_motion}", False, "environment")
    cl.add("No severe morphing, melting, warping or flicker of people or objects between frames", True, "artifact")
    cl.add("Motion is physically plausible (no limbs bending impossibly, objects teleporting)", True, "artifact")
    cl.add("No overlaid captions, watermark or logo", True, "artifact")
    return cl


def build_prompt(checklist: Checklist, context: str, image_roles: list[str]) -> str:
    roles = "\n".join(f"Image {i+1}: {r}" for i, r in enumerate(image_roles))
    items = "\n".join(f"{i+1}. {c.text}" for i, c in enumerate(checklist.checks))
    schema = ('{"description": "<what you actually see, 2-5 sentences, chronological for video frames>", '
              '"checks": [{"item": "<item number and text>", "verdict": "yes|no|unclear", "evidence": "<short>"}], '
              '"defects": ["<visible defects>"], "correction": "<concrete prompt change that would fix failures>"}')
    return f"""{INSPECTOR}

Images provided:
{roles}

Context:
{context}

First describe what you see. Then answer EVERY checklist item, in order, for the image(s) under inspection:
{items}

Reply with ONE JSON object and nothing else, using this exact shape:
{schema}"""


def parse_verdict(text: str, checklist: Checklist) -> VisualVerdict:
    """Parse VLM JSON leniently; missing items count as 'unclear' (never as 'yes')."""
    m = re.search(r"\{.*\}", text, re.S)
    if not m:
        raise ValueError("VLM returned no JSON")
    raw = m.group(0)
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        data = json.loads(re.sub(r",\s*([}\]])", r"\1", raw))  # trailing commas
    checks_in = data.get("checks", []) or []
    aligned: list[CheckItem] = []
    for i, c in enumerate(checklist.checks):
        got = checks_in[i] if i < len(checks_in) and isinstance(checks_in[i], dict) else {}
        verdict = str(got.get("verdict", "unclear")).strip().lower()
        if verdict not in ("yes", "no", "unclear"):
            verdict = "unclear"
        aligned.append(CheckItem(item=c.text, verdict=verdict, evidence=str(got.get("evidence", ""))[:300]))
    try:
        return VisualVerdict(description=str(data.get("description", ""))[:2000], checks=aligned,
                             defects=[str(d)[:200] for d in data.get("defects", []) or []][:10],
                             correction=str(data.get("correction", ""))[:1000])
    except ValidationError as exc:
        raise ValueError(str(exc)) from exc


@dataclass
class Scored:
    passed: bool
    score: float
    mandatory_failed: list[str]
    optional_ratio: float
    verdict: VisualVerdict


def score(verdict: VisualVerdict, checklist: Checklist, optional_ratio_min: float,
          unclear_mandatory_passes: bool = False) -> Scored:
    mand_fail, opt_yes, opt_n, yes = [], 0, 0, 0
    for c, v in zip(checklist.checks, verdict.checks):
        ok = v.verdict == "yes" or (unclear_mandatory_passes and c.mandatory and v.verdict == "unclear")
        yes += v.verdict == "yes"
        if c.mandatory and not ok:
            mand_fail.append(f"{c.text} -> {v.verdict}: {v.evidence}")
        elif not c.mandatory:
            opt_n += 1
            opt_yes += v.verdict == "yes"
    ratio = opt_yes / opt_n if opt_n else 1.0
    total = len(checklist.checks) or 1
    return Scored(passed=not mand_fail and ratio >= optional_ratio_min, score=round(yes / total, 3),
                  mandatory_failed=mand_fail, optional_ratio=round(ratio, 3), verdict=verdict)
