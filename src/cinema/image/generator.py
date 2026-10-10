"""Character bible portraits, location plates, and per-shot keyframes with FLUX.2 Klein (MLX-Gen).

Prompts are composed from the structured bible, not free text, so every image of a character repeats
the same face/hair/costume words. Keyframes condition on the character portraits (+ location plate)
through FLUX.2 multi-reference editing.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from ..schemas import Screenplay, Shot

NO_TEXT = "No text, no letters, no watermark, no logo."


@dataclass
class ImageTask:
    asset_id: str
    kind: str              # char_ref | loc_ref | keyframe
    prompt: str
    width: int
    height: int
    references: list[Path]
    reference_roles: list[str]
    depends_on: list[str]


def character_prompt(sp: Screenplay, cid: str, style: str, correction: str = "") -> str:
    c = sp.character(cid)
    return (f"Cinematic character reference portrait, waist-up, three-quarter view facing camera, neutral "
            f"expression, plain dark grey studio backdrop, soft key light. A {c.age}-year-old "
            f"{c.gender_presentation}, {c.ethnicity_and_skin}. Face: {c.face}. Hair: {c.hair}. Build: {c.body}. "
            f"Wearing {c.costume}. Accessories: {c.accessories}. {style}. {NO_TEXT}"
            + (f" Important: {correction}" if correction else ""))


def location_prompt(sp: Screenplay, lid: str, style: str, correction: str = "") -> str:
    loc = sp.location(lid)
    return (f"Cinematic establishing shot of an empty location, no people. {loc.name}: {loc.description}. "
            f"Lighting: {loc.lighting}. Color palette: {loc.color_palette}. Visible: {', '.join(loc.key_props)}. "
            f"Atmosphere: {loc.atmosphere}. Vertical 9:16 composition. {style}. {NO_TEXT}"
            + (f" Important: {correction}" if correction else ""))


def keyframe_prompt(sp: Screenplay, shot: Shot, style: str, ref_roles: list[str], correction: str = "") -> str:
    loc = sp.location(shot.location_id)
    people = []
    for cid in shot.character_ids:
        c = sp.character(cid)
        idx = next((i + 1 for i, r in enumerate(ref_roles) if r.startswith(f"character {cid}")), None)
        ref = f" (the same person as reference image {idx})" if idx else ""
        people.append(f"{c.name}{ref}: {c.age}-year-old {c.gender_presentation}, hair {c.hair}, wearing {c.costume}")
    loc_idx = next((i + 1 for i, r in enumerate(ref_roles) if r.startswith("location")), None)
    count = ("no people" if shot.subject_count == 0 else
             "exactly one person" if shot.subject_count == 1 else f"exactly {shot.subject_count} people")
    return (f"Cinematic film still, vertical 9:16. {shot.framing}. {shot.image_prompt}. "
            f"{'People: ' + '; '.join(people) + '. ' if people else ''}Show {count}. "
            f"Setting: {loc.name}{f' (the place from reference image {loc_idx})' if loc_idx else ''} — "
            f"{loc.description}. Props: {', '.join(shot.props) or 'none'}. Lighting: {shot.lighting}. "
            f"{style}. {NO_TEXT}" + (f" Important correction: {correction}" if correction else ""))


def keyframe_refs(sp: Screenplay, shot: Shot, char_paths: dict[str, Path], loc_paths: dict[str, Path]
                  ) -> tuple[list[Path], list[str]]:
    refs, roles = [], []
    for cid in shot.character_ids:
        if cid in char_paths:
            refs.append(char_paths[cid])
            roles.append(f"character {cid} ({sp.character(cid).name}) reference portrait")
    if shot.location_id in loc_paths:
        refs.append(loc_paths[shot.location_id])
        roles.append(f"location {shot.location_id} reference plate")
    return refs, roles
