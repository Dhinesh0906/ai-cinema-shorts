"""Wan2.2 TI2V-5B first-frame image-to-video prompts. The approved keyframe is the first frame, which
is what carries character identity into the moving shot."""
from __future__ import annotations

from ..schemas import Screenplay, Shot

BASE_NEGATIVE = ("distorted face, morphing face, identity change, extra limbs, extra fingers, duplicated person, "
                 "melting, warping, flicker, jitter, oversaturated, rainbow artifacts, cartoon, anime, CGI, blurry, "
                 "low quality, text, subtitles, watermark, logo, static image, frozen frame")


def motion_prompt(sp: Screenplay, shot: Shot, style: str, correction: str = "") -> str:
    who = ", ".join(f"{sp.character(c).name} ({sp.character(c).costume})" for c in shot.character_ids)
    seq = "; then ".join(shot.expected_action_sequence)
    return (f"{shot.motion_prompt}. {('Characters: ' + who + '. ') if who else ''}"
            f"Action: {seq}. Camera: {shot.camera_motion}. Environment: {shot.environment_motion}. "
            f"Lighting: {shot.lighting}. Natural, smooth, physically plausible motion; consistent identity; "
            f"{style}" + (f". Correction: {correction}" if correction else ""))


def negative_prompt(extra: str = "") -> str:
    return BASE_NEGATIVE + (", " + extra if extra else "")
