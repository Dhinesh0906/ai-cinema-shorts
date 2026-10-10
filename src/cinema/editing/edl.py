"""Resolve an EditPlan (AI director) into a concrete, validated EDL with timeline positions for every
video segment, audio event and caption."""
from __future__ import annotations

from dataclasses import dataclass

from ..schemas import EDL, AudioEvent, Caption, CaptionWord, EditPlan, EditShot, Screenplay

XFADE = {"crossfade", "dip_to_black"}


@dataclass
class ShotMedia:
    shot_id: str
    clip: str
    clip_duration: float


@dataclass
class SpeechMedia:
    asset_id: str
    shot_id: str
    kind: str
    path: str
    duration: float
    words: list[dict]          # whisper words relative to the file start
    text: str


@dataclass
class SoundMedia:
    asset_id: str
    kind: str                  # sfx | ambience | music
    path: str
    duration: float
    shot_id: str | None = None
    location_id: str | None = None


class EDLError(ValueError):
    pass


def validate_plan(plan: EditPlan, sp: Screenplay, media: dict[str, ShotMedia], min_len: float = 1.5) -> list[str]:
    errs = []
    ids = [s.shot_id for s in plan.shots]
    want = [s.id for s in sp.shots]
    if ids != want:
        errs.append(f"shot order/completeness {ids} != screenplay {want}")
    for es in plan.shots:
        m = media.get(es.shot_id)
        if m is None:
            continue
        if not (0 <= es.in_point < es.out_point <= m.clip_duration + 1e-3):
            errs.append(f"{es.shot_id}: in/out {es.in_point}-{es.out_point} outside clip 0-{m.clip_duration:.2f}")
        if es.out_point - es.in_point < min_len:
            errs.append(f"{es.shot_id}: segment shorter than {min_len}s")
        if es.transition_in in XFADE and es.transition_seconds > (es.out_point - es.in_point) / 3:
            errs.append(f"{es.shot_id}: transition too long for segment")
    if plan.shots and plan.shots[0].transition_in in XFADE:
        errs.append("first shot cannot crossfade from nothing")
    return errs


def resolve(job_id: str, plan: EditPlan, sp: Screenplay, media: dict[str, ShotMedia], speech: list[SpeechMedia],
            sounds: list[SoundMedia], source: str, min_total: float, max_total: float) -> EDL:
    errs = validate_plan(plan, sp, media)
    if errs:
        raise EDLError("; ".join(errs))
    # ---- video timeline
    video, t = [], 0.0
    shot_span: dict[str, tuple[float, float]] = {}
    for i, es in enumerate(plan.shots):
        seg = es.out_point - es.in_point
        overlap = es.transition_seconds if (i > 0 and es.transition_in in XFADE) else 0.0
        start = t - overlap
        video.append({"shot_id": es.shot_id, "path": media[es.shot_id].clip, "in": round(es.in_point, 3),
                      "out": round(es.out_point, 3), "timeline_start": round(start, 3),
                      "transition_in": es.transition_in,
                      "transition_seconds": round(es.transition_seconds if es.transition_in != "cut" else 0.0, 3),
                      "overlap": round(overlap, 3)})
        shot_span[es.shot_id] = (start, start + seg)
        t = start + seg
    total = round(t, 3)
    if not (min_total - 1e-6 <= total <= max_total + 1e-6):
        raise EDLError(f"edited duration {total}s outside {min_total}-{max_total}s")

    # ---- speech: sequential, never overlapping, anchored to its shot
    audio: list[AudioEvent] = []
    captions: list[Caption] = []
    cursor = 0.0
    for sm in speech:
        s0, s1 = shot_span[sm.shot_id]
        start = max(s0 + plan.narration_lead_seconds, cursor + 0.15)
        if start - s0 > 1.2 and start > s1 - 0.5:
            raise EDLError(f"{sm.asset_id} cannot start inside its shot (previous line runs too long)")
        end = start + sm.duration
        if end > total - 0.15:
            raise EDLError(f"{sm.asset_id} ends at {end:.2f}s after the film ends ({total}s)")
        audio.append(AudioEvent(kind=sm.kind, asset_id=sm.asset_id, path=sm.path, start=round(start, 3),
                                duration=round(sm.duration, 3), gain_db=0.0))
        captions.extend(chunk_captions(sm, start))
        cursor = end
    # ---- sound design
    for snd in sounds:
        if snd.kind == "music":
            audio.append(AudioEvent(kind="music", asset_id=snd.asset_id, path=snd.path, start=0.0,
                                    duration=round(min(snd.duration, total), 3), gain_db=plan.music_gain_db,
                                    fade_in=1.0, fade_out=min(2.5, total / 6)))
        elif snd.kind == "ambience":
            runs = _location_runs(sp, shot_span, snd.location_id)
            for k, (a, b) in enumerate(runs):
                audio.append(AudioEvent(kind="ambience", asset_id=f"{snd.asset_id}#{k}", path=snd.path, start=round(a, 3),
                                        duration=round(min(b - a, snd.duration), 3), gain_db=plan.ambience_gain_db,
                                        fade_in=0.4, fade_out=0.4))
        elif snd.kind == "sfx" and snd.shot_id in shot_span:
            a, b = shot_span[snd.shot_id]
            audio.append(AudioEvent(kind="sfx", asset_id=snd.asset_id, path=snd.path, start=round(a + 0.2, 3),
                                    duration=round(min(snd.duration, b - a - 0.2), 3), gain_db=plan.sfx_gain_db,
                                    fade_in=0.05, fade_out=0.3))
    return EDL(job_id=job_id, grade=plan.grade, video=video, audio=audio, captions=captions, duration=total,
               fade_out=1.0 if plan.end_with_fade else 0.0, source=source, rationale=plan.rationale)


def _location_runs(sp: Screenplay, span: dict[str, tuple[float, float]], loc_id: str | None) -> list[tuple[float, float]]:
    runs: list[list[float]] = []
    for s in sp.shots:
        if s.location_id != loc_id or s.id not in span:
            continue
        a, b = span[s.id]
        if runs and abs(runs[-1][1] - a) < 1.0:
            runs[-1][1] = b
        else:
            runs.append([a, b])
    return [(a, b) for a, b in runs]


def chunk_captions(sm: SpeechMedia, offset: float, max_words: int = 3, max_len: float = 1.4) -> list[Caption]:
    words = [w for w in sm.words if w.get("word")]
    if not words:  # even split fallback (should not happen: speech is validated by Whisper)
        toks = sm.text.split()
        n = max(1, (len(toks) + max_words - 1) // max_words)
        step = sm.duration / n
        return [Caption(start=round(offset + i * step, 3), end=round(offset + (i + 1) * step, 3),
                        text=" ".join(toks[i * max_words:(i + 1) * max_words])) for i in range(n)]
    caps, cur = [], []
    for w in words:
        if cur and (len(cur) >= max_words or w["end"] - cur[0]["start"] > max_len):
            caps.append(cur)
            cur = []
        cur.append(w)
    if cur:
        caps.append(cur)
    out = []
    for i, c in enumerate(caps):
        start = offset + c[0]["start"]
        end = offset + (caps[i + 1][0]["start"] if i + 1 < len(caps) else c[-1]["end"] + 0.25)
        out.append(Caption(start=round(start, 3), end=round(max(end, start + 0.4), 3),
                           text=" ".join(w["word"] for w in c).strip(),
                           words=[CaptionWord(word=w["word"].strip(), start=round(offset + w["start"], 3),
                                              end=round(offset + w["end"], 3)) for w in c]))
    return out


def default_plan(sp: Screenplay, media: dict[str, ShotMedia], target: float) -> EditPlan:
    """Deterministic fallback used only if the AI director fails validation after retries."""
    shots, used = [], 0.0
    for i, s in enumerate(sp.shots):
        d = min(s.duration, media[s.id].clip_duration)
        shots.append(EditShot(shot_id=s.id, in_point=0.0, out_point=round(d, 3),
                              transition_in="fade_from_black" if i == 0 else "cut",
                              transition_seconds=0.5 if i == 0 else 0.0))
        used += d
    return EditPlan(shots=shots, grade="neutral", music_gain_db=-20, ambience_gain_db=-22, sfx_gain_db=-8,
                    narration_lead_seconds=0.3, end_with_fade=True, rationale="deterministic fallback")
