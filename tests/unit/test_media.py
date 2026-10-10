"""Real FFmpeg/OpenCV on synthetic media: motion analysis, EDL resolution, rendering, captions, speech diff."""
import pytest

from cinema.audio.analysis import measure
from cinema.audio.speech_validator import compare
from cinema.editing.edl import EDLError, ShotMedia, SoundMedia, SpeechMedia, chunk_captions, default_plan, resolve
from cinema.editing.ffmpeg_renderer import render
from cinema.fixtures import good
from cinema.model_workers import wan_frames
from cinema.video.motion_analysis import analyze, stream_info
from tests.conftest import make_clip, make_tone


def test_wan_frames_are_4n_plus_1(cfg):
    p = cfg.presets["balanced"]
    for d in (2.5, 3.3, 4.0, 5.0):
        n = wan_frames(d, p)
        assert (n - 1) % 4 == 0 and n <= p.max_frames and (n >= d * p.fps or n == p.max_frames)


def test_motion_analysis_detects_frozen_black_and_passes_motion(cfg, tmp_path):
    moving = analyze(make_clip(tmp_path / "m.mp4", "moving"), thresholds=cfg.thresholds)
    still = analyze(make_clip(tmp_path / "s.mp4", "still"), thresholds=cfg.thresholds)
    black = analyze(make_clip(tmp_path / "b.mp4", "black"), thresholds=cfg.thresholds)
    assert moving["passed"], moving["problems"]
    assert not still["passed"] and any("static" in p or "frozen" in p for p in still["problems"])
    assert not black["passed"] and any("black" in p for p in black["problems"])


def test_broken_file_is_unreadable(cfg, tmp_path):
    clip = make_clip(tmp_path / "m.mp4")
    bad = tmp_path / "bad.mp4"
    bad.write_bytes(clip.read_bytes()[:2000])
    assert not analyze(bad, thresholds=cfg.thresholds)["passed"]


def test_speech_compare():
    ok = compare("Forty years. Tonight the sea answered back.", "40 years, tonight the sea answered back")
    assert ok["wer"] == 0
    bad = compare("Forty years. Tonight the sea answered back.", "Forty years tonight")
    assert bad["wer"] > 0.4 and "answered" in bad["missing_content_words"]


def _edl_inputs(tmp_path, n=3):
    sp = good()
    sp = sp.model_copy(update={"shots": sp.shots[:n], "target_seconds": 12})
    media = {s.id: ShotMedia(s.id, str(make_clip(tmp_path / f"{s.id}.mp4", seconds=4.3)), 4.3) for s in sp.shots}
    speech = [SpeechMedia(f"narr_{s.id}", s.id, "narration", str(make_tone(tmp_path / f"n{i}.wav", 1.6)), 1.6,
                          [{"word": "hello", "start": 0.1, "end": 0.6}, {"word": "there", "start": 0.7, "end": 1.4}],
                          "hello there") for i, s in enumerate(sp.shots)]
    sounds = [SoundMedia("music", "music", str(make_tone(tmp_path / "mus.wav", 14, 220, 0.2)), 14.0),
              SoundMedia("amb_loc_lamp_room", "ambience", str(make_tone(tmp_path / "amb.wav", 14, 120, 0.2)), 14.0,
                         location_id="loc_lamp_room")]
    return sp, media, speech, sounds


def test_edl_resolve_and_render(cfg, tmp_path):
    sp, media, speech, sounds = _edl_inputs(tmp_path)
    plan = default_plan(sp, media, 12)
    plan.shots[1].transition_in = "crossfade"
    plan.shots[1].transition_seconds = 0.4
    edl = resolve("j", plan, sp, media, speech, sounds, "deterministic_fallback", 9, 15)
    assert abs(edl.duration - (3 * 4.3 - 0.4)) < 0.01
    starts = [a.start for a in edl.audio if a.kind == "narration"]
    assert starts == sorted(starts) and all(b - a >= 1.6 for a, b in zip(starts, starts[1:]))
    assert edl.captions and edl.captions[0].text == "hello there"
    out = tmp_path / "final.mp4"
    render(edl, out, tmp_path / "work", tmp_path / "render.log", burn_captions=True)
    info = stream_info(out)
    assert (info["width"], info["height"], info["codec"], info["audio_codec"]) == (1080, 1920, "h264", "aac")
    assert abs(info["duration"] - edl.duration) < 0.15
    m = measure(out)
    assert m["lufs"] is not None and abs(m["lufs"] - cfg.thresholds.target_lufs) < 2.5


def test_edl_rejects_out_of_order_and_overlong(tmp_path):
    sp, media, speech, sounds = _edl_inputs(tmp_path)
    plan = default_plan(sp, media, 12)
    plan.shots = plan.shots[::-1]
    with pytest.raises(EDLError):
        resolve("j", plan, sp, media, speech, sounds, "ai_director", 9, 15)
    plan = default_plan(sp, media, 12)
    plan.shots[0].out_point = 9.0  # beyond clip
    with pytest.raises(EDLError):
        resolve("j", plan, sp, media, speech, sounds, "ai_director", 9, 15)


def test_caption_chunks_follow_words():
    sm = SpeechMedia("a", "s", "narration", "x", 3.0, [{"word": w, "start": i * 0.4, "end": i * 0.4 + 0.3}
                                                       for i, w in enumerate("one two three four five six".split())], "")
    caps = chunk_captions(sm, 10.0)
    assert [c.text for c in caps] == ["one two three", "four five six"]
    assert caps[1].words[0].word == "four" and caps[1].words[0].start == 11.2
    assert caps[0].start == 10.0 and caps[0].end <= caps[1].start + 1e-6


def test_shorts_caption_track(tmp_path):
    from PIL import Image
    from cinema.editing.captions import CaptionStyle, build_track, draw_caption
    from cinema.schemas import Caption, CaptionWord
    im = draw_caption(["the", "mirror", "lies"], 1, 1080, 1920, CaptionStyle())
    bbox = im.getbbox()
    assert bbox and 0.6 * 1920 < (bbox[1] + bbox[3]) / 2 < 0.8 * 1920          # lower-middle band
    yellow = sum(1 for p in im.getdata() if p[0] > 240 and p[1] > 200 and p[2] < 40 and p[3] > 200)
    assert yellow > 500                                                          # active word highlighted
    caps = [Caption(start=0.5, end=1.6, text="the mirror lies",
                    words=[CaptionWord(word=w, start=0.5 + i * 0.35, end=0.8 + i * 0.35)
                           for i, w in enumerate(["the", "mirror", "lies"])])]
    out = build_track(caps, 3.0, tmp_path / "c.mov")
    info = stream_info(out)
    assert (info["width"], info["height"]) == (1080, 1920) and abs(info["duration"] - 3.0) < 0.1
