"""Deterministic FFmpeg rendering of a resolved EDL to H.264/AAC 1080x1920.

Captions are a Shorts-style transparent track built by editing/captions.py and overlaid once. Native generation resolution is upscaled (lanczos) for export; the
render metadata records both sizes — upscaling adds no native detail.
"""
from __future__ import annotations

import subprocess
from pathlib import Path

from ..schemas import EDL

GRADES = {
    "neutral": "eq=contrast=1.03:saturation=1.0",
    "teal_orange": "colorbalance=rs=0.04:bs=-0.04:rh=0.05:bh=-0.06:rm=0.02:bm=-0.02,eq=contrast=1.06:saturation=1.08",
    "cold_desaturated": "colorbalance=bs=0.05:bm=0.03:rs=-0.02,eq=contrast=1.05:saturation=0.78",
    "warm_vintage": "colorbalance=rs=0.05:rm=0.04:bs=-0.04,eq=contrast=0.98:saturation=0.9:gamma=1.03",
    "noir_contrast": "eq=contrast=1.18:saturation=0.55:brightness=-0.02",
    "moonlit_blue": "colorbalance=bs=0.07:bm=0.04:bh=0.02:rs=-0.03,eq=contrast=1.08:saturation=0.85:brightness=-0.02",
}
def build_command(edl: EDL, out: Path, workdir: Path, burn_captions: bool = True,
                  target_lufs: float = -14.0) -> list[str]:
    W, H, fps = edl.width, edl.height, edl.fps
    args = ["ffmpeg", "-y", "-hide_banner"]
    filters: list[str] = []
    n = len(edl.video)
    for v in edl.video:
        args += ["-i", v["path"]]
    grade = GRADES[edl.grade]
    for i, v in enumerate(edl.video):
        seg = v["out"] - v["in"]
        chain = (f"[{i}:v]trim=start={v['in']}:end={v['out']},setpts=PTS-STARTPTS,fps={fps},"
                 f"scale={W}:{H}:force_original_aspect_ratio=increase:flags=lanczos,crop={W}:{H},setsar=1,"
                 f"{grade},format=yuv420p,settb=1/{fps}")
        if v["transition_in"] == "fade_from_black" and v["transition_seconds"] > 0:
            chain += f",fade=t=in:st=0:d={v['transition_seconds']}"
        nxt = edl.video[i + 1] if i + 1 < n else None
        if nxt and nxt["transition_in"] == "fade_to_black" and nxt["transition_seconds"] > 0:
            chain += f",fade=t=out:st={seg - nxt['transition_seconds']:.3f}:d={nxt['transition_seconds']}"
            # the next shot fades back in from black
        if nxt is None and edl.fade_out > 0:
            chain += f",fade=t=out:st={max(0, seg - edl.fade_out):.3f}:d={edl.fade_out}"
        if i > 0 and v["transition_in"] == "fade_to_black" and v["transition_seconds"] > 0:
            chain += f",fade=t=in:st=0:d={v['transition_seconds']}"
        filters.append(chain + f"[v{i}]")
    # fold segments with cuts (concat) or xfades
    cur, cur_len = "[v0]", edl.video[0]["out"] - edl.video[0]["in"]
    for i in range(1, n):
        v = edl.video[i]
        seg = v["out"] - v["in"]
        label = f"[x{i}]"
        if v["overlap"] > 0:
            kind = "fadeblack" if v["transition_in"] == "dip_to_black" else "fade"
            filters.append(f"{cur}[v{i}]xfade=transition={kind}:duration={v['overlap']}:"
                           f"offset={cur_len - v['overlap']:.3f}{label}")
            cur_len = cur_len - v["overlap"] + seg
        else:
            # concat outputs a 1/1000000 timebase; xfade needs both inputs on the same one
            filters.append(f"{cur}[v{i}]concat=n=2:v=1:a=0,settb=1/{fps}{label}")
            cur_len += seg
        cur = label
    vlabel = cur
    # captions: one transparent Shorts-style track (word highlight, pop-in), overlaid once
    idx = n
    if burn_captions and edl.captions:
        from .captions import build_track
        track = build_track(edl.captions, edl.duration, workdir / "captions.mov", W, H, fps)
        args += ["-i", str(track)]
        filters.append(f"{vlabel}[{idx}:v]overlay=0:0:format=auto:eof_action=pass[vcap]")
        vlabel, idx = "[vcap]", idx + 1
    filters.append(f"{vlabel}trim=duration={edl.duration:.3f},setpts=PTS-STARTPTS[vout]")
    # audio
    speech, beds = [], []
    for a in edl.audio:
        args += ["-i", a.path]
        ms = int(round(a.start * 1000))
        ch = (f"[{idx}:a]aresample=48000,aformat=channel_layouts=stereo,atrim=duration={a.duration:.3f},"
              f"asetpts=PTS-STARTPTS,volume={a.gain_db}dB")
        if a.fade_in:
            ch += f",afade=t=in:st=0:d={a.fade_in}"
        if a.fade_out:
            ch += f",afade=t=out:st={max(0, a.duration - a.fade_out):.3f}:d={a.fade_out}"
        ch += f",adelay={ms}|{ms}"
        lab = f"[a{idx}]"
        filters.append(ch + lab)
        (speech if a.kind in ("narration", "dialogue") else beds).append((a.kind, lab))
        idx += 1
    dur = f"{edl.duration:.3f}"
    if speech:
        filters.append("".join(l for _, l in speech) + f"amix=inputs={len(speech)}:normalize=0:duration=longest,"
                       f"apad=whole_dur={dur}[speech]")
    music = [l for k, l in beds if k == "music"]
    other = [l for k, l in beds if k != "music"]
    mix_inputs = []
    if speech:
        filters.append("[speech]asplit=2[sp_mix][sp_key]")
        mix_inputs.append("[sp_mix]")
    if other:
        filters.append("".join(other) + f"amix=inputs={len(other)}:normalize=0:duration=longest,apad=whole_dur={dur}[fx]")
        mix_inputs.append("[fx]")
    if music:
        filters.append(f"{music[0]}apad=whole_dur={dur}[mus]")
        if speech:
            filters.append("[mus][sp_key]sidechaincompress=threshold=0.03:ratio=6:attack=20:release=400[musd]")
            mix_inputs.append("[musd]")
        else:
            mix_inputs.append("[mus]")
    elif speech:
        filters.append("[sp_key]anullsink")
    if not mix_inputs:
        filters.append(f"anullsrc=r=48000:cl=stereo,atrim=duration={dur}[aout0]")
    else:
        filters.append("".join(mix_inputs) + f"amix=inputs={len(mix_inputs)}:normalize=0:duration=longest[aout0]")
    filters.append(f"[aout0]atrim=duration={dur},asetpts=PTS-STARTPTS,"
                   f"loudnorm=I={target_lufs}:TP=-1.5:LRA=11,aresample=48000[aout]")
    script = workdir / "filtergraph.txt"
    script.write_text(";\n".join(filters))
    args += ["-/filter_complex", str(script), "-map", "[vout]", "-map", "[aout]", "-c:v", "libx264",
             "-preset", "slow", "-crf", "17", "-profile:v", "high", "-pix_fmt", "yuv420p", "-r", str(fps),
             "-c:a", "aac", "-b:a", "192k", "-ar", "48000", "-movflags", "+faststart", "-t", dur, str(out)]
    return args


def render(edl: EDL, out: Path, workdir: Path, log: Path, burn_captions: bool = True, target_lufs: float = -14.0) -> None:
    workdir.mkdir(parents=True, exist_ok=True)
    cmd = build_command(edl, out, workdir, burn_captions, target_lufs)
    with log.open("a") as f:
        f.write("\n$ " + " ".join(cmd) + "\n")
        f.flush()
        r = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT)
    if r.returncode:
        raise RuntimeError(f"ffmpeg render failed ({r.returncode}); see {log}")


def write_srt(edl: EDL, path: Path) -> None:
    def ts(t: float) -> str:
        ms = int(round(t * 1000))
        h, ms = divmod(ms, 3_600_000)
        m, ms = divmod(ms, 60_000)
        s, ms = divmod(ms, 1000)
        return f"{h:02}:{m:02}:{s:02},{ms:03}"
    path.write_text("\n".join(f"{i}\n{ts(c.start)} --> {ts(c.end)}\n{c.text}\n"
                              for i, c in enumerate(edl.captions, 1)))


def thumbnail(video: Path, at: float, out: Path) -> None:
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-ss", f"{at:.2f}", "-i", str(video), "-frames:v", "1", str(out)],
                   check=True)
