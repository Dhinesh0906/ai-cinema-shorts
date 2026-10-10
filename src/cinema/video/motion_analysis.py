"""Deterministic video measurements: decode, duration/fps/aspect, freeze, black, optical flow, temporal
spikes, saturation blow-up and feature-print drift. Nonzero motion alone never passes a shot — the
semantic VLM gate is separate and also mandatory."""
from __future__ import annotations

import json
import subprocess
import tempfile
from pathlib import Path

import numpy as np


def probe(path: Path) -> dict:
    r = subprocess.run(["ffprobe", "-v", "error", "-show_format", "-show_streams", "-of", "json", str(path)],
                       capture_output=True, text=True)
    if r.returncode:
        raise ValueError(f"ffprobe failed: {r.stderr[-300:]}")
    return json.loads(r.stdout)


def stream_info(path: Path) -> dict:
    d = probe(path)
    v = next((s for s in d["streams"] if s["codec_type"] == "video"), None)
    a = next((s for s in d["streams"] if s["codec_type"] == "audio"), None)
    if v is None:
        raise ValueError("no video stream")
    num, den = map(int, v["r_frame_rate"].split("/"))
    return {"width": v["width"], "height": v["height"], "fps": round(num / den, 3), "codec": v["codec_name"],
            "frames": int(v.get("nb_frames") or 0), "duration": float(v.get("duration") or d["format"]["duration"]),
            "pix_fmt": v.get("pix_fmt"), "audio": a is not None,
            "audio_codec": a["codec_name"] if a else None,
            "audio_duration": float(a.get("duration") or d["format"]["duration"]) if a else None}


def decode_ok(path: Path) -> tuple[bool, str]:
    r = subprocess.run(["ffmpeg", "-v", "error", "-xerror", "-i", str(path), "-f", "null", "-"],
                       capture_output=True, text=True)
    return r.returncode == 0, r.stderr[-500:]


def read_frames(path: Path, width: int = 256, gray: bool = False) -> np.ndarray:
    info = stream_info(path)
    h = int(round(info["height"] * width / info["width"] / 2) * 2)
    fmt = "gray" if gray else "rgb24"
    r = subprocess.run(["ffmpeg", "-v", "error", "-i", str(path), "-vf", f"scale={width}:{h}", "-f", "rawvideo",
                        "-pix_fmt", fmt, "-"], capture_output=True, check=True)
    ch = 1 if gray else 3
    arr = np.frombuffer(r.stdout, np.uint8)
    return arr[: arr.size // (width * h * ch) * width * h * ch].reshape(-1, h, width, *(() if gray else (3,)))


def flow_magnitudes(gray: np.ndarray) -> np.ndarray:
    import cv2
    mags = []
    for a, b in zip(gray[:-1], gray[1:]):
        f = cv2.calcOpticalFlowFarneback(a, b, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        mags.append(float(np.linalg.norm(f, axis=2).mean()))
    return np.array(mags)


def extract_frames(path: Path, times: list[float], out_dir: Path, prefix: str = "f") -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for i, t in enumerate(times):
        p = out_dir / f"{prefix}{i:02d}_{t:05.2f}s.png"
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-ss", f"{t:.3f}", "-i", str(path), "-frames:v", "1", str(p)],
                       check=True)
        paths.append(p)
    return paths


def sample_times(duration: float, n: int) -> list[float]:
    if n <= 1:
        return [0.0]
    end = max(0.0, duration - 0.08)
    return [round(end * i / (n - 1), 3) for i in range(n)]


def analyze(path: Path, *, expected_duration: float | None = None, expected_aspect: float = 9 / 16,
            reference: Path | None = None, thresholds=None) -> dict:
    """Return measurements and a list of technical problems (empty list = technical pass)."""
    problems: list[str] = []
    try:
        info = stream_info(path)
    except Exception as exc:
        return {"problems": [f"unreadable: {exc}"], "passed": False}
    ok, err = decode_ok(path)
    if not ok:
        problems.append(f"decode error: {err.strip()[:200]}")
    aspect = info["width"] / info["height"]
    if abs(aspect - expected_aspect) > 0.03:
        problems.append(f"aspect {info['width']}x{info['height']} is not 9:16")
    if expected_duration is not None and info["duration"] + 0.05 < expected_duration:
        problems.append(f"duration {info['duration']:.2f}s shorter than required {expected_duration:.2f}s")
    if info["fps"] < 12:
        problems.append(f"frame rate {info['fps']} too low")

    rgb = read_frames(path, 256)
    if len(rgb) < 5:
        problems.append("fewer than 5 decodable frames")
        return {**info, "problems": problems, "passed": False}
    gray = rgb.mean(axis=3).astype(np.uint8)
    luma = gray.reshape(len(gray), -1).mean(axis=1)
    diffs = np.abs(np.diff(gray.astype(np.int16), axis=0)).mean(axis=(1, 2))
    flow = flow_magnitudes(gray)
    # Freeze: a run of adjacent near-identical frames. Wan's 4-frame latent groups produce isolated
    # low-diff pairs, so require runs of >= 3 before counting as frozen.
    still = diffs < 0.35
    frozen = 0
    run = 0
    for s in still:
        run = run + 1 if s else 0
        if run >= 3:
            frozen += 1
    freeze_frac = frozen / max(1, len(diffs))
    black_frac = float((luma < 12).mean())
    med = float(np.median(diffs)) or 1e-3
    spike = float(diffs.max() / med) if med > 0 else 0.0
    hsv_sat = _saturation(rgb)
    sat_jump = float(hsv_sat[len(hsv_sat) // 2:].mean() - hsv_sat[:2].mean())

    t = thresholds
    mean_flow = float(flow.mean())
    if t is not None:
        if mean_flow < t.min_mean_flow:
            problems.append(f"near-static footage (mean optical flow {mean_flow:.3f} px/frame)")
        if freeze_frac > t.max_freeze_fraction:
            problems.append(f"frozen for {freeze_frac:.0%} of frames")
        if black_frac > t.max_black_fraction:
            problems.append(f"black for {black_frac:.0%} of frames")
        if spike > t.max_temporal_spike and diffs.max() > 25:
            problems.append(f"abrupt temporal spike (max/median frame change {spike:.1f})")
    if sat_jump > 45:
        problems.append(f"saturation blow-up (+{sat_jump:.0f}) — likely generation collapse")

    drift = None
    if reference is not None:
        drift = _feature_drift(path, reference, info["duration"])
        if t is not None and t.max_feature_drift > 0 and drift["max"] > t.max_feature_drift:
            problems.append(f"content drift from keyframe {drift['max']:.2f} > {t.max_feature_drift}")
    return {**info, "decode_ok": ok, "mean_flow": round(mean_flow, 4), "flow_p90": round(float(np.percentile(flow, 90)), 4),
            "mean_frame_diff": round(float(diffs.mean()), 3), "temporal_spike": round(spike, 2),
            "freeze_fraction": round(freeze_frac, 3), "black_fraction": round(black_frac, 3),
            "saturation_jump": round(sat_jump, 2), "feature_drift": drift,
            "problems": problems, "passed": not problems}


def _saturation(rgb: np.ndarray) -> np.ndarray:
    f = rgb.astype(np.float32)
    mx, mn = f.max(axis=3), f.min(axis=3)
    return ((mx - mn)).reshape(len(f), -1).mean(axis=1)


def _feature_drift(path: Path, reference: Path, duration: float) -> dict:
    from ..image.identity import feature_distance
    with tempfile.TemporaryDirectory() as td:
        frames = extract_frames(path, sample_times(duration, 5), Path(td))
        d = [feature_distance(reference, f) for f in frames]
    return {"distances": d, "max": max(d), "first": d[0], "last": d[-1]}
