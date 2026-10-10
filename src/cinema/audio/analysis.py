"""FFmpeg-based technical audio measurements."""
from __future__ import annotations

import re
import subprocess
from pathlib import Path


def _ff(args: list[str]) -> str:
    return subprocess.run(["ffmpeg", "-hide_banner", "-nostats", *args], capture_output=True, text=True).stderr


def duration(path: Path) -> float:
    r = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(path)],
                       capture_output=True, text=True)
    return float(r.stdout.strip() or 0)


def measure(path: Path, silence_db: float = -45, min_silence: float = 0.6) -> dict:
    err = _ff(["-i", str(path), "-af", f"astats=metadata=0,ebur128=peak=true,silencedetect=n={silence_db}dB:d={min_silence}",
               "-f", "null", "-"])
    peak = re.findall(r"Peak level dB:\s*(-?[\d.inf]+)", err)
    rms = re.findall(r"RMS level dB:\s*(-?[\d.inf]+)", err)
    lufs = re.findall(r"I:\s*(-?[\d.]+) LUFS", err)
    true_peak = re.findall(r"Peak:\s*(-?[\d.]+) dBFS", err)
    starts = [float(x) for x in re.findall(r"silence_start:\s*(-?[\d.]+)", err)]
    ends = [float(x) for x in re.findall(r"silence_end:\s*([\d.]+)", err)]
    durs = [float(x) for x in re.findall(r"silence_duration:\s*([\d.]+)", err)]
    clip = re.findall(r"Number of samples clipped:\s*(\d+)", err)  # not always present
    total = duration(path)

    def fl(v: list[str]) -> float | None:
        try:
            return float(v[-1])
        except (IndexError, ValueError):
            return None
    silences = [{"start": max(0.0, s), "end": e, "duration": d} for s, e, d in zip(starts, ends, durs)]
    if len(starts) > len(ends):  # trailing silence runs to EOF
        silences.append({"start": starts[-1], "end": total, "duration": total - starts[-1]})
    return {"duration": round(total, 3), "peak_db": fl(peak), "rms_db": fl(rms), "lufs": fl(lufs),
            "true_peak_dbfs": fl(true_peak), "silences": silences,
            "silence_total": round(sum(s["duration"] for s in silences), 3),
            "clipped_samples": int(clip[-1]) if clip else None}


def leading_trailing_silence(path: Path, thresh_db: float = -45) -> tuple[float, float]:
    m = measure(path, thresh_db, 0.05)
    lead = next((s["duration"] for s in m["silences"] if s["start"] <= 0.01), 0.0)
    trail = next((s["duration"] for s in m["silences"] if abs(s["end"] - m["duration"]) < 0.05), 0.0)
    return round(lead, 3), round(trail, 3)
