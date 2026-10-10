"""Opportunistic cloud video: OpenRouter `alibaba/wan-2.6:free` (image-to-video), rotating API keys.

Async API (docs, verified 2026-10-10): POST /api/v1/videos -> {id, polling_url}; GET polling_url until
status completed|failed; download unsigned_urls[0]. Image-to-video uses frame_images[first_frame].

Measured 2026-10-10: the `:free` variant is listed (price Free) but had ZERO endpoints (HTTP 404 "No
endpoints found"); the paid variant returns 402 without credits. Any of 404/402/rate-limit across the
whole key ring raises CloudVideoUnavailable and the caller renders the shot locally with Wan 2.2.
"""
from __future__ import annotations

import base64
import json
import logging
import time
from pathlib import Path

import httpx

from .cloud import KeyRing, _is_rate_limited
from .config import Config

log = logging.getLogger("cinema")
API = "https://openrouter.ai/api/v1"


class CloudVideoUnavailable(RuntimeError):
    pass


def _b64(path: Path) -> str:
    return "data:image/png;base64," + base64.b64encode(path.read_bytes()).decode()


def generate(cfg: Config, prompt: str, negative: str, image: Path, out: Path, seconds: float, seed: int) -> dict:
    oc = cfg.openrouter
    ring = KeyRing(cfg, "openrouter", oc.keys_file)
    duration = min(oc.durations, key=lambda d: (d < seconds, abs(d - seconds)))  # shortest that covers the shot
    body = {"model": oc.model, "prompt": prompt, "duration": duration, "resolution": oc.resolution,
            "aspect_ratio": "9:16", "generate_audio": False,
            "frame_images": [{"type": "image_url", "image_url": {"url": _b64(image)}, "frame_type": "first_frame"}]}
    # seed / negative_prompt passthrough omitted: not in wan-2.6's supported parameters (would risk a 400)
    t0, tried = time.monotonic(), set()
    while True:
        idx, key = ring.current()
        if idx in tried and len(tried) >= len(ring.keys):
            raise CloudVideoUnavailable("every OpenRouter key was refused (rate limit / quota)")
        headers = {"Authorization": f"Bearer {key}"}
        try:
            r = httpx.post(f"{API}/videos", headers=headers, json=body, timeout=120)
        except httpx.HTTPError as exc:
            raise CloudVideoUnavailable(f"network: {exc}") from exc
        if r.status_code in (402, 404):
            raise CloudVideoUnavailable(f"HTTP {r.status_code}: {r.text[:200]}")
        if _is_rate_limited(r) or r.status_code in (401, 403):
            tried.add(idx)
            ring.exhaust(idx, f"HTTP {r.status_code} {r.text[:150]}")
            continue
        if r.status_code >= 400:
            raise CloudVideoUnavailable(f"HTTP {r.status_code}: {r.text[:300]}")
        job = r.json()
        poll = job.get("polling_url") or f"{API}/videos/{job['id']}"
        break
    while time.monotonic() - t0 < oc.timeout_s:
        time.sleep(oc.poll_s)
        try:
            st = httpx.get(poll, headers=headers, timeout=60).json()
        except (httpx.HTTPError, ValueError):
            continue
        status = st.get("status")
        if status == "completed":
            urls = st.get("unsigned_urls") or []
            if not urls:
                raise CloudVideoUnavailable("completed without a download URL")
            v = httpx.get(urls[0], headers=headers, timeout=600, follow_redirects=True)
            v.raise_for_status()
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_bytes(v.content)
            meta = {"backend": "openrouter", "model": oc.model, "job_id": job.get("id"), "duration": duration,
                    "resolution": oc.resolution, "seconds": round(time.monotonic() - t0, 1),
                    "key": ring.label(idx), "usage": st.get("usage"), "prompt": prompt, "input_reference": str(image)}
            out.with_suffix(".metadata.json").write_text(json.dumps(meta, indent=2))
            return meta
        if status in ("failed", "cancelled", "expired"):
            raise CloudVideoUnavailable(f"job {status}: {st.get('error')}")
    raise CloudVideoUnavailable(f"job did not finish within {oc.timeout_s}s")
