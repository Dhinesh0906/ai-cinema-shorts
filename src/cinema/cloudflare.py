"""FLUX.2 [dev] on Cloudflare Workers AI with a rotating multi-account credential ring.

Interface (Cloudflare docs, verified 2026-10-10): POST
  https://api.cloudflare.com/client/v4/accounts/{account_id}/ai/run/@cf/black-forest-labs/flux-2-dev
multipart/form-data: prompt, steps, width, height, input_image_0..N (each < 512x512); response JSON
{"result": {"image": <base64>}}. Cost (pricing page): 37.5 neurons per output 512x512 tile per step +
18.75 per input tile per step; 10,000 free neurons/day/account (resets 00:00 UTC).

License: FLUX.2 [dev] weights are under the FLUX Non-Commercial License; the model card states
"Generated outputs can be used for personal, scientific, and commercial purposes".

Rotation: daily-allocation/rate-limit errors exhaust the current account and move to the next; when all
are exhausted the ring resets and starts over. `on_all_exhausted: local` instead hands the image to the
local FLUX.2 klein 4B (recorded in metadata) so a film never stalls until the UTC reset.
"""
from __future__ import annotations

import base64
import io
import json
import logging
import math
import time
from pathlib import Path

import httpx
from PIL import Image

from .cloud import KeyRing, _is_rate_limited
from .config import Config

log = logging.getLogger("cinema")


class AllAccountsExhausted(RuntimeError):
    pass


def neuron_estimate(width: int, height: int, steps: int, n_refs: int) -> float:
    out_tiles = math.ceil(width / 512) * math.ceil(height / 512)
    return steps * (37.5 * out_tiles + 18.75 * n_refs)


def _ref_bytes(path: Path, max_side: int) -> bytes:
    im = Image.open(path).convert("RGB")
    scale = max_side / max(im.size)
    if scale < 1:
        im = im.resize((max(8, int(im.width * scale)) // 8 * 8, max(8, int(im.height * scale)) // 8 * 8),
                       Image.LANCZOS)
    buf = io.BytesIO()
    im.save(buf, "PNG")
    return buf.getvalue()


def flux2_dev(cfg: Config, prompt: str, out: Path, width: int, height: int, seed: int,
              references: list[Path] | None = None) -> dict:
    cc = cfg.cloudflare
    ring = KeyRing(cfg, "cloudflare", cc.keys_file)
    refs = references or []
    form = {"prompt": prompt, "steps": str(cc.steps), "width": str(width), "height": str(height)}
    if cc.send_seed:
        form["seed"] = str(seed)
    # Every field is sent as a multipart part (the API requires multipart even without images).
    parts = [(k, (None, v)) for k, v in form.items()]
    parts += [(f"input_image_{i}", (f"ref{i}.png", _ref_bytes(r, cc.ref_max_side), "image/png"))
              for i, r in enumerate(refs)]
    started, resets, server_errors = time.monotonic(), 0, 0
    while True:
        idx, acct = ring.current()
        url = f"https://api.cloudflare.com/client/v4/accounts/{acct['account_id']}/ai/run/{cc.model}"
        try:
            r = httpx.post(url, headers={"Authorization": f"Bearer {acct['token']}"}, files=parts,
                           timeout=httpx.Timeout(cc.timeout_s, connect=30))
        except httpx.HTTPError as exc:
            log.warning("cloudflare network error on %s: %s", ring.label(idx), exc)
            time.sleep(10)
            if time.monotonic() - started > cfg.cloud.max_error_seconds:
                raise RuntimeError(f"cloudflare unreachable: {exc}") from exc
            continue
        if _is_rate_limited(r) or r.status_code in (401, 403):
            if ring.exhaust(idx, f"HTTP {r.status_code} {r.text[:200]}"):
                resets += 1
                if cc.on_all_exhausted == "local":
                    raise AllAccountsExhausted("all Cloudflare accounts exhausted")
                time.sleep(cc.cycle_pause_s)
            continue
        if r.status_code >= 500:
            # Measured 2026-10-10: flux-2-dev returned HTTP 500 / code 3043 for every request (even the
            # documented example) while flux-2-klein-4b worked; bounded retries, then local fallback.
            server_errors += 1
            if server_errors >= cc.max_server_errors:
                raise AllAccountsExhausted(f"cloudflare server error x{server_errors}: {r.status_code} {r.text[:200]}")
            time.sleep(10)
            continue
        try:
            data = r.json()
        except ValueError:
            raise RuntimeError(f"cloudflare non-JSON response HTTP {r.status_code}: {r.text[:200]}")
        if r.status_code >= 400 or not data.get("success", True):
            raise RuntimeError(f"cloudflare rejected request: HTTP {r.status_code} {json.dumps(data)[:400]}")
        b64 = (data.get("result") or {}).get("image")
        if not b64:
            raise RuntimeError(f"cloudflare returned no image: {json.dumps(data)[:300]}")
        out.parent.mkdir(parents=True, exist_ok=True)
        Image.open(io.BytesIO(base64.b64decode(b64))).convert("RGB").save(out)
        meta = {"backend": "cloudflare", "model": cc.model, "account": ring.label(idx), "steps": cc.steps,
                "width": width, "height": height, "seed": seed if cc.send_seed else None,
                "references": [str(x) for x in refs], "ref_max_side": cc.ref_max_side,
                "neurons_estimate": round(neuron_estimate(width, height, cc.steps, len(refs)), 1),
                "seconds": round(time.monotonic() - started, 2), "ring_resets": resets, "prompt": prompt}
        out.with_suffix(".metadata.json").write_text(json.dumps(meta, indent=2))
        return meta
