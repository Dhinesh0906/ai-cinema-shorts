"""Instagram Reels via the official Instagram API (content publishing, resumable upload — no public URL needed).

Requires an Instagram Business/Creator account and a long-lived access token with
instagram_business_content_publish (Instagram API with Instagram Login), saved as
secrets/instagram.json: {"user_id": "...", "access_token": "..."}. Long-lived tokens last ~60 days; they are
refreshed automatically when older than 30 days.

Flow: POST /{user}/media (media_type=REELS, upload_type=resumable) -> upload bytes to the returned uri ->
poll container status_code until FINISHED -> POST /{user}/media_publish.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import httpx

from ..config import Config

GRAPH = "https://graph.instagram.com/v23.0"


class InstagramError(RuntimeError):
    pass


def _creds(cfg: Config) -> dict:
    p = cfg.resolve("secrets/instagram.json")
    if not p.exists():
        raise InstagramError("missing secrets/instagram.json ({user_id, access_token})")
    d = json.loads(p.read_text())
    if time.time() - d.get("refreshed_at", 0) > 30 * 86400:  # keep the 60-day token alive
        r = httpx.get("https://graph.instagram.com/refresh_access_token",
                      params={"grant_type": "ig_refresh_token", "access_token": d["access_token"]}, timeout=30)
        if r.status_code == 200 and "access_token" in r.json():
            d.update(access_token=r.json()["access_token"], refreshed_at=time.time())
            p.write_text(json.dumps(d))
            p.chmod(0o600)
    return d


def upload(cfg: Config, video: Path, caption: str) -> dict:
    c = _creds(cfg)
    tok, uid = c["access_token"], c["user_id"]
    r = httpx.post(f"{GRAPH}/{uid}/media", params={"media_type": "REELS", "upload_type": "resumable",
                   "caption": caption[:2200], "share_to_feed": "true", "access_token": tok}, timeout=60)
    if r.status_code != 200 or "id" not in r.json():
        raise InstagramError(f"container create failed: HTTP {r.status_code} {r.text[:400]}")
    container, uri = r.json()["id"], r.json().get("uri") or f"https://rupload.facebook.com/ig-api-upload/v23.0/{r.json()['id']}"
    data = video.read_bytes()
    up = httpx.post(uri, headers={"Authorization": f"OAuth {tok}", "offset": "0", "file_size": str(len(data))},
                    content=data, timeout=1800)
    if up.status_code != 200:
        raise InstagramError(f"video upload failed: HTTP {up.status_code} {up.text[:400]}")
    for _ in range(120):  # Instagram processes the reel before it can be published
        st = httpx.get(f"{GRAPH}/{container}", params={"fields": "status_code,status", "access_token": tok},
                       timeout=30).json()
        if st.get("status_code") == "FINISHED":
            break
        if st.get("status_code") in ("ERROR", "EXPIRED"):
            raise InstagramError(f"processing failed: {st}")
        time.sleep(10)
    else:
        raise InstagramError("processing did not finish within 20 minutes")
    pub = httpx.post(f"{GRAPH}/{uid}/media_publish", params={"creation_id": container, "access_token": tok},
                     timeout=60)
    if pub.status_code != 200 or "id" not in pub.json():
        raise InstagramError(f"publish failed: HTTP {pub.status_code} {pub.text[:400]}")
    media_id = pub.json()["id"]
    link = httpx.get(f"{GRAPH}/{media_id}", params={"fields": "permalink", "access_token": tok}, timeout=30)
    return {"platform": "instagram", "id": media_id,
            "url": link.json().get("permalink") if link.status_code == 200 else None}
