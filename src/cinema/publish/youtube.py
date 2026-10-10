"""YouTube Shorts upload via the official YouTube Data API v3 (resumable videos.insert).

Setup once: `uv run cinema publish-auth youtube` (opens the browser, stores a refresh token in
secrets/youtube_token.json). Client credentials come from a Google Cloud "Desktop app" OAuth client saved
as secrets/youtube_client.json.

Note (Google policy): videos uploaded through videos.insert by API projects that have NOT passed the
YouTube API audit are locked to private. Request the audit to publish publicly from the API.
"""
from __future__ import annotations

import http.server
import json
import secrets
import threading
import time
import urllib.parse
import webbrowser
from pathlib import Path

import httpx

from ..config import Config

SCOPE = "https://www.googleapis.com/auth/youtube.upload"
AUTH_URL = "https://accounts.google.com/o/oauth2/v2/auth"
TOKEN_URL = "https://oauth2.googleapis.com/token"
UPLOAD_URL = "https://www.googleapis.com/upload/youtube/v3/videos"


class YouTubeError(RuntimeError):
    pass


def _client(cfg: Config) -> dict:
    p = cfg.resolve("secrets/youtube_client.json")
    if not p.exists():
        raise YouTubeError("missing secrets/youtube_client.json (Google Cloud OAuth 'Desktop app' client)")
    d = json.loads(p.read_text())
    return d.get("installed") or d.get("web") or d


def authorize(cfg: Config) -> None:
    """One-time loopback OAuth (interactive, run by the owner)."""
    c = _client(cfg)
    state = secrets.token_urlsafe(16)
    result: dict = {}

    class H(http.server.BaseHTTPRequestHandler):
        def do_GET(self):  # noqa: N802
            q = urllib.parse.parse_qs(urllib.parse.urlparse(self.path).query)
            result.update({k: v[0] for k, v in q.items()})
            self.send_response(200)
            self.end_headers()
            self.wfile.write(b"YouTube authorized. You can close this tab.")

        def log_message(self, *a):
            pass

    srv = http.server.HTTPServer(("127.0.0.1", 0), H)
    redirect = f"http://127.0.0.1:{srv.server_port}/"
    url = AUTH_URL + "?" + urllib.parse.urlencode({
        "client_id": c["client_id"], "redirect_uri": redirect, "response_type": "code", "scope": SCOPE,
        "access_type": "offline", "prompt": "consent", "state": state})
    threading.Thread(target=srv.handle_request, daemon=True).start()
    print("Opening the browser for YouTube authorization…\n" + url)
    webbrowser.open(url)
    for _ in range(600):
        if result:
            break
        time.sleep(0.5)
    srv.server_close()
    if result.get("state") != state or "code" not in result:
        raise YouTubeError(f"authorization failed: {result}")
    r = httpx.post(TOKEN_URL, data={"code": result["code"], "client_id": c["client_id"],
                                    "client_secret": c["client_secret"], "redirect_uri": redirect,
                                    "grant_type": "authorization_code"}, timeout=30)
    r.raise_for_status()
    tok = r.json()
    if "refresh_token" not in tok:
        raise YouTubeError("no refresh token returned; remove the app's access at myaccount.google.com and retry")
    p = cfg.resolve("secrets/youtube_token.json")
    p.write_text(json.dumps({"refresh_token": tok["refresh_token"]}))
    p.chmod(0o600)


def _access_token(cfg: Config) -> str:
    c = _client(cfg)
    p = cfg.resolve("secrets/youtube_token.json")
    if not p.exists():
        raise YouTubeError("not authorized yet: run `uv run cinema publish-auth youtube`")
    r = httpx.post(TOKEN_URL, data={"refresh_token": json.loads(p.read_text())["refresh_token"],
                                    "client_id": c["client_id"], "client_secret": c["client_secret"],
                                    "grant_type": "refresh_token"}, timeout=30)
    if r.status_code != 200:
        raise YouTubeError(f"token refresh failed: {r.text[:300]}")
    return r.json()["access_token"]


def upload(cfg: Config, video: Path, title: str, description: str, tags: list[str], privacy: str = "public") -> dict:
    token = _access_token(cfg)
    title = title.strip()
    if "#shorts" not in title.lower() and "#shorts" not in description.lower():
        description = description.rstrip() + "\n\n#Shorts"
    meta = {"snippet": {"title": title[:100], "description": description[:4900], "tags": tags[:15],
                        "categoryId": "1"},  # Film & Animation
            "status": {"privacyStatus": privacy, "selfDeclaredMadeForKids": False,
                       "containsSyntheticMedia": True}}
    size = video.stat().st_size
    r = httpx.post(UPLOAD_URL, params={"uploadType": "resumable", "part": "snippet,status"},
                   headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json; charset=UTF-8",
                            "X-Upload-Content-Type": "video/mp4", "X-Upload-Content-Length": str(size)},
                   content=json.dumps(meta), timeout=60)
    if r.status_code != 200 or "location" not in r.headers:
        raise YouTubeError(f"upload init failed: HTTP {r.status_code} {r.text[:400]}")
    with video.open("rb") as f:
        up = httpx.put(r.headers["location"], headers={"Authorization": f"Bearer {token}",
                       "Content-Type": "video/mp4", "Content-Length": str(size)}, content=f.read(), timeout=1800)
    if up.status_code not in (200, 201):
        raise YouTubeError(f"upload failed: HTTP {up.status_code} {up.text[:400]}")
    v = up.json()
    return {"platform": "youtube", "id": v["id"], "url": f"https://youtube.com/shorts/{v['id']}",
            "privacy": v.get("status", {}).get("privacyStatus")}
