"""OpenRouter video adapter: 404/402 -> unavailable (local fallback); success path downloads the clip."""
import json

import httpx
import pytest
from PIL import Image

from cinema import openrouter_video as ov


class R:
    def __init__(self, code, body=None, content=b""):
        self.status_code, self._b, self.content = code, body or {}, content
        self.text = json.dumps(self._b)

    def json(self):
        return self._b

    def raise_for_status(self):
        pass


@pytest.fixture
def or_cfg(cfg, tmp_path):
    (tmp_path / "secrets").mkdir()
    (tmp_path / "secrets" / "openrouter_keys.json").write_text(json.dumps({"keys": ["k0", "k1"]}))
    cfg.openrouter.poll_s = 0
    img = tmp_path / "kf.png"
    Image.new("RGB", (480, 832)).save(img)
    return cfg, img


def test_no_endpoints_means_unavailable(or_cfg, tmp_path, monkeypatch):
    cfg, img = or_cfg
    monkeypatch.setattr(httpx, "post", lambda *a, **k: R(404, {"error": {"message": "No endpoints found"}}))
    with pytest.raises(ov.CloudVideoUnavailable):
        ov.generate(cfg, "p", "n", img, tmp_path / "o.mp4", 4.3, 1)


def test_rate_limit_rotates_then_succeeds(or_cfg, tmp_path, monkeypatch):
    cfg, img = or_cfg
    posts = []

    def post(url, headers=None, json=None, timeout=None):
        posts.append((headers["Authorization"], json))
        return R(429, {"error": "rate limit"}) if len(posts) == 1 else R(200, {"id": "j1", "polling_url": "P"})

    def get(url, headers=None, timeout=None, follow_redirects=False):
        return R(200, content=b"MP4DATA") if url == "U" else R(200, {"status": "completed", "unsigned_urls": ["U"]})
    monkeypatch.setattr(httpx, "post", post)
    monkeypatch.setattr(httpx, "get", get)
    monkeypatch.setattr(ov.time, "sleep", lambda s: None)
    meta = ov.generate(cfg, "p", "n", img, tmp_path / "o.mp4", 4.3, 1)
    assert [p[0] for p in posts] == ["Bearer k0", "Bearer k1"]
    body = posts[1][1]
    assert body["duration"] == 5 and body["aspect_ratio"] == "9:16"
    assert body["frame_images"][0]["frame_type"] == "first_frame"
    assert (tmp_path / "o.mp4").read_bytes() == b"MP4DATA" and meta["backend"] == "openrouter"
