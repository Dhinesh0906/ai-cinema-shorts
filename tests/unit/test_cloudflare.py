"""Cloudflare FLUX.2 dev adapter: multipart request shape, account rotation, local fallback."""
import base64
import io
import json

import httpx
import pytest
from PIL import Image

from cinema import cloudflare


def _png_b64():
    buf = io.BytesIO()
    Image.new("RGB", (480, 832), (10, 20, 30)).save(buf, "PNG")
    return base64.b64encode(buf.getvalue()).decode()


class R:
    def __init__(self, code, body):
        self.status_code, self._b = code, body
        self.text = json.dumps(body)

    def json(self):
        return self._b


@pytest.fixture
def cf_cfg(cfg, tmp_path):
    (tmp_path / "secrets").mkdir()
    accts = [{"account_id": f"acc{i}", "token": f"tok{i}-secret"} for i in range(3)]
    (tmp_path / "secrets" / "cloudflare_accounts.json").write_text(json.dumps({"accounts": accts}))
    cfg.cloudflare.cycle_pause_s = 0
    cfg.image_backend = "cloudflare"
    return cfg


def test_multipart_and_rotation(cf_cfg, tmp_path, monkeypatch):
    calls = []
    ref = tmp_path / "ref.png"
    Image.new("RGB", (768, 1024)).save(ref)

    def fake_post(url, headers=None, files=None, timeout=None):
        calls.append((url, headers["Authorization"], files))
        if len(calls) == 1:
            return R(429, {"errors": [{"message": "you have used up your daily free allocation of 10,000 neurons"}]})
        return R(200, {"success": True, "result": {"image": _png_b64()}})
    monkeypatch.setattr(httpx, "post", fake_post)
    monkeypatch.setattr(cloudflare.time, "sleep", lambda s: None)
    meta = cloudflare.flux2_dev(cf_cfg, "a test", tmp_path / "out.png", 480, 832, 7, [ref])
    assert "acc0" in calls[0][0] and "acc1" in calls[1][0]          # rotated after quota error
    names = [p[0] for p in calls[1][2]]
    assert {"prompt", "steps", "width", "height", "input_image_0"} <= set(names)
    img_part = dict(calls[1][2])["input_image_0"]
    assert max(Image.open(io.BytesIO(img_part[1])).size) < 512        # refs downscaled below 512
    assert (tmp_path / "out.png").exists() and meta["neurons_estimate"] > 0
    assert "tok1-secret" not in json.dumps(meta)


def test_all_exhausted_falls_back_to_local(cf_cfg, tmp_path, monkeypatch):
    cf_cfg.cloudflare.on_all_exhausted = "local"
    monkeypatch.setattr(httpx, "post", lambda *a, **k: R(429, {"errors": [{"message": "rate limit"}]}))
    monkeypatch.setattr(cloudflare.time, "sleep", lambda s: None)
    with pytest.raises(cloudflare.AllAccountsExhausted):
        cloudflare.flux2_dev(cf_cfg, "x", tmp_path / "o.png", 480, 832, 1)
