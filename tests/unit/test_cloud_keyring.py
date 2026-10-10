"""Key-ring rotation: advance on rate limit, reset when all exhausted, keep cycling indefinitely."""
import json
import logging

import httpx
import pytest

from cinema import cloud


class FakeResp:
    def __init__(self, code, body):
        self.status_code, self._body = code, body
        self.text = json.dumps(body)

    def json(self):
        return self._body


@pytest.fixture
def ring_cfg(cfg, tmp_path):
    keys = ["k0-aaaa", "k1-bbbb", "k2-cccc"]
    (tmp_path / "secrets").mkdir()
    (tmp_path / "secrets" / "ollama_api_keys.json").write_text(json.dumps({"keys": keys}))
    cfg.cloud.cycle_pause_s = 0
    return cfg, keys


def _script(monkeypatch, plan):
    """plan: list of (expected_key_or_None, status). Records keys used."""
    used = []

    def fake_post(url, json=None, headers=None, timeout=None):
        key = headers["Authorization"].split()[1]
        used.append(key)
        code = plan[len(used) - 1]
        body = {"error": "you have reached your rate limit"} if code == 429 else {"message": {"content": "{}"}}
        return FakeResp(code, body)
    monkeypatch.setattr(httpx, "post", fake_post)
    monkeypatch.setattr(cloud.time, "sleep", lambda s: None)
    return used


def test_rotates_to_next_key_on_rate_limit(ring_cfg, monkeypatch):
    cfg, keys = ring_cfg
    used = _script(monkeypatch, [429, 429, 200])
    cloud.chat(cfg, {"model": "gemma4:31b", "messages": []})
    assert used == keys
    # next call starts from the key that worked
    used2 = _script(monkeypatch, [200])
    cloud.chat(cfg, {"model": "gemma4:31b", "messages": []})
    assert used2 == ["k2-cccc"]


def test_resets_after_all_exhausted_and_keeps_cycling(ring_cfg, monkeypatch, caplog):
    cfg, keys = ring_cfg
    # three full cycles of rate limits, then success on the 2nd key of cycle 4
    used = _script(monkeypatch, [429] * 9 + [429, 200])
    with caplog.at_level(logging.WARNING, logger="cinema"):
        cloud.chat(cfg, {"model": "gpt-oss:120b", "messages": []})
    assert used == keys * 3 + ["k0-aaaa", "k1-bbbb"]
    state = json.loads((cfg.root / "state" / "cloud_keyring.json").read_text())
    assert state["cycles"] == 3 and state["current"] == 1
    assert not any(k in caplog.text for k in keys)  # keys never logged, only masked tails


def test_invalid_key_is_skipped(ring_cfg, monkeypatch):
    cfg, keys = ring_cfg
    used = _script(monkeypatch, [401, 200])
    cloud.chat(cfg, {"model": "gemma4:31b", "messages": []})
    assert used == keys[:2]


def test_timeout_moves_to_next_key_without_exhausting(ring_cfg, monkeypatch):
    cfg, keys = ring_cfg
    used = []

    def fake_post(url, json=None, headers=None, timeout=None):
        used.append(headers["Authorization"].split()[1])
        if len(used) == 1:
            raise httpx.ReadTimeout("The read operation timed out")
        return FakeResp(200, {"message": {"content": "{}"}})
    monkeypatch.setattr(httpx, "post", fake_post)
    monkeypatch.setattr(cloud.time, "sleep", lambda s: None)
    cloud.chat(cfg, {"model": "gemma4:31b", "messages": []})
    assert used == keys[:2]
    state = json.loads((cfg.root / "state" / "cloud_keyring.json").read_text())
    assert state["exhausted"] == []          # a timeout is not quota exhaustion
