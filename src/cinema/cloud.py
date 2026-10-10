"""Ollama Cloud client with a rotating API-key ring.

Rotation policy (requested by the operator):
  - use the current key until the API says it is rate limited (HTTP 429 / limit message);
  - then mark it exhausted and move to the next key;
  - when every key is exhausted, reset all of them and start again from the first one;
  - repeat indefinitely (a short pause between full cycles avoids hammering the API).

Ring state lives in state/cloud_keyring.json under a file lock so concurrent jobs share it and it
survives restarts. Keys are read from the gitignored secrets file and are never written to logs
(only their index and last 4 characters).
"""
from __future__ import annotations

import fcntl
import json
import logging
import time
from contextlib import contextmanager
from pathlib import Path

import httpx

from .config import Config

log = logging.getLogger("cinema")

LIMIT_WORDS = ("rate limit", "rate-limit", "ratelimit", "too many requests", "usage limit", "quota",
               "limit exceeded", "exceeded your", "daily free allocation", "neurons", "capacity")


class CloudError(RuntimeError):
    pass


def _mask(key: str) -> str:
    return "…" + key[-4:]


class KeyRing:
    """Rotating credential ring. Entries are opaque (a string key or a dict like
    {"account_id", "token"}); state is persisted per ring name under a file lock."""

    def __init__(self, cfg: Config, name: str = "ollama", keys_file: str | None = None):
        self.cfg, self.name = cfg, name
        secrets = cfg.resolve(keys_file or cfg.cloud.keys_file)
        if not secrets.exists():
            raise CloudError(f"{name} credentials file missing: {secrets}")
        data = json.loads(secrets.read_text())
        raw = data.get("keys") or data.get("accounts") or []
        self.keys: list = [k.strip() if isinstance(k, str) else k for k in raw if k]
        self.base_url: str = data.get("base_url", "https://ollama.com").rstrip("/")
        if not self.keys:
            raise CloudError(f"no credentials configured in {secrets}")
        suffix = "" if name == "ollama" else f"_{name}"
        self.state_path = cfg.root / "state" / f"cloud_keyring{suffix}.json"
        self.state_path.parent.mkdir(parents=True, exist_ok=True)

    def label(self, idx: int) -> str:
        k = self.keys[idx]
        secret = k if isinstance(k, str) else str(k.get("token", ""))
        return f"{self.name}#{idx} ({_mask(secret)})"

    @contextmanager
    def _locked(self):
        with (self.state_path.with_suffix(".lock")).open("a+") as lf:
            fcntl.flock(lf, fcntl.LOCK_EX)
            try:
                st = json.loads(self.state_path.read_text()) if self.state_path.exists() else {}
            except ValueError:
                st = {}
            st.setdefault("current", 0)
            st.setdefault("exhausted", [])
            st.setdefault("cycles", 0)
            st["current"] %= len(self.keys)
            yield st
            self.state_path.write_text(json.dumps(st))
            fcntl.flock(lf, fcntl.LOCK_UN)

    def current(self) -> tuple[int, object]:
        with self._locked() as st:
            return st["current"], self.keys[st["current"]]

    def advance(self, idx: int, reason: str) -> None:
        """Move to the next key WITHOUT marking idx exhausted (timeouts / connection errors)."""
        with self._locked() as st:
            if st["current"] == idx:
                free = [i for i in range(len(self.keys)) if i not in st["exhausted"]] or list(range(len(self.keys)))
                st["current"] = next((i for i in free if i > idx), free[0])
            log.warning("cloud key %s no answer (%s); trying the next key", self.label(idx), reason[:120])

    def exhaust(self, idx: int, reason: str) -> bool:
        """Mark key idx exhausted and advance. Returns True when a full reset just happened."""
        with self._locked() as st:
            if idx not in st["exhausted"]:
                st["exhausted"].append(idx)
            log.warning("cloud key %s exhausted: %s", self.label(idx), reason[:160])
            free = [i for i in range(len(self.keys)) if i not in st["exhausted"]]
            if free:
                st["current"] = next((i for i in free if i > idx), free[0])
                return False
            st["exhausted"] = []
            st["current"] = 0
            st["cycles"] += 1
            log.warning("all %d %s keys exhausted; resetting ring (cycle %d)", len(self.keys), self.name, st["cycles"])
            return True


def _is_rate_limited(resp: httpx.Response) -> bool:
    if resp.status_code == 429:
        return True
    if resp.status_code >= 400:
        text = resp.text.lower()
        return any(w in text for w in LIMIT_WORDS)
    return False


def chat(cfg: Config, body: dict, timeout: float = 900) -> dict:
    """POST /api/chat to Ollama Cloud, rotating keys on rate limits forever. Other failures
    (network, 5xx) are retried with backoff for up to cfg.cloud.max_error_seconds, then raised."""
    ring = KeyRing(cfg)
    error_deadline = None
    backoff = 5.0
    tries_this_round = 0
    while True:
        idx, key = ring.current()
        try:
            r = httpx.post(f"{ring.base_url}/api/chat", json={**body, "stream": False},
                           headers={"Authorization": f"Bearer {key}"},
                           timeout=httpx.Timeout(timeout, connect=30))
        except httpx.HTTPError as exc:
            error_deadline = error_deadline or time.monotonic() + cfg.cloud.max_error_seconds
            if time.monotonic() > error_deadline:
                raise CloudError(f"cloud unreachable for {cfg.cloud.max_error_seconds}s: {exc}") from exc
            ring.advance(idx, str(exc))          # timeout / connection error: next key, no exhaustion mark
            tries_this_round += 1
            if tries_this_round >= len(ring.keys):  # every key timed out once: back off before the next round
                tries_this_round = 0
                time.sleep(backoff)
                backoff = min(backoff * 2, 120)
            continue
        if _is_rate_limited(r):
            if ring.exhaust(idx, f"HTTP {r.status_code} {r.text[:200]}"):
                time.sleep(cfg.cloud.cycle_pause_s)  # every key limited: pause, then start over
            continue
        if r.status_code in (401, 403):
            # invalid/revoked key: skip it like an exhausted key so the ring keeps going
            if ring.exhaust(idx, f"auth failure HTTP {r.status_code}"):
                time.sleep(cfg.cloud.cycle_pause_s)
            continue
        if r.status_code >= 500:
            error_deadline = error_deadline or time.monotonic() + cfg.cloud.max_error_seconds
            if time.monotonic() > error_deadline:
                raise CloudError(f"cloud server errors for {cfg.cloud.max_error_seconds}s: {r.status_code} {r.text[:200]}")
            time.sleep(backoff)
            backoff = min(backoff * 2, 120)
            continue
        if r.status_code >= 400:
            raise CloudError(f"cloud request rejected: HTTP {r.status_code} {r.text[:300]}")
        return r.json()


def list_models(cfg: Config) -> list[str]:
    ring = KeyRing(cfg)
    _, key = ring.current()
    r = httpx.get(f"{ring.base_url}/api/tags", headers={"Authorization": f"Bearer {key}"}, timeout=20)
    r.raise_for_status()
    return [m["name"] for m in r.json().get("models", [])]
