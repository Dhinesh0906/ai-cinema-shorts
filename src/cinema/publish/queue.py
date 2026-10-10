"""Publish queue: each film goes to YouTube Shorts and Instagram Reels at the same time; consecutive
films are at least `gap` (2 h) apart.

State: state/publish_queue.json. A film is never queued twice for the same platform. Failed uploads are
retried every 30 min, up to 3 attempts, then marked failed (and reported).
"""
from __future__ import annotations

import fcntl
import json
import time
import uuid
from contextlib import contextmanager
from pathlib import Path

from ..config import Config

GAP_S = 2 * 3600
RETRY_S = 30 * 60
MAX_ATTEMPTS = 3
PLATFORMS = ("youtube", "instagram")  # supported; cfg.publish_platforms selects which are auto-posted


@contextmanager
def _locked(cfg: Config):
    p = cfg.root / "state" / "publish_queue.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.with_suffix(".lock").open("a+") as lf:
        fcntl.flock(lf, fcntl.LOCK_EX)
        try:
            q = json.loads(p.read_text()) if p.exists() else {}
        except ValueError:
            q = {}
        q.setdefault("items", [])
        q.setdefault("last_publish_at", 0.0)
        yield q
        p.write_text(json.dumps(q, indent=2))
        fcntl.flock(lf, fcntl.LOCK_UN)


def enqueue(cfg: Config, job_id: str, now: float | None = None, gap: float = GAP_S) -> list[dict]:
    now = now or time.time()
    added = []
    with _locked(cfg) as q:
        taken = {(i["job_id"], i["platform"]) for i in q["items"] if i["status"] != "failed"}
        other_slots = [i["not_before"] for i in q["items"] if i["status"] == "queued" and i["job_id"] != job_id]
        latest = max([q["last_publish_at"] if q.get("last_publish_job") != job_id else 0] + other_slots)
        t = max(now, latest + gap) if latest else now
        for platform in cfg.publish_platforms:  # same slot for every platform
            if (job_id, platform) in taken:
                continue
            item = {"id": uuid.uuid4().hex[:8], "job_id": job_id, "platform": platform, "not_before": t,
                    "status": "queued", "attempts": 0, "result": None, "error": None}
            q["items"].append(item)
            added.append(item)
    return added


def due(cfg: Config, now: float | None = None, gap: float = GAP_S) -> dict | None:
    now = now or time.time()
    with _locked(cfg) as q:
        in_gap = bool(q["last_publish_at"]) and now < q["last_publish_at"] + gap
        ready = sorted((i for i in q["items"] if i["status"] == "queued" and i["not_before"] <= now
                        and (not in_gap or i["job_id"] == q.get("last_publish_job"))),  # same film: no gap
                       key=lambda i: i["not_before"])
        if not ready:
            return None
        ready[0]["status"] = "publishing"
        return dict(ready[0])


def finish(cfg: Config, item_id: str, result: dict | None = None, error: str | None = None,
           now: float | None = None) -> dict:
    now = now or time.time()
    with _locked(cfg) as q:
        it = next(i for i in q["items"] if i["id"] == item_id)
        it["attempts"] += 1
        if result is not None:
            it.update(status="done", result=result, error=None, published_at=now)
            if q.get("last_publish_job") != it["job_id"]:
                q["last_publish_at"] = now  # the 2h gap is measured from a film's first post
            q["last_publish_job"] = it["job_id"]
        elif it["attempts"] >= MAX_ATTEMPTS:
            it.update(status="failed", error=error)
        else:
            it.update(status="queued", error=error, not_before=now + RETRY_S)
        return dict(it)


def items(cfg: Config) -> list[dict]:
    with _locked(cfg) as q:
        return list(q["items"])


def recover_stuck(cfg: Config) -> list[dict]:
    """Items left 'publishing' by a crash are NOT retried automatically (that could double-post);
    they are marked failed so the owner can check the platform."""
    hit = []
    with _locked(cfg) as q:
        for i in q["items"]:
            if i["status"] == "publishing":
                i.update(status="failed", error="interrupted during upload — check the platform before re-queueing")
                hit.append(dict(i))
    return hit


def publish_item(cfg: Config, item: dict) -> dict:
    out = cfg.root / "outputs" / item["job_id"]
    video = out / "final.mp4"
    if not video.exists():
        raise FileNotFoundError(f"{video} missing")
    from .metadata import build
    meta = build(cfg, item["job_id"])
    if item["platform"] == "youtube":
        from .youtube import upload
        return upload(cfg, video, meta["title"], meta["youtube_description"], meta["youtube_tags"],
                      privacy=cfg.publish_privacy)
    from .instagram import upload
    return upload(cfg, video, meta["instagram_caption"])
