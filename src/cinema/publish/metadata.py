"""Reach-oriented publishing metadata per platform, written by the cloud writer from the approved script.

Cached in outputs/JOB/publish_meta_v2.json. Deterministic fallback if the LLM is unavailable. Rules:
honest hook (no clickbait lies), teaser without spoiling the twist, few relevant hashtags, AI disclosure.
"""
from __future__ import annotations

import json
import re

from pydantic import BaseModel, Field

from ..config import Config

DISCLOSURE = "Original fiction. Visuals, voices and sound were generated with AI."


class ReachMeta(BaseModel):
    title: str = Field(description="curiosity hook, max 70 characters, true to the story, no spoilers")
    teaser: str = Field(description="1-2 sentence teaser that does NOT reveal the twist or ending")
    youtube_hashtags: list[str] = Field(min_length=2, max_length=5, description="without #, genre/topic specific")
    instagram_hashtags: list[str] = Field(min_length=4, max_length=8, description="without #, broad+genre+niche")
    keywords: list[str] = Field(min_length=5, max_length=15, description="YouTube search keywords")


def _tag(t: str) -> str:
    return "#" + re.sub(r"[^A-Za-z0-9]", "", t.title() if " " in t else t)


def _fallback(story: dict) -> ReachMeta:
    genre = re.sub(r"[^A-Za-z ]", "", story.get("genre", "story")).strip() or "story"
    return ReachMeta(title=story["title"][:70], teaser=story.get("hook") or story["logline"],
                     youtube_hashtags=[genre + " story", "short film"],
                     instagram_hashtags=["shortfilm", genre + "story", "cinematic", "storytime", "aifilm"],
                     keywords=[genre, "short film", "story", "cinematic short", "ai film"])


def build(cfg: Config, job_id: str) -> dict:
    out = cfg.root / "outputs" / job_id
    cache = out / "publish_meta_v2.json"
    if cache.exists():
        return json.loads(cache.read_text())
    story = json.loads((out / "story.json").read_text())
    try:
        from ..llm import chat_json
        m = chat_json(cfg, cfg.models.writer,
                      "You write YouTube Shorts and Instagram Reels metadata for an original fiction channel. "
                      "Maximize honest curiosity: a hook title that is TRUE to the story, a teaser that never "
                      "reveals the twist or ending, and a few highly relevant hashtags (genre + topic + format). "
                      "No clickbait lies, no real people, no brands.",
                      f"Story JSON: {json.dumps(story)}\nReturn JSON.", ReachMeta, temperature=0.6, retries=1,
                      timeout=120)
    except Exception:
        m = _fallback(story)
    ch = cfg.channel_name
    yt_tags = ["#Shorts"] + [_tag(t) for t in m.youtube_hashtags if _tag(t).lower() != "#shorts"][:4]
    ig_tags = [_tag(t) for t in m.instagram_hashtags][:8]
    meta = {
        "title": m.title.strip()[:70],
        "youtube_description": f"{m.teaser.strip()}\n\nFollow {ch} for a new short story every day.\n\n"
                               f"{DISCLOSURE}\n\n{' '.join(yt_tags)}",
        "youtube_tags": [k[:30] for k in m.keywords][:15],
        "instagram_caption": f"{m.title.strip()}\n\n{m.teaser.strip()}\n\nFollow @{re.sub(r'[^a-z0-9]', '', ch.lower())} "
                             f"for a new short story every day.\n{DISCLOSURE}\n\n{' '.join(ig_tags)}",
    }
    cache.write_text(json.dumps(meta, indent=2))
    return meta
