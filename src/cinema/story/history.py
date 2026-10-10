"""Permanent memory of finished stories so new scripts don't repeat premises or twists.

One JSON line per COMPLETE film in state/story_history.jsonl (kept even if the project is deleted)."""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

from ..config import Config
from ..schemas import Screenplay


def _path(cfg: Config) -> Path:
    return cfg.root / "state" / "story_history.jsonl"


def record(cfg: Config, job_id: str, sp: Screenplay) -> None:
    st = sp.story
    entries = load(cfg)
    if any(e.get("job_id") == job_id for e in entries):
        return
    entry = {"job_id": job_id, "date": datetime.now().strftime("%Y-%m-%d"), "title": st.title, "genre": st.genre,
             "logline": st.logline, "theme": st.theme, "hook": st.hook, "twist": st.twist, "ending": st.ending,
             "characters": [f"{c.name} ({c.role})" for c in st.characters],
             "locations": [loc.name for loc in st.locations]}
    p = _path(cfg)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("a") as f:
        f.write(json.dumps(entry) + "\n")


def load(cfg: Config) -> list[dict]:
    p = _path(cfg)
    if not p.exists():
        return []
    out = []
    for line in p.read_text().splitlines():
        try:
            out.append(json.loads(line))
        except ValueError:
            pass
    return out


def avoid_list(cfg: Config, limit: int = 40) -> list[str]:
    return [f"{e['title']} — {e['logline']} | twist: {e['twist']} | ending: {e['ending']}"
            for e in load(cfg)[-limit:]]
