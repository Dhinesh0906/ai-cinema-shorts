"""Best-of-N fallback: when an asset has used every retry, ship its best validated attempt instead of blocking.

Candidates come from the per-attempt validation reports on disk, so it also works after a pause/crash/resume.
Ranking: fewest failed mandatory checks, then highest validator score. Only attempts that reached the
semantic validator count (crashed renders and technical failures - black, frozen, wrong size - never ship),
and only attempts rendered after their inputs (an older render of a since-regenerated keyframe is stale).
"""
from __future__ import annotations

import json
from pathlib import Path

from .context import Ctx

NO_USABLE = "no usable attempt"  # feedback prefix when best-of finds nothing; resume then grants fresh tries
KINDS = ("clip", "keyframe", "char_ref", "loc_ref")


def _fresh_after(ctx: Ctx, deps: list[str]) -> float:
    t = 0.0
    for dep in deps:
        try:
            p = ctx.asset(dep).get("path")
        except KeyError:  # e.g. the screenplay is a file, not a tracked asset
            continue
        if p and Path(p).exists():
            t = max(t, Path(p).stat().st_mtime)
    return t


def pick(ctx: Ctx, pairs: list[tuple[Path, Path]], deps: list[str]) -> dict | None:
    """pairs: (output file, its validation report). Returns the best candidate or None."""
    newer_than = _fresh_after(ctx, deps)
    best = None
    for out, rep in pairs:
        if not (out.exists() and rep.exists()) or out.stat().st_mtime < newer_than:
            continue
        try:
            r = json.loads(rep.read_text())
        except ValueError:
            continue
        c = {"path": out, "score": float(r.get("score") or 0.0), "mandatory_failed": r.get("mandatory_failed", []),
             "report": rep.name}
        if best is None or (len(c["mandatory_failed"]), -c["score"]) < (len(best["mandatory_failed"]), -best["score"]):
            best = c
    return best


def clip_candidates(ctx: Ctx, shot_id: str) -> list[tuple[Path, Path]]:
    n = ctx.cfg.retries.max_attempts
    return [(ctx.dirs.work / f"{shot_id}_a{i}.mp4", ctx.dirs.reports / f"{shot_id}_a{i}_semantic.json")
            for i in range(1, n + 1)]


def image_candidates(ctx: Ctx, aid: str) -> list[tuple[Path, Path]]:
    stem, n = aid.replace(":", "_"), ctx.cfg.retries.max_attempts
    return [(ctx.dirs.work / f"{stem}_a{i}_c{k}.png", ctx.dirs.reports / f"{stem}_a{i}_c{k}.json")
            for i in range(1, n + 1) for k in range(max(ctx.cfg.image_candidates, 2))]


def summary(best: dict, attempts: int) -> str:
    issues = "; ".join(best["mandatory_failed"])[:300] or "none"
    return f"accepted best of {attempts} attempts ({best['report']}, score {best['score']:.2f}; known issues: {issues})"
