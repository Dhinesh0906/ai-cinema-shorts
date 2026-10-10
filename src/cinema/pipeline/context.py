"""Shared per-job context for stage runners."""
from __future__ import annotations

import logging
import shutil
from dataclasses import dataclass, field
from pathlib import Path

from ..config import Config, Preset
from ..resource_manager import ResourceManager
from ..schemas import Decision, Screenplay
from ..state_machine import AssetState, JobState
from ..storage.database import DB
from ..storage.files import JobDirs, atomic_json, read_json, sha256

log = logging.getLogger("cinema")


def _path(src: AssetState, dst: AssetState) -> list[AssetState]:
    """Shortest legal transition path (BFS over the asset state machine)."""
    from ..state_machine import ASSET_TRANSITIONS
    if src == dst:
        return [src, dst]
    prev, queue = {src: None}, [src]
    while queue:
        cur = queue.pop(0)
        for nxt in ASSET_TRANSITIONS[cur]:
            if nxt not in prev:
                prev[nxt] = cur
                if nxt == dst:
                    out = [dst]
                    while prev[out[-1]] is not None:
                        out.append(prev[out[-1]])
                    return out[::-1]
                queue.append(nxt)
    return [src, dst]  # let set_asset raise the TransitionError


class StageBlocked(RuntimeError):
    """Mandatory work could not pass within its retry budget (job -> BLOCKED)."""


class StageFailed(RuntimeError):
    """Recovery strategies exhausted (job -> FAILED)."""


@dataclass
class Ctx:
    cfg: Config
    db: DB
    rm: ResourceManager
    job_id: str
    job: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.job = self.db.job(self.job_id) or {}
        self.dirs = JobDirs(self.cfg.root / "projects" / self.job_id)

    # ---------------------------------------------------------------- helpers
    @property
    def preset(self) -> Preset:
        return self.cfg.presets[self.job["preset"]]

    @property
    def seed(self) -> int:
        return int(self.job["seed"])

    @property
    def target(self) -> int:
        return int(self.job["duration"])

    @property
    def style(self) -> str:
        return self.job["params"].get("style") or self.cfg.visual_style

    def state(self, s: JobState) -> None:
        self.db.set_job_status(self.job_id, s)
        self.db.event(self.job_id, f"state {s}")
        log.info("[%s] %s", self.job_id, s)

    def info(self, msg: str, **data) -> None:
        self.db.event(self.job_id, msg, **data)
        log.info("[%s] %s", self.job_id, msg)

    def warn(self, msg: str, **data) -> None:
        self.db.event(self.job_id, msg, "warn", **data)
        log.warning("[%s] %s", self.job_id, msg)

    def decide(self, d: Decision) -> None:
        self.db.log_decision(self.job_id, d)

    def asset(self, aid: str) -> dict:
        return self.db.asset(self.job_id, aid)

    def ensure(self, aid: str, kind: str, deps: list[str]) -> dict:
        return self.db.ensure_asset(self.job_id, aid, kind, deps)

    def set(self, aid: str, status: AssetState | None = None, **kw) -> None:
        self.db.set_asset(self.job_id, aid, status, **kw)

    def passed(self, aid: str) -> bool:
        """PASSED and its file still matches the recorded hash (protects resume from partial writes)."""
        try:
            a = self.asset(aid)
        except KeyError:
            return False
        if a["status"] != AssetState.PASSED:
            return False
        p = Path(a["path"]) if a["path"] else None
        if p is None or not p.exists() or (a["sha256"] and sha256(p) != a["sha256"]):
            self.warn(f"{aid} was PASSED but its file is missing/corrupt; regenerating")
            self.db.invalidate(self.job_id, aid, "file missing or hash mismatch")
            return False
        return True

    def approve(self, aid: str, src: Path, dest: Path, meta: dict | None = None) -> None:
        dest.parent.mkdir(parents=True, exist_ok=True)
        if src.resolve() != dest.resolve():
            tmp = dest.with_suffix(dest.suffix + ".part")
            shutil.copy2(src, tmp)
            tmp.replace(dest)
        self.to(aid, AssetState.PASSED, path=str(dest), sha256=sha256(dest), meta=meta or {})

    def to(self, aid: str, status: AssetState, **kw) -> None:
        """Move an asset to status, stepping through GENERATING/VALIDATING where the state machine needs it."""
        cur = AssetState(self.asset(aid)["status"])
        for s in _path(cur, status)[1:-1]:
            self.set(aid, s)
        self.set(aid, status, **kw)

    # ---------------------------------------------------------------- screenplay
    def screenplay(self) -> Screenplay:
        return Screenplay.model_validate(read_json(self.dirs.script / "screenplay.json"))

    def save_report(self, name: str, data) -> Path:
        p = self.dirs.reports / f"{name}.json"
        atomic_json(p, data)
        return p
