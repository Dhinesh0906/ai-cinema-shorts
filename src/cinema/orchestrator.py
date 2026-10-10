"""Central production controller: deterministic stage sequencing, validation gates, bounded repairs.

Terminal statuses are truthful:
  COMPLETE - every mandatory final gate passed and deliverables exist
  BLOCKED  - needs resources / models / human intervention (resumable: `cinema resume JOB_ID`)
  FAILED   - final-repair budget exhausted without a valid film
"""
from __future__ import annotations

import fcntl
import logging
import random
import shutil
import subprocess
import traceback
import uuid
from datetime import datetime
from pathlib import Path

from .config import Config
from .model_registry import autonomous_readiness
from .packaging import package
from .pipeline import audio_stage, best_of, edit_stage, final_stage, image_stage, script_stage, video_stage
from .pipeline.context import Ctx, StageBlocked, StageFailed
from .resource_manager import JobPaused, ResourceManager, ResourceUnavailable
from .state_machine import AssetState, JobState, TERMINAL_JOB
from .storage.database import DB
from .story.generator import normalize_genre

log = logging.getLogger("cinema")
MAX_RESUMES = 5


def clean_scratch(cfg: Config, job_id: str) -> int:
    """Delete a finished job's scratch: rejected attempts, unused candidates, sampled frames, validator batches.
    Approved assets (script, references, keyframes, clips, audio, edit), reports, logs and outputs stay."""
    freed = 0
    for sub in ("work", "temp"):
        d = cfg.root / "projects" / job_id / sub
        if d.is_dir():
            freed += sum(p.stat().st_size for p in d.rglob("*") if p.is_file())
            shutil.rmtree(d, ignore_errors=True)
    return freed


class Orchestrator:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.db = DB(cfg.db_path)
        self.rm = ResourceManager(cfg, self.db)

    # ------------------------------------------------------------------ jobs
    def create_job(self, genre: str | None, duration: int, preset: str | None = None, style: str | None = None,
                   seed: int | None = None, test: bool = False, brief: str | None = None) -> str:
        if not (30 <= duration <= 60 or (test and 8 <= duration <= 30)):
            raise ValueError("duration must be 30-60 seconds")
        preset = preset or self.cfg.default_preset
        if preset not in self.cfg.presets:
            raise ValueError(f"unknown preset {preset}; choose from {sorted(self.cfg.presets)}")
        seed = seed if seed is not None else random.randrange(2 ** 31)
        g = brief.strip()[:60] if brief else normalize_genre(genre, random.Random(seed))
        job_id = datetime.now().strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:6]
        self.db.create_job(job_id, g, duration, preset, seed, {"style": style, "requested_genre": genre, "test": test,
                                                                "brief": brief.strip() if brief else None})
        (self.cfg.root / "projects" / job_id).mkdir(parents=True, exist_ok=True)
        self.db.event(job_id, "job created", genre=g, duration=duration, preset=preset, seed=seed)
        return job_id

    def resume(self, job_id: str) -> JobState:
        job = self.db.job(job_id)
        if job is None:
            raise KeyError(job_id)
        status = JobState(job["status"])
        if status in (JobState.COMPLETE, JobState.FAILED):
            raise RuntimeError(f"job {job_id} is {status}; it will not be restarted")
        (self.cfg.root / "projects" / job_id / "PAUSE").unlink(missing_ok=True)
        if status == JobState.PAUSED:  # owner pause/resume is not a failure-retry: no resume budget used
            return self.run(job_id)
        n = self.db.bump_resume(job_id)
        if n > MAX_RESUMES:
            raise RuntimeError(f"job {job_id} has been resumed {n - 1} times; refusing to loop. "
                               "Inspect its failure report and start a new job.")
        if status == JobState.BLOCKED:
            for a in self.db.assets(job_id):  # one more retry budget for every unfinished asset ...
                if a["status"] in (AssetState.BLOCKED, AssetState.RETRY_PENDING):
                    # ... except visuals that used all retries but have a usable attempt: those take the best-of
                    keep = (a["kind"] in best_of.KINDS and a["attempts"] >= self.cfg.retries.max_attempts
                            and not (a["feedback"] or "").startswith(best_of.NO_USABLE))
                    self.db.invalidate(job_id, a["asset_id"], "resume after BLOCKED", reset_attempts=not keep)
        return self.run(job_id)

    # ------------------------------------------------------------------ run
    def run(self, job_id: str) -> JobState:
        lock_path = self.cfg.root / "projects" / job_id / "run.lock"
        lock_path.parent.mkdir(parents=True, exist_ok=True)
        lf = lock_path.open("w")
        try:
            fcntl.flock(lf, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError(f"job {job_id} is already running in another process")
        caff = _caffeinate()
        import os
        pid_file = lock_path.with_name("run.pid")
        pid_file.write_text(str(os.getpid()))
        try:
            return self._run(job_id)
        finally:
            pid_file.unlink(missing_ok=True)
            if caff:
                caff.terminate()
            fcntl.flock(lf, fcntl.LOCK_UN)
            lf.close()

    def _run(self, job_id: str) -> JobState:
        ctx = Ctx(self.cfg, self.db, self.rm, job_id)
        _file_log(ctx.dirs.logs / "production.log")
        if JobState(ctx.job["status"]) in TERMINAL_JOB - {JobState.BLOCKED}:
            return JobState(ctx.job["status"])
        if (self.cfg.root / "projects" / job_id / "PAUSE").exists():
            self.db.set_job_status(job_id, JobState.PAUSED)
            return JobState.PAUSED
        ctx.state(JobState.PREFLIGHT)
        try:
            problems = autonomous_readiness(self.cfg, self.rm)
            if problems:
                raise StageBlocked("preflight: " + "; ".join(problems))
            feedback = ""
            for repair_round in range(self.cfg.retries.final_repairs + 1):
                sp = script_stage.run(ctx)
                image_stage.run(ctx, sp)
                video_stage.run(ctx, sp)
                sp = audio_stage.run(ctx, sp)
                final = edit_stage.run(ctx, sp, feedback)
                report = final_stage.run(ctx, sp, final)
                if report["passed"]:
                    ctx.approve("final", final, final, meta={"repair_rounds": repair_round})
                    out = package(ctx, sp, report)
                    ctx.state(JobState.COMPLETE)
                    ctx.info(f"COMPLETE: {out / 'final.mp4'}")
                    freed = clean_scratch(self.cfg, job_id)
                    ctx.info(f"cleaned scratch files ({freed / 1e6:.0f} MB); approved assets kept")
                    return JobState.COMPLETE
                repairs = [r["repair"] for r in report["mandatory_failed"] if r.get("repair")]
                ctx.warn(f"final validation failed ({len(report['mandatory_failed'])} mandatory checks); "
                         f"repair round {repair_round + 1}/{self.cfg.retries.final_repairs}",
                         failures=[r["check"] for r in report["mandatory_failed"]])
                if repair_round == self.cfg.retries.final_repairs:
                    break
                feedback = self._apply_repairs(ctx, repairs or [{"origin": "EDITING", "asset_id": "edl",
                                                                  "reason": "unattributed final failure"}])
            package(ctx, ctx.screenplay(), report, failed=True)
            raise StageFailed("final validation still failing after "
                              f"{self.cfg.retries.final_repairs} repair rounds: "
                              + "; ".join(r["check"] for r in report["mandatory_failed"]))
        except JobPaused:
            self.db.set_job_status(job_id, JobState.PAUSED)
            ctx.info("paused by owner at a safe point; models unloaded; send resume to continue")
            return JobState.PAUSED
        except StageFailed as exc:
            return self._terminal(ctx, JobState.FAILED, str(exc))
        except (StageBlocked, ResourceUnavailable) as exc:
            return self._terminal(ctx, JobState.BLOCKED, str(exc))
        except KeyboardInterrupt:
            ctx.warn("interrupted by user/system; job left resumable")
            raise
        except Exception as exc:  # unexpected: keep resumable, never claim success
            ctx.warn("unexpected error", traceback=traceback.format_exc())
            return self._terminal(ctx, JobState.BLOCKED, f"unexpected {type(exc).__name__}: {exc}")

    def _apply_repairs(self, ctx: Ctx, repairs: list[dict]) -> str:
        """Invalidate only what the failure points at, plus its dependents (edl/final always rebuild)."""
        seen, notes = set(), []
        rank = {"SCRIPT": 0, "IMAGE": 1, "VIDEO": 2, "AUDIO": 3, "EDITING": 4}
        for r in sorted(repairs, key=lambda r: rank.get(r["origin"], 9)):
            if r["asset_id"] in seen:
                continue
            seen.add(r["asset_id"])
            ids = self.db.invalidate(ctx.job_id, r["asset_id"], f"final QA: {r['reason']}"[:900], reset_attempts=True)
            ctx.info(f"repair {r['origin']}: invalidated {ids}")
            notes.append(f"{r['origin']}: {r['reason']}"[:300])
        for aid in ("edl", "final"):
            if aid not in seen:
                self.db.invalidate(ctx.job_id, aid, "rebuild after repair", reset_attempts=True)
        if any(r["origin"] == "SCRIPT" for r in repairs):
            draft = ctx.dirs.script / "draft.json"
            if draft.exists():
                draft.rename(ctx.dirs.script / f"draft_superseded_{datetime.now():%H%M%S}.json")
        return " | ".join(notes)

    def _terminal(self, ctx: Ctx, status: JobState, reason: str) -> JobState:
        self.db.set_job_status(ctx.job_id, status, error=reason[:4000])
        ctx.db.event(ctx.job_id, f"{status}: {reason}", "error")
        log.error("[%s] %s: %s", ctx.job_id, status, reason)
        from .packaging import failure_report
        failure_report(ctx, status, reason)
        return status


def _caffeinate():
    """Scoped sleep prevention for this process only (no global power settings are changed)."""
    import os
    try:
        return subprocess.Popen(["caffeinate", "-ims", "-w", str(os.getpid())], stdout=subprocess.DEVNULL,
                                stderr=subprocess.DEVNULL)
    except OSError:
        return None


def _file_log(path: Path) -> None:
    root = logging.getLogger("cinema")
    if any(getattr(h, "baseFilename", None) == str(path) for h in root.handlers):
        return
    h = logging.FileHandler(path)
    h.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    root.addHandler(h)
    root.setLevel(logging.INFO)
