"""Central model execution manager.

Exactly one heavyweight model may run at a time on the 48 GB unified-memory Mac. Every heavy model
(Ollama LLM, MLX-Gen image/video, MLX-VLM, Whisper, Kokoro, Stable Audio, CLAP) runs either in an
isolated subprocess started here or inside the Ollama server, and only while this process holds the
exclusive heavy-model file lock. The sequence per run:

  lock -> memory/disk preflight -> launch isolated worker -> sample peak memory -> worker exit ->
  verify exit (and clean up only our own process group) -> verify memory recovery -> unlock

Measurements land in the worker_runs table so isolation can be audited (`cinema benchmark --stage memory`).
"""
from __future__ import annotations

import fcntl
import json
import os
import re
import signal
import shutil
import subprocess
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator

import httpx
import psutil

from .config import Config
from .storage.database import DB, now

GiB = 1024 ** 3

# Conservative peak estimates (GiB) used only for launch preflight; actual peaks are measured.
MODEL_ESTIMATE_GIB = {
    "writer": 22, "critic": 16, "vlm": 26, "image": 14, "video": 18, "whisper": 5,
    "tts": 2, "sfx": 8, "clap": 3,
}


class ResourceUnavailable(RuntimeError):
    """Raised when memory/disk never reached a safe level; the job becomes BLOCKED."""


class JobPaused(Exception):
    """The owner asked to pause; raised at a safe boundary (before the next heavy model starts)."""


class WorkerFailed(RuntimeError):
    def __init__(self, msg: str, result: "WorkerResult | None" = None):
        super().__init__(msg)
        self.result = result


def _sysctl_int(name: str) -> int | None:
    try:
        return int(subprocess.run(["sysctl", "-n", name], capture_output=True, text=True, timeout=5).stdout.strip())
    except (ValueError, subprocess.SubprocessError, OSError):
        return None


def memory_snapshot() -> dict:
    vm = psutil.virtual_memory()
    swap = psutil.swap_memory()
    level = _sysctl_int("kern.memorystatus_vm_pressure_level")  # 1 normal, 2 warn, 4 critical
    free_pct = _sysctl_int("kern.memorystatus_level")
    # psutil's "available" omits much of the reclaimable file cache (e.g. model weights just read
    # from disk); the kernel's memorystatus level is what macOS itself uses for pressure decisions.
    kernel_avail = vm.total * free_pct / 100 if free_pct is not None else vm.available
    return {
        "total_gib": round(vm.total / GiB, 2),
        "available_gib": round(max(vm.available, kernel_avail) / GiB, 2),
        "psutil_available_gib": round(vm.available / GiB, 2),
        "free_percent": free_pct,
        "pressure": {1: "normal", 2: "warn", 4: "critical"}.get(level or 0, f"unknown({level})"),
        "swap_used_gib": round(swap.used / GiB, 2),
    }


def disk_free_gib(path: Path) -> float:
    return shutil.disk_usage(path).free / GiB


@dataclass
class WorkerResult:
    name: str
    exit_code: int
    seconds: float
    peak_rss_gib: float
    peak_footprint_gib: float | None
    before: dict
    after: dict
    exited_clean: bool
    log: Path | None = None
    extra: dict = field(default_factory=dict)


class _Sampler(threading.Thread):
    """Samples RSS of a process tree (or of processes matching a name) until stopped."""

    def __init__(self, interval: float, pid: int | None = None, name_match: str | None = None):
        super().__init__(daemon=True)
        self.interval, self.pid, self.name_match = interval, pid, name_match
        self.peak = 0
        self._halt = threading.Event()

    def _procs(self) -> list[psutil.Process]:
        if self.pid is not None:
            try:
                root = psutil.Process(self.pid)
                return [root, *root.children(recursive=True)]
            except psutil.Error:
                return []
        out = []
        for p in psutil.process_iter(["name", "cmdline"]):
            try:
                if self.name_match and self.name_match in " ".join(p.info["cmdline"] or [p.info["name"] or ""]):
                    out.append(p)
            except psutil.Error:
                pass
        return out

    def run(self) -> None:
        while not self._halt.is_set():
            total = 0
            for p in self._procs():
                try:
                    total += p.memory_info().rss
                except psutil.Error:
                    pass
            self.peak = max(self.peak, total)
            self._halt.wait(self.interval)

    def stop(self) -> float:
        self._halt.set()
        self.join(timeout=5)
        return round(self.peak / GiB, 2)


class ResourceManager:
    def __init__(self, cfg: Config, db: DB):
        self.cfg, self.db = cfg, db
        self.lock_path = cfg.root / "state" / "heavy_model.lock"
        self.lock_path.parent.mkdir(parents=True, exist_ok=True)
        self._held = False

    # ------------------------------------------------------------ lock + preflight
    @contextmanager
    def heavy_lock(self, holder: str) -> Iterator[None]:
        if self._held:
            raise RuntimeError("heavy lock is not re-entrant: a heavy model is already running")
        f = self.lock_path.open("a+")
        try:
            fcntl.flock(f, fcntl.LOCK_EX)
            f.seek(0)
            f.truncate()
            f.write(json.dumps({"pid": os.getpid(), "holder": holder, "since": now()}))
            f.flush()
            self._held = True
            yield
        finally:
            self._held = False
            f.seek(0)
            f.truncate()
            f.flush()
            fcntl.flock(f, fcntl.LOCK_UN)
            f.close()

    def check_pause(self, job_id: str | None) -> None:
        if job_id and (self.cfg.root / "projects" / job_id / "PAUSE").exists():
            raise JobPaused(job_id)

    def preflight(self, role: str, job_id: str | None, disk_gib: float = 2.0) -> dict:
        need = MODEL_ESTIMATE_GIB.get(role, 8) + self.cfg.memory.launch_min_available_gib
        free_disk = disk_free_gib(self.cfg.root)
        if free_disk - disk_gib < self.cfg.memory.min_free_gib_disk:
            raise ResourceUnavailable(f"disk: {free_disk:.1f} GiB free; {role} needs {disk_gib} GiB "
                                      f"above the {self.cfg.memory.min_free_gib_disk} GiB reserve")
        deadline = time.monotonic() + self.cfg.memory.recovery_timeout_s
        snap = memory_snapshot()
        while time.monotonic() < deadline:
            snap = memory_snapshot()
            if snap["pressure"] == "normal" and snap["available_gib"] >= need:
                return snap
            self.db.event(job_id, f"waiting for memory before {role}", "warn", need_gib=need, **snap)
            time.sleep(10)
        raise ResourceUnavailable(f"memory never became safe for {role}: need {need} GiB available with "
                                  f"normal pressure, have {snap}")

    def _wait_recovery(self, job_id: str | None, before: dict) -> dict:
        target = min(before["available_gib"] - 3.0, before["total_gib"] * 0.5)
        deadline = time.monotonic() + self.cfg.memory.recovery_timeout_s
        snap = memory_snapshot()
        while time.monotonic() < deadline:
            snap = memory_snapshot()
            if snap["pressure"] == "normal" and snap["available_gib"] >= target:
                return snap
            time.sleep(2)
        self.db.event(job_id, "memory did not fully recover after worker", "warn", before=before, after=snap)
        if snap["pressure"] != "normal":
            raise ResourceUnavailable(f"memory pressure stayed {snap['pressure']} after worker exit")
        return snap

    # ------------------------------------------------------------ subprocess workers
    def run_worker(self, role: str, model: str, cmd: list[str], log: Path, job_id: str | None,
                   timeout: float | None = None, env: dict | None = None, cwd: Path | None = None,
                   disk_gib: float = 2.0) -> WorkerResult:
        log.parent.mkdir(parents=True, exist_ok=True)
        self.check_pause(job_id)
        with self.heavy_lock(f"{role}:{model}"):
            before = self.preflight(role, job_id, disk_gib)
            self._ensure_ollama_idle(job_id)
            start_offset = log.stat().st_size if log.exists() else 0
            started, t0 = now(), time.monotonic()
            with log.open("a") as lf:
                lf.write(f"\n### {started} {role} {model}\n$ {' '.join(map(str, cmd))}\n")
                lf.flush()
                proc = subprocess.Popen(["/usr/bin/time", "-l", *map(str, cmd)], stdout=lf,
                                        stderr=subprocess.STDOUT, start_new_session=True, cwd=cwd,
                                        env={**os.environ, **(env or {})})
                sampler = _Sampler(self.cfg.memory.sample_interval_s, pid=proc.pid)
                sampler.start()
                timed_out = False
                try:
                    proc.wait(timeout=timeout or self.cfg.memory.worker_timeout_s)
                except subprocess.TimeoutExpired:
                    timed_out = True
                finally:
                    exited_clean = self._reap(proc)  # always reap, even if sampling failed
                    try:
                        peak_rss = sampler.stop()
                    except Exception:
                        peak_rss = 0.0
            seconds = time.monotonic() - t0
            footprint = _parse_footprint(log, start_offset)
            after = self._wait_recovery(job_id, before)
            code = -9 if timed_out else proc.returncode
            res = WorkerResult(role, code, round(seconds, 2), peak_rss, footprint, before, after, exited_clean, log)
            self.db.log_worker(job_id, worker=role, model=model, command=" ".join(map(str, cmd))[:4000],
                               started_at=started, ended_at=now(), seconds=res.seconds, exit_code=code,
                               peak_rss_gib=peak_rss, peak_footprint_gib=footprint,
                               avail_before_gib=before["available_gib"], avail_after_gib=after["available_gib"],
                               pressure_before=before["pressure"], pressure_after=after["pressure"],
                               exited_clean=int(exited_clean), note="timeout" if timed_out else "")
        if timed_out:
            raise WorkerFailed(f"{role} worker timed out; see {log}", res)
        if code != 0:
            raise WorkerFailed(f"{role} worker exited {code}; see {log}", res)
        return res

    @staticmethod
    def _reap(proc: subprocess.Popen) -> bool:
        """Ensure the worker and anything it spawned (its own process group) is gone."""
        if proc.poll() is None:
            for sig, wait in ((signal.SIGTERM, 15), (signal.SIGKILL, 10)):
                try:
                    os.killpg(proc.pid, sig)
                except ProcessLookupError:
                    break
                try:
                    proc.wait(timeout=wait)
                    break
                except subprocess.TimeoutExpired:
                    continue
        leftovers = [p for p in psutil.process_iter(["pid"]) if _pgid(p.pid) == proc.pid]
        for p in leftovers:  # only processes in the group we created
            try:
                p.kill()
            except psutil.Error:
                pass
        return proc.poll() is not None and not leftovers

    # ------------------------------------------------------------ Ollama
    def ollama_loaded(self) -> list[str]:
        r = httpx.get(f"{self.cfg.ollama_url}/api/ps", timeout=10)
        r.raise_for_status()
        return [m["name"] for m in r.json().get("models", [])]

    def ollama_unload(self, model: str) -> None:
        httpx.post(f"{self.cfg.ollama_url}/api/generate", json={"model": model, "keep_alive": 0}, timeout=60)

    def _ensure_ollama_idle(self, job_id: str | None) -> None:
        try:
            loaded = self.ollama_loaded()
        except httpx.HTTPError:
            return  # server down = nothing loaded
        for m in loaded:
            self.db.event(job_id, f"unloading idle Ollama model {m} before heavy worker", "warn")
            self.ollama_unload(m)
        deadline = time.monotonic() + 60
        while time.monotonic() < deadline and self.ollama_loaded():
            time.sleep(1)
        if self.ollama_loaded():
            raise ResourceUnavailable(f"Ollama still has models loaded: {self.ollama_loaded()}")

    @contextmanager
    def ollama_session(self, role: str, model: str, job_id: str | None) -> Iterator[dict]:
        """Hold the heavy lock while an Ollama model is used; unload with keep_alive=0 and VERIFY.
        Cloud models use no local memory: no lock, no unload; the call is still logged."""
        self.check_pause(job_id)
        if self.cfg.is_cloud(model):
            started, t0, info = now(), time.monotonic(), {"calls": 0}
            try:
                yield info
            finally:
                self.db.log_worker(job_id, worker=role, model=model, command=f"ollama-cloud {info['calls']} calls",
                                   started_at=started, ended_at=now(), seconds=round(time.monotonic() - t0, 2),
                                   exit_code=0, peak_rss_gib=0.0, peak_footprint_gib=None,
                                   avail_before_gib=None, avail_after_gib=None, pressure_before=None,
                                   pressure_after=None, exited_clean=1, note="cloud (no local model loaded)")
            return
        with self.heavy_lock(f"{role}:{model}"):
            before = self.preflight(role, job_id)
            self._ensure_ollama_idle(job_id)
            sampler = _Sampler(self.cfg.memory.sample_interval_s, name_match="lib/ollama/llama-server")
            sampler.start()
            started, t0 = now(), time.monotonic()
            info: dict = {"calls": 0}
            try:
                yield info
            finally:
                peak = sampler.stop()
                self.ollama_unload(model)
                deadline = time.monotonic() + 90
                while time.monotonic() < deadline and model in self.ollama_loaded():
                    time.sleep(1)
                unloaded = model not in self.ollama_loaded()
                after = self._wait_recovery(job_id, before)
                self.db.log_worker(job_id, worker=role, model=model, command=f"ollama {info['calls']} calls",
                                   started_at=started, ended_at=now(), seconds=round(time.monotonic() - t0, 2),
                                   exit_code=0 if unloaded else 1, peak_rss_gib=peak, peak_footprint_gib=None,
                                   avail_before_gib=before["available_gib"], avail_after_gib=after["available_gib"],
                                   pressure_before=before["pressure"], pressure_after=after["pressure"],
                                   exited_clean=int(unloaded), note="ollama keep_alive=0 unload verified"
                                   if unloaded else "ollama unload NOT verified")
                if not unloaded:
                    raise ResourceUnavailable(f"Ollama did not unload {model}")


def _pgid(pid: int) -> int | None:
    try:
        return os.getpgid(pid)
    except (ProcessLookupError, PermissionError, OSError):
        return None


def _parse_footprint(log: Path, offset: int) -> float | None:
    try:
        with log.open("rb") as f:
            f.seek(offset)
            text = f.read().decode(errors="replace")
    except OSError:
        return None
    m = re.findall(r"(\d+)\s+peak memory footprint", text)
    return round(int(m[-1]) / GiB, 2) if m else None
