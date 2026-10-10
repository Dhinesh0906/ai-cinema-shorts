"""Real subprocesses + real file locks (no mocks): isolation, exit verification, timeouts, serialization."""
import sys
import threading
import time

import pytest

from cinema.resource_manager import ResourceManager, WorkerFailed, memory_snapshot
from cinema.storage.database import DB


def _rm(cfg):
    return ResourceManager(cfg, DB(cfg.db_path))


def test_snapshot_has_pressure():
    s = memory_snapshot()
    assert s["total_gib"] > 8 and s["pressure"] in ("normal", "warn", "critical")


def test_worker_runs_isolated_and_is_measured(cfg, tmp_path):
    rm = _rm(cfg)
    r = rm.run_worker("tts", "dummy", [sys.executable, "-c", "x = bytearray(200*1024*1024); print('ok')"],
                      tmp_path / "w.log", None)
    assert r.exit_code == 0 and r.exited_clean
    assert r.peak_footprint_gib is not None and r.peak_footprint_gib >= 0.15
    run = rm.db.worker_runs()[-1]
    assert run["exited_clean"] == 1 and run["pressure_before"]


def test_worker_failure_and_grandchild_cleanup(cfg, tmp_path):
    import psutil
    rm = _rm(cfg)
    pidfile = tmp_path / "child.pid"
    code = (f"import subprocess,sys; p=subprocess.Popen(['sleep','60']); open({str(pidfile)!r},'w').write(str(p.pid)); "
            "sys.exit(3)")
    with pytest.raises(WorkerFailed):
        rm.run_worker("tts", "dummy", [sys.executable, "-c", code], tmp_path / "w.log", None)
    run = rm.db.worker_runs()[-1]
    assert run["exit_code"] == 3
    assert run["exited_clean"] == 0  # orphan in our process group was found and killed
    child = int(pidfile.read_text())
    time.sleep(0.2)
    assert not psutil.pid_exists(child) or psutil.Process(child).status() == psutil.STATUS_ZOMBIE


def test_worker_timeout_kills(cfg, tmp_path):
    rm = _rm(cfg)
    t0 = time.monotonic()
    with pytest.raises(WorkerFailed):
        rm.run_worker("tts", "dummy", [sys.executable, "-c", "import time; time.sleep(30)"], tmp_path / "w.log",
                      None, timeout=1)
    assert time.monotonic() - t0 < 25


def test_heavy_workers_never_overlap(cfg, tmp_path):
    spans = []

    def go(i):
        rm = _rm(cfg)  # separate manager = separate lock file description, like separate processes
        t0 = time.monotonic()
        rm.run_worker("tts", f"d{i}", [sys.executable, "-c", "import time; time.sleep(1.0)"], tmp_path / f"{i}.log", None)
        spans.append((t0, time.monotonic()))

    ths = [threading.Thread(target=go, args=(i,)) for i in range(3)]
    for t in ths:
        t.start()
    for t in ths:
        t.join()
    runs = sorted((r["started_at"], r["ended_at"]) for r in DB(cfg.db_path).worker_runs())
    assert len(runs) == 3
    for a, b in zip(runs, runs[1:]):
        assert b[0] >= a[1], "heavy workers overlapped"


def test_lock_not_reentrant(cfg, tmp_path):
    rm = _rm(cfg)
    with rm.heavy_lock("a"):
        with pytest.raises(RuntimeError):
            with rm.heavy_lock("b"):
                pass


def test_disk_reserve_blocks(cfg, tmp_path):
    from cinema.resource_manager import ResourceUnavailable
    cfg.memory.min_free_gib_disk = 10 ** 6
    with pytest.raises(ResourceUnavailable):
        _rm(cfg).preflight("tts", None)
