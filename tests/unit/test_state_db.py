import pytest

from cinema.schemas import Decision
from cinema.state_machine import AssetState, JobState, TransitionError, can_transition_job
from cinema.storage.database import DB


def test_job_transitions():
    assert can_transition_job(JobState.PENDING, JobState.PREFLIGHT)
    assert can_transition_job(JobState.PREFLIGHT, JobState.GENERATING_VIDEO)        # resume skip-ahead
    assert can_transition_job(JobState.FINAL_VALIDATION, JobState.GENERATING_VIDEO)  # repair goes back
    assert not can_transition_job(JobState.EDITING, JobState.COMPLETE)               # must pass final validation
    assert not can_transition_job(JobState.COMPLETE, JobState.PREFLIGHT)
    assert not can_transition_job(JobState.FAILED, JobState.PREFLIGHT)
    assert can_transition_job(JobState.BLOCKED, JobState.PREFLIGHT)


def test_db_job_and_assets(tmp_path):
    db = DB(tmp_path / "x.sqlite3")
    db.create_job("j1", "horror", 45, "balanced", 7, {})
    db.set_job_status("j1", JobState.PREFLIGHT)
    with pytest.raises(TransitionError):
        db.set_job_status("j1", JobState.COMPLETE)
    db.ensure_asset("j1", "screenplay", "script", [])
    db.ensure_asset("j1", "keyframe:s1", "keyframe", ["screenplay"])
    db.ensure_asset("j1", "clip:s1", "clip", ["keyframe:s1"])
    db.ensure_asset("j1", "clip:s2", "clip", ["screenplay"])
    db.ensure_asset("j1", "edl", "edl", ["clip:s1", "clip:s2"])
    for a in ("screenplay", "keyframe:s1", "clip:s1", "clip:s2", "edl"):
        db.set_asset("j1", a, AssetState.GENERATING, attempts_inc=True)
        db.set_asset("j1", a, AssetState.VALIDATING)
        db.set_asset("j1", a, AssetState.PASSED)
    with pytest.raises(TransitionError):
        db.set_asset("j1", "edl", AssetState.GENERATING)  # PASSED must be invalidated first
    inv = db.invalidate("j1", "keyframe:s1", "bad face")
    assert set(inv) == {"keyframe:s1", "clip:s1", "edl"}          # only dependents, not clip:s2
    assert db.asset("j1", "clip:s2")["status"] == AssetState.PASSED
    assert db.asset("j1", "clip:s1")["attempts"] == 1
    db.invalidate("j1", "clip:s2", "repair", reset_attempts=True)
    assert db.asset("j1", "clip:s2")["attempts"] == 0
    db.log_decision("j1", Decision(asset_id="edl", stage="edit", criterion="x", evaluator="t", passed=False))
    assert db.decisions("j1")[0]["passed"] == 0


def test_bfs_path_through_states(cfg, tmp_path):
    from cinema.pipeline.context import _path
    assert _path(AssetState.PASSED, AssetState.GENERATING) == [AssetState.PASSED, AssetState.PENDING,
                                                               AssetState.GENERATING]
    assert _path(AssetState.VALIDATING, AssetState.GENERATING)[-1] == AssetState.GENERATING


def test_output_folder_named_by_date_and_title(tmp_path):
    from cinema.storage.files import film_dir_name, name_output_dir, output_dir
    jid = "20261010-161124-07da36"
    assert film_dir_name(jid, "Mirror: Part/2?") == "2026-10-10 Mirror Part2"           # filesystem-safe
    assert film_dir_name(jid, "The Mirror That Sees the Future 🪞") == "2026-10-10 The Mirror That Sees the Future"
    legacy = tmp_path / "outputs" / jid
    legacy.mkdir(parents=True)
    (legacy / "final.mp4").write_bytes(b"v")
    assert output_dir(tmp_path, jid) == legacy                      # before packaging: job-id folder
    named = name_output_dir(tmp_path, jid, "The Mirror That Sees the Future 🪞")
    assert named.name == "2026-10-10 The Mirror That Sees the Future" and not legacy.exists()
    assert output_dir(tmp_path, jid) == named and (named / "final.mp4").exists()
    assert name_output_dir(tmp_path, jid, "The Mirror That Sees the Future") == named   # idempotent
    other = name_output_dir(tmp_path, "20261010-170000-aaaaaa", "The Mirror That Sees the Future")
    assert other.name == "2026-10-10 The Mirror That Sees the Future (2)"           # same title, other film
