"""Telegram bot: owner-only, abort needs explicit yes, no double start, abort deletes only that job."""
import json

import pytest

from cinema import telegram_bot as tb


@pytest.fixture
def bot(cfg, tmp_path, monkeypatch):
    (tmp_path / "secrets").mkdir()
    (tmp_path / "secrets" / "telegram.json").write_text(json.dumps({"bot_token": "T", "chat_id": 42}))
    b = tb.Bot(cfg)
    b.sent = []
    monkeypatch.setattr(b, "send", lambda text: b.sent.append(text))
    return b


def _job(b, jid, status="GENERATING_VIDEO"):
    b.db.create_job(jid, "horror", 45, "balanced", 1, {})
    with b.db.conn() as c:
        c.execute("UPDATE jobs SET status=? WHERE id=?", (status, jid))
    (b.cfg.root / "projects" / jid).mkdir(parents=True)
    (b.cfg.root / "outputs" / jid).mkdir(parents=True)


def test_abort_requires_yes(bot, monkeypatch):
    _job(bot, "j1")
    monkeypatch.setattr(bot, "_job_pid", lambda jid: None)
    bot.handle("abort")
    assert bot.pending_abort == "j1" and "yes or no" in bot.sent[-1]
    bot.handle("no")
    assert bot.db.job("j1") and (bot.cfg.root / "projects" / "j1").exists()
    bot.handle("abort")
    bot.handle("YES")
    assert bot.db.job("j1") is None and not (bot.cfg.root / "projects" / "j1").exists()
    assert not (bot.cfg.root / "outputs" / "j1").exists()


def test_abort_deletes_only_that_job(bot, monkeypatch):
    _job(bot, "keep", "COMPLETE")
    _job(bot, "kill")
    monkeypatch.setattr(bot, "_job_pid", lambda jid: 123 if jid == "kill" else None)
    monkeypatch.setattr(tb.os, "kill", lambda pid, sig: None)
    calls = iter([123] + [None] * 100)
    monkeypatch.setattr(bot, "_job_pid", lambda jid: next(calls) if jid == "kill" else None)
    bot.pending_abort = "kill"
    monkeypatch.setattr(tb.time, "sleep", lambda s: None)
    bot.handle("yes")
    assert bot.db.job("keep") and bot.db.job("kill") is None


def test_no_double_start(bot, monkeypatch):
    _job(bot, "j2")
    monkeypatch.setattr(bot, "_job_pid", lambda jid: 99)
    started = []
    monkeypatch.setattr(tb.subprocess, "Popen", lambda *a, **k: started.append(a))
    bot.handle("start")
    assert not started and "already" in bot.sent[-1]


def test_publish_needs_confirmation_and_queues_both(bot, monkeypatch):
    bot.cfg.publish_platforms = ["youtube", "instagram"]
    _job(bot, "done1", "COMPLETE")
    (bot.cfg.root / "outputs" / "done1" / "final.mp4").write_bytes(b"x")
    bot.handle("publish")
    assert bot.pending_publish == "done1" and "yes or no" in bot.sent[-1]
    bot.handle("no")
    from cinema.publish import queue as pq
    assert pq.items(bot.cfg) == []
    bot.handle("publish")
    bot.handle("yes")
    its = pq.items(bot.cfg)
    assert [i["platform"] for i in its] == ["youtube", "instagram"]
    assert its[0]["not_before"] == its[1]["not_before"]                   # both platforms together
    bot.handle("publish")
    assert "already" in bot.sent[-1]
    bot.handle("hello")
    assert "Commands" in bot.sent[-1]


def _capture_popen(bot, monkeypatch):
    calls = []
    monkeypatch.setattr(bot, "running", lambda: None)
    monkeypatch.setattr(tb.subprocess, "Popen", lambda cmd, **k: calls.append(cmd))
    monkeypatch.setattr(tb.time, "sleep", lambda s: None)
    return calls


def test_start_parsing(bot, monkeypatch):
    calls = _capture_popen(bot, monkeypatch)
    bot.handle("start")
    c = calls[-1]
    assert 30 <= int(c[c.index("--duration") + 1]) <= 45 and c[c.index("--genre") + 1] == "random"
    bot.handle("start 40 fictional future story with future romance story")
    c = calls[-1]
    assert c[c.index("--duration") + 1] == "40"
    assert c[c.index("--topic") + 1] == "fictional future story with future romance story"
    bot.handle("start horror on a ghost ship")
    c = calls[-1]
    assert 30 <= int(c[c.index("--duration") + 1]) <= 45 and c[c.index("--topic") + 1] == "horror on a ghost ship"
    n = len(calls)
    bot.handle("start 90")
    assert len(calls) == n and "30-60" in bot.sent[-1]


def test_script_command(bot, monkeypatch):
    from cinema.fixtures import good
    _job(bot, "j3")
    monkeypatch.setattr(bot, "_job_pid", lambda jid: None)
    (bot.cfg.root / "projects" / "j3" / "script").mkdir()
    (bot.cfg.root / "projects" / "j3" / "script" / "screenplay.json").write_text(good().model_dump_json())
    bot.handle("script")
    assert "The Last Entry" in bot.sent[-1] and "1. [hook" in bot.sent[-1] and "Elias" in bot.sent[-1]


def test_story_history_roundtrip(cfg):
    from cinema.fixtures import good
    from cinema.story.history import avoid_list, record
    record(cfg, "jA", good())
    record(cfg, "jA", good())  # idempotent
    av = avoid_list(cfg)
    assert len(av) == 1 and "The Last Entry" in av[0] and "twist:" in av[0]



def test_queue_same_slot_per_film_2h_between_films(cfg):
    from cinema.publish import queue as pq
    cfg.publish_platforms = ["youtube", "instagram"]
    t0 = 1_000_000.0
    a = pq.enqueue(cfg, "A", now=t0)
    b = pq.enqueue(cfg, "B", now=t0)
    assert a[0]["not_before"] == a[1]["not_before"] == t0                 # YouTube + Instagram together
    assert b[0]["not_before"] == b[1]["not_before"] == t0 + 2 * 3600      # next film 2h later
    assert pq.enqueue(cfg, "A", now=t0) == []                             # never queued twice
    first = pq.due(cfg, now=t0)
    pq.finish(cfg, first["id"], result={"url": "u"}, now=t0)
    second = pq.due(cfg, now=t0 + 60)                                     # same film, other platform: no wait
    assert second["job_id"] == "A" and second["platform"] != first["platform"]
    pq.finish(cfg, second["id"], result={"url": "u"}, now=t0 + 60)
    assert pq.due(cfg, now=t0 + 3600) is None                             # next film waits for the 2h gap
    nxt = pq.due(cfg, now=t0 + 2 * 3600)
    assert nxt["job_id"] == "B"
    st = pq.finish(cfg, nxt["id"], error="boom", now=t0 + 2 * 3600)      # one platform fails
    sib = pq.due(cfg, now=t0 + 2 * 3600)                                   # the other still posts now
    assert sib["job_id"] == "B" and sib["id"] != nxt["id"]
    pq.finish(cfg, sib["id"], result={"url": "u"}, now=t0 + 2 * 3600)
    while st["status"] == "queued":                                        # failed one retries, then gives up
        again = pq.due(cfg, now=st["not_before"])
        assert again["id"] == nxt["id"]
        st = pq.finish(cfg, nxt["id"], error="boom", now=st["not_before"])
    assert st["status"] == "failed" and st["attempts"] == pq.MAX_ATTEMPTS


def test_interrupted_upload_is_not_retried(cfg):
    from cinema.publish import queue as pq
    pq.enqueue(cfg, "C", now=1.0)
    it = pq.due(cfg, now=1.0)
    hit = pq.recover_stuck(cfg)
    assert hit and hit[0]["id"] == it["id"] and pq.items(cfg)[0]["status"] == "failed"


def test_reach_metadata_fallback_format(cfg, monkeypatch):
    from cinema.publish import metadata as md
    out = cfg.root / "outputs" / "J"
    out.mkdir(parents=True)
    (out / "story.json").write_text(json.dumps({"title": "The Last Entry", "genre": "supernatural mystery",
                                                "logline": "A keeper answers a signal.", "hook": "The sea signals back."}))
    import cinema.llm as llm
    monkeypatch.setattr(llm, "chat_json", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("offline")))
    m = md.build(cfg, "J")
    assert m["title"] == "The Last Entry"
    assert "#Shorts" in m["youtube_description"] and "AI" in m["youtube_description"]
    assert "Flicker Velle" in m["youtube_description"] and "@flickervelle" in m["instagram_caption"]
    tags = [w for w in m["youtube_description"].split() if w.startswith("#")]
    assert 2 <= len(tags) <= 5                       # few, relevant hashtags
    assert (out / "publish_meta_v2.json").exists()   # cached


def test_pause_flag_stops_before_next_model(cfg, tmp_path):
    import sys
    from cinema.resource_manager import JobPaused, ResourceManager
    from cinema.storage.database import DB
    rm = ResourceManager(cfg, DB(cfg.db_path))
    (cfg.root / "projects" / "jp").mkdir(parents=True)
    rm.run_worker("tts", "d", [sys.executable, "-c", "print(1)"], tmp_path / "w.log", "jp")   # runs normally
    (cfg.root / "projects" / "jp" / "PAUSE").write_text("x")
    with pytest.raises(JobPaused):
        rm.run_worker("tts", "d", [sys.executable, "-c", "print(1)"], tmp_path / "w.log", "jp")
    assert len(rm.db.worker_runs()) == 1   # second model never started


def test_paused_state_transitions():
    from cinema.state_machine import JobState, can_transition_job
    assert can_transition_job(JobState.GENERATING_VIDEO, JobState.PAUSED)
    assert can_transition_job(JobState.PAUSED, JobState.PREFLIGHT)
    assert not can_transition_job(JobState.COMPLETE, JobState.PAUSED)


def test_default_publishes_youtube_only(cfg):
    from cinema.publish import queue as pq
    assert [i["platform"] for i in pq.enqueue(cfg, "D", now=1.0)] == ["youtube"]   # Instagram is posted by hand
