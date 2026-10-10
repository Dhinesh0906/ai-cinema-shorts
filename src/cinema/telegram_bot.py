"""Telegram control bot for the cinema pipeline (long polling; only the configured chat is obeyed).

Commands (case-insensitive):
  start [seconds] [genre]   launch a full autonomous film (default 45 random)
  status                    plain-language progress (cloud LLM summary of live job data; raw fallback)
  abort                     asks yes/no; "yes" stops the running job (and its GPU workers) and deletes its progress
  publish                   reserved (not implemented yet)
  help                      command list
Notifications: when a job ends COMPLETE (the video is sent), BLOCKED or FAILED.
"""
from __future__ import annotations

import json
import logging
import os
import shutil
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path

import httpx
from pydantic import BaseModel

from .config import Config, load_config
from .storage.database import DB

log = logging.getLogger("cinema.bot")
TERMINAL = {"COMPLETE", "BLOCKED", "FAILED"}
HELP = ("Commands:\n"
        "• start — random 30-45s film, random genre\n"
        "• start <seconds> <topic> — e.g. start 40 fictional future romance story (either part optional)\n"
        "• script — the approved screenplay of the current film\n"
        "• status — progress of the current film\n"
        "• pause — finish the current step, then stop and unload the models (nothing is lost)\n"
        "• resume — continue a paused film\n"
        "• abort — stop the current film and delete its progress (asks to confirm)\n"
        "• publish [job] — post the latest finished film (YouTube Shorts; Instagram caption included to post yourself); "
        "films are spaced 2h apart (asks to confirm)\n"
        "• queue — publishing schedule and results\n"
        "• help")


# Minutes per asset, measured on this M5 Pro (generation + validation); clips self-calibrate per run.
WEIGHT_MIN = {"script": 2.0, "char_ref": 1.5, "loc_ref": 1.5, "keyframe": 1.8, "clip": 9.0, "speech": 0.3,
              "sfx": 0.2, "ambience": 0.2, "music": 0.3, "edl": 1.0, "final": 4.0}


def progress(db: DB, root: Path, job: dict) -> dict:
    """Deterministic % done and ETA from asset states (never invented by the LLM)."""
    assets = db.assets(job["id"])
    kinds = {a["asset_id"]: a["kind"] for a in assets}
    shots = 10
    sp = root / "projects" / job["id"] / "script" / "screenplay.json"
    if sp.exists():
        try:
            shots = len(json.loads(sp.read_text())["shots"])
        except (ValueError, KeyError):
            pass
    # expected assets, including ones not created yet
    expected = {"script": 1, "keyframe": shots, "clip": shots, "edl": 1, "final": 1,
                "char_ref": max(1, sum(k == "char_ref" for k in kinds.values())),
                "loc_ref": max(1, sum(k == "loc_ref" for k in kinds.values())),
                "speech": max(shots, sum(k == "speech" for k in kinds.values()))}
    clip_min = WEIGHT_MIN["clip"]
    runs = [r for r in db.worker_runs(job["id"]) if r["worker"] == "video" and r["exit_code"] == 0]
    if runs:
        clip_min = sum(r["seconds"] for r in runs) / len(runs) / 60 + 1.5  # + VLM share
    w = dict(WEIGHT_MIN, clip=clip_min)
    total = sum(w.get(k, 0.3) * n for k, n in expected.items())
    done = sum(w.get(a["kind"], 0.3) for a in assets if a["status"] == "PASSED")
    # generated clips awaiting the batch validation already cost their render time
    done += sum(clip_min - 1.5 for a in assets if a["kind"] == "clip" and a["status"] == "VALIDATING")
    live = None
    if job["status"] == "GENERATING_VIDEO":  # partial credit for the clip being rendered
        vlog = root / "projects" / job["id"] / "logs" / "video.log"
        if vlog.exists():
            for line in reversed(vlog.read_text(errors="ignore").splitlines()[-200:]):
                if line.startswith("{") and '"total_steps"' in line:
                    try:
                        ev = json.loads(line)
                        live = ev["step"] / max(1, ev["total_steps"])
                        done += live * (clip_min - 1.5)
                    except (ValueError, KeyError):
                        pass
                    break
    if job["status"] == "COMPLETE":
        done = total
    pct = max(0, min(100, round(100 * done / total))) if total else 0
    remaining = max(0.0, total - done)
    eta = time.strftime("%H:%M", time.localtime(time.time() + remaining * 60))
    return {"percent": pct, "eta_minutes": round(remaining), "eta_clock": eta,
            "shots_done": sum(1 for a in assets if a["kind"] == "clip" and a["status"] == "PASSED"),
            "shots_generated": sum(1 for a in assets if a["kind"] == "clip" and a["status"] in ("VALIDATING", "PASSED")),
            "shots_total": shots, "current_shot_progress": round(live, 2) if live is not None else None}


def bar(pct: int, width: int = 12) -> str:
    filled = round(pct / 100 * width)
    return "▓" * filled + "░" * (width - filled)


# Expected, self-handled events that must not be reported as problems.
BENIGN = ("cloud video unavailable", "Cloudflare unavailable", "rendering locally", "waiting for memory")


def _with_timeout(fn, seconds: float):
    """Run fn in a daemon thread; raise TimeoutError if it doesn't finish (the thread is abandoned)."""
    box: dict = {}

    def run():
        try:
            box["v"] = fn()
        except Exception as exc:
            box["e"] = exc
    t = threading.Thread(target=run, daemon=True)
    t.start()
    t.join(seconds)
    if t.is_alive():
        raise TimeoutError(f"no answer within {seconds}s")
    if "e" in box:
        raise box["e"]
    return box["v"]


class StatusText(BaseModel):
    message: str


class Bot:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        sec = json.loads(cfg.resolve("secrets/telegram.json").read_text())
        self.token, self.chat_id = sec["bot_token"], int(sec["chat_id"])
        self.api = f"https://api.telegram.org/bot{self.token}"
        self.db = DB(cfg.db_path)
        self.state_path = cfg.root / "state" / "telegram_state.json"
        self.state = self._load_state()
        self.pending_abort: str | None = None
        self.pending_publish: str | None = None

    # ------------------------------------------------------------ telegram I/O
    def _load_state(self) -> dict:
        try:
            return json.loads(self.state_path.read_text())
        except (OSError, ValueError):
            return {"offset": 0, "notified": {}}

    def _save_state(self) -> None:
        self.state_path.write_text(json.dumps(self.state))

    def send(self, text: str) -> None:
        for chunk in [text[i:i + 3900] for i in range(0, len(text), 3900)] or [""]:
            try:
                httpx.post(f"{self.api}/sendMessage", data={"chat_id": self.chat_id, "text": chunk}, timeout=30)
            except httpx.HTTPError as exc:
                log.warning("send failed: %s", exc)

    def send_video(self, path: Path, caption: str) -> bool:
        if path.stat().st_size > 49 * 1024 * 1024:  # bot API upload limit is 50 MB
            self.send(f"{caption}\n(video is {path.stat().st_size / 1e6:.0f} MB — too large for Telegram; it is at {path})")
            return False
        try:
            with path.open("rb") as f:
                r = httpx.post(f"{self.api}/sendVideo", data={"chat_id": self.chat_id, "caption": caption[:1000],
                               "supports_streaming": "true"}, files={"video": (path.name, f, "video/mp4")},
                               timeout=600)
            return r.status_code == 200 and r.json().get("ok", False)
        except httpx.HTTPError as exc:
            log.warning("sendVideo failed: %s", exc)
            return False

    # ------------------------------------------------------------ jobs
    def current_job(self) -> dict | None:
        jobs = self.db.jobs()
        live = [j for j in jobs if j["status"] not in TERMINAL and self._job_pid(j["id"])]
        return live[0] if live else (jobs[0] if jobs else None)

    def _job_pid(self, job_id: str) -> int | None:
        pid_file = self.cfg.root / "projects" / job_id / "run.pid"
        if pid_file.exists():
            try:
                pid = int(pid_file.read_text())
                os.kill(pid, 0)
                return pid
            except (ValueError, ProcessLookupError, PermissionError):
                pass
        lock = self.cfg.root / "projects" / job_id / "run.lock"  # jobs started before run.pid existed
        if lock.exists():
            out = subprocess.run(["lsof", "-t", str(lock)], capture_output=True, text=True).stdout.split()
            pids = [int(p) for p in out if p.isdigit() and int(p) != os.getpid()]
            if pids:
                return pids[0]
        return None

    def running(self) -> dict | None:
        for j in self.db.jobs():
            if j["status"] not in TERMINAL and self._job_pid(j["id"]):
                return j
        return None

    # ------------------------------------------------------------ commands
    def cmd_start(self, args: list[str]) -> None:
        """start | start <seconds> | start <topic...> | start <seconds> <topic...>"""
        if (j := self.running()):
            self.send(f"A film is already being made ({j['id']}, {j['status']}). Send status or abort.")
            return
        import random
        if args and args[0].isdigit():
            duration, topic = int(args[0]), " ".join(args[1:]).strip()
            if not 30 <= duration <= 60:
                self.send("Duration must be 30-60 seconds.")
                return
        else:
            duration, topic = random.randint(30, 45), " ".join(args).strip()
        cmd = [sys.executable, "-m", "cinema.cli", "run", "--autonomous", "--duration", str(duration)]
        cmd += ["--topic", topic] if topic else ["--genre", "random"]
        log_path = self.cfg.root / "logs" / f"bot-run-{time.strftime('%Y%m%d-%H%M%S')}.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        subprocess.Popen(cmd, cwd=self.cfg.root, stdout=log_path.open("a"), stderr=subprocess.STDOUT,
                         stdin=subprocess.DEVNULL, start_new_session=True,
                         env={**os.environ, "PYTHONPATH": str(self.cfg.root / "src")})
        self.send(f"Starting a {duration}s film — " + (f"topic: \"{topic}\"" if topic else "random genre")
                  + ". Expect about 2-3 hours. I'll message you when it's done.")
        for _ in range(30):  # report the job id once it exists
            time.sleep(2)
            j = self.running()
            if j:
                self.send(f"Job {j['id']} is running. Send script once the story is approved.")
                return

    def status_data(self) -> dict | None:
        j = self.current_job()
        if not j:
            return None
        assets = self.db.assets(j["id"])
        by: dict[str, dict[str, int]] = {}
        for a in assets:
            by.setdefault(a["kind"], {}).setdefault(a["status"], 0)
            by[a["kind"]][a["status"]] += 1
        title = None
        sp = self.cfg.root / "projects" / j["id"] / "script" / "screenplay.json"
        if sp.exists():
            try:
                title = json.loads(sp.read_text())["story"]["title"]
            except (ValueError, KeyError):
                pass
        events = self.db.events(j["id"], 200)
        # only events since the latest (re)start: earlier errors were already handled by that resume
        last_start = max((i for i, e in enumerate(events) if e["message"] == "state PREFLIGHT"), default=0)
        current = events[last_start:]
        return {"job": j["id"], "title": title, "status": j["status"], "genre": j["genre"], "duration": j["duration"],
                "running": bool(self._job_pid(j["id"])), "started": j["created_at"], "error": j["error"],
                "assets": by, "recent_events": [f"{e['created_at'][11:19]} {e['message']}" for e in current[-8:]],
                "problems_since_restart": [e["message"] for e in current if e["level"] in ("warn", "error")
                                           and not any(k in e["message"] for k in BENIGN)][-5:],
                "progress": progress(self.db, self.cfg.root, j)}

    def cmd_status(self) -> None:
        data = self.status_data()
        if not data:
            self.send("No films yet. Send start to make one.")
            return
        try:
            msg = _with_timeout(lambda: self._llm_status(data), 20)
        except Exception as exc:  # cloud slow/unavailable: raw but readable
            log.warning("status LLM unavailable: %s", exc)
            msg = (f"{data['title'] or data['job']} — {data['status']}{' (running)' if data['running'] else ''}\n"
                   + "\n".join(f"{k}: {v}" for k, v in data["assets"].items())
                   + ("\nProblems: " + "; ".join(data["problems_since_restart"]) if data["problems_since_restart"]
                      else "\nProblems: none"))
        self._send_status(data, msg)

    def _llm_status(self, data: dict) -> str:
        from .llm import chat_json
        return chat_json(self.cfg, self.cfg.models.writer,
                            "You report the progress of an automated AI film pipeline to its owner on Telegram. "
                            "Be brief (3-6 short lines), concrete and honest: current stage, what is done "
                            "(e.g. 4/10 video shots approved), problems ONLY from problems_since_restart (if it is "
                            "empty, say there are no problems), and what happens next. No invented numbers.",
                            "Live job data:\n" + json.dumps(data, default=str), StatusText,
                            temperature=0.2, retries=0, timeout=20).message

    def _send_status(self, data: dict, msg: str) -> None:
        p = data["progress"]
        if data["status"] in TERMINAL:
            head = f"{data['status']} · {bar(p['percent'])} {p['percent']}%"
        else:
            hours, mins = divmod(p["eta_minutes"], 60)
            eta = f"{hours}h {mins}m" if hours else f"{mins}m"
            head = (f"{bar(p['percent'])} {p['percent']}% · ETA ~{eta} (≈{p['eta_clock']})\n"
                    f"Video shots: {p['shots_generated']}/{p['shots_total']} rendered, {p['shots_done']} approved"
                    + (f" · current shot {round(100 * p['current_shot_progress'])}%"
                       if p["current_shot_progress"] is not None else ""))
        self.send(head + "\n\n" + msg)

    def cmd_script(self) -> None:
        j = self.running() or self.current_job()
        if not j:
            self.send("No films yet. Send start to make one.")
            return
        sp = self.cfg.root / "projects" / j["id"] / "script" / "screenplay.json"
        if not sp.exists():
            self.send(f"The script for {j['id']} isn't approved yet (stage: {j['status']}). Try again in a minute.")
            return
        d = json.loads(sp.read_text())
        st = d["story"]
        lines = [f"🎬 {st['title']}  ({st['genre']}, {d['target_seconds']}s, {len(d['shots'])} shots)",
                 f"Logline: {st['logline']}", f"Twist: {st['twist']}", ""]
        lines += [f"👤 {c['name']} — {c['age']}, {c['role']}; {c['costume']}" for c in st["characters"]]
        lines += [f"📍 {loc['name']} — {loc['description']}" for loc in st["locations"]]
        lines.append("")
        for i, sh in enumerate(d["shots"], 1):
            lines.append(f"{i}. [{sh['beat']} · {sh['duration']}s] {sh['action']}")
            if sh["narration"].strip():
                lines.append(f"   🎙 “{sh['narration']}”")
            for dl in sh.get("dialogue", []):
                lines.append(f"   🗣 {dl['speaker_id']} ({dl['delivery']}): “{dl['text']}”")
        self.send("\n".join(lines))

    # ------------------------------------------------------------ publishing
    def missing_publish_creds(self) -> list[str]:
        need = {"youtube": {"secrets/youtube_client.json": "YouTube OAuth client", "secrets/youtube_token.json":
                            "YouTube sign-in (run: uv run cinema publish-auth youtube)"},
                "instagram": {"secrets/instagram.json": "Instagram user_id + access token"}}
        return [v for p in self.cfg.publish_platforms for k, v in need.get(p, {}).items()
                if not self.cfg.resolve(k).exists()]

    def cmd_publish(self, args: list[str]) -> None:
        from .publish import queue as pq
        done = [j for j in self.db.jobs() if j["status"] == "COMPLETE"
                and (self.cfg.root / "outputs" / j["id"] / "final.mp4").exists()]
        job = next((j for j in done if args and j["id"] == args[0]), done[0] if done and not args else None)
        if not job:
            self.send("No finished film to publish." if not args else f"No finished film {args[0]}.")
            return
        queued = {(i["job_id"], i["platform"]) for i in pq.items(self.cfg) if i["status"] != "failed"}
        if all((job["id"], p) in queued for p in self.cfg.publish_platforms):
            self.send(f"{job['id']} is already queued/published. Send queue.")
            return
        title_p = self.cfg.root / "outputs" / job["id"] / "youtube_title.txt"
        title = title_p.read_text().strip() if title_p.exists() else job["id"]
        missing = self.missing_publish_creds()
        warn = ("\n⚠️ Not set up yet: " + "; ".join(missing) + " — it will wait in the queue until they exist."
                if missing else "")
        try:
            from .publish.metadata import build
            meta = _with_timeout(lambda: build(self.cfg, job["id"]), 90)
            preview = (f"\n\n— YouTube —\nTitle: {meta['title']}\n{meta['youtube_description']}"
                       f"\n\n— Instagram{'' if 'instagram' in self.cfg.publish_platforms else ' (copy this when you post it yourself)'} —"
                       f"\n{meta['instagram_caption']}")
        except Exception as exc:
            preview = f"\n\n(metadata preview unavailable: {exc}; it will be generated at upload)"
        self.pending_publish = job["id"]
        where = " + ".join({"youtube": "YouTube Shorts", "instagram": "Instagram Reels"}[p]
                           for p in self.cfg.publish_platforms)
        self.send(f"Publish “{title}” ({job['id']}) PUBLICLY to {where} "
                  f"(at the next free slot; films are 2h apart)?{preview}\n\nReply yes or no.{warn}")

    def do_publish(self, job_id: str) -> None:
        from .publish import queue as pq
        added = pq.enqueue(self.cfg, job_id)
        if not added:
            self.send("Nothing new to queue.")
            return
        self.send("Queued:\n" + "\n".join(f"• {i['platform']} at {time.strftime('%a %H:%M', time.localtime(i['not_before']))}"
                                          for i in added))

    def cmd_queue(self) -> None:
        from .publish import queue as pq
        its = pq.items(self.cfg)
        if not its:
            self.send("Publishing queue is empty.")
            return
        lines = []
        for i in its[-12:]:
            when = time.strftime("%a %H:%M", time.localtime(i.get("published_at") or i["not_before"]))
            extra = (i["result"] or {}).get("url") or i["error"] or ""
            lines.append(f"{i['status']:<10} {i['platform']:<9} {when}  {i['job_id']}  {extra}"[:300])
        self.send("\n".join(lines))

    def publish_worker(self) -> None:
        from .publish import queue as pq
        for it in pq.recover_stuck(self.cfg):
            self.send(f"⚠️ {it['platform']} upload of {it['job_id']} was interrupted — check {it['platform']} "
                      "and send publish again if it isn't there.")
        while True:
            try:
                if not self.missing_publish_creds():
                    it = pq.due(self.cfg)
                    if it:
                        try:
                            res = pq.publish_item(self.cfg, it)
                            pq.finish(self.cfg, it["id"], result=res)
                            note = (" (YouTube kept it private: the API project needs Google's audit for public "
                                    "uploads)" if res.get("privacy") == "private" and self.cfg.publish_privacy == "public"
                                    else "")
                            self.send(f"📤 Published to {it['platform']}: {res.get('url')}{note}")
                        except Exception as exc:
                            st = pq.finish(self.cfg, it["id"], error=str(exc)[:500])
                            self.send(f"⚠️ {it['platform']} publish of {it['job_id']} failed ({st['status']}, "
                                      f"attempt {st['attempts']}/{pq.MAX_ATTEMPTS}): {str(exc)[:300]}")
            except Exception as exc:
                log.warning("publish worker error: %s", exc)
            time.sleep(60)

    # ------------------------------------------------------------ pause / resume
    @staticmethod
    def _worker_pids() -> set[int]:
        out = subprocess.run(["pgrep", "-f", "^/usr/bin/time -l"], capture_output=True, text=True).stdout.split()
        return {int(p) for p in out if p.isdigit()}

    def cmd_pause(self) -> None:
        j = self.running()
        if not j:
            self.send("Nothing is running.")
            return
        (self.cfg.root / "projects" / j["id"] / "PAUSE").write_text("pause requested\n")
        legacy = not (self.cfg.root / "projects" / j["id"] / "run.pid").exists()
        self.send("⏸ Pausing after the current step finishes (a video shot takes up to ~8 min). "
                  "Nothing in progress is lost. I'll confirm when the models are unloaded.")
        threading.Thread(target=self._finish_pause, args=(j["id"], legacy), daemon=True).start()

    def _finish_pause(self, job_id: str, legacy: bool) -> None:
        pid = self._job_pid(job_id)
        if legacy and pid:  # job started before pause support: stop it at the next model boundary ourselves
            current = self._worker_pids()
            while current & self._worker_pids():
                time.sleep(0.5)
            os.kill(pid, signal.SIGTERM)
            for w in self._worker_pids() - current:  # a model the old process just launched
                try:
                    os.killpg(w, signal.SIGTERM)
                except (ProcessLookupError, PermissionError):
                    pass
            for _ in range(30):
                if not self._job_pid(job_id):
                    break
                time.sleep(1)
            j = self.db.job(job_id)
            if j and j["status"] not in TERMINAL and j["status"] != "PAUSED":
                with self.db.conn() as c:
                    c.execute("UPDATE jobs SET status='PAUSED' WHERE id=?", (job_id,))
                self.db.event(job_id, "paused by owner at a safe point; models unloaded")
        else:
            for _ in range(3600):  # new-code jobs pause themselves at the next model boundary
                j = self.db.job(job_id)
                if not j or j["status"] in TERMINAL or j["status"] == "PAUSED" or not self._job_pid(job_id):
                    break
                time.sleep(2)
        time.sleep(3)
        busy = self._worker_pids()
        self.send(f"⏸ Paused {job_id}. Models unloaded — the Mac is back to normal"
                  + (" (a model process is still exiting)." if busy else ".") + " Send resume to continue.")

    def cmd_resume(self) -> None:
        if (j := self.running()):
            self.send(f"{j['id']} is already running ({j['status']}).")
            return
        paused = [j for j in self.db.jobs() if j["status"] in ("PAUSED", "BLOCKED")]
        if not paused:
            self.send("Nothing paused to resume.")
            return
        j = paused[0]
        (self.cfg.root / "projects" / j["id"] / "PAUSE").unlink(missing_ok=True)
        log_path = self.cfg.root / "logs" / f"bot-resume-{time.strftime('%Y%m%d-%H%M%S')}.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        subprocess.Popen([sys.executable, "-m", "cinema.cli", "resume", j["id"]], cwd=self.cfg.root,
                         stdout=log_path.open("a"), stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                         start_new_session=True, env={**os.environ, "PYTHONPATH": str(self.cfg.root / "src")})
        self.send(f"▶️ Resuming {j['id']} from where it stopped.")

    def cmd_abort(self) -> None:
        j = self.running() or self.current_job()
        if not j or j["status"] == "COMPLETE":
            self.send("Nothing to abort.")
            return
        self.pending_abort = j["id"]
        self.send(f"Abort {j['id']} ({j['status']}) and DELETE all its progress? Reply yes or no.")

    def do_abort(self, job_id: str) -> None:
        pid = self._job_pid(job_id)
        if pid:
            try:
                os.kill(pid, signal.SIGTERM)  # unwinds into the resource manager, which reaps GPU workers
            except ProcessLookupError:
                pass
            for _ in range(60):
                time.sleep(1)
                if not self._job_pid(job_id):
                    break
            else:
                try:
                    os.kill(pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
        for d in (self.cfg.root / "projects" / job_id, self.cfg.root / "outputs" / job_id):
            if d.exists():
                shutil.rmtree(d)
        with self.db.conn() as c:
            for table, col in (("assets", "job_id"), ("decisions", "job_id"), ("events", "job_id"),
                               ("worker_runs", "job_id"), ("jobs", "id")):
                c.execute(f"DELETE FROM {table} WHERE {col}=?", (job_id,))
        self.send(f"Aborted {job_id} and deleted its progress.")

    def handle(self, text: str) -> None:
        words = text.strip().lstrip("/").split()
        if not words:
            return
        cmd, args = words[0].lower(), words[1:]
        if self.pending_publish:
            job, self.pending_publish = self.pending_publish, None
            if cmd in ("yes", "y"):
                self.do_publish(job)
            else:
                self.send("Publishing cancelled.")
            return
        if self.pending_abort:
            job, self.pending_abort = self.pending_abort, None
            if cmd in ("yes", "y"):
                self.do_abort(job)
            else:
                self.send("Abort cancelled; the film keeps going.")
            return
        if cmd == "start":
            self.cmd_start(args)
        elif cmd == "status":
            self.cmd_status()
        elif cmd == "script":
            self.cmd_script()
        elif cmd == "pause":
            self.cmd_pause()
        elif cmd == "resume":
            self.cmd_resume()
        elif cmd == "abort":
            self.cmd_abort()
        elif cmd == "publish":
            self.cmd_publish(args)
        elif cmd == "queue":
            self.cmd_queue()
        else:
            self.send(HELP)

    # ------------------------------------------------------------ completion notifications
    def watch_jobs(self) -> None:
        if not self.state.get("notified"):  # first run: don't replay history
            self.state["notified"] = {j["id"]: j["status"] for j in self.db.jobs() if j["status"] in TERMINAL}
            self._save_state()
        while True:
            try:
                for j in self.db.jobs():
                    if j["status"] in TERMINAL and self.state["notified"].get(j["id"]) != j["status"]:
                        self.notify(j)
                        self.state["notified"][j["id"]] = j["status"]
                        self._save_state()
            except Exception as exc:
                log.warning("watcher error: %s", exc)
            time.sleep(30)

    def notify(self, j: dict) -> None:
        out = self.cfg.root / "outputs" / j["id"]
        if j["status"] == "COMPLETE":
            sp = self.cfg.root / "projects" / j["id"] / "script" / "screenplay.json"
            if sp.exists():
                try:
                    from .schemas import Screenplay
                    from .story.history import record
                    record(self.cfg, j["id"], Screenplay.model_validate_json(sp.read_text()))
                except Exception as exc:
                    log.warning("story history not recorded: %s", exc)
            title = (out / "youtube_title.txt").read_text().strip() if (out / "youtube_title.txt").exists() else j["id"]
            final = out / "final.mp4"
            self.send(f"✅ Your film is complete: {title}\nJob {j['id']} · {j['duration']}s {j['genre']}")
            if final.exists() and not self.send_video(final, title):
                self.send(f"(Couldn't upload the video; it's at {final})")
        else:
            self.send(f"⚠️ Film {j['id']} ended {j['status']}: {(j['error'] or '')[:600]}\n"
                      + ("Send start for a new one." if j["status"] == "FAILED" else
                         f"It can be resumed: uv run cinema resume {j['id']}"))

    # ------------------------------------------------------------ main loop
    def command_worker(self) -> None:
        while True:
            text = self.commands.get()
            try:
                self.handle(text)
            except Exception as exc:
                log.exception("command failed")
                self.send(f"Command failed: {exc}")

    def run(self) -> None:
        import queue
        self.commands: queue.Queue = queue.Queue()
        threading.Thread(target=self.command_worker, daemon=True).start()
        threading.Thread(target=self.watch_jobs, daemon=True).start()
        threading.Thread(target=self.publish_worker, daemon=True).start()
        log.info("bot polling")
        while True:
            try:
                r = httpx.get(f"{self.api}/getUpdates", params={"offset": self.state["offset"], "timeout": 50},
                              timeout=70)
                for u in r.json().get("result", []):
                    self.state["offset"] = u["update_id"] + 1
                    self._save_state()
                    m = u.get("message") or u.get("edited_message") or {}
                    if m.get("chat", {}).get("id") != self.chat_id:
                        continue  # only the owner's chat is obeyed
                    if m.get("text"):
                        self.commands.put(m["text"])
            except (httpx.HTTPError, ValueError) as exc:
                log.warning("poll error: %s", exc)
                time.sleep(5)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    for noisy in ("httpx", "httpcore"):  # their INFO lines contain request URLs, i.e. the bot token
        logging.getLogger(noisy).setLevel(logging.WARNING)
    Bot(load_config()).run()
