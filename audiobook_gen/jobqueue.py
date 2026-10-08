"""Scheduled, watchdogged audiobook generation.

Jobs live in work/queue/jobs.json. ONE runner executes them, one at a time, and the runner is the watchdog: it
restarts a job that crashes or makes no new clip for STALL seconds (a GPU hang looks like that), pauses it outside its
daily window, and waits while any other synthesis holds the GPU. A supervisor restarts the runner if it dies or stops
answering. So a job can only be queued with a watchdog, because nothing else runs jobs.

    python -m audiobook_gen.jobqueue add --work work/book --not-before "2026-10-09 23:00" --window 23:00-06:30
    python -m audiobook_gen.jobqueue list | cancel ID | run | supervise
"""
import argparse
import fcntl
import json
import os
import signal
import subprocess
import sys
import time
import uuid
from datetime import datetime, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
QUEUE = ROOT / "work" / "queue"
JOBS = QUEUE / "jobs.json"
HEARTBEAT = QUEUE / "heartbeat"
SUPERVISOR_PID = QUEUE / "supervisor.pid"
GUI_LOCK = QUEUE / "gui_generate.lock"          # the GUI's own Generate button holds this while it uses the GPU

STALL = 600           # seconds without a new clip before the job is restarted
START_GRACE = 900     # model loading and emotion analysis write no clips for a while
MAX_RESTARTS = 6
POLL = 20
ACTIVE = ("queued", "paused", "running")
FMT = "%Y-%m-%d %H:%M"


# ---------- the job file ----------
def _locked(fn):
    QUEUE.mkdir(parents=True, exist_ok=True)
    with open(QUEUE / "jobs.lock", "w") as lk:
        fcntl.flock(lk, fcntl.LOCK_EX)
        try:
            return fn()
        finally:
            fcntl.flock(lk, fcntl.LOCK_UN)


def load() -> list[dict]:
    try:
        return json.loads(JOBS.read_text())
    except Exception:
        return []


def _write(jobs: list[dict]) -> None:
    tmp = JOBS.with_suffix(".tmp")
    tmp.write_text(json.dumps(jobs, indent=1, ensure_ascii=False))
    tmp.replace(JOBS)


def update(job_id: str, **fields) -> None:
    def go():
        jobs = load()
        for j in jobs:
            if j["id"] == job_id:
                j.update(fields)
        _write(jobs)
    _locked(go)


def add(work: str, title: str = "", author: str = "", cover: str = "", config: str = "", out: str = "",
        chapters: list[int] | None = None, not_before: str = "", window: str = "") -> dict:
    """Queue a generation. not_before: "YYYY-MM-DD HH:MM" local time or "". window: "23:00-06:30" or ""."""
    w = Path(work).resolve()
    if not (w / "segments.json").exists():
        raise ValueError("Parse the book on the Cast tab first.")
    if not_before:
        datetime.strptime(not_before, FMT)
    if window:
        _parse_window(window)
    job = {"id": uuid.uuid4().hex[:8], "title": title or w.name, "author": author, "cover": cover, "work": str(w),
           "config": config or str(w / "config.yaml"), "out": out or str(ROOT / "out" / f"{w.name}.m4b"),
           "chapters": chapters or None, "not_before": not_before, "window": window, "status": "queued", "note": "",
           "progress": "", "restarts": 0, "created": time.strftime(FMT), "started": "", "finished": ""}

    def go():
        jobs = load(); jobs.append(job); _write(jobs)
    _locked(go)
    return job


def cancel(job_id: str) -> None:
    update(job_id, status="cancelled", note="cancelled by you")


def remove(job_id: str) -> None:
    _locked(lambda: _write([j for j in load() if j["id"] != job_id or j["status"] == "running"]))


# ---------- scheduling ----------
def _parse_window(text: str) -> tuple[int, int]:
    a, b = text.split("-")
    mins = lambda s: int(s.split(":")[0]) * 60 + int(s.split(":")[1])
    return mins(a), mins(b)


def in_window(window: str, now: datetime) -> bool:
    if not window:
        return True
    start, stop = _parse_window(window)
    m = now.hour * 60 + now.minute
    return (m >= start or m < stop) if start > stop else (start <= m < stop)


def due(job: dict, now: datetime) -> bool:
    if job["status"] not in ACTIVE:
        return False
    if job.get("not_before") and now < datetime.strptime(job["not_before"], FMT):
        return False
    return in_window(job.get("window", ""), now)


def next_start(job: dict, now: datetime) -> str:
    """When a waiting job may start (for the table)."""
    t = datetime.strptime(job["not_before"], FMT) if job.get("not_before") else now
    t = max(t, now)
    if job.get("window") and not in_window(job["window"], t):
        start, _ = _parse_window(job["window"])
        t = t.replace(hour=start // 60, minute=start % 60, second=0)
        if t < now:
            t += timedelta(days=1)
    return t.strftime(FMT)


# ---------- processes ----------
def _alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def foreign_synthesis() -> str:
    """Something other than this runner is using the GPU: a synth started by hand, or the GUI's Generate button."""
    try:
        pid = int(GUI_LOCK.read_text().split()[0])
        if _alive(pid):
            return "the Generate button in the app"
        GUI_LOCK.unlink(missing_ok=True)
    except Exception:
        pass
    out = subprocess.run(["ps", "-eo", "pid,args"], capture_output=True, text=True).stdout
    for line in out.splitlines():
        if "-m audiobook_gen synth" in line and "ps -eo" not in line:
            return "another synthesis that is already running"
    return ""


def _kill_group(proc: subprocess.Popen) -> None:
    """Stop the job and its model-worker children."""
    try:
        pg = os.getpgid(proc.pid)
        os.killpg(pg, signal.SIGTERM)
        for _ in range(30):
            if proc.poll() is not None:
                break
            time.sleep(0.5)
        if proc.poll() is None:
            os.killpg(pg, signal.SIGKILL)
        proc.wait(timeout=10)
    except (ProcessLookupError, PermissionError):
        pass


def newest_clip(work: Path) -> float:
    newest = 0.0
    try:
        with os.scandir(work / "clips") as it:
            for e in it:
                if e.name.endswith(".wav"):
                    newest = max(newest, e.stat().st_mtime)
    except OSError:
        pass
    return newest


class Runner:
    def __init__(self, poll: float = POLL, stall: float = STALL, grace: float = START_GRACE, max_restarts: int = MAX_RESTARTS,
                 foreign=foreign_synthesis, now=datetime.now, retry_wait: float = 30):
        self.poll, self.stall, self.grace, self.max_restarts = poll, stall, grace, max_restarts
        self.foreign, self.now, self.retry_wait = foreign, now, retry_wait

    # commands (replaceable in tests)
    def synth_cmd(self, job: dict) -> list[str]:
        cmd = [sys.executable, "-m", "audiobook_gen", "synth", "x", "--work", job["work"], "--config", job["config"]]
        return cmd + (["--chapters", ",".join(map(str, job["chapters"]))] if job.get("chapters") else [])

    def assemble_cmd(self, job: dict) -> list[str]:
        cmd = [sys.executable, "-m", "audiobook_gen", "assemble", "x", "--work", job["work"], "--config", job["config"],
               "--title", job["title"], "--author", job.get("author", ""), "--out", job["out"]]
        if job.get("cover"):
            cmd += ["--cover", job["cover"]]
        return cmd + (["--chapters", ",".join(map(str, job["chapters"]))] if job.get("chapters") else [])

    def _spawn(self, cmd: list[str], job: dict):
        QUEUE.mkdir(parents=True, exist_ok=True)
        log = open(QUEUE / f"{job['id']}.log", "ab")
        env = {**os.environ, "PYTHONPATH": str(ROOT), "PYTHONUNBUFFERED": "1"}
        return subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, cwd=ROOT, env=env, start_new_session=True)

    def _beat(self) -> None:
        try:
            QUEUE.mkdir(parents=True, exist_ok=True)
            HEARTBEAT.write_text(str(time.time()))
        except OSError:
            pass

    def _progress(self, job: dict) -> str:
        try:
            tail = (QUEUE / f"{job['id']}.log").read_bytes()[-4000:].decode("utf-8", "ignore")
            for line in reversed(tail.replace("\r", "\n").splitlines()):
                if line.startswith("[progress]"):
                    return line.split("]", 1)[1].strip()
        except OSError:
            pass
        return ""

    def _watch(self, job: dict, proc: subprocess.Popen, stage: str) -> str:
        """Watch one stage. Returns "ok", "failed", "stalled", "window" (closed), or "cancelled"."""
        started = time.time()
        while True:
            time.sleep(self.poll)
            self._beat()
            code = proc.poll()
            cur = next((j for j in load() if j["id"] == job["id"]), None)
            if cur is None or cur["status"] == "cancelled":
                if code is None:
                    _kill_group(proc)
                return "cancelled"
            if code is not None:
                return "ok" if code == 0 else "failed"
            if stage == "synth":
                prog = self._progress(job)
                if prog and prog != cur.get("progress"):
                    update(job["id"], progress=prog)
                if not in_window(job.get("window", ""), self.now()):
                    _kill_group(proc)
                    return "window"
                last = max(newest_clip(Path(job["work"])), started)
                if time.time() - started > self.grace and time.time() - last > self.stall:
                    _kill_group(proc)
                    return "stalled"
                if time.time() - started <= self.grace and time.time() - last > self.grace:
                    _kill_group(proc)
                    return "stalled"

    def run_job(self, job: dict) -> None:
        update(job["id"], status="running", started=job.get("started") or time.strftime(FMT), note="")
        for stage, cmd in (("synth", self.synth_cmd(job)), ("assemble", self.assemble_cmd(job))):
            while True:
                cur = next((j for j in load() if j["id"] == job["id"]), job)
                if cur["status"] == "cancelled":
                    return
                if stage == "synth" and not in_window(job.get("window", ""), self.now()):
                    update(job["id"], status="paused", note="waiting for its daily window")
                    return
                update(job["id"], note=f"{'making the speech' if stage == 'synth' else 'building the audiobook'}")
                proc = self._spawn(cmd, job)
                result = self._watch(job, proc, stage)
                if result == "ok":
                    break
                if result == "cancelled":
                    return
                if result == "window":
                    update(job["id"], status="paused", note="paused: outside its daily window (finished clips are kept)")
                    return
                restarts = int(cur.get("restarts", 0)) + 1
                update(job["id"], restarts=restarts, note=f"{stage} {result}; restart {restarts} of {self.max_restarts}")
                if restarts > self.max_restarts:
                    update(job["id"], status="failed", finished=time.strftime(FMT),
                           note=f"gave up after {self.max_restarts} restarts; see work/queue/{job['id']}.log")
                    return
                time.sleep(self.retry_wait)
        done = Path(job["out"]).exists() and Path(job["out"]).stat().st_size > 0
        update(job["id"], status="done" if done else "failed", finished=time.strftime(FMT),
               note="" if done else "the audiobook file was not created", progress="finished" if done else "")

    def step(self) -> bool:
        """One scheduling decision. Returns True if a job ran."""
        self._beat()
        for j in load():                                            # a job left "running" by a dead runner starts over
            if j["status"] == "running":
                update(j["id"], status="queued", note="resuming after a restart")
        now = self.now()
        jobs = load()
        ready = [j for j in jobs if due(j, now)]
        if not ready:
            return False
        busy = self.foreign()
        if busy:
            update(ready[0]["id"], note=f"waiting: {busy} holds the GPU")
            return False
        self.run_job(ready[0])
        return True

    def run_forever(self) -> None:
        QUEUE.mkdir(parents=True, exist_ok=True)
        print(f"[queue] runner started {time.strftime(FMT)}", flush=True)
        while True:
            try:
                if not self.step():
                    time.sleep(self.poll)
            except Exception as e:                                  # a bug in one job must not stop the runner
                print(f"[queue] error: {e!r}", flush=True)
                time.sleep(self.poll)


# ---------- supervisor ----------
def supervise(restart_after: float = 180) -> None:
    """Keep the runner alive: restart it if it exits or stops updating its heartbeat."""
    QUEUE.mkdir(parents=True, exist_ok=True)
    SUPERVISOR_PID.write_text(str(os.getpid()))
    child = None
    while True:
        stale = (time.time() - float(HEARTBEAT.read_text())) > restart_after if HEARTBEAT.exists() else False
        if child is not None and child.poll() is None and stale:
            print("[supervisor] runner not answering - restarting it", flush=True)
            _kill_group(child); child = None
        if child is None or child.poll() is not None:
            if child is not None:
                print(f"[supervisor] runner exited ({child.returncode}) - restarting", flush=True)
            HEARTBEAT.write_text(str(time.time()))
            child = subprocess.Popen([sys.executable, "-m", "audiobook_gen.jobqueue", "run"], cwd=ROOT, start_new_session=True,
                                     env={**os.environ, "PYTHONPATH": str(ROOT), "PYTHONUNBUFFERED": "1"})
        time.sleep(15)


def supervisor_running() -> bool:
    try:
        pid = int(SUPERVISOR_PID.read_text())
        return _alive(pid) and "jobqueue" in Path(f"/proc/{pid}/cmdline").read_text()
    except Exception:
        return False


def ensure_supervisor() -> bool:
    """Start the supervisor (and so the runner) if it is not running. Returns True if it was started."""
    if supervisor_running():
        return False
    QUEUE.mkdir(parents=True, exist_ok=True)
    log = open(QUEUE / "runner.log", "ab")
    subprocess.Popen([sys.executable, "-m", "audiobook_gen.jobqueue", "supervise"], cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                     start_new_session=True, env={**os.environ, "PYTHONPATH": str(ROOT), "PYTHONUNBUFFERED": "1"})
    return True


def runner_alive() -> bool:
    try:
        return time.time() - float(HEARTBEAT.read_text()) < 90
    except Exception:
        return False


def now_running() -> dict | None:
    """The job being made right now: title, clips done/total, percent, seconds since its last clip, when it started."""
    for j in load():
        if j["status"] == "running":
            done = total = pct = 0
            try:
                done, total = (int(x) for x in j.get("progress", "").split("/"))
                pct = round(100 * done / total) if total else 0
            except ValueError:
                pass
            newest = newest_clip(Path(j["work"]))
            return {"id": j["id"], "title": j["title"], "done": done, "total": total, "pct": pct, "note": j.get("note", ""),
                    "since_clip": (time.time() - newest) if newest else None, "started": j.get("started", ""), "work": j["work"]}
    return None


def table(now: datetime | None = None) -> list[list]:
    """Rows for the Queue tab."""
    now = now or datetime.now()
    rows = []
    for j in load():
        when = ""
        if j["status"] in ("queued", "paused"):
            when = next_start(j, now)
        rows.append([j["title"][:40], j["status"], j.get("progress", ""), when, j.get("window") or "any time",
                     j.get("note", ""), j["id"]])
    return rows


def main(argv=None) -> None:
    p = argparse.ArgumentParser(prog="audiobook_gen.jobqueue")
    p.add_argument("cmd", choices=["add", "list", "cancel", "run", "supervise"])
    p.add_argument("arg", nargs="?")
    p.add_argument("--work"); p.add_argument("--title", default=""); p.add_argument("--author", default="")
    p.add_argument("--cover", default=""); p.add_argument("--config", default=""); p.add_argument("--out", default="")
    p.add_argument("--chapters"); p.add_argument("--not-before", default=""); p.add_argument("--window", default="")
    a = p.parse_args(argv)
    if a.cmd == "add":
        ch = [int(x) for x in a.chapters.split(",")] if a.chapters else None
        job = add(a.work, a.title, a.author, a.cover, a.config, a.out, ch, a.not_before, a.window)
        print("queued", job["id"])
        ensure_supervisor()
    elif a.cmd == "list":
        for r in table():
            print(" | ".join(map(str, r)))
    elif a.cmd == "cancel":
        cancel(a.arg)
    elif a.cmd == "run":
        Runner().run_forever()
    else:
        supervise()


if __name__ == "__main__":
    main()
