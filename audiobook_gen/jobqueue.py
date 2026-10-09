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
import re
import signal
import threading
import subprocess
import sys
import time
import uuid
from datetime import datetime, timedelta
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
QUEUE = ROOT / "work" / "queue"
JOBS = QUEUE / "jobs.json"
HEARTBEAT = QUEUE / "heartbeat"
SUPERVISOR_PID = QUEUE / "supervisor.pid"
RUNNER_PID = QUEUE / "runner.pid"
JOB_PID = QUEUE / "job.pid"                     # the job's process group, so Stop can end it too
STOPPED = QUEUE / "stopped"                      # you stopped the runner on purpose: the app must not restart it behind your back
SERVICE = "audiobook-queue.service"
SETTINGS = QUEUE / "settings.json"
SEND_EVERY = 300                                 # seconds between tries while the phone is out of reach
GUI_LOCK = QUEUE / "gui_generate.lock"          # reserved: anything in the app that holds the GPU for long writes its pid here and the queue waits

STALL = 600           # seconds without a new clip before the job is restarted
START_GRACE = 900     # model loading and emotion analysis write no clips for a while
MAX_RESTARTS = 6
POLL = 20
MIN_RECYCLE = 3.5 * 3600     # the speech process is never recycled sooner than this
RECYCLE_MARGIN = 300         # recycle this long before the uptime at which this machine's first slowdown began
SPEECH_LIMIT = QUEUE / "speech_limit.json"   # learned on this machine: {"seconds": the longest a speech process should run}
PACE_EVERY = 60       # seconds between pace checks / memory notes
MEM_EVERY = 300
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
        chapters: list[int] | None = None, not_before: str = "", window: str = "", send_to: str = "") -> dict:
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
           "chapters": chapters or None, "not_before": not_before, "window": window, "send_to": send_to, "sent": "", "status": "queued", "note": "",
           "progress": "", "restarts": 0, "created": time.strftime(FMT), "started": "", "finished": ""}

    def go():
        jobs = load(); jobs.append(job); _write(jobs)
    _locked(go)
    return job


def cancel(job_id: str) -> None:
    update(job_id, status="cancelled", note="cancelled by you")


def remove(job_id: str) -> None:
    remove_many([job_id])


def remove_many(ids: list[str]) -> int:
    """Take jobs off the list (not one that is being made: pause or cancel it first). Returns how many were removed."""
    gone = []

    def go():
        jobs = load()
        keep = []
        for j in jobs:
            if j["id"] in ids and j["status"] != "running":
                gone.append(j["id"])
            else:
                keep.append(j)
        _write(keep)
    _locked(go)
    return len(gone)


def hold(ids: list[str]) -> int:
    """Save for later / pause: a waiting job will not start, and one being made is stopped (its clips are kept)."""
    n = []

    def go():
        jobs = load()
        for j in jobs:
            if j["id"] in ids and j["status"] in ("queued", "paused", "running"):
                j["note"] = "paused by you" if j["status"] == "running" else "saved for later"
                j["status"] = "held"; n.append(1)
        _write(jobs)
    _locked(go)
    return len(n)


def resume(ids: list[str]) -> int:
    n = []

    def go():
        jobs = load()
        for j in jobs:
            if j["id"] in ids and j["status"] == "held":
                j["status"], j["note"] = "queued", ""; n.append(1)
        _write(jobs)
    _locked(go)
    return len(n)


def move(ids: list[str], where: str) -> None:
    """Reorder: the runner takes the first job that is ready, so earlier = sooner. where: top | up | down | bottom.
    The selected jobs keep their order among themselves. (A job already being made is not interrupted.)"""
    def go():
        jobs = load()
        sel = [j for j in jobs if j["id"] in ids]
        rest = [j for j in jobs if j["id"] not in ids]
        if where == "top":
            jobs = sel + rest
        elif where == "bottom":
            jobs = rest + sel
        else:
            order = list(jobs)
            seq = range(len(order)) if where == "up" else reversed(range(len(order)))
            for i in seq:
                if order[i]["id"] in ids:
                    k = i - 1 if where == "up" else i + 1
                    if 0 <= k < len(order) and order[k]["id"] not in ids:
                        order[i], order[k] = order[k], order[i]
            jobs = order
        _write(jobs)
    _locked(go)


def set_cover(work: str, cover: str) -> int:
    """Use this cover for every not-yet-finished job of the book in `work`."""
    w = str(Path(work).resolve())
    n = []

    def go():
        jobs = load()
        for j in jobs:
            if j["work"] == w and j["status"] not in ("done", "failed", "cancelled"):
                j["cover"] = str(cover); n.append(1)
        _write(jobs)
    _locked(go)
    return len(n)


def set_schedule(ids: list[str], not_before: str = "", window: str = "") -> int:
    """Give the jobs a new start time and daily window ("" and "" = as soon as possible, any time)."""
    if not_before:
        datetime.strptime(not_before, FMT)
    if window:
        _parse_window(window)
    n = []

    def go():
        jobs = load()
        for j in jobs:
            if j["id"] in ids and j["status"] not in ("done", "failed"):
                j["not_before"], j["window"] = not_before, window; n.append(1)
        _write(jobs)
    _locked(go)
    return len(n)


# ---------- sending to a phone (KDE Connect) ----------
def kde_devices() -> list[dict]:
    """Paired devices from `kdeconnect-cli -l`: [{"id", "name", "reachable"}]. Empty if KDE Connect is not installed."""
    try:
        out = subprocess.run(["kdeconnect-cli", "-l"], capture_output=True, text=True, timeout=25).stdout
    except Exception:
        return []
    devs = []
    for line in out.splitlines():
        m = re.match(r"^- (?P<name>.+?): (?P<id>\S+)(?: on \S+ via \S+)? \((?P<state>[^)]*)\)", line.strip())
        if m and "paired" in m["state"] and "not paired" not in m["state"]:
            devs.append({"id": m["id"], "name": m["name"], "reachable": "reachable" in m["state"] and "unreachable" not in m["state"]})
    return devs


def send_file(device: str, path: str) -> bool:
    """Share a file to the device. False if the device is out of reach or the share failed (it is tried again later)."""
    if not any(d["id"] == device and d["reachable"] for d in kde_devices()):
        return False
    try:
        return subprocess.run(["kdeconnect-cli", "-d", device, "--share", str(path)], capture_output=True, timeout=900).returncode == 0
    except Exception:
        return False


def default_device() -> str:
    try:
        return json.loads(SETTINGS.read_text()).get("device", "")
    except Exception:
        return ""


def save_default_device(device: str) -> None:
    try:
        QUEUE.mkdir(parents=True, exist_ok=True)
        SETTINGS.write_text(json.dumps({"device": device}))
    except OSError:
        pass


def set_send(ids: list[str], device: str) -> int:
    """Send the finished book to this device ("" = stop sending). A finished book that was sent is sent again."""
    n = []

    def go():
        jobs = load()
        for j in jobs:
            if j["id"] in ids and j["status"] != "cancelled":
                j["send_to"] = device
                if device:
                    j["sent"], j["send_try"] = "", 0
                n.append(1)
        _write(jobs)
    _locked(go)
    return len(n)


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


def speech_limit():
    """How long a speech process may run before it is replaced, learned from this machine's first slowdown (None = not learned yet)."""
    try:
        return float(json.loads(SPEECH_LIMIT.read_text())["seconds"])
    except (OSError, ValueError, KeyError, TypeError):
        return None


def learn_speech_limit(onset: float) -> float | None:
    """A speech process began to slow down `onset` seconds after it started. Recycle a little before that next time, but not
    sooner than MIN_RECYCLE (an earlier slowdown is not about uptime, and the pace check handles it). Only ever lowers the limit."""
    if onset < MIN_RECYCLE:
        return None
    new = max(MIN_RECYCLE, onset - RECYCLE_MARGIN)
    old = speech_limit()
    if old is not None and old <= new:
        return None
    try:
        QUEUE.mkdir(parents=True, exist_ok=True)
        SPEECH_LIMIT.write_text(json.dumps({"seconds": round(new), "learned": time.strftime("%Y-%m-%d %H:%M"),
                                            "slowdown_began_after": round(onset)}))
    except OSError:
        return None
    return new


def clip_times(work: Path, since: float = 0.0) -> list[float]:
    """Modification times of the clips made since `since`."""
    out = []
    try:
        with os.scandir(work / "clips") as it:
            for e in it:
                if e.name.endswith(".wav"):
                    t = e.stat().st_mtime
                    if t >= since:
                        out.append(t)
    except OSError:
        pass
    return out


class Pace:
    """Notices a speech process that has slowed to a fraction of its own best speed (clips still trickle in, so the plain
    "no clip for 10 minutes" stall check never fires)."""
    WINDOW = 600          # seconds over which clips are counted
    WARMUP = 1200         # judge only after the process has run this long
    MIN_PEAK = 20         # clips per window the process must once have reached before a slowdown means anything
    SLOW = 0.25           # slower than this share of its best window = slow

    def __init__(self, started: float):
        self.started, self.peak = started, 0

    def slow(self, times: list[float], now: float) -> bool:
        if now - self.started < self.WARMUP:
            return False
        n = sum(1 for t in times if t > now - self.WINDOW)
        self.peak = max(self.peak, n)
        return self.peak >= self.MIN_PEAK and n < self.peak * self.SLOW


def group_rss(pgid: int) -> float:
    """Memory (GB) held by every process of a process group (the job and its worker processes)."""
    total = 0
    for d in Path("/proc").iterdir():
        if d.name.isdigit():
            try:
                stat = (d / "stat").read_text().rsplit(")", 1)[1].split()
                if int(stat[2]) == pgid:
                    total += int(stat[21]) * os.sysconf("SC_PAGE_SIZE")
            except (OSError, ValueError, IndexError):
                pass
    return total / 2**30


class Runner:
    def __init__(self, poll: float = POLL, stall: float = STALL, grace: float = START_GRACE, max_restarts: int = MAX_RESTARTS, recycle: float | None = None,
                 foreign=foreign_synthesis, now=datetime.now, retry_wait: float = 30, sender=send_file, send_every: float = SEND_EVERY):
        self.poll, self.stall, self.grace, self.max_restarts, self.recycle = poll, stall, grace, max_restarts, recycle
        self.sender, self.send_every, self._sender_thread = sender, send_every, None
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
        proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, cwd=ROOT, env=env, start_new_session=True)
        try:
            JOB_PID.write_text(str(proc.pid))
        except OSError:
            pass
        return proc

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

    def _tick(self, job: dict, seconds: float) -> None:
        """Add time a stage spent running to the job's active time (waiting for its window or the GPU is not counted)."""
        cur = next((j for j in load() if j["id"] == job["id"]), None)
        if cur is not None:
            update(job["id"], active=round(float(cur.get("active", 0)) + max(0.0, seconds), 1))

    def _record(self, job: dict) -> None:
        """A finished book teaches the time estimate: its characters per engine against its active time."""
        try:
            from .runstats import record_job
            from .synth import resolve_voice
            cfg = yaml.safe_load(Path(job["config"]).read_text()) or {}
            only = set(job.get("chapters") or [])
            chars: dict[str, int] = {}
            for sg in json.loads((Path(job["work"]) / "segments.json").read_text()):
                if only and sg["chapter"] not in only:
                    continue
                e = resolve_voice(sg["speaker"], cfg)["engine"]
                chars[e] = chars.get(e, 0) + len(sg["text"])
            cur = next((j for j in load() if j["id"] == job["id"]), job)
            record_job(job["id"], chars, float(cur.get("active", 0)))
        except Exception:
            pass

    def _note_memory(self, job: dict, proc: subprocess.Popen, started: float) -> None:
        """One line in the job log: how much memory the speech process holds after how long (to see whether it grows)."""
        try:
            with open(QUEUE / f"{job['id']}.log", "a") as f:
                f.write(f"[memory] {group_rss(proc.pid):.1f} GB after {(time.time() - started) / 60:.0f} min\n")
        except OSError:
            pass

    def _watch(self, job: dict, proc: subprocess.Popen, stage: str) -> str:
        """Watch one stage. Returns "ok", "failed", "stalled", "window" (closed), or "cancelled"."""
        started = last_tick = last_check = last_mem = time.time()
        pace = Pace(started)
        while True:
            time.sleep(self.poll)
            self._beat()
            self._tick(job, time.time() - last_tick)
            last_tick = time.time()
            code = proc.poll()
            cur = next((j for j in load() if j["id"] == job["id"]), None)
            if cur is None or cur["status"] in ("cancelled", "held"):
                if code is None:
                    _kill_group(proc)
                return "cancelled" if cur is None or cur["status"] == "cancelled" else "held"
            if code is not None:
                return "ok" if code == 0 else "failed"
            if stage == "synth":
                prog = self._progress(job)
                if prog and prog != cur.get("progress"):
                    update(job["id"], progress=prog)
                if not in_window(job.get("window", ""), self.now()):
                    _kill_group(proc)
                    return "window"
                if time.time() - last_check >= PACE_EVERY:
                    last_check = time.time()
                    limit = self.recycle if self.recycle is not None else speech_limit()
                    if limit and time.time() - started > limit:
                        _kill_group(proc)
                        return "recycle"
                    if pace.slow(clip_times(Path(job["work"]), started), time.time()):
                        learn_speech_limit(time.time() - started - Pace.WINDOW)
                        _kill_group(proc)
                        return "slow"
                if time.time() - last_mem >= MEM_EVERY:
                    last_mem = time.time()
                    self._note_memory(job, proc, started)
                last = max(newest_clip(Path(job["work"])), started)
                if time.time() - started > self.grace and time.time() - last > self.stall:
                    learn_speech_limit(last - started)
                    _kill_group(proc)
                    return "stalled"
                if time.time() - started <= self.grace and time.time() - last > self.grace:
                    _kill_group(proc)
                    return "stalled"

    def _deliver(self) -> None:
        """Send each finished book that has a device and has not been sent. Out of reach: try again in SEND_EVERY seconds."""
        for j in load():
            if (j["status"] == "done" and j.get("send_to") and not j.get("sent") and Path(j["out"]).exists()
                    and time.time() - j.get("send_try", 0) >= self.send_every):
                update(j["id"], send_try=time.time())
                if self.sender(j["send_to"], j["out"]):
                    update(j["id"], sent=time.strftime(FMT), note="sent to your phone")
                else:
                    update(j["id"], note="waiting for your phone to be reachable")

    def deliver_pending(self) -> None:
        """Run the sending on its own thread: a big file or an absent phone must never stop the heartbeat (or the queue)."""
        if self._sender_thread is None or not self._sender_thread.is_alive():
            self._sender_thread = threading.Thread(target=self._deliver, daemon=True)
            self._sender_thread.start()

    def run_job(self, job: dict) -> None:
        fresh = next((j for j in load() if j["id"] == job["id"]), None)
        if fresh is None or fresh["status"] not in ACTIVE:              # held or cancelled in the instant before it started
            return
        update(job["id"], status="running", started=job.get("started") or time.strftime(FMT), note="")
        for stage, cmd in (("synth", self.synth_cmd(job)), ("assemble", self.assemble_cmd(job))):
            while True:
                cur = next((j for j in load() if j["id"] == job["id"]), job)
                if cur["status"] in ("cancelled", "held"):
                    return
                if stage == "synth" and not in_window(job.get("window", ""), self.now()):
                    update(job["id"], status="paused", note="waiting for its daily window")
                    return
                update(job["id"], note=f"{'making the speech' if stage == 'synth' else 'building the audiobook'}")
                proc = self._spawn(cmd, job)
                result = self._watch(job, proc, stage)
                if result == "ok":
                    break
                if result in ("cancelled", "held"):
                    return
                if result == "window":
                    update(job["id"], status="paused", note="paused: outside its daily window (finished clips are kept)")
                    return
                if result in ("recycle", "slow"):                # a planned fresh start, not a failure: finished clips are kept
                    update(job["id"], note=("restarted the speech process to keep it fast" if result == "recycle"
                                            else "speech had slowed down; restarted it"))
                    continue
                restarts = int(cur.get("restarts", 0)) + 1
                update(job["id"], restarts=restarts, note=f"{stage} {result}; restart {restarts} of {self.max_restarts}")
                if restarts > self.max_restarts:
                    update(job["id"], status="failed", finished=time.strftime(FMT),
                           note=f"gave up after {self.max_restarts} restarts; see work/queue/{job['id']}.log")
                    return
                time.sleep(self.retry_wait)
        cur = next((j for j in load() if j["id"] == job["id"]), job)
        if cur["status"] in ("cancelled", "held"):
            return
        done = Path(job["out"]).exists() and Path(job["out"]).stat().st_size > 0
        update(job["id"], status="done" if done else "failed", finished=time.strftime(FMT),
               note="" if done else "the audiobook file was not created", progress="finished" if done else "")
        if done:
            self._record(job)

    def step(self) -> bool:
        """One scheduling decision. Returns True if a job ran."""
        self._beat()
        self.deliver_pending()
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
        RUNNER_PID.write_text(str(os.getpid()))
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
    STOPPED.unlink(missing_ok=True)                  # a supervisor that is running means the runner should be too
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


def service_installed() -> bool:
    return (Path.home() / ".config" / "systemd" / "user" / SERVICE).exists()


def service_active() -> bool:
    try:
        return subprocess.run(["systemctl", "--user", "is-active", "--quiet", SERVICE], timeout=10).returncode == 0
    except Exception:
        return False


def _kill_pidfile(path: Path, must_contain: str) -> None:
    """End the process (group) named in a pid file, if it is still ours."""
    try:
        pid = int(path.read_text())
        if must_contain in Path(f"/proc/{pid}/cmdline").read_text():
            pg = os.getpgid(pid)
            if pg == os.getpgrp():                       # never signal our own group
                os.kill(pid, signal.SIGTERM)
                return
            try:
                os.killpg(pg, signal.SIGTERM)
            except ProcessLookupError:
                return
            for _ in range(30):
                if not _alive(pid):
                    return
                time.sleep(0.5)
            os.killpg(pg, signal.SIGKILL)
    except Exception:
        pass
    finally:
        path.unlink(missing_ok=True)


def stop_runner() -> str:
    """Stop the supervisor, the runner and the job being made. Finished clips are kept; the job resumes when the runner starts again."""
    QUEUE.mkdir(parents=True, exist_ok=True)
    STOPPED.write_text(time.strftime(FMT))
    if service_active():
        subprocess.run(["systemctl", "--user", "stop", SERVICE], timeout=120)       # ends everything in the service
    _kill_pidfile(SUPERVISOR_PID, "jobqueue")          # first, so nothing restarts the runner
    _kill_pidfile(JOB_PID, "audiobook_gen")
    _kill_pidfile(RUNNER_PID, "jobqueue")
    HEARTBEAT.unlink(missing_ok=True)
    for j in load():
        if j["status"] == "running":
            update(j["id"], status="queued", note="paused: the queue runner was stopped")
    return "Stopped the queue runner. Nothing will be made until you start it again; clips already made are kept."


def start_runner() -> str:
    STOPPED.unlink(missing_ok=True)
    if service_installed():
        subprocess.run(["systemctl", "--user", "start", SERVICE], timeout=60)
        return "Started the queue runner (system service)."
    ensure_supervisor()
    return "Started the queue runner."


def ensure_supervisor() -> bool:
    """Start the supervisor (and so the runner) if it is not running and you have not stopped it. Returns True if it was started."""
    if supervisor_running() or STOPPED.exists():
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
        elif j["status"] == "held":
            when = "when you resume it"
        phone = ("" if not j.get("send_to") else f"sent {j['sent'][-5:]}" if j.get("sent") else
                 "waiting for the phone" if j["status"] == "done" else "sends when done")
        rows.append([j["title"][:40], j["status"], j.get("progress", ""), when, j.get("window") or "any time", phone,
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
