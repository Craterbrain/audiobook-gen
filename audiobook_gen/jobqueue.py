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
import urllib.request
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
HALTED = QUEUE / "halted"                    # the queue paused itself after repeated failures; its text says why
FAIL_HALT = 3                                # failed books in a row that halt the queue
QC_STALL = 1200                              # seconds without output from the quality check
ASSEMBLE_STALL = 1500                        # seconds in which the audiobook build touches no file
NTFY_SERVER = "https://ntfy.sh"
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


def settings() -> dict:
    try:
        return json.loads(SETTINGS.read_text())
    except Exception:
        return {}


def save_settings(**kw) -> None:
    """Change some settings and keep the others."""
    try:
        QUEUE.mkdir(parents=True, exist_ok=True)
        SETTINGS.write_text(json.dumps({**settings(), **kw}))
    except OSError:
        pass


def default_device() -> str:
    return settings().get("device", "")


def save_default_device(device: str) -> None:
    save_settings(device=device)


def ntfy_topic() -> str:
    """The ntfy topic alerts go to (AUDIOBOOK_NTFY overrides the saved one). Anyone who knows the name can read it."""
    return (os.environ.get("AUDIOBOOK_NTFY") or settings().get("ntfy_topic", "")).strip()


def notify(text: str, title: str = "Audiobook queue", priority: str = "default", tags: list[str] | None = None, desktop: bool = True) -> bool:
    """Tell the user: a message on the ntfy topic (reaches the phone) and a desktop notice. Never raises, never waits long."""
    ok = False
    topic = ntfy_topic()
    if topic:
        try:
            body = json.dumps({"topic": topic, "title": title, "message": text, "priority": {"low": 2, "default": 3, "high": 4, "urgent": 5}.get(priority, 3),
                               "tags": tags or []}).encode()
            req = urllib.request.Request(NTFY_SERVER, data=body, headers={"Content-Type": "application/json"})
            ok = urllib.request.urlopen(req, timeout=15).status == 200
        except Exception:
            ok = False
    if desktop:
        try:
            subprocess.run(["notify-send", "-a", "Audiobook queue", title, text], timeout=5, capture_output=True)
        except Exception:
            pass
    return ok


def halted() -> str:
    """Why the queue paused itself ("" = it has not)."""
    try:
        return HALTED.read_text().strip() or "paused after repeated failures"
    except OSError:
        return ""


def gpu_temp() -> float | None:
    """Hottest sensor of the Intel GPU in degrees C, if the driver shows one."""
    best = None
    for hw in Path("/sys/class/hwmon").glob("hwmon*"):
        try:
            if (hw / "name").read_text().strip() != "xe":
                continue
            for t in hw.glob("temp*_input"):
                v = int(t.read_text()) / 1000
                best = v if best is None else max(best, v)
        except (OSError, ValueError):
            pass
    return best


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


LEASE_STALE = 1800        # a lease that was not renewed for this long is forgotten (the app crashed or was left alone)


def lease_holder() -> str:
    """Who has the GPU on loan from the queue ("" = nobody). The app takes a lease to load a model while no book is being made."""
    try:
        pid, owner = (GUI_LOCK.read_text().split(None, 1) + [""])[:2]
        if _alive(int(pid)) and time.time() - GUI_LOCK.stat().st_mtime < LEASE_STALE:
            return owner.strip() or "the app"
        GUI_LOCK.unlink(missing_ok=True)
    except Exception:
        pass
    return ""


def lease_take(owner: str = "the app", wait: float = 120) -> bool:
    """Take the GPU from the queue. A book being made is paused: its speech process is stopped (finished clips are kept) and
    it carries on by itself when the lease is given back. Returns True once the GPU is free (or after `wait` seconds, False)."""
    QUEUE.mkdir(parents=True, exist_ok=True)
    GUI_LOCK.write_text(f"{os.getpid()} {owner}")
    end = time.time() + wait
    while time.time() < end:
        if not _synth_running():
            return True
        time.sleep(1)
    return False


def lease_touch() -> None:
    try:
        os.utime(GUI_LOCK)
    except OSError:
        pass


def lease_mine() -> bool:
    try:
        return int(GUI_LOCK.read_text().split()[0]) == os.getpid()
    except Exception:
        return False


def lease_idle() -> float:
    """Seconds since the lease was last used (0 if there is none)."""
    try:
        return time.time() - GUI_LOCK.stat().st_mtime
    except OSError:
        return 0.0


def lease_give_back() -> None:
    if lease_mine():
        GUI_LOCK.unlink(missing_ok=True)


def _synth_running() -> bool:
    out = subprocess.run(["ps", "-eo", "pid,args"], capture_output=True, text=True).stdout
    return any(("-m audiobook_gen synth" in l or "-m audiobook_gen qc" in l) and "ps -eo" not in l for l in out.splitlines())


def foreign_synthesis() -> str:
    """Something other than this runner is using the GPU: a synth started by hand, or the app holding a lease."""
    who = lease_holder()
    if who:
        return who + " (you are using the GPU there)"
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


def stage_activity(job: dict) -> float:
    """When the job last showed life outside the speech stage: its log, the chapter files, the audiobook being written."""
    newest = 0.0
    paths = [QUEUE / f"{job['id']}.log"]
    out = Path(job.get("out", "") or "x")
    paths += list(out.parent.glob(out.name + "*")) if out.parent.exists() else []
    chapters = Path(job["work"]) / "chapters"
    paths += list(chapters.glob("*")) if chapters.exists() else []
    for p in paths:
        try:
            newest = max(newest, p.stat().st_mtime)
        except OSError:
            pass
    return newest


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
    def __init__(self, poll: float = POLL, stall: float = STALL, grace: float = START_GRACE, max_restarts: int = MAX_RESTARTS, recycle: float | None = None, stall_scale: float = 1.0,
                 foreign=foreign_synthesis, now=datetime.now, retry_wait: float = 30, sender=send_file, send_every: float = SEND_EVERY):
        self.poll, self.stall, self.grace, self.max_restarts, self.recycle, self.stall_scale = poll, stall, grace, max_restarts, recycle, stall_scale
        self.sender, self.send_every, self._sender_thread = sender, send_every, None
        self.foreign, self.now, self.retry_wait = foreign, now, retry_wait

    # commands (replaceable in tests)
    def synth_cmd(self, job: dict) -> list[str]:
        cmd = [sys.executable, "-m", "audiobook_gen", "synth", "x", "--work", job["work"], "--config", job["config"]]
        return cmd + (["--chapters", ",".join(map(str, job["chapters"]))] if job.get("chapters") else [])

    def qc_cmd(self, job: dict) -> list[str]:
        cmd = [sys.executable, "-m", "audiobook_gen", "qc", "x", "--work", job["work"], "--config", job["config"]]
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

    def _qc_summary(self, job: dict) -> None:
        try:
            r = json.loads((Path(job["work"]) / "qc_report.json").read_text())
            text = (f"{r['fixed']} clip(s) fixed" if r["fixed"] else "all clips fine") + (f", {r['unfixed']} could not be fixed" if r["unfixed"] else "")
            update(job["id"], qc=f"{text} (of {r['checked']:,})")
        except (OSError, ValueError, KeyError):
            update(job["id"], qc="no quality report")

    def _outcome(self, job: dict, ok: bool) -> None:
        """A book finished or failed: keep the failure streak, tell the user, halt the queue if everything keeps failing."""
        st = settings()
        streak = 0 if ok else int(st.get("fail_streak", 0)) + 1
        save_settings(fail_streak=streak)
        cur = next((j for j in load() if j["id"] == job["id"]), job)
        if ok:
            qc = f" ({cur['qc']})" if cur.get("qc") and "fine" not in cur["qc"] else ""
            left = [j for j in load() if j["status"] in ACTIVE]
            if left:
                notify(f"Finished “{job['title']}”{qc}. {len(left)} more in the queue.", "Book finished", tags=["white_check_mark"])
            else:
                notify(f"Finished “{job['title']}”{qc}. The queue is empty.", "Queue finished", tags=["tada"])
            if not cur.get("send_to"):
                self._cleanup(cur)
            return
        notify(f"“{job['title']}” failed: {cur.get('note', '')}", "Book failed", priority="high", tags=["x"])
        if streak >= FAIL_HALT:
            try:
                HALTED.write_text(f"{streak} books in a row failed (last: “{job['title']}”). Fix the cause, then press Resume.")
            except OSError:
                pass
            notify(f"{streak} books in a row failed, so the queue paused itself. Open the app and press Resume once the cause is fixed.",
                   "Queue paused", priority="urgent", tags=["warning"])

    def _cleanup(self, job: dict) -> None:
        """If asked to (a setting), delete a finished and delivered book's clips and chapter files to free disk space."""
        if not settings().get("clean_after_done"):
            return
        import shutil
        for sub in ("clips", "chapters"):
            shutil.rmtree(Path(job["work"]) / sub, ignore_errors=True)

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
                t = gpu_temp()
                f.write(f"[memory] {group_rss(proc.pid):.1f} GB after {(time.time() - started) / 60:.0f} min"
                        + (f" · GPU {t:.0f} °C" if t is not None else "") + "\n")
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
            if stage != "synth":
                idle = time.time() - stage_activity(job)
                if idle > (QC_STALL if stage == "qc" else ASSEMBLE_STALL) * self.stall_scale:
                    _kill_group(proc)
                    return "stalled"
            if stage in ("synth", "qc") and lease_holder():                    # the app asked for the GPU: give it up, come back later
                _kill_group(proc)
                return "lease"
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
                    self._cleanup(j)
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
        for stage, cmd in (("synth", self.synth_cmd(job)), ("qc", self.qc_cmd(job)), ("assemble", self.assemble_cmd(job))):
            if stage == "qc" and not (Path(job["work"]) / "clips_meta.json").exists():
                continue                                           # made before the checker existed: nothing to check against
            while True:
                cur = next((j for j in load() if j["id"] == job["id"]), job)
                if cur["status"] in ("cancelled", "held"):
                    return
                if stage == "synth" and not in_window(job.get("window", ""), self.now()):
                    update(job["id"], status="paused", note="waiting for its daily window")
                    return
                update(job["id"], note={"synth": "making the speech", "qc": "checking the clips", "assemble": "building the audiobook"}[stage])
                proc = self._spawn(cmd, job)
                result = self._watch(job, proc, stage)
                if result == "ok":
                    if stage == "qc":
                        self._qc_summary(job)
                    break
                if result in ("cancelled", "held"):
                    return
                if result == "lease":
                    update(job["id"], status="queued", note="paused while you use the GPU in the app (it carries on by itself)")
                    return
                if result == "window":
                    update(job["id"], status="paused", note="paused: outside its daily window (finished clips are kept)")
                    return
                if stage == "qc":                                # a safeguard must never stop the book: build it anyway
                    update(job["id"], qc="the check could not finish; see the log", note="quality check skipped")
                    break
                if result in ("recycle", "slow"):                # a planned fresh start, not a failure: finished clips are kept
                    update(job["id"], note=("restarted the speech process to keep it fast" if result == "recycle"
                                            else "speech had slowed down; restarted it"))
                    continue
                restarts = int(cur.get("restarts", 0)) + 1
                update(job["id"], restarts=restarts, note=f"{stage} {result}; restart {restarts} of {self.max_restarts}")
                if restarts > self.max_restarts:
                    update(job["id"], status="failed", finished=time.strftime(FMT),
                           note=f"gave up after {self.max_restarts} restarts; see work/queue/{job['id']}.log")
                    self._outcome(job, False)
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
        self._outcome(job, done)

    def step(self) -> bool:
        """One scheduling decision. Returns True if a job ran."""
        self._beat()
        self.deliver_pending()
        if HALTED.exists():
            return False
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
        threading.Thread(target=listen_ntfy, daemon=True).start()
        print(f"[queue] runner started {time.strftime(FMT)}", flush=True)
        while True:
            try:
                if not self.step():
                    time.sleep(self.poll)
            except Exception as e:                                  # a bug in one job must not stop the runner
                print(f"[queue] error: {e!r}", flush=True)
                time.sleep(self.poll)


# ---------- questions over ntfy ----------
COMMANDS = {"current": "what is being made now", "queue": "the waiting books and when each should be done",
            "done": "the finished books", "help": "this list"}


def _when(t) -> str:
    return t.strftime("%a %H:%M") if t else "?"


def answer(text: str, now: datetime | None = None) -> str | None:
    """The reply to a message sent to the ntfy topic, or None if it is not one of the commands (other messages are ignored)."""
    cmd = (text or "").strip().lower().strip(".!?")
    if cmd not in COMMANDS and cmd not in ("status", "now", "finished"):
        return None
    cmd = {"status": "current", "now": "current", "finished": "done"}.get(cmd, cmd)
    now = now or datetime.now()
    jobs = load()
    if cmd == "help":
        return "Send one word: " + ", ".join(f"{k} ({v})" for k, v in COMMANDS.items() if k != "help") + "."
    est = estimate_queue(now)
    if cmd == "current":
        r = now_running()
        why = halted()
        if r:
            end = est["ends"].get(r["id"])
            return (f"Making “{r['title']}”: {r['pct']}% ({r['done']:,} of {r['total']:,} clips)" if r["total"] else f"Making “{r['title']}”: {r['note'] or 'starting'}") \
                + (f"\nExpected done {_when(end)}" if end else "") + (f"\nStep: {r['note']}" if r["note"] and r["total"] else "") + (f"\n⚠ {why}" if why else "")
        if why:
            return f"The queue paused itself: {why}"
        nxt = [j for j in jobs if j["status"] in ("queued", "paused")]
        if nxt:
            j = min(nxt, key=lambda x: next_start(x, now))
            return f"Nothing is being made right now. Next: “{j['title']}” at {next_start(j, now)}."
        return "Nothing is being made and nothing is waiting."
    if cmd == "queue":
        lines = []
        for i, j in enumerate([j for j in jobs if j["status"] in ACTIVE or j["status"] == "held"], 1):
            tail = (f"{j.get('progress', '')} · done {_when(est['ends'].get(j['id']))}" if j["status"] == "running" else
                    f"done {_when(est['ends'][j['id']])}{'~' if j['title'] in est['rough'] else ''}" if j["id"] in est["ends"] else "saved for later" if j["status"] == "held" else "no estimate yet")
            lines.append(f"{i}. {j['title'][:38]} — {j['status']}, {tail}")
        if not lines:
            return "The queue is empty."
        return "\n".join(lines) + (f"\nAll finished about {_when(est['all'])}." + (" (~ = rough: a voice engine has no measured speed yet)" if est["rough"] else "") if est["all"] else "")
    done = [j for j in jobs if j["status"] in ("done", "failed")]
    if not done:
        return "Nothing has finished yet."
    return "\n".join(f"{'✓' if j['status'] == 'done' else '✗'} {j['title'][:38]} — {j.get('finished', '')}"
                     + (f", {('sent' if j.get('sent') else 'not sent yet') if j.get('send_to') else 'not sent'}" if j["status"] == "done" else f" ({j.get('note', '')[:40]})")
                     for j in done[-12:])


def listen_ntfy() -> None:
    """Answer the commands above for as long as the runner lives: follow the topic's message stream, reconnect when it drops.
    Only messages that arrive after it starts are read, and only the exact command words get an answer."""
    since, backoff = str(int(time.time())), 5
    while True:
        topic = ntfy_topic()
        if not topic:
            time.sleep(30)
            continue
        try:
            req = urllib.request.Request(f"{NTFY_SERVER}/{topic}/json?since={since}")
            with urllib.request.urlopen(req, timeout=90) as stream:            # the server sends a keep-alive every 30 s
                backoff = 5
                for line in stream:
                    try:
                        m = json.loads(line)
                    except ValueError:
                        continue
                    if m.get("event") != "message":
                        continue
                    since = str(m.get("time", since))
                    reply = answer(m.get("message", ""))
                    if reply:
                        notify(reply, "Audiobook queue", desktop=False)
        except Exception:
            time.sleep(backoff)
            backoff = min(300, backoff * 2)


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
    HALTED.unlink(missing_ok=True)
    save_settings(fail_streak=0)
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


_CHARS: dict = {}


def book_chars(job: dict) -> dict[str, int]:
    """Characters per engine in the job's chapters (cached while the book's segments file is unchanged)."""
    path = Path(job["work"]) / "segments.json"
    try:
        key = (str(path), path.stat().st_mtime, tuple(job.get("chapters") or ()))
    except OSError:
        return {}
    if key not in _CHARS:
        from .synth import resolve_voice
        cfg = yaml.safe_load(Path(job["config"]).read_text()) or {}
        only, out = set(job.get("chapters") or []), {}
        for sg in json.loads(path.read_text()):
            if not only or sg["chapter"] in only:
                e = resolve_voice(sg["speaker"], cfg)["engine"]
                out[e] = out.get(e, 0) + len(sg["text"])
        _CHARS[key] = out
    return _CHARS[key]


def estimate_queue(now: datetime | None = None) -> dict:
    """When each waiting or running book should be finished, from the measured speed of each engine (whole books, start to file),
    in queue order, through each book's start time and daily window. {"ends": {job id: datetime}, "all": datetime | None, "unknown": [titles]}."""
    from .runstats import averages
    now = now or datetime.now()
    avg, t, ends, unknown, rough = averages(), now, {}, [], []
    for j in load():
        if j["status"] not in ACTIVE:
            continue
        try:
            per = book_chars(j)
            if not per or not avg:
                raise KeyError
            slowest = min(a["cps"] for a in avg.values())                 # an engine never measured is assumed as slow as the slowest known
            if any(e not in avg for e in per):
                rough.append(j["title"])
            secs = sum(n / avg[e]["cps"] if e in avg else n / slowest for e, n in per.items())
        except Exception:
            unknown.append(j["title"])
            continue
        try:
            done, total = (int(x) for x in j.get("progress", "").split("/"))
            if j["status"] == "running" and total:
                secs *= 1 - done / total
        except ValueError:
            pass
        t = max(t, datetime.strptime(j["not_before"], FMT)) if j.get("not_before") else t
        step = timedelta(minutes=5)
        for _ in range(60 * 24 * 12):                                 # at most 60 days ahead
            if secs <= 0:
                break
            if in_window(j.get("window", ""), t):
                secs -= step.total_seconds()
            t += step
        ends[j["id"]] = t
    return {"ends": ends, "all": max(ends.values()) if ends else None, "unknown": unknown, "rough": rough}


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
