"""Running average of how fast each engine really synthesises, in characters per second of wall time.
Every synth run adds (or updates) one record in work/run_stats.json; the Generate tab shows the average."""
import json
import time
from pathlib import Path

PATH = Path(__file__).resolve().parent.parent / "work" / "run_stats.json"
KEEP = 300             # most recent runs kept
MIN_SECONDS = 20       # shorter runs are mostly start-up noise


def _load() -> list[dict]:
    try:
        return json.loads(PATH.read_text())
    except Exception:
        return []


def record(run_id: str, engine: str, workers: int, chars: int, seconds: float) -> None:
    """Add or update this run's record (a long run is updated as it goes, so a killed run still counts)."""
    if chars <= 0 or seconds <= 0:
        return
    runs = _load()
    rec = {"id": run_id, "engine": engine, "workers": int(workers), "chars": int(chars), "seconds": round(seconds, 1),
           "date": time.strftime("%Y-%m-%d %H:%M")}
    runs = [r for r in runs if not (r.get("id") == run_id and r.get("engine") == engine)] + [rec]
    try:
        PATH.parent.mkdir(exist_ok=True)
        PATH.write_text(json.dumps(runs[-KEEP:], indent=1))
    except OSError:
        pass


def fresh_chars(work_name: str, since: float) -> dict[str, int]:
    """{engine: characters made fresh} by the speech runs of this book since `since` (epoch seconds). A rebuild that reuses clips made earlier
    has few or none, so it never passes cached clips off as speed."""
    out: dict[str, int] = {}
    for r in _load():
        rid = str(r.get("id", ""))
        stem, _, stamp = rid.rpartition("-")
        if r.get("kind") != "job" and stem == work_name and stamp.isdigit() and int(stamp) >= since:
            out[r["engine"]] = out.get(r["engine"], 0) + int(r["chars"])
    return out


def drop_cached_jobs(factor: float = 1.5) -> int:
    """One-time clean-up of old history: remove whole-book records that ran faster than `factor` times the engine's speech-only speed,
    which only happens when most clips were reused (a rebuild). Returns how many were removed."""
    synth = {e: a["cps"] for e, a in averages(jobs=False).items()}
    runs = _load()
    keep = [r for r in runs if not (r.get("kind") == "job" and r["engine"] in synth and r["chars"] / max(r["seconds"], 1) > factor * synth[r["engine"]])]
    if len(keep) != len(runs):
        PATH.write_text(json.dumps(keep, indent=1))
    return len(runs) - len(keep)


def record_job(job_id: str, chars: dict[str, int], seconds: float) -> None:
    """A whole finished book: the characters it made fresh per engine and the active time from the queue starting it to the audiobook
    being built (restarts and start-up included, waiting outside the window not). The seconds are shared between engines
    by how long each engine's characters take at its synth speed, so a mixed-voice book still gives each engine a speed."""
    chars = {e: n for e, n in chars.items() if n > 0}
    if not chars or seconds < MIN_SECONDS:
        return
    synth = {e: a["cps"] for e, a in averages(jobs=False).items()}
    weight = {e: n / synth.get(e, 20.0) for e, n in chars.items()}
    total = sum(weight.values())
    runs = [r for r in _load() if not (r.get("id") == job_id and r.get("kind") == "job")]
    for e, n in chars.items():
        runs.append({"id": job_id, "kind": "job", "engine": e, "workers": 0, "chars": int(n),
                     "seconds": round(seconds * weight[e] / total, 1), "date": time.strftime("%Y-%m-%d %H:%M")})
    try:
        PATH.parent.mkdir(exist_ok=True)
        PATH.write_text(json.dumps(runs[-KEEP:], indent=1))
    except OSError:
        pass


def averages(jobs: bool = True) -> dict[str, dict]:
    """{engine: {"cps": chars per second, "runs": n, "chars": total, "whole": bool}}.
    Whole-book timings (everything from start to the finished file) are used for an engine once it has any; until then
    the speech-only speed of its synth runs."""
    acc: dict[str, list] = {}
    whole: dict[str, list] = {}
    for r in _load():
        if r["seconds"] >= MIN_SECONDS:
            a = (whole if r.get("kind") == "job" else acc).setdefault(r["engine"], [0, 0.0, 0])
            a[0] += r["chars"]; a[1] += r["seconds"]; a[2] += 1
    out = {e: {"cps": c / s, "runs": n, "chars": c, "whole": False} for e, (c, s, n) in acc.items() if s}
    if jobs:
        out.update({e: {"cps": c / s, "runs": n, "chars": c, "whole": True} for e, (c, s, n) in whole.items() if s})
    return out


def clock(seconds: float) -> str:
    m = int(round(seconds / 60))
    return f"{m // 60} h {m % 60:02d} min" if m >= 60 else f"{max(m, 1)} min"
