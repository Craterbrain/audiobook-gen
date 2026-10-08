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


def averages() -> dict[str, dict]:
    """{engine: {"cps": chars per second, "runs": n, "chars": total}} over runs long enough to count."""
    acc: dict[str, list] = {}
    for r in _load():
        if r["seconds"] >= MIN_SECONDS:
            a = acc.setdefault(r["engine"], [0, 0.0, 0])
            a[0] += r["chars"]; a[1] += r["seconds"]; a[2] += 1
    return {e: {"cps": c / s, "runs": n, "chars": c} for e, (c, s, n) in acc.items() if s}


def clock(seconds: float) -> str:
    m = int(round(seconds / 60))
    return f"{m // 60} h {m % 60:02d} min" if m >= 60 else f"{max(m, 1)} min"
