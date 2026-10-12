"""Finished books: rate them, and clean up their work folders.

The ratings live in out/ratings.json (not in the work folder, so a rating survives cleaning the folder up). A work folder can be
  * deleted whole, or
  * archived: everything except the clips is zipped into archive/<name>.zip and the folder is removed (restore unzips it back; the
    clips are not in the zip, so the next build of that book makes them again).
Neither ever touches the finished audiobook in out/. A folder that a queued, held or running job uses is never touched, and nothing
outside work/ is.
"""
import json
import shutil
import zipfile
from pathlib import Path

from . import jobqueue

ROOT = jobqueue.ROOT
RATINGS = ROOT / "out" / "ratings.json"
ARCHIVE = ROOT / "archive"
ACTIVE = ("queued", "paused", "running", "held")
CLIP_DIRS = ("clips", "clips_before_name_check")          # what the archive leaves out (a folder named clips*)


def _load_ratings() -> dict:
    try:
        return json.loads(RATINGS.read_text())
    except (OSError, ValueError):
        return {}


def rate(name: str, stars: int, note: str = "") -> None:
    """1 to 5 stars for the book made in work/<name>; 0 clears it."""
    import time
    data = _load_ratings()
    if not 0 <= int(stars) <= 5:
        raise ValueError("A rating is 1 to 5 stars.")
    if int(stars) == 0:
        data.pop(name, None)
    else:
        data[name] = {"stars": int(stars), "note": note.strip(), "at": time.strftime("%Y-%m-%d")}
    RATINGS.parent.mkdir(exist_ok=True)
    RATINGS.write_text(json.dumps(data, indent=1, ensure_ascii=False))


def rating(name: str) -> dict:
    return _load_ratings().get(name, {})


def stars_text(n: int) -> str:
    return "★" * n + "☆" * (5 - n) if n else "not rated"


def _size(p: Path) -> int:
    return sum(f.stat().st_size for f in p.rglob("*") if f.is_file()) if p.is_dir() else 0


def clips_size(work: Path) -> int:
    return sum(_size(work / d) for d in CLIP_DIRS)


def human(n: int) -> str:
    return f"{n / 1e9:.1f} GB" if n >= 1e9 else f"{n / 1e6:.0f} MB" if n >= 1e6 else f"{n / 1e3:.0f} KB"


def done_books() -> list[dict]:
    """One row per finished book (newest finish first): {name, title, work, out, finished, stars, folder, clips, bytes, clip_bytes}."""
    seen, rows = set(), []
    for j in sorted((j for j in jobqueue.load() if j["status"] == "done"), key=lambda j: j.get("finished") or "", reverse=True):
        w = Path(j["work"]).resolve()
        if w in seen:
            continue
        seen.add(w)
        rows.append({"name": w.name, "title": j["title"], "work": str(w), "out": j.get("out", ""), "finished": j.get("finished", ""),
                     "stars": rating(w.name).get("stars", 0), "folder": w.is_dir(), "bytes": _size(w), "clip_bytes": clips_size(w) if w.is_dir() else 0})
    return rows


def archived() -> list[Path]:
    return sorted(ARCHIVE.glob("*.zip")) if ARCHIVE.exists() else []


def check_cleanable(work: Path) -> None:
    """Raise ValueError (with the reason) unless this folder may be deleted or archived."""
    w = Path(work).resolve()
    if (ROOT / "work").resolve() not in w.parents:
        raise ValueError("That is not a book folder under work/.")
    if not w.is_dir():
        raise ValueError("The folder is already gone.")
    busy = [j["title"] for j in jobqueue.load() if j["status"] in ACTIVE and Path(j["work"]).resolve() == w]
    if busy:
        raise ValueError(f"“{busy[0]}” still has a job waiting or running that uses this folder.")


def unused_clips(work: Path) -> list[Path]:
    """Clip files in clips/ that the book no longer uses: not in the current plan (clips_meta.json / clips.json) and not an original that a
    correction can still go back to (clip_overrides.json 'was'). What is left after lexicon, voice or text changes made new versions."""
    work = Path(work)
    used: set[str] = set()
    for name in ("clips_meta.json", "clips.json"):
        try:
            data = json.loads((work / name).read_text())
        except (OSError, ValueError):
            return []                                      # no plan to compare against: never guess
        used |= set(data) if name == "clips_meta.json" else {f for v in data.values() for f in v}
    try:                                                   # a plan made for only some chapters says nothing about the others' clips
        planned = {m.get("chapter") for m in json.loads((work / "clips_meta.json").read_text()).values()}
        book = {sg.get("chapter") for sg in json.loads((work / "segments.json").read_text())}
        if not book <= planned:
            return []
    except (OSError, ValueError):
        return []
    try:
        used |= {o.get("was") for o in json.loads((work / "clip_overrides.json").read_text()).values() if o.get("was")}
    except (OSError, ValueError):
        pass
    d = work / "clips"
    return sorted(f for f in d.glob("*.wav") if f.name not in used) if d.is_dir() and used else []


def remove_unused_clips(work: Path) -> tuple[int, int]:
    """Delete the unused clips of a book that is not being made. Returns (files, bytes freed)."""
    check_cleanable(work)
    gone = unused_clips(work)
    freed = 0
    for f in gone:
        freed += f.stat().st_size
        f.unlink()
    return len(gone), freed


def delete_folder(work: Path) -> int:
    """Delete the whole work folder. Returns the bytes freed."""
    check_cleanable(work)
    n = _size(Path(work))
    shutil.rmtree(work)
    return n


def archive_without_clips(work: Path) -> tuple[Path, int]:
    """Zip everything except the clips into archive/<name>.zip, check the zip, then remove the folder. Returns (zip, bytes freed)."""
    check_cleanable(work)
    w = Path(work).resolve()
    ARCHIVE.mkdir(exist_ok=True)
    dest = ARCHIVE / f"{w.name}.zip"
    if dest.exists():
        raise ValueError(f"{dest.name} is already in archive/: restore or move it first.")
    tmp = dest.with_suffix(".zip.part")
    with zipfile.ZipFile(tmp, "w", zipfile.ZIP_DEFLATED) as z:
        for f in sorted(w.rglob("*")):
            rel = f.relative_to(w)
            if f.is_file() and rel.parts[0] not in CLIP_DIRS:
                z.write(f, Path(w.name) / rel)
    with zipfile.ZipFile(tmp) as z:
        if z.testzip() is not None:
            tmp.unlink()
            raise ValueError("The zip did not verify, so nothing was deleted.")
    tmp.rename(dest)
    freed = _size(w)
    shutil.rmtree(w)
    return dest, freed - dest.stat().st_size


def restore(zip_path: Path) -> Path:
    """Unzip an archive back into work/. The clips are not in it; the next build of the book makes them again."""
    z = Path(zip_path)
    target = ROOT / "work" / z.stem
    if target.exists():
        raise ValueError(f"work/{z.stem} already exists.")
    with zipfile.ZipFile(z) as f:
        for n in f.namelist():
            if not n.startswith(z.stem + "/") or ".." in Path(n).parts:
                raise ValueError("That zip does not look like an archived book.")
        f.extractall(ROOT / "work")
    return target
