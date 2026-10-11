"""Music for the audiobook: a title intro, short bridges between sections, and an ending.

Only music that is free to use is offered: public domain or CC0 recordings on Wikimedia Commons (same rule as the cover maker).
A track is downloaded once into music/library/ with its credit line; a book's plan (work/<book>/music.json) says which track goes
where. Music plays between pieces of speech, never under the narrator, so there is no ducking to tune.

    python -m audiobook_gen.music search "Bach cello suite" [N]
    python -m audiobook_gen.music add "File:Some recording.ogg"
    python -m audiobook_gen.music list
"""
import argparse
import json
import re
import subprocess
import sys
import urllib.parse
import urllib.request
from pathlib import Path

import numpy as np

from . import covers

ROOT = Path(__file__).resolve().parent.parent
LIB = ROOT / "music" / "library"
INDEX = LIB / "index.json"
CHARS_PER_SECOND = 14.5                      # measured narration speed, used to guess a chapter's length before it is made
# what to look for, per place in the book (each query is tried on Commons; every result is still checked for its licence)
SUGGESTIONS = {
    "intro": ["Bach cello suite prelude", "Debussy Clair de lune piano", "Handel organ concerto", "Satie Gymnopedie"],
    "bridge": ["Bach prelude harp", "Satie Gymnopedie", "Chopin nocturne piano", "Pachelbel canon"],
    "outro": ["Bach Air on the G string", "Schubert Ave Maria", "Chopin nocturne piano", "Handel Largo"],
}
DEFAULT_PLAN = {
    "enabled": False,
    "level_db": -4.0,                        # music loudness against the narration (RMS), so it never jumps out
    "intro": {"track": "", "start": 0.0, "seconds": 22.0, "fade_in": 1.5, "fade_out": 4.0,
              # layered opening (used when `title` is set): music alone for `lead` s, the spoken title over the music (turned down by
              # `duck_db`), music alone again for `tail` s, then the book begins with the music turned down to `under_db` and fading out over `fade` s
              "title": "", "title_pause": 2.8, "lead": 6.0, "tail": 6.0, "fade": 12.0, "duck_db": -14.0, "under_db": -9.0},
    "bridge": {"track": "", "start": 0.0, "seconds": 8.0, "fade_in": 0.8, "fade_out": 3.0,
               # with `title_before` > 0 the bridge plays alone after `lead` s of silence, and the next section's title is spoken over its last
               # `title_before` s (the music turned down to `under_db`); the narration then resumes while the music slowly fades out
               "lead": 3.0, "title_before": 0.0, "under_db": -10.0},
    # the ending: the music fades in under the last `before` seconds of speech, stays turned down for `hold` s after the last word, swells, carries on for `after`
    # seconds and fades out over the last `fade_out` of them
    "outro": {"track": "", "start": 0.0, "before": 20.0, "after": 9.0, "fade_out": 5.0, "under_db": -11.0, "hold": 1.0},
    "titles": {"speak": True, "pause": 2.8},   # read each section's title aloud, with this pause before and after (see titles.py)
    "interval_min": 25,                      # recommended: one bridge about this many minutes apart, at the nearest section start
    "chapters": {},                          # "<chapter index>": "on" | "off" (a person's choice); missing = follow the recommendation
}


# ---- the library -------------------------------------------------------------------------------------------------------

def library() -> list[dict]:
    try:
        return json.loads(INDEX.read_text())
    except (OSError, ValueError):
        return []


def _save(entries: list[dict]) -> None:
    LIB.mkdir(parents=True, exist_ok=True)
    INDEX.write_text(json.dumps(entries, indent=1, ensure_ascii=False))


def track(track_id: str) -> dict | None:
    return next((t for t in library() if t["id"] == track_id), None)


def search(query: str, n: int = 8) -> list[dict]:
    """Free-licence recordings on Commons: [{file, title, artist, licence, seconds, mime, date, page}]."""
    r = covers._api(action="query", generator="search", gsrsearch=query + " filetype:audio", gsrnamespace=6, gsrlimit=50,
                    prop="imageinfo", iiprop="url|size|mime|extmetadata|duration",
                    iiextmetadatafilter="LicenseShortName|Artist|DateTimeOriginal|ObjectName")
    out = []
    for pg in sorted((r.get("query") or {}).get("pages", {}).values(), key=lambda x: x.get("index", 0)):
        ii = pg["imageinfo"][0]
        m = ii.get("extmetadata", {})
        lic = covers._plain(m.get("LicenseShortName", {}).get("value", ""))
        mime, secs = ii.get("mime", ""), float(ii.get("duration") or 0)
        if not covers.free_licence(lic) or "midi" in mime or secs < 20:        # a MIDI file is not a recording; under 20 s is a snippet
            continue
        out.append({"file": pg["title"], "title": covers._name(pg["title"]), "artist": covers._artist(m.get("Artist", {}).get("value", "")),
                    "licence": lic, "seconds": round(secs), "mime": mime, "date": covers._date(m.get("DateTimeOriginal", {}).get("value", "")),
                    "page": "https://commons.wikimedia.org/wiki/" + urllib.parse.quote(pg["title"].replace(" ", "_"))})
        if len(out) >= n:
            break
    return out


def add(file: str) -> dict:
    """Download a free-licence Commons recording into the library (once) and return its entry. Refuses anything else."""
    hits = [h for h in search(file.removeprefix("File:").rsplit(".", 1)[0], 20) if h["file"].lower() == ("file:" + file.removeprefix("File:")).lower()]
    if not hits:
        raise ValueError(f"{file} is not a free-licence (public domain or CC0) recording on Wikimedia Commons")
    h = hits[0]
    tid = re.sub(r"[^a-z0-9]+", "-", h["title"].lower()).strip("-")[:60]
    entries = library()
    if any(t["id"] == tid for t in entries):
        return next(t for t in entries if t["id"] == tid)
    LIB.mkdir(parents=True, exist_ok=True)
    ext = Path(h["file"]).suffix or ".ogg"
    dest = LIB / (tid + ext)
    url = "https://commons.wikimedia.org/wiki/Special:FilePath/" + urllib.parse.quote(h["file"][5:])
    dest.write_bytes(urllib.request.urlopen(urllib.request.Request(url, headers=covers.UA), timeout=300).read())
    credit = (f"{h['title']}, {h['artist'] or 'unknown artist'}" + (f", {h['date']}" if h["date"] else "") + f". {h['page']} ({h['licence']})")
    entry = {"id": tid, "title": h["title"], "artist": h["artist"], "licence": h["licence"], "seconds": h["seconds"],
             "file": dest.name, "credit": credit}
    _save(entries + [entry])
    return entry


def remove(track_id: str) -> None:
    keep = []
    for t in library():
        if t["id"] == track_id:
            (LIB / t["file"]).unlink(missing_ok=True)
        else:
            keep.append(t)
    _save(keep)


# ---- a book's plan -----------------------------------------------------------------------------------------------------

def load_plan(work: Path) -> dict:
    plan = json.loads(json.dumps(DEFAULT_PLAN))
    try:
        saved = json.loads((Path(work) / "music.json").read_text())
    except (OSError, ValueError):
        return plan
    for k, v in saved.items():
        if isinstance(v, dict) and isinstance(plan.get(k), dict) and k != "chapters":
            plan[k].update(v)
        else:
            plan[k] = v
    return plan


def save_plan(work: Path, plan: dict) -> None:
    (Path(work) / "music.json").write_text(json.dumps(plan, indent=1, ensure_ascii=False))


def chapter_seconds(work: Path) -> list[tuple[int, str, float]]:
    """(index, title, guessed seconds) for each chapter, from the text length (exact timings only exist after the book is made)."""
    meta = json.loads((Path(work) / "chapters.json").read_text())
    return [(c["index"], c["title"], len(c.get("text", "")) / CHARS_PER_SECOND) for c in meta["chapters"] if not c.get("opening")]


def recommended(chapters: list[tuple[int, str, float]], interval_min: float = 25) -> list[int]:
    """Chapter starts that should get a bridge: the first section start at or after each `interval_min` minutes of narration, so a
    long book gets music about that often, and never two in a row. The first chapter has the intro, so no bridge there."""
    out, clock, last = [], 0.0, 0.0
    for i, (idx, _, secs) in enumerate(chapters):
        if i > 0 and clock - last >= interval_min * 60:
            out.append(idx)
            last = clock
        clock += secs
    return out


def bridge_before(plan: dict, chapters: list[tuple[int, str, float]]) -> set[int]:
    """The chapter indexes that begin with a bridge: the recommendation, changed by the person's own on/off choices."""
    rec = set(recommended(chapters, plan.get("interval_min", 25)))
    for k, v in plan.get("chapters", {}).items():
        (rec.add if v == "on" else rec.discard)(int(k))
    first = chapters[0][0] if chapters else None
    rec.discard(first)
    return rec


# ---- mixing ------------------------------------------------------------------------------------------------------------

def _decode(path: Path, start: float, seconds: float, sr: int) -> np.ndarray:
    p = subprocess.run(["ffmpeg", "-v", "error", "-ss", str(max(0.0, start)), "-t", str(seconds), "-i", str(path), "-ac", "1", "-ar", str(sr),
                        "-f", "f32le", "-"], capture_output=True, check=True)
    return np.frombuffer(p.stdout, dtype=np.float32).copy()


def _fade(a: np.ndarray, sr: int, fade_in: float, fade_out: float) -> np.ndarray:
    n_in, n_out = min(len(a), int(fade_in * sr)), min(len(a), int(fade_out * sr))
    if n_in:
        a[:n_in] *= np.linspace(0, 1, n_in, dtype=np.float32) ** 1.5
    if n_out:
        a[-n_out:] *= np.linspace(1, 0, n_out, dtype=np.float32) ** 1.5
    return a


def speech_rms(audio: np.ndarray) -> float:
    """Loudness of the speech itself (silences left out)."""
    loud = audio[np.abs(audio) > 0.02]
    return float(np.sqrt(np.mean(loud ** 2))) if len(loud) else 0.1


def piece(spec: dict, sr: int, rms: float, level_db: float) -> np.ndarray | None:
    """One piece of music (intro, bridge or outro) cut, faded and set to `level_db` against the speech level `rms`; None if the
    track is not chosen or missing."""
    t = track(spec.get("track", ""))
    if not t or not (LIB / t["file"]).exists():
        return None
    secs = min(float(spec.get("seconds", 8)), max(1.0, t["seconds"] - float(spec.get("start", 0))))
    a = _decode(LIB / t["file"], float(spec.get("start", 0)), secs, sr)
    if not len(a):
        return None
    a = _fade(a, sr, float(spec.get("fade_in", 1)), float(spec.get("fade_out", 3)))
    own = speech_rms(a) or 1.0
    a = a * (rms * 10 ** (level_db / 20) / own)
    peak = float(np.abs(a).max()) or 1.0
    return (a * min(1.0, 0.7 / peak)).astype(np.float32)


def wrap(audio: np.ndarray, chapter: int, chapters: list[int], plan: dict, guess: list[tuple[int, str, float]], sr: int, layered: bool = False) -> np.ndarray:
    """Put this chapter's music around its speech: the intro before the first chapter, a bridge before a chapter chosen for one, the
    ending after the last chapter. `chapters` are the indexes being made now (all of them, or the few a partial run asked for)."""
    if not plan.get("enabled") or not chapters:
        return audio
    rms, db, pre = speech_rms(audio), float(plan.get("level_db", -4)), None
    if chapter == chapters[0]:
        pre = None if layered else piece(plan["intro"], sr, rms, db)           # a layered opening is its own chapter
    elif chapter in bridge_before(plan, [g for g in guess if g[0] in chapters]):
        spec = plan["bridge"]
        if float(spec.get("title_before", 0)) > 0:
            mixed = bridge_under_title(audio, spec, sr, rms, db)
            if mixed is not None:
                return mixed if not (chapter == chapters[-1] and plan["outro"].get("track")) else ending(mixed, plan, sr)
        pre = piece(spec, sr, rms, db)
    if chapter == chapters[-1] and plan["outro"].get("track"):
        audio = ending(audio, plan, sr)
    parts = [x for x in (pre, audio) if x is not None]
    return np.concatenate(parts) if len(parts) > 1 else audio


def _speech_start(audio: np.ndarray, sr: int) -> int:
    """Index of the first sample of speech (the silence a section begins with is skipped)."""
    loud = np.flatnonzero(np.abs(audio) > 0.02)
    return int(loud[0]) if len(loud) else 0


def bridge_under_title(audio: np.ndarray, spec: dict, sr: int, rms: float, level_db: float) -> np.ndarray | None:
    """A bridge that leads into the section: `lead` s of silence, the music alone, then the section's spoken title starts `title_before` s
    before the music ends (the music turned down to `under_db`), and the narration carries on while the music fades out slowly.
    None if the track is missing (the caller then falls back to a plain bridge)."""
    music = piece({**spec, "fade_out": 0.0}, sr, rms, level_db)
    if music is None:
        return None
    lead, n_m = int(float(spec.get("lead", 3.0)) * sr), len(music)
    s0 = _speech_start(audio, sr)
    t0 = max(0, n_m - int(float(spec["title_before"]) * sr))                  # where the title starts, within the music
    g = np.ones(n_m, np.float32)
    under, ramp = 10 ** (float(spec.get("under_db", -10)) / 20), min(int(1.0 * sr), n_m - t0)
    if ramp > 0:
        g[t0:t0 + ramp] = np.linspace(1, under, ramp, dtype=np.float32)
    rest = n_m - t0 - ramp
    if rest > 0:
        g[t0 + ramp:] = under * np.linspace(1, 0, rest, dtype=np.float32) ** 1.5      # the slow fade
    body = audio[s0:]
    out = np.zeros(max(lead + n_m, lead + t0 + len(body)), np.float32)
    out[lead:lead + n_m] += music * g
    out[lead + t0:lead + t0 + len(body)] += body
    peak = float(np.abs(out).max()) or 1.0
    return (out * min(1.0, 0.98 / peak)).astype(np.float32)


def add_opening(work: Path, text: str, pause_s: float = 0.0, after: int | None = None) -> None:
    """Make the spoken title the book's first chapter ("Opening"), so the narrator reads it and the music can be laid under it. Each line
    of `text` is read as its own paragraph; with `pause_s`, the lines after the first begin after that many seconds of silence (a dramatic
    pause after the title). With `after`, the opening comes right after that chapter (a preface, say) instead of at the start.
    Changes chapters.json and segments.json; the other chapters and their clips are left alone."""
    work = Path(work)
    lines = [" ".join(l.split()) for l in text.splitlines() if l.strip()]
    if not lines and not has_opening(work):
        return                                                           # nothing to add and nothing to take away
    meta = json.loads((work / "chapters.json").read_text())
    segs = json.loads((work / "segments.json").read_text())
    gone = {c["index"] for c in meta["chapters"] if c.get("opening")}
    meta["chapters"] = [c for c in meta["chapters"] if not c.get("opening")]
    segs = [g for g in segs if g.get("chapter") not in gone]
    if lines:
        used = {c["index"] for c in meta["chapters"]}
        idx = 0 if 0 not in used else max(used) + 1
        order = [c["index"] for c in meta["chapters"]]
        at = order.index(after) + 1 if after in order else 0
        meta["chapters"].insert(at, {"index": idx, "title": "Opening", "text": "\n\n".join(lines), "opening": True})
        new = [{"speaker": "Narrator", "kind": "narration", "text": t, "para_start": True, "chapter": idx, "id": f"{idx:03d}-{i:05d}",
                **({"pause_ms": int(pause_s * 1000)} if i and pause_s else {})} for i, t in enumerate(lines)]
        last = max((i for i, g in enumerate(segs) if after in order and g["chapter"] == after), default=-1)
        segs = segs[:last + 1] + new + segs[last + 1:]
    (work / "chapters.json").write_text(json.dumps(meta, indent=1, ensure_ascii=False))
    (work / "segments.json").write_text(json.dumps(segs, indent=1, ensure_ascii=False))


def has_opening(work: Path) -> bool:
    try:
        return any(c.get("opening") for c in json.loads((Path(work) / "chapters.json").read_text())["chapters"])
    except (OSError, ValueError):
        return False


def _raw(track_id: str, start: float, seconds: float, sr: int) -> np.ndarray | None:
    t = track(track_id)
    if not t or not (LIB / t["file"]).exists():
        return None
    a = _decode(LIB / t["file"], start, max(1.0, min(seconds, t["seconds"] - start)), sr)
    return a if len(a) else None


def _ramp(g: np.ndarray, sr: int, t0: float, t1: float, a: float, b: float) -> None:
    """Move the gain from a to b between seconds t0 and t1 (smoothly), and keep b afterwards."""
    i0, i1 = int(t0 * sr), int(t1 * sr)
    n = max(1, i1 - i0)
    x = np.linspace(0, 1, n, dtype=np.float32)
    x = x * x * (3 - 2 * x)
    g[i0:i1] = a + (b - a) * x[: len(g[i0:i1])]
    g[i1:] = b


def long_silences(audio: np.ndarray, sr: int, min_s: float = 1.2) -> list[tuple[float, float]]:
    """Stretches of near-silence inside the speech (not at its ends) at least `min_s` long: (start s, end s)."""
    win = int(0.05 * sr)
    n = len(audio) // win
    env = np.sqrt(np.mean(audio[: n * win].reshape(n, win) ** 2, axis=1))
    quiet = env < max(0.004, 0.03 * float(env.max() or 1.0))
    out, i = [], 0
    while i < n:
        if quiet[i]:
            j = i
            while j < n and quiet[j]:
                j += 1
            if i > 0 and j < n and (j - i) * 0.05 >= min_s:
                out.append((i * 0.05, j * 0.05))
            i = j
        else:
            i += 1
    return out


def opening(title_audio: np.ndarray, plan: dict, sr: int) -> tuple[np.ndarray, np.ndarray]:
    """The layered opening. Returns (the Opening chapter's audio, music to lay under the start of the next chapter):
    music alone -> the spoken title over the music (music turned down) -> music alone -> [next chapter] speech with the music faded
    down and slowly out behind it."""
    spec = plan["intro"]
    lead, tail, fade = float(spec["lead"]), float(spec["tail"]), float(spec["fade"])
    loud = np.flatnonzero(np.abs(title_audio) > 0.01)                        # the music sets the lead-in, so the chapter's own start silence goes
    if len(loud):
        title_audio = title_audio[max(0, int(loud[0]) - int(0.05 * sr)):]
    duck, under = 10 ** (float(spec["duck_db"]) / 20), 10 ** (float(spec["under_db"]) / 20)
    t_len = len(title_audio) / sr
    head_len = lead + t_len + tail
    raw = _raw(spec["track"], float(spec.get("start", 0)), head_len + fade, sr)
    if raw is None:
        return np.concatenate([np.zeros(int(lead * sr), np.float32), title_audio, np.zeros(int(tail * sr), np.float32)]), np.zeros(0, np.float32)
    total = len(raw) / sr
    fade = max(3.0, min(fade, total - head_len)) if total > head_len + 3 else max(0.0, total - head_len)
    raw = raw[: int((head_len + fade) * sr)]
    own = speech_rms(raw) or 1.0
    raw = raw * (speech_rms(title_audio) * 10 ** (float(plan.get("level_db", -4)) / 20) / own)       # music alone sits near the voice level
    g = np.ones(len(raw), np.float32)
    g[: int(float(spec.get("fade_in", 1.5)) * sr)] = np.linspace(0, 1, int(float(spec.get("fade_in", 1.5)) * sr), dtype=np.float32) ** 1.5
    _ramp(g, sr, lead - 0.6, lead + 0.6, 1.0, duck)                    # turn down as the title starts
    for a0, a1 in long_silences(title_audio, sr):                      # a dramatic pause inside the title: the music swells into it
        _ramp(g, sr, lead + a0 + 0.3, lead + a0 + 1.1, duck, 1.0)
        _ramp(g, sr, lead + a1 - 0.9, lead + a1 - 0.1, 1.0, duck)
    _ramp(g, sr, lead + t_len, lead + t_len + 1.5, duck, 1.0)          # back up after it
    start_fade = head_len                                              # the book begins here: turn down, then fade slowly out
    g2 = g.copy()
    _ramp(g2, sr, start_fade - 0.4, start_fade + 1.2, 1.0, under)
    i0 = int((start_fade + 1.2) * sr)
    n = len(g2) - i0
    if n > 0:
        g2[i0:] = under * np.linspace(1, 0, n, dtype=np.float32) ** 1.5
    mixed = raw * g2
    peak = float(np.abs(mixed).max()) or 1.0
    mixed = (mixed * min(1.0, 0.6 / peak)).astype(np.float32)
    head = np.zeros(int(head_len * sr), np.float32)
    head[int(lead * sr): int(lead * sr) + len(title_audio)] += title_audio
    n_head = len(head)
    head += mixed[:n_head]
    return head, mixed[n_head:]


def apply_tail(audio: np.ndarray, tail: np.ndarray) -> np.ndarray:
    """Lay the opening's fading music under the start of the next chapter's speech."""
    if not len(tail):
        return audio
    out = audio.copy()
    n = min(len(out), len(tail))
    out[:n] += tail[:n]
    peak = float(np.abs(out).max()) or 1.0
    return (out * min(1.0, 0.98 / peak)).astype(np.float32)           # only if the sum would clip; the speech itself is left as it is


def ending(audio: np.ndarray, plan: dict, sr: int) -> np.ndarray:
    """The last chapter with the ending music: it fades in under the final `before` seconds of speech (quietly, at `under_db`), swells
    to full level once the speech has stopped, carries on for `after` seconds and fades out. Returns audio that is `after` seconds longer."""
    spec = plan["outro"]
    before, after = min(float(spec["before"]), len(audio) / sr), float(spec["after"])
    fade_out = max(1.0, min(float(spec["fade_out"]), after - 2.0))
    raw = _raw(spec["track"], float(spec.get("start", 0)), before + after, sr)
    if raw is None:
        return audio
    n_total, n_b = int((before + after) * sr), int(before * sr)
    raw = np.concatenate([raw, np.zeros(max(0, n_total - len(raw)), np.float32)])[:n_total]
    raw = raw * (speech_rms(audio) * 10 ** (float(plan.get("level_db", -4)) / 20) / (speech_rms(raw) or 1.0))
    under = 10 ** (float(spec.get("under_db", -11)) / 20)
    g = np.zeros(n_total, np.float32)
    g[:n_b] = under * np.linspace(0, 1, n_b, dtype=np.float32) ** 1.5            # in under the speech
    g[n_b:] = under                                                            # after the last word: still there, turned down
    hold = float(spec.get("hold", 1.0))                                        # stays turned down this long after the last word
    _ramp(g, sr, before + hold, before + hold + 1.5, under, 1.0)               # then swells
    n_out = int(fade_out * sr)
    g[-n_out:] *= np.linspace(1, 0, n_out, dtype=np.float32) ** 1.5              # and out
    mixed = raw * g
    peak = float(np.abs(mixed).max()) or 1.0
    mixed = (mixed * min(1.0, 0.6 / peak)).astype(np.float32)
    out = np.concatenate([audio, np.zeros(int(after * sr), np.float32)])
    start = len(audio) - n_b
    out[start: start + n_total] += mixed[: len(out) - start]
    peak = float(np.abs(out).max()) or 1.0
    return (out * min(1.0, 0.98 / peak)).astype(np.float32)


def credits(plan: dict) -> list[str]:
    seen, out = set(), []
    for role in ("intro", "bridge", "outro"):
        t = track(plan.get(role, {}).get("track", ""))
        if t and t["id"] not in seen:
            seen.add(t["id"])
            out.append(t["credit"])
    return out


def main(argv=None) -> None:
    p = argparse.ArgumentParser(prog="audiobook_gen.music")
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("search"); s.add_argument("query"); s.add_argument("n", nargs="?", type=int, default=8)
    a = sub.add_parser("add"); a.add_argument("file")
    sub.add_parser("list")
    args = p.parse_args(argv)
    if args.cmd == "search":
        for h in search(args.query, args.n):
            print(f"{h['file']} | {h['seconds']} s | {h['licence']} | {h['artist']}")
    elif args.cmd == "add":
        print(add(args.file)["credit"])
    else:
        for t in library():
            print(f"{t['id']} | {t['seconds']} s | {t['licence']} | {t['title']}")


if __name__ == "__main__":
    main()
