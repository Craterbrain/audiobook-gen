"""A short listen to the opening, a bridge or the ending of a book, laid into clips that are already made (the final clips will differ a little:
the pauses are fixed and a section title that is not made yet is silence). Used by the Extras tab and by the Ask Claude helper."""
import json
import subprocess
from math import gcd
from pathlib import Path

SR = 24000


def make(work, plan: dict, what: str, chapter: int | None, out_dir, name: str = "") -> tuple[Path, str]:
    """Build previews/<name>.mp3 for `what` ('opening' | 'bridge' | 'ending'). Returns (mp3 path, a note about stand-ins).
    Raises ValueError when nothing can be built (no clips yet, no such section)."""
    import numpy as np
    import soundfile as sf
    from scipy.signal import resample_poly
    from . import music
    work, out_dir = Path(work), Path(out_dir)
    try:
        meta = json.loads((work / "clips_meta.json").read_text())
    except (OSError, ValueError):
        meta = {}
    clips = [(f, m) for f, m in meta.items() if (work / "clips" / f).exists()]
    if not clips:
        raise ValueError("No clips are made yet, so there is nothing to build a preview from.")
    plan = {**plan, "enabled": True}
    secs = music.chapter_seconds(work)
    if not secs:
        raise ValueError("The book has no chapters.")
    all_ch, names = [c[0] for c in secs], {c[0]: c[1] for c in secs}

    def load(f):
        x, sr = sf.read(work / "clips" / f, dtype="float32")
        x = x if x.ndim == 1 else x.mean(1)
        if sr != SR:
            g = gcd(SR, sr)
            x = resample_poly(x, SR // g, sr // g).astype(np.float32)
        return x

    def body(ch, seconds, tail=False, skip=()):
        rows = [(f, m) for f, m in clips if m.get("chapter") == ch and m["text"].strip().lower() not in skip]
        rows = rows[::-1] if tail else rows
        out, total = [], 0.0
        for f, m in rows:
            x = load(f)
            out += [x, np.zeros(int(0.45 * SR), np.float32)]
            total += len(x) / SR + 0.45
            if total >= seconds:
                break
        out = out[::-1] if tail else out
        return np.concatenate(out) if out else np.zeros(0, np.float32)

    def clip_for(text):
        t = text.strip().lower()
        return next((load(f) for f, m in clips if m["text"].strip().lower() in (t, t.rstrip(".") + ".")), None)

    note = ""
    if what == "opening":
        book_title = json.loads((work / "chapters.json").read_text()).get("title", work.name)
        line = (plan["intro"].get("title") or book_title).splitlines()[0]
        if not music.track(plan["intro"].get("track", "")):
            raise ValueError("Choose an opening song first.")
        t = clip_for(line)
        note = "" if t is not None else " (the spoken title is not made yet: silence stands in for it)"
        title = np.concatenate([np.zeros(int(0.3 * SR), np.float32), t if t is not None else np.zeros(int(3 * SR), np.float32)])
        head, tail = music.opening(title, plan, SR)
        out = np.concatenate([head, music.apply_tail(body(all_ch[0], 35, skip=(line.lower(),)), tail)])
    elif what == "bridge":
        if chapter not in names:
            raise ValueError("Choose a section for the bridge.")
        if not music.track(music.bridge_spec(plan, chapter).get("track", "")):
            raise ValueError("That section has no bridge song: choose a bridge song first.")
        t = clip_for(names[chapter])
        note = "" if t is not None else " (the section title is not made yet: silence stands in for it)"
        section = np.concatenate([np.zeros(int(1.0 * SR), np.float32), t if t is not None else np.zeros(int(3 * SR), np.float32),
                                  np.zeros(int(2.8 * SR), np.float32), body(chapter, 40, skip=(names[chapter].lower(),))])
        plan = {**plan, "bridges": {**(plan.get("bridges") or {}), str(chapter): music.bridge_spec(plan, chapter)}}      # this section gets a bridge for the preview
        out = music.wrap(section, chapter, all_ch, plan, secs, SR)
    elif what == "ending":
        if not music.track(plan["outro"].get("track", "")):
            raise ValueError("Choose an ending song first.")
        out = music.ending(body(all_ch[-1], 30, tail=True), plan, SR)
    else:
        raise ValueError("Preview the opening, a bridge or the ending.")
    out_dir.mkdir(parents=True, exist_ok=True)
    name = name or what + (f"_{int(chapter)}" if what == "bridge" else "")
    wav, mp3 = out_dir / f"{name}.wav", out_dir / f"{name}.mp3"
    sf.write(wav, out, SR)
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(wav), "-af", "loudnorm=I=-16:TP=-1.5:LRA=11", "-ar", "24000", "-ac", "1",
                    "-b:a", "96k", str(mp3)], check=True)
    wav.unlink()
    return mp3, f"{len(out) / SR:.0f} s{note}"
