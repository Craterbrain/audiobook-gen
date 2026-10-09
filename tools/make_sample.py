"""A short listening sample from the first finished clips of a book: the first stretch of clips (in book order) that all exist,
joined with the book's usual pauses, normalised, saved as an MP3.
Usage: python tools/make_sample.py [--work work/<book>] [--seconds 30] [--send]
Without --work it uses the book the queue is making right now. --send puts the file on your ntfy topic."""
import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def main() -> None:
    import numpy as np
    import soundfile as sf
    from audiobook_gen import jobqueue
    ap = argparse.ArgumentParser()
    ap.add_argument("--work"); ap.add_argument("--seconds", type=float, default=30); ap.add_argument("--send", action="store_true")
    a = ap.parse_args()
    work = Path(a.work) if a.work else None
    if work is None:
        run = jobqueue.now_running()
        if not run:
            raise SystemExit("no book is being made right now; give --work")
        work = Path(run["work"])
    meta = json.loads((work / "clips_meta.json").read_text())
    parts, total, sr = [], 0.0, None
    for f, m in meta.items():
        p = work / "clips" / f
        if not p.exists():
            if total >= a.seconds:
                break
            parts, total = [], 0.0                  # the run broke before it was long enough: start again after the gap
            continue
        audio, sr = sf.read(p, dtype="float32")
        if len(audio) / sr < 2.0 and not parts:      # skip a heading read on its own at the start
            continue
        parts.append(audio); total += len(audio) / sr + 0.4
        if total >= a.seconds:
            break
    if total < a.seconds * 0.6:
        raise SystemExit(f"only {total:.0f} s of unbroken clips so far; try again later")
    gap = np.zeros(int(sr * 0.4), np.float32)
    joined = np.concatenate([x for p in parts for x in (p, gap)])
    out = ROOT / "out" / "samples"
    out.mkdir(parents=True, exist_ok=True)
    wav, mp3 = out / f"{work.name}_sample.wav", out / f"{work.name}_sample.mp3"
    sf.write(wav, joined, sr)
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(wav), "-af", "loudnorm=I=-18:TP=-2:LRA=11", "-ac", "1", "-b:a", "96k", str(mp3)], check=True)
    wav.unlink()
    print(f"{mp3} ({len(joined) / sr:.0f} s, {mp3.stat().st_size // 1024} KB) from the first {len(parts)} clips")
    if a.send:
        ok = jobqueue.notify_file(str(mp3), f"A {len(joined) / sr:.0f}-second sample from the start of {work.name.replace('_', ' ')}.", "Sample")
        print("sent to your ntfy topic" if ok else "could not send (no topic set, or ntfy unreachable)")


if __name__ == "__main__":
    main()
