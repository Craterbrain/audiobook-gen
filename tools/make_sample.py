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
    import yaml
    from audiobook_gen.assemble import build_chapter
    from audiobook_gen.synth import load_segments
    meta = json.loads((work / "clips_meta.json").read_text())
    cfg = yaml.safe_load((work / "config.yaml").read_text())
    for k, v in yaml.safe_load((ROOT / "config.yaml").read_text()).items():      # settings the book's own config leaves out
        cfg.setdefault(k, v)
    run, total = [], 0.0                     # the first unbroken stretch of finished clips that is long enough
    for f, m in meta.items():
        p = work / "clips" / f
        if not p.exists():
            if total >= a.seconds:
                break
            run, total = [], 0.0
            continue
        d = sf.info(p).duration
        if d < 2.0 and not run:              # skip a heading read on its own at the start
            continue
        run.append(f); total += d + 0.4
        if total >= a.seconds:
            break
    if total < a.seconds * 0.6:
        raise SystemExit(f"only {total:.0f} s of unbroken clips so far; try again later")
    seg_ids = list(dict.fromkeys(meta[f]["seg"] for f in run))
    clips = {sid: [f for f in run if meta[f]["seg"] == sid] for sid in seg_ids}
    segs = [s for s in load_segments(work, cfg) if s["id"] in clips]
    joined = build_chapter(segs, clips, work / "clips", cfg, meta)       # the same pauses, trimming and joins as the real audiobook
    sr, parts = cfg["sample_rate"], run
    joined = joined[max(0, int(sr * (cfg["pacing_ms"]["chapter_start"] - 150) / 1000)):]      # the book's chapter-start silence is not wanted here
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
