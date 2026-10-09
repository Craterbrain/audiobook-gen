"""Quality check of the finished clips, before the audiobook is built.

Text-to-speech sometimes rushes, mumbles, trails off into dead air or produces nothing. Each clip is measured (how long the
speech really lasts, how much of it is silence) against what its text should take at this book's usual pace. Clips that stand out
are made again with other random seeds and the best attempt is kept. Whatever cannot be fixed is listed in qc_report.json."""
import json
import re
import time
from pathlib import Path

import numpy as np
import soundfile as sf

MIN_SECONDS_FOR_SILENCE = 1.5   # a clip this short is mostly padding by nature ("No!"): judge it by its pace and dead air, not its silent share
MIN_CHARS = 25          # shorter clips vary too much (a single word has any length) to judge by pace
FAST, SLOW = 1.6, 0.45  # speech pace outside [SLOW, FAST] x the book's median pace is suspect
DEAD_AIR = 2.0          # seconds of silence inside a clip
DASH_PAUSE = 1.5        # more allowed for each dash or ellipsis in the text: the voice pauses there (headings full of dashes do)
SILENT_DB = -45.0
MAX_SILENT_SHARE = 0.6
RETRIES = (101, 202, 303, 404)       # seeds tried for a flagged clip
REPAIRABLE = ("chatterbox", "qwen3")  # engines that take a seed


def measure(path: Path) -> dict:
    """{seconds, speech (seconds without leading/trailing silence), longest_gap, silent_share}; or {"broken": why}."""
    try:
        audio, sr = sf.read(path, dtype="float32")
    except Exception as e:                                           # missing, truncated, not a sound file
        return {"broken": f"unreadable ({type(e).__name__})"}
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if not len(audio) or not np.isfinite(audio).all():
        return {"broken": "empty or not a number"}
    win = max(1, int(sr * 0.02))
    n = len(audio) // win
    if n == 0:
        return {"broken": "too short"}
    rms = np.sqrt((audio[: n * win].reshape(n, win) ** 2).mean(axis=1) + 1e-12)
    quiet = 20 * np.log10(rms) < SILENT_DB
    loud = np.flatnonzero(~quiet)
    seconds = len(audio) / sr
    if not len(loud):
        return {"broken": "silent"}
    speech = (loud[-1] - loud[0] + 1) * win / sr
    gap = best = 0
    for q in quiet[loud[0]: loud[-1] + 1]:                          # silence between the first and last sound
        gap = gap + 1 if q else 0
        best = max(best, gap)
    return {"seconds": seconds, "speech": speech, "longest_gap": best * win / sr, "silent_share": float(quiet.mean())}


MARKUP = re.compile(r"\[([^\]]*)\]\(/[^)]*/\)")


def spoken_chars(text: str) -> int:
    """Characters as the reader would see them: the lexicon's [word](/IPA/) markup counts as just the word."""
    return len(MARKUP.sub(r"\1", text))


def pauses_in(text: str) -> int:
    return sum(text.count(d) for d in ("—", "–", "...", "…", " - "))


def problems(m: dict, chars: int, median_pace: float | None, text: str = "") -> list[str]:
    """Why this clip is suspect (empty = fine)."""
    if "broken" in m:
        return [m["broken"]]
    out, dashes = [], pauses_in(text)
    if m["longest_gap"] > DEAD_AIR + DASH_PAUSE * dashes:
        out.append(f"{m['longest_gap']:.1f} s of silence inside")
    if m["silent_share"] > MAX_SILENT_SHARE + (0.2 if dashes else 0) and m["seconds"] >= MIN_SECONDS_FOR_SILENCE:
        out.append("mostly silence")
    letters = sum(c.isalpha() for c in text) / max(1, len(text)) if text else 1.0
    if median_pace and chars >= MIN_CHARS and m["speech"] > 0 and letters >= 0.5:      # a row of asterisks has no natural pace
        pace = chars / m["speech"]
        if pace > FAST * median_pace:
            out.append(f"rushed ({pace:.0f} characters/s, usual {median_pace:.0f})")
        elif pace < SLOW * median_pace:
            out.append(f"drawn out ({pace:.0f} characters/s, usual {median_pace:.0f})")
    return out


def badness(m: dict, chars: int, median_pace: float | None) -> float:
    """How far a clip is from normal (0 = ideal); used to pick the best of several attempts."""
    if "broken" in m:
        return 1e9
    score = max(0.0, m["longest_gap"] - 0.8) + 3 * max(0.0, m["silent_share"] - 0.3)
    if median_pace and chars >= MIN_CHARS and m["speech"] > 0:
        score += abs(np.log((chars / m["speech"]) / median_pace)) * 2
    return float(score)


def scan(work: Path, progress=print) -> tuple[list[dict], dict]:
    """Measure every clip. Returns (flagged, stats). flagged: [{file, text, engine, why, chars}]."""
    meta = json.loads((work / "clips_meta.json").read_text())
    clips = work / "clips"
    measured = {}
    for i, (f, info) in enumerate(meta.items()):
        measured[f] = measure(clips / f)
        if i % 250 == 0:
            progress(f"  checked {i}/{len(meta)}")
    from .synth import load_overrides
    chosen = load_overrides(work)
    median = {}
    for eng in {i["engine"] for i in meta.values()}:
        paces = [spoken_chars(meta[f]["text"]) / m["speech"] for f, m in measured.items()
                 if meta[f]["engine"] == eng and "broken" not in m and spoken_chars(meta[f]["text"]) >= MIN_CHARS and m["speech"] > 0.5]
        median[eng] = float(np.median(paces)) if len(paces) >= 20 else None
    flagged = []
    for f, info in meta.items():
        why = problems(measured[f], spoken_chars(info["text"]), median[info["engine"]], info["text"])
        if why and info.get("key") in chosen:                    # you picked this take yourself: leave it alone
            why = []
        if why:
            flagged.append({"file": f, "text": info["text"], "engine": info["engine"], "why": why, "chars": spoken_chars(info["text"])})
    return flagged, {"checked": len(meta), "median_pace": median}


def repair(work: Path, cfg: dict, flagged: list[dict], median: dict, progress=print) -> dict:
    """Make each flagged clip again with other seeds; keep the best attempt in place. Returns the report."""
    from .synth import get_engine
    import soundfile
    meta = json.loads((work / "clips_meta.json").read_text())
    device, precision = cfg.get("device", "auto"), cfg.get("f5_precision", "float16")
    fixed, left = [], []
    for n, item in enumerate(flagged):
        info = meta[item["file"]]
        path = work / "clips" / item["file"]
        eng_name = info["engine"]
        best = measure(path)
        best_bad, tried = badness(best, item["chars"], median.get(eng_name)), 0
        if eng_name in REPAIRABLE:
            eng = get_engine(eng_name, device, 0, precision)
            for seed in RETRIES:
                if not problems(best, item["chars"], median.get(eng_name), info["text"]):
                    break
                tried += 1
                tmp = path.with_suffix(".try.wav")
                try:
                    soundfile.write(tmp, eng.synth(info["text"], {**info["voice"], "seed": seed}), eng.sample_rate)
                except Exception as e:                                       # a failed attempt just does not count
                    progress(f"  attempt failed: {e}")
                    continue
                m = measure(tmp)
                bad = badness(m, item["chars"], median.get(eng_name))
                if bad < best_bad:
                    tmp.replace(path)
                    best, best_bad = m, bad
                else:
                    tmp.unlink(missing_ok=True)
        (left if problems(best, item["chars"], median.get(eng_name), info["text"]) else fixed).append({**item, "attempts": tried})
        print(f"[progress] {n + 1}/{len(flagged)}", flush=True)
    return {"fixed": fixed, "unfixed": left}


def run(work: Path, cfg: dict, progress=print) -> dict:
    """Scan, repair what can be repaired, write qc_report.json. Returns the report."""
    t0 = time.time()
    if not (work / "clips_meta.json").exists():
        return {"checked": 0, "skipped": "no clip list (made before the checker existed)"}
    flagged, stats = scan(work, progress)
    progress(f"[qc] {len(flagged)} of {stats['checked']} clips look wrong")
    result = repair(work, cfg, flagged, stats["median_pace"], progress) if flagged else {"fixed": [], "unfixed": []}
    report = {"checked": stats["checked"], "flagged": len(flagged), "fixed": len(result["fixed"]), "unfixed": len(result["unfixed"]),
              "median_pace": stats["median_pace"], "seconds": round(time.time() - t0), "problems_left": result["unfixed"],
              "problems_fixed": result["fixed"]}
    (work / "qc_report.json").write_text(json.dumps(report, indent=1, ensure_ascii=False))
    progress(f"[qc] {report['fixed']} fixed, {report['unfixed']} could not be fixed ({report['seconds']} s)")
    return report
