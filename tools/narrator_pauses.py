"""Measure a real narrator's pauses: match a stretch of an audiobook (.m4b) to its ebook text, then time the silence at every
word boundary and group it by what the text has there (sentence end, comma, dash, paragraph, quote...).

Usage: python tools/narrator_pauses.py --m4b book.m4b --epub book.epub [--chapter N] [--minutes 10] [--out stats.json]
Only numbers are kept (counts and durations); no text or audio is saved. Runs the speech recogniser on the CPU."""
import argparse
import json
import re
import sys
from difflib import SequenceMatcher
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def classify(sep: str) -> str:
    """What the book has between two words."""
    if "\n" in sep:
        return "paragraph"
    s = sep.replace("’", "'")
    close, opn = bool(re.search(r"[”\"]", s)) and not re.match(r"^\s*[“\"]", s), bool(re.search(r"[“\"]", s.lstrip(".!?…,;:—– ”'’")))
    if "?" in s:
        base = "question"
    elif "!" in s:
        base = "exclaim"
    elif "…" in s or "..." in s:
        base = "ellipsis"
    elif "." in s:
        base = "period"
    elif "—" in s or "–" in s or " - " in s:
        base = "dash"
    elif ";" in s or ":" in s:
        base = "semicolon"
    elif "," in s:
        base = "comma"
    else:
        base = "word" if not re.search(r"[“\"”]", s) else "quote"
    if base in ("period", "question", "exclaim", "comma", "ellipsis", "dash") and re.search(r"[”\"]", s):
        base += "+closequote"
    elif base in ("period", "question", "exclaim", "ellipsis") and opn:
        base += "+openquote"
    return base


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--m4b", required=True); ap.add_argument("--epub", required=True)
    ap.add_argument("--chapter", type=int, help="audio chapter number (default: the one in the middle)")
    ap.add_argument("--skip", type=float, default=120, help="seconds into that chapter to start")
    ap.add_argument("--minutes", type=float, default=10)
    ap.add_argument("--threads", type=int, default=6)
    ap.add_argument("--max-ms", type=float, default=0, help="ignore silences longer than this (e.g. the gaps between separately made clips)")
    ap.add_argument("--out", default=str(ROOT / "work" / "narrator_pauses.json"))
    a = ap.parse_args()
    import soundfile as sf
    import torch
    from transformers import pipeline
    from audiobook_gen import m4b, sync
    from audiobook_gen.extract import extract
    torch.set_num_threads(a.threads)

    chs = m4b.chapters(a.m4b)
    ch = chs[(a.chapter - 1) if a.chapter else len(chs) // 2]
    start = ch["start"] + a.skip
    end = min(ch["end"], start + a.minutes * 60)
    print(f"audio: chapter {ch['index']} from {start:.0f} s, {(end - start) / 60:.1f} min", flush=True)
    wav = m4b.extract_wav(a.m4b, start, end, 16000, ROOT / "work" / "_m4b_cache" / "pause_excerpt.wav")
    audio, sr = sf.read(wav, dtype="float32")

    print("transcribing with word timestamps (CPU)...", flush=True)
    asr = pipeline("automatic-speech-recognition", model="openai/whisper-small.en", device="cpu", dtype=torch.float32)
    res = asr({"raw": audio, "sampling_rate": sr}, return_timestamps="word", chunk_length_s=30, stride_length_s=5)
    words = [(w["text"], w["timestamp"][0], w["timestamp"][1]) for w in res["chunks"] if w["timestamp"][0] is not None and w["timestamp"][1] is not None]
    print(f"{len(words)} words heard", flush=True)

    book = extract(a.epub, str(ROOT / "work" / "_tmp_pauses"), None)
    idx = sync.book_index(book["chapters"])
    a_tok = []                                           # recogniser words as tokens, remembering which word each came from
    for wi, (t, _, _) in enumerate(words):
        for tok, _, _ in sync.tokenize(t):
            a_tok.append((tok, wi))
    toks, spans = idx["tokens"], idx["spans"]
    chain = sync.anchors([t for t, _ in a_tok], toks)
    if not chain:
        raise SystemExit("could not find this audio in the ebook text")
    (j0, i0), (j1, i1) = chain[0], chain[-1]
    lo, hi = max(0, i0 - j0 - 6), min(len(toks), i1 + (len(a_tok) - j1) + 6)
    sm = SequenceMatcher(None, [t for t, _ in a_tok], toks[lo:hi], autojunk=False)
    pairs = {}                                           # book token index -> recogniser word index
    for b in sm.get_matching_blocks():
        for k in range(b.size):
            pairs[lo + b.b + k] = a_tok[b.a + k][1]
    print(f"matched {len(pairs)} of {len(a_tok)} words to the book ({100 * len(pairs) / len(a_tok):.0f}%)", flush=True)

    hop = int(sr * 0.01)
    rms = np.sqrt(np.convolve(audio ** 2, np.ones(hop) / hop, mode="same"))[::hop]
    level = 20 * np.log10(np.percentile(rms, 80) + 1e-9)
    quiet = 20 * np.log10(rms + 1e-9) < level - 30         # 30 dB under the usual speech level

    def pause_after(wi: int) -> float | None:
        """The silent stretch nearest the gap between two words (the recogniser's word edges are only roughly right, and often
        touch across a real pause, so look ±0.6 s around the gap's middle and take the silence that contains or is nearest to it)."""
        (_, _, e), (_, s, _) = words[wi], words[wi + 1]
        mid = (e + s) / 2
        lo_f, hi_f = int(max(0, mid - 0.6) * 100), int(min(len(rms) / 100, mid + 0.6) * 100)
        runs, start = [], None
        for f in range(lo_f, hi_f + 1):
            q = f < hi_f and quiet[f]
            if q and start is None:
                start = f
            elif not q and start is not None:
                runs.append((start, f)); start = None
        runs = [(x, y) for x, y in runs if y - x >= 2]
        if not runs:
            return 0.0
        mf = mid * 100
        x, y = min(runs, key=lambda r: 0 if r[0] <= mf <= r[1] else min(abs(r[0] - mf), abs(r[1] - mf)))
        return (y - x) * 10.0 if (x - 8 <= mf <= y + 8) else 0.0      # a silence that is not at this boundary does not count

    groups: dict[str, list[float]] = {}
    rate_chars = rate_time = 0.0
    keys = sorted(pairs)
    for b in keys:
        nb = b + 1
        if nb not in pairs or spans[nb][0] != spans[b][0] or pairs[nb] != pairs[b] + 1 and pairs[nb] != pairs[b]:
            continue
        if pairs[nb] == pairs[b]:
            continue
        sep = idx["chapters"][spans[b][0]]["text"][spans[b][2]:spans[nb][1]]
        pz = pause_after(pairs[b])
        if pz is None or len(sep) > 12 or (a.max_ms and pz > a.max_ms):
            continue
        groups.setdefault(classify(sep), []).append(pz)
    words_t = sum(max(0.0, words[i][2] - words[i][1]) for i in range(len(words)))
    chars = sum(len(re.sub(r"[^A-Za-z]", "", w[0])) for w in words)
    stats = {}
    for k, v in sorted(groups.items(), key=lambda kv: -len(kv[1])):
        v = np.array(v)
        pos = v[v > 0]
        stats[k] = {"n": int(len(v)), "median_ms": float(np.median(v)), "mean_ms": float(v.mean()), "p10": float(np.percentile(v, 10)),
                    "p25": float(np.percentile(v, 25)), "p75": float(np.percentile(v, 75)), "p90": float(np.percentile(v, 90)),
                    "log_mean": float(np.log(pos).mean()) if len(pos) > 3 else None, "log_sd": float(np.log(pos).std()) if len(pos) > 3 else None}
    out = {"minutes": round((end - start) / 60, 1), "words": len(words), "letters_per_spoken_second": round(chars / max(words_t, 1e-9), 1),
           "speech_level_db": round(level, 1), "classes": stats}
    Path(a.out).parent.mkdir(exist_ok=True)
    Path(a.out).write_text(json.dumps(out, indent=1))
    print(f"\n{'boundary':24s}{'n':>6s}{'median':>8s}{'p10':>7s}{'p90':>7s}   (ms of silence)")
    for k, s in stats.items():
        if s["n"] >= 5:
            print(f"{k:24s}{s['n']:6d}{s['median_ms']:8.0f}{s['p10']:7.0f}{s['p90']:7.0f}")
    print(f"\nspeaking rate: {out['letters_per_spoken_second']} letters per second of word time -> {a.out}")


if __name__ == "__main__":
    main()
