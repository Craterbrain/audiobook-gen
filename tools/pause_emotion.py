"""Do a narrator's pauses depend on the feeling of what was just said? Measures sentence-end and paragraph pauses over several stretches
of an audiobook (see narrator_pauses.py), labels the sentence before each pause with the text emotion classifier the pipeline uses, and
compares. Only numbers are saved; the recogniser's transcripts are cached locally under work/_m4b_cache (not committed).
Usage: python tools/pause_emotion.py --m4b book.m4b --epub book.epub [--windows 4] [--minutes 30]"""
import argparse
import hashlib
import json
import re
import sys
from difflib import SequenceMatcher
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))
CACHE = ROOT / "work" / "_m4b_cache"
SENT = re.compile(r"(?<=[.!?…])[\"”’')\]]*\s+")


def words_for(m4b: str, start: float, end: float, threads: int, cached_only: bool = False) -> list | None:
    key = hashlib.sha1(f"{m4b}|{start:.0f}|{end:.0f}".encode()).hexdigest()[:12]
    f = CACHE / f"asr_{key}.json"
    if f.exists():
        return json.loads(f.read_text())
    if cached_only:
        return None
    import soundfile as sf
    import torch
    from transformers import pipeline
    from audiobook_gen import m4b as m4blib
    torch.set_num_threads(threads)
    wav = m4blib.extract_wav(m4b, start, end, 16000, CACHE / f"asr_{key}.wav")
    audio, sr = sf.read(wav, dtype="float32")
    asr = pipeline("automatic-speech-recognition", model="openai/whisper-small.en", device="cpu", dtype=torch.float32)
    res = asr({"raw": audio, "sampling_rate": sr}, return_timestamps="word", chunk_length_s=30, stride_length_s=5)
    words = [[w["text"], w["timestamp"][0], w["timestamp"][1]] for w in res["chunks"] if w["timestamp"][0] is not None and w["timestamp"][1] is not None]
    f.write_text(json.dumps(words))
    wav.unlink(missing_ok=True)
    return words


def window_records(m4b: str, idx: dict, start: float, end: float, threads: int, cached_only: bool = False) -> list[dict]:
    """One record per matched word boundary: its pause (ms), the book's separator, the sentence before and after, and whether it is speech."""
    import soundfile as sf
    from audiobook_gen import m4b as m4blib, sync
    words = words_for(m4b, start, end, threads, cached_only)
    if words is None:
        return []
    wav = m4blib.extract_wav(m4b, start, end, 16000, CACHE / "pe_tmp.wav")
    audio, sr = sf.read(wav, dtype="float32")
    wav.unlink(missing_ok=True)
    a_tok = [(tok, wi) for wi, (t, _, _) in enumerate(words) for tok, _, _ in sync.tokenize(t)]
    toks, spans = idx["tokens"], idx["spans"]
    chain = sync.anchors([t for t, _ in a_tok], toks)
    if not chain:
        return []
    (j0, i0), (j1, i1) = chain[0], chain[-1]
    lo, hi = max(0, i0 - j0 - 6), min(len(toks), i1 + (len(a_tok) - j1) + 6)
    pairs = {}
    for b in SequenceMatcher(None, [t for t, _ in a_tok], toks[lo:hi], autojunk=False).get_matching_blocks():
        for k in range(b.size):
            pairs[lo + b.b + k] = a_tok[b.a + k][1]
    hop = int(sr * 0.01)
    rms = np.sqrt(np.convolve(audio ** 2, np.ones(hop) / hop, mode="same"))[::hop]
    level = 20 * np.log10(np.percentile(rms, 80) + 1e-9)
    quiet = 20 * np.log10(rms + 1e-9) < level - 30

    def pause(wi):
        mid = (words[wi][2] + words[wi + 1][1]) / 2
        lo_f, hi_f = int(max(0, mid - 0.6) * 100), int(min(len(rms) / 100, mid + 0.6) * 100)
        runs, st = [], None
        for f in range(lo_f, hi_f + 1):
            q = f < hi_f and quiet[f]
            if q and st is None:
                st = f
            elif not q and st is not None:
                runs.append((st, f)); st = None
        runs = [(x, y) for x, y in runs if y - x >= 2]
        if not runs:
            return 0.0
        mf = mid * 100
        x, y = min(runs, key=lambda r: 0 if r[0] <= mf <= r[1] else min(abs(r[0] - mf), abs(r[1] - mf)))
        return (y - x) * 10.0 if (x - 8 <= mf <= y + 8) else 0.0
    from narrator_pauses import classify
    sent_cache: dict[int, list[tuple[int, int]]] = {}

    def sentences(ci):
        if ci not in sent_cache:
            text, out, s = idx["chapters"][ci]["text"], [], 0
            for m in SENT.finditer(text):
                out.append((s, m.start())); s = m.end()
            out.append((s, len(text)))
            sent_cache[ci] = out
        return sent_cache[ci]
    recs = []
    for b in sorted(pairs):
        nb = b + 1
        if nb not in pairs or spans[nb][0] != spans[b][0] or pairs[nb] != pairs[b] + 1:
            continue
        ci = spans[b][0]
        text = idx["chapters"][ci]["text"]
        sep = text[spans[b][2]:spans[nb][1]]
        if len(sep) > 12:
            continue
        cls = classify(sep)
        base = cls.split("+")[0]
        if base not in ("period", "question", "exclaim", "paragraph", "comma"):
            continue
        ss = sentences(ci)
        k = next((n for n, (s0, e0) in enumerate(ss) if s0 <= spans[b][1] <= e0 + 1), None)
        if k is None:
            continue
        before = text[ss[k][0]:ss[k][1] + 1].strip()
        after = text[ss[k + 1][0]:ss[k + 1][1] + 1].strip() if k + 1 < len(ss) else ""
        para = text[text.rfind("\n", 0, spans[b][2]) + 1:spans[b][2]]            # the paragraph up to this word
        inside = (para.count("“") - para.count("”")) > 0 or (("“" not in para and "”" not in para) and para.count('"') % 2 == 1)
        recs.append({"cls": cls, "base": base, "pause": pause(pairs[b]), "before": before, "after": after,
                     "speech": bool(re.search(r"[“”\"]", before)), "in_quote": bool(inside), "words": len(before.split())})
    return recs


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--m4b", required=True); ap.add_argument("--epub", required=True)
    ap.add_argument("--windows", type=int, default=4); ap.add_argument("--minutes", type=float, default=30)
    ap.add_argument("--threads", type=int, default=6)
    ap.add_argument("--cached-only", action="store_true", help="use only stretches whose transcript is already cached")
    ap.add_argument("--out", default=str(ROOT / "work" / "pause_emotion.json"))
    a = ap.parse_args()
    from audiobook_gen import emotion, m4b as m4blib, sync
    from audiobook_gen.extract import extract
    chs = m4blib.chapters(a.m4b)
    idx = sync.book_index(extract(a.epub, str(ROOT / "work" / "_tmp_pauses"), None)["chapters"])
    picks = [chs[int(len(chs) * (k + 0.5) / a.windows)] for k in range(a.windows)]
    recs = []
    for ch in picks:
        start = ch["start"] + 60
        end = min(ch["end"], start + a.minutes * 60)
        print(f"window: chapter {ch['index']} {(end - start) / 60:.0f} min", flush=True)
        recs += window_records(a.m4b, idx, start, end, a.threads, a.cached_only)
        print(f"  records so far: {len(recs)}", flush=True)
    print("labelling the sentences with the emotion classifier...", flush=True)
    P = emotion.probabilities([r["before"][:600] or "." for r in recs])
    for r, p in zip(recs, P):
        r["emo"] = max(p, key=p.get); r["emo_p"] = round(p[r["emo"]], 3); r["feel"] = round(1 - p.get("neutral", 0), 3)
    Path(a.out).write_text(json.dumps([{k: v for k, v in r.items() if k not in ("before", "after")} for r in recs]))   # numbers only
    report(recs)


def report(recs: list[dict]) -> None:
    from scipy.stats import spearmanr
    def line(name, v):
        v = np.array(v); print(f"  {name:30s} n={len(v):4d}  median {np.median(v):5.0f}  p25 {np.percentile(v, 25):5.0f}  p75 {np.percentile(v, 75):5.0f} ms")
    for fam, bases in (("sentence ends (. ? !)", ("period", "question", "exclaim")), ("paragraph ends", ("paragraph",)), ("commas", ("comma",))):
        sub = [r for r in recs if r["base"] in bases]
        if len(sub) < 20:
            continue
        print(f"\n== {fam}: {len(sub)} boundaries; by the feeling of the sentence before")
        for e in ("neutral", "joy", "sadness", "anger", "fear", "surprise", "disgust"):
            v = [r["pause"] for r in sub if r["emo"] == e]
            if len(v) >= 5:
                line(e, v)
        rho, p = spearmanr([r["feel"] for r in sub], [r["pause"] for r in sub])
        print(f"  pause vs how emotional the sentence is (0-1): Spearman rho {rho:+.2f} (p={p:.3f})")
        rho2, p2 = spearmanr([r["words"] for r in sub], [r["pause"] for r in sub])
        print(f"  pause vs sentence length in words:           Spearman rho {rho2:+.2f} (p={p2:.3f})")
        for name, flt in (("in speech (inside quotes)", lambda r: r["speech"]), ("in narration", lambda r: not r["speech"])):
            v = [r["pause"] for r in sub if flt(r)]
            if len(v) >= 5:
                line(name, v)
        if fam == "commas":                                  # dialogue against narration, the way the text marks it
            print("  -- commas by where they sit:")
            for name, flt in (("inside quoted speech", lambda r: r["in_quote"] and "closequote" not in r["cls"]),
                              ("in narration (outside quotes)", lambda r: not r["in_quote"] and "closequote" not in r["cls"]),
                              ("last mark of a quote, before its tag", lambda r: "closequote" in r["cls"])):
                v = [r["pause"] for r in sub if flt(r)]
                if len(v) >= 5:
                    line(name, v)
                    short = sum(1 for x in v if x < 100) / len(v)
                    print(f"{'':34s}{100 * short:.0f}% have almost no pause (< 100 ms)")
        else:
            print("  -- by where they sit:")
            for name, flt in (("inside quoted speech", lambda r: r["in_quote"] or "closequote" in r["cls"]), ("in narration", lambda r: not r["in_quote"] and "closequote" not in r["cls"])):
                v = [r["pause"] for r in sub if flt(r)]
                if len(v) >= 5:
                    line(name, v)
        hi = [r["pause"] for r in sub if r["feel"] >= 0.6]; lo = [r["pause"] for r in sub if r["feel"] <= 0.2]
        if len(hi) >= 5 and len(lo) >= 5:
            line("emotional (feel >= 0.6)", hi); line("calm (feel <= 0.2)", lo)
        # does emotion still matter once length and speech/narration are accounted for?
        X = np.array([[1, r["feel"], np.log(r["words"] + 1), 1.0 if r["speech"] else 0.0] for r in sub]); y = np.log1p(np.array([r["pause"] for r in sub]))
        beta, res, *_ = np.linalg.lstsq(X, y, rcond=None)
        resid = y - X @ beta; s2 = resid @ resid / (len(y) - X.shape[1]); se = np.sqrt(np.diag(s2 * np.linalg.inv(X.T @ X)))
        print("  regression on log(1+pause): " + ", ".join(f"{n} {b:+.2f} (t={b / s:+.1f})" for n, b, s in zip(("feel", "log length", "speech"), beta[1:], se[1:])))


if __name__ == "__main__":
    main()
