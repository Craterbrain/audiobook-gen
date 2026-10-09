"""Pacing profiles: measure how a narrator paces speech from an audiobook (.m4b) and its ebook, fit a table of pauses, and keep it
with the voice (voices/library/<name>/pacing.json). The assembler uses the profile of the book's narrator voice, or the default table
(data/narrator_pacing.json) when the voice has none.

Measuring: the recogniser (Whisper, on the CPU) times every word of several stretches of the audiobook; the words are matched to the
ebook text, and the silence at each word boundary is grouped by what the text has there (sentence end, comma, dash, paragraph...) and by
whether it is inside quoted speech. Only numbers are kept (counts and durations); the recogniser's transcripts are cached locally under
work/_m4b_cache."""
import hashlib
import json
import re
import time
from difflib import SequenceMatcher
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
CACHE = ROOT / "work" / "_m4b_cache"
DEFAULT_FILE = ROOT / "data" / "narrator_pacing.json"
SENT = re.compile(r"(?<=[.!?…])[\"”’')\]]*\s+")


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


SHORT = 100          # ms: below this a pause is "almost none"
KINDS = {            # name -> which measured boundaries it is made from
    "sentence_narration": lambda r: r["base"] in ("period", "question", "exclaim") and not r["in_quote"] and "closequote" not in r["cls"] and "openquote" not in r["cls"],
    "sentence_dialogue": lambda r: r["base"] in ("period", "question", "exclaim") and (r["in_quote"] or "closequote" in r["cls"]) and "openquote" not in r["cls"],
    "paragraph": lambda r: r["base"] == "paragraph",
    "comma_narration": lambda r: r["base"] == "comma" and not r["in_quote"] and "closequote" not in r["cls"],
    "comma_dialogue": lambda r: r["base"] == "comma" and r["in_quote"] and "closequote" not in r["cls"],
    "before_tag": lambda r: r["base"] == "comma" and "closequote" in r["cls"],
    "speaker_change": lambda r: r["base"] in ("period", "question", "exclaim") and "openquote" in r["cls"],
}


def label_feelings(recs: list[dict], progress=print) -> None:
    """Add "feel" (1 - probability of neutral, from the pipeline's emotion classifier) to each record, from the sentence before it."""
    from . import emotion
    progress("labelling the sentences with the emotion classifier")
    for i in range(0, len(recs), 256):
        part = recs[i:i + 256]
        for r, p in zip(part, emotion.probabilities([r["before"][:600] or "." for r in part])):
            r["feel"] = round(1 - p.get("neutral", 0.0), 3)


def fit_table(recs: list[dict]) -> dict:
    """Records -> {"kinds": {...}, "emotion": {...}}: for each boundary kind the share of almost-no pauses and a log-normal fit of the rest;
    and how much the feeling of the sentence before shifts sentence-end and paragraph pauses."""
    table = {}
    for name, pick in KINDS.items():
        v = np.array([r["pause"] for r in recs if pick(r)], dtype=float)
        if len(v) < 15:
            continue
        short = v < SHORT
        long = v[~short]
        entry = {"n": int(len(v)), "short_share": round(float(short.mean()), 3), "median_ms": round(float(np.median(v))), "short_range_ms": [20, SHORT]}
        if len(long) >= 8:
            logs = np.log(long)
            entry.update(mu=round(float(logs.mean()), 4), sigma=round(float(logs.std()), 4),
                         lo_ms=round(float(np.percentile(long, 3))), hi_ms=round(float(np.percentile(long, 97))))
        table[name] = entry
    emotion = {}
    for name, pick in (("sentence", lambda r: r["base"] in ("period", "question", "exclaim")), ("paragraph", lambda r: r["base"] == "paragraph")):
        sub = [r for r in recs if pick(r) and "feel" in r]
        if len(sub) < 100:
            continue
        X = np.array([[1, r["feel"], np.log(r["words"] + 1), 1.0 if (r["in_quote"] or "closequote" in r["cls"]) else 0.0] for r in sub])
        y = np.log1p(np.array([r["pause"] for r in sub]))
        beta = np.linalg.lstsq(X, y, rcond=None)[0]
        resid = y - X @ beta
        se = np.sqrt(np.diag(resid @ resid / (len(y) - X.shape[1]) * np.linalg.inv(X.T @ X)))
        emotion[name] = {"coefficient": round(float(beta[1]), 3), "t": round(float(beta[1] / se[1]), 1), "n": len(sub)}
    return {"kinds": table, "emotion": emotion}


def measure_profile(m4b: str, epub: str, windows: int = 4, minutes: float = 30, threads: int = 6, progress=print) -> dict:
    """Measure a narrator from an audiobook and its ebook. Returns the fitted table (with "records": the number of boundaries)."""
    from . import m4b as m4blib, sync
    from .extract import extract
    chs = m4blib.chapters(m4b)
    idx = sync.book_index(extract(epub, str(ROOT / "work" / "_tmp_pauses"), None)["chapters"])
    picks = [chs[int(len(chs) * (k + 0.5) / windows)] for k in range(windows)]
    recs = []
    for n, ch in enumerate(picks, 1):
        start = ch["start"] + min(60.0, max(0.0, (ch["end"] - ch["start"]) / 4))
        end = min(ch["end"], start + minutes * 60)
        progress(f"stretch {n} of {windows}: chapter {ch['index']}, {(end - start) / 60:.0f} min (the recogniser runs on the CPU)")
        recs += window_records(m4b, idx, start, end, threads)
        progress(f"  {len(recs)} pauses measured so far")
    if len(recs) < 300:
        raise RuntimeError(f"only {len(recs)} pauses could be matched to the book; check that the ebook is the one that was read")
    label_feelings(recs, progress)
    table = fit_table(recs)
    table["records"] = len(recs)
    return table


# --- profiles kept with the voices -------------------------------------------------------------------------------------------
def profile_path(voice: str) -> Path:
    from .voices import LIB
    return LIB / voice / "pacing.json"


def save_profile(voice: str, table: dict, source: str = "") -> Path:
    path = profile_path(voice)
    path.write_text(json.dumps({"about": "Pauses measured on this narrator (numbers only).", "measured": time.strftime("%Y-%m-%d"),
                                "source": source, "records": table.get("records"), "kinds": table["kinds"], "emotion": table["emotion"]}, indent=1))
    return path


def load_profile(voice: str) -> dict | None:
    try:
        return json.loads(profile_path(voice).read_text())
    except (OSError, ValueError):
        return None


def remove_profile(voice: str) -> bool:
    p = profile_path(voice)
    if p.exists():
        p.unlink()
        return True
    return False


def narrator_voice_name(cfg: dict) -> str:
    """The saved voice that reads the book's narration ("" if it is not a library voice)."""
    single = cfg.get("single_voice") or {}
    voice = single["voice"] if single.get("enabled") and single.get("voice") else (cfg.get("voices") or {}).get("Narrator") or cfg.get("default_voice") or {}
    return voice.get("library", "") if isinstance(voice, dict) else ""


def tables_for(cfg: dict) -> tuple[dict, dict, str]:
    """(kinds, emotion coefficients, where they came from) for a book: its narrator voice's profile, else the default table."""
    name = narrator_voice_name(cfg)
    prof = load_profile(name) if name else None
    if prof and prof.get("kinds"):
        return prof["kinds"], prof.get("emotion", {}), name
    try:
        d = json.loads(DEFAULT_FILE.read_text())
        return d.get("kinds", {}), d.get("emotion", {}), "default"
    except (OSError, ValueError):
        return {}, {}, "none"
