"""Find a spelling that makes F5-TTS say a word the way an IPA says it.

F5 reads plain letters (its vocabulary is characters; there is no phoneme input), so the only lever is the
spelling. There is no reliable IPA -> spelling rule, so this searches: candidate spellings are generated from
the IPA, each is spoken by F5 in the chosen voice, a phoneme recognizer (wav2vec2, espeak phone set) writes
down what it heard, and candidates are ranked by how close those phones are to the IPA. Stress is not
measured; listen to the top few."""
import itertools
import random
import re
from difflib import SequenceMatcher
from pathlib import Path

import numpy as np

REC_MODEL = "facebook/wav2vec2-lv-60-espeak-cv-ft"
_EAR = {}
CARRIER = "Philip findeth {w}, and saith unto him."

# ways to write each phone with plain English letters, most likely first
VOWELS = {
    "ə": ["uh", "a", "e", "u", "ah", "i", "o"], "ʌ": ["u", "uh", "o"], "æ": ["a", "ah"], "ɛ": ["e", "eh"],
    "ɪ": ["i", "ih", "y"], "i": ["ee", "y", "ea", "ie"], "iː": ["ee", "ea", "ie"], "ɑ": ["ah", "a", "o"],
    "ɔ": ["aw", "o", "au"], "u": ["oo", "u", "ue"], "uː": ["oo", "ue"], "ʊ": ["oo", "u"], "ɜ": ["er", "ur", "ir"],
    "eɪ": ["ay", "ai", "a", "ei"], "aɪ": ["eye", "i", "igh", "y"], "oʊ": ["oh", "o", "ow", "oa"], "aʊ": ["ow", "ou"],
    "ɔɪ": ["oy", "oi"], "ɑː": ["ah", "ar"], "e": ["ay", "e"], "o": ["o", "oh"], "a": ["ah", "a"],
}
CONS = {"θ": ["th"], "ð": ["th", "dh"], "ʃ": ["sh"], "ʒ": ["zh", "j"], "tʃ": ["ch"], "dʒ": ["j", "g"], "j": ["y"],
        "ŋ": ["ng"], "k": ["k", "c", "ck"], "ɡ": ["g"], "g": ["g"], "ɹ": ["r"], "r": ["r"], "ʁ": ["r"], "x": ["kh"],
        "w": ["w"], "h": ["h"], "s": ["s", "ss"], "z": ["z", "s"], "f": ["f"], "v": ["v"], "l": ["l", "ll"],
        "m": ["m"], "n": ["n"], "p": ["p"], "b": ["b"], "t": ["t"], "d": ["d"]}
SIMILAR = [{"ə", "ʌ", "ɐ", "ɜ"}, {"ɪ", "i", "iː"}, {"u", "uː", "ʊ"}, {"æ", "a", "ɛ"}, {"eɪ", "e", "ɛ"},
           {"oʊ", "o", "ɔ", "əʊ"}, {"ɑ", "ɑː", "a", "ɒ"}, {"ɔ", "ɒ", "ɑ"}, {"l", "ɫ"}, {"ɹ", "r", "ʁ", "ɾ"}]


def ear():
    if "m" not in _EAR:
        import torch
        from transformers import AutoModelForCTC, AutoProcessor

        from .tts.device import pick_device
        dev = pick_device("auto")
        _EAR["p"] = AutoProcessor.from_pretrained(REC_MODEL)
        _EAR["m"] = AutoModelForCTC.from_pretrained(REC_MODEL).to(dev).eval()
        _EAR["dev"] = dev
    return _EAR


def free_ear() -> None:
    import gc

    import torch
    _EAR.clear()
    gc.collect()
    if hasattr(torch, "xpu") and torch.xpu.is_available():
        torch.xpu.empty_cache()


def recognize(audio: np.ndarray, sr: int) -> list[str]:
    import torch
    from scipy.signal import resample_poly

    e = ear()
    if sr != 16000:
        from math import gcd
        g = gcd(16000, sr)
        audio = resample_poly(audio, 16000 // g, sr // g).astype("float32")
    x = e["p"](audio, sampling_rate=16000, return_tensors="pt").input_values.to(e["dev"])
    with torch.no_grad():
        ids = e["m"](x).logits.argmax(-1)
    return e["p"].batch_decode(ids)[0].split()


def segment(ipa: str) -> list[str]:
    """IPA string -> phone tokens (longest match against the recognizer's own inventory); stress marks dropped."""
    vocab = set(ear()["p"].tokenizer.get_vocab())
    s = re.sub(r"[ˈˌ/‿]", "", ipa)
    out, i = [], 0
    while i < len(s):
        for n in (3, 2, 1):
            if s[i:i + n] in vocab:
                out.append(s[i:i + n]); i += n
                break
        else:
            i += 1
    split = {"əl": ["ə", "l"], "ən": ["ə", "n"], "əm": ["ə", "m"], "ɚ": ["ɜ", "ɹ"]}  # recognizer writes them apart
    return [q for t in out for q in split.get(t, [t])]


def _cost(a: str, b: str) -> float:
    if a == b:
        return 0.0
    if a.rstrip("ː") == b.rstrip("ː"):
        return 0.3                                  # only vowel length differs
    return 0.5 if any(a.rstrip("ː") in g and b.rstrip("ː") in g for g in SIMILAR) else 1.0


def distance(heard: list[str], target: list[str]) -> float:
    """How well `target` phones are found inside `heard` (a whole carrier sentence): weighted edit distance with
    free extra phones before and after the best matching stretch, divided by the target length."""
    if not heard:
        return 1.0
    prev = [float(j) for j in range(len(target) + 1)]
    prev[0] = 0.0
    best = prev[-1]
    for h in heard:
        cur = [0.0]                                  # starting later in `heard` costs nothing
        for j, t in enumerate(target, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + _cost(h, t)))
        prev = cur
        best = min(best, prev[-1])                   # ending earlier in `heard` costs nothing
    return best / max(1, len(target))


def candidates(ipa: str, word: str, limit: int = 60, seed: int = 0) -> list[str]:
    """Spellings built from the IPA's syllables (every choice list is short, so we sample if it explodes)."""
    phones = segment(ipa)
    slots = []
    for p in phones:
        if p in VOWELS:
            slots.append(VOWELS[p][:4])
        elif p.rstrip("ː") in VOWELS:
            slots.append(VOWELS[p.rstrip("ː")][:3])
        else:
            slots.append(CONS.get(p, [p])[:2])
    combos = ["".join(c) for c in itertools.product(*slots)]
    rng = random.Random(seed)
    if len(combos) > limit:
        combos = rng.sample(combos, limit)
    out = [word.lower()] + [c for c in combos if c != word.lower()]
    # no consecutive triple letters, no leftover non-letters
    return list(dict.fromkeys(c for c in out if re.fullmatch(r"[a-z]+", c) and not re.search(r"(.)\1\1", c)))


def search(word: str, ipa: str, voice: dict, f5, outdir: Path | None = None, limit: int = 60,
           top: int = 8, progress=None) -> list[dict]:
    """Rank spellings for `word` (IPA `ipa`) in `voice` (an F5 voice dict). Returns [{spelling, distance, heard}],
    best first; with outdir the top few are saved as WAVs for listening."""
    import soundfile as sf

    target = segment(ipa)
    cands = candidates(ipa, word, limit)
    rows = []
    for k, c in enumerate(cands):
        if progress:
            progress(k / len(cands), f"Trying spelling {k + 1}/{len(cands)}: {c}")
        a = np.asarray(f5.synth(CARRIER.format(w=c.capitalize()), voice), dtype=np.float32)
        heard = recognize(a, f5.sample_rate)
        rows.append({"spelling": c, "distance": round(distance(heard, target), 3), "heard": " ".join(heard), "audio": a})
    rows.sort(key=lambda r: r["distance"])
    if outdir:
        outdir.mkdir(parents=True, exist_ok=True)
        for f in outdir.glob("*.wav"):
            f.unlink()
        lines = [f"{word}: target IPA {ipa} -> phones {' '.join(target)}", ""]
        for n, r in enumerate(rows[:top], 1):
            a = r["audio"] * (0.9 / (np.abs(r["audio"]).max() or 1))
            sf.write(outdir / f"{n:02d}_{r['spelling']}.wav", a, f5.sample_rate, subtype="PCM_16")
            lines.append(f"{n:02d}_{r['spelling']}.wav   distance {r['distance']}   heard: {r['heard']}")
        (outdir / "INDEX.txt").write_text("\n".join(lines) + "\n")
    for r in rows:
        r.pop("audio", None) if not outdir else None
    return rows


def target_ipa(entry: dict) -> str:
    """The sound to aim for: the entry's own IPA (typed/seeded), else espeak in its language, else espeak US English."""
    from .lexicon import misaki_to_std, suggest_ipa
    if entry.get("ipa"):
        return entry["ipa"]
    lang = entry.get("lang") or "en-us"
    ps = suggest_ipa(entry["term"], lang)
    return ps if entry.get("lang") else misaki_to_std(ps)   # US English comes back in Kokoro's letters


def rank_lexicon(work: Path, voice: dict, f5, tries: int = 20, margin: float = 0.06, progress=None,
                 kinds=("name", "place", "demonym", "foreign", "word")) -> list[dict]:
    """Rank spellings for every flagged word in lexicon.json and keep the best. For each word: the generated
    spellings (plain spelling included) are spoken in `voice`, scored against the target IPA, and the top 3 are
    stored under "ranking". The plain spelling is kept unless another beats it by more than `margin`
    (no point re-spelling a word F5 already says right). Words you typed a respelling for are left alone;
    finished words are skipped, so an interrupted pass resumes."""
    import json
    path = work / "lexicon.json"
    lex = json.loads(path.read_text())
    todo = [e for e in lex if not e.get("known") and e.get("kind") in kinds and "ranking" not in e
            and not (e.get("source") == "user" and e.get("respell"))]
    todo.sort(key=lambda e: -e.get("count", 0))
    report = []
    for k, e in enumerate(todo):
        if progress:
            progress(k / max(1, len(todo)), f"Ranking spellings for {e['term']} ({k + 1}/{len(todo)})")
        ipa = target_ipa(e)
        try:
            rows = search(e["term"], ipa, voice, f5, None, limit=tries)
        except Exception as ex:  # a word the recognizer/espeak can't handle shouldn't stop the pass
            e["ranking"] = []
            e["rank_note"] = f"skipped: {ex}"
            path.write_text(json.dumps(lex, indent=2, ensure_ascii=False))
            continue
        raw = next((r for r in rows if r["spelling"] == e["term"].lower()), None)
        best = rows[0]
        keep_raw = raw is not None and raw["distance"] <= best["distance"] + margin
        e["ranking"] = [{"spelling": r["spelling"], "score": r["distance"]} for r in rows[:3]]
        e["rank_raw"] = raw["distance"] if raw else None
        if not e.get("ipa"):
            e["ipa"] = ipa
        if not keep_raw and not e.get("respell"):
            e["respell"], e["respell_src"] = best["spelling"], "ranked"
        report.append({"term": e["term"], "count": e.get("count", 0), "target": ipa, "kept_plain": keep_raw,
                       "chosen": e["term"].lower() if keep_raw else best["spelling"], "best": best["distance"],
                       "plain": e["rank_raw"]})
        path.write_text(json.dumps(lex, indent=2, ensure_ascii=False))
    free_ear()
    return report
