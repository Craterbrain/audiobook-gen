"""Check how a TTS engine really pronounces a word: speak it in a short carrier sentence, transcribe the audio to
phonemes (wav2vec2 espeak phoneme recogniser, CPU), and score the best-matching stretch against the target IPA.
Used to test respellings for Chatterbox against the IPA dictionary."""
import json
import re
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
MODEL = "facebook/wav2vec2-lv-60-espeak-cv-ft"
CARRIER = "He called out, {}, and then he left."
CACHE = ROOT / "work" / "respell_check_cache.json"

_REC = None


def _recogniser():
    global _REC
    if _REC is None:
        from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
        _REC = (Wav2Vec2Processor.from_pretrained(MODEL), Wav2Vec2ForCTC.from_pretrained(MODEL).eval())
    return _REC


def phonemes(audio: np.ndarray, sr: int, timed: bool = False):
    """Phoneme string; with timed=True also the start time (s) of each phoneme."""
    import torch
    import torchaudio.functional as AF
    proc, model = _recogniser()
    x = torch.from_numpy(np.asarray(audio, dtype="float32"))
    if sr != 16000:
        x = AF.resample(x, sr, 16000)
    inp = proc(x.numpy(), sampling_rate=16000, return_tensors="pt")
    with torch.no_grad():
        ids = model(inp.input_values).logits.argmax(-1)
    text = proc.batch_decode(ids)[0]
    if not timed:
        return text
    pad, prev, times = proc.tokenizer.pad_token_id, None, []
    for f, t in enumerate(ids[0].tolist()):
        if t != prev and t != pad and proc.tokenizer.convert_ids_to_tokens(t) != proc.tokenizer.word_delimiter_token:
            times.append(round(f * 0.02, 2))
        prev = t
    return text, times


# --- comparison on a reduced phoneme alphabet -------------------------------------------------------------
MAP = {"ɹ": "r", "ɾ": "t", "ɝ": "ɜr", "ɚ": "ər", "ɐ": "ə", "ɜ": "ə", "ʌ": "ə", "ɨ": "ɪ", "ᵻ": "ɪ", "ɑ": "a", "ɒ": "a",
       "ɔ": "o", "oʊ": "o", "əʊ": "o", "eɪ": "e", "ɛ": "e", "æ": "e", "ʊ": "u", "ɫ": "l", "ɡ": "g", "ʧ": "tʃ", "ʤ": "dʒ",
       "ɵ": "θ", "ɬ": "l", "ʔ": "", "ʰ": "", "ʲ": "", "ˠ": "", "̃": "", "̩": "", "̯": "", "ᵊ": "ə"}
VOWELS = set("aeiouəɪʊɔɜ")


def norm(ipa: str) -> list[str]:
    s = re.sub(r"[ˈˌːˑ͡‿.\s,;'ʼ·‧\-]", "", ipa)
    s = s.replace("tʃ", "ʧ").replace("dʒ", "ʤ")
    for k in sorted(MAP, key=len, reverse=True):
        s = s.replace(k, MAP[k])
    s = s.replace("ʧ", "tʃ").replace("ʤ", "dʒ")
    out = []
    for ch in s:
        if out and ch == out[-1] and ch in VOWELS:      # long vowels, doubled letters
            continue
        out.append(ch)
    # affricates as one unit
    tok, i = [], 0
    while i < len(out):
        if out[i] in "td" and i + 1 < len(out) and out[i + 1] in "ʃʒ":
            tok.append(out[i] + out[i + 1]); i += 2
        else:
            tok.append(out[i]); i += 1
    return tok


def _sub(a: str, b: str) -> float:
    if a == b:
        return 0.0
    if a[0] in VOWELS and b[0] in VOWELS:
        return 0.5
    if a in ("ə", "ɪ") and b in ("ə", "ɪ"):
        return 0.3
    return 1.0


def _gap(a: str) -> float:
    return 0.5 if a in ("ə", "ɪ", "h", "r") else 1.0


def local_score(target: list[str], heard: list[str]):
    """1.0 = the heard phonemes contain the target exactly; best stretch of `heard` (free ends) vs `target`."""
    n, m = len(target), len(heard)
    if not n:
        return 0.0, "", (0, 0)
    D = np.zeros((n + 1, m + 1)); start = np.zeros((n + 1, m + 1), dtype=int)
    for i in range(1, n + 1):
        D[i, 0] = D[i - 1, 0] + _gap(target[i - 1])
    for j in range(m + 1):
        start[0, j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            c = [(D[i - 1, j - 1] + _sub(target[i - 1], heard[j - 1]), start[i - 1, j - 1]),
                 (D[i - 1, j] + _gap(target[i - 1]), start[i - 1, j]),
                 (D[i, j - 1] + _gap(heard[j - 1]), start[i, j - 1])]
            D[i, j], start[i, j] = min(c, key=lambda t: t[0])
    j = int(np.argmin(D[n]))
    cost = D[n, j]
    return max(0.0, 1 - cost / sum(_gap(t) for t in target)), "".join(heard[start[n, j]:j]), (int(start[n, j]), j)


PAUSE = 0.28      # seconds between two phonemes of one name that count as a pause


def judge(target_ipa: str, heard: str, times: list | None = None) -> tuple[float, str, int]:
    """(score, matched phonemes, pauses inside the name). A pause lowers the score by 0.1."""
    toks = heard.split()
    units, utimes = [], []
    for i, tk in enumerate(toks):
        for u in norm(tk):
            if units and u == units[-1] and u in VOWELS:
                continue
            units.append(u); utimes.append(times[i] if times and i < len(times) else None)
    score, stretch, (a, b) = local_score(norm(target_ipa), units)
    pauses = 0
    if times and len(toks) == len(times):
        ts = [t for t in utimes[a:b] if t is not None]
        pauses = sum(1 for x, y in zip(ts, ts[1:]) if y - x > PAUSE)
    return max(0.0, score - 0.1 * pauses), stretch, pauses


# --- running an engine ------------------------------------------------------------------------------------
def _cache() -> dict:
    try:
        return json.loads(CACHE.read_text())
    except Exception:
        return {}


def check(items: list[tuple[str, str, str]], voice: dict, engine, progress=print, seed: int = 1) -> list[dict]:
    """items: (name, text_to_speak, target_ipa). Returns [{name, text, score, heard}]. Cached on (text, voice, seed)."""
    import soundfile as sf
    cache, out = _cache(), []
    vkey = json.dumps({k: voice[k] for k in sorted(voice) if k != "library"}, sort_keys=True, default=str)
    for n, item in enumerate(items):
        name, text, ipa = item[:3]
        frame = item[3] if len(item) > 3 else CARRIER          # a phrase with {} where the name goes
        text = frame.format(text)
        k = f"{vkey}|{seed}|{text}"
        if k not in cache or (isinstance(cache[k], str) and "-" in text):     # older entries have no timing
            audio = engine.synth(text, {**voice, "seed": seed})
            cache[k] = list(phonemes(audio, engine.sample_rate, timed=True))
            if n % 20 == 0:
                CACHE.write_text(json.dumps(cache, ensure_ascii=False))
                progress(f"  {n}/{len(items)}")
        heard, times = (cache[k], None) if isinstance(cache[k], str) else cache[k]
        score, stretch, pauses = judge(ipa, heard, times)
        out.append({"name": name, "text": text, "score": round(score, 2), "heard": stretch, "pauses": pauses})
    CACHE.write_text(json.dumps(cache, ensure_ascii=False))
    return out
