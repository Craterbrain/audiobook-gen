"""Saved cloned voices: voices/library/<name>/{ref.wav, voice.json}.

A library voice is a reference clip + its exact transcript + the F5-TTS settings that made it sound right
(pace, steps, guidance, seed...). Roles refer to it by name ({"engine": "f5", "library": "<name>"}), so
changing a voice here changes every project that uses it."""
import json
import re
import shutil
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
LIB = ROOT / "voices" / "library"
MAX_REF_SECONDS = 12  # F5-TTS clips longer references anyway

EXTRA = ("exaggeration", "cfg_weight")   # Chatterbox settings; stored only when a voice was tuned in Chatterbox
HIGGS_DEFAULTS = dict(temperature=0.7, top_k=50, top_p=0.95, max_new_tokens=1500, tag="<|prosody:expressive_high|>", pace=1.08)   # Higgs: temperature 0.7 is warmer than the 1.0 it ships with
DEFAULTS = dict(speed=1.0, nfe_step=32, cfg_strength=2.0, sway_sampling_coef=-1.0,
                cross_fade_duration=0.15, target_rms=0.1, seed=1234)


def clean_name(name: str) -> str:
    return re.sub(r"[^\w\- ]", "", name or "").strip()


def list_voices() -> list[str]:
    return sorted(p.parent.name for p in LIB.glob("*/voice.json"))


def load_voice(name: str) -> dict:
    """Voice dict ready for F5Engine.synth: engine, ref_audio (absolute), ref_text and settings."""
    d = LIB / name
    meta = json.loads((d / "voice.json").read_text())
    return {"engine": "f5", "ref_audio": str(d / "ref.wav"), "ref_text": meta["ref_text"],
            **{**DEFAULTS, **meta.get("params", {})}}


def save_voice(name: str, audio_path: str, ref_text: str, params: dict, notes: str = "", higgs: dict | None = None, gender: str | None = None) -> str:
    """Store the clip (mono 24 kHz, trimmed to 12 s) with its transcript and settings."""
    import numpy as np
    import soundfile as sf
    from scipy.signal import resample_poly

    name = clean_name(name)
    if not name:
        raise ValueError("Give the voice a name.")
    if not ref_text.strip():
        raise ValueError("The reference transcript is required: F5-TTS needs the exact words spoken in the clip.")
    from .audio_io import read_audio
    a, sr = read_audio(audio_path)                         # any format: wav, mp3, flac, ogg, opus, m4a, aac, wma, a video's soundtrack ...
    a = a[: int(MAX_REF_SECONDS * sr)]
    if sr != 24000:
        from math import gcd
        g = gcd(24000, sr)
        a = resample_poly(a, 24000 // g, sr // g).astype(np.float32)
    d = LIB / name
    d.mkdir(parents=True, exist_ok=True)
    sf.write(d / "ref.wav", a, 24000, subtype="PCM_16")
    chosen = gender if gender in GENDERS else None                  # None / "auto": from the pitch of the clip
    (d / "voice.json").write_text(json.dumps({
        "name": name, "ref_text": ref_text.strip(), "notes": notes,
        "gender": chosen or detect_gender(str(d / "ref.wav")), "gender_src": "user" if chosen else "auto",
        "params": {k: params[k] for k in (*DEFAULTS, *EXTRA) if k in params},
        **({"higgs": {k: v for k, v in higgs.items() if k in HIGGS_DEFAULTS}} if higgs else {}),
        "saved": time.strftime("%Y-%m-%d %H:%M")}, indent=2, ensure_ascii=False))
    return name


GENDERS = ("male", "female", "unknown")


def detect_gender(audio_path: str) -> str:
    """male / female / unknown from the pitch of a recording: the median fundamental frequency of its voiced frames (men sit near 85-155 Hz,
    women near 165-255 Hz; the band between is left unknown rather than guessed)."""
    import numpy as np
    import soundfile as sf
    import librosa
    try:
        y, sr = sf.read(audio_path, dtype="float32")
        y = y if y.ndim == 1 else y.mean(1)
        y = librosa.resample(y[: 20 * sr], orig_sr=sr, target_sr=16000) if sr != 16000 else y[: 20 * 16000]
        f0 = librosa.yin(y, fmin=60, fmax=400, sr=16000, frame_length=1024)
        rms = librosa.feature.rms(y=y, frame_length=1024, hop_length=256)[0]
        n = min(len(f0), len(rms))
        voiced = f0[:n][rms[:n] > 0.2 * rms.max()]
        if len(voiced) < 10:
            return "unknown"
        med = float(np.median(voiced))
    except Exception:
        return "unknown"
    return "male" if med < 155 else "female" if med > 175 else "unknown"


def voice_gender(name: str) -> str:
    """The saved gender of a clone voice. A voice saved before genders existed is detected from its clip once and the answer is kept (as "auto")."""
    path = LIB / name / "voice.json"
    try:
        meta = json.loads(path.read_text())
    except (OSError, ValueError):
        return "unknown"
    if meta.get("gender") in GENDERS:
        return meta["gender"]
    meta["gender"], meta["gender_src"] = detect_gender(str(LIB / name / "ref.wav")), "auto"
    try:
        path.write_text(json.dumps(meta, indent=2, ensure_ascii=False))
    except OSError:
        pass
    return meta["gender"]


def save_gender_auto(name: str) -> str:
    """Forget a gender you set and detect it from the clip again."""
    path = LIB / clean_name(name) / "voice.json"
    meta = json.loads(path.read_text())
    meta["gender"], meta["gender_src"] = detect_gender(str(LIB / clean_name(name) / "ref.wav")), "auto"
    path.write_text(json.dumps(meta, indent=2, ensure_ascii=False))
    return meta["gender"]


def set_gender(name: str, gender: str) -> None:
    if gender not in GENDERS:
        raise ValueError("A voice is male, female or unknown.")
    path = LIB / clean_name(name) / "voice.json"
    meta = json.loads(path.read_text())
    meta["gender"], meta["gender_src"] = gender, "user"
    path.write_text(json.dumps(meta, indent=2, ensure_ascii=False))


def delete_voice(name: str) -> None:
    shutil.rmtree(LIB / clean_name(name), ignore_errors=True)


def resolve(voice: dict) -> dict:
    """Expand {"engine": "f5", "library": name, ...overrides} into a full voice dict."""
    name = voice.get("library")
    if not name:
        return voice
    base = load_voice(name)
    out = {**base, **{k: v for k, v in voice.items() if k != "library"}}
    if out.get("engine") == "higgs":          # its own settings: the defaults, then what was saved with the voice, then what the book says
        saved = json.loads((LIB / name / "voice.json").read_text()).get("higgs", {})
        out = {**HIGGS_DEFAULTS, **saved, **{k: v for k, v in voice.items() if k not in ("library", "engine")}, "engine": "higgs",
               **{k: base[k] for k in ("ref_audio", "ref_text", "seed") if k in base and k not in voice}}
    return out
