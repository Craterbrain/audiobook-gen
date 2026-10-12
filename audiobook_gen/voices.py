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
HIGGS_DEFAULTS = dict(temperature=0.7, top_k=50, top_p=0.95, tag="<|prosody:expressive_high|>", pace=1.08)   # Higgs: temperature 0.7 is warmer than the 1.0 it ships with
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


def save_voice(name: str, audio_path: str, ref_text: str, params: dict, notes: str = "", higgs: dict | None = None) -> str:
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
    (d / "voice.json").write_text(json.dumps({
        "name": name, "ref_text": ref_text.strip(), "notes": notes,
        "params": {k: params[k] for k in (*DEFAULTS, *EXTRA) if k in params},
        **({"higgs": {k: v for k, v in higgs.items() if k in HIGGS_DEFAULTS}} if higgs else {}),
        "saved": time.strftime("%Y-%m-%d %H:%M")}, indent=2, ensure_ascii=False))
    return name


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
