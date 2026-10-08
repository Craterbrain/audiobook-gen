"""Friendly voice choices for the cast screen: Kokoro presets with readable names, plus saved clones."""
import json

from . import voices as vlib

KOKORO = {  # id: (name, description)
    "af_heart": ("Heart", "US female, warm"), "af_bella": ("Bella", "US female"),
    "af_nicole": ("Nicole", "US female, soft"), "af_sarah": ("Sarah", "US female"),
    "af_sky": ("Sky", "US female, light"), "am_adam": ("Adam", "US male"),
    "am_echo": ("Echo", "US male"), "am_eric": ("Eric", "US male"), "am_liam": ("Liam", "US male"),
    "am_michael": ("Michael", "US male, steady"), "am_onyx": ("Onyx", "US male, deep"),
    "am_puck": ("Puck", "US male, playful"), "bf_emma": ("Emma", "UK female"),
    "bf_isabella": ("Isabella", "UK female"), "bm_daniel": ("Daniel", "UK male"),
    "bm_fable": ("Fable", "UK male, storyteller"), "bm_george": ("George", "UK male, classic narrator"),
    "bm_lewis": ("Lewis", "UK male"),
}


def voice_choices() -> list[tuple[str, str]]:
    """[(label, key)] for a dropdown: saved clones first, then Kokoro presets."""
    clones = [(f"{n} — your clone ({lab})", f"clone:{n}" + suffix) for n in vlib.list_voices()
              for suffix, lab in CLONE_ENGINES if _engine_ready(suffix)]
    kokoro = [(f"{name} — {desc}", f"kokoro:{vid}") for vid, (name, desc) in KOKORO.items()]
    from .voicepack import PACKS
    packs = []
    for p in sorted(PACKS.glob("*.pt")):
        meta = json.loads(p.with_suffix(".json").read_text()) if p.with_suffix(".json").exists() else {}
        label = meta.get("label") or p.stem
        desc = f"{meta['tags']} (your voice)" if meta.get("tags") else "your voice, from a recording"
        packs.append((f"{label} — {desc}", f"kokoro:pack:{p.stem}"))
    return clones + packs + kokoro


CLONE_ENGINES = [("", "F5"), ("@qwen3", "Qwen3"), ("@chatterbox", "Chatterbox")]   # every clone can be voiced by each


def _engine_ready(suffix: str) -> bool:
    if not suffix:
        return True
    from .tts import remote_engine
    return {"@qwen3": remote_engine.Qwen3Engine, "@chatterbox": remote_engine.ChatterboxEngine}[suffix].available()


def key_of(voice: dict) -> str | None:
    if voice.get("library"):
        return f"clone:{voice['library']}" + ("" if voice.get("engine", "f5") == "f5" else f"@{voice['engine']}")
    if voice.get("engine") == "kokoro" and voice.get("voice"):
        return f"kokoro:{voice['voice']}"
    return None


def voice_of(key: str, speed: float = 1.0) -> dict:
    """Config voice dict for a dropdown key. A clone keeps its saved pace unless the slider moved."""
    kind, _, name = key.partition(":")
    if kind == "clone":
        name, _, engine = name.partition("@")
        v = {"engine": engine or "f5", "library": name}
        if abs(float(speed) - vlib.load_voice(name).get("speed", 1.0)) > 1e-6:
            v["speed"] = float(speed)
        return v
    return {"engine": "kokoro", "voice": name, "speed": float(speed)}


FEMALE = ["af_heart", "af_bella", "af_nicole", "af_sarah", "af_sky", "bf_emma", "bf_isabella"]
MALE = ["am_adam", "am_echo", "am_eric", "am_liam", "am_michael", "am_onyx", "am_puck",
        "bm_daniel", "bm_fable", "bm_george", "bm_lewis"]


def pool_for(gender: str | None) -> list[str]:
    return FEMALE if gender == "female" else MALE if gender == "male" else MALE + FEMALE


def assign_by_gender(roles: list[str], genders: dict, keep: dict | None = None) -> dict:
    """{role: Kokoro voice dict}: each character gets a voice of their gender, distinct while the pool lasts
    (roles are taken in the order given, i.e. most lines first). Roles in `keep` (e.g. clones) are untouched."""
    keep = keep or {}
    used: dict[str, int] = {}
    out = {}
    for role in roles:
        if role in keep or role == "Narrator":
            continue
        pool = pool_for(genders.get(role))
        pick = min(pool, key=lambda v: (used.get(v, 0), pool.index(v)))   # least-used, then earliest
        used[pick] = used.get(pick, 0) + 1
        out[role] = {"engine": "kokoro", "voice": pick}
    return out
