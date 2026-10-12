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


CLONE_ENGINES = [("", "F5"), ("@qwen3", "Qwen3"), ("@chatterbox", "Chatterbox"), ("@higgs", "Higgs")]   # every clone can be voiced by each


def _engine_ready(suffix: str) -> bool:
    if not suffix:
        return True
    from .tts import remote_engine
    from .tts.higgs_engine import HiggsEngine
    return {"@qwen3": remote_engine.Qwen3Engine, "@chatterbox": remote_engine.ChatterboxEngine, "@higgs": HiggsEngine}[suffix].available()


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
        if engine == "higgs":                 # Higgs has its own pace (a stretch after the clip is made): the slider scales it, 1.0 = as saved
            if abs(float(speed) - 1.0) > 1e-6:
                v["pace"] = round(float(vlib.resolve({"engine": "higgs", "library": name})["pace"]) * float(speed), 3)
            return v
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


# ---- picking a voice: the engine first, then a voice of that engine ------------------------------------------------------
ENGINES = {   # id: (label, key suffix, what it is like, GPU group: engines of different groups are loaded one after the other)
    "kokoro": ("Kokoro", None, "fast preset voices", "light"),
    "higgs": ("Higgs", "@higgs", "clone; plain text, names right, slow, 9.4 GB of GPU", "higgs"),
    "chatterbox": ("Chatterbox", "@chatterbox", "clone; expressive, slower, names need respelling", "heavy"),
    "f5": ("F5", "", "clone; light", "light"),
    "qwen3": ("Qwen3", "@qwen3", "clone", "heavy"),
}
_READY: dict[str, bool] = {}


def engine_ready(engine: str) -> bool:
    """Is this engine usable here? (Remembered until `forget_ready`: the check looks at virtualenvs and the installed transformers.)"""
    if engine not in _READY:
        suffix = ENGINES[engine][1]
        _READY[engine] = True if engine == "kokoro" else _engine_ready(suffix)
    return _READY[engine]


def forget_ready() -> None:
    _READY.clear()


def engine_of_key(key: str | None) -> str | None:
    """The engine a dropdown key belongs to: kokoro:... -> kokoro, clone:Name -> f5, clone:Name@higgs -> higgs."""
    if not key:
        return None
    kind, _, rest = key.partition(":")
    if kind == "kokoro":
        return "kokoro"
    return rest.partition("@")[2] or "f5" if kind == "clone" else None


def engine_choices(current: str | None = None) -> list[tuple[str, str]]:
    """[(label, engine)] for the Engine menu: the engines that work here, and `current` even if it does not (marked)."""
    out = []
    for e, (label, _, _, _) in ENGINES.items():
        if engine_ready(e):
            out.append((label, e))
        elif e == current:
            out.append((f"{label} ⚠ not installed", e))
    return out


def _clone_label(key: str) -> str:
    return key.partition(":")[2].partition("@")[0]


def voices_for(engine: str, current_key: str | None = None) -> list[tuple[str, str]]:
    """[(label, key)] of one engine's voices. A `current_key` that is no longer available is kept in the list, marked missing."""
    allv = voice_choices_all()
    if engine == "kokoro":
        out = [(lab, k) for lab, k in allv if k.startswith("kokoro:")]
    else:
        suffix = ENGINES[engine][1]
        out = [(_clone_label(k), k) for lab, k in allv if engine_of_key(k) == engine]
    if current_key and current_key not in {k for _, k in out} and engine_of_key(current_key) == engine:
        name = _clone_label(current_key) if current_key.startswith("clone:") else current_key.partition(":")[2]
        out.append((f"{name} (missing)", current_key))
    return out


def voice_choices_all() -> list[tuple[str, str]]:
    """Every voice of every engine whether or not the engine is installed (the dropdown for a role already set to one must still show it)."""
    clones = [(f"{n} — your clone ({lab})", f"clone:{n}" + suffix) for n in vlib.list_voices() for suffix, lab in CLONE_ENGINES]
    return clones + [x for x in voice_choices() if x[1].startswith("kokoro:")]


def switch_engine(key: str | None, engine: str, gender: str | None = None, taken=(), last_clone: str | None = None) -> str:
    """The dropdown key to use when a role moves to `engine`. A clone keeps its clone (Walter (F5) -> Walter (Higgs)); from Kokoro to a clone engine
    it takes the last clone used, else the first saved; to Kokoro it takes a preset of the role's gender that nobody else has yet."""
    if engine == "kokoro":
        if engine_of_key(key) == "kokoro":
            return key
        pool = pool_for(gender)
        taken = set(taken)
        free = [v for v in pool if f"kokoro:{v}" not in taken]
        return f"kokoro:{(free or pool)[0]}"
    name = _clone_label(key) if key and key.startswith("clone:") else (last_clone or (vlib.list_voices() or [None])[0])
    if not name:
        raise ValueError("Save a clone voice first (Voice lab): Higgs, Chatterbox, F5 and Qwen3 speak with a clone.")
    return f"clone:{name}{ENGINES[engine][1]}"


def configured_voice(role: str, cfg: dict) -> dict:
    """The voice dict a role is set to, as stored, or the default's (single-narrator mode: the narrator's). No library lookup."""
    single = cfg.get("single_voice") or {}
    if single.get("enabled") and single.get("voice"):
        return single["voice"]
    return (cfg.get("voices") or {}).get(role) or cfg.get("default_voice") or {"engine": "kokoro", "voice": "bm_george"}


def voice_problem(voice: dict) -> str | None:
    """Why this voice cannot be used right now, or None."""
    if voice.get("library") and voice["library"] not in vlib.list_voices():
        return f"{voice['library']} is no longer in the voice library"
    e = voice.get("engine", "f5" if voice.get("library") else "kokoro")
    if e in ENGINES and not engine_ready(e):
        return f"{ENGINES[e][0]} is not installed"
    return None


def mix_note(voices) -> str:
    """One line on what a cast's engines cost: nothing for light mixes, a model swap when Higgs shares a book with Chatterbox or Qwen3."""
    engines = []
    for v in voices:
        e = v.get("engine", "f5" if v.get("library") else "kokoro")
        if e in ENGINES and e not in engines:
            engines.append(e)
    if not engines:
        return ""
    names = " + ".join(ENGINES[e][0] for e in engines)
    if "higgs" in engines and any(e in ("chatterbox", "qwen3") for e in engines):
        return f"ⓘ This cast uses {names}. Higgs and {'Chatterbox' if 'chatterbox' in engines else 'Qwen3'} cannot share the GPU, so the book is made in separate passes (one model swap, about a minute or two)."
    return f"ⓘ This cast uses {names}: one GPU pass."
