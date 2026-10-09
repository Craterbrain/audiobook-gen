"""Drive TTS engines over segments with per-chunk disk caching."""
import hashlib
import json
import re
import time
from pathlib import Path

import numpy as np
import soundfile as sf

from .lexicon import load_preprocessor, normalize
from .runstats import averages, clock, record

SENT_RE = re.compile(r"(?<=[.!?;:])\s+")
POOL = ["af_sarah", "am_echo", "bf_emma", "am_liam", "af_nicole", "bm_fable", "af_sky", "am_onyx"]


def chunk_text_cuts(text: str, max_chars: int = 300) -> list[tuple[str, str]]:
    """[(chunk, how it ends)]: "end" for a natural sentence boundary, "comma" when a very long sentence had to be cut at a comma,
    semicolon or dash, "space" when it had to be cut in the middle of a phrase. The pause after a chunk follows from this."""
    out, cur = [], ""
    for sent in SENT_RE.split(text.strip()):
        while len(sent) > max_chars:  # hard-split very long sentences, at the last comma/semicolon/dash in the back half, else at a space
            window = sent[:max_chars]
            stops = [i for i, ch in enumerate(window) if ch in ",;:—" and i >= int(max_chars * 0.45)]
            cut = stops[-1] if stops else window.rfind(" ")
            cut = cut if cut > 0 else max_chars
            piece, sent = sent[:cut + 1].strip(), sent[cut + 1:].strip()
            if cur:
                out.append((cur, "end")); cur = ""
            out.append((piece, "comma" if stops else "space"))
        if cur and len(cur) + 1 + len(sent) > max_chars:
            out.append((cur, "end")); cur = ""
        cur = f"{cur} {sent}".strip()
    if cur:
        out.append((cur, "end"))
    return [(c, k) for c, k in out if c]


def chunk_text(text: str, max_chars: int = 300) -> list[str]:
    return [c for c, _ in chunk_text_cuts(text, max_chars)]


def load_segments(work: Path, cfg: dict) -> list[dict]:
    """segments.json, or in one-narrator mode the same text with every speaker merged into the Narrator."""
    segs = json.loads((work / "segments.json").read_text())
    if (cfg.get("single_voice") or {}).get("enabled"):
        from .speakers import collapse_to_narrator
        segs = collapse_to_narrator(segs)
    return segs


def resolve_voice(role: str, cfg: dict) -> dict:
    from .voices import resolve
    single = cfg.get("single_voice") or {}
    if single.get("enabled") and single.get("voice"):
        return resolve(single["voice"])
    v = (cfg.get("voices") or {}).get(role)
    if v:
        return resolve(v)
    d = cfg["default_voice"]
    if role == "Narrator" or d.get("engine") != "kokoro":
        return resolve(d)
    # stable per-role voice for unconfigured characters, matching their gender when it is known
    from .casting import pool_for
    gender = (cfg.get("genders") or {}).get(role)
    pool = pool_for(gender) if gender in ("male", "female") else POOL
    return {"engine": "kokoro", "voice": pool[int(hashlib.md5(role.encode()).hexdigest(), 16) % len(pool)]}


_ENGINES: dict = {}  # shared across calls so the GUI doesn't reload models every run
DEFAULT_WORKERS = {"kokoro": 4, "f5": 1, "qwen3": 1, "chatterbox": 1}  # Kokoro: 1 worker 59 s, 3 -> 23 s, 4 -> 17 s on John 1; F5 saturates the GPU (no gain)


def lexicon_mode(ename: str, cfg: dict) -> str:
    """How an engine receives the lexicon. Kokoro takes IPA. Chatterbox and Qwen3 take ONLY respellings you typed (Chatterbox
    also the tested Bible respellings); F5 follows the project's text_lexicon setting."""
    if ename == "kokoro":
        return "ipa"
    if ename == "chatterbox":
        return "verified"
    if ename == "qwen3":
        return "typed"
    return cfg.get("text_lexicon", "respell")


def clip_key(chunk: str, voice: dict, ename: str) -> str:
    """The name of a clip's file: from what it says, who says it (with every setting) and which engine."""
    return hashlib.sha1(json.dumps([chunk, voice, ename]).encode()).hexdigest()[:16]


def load_overrides(work: Path) -> dict:
    """Your corrections, by "segment id:chunk number": {"text": spoken text, "voice": {settings to change}}."""
    try:
        return json.loads((Path(work) / "clip_overrides.json").read_text())
    except (OSError, ValueError):
        return {}


def unload_engines() -> None:
    """Free the models this process holds (and the GPU memory with them), so the queue can have the GPU back."""
    import gc
    _ENGINES.clear()
    gc.collect()
    try:
        import torch
        if hasattr(torch, "xpu") and torch.xpu.is_available():
            torch.xpu.empty_cache()
    except Exception:
        pass


def get_engine(name: str, device_pref: str = "auto", slot: int = 0, precision: str = "float16"):
    key = (name, device_pref, slot)
    if key not in _ENGINES:  # lazy: only load what the config needs
        if name == "kokoro":
            from .tts.kokoro_engine import KokoroEngine
            _ENGINES[key] = KokoroEngine(device_pref)
        elif name in ("qwen3", "chatterbox"):   # run in their own virtualenvs
            from .tts import remote_engine
            _ENGINES[key] = {"qwen3": remote_engine.Qwen3Engine, "chatterbox": remote_engine.ChatterboxEngine}[name](device_pref)
        else:
            from .tts.f5_engine import F5Engine
            _ENGINES[key] = F5Engine(device_pref, precision=precision)
    return _ENGINES[key]


def synthesize_iter(work: Path, cfg: dict, only_chapters: set[int] | None = None):
    """Generator: yields (done_chunks, total_chunks, message) after every clip. Clips are made by several worker
    threads (cfg["workers"], default Kokoro 3 / F5 1), each with its own model copy; finished clips are cached."""
    import queue
    from concurrent.futures import ThreadPoolExecutor, as_completed

    segs = load_segments(work, cfg)
    segs = [s for s in segs if not only_chapters or s["chapter"] in only_chapters]
    clips = work / "clips"
    clips.mkdir(exist_ok=True)
    workers = {**DEFAULT_WORKERS, **(cfg.get("workers") or {})}
    device, precision = cfg.get("device", "auto"), cfg.get("f5_precision", "float16")

    plan, todo, total = [], [], 0   # plan: (seg, [clip file names]); todo: clips still to make
    meta = {}                       # clip file -> what it should say and who says it (the quality check and the Corrections tab read this)
    overrides = load_overrides(work)
    lex = {}
    pairs = [chunk_text_cuts(normalize(seg["text"]), cfg.get("max_chunk_chars", 300)) for seg in segs]
    raw = [[c for c, _ in p] for p in pairs]
    cuts = [[k for _, k in p] for p in pairs]                 # how each chunk ends: the join pauses by this
    from . import refine
    refined = refine.load(work)                      # the language pass, if the book has had one: added question marks, delivery
    if refined:
        raw = refine.apply_fixes(raw, segs, refined)
    feel = None   # per-chunk emotion settings (cfg "emotion: true"), for engines that take them
    if cfg.get("emotion") and any(resolve_voice(sg["speaker"], cfg)["engine"] == "chatterbox" for sg in segs):
        from .emotion import delivery
        yield 0, max(1, sum(map(len, raw))), "Reading the emotion of each sentence"
        feel = delivery(segs, raw, cfg.get("emotion_base", 0.0), refine.hints(segs, raw, refined) if refined else None)
    for si, seg in enumerate(segs):
        voice = resolve_voice(seg["speaker"], cfg)
        ename = voice["engine"]
        if ename not in lex:
            lex[ename] = load_preprocessor(work, lexicon_mode(ename, cfg))
        files = []
        for ci, chunk in enumerate(lex[ename].substitute(c) for c in raw[si]):
            if feel and ename == "chatterbox":
                voice = {**voice, **feel[si][ci]}
            ov = overrides.get(f"{seg['id']}:{ci}") or {}
            spoken = ov.get("text") or chunk
            vv = {**voice, **ov["voice"]} if ov.get("voice") else voice          # a correction changes this clip only
            key = clip_key(spoken, vv, ename)
            files.append(f"{key}.wav")
            meta[f"{key}.wav"] = {"text": spoken, "speaker": seg["speaker"], "engine": ename, "voice": vv, "key": f"{seg['id']}:{ci}",
                                  "orig": raw[si][ci], "chapter": seg["chapter"], "seg": seg["id"], "ci": ci, "cut": cuts[si][ci]}
            total += 1
            if not (clips / f"{key}.wav").exists():
                todo.append((seg["speaker"], spoken, vv, ename, clips / f"{key}.wav"))
        plan.append((seg, files))

    (work / "clips_meta.json").write_text(json.dumps(meta, ensure_ascii=False))
    done = total - len(todo)
    if done:
        yield done, total, f"{done} clip(s) already made"
    pools: dict[str, queue.Queue] = {}
    for ename in {t[3] for t in todo}:
        pools[ename] = queue.Queue()
        for slot in range(max(1, int(workers.get(ename, 1)))):
            pools[ename].put(get_engine(ename, device, slot, precision))

    def make(task):
        speaker, chunk, voice, ename, path = task
        eng = pools[ename].get()
        try:
            t0 = time.time()
            audio = eng.synth(chunk, voice)
            sf.write(path, audio, eng.sample_rate)
            return speaker, chunk, len(audio) / eng.sample_rate, time.time() - t0
        finally:
            pools[ename].put(eng)

    t_audio = t_wall = 0.0
    start = time.time()
    run_id, spent, made = f"{work.name}-{int(start)}", {}, {}    # per engine: clip seconds, characters made

    def note(final: bool = False):
        """Share the run's wall time between engines by the time their clips took, then record chars/second."""
        busy = sum(spent.values()) or 1.0
        for en, sec in spent.items():
            record(run_id, en, int(workers.get(en, 1)), made[en], (time.time() - start) * sec / busy)

    left = {}                                  # characters still to make, per engine
    for t in todo:
        left[t[3]] = left.get(t[3], 0) + len(t[1])
    avg = averages()

    def eta() -> str:
        """Time left from each engine's running average speed (nothing until that speed is known)."""
        if not left or not all(e in avg for e in left):
            return ""
        return f" · about {clock(sum(n / avg[e]['cps'] for e, n in left.items()))} left"

    for ename in dict.fromkeys(t[3] for t in todo):      # one engine at a time: two models computing on the GPU together can hang it
        group = [t for t in todo if t[3] == ename]
        with ThreadPoolExecutor(pools[ename].qsize()) as ex:
            futs = []
            for t in group:
                f = ex.submit(make, t); f.en = ename; futs.append(f)
            for fut in as_completed(futs):
                speaker, chunk, dur, wall = fut.result()
                done += 1
                t_audio += dur
                en = fut.en
                spent[en] = spent.get(en, 0.0) + wall
                made[en] = made.get(en, 0) + len(chunk)
                left[en] = max(0, left.get(en, 0) - len(chunk))
                if done % 20 == 0:
                    note()
                yield done, total, f"{speaker}: {chunk[:60]}{eta()}"
    t_wall = time.time() - start
    note(True)
    (work / "clips.json").write_text(json.dumps({seg["id"]: files for seg, files in plan}, indent=1))
    if t_audio:
        yield total, total, f"synth done: {t_audio:.0f}s audio in {t_wall:.0f}s (RTF {t_wall / t_audio:.2f})"


def synthesize(work: Path, cfg: dict, only_chapters: set[int] | None = None) -> None:
    msg = ""
    for done, total, msg in synthesize_iter(work, cfg, only_chapters):
        if done % 25 == 0:
            print(f"[progress] {done}/{total}", flush=True)         # the job queue reads these
    print("[synth]", msg)
