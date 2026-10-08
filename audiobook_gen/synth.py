"""Drive TTS engines over segments with per-chunk disk caching."""
import hashlib
import json
import re
import time
from pathlib import Path

import numpy as np
import soundfile as sf

from .lexicon import load_preprocessor, normalize

SENT_RE = re.compile(r"(?<=[.!?;:])\s+")
POOL = ["af_sarah", "am_echo", "bf_emma", "am_liam", "af_nicole", "bm_fable", "af_sky", "am_onyx"]


def chunk_text(text: str, max_chars: int = 300) -> list[str]:
    chunks, cur = [], ""
    for sent in SENT_RE.split(text.strip()):
        while len(sent) > max_chars:  # hard-split very long sentences at a comma/space
            cut = max(sent.rfind(",", 0, max_chars), sent.rfind(" ", 0, max_chars))
            cut = cut if cut > 0 else max_chars
            piece, sent = sent[:cut + 1].strip(), sent[cut + 1:].strip()
            if cur:
                chunks.append(cur); cur = ""
            chunks.append(piece)
        if cur and len(cur) + 1 + len(sent) > max_chars:
            chunks.append(cur); cur = ""
        cur = f"{cur} {sent}".strip()
    if cur:
        chunks.append(cur)
    return [c for c in chunks if c]


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
    lex = {}
    raw = [chunk_text(normalize(seg["text"]), cfg.get("max_chunk_chars", 300)) for seg in segs]
    feel = None   # per-chunk emotion settings (cfg "emotion: true"), for engines that take them
    if cfg.get("emotion") and any(resolve_voice(sg["speaker"], cfg)["engine"] == "chatterbox" for sg in segs):
        from .emotion import delivery
        yield 0, max(1, sum(map(len, raw))), "Reading the emotion of each sentence"
        feel = delivery(segs, raw, cfg.get("emotion_base", 0.0))
    for si, seg in enumerate(segs):
        voice = resolve_voice(seg["speaker"], cfg)
        ename = voice["engine"]
        if ename not in lex:
            lex[ename] = load_preprocessor(work, "ipa" if ename == "kokoro" else cfg.get("text_lexicon", "respell"))
        files = []
        for ci, chunk in enumerate(lex[ename].substitute(c) for c in raw[si]):
            if feel and ename == "chatterbox":
                voice = {**voice, **feel[si][ci]}
            key = hashlib.sha1(json.dumps([chunk, voice, ename]).encode()).hexdigest()[:16]
            files.append(f"{key}.wav")
            total += 1
            if not (clips / f"{key}.wav").exists():
                todo.append((seg["speaker"], chunk, voice, ename, clips / f"{key}.wav"))
        plan.append((seg, files))

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
    if todo:
        with ThreadPoolExecutor(max(q.qsize() for q in pools.values())) as ex:
            for fut in as_completed([ex.submit(make, t) for t in todo]):
                speaker, chunk, dur, wall = fut.result()
                done += 1
                t_audio += dur
                yield done, total, f"{speaker}: {chunk[:60]}"
    t_wall = time.time() - start
    (work / "clips.json").write_text(json.dumps({seg["id"]: files for seg, files in plan}, indent=1))
    if t_audio:
        yield total, total, f"synth done: {t_audio:.0f}s audio in {t_wall:.0f}s (RTF {t_wall / t_audio:.2f})"


def synthesize(work: Path, cfg: dict, only_chapters: set[int] | None = None) -> None:
    msg = ""
    for _, _, msg in synthesize_iter(work, cfg, only_chapters):
        pass
    print("[synth]", msg)
