"""Corrections: find the clips of a book, hear them, and remake single clips with other words or settings.

A book's clips are listed in work/<book>/clips_meta.json (written by every synth run). A correction is stored in
clip_overrides.json under "segment id:chunk number", which stays the same when the clip's text or settings change, and is
applied by the synth stage the next time it plans the clips. Accepting a new take also puts it in place at once, so the book
can be rebuilt without making anything again."""
import json
import re
import shutil
import time
from pathlib import Path

import soundfile as sf

from .synth import clip_key, load_overrides

TAKE_FIELDS = ("exaggeration", "cfg_weight", "seed")      # settings the Corrections tab can change (engines that lack one ignore it)


def _meta(work: Path) -> dict:
    try:
        return json.loads((Path(work) / "clips_meta.json").read_text())
    except (OSError, ValueError):
        return {}


def clips(work: Path) -> list[dict]:
    """Every clip in book order: {file, key, chapter, speaker, engine, text, orig, voice, flags, corrected}. Empty for a book
    made before the clip list existed (make it once more to get one)."""
    work = Path(work)
    over = load_overrides(work)
    flags = {}
    try:
        rep = json.loads((work / "qc_report.json").read_text())
        for item in rep.get("problems_left", []) + rep.get("problems_fixed", []):
            flags[item["file"]] = ", ".join(item["why"]) + ("" if item in rep.get("problems_left", []) else " (remade)")
    except (OSError, ValueError):
        pass
    out = []
    for f, m in _meta(work).items():
        if "key" not in m:
            continue
        out.append({"file": f, **m, "flags": flags.get(f, ""), "corrected": m["key"] in over,
                    "made": (work / "clips" / f).exists()})
    return out


def search(work: Path, query: str = "", chapter: int | None = None, speaker: str = "", flagged_only: bool = False,
           corrected_only: bool = False, limit: int = 400) -> list[dict]:
    """Clips whose original or spoken text contains the words typed (a name, a phrase), narrowed by chapter, speaker, flags."""
    words = [w for w in re.split(r"\s+", (query or "").strip().lower()) if w]
    found = []
    for c in clips(work):
        hay = (c["orig"] + " " + c["text"]).lower()
        if (words and not all(w in hay for w in words)) or (chapter and c["chapter"] != chapter) or (speaker and speaker.lower() not in c["speaker"].lower()):
            continue
        if (flagged_only and not c["flags"]) or (corrected_only and not c["corrected"]):
            continue
        found.append(c)
        if len(found) >= limit:
            break
    return found


def make_take(work: Path, clip: dict, text: str, changes: dict, engine) -> Path:
    """Speak `text` with this clip's voice plus `changes`, with an engine the caller has loaded. Saved under corrections/."""
    work = Path(work)
    voice = {**clip["voice"], **{k: v for k, v in changes.items() if v is not None}}
    audio = engine.synth(text, voice)
    folder = work / "corrections"
    folder.mkdir(exist_ok=True)
    path = folder / f"{clip['key'].replace(':', '_')}_{int(time.time() * 1000)}.wav"
    sf.write(path, audio, engine.sample_rate)
    return path


def accept(work: Path, clip: dict, text: str, changes: dict, take: Path) -> str:
    """Make a take the clip: remember the correction, put the audio under the name the synth stage will plan for it, and
    point clips.json at it. Returns the new file name."""
    work = Path(work)
    changes = {k: v for k, v in changes.items() if v is not None}
    over = load_overrides(work)
    prev = over.get(clip["key"], {})
    over[clip["key"]] = {"text": text, "voice": {**prev.get("voice", {}), **changes}, "was": prev.get("was") or clip["file"]}
    voice = {**clip["voice"], **changes}
    name = f"{clip_key(text, voice, clip['engine'])}.wav"
    shutil.copyfile(take, work / "clips" / name)
    (work / "clip_overrides.json").write_text(json.dumps(over, indent=1, ensure_ascii=False))
    try:
        index = json.loads((work / "clips.json").read_text())
        files = index[str(clip["seg"])]
        files[clip["ci"]] = name
        (work / "clips.json").write_text(json.dumps(index, indent=1))
    except (OSError, ValueError, KeyError, IndexError):
        pass                                            # the next synth run writes it
    meta = _meta(work)
    meta.pop(clip["file"], None)
    meta[name] = {k: v for k, v in clip.items() if k not in ("file", "flags", "corrected", "made")} | {"text": text, "voice": voice}
    (work / "clips_meta.json").write_text(json.dumps(meta, ensure_ascii=False))
    return name


def revert(work: Path, clip: dict) -> bool:
    """Drop a correction. The synth stage plans the original clip again (and makes it if it was deleted)."""
    work = Path(work)
    over = load_overrides(work)
    gone = over.pop(clip["key"], None)
    if gone is None:
        return False
    (work / "clip_overrides.json").write_text(json.dumps(over, indent=1, ensure_ascii=False))
    try:                                                # point the book back at the original clip if it is still there
        if gone.get("was") and (work / "clips" / gone["was"]).exists():
            index = json.loads((work / "clips.json").read_text())
            index[str(clip["seg"])][clip["ci"]] = gone["was"]
            (work / "clips.json").write_text(json.dumps(index, indent=1))
    except (OSError, ValueError, KeyError, IndexError):
        pass
    return True
