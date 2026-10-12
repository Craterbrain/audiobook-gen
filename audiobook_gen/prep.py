"""Text preparation: the steps that turn a loaded book into something the voices can read, chosen one by one.

The Book tab lists STEPS, most important first, each with a tick box and a tag saying whether it needs an AI model (and so the GPU).
The Prepare button runs the ticked ones in a fixed order (RUN_ORDER) and records what it did in prep.json. The Generate tab and the
queue refuse a book until it has been prepared, so how much preparation a book gets is a choice made here, not a side effect.

  prep.status(work)      -> {"ready", "at", "steps", "why"}
  prep.run(work, steps)  -> prepares the book (what the Prepare button does)
  prep.mark_done(work)   -> records a preparation made by a script (tools/prepare_books.py)

A book prepared before this existed counts as prepared when its prepared.json is there (the command-line tool wrote it)."""
import hashlib
import json
import time
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
AI = "🧠 needs an AI model (GPU)"
NO_AI = "no AI model"

# id, label, help, needs an AI model, ticked by default. In the order of importance for most texts.
STEPS = [
    {"id": "segments", "label": "Split the text into lines and work out who speaks each one", "ai": False, "default": True, "required": True,
     "help": "Always done: the voices read these lines. Finds the narrator and the characters from quotation marks and speech tags."},
    {"id": "lexicon", "label": "Find names and unusual words (the pronunciation list)", "ai": False, "default": True,
     "help": "Uses a small language-analysis model (spaCy) on the CPU; no generative AI. Names are what the voices get wrong most often."},
    {"id": "ipa", "label": "Look up how those names are pronounced (dictionaries, Bible names)", "ai": False, "default": True,
     "help": "Offline dictionaries and the Bible-name list. Gives the name check after generation something to judge against."},
    {"id": "language", "label": "Question marks and delivery: which sentences are plain or emotional, and which feeling", "ai": True, "default": True,
     "help": "A small language model (Qwen3-4B) and an emotion classifier decide how each sentence is delivered. Uses the GPU; a book being made steps aside."},
    {"id": "dialect", "label": "Dialect and apostrophe words (goin', 'em, jes')", "ai": False, "default": True,
     "help": "Finds elided and dialect words that have a known sound, so the name check can judge them. Only matters for books with dialect."},
    {"id": "speaker_ai", "label": "Settle unclear speakers with a small AI model (books with several voices)", "ai": True, "default": False,
     "help": "A small model (Qwen2.5-1.5B) breaks ties when the speaker of a quote is not clear. Uses the GPU; slower. Only useful for multi-voice books."},
]
RUN_ORDER = ["segments", "lexicon", "ipa", "dialect", "language"]       # speaker_ai is part of "segments"; dialect words go in after the lexicon is built
BY_ID = {s["id"]: s for s in STEPS}


def label(step: dict) -> str:
    return f"{step['label']}  —  {AI if step['ai'] else NO_AI}" + ("  (always done)" if step.get("required") else "")


def choices() -> list[tuple[str, str]]:
    return [(label(s), s["id"]) for s in STEPS]


NO_DIALECT_READERS = ("play", "verse")        # a play in verse or scripture has no dialect spelling to look up


def default_selected(reader: str = "") -> list[str]:
    """The steps ticked by default. `reader` is how the text was read (chapters.json parse_report.reader): what the text is decides what it needs."""
    return [s["id"] for s in STEPS if s["default"] and not (s["id"] == "dialect" and reader in NO_DIALECT_READERS)]


def reader_of(work) -> str:
    try:
        return (json.loads((Path(work) / "chapters.json").read_text()).get("parse_report") or {}).get("reader", "")
    except (OSError, ValueError):
        return ""


def _fingerprint(work: Path) -> str:
    try:
        return hashlib.sha1((work / "chapters.json").read_bytes()).hexdigest()[:16]
    except OSError:
        return ""


def _path(work: Path) -> Path:
    return Path(work) / "prep.json"


def status(work) -> dict:
    """{"ready": bool, "at": when, "steps": [ids], "why": what is missing}. A book is ready after Prepare (or a script's mark_done), as
    long as its chapters have not changed since and its lines exist."""
    work = Path(work)
    if not (work / "segments.json").exists():
        return {"ready": False, "at": "", "steps": [], "why": "the book has not been prepared"}
    try:
        info = json.loads(_path(work).read_text())
    except (OSError, ValueError):
        info = None
    if info is None:
        if (work / "prepared.json").exists():                                        # made by the command-line tool before this existed
            return {"ready": True, "at": "", "steps": ["earlier"], "why": "", "legacy": True}
        return {"ready": False, "at": "", "steps": [], "why": "the book has not been prepared"}
    if info.get("chapters") and info["chapters"] != _fingerprint(work):
        return {"ready": False, "at": info.get("at", ""), "steps": info.get("steps", []), "why": "the chapters changed since it was prepared"}
    return {"ready": True, "at": info.get("at", ""), "steps": info.get("steps", []), "why": ""}


def mark_done(work, steps: list[str] | None = None, source: str = "") -> None:
    """Record that the book is prepared (by the Prepare button, or by a script that did the steps itself)."""
    work = Path(work)
    _path(work).write_text(json.dumps({"at": time.strftime("%Y-%m-%d %H:%M"), "steps": steps if steps is not None else default_selected(),
                                       "chapters": _fingerprint(work), **({"source": source} if source else {})}, indent=1))


def restamp(work) -> None:
    """After a step you trust changed only chapters.json (the spoken opening, section titles): keep the book 'prepared'."""
    work = Path(work)
    try:
        info = json.loads(_path(work).read_text())
    except (OSError, ValueError):
        return
    info["chapters"] = _fingerprint(work)
    _path(work).write_text(json.dumps(info, indent=1))


def describe(work) -> str:
    """One line for the Generate tab."""
    st = status(work)
    if not st["ready"]:
        return (f"⛔ **Text not prepared** ({st['why']}). Open the **Book** tab, tick the preparation steps this book needs and press "
                "**Prepare**; generating stays blocked until then.")
    if st.get("legacy"):
        return "✅ **Text prepared** (earlier, with the command-line tool)."
    names = ", ".join(BY_ID[s]["label"].split(" (")[0].split(":")[0].lower() for s in st["steps"] if s in BY_ID)
    return f"✅ **Text prepared** {st['at']} — {names}."


def _cfg(work: Path) -> dict:
    p = work / "config.yaml"
    return yaml.safe_load((p if p.exists() else ROOT / "config.yaml").read_text()) or {}


def run(work, selected: list[str] | None = None, progress=lambda f, d: None, lang: str = "") -> dict:
    """Prepare the book with the ticked steps (the required one always). Returns {"ran": [ids], "notes": {id: text}}."""
    from . import jobqueue
    work = Path(work)
    sel = {*(selected if selected is not None else default_selected(reader_of(work))), "segments"}
    steps = [s for s in RUN_ORDER if s in sel]
    cfg = _cfg(work)
    meta = json.loads((work / "chapters.json").read_text())
    notes: dict[str, str] = {}
    ai = any(BY_ID[s]["ai"] for s in sel)
    jobqueue.activity_set("text preparation", work.name, "starting")
    if ai:
        progress(0.0, "Waiting for the GPU (a book being made steps aside)")
        jobqueue.lease_take("the text preparation")
    try:
        for n, sid in enumerate(steps):
            lo, hi = n / len(steps), (n + 1) / len(steps)
            sub = lambda f, d, lo=lo, hi=hi: progress(lo + (hi - lo) * max(0.0, min(1.0, f)), d)
            progress(lo, BY_ID[sid]["label"])
            jobqueue.activity_detail(f"step {n + 1} of {len(steps)}: {BY_ID[sid]['label'].split(' (')[0].split(':')[0].lower()}")
            if sid == "segments":
                from .speakers import parse_book
                judge = None
                if "speaker_ai" in sel:
                    from .tiebreak import Judge
                    judge = Judge(cfg.get("judge_model", "Qwen/Qwen2.5-1.5B-Instruct"), cfg.get("device", "auto"))
                try:
                    segs = parse_book(work, cfg.get("aliases"), None, cfg.get("phrase_roles"), judge, sub, cfg.get("genders"))
                finally:
                    if judge:
                        judge.close()
                notes[sid] = f"{len(segs)} lines"
            elif sid == "lexicon":
                from .lexicon import build_lexicon
                lex = build_lexicon(work, lang or cfg.get("name_lang", ""), None, meta.get("title", ""), meta.get("author") or "", sub)
                notes[sid] = f"{len(lex)} words"
            elif sid == "ipa":
                from .pronounce import fill_lexicon
                path = work / "lexicon.json"
                lex = json.loads(path.read_text())
                r = fill_lexicon(lex, lang or cfg.get("name_lang", ""), [], sub, bible=True, offline=True)
                path.write_text(json.dumps(lex, indent=2, ensure_ascii=False))
                notes[sid] = f"IPA for {r['found']} of {r['asked']}"
            elif sid == "dialect":
                from . import dialect
                r = dialect.apply(work)
                notes[sid] = f"{r['words']} words, {r['added']} with a sound"
            elif sid == "language":
                from . import refine
                rep = refine.run(work, cfg, progress=lambda m: sub(0.5, str(m)))
                notes[sid] = f"{len(rep.get('fixes', []))} question marks added"
    finally:
        if ai:
            jobqueue.lease_give_back()
        jobqueue.activity_clear()
    from . import music, titles
    t = music.load_plan(work).get("titles") or {}
    if t.get("speak"):                                                     # each section's title is read aloud unless the plan says not to
        n = titles.apply(work, True, float(t.get("pause", titles.PAUSE_S)))
        notes["titles"] = f"{n} section titles"
    mark_done(work, [s for s in RUN_ORDER if s in sel] + (["speaker_ai"] if "speaker_ai" in sel else []))
    progress(1.0, "Prepared")
    return {"ran": steps, "notes": notes}
