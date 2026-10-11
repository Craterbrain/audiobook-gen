"""Where a book is in the pipeline, and what the user should do next. Pure: reads files and is handed the job list, no Gradio.

Six steps: Book -> Voices -> Names -> Extras -> Make -> Fix. Each has a state:
  required (must happen before Make) · recommended (worth a look) · optional · done · stale (was done, an input changed) · busy.
A step that cannot start yet carries `blocked_by` (the earlier step to go to).

`workflow.json` in the work folder is the only file this module writes: what the last queued build was made from (so "stale" can say
whether clips need recording again or only the book needs rebuilding) and what the user has looked over (voices, names).
"""
import hashlib
import json
import time
from dataclasses import dataclass
from pathlib import Path

from . import prep

STAGES = ("book", "voices", "names", "extras", "make", "fix")
LABELS = {"book": "① Book", "voices": "② Voices", "names": "③ Names", "extras": "④ Extras", "make": "⑤ Make", "fix": "⑥ Fix"}
EXPLAIN = {"book": "Load it, pick chapters, prepare the text.", "voices": "Check who speaks; pick each voice.",
           "names": "Make names sound right first.", "extras": "Cover, music, spoken titles.",
           "make": "Queue it: record, check, build.", "fix": "Listen, remake clips, rebuild."}
MARK = {"done": "✓", "stale": "!", "required": "●", "recommended": "○", "optional": "·", "busy": "⟳"}
NAME_KINDS = ("name", "place", "demonym", "foreign", "dialect")
ACTIVE = ("queued", "paused", "running", "held")

# Settings that change a clip's file name (so the clip is recorded again) and settings only the final assembly reads.
SPEECH_CFG = ("voices", "single_voice", "default_voice", "genders", "emotion", "emotion_base", "text_lexicon", "max_chunk_chars")
BUILD_CFG = ("pacing_ms", "crossfade_ms", "titles")


@dataclass(frozen=True)
class Stage:
    id: str
    label: str
    state: str                 # required | recommended | optional | done | stale | busy
    reason: str = ""           # a few words for the strip
    detail: str = ""           # one line for the banner
    blocked_by: str = ""       # the earlier stage to go to first, or ""
    cost: str = ""             # for make/fix when stale: "speech" (record again) | "build" (rebuild only) | "unknown"


# ---------- fingerprints ----------
_cache: dict = {}


def _file_sha(p: Path) -> str:
    try:
        st = p.stat()
    except OSError:
        return ""
    k = (str(p), st.st_mtime_ns, st.st_size)
    if k not in _cache:
        _cache[k] = hashlib.sha1(p.read_bytes()).hexdigest()
    return _cache[k]


def _sha(obj) -> str:
    return hashlib.sha1(json.dumps(obj, sort_keys=True, ensure_ascii=False, default=str).encode()).hexdigest()[:16]


def _json(p: Path, default=None):
    try:
        return json.loads(Path(p).read_text())
    except (OSError, ValueError):
        return default


def _cfg(work: Path) -> dict:
    import yaml
    try:
        return yaml.safe_load((Path(work) / "config.yaml").read_text()) or {}
    except Exception:
        return {}


def speech_fp(work: Path, cfg: dict | None = None) -> str:
    """Anything that changes which clips the book needs. If it changed, clips must be recorded again."""
    work = Path(work)
    cfg = _cfg(work) if cfg is None else cfg
    lex = [{k: e.get(k, "") for k in ("term", "ipa", "respell", "lang")} for e in (_json(work / "lexicon.json", []) or [])]
    return _sha([_file_sha(work / "segments.json"), lex, {k: cfg.get(k) for k in SPEECH_CFG},
                 _file_sha(work / "delivery.json"), _file_sha(work / "clip_overrides.json")])


def build_fp(work: Path, cfg: dict | None = None, title: str = "", author: str = "") -> str:
    """Anything only the final assembly reads. If only this changed, the book needs a rebuild (minutes), not new clips."""
    work = Path(work)
    cfg = _cfg(work) if cfg is None else cfg
    cover = work / "cover_custom.jpg"
    c = (cover.stat().st_size, cover.stat().st_mtime_ns) if cover.exists() else None
    return _sha([{k: cfg.get(k) for k in BUILD_CFG}, _file_sha(work / "music.json"), c, title, author])


# ---------- workflow.json ----------
def _wf(work: Path) -> dict:
    return _json(Path(work) / "workflow.json", {}) or {}


def _wf_save(work: Path, data: dict) -> None:
    (Path(work) / "workflow.json").write_text(json.dumps(data, indent=1))


def stamp_queued(work: Path, job_id: str = "", title: str = "", author: str = "") -> None:
    """Record what a build was queued from, so later changes can be called speech-stale or build-stale."""
    work = Path(work)
    cfg = _cfg(work)
    data = _wf(work)
    data["queued"] = {"at": time.strftime("%Y-%m-%d %H:%M"), "speech": speech_fp(work, cfg), "build": build_fp(work, cfg, title, author),
                      "job": job_id}
    _wf_save(work, data)


def _roles(work: Path) -> list[str]:
    return sorted({s.get("speaker", "") for s in (_json(work / "segments.json", []) or [])})


def _names(work: Path) -> list[str]:
    return sorted(e["term"] for e in (_json(work / "lexicon.json", []) or []) if e.get("kind") in NAME_KINDS)


def _cast_fp(work: Path, cfg: dict) -> str:
    return _sha([_roles(work), {k: cfg.get(k) for k in ("voices", "single_voice", "genders")}])


def ack(work: Path, which: str) -> None:
    """The user looked over the voices or the names and is happy: stop recommending it until it changes."""
    work = Path(work)
    data = _wf(work)
    a = data.setdefault("ack", {})
    if which == "voices":
        a["voices"] = {"fp": _cast_fp(work, _cfg(work)), "roles": _roles(work)}
    elif which == "names":
        a["names"] = {"terms": _names(work)}
    else:
        raise ValueError(which)
    _wf_save(work, data)


# ---------- status ----------
def _job_for(work: Path, jobs: list[dict]) -> dict | None:
    w = str(Path(work).resolve())
    mine = [j for j in jobs or [] if str(Path(j.get("work", "")).resolve()) == w and j.get("status") in ACTIVE]
    mine.sort(key=lambda j: j.get("status") != "running")
    return mine[0] if mine else None


def _mt(p: Path) -> float:
    try:
        return p.stat().st_mtime
    except OSError:
        return 0.0


def stage_status(work: Path | None, jobs: list[dict] | None = None, ticks: set[int] | None = None, out: Path | None = None) -> list[Stage]:
    """The six steps for the book in `work` (None = no book open). `ticks` are the chapter numbers ticked in the UI, if known.
    `out` is the finished M4B, if it should be looked for somewhere else than out/<work name>.m4b."""
    jobs = jobs or []
    if work is None or not (Path(work) / "chapters.json").exists():
        return [Stage("book", LABELS["book"], "required", "load a book", "Load a book: upload an epub or text file.")] + [
            Stage(s, LABELS[s], "optional", "", "", blocked_by="book") for s in STAGES[1:]]
    work = Path(work)
    cfg = _cfg(work)
    wf = _wf(work)
    job = _job_for(work, jobs)
    running = bool(job and job.get("status") == "running")
    st = prep.status(work)
    meta = _json(work / "chapters.json", {}) or {}
    chapters = {c.get("index") for c in meta.get("chapters", []) if not c.get("opening")}

    # 1 Book
    if running and job.get("note", "").startswith(("text", "preparing")):
        book = Stage("book", LABELS["book"], "busy", "preparing")
    elif not st["ready"] and "not been prepared" in st["why"]:
        book = Stage("book", LABELS["book"], "required", "prepare text", "Tick what this book needs and press Prepare.")
    elif not st["ready"]:
        book = Stage("book", LABELS["book"], "stale", "chapters changed", f"Press Prepare again: {st['why']}.")
    elif ticks is not None and {t for t in ticks} != chapters:
        book = Stage("book", LABELS["book"], "stale", "ticks changed", "The ticked chapters differ from the prepared ones: press Prepare again.")
    else:
        rep = (meta.get("parse_report") or {})
        warn = f" ⚠️ {rep['notes'][0]}" if rep.get("notes") else ""
        book = Stage("book", LABELS["book"], "done", "prepared" if not warn else "prepared, check parse",
                     f"Prepared {st.get('at', '')}.".replace(" .", ".") + warn)
    blocked = "" if book.state in ("done", "busy") else "book"

    # 2 Voices
    roles = _roles(work)
    a = wf.get("ack", {})
    if blocked:
        voices = Stage("voices", LABELS["voices"], "optional", blocked_by="book")
    elif (cfg.get("single_voice") or {}).get("enabled") and (cfg.get("single_voice") or {}).get("voice"):
        voices = Stage("voices", LABELS["voices"], "done", "one narrator", "One narrator reads everything.")
    elif a.get("voices") and a["voices"].get("fp") == _cast_fp(work, cfg):
        voices = Stage("voices", LABELS["voices"], "done", "checked", f"{len(roles)} characters checked.")
    elif a.get("voices") and set(roles) - set(a["voices"].get("roles", [])):
        new = len(set(roles) - set(a["voices"].get("roles", [])))
        voices = Stage("voices", LABELS["voices"], "stale", f"{new} new", f"{new} new character(s) since you checked the voices.")
    else:
        auto = sum(1 for r in roles if r not in (cfg.get("voices") or {}))
        voices = Stage("voices", LABELS["voices"], "recommended", "check voices",
                       f"{auto} of {len(roles)} characters have automatic voices." if auto else f"{len(roles)} characters: look them over.")

    # 3 Names
    lex = _json(work / "lexicon.json", None)
    if blocked or lex is None:
        names = Stage("names", LABELS["names"], "optional", "", "No names list yet: Prepare builds it." if not blocked else "", blocked_by=blocked)
    else:
        terms = _names(work)
        unresolved = [e for e in lex if e.get("kind") in NAME_KINDS and not e.get("known") and e.get("count", 0) >= 2
                      and not e.get("ipa") and not e.get("respell") and not e.get("name_ok")]
        an = a.get("names")
        if an and set(terms) - set(an.get("terms", [])):
            n = len(set(terms) - set(an["terms"]))
            names = Stage("names", LABELS["names"], "stale", f"{n} new", f"{n} new name(s) since you checked.")
        elif not unresolved or an:
            names = Stage("names", LABELS["names"], "done", "checked" if an else "none open",
                          "Every name has a sound." if not unresolved else "You checked the names.")
        else:
            names = Stage("names", LABELS["names"], "recommended", f"{len(unresolved)} unsounded",
                          f"{len(unresolved)} names have no sound yet.")

    # 4 Extras
    plan = _json(work / "music.json", {}) or {}
    intro = plan.get("intro") or {}
    speak = bool((plan.get("titles") or {}).get("speak"))
    segs = _json(work / "segments.json", []) or []
    lost = (plan.get("enabled") and intro.get("track") and intro.get("title") and not any(c.get("opening") for c in meta.get("chapters", []))) \
        or (speak and not any(str(s.get("id", "")).endswith("-title") for s in segs))
    if blocked:
        extras = Stage("extras", LABELS["extras"], "optional", blocked_by="book")
    elif lost:
        extras = Stage("extras", LABELS["extras"], "stale", "titles lost", "The spoken title or section titles were lost: save the Music page again.")
    elif plan.get("enabled") or (work / "cover_custom.jpg").exists():
        extras = Stage("extras", LABELS["extras"], "done", "set", "Cover or music chosen.")
    else:
        extras = Stage("extras", LABELS["extras"], "optional", "cover, music", "Cover, music and spoken titles are optional.")

    # 5 Make
    outm4b = Path(out) if out else None
    if outm4b is None:
        from .jobqueue import ROOT
        outm4b = ROOT / "out" / f"{work.name}.m4b"
    built = outm4b.exists()
    q = wf.get("queued")
    if blocked:
        make = Stage("make", LABELS["make"], "required", "prepare first", blocked_by="book")
    elif job:
        pct = f" {job.get('progress', '')}" if job.get("progress") else ""
        make = Stage("make", LABELS["make"], "busy", "making" if running else "queued", f"{job.get('status', '')}{pct}: {job.get('note', '')}".strip(": "))
    elif not built:
        make = Stage("make", LABELS["make"], "required", "not made yet", "Queue this book.")
    elif q:
        if q.get("speech") != speech_fp(work, cfg):
            make = Stage("make", LABELS["make"], "stale", "re-record", "Voices, names, text or emotion changed since it was made: clips will be recorded again.", cost="speech")
        elif q.get("build") != build_fp(work, cfg, meta.get("title", ""), meta.get("author", "")):
            make = Stage("make", LABELS["make"], "stale", "rebuild", "Only pauses, music or cover changed: a rebuild takes minutes.", cost="build")
        else:
            make = Stage("make", LABELS["make"], "done", "made", "Made, nothing changed since.")
    else:
        newest = max(_mt(work / n) for n in ("segments.json", "lexicon.json", "config.yaml", "music.json", "cover_custom.jpg", "clip_overrides.json"))
        if newest > _mt(outm4b):
            make = Stage("make", LABELS["make"], "stale", "changed", "Changed since this was built; how much is unknown.", cost="unknown")
        else:
            make = Stage("make", LABELS["make"], "done", "made", "Made, nothing changed since.")

    # 6 Fix
    if not (work / "clips_meta.json").exists():
        fix = Stage("fix", LABELS["fix"], "optional", blocked_by="make", detail="Available once clips exist.")
    else:
        qc = _json(work / "qc_report.json", {}) or {}
        nr = _json(work / "name_report.json", {}) or {}
        left, weak = len(qc.get("problems_left") or []), int(nr.get("clips_below_good") or 0)
        if built and _mt(work / "clip_overrides.json") > _mt(outm4b):
            fix = Stage("fix", LABELS["fix"], "stale", "rebuild", "Your fixes are not in the audiobook yet: rebuild.", cost="build")
        elif left or weak:
            fix = Stage("fix", LABELS["fix"], "recommended", "review", f"{left} clips flagged · {weak} weak-name clips.".replace("0 clips flagged · ", ""))
        else:
            fix = Stage("fix", LABELS["fix"], "optional", "listen", "Nothing flagged. Listen and fix anything you hear.")
    return [book, voices, names, extras, make, fix]


def next_step(stages: list[Stage], here: str = "") -> Stage | None:
    """The first step that still needs the user, at or before the step they are looking at (so a banner can point backwards)."""
    order = [s.id for s in stages]
    upto = order.index(here) + 1 if here in order else len(order)
    for s in stages[:upto]:
        if s.id != here and s.state in ("required", "stale"):
            return s
    for s in stages[:upto]:
        if s.id != here and s.state == "recommended":
            return s
    return None


def strip_markdown(stages: list[Stage], title: str = "", author: str = "") -> str:
    """The header strip: one cell per step, a mark for its state, and a lock where an earlier step must come first."""
    cells = []
    for s in stages:
        mark = "🔒" if s.blocked_by else MARK.get(s.state, "")
        cells.append(f"{s.label} {mark}" + (f" _{s.reason}_" if s.reason else ""))
    head = f"**{title}**" + (f" · {author}" if author else "") + "\n\n" if title else ""
    return head + "  →  ".join(cells)


def banner_markdown(s: Stage, nxt: Stage | None = None) -> str:
    """The page banner: the step's one-line purpose, where it stands, and what to do first if something earlier is open."""
    line = f"**{s.label}** — {EXPLAIN[s.id]}"
    status = f"{MARK.get(s.state, '')} {s.detail}" if s.detail else ""
    if s.blocked_by:
        status = f"🔒 Needs {LABELS[s.blocked_by]} first."
    elif nxt and s.id in ("make", "fix") and nxt.state in ("required", "stale"):
        status += f"  \n⚠️ {nxt.label} {nxt.reason or nxt.state}: {nxt.detail}"
    elif nxt and s.id == "make" and nxt.state == "recommended":
        status += f"  \nWorth a look first: {nxt.label} ({nxt.detail})"
    return line + ("  \n" + status if status else "")
