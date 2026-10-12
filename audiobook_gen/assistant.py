"""The 'Ask Claude' helper: runs Claude Code headless (`claude -p`) in a scratch copy of a project's settings.
It may edit only config.yaml, lexicon.json and music.json there, and run two commands: ./cover (search free pictures on Wikimedia
Commons, download one, lay out the cover) and ./tools (look at the book, clean the lexicon, plan and preview the music, ask for a rebuild
or a clean-up: see assistant_tools.py). The app shows the changes and the requested actions and applies them only when the person
accepts, after validating them. It cannot run other commands, browse the web, or start a generation by itself."""
import difflib
import json
import os
import re
import shutil
import stat
import subprocess
import sys
import time
from pathlib import Path

import yaml

EDITABLE = ("config.yaml", "lexicon.json", "music.json")
ACTIONS = ("rebuild", "delete-unused-clips", "origin-check", "hold", "resume")
GUIDE = Path(__file__).with_name("assistant_guide.md")
ENGINES = {"kokoro", "f5", "qwen3", "chatterbox", "higgs"}
LEX_MODES = {"verified", "respell", "rawipa", "plain", "ipa"}
KOKORO_ID = re.compile(r"^[abefhijpz][fm]_\w+$|^pack:\w+|^\w+$")


def claude_bin() -> str | None:
    return os.environ.get("AUDIOBOOK_CLAUDE") or shutil.which("claude")


def scratch_dir(project: Path) -> Path:
    return Path(project) / "assistant"


def _reference(project: Path, scratch: Path) -> None:
    from . import casting, voices as vlib
    ref = scratch / "reference"
    ref.mkdir(parents=True, exist_ok=True)
    cfg = yaml.safe_load((project / "config.yaml").read_text()) if (project / "config.yaml").exists() else {}
    chapters = json.loads((project / "chapters.json").read_text()) if (project / "chapters.json").exists() else {}
    chs = chapters.get("chapters", [])
    (ref / "book.md").write_text(f"# {chapters.get('title', project.name)}\nAuthor: {chapters.get('author') or 'unknown'}\n"
                                 f"{len(chs)} chapters, {sum(len(c['text']) for c in chs):,} characters.\n\n"
                                 + "\n".join(f"- {c['index']}. {c['title']}" for c in chs[:80]) + "\n")
    lines = ["# Characters (most lines first)\n"]
    if (project / "segments.json").exists():
        segs = json.loads((project / "segments.json").read_text())
        by: dict[str, list] = {}
        for s in segs:
            by.setdefault(s["speaker"], []).append(s["text"])
        for role in sorted(by, key=lambda r: -len(by[r]))[:40]:
            g = (cfg.get("genders") or {}).get(role, "unknown")
            lines.append(f"## {role} — {len(by[role])} lines, gender {g}")
            lines += [f"- “{t[:140]}”" for t in by[role][:2]]
    (ref / "cast.md").write_text("\n".join(lines) + "\n")
    kokoro = "\n".join(f"- {vid}: {name}, {desc}" for vid, (name, desc) in casting.KOKORO.items())
    clones = "\n".join(f"- {n}" for n in vlib.list_voices()) or "- (none saved)"
    (ref / "voices.md").write_text(f"# Kokoro preset voices (engine: kokoro, voice: <id>)\n{kokoro}\n\n"
                                   f"# Saved clone voices (library: <name>, with engine chatterbox, f5, qwen3 or higgs)\n{clones}\n")


COVER_FILES = ("cover.jpg", "cover.json")


def _write_cover_tool(scratch: Path) -> None:
    """The one command the assistant may run: ./cover search | fetch | make (see covers.py). It works inside the scratch folder."""
    root = Path(__file__).resolve().parent.parent
    tool = scratch / "cover"
    tool.write_text(f'#!/bin/bash\ncd "$(dirname "$0")"\nPYTHONPATH={root} exec {sys.executable} -m audiobook_gen.covers "$@"\n')
    tool.chmod(tool.stat().st_mode | stat.S_IEXEC)


def _write_tools_command(scratch: Path, project: Path) -> None:
    """./tools: the book's helper commands (assistant_tools.py), pointed at this book."""
    root = Path(__file__).resolve().parent.parent
    tool = scratch / "tools"
    tool.write_text(f'#!/bin/bash\ncd "$(dirname "$0")"\nAUDIOBOOK_PROJECT={Path(project).resolve()} PYTHONPATH={root} exec {sys.executable} -m audiobook_gen.assistant_tools "$@"\n')
    tool.chmod(tool.stat().st_mode | stat.S_IEXEC)


def prepare(project: Path, fresh: bool = False) -> Path:
    """Scratch folder with the editable files, reference notes and the guide. Pending edits are kept unless fresh."""
    project = Path(project)
    scratch = scratch_dir(project)
    scratch.mkdir(parents=True, exist_ok=True)
    if fresh:
        for name in COVER_FILES + ("actions.json",):
            (scratch / name).unlink(missing_ok=True)
        shutil.rmtree(scratch / "pictures", ignore_errors=True)
        shutil.rmtree(scratch / "previews", ignore_errors=True)
        (scratch / "music.json").unlink(missing_ok=True)
    _write_cover_tool(scratch)
    _write_tools_command(scratch, project)
    src_cfg = project / "config.yaml"
    for name in EDITABLE:
        src = src_cfg if name == "config.yaml" and src_cfg.exists() else (project / name if name != "config.yaml" else Path(__file__).resolve().parent.parent / "config.yaml")
        if src.exists() and (fresh or not (scratch / name).exists()):
            shutil.copy(src, scratch / name)
    shutil.copy(GUIDE, scratch / "CLAUDE.md")
    _reference(project, scratch)
    return scratch


def _current(project: Path, name: str) -> str:
    p = Path(project) / name
    if not p.exists() and name == "config.yaml":
        p = Path(__file__).resolve().parent.parent / "config.yaml"
    return p.read_text() if p.exists() else ""


def _pretty(name: str, text: str) -> str:
    if name == "lexicon.json" and text.strip():
        try:
            return "\n".join(json.dumps(e, ensure_ascii=False) for e in json.loads(text)) + "\n"
        except Exception:
            pass
    return text


def diff(project: Path) -> dict[str, str]:
    """{file: unified diff} for the files the assistant changed in the scratch copy."""
    scratch, out = scratch_dir(project), {}
    for name in EDITABLE:
        new = (scratch / name).read_text() if (scratch / name).exists() else ""
        old = _current(project, name)
        if new and new != old:
            d = "".join(difflib.unified_diff(_pretty(name, old).splitlines(True), _pretty(name, new).splitlines(True),
                                             f"current {name}", f"proposed {name}", n=1))
            if d:
                out[name] = d
    return out


def actions(project: Path) -> list[dict]:
    """What the helper asked to have done (./tools request ...), kept to the actions the app knows."""
    try:
        acts = json.loads((scratch_dir(project) / "actions.json").read_text())
    except (OSError, ValueError):
        return []
    return [a for a in acts if isinstance(a, dict) and a.get("action") in ACTIONS]


def previews(project: Path) -> list[Path]:
    d = scratch_dir(project) / "previews"
    return sorted(d.glob("*.mp3")) if d.is_dir() else []


def _music_problems(project: Path, scratch: Path) -> list[str]:
    """Reasons the proposed music.json is not usable."""
    from . import music
    try:
        plan = json.loads((scratch / "music.json").read_text())
        assert isinstance(plan, dict)
    except Exception:
        return ["music.json is not valid JSON"]
    out, tracks = [], {t["id"] for t in music.library()}
    chapters = {c["index"] for c in json.loads((Path(project) / "chapters.json").read_text()).get("chapters", []) if not c.get("opening")} \
        if (Path(project) / "chapters.json").exists() else set()

    def spec(label, d, keys=("seconds", "start", "lead", "tail", "fade", "before", "after")):
        if not isinstance(d, dict):
            return
        if d.get("track") and d["track"] not in tracks:
            out.append(f"{label}: there is no track '{d['track']}' in the music library")
        for k in keys:
            if k in d and not (isinstance(d[k], (int, float)) and 0 <= d[k] <= 1200):
                out.append(f"{label}.{k} must be a number from 0 to 1200")
    for role in ("intro", "bridge", "outro"):
        spec(role, plan.get(role))
    for ch, d in (plan.get("bridges") or {}).items():
        if not str(ch).isdigit() or (chapters and int(ch) not in chapters):
            out.append(f"bridges: section {ch} is not one of the book's chapters")
        spec(f"bridge before section {ch}", d)
    if "enabled" in plan and not isinstance(plan["enabled"], bool):
        out.append("music.json: enabled must be true or false")
    if "level_db" in plan and not (isinstance(plan["level_db"], (int, float)) and -30 <= plan["level_db"] <= 0):
        out.append("music.json: level_db must be a number from -30 to 0")
    return out


def cover_proposal(project: Path) -> tuple[Path, str] | None:
    """(cover picture, credit line) if the assistant made a cover that differs from the project's current one."""
    scratch = scratch_dir(project)
    new, cur = scratch / "cover.jpg", Path(project) / "cover_custom.jpg"
    if not new.exists() or (cur.exists() and cur.read_bytes() == new.read_bytes()):
        return None
    credit = (scratch / "pictures" / "credit.txt")
    return new, (credit.read_text().strip() if credit.exists() else "")


def validate(project: Path) -> list[str]:
    """Reasons the proposed files must not be applied (empty = fine)."""
    from . import casting, voices as vlib
    scratch, problems = scratch_dir(project), []
    try:
        cfg = yaml.safe_load((scratch / "config.yaml").read_text())
        assert isinstance(cfg, dict)
    except Exception:
        return ["config.yaml is not valid YAML"]
    clones = set(vlib.list_voices())

    def check_voice(label, v):
        if not isinstance(v, dict) or v.get("engine") not in ENGINES:
            problems.append(f"{label}: needs an engine of {sorted(ENGINES)}")
        elif v.get("library"):
            if v["library"] not in clones:
                problems.append(f"{label}: there is no saved clone voice called “{v['library']}”")
        elif v["engine"] == "kokoro" and not (isinstance(v.get("voice"), str) and KOKORO_ID.match(v["voice"])):
            problems.append(f"{label}: a Kokoro voice needs a voice id")
        elif v["engine"] != "kokoro" and not v.get("ref_audio"):
            problems.append(f"{label}: a clone voice needs a library name")
    for role, v in (cfg.get("voices") or {}).items():
        check_voice(f"voice for {role}", v)
    if cfg.get("default_voice"):
        check_voice("default_voice", cfg["default_voice"])
    sv = cfg.get("single_voice") or {}
    if sv.get("voice"):
        check_voice("single_voice", sv["voice"])
    if cfg.get("text_lexicon", "respell") not in LEX_MODES:
        problems.append(f"text_lexicon must be one of {sorted(LEX_MODES)}")
    for k, v in (cfg.get("pacing_ms") or {}).items():
        if not isinstance(v, (int, float)) or not 0 <= v <= 5000:
            problems.append(f"pacing_ms.{k} must be a number from 0 to 5000")
    for k, v in (cfg.get("workers") or {}).items():
        if not isinstance(v, int) or not 1 <= v <= 6:
            problems.append(f"workers.{k} must be a whole number from 1 to 6")
    for r, g in (cfg.get("genders") or {}).items():
        if g not in ("male", "female", "unknown"):
            problems.append(f"gender of {r} must be male, female or unknown")
    if (scratch / "music.json").exists():
        problems += _music_problems(project, scratch)
    raw = (scratch / "actions.json")
    if raw.exists():
        try:
            bad = [a for a in json.loads(raw.read_text()) if not (isinstance(a, dict) and a.get("action") in ACTIONS)]
            problems += [f"unknown requested action: {a}" for a in bad]
        except Exception:
            problems.append("actions.json is not valid JSON")
    prop = cover_proposal(project)
    if prop:
        try:
            from PIL import Image
            if min(Image.open(prop[0]).size) < 600:
                problems.append("the proposed cover picture is too small")
        except Exception:
            problems.append("the proposed cover is not a readable image")
    if (scratch / "lexicon.json").exists():
        try:
            lex = json.loads((scratch / "lexicon.json").read_text())
            if not (isinstance(lex, list) and all(isinstance(e, dict) and e.get("term") for e in lex)):
                problems.append("lexicon.json must be a list of entries that each have a term")
        except Exception:
            problems.append("lexicon.json is not valid JSON")
    return problems


def apply(project: Path) -> str:
    """Copy the accepted files into the project (originals are kept in assistant/backup-<time>/)."""
    project = Path(project)
    changes, cover, todo = diff(project), cover_proposal(project), actions(project)
    if not changes and not cover and not todo:
        return "Nothing to apply."
    problems = validate(project)
    if problems:
        return "Not applied: " + "; ".join(problems)
    if "music.json" in changes:
        from . import operations
        try:
            operations.refuse_while_making(project, "change its music or spoken titles")
        except ValueError as e:
            return "Not applied: " + str(e)
    backup = scratch_dir(project) / f"backup-{time.strftime('%Y%m%d-%H%M%S')}"
    backup.mkdir(parents=True)
    for name in changes:
        old = project / name
        if old.exists():
            shutil.copy(old, backup / name)
        shutil.copy(scratch_dir(project) / name, old)
    done = list(changes)
    if "music.json" in changes:                                       # the opening and section titles the plan asks for go into the book's lines
        from . import operations
        operations.reapply_extras(project)
    if cover:
        if (project / "cover_custom.jpg").exists():
            shutil.copy(project / "cover_custom.jpg", backup / "cover_custom.jpg")
        shutil.copy(cover[0], project / "cover_custom.jpg")
        (project / "cover_credit.txt").write_text(cover[1] + "\n")
        done.append("the cover")
    msg = f"Applied changes to {', '.join(done)}. The previous version is kept in assistant/{backup.name}/." if done else ""
    if todo:
        msg = (msg + " " if msg else "") + run_actions(project, todo)
        (scratch_dir(project) / "actions.json").unlink(missing_ok=True)
    return msg


def run_actions(project: Path, todo: list[dict]) -> str:
    """Do what the person approved. Each result is one sentence; a refused action says why and the rest still run."""
    from . import jobqueue, library, operations
    project, out = Path(project), []
    for a in todo:
        kind = a["action"]
        try:
            if kind == "rebuild":
                job, updated = operations.rebuild(project)
                out.append(f"{'Updated the waiting job' if updated else 'Queued a rebuild'} for “{job['title']}”.")
            elif kind == "delete-unused-clips":
                operations.refuse_while_making(project, "delete its clips")
                n, freed = library.remove_unused_clips(project)
                out.append(f"Deleted {n} unused clips, freeing {library.human(freed)}.")
            elif kind == "hold":
                ids = [j["id"] for j in jobqueue.jobs_for(str(project)) if j["status"] in ("queued", "paused", "running")]
                out.append(f"Paused {jobqueue.hold(ids)} job(s) (clips already made are kept).")
            elif kind == "resume":
                ids = [j["id"] for j in jobqueue.jobs_for(str(project)) if j["status"] in ("held", "paused")]
                out.append(f"Resumed {jobqueue.resume(ids)} job(s).")
            elif kind == "origin-check":
                from .lexicon import redo_origins
                from .refine import MODEL, Scorer
                cfg = yaml.safe_load((project / "config.yaml").read_text()) if (project / "config.yaml").exists() else {}
                jobqueue.lease_take("the name origin check")
                scorer = Scorer(cfg.get("origin_model", MODEL), cfg.get("device", "auto"))
                try:
                    r = redo_origins(project, scorer)
                finally:
                    scorer.close()
                    jobqueue.lease_give_back()
                out.append(f"Placed {r['asked']} names: " + ", ".join(f"{n} {c}" for c, n in sorted(r["by_setting"].items(), key=lambda kv: -kv[1])) + ".")
        except Exception as e:
            out.append(f"{kind}: not done ({e}).")
    return " ".join(out)


def discard(project: Path) -> str:
    prepare(project, fresh=True)
    return "Proposed changes discarded."


def ask(project: Path, message: str, session: str | None = None, timeout: int = 420) -> dict:
    """Ask Claude. Returns {"reply", "session", "cost", "error"}; the proposed edits are in the scratch folder."""
    exe = claude_bin()
    if not exe:
        return {"reply": "", "session": session, "cost": 0.0,
                "error": "Claude Code is not installed (the `claude` command was not found)."}
    scratch = prepare(project)
    cmd = [exe, "-p", message, "--output-format", "json", "--permission-mode", "acceptEdits",
           "--allowedTools", "Read", "Edit", "Write", "Glob", "Grep", "Bash(./cover:*)", "Bash(./tools:*)",      # the only commands: the cover and book tools
           "--disallowedTools", "WebFetch", "WebSearch", "Task", "--max-turns", "25"]
    if session:
        cmd += ["--resume", session]
    try:
        r = subprocess.run(cmd, cwd=scratch, capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return {"reply": "", "session": session, "cost": 0.0, "error": "Claude took too long (over 7 minutes); try a smaller request."}
    try:
        data = json.loads(r.stdout)
    except Exception:
        return {"reply": "", "session": session, "cost": 0.0, "error": (r.stderr or r.stdout or "no answer").strip()[:400]}
    return {"reply": data.get("result", ""), "session": data.get("session_id", session),
            "cost": float(data.get("total_cost_usd") or 0), "error": data.get("result", "") if data.get("is_error") else ""}
