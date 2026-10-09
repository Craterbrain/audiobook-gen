"""The 'Ask Claude' helper: runs Claude Code headless (`claude -p`) in a scratch copy of a project's settings.
It may edit only config.yaml and lexicon.json there, and make a cover with the one command it is given (./cover: search free
pictures on Wikimedia Commons, download one, lay out the cover). The app shows the changes and copies them back only when the
person accepts, after validating them. It cannot run other commands, browse the web, or start a generation."""
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

EDITABLE = ("config.yaml", "lexicon.json")
GUIDE = Path(__file__).with_name("assistant_guide.md")
ENGINES = {"kokoro", "f5", "qwen3", "chatterbox"}
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
                                   f"# Saved clone voices (library: <name>, with engine chatterbox, f5 or qwen3)\n{clones}\n")


COVER_FILES = ("cover.jpg", "cover.json")


def _write_cover_tool(scratch: Path) -> None:
    """The one command the assistant may run: ./cover search | fetch | make (see covers.py). It works inside the scratch folder."""
    root = Path(__file__).resolve().parent.parent
    tool = scratch / "cover"
    tool.write_text(f'#!/bin/bash\ncd "$(dirname "$0")"\nPYTHONPATH={root} exec {sys.executable} -m audiobook_gen.covers "$@"\n')
    tool.chmod(tool.stat().st_mode | stat.S_IEXEC)


def prepare(project: Path, fresh: bool = False) -> Path:
    """Scratch folder with the editable files, reference notes and the guide. Pending edits are kept unless fresh."""
    project = Path(project)
    scratch = scratch_dir(project)
    scratch.mkdir(parents=True, exist_ok=True)
    if fresh:
        for name in COVER_FILES:
            (scratch / name).unlink(missing_ok=True)
        shutil.rmtree(scratch / "pictures", ignore_errors=True)
    _write_cover_tool(scratch)
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
    changes, cover = diff(project), cover_proposal(project)
    if not changes and not cover:
        return "Nothing to apply."
    problems = validate(project)
    if problems:
        return "Not applied: " + "; ".join(problems)
    backup = scratch_dir(project) / f"backup-{time.strftime('%Y%m%d-%H%M%S')}"
    backup.mkdir(parents=True)
    for name in changes:
        old = project / name
        if old.exists():
            shutil.copy(old, backup / name)
        shutil.copy(scratch_dir(project) / name, old)
    done = list(changes)
    if cover:
        if (project / "cover_custom.jpg").exists():
            shutil.copy(project / "cover_custom.jpg", backup / "cover_custom.jpg")
        shutil.copy(cover[0], project / "cover_custom.jpg")
        (project / "cover_credit.txt").write_text(cover[1] + "\n")
        done.append("the cover")
    return f"Applied changes to {', '.join(done)}. The previous version is kept in assistant/{backup.name}/."


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
           "--allowedTools", "Read", "Edit", "Write", "Glob", "Grep", "Bash(./cover:*)",      # the only command: the cover tool
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
