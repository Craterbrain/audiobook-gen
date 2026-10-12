"""Recommended settings for what a text turns out to be. The Book tab offers one right after a book is loaded; the user says yes or no.

    recommend(reader) -> {"title", "why", "changes": [plain-language lines], "prep": [step ids]} or None
    apply(work, reader) -> changes config.yaml (voice style); returns the Prepare steps to tick
    answer(work, "applied" | "declined") -> remembered in chapters.json so the same book is not asked again
"""
import json
from pathlib import Path

import yaml

from . import prep

PRESETS = {
    "verse": {"title": "Scripture", "why": "This reads like chapter:verse scripture.",
              "changes": ["One narrator reads everything (no separate voices for speakers)", "Bible-name pronunciations on",
                          "Dialect-word lookup off (scripture has none)", "No AI speaker guessing"],
              "single": True},
    "play": {"title": "Play", "why": "This reads like a play: speaker names, then their lines, in scenes.",
             "changes": ["A separate voice for each character (the stage directions go to the narrator)", "Each scene is a chapter",
                         "Dialect-word lookup off", "No AI speaker guessing (the names are printed in the text)"],
             "single": False},
}


def recommend(reader: str) -> dict | None:
    p = PRESETS.get(reader)
    return None if p is None else {**p, "prep": [s for s in prep.default_selected(reader) if s != "speaker_ai"]}


def _chapters(work: Path) -> dict:
    return json.loads((Path(work) / "chapters.json").read_text())


def pending(work: Path) -> dict | None:
    """The preset to offer for this book, unless it was already answered."""
    try:
        d = _chapters(work)
    except (OSError, ValueError):
        return None
    rep = d.get("parse_report") or {}
    return None if rep.get("preset") else recommend(rep.get("reader", ""))


def answer(work: Path, what: str) -> None:
    p = Path(work) / "chapters.json"
    d = _chapters(work)
    d.setdefault("parse_report", {})["preset"] = what
    p.write_text(json.dumps(d, indent=2, ensure_ascii=False))


def apply(work: Path, reader: str) -> list[str]:
    """Set the voice style for this reading in config.yaml. Returns the Prepare steps to tick."""
    pr = recommend(reader)
    if pr is None:
        return prep.default_selected()
    cfgp = Path(work) / "config.yaml"
    cfg = yaml.safe_load(cfgp.read_text()) or {}
    if pr["single"]:
        voice = (cfg.get("voices") or {}).get("Narrator") or cfg.get("default_voice") or {"engine": "kokoro", "voice": "bm_george"}
        cfg["single_voice"] = {"enabled": True, "voice": voice}
    else:
        cfg["single_voice"] = {**(cfg.get("single_voice") or {}), "enabled": False}
    cfgp.write_text(yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True))
    answer(work, "applied")
    return pr["prep"]
