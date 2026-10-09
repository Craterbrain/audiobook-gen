"""Prepare Gutenberg epubs for an overnight run: extract, cast (or single narrator), lexicon + IPA lookups, config.
Usage: python tools/prepare_books.py [--single] [--walter] books/pg35.epub[:lang] ...
  --single  read the whole book in one narrator voice          --walter  narrate with your Walter clone on Chatterbox, emotion on
Writes work/<slug>/ and appends the slug to books/queue.txt."""
import argparse
import json
import re
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from audiobook_gen import casting                          # noqa: E402
from audiobook_gen.extract import extract                  # noqa: E402
from audiobook_gen.lexicon import build_lexicon            # noqa: E402
from audiobook_gen.pronounce import fill_lexicon           # noqa: E402
from audiobook_gen.speakers import parse_book              # noqa: E402

MIN_DIALOGUE = 0.07               # share of the text that is dialogue before a book gets a multi-voice cast
MIN_LINES, MIN_CAST = 8, 2        # a character needs this many lines; this many characters make it a dynamic (multi-voice) book


def prepare(path: str, lang: str = "", single: bool = False, walter: bool = False, refine_text: bool = True) -> dict:
    src = Path(path)
    book = extract(str(src), str(ROOT / "work" / "_tmp"), None)
    slug = re.sub(r"\W+", "_", book["title"].lower()).strip("_")[:40]
    work = ROOT / "work" / slug
    book = extract(str(src), str(work), None)
    cfg = yaml.safe_load((ROOT / "config.yaml").read_text())
    WALTER = {"engine": "chatterbox", "library": "Walter"}
    narrator = WALTER if walter else {"engine": "kokoro", "voice": "bm_george"}
    cfg.update(voices={"Narrator": dict(narrator)}, name_lang=lang)
    if walter:
        cfg.update(emotion=True)
    print(f"[{slug}] {len(book['chapters'])} chapters, {sum(len(c['text']) for c in book['chapters']):,} characters", flush=True)
    segs = parse_book(work, None, None, None, None, lambda f, d: None, None)
    counts: dict = {}
    for s in segs:
        counts[s["speaker"]] = counts.get(s["speaker"], 0) + 1
    cast = [r for r, n in sorted(counts.items(), key=lambda x: -x[1]) if r != "Narrator" and n >= MIN_LINES]
    genders = {}
    total = sum(len(s["text"]) for s in segs) or 1
    share = sum(len(s["text"]) for s in segs if s["speaker"] != "Narrator") / total
    dynamic = len(cast) >= MIN_CAST and share >= MIN_DIALOGUE and not single
    if dynamic:
        from audiobook_gen.tiebreak import Judge
        j = Judge(cfg.get("judge_model", "Qwen/Qwen2.5-1.5B-Instruct"), "cpu")      # CPU: leaves the GPU to the running job
        try:
            found = j.genders(book["title"], cast)
        finally:
            j.close()
        genders = {n: g for n, (g, _) in found.items()}
        cfg["genders"] = genders
        cfg["voices"].update(casting.assign_by_gender(cast, genders))
        cfg["default_voice"] = dict(narrator)                                 # minor characters share the narrator's voice
    else:
        cfg["single_voice"] = {"enabled": True, "voice": dict(narrator)}
    lex = build_lexicon(work, lang, None, book["title"], book["author"] or "", lambda f, d: None)
    r = {"found": 0, "asked": 0} if walter else fill_lexicon(lex, lang, [], lambda f, d: None, bible=False, offline=False)   # Chatterbox ignores IPA
    for e in lex:                       # Kokoro already reads ordinary words well; a wrong lookup does more harm than good
        if e.get("kind") != "name" and e.get("source") != "user":
            e["ipa"] = e["respell"] = ""
    (work / "lexicon.json").write_text(json.dumps(lex, indent=2, ensure_ascii=False))
    (work / "config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True))
    info = {"slug": slug, "title": book["title"], "author": book["author"] or "", "cover": book["cover"] or "",
            "chapters": len(book["chapters"]), "dynamic": dynamic, "cast": cast[:12], "lexicon": len(lex), "ipa_found": r["found"], "ipa_asked": r["asked"]}
    (work / "prepared.json").write_text(json.dumps(info, indent=1, ensure_ascii=False))
    if refine_text:             # language pass: question marks and delivery, decided by a small model (uses the GPU; a running book steps aside)
        from audiobook_gen import jobqueue, refine
        jobqueue.lease_take("the language pass")
        try:
            rep = refine.run(work, cfg, progress=lambda m: print(m, flush=True))
        finally:
            jobqueue.lease_give_back()
        info["question_marks_added"] = len(rep["fixes"])
        (work / "prepared.json").write_text(json.dumps(info, indent=1, ensure_ascii=False))
    print(f"[{slug}] {'multi-voice, ' + str(len(cast)) + ' characters' if dynamic else 'single narrator'}; lexicon {len(lex)} entries, IPA for {r['found']}/{r['asked']}", flush=True)
    return info


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("books", nargs="+", help="epub[:lang]")
    ap.add_argument("--single", action="store_true"); ap.add_argument("--walter", action="store_true")
    ap.add_argument("--no-refine", action="store_true", help="skip the language pass (question marks and delivery)")
    a = ap.parse_args()
    for arg in a.books:
        p, _, lang = arg.partition(":")
        info = prepare(p, lang, a.single, a.walter, not a.no_refine)
        queue_txt = ROOT / "books" / "queue.txt"
        if info["slug"] not in (queue_txt.read_text().split() if queue_txt.exists() else []):
            with open(queue_txt, "a") as q:
                q.write(info["slug"] + "\n")
