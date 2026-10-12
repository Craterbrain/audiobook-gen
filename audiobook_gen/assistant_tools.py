"""The ./tools command the Ask Claude helper may run (see assistant.py and assistant_guide.md). It works in the helper's scratch folder, for
the book named in $AUDIOBOOK_PROJECT:

  Look (read only)
    status                      where the book stands, its last parse check, job, clips, unused clips, music plan
    chapters                    the chapters, with their length and any music before them
    names [--setting S] [--search T] [--limit N]    lexicon entries and where the model placed each (us, gb, es, fr ...)
    unused-clips                how many clips and how much space the book's old versions take (nothing is deleted)
  Prepare a change (written to the scratch files; the person reviews the diff and accepts)
    lexicon-fix numerals|phrases|language           clean the lexicon (Roman numerals, sentence-initial words in phrases, a guessed language)
    music plan | list | search "words" [n] | add "File:x.ogg" | add-url URL --title T ...
    music bridge CHAPTER TRACK [--seconds S] [--start S] | unbridge CHAPTER | opening TRACK [--title TEXT] | ending TRACK | enable on|off
    preview opening | bridge CHAPTER | ending         a short mp3 from the clips already made, with the music plan laid in
  Ask for something to be done (nothing happens until the person presses Apply)
    request rebuild | delete-unused-clips | origin-check | hold | resume   [--why TEXT]
"""
import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path

REQUESTABLE = {"rebuild": "Queue the book again (clips that exist are reused, new ones are made, then the book is assembled)",
               "delete-unused-clips": "Delete the clips this book no longer uses (the book's current clips and corrections' originals stay)",
               "origin-check": "Ask the language model where each name is from (US/UK English, Spanish, French ...); a few minutes on the GPU",
               "hold": "Pause this book's queued or running job (clips already made are kept)",
               "resume": "Put this book's paused job back in the queue"}


def project() -> Path:
    p = os.environ.get("AUDIOBOOK_PROJECT")
    if not p or not Path(p).is_dir():
        sys.exit("No book is set for these tools.")
    return Path(p)


def scratch() -> Path:
    return Path.cwd()


def _load(p: Path, default):
    try:
        return json.loads(p.read_text())
    except (OSError, ValueError):
        return default


# ---- looking ----------------------------------------------------------------------------------------------------------

def cmd_status(a):
    from . import jobqueue, library, music, workflow
    w = project()
    meta = _load(w / "chapters.json", {})
    print(f"Book: {meta.get('title', w.name)} by {meta.get('author') or 'unknown'} ({len(meta.get('chapters', []))} chapters)")
    stages = workflow.stage_status(w, jobqueue.load())
    for s in stages:
        print(f"  {s.label}: {s.state}{' (needs ' + workflow.LABELS[s.blocked_by] + ')' if s.blocked_by else ''}{' - ' + s.detail if s.detail else ''}")
    rep = meta.get("parse_report") or {}
    if rep:
        print(f"Text read as: {rep.get('reader')} ({rep.get('level', 'ok')})" + "".join(f"\n  note: {n}" for n in rep.get("notes", [])))
    for j in jobqueue.jobs_for(str(w))[:2]:
        print(f"Job {j['id']}: {j['status']} {j.get('progress', '')} {j.get('note', '')}".rstrip())
    meta_clips = _load(w / "clips_meta.json", {})
    files = len(list((w / "clips").glob("*.wav"))) if (w / "clips").is_dir() else 0
    un = library.unused_clips(w)
    print(f"Clips: {len(meta_clips)} in the plan, {files} files on disk, {len(un)} unused ({library.human(sum(f.stat().st_size for f in un))})")
    lex = _load(w / "lexicon.json", [])
    if lex:
        from collections import Counter
        print("Lexicon:", len(lex), "entries; placed by the model:", dict(Counter(e.get("setting") for e in lex if e.get("setting"))) or "not yet")
    plan = music.load_plan(w)
    print(f"Music: {'ON' if plan['enabled'] else 'off'}; opening {plan['intro'].get('track') or 'none'}; {len(plan.get('bridges') or {})} own bridges; "
          f"shared bridge {plan['bridge'].get('track') or 'none'}; ending {plan['outro'].get('track') or 'none'}; spoken titles {'on' if (plan.get('titles') or {}).get('speak') else 'off'}")


def cmd_chapters(a):
    from . import music
    w = project()
    plan = music.load_plan(scratch() if (scratch() / "music.json").exists() else w)
    own = plan.get("bridges") or {}
    secs = music.chapter_seconds(w)
    for idx, title, s in secs:
        mark = ""
        spec = own.get(str(idx))
        if spec:
            t = music.track(spec.get("track", ""))
            mark = f"   <- bridge: {t['title'] if t else spec.get('track')} ({spec.get('seconds', plan['bridge']['seconds'])} s)"
        print(f"{idx:>3}. {title[:90]}  [{s / 3600:.1f} h]{mark}")


def cmd_names(a):
    lex = _load(scratch() / "lexicon.json", [])
    out = [e for e in lex if (not a.setting or e.get("setting") == a.setting or (a.setting == "us" and not e.get("lang") and e.get("setting")))
           and (not a.search or a.search.lower() in e["term"].lower())]
    out.sort(key=lambda e: -e.get("count", 0))
    for e in out[: a.limit]:
        print(f"{e['term']:<28} x{e.get('count', 0):<4} {e.get('kind', ''):<8} setting={e.get('setting', '-'):<5} lang={e.get('lang') or '-':<4} "
              f"ipa={e.get('ipa') or '-'} respell={e.get('respell') or '-'}")
    print(f"({len(out)} match; showing {min(len(out), a.limit)})")


def cmd_unused(a):
    from . import library
    un = library.unused_clips(project())
    print(f"{len(un)} unused clips, {library.human(sum(f.stat().st_size for f in un))}. The book's current clips and corrections' originals are not counted. "
          "Use `request delete-unused-clips` to ask for them to be deleted.")


# ---- preparing changes --------------------------------------------------------------------------------------------------

def cmd_lexicon_fix(a):
    from . import lexicon
    p = scratch() / "lexicon.json"
    lex = _load(p, None)
    if lex is None:
        sys.exit("There is no lexicon.json to fix.")
    n = {"numerals": lexicon.drop_numerals, "phrases": lexicon.repair_phrases, "language": lexicon.drop_guessed_language}[a.what](lex)
    p.write_text(json.dumps(lex, indent=2, ensure_ascii=False))
    print(f"{a.what}: {n} entries changed. The person will see the diff.")


def _plan():
    from . import music
    p = scratch() / "music.json"
    if p.exists():
        return music.load_plan(scratch())
    return music.load_plan(project())


def _save_plan(plan):
    from . import music
    music.save_plan(scratch(), plan)


def cmd_music(a):
    from . import music
    if a.op == "list":
        for t in music.library():
            print(f"{t['id']} | {t['seconds']} s | {t['licence']} | {t['title']}")
    elif a.op == "plan":
        print(json.dumps(_plan(), indent=1, ensure_ascii=False))
    elif a.op == "search":
        for h in music.search(" ".join(a.args), 8):
            print(f"{h['file']} | {h['seconds']} s | {h['licence']} | {h['date'] or '-'} | {h['artist'][:50]}")
    elif a.op == "add":
        e = music.add(" ".join(a.args))
        print(f"Added to the library as '{e['id']}' ({e['seconds']} s, {e['licence']}).")
    elif a.op == "add-url":
        if not a.url or not a.url.startswith("https://") or not a.title:
            sys.exit("add-url needs an https:// link and --title. Only for a link the person gave you and said is public domain.")
        e = music.add_url(a.url, a.title, a.artist or "", a.year or "", a.page or "", a.licence or "Public domain (the person confirmed)")
        print(f"Added to the library as '{e['id']}' ({e['seconds']} s).")
    elif a.op == "bridge":
        plan = _plan()
        ch, tid = a.args[0], a.args[1]
        if not music.track(tid):
            sys.exit(f"No track '{tid}' in the library (see `music list`).")
        plan.setdefault("bridges", {})[str(int(ch))] = {"track": tid, "seconds": float(a.seconds), "start": float(a.start)}
        _save_plan(plan); print(f"Section {ch} will have {tid} ({a.seconds} s from {a.start} s) before it.")
    elif a.op == "unbridge":
        plan = _plan(); plan.setdefault("bridges", {}).pop(str(int(a.args[0])), None); _save_plan(plan); print("Removed.")
    elif a.op == "opening":
        plan = _plan()
        if not music.track(a.args[0]):
            sys.exit(f"No track '{a.args[0]}' in the library.")
        plan["intro"].update(track=a.args[0], start=float(a.start))
        if a.title is not None:
            plan["intro"]["title"] = a.title.replace("\\n", "\n")
        _save_plan(plan); print("Opening set.")
    elif a.op == "ending":
        plan = _plan()
        if not music.track(a.args[0]):
            sys.exit(f"No track '{a.args[0]}' in the library.")
        plan["outro"].update(track=a.args[0], start=float(a.start)); _save_plan(plan); print("Ending set.")
    elif a.op == "enable":
        plan = _plan(); plan["enabled"] = a.args[0].lower() in ("on", "true", "yes", "1"); _save_plan(plan); print("Music is", "ON" if plan["enabled"] else "off")


def cmd_preview(a):
    """A short mp3 of the opening, a bridge or the ending, laid into clips that are already made (the final clips will differ a little)."""
    import numpy as np
    import soundfile as sf
    from . import music
    w, SR = project(), 24000
    meta = _load(w / "clips_meta.json", {})
    clips = [(f, m) for f, m in meta.items() if (w / "clips" / f).exists()]
    if not clips:
        sys.exit("No clips are made yet, so there is nothing to build a preview from.")
    plan = _plan(); plan["enabled"] = True
    secs = music.chapter_seconds(w); all_ch = [c[0] for c in secs]; names = {c[0]: c[1] for c in secs}

    def load(f):
        x, sr = sf.read(w / "clips" / f, dtype="float32")
        x = x if x.ndim == 1 else x.mean(1)
        if sr != SR:
            from math import gcd
            from scipy.signal import resample_poly
            g = gcd(SR, sr); x = resample_poly(x, SR // g, sr // g).astype(np.float32)
        return x

    def body(ch, seconds, tail=False, skip=()):
        rows = [(f, m) for f, m in clips if m.get("chapter") == ch and m["text"].strip().lower() not in skip]
        rows = rows[::-1] if tail else rows
        out, total = [], 0.0
        for f, m in rows:
            x = load(f); out += [x, np.zeros(int(0.45 * SR), np.float32)]; total += len(x) / SR + 0.45
            if total >= seconds:
                break
        out = out[::-1] if tail else out
        return np.concatenate(out) if out else np.zeros(0, np.float32)

    def clip_for(text):
        t = text.strip().lower()
        return next((load(f) for f, m in clips if m["text"].strip().lower() in (t, t.rstrip(".") + ".")), None)

    note = ""
    if a.what == "opening":
        line = (plan["intro"].get("title") or meta_title(w)).splitlines()[0]
        t = clip_for(line)
        note = "" if t is not None else " (the spoken title is not made yet: silence stands in for it)"
        title = np.concatenate([np.zeros(int(0.3 * SR), np.float32), t if t is not None else np.zeros(int(3 * SR), np.float32)])
        head, tail = music.opening(title, plan, SR)
        first = secs[0][0]
        out = np.concatenate([head, music.apply_tail(body(first, 35, skip=(line.lower(),)), tail)])
    elif a.what == "bridge":
        ch = int(a.chapter)
        t = clip_for(names.get(ch, ""))
        note = "" if t is not None else " (the section title is not made yet: silence stands in for it)"
        chapter = np.concatenate([np.zeros(int(1.0 * SR), np.float32), t if t is not None else np.zeros(int(3 * SR), np.float32),
                                  np.zeros(int(2.8 * SR), np.float32), body(ch, 40, skip=(names.get(ch, "").lower(),))])
        out = music.wrap(chapter, ch, all_ch, plan, secs, SR)
    else:
        out = music.ending(body(all_ch[-1], 30, tail=True), plan, SR)
    d = scratch() / "previews"; d.mkdir(exist_ok=True)
    name = a.what + (f"_{int(a.chapter)}" if a.what == "bridge" else "")
    wav, mp3 = d / f"{name}.wav", d / f"{name}.mp3"
    sf.write(wav, out, SR)
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(wav), "-af", "loudnorm=I=-16:TP=-1.5:LRA=11", "-ar", "24000", "-ac", "1", "-b:a", "96k", str(mp3)], check=True)
    wav.unlink()
    print(f"Preview made: {mp3.name}, {len(out) / SR:.0f} s{note}. The person can listen to it in the app.")


def meta_title(w: Path) -> str:
    return _load(w / "chapters.json", {}).get("title", w.name)


def cmd_request(a):
    if a.what not in REQUESTABLE:
        sys.exit(f"Unknown request. Choose from: {', '.join(REQUESTABLE)}")
    p = scratch() / "actions.json"
    acts = _load(p, [])
    if not any(x["action"] == a.what for x in acts):
        acts.append({"action": a.what, "why": a.why or ""})
    p.write_text(json.dumps(acts, indent=1, ensure_ascii=False))
    print(f"Requested: {REQUESTABLE[a.what]}. It happens only if the person presses Apply.")


def main(argv=None):
    p = argparse.ArgumentParser(prog="tools")
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("status").set_defaults(fn=cmd_status)
    sub.add_parser("chapters").set_defaults(fn=cmd_chapters)
    n = sub.add_parser("names"); n.add_argument("--setting"); n.add_argument("--search"); n.add_argument("--limit", type=int, default=60); n.set_defaults(fn=cmd_names)
    sub.add_parser("unused-clips").set_defaults(fn=cmd_unused)
    lf = sub.add_parser("lexicon-fix"); lf.add_argument("what", choices=["numerals", "phrases", "language"]); lf.set_defaults(fn=cmd_lexicon_fix)
    m = sub.add_parser("music"); m.add_argument("op", choices=["list", "plan", "search", "add", "add-url", "bridge", "unbridge", "opening", "ending", "enable"])
    m.add_argument("args", nargs="*"); m.add_argument("--seconds", type=float, default=45.0); m.add_argument("--start", type=float, default=0.0)
    m.add_argument("--title"); m.add_argument("--artist"); m.add_argument("--year"); m.add_argument("--page"); m.add_argument("--licence"); m.add_argument("--url")
    m.set_defaults(fn=cmd_music)
    pv = sub.add_parser("preview"); pv.add_argument("what", choices=["opening", "bridge", "ending"]); pv.add_argument("chapter", nargs="?"); pv.set_defaults(fn=cmd_preview)
    r = sub.add_parser("request"); r.add_argument("what"); r.add_argument("--why"); r.set_defaults(fn=cmd_request)
    args = p.parse_args(argv)
    if args.cmd == "music" and args.op == "add-url" and not args.url and args.args:
        args.url = args.args[0]
    args.fn(args)


if __name__ == "__main__":
    main()
