"""python -m audiobook_gen {extract,parse,lexicon,synth,assemble,run} ..."""
import argparse
import json
import re
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent


def _cfg(path):
    return yaml.safe_load(Path(path).read_text())


def _work(args):
    return Path(args.work or ROOT / "work" / re.sub(r"\W+", "_", Path(args.input).stem))


def _chapters(s):
    return {int(x) for x in s.split(",")} if s else None


def _printer():
    last = [None, -1.0]

    def show(frac, desc):
        if desc != last[0] or frac - last[1] >= 0.1 or frac >= 1.0:  # new stage, or 10% further along
            print(f"[{frac:4.0%}] {desc}", flush=True)
            last[0], last[1] = desc, frac
    return show


def main(argv=None):
    p = argparse.ArgumentParser(prog="audiobook_gen")
    p.add_argument("cmd", choices=["extract", "parse", "lexicon", "lookup", "bible-dict", "rank", "synth", "qc", "refine", "assemble", "run"])
    p.add_argument("input", help=".epub or .txt")
    p.add_argument("--work", help="working dir (default work/<name>)")
    p.add_argument("--config", default=ROOT / "config.yaml")
    p.add_argument("--max-chapters", type=int, help="only first N chapters")
    p.add_argument("--chapters", help="comma list of chapter numbers to synth/assemble")
    p.add_argument("--lang", help="language of names for auto IPA (fr, it, de, es, la, he...)")
    p.add_argument("--wikis", help="Fandom wikis to search for names (comma list, e.g. lotr,harrypotter)")
    p.add_argument("--offline", action="store_true", help="lookup / bible-dict: local sources only, no network")
    p.add_argument("--bible", action="store_true", help="lookup: use the 1900 scripture-names pronouncing vocabulary too")
    p.add_argument("--tries", type=int, default=20, help="spellings tried per word in the ranking pass")
    p.add_argument("--single-voice", action="store_true", help="read everything in the Narrator voice")
    p.add_argument("--seed-macbeth", action="store_true")
    p.add_argument("--seed-monte-cristo", action="store_true")
    p.add_argument("--seed-john", action="store_true", help="seed sample pronunciations")
    p.add_argument("--judge", nargs="?", const="Qwen/Qwen2.5-1.5B-Instruct",
                   help="small local LLM that breaks ties on ambiguous quotes (HF model id)")
    p.add_argument("--llm-endpoint", help="OpenAI-compatible chat URL for speaker refinement")
    p.add_argument("--llm-model", default="local")
    p.add_argument("--cover"); p.add_argument("--title"); p.add_argument("--author")
    p.add_argument("--out", help="output .m4b path")
    a = p.parse_args(argv)

    cfg, work = _cfg(a.config), _work(a)
    steps = ["extract", "parse", "lexicon", "synth", "assemble"] if a.cmd == "run" else [a.cmd]
    only = _chapters(a.chapters)
    for step in steps:
        print(f"== {step}")
        if step == "extract":
            from .extract import extract
            d = extract(a.input, work, a.max_chapters)
            print(f"{len(d['chapters'])} chapters")
        elif step == "parse":
            from .speakers import parse_book
            llm = (a.llm_endpoint, a.llm_model) if a.llm_endpoint else None
            judge = None
            if a.judge:
                from .tiebreak import Judge
                judge = Judge(a.judge, cfg.get("device", "auto"))
            segs = parse_book(work, cfg.get("aliases"), llm, cfg.get("phrase_roles"), judge, _printer())
            if judge:
                judge.close()
            if a.single_voice:
                from .speakers import collapse_to_narrator
                segs = collapse_to_narrator(segs)
                (work / "segments.json").write_text(json.dumps(segs, indent=2, ensure_ascii=False))
            print(f"{len(segs)} segments, roles: {sorted({s['speaker'] for s in segs})}")
        elif step == "lexicon":
            from .lexicon import build_lexicon, seed
            judge = None
            if a.judge:
                from .tiebreak import Judge
                judge = Judge(a.judge, cfg.get("device", "auto"))
            lex = build_lexicon(work, a.lang or cfg.get("name_lang", ""), judge, a.title or "", a.author or "", _printer())
            if judge:
                judge.close()
            if a.seed_john:
                seed(work)
            if a.seed_monte_cristo:
                from .lexicon import MONTE_CRISTO_SEED
                seed(work, MONTE_CRISTO_SEED)
            if a.seed_macbeth:
                from .lexicon import MACBETH_SEED
                seed(work, MACBETH_SEED)
            print(f"{len(lex)} entries -> {work / 'lexicon.json'}")
        elif step == "bible-dict":
            from .bible_dict import build
            if a.input == "apply":      # re-apply data/bible_ipa_user.txt / bible_ipa_claude_*.txt without rebuilding
                from .bible_dict import apply_overrides
                print(f"{apply_overrides()} entries changed")
            else:
                build(a.input, offline=a.offline)          # input = the verse-per-line Bible text; writes data/bible_ipa.json
        elif step == "lookup":
            from .pronounce import fill_lexicon
            path = work / "lexicon.json"
            lex = json.loads(path.read_text())
            r = fill_lexicon(lex, a.lang or cfg.get("name_lang", ""), [w for w in (a.wikis or "").split(",") if w],
                             lambda f, d: print(f"  {d}", flush=True), bible=a.bible, offline=a.offline)
            path.write_text(json.dumps(lex, indent=2, ensure_ascii=False))
            print(f"found IPA for {r['found']} of {r['asked']} words; still without: {', '.join(r['missing'][:25])}")
        elif step == "rank":
            from . import voices as vlib
            from .respell_search import rank_lexicon
            from .synth import get_engine, resolve_voice
            voice = resolve_voice("Narrator", cfg)
            if voice["engine"] != "f5":
                raise SystemExit("rank needs an F5 voice for the Narrator (spellings only matter for F5)")
            report = rank_lexicon(work, voice, get_engine("f5", cfg.get("device", "auto")), a.tries, progress=_printer())
            (work / "pronunciation_report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False))
            print(f"{len(report)} words ranked; respelled {sum(not r['kept_plain'] for r in report)}")
        elif step == "synth":
            from .synth import synthesize
            synthesize(work, cfg, only)
        elif step == "refine":
            from . import jobqueue, refine
            jobqueue.lease_take("the language pass")             # a book being made steps aside while the model is on the GPU
            try:
                refine.run(work, cfg)
            finally:
                jobqueue.lease_give_back()
        elif step == "qc":
            from . import qc
            qc.run(work, cfg)
        elif step == "assemble":
            from .assemble import assemble
            out = Path(a.out) if a.out else ROOT / "out" / f"{work.name}.m4b"
            print("->", assemble(work, cfg, out, a.cover, a.title, a.author, only))


if __name__ == "__main__":
    main()
