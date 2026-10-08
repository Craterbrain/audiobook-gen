"""Build data/bible_ipa.json: every proper name and place in a verse-per-line Bible text, with IPA.

Names are found without a name list: a word counts when it is capitalised in the middle of a sentence and its
lowercase form is (almost) never seen, which drops He/Him/Lord/Father-style words. Each name then goes through
pronounce.lookup(bible=True): Wiktionary, ipa-dict, WikiPron (IPA as the source wrote it; nothing is derived from spelling). Names none of them knew
get espeak's guess, flagged "guess" so they are easy to review. Only the names and their IPA are stored, not
the Bible text itself."""
import json
import re
import time
from collections import Counter
from pathlib import Path

from .pronounce import ROOT, lookup

OUT = ROOT / "data" / "bible_ipa.json"
WORD = re.compile(r"[A-Za-z][A-Za-z'’\-]*[A-Za-z]|[A-Za-z]")
LINE = re.compile(r"^((?:[1-3]\s)?[A-Za-z][A-Za-z ]+?)\s+(\d+):(\d+)\s+(.*)$")
STOP = {"I", "A", "O", "Admin", "Alas", "Almighty", "Angle", "Assassins", "Bandits", "Changing", "Climbing", "Creator",
        "Deity", "Destiny", "Emperor", "Ether", "Sovereign", "Master", "Teacher", "Savior", "Redeemer", "Counselor", "Lamb",
        "Holy", "Spirit", "Father", "Mighty", "Eternal", "Judge", "King", "Prince", "Shepherd", "Rock", "Hallelujah"}   # titles and common words
SENTENCE_END = re.compile(r"[.!?:;]\s*[\"“”'‘’)]*\s*$")


def find_names(path: str, min_count: int = 1, max_lower_share: float = 0.02) -> dict[str, dict]:
    upper, lower, first = Counter(), Counter(), {}
    for raw in Path(path).read_text(encoding="utf-8").splitlines():
        m = LINE.match(raw.strip())
        if not m:
            continue
        book, ch, vs, text = m.group(1), m.group(2), m.group(3), m.group(4).replace("*", "")
        for w in WORD.finditer(text):
            tok = w.group().strip("-'’")
            if not tok:
                continue
            if tok[0].islower():
                lower[tok] += 1
                continue
            before = text[:w.start()]
            at_start = not before.strip(" \"“‘'(") or bool(SENTENCE_END.search(before))
            if at_start:        # sentence-initial capitals tell us nothing: count the lowercase form instead
                lower[tok.lower()] += 0
                continue
            base = re.sub(r"['’]s$", "", tok)
            upper[base] += 1
            first.setdefault(base, f"{book} {ch}:{vs}")
    for w in [w for w in upper if w.isupper() and len(w) > 2]:      # JESUS, JEWS (small caps): count under Jesus, Jews
        cap = w.capitalize()
        if cap in upper:
            upper[cap] += upper.pop(w)
        else:
            upper[cap] = upper.pop(w)
            first[cap] = first.get(w, "")
    names = {}
    for w, n in upper.items():
        lc = lower[w.lower()] + lower[w.lower() + "s"]
        if n >= min_count and w not in STOP and len(w) > 2 and lc <= max_lower_share * (n + lc):
            names[w] = {"count": n, "first": first[w]}
    return names


DROP = {"Desolating", "Fenced", "Furnaces", "Havens", "Ingathering", "Marshalled", "Benefactors", "Besiegers", "Christs",
        "Perfumed", "Selects", "Sirs", "Tracked", "God-fearing", "Scriptures", "Freedmen", "Conceiving", "Conjure",
        "Dispense", "Settling", "Unbind", "Tek"}      # capitalised ordinary words that slipped through the name filter


def _read_pairs(path: Path) -> dict[str, str]:
    out = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if "|" in line:
            k, v = line.split("|", 1)
            out[k.strip()] = v.strip()
    return out


def manual_entries() -> dict[str, tuple[str, str]]:
    """{name: (ipa, source)} from hand-written files: data/bible_ipa_claude_*.txt (written by Claude) and
    data/bible_ipa_user.txt (yours; wins over everything). Lines are `Name|IPA`."""
    out = {}
    for f in sorted((ROOT / "data").glob("bible_ipa_claude_*.txt")):
        out.update({k: (v, "claude") for k, v in _read_pairs(f).items()})
    user = ROOT / "data" / "bible_ipa_user.txt"
    if user.exists():
        out.update({k: (v, "user") for k, v in _read_pairs(user).items()})
    return out


def apply_overrides(path: Path = OUT) -> int:
    """Re-apply the hand-written files to an existing data/bible_ipa.json (no lookups). Returns entries changed."""
    data = json.loads(path.read_text(encoding="utf-8"))
    n = 0
    for w, (ipa, source) in manual_entries().items():
        e = data["entries"].get(w)
        if e and (e["ipa"], e["source"]) != (ipa, source):
            e["ipa"], e["source"] = ipa, source
            n += 1
    c = {}
    for e in data["entries"].values():
        c[e["source"].split(":")[0]] = c.get(e["source"].split(":")[0], 0) + 1
    data["_meta"]["counts"] = c
    data["_meta"]["sources"]["user"] = "typed by you; overrides everything"
    path.write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")
    return n


def _norm(w: str) -> str:
    return re.sub(r"[^a-z]", "", w.lower())


def build(nasb_path: str, out: Path = OUT, progress=print, offline: bool = False) -> dict:
    """nasb_path: one verse-per-line Bible text, or several joined with commas (e.g. NASB and KJV). Names are the union;
    each entry keeps its count in every text."""
    t0 = time.time()
    paths = [x for x in str(nasb_path).split(",") if x]
    names: dict[str, dict] = {}
    for path in paths:
        label = Path(path).stem
        for w, v in find_names(path).items():
            if w in DROP:
                continue
            e = names.setdefault(w, {"count": 0, "first": v["first"], "texts": {}})
            e["count"] += v["count"]
            e["texts"][label] = v["count"]
    manual = manual_entries()
    progress(f"{len(names)} candidate names")
    hits = lookup(sorted(names), "", None, lambda f, d: progress(f"  {d}") if int(f * 100) % 10 == 0 else None, bible=True, offline=offline, dictionary=False)
    from .lexicon import guess_ipa
    entries, src = {}, Counter()
    for w in sorted(names):
        h = hits.get(w)
        if w in manual:
            ipa, source = manual[w]
        elif h:
            ipa, source = h["ipa"], h["source"].split(" ")[0]
        else:
            ipa, source = "", "none"
        entries[w] = {"ipa": ipa, "source": source, **names[w]}
    # the same name spelled solid or hyphenated (Bethshemesh / Beth-shemesh) shares one pronunciation
    by_norm = {}
    for w, e in entries.items():
        if e["ipa"] and e["source"] not in ("guess", "none"):
            by_norm.setdefault(_norm(w), (w, e["ipa"]))
    for w, e in entries.items():
        if e["source"] in ("guess", "none"):
            twin = by_norm.get(_norm(w))
            if twin and twin[0] != w:
                e["ipa"], e["source"] = twin[1], f"alias:{twin[0]}"
            else:
                try:
                    e["ipa"], e["source"] = guess_ipa(w), "guess"
                except Exception:
                    e["ipa"], e["source"] = "", "none"
    for e in entries.values():
        src[e["source"].split(":")[0]] += 1
    data = {"_meta": {
        "what": "IPA for proper names and places in the Bible (single words), General American English where known",
        "sources": {"wiktionary": "en.wiktionary.org, CC BY-SA 4.0", "wikipron": "CUNY-CL/WikiPron, from Wiktionary, CC BY-SA",
                    "ford1900": "S. V. R. Ford, Pronouncing Vocabulary of Scripture Proper Names (1900), public domain; "
                                "stress and syllables from the book, vowels from rules (approximate)",
                    "claude": "written by Claude from conventional American English readings; unreviewed, check by ear",
                    "alias": "same name spelled solid or hyphenated in another text; shares its pronunciation",
                    "guess": "espeak via misaki; unreviewed"},
        "license": "CC BY-SA 4.0 (contains Wiktionary/WikiPron data); see NOTICE.md", "counts": dict(src), "names": len(entries), "texts": [Path(x).stem for x in paths], "dropped_not_names": sorted(DROP), "built": time.strftime("%Y-%m-%d")},
        "entries": entries}
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(data, ensure_ascii=False, indent=1))
    progress(f"{len(entries)} names -> {out}  {dict(src)}  ({time.time() - t0:.0f}s)")
    return data
