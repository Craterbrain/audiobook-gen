"""Dialect and elided words (goin', 'em, jes', reg'lar, Chris'mus): find them in a book and give them a target sound, so the name pass
(namecheck.py) can judge how the voice reads them, retry with other seeds, and try a respelling when they keep coming out wrong.

Chatterbox reads plain letters, so a dropped letter written as an apostrophe is a guess for it. This module does not decide how a word is
said; it only finds the candidates and looks up the sound they are meant to have:

  data/dialect_ipa.json        token -> IPA, written by Claude from how the word is conventionally said (unreviewed, check by ear)
  data/dialect_ipa_user.txt    `token|IPA` lines you typed; they always win
  work/<book>/dialect_ipa.txt  the same, for one book only (wins over both)

`scan` lists the apostrophe words of a book that are neither a standard contraction nor a plural possessive (days', Epps'), with their
counts and whether a sound is known. `apply` adds the known ones to lexicon.json as kind "dialect" (with ipa_src saying where the
sound came from); the ones with no sound are listed in dialect_scan.json so one can be written. Possessives and standard contractions
are left alone: the voice reads them fine."""
import json
import re
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "dialect_ipa.json"
USER = ROOT / "data" / "dialect_ipa_user.txt"
LETTER = "A-Za-zÀ-ÖØ-öø-ÿ"
TOKEN = re.compile(rf"(?<![{LETTER}'’])['’]?[{LETTER}]+(?:['’][{LETTER}]+)*['’]?(?![{LETTER}])")
STANDARD = re.compile(r"^(?:[a-z]+n't|[a-z]+'(?:s|d|ll|re|ve|m)|i'm|o'clock|ma'am|ne'er|e'er|o'er|'tis|'twas|'twere|'twill|'twould)$")


def key(token: str) -> str:
    return token.replace("’", "'").lower()


def _pairs(path: Path) -> dict[str, str]:
    out = {}
    try:
        for line in path.read_text(encoding="utf-8").splitlines():
            if "|" in line and not line.lstrip().startswith("#"):
                t, ipa = line.split("|", 1)
                if t.strip() and ipa.strip():
                    out[key(t.strip())] = ipa.strip()
    except OSError:
        pass
    return out


def sounds(work: Path | None = None) -> dict[str, tuple[str, str]]:
    """token -> (IPA, where it came from): the shipped list, then your global file, then this book's file."""
    out = {}
    try:
        for t, v in json.loads(DATA.read_text(encoding="utf-8")).items():
            if not t.startswith("_"):
                out[key(t)] = (v, "dialect-dict:claude")
    except (OSError, ValueError):
        pass
    for t, v in _pairs(USER).items():
        out[t] = (v, "dialect-dict:user")
    if work is not None:
        for t, v in _pairs(Path(work) / "dialect_ipa.txt").items():
            out[t] = (v, "dialect-dict:book")
    return out


def is_dialect(token: str, known: dict | set = ()) -> bool:
    """An apostrophe word that is not a standard contraction, a possessive (days', Epps', John's) or a capitalised name (see lexicon).
    A word in the dialect lists (jes', las') is dialect even where it looks like a plural possessive."""
    k = key(token)
    if "'" not in k or STANDARD.match(k) or k.strip("'") == "":
        return False
    if k in known:
        return True
    if re.search(r"s'$", k) or re.search(r"[a-z]'s$", k):                       # plural and ordinary possessives
        return False
    t = token.replace("’", "'")
    if t.startswith("'") and t.endswith("'") and len(t) > 2:                       # a word in single quotes: 'many'
        return False
    first = t.lstrip("'")[:1]
    return not first.isupper() and not (token.startswith(("'", "’")) and token.lstrip("'’")[:1].isupper())


def scan(work: Path) -> list[dict]:
    """The dialect words of a book: [{token, count, ipa, ipa_src}], most used first."""
    work = Path(work)
    meta = json.loads((work / "chapters.json").read_text(encoding="utf-8"))
    count = Counter()
    known = sounds(work)
    for ch in meta["chapters"]:
        for m in TOKEN.finditer(ch.get("text", "")):
            if is_dialect(m.group(), known):
                count[key(m.group())] += 1
    return [{"token": t, "count": n, "ipa": known.get(t, ("", ""))[0], "ipa_src": known.get(t, ("", ""))[1]} for t, n in count.most_common()]


def apply(work: Path) -> dict:
    """Add the dialect words that have a sound to lexicon.json (kind "dialect"); write the rest to dialect_scan.json. Entries you made
    yourself, or already gave a sound, are not changed. Returns {"added", "missing", "words"}."""
    work = Path(work)
    found = scan(work)
    path = work / "lexicon.json"
    lex = json.loads(path.read_text(encoding="utf-8")) if path.exists() else []
    have = {key(e["term"]): e for e in lex}
    added = 0
    for w in found:
        if not w["ipa"]:
            continue
        e = have.get(w["token"])
        if e is None:
            lex.append({"term": w["token"], "count": w["count"], "kind": "dialect", "known": False, "lang": "", "lang_src": "", "lang_conf": 0.0,
                        "guess": "", "ipa": w["ipa"], "ipa_src": w["ipa_src"], "respell": "", "source": "auto", "zipf": 0.0})
            added += 1
        elif e.get("source") != "user" and not e.get("ipa"):
            e.update(ipa=w["ipa"], ipa_src=w["ipa_src"], kind=e.get("kind") or "dialect")
    path.write_text(json.dumps(lex, indent=2, ensure_ascii=False), encoding="utf-8")
    missing = [w for w in found if not w["ipa"]]
    (work / "dialect_scan.json").write_text(json.dumps({"words": found, "missing_a_sound": [w["token"] for w in missing]}, indent=1, ensure_ascii=False),
                                            encoding="utf-8")
    return {"added": added, "missing": len(missing), "words": len(found)}
