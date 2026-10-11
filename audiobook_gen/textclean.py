"""Clean-up of book text that comes out of an epub (or a work folder made from one) looking wrong to a voice:

  * small-capital names split into pieces: "D OUGLASS", "P ATRICK H ENRY", "B OSTON" (the epub writes D<small>OUGLASS</small>, and the
    text extractor puts a space where the tag was) -> "Douglass", "Patrick Henry", "Boston";
  * names and words in ALL CAPITALS: "FREDERICK DOUGLASS", "NO COMPROMISE WITH SLAVERY!" -> "Frederick Douglass", "No compromise with slavery!"
    (a name keeps its capital letter; acronyms and roman numerals are left alone);
  * "Esq." / "ESQ." -> "Esquire" (the full stop is kept only where it ends a sentence);
  * chapter titles written in capitals ("CHAPTER VII", "LETTER FROM WENDELL PHILLIPS, ESQ.") -> Title Case;
  * a space before punctuation ("Douglass ," "Institution !");
  * paragraphs that are not part of the book: the Project Gutenberg file note and a "Contents" line.

`fix_text` works on one text with the rest of the book as evidence for what is a name; `clean_chapters` does a whole book; `clean_work` fixes a
work folder that was made before this existed (chapters.json and segments.json) without touching anything else."""
import json
import re
from pathlib import Path

LETTER = "A-Za-zÀ-ÖØ-öø-ÿ"
CRUFT = re.compile(r"^\s*(?:Note from the original file\b.*|(?:Table of )?Contents\s*|Transcriber.s Note:?\s*)$", re.I | re.S)
ROMAN = re.compile(r"^[IVXLCDM]+$")
SPLIT = re.compile(r"\b([B-HJ-Z]) ([A-Z]{2,})\b")                     # not A and I: they are words on their own ("A PARODY", "I AM")
CAPS = re.compile(rf"\b[A-Z]{{3,}}\b")
SMALL_WORDS = {"a", "an", "and", "as", "at", "by", "for", "from", "in", "of", "on", "or", "the", "to", "with"}
KEEP = {"USA", "CSA", "YMCA", "NAACP", "AME", "KJV", "NASB", "BBC", "UFO", "DNA", "TNT", "NASA", "FBI", "CIA", "NATO", "UNESCO", "OPEC", "AIDS", "USSR", "CEO", "HTML", "PDF"}


def _zipf(word: str) -> float:
    try:
        from wordfreq import zipf_frequency
        return zipf_frequency(word.lower(), "en")
    except Exception:
        return 0.0


ESQ = re.compile(r"\bEsq\b\.?", re.I)


def esquire(text: str) -> str:
    """Esq. -> Esquire; the full stop stays when the sentence ends there (end of text, or a capital letter / quote after it)."""
    def sub(m):
        end = text[m.end():m.end() + 3]
        stop = m.group(0).endswith(".") and (not end.strip() or re.match(r"\s*[\"“‘'(]?[A-Z]", end) or end[:1] in "\n")
        return "Esquire" + ("." if stop else "")
    return ESQ.sub(sub, text)


def title_case(title: str) -> str:
    """A title written in capitals becomes Title Case (Esq. is spelled out); any other title is only given the Esquire fix."""
    t = _heading(title.strip()) or title
    m = re.match(r"^(M{0,3}(?:CM|CD|D?C{0,3})(?:XC|XL|L?X{0,3})(?:IX|IV|V?I{0,3}))\.\s+(\S.*)$", t, re.I)
    if m and m[1] and (m[1] != m[1].upper() or m[2][0].islower()):                       # an earlier version wrote "II. THE X" as "Ii. the X": put it right
        rest = m[2][0].upper() + m[2][1:] if m[2][0].islower() else m[2]
        t = f"{m[1].upper()}. {rest}"
    return esquire(t)


def evidence(texts: list[str]) -> set[str]:
    """Words the book itself writes with a capital letter and lower-case after it, in the middle of a sentence: its names."""
    seen = set()
    for t in texts:
        for m in re.finditer(rf"(?<![.!?\"“”‘’\n] )(?<!^)\b([A-Z][a-z]+)\b", t):
            seen.add(m.group(1).lower())
    return seen


def _heading(par: str) -> str | None:
    """A paragraph that is a short line in capitals ("LETTER FROM WENDELL PHILLIPS, ESQ.") written as a heading: Title Case."""
    letters = re.sub(r"[^A-Za-z]", "", par)
    words = par.split()
    if not letters or letters != letters.upper() or len(words) > 12 or len(letters) < 3:
        return None
    out = []
    for i, w in enumerate(words):
        core = re.sub(r"[^A-Za-z]", "", w)
        if ROMAN.match(core) and (len(core) == 1 or core in ("II", "III", "IV", "VI", "VII", "VIII", "IX", "XI", "XII")) and i > 0:
            out.append(w)
        elif i == 0 and w.endswith(".") and len(words) > 1 and ROMAN.fullmatch(core):      # "XII. WHAT I SAW" keeps its numeral
            out.append(w)
        elif core.upper() in KEEP:
            out.append(w)
        elif i and not words[i - 1].endswith((".", ":")) and core.lower() in SMALL_WORDS:
            out.append(w.lower())
        else:
            out.append(re.sub(r"[A-Za-z]+", lambda m: m.group(0).capitalize(), w))
    return " ".join(out)


def fix_text(text: str, names: set[str] | None = None) -> str:
    """The fixes above on one text. `names`: lower-case words known to be names in this book (see evidence())."""
    names = names or set()

    def join(m):
        merged = m.group(1) + m.group(2)
        return merged.capitalize() if merged.lower() in names or _zipf(merged) >= 2.5 else m.group(0)       # a small-capital word is a name
    text = SPLIT.sub(join, text)
    if "\n" in text:                                         # headings: a short line in capitals becomes Title Case
        text = "\n\n".join(_heading(p.strip()) or p for p in re.split(r"\n\s*\n", text))
    elif _heading(text.strip()):
        text = _heading(text.strip())

    def sentence_start(pos: int) -> bool:
        tail = text[max(0, pos - 6):pos]
        if pos == 0 or tail[-1:] in ("“", "\"", "‘", "'", "(") or not tail.strip():
            return True                                                # the start of the text, or of quoted speech
        return tail.rstrip(" \t").endswith((".", "!", "?", ":", "\n")) or tail.endswith("\n")

    def word(w: str, pos: int, single: bool = False) -> str:
        if ROMAN.match(w) or w in KEEP:
            return w
        low = w.lower()
        if low in names or single:
            return w.capitalize()                                     # a name, or one capital word: Capitalised (JOHN -> John, MAN -> Man)
        if len(w) >= 3 and _zipf(low) < 2.0:
            return w                                                  # an acronym, or a word nobody knows: left as it is
        return w.capitalize() if sentence_start(pos) else low

    def run(m):                                               # "NO COMPROMISE WITH SLAVERY!": one shouted phrase, only its first word capitalised
        out, pos = [], m.start()
        for part in re.split(r"( +)", m.group(0)):
            out.append(part if part.strip() == "" else word(part, pos))
            pos += len(part)
        return "".join(out)
    text = re.sub(r"\b[A-Z]{2,}(?: +[A-Z]{2,})+\b", run, text)                       # two or more capital words together
    text = CAPS.sub(lambda m: word(m.group(0), m.start(), True) if _zipf(m.group(0).lower()) >= 2.0 or m.group(0).lower() in names else m.group(0), text)                       # a single capital word of three letters or more
    text = esquire(text)
    text = re.sub(r"(?<=[A-Za-z0-9”’\"')]) +([,;:!?])", r"\1", text)
    text = re.sub(rf"(?<=[{LETTER}0-9]) +\.(?=\s|$|[”’\"'])", ".", text)
    return text


def drop_cruft(text: str) -> str:
    """Remove paragraphs that are not part of the book."""
    paras = re.split(r"\n\s*\n", text)
    return "\n\n".join(p for p in paras if not CRUFT.match(p.strip()))


def clean_chapters(chapters: list[dict]) -> list[dict]:
    """Clean every chapter of a book; chapters that are nothing but cruft and their own title disappear."""
    cleaned = [dict(c, text=drop_cruft(c["text"]), **({"title": title_case(c["title"])} if c.get("title") else {})) for c in chapters]
    names = evidence([c["text"] for c in cleaned])
    for c in cleaned:
        c["text"] = fix_text(c["text"], names)
    return cleaned


def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", s.lower()).strip()


def clean_work(work: Path) -> dict:
    """Fix a work folder made before this existed: chapters.json and segments.json. Returns what changed."""
    work = Path(work)
    meta = json.loads((work / "chapters.json").read_text())
    segs = json.loads((work / "segments.json").read_text()) if (work / "segments.json").exists() else []
    names = evidence([c.get("text", "") for c in meta["chapters"]])
    report = {"changed_text": 0, "dropped_segments": 0, "dropped_chapters": []}
    for c in meta["chapters"]:
        before = c["text"]
        c["text"] = fix_text(drop_cruft(c["text"]), names)
        report["changed_text"] += before != c["text"]
        if c.get("title") and title_case(c["title"]) != c["title"]:
            c["title"] = title_case(c["title"])
            report["changed_titles"] = report.get("changed_titles", 0) + 1
    kept = []
    for g in segs:
        if g["id"].endswith("-title") or g.get("kind") == "title":
            kept.append(g)
            continue
        if CRUFT.match(g["text"].strip()):
            report["dropped_segments"] += 1
            continue
        fixed = fix_text(g["text"], names)
        if fixed != g["text"]:
            report["changed_text"] += 1
        kept.append({**g, "text": fixed})
    # a chapter left with nothing but its title (the title page: title, file note, contents) is not a chapter
    titles = {c["index"]: _norm(c.get("title", "")) for c in meta["chapters"]}
    body = {}
    for g in kept:
        if not g["id"].endswith("-title") and _norm(g["text"]) != titles.get(g["chapter"]):
            body[g["chapter"]] = body.get(g["chapter"], 0) + 1
    gone = {c["index"] for c in meta["chapters"] if not c.get("opening") and not body.get(c["index"]) and c["index"] in {g["chapter"] for g in kept}}
    if gone:
        report["dropped_chapters"] = sorted(gone)
        meta["chapters"] = [c for c in meta["chapters"] if c["index"] not in gone]
        kept = [g for g in kept if g["chapter"] not in gone]
    if segs:
        (work / "segments.json").write_text(json.dumps(kept, indent=1, ensure_ascii=False))
    (work / "chapters.json").write_text(json.dumps(meta, indent=1, ensure_ascii=False))
    return report
