"""Read each section's title aloud, with a pause before and after it.

The title becomes the first segment of its chapter ("Part 1: The Word Became Flesh" is read as "Part one. The Word Became Flesh."). The pause
before it is the chapter's start silence (set from the same number in assemble); the pause after it is carried by the segment that follows.
The Opening chapter (a spoken book title over music) is left alone.
"""
import json
import re
from pathlib import Path

PAUSE_S = 2.8
SMALL = {"a", "an", "and", "as", "at", "by", "for", "from", "in", "of", "on", "or", "the", "to", "with"}


def _cap(t: str) -> str:
    """ALL-CAPS heading to title case, keeping small words lower-case (the first word is always capitalised)."""
    up = lambda w: re.sub(r"[a-z]", lambda m: m[0].upper(), w, count=1)         # capitalise the first letter, past an opening quote
    words = t.lower().split(" ")
    return " ".join(up(w) if i == 0 or w.strip("“”\"'(") not in SMALL else w for i, w in enumerate(words))


def _roman(r: str) -> int:
    v = {"I": 1, "V": 5, "X": 10, "L": 50, "C": 100}
    n = [v[c] for c in r]
    return sum(-a if a < b else a for a, b in zip(n, n[1:] + [0]))


WORDS = ("one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen seventeen eighteen nineteen "
         "twenty").split()


def _body(t: str) -> str:
    """The words of a title: an ALL-CAPS heading would be read letter by letter, so it becomes title case."""
    t = re.sub(r"\bEsq\b\.?", "Esquire", t, flags=re.I)
    t = t[:1].upper() + t[1:]                       # a heading after its number starts with a capital ("Book one. The Coming ...")
    if t.isupper():
        return _cap(t)
    return re.sub(r"\b[A-Z]{2,}\b", lambda x: x[0].capitalize(), t)


def spoken(title: str) -> str:
    """How a chapter title is read: numbers in words, a full stop after each part."""
    from num2words import num2words
    t = " ".join(title.split())
    lead = re.match(r"^([IVXLC]+)\.\s+(\S.*)$", t)                   # "II. THE FALLING STAR." -> "Two. The Falling Star."
    if lead:
        return f"{num2words(_roman(lead[1])).capitalize()}. {_body(lead[2]).rstrip('.')}."
    wb = re.match(r"^(book|part|chapter|section)\s+(" + "|".join(WORDS) + r"|[IVXLC]+)\s+(?=\w)(.+)$", t, re.I)    # "BOOK ONE THE COMING OF ..."
    if wb:
        num = wb[2].lower() if wb[2].lower() in WORDS else num2words(_roman(wb[2].upper()))
        return f"{wb[1].capitalize()} {num}. {_body(wb[3]).rstrip('.')}."
    r = re.match(r"^(part|chapter|book|section)\s+([IVXLC]+)\.?$", t, re.I)
    if r and r[2].upper() == r[2]:                        # "CHAPTER IV" -> "Chapter 4"
        t = f"{r[1]} {_roman(r[2])}"
    t = re.sub(r"\bEsq\b\.?", "Esquire", t, flags=re.I)
    if t.isupper():                                       # "PREFACE" would be read letter by letter
        t = _cap(t)
    else:
        t = re.sub(r"\b[A-Z]{2,}\b", lambda x: x[0].capitalize(), t)
    m = re.match(r"^(part|chapter|book|section)\s+(\d+)\s*[:.\-–—]?\s*(.*)$", t, re.I)
    if m:
        head = f"{m[1].capitalize()} {num2words(int(m[2]))}."
        return head + (f" {m[3].rstrip('.')}." if m[3] else "")
    t = re.sub(r"\b(\d+)\b", lambda x: num2words(int(x[1])), t)
    return t.rstrip(".") + "."


def _same(text: str, title: str) -> bool:
    norm = lambda x: re.sub(r"[^a-z0-9]+", " ", x.lower()).strip()
    return bool(norm(title)) and norm(text) == norm(title)


def apply(work: Path, speak: bool, pause_s: float = PAUSE_S) -> int:
    """Put the spoken titles into (or take them out of) segments.json. Returns how many titles are in the book afterwards."""
    work = Path(work)
    if not (work / "segments.json").exists():
        return 0
    meta = json.loads((work / "chapters.json").read_text())
    segs = json.loads((work / "segments.json").read_text())
    segs = [g for g in segs if not str(g.get("id", "")).endswith("-title")]
    for g in segs:
        g.pop("title_pause", None)
    n = 0
    if speak:
        out, seen = [], set()
        opening = {c["index"] for c in meta["chapters"] if c.get("opening")}
        names = {c["index"]: c["title"] for c in meta["chapters"]}
        after = None                                                    # the chapter whose next line follows its title and takes the pause
        for g in segs:
            ch = g["chapter"]
            if ch not in seen and ch not in opening and ch in names:
                seen.add(ch)
                out.append({"speaker": "Narrator", "kind": "narration", "text": spoken(names[ch]), "para_start": True, "chapter": ch,
                            "id": f"{ch:03d}-title"})
                n += 1
                if _same(g["text"], names[ch]):                         # the chapter's own first line is its title: it is read once, not twice
                    after = ch
                    continue
                g = {**g, "pause_ms": int(pause_s * 1000)}              # the pause after the title
            elif after == ch:
                g, after = {**g, "pause_ms": int(pause_s * 1000)}, None
            out.append(g)
        segs = out
    else:
        for g in segs:                                                  # a pause set only for a title goes with it
            if g.get("pause_ms") and g["chapter"] != 0 and g.get("pause_ms") == int(pause_s * 1000):
                g.pop("pause_ms")
    (work / "segments.json").write_text(json.dumps(segs, indent=1, ensure_ascii=False))
    return n
