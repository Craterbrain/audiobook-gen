"""Is this the right way to read this text? Look at what the text is made of, pick the reader that fits (a play, chapter:verse scripture,
'Chapter N' headings, or none), and afterwards check that what came out looks like a book.

    kind(raw)              -> {"kind": play|verse|chapters|single, "confidence": 0..1, "evidence": {...}, "reasons": [...]}
    validate(book, raw_len)-> {"level": ok|check|bad, "notes": [plain-language findings], "stats": {...}}

The report is saved in chapters.json as "parse_report" and shown on the Book page when something looks off.
"""
import re
import statistics

SPEAKER_LINE = re.compile(r"^[A-Z][A-Z .’'-]+\.$", re.M)
SCENE_LINE = re.compile(r"^(?:SCENE|ACT) [IVX]+\b", re.M)
VERSE_RE = re.compile(r"(?<!\d)(\d{1,3}):(\d{1,3})\s+")
NUM = r"(?:\d{1,4}|[IVXLC]{1,7}|(?:one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|thirteen|fourteen|fifteen|sixteen|seventeen|eighteen|nineteen|twenty)[\w-]*)"
# a heading is a short line of its own: "CHAPTER XV.", "Book Two: The Earth under the Martians", "Part 3 - Dawn"
STRICT_HEADING = re.compile(rf"^[ \t]*((?:chapter|book|part|section|letter|canto)[ \t]+{NUM}\b[ \t]*[.:\-–—]?[ \t]*[^\n]{{0,90}})$", re.I | re.M)
LOOSE_HEADING = re.compile(r"^\s*((?:chapter|book|part)\s+[\w.]+.*)$", re.I | re.M)


def _standalone(m: re.Match, raw: str) -> bool:
    """A heading line, not a sentence that happens to begin with 'Part': short, not ending like running text, and set off by a blank line."""
    line = m.group(1).strip()
    if len(line) > 110 or line.endswith((",", ";")) or len(line.split()) > 18:
        return False
    before = raw[:m.start()].rstrip(" \t")
    after = raw[m.end():m.end() + 3]
    return before.endswith("\n\n") or before == "" or before.endswith("\n") and after.startswith("\n")


def headings(raw: str) -> list[re.Match]:
    return [m for m in STRICT_HEADING.finditer(raw) if _standalone(m, raw)]


def verse_sequence(raw: str) -> tuple[int, float]:
    """(how many chapter:verse markers, the share that follow on from the one before: 1:1 1:2 1:3 ... 2:1). Scripture is nearly all in sequence;
    times of day, ratios and scores are not."""
    marks = [(int(a), int(b)) for a, b in VERSE_RE.findall(raw) if int(a) <= 150 and 1 <= int(b) <= 176 and int(a) >= 1]
    if len(marks) < 2:
        return len(marks), 0.0
    seq = sum(1 for (c0, v0), (c1, v1) in zip(marks, marks[1:]) if (c1 == c0 and v1 == v0 + 1) or (v1 == 1 and c1 in (c0, c0 + 1)))
    return len(marks), seq / (len(marks) - 1)


def kind(raw: str) -> dict:
    """Which reader fits this text. Order matters: a play and scripture are very specific; chapter headings are next; else it is one block."""
    speakers = len(SPEAKER_LINE.findall(raw))
    scenes = len(re.findall(r"^SCENE [IVX]+\.", raw, re.M))
    marks, seq = verse_sequence(raw)
    heads = headings(raw)
    loose = len(LOOSE_HEADING.findall(raw))
    ev = {"speaker_lines": speakers, "scene_lines": scenes, "verse_marks": marks, "verse_in_sequence": round(seq, 2),
          "chapter_headings": len(heads), "loose_heading_lines": loose}
    reasons = []
    if speakers > 50 and scenes >= 1:
        reasons.append(f"{speakers} lines of the form SPEAKER. and {scenes} SCENE lines: a play")
        return {"kind": "play", "confidence": min(1.0, 0.6 + speakers / 500), "evidence": ev, "reasons": reasons}
    if marks > 20 and seq >= 0.6:
        reasons.append(f"{marks} chapter:verse numbers, {seq:.0%} of them in order: scripture")
        return {"kind": "verse", "confidence": min(1.0, seq), "evidence": ev, "reasons": reasons}
    if marks > 20:
        reasons.append(f"{marks} things like 12:30 but only {seq:.0%} follow each other, so they are times or ratios, not verses")
    if len(heads) >= 2:
        reasons.append(f"{len(heads)} headings of the form “Chapter N” on lines of their own")
        return {"kind": "chapters", "confidence": min(1.0, 0.5 + len(heads) / 40), "evidence": ev, "reasons": reasons}
    if loose >= 2:
        reasons.append(f"{loose} lines begin with Chapter/Book/Part, but none looks like a heading on its own line")
        return {"kind": "chapters_loose", "confidence": 0.3, "evidence": ev, "reasons": reasons}
    reasons.append("no chapter headings, scenes or verse numbers: one block of text")
    return {"kind": "single", "confidence": 0.6, "evidence": ev, "reasons": reasons}


def validate(book: dict, raw_len: int = 0, reader: str = "chapters") -> dict:
    """Does the result look like a book? Plain-language notes; level ok / check / bad."""
    ch = book.get("chapters") or []
    sizes = [len(c.get("text", "")) for c in ch]
    total = sum(sizes)
    notes, bad = [], False
    stats = {"chapters": len(ch), "chars": total}
    if not ch or total < 200:
        return {"level": "bad", "notes": ["Almost no text came out."], "stats": stats}
    med = statistics.median(sizes)
    stats["median_chapter_chars"] = int(med)
    if len(ch) == 1 and total > 60_000:
        notes.append(f"The whole book ({total:,} characters) is one chapter: no chapter headings were found. Chapters give you navigation and music between sections.")
    if len(ch) > 3 and med < 500:
        notes.append(f"Most chapters are tiny (median {int(med)} characters): headings may be matching ordinary lines, or this is a contents list.")
        bad = bad or med < 150
    if len(ch) > 5 and max(sizes) > 12 * med and max(sizes) > 50_000:
        notes.append(f"One chapter is {max(sizes) // max(1, int(med))} times the typical size: a heading may have been missed there.")
    empty = sum(1 for s in sizes if s < 40)
    if empty:
        notes.append(f"{empty} chapter(s) have no text.")
    if raw_len and total < 0.8 * raw_len and reader in ("chapters", "single"):
        notes.append(f"Only {total / raw_len:.0%} of the text ended up in chapters: part of the file was dropped.")
        stats["coverage"] = round(total / raw_len, 2)
    titles = [c.get("title", "") for c in ch]
    if len(ch) > 3 and len(set(titles)) < 0.6 * len(titles):
        notes.append("Many chapters share a title: a contents list or a repeated page header may have been read as headings.")
    return {"level": "bad" if bad else ("check" if notes else "ok"), "notes": notes, "stats": stats}
