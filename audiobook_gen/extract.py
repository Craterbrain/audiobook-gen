"""Module 1a: raw text + chapter structure from .epub / .txt."""
import json
import re
from pathlib import Path

VERSE_RE = re.compile(r"(?<!\d)(\d{1,3}):(\d{1,3})\s+")
HEADING_RE = re.compile(r"^\s*((?:chapter|book|part)\s+[\w.]+.*)$", re.I | re.M)


def _clean(text: str) -> str:
    text = text.replace("\r", "")
    text = re.sub(r"^\s*\d{3,4}m?\s*$", "", text, flags=re.M)  # Gutenberg illustration page markers
    text = re.sub(r"[ \t]+", " ", text)
    return re.sub(r"\n{3,}", "\n\n", text).strip()


SPEAKER_LINE = re.compile(r"^[A-Z][A-Z .’'-]+\.$", re.M)
ROMAN = {"I": "one", "II": "two", "III": "three", "IV": "four", "V": "five", "VI": "six",
         "VII": "seven", "VIII": "eight", "IX": "nine", "X": "ten"}


def _spoken_heading(act: str, scene: str, place: str) -> str:
    return f"Act {ROMAN.get(act, act)}, scene {ROMAN.get(scene, scene)}. {place}"


def extract_play(raw: str, title: str) -> dict:
    """Gutenberg-style play: 'SPEAKER.' on its own line, then verse; one chapter per scene."""
    raw = raw.split("*** END OF")[0]
    start = [m.start() for m in re.finditer(r"^ACT I\s*$", raw, re.M)]
    body = raw[start[-1]:] if start else raw  # skip the table of contents
    body = re.sub(r"\[_.*?_\]", "", body, flags=re.S)             # [_Exeunt._]
    body = re.sub(r"^ .*(?:\n .*)*", "", body, flags=re.M)           # indented stage directions
    chapters, act, cur = [], "I", None
    for block in re.split(r"\n\s*\n", body):
        block = block.strip()
        if not block:
            continue
        if (m := re.match(r"ACT ([IVX]+)\b", block)) and "SCENE" not in block:
            act = m.group(1)
            continue
        if m := re.match(r"SCENE ([IVX]+)\.\s*(.*)", block, re.S):
            place = re.sub(r"\s+", " ", m.group(2)).strip().rstrip(".")
            cur = {"index": len(chapters) + 1, "title": f"Act {act}, Scene {m.group(1)}",
                   "format": "play", "heading": _spoken_heading(act, m.group(1), place),
                   "text": ""}
            chapters.append(cur)
        elif cur is not None:
            cur["text"] += block + "\n\n"
    return {"title": title, "author": "", "cover": None, "chapters": chapters}


def extract_txt(path: Path) -> dict:
    raw = path.read_text(encoding="utf-8", errors="replace")
    title = raw.strip().splitlines()[0].strip()
    if len(SPEAKER_LINE.findall(raw)) > 50 and re.search(r"^SCENE [IVX]+\.", raw, re.M):
        m = re.search(r"Title:\s*(.+)", raw)
        author = re.search(r"Author:\s*(.+)", raw)
        d = extract_play(raw, m.group(1).strip() if m else title)
        d["author"] = author.group(1).strip() if author else ""
        return d
    # Bible-style "chapter:verse" text (the KJV sample): chapter = number before the colon.
    if len(VERSE_RE.findall(raw)) > 20:
        chapters, cur = {}, None
        pieces = VERSE_RE.split(raw)  # [pre, ch, v, text, ch, v, text, ...]
        for i in range(1, len(pieces) - 2, 3):
            ch = int(pieces[i])
            body = re.sub(r"\s+", " ", pieces[i + 2]).strip()
            chapters.setdefault(ch, []).append(body)
        return {
            "title": title, "author": "", "cover": None,
            "chapters": [
                {"index": n, "title": f"Chapter {n}", "text": "\n\n".join(v)}
                for n, v in sorted(chapters.items())
            ],
        }
    heads = list(HEADING_RE.finditer(raw))
    if not heads:
        return {"title": title, "author": "", "cover": None,
                "chapters": [{"index": 1, "title": title, "text": _clean(raw)}]}
    chapters = []
    for i, m in enumerate(heads):
        end = heads[i + 1].start() if i + 1 < len(heads) else len(raw)
        chapters.append({"index": i + 1, "title": m.group(1).strip(),
                         "text": _clean(raw[m.end():end])})
    return {"title": title, "author": "", "cover": None, "chapters": chapters}


def extract_epub(path: Path, out_dir: Path) -> dict:
    import ebooklib
    from bs4 import BeautifulSoup
    from ebooklib import epub

    book = epub.read_epub(str(path))
    meta = lambda k: (book.get_metadata("DC", k) or [[""]])[0][0]

    # Map href -> TOC title
    toc_titles = {}
    def walk(items):
        for it in items:
            if isinstance(it, tuple):
                walk(it[1])
            elif hasattr(it, "href"):
                toc_titles.setdefault(it.href.split("#")[0], it.title)
    walk(book.toc)

    chapters = []
    for item_id, _ in book.spine:
        item = book.get_item_with_id(item_id)
        if item is None or item.get_type() != ebooklib.ITEM_DOCUMENT:
            continue
        soup = BeautifulSoup(item.get_content(), "lxml")
        head = soup.find(["h1", "h2", "h3"])
        paras = [p.get_text(" ", strip=True) for p in soup.find_all(["p", "h1", "h2", "h3"])]
        text = _clean("\n\n".join(p for p in paras if p))
        if len(text) < 200:  # skip copyright/blank pages
            continue
        title = toc_titles.get(item.get_name()) or (head.get_text(strip=True) if head else None)
        chapters.append({"index": len(chapters) + 1,
                         "title": title or f"Chapter {len(chapters) + 1}", "text": text})

    cover = None
    for item in book.get_items():
        if item.get_type() == ebooklib.ITEM_COVER or "cover" in item.get_name().lower() \
                and item.media_type.startswith("image/"):
            out_dir.mkdir(parents=True, exist_ok=True)
            cover = out_dir / ("cover" + Path(item.get_name()).suffix)
            cover.write_bytes(item.get_content())
            break
    return {"title": meta("title") or path.stem, "author": meta("creator"),
            "cover": str(cover) if cover else None, "chapters": chapters}


def extract(path: str, out_dir: str, max_chapters: int | None = None) -> dict:
    path, out_dir = Path(path), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    data = extract_epub(path, out_dir) if path.suffix.lower() == ".epub" else extract_txt(path)
    if max_chapters:
        data["chapters"] = data["chapters"][:max_chapters]
    (out_dir / "chapters.json").write_text(json.dumps(data, indent=2, ensure_ascii=False))
    return data
