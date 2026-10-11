"""Module 1a: raw text + chapter structure from .epub / .txt / .md / .html / .docx."""
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


GUTENBERG_END = re.compile(r"THE FULL PROJECT GUTENBERG|\*\*\* ?END OF (?:THE |THIS )?PROJECT GUTENBERG", re.I)
HEAD_LINE = re.compile(r"^(The Project Gutenberg eBook of|Title|Author|Editor|Translator|Illustrator|Release date|Language|Credits?|"
                       r"Produced by|Other information|Character set|Original publication|Contributor|Most recently|www\.|https?:)", re.I)
FRONT_TITLES = re.compile(r"^\s*(contents|table of contents|illustrations|list of illustrations|title page|"
                          r"copyright|transcriber.s note|footnotes)\s*\.?\s*$", re.I)


def strip_gutenberg(chapters: list[dict]) -> list[dict]:
    """Remove Project Gutenberg's header and licence text, and contents/illustration pages, so none of it is read aloud."""
    out = []
    for ch in chapters:
        text = ch["text"]
        if re.match(r"\s*The Project Gutenberg eBook of", text, re.I):       # drop the title block line by line
            lines = text.split("\n")
            k = 0
            while k < len(lines) and (not lines[k].strip() or HEAD_LINE.match(lines[k].strip())):
                k += 1
            text = "\n".join(lines[k:])
        m = GUTENBERG_END.search(text)
        if m:
            text = text[:m.start()]
        text = text.strip()
        body = re.sub(r"^\s*" + re.escape(ch["title"]) + r"\s*", "", text, flags=re.I)
        if len(text) < 200 or FRONT_TITLES.match(ch["title"]) or re.match(r"\s*(table of )?contents\s*\n", body, re.I):
            continue
        title = ch["title"]
        if GUTENBERG_END.search(title):                      # the table of contents named this file after the licence
            title = text.split("\n", 1)[0].strip(" .").title()[:40] or "Note"
        out.append({**ch, "title": title, "text": text})
    return [{**c, "index": i + 1} for i, c in enumerate(out)]


NOTE_LABEL = re.compile(r"\[?\s*(\d{1,3})\s*\]?\.?")


def _footnotes(soup, pages: dict | None = None) -> int:
    """Gutenberg-style notes: a numbered link in the text ("[1]") that points at a note, on this page or another one (`pages` maps file
    names to their parsed pages). The marker is removed and the note is read right after the paragraph that cites it, called out as
    "Footnote one. <note>". Returns how many notes were moved."""
    from num2words import num2words
    pages = pages or {}
    moved = 0
    for a in list(soup.find_all("a", href=True)):
        if a.decomposed:                                          # already taken away with its note
            continue
        m = NOTE_LABEL.fullmatch(a.get_text(strip=True))
        file, _, frag = a["href"].partition("#")
        if not m or not frag or a.parent is None:
            continue
        page = pages.get(Path(file).name, soup) if file else soup
        target = page.find(id=frag)
        block = target.find_parent(["p", "div", "li"]) if target is not None else None
        para = a.find_parent(["p", "div", "li"])
        if block is None or block is para:
            continue
        prev = a.find_previous_sibling("a")
        ids = {x for x in (a.get("id"), prev.get("id") if prev is not None and not prev.get_text(strip=True) else None) if x}
        back = block.find("a", href=lambda h: bool(h) and h.partition("#")[2] in ids)                 # the note links back to this marker
        text0 = block.get_text(" ", strip=True)
        starts = NOTE_LABEL.match(text0) is not None and NOTE_LABEL.match(text0).group(1) == m.group(1)
        if back is None and not starts:
            continue                                            # a link to something else that happens to have a number
        for x in block.find_all("a", href=True):                 # the "[1]" at the start of the note itself
            if NOTE_LABEL.fullmatch(x.get_text(strip=True)):
                x.decompose()
                break
        note = re.sub(r"\s+", " ", block.get_text(" ", strip=True))
        note = re.sub(r"\s+([,.;:!?])", r"\1", note)
        note = re.sub(r"^\[?\s*" + m.group(1) + r"\s*\]?\.?\s*", "", note).strip()
        sup = a.find_parent("sup")
        (sup or a).decompose()
        if prev is not None and not prev.decomposed and not prev.get_text(strip=True):
            prev.decompose()
        block.decompose()
        if note and para is not None:
            new = soup.new_tag("p")
            new.string = f"Footnote {num2words(int(m.group(1)))}. {note}"
            after = para
            while after.next_sibling is not None and getattr(after.next_sibling, "get", None) and "footnote-spoken" in (after.next_sibling.get("class") or []):
                after = after.next_sibling                      # several notes in one paragraph keep their order
            new["class"] = ["footnote-spoken"]
            after.insert_after(new)
            moved += 1
    for page in [soup, *pages.values()]:                         # what is left of the "FOOTNOTES:" lists
        for d in page.find_all(class_=["footnote", "footnotes"]):
            if not d.decomposed and re.fullmatch(r"(FOOTNOTES:?)?", d.get_text(strip=True), re.I):
                d.decompose()
    return moved


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
    pages = {}
    for item_id, _ in book.spine:
        item = book.get_item_with_id(item_id)
        if item is not None and item.get_type() == ebooklib.ITEM_DOCUMENT:
            pages[item_id] = (item, BeautifulSoup(item.get_content(), "lxml"))
    by_file = {Path(it.get_name()).name: sp for it, sp in pages.values()}       # a note may sit on a different page from its marker
    for it, sp in pages.values():
        _footnotes(sp, by_file)
    for item_id, _ in book.spine:
        if item_id not in pages:
            continue
        item, soup = pages[item_id]
        head = soup.find(["h1", "h2", "h3"])
        paras = [p.get_text(" ", strip=True) for p in soup.find_all(["p", "h1", "h2", "h3"])]
        text = _clean("\n\n".join(p for p in paras if p))
        if len(text) < 200:  # skip copyright/blank pages
            continue
        title = toc_titles.get(item.get_name()) or (head.get_text(strip=True) if head else None)
        chapters.append({"index": len(chapters) + 1,
                         "title": title or f"Chapter {len(chapters) + 1}", "text": text})

    from .textclean import clean_chapters
    chapters = [c for c in clean_chapters(chapters) if len(c["text"]) >= 200]       # small-capital names put right, capitals tamed, the file note gone
    for n, c in enumerate(chapters, 1):
        c["index"] = n
    cover = None
    for item in book.get_items():
        if item.get_type() == ebooklib.ITEM_COVER or "cover" in item.get_name().lower() \
                and item.media_type.startswith("image/"):
            out_dir.mkdir(parents=True, exist_ok=True)
            cover = out_dir / ("cover" + Path(item.get_name()).suffix)
            cover.write_bytes(item.get_content())
            break
    if "gutenberg" in (chapters[0]["text"][:300] + chapters[-1]["text"][-3000:]).lower() if chapters else False:
        chapters = strip_gutenberg(chapters)
    return {"title": meta("title") or path.stem, "author": meta("creator"),
            "cover": str(cover) if cover else None, "chapters": chapters}


TEXT_EXTS = [".epub", ".txt", ".text", ".md", ".markdown", ".html", ".htm", ".docx"]


def _pretty(stem: str) -> str:
    return re.sub(r"[_\-]+", " ", stem).strip().title() or "Untitled"


def _from_blocks(blocks: list[tuple[int, str]], fallback: str, title: str = "", author: str = "") -> dict:
    """Chapters from a flat list of (heading level, or 0 for a paragraph; text). Chapters start at the shallowest heading level that occurs
    at least twice; a single heading above that level is the book's title; with no such level the whole text is one chapter."""
    blocks = [(lv, re.sub(r"\s+", " ", t).strip()) for lv, t in blocks]
    blocks = [(lv, t) for lv, t in blocks if t]
    counts: dict[int, int] = {}
    for lv, _ in blocks:
        if lv:
            counts[lv] = counts.get(lv, 0) + 1
    split = min((lv for lv, n in counts.items() if n >= 2), default=0)
    skip = -1
    if counts:
        top = min(counts)
        if counts[top] == 1 and (top < split or not split):
            skip = next(i for i, (lv, _) in enumerate(blocks) if lv == top)         # the one heading above the chapters: the title
            title = title or blocks[skip][1]
    title = title or fallback
    chapters, cur = [], None
    for i, (lv, text) in enumerate(blocks):
        if i == skip:
            continue
        if split and lv and lv <= split:
            cur = {"index": len(chapters) + 1, "title": text, "text": ""}
            chapters.append(cur)
            continue
        if cur is None:
            cur = {"index": 1, "title": title, "text": ""}
            chapters.append(cur)
        cur["text"] += ("\n\n" if cur["text"] else "") + text                        # other headings are read as a line of text
    chapters = [{**c, "text": _clean(c["text"])} for c in chapters if c["text"].strip()]
    for n, c in enumerate(chapters, 1):
        c["index"] = n
    return {"title": title, "author": author, "cover": None, "chapters": chapters}


def extract_markdown(path: Path) -> dict:
    raw = path.read_text(encoding="utf-8", errors="replace").replace("\r", "")
    meta = {}
    m = re.match(r"^---\n(.*?)\n---\n", raw, re.S)                                  # YAML front matter: title / author
    if m:
        for line in m.group(1).splitlines():
            k, _, v = line.partition(":")
            meta[k.strip().lower()] = v.strip().strip("\"'")
        raw = raw[m.end():]
    raw = re.sub(r"^(```|~~~).*?^\1[ \t]*$", "", raw, flags=re.S | re.M)              # code blocks are not read aloud
    raw = re.sub(r"<!--.*?-->", "", raw, flags=re.S)
    raw = re.sub(r"^(.+)\n=+[ \t]*$", r"# \1", raw, flags=re.M)                        # setext headings
    raw = re.sub(r"^(.+)\n-{2,}[ \t]*$", r"## \1", raw, flags=re.M)
    blocks: list[tuple[int, str]] = []
    para: list[str] = []

    def flush():
        if para:
            blocks.append((0, " ".join(para)))
            para.clear()
    for line in raw.split("\n"):
        h = re.match(r"^\s{0,3}(#{1,6})\s+(.*?)\s*#*\s*$", line)
        if h:
            flush()
            blocks.append((len(h[1]), _md_inline(h[2])))
        elif not line.strip() or re.match(r"^\s{0,3}([-*_]\s*){3,}$", line) or line.lstrip().startswith("|"):    # blank, rule, table row
            flush()
        elif re.match(r"^\s*([-*+]|\d+[.)])\s+", line):                                  # each list item is a paragraph
            flush()
            para.append(_md_inline(re.sub(r"^\s*([-*+]|\d+[.)])\s+", "", line)))
            flush()
        else:
            para.append(_md_inline(re.sub(r"^\s*>\s?", "", line)))
    flush()
    return _from_blocks(blocks, _pretty(path.stem), meta.get("title", ""), meta.get("author", ""))


def _md_inline(t: str) -> str:
    t = re.sub(r"!\[[^\]]*\]\([^)]*\)", "", t)                                     # images
    t = re.sub(r"\[([^\]]*)\]\([^)]*\)", r"\1", t)                                  # [text](url) -> text
    t = re.sub(r"\[\^[^\]]*\]", "", t)                                              # footnote marks
    t = re.sub(r"<(https?://[^>]*)>", "", t)
    t = re.sub(r"<[^>]+>", "", t)
    t = re.sub(r"(\*\*|__|\*|_|`|~~)(.+?)\1", r"\2", t)
    return t.replace("\\", "").strip()


def extract_html(path: Path) -> dict:
    from bs4 import BeautifulSoup
    soup = BeautifulSoup(path.read_bytes(), "lxml")
    for t in soup(["script", "style", "nav", "header", "footer", "aside", "noscript"]):
        t.decompose()
    title = (soup.title.get_text(strip=True) if soup.title else "")
    au = soup.find("meta", attrs={"name": "author"})
    blocks = []
    for el in (soup.body or soup).find_all(["h1", "h2", "h3", "h4", "h5", "h6", "p", "li", "blockquote"]):
        if el.name == "blockquote" and el.find(["p", "li"]):
            continue                                                                  # its paragraphs are read on their own
        blocks.append((int(el.name[1]) if el.name[0] == "h" else 0, el.get_text(" ", strip=True)))
    return _from_blocks(blocks, title or _pretty(path.stem), "", au.get("content", "") if au else "")


def extract_docx(path: Path) -> dict:
    import zipfile
    import xml.etree.ElementTree as ET
    W = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"
    with zipfile.ZipFile(path) as z:
        root = ET.fromstring(z.read("word/document.xml"))
        core = ET.fromstring(z.read("docProps/core.xml")) if "docProps/core.xml" in z.namelist() else None
    blocks, title = [], ""
    for p in root.iter(W + "p"):
        style = p.find(f"{W}pPr/{W}pStyle")
        sid = (style.get(W + "val") if style is not None else "") or ""
        text = "".join(t.text or "" for t in p.iter(W + "t"))
        m = re.match(r"(?i)heading\s*(\d)", sid)
        if sid.lower() == "title":
            title = title or text.strip()
        elif m:
            blocks.append((int(m[1]), text))
        else:
            blocks.append((0, text))
    author = ""
    if core is not None:
        for el in core:
            if el.tag.endswith("}creator") and el.text:
                author = el.text
            if el.tag.endswith("}title") and el.text and not title:
                title = el.text
    return _from_blocks(blocks, _pretty(path.stem), title, author)


def extract(path: str, out_dir: str, max_chapters: int | None = None) -> dict:
    path, out_dir = Path(path), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ext = path.suffix.lower()
    if ext not in TEXT_EXTS:
        raise ValueError(f"Cannot read {path.suffix or 'that file'}: use one of {', '.join(TEXT_EXTS)}.")
    data = (extract_epub(path, out_dir) if ext == ".epub" else extract_markdown(path) if ext in (".md", ".markdown") else
            extract_html(path) if ext in (".html", ".htm") else extract_docx(path) if ext == ".docx" else extract_txt(path))
    if max_chapters:
        data["chapters"] = data["chapters"][:max_chapters]
    (out_dir / "chapters.json").write_text(json.dumps(data, indent=2, ensure_ascii=False))
    return data
