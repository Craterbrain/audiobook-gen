"""Extract books from a verse-per-line Bible text file into chapters.json.

Format: `Book Chapter:Verse text` one line per verse.
Chapters become paragraphs; verse numbers are stripped (no "1:1" spoken).
Adjacent verses are joined into flowing prose paragraphs of ~5 verses each
so the TTS produces natural reading rather than choppy single sentences.
"""
import collections
import json
import re
from pathlib import Path


def _parse(path: str) -> dict[str, dict[int, dict[int, str]]]:
    books: dict[str, dict] = {}
    with open(path) as f:
        for line in f:
            m = re.match(r'^([1-3]?\s*[A-Za-z ]+?)\s+(\d+):(\d+)\s+(.*)', line.strip())
            if m:
                b = m.group(1).strip()
                ch, vs, text = int(m.group(2)), int(m.group(3)), m.group(4).strip()
                books.setdefault(b, collections.defaultdict(dict))[ch][vs] = text
    return books


def _clean(text: str) -> str:
    """Strip editorial marks: asterisks (Gr. verb form), brackets for implied words,
    and footnote-style parenthetical references like (cf. Jn 3:16)."""
    text = text.replace("*", "")
    text = re.sub(r"\[([^\]]*)\]", r"\1", text)   # keep implied words, lose brackets
    return text.strip()


def extract(bible_path: str, book: str, work_dir: str, verses_per_para: int = 5) -> None:
    """Write chapters.json (and chapters.full.json) to work_dir for one book."""
    books = _parse(bible_path)
    data = books.get(book)
    if not data:
        raise ValueError(f"Book '{book}' not found. Available: {sorted(books)}")

    chapters = []
    for ch_num in sorted(data):
        verses = data[ch_num]
        # Group consecutive verses into paragraphs
        paras, group = [], []
        for vs in sorted(verses):
            group.append(_clean(verses[vs]))
            if len(group) >= verses_per_para:
                paras.append(" ".join(group))
                group = []
        if group:
            paras.append(" ".join(group))
        chapters.append({
            "index": ch_num - 1,
            "title": f"Chapter {ch_num}",
            "text": "\n\n".join(paras),
        })

    Path(work_dir).mkdir(parents=True, exist_ok=True)
    p = Path(work_dir) / "chapters.json"
    p.write_text(json.dumps(chapters, indent=2, ensure_ascii=False))
    (Path(work_dir) / "chapters.full.json").write_text(
        json.dumps({"book": book, "source": "bible text", "chapters": chapters}, indent=2, ensure_ascii=False)
    )
    print(f"[extract_bible] {book}: {len(chapters)} chapters -> {p}")
