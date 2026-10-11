"""Make a plain-text EPUB (no audio) from a work folder: python -m audiobook_gen epub x --work work/<book> [--out book.epub]
One chapter per entry in chapters.json, paragraphs split on blank lines, the work's cover.jpg as cover. Standard library only."""
import json
import sys
import uuid
import zipfile
from html import escape
from pathlib import Path

CSS = "body{font-family:serif;line-height:1.5}h1{margin:2em 0 1em;font-size:1.5em}p{margin:0 0 .8em;text-indent:0}"


def build(work: Path, out: Path) -> Path:
    meta = json.loads((work / "chapters.json").read_text(encoding="utf-8"))
    title, author = meta.get("title", work.name), meta.get("author") or ""
    cover = work / "cover.jpg"
    chapters = [c for c in meta["chapters"] if c.get("text", "").strip()]
    page = lambda t, body: ('<?xml version="1.0" encoding="utf-8"?>\n<!DOCTYPE html>\n<html xmlns="http://www.w3.org/1999/xhtml" xmlns:epub="http://www.idpf.org/2007/ops">'
                            f'<head><meta charset="utf-8"/><title>{escape(t)}</title><link rel="stylesheet" href="style.css"/></head><body>{body}</body></html>')
    files, manifest, spine, nav = {}, [], [], []
    for n, c in enumerate(chapters, 1):
        name = f"c{n:03d}.xhtml"
        paras = [p.strip() for p in c["text"].split("\n\n") if p.strip()]
        opening = c.get("opening") in (True, "True")
        body = (f"<h1>{escape(c.get('title', ''))}</h1>" if not opening else "") + "".join(f"<p>{escape(p).replace(chr(10), '<br/>')}</p>" for p in paras)
        files[f"OEBPS/{name}"] = page(c.get("title", ""), body)
        manifest.append(f'<item id="c{n}" href="{name}" media-type="application/xhtml+xml"/>')
        spine.append(f'<itemref idref="c{n}"/>')
        nav.append(f'<li><a href="{name}">{escape(c.get("title", "") or "Opening")}</a></li>')
    files["OEBPS/nav.xhtml"] = page("Contents", '<nav epub:type="toc"><h1>Contents</h1><ol>' + "".join(nav) + "</ol></nav>")
    files["OEBPS/style.css"] = CSS
    cover_meta = ""
    if cover.exists():
        cover_meta = '<meta name="cover" content="cover-img"/>'
        manifest.append('<item id="cover-img" href="cover.jpg" media-type="image/jpeg" properties="cover-image"/>')
    uid = f"urn:uuid:{uuid.uuid5(uuid.NAMESPACE_URL, str(work.name))}"
    files["OEBPS/content.opf"] = ('<?xml version="1.0" encoding="utf-8"?><package xmlns="http://www.idpf.org/2007/opf" version="3.0" unique-identifier="id">'
        f'<metadata xmlns:dc="http://purl.org/dc/elements/1.1/"><dc:identifier id="id">{uid}</dc:identifier><dc:title>{escape(title)}</dc:title>'
        f'<dc:language>en</dc:language>{f"<dc:creator>{escape(author)}</dc:creator>" if author else ""}{cover_meta}</metadata>'
        '<manifest><item id="nav" href="nav.xhtml" media-type="application/xhtml+xml" properties="nav"/><item id="css" href="style.css" media-type="text/css"/>'
        + "".join(manifest) + "</manifest><spine>" + "".join(spine) + "</spine></package>")
    out.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(out, "w") as z:
        z.writestr("mimetype", "application/epub+zip", zipfile.ZIP_STORED)
        z.writestr("META-INF/container.xml", '<?xml version="1.0"?><container version="1.0" xmlns="urn:oasis:names:tc:opendocument:xmlns:container">'
                   '<rootfiles><rootfile full-path="OEBPS/content.opf" media-type="application/oebps-package+xml"/></rootfiles></container>', zipfile.ZIP_DEFLATED)
        for k, v in files.items():
            z.writestr(k, v, zipfile.ZIP_DEFLATED)
        if cover.exists():
            z.write(cover, "OEBPS/cover.jpg")
    return out


if __name__ == "__main__":
    w = Path(sys.argv[1])
    print(build(w, Path(sys.argv[2]) if len(sys.argv) > 2 else Path("out") / f"{w.name}.epub"))
