"""Cover maker: find a picture on Wikimedia Commons (public domain / CC0 only), and lay out a cover.

Two layouts: "full" fills the cover with a scenery picture behind the title; "framed" sets the whole picture like a window under
the title, on a book-cloth colour (for portraits and groups of people). "plain" has no picture.

The same tool serves the app's Cover tab and the Assistant (which may run only this command):
    python -m audiobook_gen.covers search "Frederick Douglass daguerreotype" [N]
    python -m audiobook_gen.covers fetch "File:Some name.jpg" --to folder      (prints the saved path and a credit line)
    python -m audiobook_gen.covers make spec.json --out cover.jpg
spec.json: {"title", "author", "layout": "full"|"framed"|"plain", "picture": "path", "color": "navy"|"#1a2b3c"|[r,g,b],
            "focus": [0.5, 0.5], "zoom": 1.0}
"""
import argparse
import json
import re
import sys
import urllib.parse
import urllib.request
from pathlib import Path

UA = {"User-Agent": "audiobook-gen cover maker (https://github.com/Craterbrain/audiobook-gen)"}
API = "https://commons.wikimedia.org/w/api.php"
CLOTH = {"navy": (20, 34, 70), "oxblood red": (78, 14, 22), "forest green": (22, 56, 36), "dark green": (18, 52, 38),
         "burgundy": (90, 22, 34), "brick red": (112, 42, 30), "deep olive": (52, 56, 28), "chocolate brown": (62, 38, 26),
         "slate blue": (30, 44, 66), "dark teal": (16, 56, 62), "ocean blue": (14, 42, 70), "charcoal": (36, 36, 40), "black": (0, 0, 0)}
LAYOUTS = ("full", "framed", "plain")


def colour(c) -> tuple[int, int, int]:
    """A cloth colour name, "#rrggbb", or [r, g, b]."""
    if isinstance(c, (list, tuple)) and len(c) == 3:
        return tuple(max(0, min(255, int(x))) for x in c)
    c = str(c or "black").strip().lower()
    if c in CLOTH:
        return CLOTH[c]
    m = re.fullmatch(r"#?([0-9a-f]{6})", c)
    if m:
        return tuple(int(m.group(1)[i:i + 2], 16) for i in (0, 2, 4))
    raise ValueError(f"unknown colour “{c}”; use one of {', '.join(CLOTH)} or #rrggbb")


def free_licence(name: str) -> bool:
    """Only pictures that need no credit on the cover: public domain and CC0."""
    n = (name or "").lower()
    return "public domain" in n or n.startswith("cc0") or n in ("pd", "pdm") or n.startswith("pd-")


def _api(**p) -> dict:
    url = API + "?" + urllib.parse.urlencode({"format": "json", **p})
    return json.load(urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=40))


def _plain(s: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"<[^>]+>", "", s or "")).strip()


def _date(s: str) -> str:
    """Commons dates carry wiki markup after the date ("1836date QS:P571,+1836..."): keep only the date."""
    return re.split(r"date QS|\s*\bQS:", _plain(s))[0].strip()[:30]


def _name(file_title: str) -> str:
    """A readable picture name from its file name."""
    return re.sub(r"\s+", " ", re.sub(r"\.\w{3,4}$", "", file_title[5:] if file_title.lower().startswith("file:") else file_title).replace("_", " ")).strip()


def _artist(s: str) -> str:
    a = _plain(s)
    return re.split(r"\bQS:|\bunknown author", a, flags=re.I)[0].strip(" ,;")[:80]


def search(query: str, n: int = 8) -> list[dict]:
    """Free-licence pictures on Commons: [{file, title, width, height, licence, artist, date, thumb, page}]."""
    r = _api(action="query", generator="search", gsrsearch=query + " filetype:bitmap", gsrnamespace=6, gsrlimit=50,
             prop="imageinfo", iiprop="url|size|extmetadata", iiurlwidth=420,
             iiextmetadatafilter="LicenseShortName|Artist|DateTimeOriginal|ObjectName")
    out = []
    for pg in sorted((r.get("query") or {}).get("pages", {}).values(), key=lambda x: x.get("index", 0)):
        ii = pg["imageinfo"][0]
        m = ii.get("extmetadata", {})
        lic = _plain(m.get("LicenseShortName", {}).get("value", ""))
        if not free_licence(lic) or min(ii["width"], ii["height"]) < 600:
            continue
        out.append({"file": pg["title"], "title": _name(pg["title"]),
                    "width": ii["width"], "height": ii["height"], "licence": lic,
                    "artist": _artist(m.get("Artist", {}).get("value", "")), "date": _date(m.get("DateTimeOriginal", {}).get("value", "")),
                    "thumb": ii.get("thumburl", ""), "page": "https://commons.wikimedia.org/wiki/" + urllib.parse.quote(pg["title"].replace(" ", "_"))})
        if len(out) >= n:
            break
    return out


def fetch(file: str, folder: Path, width: int = 2400) -> tuple[Path, str]:
    """Download a free-licence Commons picture (a File: title) into `folder`. Returns (path, credit line). Refuses anything else."""
    title = file if file.lower().startswith("file:") else "File:" + file
    r = _api(action="query", titles=title, prop="imageinfo", iiprop="size|extmetadata",
             iiextmetadatafilter="LicenseShortName|Artist|DateTimeOriginal|ObjectName")
    pages = list((r.get("query") or {}).get("pages", {}).values())
    if not pages or "imageinfo" not in pages[0]:
        raise ValueError(f"{title} was not found on Wikimedia Commons")
    m = pages[0]["imageinfo"][0].get("extmetadata", {})
    lic = _plain(m.get("LicenseShortName", {}).get("value", ""))
    if not free_licence(lic):
        raise ValueError(f"{title} is “{lic or 'unknown licence'}”; only public domain and CC0 pictures are used")
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    dest = folder / (re.sub(r"[^\w.-]+", "_", title[5:])[:70].rsplit(".", 1)[0] + ".jpg")
    url = "https://commons.wikimedia.org/wiki/Special:FilePath/" + urllib.parse.quote(title[5:]) + f"?width={int(width)}"
    dest.write_bytes(urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=120).read())
    who, when = _artist(m.get("Artist", {}).get("value", "")) or "unknown artist", _date(m.get("DateTimeOriginal", {}).get("value", ""))
    credit = (f"{_name(title)}, {who}" + (f", {when}" if when else "") +
              f". https://commons.wikimedia.org/wiki/{urllib.parse.quote(title.replace(' ', '_'))} ({lic})")
    return dest, credit


def render(spec: dict, out: Path) -> Path:
    """Lay out the cover described by `spec` and save it as a JPEG."""
    from .assemble import make_cover, make_cover_portrait
    layout = spec.get("layout") or ("full" if spec.get("picture") else "plain")
    if layout not in LAYOUTS:
        raise ValueError(f"layout must be one of {LAYOUTS}")
    title, author = spec.get("title") or "Untitled", spec.get("author") or ""
    pic = spec.get("picture")
    if layout != "plain" and not (pic and Path(pic).exists()):
        raise ValueError("this layout needs a picture; give “picture” the path of an image file")
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    focus = tuple(spec.get("focus") or (0.5, 0.5))
    if layout == "framed":
        return make_cover_portrait(out, title, author, pic, colour(spec.get("color", "black")))
    return make_cover(out, title, author, pic if layout == "full" else None, (float(focus[0]), float(focus[1])), float(spec.get("zoom", 1.0)))


def main(argv=None) -> None:
    p = argparse.ArgumentParser(prog="audiobook_gen.covers")
    p.add_argument("cmd", choices=["search", "fetch", "make"])
    p.add_argument("arg"); p.add_argument("n", nargs="?", type=int, default=8)
    p.add_argument("--to", default="pictures"); p.add_argument("--out", default="cover.jpg")
    a = p.parse_args(argv)
    try:
        if a.cmd == "search":
            hits = search(a.arg, a.n)
            for h in hits:
                print(f"{h['file']} | {h['width']}x{h['height']} | {h['licence']} | {h['artist']} | {h['date']}")
            if not hits:
                print("no free-licence pictures found; try other words")
        elif a.cmd == "fetch":
            path, credit = fetch(a.arg, Path(a.to))
            Path(a.to, "credit.txt").write_text(credit + "\n")
            print(f"saved {path}\ncredit: {credit}")
        else:
            spec = json.loads(Path(a.arg).read_text())
            print(f"saved {render(spec, Path(a.out))}")
    except Exception as e:
        print(f"error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
