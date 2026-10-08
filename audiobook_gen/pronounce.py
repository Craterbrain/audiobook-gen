"""Look up real IPA for lexicon words instead of guessing it.

Sources, tried in this order for each word (first hit wins):
  0. Bible books only (bible=True): the finished dictionary data/bible_ipa.json, instantly and offline.
  1. Wiktionary (MediaWiki API, batched; the only step that needs the network): the {{IPA|lang|/…/|a=US}}
     templates on the word's page, from the book's language when the page has that section, else General
     American English.
  2. ipa-dict (open-dict-data, MIT): word -> IPA lists per language, US English from CMUdict, with stress marks.
  3. WikiPron (CUNY-CL, scraped from Wiktionary): per-language word -> phones tables, downloaded once to
     data/wikipron/ and searched offline. Good for plain words, but often without stress marks.
  4. Fandom wikis you name (MediaWiki API too): the page for a character/place, looking for IPA in a
     "pronunciation" line. Wikis differ a lot, so this is a last resort and only for the wikis you list.
offline=True skips the network steps (1 and 4).
IPA is always used as the source wrote it. Anglicised Hebrew and Greek names don't follow spelling rules, so nothing
is derived from spelling; Ford's 1900 book (public domain) only supplies syllables and stress as a hint (ford_hint).
Everything is cached in data/pronunciation_cache.json so repeat runs make no requests. Text is CC BY-SA from
those sites; only the IPA strings are kept here."""
import json
import re
import time
import urllib.parse
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
CACHE = DATA / "pronunciation_cache.json"
UA = "audiobook-gen/0.1 (personal local audiobook tool)"

LANG_NAMES = {"English": "en", "French": "fr", "Italian": "it", "Spanish": "es", "German": "de", "Portuguese": "pt",
              "Dutch": "nl", "Latin": "la", "Hebrew": "he", "Greek": "el", "Ancient Greek": "grc", "Russian": "ru",
              "Polish": "pl", "Welsh": "cy", "Irish": "ga", "Arabic": "ar"}
WIKIPRON = {"en": "eng_latn_us_broad", "fr": "fra_latn_broad", "it": "ita_latn_broad", "es": "spa_latn_la_broad",
            "de": "deu_latn_broad", "pt": "por_latn_bz_broad", "nl": "nld_latn_broad", "he": "heb_hebr_broad",
            "el": "ell_grek_broad"}
IPA_ONLY = set("ˈˌəɪʊɛɔæɑθðʃʒŋɹɾɡɜɝɚʌɒøœɨɐʁɲʎɥ")    # characters that don't occur in ordinary spelling
US = re.compile(r"\b(US|GA|General American|American)\b", re.I)
NOT_US = re.compile(r"\b(UK|RP|Received|Australia|NZ|Ireland|Scotland|Canada|Quebec|Cockney|Southern)\b", re.I)


# ---------- plumbing ----------

def _cache() -> dict:
    try:
        return json.loads(CACHE.read_text())
    except (OSError, ValueError):
        return {}


def _save(c: dict) -> None:
    DATA.mkdir(exist_ok=True)
    CACHE.write_text(json.dumps(c, ensure_ascii=False, indent=0, sort_keys=True))


def _api(host: str, params: dict, tries: int = 3) -> dict:
    url = f"https://{host}/w/api.php?" + urllib.parse.urlencode({**params, "format": "json", "formatversion": 2}) \
        if host.endswith("wikipedia.org") or host.endswith("wiktionary.org") \
        else f"https://{host}/api.php?" + urllib.parse.urlencode({**params, "format": "json", "formatversion": 2})
    for i in range(tries):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": UA}), timeout=25) as r:
                return json.loads(r.read())
        except Exception:
            time.sleep(1.5 * (i + 1))
    return {}


def clean_ipa(s: str) -> str:
    """Bare US-style IPA: no slashes/brackets, syllable dots, tie bars or optional-sound parentheses; syllabic
    consonants spelled out (l̩ -> əl) and British-only vowels mapped to their American equivalents."""
    s = s.strip().strip("/[]")
    s = re.sub(r"[().‿̯̪͜͡]", "", s)
    s = s.replace("l̩", "əl").replace("n̩", "ən").replace("m̩", "əm").replace("ɫ", "l")
    s = s.replace("ɒ", "ɑ").replace("ɜː", "ɜɹ").replace("əʊ", "oʊ").replace("ɛə", "ɛɹ")
    return re.sub(r"\s+", " ", s).strip()


def looks_like_ipa(s: str) -> bool:
    return len(s) >= 2 and any(ch in IPA_ONLY for ch in s)


# ---------- Wiktionary ----------

def parse_wiktionary(text: str, langs: list[str]) -> tuple[str, str, str] | None:
    """(ipa, language code, accent) from a page's wikitext, preferring the languages in order."""
    secs = {}
    parts = re.split(r"^==\s*([^=\n][^=\n]*?)\s*==\s*$", text, flags=re.M)
    for name, body in zip(parts[1::2], parts[2::2]):
        if name in LANG_NAMES:
            secs[LANG_NAMES[name]] = body
    for lang in langs:
        body = secs.get(lang)
        if not body:
            continue
        found = []
        for m in re.finditer(r"\{\{IPA\|" + lang + r"\|([^}]*)\}\}", body):
            for p in m.group(1).split("|"):
                g = re.match(r"([/\[])([^/\]<{]+)[/\]]", p)      # stop at the closing slash: refs/templates may follow
                if g:
                    acc = next((q[2:] for q in m.group(1).split("|") if q.startswith("a=")), "")
                    if looks_like_ipa(g.group(2)) or lang != "en":
                        found.append((clean_ipa(g.group(2)), g.group(1) == "/", acc))
        if not found:
            continue
        found.sort(key=lambda f: (not f[1],                                   # phonemic /…/ before phonetic […]
                                  0 if US.search(f[2]) else 2 if NOT_US.search(f[2]) else 1))
        return found[0][0], lang, found[0][2]
    return None


def wiktionary(terms: list[str], langs: list[str], progress=None) -> dict[str, dict]:
    out: dict[str, dict] = {}
    todo = list(dict.fromkeys(terms))
    STEP = 16      # the API takes at most 50 titles per request, and each word is asked in up to three spellings
    for i in range(0, len(todo), STEP):
        batch = todo[i:i + STEP]
        titles = list(dict.fromkeys(t for w in batch for t in (w, w.lower(), w.capitalize())))
        d = _api("en.wiktionary.org", {"action": "query", "prop": "revisions", "rvprop": "content", "rvslots": "main",
                                      "titles": "|".join(titles), "redirects": 1})
        pages = {p["title"]: (p.get("revisions") or [{}])[0].get("slots", {}).get("main", {}).get("content", "")
                 for p in d.get("query", {}).get("pages", []) if not p.get("missing")}
        redirects = {r["from"]: r["to"] for r in d.get("query", {}).get("redirects", [])}
        for w in batch:
            for lg in langs:       # the book's language first, across every spelling of the title
                for t in (w, w.capitalize(), w.lower()):
                    text = pages.get(redirects.get(t, t), "")
                    hit = parse_wiktionary(text, [lg]) if text else None
                    if hit:
                        out[w] = {"ipa": hit[0], "source": f"wiktionary:{hit[1]}" + (f" ({hit[2]})" if hit[2] else "")}
                        break
                if w in out:
                    break
        if progress:
            progress(min(1, (i + STEP) / len(todo)), f"Wiktionary: {min(len(todo), i + STEP)} of {len(todo)} words")
        time.sleep(0.3)
    return out


# ---------- WikiPron ----------

def _wikipron_table(lang: str) -> dict[str, str]:
    name = WIKIPRON.get(lang)
    if not name:
        return {}
    path = DATA / "wikipron" / f"{name}.tsv"
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        url = f"https://raw.githubusercontent.com/CUNY-CL/wikipron/master/data/scrape/tsv/{name}.tsv"
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": UA}), timeout=60) as r:
                path.write_bytes(r.read())
        except Exception:
            return {}
    table: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        w, _, ph = line.partition("\t")
        if w and ph and w.lower() not in table:
            table[w.lower()] = ph
    return table


def wikipron(terms: list[str], lang: str) -> dict[str, dict]:
    table = _wikipron_table(lang)
    out = {}
    for w in terms:
        ph = table.get(w.lower())
        if ph:      # "ˈ f ɛ ɹ oʊ" -> phones joined; WikiPron separates every phone with a space
            out[w] = {"ipa": clean_ipa(ph.replace(" ", "")), "source": f"wikipron:{WIKIPRON[lang]}"}
    return out


# ---------- ipa-dict (open-dict-data, MIT) ----------

IPADICT = {"en": "en_US", "fr": "fr_FR", "de": "de", "es": "es_ES", "pt": "pt_BR", "nl": "nl"}
_IPADICT: dict[str, dict[str, str]] = {}


def _ipadict_table(lang: str) -> dict[str, str]:
    name = IPADICT.get(lang)
    if not name:
        return {}
    if name not in _IPADICT:
        path = DATA / "ipa-dict" / f"{name}.txt"
        if not path.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
            url = f"https://raw.githubusercontent.com/open-dict-data/ipa-dict/master/data/{name}.txt"
            try:
                with urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": UA}), timeout=60) as r:
                    path.write_bytes(r.read())
            except Exception:
                _IPADICT[name] = {}
                return {}
        table: dict[str, str] = {}
        for line in path.read_text(encoding="utf-8").splitlines():
            w, _, ph = line.partition("\t")
            first = ph.split(",")[0].strip()          # several readings are listed: take the first
            if w and first:
                table.setdefault(w.lower(), clean_ipa(first).replace("ɫ", "l"))   # dark l is an allophone, not a phoneme
        _IPADICT[name] = table
    return _IPADICT[name]


def ipadict(terms: list[str], lang: str) -> dict[str, dict]:
    table = _ipadict_table(lang)
    return {w: {"ipa": table[w.lower()], "source": f"ipa-dict:{IPADICT[lang]}"} for w in terms if w.lower() in table}


# ---------- Bible names: Ford's pronouncing vocabulary (1900, public domain) ----------

FORD_URL = "https://data.labs.loc.gov/digitized-books/data/00006214.txt"
_FORD: dict[str, list[str]] | None = None


def _ford_table() -> dict[str, list[str]]:
    """{lowercase headword letters: syllables} where an accented syllable ends with an apostrophe."""
    global _FORD
    if _FORD is not None:
        return _FORD
    path = DATA / "ford_scripture_names_1900.txt"
    if not path.exists():
        DATA.mkdir(exist_ok=True)
        try:
            with urllib.request.urlopen(urllib.request.Request(FORD_URL, headers={"User-Agent": UA}), timeout=60) as r:
                path.write_bytes(r.read())
        except Exception:
            _FORD = {}
            return _FORD
    text = path.read_text(encoding="utf-8", errors="replace")
    start = text.find("Prefatory Note to the Pronouncing")
    table: dict[str, list[str]] = {}
    for line in text[start:].splitlines():
        t = line.strip().replace("’", "'").replace("‘", "'").replace("´", "'")
        if not re.fullmatch(r"[A-Za-z][A-Za-z'\-–]{2,28}", t) or "'" not in t:
            continue          # OCR junk, page headers and prose lines have no stress mark
        syl = [x for x in re.split(r"(?<=')|[-–]", t) if x]
        word = re.sub(r"[^a-z]", "", t.lower())
        if len(word) >= 3 and word not in table:
            table[word] = syl
    _FORD = table
    return table


def ford_hint(word: str) -> str:
    """Ford's 1900 syllables and stress for a Scripture name, e.g. "Na-than'-a-el" (the apostrophe follows the stressed
    syllable). It is only a hint for whoever writes the IPA: no sounds are derived from it."""
    syl = _ford_table().get(re.sub(r"[^a-z]", "", word.lower()))
    return "-".join(syl) if syl else ""


# ---------- Fandom ----------

def fandom(terms: list[str], wikis: list[str], progress=None) -> dict[str, dict]:
    out = {}
    for n, w in enumerate(terms):
        for host in wikis:
            host = host if "." in host else f"{host}.fandom.com"
            d = _api(host, {"action": "query", "prop": "revisions", "rvprop": "content", "rvslots": "main",
                            "titles": w, "redirects": 1})
            pages = d.get("query", {}).get("pages", [])
            text = (pages[0].get("revisions") or [{}])[0].get("slots", {}).get("main", {}).get("content", "") if pages and not pages[0].get("missing") else ""
            hit = None
            for m in re.finditer(r"(?:pronounc|IPA|pronunciation)[^\n]{0,200}", text, re.I):
                for cand in re.findall(r"[/\[]([^/\[\]\n|]{2,40})[/\]]", m.group()):
                    if looks_like_ipa(cand):
                        hit = clean_ipa(cand)
                        break
                if hit:
                    break
            if hit:
                out[w] = {"ipa": hit, "source": f"fandom:{host.split('.')[0]}"}
                break
            time.sleep(0.2)
        if progress and n % 5 == 0:
            progress(n / max(1, len(terms)), f"Fandom: {n} of {len(terms)} words")
    return out


# ---------- entry point ----------

def _bible_dictionary() -> dict[str, dict]:
    path = DATA / "bible_ipa.json"
    try:
        return json.loads(path.read_text(encoding="utf-8"))["entries"]
    except (OSError, ValueError, KeyError):
        return {}


def lookup(terms: list[str], lang: str = "", wikis: list[str] | None = None, progress=None,
           use_cache: bool = True, bible: bool = False, offline: bool = False, dictionary: bool = True) -> dict[str, dict]:
    """{term: {"ipa", "source"}} for the terms found. `lang` is the book's language code (fr, it, ...)."""
    P = progress or (lambda f, d: None)
    cache = _cache() if use_cache else {}
    pre: dict[str, dict] = {}
    if bible and dictionary:      # the finished Bible dictionary (data/bible_ipa.json) answers instantly, offline
        bd = _bible_dictionary()
        pre = {w: {"ipa": bd[w]["ipa"], "source": f"bible-dict:{bd[w]['source'].split(' ')[0]}"}
               for w in terms if w in bd and bd[w]["source"] not in ("guess", "none") and bd[w]["ipa"]}
        terms = [w for w in terms if w not in pre]
    ckey = lambda w: f"{lang or 'en'}|{','.join(sorted(wikis or []))}|{'bible' if bible else ''}|{w}"
    result = {w: cache[ckey(w)] for w in terms if ckey(w) in cache and cache[ckey(w)]}
    rest = [w for w in terms if ckey(w) not in cache]
    langs = list(dict.fromkeys([lang, "en"] if lang else ["en"]))
    if rest:
        found, left = {}, rest
        for n, lg in enumerate(langs):      # the book's language first, then English
            if left:
                got = {}
                if not offline:      # the only step that needs the network
                    got.update(wiktionary([w for w in left if w not in got], [lg], lambda f, d: P((n + f) * 0.35, d)))
                got.update(ipadict([w for w in left if w not in got], lg))
                got.update(wikipron([w for w in left if w not in got], lg))
                found.update(got)
                left = [w for w in left if w not in got]
        if wikis and left and not offline:
            found.update(fandom(left, wikis, lambda f, d: P(0.75 + 0.25 * f, d)))
        for w in rest:
            if found.get(w, {}).get("source") == "ford1900":
                continue                        # computed locally from the rules: never cached, so rule fixes apply at once
            cache[ckey(w)] = found.get(w, {})   # online results and misses are cached, so they aren't asked again
        if use_cache and not offline:      # offline misses mustn't be remembered as "not on Wiktionary"
            _save(cache)
        result.update(found)
    result.update(pre)
    return result


def fill_lexicon(entries: list[dict], lang: str = "", wikis: list[str] | None = None, progress=None,
                 overwrite: bool = False, bible: bool = False, offline: bool = False) -> dict:
    """Fill empty `ipa` on entries (never ones typed by the user) and mark where it came from in `ipa_src`."""
    todo = [e for e in entries if e.get("source") != "user" and (overwrite or not e.get("ipa"))]
    terms = [e["term"] for e in todo]
    hits = lookup(terms, lang, wikis, progress, bible=bible, offline=offline)
    n = 0
    for e in todo:
        h = hits.get(e["term"])
        if h:
            e["ipa"], e["ipa_src"] = h["ipa"], h["source"]
            n += 1
    return {"asked": len(terms), "found": n, "missing": [t for t in terms if t not in hits]}
