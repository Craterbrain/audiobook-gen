"""Module 2: pronunciation lexicon.

Detection asks the TTS front end itself what it can't pronounce: a word is flagged when Kokoro's
G2P (misaki) has no dictionary entry for it and must guess. That finds invented/foreign names,
places and archaic words alike, with none of the noise a pure frequency cut-off gives
(ordinary words such as "murmured" are fine and stay out). spaCy NER only labels what kind of
thing a word is. Each entry carries Kokoro's current guess, an optional language hint, and an
IPA/respelling that is either auto-suggested (espeak, in that language) or typed by the user.
"""
import json
import re
from collections import Counter
from pathlib import Path

LETTER = "A-Za-zÀ-ÖØ-öø-ÿ"
WORD_RE = re.compile(rf"[{LETTER}]+(?:[’'-][{LETTER}]+)*")
POSS_RE = re.compile(r"[’']s$", re.I)
LABEL_KIND = {"PERSON": "name", "GPE": "place", "LOC": "place", "FAC": "place", "ORG": "name",
              "NORP": "demonym"}
ESPEAK_LANG = {"el": "el", "ar": "ar", "fr": "fr-fr", "it": "it", "de": "de", "es": "es", "la": "la", "pt": "pt-br",
               "ru": "ru", "he": "he", "nl": "nl", "pl": "pl", "gb": "en-gb", "en": "en-us"}      # plain "en" would be espeak's British default

_G2P = None
_ESPEAK: dict = {}
_VOCAB = None


def _g2p():
    global _G2P
    if _G2P is None:
        from misaki import en, espeak
        _G2P = en.G2P(trf=False, british=False, fallback=espeak.EspeakFallback(british=False))
    return _G2P


def kokoro_vocab() -> set:
    """Phoneme symbols Kokoro understands (from its model config)."""
    global _VOCAB
    if _VOCAB is None:
        import huggingface_hub
        _VOCAB = set(json.load(open(huggingface_hub.hf_hub_download("hexgrad/Kokoro-82M", "config.json")))["vocab"])
    return _VOCAB


def guess_ipa(word: str) -> str:
    """What Kokoro will say for `word` today."""
    return _g2p()(word)[0]


def suggest_ipa(word: str, lang: str) -> str:
    """IPA from espeak in `lang` (e.g. 'fr'), restricted to symbols Kokoro knows."""
    code = ESPEAK_LANG.get(lang, lang)
    if code not in _ESPEAK:
        from misaki import espeak
        _ESPEAK[code] = espeak.EspeakG2P(language=code)
    ps = _ESPEAK[code](word)[0]
    vocab = kokoro_vocab()
    return "".join(c for c in ps if c in vocab)


SKIP = {"tis", "twas", "twere", "twill", "ere", "oft", "nay", "aye"}


def _stems(w: str):
    """The word plus plausible un-inflected forms (towards->toward, inquired->inquire, cities->city)."""
    yield w
    for suf, rep in (("ies", "y"), ("ied", "y"), ("ing", ""), ("ing", "e"), ("ed", ""), ("ed", "e"),
                     ("es", ""), ("s", ""), ("d", ""), ("ly", ""), ("er", ""), ("est", ""), ("st", ""), ("eth", ""),
                     ("eth", "e"), ("th", "")):
        if w.endswith(suf) and len(w) - len(suf) >= 3:
            yield w[:-len(suf)] + rep


def _is_known(word: str) -> bool:
    lx = _g2p().lexicon
    w = word.replace("’", "'").lower()
    if w in SKIP:
        return True
    ok = lambda x: lx.is_known(x, None) or lx.is_known(x.capitalize(), None)
    expanded = re.sub(r"'(d|st|dst|t)$", lambda m: {"d": "ed", "st": "est", "dst": "edst", "t": "t"}[m.group(1)], w)
    if expanded != w and ok(expanded):
        return True  # Elizabethan elision: promis'd -> promised
    parts = [p for p in re.split(r"[-']", w) if p]  # good-bye, o'clock, don't
    if "'" in w and len(parts) == 2 and len(parts[0]) == 1 and len(parts[1]) > 2 and word[:1].isupper():
        return False        # a name like M'ling or O'Niel: the one-letter piece and the ordinary word beside it do not make it a known word
    return ok(w) or all(any(ok(s) for s in _stems(p)) or len(p) <= 2 for p in parts)


def _base(word: str) -> str:
    return POSS_RE.sub("", word)


WF_LANGS = ["fr", "it", "es", "de", "pt", "nl", "pl", "sv", "ca"]  # Latin-script languages wordfreq covers


def detect_wordfreq(word: str, is_name: bool) -> tuple[str, float]:
    """Language whose word-frequency list knows `word` clearly better than English's does."""
    from wordfreq import zipf_frequency
    w = word.lower()
    en = zipf_frequency(w, "en")
    best, bz = "", 0.0
    for l in WF_LANGS:
        z = zipf_frequency(w, l)
        if z > bz:
            best, bz = l, z
    if best and len(w) >= 4 and bz >= 2.5 and bz >= en + (1.0 if is_name else 0.5):
        return best, bz
    return "", 0.0


def _sentence_with(texts: list[str], word: str, width: int = 160) -> str:
    for t in texts:
        m = re.search(rf"(?<![{LETTER}]){re.escape(word)}", t)
        if m:
            return re.sub(r"\s+", " ", t[max(0, m.start() - width):m.end() + width]).strip()
    return word


VOTE_SHARE = 0.15                  # the top language must have this share of the not-well-known names (Monte Cristo: 0.23; English books: <= 0.11)
ENGLISH_PHRASE_ZIPF = 3.2          # a phrase made only of words this common in English is English ("Clock Tower", "Hyde Park")
BOOK_LANGS = {"fr", "it", "es", "de", "pt", "nl"}  # Latin-script languages whose names espeak can voice


def _english_phrase(term: str) -> bool:
    """Every word is an everyday English word (not a name in disguise): leave the phrase to the English reader."""
    from wordfreq import zipf_frequency
    words = [w.lower() for w in WORD_RE.findall(term.replace("’s", "").replace("'s", "")) if w.lower() not in CONNECTORS]
    return bool(words) and all(zipf_frequency(w, "en") >= ENGLISH_PHRASE_ZIPF for w in words)


def drop_guessed_language(entries: list[dict]) -> int:
    """Undo a book language that was only a weak guess (lang_conf 0.5): clear the language and any IPA/respelling espeak made from it.
    Things you typed (source: user) and IPA that came from a dictionary are kept. Returns how many entries were reset."""
    n = 0
    for e in entries:
        if e.get("lang_src") == "book" and e.get("lang_conf") == 0.5 and e.get("source") != "user":
            e.update(lang="", lang_src="", lang_conf=0.0)
            if not e.get("ipa_src") and e.get("source") != "user":
                e["ipa"] = ""
                if e.get("respell_src") in (None, "", "auto"):
                    e["respell"], e["respell_src"] = "", ""
            n += 1
    return n


ORIGIN_KINDS = ("name", "place", "phrase", "demonym", "foreign")
ORIGIN_CONTEXTS = 3                # sentences asked about per name: early, middle and late mentions (a general who once served in Mexico is not Spanish)
NONUS_MIN = 0.6                    # a tradition other than American English needs at least this probability (averaged over the sentences) ...
NONUS_MIN_KNOWN = 0.9              # ... and this much when the name is in the US English dictionary (Meade, Halleck, Longstreet are, Contreras too)
GB_MIN = 0.8                       # British English is over-guessed for Anglo-looking American surnames, so it needs more


_NO_SPLIT = {"mr", "mrs", "ms", "dr", "gen", "col", "capt", "lieut", "lt", "maj", "brig", "sgt", "st", "mt", "ft", "vs", "no", "jr", "sr", "prof",
             "hon", "rev", "messrs", "gov", "sen", "rep", "esq", "etc", "viz", "wm", "geo", "chas", "jno", "thos"}
_SENT_END = re.compile(r"(?:(?<=[.!?])|(?<=[.!?][\"”’']))\s+(?=[\"“‘'(]?[A-Z0-9])")


def _split_sentences(par: str) -> list[str]:
    """Sentences of one paragraph. A full stop after Mr., Gen., Lieut. or a single initial does not end one."""
    out, start = [], 0
    for m in _SENT_END.finditer(par):
        before = par[start:m.start()]
        last = re.findall(rf"[{LETTER}]+\.?[\"”’']?$", before)
        word = (last[0].rstrip(".\"”’'").lower() if last else "")
        if before.rstrip().endswith(".") and (word in _NO_SPLIT or len(word) == 1):
            continue
        out.append(par[start:m.start()].strip())
        start = m.end()
    out.append(par[start:].strip())
    return [x for x in out if x]


def sentence_context(par: str, pos: int, before: int = 1, after: int = 1, limit: int = 900) -> str:
    """The sentence of `par` that holds position `pos`, with `before` sentences ahead of it and `after` behind it (so three by default).
    The same place always gives the same text."""
    sents, at, idx = _split_sentences(par), 0, 0
    for i, sent in enumerate(sents):
        j = par.find(sent, at)
        if j < 0:
            continue
        if j <= pos < j + len(sent) + 1:
            idx = i
            break
        at = j + len(sent)
    return " ".join(sents[max(0, idx - before): idx + after + 1])[:limit]


def _sentences_with(texts: list[str], word: str, k: int = ORIGIN_CONTEXTS, limit: int = 900) -> list[str]:
    """Up to k passages spread through the book (first, middle, last mention). A passage is the sentence holding the word and the sentence
    either side of it, even across a paragraph break: three sentences, the same text every time."""
    import bisect
    hits = []                                       # (sentences of that chapter, index of the one holding the word)
    for t in texts:
        found = [m.start() for m in re.finditer(rf"(?<![{LETTER}]){re.escape(word)}", t)]
        if not found:
            continue
        sents, starts, at = [], [], 0
        for par in re.split(r"\n\s*\n", t):
            j = t.find(par, at)
            at = j + len(par)
            cur = j
            for sent in _split_sentences(par):
                i = t.find(sent, cur)
                sents.append(sent); starts.append(i); cur = i + len(sent)
        last = -1
        for pos in found:
            i = bisect.bisect_right(starts, pos) - 1
            if i != last and i >= 0:
                hits.append((sents, i)); last = i
        if len(hits) >= 600:
            break
    if not hits:
        return [word]
    idx = sorted({round(i * (len(hits) - 1) / max(1, k - 1)) for i in range(k)}) if len(hits) > 1 else [0]
    return [" ".join(hits[i][0][max(0, hits[i][1] - 1): hits[i][1] + 2])[:limit] for i in idx]


def _in_us_dictionary(term: str) -> bool:
    from .pronounce import _ipadict_table
    table = _ipadict_table("en")
    words = [w.lower() for w in WORD_RE.findall(term.replace("’s", "").replace("'s", ""))]
    return bool(words) and all(w in table for w in words)


def assign_origins(entries: list[dict], texts: list[str], judge, title: str = "", author: str = "", tick=None) -> dict:
    """Ask the model, name by name, where each belongs (American English, British English, Spanish, French ...), from a few sentences it
    appears in and the book it is from. Replaces the guessed book-wide language. Entries you edited (source: user) and the very well known
    names are left alone. Entries that end up non-English get espeak IPA in their language; the rest lose any IPA/respelling made from a
    wrong language. Anything but American English must be convincing (see NONUS_MIN...). Returns {"asked", "changed", "by_setting"}."""
    drop_numerals(entries)                                      # chapter numbers (XVII) are not names, whatever tradition they look like
    todo = [e for e in entries if e.get("kind") in ORIGIN_KINDS and e.get("source") != "user" and not e.get("known")]
    by: Counter = Counter()
    changed = 0
    passages = [_sentences_with(texts, e["term"]) for e in todo]
    requests = [(e["term"], p) for e, ps in zip(todo, passages) for p in ps]
    answers = judge.settings(requests, title, author,
                             (lambda d, t: tick(d / max(1, t), f"Asking where each name is from ({int(d / max(1, t) * len(todo))}/{len(todo)})")) if tick else None)
    pos = 0
    for k, e in enumerate(todo):
        n = len(passages[k])
        ps, pos = answers[pos:pos + n], pos + n
        avg = {c: sum(p[c] for p in ps) / len(ps) for c in ps[0]}
        code, conf = max(avg.items(), key=lambda kv: kv[1])
        need = NONUS_MIN_KNOWN if _in_us_dictionary(e["term"]) else NONUS_MIN
        if code == "gb":
            need = max(need, GB_MIN)
        if code in ("us", "other") or conf < need:
            code = "us"
        by[code] += 1
        lang = "" if code == "us" else code
        old = e.get("lang") or ""
        if lang != old:
            changed += 1
        if e.get("lang_src") == "wordfreq" and old and lang == "":
            continue                                          # a word list says it is foreign and the model is not convinced otherwise: keep that
        e.update(lang=lang, lang_src="model" if lang else "", lang_conf=round(conf, 2), setting=code)
        if lang != old:                                       # the sound made from the old language no longer applies
            if not e.get("ipa_src") and e.get("source") != "user":
                e["ipa"] = ""
            if e.get("respell_src") in (None, "", "auto") and e.get("source") != "user":
                e["respell"], e["respell_src"] = "", ""
    todo2 = [e for e in todo if e.get("lang") and not e.get("ipa")]
    for e in todo2:
        e["source"] = "auto"
    autofill(todo2)
    return {"asked": len(todo), "changed": changed, "by_setting": dict(by)}


def redo_origins(work, judge, progress=None) -> dict:
    """assign_origins for a book that already has a lexicon (the old one is kept as lexicon.before_origins.json)."""
    import shutil
    work = Path(work)
    data = json.loads((work / "chapters.json").read_text())
    entries = json.loads((work / "lexicon.json").read_text())
    shutil.copy(work / "lexicon.json", work / "lexicon.before_origins.json")
    rep = assign_origins(entries, [c["text"] for c in data["chapters"]], judge, data.get("title", ""), data.get("author") or "", progress)
    (work / "lexicon.json").write_text(json.dumps(entries, indent=2, ensure_ascii=False))
    return rep


def detect_languages(entries: dict, texts: list[str], hint: str = "", judge=None,
                     title: str = "", author: str = "", tick=None) -> dict:
    """Assume English; move words out of it only on evidence.

    1. Word-frequency lists: a language that knows the word clearly better than English's list does.
    2. The book's language: a user hint, else the LLM judge reading the title and author (it is reliable
       at that: Monte Cristo -> French, Macbeth -> English), else word-frequency votes. An English
       verdict (>= 0.85) locks the book to English; only fr/it/es/de/pt/nl are accepted otherwise.
    3. Names, places and accented words that match nothing take the book's language; plain lowercase
       words must also pass the judge's debiased "ordinary English, or foreign?" check. With no book
       language everything unmatched stays English.
    Well-known names (Edmond, Napoleon), words with apostrophes (promis'd) and anything the user set
    are left alone."""
    stats = {"wordfreq": 0, "book": 0, "english": 0}
    pending = []
    for e in entries.values():
        if e.get("source") == "user" or e.get("lang") or e.get("known"):
            continue
        is_name = e["kind"] in ("name", "place", "demonym", "phrase")
        lang, conf = detect_wordfreq(e["term"], is_name) if e["kind"] != "phrase" else ("", 0.0)
        if lang:
            e.update(lang=lang, lang_src="wordfreq", lang_conf=round(conf / 7, 2))
            stats["wordfreq"] += 1
        elif "'" not in e["term"] and "’" not in e["term"] or e["kind"] == "phrase":
            pending.append(e)
    votes = Counter(e["lang"] for e in entries.values() if e.get("lang_src") == "wordfreq")

    book, how = hint, "hint"
    if not book and judge is not None and title:
        code, p = judge.book_language(title, author)
        stats["book_llm"] = f"{code} {p:.2f}"
        if p >= 0.85:
            book, how = (code if code in BOOK_LANGS else ""), "llm"
            if code == "en":
                votes = Counter()  # English-first: stray loanwords don't make a French book
        else:
            how = "votes"
    elif not book:
        how = "votes"
    if not book and how == "votes" and votes:
        top, n = votes.most_common(1)[0]
        unknown = sum(1 for e in entries.values() if not e.get("known"))
        # A few foreign-looking names do not make a book foreign (War of the Worlds got French from 5 of 232 names, and its London
        # parks were then read by espeak's French voice, which says English words in British phonemes). A foreign book has many.
        if n >= 3 and n / sum(votes.values()) >= 0.5 and n >= VOTE_SHARE * max(1, unknown):
            book = top
    for e in entries.values():  # a lone hit in some other language is more likely coincidence
        if e.get("lang_src") == "wordfreq" and e["lang"] != book and votes[e["lang"]] < 2 \
                and not re.search(r"[^\x00-\x7f]", e["term"]):
            e.update(lang="", lang_src="", lang_conf=0.0)
            stats["wordfreq"] -= 1
    for k, e in enumerate(pending):
        if tick and judge is not None and book and k % 5 == 0:
            tick(k / max(1, len(pending)), f"Asking the small model about {len(pending)} unmatched words ({k}/{len(pending)})")
        if not book:
            continue
        diacritic = bool(re.search(r"[^\x00-\x7f]", e["term"]))
        if e["kind"] == "phrase" and not diacritic and _english_phrase(e["term"]):
            stats["english"] += 1
            continue
        if e["kind"] in ("name", "place", "demonym", "foreign", "phrase") or diacritic:
            e.update(lang=book, lang_src="book", lang_conf=1.0 if how != "votes" else 0.5)
            stats["book"] += 1
        elif judge is not None and judge.is_foreign(e["term"], _sentence_with(texts, e["term"])) >= 0.6:
            e.update(lang=book, lang_src="llm", lang_conf=0.6)
            stats["book"] += 1
        else:
            stats["english"] += 1
    stats["book_language"] = book or "en"
    entries["__stats__"] = stats
    return entries


_HEADING_START = re.compile(r"^\s*(?:(?:chapter|book|part|section|letter|canto|act|scene)\b|[IVXLC]+\s*[.:]|\d{1,3}\s*[.:]\s*[A-Z])", re.I)


def is_heading_line(par: str, title: str = "") -> bool:
    """A short line that is the chapter's own heading: its title again, or "XVII. THE THUNDER CHILD.", "Chapter 4", "BOOK ONE ..."."""
    line = " ".join(par.split())
    if not line or len(line) > 100:
        return False
    norm = lambda x: re.sub(r"[^a-z0-9]+", " ", x.lower()).strip()
    if title and norm(title) and (norm(line) == norm(title) or norm(line) in norm(title) or norm(title) in norm(line)):
        return True
    return bool(_HEADING_START.match(line)) and not re.search(r"[a-z]{3,}[,;]", line) and line.count(".") <= 2


def chapter_body(ch: dict) -> str:
    """The chapter's text without its heading lines at the start. The headings are read by the titles step (numbers spelled out), so
    "XVII" and the like are never lexicon material: the headings are taken out first and the word scan runs on what is left."""
    pars = re.split(r"\n\s*\n", ch["text"])
    k = 0
    while k < min(3, len(pars)) and is_heading_line(pars[k], ch.get("title", "")):
        k += 1
    return "\n\n".join(pars[k:])


def build_lexicon(work: Path, lang: str = "", judge=None, title: str = "", author: str = "", progress=None) -> list[dict]:
    """Scan chapters.json -> lexicon.json (sorted by count). User-edited entries are preserved.
    progress(fraction 0..1, description) reports each stage."""
    P = progress or (lambda f, d: None)
    P(0.0, "Loading language tools")
    import spacy
    from wordfreq import zipf_frequency

    nlp = spacy.load("en_core_web_sm", disable=["parser", "lemmatizer"])
    nlp.max_length = 5_000_000
    data = json.loads((work / "chapters.json").read_text())

    stat: dict[str, dict] = {}
    ents: list[tuple[str, str]] = []  # (label, text) from spaCy, reused for place-name phrases
    n_ch = len(data["chapters"])
    for ci, ch in enumerate(data["chapters"]):
        P(0.02 + 0.30 * ci / n_ch, f"Reading chapter {ci + 1} of {n_ch} (finding names)")
        text = chapter_body(ch)                                  # headings first: they are not scanned for names
        if ch.get("format") == "play":  # speaker labels aren't spoken
            text = re.sub(r"^[A-Z][A-Z .’'-]+\.$", "", text, flags=re.M)
        for m in WORD_RE.finditer(text):
            w = _base(m.group())
            if len(w) < 3 or is_numeral(w):
                continue
            st = stat.setdefault(w.lower(), {"surface": Counter(), "mid": 0, "init": 0, "lower": 0,
                                             "labels": Counter()})
            st["surface"][w] += 1
            if w[0].islower():
                st["lower"] += 1
            else:
                before = text[:m.start()].rstrip()
                prev = before[-1:]
                after_abbr = re.search(r"\b(?:M|MM|Mme|Mlle|Mr|Mrs|Dr|St|Capt|Col)\.$", before)
                st["init" if (not prev or prev in ".!?\"“”‘\n") and not after_abbr else "mid"] += 1
        for ent in nlp(text).ents:
            ents.append((ent.label_, ent.text))
            kind = LABEL_KIND.get(ent.label_)
            if kind:
                for m in WORD_RE.finditer(ent.text):
                    key = _base(m.group()).lower()
                    if key in stat:
                        stat[key]["labels"][kind] += 1

    entries: dict[str, dict] = {}
    for n, (key, st) in enumerate(stat.items()):
        if n % 50 == 0:
            P(0.33 + 0.27 * n / max(1, len(stat)), f"Checking {len(stat)} different words against Kokoro's dictionary")
        surface = next((w for w, _ in st["surface"].most_common() if not w.isupper()), None) \
            or st["surface"].most_common(1)[0][0].capitalize()
        known = _is_known(surface)
        if known and not (st["labels"].get("name") or st["labels"].get("place")) :
            continue  # Kokoro already has this word; nothing to fix
        capital = st["mid"] >= 1 and st["mid"] >= st["lower"]  # capitalised away from sentence starts
        non_ascii = bool(re.search(r"[^\x00-\x7f]", surface))
        if capital:
            kind = st["labels"].most_common(1)[0][0] if st["labels"] else "name"
        else:
            kind = "foreign" if non_ascii else "word"
        if known and not (capital and kind in ("name", "place") and st["lower"] == 0
                          and zipf_frequency(key, "en") < 4.5):
            continue  # common English words that merely happen to be capitalised (North, Justice, Fort)
        entries[key] = {"term": surface, "count": sum(st["surface"].values()), "kind": kind, "known": known,
                        "lang": "", "lang_src": "", "lang_conf": 0.0,
                        "guess": guess_ipa(surface), "ipa": "", "respell": "", "source": "auto",
                        "zipf": round(zipf_frequency(key, "en"), 2)}

    path = work / "lexicon.json"
    if path.exists():  # keep what the user decided
        for old in json.loads(path.read_text()):
            k = old["term"].lower()
            if old.get("source") == "user":
                if k in entries:
                    entries[k].update({f: old[f] for f in ("ipa", "respell", "lang") if f in old}, source="user")
                else:
                    entries[k] = old  # hand-added term
    texts = [chapter_body(c) for c in data["chapters"]]
    P(0.62, "Finding place-name phrases")
    add_phrases(entries, stat, texts, ents)
    P(0.66, "Working out the language of each word" + (" (asking the small model)" if judge else ""))
    detect_languages(entries, texts, lang, judge, title or data.get("title", ""), author or data.get("author", ""),
                     tick=lambda f, d: P(0.66 + 0.29 * f, d))
    stats = entries.pop("__stats__")
    entries = {k: e for k, e in entries.items() if not (e["kind"] == "phrase" and not e["lang"])}  # no use in English
    print(f"[lexicon] {stats}")
    P(0.96, "Suggesting pronunciations")
    autofill(entries.values())
    P(1.0, "Saving")
    out = sorted(entries.values(), key=lambda e: (-e["count"], e["term"].lower()))
    path.write_text(json.dumps(out, indent=2, ensure_ascii=False))
    return out


def f5_text(respell: str, surface: str = "") -> str:
    """What a plain-text engine should be given. F5 reads letter by letter: hyphens split a name into separate
    words ("Nuh-than-yel" was heard as "no, then yell") and capitals are spelled out ("dahn-GLAHR" -> "Don G L H R").
    So syllable breaks and stress capitals are dropped: "Nuh-THAN-yel" -> "Nuhthanyel"."""
    out = " ".join(re.sub(r"[-·‧]", "", w).lower() for w in respell.split())
    return out[:1].upper() + out[1:] if surface[:1].isupper() else out


def auto_respell(e: dict, use_guess: bool = False) -> str:
    """Respelling for plain-text engines, made from the entry's IPA (typed, seeded, or from espeak in its
    language). With no IPA, F5 keeps reading the word's own spelling, unless use_guess is set: then the
    spelling is made from `guess` (Kokoro's current pronunciation), which is what the Lexicon tab's
    "Auto-spell for F5" button does."""
    ps = e.get("ipa") or (e.get("guess") if use_guess else "")
    return f5_text(ipa_to_respell(ps, f5=True)) if ps else ""


def fill_respell(entries, refresh: bool = False, use_guess: bool = False) -> int:
    """Auto-spell: give entries without a respelling one made from their IPA (or the G2P guess).
    A respelling the user typed is never touched; with refresh=True, ones made automatically earlier are
    regenerated (use after the IPA changed)."""
    n = 0
    for e in entries:
        mine = e.get("respell_src") in ("auto", "auto-guess") or (not e.get("respell_src") and e.get("source") != "user")
        if e.get("respell") and not (refresh and mine):
            continue
        r = auto_respell(e, use_guess)
        if r and r != e.get("respell"):
            e["respell"], e["respell_src"] = r, "auto-guess" if (use_guess and not e.get("ipa")) else "auto"
            n += 1
    return n


def autofill(entries, include_known: bool = False) -> int:
    """Fill empty IPA for entries that have a language, using espeak in that language.
    Well-known names (Edmond, Napoleon) are skipped unless include_known: Kokoro's reading may be fine."""
    n = 0
    for e in entries:
        if e.get("lang") and not e.get("ipa") and e.get("source") != "user" \
                and (include_known or not e.get("known")):
            try:
                e["ipa"] = suggest_ipa(e["term"], e["lang"])
                n += bool(e["ipa"])
            except Exception:  # unknown language code / espeak failure: leave blank
                pass
    fill_respell(entries)
    return n


CONNECTORS = {"de", "du", "des", "la", "le", "les", "of", "von", "van", "der", "di", "del", "della", "da",
              "dos", "y", "el", "al", "the", "d", "l"}
PLACE_LABELS = {"GPE", "LOC", "FAC"}


LEADING_STOP = frozenset("""from the in on at to and but then when as by for with after before of if so or nor yet while until upon into over under
through about near between among during against along across toward towards within without dear dearest my our your his her their this that these
those there here now thus once all some""".split())      # a capital only because it starts a sentence ("From La Grange"), never part of a name


def trim_phrase(term: str) -> str:
    """The phrase without sentence-initial function words at its front ("From La Grange" -> "La Grange"); "" if nothing is left of it."""
    words = term.split()
    while words and words[0].lower().strip("’'") in LEADING_STOP:
        words = words[1:]
    return " ".join(words) if len(words) >= 2 else ""


def repair_phrases(entries: list[dict]) -> int:
    """Fix phrase entries made before the leading-word trim: rename them (merging with an entry of the new name), drop what is left of one word."""
    by = {e["term"].lower(): e for e in entries}
    gone, n = [], 0
    for e in entries:
        if e.get("kind") != "phrase" or e.get("source") == "user":
            continue
        new = trim_phrase(e["term"])
        if new == e["term"]:
            continue
        n += 1
        other = by.get(new.lower()) if new else None
        if other is not None and other is not e:
            other["count"] = other.get("count", 0) + e.get("count", 0)
            gone.append(e)
        elif new:
            e["term"] = new
        else:
            gone.append(e)
    for e in gone:
        entries.remove(e)
    return n


ROMAN_NUMERAL = re.compile(r"M{0,3}(?:CM|CD|D?C{0,3})(?:XC|XL|L?X{0,3})(?:IX|IV|V?I{0,3})", re.I)


def is_numeral(term: str) -> bool:
    """A Roman numeral (XVII, Xii, IV): chapter and section numbers. The chapter titles are read as numbers by titles.spoken, so they are not
    lexicon material. Two mixed-case letters ("Li", "Xi", "Mi") are left alone: they are surnames."""
    if not term or not ROMAN_NUMERAL.fullmatch(term) or not term.isalpha():
        return False
    if not term.isupper():
        if len(term) < 3:
            return False
        from .pronounce import _ipadict_table
        if term.lower() in _ipadict_table("en"):                       # Dix, Mix, Mil: a surname or an English word that happens to be valid
            return False
    return True


def drop_numerals(entries: list[dict]) -> int:
    """Remove Roman-numeral entries (not ones you wrote yourself) from a lexicon made before they were skipped. Returns how many."""
    gone = [e for e in entries if is_numeral(e.get("term", "")) and e.get("source") != "user"]
    for e in gone:
        entries.remove(e)
    return len(gone)


def _phrases(text: str):
    """Capitalised word runs, allowing lowercase connectors inside (Allées de Meilhan, Château d'If)."""
    toks = list(WORD_RE.finditer(text))
    i = 0
    while i < len(toks):
        if not toks[i].group()[0].isupper():
            i += 1
            continue
        j, end = i, i
        while j + 1 < len(toks):
            gap = text[toks[j].end():toks[j + 1].start()]
            nxt = toks[j + 1].group()
            if gap.strip(" \n") or len(gap) > 2:
                break
            if nxt[0].isupper() or re.match(r"^(?:d|l|de)['’][A-Z]", nxt):
                j += 1
                end = j
            elif nxt.lower() in CONNECTORS and j + 2 < len(toks) and toks[j + 2].group()[0].isupper():
                j += 1  # connector only counts if a capital follows
            else:
                break
        if end > i:
            first = i
            while first < end and toks[first].group().lower().strip("’'") in LEADING_STOP:
                first += 1
            if end > first:
                yield text[toks[first].start():toks[end].end()]
        i = end + 1


def add_phrases(entries: dict, stat: dict, texts: list[str], ents: list[tuple[str, str]]) -> None:
    """Multi-word names, especially places (Fort Saint Nicholas, Palais de Justice, Château d'If), get an
    entry of their own so they are pronounced as a whole: liaison and elision follow the phrase."""
    from collections import Counter
    from wordfreq import zipf_frequency

    found: Counter = Counter()
    labelled: set = set()
    for label, etext in ents:
        if label in PLACE_LABELS and len(etext.split()) >= 2:
            labelled.add(re.sub(r"\s+", " ", re.sub(r"^(?:the|The)\s+", "", etext)).strip())
    for t in texts:
        for ph in _phrases(t):
            found[re.sub(r"\s+", " ", ph)] += 1
    foreign_conn = CONNECTORS - {"of", "the", "and", "to"}
    for ph, n in found.items():
        toks = [w.lower().replace("’", "'") for w in WORD_RE.findall(ph)]
        conns = {t.split("'")[0] for t in toks if t in CONNECTORS or re.match(r"^(d|l|de)'", t)}
        content = [_base(w).lower() for w in WORD_RE.findall(ph) if w.lower() not in CONNECTORS
                   and not re.match(r"^(d|l|de)['’]$", w.lower())]
        if len(content) < 2 and not (conns & foreign_conn) and not any(t.startswith(("d'", "l'")) for t in toks):
            continue
        english_conn = bool(conns & {"of", "the", "and", "to"})
        flagged = sum(k in entries for k in content)
        place = ph in labelled and not english_conn
        foreign_form = bool(conns & foreign_conn or any(t.startswith(("d'", "l'")) for t in toks)) and flagged >= 1
        if not (place or foreign_form):
            continue  # persons are covered word by word; English-built phrases (Island of Elba) too
        entries[ph.lower()] = {"term": ph, "count": n, "kind": "phrase", "known": False, "lang": "", "lang_src": "",
                               "lang_conf": 0.0, "guess": guess_ipa(ph), "ipa": "", "respell": "",
                               "source": "auto", "zipf": 0.0}


_V = {"a": "ah", "ɑ": "ah", "æ": "a", "ɛ": "eh", "e": "ay", "i": "ee", "ɪ": "ih", "ɔ": "aw", "o": "oh",
      "u": "oo", "ʊ": "uu", "y": "ew", "ø": "er", "œ": "er", "ə": "uh", "ɜ": "er", "ʌ": "uh", "ɐ": "uh",
      "ɒ": "o", "ᵻ": "ih", "ɚ": "er"}
_C = {"ʁ": "r", "ɹ": "r", "ɾ": "r", "ʃ": "sh", "ʒ": "zh", "ŋ": "ng", "θ": "th", "ð": "dh", "j": "y", "ɡ": "g",
      "x": "kh", "ɲ": "ny", "ʎ": "ly", "ç": "h", "ɣ": "g", "ʧ": "ch", "ʤ": "j", "w": "w", "ɫ": "l"}


_MISAKI = {"A": "eɪ", "I": "aɪ", "O": "oʊ", "Q": "əʊ", "W": "aʊ", "Y": "ɔɪ", "ʤ": "dʒ", "ʧ": "tʃ",
           "ᵻ": "ɪ", "ᵊ": "ə", "T": "t", "ɾ": "t"}


def misaki_to_std(ps: str) -> str:
    """Kokoro's own phoneme letters (A=eɪ, I=aɪ, O=oʊ, W=aʊ, Y=ɔɪ, T=flapped t...) -> ordinary IPA."""
    return "".join(_MISAKI.get(c, c) for c in ps)


_DIPH = {"eɪ": "ay", "aɪ": "eye", "oʊ": "oh", "əʊ": "oh", "aʊ": "ow", "ɔɪ": "oy", "ɜɹ": "er", "ɑɹ": "ar", "ɔɹ": "or",
         "ɛɹ": "air", "ɪɹ": "eer", "ʊɹ": "oor", "ɚ": "er", "ɐɹ": "ar"}
_ONSETS = {"pl", "bl", "kl", "gl", "fl", "sl", "pr", "br", "tr", "dr", "kr", "gr", "fr", "θr", "ʃr", "sp", "st", "sk",
           "sm", "sn", "sw", "tw", "kw", "dw", "str", "spr", "skr", "spl", "skw"}


# F5 reads letters, and "uh" comes out as a long "oo" (Nuhthanyul -> "nootHANyool"). The spelling that tested
# right for Nathanael writes the schwa as "a" (nathanyal), so the F5 style does the same.
_F5_VOWELS = {"ə": "a", "ɐ": "a", "ʌ": "u", "ᵻ": "i", "ɜ": "er"}


def _respell_word(word: str, f5: bool = False) -> str:
    units, stress_next, i = [], False, 0   # unit = (kind V/C, text, stressed)
    while i < len(word):
        c, two = word[i], word[i:i + 2]
        if c == "ˈ":
            stress_next = True
        elif c in "ˌːˑ͡‿.":
            pass
        elif c == "̃":
            if units and units[-1][0] == "V":
                units.append(("C", "n", False))
        elif two in _DIPH:
            units.append(("V", _DIPH[two], stress_next)); stress_next = False; i += 1
        elif c == "ɚ":
            units.append(("V", "er", stress_next)); stress_next = False
        elif c in "tdTD" and word[i + 1:i + 2] in ("ʃ", "ʒ"):
            units.append(("C", "ch" if c == "t" else "j", False)); i += 1
        elif c in _V:
            units.append(("V", (_F5_VOWELS.get(c) if f5 else None) or _V[c], stress_next)); stress_next = False
        else:
            units.append(("C", _C.get(c, c), False))
        i += 1
    # syllables: one vowel each; consonants between vowels go to the next syllable when they form a
    # legal onset (dr, st, pl...), otherwise the first closes the previous syllable
    sylls, cur, stressed, seen_v, k = [], "", False, False, 0
    while k < len(units):
        kind, txt, st = units[k]
        if kind == "V":
            if seen_v:
                sylls.append((cur, stressed)); cur, stressed = "", False
            cur += txt; seen_v = True; stressed = stressed or st
            k += 1
            continue
        run = 0
        while k + run < len(units) and units[k + run][0] == "C":
            run += 1
        cons = [u[1] for u in units[k:k + run]]
        if not (seen_v and k + run < len(units)):      # leading or trailing consonants
            cur += "".join(cons)
        else:
            take = 1
            for n in (3, 2):                            # longest legal onset
                if run >= n and "".join(cons[-n:]) in _ONSETS:
                    take = n
                    break
            cur += "".join(cons[:-take]); sylls.append((cur, stressed))
            cur, stressed, seen_v = "".join(cons[-take:]), False, False
        k += run
    if cur:
        sylls.append((cur, stressed))
    return "-".join(t.upper() if st and len(sylls) > 1 else t for t, st in sylls)


def ipa_to_respell(ipa: str, f5: bool = False) -> str:
    """Rough English-readable respelling of an IPA string: syllables joined by '-', stressed one in CAPS
    (dɑ̃ɡlˈaʁ -> dahn-GLAHR). Nasal vowels become vowel + n. Kokoro's own letters (A, I, O, W, Y) are accepted.
    Meant for people and for plain-text TTS such as F5."""
    ipa = ipa.strip("/")
    if re.search(r"[AIOQWYT]", ipa):
        ipa = misaki_to_std(ipa)
    return " ".join(_respell_word(w, f5) for w in ipa.split())


# Abbreviations and numerals are not names but are read badly; expand them before synthesis.
ABBREVIATIONS = [
    (r"\bMM\.\s+(?=[A-Z])", "Messieurs "), (r"\bMme\.?\s+(?=[A-Z])", "Madame "),
    (r"\bMlle\.?\s+(?=[A-Z])", "Mademoiselle "), (r"\bM\.\s+(?=[A-Z])", "Monsieur "),
    (r"\bDr\.\s+(?=[A-Z])", "Doctor "), (r"\bMr\.\s+(?=[A-Z])", "Mister "),
    (r"\bMrs\.\s+(?=[A-Z])", "Missus "), (r"\bSt\.\s+(?=[A-Z])", "Saint "),
    (r"\bCapt\.\s+(?=[A-Z])", "Captain "), (r"\bCol\.\s+(?=[A-Z])", "Colonel "),
    # first names written short in older books ("Wm. Lloyd Garrison"); read as written they come out as noise
    (r"\bWm\.\s+(?=[A-Z])", "William "), (r"\bChas\.\s+(?=[A-Z])", "Charles "), (r"\bGeo\.\s+(?=[A-Z])", "George "),
    (r"\bThos\.\s+(?=[A-Z])", "Thomas "), (r"\bJas\.\s+(?=[A-Z])", "James "), (r"\bJno\.\s+(?=[A-Z])", "John "),
    # titles and ranks
    (r"\bHon\.\s+(?=[A-Z])", "Honorable "), (r"\bRev\.\s+(?=[A-Z])", "Reverend "), (r"\bMessrs\.\s+(?=[A-Z])", "Messieurs "),
    (r"\bLieut\.\s*-?\s*(?=[A-Z])", "Lieutenant "), (r"\bMaj\.\s*-?\s*(?=[A-Z])", "Major "), (r"\bBrig\.\s*-?\s*(?=[A-Z])", "Brigadier "),
    (r"\bProf\.\s+(?=[A-Z])", "Professor "), (r"\bGov\.\s+(?=[A-Z])", "Governor "),
    (r"(?<=[a-z])\s+Jr\.", " Junior"), (r"(?<=[a-z])\s+Sr\.", " Senior"), (r"\bEsq\.(?=[,;])", "Esquire"), (r"\bEsq\.", "Esquire."),
    (r"\bviz\.,?", "namely,"), (r"\bi\.e\.,?", "that is,"), (r"\be\.g\.,?", "for example,"), (r"\betc\.(?=[,;])", "et cetera"), (r"\betc\.", "et cetera."),
    # states after a place name ("Memphis, Tenn.", "Malden, W. Va.")
    (r"(?<=[,.] )W\. Va\.", "West Virginia"), (r"(?<=, )Va\.", "Virginia"), (r"(?<=, )Tenn\.", "Tennessee"), (r"(?<=, )Mass\.", "Massachusetts"),
    (r"(?<=, )Ga\.", "Georgia"), (r"(?<=, )Ala\.", "Alabama"), (r"(?<=, )Wis\.", "Wisconsin"), (r"(?<=, )Conn\.", "Connecticut"),
    (r"(?<=, )Pa\.", "Pennsylvania"), (r"(?<=, )Ky\.", "Kentucky"), (r"(?<=, )Mo\.", "Missouri"),
    (r"\bSaml\.\s+(?=[A-Z])", "Samuel "), (r"\bBenj\.\s+(?=[A-Z])", "Benjamin "), (r"\bGen\.\s+(?=[A-Z])", "General "),
]
ROMAN_RE = re.compile(r"\b(?:(Act|Scene|Chapter|Book|Part|Canto)\s+|([A-Z][a-z]+)\s+)([IVXLC]{2,})\b")


def _roman(s: str) -> int:
    vals = dict(I=1, V=5, X=10, L=50, C=100)
    tot = 0
    for a, b in zip(s, s[1:] + " "):
        tot += -vals[a] if vals.get(b, 0) > vals[a] else vals[a]
    return tot


# --- degrees, minutes, seconds: "42° 15′ N. lat." -> "forty-two degrees fifteen minutes north latitude" (engines stumble on the symbols)
_NUM = r"\d+(?:\.\d+)?"
_DEG = r"(?:°|\bdeg(?:rees?)?\b\.?)"
_ANGLE_RE = re.compile(
    rf"(?P<d>{_NUM})\s*{_DEG}"
    rf"(?:\s*(?P<m>{_NUM})\s*[′’'](?!\w))?"
    rf"(?:\s*(?P<s>{_NUM})\s*(?:″|”|\"|′′|''))?"
    rf"(?:\s*(?P<f>[CF])\b\.?)?"                                       # 98° F.  ->  degrees Fahrenheit
    rf"(?:\s+(?P<dir>[NSEW])\.(?=[\s,;)]|$))?"
    rf"(?:\s+(?P<ll>lat|long)\b\.?)?")
_COMPASS_RE = re.compile(r"(?<![\w.])(?:[NSEW]\.){2,4}(?!\w)")
_COMPASS = {"N": "north", "S": "south", "E": "east", "W": "west"}


def _say_number(x: str) -> str:
    from num2words import num2words
    return num2words(float(x)) if "." in x else num2words(int(x))


def expand_angles(text: str) -> str:
    """Spell out degree/minute/second marks, compass letters and lat./long. so the voice reads them as speech."""
    def unit(n: str, name: str) -> str:
        return f"{_say_number(n)} {name}" + ("" if n == "1" else "s")

    def angle(m):
        parts = [unit(m["d"], "degree")]
        if m["m"]:
            parts.append(unit(m["m"], "minute"))
        if m["s"]:
            parts.append(unit(m["s"], "second"))
        out = " ".join(parts)
        if m["f"]:
            out += " " + {"C": "Celsius", "F": "Fahrenheit"}[m["f"]]
        if m["dir"]:
            out += " " + _COMPASS[m["dir"]]
        if m["ll"]:
            out += " " + {"lat": "latitude", "long": "longitude"}[m["ll"]]
        # an abbreviation's period that was swallowed is put back when a new sentence starts right after it ("... W. long. In the")
        if m.group(0).endswith(".") and re.match(r"\s+[A-Z“\"‘']", m.string[m.end():]):
            out += "."
        return out

    def compass(m):
        return "-".join(_COMPASS[c] for c in re.findall(r"[NSEW]", m.group(0)))
    return _COMPASS_RE.sub(compass, _ANGLE_RE.sub(angle, text))


def normalize(text: str) -> str:
    from num2words import num2words
    text = expand_angles(text)
    for rx, rep in ABBREVIATIONS:
        text = re.sub(rx, rep, text)

    def roman(m):
        n = _roman(m.group(3))
        if m.group(1):  # "Act II" -> "Act two"
            return f"{m.group(1)} {num2words(n)}"
        return f"{m.group(2)} the {num2words(n, to='ordinal')}"  # "Louis XVIII" -> "Louis the eighteenth"
    return ROMAN_RE.sub(roman, text)


def verified_respellings() -> dict:
    """{name: {"respell", "score", ...}} from data/bible_respell.json (tested against Chatterbox), or {}."""
    p = Path(__file__).resolve().parent.parent / "data" / "bible_respell.json"
    try:
        return json.loads(p.read_text(encoding="utf-8"))["entries"]
    except Exception:
        return {}


class Preprocessor:
    """normalize() expands abbreviations; substitute() applies the lexicon with one compiled regex
    (longest term first).

    mode="respell": replace with phonetic respelling (F5-TTS reads plain text).
    mode="ipa":     replace with Kokoro/misaki markup  [term](/IPA/).
    mode="rawipa":  replace the word with its bare IPA, for text models that may read it (Chatterbox).
    mode="plain":   no substitutions at all.
    mode="typed":   plain spelling, except respellings you typed in the lexicon.
    mode="checked": as "typed", plus respellings the name check found and scored (Higgs; not the Bible respellings tuned for Chatterbox).
    mode="verified": plain spelling, except names whose respelling in data/bible_respell.json was tested to sound
                    better (Chatterbox) and respellings the name check (namecheck.py) scored better than the plain spelling;
                    respellings you typed into the project lexicon win over those.
    Entries lacking the needed field are skipped. Possessives keep their suffix (Edmond's)."""

    def __init__(self, lexicon: list[dict], mode: str = "respell"):
        field = "ipa" if mode in ("ipa", "rawipa") else "respell"
        self.map = {}
        if mode in ("verified", "typed", "checked"):
            lexicon = ([{"term": k, "respell": v["respell"]} for k, v in verified_respellings().items()] if mode == "verified" else []) + \
                      [e for e in lexicon if e.get("respell") and (e.get("respell_src") == "user" or
                                                                   (not e.get("respell_src") and e.get("source") == "user") or
                                                                   (mode in ("verified", "checked") and e.get("respell_src") == "checked"))]   # typed by you (or, for Chatterbox and Higgs, found and scored by the name check), not auto-made from IPA
        for e in ([] if mode == "plain" else lexicon):
            val = (e.get(field) or "").strip()
            if mode == "respell":
                val = f5_text(val) if val else auto_respell(e)  # plain-text engines: F5-safe spelling, made from the IPA if blank
            if val:
                self.map[e["term"].lower()] = (e["term"], val)
        self.mode = mode
        self.map = {self._key(k): v for k, v in self.map.items()}
        terms = sorted(self.map, key=len, reverse=True)
        pat = lambda t: re.escape(t).replace("\\ ", r"\s+").replace("'", "['’]")  # phrases span line breaks
        self.rx = (re.compile(rf"(?<![{LETTER}])(" + "|".join(map(pat, terms)) + rf")(?![{LETTER}])", re.I)
                   if terms else None)

    @staticmethod
    def _key(t: str) -> str:
        return re.sub(r"\s+", " ", t.lower().replace("’", "'"))

    def _sub(self, m: re.Match) -> str:
        orig, val = self.map[self._key(m.group(1))]
        surface = re.sub(r"\s+", " ", m.group(1))
        if self.mode == "ipa":
            return f"[{surface}](/{val.strip('/')}/)"
        if self.mode == "rawipa":
            return val.strip("/")
        if surface.isupper() and len(surface) > 1:
            return val.upper()
        return val[0].upper() + val[1:] if surface[0].isupper() else val

    def substitute(self, text: str) -> str:
        return self.rx.sub(self._sub, text) if self.rx else text

    normalize = staticmethod(normalize)

    def __call__(self, text: str) -> str:
        return self.substitute(normalize(text))


def load_preprocessor(work: Path, mode: str) -> Preprocessor:
    p = work / "lexicon.json"
    return Preprocessor(json.loads(p.read_text()) if p.exists() else [], mode)


# Starter pronunciations for the samples, applied by `lexicon --seed-*`.
JOHN_SEED = {   # IPA targets; the ranking pass finds F5 spellings (Nathanyal was picked by ear)
    "Nathanael": ("nəˈθænjəl", "Nathanyal"),
    "Bethabara": ("bɛˈθæbərə", ""),
    "Siloam": ("sɪˈloʊəm", ""),
    "Didymus": ("ˈdɪdɪməs", ""),
    "Cephas": ("ˈsiːfəs", ""),
    "Nicodemus": ("ˌnɪkəˈdiːməs", ""),
    "Aenon": ("ˈiːnɒn", ""),
    "Sychar": ("ˈsaɪkɑːr", ""),
}


MACBETH_SEED = {
    "Macbeth": ("məkˈbɛθ", "Mak-beth"), "Banquo": ("ˈbæŋkwoʊ", "Bang-kwo"),
    "Duncan": ("ˈdʌŋkən", "Dun-kin"), "Malcolm": ("ˈmælkəm", "Mal-kum"),
    "Donalbain": ("ˈdɒnəlbeɪn", "Don-al-bane"), "Macdonwald": ("məkˈdɒnəld", "Mak-don-ald"),
    "Macduff": ("məkˈdʌf", "Mak-duff"), "Lennox": ("ˈlɛnəks", "Len-ox"),
    "Fleance": ("ˈfliːəns", "Flee-unce"), "Graymalkin": ("ɡreɪˈmælkɪn", "Gray-mal-kin"),
    "Paddock": ("ˈpædək", "Pad-uk"), "Sinel": ("ˈsaɪnəl", "Sigh-nel"),
    "Glamis": ("ɡlɑːmz", "Glahmz"), "Cawdor": ("ˈkɔːdər", "Kaw-der"),
    "Forres": ("ˈfɒrɪs", "Forr-iss"), "Bellona": ("bəˈloʊnə", "Bel-lo-na"),
    "Norweyan": ("nɔːrˈweɪən", "Nor-way-un"), "Colmekill": ("ˈkoʊmkɪl", "Kolm-kill"),
    "Saint Colme": ("seɪnt koʊm", "Saint Kolm"), "Hecate": ("ˈhɛkət", "Heck-it"),
    "Thane": ("θeɪn", "Thane"), "Ross": ("rɒs", "Ross"),
}


MONTE_CRISTO_SEED = {
    "Dantès": ("dɑ̃ˈtɛs", "Dahn-tess"), "Edmond": ("ɛdˈmɒ̃", "Ed-mohn"),
    "Mercédès": ("mɛrseɪˈdɛs", "Mair-say-dess"), "Caderousse": ("kɑːdˈruːs", "Kad-roose"),
    "Danglars": ("dɑ̃ˈɡlɑːr", "Dahn-glar"), "Fernand": ("fɛrˈnɑ̃", "Fair-nahn"),
    "Villefort": ("vilˈfɔːr", "Veel-for"), "Morrel": ("mɔˈrɛl", "Mor-rell"),
    "Marseilles": ("mɑːrˈseɪ", "Mar-say"), "Pharaon": ("faraˈɔ̃", "Fah-rah-ohn"),
    "Réserve": ("reɪˈzɛrv", "Ray-zerv"), "Catalans": ("ˈkætələnz", "Cat-uh-lunz"),
    "Arlesian": ("ɑːrˈliːʒən", "Ar-lee-zhun"), "Arlesienne": ("ɑːrlɛzˈjɛn", "Ar-lez-yen"),
    "Cyprus": ("ˈsaɪprəs", "Sigh-prus"), "Chios": ("ˈkaɪɒs", "Kigh-oss"),
}


def seed(work: Path, seeds: dict = JOHN_SEED) -> None:
    p = work / "lexicon.json"
    lex = json.loads(p.read_text())
    for e in lex:
        if e["term"] in seeds:
            e["ipa"], e["respell"], e["source"] = *seeds[e["term"]], "user"
    p.write_text(json.dumps(lex, indent=2, ensure_ascii=False))
