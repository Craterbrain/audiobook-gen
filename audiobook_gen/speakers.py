"""Module 1b: split chapters into narration/dialogue segments with speaker roles.

Heuristic parser handles both quoted fiction ("..." said Anna) and unquoted
KJV-style speech (Jesus answered and said unto him, Verily...). An optional
local-LLM pass (any OpenAI-compatible endpoint) can refine speaker labels.
"""
import json
import re
from pathlib import Path

VERBS = r"(?:answered|said|saith|cried|spake|asked|asketh|saying|replied|shouted|whispered|exclaimed|sighed|returned|added|inquired|continued|observed|remarked|muttered|murmured|rejoined|interrupted|stammered|responded|demanded|resumed|declared|rejoined|retorted|exclaiming|repeated|echoed|laughed|sobbed|groaned|gasped|whimpered|thundered|uttered|ejaculated|faltered|cried|shouted|answer)"
# attribution = sentence start ... speech verb ... comma, then the spoken text
ATTR = re.compile(
    rf"(?:^|(?<=[.?!]\s))((?:[^.?!;:]{{0,70}}?)\b{VERBS}\b[^.?!;:,]{{0,50}},\s+)"
)
LEAD_SAYING = re.compile(r"^(?:and\s+)?(?:saying|said|saith)[, ]+\s*", re.I)
QUOTE_RE = re.compile(r"[“\"]([^”\"]+)[”\"]")
STOP = {"Then", "And", "But", "The", "Now", "When", "So", "Verily", "Lord", "God", "Father",
        "Son", "Spirit", "Word", "He", "She", "They", "It", "I", "His", "Her", "Their", "Therefore",
        "Again", "While", "Jesus's"}
PRONOUN = re.compile(r"\b(he|she|they|the other)\b", re.I)
GROUP = {"jews": "Jews", "disciples": "Disciples", "pharisees": "Pharisees", "woman": "Woman",
         "servants": "Servants", "officers": "Officers", "people": "Crowd", "multitude": "Crowd"}
DEFAULT_ALIASES = {"John": "John the Baptist"}  # in the Gospel, quoted "John" is the Baptist
NARRATOR = "Narrator"
SPEECHY = re.compile(r"\b(I|thou|thee|thy|ye|we|you|my|me)\b")
NARRATIVE_START = re.compile(r"(Then|Now|After|When|So|Jesus|And (?:he|they|Jesus|the|there|John)\b)")


def _speaker(prefix: str, state: dict, aliases: dict, ctx: dict | None = None,
             allow_pron: bool = True) -> str | None:
    """Role named in an attribution fragment; a bare pronoun means someone already speaking.

    ctx (prose mode): {"names": set of known character names, "phrases": [(regex, role)]}.
    Without ctx (KJV mode) capitalised words are taken as names and first/second person
    words reject the fragment (it is inside speech)."""
    names_ok = ctx["names"] if ctx else None
    if not ctx and re.search(r"\b(I|thou|ye|we|thee)\b", prefix):
        return None
    if ctx:
        for rx, role in ctx["phrases"]:
            if re.search(rx, prefix, re.I):
                return role
    for g, role in GROUP.items():
        if re.search(rf"\b{g}\b", prefix, re.I):
            return role
    v = re.search(rf"\b{VERBS}\b", prefix)
    before, after = (prefix[:v.start()], prefix[v.end():]) if v else (prefix, "")
    if ctx and (re.search(r"\b(he|she|they)\s+(?:\w+ly\s+)?$", before, re.I)
                or re.match(r"\s*(?:\w+ly\s+)?(he|she|they)\b", after, re.I)):
        # pronoun is the subject ("turning to Mercédès, he inquired"): any name is an object
        return None if not allow_pron else state["pron"]()
    pick = lambda t: [x for x in re.findall(r"\b[A-Z][a-zà-ÿ]+\b", t)
                      if (x in names_ok if names_ok is not None else x not in STOP)]
    names = pick(before) or ([] if re.match(r"\s*(he|she|they|the other)\b", after, re.I) else pick(after)[:1])
    if names:
        return aliases.get(names[0], names[0])
    if allow_pron and PRONOUN.search(prefix):
        return state["pron"]()
    return None


def _frag_after(gap: str) -> str:
    """Attribution fragment that follows a quote: the gap up to its first sentence end."""
    return re.split(r"(?<=[a-zà-ÿ’”][.!?])\s", gap.strip(), maxsplit=1)[0][:140]


def _frag_before(gap: str) -> str:
    """Attribution fragment that precedes a quote: the last sentence of the gap."""
    parts = [p for p in re.split(r"(?<=[a-zà-ÿ’”][.!?])\s", gap.strip()) if p]
    return parts[-1][-140:] if parts else ""


def parse_quoted(text: str, state: dict, aliases: dict, ctx: dict | None) -> list[tuple[str, str]]:
    """Quoted prose: attribute each quote from the verb phrase around it ("...," said Mercédès)."""
    pieces, pos = [], 0
    for m in QUOTE_RE.finditer(text):
        pieces += [text[pos:m.start()], m.group(1).strip()]
        pos = m.end()
    pieces.append(text[pos:])  # gaps at even indexes, quotes at odd
    n = len(pieces) // 2
    first = {"yes": True}
    if ctx:  # remember the last character named in narration, for unattributed openers
        for g in pieces[0::2]:
            nm = [x for x in re.findall(r"\b[A-Z][a-zà-ÿ]+\b", g) if x in ctx["names"]]
            if nm:
                state["mention"] = aliases.get(nm[0], nm[0])
    state["pron"] = lambda: (state["prev"] or state["last"]) if first["yes"] else state["last"]
    who_of: list[str] = []
    for qi in range(n):
        before, quote, after = pieces[2 * qi], pieces[2 * qi + 1], pieces[2 * qi + 2]
        who = None
        fa, fb = _frag_after(after), _frag_before(before)
        if re.search(rf"\b{VERBS}\b", fa):
            who = _speaker(fa, state, aliases, ctx)
        if not who and re.search(rf"\b{VERBS}\b", fb):
            who = _speaker(fb, state, aliases, ctx)
        if not who and qi and quote[:1].islower() and re.search(rf"\b{VERBS}\b", before):
            who = who_of[-1]  # split quote: "Father," said X, "sit..."
        if not who:  # unattributed: continue within a paragraph, alternate across paragraphs
            who = (state["prev"] if first["yes"] and state["prev"] not in (None, state["last"])
                   else state["last"]) or state.get("mention") or "Speaker A"
        who_of.append(who)
        if who != state["last"]:
            state["prev"], state["last"] = state["last"], who
        first["yes"] = False
    out = []
    for i, piece in enumerate(pieces):
        if i % 2:
            out.append((who_of[i // 2], piece))
        elif re.search(r"\w", piece):
            out.append((NARRATOR, piece.strip()))
    return out


def parse_paragraph(text: str, state: dict, aliases: dict, ctx: dict | None = None) -> list[dict]:
    segs: list[tuple[str, str]] = []  # (speaker, text)

    if QUOTE_RE.search(text):
        segs = parse_quoted(text, state, aliases, ctx)
    else:
        matches = []
        for m in ATTR.finditer(text):
            who = _speaker(m.group(1), state, aliases, ctx)
            if who:
                matches.append((m, who))
                state["prev"], state["last"] = state["last"], who  # provisional; keeps pronouns flowing
        if not matches:
            segs.append((NARRATOR, text))
        else:
            if matches[0][0].start() > 0:
                segs.append((NARRATOR, text[:matches[0][0].start()].strip()))
            for i, (m, who) in enumerate(matches):
                prefix = m.group(1)
                end = matches[i + 1][0].start() if i + 1 < len(matches) else len(text)
                speech = text[m.end():end].strip()
                while (lm := LEAD_SAYING.match(speech)):
                    prefix += lm.group(0)
                    speech = speech[lm.end():]
                segs.append((NARRATOR, prefix.strip().rstrip(",") + ","))
                if speech:
                    segs.append((who, speech))
    return [{"speaker": s, "kind": "narration" if s == NARRATOR else "dialogue", "text": t}
            for s, t in segs if t.strip()]


def _mentions(text: str, aliases: dict, ctx: dict) -> list[str]:
    """Roles named in `text`, in order: phrase roles, group nouns, then known character names."""
    found = []
    for rx, role in ctx["phrases"]:
        if re.search(rx, text, re.I):
            found.append(role)
    for g, role in GROUP.items():
        if re.search(rf"\b{g}\b", text, re.I):
            found.append(role)
    for x in re.findall(r"\b[A-Z][a-zà-ÿ]+\b", text):
        if x in ctx["names"]:
            found.append(aliases.get(x, x))
    return list(dict.fromkeys(found))


FEM_NOUN = r"(?:girl|woman|lady|maiden|damsel|mother|wife|bride|daughter|sister|queen)"
MASC_NOUN = r"(?:man|gentleman|boy|youth|fellow|father|husband|bridegroom|son|brother|lad|king)"


def _pron_gender(frag: str) -> str | None:
    if re.search(r"\b(he|his|him)\b|\b(?:the|a)\s+(?:\w+\s+)?" + MASC_NOUN + r"\b", frag, re.I):
        return "m"
    if re.search(r"\b(she|her)\b|\b(?:the|a)\s+(?:\w+\s+)?" + FEM_NOUN + r"\b", frag, re.I):
        return "f"
    return "n" if re.search(r"\b(the other|they)\b", frag, re.I) else None


def parse_prose_chapter(ch: dict, aliases: dict, ctx: dict) -> list[dict]:
    """Quoted prose: gather evidence for every quote in the chapter, decode jointly, then emit segments."""
    from .attribution import decode, infer_genders

    paras = [re.sub(r"\s+", " ", p).strip() for p in re.split(r"\n\s*\n", ch["text"])]
    paras = [p for p in paras if p]
    matchers = [(rf"\b{re.escape(n)}\b", aliases.get(n, n)) for n in ctx["names"]] + \
               [(rx, role) for rx, role in ctx["phrases"]]
    gender = {**infer_genders(ch["text"], matchers), **ctx.get("genders", {})}

    layout, items = [], []  # layout: per paragraph, list of ("N", text) | ("Q", item index)
    para_mentions = [_mentions(QUOTE_RE.sub(" ", p), aliases, ctx) for p in paras]  # narration only
    window: list[str] = []  # roles seen as explicit speakers recently
    for pi, para in enumerate(paras):
        pieces, pos = [], 0
        for m in QUOTE_RE.finditer(para):
            pieces += [para[pos:m.start()], m.group(1).strip()]
            pos = m.end()
        pieces.append(para[pos:])
        row = []
        for gi, gap in enumerate(pieces[0::2]):
            if re.search(r"\w", gap):
                row.append(("N", gap.strip()))
            if gi < len(pieces) // 2:
                quote = pieces[2 * gi + 1]
                before, after = pieces[2 * gi], pieces[2 * gi + 2]
                fa, fb = _frag_after(after), _frag_before(before)
                has = lambda f: re.search(rf"\b{VERBS}\b", f)
                explicit = (_speaker(fa, {}, aliases, ctx, False) if has(fa) else None) or \
                           (_speaker(fb, {}, aliases, ctx, False) if has(fb) else None)
                frag = fa if has(fa) else fb if has(fb) else ""
                near = _mentions(before, aliases, ctx)
                lead = next((para_mentions[k][0] for k in (pi - 1, pi - 2)
                             if k >= 0 and para_mentions[k] and not QUOTE_RE.search(paras[k])), None)
                prev_par = [r for k in range(max(0, pi - 3), pi) for r in para_mentions[k]]
                cands = list(dict.fromkeys(near + para_mentions[pi] + prev_par + window[-4:]))
                if explicit:
                    window.append(explicit)
                pron = None if explicit else _pron_gender(frag)
                prev_item = items[-1] if items and items[-1]["para"] == pi else None
                items.append({
                    "para": pi, "text": quote, "explicit": explicit, "pron": pron, "lead": lead,
                    "split": bool(gi and has(before) and re.search(r"[,;:—-]\s*$", before)),
                    "cont": bool(prev_item and prev_item["explicit"] and not explicit and not pron),
                    "other": bool(re.search(r"\bthe other\b", frag, re.I)) and not explicit,
                    "addressed": set(_mentions(quote, aliases, ctx)) | (set(_mentions(frag, aliases, ctx)) - {explicit}),
                    "recent": set(near), "cands": cands[:10]})
                row.append(("Q", len(items) - 1))
        layout.append(row)

    # cast discovery: only roles that are explicitly named as speaking (here or in earlier chapters)
    # or configured phrase/group roles can be chosen; place names and bystanders drop out
    cast = ctx.setdefault("cast", {})
    for it in items:
        if it["explicit"]:
            cast[it["explicit"]] = cast.get(it["explicit"], 0) + 1
    allowed = set(cast) | {r for _, r in ctx["phrases"]} | set(GROUP.values())
    top = [r for r, _ in sorted(cast.items(), key=lambda kv: -kv[1])][:4]
    for it in items:
        it["cands"] = [c for c in it["cands"] if c in allowed]
        if len(it["cands"]) < 2:  # nothing named nearby: fall back to the main speakers
            it["cands"] = list(dict.fromkeys(it["cands"] + top))
    who, post = decode(items, gender, posteriors=True)
    judge = ctx.get("judge")
    if judge:
        from .tiebreak import context_for
        extra, thr = {}, ctx.get("judge_threshold", 0.8)
        todo = [i for i, it in enumerate(items)
                if not (it["explicit"] or max(post[i].values()) >= thr or len(post[i]) < 2)]  # rest are clear already
        tick = ctx.get("tick")
        for k, i in enumerate(todo):
            if tick:
                tick(k / max(1, len(todo)), f"Asking the small model about ambiguous quotes ({k + 1}/{len(todo)})")
            it = items[i]
            opts = [r for r, _ in sorted(post[i].items(), key=lambda kv: -kv[1])][:6]
            lp = judge.score(context_for(paras, it["para"], it["text"]), it["text"][:300], opts)
            extra[i] = {r: ctx.get("judge_weight", 0.5) * max(v, -6.0) for r, v in lp.items()}
            if ctx.get("raw") is not None:
                ctx["raw"][i] = lp
        if extra:
            before_who = who
            who = decode(items, gender, extra)
            ctx["judge_stats"] = ctx.get("judge_stats", [0, 0])
            ctx["judge_stats"][0] += len(extra)
            ctx["judge_stats"][1] += sum(a != b for a, b in zip(before_who, who))
    ctx["_items"], ctx["_gender"], ctx["_post"] = items, gender, post
    if ctx.get("debug") is not None:
        ctx["debug"].extend({**it, "who": w} for it, w in zip(items, who)); ctx["debug_gender"] = gender
    out = []
    for row in layout:
        segs = [(NARRATOR, v) if k == "N" else (who[v], items[v]["text"]) for k, v in row]
        for i, (sp, t) in enumerate(segs):
            out.append({"speaker": sp, "kind": "narration" if sp == NARRATOR else "dialogue",
                        "text": t, "para_start": i == 0})
    for n, sg in enumerate(out):
        sg.update(chapter=ch["index"], id=f"{ch['index']:03d}-{n:05d}")
    return out


def parse_play_chapter(ch: dict) -> list[dict]:
    """'SPEAKER.' header line followed by the speech; heading is read by the narrator."""
    out = [{"speaker": NARRATOR, "kind": "narration", "text": ch["heading"], "para_start": True}]
    for block in re.split(r"\n\s*\n", ch["text"]):
        lines = [l.strip() for l in block.strip().splitlines()]
        if len(lines) < 2 or not re.fullmatch(r"[A-Z][A-Z .’'-]+\.", lines[0]):
            continue  # stray text / unattributed direction
        role = lines[0].rstrip(".").title().replace("’S", "’s")
        text = re.sub(r"\s+", " ", " ".join(lines[1:])).strip()
        if out[-1]["speaker"] == role and len(out) > 1:
            out[-1]["text"] += " " + text
        else:
            out.append({"speaker": role, "kind": "dialogue", "text": text, "para_start": True})
    for n, s in enumerate(out):
        s.update(chapter=ch["index"], id=f"{ch['index']:03d}-{n:05d}")
    return out


def parse_chapter(ch: dict, aliases: dict, ctx: dict | None = None) -> list[dict]:
    if ch.get("format") == "play":
        return parse_play_chapter(ch)
    if ctx and QUOTE_RE.search(ch["text"]):
        return parse_prose_chapter(ch, aliases, ctx)
    state = {"last": None, "prev": None, "pron": lambda: state["last"]}
    out = []
    for p in re.split(r"\n\s*\n", ch["text"]):
        p = re.sub(r"\s+", " ", p).strip()
        if not p:
            continue
        segs = parse_paragraph(p, state, aliases, ctx)
        # unquoted speech that runs on across verse/paragraph breaks
        if (len(segs) == 1 and segs[0]["kind"] == "narration" and state.get("open")
                and SPEECHY.search(p) and not NARRATIVE_START.match(p)):
            segs = [{"speaker": state["open"], "kind": "dialogue", "text": p}]
        state["open"] = segs[-1]["speaker"] if segs[-1]["kind"] == "dialogue" else None
        for i, seg in enumerate(segs):
            seg["para_start"] = i == 0
            out.append(seg)
    # merge adjacent narration pieces from one paragraph (keeps flow natural)
    merged = []
    for s in out:
        if merged and not s["para_start"] and merged[-1]["speaker"] == s["speaker"]:
            merged[-1]["text"] += " " + s["text"]
        else:
            merged.append(s)
    for n, s in enumerate(merged):
        s.update(chapter=ch["index"], id=f"{ch['index']:03d}-{n:05d}")
    return merged


def llm_refine(segments: list[dict], endpoint: str, model: str) -> list[dict]:
    """Optional: ask a local OpenAI-compatible server to relabel dialogue speakers.
    Falls back to heuristic labels on any failure."""
    import urllib.request
    roles = sorted({s["speaker"] for s in segments})
    for i in range(0, len(segments), 40):
        window = segments[max(0, i - 3):i + 40]
        prompt = ("Known characters: " + ", ".join(roles) + ".\nFor each numbered line, return a JSON "
                  'list of speaker names (use "Narrator" for narration).\n' +
                  "\n".join(f"{j}. {s['text'][:200]}" for j, s in enumerate(window)))
        body = json.dumps({"model": model, "temperature": 0,
                           "messages": [{"role": "user", "content": prompt}]}).encode()
        try:
            req = urllib.request.Request(endpoint, body, {"Content-Type": "application/json"})
            reply = json.load(urllib.request.urlopen(req, timeout=120))["choices"][0]["message"]["content"]
            labels = json.loads(reply[reply.index("["):reply.rindex("]") + 1])
            if len(labels) == len(window):
                for s, lab in zip(window, labels):
                    if s["kind"] == "dialogue" and isinstance(lab, str):
                        s["speaker"] = lab
        except Exception as e:  # noqa: BLE001 - keep heuristic result
            print(f"[llm] window {i} skipped: {e}")
    return segments


def find_names(text: str, min_count: int = 2) -> set[str]:
    """Character-name candidates: capitalised mid-sentence words seen repeatedly and uncommon in English."""
    from collections import Counter
    from wordfreq import zipf_frequency

    c: Counter = Counter()
    for m in re.finditer(r"\b[A-Z][a-zà-ÿ]{2,}\b", text):
        pre = text[max(0, m.start() - 6):m.start()]
        if re.search(r"[.!?“”\"]\s*$", pre) and not re.search(r"\b(M|Mme|Mlle|Dr|St)\.\s*$", pre):
            continue
        c[m.group()] += 1
    return {w for w, n in c.items() if n >= min_count and zipf_frequency(w.lower(), "en") < 4.0}


def collapse_to_narrator(segs: list[dict]) -> list[dict]:
    """Single-voice mode: everything is the Narrator, and pieces of one paragraph (verse) are rejoined so
    the reading flows instead of pausing at every change of speaker."""
    out = []
    for s in segs:
        s = {**s, "speaker": NARRATOR, "kind": "narration"}
        if out and not s["para_start"] and out[-1]["chapter"] == s["chapter"]:
            out[-1]["text"] += " " + s["text"]
        else:
            out.append(s)
    for n, s in enumerate(out):
        s["id"] = f"{s['chapter']:03d}-{n:05d}"
    return out


def parse_book(work: Path, aliases: dict | None = None, llm: tuple[str, str] | None = None,
               phrase_roles: dict | None = None, judge=None, progress=None, genders: dict | None = None) -> list[dict]:
    """progress(fraction 0..1, description) is called as chapters (and ambiguous quotes) are processed."""
    data = json.loads((work / "chapters.json").read_text())
    al = {**DEFAULT_ALIASES, **(aliases or {})}
    full = "\n".join(c["text"] for c in data["chapters"])
    # quoted prose gets the name-aware parser; unquoted (KJV-style) keeps the original rules
    ctx = None
    if QUOTE_RE.search(full) and not any(c.get("format") == "play" for c in data["chapters"]):
        ctx = {"names": find_names(full),
               "phrases": [(k, v) for k, v in (phrase_roles or {}).items()], "judge": judge,
               "genders": {r: ("m" if g == "male" else "f") for r, g in (genders or {}).items() if g in ("male", "female")}}
        al = {k: v for k, v in al.items() if k != "John"} | (aliases or {})
    segs, n = [], len(data["chapters"])
    for i, ch in enumerate(data["chapters"]):
        if progress:
            progress(i / n, f"Reading chapter {i + 1} of {n}: {ch.get('title', '')}")
            if ctx is not None:
                ctx["tick"] = lambda f, d, i=i: progress((i + f) / n, d)
        segs += parse_chapter(ch, al, ctx)
    if progress:
        progress(1.0, "Saving")
    if ctx and ctx.get("judge_stats"):
        print(f"[judge] asked about {ctx['judge_stats'][0]} uncertain quotes, changed {ctx['judge_stats'][1]} labels")
    if llm:
        segs = llm_refine(segs, *llm)
    (work / "segments.json").write_text(json.dumps(segs, indent=2, ensure_ascii=False))
    return segs
