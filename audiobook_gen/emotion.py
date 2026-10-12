"""Per-sentence delivery for emotion-aware engines (Chatterbox: exaggeration + cfg_weight).

Two signals, combined:
  1. a pretrained 7-emotion text classifier (j-hartmann/emotion-english-distilroberta-base, CPU), averaged over
     the sentence and its neighbours so one flat sentence in a tense scene doesn't snap back to neutral;
  2. the dialogue tag next to a quote ("cried Caderousse, smiling", "whispered"), which says outright how it
     was spoken.
Narration is pulled toward a calm reading; quoted speech gets the full range."""
import re

MODEL = "j-hartmann/emotion-english-distilroberta-base"

#            exaggeration, cfg_weight (lower cfg = faster, more energetic delivery)
TABLE = {"neutral": (0.45, 0.45), "joy": (0.62, 0.40), "sadness": (0.40, 0.55), "anger": (0.82, 0.30),
         "fear": (0.68, 0.35), "surprise": (0.72, 0.35), "disgust": (0.66, 0.40)}

# tag word -> (exaggeration shift, cfg shift)
TAGS = {"cried": (.15, -.05), "shouted": (.2, -.08), "exclaimed": (.15, -.05), "roared": (.25, -.1), "screamed": (.25, -.1),
        "yelled": (.2, -.08), "thundered": (.2, -.08), "declared": (.08, 0), "demanded": (.1, -.05), "snapped": (.12, -.05),
        "whispered": (-.2, .1), "murmured": (-.15, .1), "muttered": (-.12, .08), "sighed": (-.15, .1), "sobbed": (-.05, .1),
        "faltered": (-.1, .1), "stammered": (-.05, .05), "asked": (0, 0), "inquired": (0, 0), "said": (0, 0), "replied": (0, 0),
        "answered": (0, 0), "added": (0, 0), "laughed": (.1, -.05), "laughingly": (.1, -.05), "smiling": (.05, 0),
        "gently": (-.12, .08), "softly": (-.15, .1), "quietly": (-.15, .1), "timid": (-.12, .08), "timidly": (-.12, .08),
        "angrily": (.2, -.1), "furiously": (.25, -.1), "eagerly": (.1, -.05), "joyfully": (.12, -.05), "sadly": (-.1, .1),
        "bitterly": (.1, .0), "coldly": (-.05, .1), "trembling": (-.05, .05), "pale": (-.05, .05)}
SHORT_WORDS = 6          # chunks shorter than this are pulled toward calm in proportion to their length
SHORT_MAX_EXAG, SHORT_MIN_CFG = 0.55, 0.40
EXCLAIM_EXAG = 0.70      # one- or two-word exclamations
MAX_EXAG = 0.9
WORD = re.compile(r"[a-z]+")

_pipe = None


def _classifier():
    global _pipe
    if _pipe is None:
        from transformers import pipeline
        _pipe = pipeline("text-classification", model=MODEL, top_k=None, device=-1, truncation=True, max_length=256)
    return _pipe


def probabilities(texts: list[str]) -> list[dict]:
    out = _classifier()(texts, batch_size=16)
    return [{d["label"]: d["score"] for d in row} for row in out]


HIGGS_TAGS = {"joy": "contentment", "sadness": "sadness", "anger": "anger", "fear": "fear", "surprise": "surprise", "disgust": "disgust"}


def higgs_tags(chunks: list[list[str]], minimum: float = 0.45) -> list[list[str]]:
    """An emotion tag for each chunk, for Higgs (which takes `<|emotion:...|>` at the start of a sentence): the strongest non-neutral feeling of
    the chunk averaged with its neighbours, when it reaches `minimum`; empty for everything calmer. Strong joy is elation."""
    flat = [c for seg in chunks for c in seg]
    if not flat:
        return [[] for _ in chunks]
    probs = probabilities(flat)
    tags, i = [], 0
    for seg in chunks:
        row = []
        for _ in seg:
            nb = [probs[j] for j in range(max(0, i - 1), min(len(flat), i + 2))]
            avg = {k: sum(p.get(k, 0) for p in nb) / len(nb) for k in HIGGS_TAGS}
            k = max(avg, key=avg.get)
            row.append("" if avg[k] < minimum else "<|emotion:elation|>" if k == "joy" and avg[k] >= 0.6 else f"<|emotion:{HIGGS_TAGS[k]}|>")
            i += 1
        tags.append(row)
    return tags


# The narrator's feelings the language model chooses between for Higgs: (Higgs tag, how it is described to the model)
DIRECTOR = [("none", "No particular feeling: plain, informational or calm narration"),
            ("awe", "Awe, wonder or reverence"),
            ("sadness", "Sadness, grief or regret"),
            ("determination", "Firm resolve, conviction or authority"),
            ("elation", "Joy, delight or triumph"),
            ("enthusiasm", "Eager excitement"),
            ("affection", "Warmth, tenderness or love"),
            ("contentment", "Calm gladness, peace or comfort"),
            ("surprise", "Surprise or astonishment"),
            ("anger", "Anger, indignation or rebuke"),
            ("fear", "Fear, alarm or dread"),
            ("contemplation", "Thoughtful, reflective, solemn")]
DIRECTOR_SYSTEM = "You are directing a narrator who reads a book aloud. You decide which feeling, if any, the narrator should carry in a passage."


def director_question(before: str, passage: str, after: str, title: str = "", author: str = "") -> str:
    book = (f'In the audiobook "{title}"' + (f" by {author}" if author else "")) if title else "In an audiobook"
    ctx = (f'It follows: "{before}"\n' if before else "") + (f'It is followed by: "{after}"\n' if after else "")
    return f'{book}, the narrator reads this passage aloud: "{passage}"\n{ctx}\nWhich feeling should the narrator carry while reading the passage?'


def director_probs(work, segs: list[dict], raw: list[list[str]], progress=None) -> dict[str, dict[str, float]]:
    """{"segment:chunk": {tag: probability}} for every chunk, from the language model, cached in work/higgs_emotion.json by the chunk's text."""
    import hashlib
    import json
    from pathlib import Path
    path = Path(work) / "higgs_emotion.json"
    try:
        cache = json.loads(path.read_text())
    except (OSError, ValueError):
        cache = {}
    flat = [(f"{sg['id']}:{ci}", c) for sg, chunks in zip(segs, raw) for ci, c in enumerate(chunks)]
    stamp = lambda c: hashlib.sha1(c.encode()).hexdigest()[:10]
    todo = [i for i, (k, c) in enumerate(flat) if cache.get(k, {}).get("h") != stamp(c)]
    if todo:
        try:
            meta = json.loads((Path(work) / "chapters.json").read_text())
        except (OSError, ValueError):
            meta = {}
        from .refine import Scorer
        sc = Scorer()
        try:
            clip = lambda t: t if len(t) <= 220 else t[:217] + "..."
            reqs = [(clip(flat[i - 1][1]) if i else "", flat[i][1], clip(flat[i + 1][1]) if i + 1 < len(flat) else "") for i in todo]
            out = sc.emotions(reqs, meta.get("title", ""), meta.get("author", ""), progress=progress)
        finally:
            sc.close()
        for i, p in zip(todo, out):
            cache[flat[i][0]] = {"h": stamp(flat[i][1]), "p": p}
        path.write_text(json.dumps(cache))
    return {k: cache[k]["p"] for k, _ in flat}


def director_tags(probs: dict[str, dict[str, float]], share: float = 0.2, max_per_tag: float = 0.35) -> dict[str, str]:
    """The tag for the chunks the model feels most strongly about: those whose best feeling (other than none) also beats "none", up to `share`
    of all chunks, taken from the most to the least certain. No one feeling takes more than `max_per_tag` of what is tagged, so the delivery
    does not turn into one note (a scripture passage would otherwise be all awe). A lower share tunes the emotion down; 0 gives none."""
    by_tag: dict[str, list] = {}
    for k, p in probs.items():
        best = max((t for t in p if t != "none"), key=p.get)
        if p[best] > p.get("none", 0):
            by_tag.setdefault(best, []).append((p[best] - p.get("none", 0), k))
    for rows in by_tag.values():
        rows.sort(reverse=True)
    want = int(round(share * len(probs)))
    cap = max(1, int(round(max_per_tag * want)))
    picked: dict[str, str] = {}
    taken = {t: 0 for t in by_tag}
    pos = {t: 0 for t in by_tag}
    while len(picked) < want:
        live = [t for t in by_tag if pos[t] < len(by_tag[t]) and taken[t] < cap]
        if not live:
            break
        t = max(live, key=lambda x: by_tag[x][pos[x]][0])            # the most certain of what is left, among feelings still under their cap
        k = by_tag[t][pos[t]][1]
        pos[t] += 1; taken[t] += 1
        picked[k] = f"<|emotion:{t}|>"
    return picked


def tag_shift(text: str) -> tuple[float, float]:
    """Sum of the shifts of every tag word in a short narration snippet (capped)."""
    ex = cfg = 0.0
    for w in WORD.findall(text.lower()):
        if w in TAGS:
            ex += TAGS[w][0]; cfg += TAGS[w][1]
    return max(-.3, min(.3, ex)), max(-.15, min(.15, cfg))


LLM_WEIGHT = 0.65          # how much the language pass (refine.py) counts against the sentence classifier
STRENGTH = {1: 0.5, 2: 0.8, 3: 1.0}


def blend(classifier: dict, hint: dict | None) -> dict:
    """The classifier's probabilities mixed with the language pass's reading of the passage (if there is one)."""
    if not hint:
        return classifier
    s = 0.0 if hint["emo"] == "neutral" else STRENGTH.get(hint.get("lvl", 2), 0.8)
    llm = {"neutral": 1.0 - s, hint["emo"]: s} if s else {"neutral": 1.0}
    keys = set(classifier) | set(llm)
    return {k: LLM_WEIGHT * llm.get(k, 0.0) + (1 - LLM_WEIGHT) * classifier.get(k, 0.0) for k in keys}


def delivery(segs: list[dict], chunks: list[list[str]], base: float = 0.0, hints: list[list[dict | None]] | None = None) -> list[list[dict]]:
    """For each segment, one {"exaggeration", "cfg_weight"} per chunk. `chunks[i]` are segment i's text chunks.
    `hints` (same shape) are the language pass's reading of each chunk."""
    flat = [(i, j, c) for i, cs in enumerate(chunks) for j, c in enumerate(cs)]
    probs = probabilities([c for _, _, c in flat]) if flat else []
    if hints:
        probs = [blend(p, hints[i][j]) for p, (i, j, _) in zip(probs, flat)]
    out: list[list[dict]] = [[{} for _ in cs] for cs in chunks]
    for n, (i, j, _) in enumerate(flat):
        seg = segs[i]
        # blend with the neighbouring sentences (context window)
        p = {k: 0.6 * v for k, v in probs[n].items()}
        for m, w in ((n - 1, 0.2), (n + 1, 0.2)):
            if 0 <= m < len(flat):
                for k, v in probs[m].items():
                    p[k] = p.get(k, 0) + w * v
        ex = sum(p.get(k, 0) * TABLE[k][0] for k in TABLE)
        cfg = sum(p.get(k, 0) * TABLE[k][1] for k in TABLE)
        dialogue = seg.get("kind") == "dialogue"
        words = len(chunks[i][j].split())
        if not dialogue:   # narration stays close to calm
            ex = TABLE["neutral"][0] + 0.35 * (ex - TABLE["neutral"][0])
            cfg = TABLE["neutral"][1] + 0.35 * (cfg - TABLE["neutral"][1])
        else:   # dialogue tag from the short narration segments on either side
            tag = ""
            for k, edge in ((i - 1, -90), (i + 1, 90)):
                if 0 <= k < len(segs) and segs[k].get("kind") == "narration" and len(segs[k]["text"]) < 90:
                    tag += " " + segs[k]["text"]
            dex, dcfg = tag_shift(tag)
            ex, cfg = ex + dex, cfg + dcfg
            if chunks[i][j].rstrip().endswith("!") and words >= 4:
                ex += 0.06
        if words < SHORT_WORDS:     # a word or two says little about the feeling, and extremes on so little audio screech
            if chunks[i][j].rstrip().endswith("!"):     # a short exclamation is loud on purpose: about 0.7, never more
                ex = EXCLAIM_EXAG if words <= 2 else max(0.55, min(ex, EXCLAIM_EXAG))
                cfg = max(min(cfg, 0.40), 0.35)
            else:
                f = words / SHORT_WORDS
                ex = TABLE["neutral"][0] + f * (ex - TABLE["neutral"][0])
                cfg = TABLE["neutral"][1] + f * (cfg - TABLE["neutral"][1])
                ex, cfg = min(ex, SHORT_MAX_EXAG), max(cfg, SHORT_MIN_CFG)
            ex = min(ex + base, EXCLAIM_EXAG) if chunks[i][j].rstrip().endswith("!") else ex + base
            out[i][j] = {"exaggeration": round(max(0.25, ex), 2), "cfg_weight": round(max(0.15, min(0.7, cfg)), 2)}
            continue
        ex = min(ex + base, MAX_EXAG)
        out[i][j] = {"exaggeration": round(max(0.25, ex), 2), "cfg_weight": round(max(0.15, min(0.7, cfg)), 2)}
    return out
