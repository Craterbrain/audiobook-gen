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
