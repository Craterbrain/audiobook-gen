"""Language pass: a small instruction model reads the book once, before the speech is made, and decides two things the plain
text only half says.

  1. Questions. A sentence that is asked but ends in "." (or nothing) gets a "?", because the voice follows the punctuation.
     Only the final mark of a sentence is ever changed; every word stays as it is.
  2. Delivery. Each passage gets an emotion and a strength, read with the passages before it and the speaker in view. The
     delivery step (emotion.py) blends this with its own sentence classifier.

The model never writes text: it answers multiple-choice questions by the probability of a letter (one forward pass each,
batched), so there is nothing to parse and nothing to go wrong. Results go to delivery.json beside the book's other files."""
import json
import math
import re
import time
from pathlib import Path

MODEL = "Qwen/Qwen3-4B-Instruct-2507"      # measured against Qwen2.5-3B on a book with its question marks hidden: 89 of 91 found vs 90 of 100, fewer false alarms
EMOTIONS = ["neutral", "joy", "sadness", "anger", "fear", "surprise", "disgust"]
EMO_WORDS = ["calm, matter-of-fact", "happy, warm, amused", "sad, sorrowful, grieving", "angry, harsh, upset",
             "afraid, anxious, alarmed", "surprised, astonished", "disgusted, contemptuous"]
LEVELS = ["mild", "moderate", "intense"]
LETTERS = "ABCDEFGHIJ"
AUX = set(("is are was were am art do does did dost didst can could canst will wilt would shall shalt should may might must have has had hast hath "
           "isn't aren't wasn't weren't don't doesn't didn't can't won't wouldn't shouldn't couldn't haven't hasn't ain't isn’t aren’t wasn’t "
           "weren’t don’t doesn’t didn’t can’t won’t wouldn’t shouldn’t couldn’t haven’t hasn’t").split())
WH = set("who whom whose what where when why how which".split())
SUBJECT = set("you thou ye he she it they we i there this that these those the a an my your thy his her our their".split())
PRONOUN = set("you thou ye he she it they we i".split())
QUOTE_OPEN = "\"“‘'([ "


def looks_like_question(sentence: str) -> bool:
    """A sentence that is a question in form: an auxiliary then its subject ("Are you there"), or a question word then an
    auxiliary ("Where is your brother", "What time is it"). "When I awoke" and "What you will want we do not know" are not."""
    words = re.findall(r"[\w’']+", sentence.lstrip(QUOTE_OPEN))[:5]
    low = [w.lower() for w in words]
    if len(words) < 2:
        return False
    if low[0] in AUX:
        return low[1] in SUBJECT or words[1][0].isupper()
    if low[0] in WH:
        return any(low[k] in AUX and not any(w in PRONOUN for w in low[1:k]) for k in range(1, min(4, len(low))))
    return False


SENT_SPLIT = re.compile(r"(?<=[.!?])[\"”’']*\s+")
END_MARK = re.compile(r"([.…]*)([\"”’')\]]*)$")
EMOTIONAL_ABOVE_SPEECH, EMOTIONAL_ABOVE_NARRATION = 0.60, 0.75      # how sure the model must be that a passage is emotional
YES_ABOVE = 0.80          # how sure the model must be that a sentence is a question before its mark is changed
SYS_Q = "You are a careful copy editor. You decide whether a sentence is a direct question that is being asked."
SYS_E = ("You are an experienced audiobook director. You decide how a passage should be performed by the voice actor, reading it "
         "in the context of the scene and who is speaking.")


def sentences(chunk: str) -> list[str]:
    return [s for s in SENT_SPLIT.split(chunk) if s.strip()]


def question_candidates(chunk: str) -> list[int]:
    """Sentences that read like a question (they open like one) but do not end in a question mark."""
    out = []
    for i, s in enumerate(sentences(chunk)):
        t = s.strip()
        if looks_like_question(t) and re.search(r"[A-Za-z0-9][.…]*[\"”’')\]]*$", t) and "?" not in t[-3:] and not t.rstrip("\"”’')]").endswith(("!", ",", ";", ":", "—", "-")):
            out.append(i)
    return out


def with_question_marks(chunk: str, which: set[int]) -> str:
    """The chunk with the final mark of the chosen sentences made a "?". Words are not touched."""
    parts, out = SENT_SPLIT.split(chunk), []
    seps = [m.group(0) for m in SENT_SPLIT.finditer(chunk)]
    for i, p in enumerate(parts):
        if i in which and p.strip():
            p = END_MARK.sub(lambda m: "?" + m.group(2), p.rstrip(), count=1) if p.rstrip() else p
        out.append(p + (seps[i] if i < len(seps) else ""))
    return "".join(out)


class Scorer:
    """Qwen (or another chat model) used as a letter-probability classifier, batched."""

    def __init__(self, model: str = MODEL, device_pref: str = "auto"):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        from .tts.device import pick_device
        self.device = pick_device(device_pref)
        self.tok = AutoTokenizer.from_pretrained(model, padding_side="left")
        self.model = AutoModelForCausalLM.from_pretrained(
            model, dtype=torch.bfloat16 if self.device != "cpu" else torch.float32).to(self.device).eval()
        self.ids = [self.tok.encode(l, add_special_tokens=False)[0] for l in LETTERS]

    def close(self) -> None:
        import gc
        del self.model
        gc.collect()
        try:
            import torch
            if hasattr(torch, "xpu") and torch.xpu.is_available():
                torch.xpu.empty_cache()
        except Exception:
            pass

    def letters(self, items: list[tuple[str, str]], n_options: int, batch: int = 4, progress=None) -> list[list[float]]:
        """items: [(system, user)] where each user text ends with its lettered options. Returns probabilities over the letters."""
        import torch
        texts = [self.tok.apply_chat_template([{"role": "system", "content": s}, {"role": "user", "content": u}],
                                              tokenize=False, add_generation_prompt=True) for s, u in items]
        order = sorted(range(len(texts)), key=lambda i: len(texts[i]))          # similar lengths together: little padding
        out = [None] * len(texts)
        for b in range(0, len(order), batch):
            idx = order[b:b + batch]
            enc = self.tok([texts[i] for i in idx], return_tensors="pt", padding=True).to(self.device)
            with torch.no_grad():                     # only the last position is needed: skips a (batch x length x 150k) logits tensor
                logits = self.model(**enc, logits_to_keep=1).logits[:, -1].float()
            probs = torch.softmax(logits[:, self.ids[:n_options]], dim=-1).cpu().tolist()
            for i, p in zip(idx, probs):
                out[i] = p
            if progress and (b // batch) % 25 == 0:
                progress(b, len(order))
        return out


def _listing(options: list[str]) -> str:
    return "\n".join(f"{LETTERS[i]}. {o}" for i, o in enumerate(options))


def run(work: Path, cfg: dict, scorer=None, progress=print) -> dict:
    """Read the book, write delivery.json, return the summary. `scorer` is replaceable for tests."""
    from .lexicon import normalize
    from .synth import chunk_text, load_segments
    work = Path(work)
    t0 = time.time()
    segs = load_segments(work, cfg)
    max_chars = cfg.get("max_chunk_chars", 300)
    raw = [chunk_text(normalize(s["text"]), max_chars) for s in segs]
    flat = [(i, j, c) for i, cs in enumerate(raw) for j, c in enumerate(cs)]
    own = scorer is None
    scorer = scorer or Scorer(cfg.get("refine_model", MODEL), cfg.get("device", "auto"))
    try:
        # 1. questions
        qitems, qkeys = [], []
        for i, j, c in flat:
            sents = sentences(c)
            for k in question_candidates(c):
                before = " ".join(sents[max(0, k - 1):k])
                qitems.append((SYS_Q, f"Context: {before}\n\nSentence: {sents[k]}\n\nIs the sentence a direct question being asked "
                                      f"(not a statement, a command or a report of a question)?\n{_listing(['Yes, it is a question', 'No'])}\n"
                                      "Answer with a single letter."))
                qkeys.append((i, j, k))
        qprob = scorer.letters(qitems, 2, progress=lambda d, t: progress(f"  questions {d}/{t}")) if qitems else []
        fixes: dict[str, dict] = {}
        picked: dict[tuple[int, int], set[int]] = {}
        for (i, j, k), p in zip(qkeys, qprob):
            if p[0] >= YES_ABOVE:
                picked.setdefault((i, j), set()).add(k)
        for (i, j), ks in picked.items():
            fixed = with_question_marks(raw[i][j], ks)
            if fixed != raw[i][j]:
                fixes[f"{segs[i]['id']}:{j}"] = {"from": raw[i][j], "to": fixed}
        # 2. delivery: first "is this passage performed plainly or with feeling?", then which feeling, then how strongly.
        #    (Asking for the feeling straight away made the model call almost everything emotional: six feelings against one "calm".)
        def context(n):
            return " ".join(x[2] for x in flat[max(0, n - 3):n])[-500:] or "(the start)"

        def who(i):
            return segs[i]["speaker"] + (" (speaking aloud)" if segs[i].get("kind") == "dialogue" else " (narration)")
        gate_opts = ["Plain, even delivery: narration, description or ordinary conversation",
                     "Noticeably emotional or emphatic delivery: fear, anger, grief, joy, shock, urgency"]
        gitems = [(SYS_E, f"Earlier in the scene: {context(n)}\n\nPassage, read by {who(i)}:\n{c}\n\nHow should this passage be performed?\n"
                          f"{_listing(gate_opts)}\nAnswer with a single letter.") for n, (i, j, c) in enumerate(flat)]
        gprob = scorer.letters(gitems, 2, progress=lambda d, t: progress(f"  plain or emotional {d}/{t}")) if gitems else []
        felt = [n for n, (i, j, c) in enumerate(flat)
                if len(c.split()) >= 4 and gprob[n][1] >= (EMOTIONAL_ABOVE_SPEECH if segs[i].get("kind") == "dialogue" else EMOTIONAL_ABOVE_NARRATION)]
        eprob = scorer.letters([(SYS_E, f"Earlier in the scene: {context(n)}\n\nPassage, read by {who(flat[n][0])}:\n{flat[n][2]}\n\n"
                                       f"Which feeling should the voice carry?\n{_listing(EMO_WORDS[1:])}\nAnswer with a single letter.") for n in felt],
                               len(EMOTIONS) - 1, progress=lambda d, t: progress(f"  which feeling {d}/{t}")) if felt else []
        items: dict[str, dict] = {}
        # Strength from the text itself: the model's confidence saturates near 1.0 and asking "mild / moderate / intense" always got
        # "intense". Speech is moderate, loud with "!" or capitals; narration is mild (moderate with "!" or capitals).
        def strength(i, c):
            loud = "!" in c or re.search(r"\b[A-Z]{3,}\b", c) is not None
            return (3 if loud else 2) if segs[i].get("kind") == "dialogue" else (2 if loud else 1)
        emotional = {n: (EMOTIONS[1 + max(range(6), key=ep.__getitem__)], strength(flat[n][0], flat[n][2]), ep) for n, ep in zip(felt, eprob)}
        for n, (i, j, c) in enumerate(flat):
            e = {"emo": "neutral", "conf": round(gprob[n][0], 2)}
            if n in emotional:
                emo, lvl, ep = emotional[n]
                e = {"emo": emo, "lvl": lvl, "conf": round(gprob[n][1], 2)}
            items[f"{segs[i]['id']}:{j}"] = e
    finally:
        if own:
            scorer.close()
    report = {"model": cfg.get("refine_model", MODEL), "made": time.strftime("%Y-%m-%d %H:%M"), "seconds": round(time.time() - t0),
              "chunks": len(flat), "fixes": fixes, "items": items}
    (work / "delivery.json").write_text(json.dumps(report, ensure_ascii=False, indent=1))
    kinds = {k: sum(1 for v in items.values() if v["emo"] == k) for k in EMOTIONS}
    progress(f"[refine] {len(flat)} passages in {report['seconds']} s: {len(fixes)} question mark(s) added; "
             + ", ".join(f"{k} {n}" for k, n in kinds.items() if n))
    return report


def load(work: Path) -> dict:
    try:
        return json.loads((Path(work) / "delivery.json").read_text())
    except (OSError, ValueError):
        return {}


def apply_fixes(raw: list[list[str]], segs: list[dict], data: dict) -> list[list[str]]:
    """Put the added question marks into the chunk texts (only where the chunk is still the one that was read)."""
    fixes = data.get("fixes") or {}
    for i, cs in enumerate(raw):
        for j, c in enumerate(cs):
            f = fixes.get(f"{segs[i]['id']}:{j}")
            if f and f["from"] == c:
                raw[i][j] = f["to"]
    return raw


def hints(segs: list[dict], raw: list[list[str]], data: dict) -> list[list[dict | None]]:
    """The model's delivery for each chunk, shaped like `raw`."""
    items = data.get("items") or {}
    return [[items.get(f"{segs[i]['id']}:{j}") for j, _ in enumerate(cs)] for i, cs in enumerate(raw)]
