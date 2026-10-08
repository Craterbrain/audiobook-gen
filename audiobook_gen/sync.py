"""Find the ebook text that was spoken in a clip, so the voice cloner gets the exact words.

Whisper's transcript of a short clip is fuzzy-matched against the ebook: 3-gram anchors say where in the book
the words are, difflib aligns that stretch, and the ebook's own text (punctuation included) between the first
and last matched word is returned along with a similarity score."""
import bisect
import re
from collections import defaultdict
from difflib import SequenceMatcher

TOK = re.compile(r"[a-z0-9]+(?:'[a-z]+)?")


def tokenize(text: str) -> list[tuple[str, int, int]]:
    t = text.lower().replace("’", "'")
    return [(m.group(), m.start(), m.end()) for m in TOK.finditer(t)]


def book_index(chapters: list[dict]) -> dict:
    """Flatten the book into one token list, remembering each token's chapter and character span."""
    toks, spans = [], []
    for ci, ch in enumerate(chapters):
        for t, s, e in tokenize(ch["text"]):
            toks.append(t)
            spans.append((ci, s, e))
    return {"tokens": toks, "spans": spans, "chapters": chapters}


def anchors(asr: list[str], book: list[str], k: int = 3) -> list[tuple[int, int]]:
    """Increasing chain of (asr index, book index) pairs whose k-grams match (longest increasing subsequence)."""
    index = defaultdict(list)
    for i in range(len(book) - k + 1):
        index[tuple(book[i:i + k])].append(i)
    hits = []
    for j in range(len(asr) - k + 1):
        pos = index.get(tuple(asr[j:j + k]), [])
        if 0 < len(pos) <= 8:  # skip refrains that occur everywhere
            hits += [(j, i) for i in pos]
    if not hits:
        return []
    hits.sort(key=lambda h: (h[0], -h[1]))
    tails, tail_idx, prev = [], [], [-1] * len(hits)
    for n, (_, i) in enumerate(hits):
        k_ = bisect.bisect_left(tails, i)
        if k_ == len(tails):
            tails.append(i); tail_idx.append(n)
        else:
            tails[k_] = i; tail_idx[k_] = n
        prev[n] = tail_idx[k_ - 1] if k_ else -1
    chain, n = [], tail_idx[-1]
    while n != -1:
        chain.append(hits[n]); n = prev[n]
    return chain[::-1]


def match(asr_text: str, idx: dict, only_chapter: int | None = None) -> dict | None:
    """Best ebook passage for a transcript: {chapter, text, similarity, coverage} or None if nothing lines up.
    similarity = share of the spoken words found, in order, in the book passage;
    coverage   = how much of the book passage those words account for (low = the passage is longer than the clip)."""
    a = [t for t, _, _ in tokenize(asr_text)]
    if len(a) < 4:
        return None
    toks, spans = idx["tokens"], idx["spans"]
    lo_b, hi_b = 0, len(toks)
    if only_chapter is not None:
        pos = [i for i, sp in enumerate(spans) if sp[0] == only_chapter]
        if not pos:
            return None
        lo_b, hi_b = pos[0], pos[-1] + 1
    chain = anchors(a, toks[lo_b:hi_b])
    if not chain:
        return None
    first_i, first_j = chain[0][1] + lo_b, chain[0][0]
    last_i, last_j = chain[-1][1] + lo_b, chain[-1][0]
    lo = max(0, first_i - first_j - 6)
    hi = min(len(toks), last_i + (len(a) - last_j) + 6)
    window = toks[lo:hi]
    blocks = [b for b in SequenceMatcher(None, a, window, autojunk=False).get_matching_blocks() if b.size]
    if not blocks:
        return None
    matched = sum(b.size for b in blocks)
    b0, b1 = lo + blocks[0].b, lo + blocks[-1].b + blocks[-1].size - 1  # book token range
    ci = spans[b0][0]
    if spans[b1][0] != ci:  # clip straddles a chapter break: keep the first part only
        b1 = max(i for i in range(b0, b1 + 1) if spans[i][0] == ci)
    text = idx["chapters"][ci]["text"]
    s, e = spans[b0][1], spans[b1][2]
    tail = re.match(r"""["”’')\]]*[.!?;:,]?["”’')\]]*""", text[e:e + 4])
    snippet = re.sub(r"\s+", " ", text[s:e + (len(tail.group()) if tail else 0)]).strip()
    return {"chapter": ci, "text": snippet, "similarity": matched / len(a),
            "coverage": matched / max(1, b1 - b0 + 1)}
