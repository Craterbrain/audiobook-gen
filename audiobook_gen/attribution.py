"""Global speaker attribution for quoted prose.

Per-quote evidence (explicit "said X", pronoun gender, names being addressed, who was just
mentioned) is combined with dialogue structure (turn-taking, split quotes, A-B-A alternation)
and decoded jointly with Viterbi, instead of deciding each quote in isolation.

Each item: {para, text, explicit: role|None, pron: 'm'|'f'|'n'|None, split, cont, other: bool, lead,
            addressed: set[role], recent: set[role], cands: list[role]}
"""
import math
import re
from collections import Counter

W = dict(explicit=12.0, explicit_miss=-6.0, gender_ok=2.5, gender_bad=-3.5, addressed=-3.5,
         recent=1.2, same_para_repeat=-0.5, new_para_repeat=-2.5, aba=2.0, replies_to_addressed=2.0,
         split_same=6.0, split_diff=-6.0, group_other=-1.0, cont_same=4.0, cont_diff=-2.0,
         other_ok=4.0, other_bad=-3.0, lead=0.6, prior=0.7)
GROUP_ROLES = {"Crowd", "Guest", "Guests", "Servants", "Officers", "Jews", "Disciples", "Pharisees"}


def infer_genders(text: str, matchers: list[tuple[str, str]]) -> dict[str, str]:
    """role -> 'm'/'f' from the first he/she pronoun that follows each mention within a sentence."""
    from collections import Counter
    votes: dict[str, Counter] = {}
    for rx, role in matchers:
        for m in re.finditer(rx, text):
            tail = re.split(r"[.!?]", text[m.end():m.end() + 140], maxsplit=1)[0]
            p = re.search(r"\b(he|his|him|himself|she|her|herself)\b", tail, re.I)
            if p:
                votes.setdefault(role, Counter())["f" if p.group(1).lower().startswith(("she", "her")) else "m"] += 1
    return {r: c.most_common(1)[0][0] for r, c in votes.items() if sum(c.values()) >= 2}


def _emit(it: dict, role: str, gender: dict, prior: dict) -> float:
    s = W["prior"] * math.log1p(prior.get(role, 0))
    if it["explicit"]:
        s += W["explicit"] if role == it["explicit"] else W["explicit_miss"]
    elif it["pron"] in ("m", "f"):
        g = gender.get(role)
        if g:
            s += W["gender_ok"] if g == it["pron"] else W["gender_bad"]
        if role in GROUP_ROLES:
            s += W["group_other"]
    if role in it["addressed"] and role != it["explicit"]:
        s += W["addressed"]
    if role in it["recent"]:
        s += W["recent"]
    if role == it.get("lead") and it["pron"] in ("m", "f") and not it["explicit"]:
        s += W["lead"]  # pronoun right after a narration paragraph about this character
    return s


def _trans(prev: str, cur: str, p: dict, c: dict, prev2: str | None) -> float:
    same_para = p["para"] == c["para"]
    if c["split"] and same_para:
        return W["split_same"] if cur == prev else W["split_diff"]
    s = 0.0
    if c.get("other"):  # "answered the other": the counterpart of the last exchange
        s += W["other_ok"] if (cur != prev and (cur == prev2 or cur in p["addressed"])) else W["other_bad"]
        return s
    if same_para and c.get("cont"):  # quote after an attributed quote, no new subject
        return W["cont_same"] if cur == prev else W["cont_diff"]
    if cur == prev:
        s += W["same_para_repeat"] if same_para else W["new_para_repeat"]
    elif prev2 == cur and not same_para:
        s += W["aba"]
    if cur in p["addressed"] and cur != prev:
        s += W["replies_to_addressed"]
    return s


def _lse(xs):
    m = max(xs)
    return m + math.log(sum(math.exp(x - m) for x in xs))


def decode(items: list[dict], gender: dict[str, str], extra: dict[int, dict[str, float]] | None = None,
           posteriors: bool = False):
    """Viterbi over each quote's candidate roles; returns one role per item.

    extra: {item index: {role: added log-score}} (e.g. from the LLM judge).
    posteriors=True also returns, per item, {role: probability} from forward-backward, so callers
    can tell which quotes the structural evidence leaves ambiguous."""
    if not items:
        return ([], []) if posteriors else []
    extra = extra or {}
    prior = Counter(it["explicit"] for it in items if it["explicit"])  # how often each role is named as speaker
    cands = []
    for it in items:
        c = list(dict.fromkeys(([it["explicit"]] if it["explicit"] else []) + it["cands"]))
        cands.append(c or ["Speaker A"])
    emit = lambda i, r: _emit(items[i], r, gender, prior) + extra.get(i, {}).get(r, 0.0)
    # state = (role, role_two_back) so A-B-A alternation can be rewarded
    best = {(r, None): (emit(0, r), None) for r in cands[0]}
    back = [best]
    for i in range(1, len(items)):
        nxt = {}
        for cur in cands[i]:
            e = emit(i, cur)
            for (prev, prev2), (score, _) in back[-1].items():
                sc = score + e + _trans(prev, cur, items[i - 1], items[i], prev2)
                key = (cur, prev)
                if key not in nxt or sc > nxt[key][0]:
                    nxt[key] = (sc, (prev, prev2))
        back.append(nxt)
    state = max(back[-1], key=lambda k: back[-1][k][0])
    out = [state[0]]
    for i in range(len(items) - 1, 0, -1):
        state = back[i][state][1]
        out.append(state[0])
    out = out[::-1]
    if not posteriors:
        return out

    # forward-backward (sum-product) for per-quote role probabilities
    fwd = [{(r, None): emit(0, r) for r in cands[0]}]
    for i in range(1, len(items)):
        nxt: dict = {}
        for cur in cands[i]:
            e = emit(i, cur)
            for (prev, prev2), sc in fwd[-1].items():
                nxt.setdefault((cur, prev), []).append(sc + e + _trans(prev, cur, items[i - 1], items[i], prev2))
        fwd.append({k: _lse(v) for k, v in nxt.items()})
    bwd = [None] * len(items)
    bwd[-1] = {k: 0.0 for k in fwd[-1]}
    for i in range(len(items) - 2, -1, -1):
        cur_b = {}
        for (cur, prev) in fwd[i]:
            terms = []
            for nxt_role in cands[i + 1]:
                key = (nxt_role, cur)
                if key in bwd[i + 1]:
                    terms.append(emit(i + 1, nxt_role) + _trans(cur, nxt_role, items[i], items[i + 1], prev)
                                 + bwd[i + 1][key])
            cur_b[(cur, prev)] = _lse(terms) if terms else 0.0
        bwd[i] = cur_b
    post = []
    for i in range(len(items)):
        by_role: dict = {}
        for k, f in fwd[i].items():
            by_role.setdefault(k[0], []).append(f + bwd[i][k])
        lp = {r: _lse(v) for r, v in by_role.items()}
        z = _lse(list(lp.values()))
        post.append({r: math.exp(v - z) for r, v in lp.items()})
    return out, post
