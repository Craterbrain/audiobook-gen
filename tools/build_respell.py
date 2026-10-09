import json, random, re, sys, time
from pathlib import Path
from audiobook_gen import synth, respell_check as rc, respell_rules as rr
from audiobook_gen.voices import resolve
from audiobook_gen.bible_dict import _read_pairs
N, THRESH = 400, 0.70
d = json.load(open('data/bible_ipa.json'))['entries']
top = [n for n, _ in sorted(d.items(), key=lambda x: -x[1]['count'])[:N]]
opt = {**rr.OPTIONS, **json.load(open('data/respell_options.json'))}
hand = {}
for f in sorted(Path('data').glob('bible_respell_claude_*.txt')): hand.update(_read_pairs(f))
verses = [re.sub(r'^.*?\d+:\d+\s+', '', l).replace('*', '') for l in open(sys.argv[1] if len(sys.argv) > 1 else 'bible_text.txt', encoding='utf-8').read().splitlines() if re.search(r'\d+:\d+\s', l)]
def frames(name, count=2):
    rx = re.compile(r'\b' + re.escape(name) + r'\b'); order = list(range(len(verses))); random.Random(name).shuffle(order); out = []
    for i in order:
        v = verses[i]; m = rx.search(v)
        if m and 30 <= len(v) <= 160 and not rx.search(v, m.end()):
            a = max(0, v.rfind(',', 0, m.start()) + 1) if m.start() > 60 else 0
            ph = v[a:].strip(); s = ph.find(m.group())
            if len(ph) > s + len(m.group()) + 55: ph = ph[:s + len(m.group()) + 55].rsplit(' ', 1)[0]
            if m.group() in ph and '{' not in ph and '}' not in ph:
                out.append(ph.replace(m.group(), '{}', 1))
                if len(out) == count: return out
    return out
eng = synth.get_engine('chatterbox'); voice = resolve({"engine": "chatterbox", "library": "Walter"})
F = {n: frames(n) for n in top}
def score(cands):      # {name: text} -> {name: mean score over its phrases}
    items = [(n, t, d[n]['ipa'], f) for n, t in cands.items() for f in F[n]]
    res = rc.check(items, voice, eng, progress=lambda m: print(' ', m, flush=True))
    acc = {}
    for x in res: acc.setdefault(x['name'], []).append(x['score'] - 0.0)
    return {n: sum(v) / len(v) for n, v in acc.items()}
names = [n for n in top if F[n]]
print(len(names), "names with phrases", flush=True)
sp = score({n: n for n in names})
low = [n for n in names if sp[n] < THRESH]
print(f"plain below {THRESH}: {len(low)} of {len(names)}", flush=True)
cands = {"tuned": {n: rr.respell(d[n]['ipa'], n, opt) for n in low}, "hand": {n: hand[n] for n in low if n in hand}}
sc = {k: score(v) for k, v in cands.items() if v}
entries = {}
for n in low:
    options = [(sp[n], n, "plain")] + [(sc[k][n], cands[k][n], k) for k in sc if n in sc[k]]
    s, text, src = max(options)
    if src != "plain":
        entries[n] = {"respell": text, "score": round(s, 2), "plain_score": round(sp[n], 2), "source": src}
meta = {"what": "Respellings for Chatterbox: used only where plain spelling scored under 0.70 when spoken in two Bible phrases and transcribed to phonemes",
        "method": "see audiobook_gen/respell_check.py", "threshold": THRESH, "names_tested": len(names), "built": time.strftime("%Y-%m-%d"),
        "license": "CC BY-SA 4.0 (derived from Wiktionary/WikiPron IPA)"}
Path('data/bible_respell.json').write_text(json.dumps({"_meta": meta, "entries": entries}, ensure_ascii=False, indent=1))
allsc = [max([sp[n]] + [sc[k][n] for k in sc if n in sc[k]]) if n in low else sp[n] for n in names]
print(f"plain mean {sum(sp.values())/len(sp):.3f} -> with overrides {sum(allsc)/len(allsc):.3f}; overrides {len(entries)}; names >= .7: {sum(x >= THRESH for x in allsc)}/{len(names)}", flush=True)
