import json
from audiobook_gen import synth, respell_check as rc, respell_rules as rr
from audiobook_gen.voices import resolve
d = json.load(open('data/bible_ipa.json'))['entries']
plain = {x['name']: x['score'] for x in json.load(open('work/respell_run1.json'))['plain']}
bad = [n for n in plain if plain[n] < .8]; good = [n for n in plain if plain[n] >= .8]
train = bad[0::2] + good[0::6]; test = bad[1::2]; regress = good[3::6]
eng = synth.get_engine('chatterbox'); voice = resolve({"engine": "chatterbox", "library": "Walter"})
def run(names, opt):
    res = rc.check([(n, rr.respell(d[n]['ipa'], n, opt), d[n]['ipa']) for n in names], voice, eng, progress=lambda m: None)
    return sum(x['score'] for x in res) / len(res), sum(x.get('pauses', 0) for x in res)
opt = {**rr.OPTIONS, **json.load(open('data/respell_options.json'))}
best, p = run(train, opt)
print(f"start: train {best:.3f} pauses {p}", flush=True)
for sweep in range(3):
    changed = False
    for key, choices in rr.CHOICES.items():
        for c in choices:
            if c == opt[key]: continue
            s, p = run(train, {**opt, key: c})
            if s > best + 0.003:
                best, opt[key], changed = s, c, True
                print(f"  sweep {sweep}: {key} -> {c!r}  train {s:.3f} pauses {p}", flush=True)
    if not changed: break
json.dump(opt, open('data/respell_options.json', 'w'), indent=1)
for label, names in (("train", train), ("TEST (unseen failures)", test), ("regress (unseen good)", regress)):
    s, p = run(names, opt); print(f"{label}: {s:.3f} pauses {p} of {len(names)}", flush=True)
print(f"plain: test {sum(plain[n] for n in test)/len(test):.3f} regress {sum(plain[n] for n in regress)/len(regress):.3f}", flush=True)
print(json.dumps(opt), flush=True)
