"""Turn pause measurements (tools/pause_emotion.py -> work/pause_emotion*.json) into data/narrator_pacing.json, the table the
assembler draws its pauses from. Each boundary kind gets: the share of pauses that are almost none, and a log-normal fit of the rest.
Usage: python tools/fit_pacing.py [work/pause_emotion_partial.json]"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
SHORT = 100          # ms: below this a pause is "almost none"
KINDS = {            # name -> which measured boundaries it is made from
    "sentence_narration": lambda r: r["base"] in ("period", "question", "exclaim") and not r["in_quote"] and "closequote" not in r["cls"] and "openquote" not in r["cls"],
    "sentence_dialogue": lambda r: r["base"] in ("period", "question", "exclaim") and (r["in_quote"] or "closequote" in r["cls"]) and "openquote" not in r["cls"],
    "paragraph": lambda r: r["base"] == "paragraph",
    "comma_narration": lambda r: r["base"] == "comma" and not r["in_quote"] and "closequote" not in r["cls"],
    "comma_dialogue": lambda r: r["base"] == "comma" and r["in_quote"] and "closequote" not in r["cls"],
    "before_tag": lambda r: r["base"] == "comma" and "closequote" in r["cls"],
    "speaker_change": lambda r: r["base"] in ("period", "question", "exclaim") and "openquote" in r["cls"],
}


def main() -> None:
    src = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "work" / "pause_emotion_partial.json"
    recs = json.loads(src.read_text())
    table = {}
    for name, pick in KINDS.items():
        v = np.array([r["pause"] for r in recs if pick(r)], dtype=float)
        if len(v) < 15:
            print(f"{name}: only {len(v)} measurements, skipped"); continue
        short = v < SHORT
        long = v[~short]
        entry = {"n": int(len(v)), "short_share": round(float(short.mean()), 3), "median_ms": round(float(np.median(v))),
                 "short_range_ms": [20, SHORT]}
        if len(long) >= 8:
            logs = np.log(long)
            entry.update(mu=round(float(logs.mean()), 4), sigma=round(float(logs.std()), 4),
                         lo_ms=round(float(np.percentile(long, 3))), hi_ms=round(float(np.percentile(long, 97))))
        table[name] = entry
        print(f"{name:20s} n={entry['n']:4d}  almost none {100 * entry['short_share']:3.0f}%  rest median {np.exp(entry.get('mu', 0)):5.0f} ms (sigma {entry.get('sigma', 0):.2f})")
    emotion = {}                  # how much the feeling of the sentence before shifts a pause: log(pause) changes by coefficient x feel (0-1)
    for name, pick in (("sentence", lambda r: r["base"] in ("period", "question", "exclaim")), ("paragraph", lambda r: r["base"] == "paragraph")):
        sub = [r for r in recs if pick(r) and "feel" in r]
        if len(sub) < 100:
            continue
        X = np.array([[1, r["feel"], np.log(r["words"] + 1), 1.0 if (r["in_quote"] or "closequote" in r["cls"]) else 0.0] for r in sub])
        y = np.log1p(np.array([r["pause"] for r in sub]))
        beta = np.linalg.lstsq(X, y, rcond=None)[0]
        resid = y - X @ beta
        se = np.sqrt(np.diag(resid @ resid / (len(y) - X.shape[1]) * np.linalg.inv(X.T @ X)))
        emotion[name] = {"coefficient": round(float(beta[1]), 3), "t": round(float(beta[1] / se[1]), 1), "n": len(sub)}
        print(f"emotion on {name} pauses: {beta[1]:+.2f} per unit of feeling (t={beta[1] / se[1]:+.1f}, n={len(sub)})")
    out = {"about": "Pauses between spoken units measured on one professional narrator (about 90 minutes of audio matched to the book text). "
                    "Numbers only. Refit with tools/fit_pacing.py.", "kinds": table, "emotion": emotion}
    (ROOT / "data" / "narrator_pacing.json").write_text(json.dumps(out, indent=1))
    print("-> data/narrator_pacing.json")


if __name__ == "__main__":
    main()
