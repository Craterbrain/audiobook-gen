"""Score speaker attribution against hand-labelled quotes. Run: python tests/eval_attribution.py"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def score(segments_path: Path, gold_path: Path, verbose=False) -> tuple[int, int]:
    segs = [s for s in json.loads(segments_path.read_text()) if s["kind"] == "dialogue"]
    gold = json.loads(gold_path.read_text())
    ok = total = 0
    for prefix, role in gold.items():
        hits = [s for s in segs if s["text"].startswith(prefix)]
        if not hits:
            continue
        total += 1
        got = hits[0]["speaker"]
        ok += got == role
        if verbose and got != role:
            print(f"  ✗ {prefix!r}: got {got}, want {role}")
    return ok, total


if __name__ == "__main__":
    runs = [("monte_cristo_ch5", "gold_monte_cristo.json"), ("monte_cristo_ch3", "gold_monte_cristo_ch3.json")]
    tot = [0, 0]
    for work, gold in runs:
        ok, n = score(ROOT / "work" / work / "segments.json", ROOT / "tests" / gold, True)
        print(f"{work}: {ok}/{n} = {ok / n:.0%}")
        tot[0] += ok; tot[1] += n
    print(f"TOTAL {tot[0]}/{tot[1]} = {tot[0] / tot[1]:.0%}")
