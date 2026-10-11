"""How does the voice read dialect words? Speak each word with its apostrophe and without it, score both against the word's IPA
(the same phoneme check as the name pass) and print which form wins. Run it for a book's voice before deciding on a default rewrite.

    python tools/dialect_baseline.py --config work/<book>/config.yaml [--n 40] [--seeds 3]
Needs the GPU free (stop the queue's book first). Writes work/dialect_baseline.json."""
import argparse
import json
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from audiobook_gen import dialect, respell_check          # noqa: E402
from audiobook_gen.synth import get_engine, resolve_voice  # noqa: E402

FRAME = "He looked at me and said, {}, and then he left."


def variants(token: str) -> dict[str, str]:
    bare = token.replace("'", "")
    return {"as written": token, "no apostrophe": bare}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--seeds", type=int, default=3)
    a = ap.parse_args()
    cfg = yaml.safe_load(Path(a.config).read_text())
    voice = resolve_voice("Narrator", cfg)
    engine = get_engine(voice["engine"], cfg.get("device", "auto"), 0, cfg.get("f5_precision", "float16"))
    words = list(dialect.sounds().items())[:a.n]
    rows = []
    for token, (ipa, _) in words:
        row = {"token": token, "ipa": ipa}
        for label, text in variants(token).items():
            scores = [respell_check.check([(token, text, ipa, FRAME)], voice, engine, progress=lambda *_: None, seed=s)[0]["score"]
                      for s in range(1, a.seeds + 1)]
            row[label] = round(sum(scores) / len(scores), 2)
        rows.append(row)
        print(f"{token:12} written {row['as written']:.2f}   bare {row['no apostrophe']:.2f}   {ipa}", flush=True)
    wins = sum(r["no apostrophe"] > r["as written"] + 0.05 for r in rows)
    loses = sum(r["as written"] > r["no apostrophe"] + 0.05 for r in rows)
    print(f"\nbare spelling better for {wins}, apostrophe better for {loses}, of {len(rows)}")
    (ROOT / "work" / "dialect_baseline.json").write_text(json.dumps(rows, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
