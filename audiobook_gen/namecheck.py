"""Name pronunciation pass, between the quality check and the build.

Chatterbox reads plain letters, so a name can come out wrong and there is no way to tell it the sound. This pass listens instead:
every clip that says a name from the lexicon is written down as phonemes (the wav2vec2 espeak recogniser, as in respell_check) and the
stretch where the name is spoken is scored against the name's IPA (0 to 1, from the dictionary or what you typed; names with no IPA
from a real source are skipped, never guessed from the spelling).

  * A clip with a name scoring under GOOD (0.70) is made again with other seeds until every name in it reaches GOOD; the best attempt
    stays in place, so the improvement survives later rebuilds.
  * The worse a take sounds, the sooner a respelling is tried instead of another seed. Counting the takes of one clip (the first one and
    its retries): one take under 30% is enough, two under 40%, three under 50%, four under 60%, five under 70% (BANDS). Candidate
    spellings are then spoken in the book's voice and scored the same way, and the best one is used if it clearly beats the plain
    spelling (lexicon.json: respell, respell_src "checked"). The clips with that name are made again with the new spelling.

Everything is written to name_report.json. A name you gave a respelling yourself is never re-spelled. Only Chatterbox clips are
judged (Kokoro gets the IPA directly)."""
import hashlib
import json
import re
import shutil
import time
from pathlib import Path

import numpy as np
import soundfile as sf

GOOD = 0.70
SEEDS = (101, 202, 303, 404, 505, 606)
KINDS = ("name", "place", "demonym", "foreign", "dialect")
BANDS = ((0.3, 1), (0.4, 2), (0.5, 3), (0.6, 4), (0.7, 5))     # (score, how many takes under it make a respelling due)
DUR_RANGE = (0.70, 1.30)   # a new take must last about as long as the one it replaces...
RMS_DB = 4.0               # ...and be about as loud, so a take with a runaway delivery (shouting, rushing, a drawl) is never swapped in
WIN = 0.30                 # a name is looked for within this share of the clip around where it should be
MARGIN = 0.15              # a respelling must beat the plain spelling by this much to be adopted
ROUNDS = 2
LETTER = "A-Za-zÀ-ÖØ-öø-ÿ"


def targets(work: Path) -> list[dict]:
    """Lexicon entries that have an IPA from a real source (typed by you, the Bible dictionary, a dictionary), with a pattern that finds
    the name in spoken text. Nothing is written back to lexicon.json."""
    from .pronounce import fill_lexicon
    path = work / "lexicon.json"
    lex = json.loads(path.read_text()) if path.exists() else []
    ents = [dict(e) for e in lex if e.get("kind") in KINDS]
    for e in ents:                                          # an IPA with no recorded source, on a name whose language was guessed (de, fr, nl...), is a guess
        if e.get("lang") and not e.get("ipa_src") and e.get("source") != "user":                  # from that language's spelling rules: never judge by it
            e["ipa"] = ""
    fill_lexicon(ents, bible=True, offline=True)
    out = []
    for e in ents:
        ipa = (e.get("ipa") or "").strip()
        if not ipa or str(e.get("ipa_src", "")).startswith("guess"):
            continue
        out.append(e)
    return out


def pattern(spoken: str) -> re.Pattern:
    """The name as written in the clip's text, with its possessive."""
    body = re.escape(spoken).replace("\\ ", r"\s+").replace("'", "['’]")
    return re.compile(rf"(?<![{LETTER}])({body})(?:['’]s)?(?![{LETTER}])", re.I)


def spoken_form(entry: dict, work: Path, mode: str = "verified") -> str:
    """How the book's engine is given this name: the respelling if one is in use, else the name itself."""
    from .lexicon import load_preprocessor
    pre = load_preprocessor(work, mode)
    hit = pre.map.get(pre._key(entry["term"]))
    return hit[1] if hit else entry["term"]


def score_text(text: str, heard: str, times: list, ents: list[tuple[dict, re.Pattern]]) -> dict[str, float]:
    """Score each name in `text` against the phonemes `heard` (with their start times): the worst of its occurrences."""
    from .respell_check import judge
    toks = heard.split()
    n = len(toks)
    out = {}
    for e, rx in ents:
        scores = []
        for m in rx.finditer(text):
            centre = int(m.start() / max(1, len(text)) * n)
            w = max(10, int(WIN * n))
            a, b = max(0, centre - w), min(n, centre + w)
            scores.append(judge(e["ipa"], " ".join(toks[a:b]), times[a:b] if len(times) == n else None)[0])
        if scores:
            out[e["term"]] = min(scores)
    return out


def note(detail: str) -> None:
    """Say what the pass is doing, for the ntfy `current` reply."""
    try:
        from . import jobqueue
        jobqueue.activity_detail(detail)
    except Exception:
        pass


def known_good(entry: dict, fingerprint: str | None) -> bool:
    """A name to leave out of the check: marked good by you (`good <name>` on ntfy), or judged good before and its clips have not changed since
    (the fingerprint is the list of clip files that say it)."""
    ok = entry.get("name_ok")
    return bool(ok) and (bool(ok.get("manual")) or ok.get("fp") == fingerprint)


def mark_ok(work: Path, term: str, fingerprint: str = "", score: float = 1.0, manual: bool = False) -> bool:
    """Record a name as good in this book's lexicon. Returns False if the book has no such entry."""
    path = Path(work) / "lexicon.json"
    try:
        lex = json.loads(path.read_text())
    except (OSError, ValueError):
        return False
    hit = False
    for e in lex:
        if str(e.get("term", "")).lower() == term.lower():
            e["name_ok"] = {"score": round(score, 2), "fp": fingerprint, "manual": manual, "at": time.strftime("%Y-%m-%d")}
            hit = True
    if hit:
        path.write_text(json.dumps(lex, indent=2, ensure_ascii=False))
    return hit


def mark_good_everywhere(name: str) -> list[str]:
    """`good <name>`: leave this name out of the name check in every book that has it. Returns the books."""
    return [p.parent.name for p in sorted((ROOT / "work").glob("*/lexicon.json")) if mark_ok(p.parent, name, manual=True)]


def keep_gpu() -> None:
    """Renew the GPU lease if this process holds one (the queue forgets a lease that is not renewed for half an hour)."""
    try:
        from . import jobqueue
        if jobqueue.lease_mine():
            jobqueue.lease_touch()
    except Exception:
        pass


def hear(path: Path, cache: dict) -> tuple[str, list]:
    from .respell_check import phonemes
    st = path.stat()
    key = f"{path.name}|{st.st_size}|{st.st_mtime_ns}"
    if key not in cache:
        audio, sr = sf.read(path, dtype="float32")
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        heard, times = phonemes(audio, sr, timed=True)
        cache[key] = [heard, list(times)]
    return cache[key][0], cache[key][1]


def respell_due(takes: list[float], good: float = GOOD, bands: tuple = BANDS) -> bool:
    """True when a name's takes in one clip have failed often enough for how badly: 1 under 0.3, 2 under 0.4, ... 5 under 0.7."""
    return any(sum(1 for t in takes if t < b) >= need for b, need in bands if b <= good + 1e-9)


def _level(path: Path) -> float:
    """Loudness of a clip in dB (RMS over the whole file)."""
    a, _ = sf.read(path, dtype="float32")
    return float(20 * np.log10(np.sqrt(np.mean(np.square(a))) + 1e-9))


def _median(xs: list[float]) -> float:
    return float(np.median(xs)) if xs else 0.0


def pass_once(work: Path, cfg: dict, progress=print, good: float = GOOD, seeds: tuple | None = None, respell=None,
              bands: tuple = BANDS, skip_known: bool = True, skipped: list | None = None) -> dict:
    """Judge every clip that says a known name and make the weak ones again. `respell(term, ipa)` is called when a name's takes have
    failed often enough for how badly (respell_due); it returns True if a new spelling was adopted, and the clips of that name are then
    left for the next round (their text has changed). Returns {term: {...}} statistics per name."""
    from . import qc
    from .synth import get_engine, load_overrides
    seeds = seeds or SEEDS
    meta = json.loads((work / "clips_meta.json").read_text())
    ents = targets(work)
    by_term = {e["term"]: e for e in ents}
    pats = [(e, pattern(spoken_form(e, work))) for e in ents]
    chosen = load_overrides(work)
    cache_path = work / "namecheck_cache.json"
    try:
        cache = json.loads(cache_path.read_text())
    except (OSError, ValueError):
        cache = {}
    stats = {t: {"term": t, "ipa": e["ipa"], "ipa_src": e.get("ipa_src", ""), "clips": 0, "first": [], "tries": [], "final": [], "regenerated": 0}
             for t, e in by_term.items()}
    can_respell = {t for t, e in by_term.items() if respell and not e.get("respell_tried") and e.get("respell_src") not in ("user", "checked")
                   and not (e.get("source") == "user" and e.get("respell"))}
    adopted = set()                                        # names given a new spelling in this pass: their clips wait for the next round
    todo = []
    for f, info in meta.items():
        if info.get("engine") != "chatterbox" or not (work / "clips" / f).exists():
            continue
        here = [(e, rx) for e, rx in pats if rx.search(info["text"])]
        if here:
            todo.append((f, info, here))
    files: dict[str, list[str]] = {}
    for f, info, here in todo:
        for e, rx in here:
            files.setdefault(e["term"], []).append(f)
    prints = {t: hashlib.sha1("|".join(sorted(v)).encode()).hexdigest()[:12] for t, v in files.items()}
    left_out = {t for t, e in by_term.items() if skip_known and known_good(e, prints.get(t))}
    if left_out:                                            # names already known to be good: their clips are not listened to again
        todo = [(f, info, h2) for f, info, here in todo if (h2 := [(e, rx) for e, rx in here if e["term"] not in left_out])]
        if skipped is not None:
            skipped.extend(sorted(left_out & set(files)))
    progress(f"[names] {len(todo)} clips say a name that has an IPA ({len(ents)} such names in the lexicon"
             + (f", {len(left_out & set(files))} left out as known good" if left_out else "") + ")")
    engine = None

    def check_due(takes: dict) -> bool:
        """Ask for a respelling for each name whose takes in this clip have failed enough. True if one was adopted."""
        got = False
        for t, tk in takes.items():
            if t in can_respell and respell_due(tk, good, bands):
                can_respell.discard(t)                     # one search per name
                if respell(t, by_term[t]["ipa"]):
                    adopted.add(t)
                    got = True
        return got

    for n, (f, info, here) in enumerate(todo):
        keep_gpu()
        if n % 5 == 0:
            note(f"judging clip {n + 1} of {len(todo)}")
        here = [(e, rx) for e, rx in here if e["term"] not in adopted]
        if not here:
            continue
        path = work / "clips" / f
        heard, times = hear(path, cache)
        scores = score_text(info["text"], heard, times, here)
        if not scores:
            continue
        takes = {t: [s] for t, s in scores.items()}
        for t, s in scores.items():
            stats[t]["clips"] += 1
            stats[t]["first"].append(s)
            stats[t]["tries"].append(s)
        best, best_min = scores, min(scores.values())
        if best_min < good and info.get("key") not in chosen:
            stop = check_due(takes)
            engine = engine or get_engine("chatterbox", cfg.get("device", "auto"), 0, cfg.get("f5_precision", "float16"))
            orig_len = (qc.measure(path) or {}).get("seconds") or 0
            orig_db = _level(path)
            for seed in ([] if stop else seeds):
                keep_gpu()
                tmp = path.with_suffix(".try.wav")
                try:
                    sf.write(tmp, engine.synth(info["text"], {**info["voice"], "seed": seed}), engine.sample_rate)
                    m = qc.measure(tmp)
                    if "broken" in m or (orig_len and not DUR_RANGE[0] * orig_len <= m["seconds"] <= DUR_RANGE[1] * orig_len) or abs(_level(tmp) - orig_db) > RMS_DB:
                        tmp.unlink(missing_ok=True)
                        continue
                    h2, t2 = hear(tmp, cache)
                    s2 = score_text(info["text"], h2, t2, here)
                except Exception as ex:                                      # a failed attempt just does not count
                    progress(f"  attempt failed: {ex}")
                    tmp.unlink(missing_ok=True)
                    continue
                for t, s in s2.items():
                    stats[t]["tries"].append(s)
                    takes.setdefault(t, []).append(s)
                if s2 and min(s2.values()) > best_min:
                    keep = work / "clips_before_name_check" / f                # the take you had is kept, once, in case the new one is not better to the ear
                    if not keep.exists():
                        keep.parent.mkdir(exist_ok=True)
                        shutil.copy2(path, keep)
                    tmp.replace(path)
                    best, best_min = s2, min(s2.values())
                    for t in s2:
                        stats[t]["regenerated"] += 1
                else:
                    tmp.unlink(missing_ok=True)
                if best_min >= good or check_due(takes):
                    break
        for t, s in best.items():
            stats[t]["final"].append(s)
        if n % 25 == 0:
            print(f"[progress] {n + 1}/{len(todo)}", flush=True)
            cache_path.write_text(json.dumps(cache))
    cache_path.write_text(json.dumps(cache))
    for t, s in stats.items():
        s["fp"] = prints.get(t, "")
    return {t: s for t, s in stats.items() if s["clips"]}


ROOT = Path(__file__).resolve().parent.parent
HINTS = ROOT / "data" / "respell_hints.txt"
BATCHES = (5, 10, 15)       # candidate spellings spoken per round: five first, ten more if those were not enough, then fifteen more
TARGET = 0.80               # the first rounds aim for this: a spelling at this score that also beats the plain one ends the search early


def hints(term: str, work: Path | None = None) -> list[str]:
    """Spellings you suggested for a name (`Name|spelling` lines in data/respell_hints.txt, and in the book's own respell_hints.txt).
    They are always spoken and scored in the first round, alongside the generated ones, and win only if they sound better."""
    out = []
    for path in (HINTS, (Path(work) / "respell_hints.txt") if work else None):
        try:
            for line in path.read_text(encoding="utf-8").splitlines() if path else []:
                if "|" in line and not line.lstrip().startswith("#"):
                    name, spelling = (x.strip() for x in line.split("|", 1))
                    if name.lower() == term.lower() and spelling and spelling.lower() not in (o.lower() for o in out):
                        out.append(spelling)
        except OSError:
            pass
    return out


def add_hint(name: str, spelling: str) -> bool:
    """Add a suggested spelling to data/respell_hints.txt (False if it is already there)."""
    if any(spelling.lower() == h.lower() for h in hints(name)):
        return False
    HINTS.parent.mkdir(exist_ok=True)
    with open(HINTS, "a", encoding="utf-8") as f:
        f.write(f"{name}|{spelling}\n")
    return True


def reopen(name: str) -> int:
    """Let a name be searched and judged again: forget that respellings were tried and which suggestions were tested, in every book.
    Returns how many lexicons changed."""
    n = 0
    for path in (ROOT / "work").glob("*/lexicon.json"):
        try:
            lex = json.loads(path.read_text())
        except (OSError, ValueError):
            continue
        hit = False
        for e in lex:
            if str(e.get("term", "")).lower() == name.lower() and (e.get("respell_tried") or e.get("hints_tested") or e.get("name_ok")):
                e.pop("respell_tried", None); e.pop("hints_tested", None); e.pop("name_ok", None); hit = True
        if hit:
            path.write_text(json.dumps(lex, indent=2, ensure_ascii=False))
            n += 1
    return n


def test_hints(work: Path, cfg: dict, progress=print) -> dict:
    """Speak and score the spellings you suggested (not yet tested) for names in this book, against the spelling in use now. A suggestion is
    adopted when it beats it by MARGIN. Names with no IPA from a real source cannot be judged and are only reported. Returns {term: spelling}."""
    from . import respell_check
    from .synth import get_engine
    meta = json.loads((work / "clips_meta.json").read_text())
    path = work / "lexicon.json"
    lex = json.loads(path.read_text())
    ipa_of = {e["term"]: e for e in targets(work)}
    adopted = {}
    for e in lex:
        tested = set(e.get("hints_tested") or [])
        todo = [h for h in hints(e["term"], work) if h.lower() not in {t.lower() for t in tested}]
        if not todo:
            continue
        t = ipa_of.get(e["term"])
        rx = pattern(spoken_form(e, work))
        voice = next((i["voice"] for i in meta.values() if i.get("engine") == "chatterbox" and rx.search(i["text"])), None)
        if t is None or voice is None:
            progress(f"[names] {e['term']}: suggestion(s) {', '.join(todo)} kept for later ({'no IPA to judge by' if t is None else 'no clip says it'})")
            continue
        current = e["respell"] if e.get("respell_src") == "checked" and e.get("respell") else e["term"]
        same = [h for h in todo if h.lower() == current.lower()]                       # already the spelling in use
        todo = [h for h in todo if h not in same]
        if same:
            e["hints_tested"] = sorted(tested | set(same))
        if not todo:
            continue
        note(f"testing your suggestion for {e['term']}")
        engine = get_engine("chatterbox", cfg.get("device", "auto"), 0, cfg.get("f5_precision", "float16"))
        base = respell_check.check([(current.lower(), current, t["ipa"])], voice, engine, progress=lambda *_: None, seed=1)[0]
        rows = respell_check.check([(h.lower(), h[:1].upper() + h[1:], t["ipa"]) for h in todo], voice, engine, progress=lambda *_: None, seed=1)
        best = max(rows, key=lambda r: r["score"])
        e["hints_tested"] = sorted(tested | set(todo))
        progress(f"[names] {e['term']}: your suggestion '{best['name']}' scores {best['score']} against {base['score']} for '{current}'")
        if best["score"] >= base["score"] + MARGIN:
            e["respell"], e["respell_src"], e["respell_score"], e["plain_score"], e["respell_hint"] = best["name"], "checked", best["score"], base["score"], True
            adopted[e["term"]] = best["name"]
    path.write_text(json.dumps(lex, indent=2, ensure_ascii=False))
    return adopted


def try_respell(work: Path, cfg: dict, entry: dict, progress=print, batches: tuple | None = None, target: float | None = None) -> dict:
    """Speak candidate spellings of the name in the voice of a clip that says it and score them, in rounds (BATCHES): the next round is
    only spoken if the best spelling so far is not yet at the target (TARGET, 80%) and ahead of the plain spelling by MARGIN.
    Returns the best: {"spelling", "score", "plain", "tried"}. Does not touch lexicon.json."""
    from . import respell_check, respell_search
    from .synth import get_engine
    batches = tuple(batches or cfg.get("name_respell_batches") or BATCHES)
    target = float(target if target is not None else cfg.get("name_respell_target", TARGET))
    meta = json.loads((work / "clips_meta.json").read_text())
    rx = pattern(spoken_form(entry, work))
    voice = next((i["voice"] for i in meta.values() if i.get("engine") == "chatterbox" and rx.search(i["text"])), None)
    if voice is None:
        return {}
    engine = get_engine("chatterbox", cfg.get("device", "auto"), 0, cfg.get("f5_precision", "float16"))
    plain_key = entry["term"].lower()
    suggested = [h for h in hints(entry["term"], work) if h.lower() != plain_key]                  # yours, spoken in the first round
    cands = [c for c in respell_search.candidates(entry["ipa"], entry["term"], sum(batches)) if c != plain_key and c not in {h.lower() for h in suggested}]
    if not cands and not suggested:                          # nothing to compare the plain spelling with
        return {}
    rows = respell_check.check([(plain_key, entry["term"], entry["ipa"])], voice, engine, progress=lambda *_: None, seed=1)   # the name as written is the baseline
    plain = rows[0]["score"]
    best, start = rows[0], 0
    for rnd, size in enumerate(batches):
        chunk = cands[start:start + size]
        start += size
        items = [(c, c.capitalize(), entry["ipa"]) for c in chunk]
        if rnd == 0:
            items = [(h.lower(), h[:1].upper() + h[1:], entry["ipa"]) for h in suggested] + items
        if not items:
            break
        rows = respell_check.check(items, voice, engine, progress=lambda *_: None, seed=1)
        best = max([best, *rows], key=lambda r: r["score"])
        if best["score"] >= target and best["score"] >= plain + MARGIN:
            break
    respell_search.free_ear()
    return {"spelling": best["name"], "score": best["score"], "plain": plain, "tried": min(start, len(cands)) + len(suggested)}


def note_tried(work: Path, term: str, found: dict) -> None:
    """Remember that respellings were tried for this name and did not help, so they are not tried again."""
    path = work / "lexicon.json"
    lex = json.loads(path.read_text())
    for e in lex:
        if e.get("term") == term:
            e["respell_tried"] = {k: found.get(k) for k in ("spelling", "score", "plain", "tried", "seconds")} if found else {"none": True}
    path.write_text(json.dumps(lex, indent=2, ensure_ascii=False))


def adopt(work: Path, term: str, spelling: str, score: float, plain: float | None) -> None:
    """Write a checked respelling into lexicon.json (never over one you typed)."""
    path = work / "lexicon.json"
    lex = json.loads(path.read_text())
    for e in lex:
        if e.get("term") == term and not (e.get("respell") and e.get("respell_src") == "user") and not (e.get("source") == "user" and e.get("respell")):
            e["respell"], e["respell_src"], e["respell_score"], e["plain_score"] = spelling, "checked", score, plain
    path.write_text(json.dumps(lex, indent=2, ensure_ascii=False))


def run(work: Path, cfg: dict, progress=print) -> dict:
    """Judge, retry, respell names that keep failing, make the clips with a new spelling, judge those again. Writes name_report.json."""
    t0 = time.time()
    if not (work / "clips_meta.json").exists():
        return {"skipped": "no clip list"}
    from . import jobqueue
    jobqueue.activity_set("the name check", work.name, "starting")
    try:
        return _run(work, cfg, progress, t0)
    finally:
        jobqueue.activity_clear()


def _run(work: Path, cfg: dict, progress, t0: float) -> dict:
    good = float(cfg.get("name_good", GOOD))
    bands = tuple((float(b), int(k)) for b, k in cfg.get("name_bands", BANDS))
    respelled, tried, stats = {}, {}, {}

    def respell(term: str, ipa: str) -> bool:
        entry = next((e for e in json.loads((work / "lexicon.json").read_text()) if e.get("term") == term), None)
        if entry is None:
            return False
        progress(f"[names] {term}: takes keep failing, trying respellings")
        note(f"trying respellings for {term}")
        t1 = time.time()
        try:
            found = try_respell(work, cfg, {**entry, "ipa": ipa}, progress)
            if found:
                found["seconds"] = round(time.time() - t1, 1)
        except Exception as ex:                                  # one name that cannot be searched must not stop the pass
            progress(f"[names] {term}: respelling search failed ({type(ex).__name__}: {ex})")
            found = {}
        tried[term] = found
        if found and found["score"] >= (found["plain"] or 0) + MARGIN:
            adopt(work, term, found["spelling"], found["score"], found["plain"])
            respelled[term] = found["spelling"]
            progress(f"[names] {term}: {found['plain']} -> {found['score']} with '{found['spelling']}' ({found['tried']} spellings tried, {found['seconds']} s)")
            return True
        note_tried(work, term, found)
        return False

    suggested = test_hints(work, cfg, progress)
    if suggested:
        respelled.update(suggested)
        from .synth import synthesize
        progress(f"[names] making the clips again with {len(suggested)} of your suggested spelling(s)")
        synthesize(work, cfg)
    for rnd in range(ROUNDS + 1):
        keep_gpu()
        before = len(respelled)
        skipped = []
        stats = pass_once(work, cfg, progress, good, respell=respell if rnd < ROUNDS else None, bands=bands,
                          skip_known=bool(cfg.get("name_skip_known", True)), skipped=skipped)
        if rnd == ROUNDS or len(respelled) == before:
            break
        from .synth import synthesize
        progress(f"[names] making the clips again with {len(respelled)} new spelling(s)")
        note(f"making the clips again with {len(respelled)} new spelling(s)")
        synthesize(work, cfg)                                               # only the clips whose text changed are made
    for t, s in stats.items():                                                  # names that came out good are left out next time
        if s["final"] and all(x >= good for x in s["final"]):
            mark_ok(work, t, s["fp"], _median(s["final"]))
    rows = []
    for t, s in sorted(stats.items(), key=lambda kv: _median(kv[1]["final"])):
        rows.append({"name": t, "ipa": s["ipa"], "ipa_src": s["ipa_src"], "clips": s["clips"], "first_median": round(_median(s["first"]), 2),
                     "final_median": round(_median(s["final"]), 2), "below_good": sum(1 for x in s["final"] if x < good),
                     "regenerated": s["regenerated"], "respell": respelled.get(t, ""), "respell_try": tried.get(t)})
    report = {"good": good, "bands": bands, "names": len(rows), "clips_below_good": sum(r["below_good"] for r in rows),
              "respelled": respelled, "left_out_known_good": skipped, "seconds": round(time.time() - t0), "per_name": rows}
    (work / "name_report.json").write_text(json.dumps(report, indent=1, ensure_ascii=False))
    progress(f"[names] {len(rows)} names judged, {report['clips_below_good']} clip(s) still under {good:.0%}, {len(respelled)} respelled ({report['seconds']} s)")
    return report
