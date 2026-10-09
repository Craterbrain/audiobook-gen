"""Gradio GUI: Book -> Cast -> Lexicon -> Generate. Run: python -m audiobook_gen.gui"""
import json
import os
from datetime import datetime
import re
import shutil
import tempfile
import time
from pathlib import Path

import gradio as gr
import pandas as pd
import soundfile as sf
import yaml

from .assemble import assemble
from .extract import extract
from .lexicon import MACBETH_SEED, MONTE_CRISTO_SEED, autofill, build_lexicon, fill_respell, seed
from .speakers import NARRATOR, parse_book
from . import casting
from . import m4b as m4blib
from . import sync as synclib
from . import voicepack as vpk
from . import voices as vlib
from . import respell_search as rs
from . import covers as covers_mod
from .synth import get_engine, resolve_voice, synthesize_iter

ROOT = Path(__file__).resolve().parent.parent
KOKORO_VOICES = ["af_heart", "af_bella", "af_nicole", "af_sarah", "af_sky", "am_adam", "am_echo",
                 "am_liam", "am_michael", "am_onyx", "am_puck", "bf_emma", "bf_isabella",
                 "bm_daniel", "bm_fable", "bm_george", "bm_lewis"]
MAX_ROLES = 30
MODE_MULTI = "Different voice for each character"
MODE_SINGLE = "One narrator reads everything"
ENGINE_LABELS = {"F5-TTS": "f5", "Chatterbox": "chatterbox", "Qwen3-TTS": "qwen3"}
LEX_MODES = {"Phonetic respelling — made from the IPA where blank": "respell",
             "Plain spelling — no substitutions": "plain"}
CHAP_COLS = ["Include", "#", "Title", "Characters"]
SEG_COLS = ["id", "chapter", "speaker", "kind", "text"]


def _wav(audio, sr: int, name: str = "clip.wav") -> str:
    """Previews: Kokoro's raw output peaks around -10 dBFS, which is easy to miss; normalise to -1 dBFS
    and write plain 16-bit PCM so every browser plays it."""
    import numpy as np
    audio = np.asarray(audio, dtype=np.float32)
    peak = float(np.abs(audio).max()) or 1.0
    out = Path(tempfile.mkdtemp()) / name
    sf.write(out, audio * (0.9 / peak), sr, subtype="PCM_16")
    return str(out)


def _wd(project: str) -> Path:
    if not project:
        raise gr.Error("Load a book on the Book tab first.")
    return Path(project)


def _cfg(project: str) -> dict:
    p = _wd(project) / "config.yaml"
    return yaml.safe_load((p if p.exists() else ROOT / "config.yaml").read_text())


# ---------- Book ----------
def load_book(file, max_chapters):
    if file is None:
        raise gr.Error("Choose an .epub or .txt file.")
    src = Path(file if isinstance(file, str) else file.name)
    work = ROOT / "work" / re.sub(r"\W+", "_", src.stem)
    data = extract(str(src), str(work), int(max_chapters) or None)
    rows = [[True, c["index"], c["title"], len(c["text"])] for c in data["chapters"]]
    return (str(work), pd.DataFrame(rows, columns=CHAP_COLS), data["title"], data["author"] or "",
            data["cover"], f"Loaded **{data['title']}**: {len(rows)} chapters → `{work}`")


def open_project(work, title="", author="", cover=""):
    """Open an existing project folder in the tabs without reading the book again. Returns what load_book returns."""
    p = Path(work)
    meta = json.loads(((p / "chapters.full.json") if (p / "chapters.full.json").exists() else (p / "chapters.json")).read_text())
    kept = {c["index"] for c in json.loads((p / "chapters.json").read_text())["chapters"]}
    rows = [[c["index"] in kept, c["index"], c["title"], len(c["text"])] for c in meta["chapters"]]
    cov = cover or meta.get("cover") or None
    return (str(p), pd.DataFrame(rows, columns=CHAP_COLS), title or meta.get("title", p.name), author or meta.get("author") or "",
            cov if cov and Path(cov).exists() else None, f"Opened **{title or meta.get('title', p.name)}** from `{p}` — {len(rows)} chapters.")


def _selected(df) -> set[int]:
    return {int(r["#"]) for _, r in df.iterrows() if r["Include"]}


def _filter_chapters(project, chap_df):
    """Drop unticked chapters from chapters.json (originals kept in chapters.full.json)."""
    work = _wd(project)
    full = work / "chapters.full.json"
    if not full.exists():
        shutil.copy(work / "chapters.json", full)
    data = json.loads(full.read_text())
    keep = _selected(chap_df)
    data["chapters"] = [c for c in data["chapters"] if c["index"] in keep]
    (work / "chapters.json").write_text(json.dumps(data, ensure_ascii=False, indent=2))


# ---------- Cast ----------
def _roles_list(segs):
    """[[role, line count]] with the Narrator first, then by number of lines."""
    counts: dict = {}
    for s in segs:
        counts[s["speaker"]] = counts.get(s["speaker"], 0) + 1
    return [[r, counts[r]] for r in sorted(counts, key=lambda r: (r != NARRATOR, -counts[r]))]


def _save_cfg(project, cfg):
    (_wd(project) / "config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True))


def voice_settings(project):
    """Mode radio, one-narrator dropdown/pace and the group's visibility for the loaded project."""
    cfg = _cfg(project)
    single = cfg.get("single_voice") or {}
    on = bool(single.get("enabled"))
    v = resolve_voice(NARRATOR, {**cfg, "single_voice": {}}) if not single.get("voice") else resolve_voice(NARRATOR, cfg)
    return (MODE_SINGLE if on else MODE_MULTI, gr.update(choices=casting.voice_choices(), value=casting.key_of(single.get("voice") or v)),
            v.get("speed", 1.0), gr.update(visible=on))


def set_mode(project, mode):
    cfg = _cfg(project)
    single = cfg.setdefault("single_voice", {})
    single["enabled"] = mode == MODE_SINGLE
    if "voice" not in single:  # start from whoever narrates now
        single["voice"] = casting.voice_of(casting.key_of(resolve_voice(NARRATOR, {**cfg, "single_voice": {}})) or "kokoro:bm_george")
    _save_cfg(project, cfg)
    return gr.update(visible=single["enabled"]), ("Everything will be read by one narrator, as continuous text." if single["enabled"]
                                                  else "Each character has their own voice.")


def set_single(project, key, speed):
    if not key:
        return "Choose a narrator."
    cfg = _cfg(project)
    cfg.setdefault("single_voice", {})["voice"] = casting.voice_of(key, speed)
    _save_cfg(project, cfg)
    return "Narrator saved."


def set_role_voice(project, role, key, speed):
    if not key:
        return gr.skip()
    cfg = _cfg(project)
    cfg.setdefault("voices", {})[role] = casting.voice_of(key, speed)
    _save_cfg(project, cfg)
    return f"{role} → {key.split(':', 1)[1]}"


def hear_voice(project, key, speed, text):
    if not key:
        raise gr.Error("Choose a voice first.")
    v = casting.voice_of(key, speed)
    from .voices import resolve
    v = resolve(v)
    eng = get_engine(v["engine"], _cfg(project).get("device", "auto"))
    return _wav(eng.synth(text or "In the beginning was the Word.", v), eng.sample_rate, "voice.wav")


GENDER_CHOICES = ["male", "female", "unknown"]


def find_genders(project, roles, title, progress=gr.Progress()):
    """Ask the small model for each character's gender, in the context of the book's title."""
    if not roles:
        raise gr.Error("Parse the book first.")
    from .tiebreak import Judge
    cfg = _cfg(project)
    book = (title or "").strip() or json.loads((_wd(project) / "chapters.json").read_text()).get("title", "this book")
    names = [r for r, _ in roles]
    progress(0.0, desc="Loading the small model (first time downloads it)")
    j = Judge(cfg.get("judge_model", "Qwen/Qwen2.5-1.5B-Instruct"), cfg.get("device", "auto"))
    try:
        found = j.genders(book, names, lambda f, d: progress(0.1 + 0.88 * f, desc=d))
    finally:
        j.close()
    cfg = _cfg(project)
    cfg["genders"] = {n: g for n, (g, _) in found.items()}
    _save_cfg(project, cfg)
    summary = ", ".join(f"{n}: {g}" for n, (g, _) in list(found.items())[:12])
    return f"Genders found for “{book}” — {summary}{'…' if len(found) > 12 else ''}. Check them in the menus."


def set_gender(project, role, gender):
    cfg = _cfg(project)
    cfg.setdefault("genders", {})[role] = gender
    _save_cfg(project, cfg)
    return f"{role}: {gender}"


def assign_by_gender_voices(project, roles):
    """Give each character a Kokoro voice of their gender (distinct while the list lasts); clones are kept."""
    if not roles:
        raise gr.Error("Parse the book first.")
    cfg = _cfg(project)
    keep = {r for r, v in (cfg.get("voices") or {}).items() if v.get("library")}
    new = casting.assign_by_gender([r for r, _ in roles], cfg.get("genders") or {}, keep)
    cfg.setdefault("voices", {}).update(new)
    _save_cfg(project, cfg)
    return f"Assigned {len(new)} voice(s) by gender" + (f"; kept your clone for {', '.join(sorted(keep))}" if keep else "") + "."


def refresh_voices(project):
    cfg = _cfg(project) if project else {}
    return gr.update(choices=casting.voice_choices())


def run_parse(project, chap_df, endpoint, use_judge=False, progress=gr.Progress()):
    work = _wd(project)
    progress(0.0, desc="Preparing chapters")
    _filter_chapters(project, chap_df)
    llm = (endpoint.strip(), "local") if endpoint and endpoint.strip() else None
    judge = None
    if use_judge:
        progress(0.02, desc="Loading the small model (first time downloads it)")
        from .tiebreak import Judge
        judge = Judge(_cfg(project).get("judge_model", "Qwen/Qwen2.5-1.5B-Instruct"),
                      _cfg(project).get("device", "auto"))
    try:
        segs = parse_book(work, _cfg(project).get("aliases"), llm, _cfg(project).get("phrase_roles"), judge,
                          lambda f, d: progress(0.05 + 0.92 * f, desc=d), _cfg(project).get("genders"))
    finally:
        if judge:
            judge.close()
    roles = _roles_list(segs)
    return (roles, pd.DataFrame(segs)[SEG_COLS], f"{len(segs)} segments, {len(roles)} roles",
            gr.update(choices=["All"] + _roles(segs), value="All"), gr.update(choices=_roles(segs)),
            show_lines(project, "All"), hints_text(project))


def load_parsed(project):
    """The Cast tab's contents for an already-parsed project (what run_parse returns, without parsing again)."""
    work = _wd(project)
    if not (work / "segments.json").exists():
        return [], pd.DataFrame(columns=SEG_COLS), "Not parsed yet.", gr.update(choices=["All"], value="All"), gr.update(choices=[]), \
            pd.DataFrame(columns=LINE_COLS), hints_text(project)
    segs = json.loads((work / "segments.json").read_text())
    roles = _roles_list(segs)
    return (roles, pd.DataFrame(segs)[SEG_COLS], f"{len(segs)} segments, {len(roles)} roles",
            gr.update(choices=["All"] + _roles(segs), value="All"), gr.update(choices=_roles(segs)),
            show_lines(project, "All"), hints_text(project))


def open_everything(work, title="", author="", cover=""):
    """Everything the tabs show for a project: book, cast, voices, generate settings (23 values)."""
    book = open_project(work, title, author, cover)
    return (*book, *load_parsed(book[0]), *voice_settings(book[0]), *gen_settings(book[0]))


def save_segments(project, seg_df):
    work = _wd(project)
    old = {s["id"]: s for s in json.loads((work / "segments.json").read_text())}
    out = []
    for _, r in seg_df.iterrows():
        s = dict(old[r["id"]])
        s["speaker"], s["text"] = str(r["speaker"]).strip() or NARRATOR, str(r["text"])
        s["kind"] = "narration" if s["speaker"] == NARRATOR else "dialogue"
        out.append(s)
    (work / "segments.json").write_text(json.dumps(out, ensure_ascii=False, indent=2))
    roles = _roles(out)
    return _roles_list(out), gr.update(choices=["All"] + roles), gr.update(choices=roles), "Segments saved (cast refreshed)."


LINE_COLS = ["Pick", "id", "ch", "speaker", "before", "text"]


def _segs(project):
    return json.loads((_wd(project) / "segments.json").read_text())


def _roles(segs):
    return sorted({s["speaker"] for s in segs}, key=lambda r: (r != NARRATOR, r))


def show_lines(project, role):
    """All lines currently assigned to `role` (or everything), with the preceding line for context."""
    segs = _segs(project)
    rows = [[False, s["id"], s["chapter"], s["speaker"],
             (segs[i - 1]["text"][-70:] if i else ""), s["text"]]
            for i, s in enumerate(segs) if role in (None, "", "All") or s["speaker"] == role]
    return pd.DataFrame(rows, columns=LINE_COLS)


def _refresh(project, role):
    segs = _segs(project)
    roles = _roles(segs)
    return (show_lines(project, role), _roles_list(segs),
            gr.update(choices=["All"] + roles, value=role if role in roles else "All"),
            gr.update(choices=roles))


def reassign(project, lines_df, target, role):
    target = (target or "").strip()
    picked = set(lines_df.loc[lines_df["Pick"].astype(bool), "id"])
    if not target or not picked:
        raise gr.Error("Tick some lines and choose (or type) the character to move them to.")
    work = _wd(project)
    segs = _segs(project)
    for s in segs:
        if s["id"] in picked:
            s["speaker"] = target
            s["kind"] = "narration" if target == NARRATOR else "dialogue"
    (work / "segments.json").write_text(json.dumps(segs, ensure_ascii=False, indent=2))
    return (*_refresh(project, role), f"Moved {len(picked)} line(s) to **{target}**.")


def merge_role(project, role, target):
    target = (target or "").strip()
    if role in (None, "", "All") or not target or role == target:
        raise gr.Error("Pick a specific character, and a different target.")
    lines = show_lines(project, role)
    lines["Pick"] = True
    return reassign(project, lines, target, target)


def preview_line(project, lines_df, target):
    picked = lines_df[lines_df["Pick"].astype(bool)]
    row = picked.iloc[0] if len(picked) else (lines_df.iloc[0] if len(lines_df) else None)
    if row is None:
        raise gr.Error("No line to preview.")
    role = (target or "").strip() or row["speaker"]
    cfg = _cfg(project)
    v = resolve_voice(role, cfg)
    eng = get_engine(v["engine"], cfg.get("device", "auto"))
    return _wav(eng.synth(row["text"][:400], v), eng.sample_rate, "line.wav"), f"Previewing line {row['id']} as **{role}**."


def hints_text(project) -> str:
    cfg = _cfg(project)
    return yaml.safe_dump({"aliases": cfg.get("aliases", {}), "phrase_roles": cfg.get("phrase_roles", {})},
                          sort_keys=False, allow_unicode=True)


def save_hints(project, text):
    try:
        h = yaml.safe_load(text) or {}
    except yaml.YAMLError as e:
        raise gr.Error(f"Invalid YAML: {e}")
    cfg = _cfg(project)
    cfg["aliases"], cfg["phrase_roles"] = h.get("aliases") or {}, h.get("phrase_roles") or {}
    (_wd(project) / "config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True))
    return "Hints saved. Click “Parse speakers” to re-run attribution with them."


# ---------- Lexicon ----------
LEX_COLS = ["term", "count", "kind", "known", "lang", "lang_src", "guess", "ipa", "respell"]
NAME_KINDS = ("name", "place", "demonym", "foreign")


def _lex_df(lex):
    return pd.DataFrame(lex)[LEX_COLS] if lex else pd.DataFrame(columns=LEX_COLS)


def run_lexicon(project, seed_name, lang, use_judge=False, title="", author="", progress=gr.Progress()):
    work = _wd(project)
    judge = None
    if use_judge:
        progress(0.0, desc="Loading the small model (first time downloads it)")
        from .tiebreak import Judge
        judge = Judge(_cfg(project).get("judge_model", "Qwen/Qwen2.5-1.5B-Instruct"), _cfg(project).get("device", "auto"))
    try:
        lex = build_lexicon(work, (lang or "").strip(), judge, (title or "").strip(), (author or "").strip(),
                            lambda f, d: progress(0.05 + 0.93 * f, desc=d))
    finally:
        if judge:
            judge.close()
    if seed_name == "John":
        seed(work)
    elif seed_name == "Macbeth":
        seed(work, MACBETH_SEED)
    elif seed_name == "Monte Cristo":
        seed(work, MONTE_CRISTO_SEED)
    lex = json.loads((work / "lexicon.json").read_text())
    flagged = sum(not e.get("known") for e in lex)
    return _lex_df(lex), f"{len(lex)} entries ({flagged} words Kokoro would guess at, {len(lex) - flagged} well-known names)"


def lex_autofill(project, lex_df, lang, include_known):
    """Apply the language hint to names (optionally well-known ones too), then fill empty IPA via espeak."""
    lang = (lang or "").strip()
    if not lang:
        raise gr.Error("Enter a language code first (fr, it, de, es, la, he, ...).")
    lex = lex_df.fillna("").to_dict("records")
    for e in lex:
        if not e["lang"] and e["kind"] in NAME_KINDS and (include_known or not e["known"]):
            e["lang"] = lang
    todo = [e for e in lex if e["lang"] and not e["ipa"]]
    for e in todo:
        e["source"] = "auto"
    n = autofill(todo)
    return pd.DataFrame(lex)[LEX_COLS], f"Filled {n} pronunciation(s) from espeak ({lang}). Review, hear, then Save."


def lex_lookup(project, lex_df, lang, wikis, bible, offline=False, progress=gr.Progress()):
    """Fill empty IPA from Wiktionary / WikiPron (and any Fandom wikis named), never overwriting what you typed."""
    from .pronounce import fill_lexicon
    rows = lex_df.fillna("").to_dict("records")
    full = {e["term"]: e for e in json.loads((_wd(project) / "lexicon.json").read_text())}
    for r in rows:
        r["source"] = full.get(r["term"], {}).get("source", "auto")
    r = fill_lexicon(rows, (lang or "").strip(), [w.strip() for w in (wikis or "").split(",") if w.strip()],
                     lambda f, d: progress(f, desc=d), bible=bool(bible), offline=bool(offline))
    srcs = {}
    for e in rows:
        if e.get("ipa_src"):
            k = e["ipa_src"].split(" ")[0].split(":")[0]
            srcs[k] = srcs.get(k, 0) + 1
    miss = ", ".join(r["missing"][:20]) + ("…" if len(r["missing"]) > 20 else "")
    return (pd.DataFrame(rows)[LEX_COLS],
            f"Found IPA for {r['found']} of {r['asked']} words ({', '.join(f'{v} from {k}' for k, v in srcs.items()) or 'none'}). "
            f"No entry for: {miss or '—'}. Review, hear, then Save.")


def save_lexicon(project, lex_df):
    work = _wd(project)
    old = {e["term"]: e for e in json.loads((work / "lexicon.json").read_text())}
    for _, r in lex_df.fillna("").iterrows():
        e = old.get(r["term"])
        if e is None:
            continue
        new = {f: str(r[f]).strip() for f in ("ipa", "respell", "lang")}
        if new["respell"] != (e.get("respell") or ""):
            e["respell_src"] = "user"
        if any(new[f] != (e.get(f) or "") for f in new):
            e.update(new, source="user")
    fill_respell(old.values(), refresh=True)   # IPA you typed -> an auto-made respelling for F5
    (work / "lexicon.json").write_text(json.dumps(list(old.values()), ensure_ascii=False, indent=2))
    return ("Lexicon saved. Kokoro uses your IPA; Chatterbox and Qwen3 use only respellings you typed; "
            "F5 also gets a respelling made from the IPA where you left it blank.")


def lex_autospell(project, lex_df):
    """Fill every blank respelling: from the row's IPA, or, where there is no IPA, from Kokoro's guess."""
    rows = lex_df.fillna("").to_dict("records")
    full = {e["term"]: e for e in json.loads((_wd(project) / "lexicon.json").read_text())}
    for r in rows:
        e = full.get(r["term"], {})
        r.update(source=e.get("source", "auto"), respell_src=e.get("respell_src", ""), known=False)  # known names too
    from_guess = sum(1 for r in rows if not r.get("ipa") and not r.get("respell") and r.get("guess"))
    n = fill_respell(rows, refresh=True, use_guess=True)
    for r in rows:
        r["known"] = bool(lex_df.loc[lex_df["term"] == r["term"], "known"].iloc[0])
    return (pd.DataFrame(rows)[LEX_COLS],
            f"Auto-spelled {n} word(s) for F5 — {from_guess} of them from Kokoro's guess because they had no IPA. "
            "Guess-based spellings are a starting point: hear them, or use the spelling search for a better one. Then Save.")


def fw_hear(project, word, ipa, respell, voice_key, template):
    """Speak `word` in a sentence with the chosen voice: Kokoro gets IPA markup, Chatterbox/Qwen3 the bare IPA, F5 the respelling."""
    from .lexicon import auto_respell, f5_text
    word = (word or "").strip()
    if not word or not voice_key:
        raise gr.Error("Type the word and choose a voice.")
    v = casting.voice_of(voice_key, 1.0)
    from .voices import resolve
    v = resolve(v)
    if v["engine"] == "kokoro":
        spoken = f"[{word}](/{ipa.strip().strip('/')}/)" if (ipa or "").strip() else word
    elif v["engine"] in ("chatterbox", "qwen3"):     # these read the IPA itself
        spoken = ipa.strip().strip("/") if (ipa or "").strip() else word
    else:
        spoken = (f5_text(respell, word) if (respell or "").strip()
                  else auto_respell({"ipa": ipa}) and f5_text(auto_respell({"ipa": ipa}), word)) or word
    eng = get_engine(v["engine"], _cfg(project).get("device", "auto") if project else "auto")
    text = (template or "Then said {word} unto him, Come and see.").replace("{word}", spoken)
    return _wav(eng.synth(text, v), eng.sample_rate, "word.wav"), f"Spoke it as: `{spoken}`"


def fw_save(project, word, ipa, respell):
    from .lexicon import auto_respell
    word = (word or "").strip()
    if not word:
        raise gr.Error("Type the word first.")
    path = _wd(project) / "lexicon.json"
    lex = json.loads(path.read_text()) if path.exists() else []
    e = next((x for x in lex if x["term"].lower() == word.lower()), None)
    if e is None:
        e = {"term": word, "count": 0, "kind": "name", "known": False, "lang": "", "lang_src": "", "lang_conf": 0.0,
             "guess": "", "ipa": "", "respell": "", "zipf": 0.0}
        lex.append(e)
    e.update(ipa=(ipa or "").strip(), respell=(respell or "").strip(), source="user", respell_src="user")
    if not e["respell"] and e["ipa"]:
        e["respell"], e["respell_src"] = auto_respell(e), "auto"
    path.write_text(json.dumps(lex, ensure_ascii=False, indent=2))
    return _lex_df(lex), f"Saved **{word}** to the lexicon (IPA for Kokoro, respelling for F5). Regenerate to hear it in the book."


def fw_search(project, word, ipa, voice_key, n, progress=gr.Progress()):
    """Try many spellings in the chosen F5 voice and rank them by how close the recognised phones are to the IPA."""
    word, ipa = (word or "").strip(), (ipa or "").strip()
    if not word or not ipa:
        raise gr.Error("Type the word and its IPA first (the IPA says how it should sound).")
    if not (voice_key or "").startswith("clone:") or "@" in voice_key:
        raise gr.Error("Choose one of your F5 clone voices in “Test with voice” — the spelling search only runs on F5.")
    v = vlib.load_voice(voice_key.split(":", 1)[1])
    f5 = get_engine("f5", "auto")
    outdir = ROOT / "work" / "_spell_search" / re.sub(r"\W+", "_", word)
    progress(0, desc="Loading the phoneme recognizer")
    try:
        rows = rs.search(word, ipa, v, f5, outdir, limit=int(n), top=10, progress=lambda f, d: progress(f, desc=d))
    finally:
        rs.free_ear()
    files = sorted(outdir.glob("[0-9][0-9]_*.wav"))
    df = pd.DataFrame([{"#": i + 1, "spelling": r["spelling"], "score (0 = exact)": r["distance"], "heard": r["heard"][:60]}
                       for i, r in enumerate(rows[:10])])
    return df, [str(f) for f in files], f"Best: **{rows[0]['spelling']}** (score {rows[0]['distance']}). Click a row to hear it and use it as the respelling — scores are a guide, your ear decides."


def fw_search_pick(table, files, evt: gr.SelectData):
    i = evt.index[0]
    return (files[i] if files and i < len(files) else None), str(table.iloc[i]["spelling"])


def fw_pick(lex_df, evt: gr.SelectData):
    r = lex_df.iloc[evt.index[0]]
    return r["term"], str(r["ipa"] or ""), str(r["respell"] or "")


def hear_term(project, lex_df, term, which):
    row = lex_df[lex_df["term"] == term]
    if row.empty:
        raise gr.Error("Choose a term from the table.")
    r = row.iloc[0]
    ipa = (r["ipa"] or "").strip().strip("/")
    spoken = f"{term}. [{term}](/{ipa}/). {term}." if (which == "Compare" and ipa) else \
             f"[{term}](/{ipa}/). [{term}](/{ipa}/)." if ipa and which == "My IPA" else f"{term}. {term}."
    cfg = _cfg(project)
    voice = resolve_voice(NARRATOR, cfg)
    if voice["engine"] != "kokoro":
        voice = {"engine": "kokoro", "voice": "af_heart"}
    eng = get_engine("kokoro", cfg.get("device", "auto"))
    return _wav(eng.synth(spoken, voice), eng.sample_rate, "term.wav")


# ---------- Clone ----------
CLONE_KEYS = ["speed", "nfe_step", "cfg_strength", "sway_sampling_coef", "cross_fade_duration", "target_rms", "seed"]
TEST_TEXT = ("It was the best of times, it was the worst of times. \u201cWhy, what ails you?\u201d he asked, "
             "and nobody could say.")


def _params(speed, steps, cfg, sway, xfade, rms, seed) -> dict:
    return dict(speed=float(speed), nfe_step=int(steps), cfg_strength=float(cfg), sway_sampling_coef=float(sway),
                cross_fade_duration=float(xfade), target_rms=float(rms), seed=int(seed))


def clone_transcribe(audio):
    if not audio:
        raise gr.Error("Add a reference clip first.")
    from f5_tts.infer.utils_infer import preprocess_ref_audio_text
    _, text = preprocess_ref_audio_text(audio, "")  # Whisper; downloads its model the first time
    return text.strip()


def _engine_key(label) -> str:
    return ENGINE_LABELS.get(label, "f5")


def _need_engine(key: str):
    from .tts import remote_engine as re_
    cls = {"chatterbox": re_.ChatterboxEngine, "qwen3": re_.Qwen3Engine}.get(key)
    if cls and not cls.available():
        raise gr.Error(f"{key} is not installed on this machine (its environment `{cls.venv}` is missing).")


def clone_generate(audio, ref_text, text, speed, steps, cfg, sway, xfade, rms, seed, takes,
                   engine="F5-TTS", exag=0.5, cfgw=0.4):
    key = _engine_key(engine)
    if not audio:
        raise gr.Error("Add a reference clip first.")
    if key != "chatterbox" and not (ref_text or "").strip():
        raise gr.Error("Type exactly what is said in the clip (F5-TTS and Qwen3 need it; Chatterbox does not).")
    _need_engine(key)
    import random
    import time
    base = int(seed) if int(seed) >= 0 else random.randint(0, 10**6)
    eng = get_engine(key, "auto")
    outs, seeds, notes = [], [], []
    for i in range(int(takes)):
        if key == "f5":
            voice = {"engine": "f5", "ref_audio": audio, "ref_text": ref_text.strip(),
                     **_params(speed, steps, cfg, sway, xfade, rms, base + i)}
        else:
            voice = {"engine": key, "ref_audio": audio, "ref_text": (ref_text or "").strip(), "seed": base + i}
            if key == "chatterbox":
                voice.update(exaggeration=float(exag), cfg_weight=float(cfgw))
        t0 = time.time()
        wav = eng.synth(text, voice)
        outs.append(_wav(wav, eng.sample_rate, f"take{i + 1}.wav"))
        seeds.append(base + i)
        notes.append(f"take {i + 1} ({engine}): seed {base + i}, {len(wav) / eng.sample_rate:.1f}s in {time.time() - t0:.1f}s")
    outs += [None] * (3 - len(outs))
    return (*outs[:3], seeds, "  \n".join(notes))


def clone_compare(name, text):
    """The same saved voice, the same sentence, in every engine that is installed."""
    if not name:
        raise gr.Error("Choose a saved voice.")
    from .voices import resolve
    outs = []
    for label in ENGINE_LABELS:
        key = ENGINE_LABELS[label]
        try:
            _need_engine(key)
        except gr.Error:
            outs.append(None)
            continue
        v = resolve({"engine": key, "library": name})
        eng = get_engine(key, "auto")
        outs.append(_wav(eng.synth(text or TEST_TEXT, v), eng.sample_rate, f"compare_{key}.wav"))
    return (*outs, f"**{name}** in F5-TTS, Chatterbox and Qwen3-TTS (an engine that isn't installed is left blank).")


def clone_save(name, audio, ref_text, speed, steps, cfg, sway, xfade, rms, seed, seeds, pick, notes,
               engine="F5-TTS", exag=0.5, cfgw=0.4):
    if not audio:
        raise gr.Error("Add a reference clip first.")
    idx = {"Take 1": 0, "Take 2": 1, "Take 3": 2}.get(pick, 0)
    use_seed = seeds[idx] if seeds and idx < len(seeds) else int(seed)
    try:
        params = _params(speed, steps, cfg, sway, xfade, rms, use_seed)
        if _engine_key(engine) == "chatterbox":     # remember the emotion settings that were tuned
            params.update(exaggeration=float(exag), cfg_weight=float(cfgw))
        saved = vlib.save_voice(name, audio, ref_text, params, notes or "")
    except ValueError as e:
        raise gr.Error(str(e))
    return gr.update(choices=vlib.list_voices(), value=saved), f"Saved **{saved}** (seed {use_seed}) to `voices/library/{saved}`."


def clone_load(name):
    if not name:
        raise gr.Error("Choose a saved voice.")
    v = vlib.load_voice(name)
    meta = json.loads((vlib.LIB / name / "voice.json").read_text())
    return (v["ref_audio"], v["ref_text"], name, meta.get("notes", ""),
            *[v[k] for k in CLONE_KEYS], v.get("exaggeration", 0.5), v.get("cfg_weight", 0.4),
            f"Loaded **{name}** (saved {meta.get('saved', '?')}).")


def clone_delete(name):
    if not name:
        raise gr.Error("Choose a saved voice.")
    vlib.delete_voice(name)
    return gr.update(choices=vlib.list_voices(), value=None), f"Deleted **{name}**."


def clone_roles(project):
    return gr.update(choices=_roles(_segs(project)))


def clone_assign(project, name, role, engine="F5-TTS"):
    if not name or not role:
        raise gr.Error("Choose a saved voice and a role.")
    cfg = _cfg(project)
    cfg.setdefault("voices", {})[role] = {"engine": _engine_key(engine), "library": name}
    (_wd(project) / "config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True))
    return f"**{role}** now speaks as **{name}** in {engine}. Open the Cast tab (re-parse or reload) to see it."


# ---------- Reference from an audiobook ----------
def _fpath(f):
    return None if f is None else (f if isinstance(f, str) else f.name)


def _mmss(t: float) -> str:
    return f"{int(t // 3600)}:{int(t % 3600 // 60):02d}:{int(t % 60):02d}"


def ab_read(file):
    path = _fpath(file)
    if not path:
        raise gr.Error("Open a .m4b first.")
    chs = m4blib.chapters(path)
    labels = [f"{c['index']} · {c['title']} ({_mmss(c['start'])}–{_mmss(c['end'])})" for c in chs]
    return gr.update(choices=labels, value=labels[0]), f"{len(chs)} chapter(s) found."


def ab_find(file, chapter, ebook, offset, minutes, target, model, count):
    """Cut a few clean 6-12 s clips from the chapter (waveform pauses only), have Whisper write down each one,
    and, if an ebook is given, replace Whisper's words with the book's exact text where they line up."""
    path = _fpath(file)
    if not path or not chapter:
        raise gr.Error("Open a .m4b and choose a chapter.")
    ch = next(c for c in m4blib.chapters(path) if c["index"] == int(chapter.split(" · ")[0]))
    start = ch["start"] + float(offset)
    end = min(ch["end"], start + 60 * float(minutes))
    if end - start < 15:
        raise gr.Error("That window is too short; lower the start offset or pick a longer chapter.")
    target = float(target)
    window = m4blib.extract_wav(path, start, end, 24000, m4blib.CACHE / "window.wav")
    spans = m4blib.speech_windows(window, target, max(4.0, target - 3), min(12.0, target + 3), int(count))
    if not spans:
        raise gr.Error("No clean 6–12 s stretches of speech in that window (too much music or no pauses). Try another offset.")
    clips = []
    try:
        for k, (s0, e0) in enumerate(spans):
            clip = m4blib.cut(path, start + s0, start + e0, m4blib.CACHE / f"cand_{k}.wav")
            heard = m4blib.transcribe_clip(clip, model, "auto")
            clips.append({"clip": str(clip), "at": start + s0, "dur": e0 - s0, "heard": heard,
                          "text": heard, "source": "whisper", "sim": None})
    finally:
        m4blib.free_asr()
    msg = ""
    if ebook:
        src = Path(_fpath(ebook))
        book = extract(str(src), str(ROOT / "work" / "_ebook_sync" / re.sub(r"\W+", "_", src.stem)))["chapters"]
        idx = synclib.book_index(book)
        votes = {}
        for c in clips:  # pass 1: anywhere in the book; the audio's chapter is where most clips land
            m = synclib.match(c["heard"], idx)
            if m and m["similarity"] >= 0.6:
                votes[m["chapter"]] = votes.get(m["chapter"], 0) + 1
        home = max(votes, key=votes.get) if votes else None
        for c in clips:  # pass 2: re-match inside that chapter, which is more precise
            m = synclib.match(c["heard"], idx, home) if home is not None else None
            if m and m["similarity"] >= 0.7:
                c.update(text=m["text"], source="ebook", sim=m["similarity"])
            elif m:
                c["sim"] = m["similarity"]
        msg = (f"Audio found in ebook **{book[home]['title']}** ({votes[home]}/{len(clips)} clips). " if home is not None
               else "**These clips don't match the ebook** — using Whisper's text. ")
    labels = [(f"{i + 1} · {_mmss(c['at'])} · {c['dur']:.1f}s · "
               + (f"book match {c['sim']:.0%} · " if c["sim"] is not None else "") + c["text"][:60], i)
              for i, c in enumerate(clips)]
    n_book = sum(c["source"] == "ebook" for c in clips)
    status = f"{msg}{len(clips)} sample(s); {n_book} with the ebook's exact text, {len(clips) - n_book} with Whisper's."
    return clips, gr.update(choices=labels, value=0), status


def ab_pick(clips, idx):
    if not clips or idx is None:
        return None, "", "", ""
    c = clips[int(idx)]
    sim = f" · similarity to book {c['sim']:.0%}" if c["sim"] is not None else ""
    return c["clip"], c["text"], c["heard"], f"{c['dur']:.1f}s · transcript from **{c['source']}**{sim}"


def ab_use(clips, idx):
    if not clips or idx is None:
        raise gr.Error("Find samples first.")
    c = clips[int(idx)]
    return c["clip"], c["text"], "Sample sent to the reference fields above. Generate takes to test it."


# ---------- Kokoro voicepack from a recording ----------
VP_SOURCES = ["A saved clone voice", "An audio file", "A random snippet of the audiobook above (.m4b)"]
VP_TEXT = "The next day he went down to the river, and there was no one there but the old man."


def _mine():
    return sorted(p.stem for p in vpk.PACKS.glob("*.pt"))


def vp_make(source, clone_name, audio_file, m4b_file, around, name, steps, warm, progress=gr.Progress()):
    """Search a Kokoro voicepack that sounds like the chosen voice; write voices/kokoro/<name>.pt."""
    import gc

    import numpy as np
    import soundfile as sf
    name = vlib.clean_name(name)
    if not name:
        raise gr.Error("Give the new voice a name.")
    progress(0.0, desc="Collecting the target recordings")
    if source == VP_SOURCES[0]:
        if not clone_name:
            raise gr.Error("Choose a saved clone voice (or another source).")
        v = vlib.load_voice(clone_name)
        a, sr = sf.read(v["ref_audio"], dtype="float32")
        clips = [(a, sr)]
        eng = get_engine("f5", "auto")      # a few sentences in that voice make a richer target than 12 s of reference
        for t in ["And Jesus answered and said unto them, The hour is come, that the Son of man should be glorified.",
                  "Then said the Jews, Forty and six years was this temple in building, and wilt thou rear it up?",
                  "The next day John seeth Jesus coming unto him, and saith, Behold the Lamb of God.",
                  "There was a man sent from God, whose name was John. The same came for a witness.",
                  "Nicodemus saith unto him, How can a man be born when he is old?"]:
            clips.append((np.asarray(eng.synth(t, v), dtype=np.float32), eng.sample_rate))
    elif source == VP_SOURCES[1]:
        path = audio_file if isinstance(audio_file, str) else getattr(audio_file, "name", None)
        if not path:
            raise gr.Error("Upload an audio file (a minute or more of one person speaking is best).")
        a, sr = sf.read(path, dtype="float32", always_2d=True)
        a = a.mean(1)
        step = 10 * sr
        clips = [(a[i:i + step], sr) for i in range(0, min(len(a), 8 * step), step) if len(a[i:i + step]) > 4 * sr]
    else:
        path = _fpath(m4b_file)
        if not path:
            raise gr.Error("Open an .m4b in “Take a reference from an audiobook” above first.")
        clips, info = vpk.target_from_audiobook(path, around=float(around), spread=0.05, minutes=4, clips=6,
                                                outdir=ROOT / "work" / "_vp_target")
        if not clips:
            raise gr.Error("Found no clean speech near that spot; try again (it picks a new random spot).")
    init = (vpk.PACKS / f"{warm}.pt") if warm and warm != "(start from the closest stock voice)" else None
    try:
        rep = vpk.search(clips, name, int(steps), progress=lambda f, d: progress(0.05 + 0.85 * f, desc=d), init_pack=init)
    finally:
        gc.collect()
        try:
            import torch
            torch.xpu.empty_cache()
        except Exception:
            pass
    progress(0.95, desc="Making comparison samples")
    kok = get_engine("kokoro", "auto")
    new = _wav(kok.synth(VP_TEXT, {"voice": f"pack:{name}"}), kok.sample_rate, "new.wav")
    stock = _wav(kok.synth(VP_TEXT, {"voice": rep["top_stock"][0][0]}), kok.sample_rate, "stock.wav")
    real = _wav(clips[0][0], clips[0][1], "real.wav")
    msg = (f"**{name}** saved to `voices/kokoro/{name}.pt` — started from {rep['started_from']}. Similarity to the target: "
           f"stock best **{rep['stock_similarity']:.3f}** → new **{rep['final_similarity']:.3f}**; on sentences it never saw: "
           f"**{rep['held_out_stock']:.3f}** → **{rep['held_out_new']:.3f}** ({rep['seconds']} s). "
           "This score is a computer's measure of speaker likeness; compare the three samples by ear. "
           "It now appears in the voice menus (Cast tab → Refresh voice list).")
    return (msg, real, new, stock, gr.update(choices=_mine(), value=name),
            gr.update(choices=["(start from the closest stock voice)"] + _mine()), gr.update(choices=casting.voice_choices()))


def vp_load_tag(name):
    if not name:
        return "", ""
    meta = json.loads((vpk.PACKS / f"{name}.json").read_text()) if (vpk.PACKS / f"{name}.json").exists() else {}
    return meta.get("label", ""), meta.get("tags", "")


def vp_tag(name, label, tags):
    """Give a voice a display name and description for the menus, e.g. John — US male."""
    if not name:
        raise gr.Error("Choose one of your Kokoro voices.")
    path = vpk.PACKS / f"{name}.json"
    meta = json.loads(path.read_text()) if path.exists() else {}
    meta.update(label=(label or "").strip(), tags=(tags or "").strip())
    path.write_text(json.dumps(meta, indent=2))
    return f"Tagged **{name}** as “{meta['label'] or name} — {meta['tags'] or 'your Kokoro voice'}”. Cast tab → Refresh voice list.", gr.update(choices=casting.voice_choices())


def vp_hear(name, text):
    if not name:
        raise gr.Error("Choose one of your Kokoro voices.")
    kok = get_engine("kokoro", "auto")
    return _wav(kok.synth(text or VP_TEXT, {"voice": f"pack:{name}"}), kok.sample_rate, "pack.wav")


def vp_delete(name):
    if not name:
        raise gr.Error("Choose one of your Kokoro voices.")
    for ext in (".pt", ".json"):
        (vpk.PACKS / f"{name}{ext}").unlink(missing_ok=True)
    return (gr.update(choices=_mine(), value=None), gr.update(choices=["(start from the closest stock voice)"] + _mine()),
            f"Deleted **{name}**.")


# ---------- Generate ----------
def gpu_vram_gib() -> float | None:
    """Memory of the biggest GPU, or None when there is no XPU."""
    try:
        import torch
        if torch.xpu.is_available():
            return max(torch.xpu.get_device_properties(i).total_memory for i in range(torch.xpu.device_count())) / 2**30
    except Exception:
        pass
    return None


def engine_status() -> str:
    from .tts import remote_engine as re_
    ok = lambda b: "ready" if b else "not installed"
    v = gpu_vram_gib()
    return (f"**Engines on this machine** — Kokoro: ready · F5-TTS: ready · "
            f"Chatterbox: {ok(re_.ChatterboxEngine.available())} · Qwen3-TTS: {ok(re_.Qwen3Engine.available())}"
            + (f" · GPU memory: {v:.0f} GB" if v else " · no Intel GPU found (CPU only)"))


ENGINE_NAMES = {"kokoro": "Kokoro", "f5": "F5-TTS", "chatterbox": "Chatterbox", "qwen3": "Qwen3-TTS"}


def speed_text(project=None) -> str:
    """Running average speed of every engine from past runs, plus an estimate for the open book."""
    from .runstats import averages, clock
    avg = averages()
    if not avg:
        return "**Speed** — no runs recorded yet. After a few runs this shows the average speed of each engine and how long this book should take."
    parts = [f"{ENGINE_NAMES.get(e, e)} {a['cps']:.1f} characters/s ({a['runs']} {'whole book' if a['whole'] else 'run'}{'s' if a['runs'] != 1 else ''})"
             for e, a in sorted(avg.items())]
    text = "**Speed, running average** — " + " · ".join(parts)
    try:
        cfg = _cfg(project)
        segs = json.loads((_wd(project) / "segments.json").read_text())
        per: dict[str, int] = {}
        for sg in segs:
            e = resolve_voice(sg["speaker"], cfg)["engine"]
            per[e] = per.get(e, 0) + len(sg["text"])
        if per and all(e in avg for e in per):
            secs = sum(n / avg[e]["cps"] for e, n in per.items())
            text += f"\n\n**This book:** {sum(per.values()):,} characters, about {clock(secs)} from scratch, start to finished file (clips already made are skipped)."
    except Exception:
        pass
    return text


def gen_settings(project):
    """Generate-tab engine options as saved in the project's config."""
    cfg = _cfg(project)
    mode = cfg.get("text_lexicon", "respell")
    label = next((k for k, v in LEX_MODES.items() if v == mode), list(LEX_MODES)[0])
    pm = cfg.get("pacing_ms") or {}
    return (bool(cfg.get("emotion")), float(cfg.get("emotion_base", 0.0)), label,
            int(pm.get("continuation", 140)), int(pm.get("tag", 120)),
            int((cfg.get("workers") or {}).get("chatterbox", 1)) >= 2)


def chatterbox_workers(toggle) -> int:
    """The Chatterbox toggle: off = 1 worker, on = 2. (A plain number is accepted too.)"""
    return (2 if toggle else 1) if isinstance(toggle, bool) else max(1, min(2, int(toggle or 1)))


def _prepare_generation(project, chap_df, crossfade, p_sent, p_para, p_speaker, kokoro_workers, f5_half,
                        emotion, emo_base, lex_label, p_cont, p_tag, cb_workers):
    """Save the Generate-tab settings into the project's config. Returns (work dir, config, chapters to make)."""
    work = _wd(project)
    if not (work / "segments.json").exists():
        raise gr.Error("Parse the book on the Cast tab first.")
    cfg = _cfg(project)
    cfg.update(crossfade_ms=int(crossfade), workers={**(cfg.get("workers") or {}), "kokoro": int(kokoro_workers)},
               f5_precision="float16" if f5_half else "float32")
    cfg["pacing_ms"] = {**(cfg.get("pacing_ms") or {}), "sentence": int(p_sent), "paragraph": int(p_para),
                        "speaker_change": int(p_speaker), "continuation": int(p_cont), "tag": int(p_tag)}
    cfg["emotion"], cfg["emotion_base"] = bool(emotion), float(emo_base)
    cfg["workers"] = {**cfg["workers"], "chatterbox": chatterbox_workers(cb_workers)}
    if lex_label in LEX_MODES:
        cfg["text_lexicon"] = LEX_MODES[lex_label]
    _save_cfg(project, {**_cfg(project), **{k: cfg[k] for k in ("emotion", "emotion_base", "text_lexicon", "crossfade_ms", "pacing_ms") if k in cfg},
                        "workers": cfg["workers"], "f5_precision": cfg["f5_precision"]})
    return work, cfg, _selected(chap_df)


def generate_queued(project, chap_df, title, author, cover, crossfade, p_sent, p_para, p_speaker, kokoro_workers, f5_half,
                    emotion, emo_base, lex_label, p_cont, p_tag, cb_workers, when="", window_on=False, w_start="23:00", w_stop="06:30",
                    send=False, device=""):
    """The Generate button: add the book to the queue to start as soon as the GPU is free. The queue runner makes it, watches it,
    and keeps going if this app is closed."""
    if send and not device:
        raise gr.Error("Choose which phone to send it to (or untick “Send the finished audiobook to my phone”).")
    vram = gpu_vram_gib()
    warn = (f" ⚠️ Two Chatterbox workers need about 16 GB of GPU memory and this card has {vram:.0f} GB; if it stalls, turn that off."
            if chatterbox_workers(cb_workers) > 1 and vram and vram < 16 else "")
    msg, *_ = queue_add(project, chap_df, title, author, cover, crossfade, p_sent, p_para, p_speaker, kokoro_workers, f5_half,
                        emotion, emo_base, lex_label, p_cont, p_tag, cb_workers, when, window_on, w_start, w_stop,
                        device if send else "")
    return msg + " Follow it on the **Queue** tab; the finished file appears here." + warn


def gen_stop(project):
    from . import jobqueue
    work = str(_wd(project).resolve())
    mine = [j for j in jobqueue.load() if j["work"] == work and j["status"] in ("queued", "paused", "running")]
    for j in mine:
        jobqueue.cancel(j["id"])
    return f"Cancelled {len(mine)} job(s) for this book. Clips already made are kept." if mine else "Nothing of this book is queued."


def gen_panel(project, shown):
    """The Generate tab's status for this book's newest job: progress while it runs, the file when it is done."""
    from . import jobqueue
    if not project:
        return gr.skip(), gr.skip(), gr.skip(), shown
    work = str(Path(project).resolve())
    jobs = [j for j in jobqueue.load() if j["work"] == work]
    if not jobs:
        return gr.skip(), gr.skip(), gr.skip(), shown
    j = jobs[-1]
    if j["status"] == "done":
        out = Path(j["out"])
        if shown == str(out):
            return f"Done → `{out}`", gr.skip(), gr.skip(), shown
        first = sorted((Path(project) / "chapters").glob("*.wav"))
        return f"Done → `{out}`", str(out), (str(first[0]) if first else None), str(out)
    if j["status"] == "running":
        r = jobqueue.now_running() or {}
        return (f"Making it — {r.get('pct', 0)}% ({r.get('done', 0):,} of {r.get('total', 0):,} clips)" if r.get("total") else
                f"Making it — {r.get('note') or 'starting'}"), gr.skip(), gr.skip(), shown
    return f"{j['status'].capitalize()}: {j.get('note') or 'waiting for its turn'} (starts {jobqueue.next_start(j, datetime.now())})", gr.skip(), gr.skip(), shown


# ---------- Queue ----------
QUEUE_HEADERS = ["#", "Book", "Status", "Progress", "Starts", "Window", "Phone", "Note", "id"]


def device_choices() -> list[tuple[str, str]]:
    """Paired KDE Connect devices for the dropdown, with whether each can be reached right now."""
    from . import jobqueue
    return [(f"{d['name']} — {'ready now' if d['reachable'] else 'not reachable right now'}", d["id"]) for d in jobqueue.kde_devices()]


def device_default(choices=None) -> str:
    from . import jobqueue
    ids = [v for _, v in (choices if choices is not None else device_choices())]
    want = jobqueue.default_device()
    return want if want in ids else (ids[0] if ids else "")


def devices_refresh():
    ch = device_choices()
    return gr.update(choices=ch, value=device_default(ch))


def queue_status() -> str:
    from . import jobqueue
    jobs = jobqueue.load()
    alive = jobqueue.runner_alive()
    waiting = sum(j["status"] in ("queued", "paused") for j in jobs)
    why = jobqueue.halted()
    return (("⚠️ **The queue paused itself:** " + why if why else "") + "\n\n" if why else "") + (("🟢 **Queue runner is running and watching every job.**" if alive else
             ("🔴 **The queue runner is stopped** (you stopped it) — press “Start the queue runner”. Nothing is made until you do."
              if jobqueue.STOPPED.exists() else
              "🔴 **The queue runner is not running** — press “Start the queue runner”. Jobs wait until it is."))
            + f"  {waiting} waiting · {sum(j['status'] == 'running' for j in jobs)} running · "
              f"{sum(j['status'] == 'held' for j in jobs)} saved for later · {sum(j['status'] == 'done' for j in jobs)} done")


def queue_eta() -> str:
    from . import jobqueue
    e = jobqueue.estimate_queue()
    if e["all"] is None:
        return ""
    text = f"🏁 **Everything should be finished about {e['all']:%a %d %b, %H:%M}** (from the measured speed of each voice engine, through each book's hours)."
    return (text + (f" Rough for: {', '.join(e['rough'])} (a voice engine in them has no measured speed yet)." if e["rough"] else "")
            + (f" No estimate yet for: {', '.join(e['unknown'])}." if e["unknown"] else ""))


def queue_table():
    from . import jobqueue
    import pandas as _pd
    return _pd.DataFrame([[i + 1, *r] for i, r in enumerate(jobqueue.table())], columns=QUEUE_HEADERS)


def queue_now() -> str:
    from . import jobqueue
    r = jobqueue.now_running()
    if r:
        prog = f"{r['pct']}% ({r['done']:,} of {r['total']:,} clips)" if r["total"] else (r["note"] or "starting")
        live = ("" if r["since_clip"] is None else
                f" · last clip {int(r['since_clip'])} s ago" + (" ⚠️ stalled? the watchdog will restart it" if r["since_clip"] > 300 else ""))
        return f"▶️ **Making “{r['title']}”** — {prog}{live} · started {r['started']}\n\n{queue_eta()}"
    nxt = [j for j in jobqueue.load() if j["status"] in ("queued", "paused")]
    if nxt:
        j = min(nxt, key=lambda x: jobqueue.next_start(x, datetime.now()))
        return f"⏳ Nothing is being made right now. Next: **{j['title']}** at {jobqueue.next_start(j, datetime.now())}.\n\n{queue_eta()}"
    return "Nothing is queued."


def runner_button_args() -> dict:
    from . import jobqueue
    if jobqueue.halted():
        return {"value": "Resume the queue", "variant": "primary"}
    return ({"value": "Stop the queue runner", "variant": "stop"} if jobqueue.runner_alive()
            else {"value": "Start the queue runner", "variant": "primary"})


def runner_button():
    """The button shows the opposite of the runner's state: Stop while it runs, Start while it does not."""
    from . import jobqueue
    return gr.update(**runner_button_args())


def queue_pick_update(picked=None):
    """The tick list of books (in queue order). Ticks you made are kept while the list refreshes."""
    from . import jobqueue
    jobs = jobqueue.load()
    choices = [(f"{i + 1}. {j['title'][:44]} — {j['status']}", j["id"]) for i, j in enumerate(jobs)]
    return gr.update(choices=choices, value=[x for x in (picked or []) if x in {j["id"] for j in jobs}])


def queue_refresh(picked=None):
    return queue_status(), queue_table(), queue_now(), runner_button(), queue_pick_update(picked)


def _need(picked):
    if not picked:
        raise gr.Error("Tick one or more books in the list first.")
    return list(picked)


def queue_move(picked, where):
    from . import jobqueue
    jobqueue.move(_need(picked), where)
    return {"top": "Moved to the top.", "up": "Moved up.", "down": "Moved down.", "bottom": "Moved to the bottom."}[where] + \
           " (A book that is already being made is not interrupted; pause it to start another first.)", *queue_refresh(picked)


def queue_hold(picked):
    from . import jobqueue
    n = jobqueue.hold(_need(picked))
    return (f"Paused / saved for later: {n} book(s). Clips already made are kept; press Resume to put them back in line." if n else
            "Nothing to pause (finished books can’t be paused)."), *queue_refresh(picked)


def queue_resume(picked):
    from . import jobqueue
    n = jobqueue.resume(_need(picked))
    return (f"Resumed {n} book(s)." if n else "None of those were paused or saved for later."), *queue_refresh(picked)


def queue_reschedule(picked, when, window_on, w_start, w_stop):
    from . import jobqueue
    not_before, window = _parse_schedule(when, window_on, w_start, w_stop)
    n = jobqueue.set_schedule(_need(picked), not_before, window)
    return (f"New schedule for {n} book(s): " + (f"from {not_before}" if not_before else "as soon as possible") +
            (f", only between {window}." if window else ", any time of day.")), *queue_refresh(picked)


def queue_send(picked, device):
    from . import jobqueue
    if not device:
        raise gr.Error("Choose a device on the Generate tab first (paired in KDE Connect).")
    jobqueue.save_default_device(device)
    n = jobqueue.set_send(_need(picked), device)
    return f"{n} book(s) will be sent to your phone when done (finished ones are sent now, and again if your phone is out of reach).", *queue_refresh(picked)


def queue_nosend(picked):
    from . import jobqueue
    n = jobqueue.set_send(_need(picked), "")
    return f"{n} book(s) will not be sent to your phone.", *queue_refresh(picked)


def queue_asap(picked):
    from . import jobqueue
    n = jobqueue.set_schedule(_need(picked), "", "")
    return f"{n} book(s) will start as soon as the GPU is free, at any time of day.", *queue_refresh(picked)


def alerts_settings():
    from . import jobqueue
    st = jobqueue.settings()
    return st.get("ntfy_topic", ""), bool(st.get("clean_after_done"))


def alerts_save(topic, clean):
    from . import jobqueue
    jobqueue.save_settings(ntfy_topic=(topic or "").strip(), clean_after_done=bool(clean))
    return "Saved."


def alerts_test(topic):
    from . import jobqueue
    jobqueue.save_settings(ntfy_topic=(topic or "").strip())
    if not jobqueue.ntfy_topic():
        return "Type a topic name first."
    return "Sent. Check your phone." if jobqueue.notify("This is a test from the audiobook queue.", "Test", tags=["bell"]) else "Could not reach ntfy; check the topic and your connection."


def queue_toggle_runner():
    from . import jobqueue
    running = jobqueue.runner_alive() and not jobqueue.halted()
    msg = jobqueue.stop_runner() if running else jobqueue.start_runner()
    if not running:                                   # give it a moment to report in
        for _ in range(20):
            if jobqueue.runner_alive():
                break
            time.sleep(0.5)
    return msg, *queue_refresh()


def _parse_schedule(when, window_on, w_start, w_stop) -> tuple[str, str]:
    """(start time "YYYY-MM-DD HH:MM" or "", window "23:00-06:30" or "") from the schedule controls."""
    from . import jobqueue
    not_before = ""
    if when:
        try:
            not_before = datetime.fromtimestamp(float(when)).strftime(jobqueue.FMT) if not isinstance(when, str) else \
                datetime.strptime(str(when)[:16], jobqueue.FMT).strftime(jobqueue.FMT)
        except Exception:
            raise gr.Error("Start time not understood. Pick a date and time, or clear the box to start as soon as the GPU is free.")
    window = ""
    if window_on:
        if not (re.fullmatch(r"\d{1,2}:\d{2}", (w_start or "").strip()) and re.fullmatch(r"\d{1,2}:\d{2}", (w_stop or "").strip())):
            raise gr.Error("Write the window as times like 23:00 and 06:30.")
        window = f"{w_start.strip()}-{w_stop.strip()}"
    return not_before, window


def queue_add(project, chap_df, title, author, cover, crossfade, p_sent, p_para, p_speaker, kokoro_workers, f5_half,
              emotion, emo_base, lex_label, p_cont, p_tag, cb_workers, when, window_on, w_start, w_stop, send_to=""):
    from . import jobqueue
    work, cfg, only = _prepare_generation(project, chap_df, crossfade, p_sent, p_para, p_speaker, kokoro_workers, f5_half,
                                          emotion, emo_base, lex_label, p_cont, p_tag, cb_workers)
    not_before, window = _parse_schedule(when, window_on, w_start, w_stop)
    all_chapters = {int(r["#"]) for _, r in chap_df.iterrows()}
    chapters = sorted(only) if only and only != all_chapters else None
    if send_to:
        jobqueue.save_default_device(send_to)
    jq_job = jobqueue.add(str(work), title or work.name, author or "", cover or "", "", "", chapters, not_before, window, send_to)
    jobqueue.ensure_supervisor()
    when_txt = f"from {not_before}" if not_before else "as soon as the GPU is free"
    return (f"Queued “{jq_job['title']}” — starts {when_txt}" + (f", only between {window}" if window else "") +
            ". It is watched: if it stalls or crashes it is restarted." + (" It will be sent to your phone when it is done." if send_to else "")), *queue_refresh()


def queue_open(picked):
    from . import jobqueue
    j = next((x for x in jobqueue.load() if x["id"] in _need(picked)), None)
    if not j:
        raise gr.Error("Tick a book in the list first.")
    return open_everything(j["work"], j["title"], j.get("author", ""), j.get("cover", ""))


def reconnect_on_load(project):
    """When the page opens: if a book is being made and nothing is open, open that book, so the tabs show what the queue is doing."""
    from . import jobqueue
    run = jobqueue.now_running()
    if project or not run:
        return (gr.skip(),) * 23
    j = next(x for x in jobqueue.load() if x["id"] == run["id"])
    return open_everything(j["work"], j["title"], j.get("author", ""), j.get("cover", ""))


def queue_cancel(picked):
    from . import jobqueue
    n = 0
    for j in jobqueue.load():
        if j["id"] in _need(picked) and j["status"] not in ("done", "failed", "cancelled"):
            jobqueue.cancel(j["id"]); n += 1
    return (f"Cancelled {n} book(s). Clips already made are kept." if n else "Nothing to cancel."), *queue_refresh(picked)


def queue_remove(picked):
    from . import jobqueue
    ids = _need(picked)
    n = jobqueue.remove_many(ids)
    left = len(ids) - n
    return (f"Removed {n} book(s) from the list." + (f" {left} is being made right now: pause or cancel it first." if left else "")), *queue_refresh([])


# ---------- Cover ----------
COVER_LAYOUTS = {"Picture window — for people (portrait, group)": "framed", "Full background — for scenery": "full",
                 "Plain — no picture": "plain"}


def _cover_dir(project) -> Path:
    return (Path(project) if project else ROOT / "work" / "_covers") / "cover_src"


def cover_search(query):
    from . import covers
    if not (query or "").strip():
        raise gr.Error("Type what to look for, e.g. “Civil War battle painting” or “Frederick Douglass portrait”.")
    try:
        hits = covers.search(query.strip(), 12)
    except Exception as e:
        raise gr.Error(f"Could not reach Wikimedia Commons: {e}")
    gallery = [(h["thumb"], f"{h['title'][:48]} — {h['artist'][:28]} {h['date']}".strip(" —")) for h in hits if h["thumb"]]
    return hits, gallery, (f"{len(hits)} free-to-use pictures (public domain or CC0). Click one to use it." if hits
                           else "No free-licence pictures found — try other words.")


def cover_pick(project, hits, evt: gr.SelectData):
    from . import covers
    h = hits[evt.index]
    try:
        path, credit = covers.fetch(h["file"], _cover_dir(project))
    except Exception as e:
        raise gr.Error(str(e))
    return str(path), credit, f"Using “{h['title']}” ({h['licence']}). Adjust the look below, then **Use this cover**."


def cover_render(project, title, author, upload, picked, layout_label, colour_name, fx, fy, zoom):
    from . import covers
    layout = COVER_LAYOUTS.get(layout_label, "framed")
    picture = upload or picked or None
    if layout != "plain" and not picture:
        raise gr.Error("Add a picture first: upload one, or search for one.")
    out = (Path(project) if project else ROOT / "work" / "_covers") / "cover_preview.jpg"
    try:
        return str(covers.render({"title": title or "Untitled", "author": author or "", "layout": layout, "picture": picture,
                                  "color": colour_name or "black", "focus": [fx, fy], "zoom": zoom}, out))
    except Exception as e:
        raise gr.Error(str(e))


def cover_refresh(project, title, author, upload, picked, layout_label, colour_name, fx, fy, zoom):
    """Redraw the preview when a control changes; before a picture is chosen, quietly do nothing."""
    if COVER_LAYOUTS.get(layout_label, "framed") != "plain" and not (upload or picked):
        return gr.skip()
    return cover_render(project, title, author, upload, picked, layout_label, colour_name, fx, fy, zoom)


def cover_use(project, preview, credit, upload):
    from . import jobqueue
    work = _wd(project)
    if not preview:
        raise gr.Error("Make a preview first.")
    dest = work / "cover_custom.jpg"
    shutil.copy(preview, dest)
    (work / "cover_credit.txt").write_text((credit if not upload else "your own picture") + "\n")
    n = jobqueue.set_cover(str(work), str(dest))
    return str(dest), f"This is now the book’s cover" + (f", and {n} queued job(s) will use it." if n else ". It is used when you generate.")


# ---------- Assistant ----------
def assistant_ask(project, message, history, session, spent):
    from . import assistant
    if not (message or "").strip():
        raise gr.Error("Type what you would like set up.")
    work = _wd(project)
    r = assistant.ask(work, message.strip(), session or None)
    history = (history or []) + [{"role": "user", "content": message.strip()}]
    if r["error"]:
        history.append({"role": "assistant", "content": f"⚠️ {r['error']}"})
    else:
        history.append({"role": "assistant", "content": r["reply"] or "(no reply)"})
    spent = float(spent or 0) + r["cost"]
    diffs = assistant.diff(work)
    prop = assistant.cover_proposal(work)
    shown = "\n\n".join(f"**{n}**\n```diff\n{d[:6000]}{'…' if len(d) > 6000 else ''}\n```" for n, d in diffs.items())
    problems = assistant.validate(work) if (diffs or prop) else []
    pending = bool(diffs) or bool(prop)
    note = (f"⚠️ These changes can’t be applied yet: {'; '.join(problems)}" if problems else
            "Review the changes below, then **Apply** them or **Discard** them." if pending else "No changes proposed.")
    if prop and prop[1]:
        shown = (shown + "\n\n" if shown else "") + f"**Cover picture credit:** {prop[1]}"
    return (history, r["session"] or "", spent, f"Claude usage this session: about ${spent:.2f}", note, shown or "_Nothing to show._",
            gr.update(interactive=pending and not problems), gr.update(interactive=pending), "",
            gr.update(value=str(prop[0]) if prop else None, visible=bool(prop)))


def assistant_apply(project):
    from . import assistant, jobqueue
    work = _wd(project)
    had_cover = assistant.cover_proposal(work) is not None
    msg = assistant.apply(work)
    new_cover = gr.skip()
    if had_cover and (work / "cover_custom.jpg").exists() and msg.startswith("Applied"):
        new_cover = str(work / "cover_custom.jpg")
        n = jobqueue.set_cover(str(work), new_cover)
        msg += f" The new cover is used for {n} queued job(s) and when you generate." if n else " The new cover is used when you generate."
    return msg, "_Nothing to show._", gr.update(interactive=False), gr.update(interactive=False), gr.update(value=None, visible=False), new_cover


def assistant_discard(project):
    from . import assistant
    return (assistant.discard(_wd(project)), "_Nothing to show._", gr.update(interactive=False), gr.update(interactive=False),
            gr.update(value=None, visible=False))


# The Clone tab is a workshop rather than a step in the book flow, so its button is set apart from the numbered tabs.
CSS = """
#clone-tab-button {
    background: linear-gradient(135deg, #7c3aed, #db2777);
    color: #fff !important;
    font-weight: 700;
    border: none !important;
    border-radius: 999px;
    padding: 4px 18px;
    margin-left: 14px;
    box-shadow: 0 0 0 2px rgba(124, 58, 237, .30);
}
#clone-tab-button:hover { filter: brightness(1.12); }
#clone-tab-button.selected, #clone-tab-button[aria-selected="true"] {
    box-shadow: 0 0 0 3px rgba(219, 39, 119, .55);
    filter: brightness(1.08);
}
"""


def build_ui() -> gr.Blocks:
    with gr.Blocks(title="Audiobook Gen") as ui:
        project = gr.State("")
        gr.Markdown("# Audiobook Gen\nEPUB/TXT → multi-voice chaptered M4B, on Intel Arc (XPU).")
        with gr.Tabs():
            with gr.Tab("1 · Book"):
                with gr.Row():
                    book = gr.File(label="EPUB or TXT", file_types=[".epub", ".txt"])
                    with gr.Column():
                        maxch = gr.Number(label="Only first N chapters (0 = all)", value=0, precision=0)
                        load_btn = gr.Button("Load book", variant="primary")
                        status1 = gr.Markdown()
                with gr.Row():
                    title = gr.Textbox(label="Title"); author = gr.Textbox(label="Author")
                    cover = gr.Image(label="Cover (optional)", type="filepath", height=160)
                chap_df = gr.Dataframe(headers=CHAP_COLS, datatype=["bool", "number", "str", "number"],
                                       interactive=True, label="Chapters (untick to skip)")
            with gr.Tab("2 · Cast"):
                with gr.Row():
                    parse_btn = gr.Button("Parse speakers", variant="primary", scale=1)
                    status2 = gr.Markdown(scale=3)
                with gr.Accordion("Parsing options", open=False):
                    endpoint = gr.Textbox(label="Local LLM endpoint (optional, OpenAI-compatible)",
                                          placeholder="http://localhost:8080/v1/chat/completions")
                    judge_cb = gr.Checkbox(label="Small-LLM tie-breaker for ambiguous quotes (Qwen2.5-1.5B, ~3 GB)")
                gr.Markdown("### Who reads the book?")
                mode = gr.Radio([MODE_MULTI, MODE_SINGLE], value=MODE_MULTI, label="Narration style")
                with gr.Accordion("Voice preview & menu help", open=False):
                    gr.Markdown("Every saved clone shows up three times in the voice menus: **(F5)**, **(Chatterbox)** and **(Qwen3)**. "
                                "The **Pace** slider applies to Kokoro and F5; Chatterbox and Qwen3 choose their own pace "
                                "(Chatterbox gets emotion from the text — see the Generate tab).")
                    with gr.Row():
                        pv_text = gr.Textbox(label="Sample text for ▶ Hear", scale=3,
                                             value="In the beginning was the Word, and the Word was with God.")
                        refresh_btn = gr.Button("Refresh voice list (after saving a clone)", scale=1)
                pv_audio = gr.Audio(label="Voice preview", autoplay=True)
                roles_state = gr.State([])
                with gr.Row():
                    gender_btn = gr.Button("Find genders with the small model (Qwen)")
                    gv_btn = gr.Button("Assign voices by gender")
                with gr.Group(visible=False) as single_group:
                    gr.Markdown("**One narrator for the entire cast** — all characters and narration, read straight through.")
                    with gr.Row():
                        single_dd = gr.Dropdown(label="Narrator", choices=casting.voice_choices(), scale=3)
                        single_speed = gr.Slider(0.6, 1.4, 1.0, step=0.05, label="Pace", scale=2)
                        single_hear = gr.Button("▶ Hear", scale=1)

                # Fixed pool of rows (shown/hidden as needed): more dependable than rebuilding widgets on every change.
                role_note = gr.Markdown("*Load a book and click “Parse speakers” — each character appears here with a voice menu.*")
                role_rows = []
                for _ in range(MAX_ROLES):
                    with gr.Row(visible=False, equal_height=True) as row:
                        lab = gr.Markdown(min_height=0)
                        gd = gr.Dropdown(choices=GENDER_CHOICES, show_label=False, scale=1, min_width=90)
                        dd = gr.Dropdown(choices=casting.voice_choices(), show_label=False, scale=3)
                        sp = gr.Slider(0.6, 1.4, 1.0, step=0.05, label="Pace", scale=2)
                        hear = gr.Button("▶ Hear", scale=1, min_width=80)
                    who = gr.State("")
                    gd.input(set_gender, [project, who, gd], status2)
                    dd.input(set_role_voice, [project, who, dd, sp], status2)      # .input = the user's own choice
                    sp.release(set_role_voice, [project, who, dd, sp], status2)
                    hear.click(hear_voice, [project, dd, sp, pv_text], pv_audio)
                    role_rows.append((row, lab, gd, dd, sp, who))
                row_outputs = [role_note] + [c for r in role_rows for c in r]

                def fill_rows(roles, mode_value, proj):
                    multi = bool(roles) and mode_value == MODE_MULTI
                    cfg = _cfg(proj) if (proj and multi) else {}
                    choices = casting.voice_choices()
                    if not roles:
                        note = gr.update(visible=True)
                    elif len(roles) > MAX_ROLES and multi:
                        note = gr.update(visible=True, value=f"*Showing the {MAX_ROLES} characters with the most lines; "
                                                             "the rest use automatic voices.*")
                    else:
                        note = gr.update(visible=False)
                    out = [note]
                    for i in range(MAX_ROLES):
                        if multi and i < len(roles):
                            role, n = roles[i]
                            v = resolve_voice(role, cfg)
                            out += [gr.update(visible=True), f"**{role}**  \n{n} line{'s' if n != 1 else ''}",
                                    (cfg.get("genders") or {}).get(role, "unknown"),
                                    gr.update(choices=choices, value=casting.key_of((cfg.get("voices") or {}).get(role) or v)), v.get("speed", 1.0), role]
                        else:
                            out += [gr.update(visible=False), "", gr.skip(), gr.skip(), gr.skip(), ""]
                    return out

                for trig in (roles_state.change, mode.change, refresh_btn.click):
                    trig(fill_rows, [roles_state, mode, project], row_outputs)
                gender_btn.click(find_genders, [project, roles_state, title], status2).then(
                    fill_rows, [roles_state, mode, project], row_outputs)
                gv_btn.click(assign_by_gender_voices, [project, roles_state], status2).then(
                    fill_rows, [roles_state, mode, project], row_outputs)
                with gr.Accordion("Review & fix lines — move lines between characters, edit segments, hints", open=False):
                    with gr.Accordion("Attribution hints (aliases & phrase roles)", open=False):
                        gr.Markdown("`aliases`: name → character (e.g. `Dantès: Edmond`). `phrase_roles`: regex → "
                                    "character (e.g. `old man|old father: Old Dantès`).")
                        hints = gr.Code(language="yaml", label="Hints")
                        save_hints_btn = gr.Button("Save hints")
                    gr.Markdown("### Review & reassign lines")
                    with gr.Row():
                        char_dd = gr.Dropdown(label="Show lines of", choices=["All"], value="All")
                        target_dd = gr.Dropdown(label="Move picked lines to (type a new name to create one)",
                                                choices=[], allow_custom_value=True)
                    with gr.Row():
                        move_btn = gr.Button("Move picked lines", variant="primary")
                        pv_line_btn = gr.Button("Hear first picked line (as target voice)")
                        merge_btn = gr.Button("Merge ALL of this character into target")
                    lines_df = gr.Dataframe(headers=LINE_COLS, interactive=True, wrap=True, max_height=420,
                                            datatype=["bool", "str", "number", "str", "str", "str"],
                                            static_columns=[1, 2, 3, 4, 5])
                    line_audio = gr.Audio(label="Line preview", autoplay=True)
                    with gr.Accordion("Segments (fix speaker labels / text)", open=False):
                        seg_df = gr.Dataframe(headers=SEG_COLS, interactive=True, wrap=True,
                                              datatype=["str", "number", "str", "str", "str"])
                        save_seg_btn = gr.Button("Save segments")
            with gr.Tab("3 · Lexicon"):
                gr.Markdown("How names and hard words are pronounced. **Build lexicon**, fill in the IPA (try **Look up**), review, then **Save**.\n\n"
                            "**Who uses what:** Kokoro voices use the **IPA** column. **Chatterbox and Qwen3 use only the respelling you type** in the "
                            "*respell* column — IPA and auto-made spellings never reach them. F5-TTS uses the respelling, made from the IPA where blank.")
                with gr.Accordion("F5-TTS only — how its names are spelled", open=False):
                    lex_rd = gr.Radio(list(LEX_MODES), value=list(LEX_MODES)[0], label="How names and hard words are spoken by F5-TTS",
                                      info="Does not affect Chatterbox, Qwen3 or Kokoro. Saved when you generate.")
                with gr.Row():
                    lex_btn = gr.Button("Build lexicon", variant="primary", scale=1)
                    lookup_btn = gr.Button("Look up IPA online (Wiktionary / WikiPron / Bible dictionary)", scale=2)
                    save_lex_btn = gr.Button("Save lexicon", variant="primary", scale=1)
                status3 = gr.Markdown()
                lex_df = gr.Dataframe(headers=LEX_COLS, interactive=True, max_height=480,
                                      datatype=["str", "number", "str", "bool", "str", "str", "str", "str", "str"],
                                      static_columns=[0, 1, 2, 3, 5, 6])
                with gr.Accordion("Build & lookup options — language, Bible dictionary, offline", open=False):
                    with gr.Row():
                        lang_tb = gr.Textbox(label="Language of the names (blank = English first; or set fr, it, de, es, la, he…)", scale=2)
                        seed_dd = gr.Dropdown(label="Starter pronunciations", choices=["None", "John", "Macbeth", "Monte Cristo"],
                                              value="None", scale=1)
                        detect_cb = gr.Checkbox(label="Ask the small LLM which language the names are in (uses the Book tab title + author)", scale=2)
                    with gr.Row():
                        bible_cb = gr.Checkbox(label="Bible book: use the Bible IPA dictionary (4,000 names)")
                        offline_cb = gr.Checkbox(label="Offline only (skip Wiktionary)")
                        wiki_tb = gr.Textbox(label="Fandom wikis to search too (optional: lotr, harrypotter…)", scale=3)
                with gr.Accordion("Auto-fill tools", open=False):
                    gr.Markdown("**guess** is what Kokoro says today; **ipa** is read by Kokoro, Chatterbox and Qwen3; **respell** is for F5.")
                    with gr.Row():
                        known_cb = gr.Checkbox(label="Also apply to well-known names (Edmond, Napoleon…)")
                        fill_btn = gr.Button("Auto-fill IPA from language")
                        spell_btn = gr.Button("Auto-spell for F5 (from IPA, or from the guess if no IPA)")
                with gr.Accordion("Fix or add a word (hear it in the real voice, then save)", open=False):
                    gr.Markdown("Click a row above to load it here, or type any word. **IPA** is what Kokoro, Chatterbox and Qwen3 use; **Respelling** is "
                                "what F5 uses — write it like a plain word with no hyphens or capitals (e.g. `Nuhthanyel`; hyphens "
                                "split a name into separate words and capitals get spelled out). Leave one blank and it is made from the other.")
                    with gr.Row():
                        fw_word = gr.Textbox(label="Word", scale=2)
                        fw_ipa = gr.Textbox(label="IPA (Kokoro)", scale=2)
                        fw_resp = gr.Textbox(label="Respelling (F5)", scale=2)
                    with gr.Row():
                        fw_voice = gr.Dropdown(label="Test with voice", choices=casting.voice_choices(), scale=2)
                        fw_tmpl = gr.Textbox(label="Test sentence ({word} is replaced)", value="Philip findeth {word}, and saith unto him, We have found him.", scale=4)
                    with gr.Row():
                        fw_hear_btn = gr.Button("▶ Hear"); fw_save_btn = gr.Button("Save to lexicon", variant="primary")
                        fw_n = gr.Slider(20, 100, 60, step=10, label="Spellings to try", scale=1)
                        fw_search_btn = gr.Button("Search for the best F5 spelling (a few minutes)")
                    fw_table = gr.Dataframe(label="Ranked spellings — click one to hear it", interactive=False, wrap=True)
                    fw_files = gr.State([])
                    fw_audio = gr.Audio(label="Word in context", autoplay=True)
                    fw_note = gr.Markdown()
                with gr.Accordion("Hear a term from the table", open=False):
                    with gr.Row():
                        term_dd = gr.Dropdown(label="Hear a term", choices=[], scale=2)
                        which_rd = gr.Radio(["Kokoro's guess", "My IPA", "Compare"], value="Compare", label="Version", scale=2)
                        hear_btn = gr.Button("Hear", scale=1)
                        term_audio = gr.Audio(label="Pronunciation", scale=2, autoplay=True)
            with gr.Tab("4 · Generate"):
                eng_status = gr.Markdown(engine_status())
                speed_md = gr.Markdown(speed_text())
                with gr.Row():
                    emo_cb = gr.Checkbox(label="Emotion from the text (Chatterbox)", scale=1,
                                         info="Each sentence gets its own expressiveness, from its mood and tags like “cried” or “whispered”.")
                with gr.Row():
                    g_when = gr.DateTime(label="Start at (leave empty to start as soon as the GPU is free)", include_time=True,
                                         type="string", scale=2)
                    g_win = gr.Checkbox(label="Only run overnight", value=False, scale=1,
                                        info="Pauses outside these hours and carries on the next night.")
                    g_w1 = gr.Textbox("23:00", label="from", scale=1)
                    g_w2 = gr.Textbox("06:30", label="until", scale=1)
                with gr.Row():
                    g_send = gr.Checkbox(label="Send the finished audiobook to my phone (KDE Connect)", value=bool(device_choices()), scale=2,
                                         info="Sent as soon as it is done; if your phone is out of reach it keeps trying.")
                    g_dev = gr.Dropdown(choices=device_choices(), value=device_default(), label="Phone", scale=2, allow_custom_value=True)
                    g_find = gr.Button("↻ Find devices", scale=1)
                with gr.Row():
                    go = gr.Button("Generate audiobook", variant="primary"); stop = gr.Button("Cancel")
                status4 = gr.Markdown()
                with gr.Row():
                    m4b = gr.File(label="M4B"); ch1 = gr.Audio(label="First chapter (preview)")
                with gr.Accordion("Advanced — pauses, emotion offset, speed", open=False):
                    gr.Markdown("### Every voice\nPauses and crossfade are applied when the chapters are put together, so they work the same for "
                                "Kokoro, F5-TTS, Chatterbox and Qwen3-TTS.")
                    with gr.Row():
                        xf = gr.Slider(0, 200, 60, step=10, label="Crossfade (ms) — all voices")
                        ps = gr.Slider(0, 1500, 350, step=50, label="Sentence pause (ms) — all voices")
                        pp = gr.Slider(0, 2000, 700, step=50, label="Paragraph pause (ms) — all voices")
                        pc = gr.Slider(0, 1500, 250, step=50, label="Speaker-change pause (ms) — all voices")
                    with gr.Row():
                        p_cont = gr.Slider(0, 600, 140, step=10, label="Pause when a sentence carries on after a quote (ms) — all voices")
                        p_tag = gr.Slider(0, 600, 120, step=10, label="Pause before a speaker tag (“said Danglars”) (ms) — all voices")
                    gr.Markdown("### Chatterbox only")
                    emo_base = gr.Slider(-0.3, 0.3, 0.0, step=0.05, label="Emotion offset (− calmer, + more dramatic) — Chatterbox only",
                                         info="Works with “Emotion from the text” ticked. The other voices have no emotion control.")
                    cb_workers = gr.Checkbox(value=False, label="Use two workers at once — Chatterbox only",
                                             info="About 22% faster, but needs ~4–8 GB of GPU memory each. Not recommended for GPUs with less than 16 GB of VRAM. "
                                                  "Off = one worker.")
                    with gr.Row():
                        with gr.Column():
                            gr.Markdown("### Kokoro only")
                            k_workers = gr.Slider(1, 6, 4, step=1, label="Voices made at once (more = faster, ~0.4 GB each) — Kokoro only")
                        with gr.Column():
                            gr.Markdown("### F5-TTS only")
                            f5_half = gr.Checkbox(value=True, label="Half precision (about 4.8× faster on Arc, same sound) — F5-TTS only")
                    gr.Markdown("*Qwen3-TTS has no settings of its own here.*")

            with gr.Tab("5 · Queue"):
                gr.Markdown("Every book you generate is added here and made by the queue runner, which watches it: if it stalls (a GPU hang) "
                            "or crashes it is restarted, it waits while anything else is using the GPU, and it carries on if you close the app. "
                            "Set a start time or overnight hours on the **Generate** tab.")
                q_state = gr.Markdown(queue_status())
                q_now = gr.Markdown(queue_now())
                with gr.Row():
                    q_runner = gr.Button(**runner_button_args())
                q_msg = gr.Markdown()
                with gr.Accordion("Alerts and disk space", open=False):
                    gr.Markdown("Get a message on your phone when a book finishes or fails, when the queue is empty, and when it paused itself "
                                "after repeated failures. Install the free **ntfy** app and subscribe to a topic name of your own choosing; "
                                "type the same name here. Anyone who knows the name can read the messages, so make it long and random.")
                    with gr.Row():
                        q_ntfy = gr.Textbox(value=alerts_settings()[0], label="ntfy topic", scale=3)
                        q_ntfy_save = gr.Button("Save", scale=1); q_ntfy_test = gr.Button("Send a test", scale=1)
                    q_clean = gr.Checkbox(value=alerts_settings()[1], label="Free disk space: delete a book's clips and chapter files once it is finished "
                                                                            "(and sent to the phone, if sending). You would have to remake the speech to change anything later.")
                    q_alert_msg = gr.Markdown()
                q_tbl = gr.Dataframe(value=queue_table(), headers=QUEUE_HEADERS, interactive=False, wrap=True,
                                     label="Queue — the book at the top is made first")
                q_pick = gr.CheckboxGroup(choices=[], value=[], label="Select books (tick one or more, then use the buttons below)")
                with gr.Row():
                    q_top = gr.Button("⏫ To the top"); q_up = gr.Button("🔼 Up"); q_down = gr.Button("🔽 Down"); q_bottom = gr.Button("⏬ To the bottom")
                with gr.Row():
                    q_hold = gr.Button("⏸ Pause / save for later"); q_resume = gr.Button("▶️ Resume")
                    q_cancel = gr.Button("Cancel"); q_remove = gr.Button("Remove from list")
                    q_open = gr.Button("Open in the other tabs")
                with gr.Row():
                    q_send = gr.Button("📱 Send to my phone when done"); q_nosend = gr.Button("Don’t send to my phone")
                with gr.Accordion("Change the schedule of the ticked books", open=False):
                    gr.Markdown("Switch one book or a whole batch to a new start time or overnight hours.")
                    with gr.Row():
                        qs_when = gr.DateTime(label="Start at", include_time=True, type="string", scale=2)
                        qs_win = gr.Checkbox(label="Only run overnight", value=False, scale=1)
                        qs_w1 = gr.Textbox("23:00", label="from", scale=1)
                        qs_w2 = gr.Textbox("06:30", label="until", scale=1)
                    with gr.Row():
                        qs_apply = gr.Button("Apply this schedule", variant="primary")
                        qs_asap = gr.Button("Start as soon as possible, any time")
                q_timer = gr.Timer(10)

            with gr.Tab("✨ Assistant"):
                gr.Markdown("Ask Claude to set up this book — “give the women different voices”, “make the narration calmer”, "
                            "“fix how Morlock is said”. Claude works on a **copy** of the settings; you see exactly what would change "
                            "and nothing is touched until you press Apply. It cannot start a generation, run commands or hear audio. "
                            "Each question uses your Claude Code login (usually a few cents of usage); the cast, a few sample lines "
                            "and the lexicon are sent to Anthropic.")
                a_chat = gr.Chatbot(height=320, label="Conversation")
                with gr.Row():
                    a_in = gr.Textbox(label="What would you like set up?", scale=4, lines=2,
                                      placeholder="e.g. Give each female character a different female voice, and make the narrator a little slower")
                    a_go = gr.Button("Ask Claude", variant="primary", scale=1)
                a_cost = gr.Markdown()
                a_note = gr.Markdown()
                a_diff = gr.Markdown("_Nothing to show._")
                a_cover = gr.Image(label="Proposed cover", type="filepath", height=300, interactive=False, visible=False)
                with gr.Row():
                    a_apply = gr.Button("Apply these changes", variant="primary", interactive=False)
                    a_discard = gr.Button("Discard", interactive=False)
                a_session, a_spent = gr.State(""), gr.State(0.0)

            with gr.Tab("🎨 Cover"):
                gr.Markdown("Make the cover. Pick a picture, choose how it is laid out, and press **Use this cover**. "
                            "Uses the title and author from the Book tab. Searches only pictures that are free to use "
                            "(public domain or CC0) on Wikimedia Commons, and avoids book covers. For people, the picture sits like a "
                            "window on a book-cloth colour; for scenery, it fills the background.")
                with gr.Row():
                    with gr.Column(scale=3):
                        gr.Markdown("### 1 · Picture")
                        cv_upload = gr.Image(label="Use your own picture (optional)", type="filepath", height=140)
                        with gr.Row():
                            cv_query = gr.Textbox(label="…or search free pictures", scale=4,
                                                  placeholder="e.g. Civil War painting, Frederick Douglass portrait, Martian landscape")
                            cv_search = gr.Button("Search", scale=1)
                        cv_status = gr.Markdown()
                        cv_gallery = gr.Gallery(label="Results (click one)", columns=4, height=260, object_fit="contain", allow_preview=False)
                        gr.Markdown("### 2 · Look")
                        cv_layout = gr.Radio(list(COVER_LAYOUTS), value=list(COVER_LAYOUTS)[0], label="Layout")
                        cv_colour = gr.Dropdown(list(covers_mod.CLOTH), value="navy", label="Cloth colour (picture-window layout)")
                        with gr.Row():
                            cv_fx = gr.Slider(0, 1, 0.5, step=0.05, label="Focus left ↔ right (full background)")
                            cv_fy = gr.Slider(0, 1, 0.5, step=0.05, label="Focus up ↕ down (full background)")
                            cv_zoom = gr.Slider(0.8, 2.5, 1.0, step=0.05, label="Zoom (full background)")
                        cv_make = gr.Button("Update preview", variant="primary")
                    with gr.Column(scale=2):
                        gr.Markdown("### 3 · Preview")
                        cv_preview = gr.Image(label="Cover preview", type="filepath", height=420, interactive=False)
                        cv_use = gr.Button("Use this cover", variant="primary")
                        cv_msg = gr.Markdown()
                cv_hits, cv_pic, cv_credit = gr.State([]), gr.State(""), gr.State("")

            with gr.Tab("🎙 Clone", elem_id="clone-tab"):
                gr.Markdown("Clone a voice from a short clean clip (5–12 s, one speaker, no music), compare the takes, then save it. "
                            "A saved voice can be spoken by F5-TTS, Chatterbox or Qwen3-TTS.")
                with gr.Row(equal_height=False):
                    with gr.Column(scale=1):
                        with gr.Group():
                            gr.Markdown("### 1 · Reference clip")
                            c_audio = gr.Audio(label="Clip", type="filepath", sources=["upload", "microphone"])
                            c_text = gr.Textbox(label="Exact transcript of the clip", lines=2)
                            c_trans = gr.Button("Auto-transcribe with Whisper", size="sm")
                        with gr.Accordion("Take a reference from an audiobook (.m4b)", open=False):
                            gr.Markdown("Open a narrated audiobook and pick a chapter. Clean 6–12 s clips are cut on pauses; Whisper writes "
                                        "down each short clip, and if you add the ebook, the book's exact words replace what Whisper heard.")
                            with gr.Row():
                                ab_file = gr.File(label="Audiobook (.m4b / .m4a / .mp3)", file_types=[".m4b", ".m4a", ".mp3", ".mp4"])
                                ab_ebook = gr.File(label="Ebook (optional, .epub / .txt)", file_types=[".epub", ".txt"])
                            with gr.Row():
                                ab_read_btn = gr.Button("Read chapters")
                                ab_chapter = gr.Dropdown(label="Chapter", choices=[], scale=3)
                            with gr.Row():
                                ab_offset = gr.Number(label="Start N seconds into the chapter", value=15, precision=0)
                                ab_minutes = gr.Slider(1, 10, 3, step=1, label="Minutes to analyse")
                                ab_target = gr.Slider(6, 12, 9, step=1, label="Sample length (s)")
                                ab_count = gr.Slider(2, 8, 5, step=1, label="Samples to find")
                                ab_model = gr.Dropdown(["openai/whisper-base.en", "openai/whisper-small.en", "openai/whisper-medium.en"],
                                                       value="openai/whisper-small.en", label="Whisper model")
                            ab_find_btn = gr.Button("Transcribe & find samples", variant="primary")
                            ab_status = gr.Markdown()
                            ab_clips = gr.State([])
                            with gr.Row():
                                ab_dd = gr.Dropdown(label="Sample", choices=[], scale=3, type="value")
                                ab_use_btn = gr.Button("Use as reference", variant="primary")
                            with gr.Row():
                                ab_audio = gr.Audio(label="Sample preview")
                                ab_text = gr.Textbox(label="Sample transcript", lines=3)
                            ab_info = gr.Markdown()
                            ab_asr = gr.Textbox(label="What Whisper heard in this sample", lines=2, interactive=False)
                    with gr.Column(scale=1):
                        with gr.Group():
                            gr.Markdown("### 2 · Generate takes")
                            c_engine = gr.Radio(list(ENGINE_LABELS), value="F5-TTS", label="Engine")
                            c_test = gr.Textbox(label="Test text", value=TEST_TEXT, lines=3)
                            c_go = gr.Button("Generate takes", variant="primary")
                            c_status = gr.Markdown()
                        with gr.Accordion("Tuning for this engine — emotion, pace, quality, seed", open=False):
                            with gr.Row():
                                c_seed = gr.Number(label="Seed (-1 = random)", value=1234, precision=0)
                                c_takes = gr.Slider(1, 3, 3, step=1, label="Takes to compare")
                            with gr.Group(visible=False) as cb_group:
                                c_exag = gr.Slider(0.0, 1.5, 0.5, step=0.05, label="Chatterbox emotion / exaggeration (0.5 = natural)")
                                c_cfgw = gr.Slider(0.0, 1.0, 0.4, step=0.05, label="Chatterbox pace & guidance (lower = faster, more energetic)")
                            with gr.Group() as f5_group:
                                c_speed = gr.Slider(0.5, 1.6, 1.0, step=0.05, label="F5 pace — speaking speed (the voice's default cadence)")
                                c_steps = gr.Slider(8, 64, 32, step=1, label="F5 quality steps (more = cleaner, slower)")
                                c_cfg = gr.Slider(0.0, 5.0, 2.0, step=0.1, label="F5 voice guidance (higher = closer to the clip)")
                                c_sway = gr.Slider(-2.0, 1.0, -1.0, step=0.1, label="F5 sway sampling (prosody smoothing)")
                                c_xfade = gr.Slider(0.0, 0.5, 0.15, step=0.01, label="F5 cross-fade between long-text pieces (s)")
                                c_rms = gr.Slider(0.03, 0.3, 0.1, step=0.01, label="F5 loudness of reference (target RMS)")
                            c_reset = gr.Button("Reset to defaults", size="sm")
                c_seeds = gr.State([])
                with gr.Group():
                    gr.Markdown("### Takes")
                    with gr.Row():
                        take1 = gr.Audio(label="Take 1", autoplay=False)
                        take2 = gr.Audio(label="Take 2", autoplay=False)
                        take3 = gr.Audio(label="Take 3", autoplay=False)
                with gr.Group():
                    gr.Markdown("### 3 · Save this voice")
                    with gr.Row():
                        c_name = gr.Textbox(label="Voice name", scale=3)
                        c_pick = gr.Radio(["Take 1", "Take 2", "Take 3"], value="Take 1", label="Keep the take", scale=3)
                        c_save = gr.Button("Save voice", variant="primary", scale=1)
                    c_notes = gr.Textbox(label="Notes (optional)")
                with gr.Group():
                    gr.Markdown("### 4 · Saved voices")
                    with gr.Row():
                        lib_dd = gr.Dropdown(label="Saved voice", choices=vlib.list_voices(), scale=4)
                        c_load = gr.Button("Load into editor", scale=1); c_del = gr.Button("Delete", scale=1)
                    with gr.Row():
                        lib_role = gr.Dropdown(label="Give it to a role (needs a parsed book)", choices=[], scale=3)
                        lib_engine = gr.Dropdown(list(ENGINE_LABELS), value="F5-TTS", label="…voiced by", scale=2)
                        c_roles = gr.Button("Refresh roles", scale=1); c_assign = gr.Button("Assign to role", scale=1)
                    with gr.Accordion("Hear this voice in every engine", open=False):
                        c_compare = gr.Button("Compare engines")
                        with gr.Row():
                            cmp_f5 = gr.Audio(label="F5-TTS"); cmp_cb = gr.Audio(label="Chatterbox"); cmp_qw = gr.Audio(label="Qwen3-TTS")
                gr.Markdown("### Extra")
                with gr.Accordion("Make a Kokoro voice from a recording (a voice pack that works natively in Kokoro)", open=False):
                    gr.Markdown("Kokoro can't clone from audio by itself, so this *searches* for a Kokoro voice whose speech scores as "
                                "close as possible to the speaker (WavLM speaker similarity), starting from the nearest stock voice. "
                                "The result is a normal Kokoro voice: it takes IPA, runs about 7× faster than F5, and appears in the "
                                "voice menus. It captures timbre and general delivery, not a perfect copy — compare by ear.")
                    with gr.Row():
                        vp_source = gr.Radio(VP_SOURCES, value=VP_SOURCES[0], label="Make it sound like", scale=3)
                        vp_around = gr.Slider(0.05, 0.95, 0.5, step=0.05, label="…snippet near this point of the book (0.5 = middle)", scale=2)
                    with gr.Row():
                        vp_clone = gr.Dropdown(label="Saved clone voice", choices=vlib.list_voices(), scale=2)
                        vp_audio = gr.File(label="…or an audio file", file_types=["audio"], scale=2)
                    with gr.Row():
                        vp_name = gr.Textbox(label="Name for the new Kokoro voice", scale=2)
                        vp_steps = gr.Slider(20, 200, 60, step=10, label="Search steps (more = closer, slower)", scale=2)
                        vp_warm = gr.Dropdown(label="Start from", choices=["(start from the closest stock voice)"] + _mine(),
                                              value="(start from the closest stock voice)", scale=2)
                    vp_go = gr.Button("Make the Kokoro voice", variant="primary")
                    vp_note = gr.Markdown()
                    with gr.Row():
                        vp_real = gr.Audio(label="Target (real recording)")
                        vp_new = gr.Audio(label="New Kokoro voice")
                        vp_stock = gr.Audio(label="Closest stock voice")
                    with gr.Row():
                        vp_mine = gr.Dropdown(label="My Kokoro voices", choices=_mine(), scale=2)
                        vp_text = gr.Textbox(label="Say", value=VP_TEXT, scale=4)
                        vp_hear_btn = gr.Button("▶ Hear"); vp_del_btn = gr.Button("Delete")
                    vp_audio_out = gr.Audio(label="My Kokoro voice")
                    with gr.Row():
                        vp_label = gr.Textbox(label="Menu name (e.g. John)", scale=2)
                        vp_tags = gr.Textbox(label="Description (e.g. US male)", scale=2)
                        vp_tag_btn = gr.Button("Tag this voice")

        load_btn.click(load_book, [book, maxch], [project, chap_df, title, author, cover, status1]).then(
            voice_settings, project, [mode, single_dd, single_speed, single_group]).then(
            gen_settings, project, [emo_cb, emo_base, lex_rd, p_cont, p_tag, cb_workers]).then(speed_text, project, speed_md)
        parse_btn.click(run_parse, [project, chap_df, endpoint, judge_cb],
                        [roles_state, seg_df, status2, char_dd, target_dd, lines_df, hints]).then(
            voice_settings, project, [mode, single_dd, single_speed, single_group])
        save_hints_btn.click(save_hints, [project, hints], status2)
        char_dd.change(show_lines, [project, char_dd], lines_df)
        review_out = [lines_df, roles_state, char_dd, target_dd]
        move_btn.click(reassign, [project, lines_df, target_dd, char_dd], review_out + [status2])
        merge_btn.click(merge_role, [project, char_dd, target_dd], review_out + [status2])
        pv_line_btn.click(preview_line, [project, lines_df, target_dd], [line_audio, status2])
        mode.change(set_mode, [project, mode], [single_group, status2])
        single_dd.change(set_single, [project, single_dd, single_speed], status2)
        single_speed.release(set_single, [project, single_dd, single_speed], status2)
        single_hear.click(hear_voice, [project, single_dd, single_speed, pv_text], pv_audio)
        refresh_btn.click(refresh_voices, project, single_dd)
        save_seg_btn.click(save_segments, [project, seg_df], [roles_state, char_dd, target_dd, status2])
        save_seg_btn.click(show_lines, [project, char_dd], lines_df)
        lex_btn.click(run_lexicon, [project, seed_dd, lang_tb, detect_cb, title, author], [lex_df, status3]).then(
            lambda df: gr.update(choices=list(df["term"])), lex_df, term_dd)
        fill_btn.click(lex_autofill, [project, lex_df, lang_tb, known_cb], [lex_df, status3])
        lex_df.select(fw_pick, lex_df, [fw_word, fw_ipa, fw_resp])
        fw_hear_btn.click(fw_hear, [project, fw_word, fw_ipa, fw_resp, fw_voice, fw_tmpl], [fw_audio, fw_note])
        fw_search_btn.click(fw_search, [project, fw_word, fw_ipa, fw_voice, fw_n], [fw_table, fw_files, fw_note])
        fw_table.select(fw_search_pick, [fw_table, fw_files], [fw_audio, fw_resp])
        fw_save_btn.click(fw_save, [project, fw_word, fw_ipa, fw_resp], [lex_df, fw_note])
        spell_btn.click(lex_autospell, [project, lex_df], [lex_df, status3])
        lookup_btn.click(lex_lookup, [project, lex_df, lang_tb, wiki_tb, bible_cb, offline_cb], [lex_df, status3])
        save_lex_btn.click(save_lexicon, [project, lex_df], status3)
        hear_btn.click(hear_term, [project, lex_df, term_dd, which_rd], term_audio)
        QOUT = [q_state, q_tbl, q_now, q_runner, q_pick]
        QMSG = [q_msg] + QOUT
        go.click(generate_queued, [project, chap_df, title, author, cover, xf, ps, pp, pc, k_workers, f5_half, emo_cb, emo_base, lex_rd, p_cont, p_tag, cb_workers, g_when, g_win, g_w1, g_w2, g_send, g_dev],
                 status4).then(queue_refresh, q_pick, QOUT)
        stop.click(gen_stop, project, status4)
        gen_shown = gr.State("")
        q_timer.tick(gen_panel, [project, gen_shown], [status4, m4b, ch1, gen_shown])
        q_timer.tick(speed_text, project, speed_md)
        gen_inputs = [project, chap_df, title, author, cover, xf, ps, pp, pc, k_workers, f5_half, emo_cb, emo_base, lex_rd, p_cont, p_tag, cb_workers]
        q_runner.click(queue_toggle_runner, None, QMSG)
        q_ntfy_save.click(alerts_save, [q_ntfy, q_clean], q_alert_msg)
        q_clean.change(alerts_save, [q_ntfy, q_clean], q_alert_msg)
        q_ntfy_test.click(alerts_test, q_ntfy, q_alert_msg)
        for btn, where in ((q_top, "top"), (q_up, "up"), (q_down, "down"), (q_bottom, "bottom")):
            btn.click(lambda picked, w=where: queue_move(picked, w), q_pick, QMSG)
        g_find.click(devices_refresh, None, g_dev)
        q_send.click(queue_send, [q_pick, g_dev], QMSG)
        q_nosend.click(queue_nosend, q_pick, QMSG)
        q_hold.click(queue_hold, q_pick, QMSG)
        q_resume.click(queue_resume, q_pick, QMSG)
        q_cancel.click(queue_cancel, q_pick, QMSG)
        q_remove.click(queue_remove, q_pick, QMSG)
        qs_apply.click(queue_reschedule, [q_pick, qs_when, qs_win, qs_w1, qs_w2], QMSG)
        qs_asap.click(queue_asap, q_pick, QMSG)
        open_outputs = [project, chap_df, title, author, cover, status1, roles_state, seg_df, status2, char_dd, target_dd, lines_df, hints,
                        mode, single_dd, single_speed, single_group, emo_cb, emo_base, lex_rd, p_cont, p_tag, cb_workers]
        q_open.click(queue_open, q_pick, open_outputs)
        ui.load(queue_refresh, q_pick, QOUT)          # reconnect: show the queue as it is right now
        ui.load(reconnect_on_load, project, open_outputs)              # and open the book that is being made
        q_timer.tick(queue_refresh, q_pick, QOUT)
        cv_search.click(cover_search, cv_query, [cv_hits, cv_gallery, cv_status])
        cv_query.submit(cover_search, cv_query, [cv_hits, cv_gallery, cv_status])
        cv_look = [project, title, author, cv_upload, cv_pic, cv_layout, cv_colour, cv_fx, cv_fy, cv_zoom]
        cv_gallery.select(cover_pick, [project, cv_hits], [cv_pic, cv_credit, cv_status]).then(cover_render, cv_look, cv_preview)
        cv_make.click(cover_render, cv_look, cv_preview)
        cv_upload.upload(cover_render, cv_look, cv_preview)
        cv_layout.change(cover_refresh, cv_look, cv_preview)
        cv_colour.change(cover_refresh, cv_look, cv_preview)
        cv_use.click(cover_use, [project, cv_preview, cv_credit, cv_upload], [cover, cv_msg])
        a_go.click(assistant_ask, [project, a_in, a_chat, a_session, a_spent],
                   [a_chat, a_session, a_spent, a_cost, a_note, a_diff, a_apply, a_discard, a_in, a_cover])
        a_in.submit(assistant_ask, [project, a_in, a_chat, a_session, a_spent],
                    [a_chat, a_session, a_spent, a_cost, a_note, a_diff, a_apply, a_discard, a_in, a_cover])
        a_apply.click(assistant_apply, project, [a_note, a_diff, a_apply, a_discard, a_cover, cover]).then(
            gen_settings, project, [emo_cb, emo_base, lex_rd, p_cont, p_tag, cb_workers]).then(
            voice_settings, project, [mode, single_dd, single_speed, single_group])
        a_discard.click(assistant_discard, project, [a_note, a_diff, a_apply, a_discard, a_cover])
        ab_read_btn.click(ab_read, ab_file, [ab_chapter, ab_status])
        ab_find_btn.click(ab_find, [ab_file, ab_chapter, ab_ebook, ab_offset, ab_minutes, ab_target, ab_model, ab_count],
                          [ab_clips, ab_dd, ab_status])
        ab_dd.change(ab_pick, [ab_clips, ab_dd], [ab_audio, ab_text, ab_asr, ab_info])
        ab_use_btn.click(ab_use, [ab_clips, ab_dd], [c_audio, c_text, ab_info])
        vp_go.click(vp_make, [vp_source, vp_clone, vp_audio, ab_file, vp_around, vp_name, vp_steps, vp_warm],
                    [vp_note, vp_real, vp_new, vp_stock, vp_mine, vp_warm, single_dd])
        vp_hear_btn.click(vp_hear, [vp_mine, vp_text], vp_audio_out)
        vp_mine.change(vp_load_tag, vp_mine, [vp_label, vp_tags])
        vp_tag_btn.click(vp_tag, [vp_mine, vp_label, vp_tags], [vp_note, single_dd])
        vp_del_btn.click(vp_delete, vp_mine, [vp_mine, vp_warm, vp_note])
        sliders = [c_speed, c_steps, c_cfg, c_sway, c_xfade, c_rms, c_seed]
        c_trans.click(clone_transcribe, c_audio, c_text)
        c_engine.change(lambda e: (gr.update(visible=e == "Chatterbox"), gr.update(visible=e == "F5-TTS")), c_engine, [cb_group, f5_group])
        c_go.click(clone_generate, [c_audio, c_text, c_test] + sliders + [c_takes, c_engine, c_exag, c_cfgw], [take1, take2, take3, c_seeds, c_status])
        c_reset.click(lambda: [vlib.DEFAULTS[k] for k in CLONE_KEYS], None, sliders)
        c_save.click(clone_save, [c_name, c_audio, c_text] + sliders + [c_seeds, c_pick, c_notes, c_engine, c_exag, c_cfgw], [lib_dd, c_status])
        c_load.click(clone_load, lib_dd, [c_audio, c_text, c_name, c_notes] + sliders + [c_exag, c_cfgw, c_status])
        c_del.click(clone_delete, lib_dd, [lib_dd, c_status])
        c_roles.click(clone_roles, project, lib_role)
        c_assign.click(clone_assign, [project, lib_dd, lib_role, lib_engine], c_status)
        c_compare.click(clone_compare, [lib_dd, c_test], [cmp_f5, cmp_cb, cmp_qw, c_status])
    return ui


def main():
    from . import jobqueue
    jobqueue.ensure_supervisor()                 # queued books are always watched, even when the app was closed
    build_ui().queue().launch(server_name="127.0.0.1", server_port=7860, css=CSS)


if __name__ == "__main__":
    main()
