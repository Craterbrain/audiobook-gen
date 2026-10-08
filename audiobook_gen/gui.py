"""Gradio GUI: Book -> Cast -> Lexicon -> Generate. Run: python -m audiobook_gen.gui"""
import json
import re
import shutil
import tempfile
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
from .synth import get_engine, resolve_voice, synthesize_iter

ROOT = Path(__file__).resolve().parent.parent
KOKORO_VOICES = ["af_heart", "af_bella", "af_nicole", "af_sarah", "af_sky", "am_adam", "am_echo",
                 "am_liam", "am_michael", "am_onyx", "am_puck", "bf_emma", "bf_isabella",
                 "bm_daniel", "bm_fable", "bm_george", "bm_lewis"]
MAX_ROLES = 30
MODE_MULTI = "Different voice for each character"
MODE_SINGLE = "One narrator reads everything"
ENGINE_LABELS = {"F5-TTS": "f5", "Chatterbox": "chatterbox", "Qwen3-TTS": "qwen3"}
LEX_MODES = {"Tested respellings — plain spelling, fixed where tested (Chatterbox, recommended)": "verified",
             "Real IPA — for Qwen3": "rawipa",
             "Phonetic respelling — F5 style": "respell",
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
    return "Lexicon saved (respellings for F5 are made from the IPA where you left them blank)."


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


def gen_settings(project):
    """Generate-tab engine options as saved in the project's config."""
    cfg = _cfg(project)
    mode = cfg.get("text_lexicon", "respell")
    label = next((k for k, v in LEX_MODES.items() if v == mode), list(LEX_MODES)[2])
    pm = cfg.get("pacing_ms") or {}
    return (bool(cfg.get("emotion")), float(cfg.get("emotion_base", 0.0)), label,
            int(pm.get("continuation", 140)), int(pm.get("tag", 120)),
            int((cfg.get("workers") or {}).get("chatterbox", 1)))


def generate(project, chap_df, title, author, cover, crossfade, p_sent, p_para, p_speaker, kokoro_workers=4, f5_half=True,
             emotion=False, emo_base=0.0, lex_label=None, p_cont=140, p_tag=120, cb_workers=1):
    work = _wd(project)
    if not (work / "segments.json").exists():
        raise gr.Error("Parse the book on the Cast tab first.")
    cfg = _cfg(project)
    cfg.update(crossfade_ms=int(crossfade), workers={**(cfg.get("workers") or {}), "kokoro": int(kokoro_workers)},
               f5_precision="float16" if f5_half else "float32")
    cfg["pacing_ms"] = {**cfg["pacing_ms"], "sentence": int(p_sent), "paragraph": int(p_para),
                        "speaker_change": int(p_speaker), "continuation": int(p_cont), "tag": int(p_tag)}
    cfg["emotion"], cfg["emotion_base"] = bool(emotion), float(emo_base)
    cfg["workers"] = {**cfg["workers"], "chatterbox": int(cb_workers)}
    if lex_label in LEX_MODES:
        cfg["text_lexicon"] = LEX_MODES[lex_label]
    _save_cfg(project, {**_cfg(project), **{k: cfg[k] for k in ("emotion", "emotion_base", "text_lexicon") if k in cfg},
                        "workers": {**(_cfg(project).get("workers") or {}), "chatterbox": int(cb_workers)}})
    vram = gpu_vram_gib()
    if int(cb_workers) > 1 and vram and vram < 16:
        yield (f"Heads up: Chatterbox with {int(cb_workers)} workers needs about 16 GB of GPU memory and this card has {vram:.0f} GB. "
               "It may run out of memory; if it stalls, set it back to 1."), None, None
    only = _selected(chap_df)
    for done, total, msg in synthesize_iter(work, cfg, only):
        yield f"Synthesizing {done}/{total} — {msg}", None, None
    yield "Assembling chapters + M4B…", None, None
    out = assemble(work, cfg, ROOT / "out" / f"{work.name}.m4b", cover, title, author, only)
    first = sorted((work / "chapters").glob("*.wav"))
    yield f"Done → `{out}`", str(out), (str(first[0]) if first else None)


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
                gr.Markdown("How names and hard words are pronounced. **Build lexicon**, fill in the IPA (try **Look up**), review, then **Save**.")
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
                with gr.Row():
                    emo_cb = gr.Checkbox(label="Emotion from the text (Chatterbox)", scale=1,
                                         info="Each sentence gets its own expressiveness, from its mood and tags like “cried” or “whispered”.")
                    lex_rd = gr.Radio(list(LEX_MODES), value=list(LEX_MODES)[0], scale=3,
                                      label="How names and hard words are spoken by F5, Chatterbox and Qwen3")
                with gr.Row():
                    go = gr.Button("Generate audiobook", variant="primary"); stop = gr.Button("Stop")
                status4 = gr.Markdown()
                with gr.Row():
                    m4b = gr.File(label="M4B"); ch1 = gr.Audio(label="First chapter (preview)")
                with gr.Accordion("Advanced — pauses, emotion offset, speed", open=False):
                    emo_base = gr.Slider(-0.3, 0.3, 0.0, step=0.05, label="Emotion offset (− calmer, + more dramatic)")
                    with gr.Row():
                        xf = gr.Slider(0, 200, 60, step=10, label="Crossfade (ms)")
                        ps = gr.Slider(0, 1500, 350, step=50, label="Sentence pause (ms)")
                        pp = gr.Slider(0, 2000, 700, step=50, label="Paragraph pause (ms)")
                        pc = gr.Slider(0, 1500, 250, step=50, label="Speaker-change pause (ms)")
                    with gr.Row():
                        p_cont = gr.Slider(0, 600, 140, step=10, label="Pause when a sentence carries on after a quote (ms)")
                        p_tag = gr.Slider(0, 600, 120, step=10, label="Pause before a speaker tag (“said Danglars”) (ms)")
                    with gr.Row():
                        k_workers = gr.Slider(1, 6, 4, step=1, label="Kokoro: voices made at once (more = faster, ~0.4 GB each)")
                        f5_half = gr.Checkbox(value=True, label="F5: half precision (about 4.8× faster on Arc, same sound)")
                    cb_workers = gr.Slider(1, 2, 1, step=1, label="Chatterbox: workers at once (2 is about 22% faster, ~4–8 GB of GPU memory each)",
                                           info="Not recommended for GPUs with less than 16 GB of VRAM.")

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
            gen_settings, project, [emo_cb, emo_base, lex_rd, p_cont, p_tag, cb_workers])
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
        run = go.click(generate, [project, chap_df, title, author, cover, xf, ps, pp, pc, k_workers, f5_half, emo_cb, emo_base, lex_rd, p_cont, p_tag, cb_workers],
                       [status4, m4b, ch1])
        stop.click(None, cancels=[run])
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
    build_ui().queue().launch(server_name="127.0.0.1", server_port=7860, css=CSS)


if __name__ == "__main__":
    main()
