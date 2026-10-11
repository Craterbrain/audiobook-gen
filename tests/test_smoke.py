import json, sys
import pytest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from audiobook_gen.lexicon import Preprocessor
from audiobook_gen.speakers import parse_chapter
from audiobook_gen.synth import chunk_text
from audiobook_gen.assemble import ffmeta


def test_preprocessor_respell_and_case():
    pp = Preprocessor([{"term": "Cephas", "ipa": "ˈsiːfəs", "respell": "See-fus"}])
    assert pp("Thou shalt be called Cephas, not cephasian.") == "Thou shalt be called Seefus, not cephasian."


def test_preprocessor_ipa():
    pp = Preprocessor([{"term": "Siloam", "ipa": "sɪˈloʊəm", "respell": ""}], "ipa")
    assert pp("pool of Siloam") == "pool of [Siloam](/sɪˈloʊəm/)"


def test_parser_unquoted_and_quoted():
    ch = {"index": 1, "text": "Jesus answered and said unto him, Verily I say unto thee.\n\n“Go home,” said Anna."}
    segs = parse_chapter(ch, {})
    assert [(s["speaker"]) for s in segs][:2] == ["Narrator", "Jesus"]
    assert any(s["speaker"] == "Anna" and s["text"] == "Go home," for s in segs)


def test_chunking():
    chunks = chunk_text("One. " * 200, 100)
    assert all(len(c) <= 100 for c in chunks) and len(chunks) > 5


def test_ffmeta():
    out = ffmeta([("A", 1.5), ("B", 2.0)], "T", "Au")
    assert "START=0\nEND=1500" in out and "START=1500\nEND=3500" in out


def test_normalize_abbreviations_and_numerals():
    from audiobook_gen.lexicon import normalize
    assert normalize("M. Morrel and Mme. Dantès") == "Monsieur Morrel and Madame Dantès"
    assert normalize("King Louis XVIII") == "King Louis the eighteenth"
    assert normalize("Act II, Scene III") == "Act two, Scene three"


def test_possessive_keeps_suffix():
    pp = Preprocessor([{"term": "Edmond", "ipa": "ɛdmɔ̃", "respell": ""}], "ipa")
    assert pp("Edmond’s father") == "[Edmond](/ɛdmɔ̃/)’s father"


def test_oov_detection_and_french_suggestion():
    from audiobook_gen.lexicon import _is_known, suggest_ipa
    assert _is_known("murmured") and _is_known("inquired") and _is_known("don’t")
    assert not _is_known("Danglars") and not _is_known("Caderousse")
    assert suggest_ipa("Mercédès", "fr").startswith("m")


def test_viterbi_uses_turn_taking():
    from audiobook_gen.attribution import decode
    mk = lambda para, explicit=None: dict(para=para, text="x", explicit=explicit, pron=None, split=False,
                                          cont=False, other=False, lead=None, addressed=set(), recent=set(),
                                          cands=["Ann", "Bob"])
    items = [mk(0, "Ann"), mk(1), mk(2), mk(3)]   # unattributed replies alternate
    assert decode(items, {}) == ["Ann", "Bob", "Ann", "Bob"]


def test_language_detection_uses_book_consensus():
    from audiobook_gen.lexicon import detect_languages, detect_wordfreq
    assert detect_wordfreq("procureur", False)[0] == "fr" and detect_wordfreq("overtook", False)[0] == ""
    mk = lambda t, k="name": dict(term=t, kind=k, known=False, lang="", lang_src="", lang_conf=0.0, source="auto")
    fr = {t: mk(t, "word" if t.islower() else "name") for t in
          ("Réserve", "Allées", "Palais", "procureur", "Danglars", "overtook")}
    out = detect_languages(fr, [], "", None)
    assert out["Réserve"]["lang"] == "fr"
    assert out["Danglars"]["lang"] == "fr"        # unmatched name takes the book's language (French votes)
    assert out["overtook"]["lang"] == ""          # unmatched plain word stays English
    out = detect_languages({t: mk(t) for t in ("Banquo", "Cawdor")}, [], "", None)
    assert all(e["lang"] == "" for k, e in out.items() if k != "__stats__")


def test_respelling_and_phrases():
    from audiobook_gen.lexicon import ipa_to_respell
    assert ipa_to_respell("dɑ̃ɡlˈaʁ") == "dahn-GLAHR"
    assert ipa_to_respell("alˈe də mɛilˈɑ̃") == "ah-LAY duh meh-ee-LAHN"
    pp = Preprocessor([{"term": "Allées de Meilhan", "ipa": "alˈe də mɛilˈɑ̃", "respell": ""}], "ipa")
    assert pp("the Allées\nde Meilhan.") == "the [Allées de Meilhan](/alˈe də mɛilˈɑ̃/)."


def test_voice_library_roundtrip(tmp_path, monkeypatch):
    import numpy as np
    import soundfile as sf
    from audiobook_gen import voices
    monkeypatch.setattr(voices, "LIB", tmp_path)
    ref = tmp_path / "in.wav"
    sf.write(ref, np.zeros(48000, np.float32), 48000)          # 1 s at 48 kHz -> stored as 24 kHz mono
    voices.save_voice("My Voice!", str(ref), "hello world", {"speed": 0.9, "seed": 7, "bogus": 1})
    assert voices.list_voices() == ["My Voice"]
    v = voices.load_voice("My Voice")
    assert v["speed"] == 0.9 and v["seed"] == 7 and v["nfe_step"] == 32 and "bogus" not in v
    assert sf.info(v["ref_audio"]).samplerate == 24000
    assert voices.resolve({"engine": "f5", "library": "My Voice", "speed": 1.2})["speed"] == 1.2
    voices.delete_voice("My Voice")
    assert voices.list_voices() == []


def test_ebook_text_match_and_pause_windows(tmp_path):
    import numpy as np
    import soundfile as sf
    from audiobook_gen import m4b, sync
    book = [{"text": "The sea was calm. Jesus saith unto them, Fill the waterpots with water. And they filled them up to the brim."},
            {"text": "Nothing here resembles the clip at all, only other words."}]
    r = sync.match("jesus saith unto them fill the water pots with water and they filled them up", sync.book_index(book))
    assert r["chapter"] == 0 and r["text"].startswith("Jesus saith unto them, Fill") and r["similarity"] > 0.8
    assert sync.match("quarterly tax filings for the committee", sync.book_index(book)) is None
    sr = 24000                                      # 2 s tone, 0.5 s gap, 8 s tone, 0.5 s gap, 2 s tone
    tone = lambda s: 0.3 * np.sin(2 * np.pi * 220 * np.arange(int(s * sr)) / sr).astype("float32")
    gap = np.zeros(int(0.5 * sr), "float32")
    sf.write(tmp_path / "w.wav", np.concatenate([tone(2), gap, tone(8), gap, tone(2)]), sr)
    (s, e), = m4b.speech_windows(tmp_path / "w.wav", target=8, lo=6, hi=12, top=1)
    assert abs(s - 2.5) < 0.1 and abs(e - 10.5) < 0.1


def test_auto_spelling_for_plain_text_engines():
    from audiobook_gen.lexicon import auto_respell, f5_text, fill_respell, ipa_to_respell
    assert ipa_to_respell("kˈAdɹWs") == "KAY-drows"           # readable form keeps stress marks
    assert f5_text("Nuh-THAN-yel", "Nathanael") == "Nuhthanyel"            # F5 gets no hyphens or capitals (it spells capitals out)
    assert f5_text("ah-LAY duh meh-ee-LAHN") == "ahlay duh meheelahn"
    e = {"term": "Danglars", "ipa": "dɑ̃ɡlˈaʁ", "respell": "", "guess": "dˈæŋɡləɹz", "known": False, "source": "auto"}
    assert auto_respell(e) == "dahnglahr"
    assert Preprocessor([e], "respell")("Danglars spoke.") == "Dahnglahr spoke."   # derived on the fly
    assert Preprocessor([{"term": "Nathanael", "ipa": "", "respell": "Nuh-than-yel"}], "respell")("Nathanael came") == "Nuhthanyel came"
    assert Preprocessor([e], "ipa")("Danglars spoke.") == "[Danglars](/dɑ̃ɡlˈaʁ/) spoke."
    english = {"term": "Graymalkin", "ipa": "", "respell": "", "guess": "ɡɹˈAmˌælkɪn", "known": False, "source": "auto"}
    assert auto_respell(english) == ""                         # no IPA: F5 keeps the word's own spelling
    typed = {"term": "Foo", "ipa": "fˈu", "respell": "Foooo", "source": "user", "respell_src": "user"}
    fill_respell([typed], refresh=True)
    assert typed["respell"] == "Foooo"                         # never overwrite what you typed


def test_phone_distance_ranks_closer_pronunciations_first():
    from audiobook_gen.respell_search import distance
    target = ["n", "ə", "θ", "æ", "n", "j", "ə", "l"]
    carrier = lambda name: ["f", "ɪ", "n", "d", "ə", "θ"] + name + ["æ", "n", "d"]
    good = distance(carrier(["n", "ə", "θ", "æ", "n", "j", "ə", "l"]), target)
    near = distance(carrier(["n", "ɐ", "θ", "æ", "n", "j", "ə", "l"]), target)      # similar vowel: small cost
    far = distance(carrier(["n", "uː", "θ", "æ", "n", "j", "oʊ"]), target)          # long u + wrong ending
    assert good == 0 < near < far


def test_parallel_synthesis_caches_and_keeps_every_clip(tmp_path, monkeypatch):
    import json, threading
    import numpy as np
    from audiobook_gen import synth

    seen, lock = set(), threading.Lock()

    class Fake:
        sample_rate = 24000
        def synth(self, text, voice):
            with lock:
                seen.add(threading.current_thread().name)
            return np.full(2400, 0.1, np.float32)

    monkeypatch.setattr(synth, "get_engine", lambda *a, **k: Fake())
    segs = [{"id": f"001-{i:05d}", "chapter": 1, "speaker": "Narrator", "kind": "narration",
             "text": f"Sentence number {i}.", "para_start": True} for i in range(12)]
    (tmp_path / "segments.json").write_text(json.dumps(segs))
    cfg = {"voices": {}, "default_voice": {"engine": "kokoro", "voice": "bm_george"}, "workers": {"kokoro": 3}}
    msgs = list(synth.synthesize_iter(tmp_path, cfg))
    assert msgs[-1][0] == msgs[-1][1] == 12
    assert len(json.loads((tmp_path / "clips.json").read_text())) == 12 and len(list((tmp_path / "clips").glob("*.wav"))) == 12
    assert len(seen) > 1                                   # more than one worker thread did the work
    seen.clear()
    list(synth.synthesize_iter(tmp_path, cfg))
    assert not seen                                        # second run: everything cached, no synthesis


def test_gender_aware_voices():
    from audiobook_gen import casting
    from audiobook_gen.synth import resolve_voice
    genders = {"Mercédès": "female", "Danglars": "male", "Edmond": "male", "Martha": "female", "Mary": "female"}
    roles = ["Narrator", "Edmond", "Danglars", "Mercédès", "Martha", "Mary", "Crowd"]
    out = casting.assign_by_gender(roles, genders, keep={"Edmond": 1})
    assert "Narrator" not in out and "Edmond" not in out                      # narrator and kept (clone) roles untouched
    assert out["Mercédès"]["voice"] in casting.FEMALE and out["Danglars"]["voice"] in casting.MALE
    assert len({out[r]["voice"] for r in ("Mercédès", "Martha", "Mary")}) == 3  # distinct while the pool lasts
    cfg = {"voices": {}, "default_voice": {"engine": "kokoro", "voice": "af_heart"}, "genders": genders}
    assert resolve_voice("Mary", cfg)["voice"] in casting.FEMALE               # unconfigured roles follow their gender
    assert resolve_voice("Danglars", cfg)["voice"] in casting.MALE


def test_auto_spell_from_guess_when_no_ipa():
    from audiobook_gen.lexicon import auto_respell, fill_respell
    e = {"term": "Graymalkin", "ipa": "", "respell": "", "guess": "ɡɹˈAmˌælkɪn", "known": False, "source": "auto"}
    assert auto_respell(e) == ""                              # default: F5 keeps the plain spelling
    assert auto_respell(e, use_guess=True) == "graymalkihn"   # the button's behaviour: spelled from the guess
    withipa = {**e, "ipa": "ɡreɪmˈælkɪn"}
    assert auto_respell(withipa, use_guess=True) == auto_respell(withipa)   # IPA wins over the guess
    rows = [dict(e)]
    assert fill_respell(rows, use_guess=True) == 1 and rows[0]["respell_src"] == "auto-guess"


def test_f5_style_writes_schwa_as_a():
    from audiobook_gen.lexicon import auto_respell, ipa_to_respell
    assert auto_respell({"ipa": "nəˈθænjəl"}) == "nathanyal"          # the spelling that sounded right in F5
    assert ipa_to_respell("nəˈθænjəl") == "nuh-THAN-yuhl"             # the human-readable form keeps "uh"
    assert auto_respell({"ipa": "dˈʌŋkən"}) == "dungkan"


def test_voice_packs_appear_in_voice_menus(tmp_path, monkeypatch):
    from audiobook_gen import casting, voicepack
    monkeypatch.setattr(voicepack, "PACKS", tmp_path)
    (tmp_path / "wakers_narrator.pt").write_bytes(b"x")
    labels = dict((k, l) for l, k in casting.voice_choices())
    assert "kokoro:pack:wakers_narrator" in labels and "recording" in labels["kokoro:pack:wakers_narrator"]
    v = casting.voice_of("kokoro:pack:wakers_narrator", 1.0)
    assert v == {"engine": "kokoro", "voice": "pack:wakers_narrator", "speed": 1.0}   # what KokoroEngine turns into the .pt
    assert casting.key_of(v) == "kokoro:pack:wakers_narrator"


def test_tone_eq_and_pace_measures():
    import numpy as np
    from audiobook_gen import voicepack as vp
    sr = 24000
    t = np.arange(sr * 2) / sr
    voice = (0.3 * np.sin(2 * np.pi * 150 * t) + 0.3 * np.sin(2 * np.pi * 5000 * t)).astype("float32")
    flat = {"freqs": vp.THIRDS, "gains_db": [0.0] * len(vp.THIRDS)}
    assert np.allclose(vp.apply_eq(voice, sr, flat), voice, atol=1e-3)
    cut = {"freqs": vp.THIRDS, "gains_db": [0.0] * 16 + [-6.0] * 4}          # 4 kHz and up down 6 dB
    out = vp.apply_eq(voice, sr, cut)
    hi = lambda a: float(np.abs(np.fft.rfft(a))[int(4.9 * 2 * 1000):int(5.1 * 2 * 1000)].max())
    assert 0.4 < hi(out) / hi(voice) < 0.6                                     # about -6 dB at 5 kHz
    # match_eq turns a dark voice brighter, within the +3 dB cap above 4 kHz
    dark = vp.apply_eq(voice, sr, {"freqs": vp.THIRDS, "gains_db": [0.0] * 16 + [-6.0] * 4})
    eq = vp.match_eq([dark], [voice], sr)
    assert eq["gains_db"][17] > 0 and max(eq["gains_db"]) <= 6.0
    # pace: 10 words in 2 s of speech inside 4 s of audio
    clip = np.concatenate([voice, np.zeros(sr * 2, "float32")])
    rate, pause = vp.pace_of([clip], sr, 10)
    assert abs(rate - 5.0) < 0.3 and abs(pause - 0.5) < 0.05


def test_tagged_voice_shows_name_and_description(tmp_path, monkeypatch):
    import json
    from audiobook_gen import casting, voicepack
    monkeypatch.setattr(voicepack, "PACKS", tmp_path)
    (tmp_path / "wakers_narrator_v4.pt").write_bytes(b"x")
    (tmp_path / "wakers_narrator_v4.json").write_text(json.dumps({"label": "John", "tags": "US male"}))
    (tmp_path / "plain.pt").write_bytes(b"x")
    labels = dict((k, l) for l, k in casting.voice_choices())
    assert labels["kokoro:pack:wakers_narrator_v4"].startswith("John — US male")
    assert labels["kokoro:pack:plain"].startswith("plain —")                  # untagged voices keep their file name


def test_text_engine_lexicon_modes():
    from audiobook_gen.lexicon import Preprocessor
    lex = [{"term": "Pharaon", "ipa": "faʁaˈɔ̃", "respell": "fahrahawn"}]
    assert Preprocessor(lex, "plain").substitute("the Pharaon sailed") == "the Pharaon sailed"
    assert Preprocessor(lex, "rawipa").substitute("the Pharaon sailed") == "the faʁaˈɔ̃ sailed"
    assert "fahrahawn" in Preprocessor(lex, "respell").substitute("the Pharaon sailed").lower()


def test_wiktionary_ipa_parsing():
    from audiobook_gen.pronounce import clean_ipa, parse_wiktionary
    wt = ("==English==\n===Pronunciation===\n* {{IPA|en|/ˈæntɪˌɒk/<a:UK><ref:{{R:Collins}}>|a=UK}}\n"
          "* {{IPA|en|/ˈæn.ti.ˌɑk/|a=US}}\n==French==\n* {{IPA|fr|/ɑ̃.tjɔk/}}\n")
    assert parse_wiktionary(wt, ["en"]) == ("ˈæntiˌɑk", "en", "US")   # General American wins, dots removed
    assert parse_wiktionary(wt, ["fr", "en"])[:2] == ("ɑ̃tjɔk", "fr")                  # the book's language comes first
    assert clean_ipa("/mɛʁ.se.dɛs/") == "mɛʁsedɛs"


def test_ford_hint_is_only_syllables_and_stress():
    from audiobook_gen import pronounce as pr
    pr._FORD = {"nathanael": ["Na", "than'", "a", "el"]}
    assert pr.ford_hint("Nathanael") == "Na-than'-a-el" and pr.ford_hint("Zzyzx") == ""
    pr._FORD = None


def test_library_voice_keeps_chatterbox_settings(tmp_path, monkeypatch):
    import numpy as np
    import soundfile as sf
    from audiobook_gen import voices as vlib
    monkeypatch.setattr(vlib, "LIB", tmp_path)
    sf.write(tmp_path / "ref.wav", np.zeros(24000, "float32"), 24000)
    vlib.save_voice("T", str(tmp_path / "ref.wav"), "hello there", {"speed": 1.0, "exaggeration": 0.7, "cfg_weight": 0.3})
    v = vlib.load_voice("T")
    assert v["exaggeration"] == 0.7 and v["cfg_weight"] == 0.3
    vlib.save_voice("F", str(tmp_path / "ref.wav"), "hello there", {"speed": 1.0})
    assert "exaggeration" not in vlib.load_voice("F")      # F5-only voices keep their cache keys


def test_interface_builds_with_every_engine_option():
    from audiobook_gen import gui
    assert gui.build_ui() is not None
    assert set(gui.ENGINE_LABELS.values()) == {"f5", "chatterbox", "qwen3"}
    assert set(gui.LEX_MODES.values()) == {"respell", "plain"}


def test_placeholder_cover_has_large_readable_title(tmp_path):
    from PIL import Image
    from audiobook_gen.assemble import make_cover
    p = make_cover(tmp_path / "c.jpg", "Genesis", "Test author")
    im = Image.open(p).convert("L")
    assert im.size == (1400, 1400)
    mid = im.crop((200, 480, 1200, 720))                 # where a one-word title is set
    bright = sum(1 for v in mid.getdata() if v > 200)
    assert bright > 20000                                # big cream letters, not the tiny default font


def test_voice_menu_keys_round_trip_through_the_config():
    from audiobook_gen import casting
    for key in ("kokoro:am_onyx", "kokoro:pack:wakers_ch16", "clone:Walter", "clone:Walter@chatterbox", "clone:Walter@qwen3"):
        assert casting.key_of(casting.voice_of(key, 1.0)) == key


def test_verified_lexicon_mode_only_respells_tested_names():
    from audiobook_gen.lexicon import Preprocessor, verified_respellings
    v = verified_respellings()
    assert v, "data/bible_respell.json should load"
    name = next(iter(v))
    pre = Preprocessor([], "verified")
    assert pre(f"Then {name} spoke to Moses.").startswith("Then " + v[name]["respell"])
    assert "Moses" in pre("Moses spoke.")                       # untested names stay as written
    user = Preprocessor([{"term": name, "respell": "Custom", "source": "user"}], "verified")
    assert "Custom" in user(f"{name} spoke.")                   # your own respelling wins


def test_respell_rules_make_readable_spellings():
    from audiobook_gen.respell_rules import respell
    assert respell("ˈheɪɡɑɹ").lower().startswith("haygar")


def test_run_stats_average_and_estimate(tmp_path, monkeypatch):
    from audiobook_gen import runstats
    monkeypatch.setattr(runstats, "PATH", tmp_path / "run_stats.json")
    runstats.record("a", "chatterbox", 1, 3000, 300)      # 10 chars/s
    runstats.record("b", "chatterbox", 1, 1000, 50)       # 20 chars/s
    runstats.record("b", "chatterbox", 1, 2000, 100)      # same run again: updated, not added
    runstats.record("c", "kokoro", 4, 100, 5)             # too short to count
    avg = runstats.averages()
    assert avg["chatterbox"]["runs"] == 2 and abs(avg["chatterbox"]["cps"] - 5000 / 400) < 1e-6
    assert "kokoro" not in avg
    assert runstats.clock(90 * 60) == "1 h 30 min"


def test_cover_with_a_picture_background(tmp_path):
    from PIL import Image
    from audiobook_gen.assemble import make_cover
    Image.new("RGB", (900, 600), (230, 230, 230)).save(tmp_path / "bg.jpg")      # a bright, wide picture
    out = make_cover(tmp_path / "c.jpg", "A Test Title", "An Author", str(tmp_path / "bg.jpg"), (0.4, 0.5), 1.2)
    im = Image.open(out)
    assert im.size == (1400, 1400)
    assert im.convert("L").resize((1, 1)).getpixel((0, 0)) < 110                  # toned down so the title can be read
    far = make_cover(tmp_path / "d.jpg", "Zoomed Out", "", str(tmp_path / "bg.jpg"), (0.5, 0.5), 0.8)
    assert Image.open(far).size == (1400, 1400)


def test_portrait_cover_layout(tmp_path):
    from PIL import Image
    from audiobook_gen.assemble import make_cover_portrait
    Image.new("RGB", (600, 800), (90, 90, 90)).save(tmp_path / "p.png")
    out = make_cover_portrait(tmp_path / "c.jpg", "A Long Narrative of a Person's Life", "Some Author", str(tmp_path / "p.png"))
    im = Image.open(out)
    assert im.size == (1400, 1400)
    assert im.getpixel((700, 700))[0] > 60            # the picture is in the middle...
    assert im.getpixel((200, 700)) == (0, 0, 0)       # ...on a black page


# ---------- job queue ----------
def _queue_env(tmp_path, monkeypatch):
    from audiobook_gen import jobqueue as jq
    monkeypatch.setattr(jq, "QUEUE", tmp_path / "queue")
    monkeypatch.setattr(jq, "JOBS", tmp_path / "queue" / "jobs.json")
    monkeypatch.setattr(jq, "HEARTBEAT", tmp_path / "queue" / "heartbeat")
    for name in ("RUNNER_PID", "JOB_PID", "SUPERVISOR_PID", "STOPPED", "SPEECH_LIMIT", "SETTINGS", "HALTED", "ACTIVITY", "GUI_LOCK"):   # never touch the real queue's files
        monkeypatch.setattr(jq, name, tmp_path / "queue" / name.lower())
    sent = []                                                                   # alerts are recorded, never sent
    monkeypatch.setattr(jq, "notify", lambda text, title="", **k: sent.append((title, text)) or True)
    jq.sent_alerts = sent
    work = tmp_path / "book"; work.mkdir()
    (work / "segments.json").write_text('[{"id": 1, "speaker": "Narrator", "kind": "narration", "text": "Hello.", "chapter": 1}]')
    from audiobook_gen import prep
    prep.mark_done(work)                                                        # a prepared book: the queue tests are about the queue
    return jq, work


def test_queue_windows_and_start_times():
    from datetime import datetime
    from audiobook_gen import jobqueue as jq
    t = lambda s: datetime.strptime(s, jq.FMT)
    assert jq.in_window("23:00-06:30", t("2026-10-09 23:30")) and jq.in_window("23:00-06:30", t("2026-10-10 06:00"))
    assert not jq.in_window("23:00-06:30", t("2026-10-09 12:00")) and not jq.in_window("23:00-06:30", t("2026-10-10 06:30"))
    job = {"status": "queued", "not_before": "2026-10-09 23:00", "window": ""}
    assert not jq.due(job, t("2026-10-09 22:59")) and jq.due(job, t("2026-10-09 23:00"))
    assert not jq.due({**job, "status": "done"}, t("2026-10-10 01:00"))
    assert jq.next_start({"not_before": "", "window": "23:00-06:30"}, t("2026-10-09 12:00")) == "2026-10-09 23:00"


def test_queue_runs_a_job_to_completion(tmp_path, monkeypatch):
    jq, work = _queue_env(tmp_path, monkeypatch)
    out = tmp_path / "book.m4b"
    job = jq.add(str(work), "Book", out=str(out))
    class R(jq.Runner):
        synth_cmd = lambda self, j: [sys.executable, "-c", "print('[progress] 5/10')"]
        assemble_cmd = lambda self, j: [sys.executable, "-c", f"open({str(out)!r}, 'w').write('x')"]
    import sys
    assert R(poll=0.1, foreign=lambda: "").step() is True
    done = [j for j in jq.load() if j["id"] == job["id"]][0]
    assert done["status"] == "done" and out.exists()


def test_queue_watchdog_restarts_a_stalled_job_then_gives_up(tmp_path, monkeypatch):
    import sys
    jq, work = _queue_env(tmp_path, monkeypatch)
    job = jq.add(str(work), "Hung book", out=str(tmp_path / "x.m4b"))
    class R(jq.Runner):
        synth_cmd = lambda self, j: [sys.executable, "-c", "import time; time.sleep(60)"]      # never writes a clip
    r = R(poll=0.1, stall=0.3, grace=0.3, max_restarts=2, retry_wait=0, foreign=lambda: "")
    r.step()
    j = [x for x in jq.load() if x["id"] == job["id"]][0]
    assert j["status"] == "failed" and j["restarts"] == 3 and "gave up" in j["note"]


def test_queue_waits_while_another_synthesis_holds_the_gpu(tmp_path, monkeypatch):
    jq, work = _queue_env(tmp_path, monkeypatch)
    job = jq.add(str(work), "Waiting book")
    assert jq.Runner(poll=0.1, foreign=lambda: "another synthesis").step() is False
    assert "holds the GPU" in [x for x in jq.load() if x["id"] == job["id"]][0]["note"]


# ---------- assistant ----------
def _assistant_project(tmp_path):
    import json, yaml
    p = tmp_path / "proj"; p.mkdir()
    (p / "config.yaml").write_text(yaml.safe_dump({"voices": {"Narrator": {"engine": "kokoro", "voice": "bm_george"}}, "genders": {}}))
    (p / "lexicon.json").write_text(json.dumps([{"term": "Weena", "ipa": "", "source": "auto"}]))
    (p / "segments.json").write_text(json.dumps([{"speaker": "Narrator", "kind": "narration", "text": "Hello.", "id": 1, "chapter": 1}]))
    (p / "chapters.json").write_text(json.dumps({"title": "T", "author": "A", "chapters": [{"index": 1, "title": "One", "text": "Hello."}]}))
    from audiobook_gen import prep
    prep.mark_done(p)
    return p


def test_assistant_proposes_validates_and_applies_changes(tmp_path):
    import json, yaml
    from audiobook_gen import assistant
    p = _assistant_project(tmp_path)
    scratch = assistant.prepare(p)
    assert (scratch / "CLAUDE.md").exists() and (scratch / "reference" / "voices.md").exists()
    assert assistant.diff(p) == {}
    cfg = yaml.safe_load((scratch / "config.yaml").read_text()); cfg["voices"]["Narrator"]["voice"] = "am_adam"
    (scratch / "config.yaml").write_text(yaml.safe_dump(cfg))
    assert "am_adam" in assistant.diff(p)["config.yaml"] and assistant.validate(p) == []
    assert "Applied" in assistant.apply(p)
    assert yaml.safe_load((p / "config.yaml").read_text())["voices"]["Narrator"]["voice"] == "am_adam"
    assert list((scratch).glob("backup-*"))                                  # the old file is kept


def test_assistant_refuses_bad_proposals(tmp_path):
    import yaml
    from audiobook_gen import assistant
    p = _assistant_project(tmp_path)
    scratch = assistant.prepare(p)
    cfg = yaml.safe_load((scratch / "config.yaml").read_text())
    cfg["voices"]["Narrator"] = {"engine": "chatterbox", "library": "NoSuchVoice"}; cfg["workers"] = {"chatterbox": 99}
    (scratch / "config.yaml").write_text(yaml.safe_dump(cfg))
    problems = assistant.validate(p)
    assert any("NoSuchVoice" in x for x in problems) and any("workers" in x for x in problems)
    assert assistant.apply(p).startswith("Not applied")
    assert "bm_george" in (p / "config.yaml").read_text()                    # untouched
    assert "discarded" in assistant.discard(p) and assistant.diff(p) == {}


def test_assistant_runs_claude_in_the_scratch_folder_only(tmp_path, monkeypatch):
    import stat
    from audiobook_gen import assistant
    p = _assistant_project(tmp_path)
    fake = tmp_path / "fakeclaude"
    fake.write_text('#!/bin/bash\necho "$@" > args.txt\npwd > where.txt\necho \'{"result":"done","session_id":"s1","total_cost_usd":0.05,"is_error":false}\'\n')
    fake.chmod(fake.stat().st_mode | stat.S_IEXEC)
    monkeypatch.setenv("AUDIOBOOK_CLAUDE", str(fake))
    r = assistant.ask(p, "make it spooky")
    assert r["reply"] == "done" and r["session"] == "s1" and r["cost"] == 0.05
    args = (assistant.scratch_dir(p) / "args.txt").read_text()
    assert "make it spooky" in args
    allowed, _, blocked = args.partition("--disallowedTools")
    assert "Bash(./cover:*)" in allowed and allowed.count("Bash") == 1         # the cover tool is the only command it may run
    assert "WebFetch" in blocked and "WebSearch" in blocked and "Bash" not in blocked
    assert (assistant.scratch_dir(p) / "where.txt").read_text().strip().endswith("proj/assistant")


def test_gui_reconnects_to_a_running_job(tmp_path, monkeypatch):
    import json, time
    from audiobook_gen import gui, jobqueue as jq
    jq_dir = tmp_path / "queue"
    monkeypatch.setattr(jq, "QUEUE", jq_dir); monkeypatch.setattr(jq, "JOBS", jq_dir / "jobs.json")
    p = _assistant_project(tmp_path)
    (p / "clips").mkdir(); (p / "clips" / "a.wav").write_bytes(b"x")
    job = jq.add(str(p), "Running book", "An Author")
    jq.update(job["id"], status="running", progress="30/120", started="2026-10-09 23:00")
    r = jq.now_running()
    assert r["title"] == "Running book" and r["pct"] == 25 and r["since_clip"] < 60
    assert "Making “Running book”" in gui.queue_now() and "25%" in gui.queue_now()
    out = gui.reconnect_on_load("")                          # nothing open: it opens the running book
    assert len(out) == 29 and out[0] == str(p) and out[2] == "Running book" and out[3] == "An Author"
    assert out[1]["Include"].all()                           # every chapter ticked, as it was queued
    assert all(x == gui.gr.skip() for x in gui.reconnect_on_load(str(p)))     # a book is already open: leave it alone
    jq.update(job["id"], status="done")
    assert all(x == gui.gr.skip() for x in gui.reconnect_on_load(""))         # nothing running: leave the tabs alone
    assert "Nothing is queued" in gui.queue_now()


def test_gui_opens_a_saved_project_without_reading_the_book_again(tmp_path):
    from audiobook_gen import gui
    p = _assistant_project(tmp_path)
    proj, chap, title, author, cover, msg = gui.open_project(str(p))
    assert proj == str(p) and title == "T" and list(chap["#"]) == [1] and "Opened" in msg
    roles, seg_df, status, *_ = gui.load_parsed(proj)
    assert roles[0][0] == "Narrator" and len(seg_df) == 1


def test_generate_button_queues_the_book_and_cancel_stops_it(tmp_path, monkeypatch):
    import pandas as pd
    from audiobook_gen import gui, jobqueue as jq
    monkeypatch.setattr(jq, "QUEUE", tmp_path / "queue"); monkeypatch.setattr(jq, "JOBS", tmp_path / "queue" / "jobs.json")
    monkeypatch.setattr(jq, "ensure_supervisor", lambda: False)
    p = _assistant_project(tmp_path)
    chap = pd.DataFrame([[True, 1, "One", 6]], columns=gui.CHAP_COLS)
    msg = gui.generate_queued(str(p), chap, "My Book", "Me", "", 60, 350, 700, 250, 4, True, True, 0.0,
                              list(gui.LEX_MODES)[0], 140, 120, 1)
    jobs = jq.load()
    assert len(jobs) == 1 and jobs[0]["status"] == "queued" and jobs[0]["not_before"] == "" and jobs[0]["window"] == ""
    assert "Queue" in msg
    status, *_ = gui.gen_panel(str(p), "")
    assert status.startswith("Queued")
    assert "Cancelled 1" in gui.gen_stop(str(p)) and jq.load()[0]["status"] == "cancelled"
    jq.update(jobs[0]["id"], status="done", out=str(tmp_path / "x.m4b"))
    (tmp_path / "x.m4b").write_bytes(b"x")
    assert gui.gen_panel(str(p), "")[0].startswith("Done")


def test_chatterbox_lexicon_uses_only_respellings_you_typed():
    from audiobook_gen.lexicon import Preprocessor
    lex = [{"term": "Weena", "respell": "Weenuh", "respell_src": "user", "source": "user"},
           {"term": "Filby", "ipa": "ˈfɪlbi", "respell": "filbee", "respell_src": "auto", "source": "user"},   # made from IPA you typed
           {"term": "Eloi", "respell": "eloy", "source": "auto"}]
    out = Preprocessor(lex, "verified")("Weena met Filby and the Eloi.")
    assert "Weenuh" in out and "Filby" in out and "filbee" not in out and "Eloi" in out


def test_chatterbox_and_qwen3_get_only_typed_respellings():
    from audiobook_gen.lexicon import Preprocessor
    from audiobook_gen.synth import lexicon_mode
    cfg = {"text_lexicon": "rawipa"}                                   # even a project set to raw IPA
    assert lexicon_mode("kokoro", cfg) == "ipa"
    assert lexicon_mode("chatterbox", cfg) == "verified" and lexicon_mode("qwen3", cfg) == "typed"
    assert lexicon_mode("f5", cfg) == "rawipa" and lexicon_mode("f5", {}) == "respell"
    lex = [{"term": "Weena", "ipa": "ˈwiːnə", "respell": "Weenuh", "respell_src": "user", "source": "user"},
           {"term": "Eloi", "ipa": "ˈiːlɔɪ", "respell": "eeloy", "respell_src": "auto", "source": "auto"},
           {"term": "Morlock", "ipa": "ˈmɔɹlɑk", "respell": "", "source": "user"}]                # IPA only
    for mode in ("verified", "typed"):
        out = Preprocessor(lex, mode)("Weena, Eloi and Morlock.")
        assert out == "Weenuh, Eloi and Morlock."                      # no IPA, no auto-made spellings
    assert "Shimmee-eye" in Preprocessor([], "verified")("Shimei came.") and "Shimei" in Preprocessor([], "typed")("Shimei came.")


def test_generate_button_uses_the_schedule_on_the_generate_tab(tmp_path, monkeypatch):
    import pandas as pd
    from audiobook_gen import gui, jobqueue as jq
    monkeypatch.setattr(jq, "QUEUE", tmp_path / "queue"); monkeypatch.setattr(jq, "JOBS", tmp_path / "queue" / "jobs.json")
    monkeypatch.setattr(jq, "ensure_supervisor", lambda: False)
    p = _assistant_project(tmp_path)
    chap = pd.DataFrame([[True, 1, "One", 6]], columns=gui.CHAP_COLS)
    msg = gui.generate_queued(str(p), chap, "Late book", "Me", "", 60, 350, 700, 250, 4, True, True, 0.0,
                              list(gui.LEX_MODES)[0], 140, 120, 1, "2026-10-09 23:00:00", True, "23:00", "06:30")
    job = jq.load()[0]
    assert job["not_before"] == "2026-10-09 23:00" and job["window"] == "23:00-06:30" and "2026-10-09 23:00" in msg


def test_chatterbox_workers_toggle(tmp_path, monkeypatch):
    import pandas as pd, yaml
    from audiobook_gen import gui, jobqueue as jq
    monkeypatch.setattr(jq, "QUEUE", tmp_path / "queue"); monkeypatch.setattr(jq, "JOBS", tmp_path / "queue" / "jobs.json")
    monkeypatch.setattr(jq, "ensure_supervisor", lambda: False)
    p = _assistant_project(tmp_path)
    chap = pd.DataFrame([[True, 1, "One", 6]], columns=gui.CHAP_COLS)
    for toggle, want in ((True, 2), (False, 1)):
        gui.generate_queued(str(p), chap, "B", "A", "", 60, 350, 700, 250, 4, True, True, 0.0, list(gui.LEX_MODES)[0], 140, 120, toggle)
        assert yaml.safe_load((p / "config.yaml").read_text())["workers"]["chatterbox"] == want
    assert gui.gen_settings(str(p))[5] is False                       # the toggle shows what is saved


def test_stop_and_start_the_queue_runner(tmp_path, monkeypatch):
    import subprocess, sys, time
    from audiobook_gen import jobqueue as jq, gui
    q = tmp_path / "queue"; q.mkdir()
    for name, val in (("QUEUE", q), ("JOBS", q / "jobs.json"), ("HEARTBEAT", q / "heartbeat"), ("RUNNER_PID", q / "runner.pid"),
                      ("JOB_PID", q / "job.pid"), ("SUPERVISOR_PID", q / "supervisor.pid"), ("STOPPED", q / "stopped")):
        monkeypatch.setattr(jq, name, val)
    monkeypatch.setattr(jq, "service_active", lambda: False)
    monkeypatch.setattr(jq, "service_installed", lambda: False)
    monkeypatch.setattr(jq, "supervisor_running", lambda: False)
    # three stand-ins for the supervisor, the runner and a job in progress, each in its own process group
    procs = {}
    for name, tag, path in (("sup", "jobqueue", jq.SUPERVISOR_PID), ("run", "jobqueue", jq.RUNNER_PID), ("job", "audiobook_gen", jq.JOB_PID)):
        pr = subprocess.Popen([sys.executable, "-c", f"import time  # {tag}\ntime.sleep(60)"], start_new_session=True)
        path.write_text(str(pr.pid)); procs[name] = pr
    work = tmp_path / "book"; work.mkdir(); (work / "segments.json").write_text("[]")
    job = jq.add(str(work), "Book"); jq.update(job["id"], status="running")
    jq.HEARTBEAT.write_text(str(time.time()))
    started = []
    real_popen = jq.subprocess.Popen
    assert jq.runner_alive() and gui.runner_button()["value"] == "Stop the queue runner"
    assert "Stopped" in jq.stop_runner()
    monkeypatch.setattr(jq.subprocess, "Popen", lambda *a, **k: started.append(a) or None)     # from here on, record launches only
    for pr in procs.values():
        assert pr.wait(timeout=20) is not None                         # all three ended, including the job
    assert not jq.runner_alive() and gui.runner_button()["value"] == "Start the queue runner"
    assert [x for x in jq.load() if x["id"] == job["id"]][0]["status"] == "queued"      # the job resumes later
    assert jq.STOPPED.exists() and jq.ensure_supervisor() is False and not started          # nothing restarts it behind your back
    assert "stopped" in gui.queue_status()
    jq.start_runner()
    assert not jq.STOPPED.exists() and started                              # now it may start again


def test_queue_reorder_hold_resume_remove_and_reschedule(tmp_path, monkeypatch):
    jq, work = _queue_env(tmp_path, monkeypatch)
    ids = [jq.add(str(work), t)["id"] for t in "ABCDE"]
    names = lambda: "".join(j["title"] for j in jq.load())
    jq.move([ids[3]], "top");               assert names() == "DABCE"
    jq.move([ids[0], ids[2]], "down");      assert names() == "DBAEC"
    jq.move([ids[4]], "up");                assert names() == "DBEAC"
    jq.move([ids[3]], "bottom");            assert names() == "BEACD"
    jq.move([ids[1]], "up");                assert names() == "BEACD"            # already first: stays put
    assert jq.hold([ids[1], ids[2]]) == 2
    st = {j["title"]: j["status"] for j in jq.load()}
    assert st["B"] == st["C"] == "held" and st["A"] == "queued"
    from datetime import datetime
    assert not jq.due([j for j in jq.load() if j["title"] == "B"][0], datetime.now())     # a held job never starts
    assert jq.resume([ids[1]]) == 1 and [j for j in jq.load() if j["title"] == "B"][0]["status"] == "queued"
    assert jq.set_schedule([ids[0], ids[4]], "2026-10-12 23:00", "22:00-05:00") == 2
    a = [j for j in jq.load() if j["title"] == "A"][0]
    assert a["not_before"] == "2026-10-12 23:00" and a["window"] == "22:00-05:00"
    jq.set_schedule([ids[0]], "", "");  assert [j for j in jq.load() if j["title"] == "A"][0]["window"] == ""
    import pytest
    with pytest.raises(ValueError):
        jq.set_schedule([ids[0]], "tomorrow-ish", "")
    jq.update(ids[2], status="running")
    assert jq.remove_many([ids[2], ids[3]]) == 1 and "C" in names() and "D" not in names()      # a running book is never removed


def test_queue_pausing_a_running_book_stops_it_and_keeps_it_for_later(tmp_path, monkeypatch):
    import sys, threading, time
    jq, work = _queue_env(tmp_path, monkeypatch)
    job = jq.add(str(work), "Long book", out=str(tmp_path / "x.m4b"))
    class R(jq.Runner):
        synth_cmd = lambda self, j: [sys.executable, "-c", "import time; (open(%r,'w')).write('x'); time.sleep(60)" % str(work / "clips" / "a.wav")]
    (work / "clips").mkdir()
    threading.Timer(1.0, lambda: jq.hold([job["id"]])).start()
    t0 = time.time()
    R(poll=0.2, stall=30, grace=30, foreign=lambda: "").step()
    j = [x for x in jq.load() if x["id"] == job["id"]][0]
    assert time.time() - t0 < 15 and j["status"] == "held" and j["note"] == "paused by you"      # stopped promptly, not failed
    assert R(foreign=lambda: "").step() is False                                              # and it does not start again by itself
    jq.resume([job["id"]])
    assert [x for x in jq.load() if x["id"] == job["id"]][0]["status"] == "queued"


def test_queue_tab_actions(tmp_path, monkeypatch):
    from audiobook_gen import gui
    jq, work = _queue_env(tmp_path, monkeypatch)
    monkeypatch.setattr(jq, "runner_alive", lambda: True)
    ids = [jq.add(str(work), t)["id"] for t in ("One", "Two", "Three")]
    titles = lambda: [j["title"] for j in jq.load()]
    import pytest
    with pytest.raises(Exception):
        gui.queue_move([], "top")                                                   # nothing ticked: a clear message
    msg, state, table, now, btn, pick = gui.queue_move([ids[2]], "top")
    assert titles() == ["Three", "One", "Two"] and list(table["#"]) == [1, 2, 3] and list(table["Book"]) == titles()
    assert pick == [ids[2]]                                                           # the ticks survive a refresh
    assert [bool(x) for x in table["✔"]] == [True, False, False] and gui.ticked_ids(table) == [ids[2]]   # and show in the table's first column
    assert "Paused" in gui.queue_hold([ids[0], ids[1]])[0] and "2 saved for later" in gui.queue_status()
    assert gui.queue_table().set_index("id").loc[ids[0], "Starts"] == "when you resume it"
    assert "Resumed 1" in gui.queue_resume([ids[0]])[0]
    r = gui.queue_reschedule([ids[0], ids[1]], "2026-10-12 23:00:00", True, "22:00", "05:00")
    assert "2 book(s)" in r[0] and all(j["window"] == "22:00-05:00" for j in jq.load() if j["id"] in ids[:2])
    assert "as soon as the GPU is free" in gui.queue_asap([ids[0]])[0] and [j for j in jq.load() if j["id"] == ids[0]][0]["window"] == ""
    assert "Cancelled 1" in gui.queue_cancel([ids[1]])[0]
    assert "Removed 2" in gui.queue_remove([ids[1], ids[2]])[0] and titles() == ["One"]


# ---------- sending to the phone ----------
def test_kde_devices_are_read_from_kdeconnect(monkeypatch):
    from audiobook_gen import jobqueue as jq
    out = ("- Pixel 6a: phone-aaaa1111 on 10.0.0.21 via LAN (paired and reachable)\\n"
           "- Pixel 8: phone-bbbb2222 on 10.0.0.22 via LAN (reachable)\\n"
           "- ThinkPad: _cccc3333_ (paired)\\n4 devices found\\n").replace("\\n", "\n")
    monkeypatch.setattr(jq.subprocess, "run", lambda *a, **k: type("R", (), {"stdout": out, "returncode": 0})())
    devs = jq.kde_devices()
    assert [(d["name"], d["reachable"]) for d in devs] == [("Pixel 6a", True), ("ThinkPad", False)]       # unpaired Pixel 8 is left out


def test_finished_book_is_sent_to_the_phone_and_retried_while_it_is_away(tmp_path, monkeypatch):
    jq, work = _queue_env(tmp_path, monkeypatch)
    out = tmp_path / "book.m4b"; out.write_bytes(b"audio")
    job = jq.add(str(work), "Sent book", out=str(out), send_to="phone1")
    jq.update(job["id"], status="done")
    reachable, shared = [False], []
    def sender(device, path):
        if not reachable[0]:
            return False
        shared.append((device, path)); return True
    r = jq.Runner(poll=0.1, foreign=lambda: "", sender=sender, send_every=0)
    r._deliver()
    j = [x for x in jq.load() if x["id"] == job["id"]][0]
    assert j["sent"] == "" and "waiting for your phone" in j["note"] and not shared
    assert jq.table()[0][5] == "waiting for the phone"
    reachable[0] = True
    r._deliver()
    j = [x for x in jq.load() if x["id"] == job["id"]][0]
    assert j["sent"] and shared == [("phone1", str(out))] and jq.table()[0][5].startswith("sent ")
    r._deliver()
    assert len(shared) == 1                                                      # once is enough
    assert jq.set_send([job["id"]], "phone2") == 1 and [x for x in jq.load() if x["id"] == job["id"]][0]["sent"] == ""   # "send again" re-arms it


def test_a_slow_send_never_blocks_the_queue(tmp_path, monkeypatch):
    import time
    jq, work = _queue_env(tmp_path, monkeypatch)
    out = tmp_path / "b.m4b"; out.write_bytes(b"x")
    job = jq.add(str(work), "Big book", out=str(out), send_to="p"); jq.update(job["id"], status="done")
    r = jq.Runner(poll=0.1, foreign=lambda: "", sender=lambda d, p: time.sleep(2) or True, send_every=0)
    t0 = time.time()
    r.step()                                                                      # starts the send on its own thread...
    assert time.time() - t0 < 1.0 and jq.HEARTBEAT.exists()                       # ...and returns at once, heartbeat written
    r._sender_thread.join(timeout=10)
    assert [x for x in jq.load() if x["id"] == job["id"]][0]["sent"]


def test_generate_tab_can_send_to_a_phone(tmp_path, monkeypatch):
    import pandas as pd
    from audiobook_gen import gui
    jq, work = _queue_env(tmp_path, monkeypatch)
    monkeypatch.setattr(jq, "SETTINGS", tmp_path / "queue" / "settings.json")
    monkeypatch.setattr(jq, "ensure_supervisor", lambda: False)
    chap = pd.DataFrame([[True, 1, "One", 6]], columns=gui.CHAP_COLS)
    args = (str(work), chap, "B", "A", "", 60, 350, 700, 250, 4, True, True, 0.0, list(gui.LEX_MODES)[0], 140, 120, False, "", False, "23:00", "06:30")
    import pytest
    with pytest.raises(Exception):
        gui.generate_queued(*args, True, "")                                        # ticked but no device chosen
    assert jq.load() == []
    gui.generate_queued(*args, True, "phone-aaaa1111")
    assert jq.load()[0]["send_to"] == "phone-aaaa1111" and jq.default_device() == "phone-aaaa1111"
    assert len(jq.load()) == 1                                                       # the refused click queued nothing
    assert "will not be sent" in gui.queue_nosend([jq.load()[0]["id"]])[0] and jq.load()[0]["send_to"] == ""
    assert "will be sent" in gui.queue_send([jq.load()[0]["id"]], "dev2")[0] and jq.load()[0]["send_to"] == "dev2"


def test_framed_cover_takes_a_book_cloth_colour(tmp_path):
    from PIL import Image
    from audiobook_gen.assemble import make_cover_portrait
    Image.new("RGB", (800, 500), (200, 200, 200)).save(tmp_path / "wide.png")                      # a rectangular picture
    oval = Image.new("RGBA", (500, 700), (0, 0, 0, 0))                                             # an oval portrait with see-through corners
    from PIL import ImageDraw
    ImageDraw.Draw(oval).ellipse([0, 0, 499, 699], fill=(160, 160, 160, 255)); oval.save(tmp_path / "oval.png")
    for pic in ("wide.png", "oval.png"):
        out = make_cover_portrait(tmp_path / f"{pic}.jpg", "A Title", "An Author", str(tmp_path / pic), (18, 52, 38))
        im = Image.open(out).convert("RGB")
        assert im.size == (1400, 1400) and all(abs(a - b) < 12 for a, b in zip(im.getpixel((150, 700)), (18, 52, 38)))   # the colour shows
        assert sum(im.getpixel((700, 700))) > 300                                                   # the picture is in the middle
    im = Image.open(tmp_path / "oval.png.jpg").convert("RGB")
    near = lambda c, ref=(18, 52, 38): all(abs(a - b) < 14 for a, b in zip(c, ref))
    ys = [y for y in range(300, 1250) if not near(im.getpixel((700, y)))]       # the picture's top and bottom (column through its centre)
    xs = [x for x in range(150, 1250) if not near(im.getpixel((x, (ys[0] + ys[-1]) // 2)))]                      # and its left edge
    assert im.getpixel((xs[0] + 6, ys[0] + 6)) is not None and near(im.getpixel((xs[0] + 6, ys[0] + 6)))        # the oval's corner shows the colour, not black


def test_assistant_cover_proposal_is_validated_and_applied(tmp_path):
    from PIL import Image
    from audiobook_gen import assistant
    p = _assistant_project(tmp_path)
    scratch = assistant.prepare(p)
    tool = scratch / "cover"
    assert tool.exists() and tool.stat().st_mode & 0o100 and "audiobook_gen.covers" in tool.read_text()      # the one command, runnable
    assert assistant.cover_proposal(p) is None
    Image.new("RGB", (300, 300), (9, 9, 9)).save(scratch / "cover.jpg")                                      # too small to be a cover
    assert any("too small" in x for x in assistant.validate(p)) and assistant.apply(p).startswith("Not applied")
    Image.new("RGB", (1400, 1400), (30, 60, 90)).save(scratch / "cover.jpg")
    (scratch / "pictures").mkdir(); (scratch / "pictures" / "credit.txt").write_text("A Painting, A. Painter, 1850. https://example.org (Public domain)\n")
    assert assistant.validate(p) == [] and "the cover" in assistant.apply(p)
    assert (p / "cover_custom.jpg").exists() and "A. Painter" in (p / "cover_credit.txt").read_text()
    assert assistant.cover_proposal(p) is None                                                               # applied: nothing pending
    assert "discarded" in assistant.discard(p) and not (scratch / "cover.jpg").exists()


def test_cover_tab_flow(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from PIL import Image
    from audiobook_gen import gui, covers, jobqueue as jq
    jq_, work = _queue_env(tmp_path, monkeypatch)
    pic = tmp_path / "art.jpg"; Image.new("RGB", (1600, 1000), (120, 90, 60)).save(pic)
    hit = {"file": "File:Art.jpg", "title": "Art", "width": 1600, "height": 1000, "licence": "Public domain", "artist": "A. Painter",
           "date": "1850", "thumb": "http://example/thumb.jpg", "page": "http://example/p"}
    monkeypatch.setattr(covers, "search", lambda q, n=8: [hit])
    monkeypatch.setattr(covers, "fetch", lambda f, folder, width=2400: (pic, "Art, A. Painter, 1850 (Public domain)"))
    hits, gallery, status = gui.cover_search("old painting")
    assert hits == [hit] and gallery[0][0] == "http://example/thumb.jpg" and "free-to-use" in status
    path, credit, msg = gui.cover_pick(str(work), hits, SimpleNamespace(index=0))
    assert path == str(pic) and "Public domain" in msg
    layout = list(gui.COVER_LAYOUTS)
    prev = gui.cover_render(str(work), "Some Title", "An Author", None, path, layout[0], "forest green", 0.5, 0.5, 1.0)
    assert Image.open(prev).size == (1400, 1400)
    assert Image.open(gui.cover_render(str(work), "Some Title", "An Author", None, path, layout[1], "navy", 0.3, 0.4, 1.2, )).size == (1400, 1400)
    import pytest
    with pytest.raises(Exception):
        gui.cover_render(str(work), "T", "A", None, "", layout[0], "navy", 0.5, 0.5, 1.0)             # a picture is needed
    assert gui.cover_refresh(str(work), "T", "A", None, "", layout[0], "navy", 0.5, 0.5, 1.0) == gui.gr.skip()   # but the pickers stay quiet
    job = jq_.add(str(work), "Book")
    dest, note = gui.cover_use(str(work), prev, credit, None)
    assert Path(dest).exists() and "1 queued job" in note and jq_.load()[0]["cover"] == dest
    assert "A. Painter" in (work / "cover_credit.txt").read_text()


def test_assistant_ask_returns_a_proposed_cover(tmp_path, monkeypatch):
    import stat, sys
    from audiobook_gen import assistant, gui
    p = _assistant_project(tmp_path)
    fake = tmp_path / "fakeclaude"
    fake.write_text(f"#!/bin/bash\n{sys.executable} -c \"from PIL import Image; Image.new('RGB',(1400,1400),(40,70,50)).save('cover.jpg')\"\n"
                    "echo '{\"result\":\"Made a green cover.\",\"session_id\":\"s2\",\"total_cost_usd\":0.1,\"is_error\":false}'\n")
    fake.chmod(fake.stat().st_mode | stat.S_IEXEC)
    monkeypatch.setenv("AUDIOBOOK_CLAUDE", str(fake))
    out = gui.assistant_ask(str(p), "make me a cover", [], "", 0.0)
    assert out[-1]["visible"] is True and out[-1]["value"].endswith("assistant/cover.jpg")        # the preview is shown
    assert out[6]["interactive"] is True                                                           # and it can be applied
    res = gui.assistant_apply(str(p))
    assert res[0].startswith("Applied") and res[5].endswith("cover_custom.jpg")                    # the Book tab gets the new cover


def test_whole_book_time_feeds_the_estimate(tmp_path, monkeypatch):
    from audiobook_gen import runstats
    monkeypatch.setattr(runstats, "PATH", tmp_path / "run_stats.json")
    runstats.record("r1", "chatterbox", 1, 2000, 100)                      # speech only: 20 chars/s
    assert round(runstats.averages()["chatterbox"]["cps"]) == 20 and not runstats.averages()["chatterbox"]["whole"]
    runstats.record_job("j1", {"chatterbox": 1800}, 100)                   # whole book incl. assembly: 18 chars/s
    a = runstats.averages()["chatterbox"]
    assert round(a["cps"]) == 18 and a["whole"]
    runstats.record_job("j2", {"chatterbox": 1000, "kokoro": 1000}, 50)    # mixed book shares its time between engines
    assert set(runstats.averages()) == {"chatterbox", "kokoro"}
    runstats.record_job("j3", {"chatterbox": 100}, 5)                      # too short to count
    assert runstats.averages()["chatterbox"]["runs"] == 2


def test_wait_for_does_not_match_itself(tmp_path):
    import subprocess, sys
    if not Path("tools/wait_for.py").exists():          # a local helper, not part of the repository
        pytest.skip("tools/wait_for.py is local only")
    script = tmp_path / "some_unique_job.py"
    script.write_text("import time; time.sleep(30)")
    waiter = [sys.executable, "tools/wait_for.py", "script", "some_unique_job.py", "--every", "0.2", "--timeout", "2"]
    assert subprocess.run(waiter).returncode == 0            # nothing running: its own command line must not count
    p = subprocess.Popen([sys.executable, str(script)])
    try:
        assert subprocess.run(waiter).returncode == 2        # really running: waits, then times out
    finally:
        p.kill()


def test_pace_notices_a_crawl_but_not_a_normal_run():
    from audiobook_gen.jobqueue import Pace
    p = Pace(started=0)
    fast = [i * 20.0 for i in range(100)]                       # a clip every 20 s: 30 per window
    assert not p.slow(fast, 1500)                               # fine
    assert not Pace(started=0).slow([], 100)                    # warm-up: too early to judge
    assert p.slow(fast[:50], 3000) is True                      # nothing new for a long while after being fast
    assert not Pace(started=0).slow([], 3000)                   # never got fast: nothing to compare with


def test_group_rss_counts_a_process_group():
    import os, subprocess, sys
    from audiobook_gen.jobqueue import group_rss
    p = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(5)"], start_new_session=True)
    try:
        assert group_rss(p.pid) > 0.001
    finally:
        p.kill()


def test_queue_restarts_a_long_running_speech_process_without_counting_a_failure(tmp_path, monkeypatch):
    import sys
    jq, work = _queue_env(tmp_path, monkeypatch)
    monkeypatch.setattr(jq, "PACE_EVERY", 0.1)
    out, marker = tmp_path / "x.m4b", tmp_path / "second"
    job = jq.add(str(work), "Long book", out=str(out))
    class R(jq.Runner):
        synth_cmd = lambda self, j: [sys.executable, "-c", f"import os, time\nif os.path.exists({str(marker)!r}): print('[progress] 1/1')\nelse:\n    open({str(marker)!r}, 'w').write('x'); time.sleep(60)"]
        assemble_cmd = lambda self, j: [sys.executable, "-c", f"open({str(out)!r}, 'w').write('x')"]
    R(poll=0.1, recycle=0.5, stall=60, grace=60, retry_wait=0, foreign=lambda: "").step()
    j = [x for x in jq.load() if x["id"] == job["id"]][0]
    assert j["status"] == "done" and j.get("restarts", 0) == 0


def test_speech_limit_is_learned_from_the_first_slowdown(tmp_path, monkeypatch):
    jq, work = _queue_env(tmp_path, monkeypatch)
    assert jq.speech_limit() is None                                   # nothing learned: no scheduled restarts
    assert jq.learn_speech_limit(2 * 3600) is None and jq.speech_limit() is None      # an early slowdown is not about uptime
    assert jq.learn_speech_limit(4 * 3600 + 49 * 60) == 4 * 3600 + 44 * 60            # onset minus 5 minutes
    assert jq.speech_limit() == 4 * 3600 + 44 * 60
    assert jq.learn_speech_limit(5 * 3600) is None and jq.speech_limit() == 4 * 3600 + 44 * 60   # never raised
    assert jq.learn_speech_limit(3 * 3600 + 32 * 60) == 3.5 * 3600                    # never below 3 h 30


def _wavs(folder, specs, sr=16000):
    """specs: [(name, seconds, quiet gap in the middle in seconds)] -> noise 'speech' files."""
    import numpy as np, soundfile as sf
    folder.mkdir(exist_ok=True)
    rng = np.random.default_rng(1)
    for name, secs, gap in specs:
        a = (rng.standard_normal(int(sr * secs)) * 0.1).astype("float32")
        if gap:
            mid = len(a) // 2
            a[mid: mid + int(sr * gap)] = 0
        sf.write(folder / name, a, sr)


def test_quality_check_flags_the_odd_clips(tmp_path):
    import json
    from audiobook_gen import qc
    text = "x" * 100                                           # 100 characters: about 5 s at the usual pace
    specs = [(f"ok{i}.wav", 5.0 + (i % 3) * 0.2, 0) for i in range(30)]
    specs += [("rushed.wav", 1.5, 0), ("dead.wav", 6.0, 2.6), ("slow.wav", 14.0, 0)]
    _wavs(tmp_path / "clips", specs)
    import soundfile as sf
    sf.write(tmp_path / "clips" / "empty.wav", [0.0] * 16000, 16000)
    specs.append(("empty.wav", 1, 0))
    (tmp_path / "clips_meta.json").write_text(json.dumps({n: {"text": text, "speaker": "N", "engine": "kokoro", "voice": {}} for n, *_ in specs}))
    flagged, stats = qc.scan(tmp_path, progress=lambda *_: None)
    got = {f["file"]: f["why"] for f in flagged}
    assert set(got) == {"rushed.wav", "dead.wav", "slow.wav", "empty.wav"}, got
    assert "rushed" in got["rushed.wav"][0] and "silence" in got["dead.wav"][0] and "drawn out" in got["slow.wav"][0]
    # engines without a seed are reported, not remade
    rep = qc.run(tmp_path, {}, progress=lambda *_: None)
    assert rep["flagged"] == 4 and rep["unfixed"] == 4 and (tmp_path / "qc_report.json").exists()


def test_queue_builds_the_book_even_if_the_quality_check_breaks(tmp_path, monkeypatch):
    import sys
    jq, work = _queue_env(tmp_path, monkeypatch)
    (work / "clips_meta.json").write_text("{}")
    out = tmp_path / "x.m4b"
    job = jq.add(str(work), "Checked book", out=str(out))
    class R(jq.Runner):
        synth_cmd = lambda self, j: [sys.executable, "-c", "print('[progress] 1/1')"]
        qc_cmd = lambda self, j: [sys.executable, "-c", "raise SystemExit(3)"]
        assemble_cmd = lambda self, j: [sys.executable, "-c", f"open({str(out)!r}, 'w').write('x')"]
    R(poll=0.1, foreign=lambda: "", retry_wait=0).step()
    j = [x for x in jq.load() if x["id"] == job["id"]][0]
    assert j["status"] == "done" and "could not finish" in j["qc"]
    assert any(t == "Queue finished" for t, _ in jq.sent_alerts)


def test_a_hung_audiobook_build_is_restarted_then_the_job_fails(tmp_path, monkeypatch):
    import sys
    jq, work = _queue_env(tmp_path, monkeypatch)
    job = jq.add(str(work), "Hung build", out=str(tmp_path / "x.m4b"))
    class R(jq.Runner):
        synth_cmd = lambda self, j: [sys.executable, "-c", "print('[progress] 1/1')"]
        assemble_cmd = lambda self, j: [sys.executable, "-c", "import time; time.sleep(60)"]
    R(poll=0.1, max_restarts=1, retry_wait=0, stall_scale=0.3 / jq.ASSEMBLE_STALL, foreign=lambda: "").step()
    j = [x for x in jq.load() if x["id"] == job["id"]][0]
    assert j["status"] == "failed" and j["restarts"] == 2
    assert any(t == "Book failed" for t, _ in jq.sent_alerts)


def test_the_queue_pauses_itself_after_repeated_failures_until_resumed(tmp_path, monkeypatch):
    import sys
    jq, work = _queue_env(tmp_path, monkeypatch)
    monkeypatch.setattr(jq, "service_installed", lambda: False)
    monkeypatch.setattr(jq, "ensure_supervisor", lambda: False)
    for n in range(4):
        jq.add(str(work), f"Bad {n}", out=str(tmp_path / f"{n}.m4b"))
    class R(jq.Runner):
        synth_cmd = lambda self, j: [sys.executable, "-c", "raise SystemExit(1)"]
    r = R(poll=0.1, max_restarts=0, retry_wait=0, foreign=lambda: "")
    while r.step():
        pass
    jobs = jq.load()
    assert [j["status"] for j in jobs] == ["failed"] * 3 + ["queued"]
    assert "3 books in a row" in jq.halted()
    assert r.step() is False                                       # stays paused
    assert any(t == "Queue paused" for t, _ in jq.sent_alerts)
    jq.start_runner()
    assert jq.halted() == "" and jq.settings()["fail_streak"] == 0


def test_a_finished_book_keeps_its_clips(tmp_path, monkeypatch):
    import sys
    jq, work = _queue_env(tmp_path, monkeypatch)
    (work / "clips").mkdir(); (work / "clips" / "a.wav").write_text("x")
    out = tmp_path / "x.m4b"
    jq.add(str(work), "Tidy book", out=str(out))
    jq.save_settings(clean_after_done=True)                       # an old setting must not delete anything any more
    class R(jq.Runner):
        synth_cmd = lambda self, j: [sys.executable, "-c", "print('[progress] 1/1')"]
        assemble_cmd = lambda self, j: [sys.executable, "-c", f"open({str(out)!r}, 'w').write('x')"]
    R(poll=0.1, foreign=lambda: "").step()
    assert out.exists() and (work / "clips" / "a.wav").exists()


def test_whole_queue_estimate_follows_the_windows(tmp_path, monkeypatch):
    import json, yaml
    from datetime import datetime
    from audiobook_gen import runstats
    jq, work = _queue_env(tmp_path, monkeypatch)
    monkeypatch.setattr(runstats, "PATH", tmp_path / "run_stats.json")
    runstats.record_job("j", {"kokoro": 3600}, 60)                          # 60 characters per second
    (work / "segments.json").write_text(json.dumps([{"chapter": 1, "id": 1, "speaker": "Narrator", "text": "x" * 360000}]))   # 6000 s = 100 min
    (tmp_path / "c.yaml").write_text(yaml.safe_dump({"voices": {"Narrator": {"engine": "kokoro", "voice": "bm_george"}}, "default_voice": {"engine": "kokoro", "voice": "bm_george"}}))
    job = jq.add(str(work), "Timed", out=str(tmp_path / "x.m4b"), config=str(tmp_path / "c.yaml"), window="23:00-06:30")
    e = jq.estimate_queue(datetime.strptime("2026-10-09 22:00", jq.FMT))
    assert e["unknown"] == [] and e["all"] == datetime.strptime("2026-10-10 00:40", jq.FMT)     # waits for 23:00, then 100 min


def test_ntfy_commands_answer_with_queue_state(tmp_path, monkeypatch):
    import json
    from datetime import datetime
    jq, work = _queue_env(tmp_path, monkeypatch)
    now = datetime.strptime("2026-10-09 12:00", jq.FMT)
    assert jq.answer("hello there") is None and jq.answer("") is None and jq.answer("Queue is long") is None
    assert "Nothing is being made and nothing is waiting" in jq.answer("current", now)
    assert "queue is empty" in jq.answer("QUEUE!", now) and "Nothing has finished" in jq.answer("done", now)
    a = jq.add(str(work), "First", out=str(tmp_path / "a.m4b")); b = jq.add(str(work), "Second", out=str(tmp_path / "b.m4b"), window="23:00-06:30")
    jq.update(a["id"], status="running", progress="50/100", note="making the speech")
    text = jq.answer("current", now)
    assert "“First”" in text and "50%" in text
    q = jq.answer("queue", now)
    assert "1. First — running" in q and "2. Second — queued" in q
    jq.update(a["id"], status="done", finished="2026-10-09 11:00", send_to="dev", sent="2026-10-09 11:05")
    assert "✓ First" in jq.answer("done", now) and "sent" in jq.answer("done", now)
    assert "now <number>" in jq.answer("help", now) and jq.answer("now", now) is None


def test_corrections_find_remake_and_keep_the_new_take(tmp_path, monkeypatch):
    import json
    import numpy as np
    from audiobook_gen import corrections as corr, synth

    made = []
    class Fake:
        sample_rate = 24000
        def synth(self, text, voice):
            made.append((text, voice.get("seed")))
            return np.full(2400, 0.1, np.float32)
    monkeypatch.setattr(synth, "get_engine", lambda *a, **k: Fake())
    segs = [{"id": f"001-{i:05d}", "chapter": 1 + i // 3, "speaker": "Narrator", "kind": "narration",
             "text": t, "para_start": True} for i, t in enumerate(["Vicksburg fell.", "Grant rode on.", "Vicksburg again.", "The end."])]
    (tmp_path / "segments.json").write_text(json.dumps(segs))
    cfg = {"voices": {}, "default_voice": {"engine": "kokoro", "voice": "bm_george"}, "workers": {"kokoro": 1}}
    list(synth.synthesize_iter(tmp_path, cfg))
    assert len(made) == 4
    hits = corr.search(tmp_path, "vicksburg")
    assert [h["orig"] for h in hits] == ["Vicksburg fell.", "Vicksburg again."] and hits[0]["key"] == "001-00000:0"
    assert [h["chapter"] for h in corr.search(tmp_path, chapter=2)] == [2]
    # remake one clip with other words and a seed, accept it
    clip = hits[0]
    take = corr.make_take(tmp_path, clip, "Vicks-burg fell.", {"seed": 7}, Fake())
    assert take.exists() and made[-1] == ("Vicks-burg fell.", 7)
    name = corr.accept(tmp_path, clip, "Vicks-burg fell.", {"seed": 7}, take)
    assert (tmp_path / "clips" / name).exists() and json.loads((tmp_path / "clips.json").read_text())["001-00000"] == [name]
    assert corr.search(tmp_path, "vicks-burg")[0]["corrected"]
    # the next synth run plans exactly that file for the clip: nothing is made again
    made.clear()
    list(synth.synthesize_iter(tmp_path, cfg))
    assert not made and json.loads((tmp_path / "clips.json").read_text())["001-00000"] == [name]
    # a different correction elsewhere does not leak into the neighbours
    other = corr.search(tmp_path, "grant")[0]
    assert other["voice"].get("seed") is None
    # going back to the original
    assert corr.revert(tmp_path, corr.search(tmp_path, "vicks-burg")[0])
    list(synth.synthesize_iter(tmp_path, cfg))
    assert not made and json.loads((tmp_path / "clips.json").read_text())["001-00000"] != [name]


def test_the_app_asks_before_taking_the_gpu_from_a_running_book(tmp_path, monkeypatch):
    import pytest
    from audiobook_gen import gui
    jq, work = _queue_env(tmp_path, monkeypatch)
    monkeypatch.setattr(jq, "GUI_LOCK", tmp_path / "queue" / "gui_lock")
    job = jq.add(str(work), "Busy book", out=str(tmp_path / "x.m4b"))
    jq.update(job["id"], status="running", note="making the speech")
    with pytest.raises(gui.gr.Error) as e:
        gui._need_gpu_real()
    assert "Busy book" in str(e.value) and "Pause the book" in str(e.value)
    # once the user pauses the book (takes the lease) the app can go on, and the queue waits for it
    monkeypatch.setattr(jq, "_synth_running", lambda: False)
    assert "You have the GPU" in gui.gpu_take()
    gui._need_gpu_real()
    assert "(you are using the GPU there)" in jq.foreign_synthesis()
    gui.gpu_give()
    assert jq.lease_holder() == ""


def test_a_running_book_is_paused_when_the_app_takes_the_gpu(tmp_path, monkeypatch):
    import sys, time
    jq, work = _queue_env(tmp_path, monkeypatch)
    monkeypatch.setattr(jq, "GUI_LOCK", tmp_path / "queue" / "gui_lock")
    job = jq.add(str(work), "Long book", out=str(tmp_path / "x.m4b"))
    class R(jq.Runner):
        synth_cmd = lambda self, j: [sys.executable, "-c", "import time; time.sleep(60)"]
    (tmp_path / "queue").mkdir(exist_ok=True)
    import threading
    threading.Timer(1.0, lambda: (tmp_path / "queue" / "gui_lock").write_text(f"{__import__('os').getpid()} the app")).start()
    R(poll=0.1, stall=60, grace=60, foreign=lambda: "").step()
    j = [x for x in jq.load() if x["id"] == job["id"]][0]
    assert j["status"] == "queued" and "paused while you use the GPU" in j["note"] and j.get("restarts", 0) == 0


def test_ntfy_now_and_stop(tmp_path, monkeypatch):
    jq, work = _queue_env(tmp_path, monkeypatch)
    a = jq.add(str(work), "First", out=str(tmp_path / "a.m4b"))
    b = jq.add(str(work), "Second", out=str(tmp_path / "b.m4b"), window="23:00-06:30", not_before="2030-01-01 00:00")
    c = jq.add(str(work), "Third", out=str(tmp_path / "c.m4b"))
    jq.update(a["id"], status="running", progress="10/100")
    assert "no number 9" in jq.answer("now 9")
    assert "already being made" in jq.answer("now 1")
    text = jq.answer("now 3")
    assert "“Third” goes first" in text and "“First” steps aside" in text
    jobs = {j["id"]: j for j in jq.load()}
    assert [j["title"] for j in jq.load()][0] == "Third" and jobs[a["id"]]["preempt"] is True
    jq.answer("now #3")                                              # a scheduled book loses its start time and window
    second = [j for j in jq.load() if j["title"] == "Second"][0]
    assert second["not_before"] == "" and second["window"] == ""
    assert "Stopped “First”" in jq.answer("stop") and [j for j in jq.load() if j["id"] == a["id"]][0]["status"] == "held"
    assert jq.answer("stop") == "Nothing is being made."


def test_corrections_tab_is_wired_to_its_own_boxes():
    """Regression: the Corrections boxes once shared variable names with the Clone tab, so clicking a clip filled the wrong boxes."""
    from audiobook_gen import gui
    cfg = gui.build_ui().get_config_file()
    comps = {c["id"]: c for c in cfg["components"]}
    label = lambda i: comps[i]["props"].get("label")
    pick = [d for d in cfg["dependencies"] if any(t[1] == "select" for t in d["targets"]) and "This clip now" in [label(o) for o in d["outputs"]]]
    assert len(pick) == 1
    assert [label(o) for o in pick[0]["outputs"] if label(o)] == ["This clip now", "Original text of the passage",
        "Text to speak (change a spelling or respelling here)", "Emotion (exaggeration)", "Pace / adherence (cfg weight)",
        "Seed (-1 = keep as is)", "New take"]
    take = [d for d in cfg["dependencies"] if "New take" in [label(o) for o in d["outputs"]] and any("Make a new take" in (comps[t[0]]["props"].get("value") or "") for t in d["targets"])]
    ins = [label(i) for i in take[0]["inputs"] if label(i)]
    assert "Text to speak (change a spelling or respelling here)" in ins and "Emotion (exaggeration)" in ins and "Seed (-1 = keep as is)" in ins


def test_language_pass_marks_questions_and_never_changes_words(tmp_path, monkeypatch):
    import json
    import numpy as np
    from audiobook_gen import refine, synth, emotion
    text = "Where is your brother? I do not know. Am I my brother's keeper. He went away."
    assert refine.question_candidates(text) == [2]                               # only the one that opens like a question and has no "?"
    assert refine.with_question_marks(text, {2}) == "Where is your brother? I do not know. Am I my brother's keeper? He went away."
    assert refine.with_question_marks("“Are you there.” he said.", {0}) == "“Are you there?” he said."
    assert refine.question_candidates("Why, he said, I am here!") == []         # an exclamation is left alone

    class Fake:                                                                  # says "yes" to questions; "angry, intense" for passages with "Cain"
        def letters(self, items, n, **k):
            out = []
            for s, u in items:
                if n == 2 and "direct question" in u:
                    out.append([0.95, 0.05])
                elif n == 2:                                                      # plain or emotional?
                    out.append([0.01, 0.99] if "Cain" in u.split("Passage")[-1] else [0.9, 0.1])
                elif n == 6:                                                      # which feeling: joy, sadness, anger, fear, surprise, disgust
                    out.append([0.02, 0.03, 0.85, 0.04, 0.03, 0.03])
                else:
                    out.append([0.1, 0.1, 0.8])
            return out
    segs = [{"id": "001-00000", "chapter": 1, "speaker": "Narrator", "kind": "narration", "text": "Am I my brother's keeper. Said Cain.", "para_start": True},
            {"id": "001-00001", "chapter": 1, "speaker": "Narrator", "kind": "narration", "text": "A quiet evening.", "para_start": True}]
    (tmp_path / "segments.json").write_text(json.dumps(segs))
    cfg = {"voices": {}, "default_voice": {"engine": "kokoro", "voice": "bm_george"}}
    rep = refine.run(tmp_path, cfg, scorer=Fake(), progress=lambda *_: None)
    assert list(rep["fixes"].values())[0]["to"].startswith("Am I my brother's keeper?")
    assert rep["items"]["001-00000:0"]["emo"] == "anger" and rep["items"]["001-00000:0"]["lvl"] == 1      # narration, no "!": mild
    assert rep["items"]["001-00001:0"]["emo"] == "neutral"
    # words are identical before and after
    for f in rep["fixes"].values():
        assert f["from"].replace("?", ".") == f["to"].replace("?", ".")
    # the speech step speaks the fixed text
    spoken = []
    class Eng:
        sample_rate = 24000
        def synth(self, t, v): spoken.append(t); return np.full(2400, 0.1, np.float32)
    monkeypatch.setattr(synth, "get_engine", lambda *a, **k: Eng())
    list(synth.synthesize_iter(tmp_path, cfg))
    assert any("keeper?" in t for t in spoken)
    # delivery: the language pass pulls a calm-looking passage toward its feeling
    p = emotion.blend({"neutral": 0.9, "anger": 0.1}, {"emo": "anger", "lvl": 3})
    assert p["anger"] > 0.6 and abs(sum(p.values()) - 1) < 1e-9
    assert emotion.blend({"neutral": 1.0}, None) == {"neutral": 1.0}


def test_degrees_minutes_and_compass_points_are_spelled_out():
    from audiobook_gen.lexicon import normalize
    assert normalize("in latitude 5° 3′ S. and longitude 101° W. in a small boat") == \
        "in latitude five degrees three minutes south and longitude one hundred and one degrees west in a small boat"
    assert normalize("latitude 1° S. and") == "latitude one degree south and"
    assert normalize("42° 15′ N. lat. and 60° 35′ W. long. In the") == \
        "forty-two degrees fifteen minutes north latitude and sixty degrees thirty-five minutes west longitude. In the"
    assert normalize("69° 50′ 72″ E.") == "sixty-nine degrees fifty minutes seventy-two seconds east"
    assert normalize("2 deg. or 3° below zero") == "two degrees or three degrees below zero"
    assert normalize("98° F. in the shade") == "ninety-eight degrees Fahrenheit in the shade"
    assert normalize("W.N.W., making") == "west-north-west, making"
    assert normalize("N. Smith met E. Nesbit") == "N. Smith met E. Nesbit"        # initials are left alone


def test_a_chunk_cut_mid_phrase_is_not_followed_by_a_sentence_pause(tmp_path):
    import numpy as np, soundfile as sf
    from audiobook_gen.assemble import build_chapter, chunk_pause
    p = {"sentence": 350, "paragraph": 700, "speaker_change": 250, "chapter_start": 0, "continuation": 140, "split": 30}
    assert chunk_pause("It ended here.", p) == 350 and chunk_pause("“Is it?”", p) == 350 and chunk_pause("so, then,", p) == 140
    assert chunk_pause("one hundred and one", p) == 30 and chunk_pause("", p) == 350                    # no record: decided by the ending
    assert chunk_pause({"text": "x, y", "cut": "end"}, p) == 350 and chunk_pause({"text": "x.", "cut": "space"}, p) == 30      # the record wins
    from audiobook_gen.synth import chunk_text_cuts
    long = "A short one. " + "word " * 14 + "clause, " + "tail " * 40 + "end. Last."
    kinds = [k for _, k in chunk_text_cuts(long, 100)]
    assert kinds[-1] == "end" and set(kinds) <= {"end", "comma", "space"} and "comma" in kinds
    sr = 24000
    for n in "ab":
        sf.write(tmp_path / f"{n}.wav", np.full(sr, 0.1, np.float32), sr)
    cfg = {"sample_rate": sr, "pacing_ms": p, "crossfade_ms": 60, "pacing_style": "fixed"}
    seg = [{"id": "s", "text": "x", "kind": "narration", "para_start": True}]
    clips = {"s": ["a.wav", "b.wav"]}
    cut = build_chapter(seg, clips, tmp_path, cfg, {"a.wav": {"text": "longitude one hundred and one", "cut": "space"}, "b.wav": {"text": "degrees west."}})
    full = build_chapter(seg, clips, tmp_path, cfg, {"a.wav": {"text": "It ended.", "cut": "end"}, "b.wav": {"text": "Next."}})
    assert len(full) - len(cut) == int(sr * 0.32)                  # 350 ms vs 30 ms
    assert len(build_chapter(seg, clips, tmp_path, cfg)) == len(full)      # no clip list: the old sentence pause


def test_a_finished_book_is_sent_while_the_next_one_is_being_made(tmp_path, monkeypatch):
    import sys, time
    jq, work = _queue_env(tmp_path, monkeypatch)
    done_out = tmp_path / "done.m4b"; done_out.write_text("x")
    a = jq.add(str(work), "Finished", out=str(done_out), send_to="dev")
    jq.update(a["id"], status="done", send_try=time.time())     # just tried: the step's own delivery pass skips it
    b = jq.add(str(work), "Running", out=str(tmp_path / "b.m4b"))
    sent = []
    class R(jq.Runner):
        synth_cmd = lambda self, j: [sys.executable, "-c", "import time; time.sleep(2.5); print('[progress] 1/1')"]
        assemble_cmd = lambda self, j: [sys.executable, "-c", f"open({str(tmp_path / 'b.m4b')!r}, 'w').write('x')"]
    r = R(poll=0.2, foreign=lambda: "", sender=lambda d, p: sent.append(p) or True, send_every=1.0)
    r.step()                                   # makes "Running"; delivery of "Finished" must happen during it, not only before it
    assert sent and [j for j in jq.load() if j["id"] == a["id"]][0]["sent"]


def test_narrator_style_pauses_follow_the_measured_spread():
    import numpy as np
    from audiobook_gen.assemble import draw_pause, narrator_pacing
    table = narrator_pacing()
    assert {"sentence_narration", "sentence_dialogue", "paragraph", "comma_narration", "comma_dialogue", "before_tag", "speaker_change"} <= set(table)
    draw = lambda kind, n=3000: np.array([draw_pause(kind, f"clip{i}.wav", table) for i in range(n)])
    narr, dial = draw("sentence_narration"), draw("sentence_dialogue")
    assert 640 <= np.median(narr) <= 780 and np.median(dial) < np.median(narr) - 80          # speech sentences are paused shorter
    assert (draw("comma_dialogue") < 100).mean() > (draw("comma_narration") < 100).mean() > 0.2    # many commas get almost no pause
    assert np.median(draw("before_tag")) < 100                                                      # a quote runs straight into its tag
    assert 650 <= np.median(draw("paragraph")) <= 850
    assert all(x >= 40 for x in draw("before_tag", 500))                                            # never so short it would crossfade
    assert draw_pause("sentence_narration", "a.wav", table) == draw_pause("sentence_narration", "a.wav", table)      # same clip, same pause
    assert draw_pause("no_such_kind", "a.wav", table) is None


def test_narrator_style_chooses_the_kind_from_speech_or_narration(tmp_path, monkeypatch):
    import numpy as np, soundfile as sf
    from audiobook_gen import assemble
    sr = 24000
    for n in "abc":
        sf.write(tmp_path / f"{n}.wav", np.full(sr, 0.1, np.float32), sr)
    asked = []
    monkeypatch.setattr(assemble, "draw_pause", lambda kind, key, table=None: asked.append(kind) or 100)
    monkeypatch.setattr(assemble, "tables_for", lambda cfg: ({"x": 1}, {}, "test"))
    p = {"sentence": 350, "paragraph": 700, "speaker_change": 250, "chapter_start": 0, "continuation": 140, "split": 30}
    cfg = {"sample_rate": sr, "pacing_ms": p, "crossfade_ms": 60}
    segs = [{"id": "s1", "text": "x", "kind": "dialogue", "para_start": True}, {"id": "s2", "text": "x", "kind": "narration", "para_start": False}]
    meta = {"a.wav": {"text": "Hello.", "cut": "end"}, "b.wav": {"text": "x"}, "c.wav": {"text": "y"}}
    assemble.build_chapter(segs[:1] + [{**segs[0], "id": "s3"}], {"s1": ["a.wav", "b.wav"], "s3": ["c.wav"]}, tmp_path, cfg, meta)
    assert asked == ["sentence_dialogue", "paragraph"]                       # a join inside speech, then a new paragraph
    asked.clear()
    meta["a.wav"] = {"text": "Hello,", "cut": "end"}
    assemble.build_chapter([segs[0], segs[1]], {"s1": ["a.wav"], "s2": ["b.wav"]}, tmp_path, cfg, meta)
    assert asked == ["before_tag"]                                           # speech running into its narration tag


def test_emotion_scales_sentence_and_paragraph_pauses_gently():
    from audiobook_gen.assemble import emotion_scale
    c = {"sentence": {"coefficient": -0.38}, "paragraph": {"coefficient": 0.22}}
    assert emotion_scale("sentence_narration", 0.0, c) == 1.0 and emotion_scale("sentence_dialogue", None, c) == 1.0
    assert 0.65 < emotion_scale("sentence_narration", 1.0, c) < 0.72 or emotion_scale("sentence_narration", 1.0, c) == 0.7      # shorter, never below 0.7
    assert 1.2 < emotion_scale("paragraph", 1.0, c) < 1.3                                                                   # a little longer
    assert emotion_scale("comma_narration", 1.0, c) == 1.0 and emotion_scale("before_tag", 1.0, c) == 1.0                   # no effect worth using


def test_pauses_use_the_feeling_of_the_sentence_before(tmp_path):
    import numpy as np, soundfile as sf
    from audiobook_gen import assemble
    sr = 24000
    for n in "ab":
        sf.write(tmp_path / f"{n}.wav", np.full(sr, 0.1, np.float32), sr)
    p = {"sentence": 350, "paragraph": 700, "speaker_change": 250, "chapter_start": 0, "continuation": 140, "split": 30}
    cfg = {"sample_rate": sr, "pacing_ms": p, "crossfade_ms": 60}
    seg = [{"id": "s", "text": "x", "kind": "narration", "para_start": True}]
    clips, meta = {"s": ["a.wav", "b.wav"]}, {"a.wav": {"text": "It ended.", "cut": "end"}, "b.wav": {"text": "Next."}}
    calm = assemble.build_chapter(seg, clips, tmp_path, cfg, meta, {"a.wav": 0.0})
    upset = assemble.build_chapter(seg, clips, tmp_path, cfg, meta, {"a.wav": 1.0})
    assert len(calm) > len(upset)                                    # the same drawn pause, shortened after an emotional sentence


def _pacing_records(n=400, seed=3):
    import random
    rng = random.Random(seed)
    recs = []
    for i in range(n):
        base = rng.choice(["period", "comma", "paragraph"])
        in_quote = rng.random() < 0.4
        feel = rng.random()
        pause = (800 if base == "period" else 500 if base == "comma" else 900) * (0.6 if in_quote else 1.0) * (1 - 0.3 * feel) * rng.uniform(0.8, 1.2)
        if base == "comma" and rng.random() < 0.4:
            pause = rng.uniform(0, 80)
        recs.append({"cls": base, "base": base, "pause": pause, "in_quote": in_quote, "words": rng.randint(4, 30), "feel": feel})
    return recs


def test_a_pacing_profile_is_fitted_saved_with_the_voice_and_used_by_its_books(tmp_path, monkeypatch):
    import numpy as np, soundfile as sf
    from audiobook_gen import assemble, pacing, voices
    monkeypatch.setattr(voices, "LIB", tmp_path / "library")
    (tmp_path / "library" / "Slowpoke").mkdir(parents=True)
    table = pacing.fit_table(_pacing_records())
    assert {"sentence_narration", "comma_narration", "paragraph"} <= set(table["kinds"]) and table["emotion"]["sentence"]["coefficient"] < 0
    table["records"] = 400
    pacing.save_profile("Slowpoke", table, "book.m4b")
    assert pacing.load_profile("Slowpoke")["records"] == 400
    narrated_by = lambda name: {"voices": {"Narrator": {"engine": "chatterbox", "library": name}}}
    assert pacing.tables_for(narrated_by("Slowpoke"))[2] == "Slowpoke"
    assert pacing.tables_for(narrated_by("NoProfileVoice"))[2] == "default"           # voices without a profile use the default table
    assert pacing.tables_for({"single_voice": {"enabled": True, "voice": {"library": "Slowpoke"}}})[2] == "Slowpoke"
    # a book narrated by the voice is paced by its profile: here every sentence pause is about 3 s
    slow = {"kinds": {"sentence_narration": {"n": 99, "short_share": 0.0, "median_ms": 3000, "mu": 8.0, "sigma": 0.01, "lo_ms": 2900, "hi_ms": 3100, "short_range_ms": [20, 100]}},
            "emotion": {}}
    pacing.save_profile("Slowpoke", {**slow, "records": 99}, "x")
    sr = 24000
    for n in "ab":
        sf.write(tmp_path / f"{n}.wav", np.full(sr, 0.1, np.float32), sr)
    p = {"sentence": 350, "paragraph": 700, "speaker_change": 250, "chapter_start": 0, "continuation": 140, "split": 30}
    seg = [{"id": "s", "text": "x", "kind": "narration", "para_start": True}]
    clips, meta = {"s": ["a.wav", "b.wav"]}, {"a.wav": {"text": "It ended.", "cut": "end"}, "b.wav": {"text": "Next."}}
    slow_book = assemble.build_chapter(seg, clips, tmp_path, {"sample_rate": sr, "pacing_ms": p, "crossfade_ms": 60, **narrated_by("Slowpoke")}, meta)
    other = assemble.build_chapter(seg, clips, tmp_path, {"sample_rate": sr, "pacing_ms": p, "crossfade_ms": 60, **narrated_by("Other")}, meta)
    assert len(slow_book) - len(other) > sr * 1.5
    assert pacing.remove_profile("Slowpoke") and pacing.load_profile("Slowpoke") is None


def test_the_clone_tab_starts_and_reports_a_pacing_measurement(tmp_path, monkeypatch):
    import pytest
    from audiobook_gen import gui, pacing, voices
    monkeypatch.setattr(voices, "LIB", tmp_path / "library")
    (tmp_path / "library" / "V").mkdir(parents=True)
    monkeypatch.setattr(pacing, "ROOT", tmp_path)
    with pytest.raises(gui.gr.Error):
        gui.pacing_start("V", str(tmp_path / "missing.m4b"), str(tmp_path / "missing.epub"), 4, 30)
    with pytest.raises(gui.gr.Error):
        gui.pacing_start("", "a", "b", 4, 30)
    text, table = gui.pacing_status("V")
    assert "no pacing profile" in text and len(table) == 0
    pacing.save_profile("V", {**pacing.fit_table(_pacing_records()), "records": 400}, "x.m4b")
    text, table = gui.pacing_status("V")
    assert "has its own pacing profile" in text and len(table) >= 4


def test_chapters_encoded_side_by_side_match_the_single_pass_encode(tmp_path, monkeypatch):
    import json, subprocess
    import numpy as np
    from audiobook_gen import assemble, synth
    class Eng:
        sample_rate = 24000
        def synth(self, text, voice):
            t = np.arange(24000) / 24000
            return (0.2 * np.sin(2 * np.pi * (200 + 40 * len(text)) * t)).astype(np.float32)
    monkeypatch.setattr(synth, "get_engine", lambda *a, **k: Eng())
    segs = [{"id": f"{c:03d}-{i:05d}", "chapter": c, "speaker": "Narrator", "kind": "narration", "text": f"Chapter {c} sentence {i}.", "para_start": i == 0}
            for c in (1, 2, 3) for i in range(3)]
    (tmp_path / "segments.json").write_text(json.dumps(segs))
    (tmp_path / "chapters.json").write_text(json.dumps({"title": "T", "author": "A", "chapters": [{"index": c, "title": f"Ch {c}", "text": ""} for c in (1, 2, 3)]}))
    cfg = {"voices": {}, "default_voice": {"engine": "kokoro", "voice": "bm_george"}, "workers": {"kokoro": 1}, "sample_rate": 24000,
           "pacing_ms": {"sentence": 350, "paragraph": 700, "speaker_change": 250, "chapter_start": 500}, "crossfade_ms": 60, "pacing_style": "fixed"}
    list(synth.synthesize_iter(tmp_path, cfg))
    probe = lambda f: json.loads(subprocess.run(["ffprobe", "-v", "error", "-print_format", "json", "-show_chapters", "-show_format", str(f)], capture_output=True, text=True).stdout)
    outs = {}
    cfg = {**cfg, "audio_codec": "aac"}                                       # the AAC path joins separately encoded chapters; Opus is encoded in one pass
    for label, workers in (("serial", 1), ("parallel", 3)):
        outs[label] = probe(assemble.assemble(tmp_path, {**cfg, "encode_workers": workers}, tmp_path / f"{label}.m4b", None, "T", "A"))
    assert abs(float(outs["serial"]["format"]["duration"]) - float(outs["parallel"]["format"]["duration"])) < 0.2
    assert len(outs["parallel"]["chapters"]) == 3
    for a, b in zip(outs["serial"]["chapters"], outs["parallel"]["chapters"]):
        assert abs(float(a["start_time"]) - float(b["start_time"])) < 0.15
    # if the pieces do not join to the expected length the assembler falls back to one pass instead of shipping a skewed book
    monkeypatch.setattr(assemble, "AAC_FRAME", 48000)
    out = probe(assemble.assemble(tmp_path, {**cfg, "encode_workers": 3}, tmp_path / "fallback.m4b", None, "T", "A"))
    assert abs(float(out["format"]["duration"]) - float(outs["serial"]["format"]["duration"])) < 0.2
    # Opus is the default codec: the same book, in an .m4b, with the same chapters and length
    opus_cfg = {k: v for k, v in cfg.items() if k != "audio_codec"}
    opus_path = assemble.assemble(tmp_path, {**opus_cfg, "encode_workers": 3}, tmp_path / "opus.m4b", None, "T", "A")
    streams = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "a", "-show_entries", "stream=codec_name", "-of", "csv=p=0", str(opus_path)],
                             capture_output=True, text=True).stdout.split()
    o = probe(opus_path)
    assert streams == ["opus"] and len(o["chapters"]) == 3 and abs(float(o["format"]["duration"]) - float(outs["serial"]["format"]["duration"])) < 0.3
    assert subprocess.run(["ffmpeg", "-v", "error", "-i", str(opus_path), "-f", "null", "-"], capture_output=True).stderr == b""      # decodes clean
    assert assemble.audio_codec({"audio_codec": "AAC"}) == "aac" and "loudnorm=I=-16.0" in assemble.loudnorm({})


def test_a_send_the_phone_never_fetched_counts_as_failed(monkeypatch):
    import subprocess
    from audiobook_gen import jobqueue as jq
    monkeypatch.setattr(jq, "kde_devices", lambda: [{"id": "dev", "name": "Phone", "reachable": True}])
    class Done:
        returncode = 0
    monkeypatch.setattr(jq.subprocess, "run", lambda *a, **k: Done())
    assert jq.send_file("dev", "/x", journal=lambda since: "", verify_seconds=0.2) is True                          # nothing logged: it was fetched
    assert jq.send_file("dev", "/x", journal=lambda since: "kdeconnectd: CompositeUploadJob::timeoutTriggered() - no connection received", verify_seconds=0.2) is False
    monkeypatch.setattr(jq, "kde_devices", lambda: [{"id": "dev", "name": "Phone", "reachable": False}])
    assert jq.send_file("dev", "/x", journal=lambda since: "", verify_seconds=0.2) is False                         # out of reach


def test_notify_file_accepts_a_multi_line_caption(tmp_path, monkeypatch):
    from audiobook_gen import jobqueue as jq
    seen = {}
    class Resp:
        status = 200
    def fake_urlopen(req, timeout=0):
        seen.update(req.headers); return Resp()
    monkeypatch.setattr(jq.urllib.request, "urlopen", fake_urlopen)
    monkeypatch.setenv("AUDIOBOOK_NTFY", "some-topic")
    f = tmp_path / "x.mp3"; f.write_bytes(b"abc")
    assert jq.notify_file(str(f), "line one\nline two\n3. three", "A title") is True
    assert "\n" not in seen["Message"] and seen["Message"].startswith("line one | line two")


def test_quality_check_ignores_lexicon_markup_and_short_interjections():
    from audiobook_gen import qc
    assert qc.spoken_chars("But, [Montgomery](/məntˈɡʌməɹi/), come.") == len("But, Montgomery, come.")
    quiet_short = {"seconds": 0.5, "speech": 0.15, "longest_gap": 0.0, "silent_share": 0.7}          # "No!" with padding
    assert qc.problems(quiet_short, 3, 20.0, "No!") == []
    quiet_long = {"seconds": 4.0, "speech": 1.0, "longest_gap": 1.2, "silent_share": 0.75}
    assert "mostly silence" in qc.problems(quiet_long, 60, 20.0, "A long line that came out almost silent here.")
    assert qc.problems({"seconds": 1.2, "speech": 0.5, "longest_gap": 0.0, "silent_share": 0.1}, qc.spoken_chars("[Montgomery](/məntˈɡʌməɹi/),"), 20.0,
                       "[Montgomery](/məntˈɡʌməɹi/),") == []       # 11 spoken characters: too short to judge by pace


def test_wait_for_reply_skips_our_posts_and_the_command_words(monkeypatch):
    import io, json
    from audiobook_gen import jobqueue as jq
    monkeypatch.setenv("AUDIOBOOK_NTFY", "t")
    now = 1000
    msgs = [{"event": "open"}, {"event": "message", "id": "1", "time": now + 1, "title": "Which Montgomery?", "message": "Option 1..."},
            {"event": "message", "id": "2", "time": now + 2, "message": "queue"}, {"event": "message", "id": "3", "time": now + 3, "message": "now 3"},
            {"event": "message", "id": "4", "time": now + 4, "message": "Option 2"}]
    monkeypatch.setattr(jq.urllib.request, "urlopen", lambda *a, **k: io.StringIO("\n".join(json.dumps(m) for m in msgs)))
    assert jq.wait_for_reply(now, timeout=2, every=0.01) == "Option 2"
    monkeypatch.setattr(jq.urllib.request, "urlopen", lambda *a, **k: io.StringIO(json.dumps(msgs[1])))
    assert jq.wait_for_reply(now, timeout=0.1, every=0.01) is None


def test_paragraph_pause_scale_and_the_longer_gap_after_a_section_background(tmp_path):
    import numpy as np, soundfile as sf
    from audiobook_gen import assemble
    sr = 24000
    sf.write(tmp_path / "a.wav", np.full(sr, 0.1, np.float32), sr)
    p = {"sentence": 350, "paragraph": 700, "speaker_change": 250, "chapter_start": 0, "continuation": 140, "split": 30}
    segs = [{"id": "s1", "text": "One.", "kind": "narration", "para_start": True},
            {"id": "s2", "text": "Two.", "kind": "narration", "para_start": True}]
    clips = {"s1": ["a.wav"], "s2": ["a.wav"]}

    def pause(extra=None, mark=False):
        cfg = {"sample_rate": sr, "pacing_ms": p, "crossfade_ms": 0, "pacing_style": "fixed", **(extra or {})}
        two = [segs[0], {**segs[1], **({"section_break": True} if mark else {})}]
        return len(assemble.build_chapter(two, clips, tmp_path, cfg)) / sr - 2.0
    base = pause()
    assert abs(pause({"pacing_paragraph_scale": 2.0}) - 2 * base) < 0.01                                             # twice as long
    assert abs(pause({"pacing_paragraph_scale": 2.0, "pacing_section_gap_scale": 2.0}) - 2 * base) < 0.01           # no mark: no extra
    assert abs(pause({"pacing_paragraph_scale": 2.0, "pacing_section_gap_scale": 2.0}, mark=True) - 4 * base) < 0.01   # marked: twice that again
    assert abs(pause({"pacing_section_gap_s": 3.0}, mark=True) - 3.0) < 0.01                                          # a fixed gap in seconds at the break
    assert abs(pause({"pacing_section_gap_s": 3.0}) - base) < 0.01                                                    # but not elsewhere


def test_the_assemble_step_uses_the_cover_set_while_the_clips_were_being_made(monkeypatch):
    from audiobook_gen import jobqueue
    started = {"id": "j1", "title": "T", "author": "A", "work": "w", "config": "c", "out": "o", "cover": ""}
    monkeypatch.setattr(jobqueue, "load", lambda: [{**started, "cover": "/covers/new.jpg", "title": "New title"}])
    cmd = jobqueue.Runner.assemble_cmd(object.__new__(jobqueue.Runner), started)
    assert cmd[cmd.index("--cover") + 1] == "/covers/new.jpg" and cmd[cmd.index("--title") + 1] == "New title"


def _music_library(tmp_path, monkeypatch, seconds=40):
    import numpy as np, soundfile as sf
    from audiobook_gen import music
    lib = tmp_path / "lib"
    lib.mkdir()
    t = np.arange(24000 * seconds) / 24000
    sf.write(lib / "tone.wav", (0.3 * np.sin(2 * np.pi * 220 * t)).astype("float32"), 24000)
    (lib / "index.json").write_text('[{"id": "tone", "title": "Tone", "artist": "x", "licence": "CC0", "seconds": %d, "file": "tone.wav", "credit": "Tone, x (CC0)"}]' % seconds)
    monkeypatch.setattr(music, "LIB", lib)
    monkeypatch.setattr(music, "INDEX", lib / "index.json")
    return music


def test_music_goes_before_the_first_chapter_at_recommended_bridges_and_after_the_last(tmp_path, monkeypatch):
    import numpy as np
    music = _music_library(tmp_path, monkeypatch)
    sr = 24000
    guess = [(1, "a", 1000.0), (2, "b", 1000.0), (3, "c", 1000.0), (4, "d", 1000.0), (5, "e", 1000.0)]
    assert music.recommended(guess, 25) == [3, 5] or music.recommended(guess, 25) == [3]       # about every 25 min, never at the first chapter
    plan = music.load_plan(tmp_path)
    plan.update(enabled=True, interval_min=25)
    plan["intro"].update(track="tone", seconds=10)
    plan["bridge"].update(track="tone", seconds=5)
    plan["outro"].update(track="tone", before=2, after=6, fade_out=3)
    speech = (0.2 * np.random.default_rng(1).standard_normal(sr * 3)).astype("float32")
    made = [1, 2, 3, 4, 5]
    n = lambda ch: len(music.wrap(speech, ch, made, plan, guess, sr)) / sr
    assert abs(n(1) - 13) < 0.1 and abs(n(2) - 3) < 0.1                                           # intro before 1; nothing before 2
    bridged = music.bridge_before(plan, guess)
    assert 3 in bridged and abs(n(3) - 8) < 0.1
    assert abs(n(5) - (3 + 6 + (5 if 5 in bridged else 0))) < 0.1                               # the ending after the last
    plan["chapters"] = {"2": "on", "3": "off"}                                                  # a person's own choices win
    assert abs(n(2) - 8) < 0.1 and abs(n(3) - 3) < 0.1
    plan["enabled"] = False
    assert len(music.wrap(speech, 1, made, plan, guess, sr)) == len(speech)
    assert music.credits({**plan, "intro": {"track": "tone"}}) == ["Tone, x (CC0)"]


def test_music_is_set_against_the_speech_level_and_never_clips(tmp_path, monkeypatch):
    import numpy as np
    music = _music_library(tmp_path, monkeypatch)
    sr = 24000
    speech = (0.1 * np.random.default_rng(2).standard_normal(sr * 2)).astype("float32")
    spec = {"track": "tone", "seconds": 8, "start": 0, "fade_in": 1, "fade_out": 2}
    a = music.piece(spec, sr, music.speech_rms(speech), -6.0)
    assert abs(a).max() <= 0.7001 and abs(a[:200]).max() < 0.05 and abs(a[-200:]).max() < 0.05      # faded at both ends
    mid = a[sr * 2: sr * 5]
    assert 0.4 < music.speech_rms(mid) / music.speech_rms(speech) < 0.6                            # about -6 dB under the speech
    assert music.piece({"track": "missing"}, sr, 0.1, -4.0) is None


def test_music_search_only_offers_free_recordings(monkeypatch):
    from audiobook_gen import music, covers
    pages = {"1": {"index": 1, "title": "File:Good piece.ogg", "imageinfo": [{"mime": "application/ogg", "duration": 120, "extmetadata": {"LicenseShortName": {"value": "CC0"}}}]},
             "2": {"index": 2, "title": "File:Restricted.ogg", "imageinfo": [{"mime": "application/ogg", "duration": 120, "extmetadata": {"LicenseShortName": {"value": "CC BY-SA 4.0"}}}]},
             "3": {"index": 3, "title": "File:Notes.mid", "imageinfo": [{"mime": "audio/midi", "duration": 120, "extmetadata": {"LicenseShortName": {"value": "Public domain"}}}]},
             "4": {"index": 4, "title": "File:Snippet.ogg", "imageinfo": [{"mime": "application/ogg", "duration": 8, "extmetadata": {"LicenseShortName": {"value": "Public domain"}}}]}}
    monkeypatch.setattr(covers, "_api", lambda **p: {"query": {"pages": pages}})
    assert [h["file"] for h in music.search("x")] == ["File:Good piece.ogg"]


def test_the_music_tab_saves_and_reloads_a_books_plan(tmp_path, monkeypatch):
    import json
    from audiobook_gen import gui
    music = _music_library(tmp_path, monkeypatch)
    work = tmp_path / "book"
    work.mkdir()
    (work / "chapters.json").write_text(json.dumps({"title": "T", "chapters": [{"index": i, "title": f"Part {i}", "text": "x" * 13050} for i in range(1, 8)]}))
    out = gui.music_load(str(work))
    assert out[0] is False and "7 sections" in out[13]                                       # nothing saved yet; the note counts the sections
    rec = music.recommended(music.chapter_seconds(work), 25)
    assert rec and 1 not in rec
    msg, credits = gui.music_save(str(work), True, "tone", "tone", "tone", 20, 0, 8, 0, 30, 0, -4, 25, [rec[0], 6])
    assert "Music is ON" in msg and credits == "Tone, x (CC0)"
    plan = music.load_plan(work)
    assert plan["enabled"] and plan["intro"]["track"] == "tone" and plan["chapters"].get("6") == "on"      # a tick the recommendation lacks is kept
    again = gui.music_load(str(work))
    assert again[0] is True and set(again[12]["value"]) >= {rec[0], 6}
    assert gui.music_hear("intro", "tone", 6, 0, -4).endswith("music_preview_intro.wav")


def test_the_layered_opening_puts_the_title_over_quiet_music_and_fades_out_behind_the_book(tmp_path, monkeypatch):
    import json
    import numpy as np
    music = _music_library(tmp_path, monkeypatch, seconds=60)
    sr = 24000
    plan = music.load_plan(tmp_path)
    plan.update(enabled=True, level_db=0.0)
    plan["intro"].update(track="tone", title="The Gospel of John", lead=6, tail=6, fade=10, duck_db=-14, under_db=-9, fade_in=0.5)
    title = (0.2 * np.sin(2 * np.pi * 440 * np.arange(sr * 4) / sr)).astype("float32")               # 4 s of "speech"
    head, tail = music.opening(title, plan, sr)
    assert abs(len(head) / sr - 16) < 0.05 and abs(len(tail) / sr - 10) < 0.3                       # lead + title + tail; then the fade
    rms = lambda a: float(np.sqrt(np.mean(a ** 2)))
    alone, under_title, after = head[sr * 2: sr * 5], head[sr * 7: sr * 9], head[sr * 12: sr * 15]
    mix_only = head[: sr * 6]
    assert rms(mix_only) > 0                                                                          # music before the title
    assert rms(after) > 0.3 * rms(alone)                                                              # music alone again after it
    assert abs(tail[-sr // 10:]).max() < 0.01 and rms(tail[: sr]) > rms(tail[-sr * 2: -sr])           # slowly out
    assert abs(head[: sr // 50]).max() < 0.05                                                          # begins with a gentle fade-in
    speech = (0.1 * np.random.default_rng(3).standard_normal(sr * 20)).astype("float32")
    laid = music.apply_tail(speech, tail)
    assert len(laid) == len(speech) and rms(laid[:sr * 3]) > rms(speech[:sr * 3]) and abs(laid.max()) <= 0.9801
    assert len(music.apply_tail(speech, np.zeros(0, np.float32))) == len(speech)
    # the Opening chapter is added once, read by the narrator, and can be taken away again
    (tmp_path / "chapters.json").write_text(json.dumps({"title": "T", "chapters": [{"index": 1, "title": "Part 1", "text": "x"}]}))
    (tmp_path / "segments.json").write_text(json.dumps([{"speaker": "Narrator", "kind": "narration", "text": "x", "para_start": True, "chapter": 1, "id": "001-00000"}]))
    music.add_opening(tmp_path, "The Gospel of John, a study edition.")
    music.add_opening(tmp_path, "The Gospel of John, a study edition.")
    meta, segs = json.loads((tmp_path / "chapters.json").read_text()), json.loads((tmp_path / "segments.json").read_text())
    assert [c["title"] for c in meta["chapters"]] == ["Opening", "Part 1"] and meta["chapters"][0]["opening"] and len(segs) == 2 and segs[0]["chapter"] == 0
    assert [c[0] for c in music.chapter_seconds(tmp_path)] == [1] and music.has_opening(tmp_path)           # the opening is not a section for bridges
    music.add_opening(tmp_path, "")
    assert not music.has_opening(tmp_path) and len(json.loads((tmp_path / "segments.json").read_text())) == 1
    # an opening can also come after a chosen chapter (a preface), in both the chapter list and the segment order
    (tmp_path / "chapters.json").write_text(json.dumps({"title": "T", "chapters": [{"index": 1, "title": "Preface", "text": "x"}, {"index": 2, "title": "One", "text": "y"}]}))
    (tmp_path / "segments.json").write_text(json.dumps([{"speaker": "Narrator", "kind": "narration", "text": t, "para_start": True, "chapter": c, "id": f"{c:03d}-00000"} for c, t in ((1, "x"), (2, "y"))]))
    music.add_opening(tmp_path, "Title.\nBy Someone.", 2.8, after=1)
    assert [c["title"] for c in json.loads((tmp_path / "chapters.json").read_text())["chapters"]] == ["Preface", "Opening", "One"]
    assert [g["chapter"] for g in json.loads((tmp_path / "segments.json").read_text())] == [1, 0, 0, 2]


def test_the_ending_fades_in_under_the_last_words_and_carries_on_after_them(tmp_path, monkeypatch):
    import numpy as np
    music = _music_library(tmp_path, monkeypatch, seconds=60)
    sr = 24000
    plan = music.load_plan(tmp_path)
    plan.update(enabled=True, level_db=0.0)
    plan["outro"].update(track="tone", before=20, after=9, fade_out=5, under_db=-11)
    speech = np.clip(0.2 * np.random.default_rng(5).standard_normal(sr * 60), -0.6, 0.6).astype("float32")
    out = music.ending(speech, plan, sr)
    assert abs(len(out) / sr - 69) < 0.05                                                  # nine seconds longer
    rms = lambda a: float(np.sqrt(np.mean(a ** 2)))
    assert np.allclose(out[: sr * 30], speech[: sr * 30])                                  # untouched until 20 s before the end
    diff = out[: len(speech)] - speech
    early, late = rms(diff[sr * 41: sr * 44]), rms(diff[sr * 56: sr * 59])
    assert 0 < early < late                                                                # fading in under the speech
    tail = out[len(speech):]
    assert 0.5 * rms(diff[sr * 58: sr * 60]) < rms(tail[: sr]) < 1.5 * rms(diff[sr * 58: sr * 60])     # still playing, turned down, for the first second: no gap
    assert rms(tail[sr * 3: sr * 5]) > 2 * rms(tail[: sr])                                # then swells
    assert abs(tail[-sr // 20:]).max() < 0.01 and abs(out).max() <= 0.9801                 # out at the very end, no clipping
    plan["outro"]["track"] = ""
    assert len(music.wrap(speech, 1, [1], plan, [(1, "a", 60.0)], sr)) == len(speech)     # no track, no change


def test_a_dramatic_pause_after_the_title_is_set_per_segment_and_the_music_swells_into_it(tmp_path, monkeypatch):
    import json
    import numpy as np
    import soundfile as sf
    from audiobook_gen import assemble
    music = _music_library(tmp_path, monkeypatch, seconds=60)
    sr = 24000
    (tmp_path / "chapters.json").write_text(json.dumps({"title": "T", "chapters": [{"index": 1, "title": "Part 1", "text": "x"}]}))
    (tmp_path / "segments.json").write_text(json.dumps([{"speaker": "Narrator", "kind": "narration", "text": "x", "para_start": True, "chapter": 1, "id": "001-00000"}]))
    music.add_opening(tmp_path, "The Gospel of John, a study edition.\nA copyright-free collection of introductions.", pause_s=2.8)
    segs = json.loads((tmp_path / "segments.json").read_text())
    assert [g["id"] for g in segs[:3]] == ["000-00000", "000-00001", "001-00000"] and segs[1]["pause_ms"] == 2800 and "pause_ms" not in segs[0]
    # build_chapter honours the set pause instead of drawing a paragraph pause
    sf.write(tmp_path / "a.wav", np.full(sr, 0.1, np.float32), sr)
    cfg = {"sample_rate": sr, "pacing_ms": {"sentence": 350, "paragraph": 700, "speaker_change": 250, "chapter_start": 0, "continuation": 140, "split": 30},
           "crossfade_ms": 0, "pacing_style": "fixed"}
    two = [{**g, "kind": "narration"} for g in segs[:2]]
    out = assemble.build_chapter(two, {"000-00000": ["a.wav"], "000-00001": ["a.wav"]}, tmp_path, cfg)
    assert abs(len(out) / sr - (2.0 + 2.8)) < 0.02
    # the music turns up in that silence and down again for the second part
    talk = np.concatenate([0.3 * np.sin(2 * np.pi * 300 * np.arange(sr * 3) / sr), np.zeros(int(sr * 2.8)), 0.3 * np.sin(2 * np.pi * 300 * np.arange(sr * 3) / sr)]).astype("float32")
    assert len(music.long_silences(talk, sr)) == 1 and abs(music.long_silences(talk, sr)[0][0] - 3.0) < 0.2
    plan = music.load_plan(tmp_path)
    plan.update(enabled=True, level_db=0.0)
    plan["intro"].update(track="tone", title="x", lead=4, tail=4, fade=6, duck_db=-18, under_db=-9)
    head, _ = music.opening(talk, plan, sr)
    music_only = np.zeros_like(head)
    rms = lambda a: float(np.sqrt(np.mean(a ** 2)))
    mixed = head.copy()
    mixed[sr * 4: sr * 4 + len(talk)] -= talk                                                        # what is left is the music
    during_words, in_the_pause = rms(mixed[sr * 5: sr * 6]), rms(mixed[sr * 4 + int(sr * 4.2): sr * 4 + int(sr * 5.2)])
    assert in_the_pause > 2 * during_words                                                         # swells into the pause


def test_section_titles_are_read_aloud_with_a_pause_and_can_be_taken_out_again(tmp_path):
    import json
    from audiobook_gen import titles
    assert titles.spoken("Part 1: The Word Became Flesh") == "Part one. The Word Became Flesh."
    assert titles.spoken("Part 48: Jesus Restores Peter") == "Part forty-eight. Jesus Restores Peter."
    assert titles.spoken("The Epilogue") == "The Epilogue."
    chapters = [{"index": 0, "title": "Opening", "text": "t", "opening": True}, {"index": 1, "title": "Part 1: A", "text": "x"}, {"index": 2, "title": "Part 2: B", "text": "y"}]
    (tmp_path / "chapters.json").write_text(json.dumps({"title": "T", "chapters": chapters}))
    seg = lambda i, ch, t: {"speaker": "Narrator", "kind": "narration", "text": t, "para_start": True, "chapter": ch, "id": i}
    (tmp_path / "segments.json").write_text(json.dumps([seg("000-00000", 0, "Title."), seg("001-00000", 1, "a"), seg("001-00001", 1, "b"), seg("002-00000", 2, "c")]))
    assert titles.apply(tmp_path, True, 2.8) == 2 and titles.apply(tmp_path, True, 2.8) == 2                    # twice gives the same result
    segs = json.loads((tmp_path / "segments.json").read_text())
    assert [g["id"] for g in segs] == ["000-00000", "001-title", "001-00000", "001-00001", "002-title", "002-00000"]
    assert segs[1]["text"] == "Part one. A." and segs[2]["pause_ms"] == 2800 and "pause_ms" not in segs[3] and "pause_ms" not in segs[0]
    assert titles.apply(tmp_path, False, 2.8) == 0
    segs = json.loads((tmp_path / "segments.json").read_text())
    assert [g["id"] for g in segs] == ["000-00000", "001-00000", "001-00001", "002-00000"] and not any("pause_ms" in g for g in segs)


def test_ntfy_send_command_shares_finished_books(tmp_path, monkeypatch):
    jq, work = _queue_env(tmp_path, monkeypatch)
    a = jq.add(str(work), "First Book", out=str(tmp_path / "a.m4b")); b = jq.add(str(work), "Second Tale", out=str(tmp_path / "b.m4b"), send_to="dev")
    (tmp_path / "a.m4b").write_bytes(b"x"); (tmp_path / "b.m4b").write_bytes(b"x")
    jq.update(a["id"], status="done", finished="2026-10-09 11:00"); jq.update(b["id"], status="done", finished="2026-10-09 12:00", send_to="dev")
    sent, said = [], []
    monkeypatch.setattr(jq, "send_file", lambda dev, path, *a, **k: sent.append((dev, path)) or True)
    monkeypatch.setattr(jq, "notify", lambda text, *a, **k: said.append(text) or True)
    monkeypatch.setattr(jq, "_spawn", lambda fn: fn())                                  # run the sending at once
    assert "1. ✓ Second Tale" in jq.answer("done") and "2. ✓ First Book" in jq.answer("done")      # the newest is number 1
    assert "no number 5" in jq.answer("send 5") and "No finished book" in jq.answer("send nothing") and "number" in jq.answer("send")
    assert "No phone" in jq.answer("send 2")                                             # First Book has no device and none is saved
    jq.save_default_device("phone")
    assert "Sending “First Book”" in jq.answer("Send 2!") and sent[-1] == ("phone", str(tmp_path / "a.m4b"))
    assert "Sending “Second Tale”" in jq.answer("send tale") and sent[-1][0] == "dev" and said[-1].startswith("Sent “Second Tale”")
    assert "Sending" in jq.answer("send 1 2") and len(sent) == 4 and jq.load()[0]["sent"]
    monkeypatch.setattr(jq, "send_file", lambda *a, **k: False)
    jq.answer("send 1"); assert "Could not send" in said[-1]
    jq.update(b["id"], status="failed", note="boom")
    assert "no finished file" in jq.answer("send 1")
    assert "send <number or name>" in jq.answer("help")


def test_footnotes_are_called_out_after_their_paragraph():
    from bs4 import BeautifulSoup
    from audiobook_gen import extract
    # a note on the same page, with a link back (Project Gutenberg's usual markup)
    page = BeautifulSoup('<body><p>Constans, who was “seen by the captain,”<a href="x.html#fn1" id="fnref1" class="pginternal"><sup>[1]</sup></a> '
                         'luckily did not reach us.</p><p class="footnote"><a id="fn1"/> <a href="x.html#fnref1" class="pginternal">[1]</a> '
                         '<i>Daily News</i> , March 17, 1887.</p><p>Next paragraph.</p></body>', "lxml")
    assert extract._footnotes(page, {"x.html": page}) == 1
    texts = [p.get_text(" ", strip=True) for p in page.find_all("p")]
    assert texts == ["Constans, who was “seen by the captain,” luckily did not reach us.", "Footnote one. Daily News, March 17, 1887.", "Next paragraph."]
    # the note on another page, its anchor an empty <a id> before the marker, and the leftover "FOOTNOTES:" heading removed
    a = BeautifulSoup('<body><p>The act.<a id="FNanchor_2"/><a class="fnanchor pginternal" href="b.html#Footnote_2">[2]</a></p><p>Then more.</p></body>', "lxml")
    b = BeautifulSoup('<body><div class="footnotes"><h3>FOOTNOTES:</h3><div class="footnote"><p><a id="Footnote_2"/><a href="a.html#FNanchor_2" class="pginternal">'
                      '<span class="label">[2]</span></a> See Appendix B.</p></div></div></body>', "lxml")
    assert extract._footnotes(a, {"a.html": a, "b.html": b}) == 1
    assert [p.get_text(" ", strip=True) for p in a.find_all("p")] == ["The act.", "Footnote two. See Appendix B.", "Then more."]
    assert "FOOTNOTES" not in b.get_text()
    # an ordinary numbered link that is not a note is left alone
    c = BeautifulSoup('<body><p>See <a href="#ch2">[2]</a> below.</p><p id="ch2">Chapter two text.</p></body>', "lxml")
    assert extract._footnotes(c, {}) == 0 and "[2]" in c.get_text()


def _name_env(tmp_path, monkeypatch, n_clips=1):
    """A work folder with a name that has an IPA, clips that say it, and fake engines: the clip's first sample stands in for its seed."""
    import json
    import numpy as np
    import soundfile as sf
    from audiobook_gen import jobqueue, respell_check, synth
    monkeypatch.setattr(jobqueue, "ACTIVITY", tmp_path / "activity.json")                # never touch the real queue's files
    work = tmp_path / "w"; (work / "clips").mkdir(parents=True)
    (work / "lexicon.json").write_text(json.dumps([{"term": "Zorbax", "kind": "name", "ipa": "zɔɹbæks", "source": "user", "count": 5}]))
    meta = {}
    for i in range(n_clips):
        first = np.full(24000, 0.1, np.float32); first[0] = 0.1234
        sf.write(work / "clips" / f"c{i}.wav", first, 24000)
        meta[f"c{i}.wav"] = {"text": "He met Zorbax at the gate.", "engine": "chatterbox", "voice": {"seed": 1234}, "key": f"001-0000{i}:0"}
    (work / "clips_meta.json").write_text(json.dumps(meta))
    heard = {1234: "z ɔ b", 101: "z ɔ b", 202: "z ɔ ɹ b æ", 303: "z ɔ ɹ b æ k s"}      # scores 0.46, 0.46, 0.69, 1.0
    monkeypatch.setattr(respell_check, "phonemes", lambda a, sr, timed=False: (heard.get(int(round(float(a[0]) * 1e4)), "h ɛ l oʊ"), [0.1 * k for k in range(len(heard.get(int(round(float(a[0]) * 1e4)), "h ɛ l oʊ").split()))]))
    made = []

    class Eng:
        sample_rate = 24000
        def synth(self, text, voice):
            made.append(voice["seed"])
            a = np.full(24000, 0.1, np.float32)
            a[0] = voice["seed"] * 1e-4                                       # the first sample says which seed made it
            return a
    monkeypatch.setattr(synth, "get_engine", lambda *a, **k: Eng())
    return work, made, heard


def test_names_below_the_good_score_are_made_again_until_they_match(tmp_path, monkeypatch):
    import soundfile as sf
    from audiobook_gen import namecheck
    work, made, heard = _name_env(tmp_path, monkeypatch)
    stats = namecheck.pass_once(work, {})
    s = stats["Zorbax"]
    assert made == [101, 202, 303]                                           # stops at the first attempt that reaches the good score
    assert s["first"][0] < namecheck.GOOD <= s["final"][0] and len(s["tries"]) == 4 and s["regenerated"] >= 1
    assert abs(float(sf.read(work / "clips" / "c0.wav")[0][0]) - 0.0303) < 1e-4 and (work / "clips_before_name_check" / "c0.wav").exists()   # the best take stays in place
    made.clear()
    assert namecheck.pass_once(work, {})["Zorbax"]["first"][0] >= namecheck.GOOD and made == []        # a name that is right is left alone
    # a name without an IPA from a real source is not judged at all
    import json
    (work / "lexicon.json").write_text(json.dumps([{"term": "Zorbax", "kind": "name", "ipa": "", "source": "auto", "count": 5}]))
    assert namecheck.pass_once(work, {}) == {}


def test_the_worse_a_name_sounds_the_sooner_a_respelling_is_due():
    from audiobook_gen.namecheck import respell_due as due
    assert due([0.25]) and not due([0.35]) and due([0.35, 0.38]) and not due([0.45, 0.44]) and due([0.45, 0.44, 0.46])
    assert not due([0.55] * 3) and due([0.55] * 4) and not due([0.65] * 4) and due([0.65] * 5) and not due([0.8] * 9)
    assert due([0.9, 0.2]) and not due([0.55] * 4, good=0.5)                   # bands above the good score do not count


def test_a_name_that_keeps_failing_gets_a_checked_respelling(tmp_path, monkeypatch):
    import json
    from audiobook_gen import namecheck, respell_check, respell_search, synth
    from audiobook_gen.lexicon import Preprocessor
    work, made, heard = _name_env(tmp_path, monkeypatch, n_clips=3)
    heard.update({1234: "z ɔ b", 101: "z ɔ b", 202: "z ɔ b", 303: "z ɔ b"})        # 0.46 every time: three takes under 50% make it due
    monkeypatch.setattr(respell_search, "candidates", lambda ipa, word, limit=30, seed=0: ["zorbax", "zawrbaks"])
    monkeypatch.setattr(respell_search, "free_ear", lambda: None)
    monkeypatch.setattr(respell_check, "check", lambda items, voice, engine, progress=print, seed=1: [
        {"name": "zorbax", "score": 0.1}, {"name": "zawrbaks", "score": 0.8}])
    again = []
    monkeypatch.setattr(synth, "synthesize", lambda work, cfg, only=None: again.append(1))
    report = namecheck.run(work, {})
    assert made == [101, 202]                                                  # the first take and two retries, then the respelling; the other clips wait
    lex = json.loads((work / "lexicon.json").read_text())[0]
    assert lex["respell"] == "zawrbaks" and lex["respell_src"] == "checked" and lex["respell_score"] == 0.8 and again
    assert report["respelled"] == {"Zorbax": "zawrbaks"} and (work / "name_report.json").exists()
    assert Preprocessor([lex], "verified").substitute("He met Zorbax.") == "He met Zawrbaks."        # Chatterbox uses it
    assert Preprocessor([lex], "typed").substitute("He met Zorbax.") == "He met Zorbax."
    # a name scoring under 30% gets the respelling at once, with no retries at all
    work2, made2, heard2 = _name_env(tmp_path / "b", monkeypatch)
    heard2.update({1234: "h ɛ l oʊ"})
    namecheck.run(work2, {})
    assert made2 == [] and json.loads((work2 / "lexicon.json").read_text())[0]["respell"] == "zawrbaks"
    # a respelling you typed yourself is never replaced, and one that did not help is not tried twice
    lex.update(respell="zor-bax", respell_src="user", source="user")
    (work / "lexicon.json").write_text(json.dumps([lex]))
    assert namecheck.run(work, {})["respelled"] == {}
    work3, made3, heard3 = _name_env(tmp_path / "c", monkeypatch)
    heard3.update({1234: "h ɛ l oʊ"})
    monkeypatch.setattr(respell_check, "check", lambda items, voice, engine, progress=print, seed=1: [{"name": "zorbax", "score": 0.1}, {"name": "zawrbaks", "score": 0.15}])
    assert namecheck.run(work3, {})["respelled"] == {} and "respell_tried" in json.loads((work3 / "lexicon.json").read_text())[0]


def test_a_retry_with_a_runaway_delivery_is_never_swapped_in(tmp_path, monkeypatch):
    import numpy as np
    from audiobook_gen import namecheck, synth
    work, made, heard = _name_env(tmp_path, monkeypatch)

    class Loud:                                    # sounds perfect to the recogniser but is three times as loud and twice as long
        sample_rate = 24000
        def synth(self, text, voice):
            a = np.full(48000, 0.3, np.float32)
            a[0] = 0.0303
            return a
    monkeypatch.setattr(synth, "get_engine", lambda *a, **k: Loud())
    s = namecheck.pass_once(work, {})["Zorbax"]
    assert s["final"][0] < namecheck.GOOD and s["regenerated"] == 0


def test_a_name_with_no_plain_candidates_does_not_stop_the_pass(tmp_path, monkeypatch):
    import json
    from audiobook_gen import namecheck, respell_search
    work, made, heard = _name_env(tmp_path, monkeypatch)
    heard[1234] = "h ɛ l oʊ"
    monkeypatch.setattr(respell_search, "candidates", lambda ipa, word, limit=30, seed=0: [])
    monkeypatch.setattr(respell_search, "free_ear", lambda: None)
    report = namecheck.run(work, {})
    assert report["respelled"] == {} and "respell_tried" in json.loads((work / "lexicon.json").read_text())[0]


def test_a_hyphenated_name_is_spelled_part_by_part(monkeypatch):
    from audiobook_gen import respell_search as rs
    phones = ["b", "iː", "ɹ", "ʃ", "iː", "b", "ə"]                          # Beer-sheba
    monkeypatch.setattr(rs, "segment", lambda ipa: phones)
    assert rs.split_phones(phones, ["beer", "sheba"]) == [["b", "iː", "ɹ"], ["ʃ", "iː", "b", "ə"]]   # cut at the start of the second syllable
    c = rs.candidates("biːɹʃibə", "Beer-sheba", 40)
    assert c[0] == "beer-sheba" and any("-" in x for x in c) and any("-" not in x for x in c) and len(c) > 4
    assert all(x.replace("-", "").isalpha() and x.islower() for x in c) and len(set(c)) == len(c)
    assert "beersheba" in c or any(x.startswith("bee") and "sh" in x for x in c)
    assert rs.candidates("biːɹ", "Beer", 10)[0] == "beer"                       # an ordinary name is unchanged


def test_names_with_an_apostrophe_are_found_and_spelled_part_by_part(monkeypatch):
    from audiobook_gen import lexicon, respell_search as rs
    assert not lexicon._is_known("M’ling") and not lexicon._is_known("M'ling")          # a main character was being skipped as "known"
    assert lexicon._is_known("don't") and lexicon._is_known("o'clock") and lexicon._is_known("goin'")
    monkeypatch.setattr(rs, "segment", lambda ipa: ["m", "ə", "l", "ɪ", "ŋ"])
    c = rs.candidates("məlɪŋ", "M’ling", 30)
    assert c[0] == "m-ling" and any("-" in x for x in c) and any(x == "muhling" or x.startswith("m") and "-" not in x for x in c)
    assert all(x.replace("-", "").isalpha() for x in c)


def test_dialect_words_are_found_and_given_a_sound(tmp_path, monkeypatch):
    import json
    from audiobook_gen import dialect, namecheck
    work = tmp_path / "w"; work.mkdir()
    text = ("Mars' Epps said the days' work was done. He was goin' to the mill, an' they asked 'em what 'many' meant. "
            "Don't you know it's six o'clock? M’ling came. 'Yes, I do,' she said. The cap'n was bilin' mad, jes' like Moxie'd said. "
            "That mornin' he shouted fo' the zzz'n.")
    (work / "chapters.json").write_text(json.dumps({"chapters": [{"index": 1, "title": "x", "text": text}]}))
    found = {w["token"]: w for w in dialect.scan(work)}
    assert {"goin'", "an'", "'em", "cap'n", "bilin'", "jes'", "fo'", "zzz'n"} <= set(found)
    assert not any(t in found for t in ("mars'", "days'", "don't", "o'clock", "'many'", "m'ling", "moxie'd", "'yes"))     # possessives, contractions, names, quotes
    assert found["goin'"]["ipa"] and found["goin'"]["ipa_src"] == "dialect-dict:claude" and found["zzz'n"]["ipa"] == ""
    (work / "dialect_ipa.txt").write_text("zzz'n|zɪzən\n")                                                         # a reading for this book only
    monkeypatch.setattr(dialect, "USER", tmp_path / "user.txt")
    (tmp_path / "user.txt").write_text("goin'|ɡoʊən\n")                                                            # yours wins over the shipped list
    assert dialect.scan(work)[0]["ipa_src"].startswith("dialect-dict")
    lex = [{"term": "Moxie", "kind": "name", "source": "auto", "ipa": ""}]
    (work / "lexicon.json").write_text(json.dumps(lex))
    r = dialect.apply(work)
    out = {e["term"]: e for e in json.loads((work / "lexicon.json").read_text())}
    assert out["goin'"]["kind"] == "dialect" and out["goin'"]["ipa"] == "ɡoʊən" and out["goin'"]["ipa_src"] == "dialect-dict:user"
    assert out["zzz'n"]["ipa_src"] == "dialect-dict:book" and "Moxie" in out and r["added"] >= 7
    # the name check judges them like any name, and finds them in the spoken text
    assert "dialect" in namecheck.KINDS and namecheck.pattern("goin'").search("He was goin' to town.") and not namecheck.pattern("goin'").search("goings")


def test_the_name_check_renews_its_gpu_lease(tmp_path, monkeypatch):
    import os, time
    from audiobook_gen import jobqueue as jq, namecheck
    monkeypatch.setattr(jq, "GUI_LOCK", tmp_path / "lock")
    jq.GUI_LOCK.write_text(f"{os.getpid()} the name check")
    os.utime(jq.GUI_LOCK, (time.time() - 3000, time.time() - 3000))                  # older than the half hour after which the queue forgets it
    assert jq.lease_holder() == ""
    jq.GUI_LOCK.write_text(f"{os.getpid()} the name check")
    os.utime(jq.GUI_LOCK, (time.time() - 1700, time.time() - 1700))
    namecheck.keep_gpu()
    assert time.time() - jq.GUI_LOCK.stat().st_mtime < 5 and jq.lease_holder() == "the name check"


def test_respellings_are_tried_in_rounds_of_five_ten_and_fifteen(tmp_path, monkeypatch):
    from audiobook_gen import namecheck, respell_check, respell_search
    work, made, heard = _name_env(tmp_path, monkeypatch)
    monkeypatch.setattr(respell_search, "candidates", lambda ipa, word, limit=30, seed=0: [f"zorb{i}" for i in range(limit)])
    monkeypatch.setattr(respell_search, "free_ear", lambda: None)
    entry = {"term": "Zorbax", "ipa": "zɔɹbæks"}
    asked, win = [], {}
    def fake(items, voice, engine, progress=print, seed=1):
        asked.append(len(items))
        return [{"name": n, "score": win.get(n, 0.2)} for n, *_ in items]
    monkeypatch.setattr(respell_check, "check", fake)
    r = namecheck.try_respell(work, {}, entry)                                  # nothing is good enough: 1 baseline + 5, 10, 15
    assert asked == [1, 5, 10, 15] and r["tried"] == 30 and r["score"] == 0.2
    asked.clear(); win["zorb2"] = 0.75                                         # 75% is not the 80% aimed for: the search goes on
    assert namecheck.try_respell(work, {}, entry)["tried"] == 30 and asked == [1, 5, 10, 15]
    asked.clear(); win["zorb2"] = 0.9                                          # 90% in the first five: stop there
    r = namecheck.try_respell(work, {}, entry)
    assert asked == [1, 5] and r["spelling"] == "zorb2" and r["tried"] == 5
    asked.clear(); win.clear(); win["zorb7"] = 0.9                             # found in the second round
    r = namecheck.try_respell(work, {}, entry)
    assert asked == [1, 5, 10] and r["spelling"] == "zorb7"
    asked.clear()
    assert namecheck.try_respell(work, {}, entry, batches=(2, 3))["tried"] == 5 and asked == [1, 2, 3]       # the sizes can be set


def test_a_book_cannot_be_generated_until_it_has_been_prepared(tmp_path, monkeypatch):
    import json
    import pandas as pd
    import pytest
    from audiobook_gen import gui, jobqueue as jq, prep
    jq, work = _queue_env(tmp_path, monkeypatch)
    (work / "prep.json").unlink()                                                 # not prepared yet
    (work / "chapters.json").write_text(json.dumps({"chapters": [{"index": 1, "title": "One", "text": "Hello."}]}))
    monkeypatch.setattr(jq, "ensure_supervisor", lambda: False)
    chap = pd.DataFrame([[True, 1, "One", 6]], columns=gui.CHAP_COLS)
    args = (str(work), chap, "B", "A", "", 60, 350, 700, 250, 4, True, True, 0.0, list(gui.LEX_MODES)[0], 140, 120, False)
    with pytest.raises(Exception, match="not been prepared"):
        gui.generate_queued(*args)
    assert jq.load() == [] and not prep.status(work)["ready"] and "not prepared" in prep.describe(work)
    ticked, status, note, go = gui.prep_state(str(work))
    assert go["interactive"] is False and "Prepare" in note and set(ticked["value"]) >= {"segments", "lexicon"}
    prep.mark_done(work, ["segments"])                                            # what the Prepare button records
    gui.generate_queued(*args)
    assert len(jq.load()) == 1 and "prepared" in prep.describe(work) and gui.prep_state(str(work))[3]["interactive"] is True
    (work / "chapters.json").write_text(json.dumps({"chapters": [{"index": 1, "title": "Changed", "text": "Hello."}]}))
    assert not prep.status(work)["ready"] and "chapters changed" in prep.status(work)["why"]       # new chapters, so prepare again
    with pytest.raises(Exception, match="chapters changed"):
        gui.generate_queued(*args)
    (work / "prep.json").unlink()                                                 # a book the command-line tool prepared before this existed
    (work / "prepared.json").write_text("{}")
    assert prep.status(work)["ready"] and prep.status(work)["legacy"]


def test_the_steps_are_listed_by_importance_and_the_ai_ones_are_marked(monkeypatch):
    from audiobook_gen import prep
    ids = [s["id"] for s in prep.STEPS]
    assert ids[0] == "segments" and prep.BY_ID["segments"]["required"] and set(prep.default_selected()) >= {"segments", "lexicon", "ipa", "language"}
    labels = dict((v, l) for l, v in prep.choices())
    assert prep.AI in labels["language"] and prep.AI in labels["speaker_ai"]
    assert all(prep.NO_AI in labels[i] for i in ("segments", "lexicon", "ipa", "dialect"))
    assert "speaker_ai" not in prep.default_selected()


def test_prepare_runs_only_the_ticked_steps_in_order_and_asks_for_the_gpu_only_for_ai(tmp_path, monkeypatch):
    import json
    from audiobook_gen import dialect, jobqueue, lexicon, prep, pronounce, refine, speakers
    monkeypatch.setattr(jobqueue, "ACTIVITY", tmp_path / "activity.json")
    work = tmp_path / "w"; work.mkdir()
    (work / "chapters.json").write_text(json.dumps({"title": "T", "author": "A", "chapters": [{"index": 1, "title": "One", "text": "Hello."}]}))
    (work / "lexicon.json").write_text("[]")
    calls, lease = [], []
    def fake_parse(*a, **k):
        calls.append("segments")
        seg = [{"id": "001-00000", "speaker": "Narrator", "kind": "narration", "text": "Hello.", "chapter": 1}]
        (work / "segments.json").write_text(json.dumps(seg))
        return seg
    monkeypatch.setattr(speakers, "parse_book", fake_parse)
    monkeypatch.setattr(lexicon, "build_lexicon", lambda *a, **k: calls.append("lexicon") or [])
    monkeypatch.setattr(pronounce, "fill_lexicon", lambda *a, **k: calls.append("ipa") or {"found": 0, "asked": 0})
    monkeypatch.setattr(dialect, "apply", lambda w: calls.append("dialect") or {"words": 0, "added": 0, "missing": 0})
    monkeypatch.setattr(refine, "run", lambda *a, **k: calls.append("language") or {"fixes": []})
    monkeypatch.setattr(jobqueue, "lease_take", lambda *a, **k: lease.append("take"))
    monkeypatch.setattr(jobqueue, "lease_give_back", lambda: lease.append("back"))
    r = prep.run(work, ["dialect", "lexicon"])                                    # no AI step: no lease
    assert calls == ["segments", "lexicon", "dialect"] and lease == [] and r["ran"] == ["segments", "lexicon", "dialect"]
    assert prep.status(work)["ready"] and json.loads((work / "prep.json").read_text())["steps"] == ["segments", "lexicon", "dialect"]
    calls.clear()
    prep.run(work, ["language", "ipa", "lexicon", "segments", "dialect"])
    assert calls == ["segments", "lexicon", "ipa", "dialect", "language"] and lease == ["take", "back"]      # the same order whatever the ticking order
    calls.clear(); lease.clear()
    prep.run(work, [])                                                            # the required step is done even when nothing is ticked
    assert calls == ["segments"]


def test_current_status_names_whatever_else_holds_the_gpu(tmp_path, monkeypatch):
    import os
    from datetime import datetime
    jq, work = _queue_env(tmp_path, monkeypatch)
    now = datetime.strptime("2026-10-09 12:00", jq.FMT)
    assert jq.answer("current", now) == "Nothing is being made and nothing is waiting."                 # nobody has the GPU
    a = jq.add(str(work), "Acts", out=str(tmp_path / "a.m4b"))
    jq.update(a["id"], note="waiting: the name check (you are using the GPU)")
    jq.activity_set("the name check", "nasb_genesis", "judging clip 301 of 793")                         # a respelling pass, no book running
    jq.GUI_LOCK.write_text(f"{os.getpid()} the name check")
    text = jq.answer("current", now)
    assert "Nothing is being made right now. Next: “Acts”" in text and "waiting: waiting: the name check" not in text
    assert "🔒 The name check on nasb_genesis is using the GPU — judging clip 301 of 793 (since " in text
    jq.activity_detail("trying respellings for Joktan")
    assert "trying respellings for Joktan" in jq.answer("current", now)
    jq.update(a["id"], status="running", progress="50/100", note="making the speech")
    both = jq.answer("current", now)
    assert "Making “Acts”" in both and "🔒 The name check" in both                                       # alongside a running book too
    jq.lease_give_back()
    assert "🔒" not in jq.answer("current", now) and jq.activity_get() == {}                              # given back: the line goes away
    jq.activity_set("text preparation", "w", "step 2 of 4")
    assert "Text preparation on w is using the GPU — step 2 of 4" in jq.answer("current", now)
    jq.activity_clear()
    jq.ACTIVITY.write_text('{"pid": 999999999, "what": "gone"}')                                         # a process that died leaves nothing behind
    assert jq.activity_get() == {} and "🔒" not in jq.answer("current", now)


def test_your_suggested_spellings_are_tried_in_the_first_round(tmp_path, monkeypatch):
    from audiobook_gen import namecheck, respell_check, respell_search
    work, made, heard = _name_env(tmp_path, monkeypatch)
    (work / "respell_hints.txt").write_text("Zorbax|Zawrbaks\n# a comment\nOther|nope\n")
    monkeypatch.setattr(respell_search, "candidates", lambda ipa, word, limit=30, seed=0: [f"zorb{i}" for i in range(limit)])
    monkeypatch.setattr(respell_search, "free_ear", lambda: None)
    asked = []
    def fake(items, voice, engine, progress=print, seed=1):
        asked.append([i[1] for i in items])
        return [{"name": n, "score": 0.9 if n == "zawrbaks" else 0.2} for n, *_ in items]
    monkeypatch.setattr(respell_check, "check", fake)
    r = namecheck.try_respell(work, {}, {"term": "Zorbax", "ipa": "zɔɹbæks"})
    assert namecheck.hints("Zorbax", work) == ["Zawrbaks"] and namecheck.hints("Achaia") == ["Uhkiah"]       # the shipped list has Achaia
    assert asked[1][:1] == ["Zawrbaks"] and len(asked[1]) == 6 and r["spelling"] == "zawrbaks" and len(asked) == 2     # yours first, with the five generated; 90% ends it


def test_respell_suggestions_by_ntfy_are_saved_and_tested(tmp_path, monkeypatch):
    import json
    from audiobook_gen import namecheck, respell_check, synth
    (tmp_path / "q").mkdir()
    jq, _ = _queue_env(tmp_path / "q", monkeypatch)
    monkeypatch.setattr(namecheck, "HINTS", tmp_path / "hints.txt")
    monkeypatch.setattr(namecheck, "ROOT", tmp_path)                                  # never touch the real hint list or the real books
    (tmp_path / "work" / "b").mkdir(parents=True)
    (tmp_path / "work" / "b" / "lexicon.json").write_text(json.dumps([{"term": "Zorbax", "respell_tried": {"none": True}}]))
    assert jq.parse_respell("respell Zorbax zawrbaks") == [("Zorbax", "zawrbaks")] and jq.parse_respell("just a message") == [] and jq.parse_respell("Zorbax zawrbaks") == []
    reply = jq.answer("Zorbax|Zawrbaks")
    assert "Saved 1 respell suggestion" in reply and "Zorbax → Zawrbaks" in reply
    assert namecheck.hints("zorbax") == ["Zawrbaks"] and "respell_tried" not in json.loads((tmp_path / "work" / "b" / "lexicon.json").read_text())[0]
    assert "Already had" in jq.answer("respell zorbax zawrbaks") and namecheck.hints("Zorbax") == ["Zawrbaks"]         # not added twice
    assert "respell <name> <spelling>" in jq.answer("help")
    # the name check speaks the suggestion, compares it with the spelling in use, and adopts it when it is clearly better
    work, made, heard = _name_env(tmp_path / "n", monkeypatch)
    monkeypatch.setattr(namecheck, "HINTS", tmp_path / "hints.txt")
    scores = {"zorbax": 0.4, "zawrbaks": 0.85}
    monkeypatch.setattr(respell_check, "check", lambda items, voice, engine, progress=print, seed=1: [{"name": n, "score": scores[n]} for n, *_ in items])
    monkeypatch.setattr(synth, "synthesize", lambda *a, **k: None)
    assert namecheck.test_hints(work, {}) == {"Zorbax": "zawrbaks"}
    lex = json.loads((work / "lexicon.json").read_text())[0]
    assert lex["respell"] == "zawrbaks" and lex["respell_src"] == "checked" and lex["respell_hint"] and lex["hints_tested"] == ["Zawrbaks"]
    assert namecheck.test_hints(work, {}) == {}                                          # tested once, not again
    scores.update(zawrbaks=0.45); lex["hints_tested"] = []; lex["respell_src"] = ""; lex["respell"] = ""
    (work / "lexicon.json").write_text(json.dumps([lex]))
    assert namecheck.test_hints(work, {}) == {}                                          # only slightly better: the spelling in use stays


def test_a_guessed_foreign_ipa_is_never_what_a_name_is_judged_by(tmp_path, monkeypatch):
    import json
    from audiobook_gen import namecheck, pronounce
    work = tmp_path / "w"; work.mkdir()
    (work / "lexicon.json").write_text(json.dumps([
        {"term": "Derbe", "kind": "name", "lang": "de", "ipa": "dˈɛɾbə", "source": "auto"},                  # a German guess with no source
        {"term": "Paulus", "kind": "name", "lang": "nl", "ipa": "pˈʌʊls", "source": "auto", "ipa_src": ""},
        {"term": "Kept", "kind": "name", "lang": "", "ipa": "kɛpt", "source": "auto"},                         # no language guess: left as it was
        {"term": "Typed", "kind": "name", "lang": "de", "ipa": "taɪpt", "source": "user"}]))                   # typed by you: always trusted
    monkeypatch.setattr(pronounce, "lookup", lambda terms, *a, **k: {"Derbe": {"ipa": "ˈdɜɹbi", "source": "bible-dict:claude"}} if "Derbe" in terms else {})
    got = {e["term"]: e for e in namecheck.targets(work)}
    assert got["Derbe"]["ipa"] == "ˈdɜɹbi" and got["Derbe"]["ipa_src"] == "bible-dict:claude"             # the dictionary's reading replaces the guess
    assert "Paulus" not in got                                                                              # nothing better is known: not judged
    assert got["Kept"]["ipa"] == "kɛpt" and got["Typed"]["ipa"] == "taɪpt"


def test_names_known_to_be_good_are_left_out_of_the_name_check(tmp_path, monkeypatch):
    import json
    from audiobook_gen import namecheck
    jq, _ = _queue_env((tmp_path / "q").__class__(tmp_path / "q") if (tmp_path / "q").mkdir() is None else tmp_path / "q", monkeypatch)
    monkeypatch.setattr(namecheck, "ROOT", tmp_path); monkeypatch.setattr(jq, "ROOT", tmp_path)
    work, made, heard = _name_env(tmp_path / "n", monkeypatch)
    heard.update({1234: "z ɔ ɹ b æ k s"})                                          # the name sounds right (1.0)
    assert namecheck.pass_once(work, {})["Zorbax"]["first"] == [1.0]
    rep = namecheck.run(work, {})                                                   # a good name is recorded as good...
    ok = json.loads((work / "lexicon.json").read_text())[0]["name_ok"]
    assert ok["fp"] and not ok["manual"] and ok["score"] == 1.0
    skipped = []
    assert namecheck.pass_once(work, {}, skipped=skipped) == {} and skipped == ["Zorbax"]          # ...and left out next time
    rep = namecheck.run(work, {})
    assert rep["left_out_known_good"] == ["Zorbax"] and rep["names"] == 0
    assert namecheck.pass_once(work, {}, skip_known=False)["Zorbax"]["clips"] == 1                    # unless asked to look again
    # its clips changed (a new clip says the name): judged again
    import soundfile as sf, numpy as np
    meta = json.loads((work / "clips_meta.json").read_text())
    meta["c9.wav"] = {**meta["c0.wav"], "key": "001-00009:0"}
    sf.write(work / "clips" / "c9.wav", np.full(24000, 0.1, np.float32), 24000)
    (work / "clips_meta.json").write_text(json.dumps(meta))
    assert namecheck.pass_once(work, {})["Zorbax"]["clips"] == 2
    # `good <name>` on ntfy: left out for good, whatever its clips do
    (tmp_path / "work" / "b").mkdir(parents=True)
    (tmp_path / "work" / "b" / "lexicon.json").write_text(json.dumps([{"term": "Derbe"}]))
    assert jq.parse_good("good Derbe") == "Derbe" and jq.parse_good("known good derbe") == "Derbe" and jq.parse_good("good morning") == ""
    assert "Marked Derbe as known good in 1 book" in jq.answer("good Derbe") and json.loads((tmp_path / "work" / "b" / "lexicon.json").read_text())[0]["name_ok"]["manual"]
    assert namecheck.known_good({"name_ok": {"manual": True, "fp": "x"}}, "different") and not namecheck.known_good({}, "x")
    assert "good <name>" in jq.answer("help")


def test_book_text_is_cleaned_of_split_names_capitals_and_file_notes(tmp_path):
    import json
    from audiobook_gen import textclean as tc, titles
    names = {"douglass"}
    assert tc.fix_text("Mr. D OUGLASS and P ATRICK H ENRY , of old , said JOHN went.", names) == "Mr. Douglass and Patrick Henry, of old, said John went."
    assert tc.fix_text("B OSTON , April 22, 1845.", names) == "Boston, April 22, 1845." and tc.fix_text("A PARODY", names) == "A Parody"
    assert tc.fix_text("LETTER FROM WENDELL PHILLIPS, ESQ.", names) == "Letter from Wendell Phillips, Esquire." and tc.fix_text("CHAPTER VII", names) == "Chapter VII"
    assert tc.fix_text("He said “NO COMPROMISE WITH SLAVERY!” and left.", names) == "He said “No compromise with slavery!” and left."
    assert tc.fix_text("The NASA team and the USA, I AM sure.", names) == "The NASA team and the USA, I AM sure."          # acronyms are left alone
    assert tc.fix_text("Institution !” and it is .", names) == "Institution!” and it is."
    assert tc.drop_cruft("Real text.\n\nNote from the original file: This electronic book is being\nreleased [Born 1929]\n\nContents\n\nMore.") == "Real text.\n\nMore."
    # a work folder made before this existed: the title page goes, names and punctuation are put right, the title is not read twice
    work = tmp_path / "w"; work.mkdir()
    chapters = [{"index": 1, "title": "Narrative of the Life of FREDERICK DOUGLASS", "text": "Narrative of the Life of FREDERICK DOUGLASS\\n\\nNote from the original file: x\\n\\nContents".replace("\\\\n", "\\n")},
                {"index": 2, "title": "PREFACE", "text": "PREFACE\n\nI met Frederick Douglass , of D OUGLASS fame."}]
    (work / "chapters.json").write_text(json.dumps({"title": "T", "chapters": chapters}))
    seg = lambda i, ch, t: {"speaker": "Narrator", "kind": "narration", "text": t, "para_start": True, "chapter": ch, "id": i}
    (work / "segments.json").write_text(json.dumps([seg("001-00000", 1, "Narrative of the Life of FREDERICK DOUGLASS"), seg("001-00001", 1, "Note from the original file: x"),
                                                    seg("001-00002", 1, "Contents"), seg("002-00000", 2, "PREFACE"), seg("002-00001", 2, "I met Frederick Douglass , of D OUGLASS fame.")]))
    r = tc.clean_work(work)
    assert r["dropped_chapters"] == [1] and r["dropped_segments"] == 2
    segs = json.loads((work / "segments.json").read_text())
    assert [g["text"] for g in segs] == ["Preface", "I met Frederick Douglass, of Douglass fame."]
    assert titles.apply(work, True) == 1
    segs = json.loads((work / "segments.json").read_text())
    assert [g["text"] for g in segs] == ["Preface.", "I met Frederick Douglass, of Douglass fame."] and segs[1]["pause_ms"] == 2800   # read once, with the pause after it


def test_done_lists_the_newest_book_first(tmp_path, monkeypatch):
    jq, work = _queue_env(tmp_path, monkeypatch)
    for title, when in (("Old", "2026-10-09 08:00"), ("Newest", "2026-10-10 20:00"), ("Middle", "2026-10-10 09:00")):
        j = jq.add(str(work), title, out=str(tmp_path / f"{title}.m4b"))
        jq.update(j["id"], status="done", finished=when)
    lines = jq.answer("done").splitlines()
    assert [l.split(" — ")[0] for l in lines] == ["1. ✓ Newest", "2. ✓ Middle", "3. ✓ Old"]               # by when they finished, not the queue order
    assert [j["title"] for j in jq.finished_list()] == ["Newest", "Middle", "Old"]


def test_a_rebuilt_book_appears_once_in_the_done_list(tmp_path, monkeypatch):
    jq, work = _queue_env(tmp_path, monkeypatch)
    for title, out, status, when in (("Book", "b.m4b", "done", "2026-10-09 08:00"), ("Book", "b.m4b", "done", "2026-10-10 08:00"),
                                     ("Book", "b.m4b", "failed", "2026-10-11 08:00"), ("Other", "o.m4b", "done", "2026-10-09 12:00")):
        j = jq.add(str(work), title, out=str(tmp_path / out))
        jq.update(j["id"], status=status, finished=when)
    got = jq.finished_list()
    assert [(j["title"], j["finished"]) for j in got] == [("Book", "2026-10-10 08:00"), ("Other", "2026-10-09 12:00")]       # the failed rebuild does not hide the finished book


def test_epub_from_a_work_folder(tmp_path):
    import json, zipfile
    from audiobook_gen.epubout import build
    (tmp_path / "chapters.json").write_text(json.dumps({"title": "T & Co", "author": "", "chapters": [
        {"index": 0, "title": "Opening", "text": "Hello.", "opening": True}, {"index": 1, "title": "Part 1", "text": "One.\n\nTwo <b>."}]}))
    out = build(tmp_path, tmp_path / "o.epub")
    z = zipfile.ZipFile(out)
    assert z.namelist()[0] == "mimetype" and z.testzip() is None
    assert "Two &lt;b&gt;." in z.read("OEBPS/c002.xhtml").decode() and "<h1>Part 1</h1>" in z.read("OEBPS/c002.xhtml").decode()


def test_titles_in_capitals_become_title_case_and_esq_is_spelled_out():
    from audiobook_gen.textclean import title_case, esquire
    assert title_case("LETTER FROM WENDELL PHILLIPS, ESQ.") == "Letter from Wendell Phillips, Esquire."
    assert title_case("CHAPTER VII") == "Chapter VII" and title_case("Part 3: The First Disciples") == "Part 3: The First Disciples"
    assert esquire("Phillips, Esq., said. Smith, Esq. He went.") == "Phillips, Esquire, said. Smith, Esquire. He went."


def test_a_bridge_can_lead_into_the_title(tmp_path):
    import numpy as np
    from audiobook_gen import music
    sr = 8000
    wav = tmp_path / "t.wav"
    import soundfile as sf
    sf.write(wav, (np.sin(np.arange(40 * sr) * 0.05) * 0.3).astype(np.float32), sr)
    music.LIB = tmp_path
    music.INDEX = tmp_path / "index.json"
    music._save([{"id": "t", "file": "t.wav", "seconds": 40.0, "credit": "x", "title": "t"}])
    audio = np.concatenate([np.zeros(2 * sr), np.ones(20 * sr, np.float32) * 0.2])
    spec = {**music.DEFAULT_PLAN["bridge"], "track": "t", "seconds": 30.0, "title_before": 10.0}
    out = music.bridge_under_title(audio, spec, sr, 0.2, -4)
    assert abs(len(out) / sr - (3 + 20 + 20)) < 0.01                      # 3 s lead, 20 s of music alone, then the 20 s of speech
    assert np.abs(out[:3 * sr]).max() == 0 and np.abs(out[5 * sr:20 * sr]).max() > 0


def test_short_first_names_are_spelled_out_before_synthesis():
    from audiobook_gen.lexicon import normalize
    assert normalize("Wm. Lloyd Garrison Boston,") == "William Lloyd Garrison Boston,"
    assert normalize("Geo. Washington and Thos. Jefferson") == "George Washington and Thomas Jefferson"


def test_titles_ranks_and_states_are_spelled_out_before_synthesis():
    from audiobook_gen.lexicon import normalize
    assert normalize("the Hon. B. K. Bruce and the Rev. Daniel Weeden") == "the Honorable B. K. Bruce and the Reverend Daniel Weeden"
    assert normalize("U. S. GRANT, Lieut.-General. At Memphis, Tenn., and Malden, W. Va. He") == "U. S. GRANT, Lieutenant General. At Memphis, Tennessee, and Malden, West Virginia He"
    assert normalize("John Melbourne, Esq., and Smith, Esq. He, etc., came, i.e., late.") == "John Melbourne, Esquire, and Smith, Esquire. He, et cetera, came, that is, late."
    assert normalize("the mass. No. 5 in Massachusetts.") == "the mass. No. 5 in Massachusetts."


def test_the_clip_count_is_full_once_the_speech_stage_is_over(tmp_path, monkeypatch):
    jq, work = _queue_env(tmp_path, monkeypatch)
    j = jq.add(str(work), "Book", out=str(tmp_path / "b.m4b"))
    jq.update(j["id"], status="running", progress="0/997")
    jq.Runner._speech_complete(None, j)
    assert next(x for x in jq.load() if x["id"] == j["id"])["progress"] == "997/997"
    jq.update(j["id"], progress="")
    jq.Runner._speech_complete(None, j)                                      # nothing to complete: left alone
    assert next(x for x in jq.load() if x["id"] == j["id"])["progress"] == ""


def test_the_quality_check_note_says_how_many_clips(tmp_path, monkeypatch):
    import json
    jq, work = _queue_env(tmp_path, monkeypatch)
    (work / "clips_meta.json").write_text(json.dumps({f"c{i}.wav": {} for i in range(1234)}))
    assert jq.Runner._clip_count(None, {"work": str(work)}) == " (1,234 clips)"
    assert jq.Runner._clip_count(None, {"work": str(tmp_path / "nowhere")}) == ""


def test_reopened_book_shows_its_saved_generate_settings(tmp_path):
    import yaml
    from audiobook_gen import gui
    p = _assistant_project(tmp_path)
    cfg = yaml.safe_load((p / "config.yaml").read_text())
    cfg.update(crossfade_ms=90, f5_precision="float32", pacing_ms={"sentence": 400, "paragraph": 900, "speaker_change": 300},
               workers={"kokoro": 2, "chatterbox": 1})
    (p / "config.yaml").write_text(yaml.safe_dump(cfg))
    g = gui.gen_settings(str(p))
    assert len(g) == 12 and g[6:] == (90, 400, 900, 300, 2, False)


def test_saving_a_stale_lexicon_table_keeps_what_the_name_check_wrote(tmp_path):
    import json, pandas as pd
    from audiobook_gen import gui
    p = _assistant_project(tmp_path)
    snap = gui.lex_snapshot(str(p))
    table = pd.DataFrame([{"term": "Weena", "ipa": "", "respell": "", "lang": ""}])      # loaded while Weena had no respelling
    lex = json.loads((p / "lexicon.json").read_text()); lex[0]["respell"] = "wee-nah"; lex[0]["respell_src"] = "namecheck"
    (p / "lexicon.json").write_text(json.dumps(lex))                           # the name check finishes meanwhile
    gui.save_lexicon(str(p), table, snap)
    e = json.loads((p / "lexicon.json").read_text())[0]
    assert e["respell"] == "wee-nah" and e.get("respell_src") == "namecheck"
    table.loc[0, "ipa"] = "wiːnə"                                              # a real edit still saves
    gui.save_lexicon(str(p), table, snap)
    e = json.loads((p / "lexicon.json").read_text())[0]
    assert e["ipa"] == "wiːnə" and e["respell"] == "wee-nah" and e["source"] == "user"


def test_a_book_is_queued_once_and_rebuild_keeps_its_settings(tmp_path, monkeypatch):
    from audiobook_gen import gui, jobqueue as jq
    monkeypatch.setattr(jq, "QUEUE", tmp_path / "queue"); monkeypatch.setattr(jq, "JOBS", tmp_path / "queue" / "jobs.json")
    monkeypatch.setattr(jq, "ensure_supervisor", lambda: False)
    p = _assistant_project(tmp_path)
    a, upd = gui._enqueue(p, "T", "A", "", [1], "", "", "")
    b, upd2 = gui._enqueue(p, "T2", "A", "", None, "", "23:00-06:30", "")
    assert not upd and upd2 and a["id"] == b["id"] and len(jq.load()) == 1
    assert jq.load()[0]["title"] == "T2" and jq.load()[0]["window"] == "23:00-06:30"
    jq.update(a["id"], status="done", chapters=[1])
    assert "Queued" in gui.corr_rebuild(str(p)) and len(jq.load()) == 2
    assert jq.load()[-1]["chapters"] == [1] and jq.load()[-1]["title"] == "T2" and jq.load()[-1]["send_to"] == ""
    (p / "prep.json").unlink()                                                 # not prepared: refused, not queued
    import pytest, gradio as gr
    with pytest.raises(gr.Error):
        gui._enqueue(p, "T", "A", "", None, "", "", "")


def test_segments_are_backed_up_and_extras_survive_prepare(tmp_path):
    import json
    from audiobook_gen import gui, music, prep
    from audiobook_gen.speakers import backup_segments
    p = _assistant_project(tmp_path)
    assert backup_segments(p) and backup_segments(p) is None                   # a second identical copy is not made
    (p / "segments.json").write_text(json.dumps([{"speaker": "Bob", "kind": "dialogue", "text": "Hi.", "id": 1, "chapter": 1}]))
    assert backup_segments(p) and len(list((p / "segments.backup").glob("*.json"))) == 2
    plan = music.load_plan(p); plan["enabled"] = True; plan["intro"].update(track="x", title="A Tale"); music.save_plan(p, plan)
    gui._reapply_extras(p)
    assert any(c.get("opening") for c in json.loads((p / "chapters.json").read_text())["chapters"]) and prep.status(p)["ready"]


def test_book_being_made_cannot_be_edited_and_new_ticks_need_prepare(tmp_path, monkeypatch):
    import pandas as pd, pytest, gradio as gr
    from audiobook_gen import gui, jobqueue as jq
    monkeypatch.setattr(jq, "QUEUE", tmp_path / "queue"); monkeypatch.setattr(jq, "JOBS", tmp_path / "queue" / "jobs.json")
    monkeypatch.setattr(jq, "ensure_supervisor", lambda: False)
    p = _assistant_project(tmp_path)
    chap = pd.DataFrame([[True, 1, "One", 6], [True, 2, "Two", 6]], columns=gui.CHAP_COLS)
    args = (60, 350, 700, 250, 4, True, True, 0.0, list(gui.LEX_MODES)[0], 140, 120, False)
    with pytest.raises(gr.Error, match="no lines yet"):                   # chapter 2 was ticked after Prepare
        gui.generate_queued(str(p), chap, "B", "A", "", *args)
    j = jq.add(str(p), "B", "A")
    jq.update(j["id"], status="running")
    with pytest.raises(gr.Error, match="being made right now"):
        gui.reassign(str(p), pd.DataFrame({"Pick": [True], "id": [1]}), "Bob", "")
    with pytest.raises(gr.Error, match="being made right now"):
        gui.run_prepare(str(p), chap, [])
    jq.update(j["id"], status="done")
    assert "Queued" in gui.corr_rebuild(str(p), True, "phone1")
    assert jq.load()[-1]["send_to"] == "phone1"


def test_corrections_and_hints_follow_their_line_not_its_position(tmp_path):
    import json
    from audiobook_gen import refine
    segs = [{"id": "s1"}, {"id": "s2"}]
    tag = refine.text_tag("Run!")
    data = {"items": {"s1:0": {"emo": "fear", "h": tag}, "s2:0": {"emo": "joy"}}}
    assert refine.hints(segs, [["Run!"], ["Hello there."]], data)[0][0]["emo"] == "fear"
    assert refine.hints(segs, [["A different line."], ["Hello there."]], data)[0][0] is None      # the line changed: no stale feeling
    assert refine.hints(segs, [["x"], ["y"]], data)[1][0]["emo"] == "joy"                           # old data without a tag still works


def test_override_is_ignored_when_the_line_has_moved(tmp_path, monkeypatch):
    import json
    import numpy as np
    from audiobook_gen import synth
    class Fake:
        sample_rate = 24000
        def synth(self, text, voice): return np.zeros(2400, np.float32)
    monkeypatch.setattr(synth, "get_engine", lambda *a, **k: Fake())
    work = tmp_path / "w"; work.mkdir()
    (work / "clip_overrides.json").write_text(json.dumps({"1:0": {"text": "Fixed text.", "orig": "Old line.", "voice": {}}}))
    cfg = {"voices": {"Narrator": {"engine": "kokoro", "voice": "bm_george"}}}
    def spoken(text):
        (work / "segments.json").write_text(json.dumps([{"id": 1, "speaker": "Narrator", "kind": "narration", "text": text, "chapter": 1}]))
        list(synth.synthesize_iter(work, cfg))
        return [m["text"] for m in json.loads((work / "clips_meta.json").read_text()).values()]
    assert spoken("Old line.") == ["Fixed text."]                         # the same line: the correction applies
    assert spoken("A new line took its place.") == ["A new line took its place."]      # another line now: it does not


def test_section_titles_are_spoken_by_default(tmp_path, monkeypatch):
    import json
    from audiobook_gen import music, prep, speakers
    work = tmp_path / "w"; work.mkdir()
    (work / "chapters.json").write_text(json.dumps({"title": "T", "chapters": [{"index": 1, "title": "Chapter 1", "text": "Hello."}]}))
    (work / "lexicon.json").write_text("[]")
    seg = [{"id": "001-00000", "speaker": "Narrator", "kind": "narration", "text": "Hello.", "chapter": 1}]
    monkeypatch.setattr(speakers, "parse_book", lambda *a, **k: ((work / "segments.json").write_text(json.dumps(seg)), seg)[1])
    assert music.load_plan(work)["titles"]["speak"] is True
    r = prep.run(work, [])
    ids = [s["id"] for s in json.loads((work / "segments.json").read_text())]
    assert ids == ["001-title", "001-00000"] and "section titles" in r["notes"]["titles"]
    (work / "music.json").write_text(json.dumps({"titles": {"speak": False}}))     # a book that opts out keeps its plan
    prep.run(work, [])
    assert [s["id"] for s in json.loads((work / "segments.json").read_text())] == ["001-00000"]


def test_spoken_titles_read_roman_numbers_and_book_headings():
    from audiobook_gen import titles
    assert titles.spoken("BOOK ONE THE COMING OF THE MARTIANS") == "Book one. The Coming of the Martians."
    assert titles.spoken("II. THE FALLING STAR.") == "Two. The Falling Star."
    assert titles.spoken("XVII. THE “THUNDER CHILD”.") == "Seventeen. The “Thunder Child”."
    assert titles.spoken("Book of Kells") == "Book of Kells." and titles.spoken("CHAPTER IV") == "Chapter four."


def test_title_case_keeps_roman_numerals_and_capitalises_after_them():
    from audiobook_gen import textclean
    assert textclean.title_case("II. THE FALLING STAR.") == "II. The Falling Star."
    assert textclean.title_case("XII. WHAT I SAW OF THE DESTRUCTION OF WEYBRIDGE") == "XII. What I Saw of the Destruction of Weybridge"
    assert textclean.title_case("CHAPTER I") == "Chapter I"


def test_spoken_title_after_a_book_number_starts_with_a_capital():
    from audiobook_gen import titles
    assert titles.spoken("Book One the Coming of the Martians") == "Book one. The Coming of the Martians."


def test_removing_from_the_queue_asks_first(tmp_path, monkeypatch):
    import pytest, gradio as gr
    from audiobook_gen import gui
    jq, work = _queue_env(tmp_path, monkeypatch)
    ids = [jq.add(str(work), t)["id"] for t in ("One", "Two")]
    with pytest.raises(gr.Error):
        gui.queue_remove_ask([])                                                  # nothing ticked
    show, text = gui.queue_remove_ask([ids[0]])
    assert show["visible"] is True and "“One”" in text and "stay on disk" in text and len(jq.load()) == 2   # asking removes nothing
    assert gui.queue_remove_cancel()[0]["visible"] is False and len(jq.load()) == 2
    out = gui.queue_remove_yes([ids[0]])
    assert out[0]["visible"] is False and "Removed 1" in out[2] and [j["title"] for j in jq.load()] == ["Two"]


def test_same_book_from_another_folder_asks_and_can_replace_the_earlier_job(tmp_path, monkeypatch):
    import shutil
    import pandas as pd
    from audiobook_gen import gui, jobqueue as jq
    monkeypatch.setattr(jq, "ensure_supervisor", lambda: False)
    jq_, work_a = _queue_env(tmp_path, monkeypatch)
    work_b = tmp_path / "book_copy"; shutil.copytree(work_a, work_b)
    old = jq.add(str(work_a), "The war of the worlds", "H. G. Wells")
    chap = pd.DataFrame([[True, 1, "One", 6]], columns=gui.CHAP_COLS)
    args = (str(work_b), chap, "The War of the Worlds", "H. G. Wells", "", 60, 350, 700, 250, 4, True, True, 0.0, list(gui.LEX_MODES)[0], 140, 120, False)
    out = gui.generate_flow(*args)                                           # same book, other folder: asks, queues nothing
    assert out[1]["visible"] is True and "already in the queue" in out[2] and len(jq.load()) == 1
    out = gui.generate_flow(*args, dup="add")                                # queue it as well
    assert len(jq.load()) == 2 and out[1]["visible"] is False
    jq.remove_many([jq.load()[-1]["id"]])
    out = gui.generate_flow(*args, dup="replace")                            # replace: the earlier job is cancelled, then asks about removing it
    jobs = {j["id"]: j for j in jq.load()}
    assert jobs[old["id"]]["status"] == "cancelled" and "replaced" in jobs[old["id"]]["note"]
    assert out[3]["visible"] is True and out[5] == [old["id"]] and len(jobs) == 2
    assert "work folder" in out[4] and "gone for good" in out[4]
    msg, box = gui.dup_remove(out[5], str(work_b))                           # list only: the folder stays
    assert "Removed 1" in msg and old["id"] not in {j["id"] for j in jq.load()} and work_a.exists()
    assert jq.same_book(str(work_b), "The War of the Worlds") == []          # the book itself, from its own folder, is not a duplicate


def test_deleting_the_earlier_jobs_work_folder_is_guarded(tmp_path, monkeypatch):
    import shutil
    from audiobook_gen import gui, jobqueue as jq
    monkeypatch.setattr(jq, "ensure_supervisor", lambda: False)
    jq_, work_a = _queue_env(tmp_path, monkeypatch)
    monkeypatch.setattr(jq, "ROOT", tmp_path)                                # work folders are checked against <root>/work
    (tmp_path / "work").mkdir(); inside = tmp_path / "work" / "old"; shutil.copytree(work_a, inside)
    keep = tmp_path / "work" / "new"; shutil.copytree(work_a, keep)
    a = jq.add(str(inside), "T"); b = jq.add(str(keep), "T")
    msg, _ = gui.dup_remove([a["id"]], str(keep), delete=True)               # the old folder goes
    assert "Deleted the work folder: old" in msg and not inside.exists() and keep.exists()
    c = jq.add(str(keep), "T2")
    msg, _ = gui.dup_remove([c["id"]], str(keep), delete=True)               # the open book's folder is never deleted
    assert "No folder was deleted" in msg and keep.exists()
    outside = tmp_path / "elsewhere"; shutil.copytree(work_a, outside)
    d = jq.add(str(outside), "T3")
    msg, _ = gui.dup_remove([d["id"]], "", delete=True)                      # not under work/: kept
    assert outside.exists()


def test_done_tab_rates_books_and_cleans_up_work_folders(tmp_path, monkeypatch):
    import json, zipfile
    import pytest, gradio as gr
    from audiobook_gen import gui, jobqueue as jq, library
    jq_, work = _queue_env(tmp_path, monkeypatch)
    monkeypatch.setattr(jq, "ROOT", tmp_path); monkeypatch.setattr(library, "ROOT", tmp_path)
    monkeypatch.setattr(library, "RATINGS", tmp_path / "out" / "ratings.json"); monkeypatch.setattr(library, "ARCHIVE", tmp_path / "archive")
    (tmp_path / "work").mkdir(); book = tmp_path / "work" / "mybook"; (book / "clips").mkdir(parents=True)
    (book / "segments.json").write_text("[]"); (book / "chapters.json").write_text("{}"); (book / "clips" / "a.wav").write_bytes(b"x" * 5000)
    (tmp_path / "out").mkdir(); (tmp_path / "out" / "mybook.m4b").write_bytes(b"audio")
    j = jq.add(str(book), "My Book"); jq.update(j["id"], status="done", finished="2026-10-11 12:00")
    tbl = gui.done_table()
    assert list(tbl["Book"]) == ["My Book"] and tbl.loc[0, "State"] == "work folder" and tbl.loc[0, "Rating"] == "not rated"
    tbl.loc[0, "✔"] = True
    msg, tbl = gui.done_rate(tbl, "★★★★", "great")
    assert "★★★★☆" in msg and library.rating("mybook")["stars"] == 4 and tbl.loc[0, "Rating"] == "★★★★☆"
    show, text, pending = gui.done_ask(tbl.assign(**{"✔": [True]}), "zip")                  # asks first; nothing is touched
    assert show["visible"] is True and "zipped" in text and book.exists()
    msg, box, _ = gui.done_do(pending)
    z = tmp_path / "archive" / "mybook.zip"
    assert "zipped to archive/mybook.zip" in msg and z.exists() and not book.exists() and (tmp_path / "out" / "mybook.m4b").exists()
    names = zipfile.ZipFile(z).namelist()
    assert "mybook/segments.json" in names and not any("clips" in n for n in names)          # clips are left out
    assert gui.done_table().loc[0, "State"] == "archived (zip)" and library.rating("mybook")["stars"] == 4   # the rating outlives the folder
    assert "Restored" in gui.done_restore("mybook.zip") and (book / "segments.json").exists() and not (book / "clips").exists()
    z.unlink()
    j2 = jq.add(str(book), "Again")                                                         # a waiting job on the folder blocks any cleanup
    with pytest.raises(gr.Error, match="still has a job"):
        gui.done_ask(gui.done_table().assign(**{"✔": [True]}), "delete")
    jq.remove_many([j2["id"]])
    _, _, pending = gui.done_ask(gui.done_table().assign(**{"✔": [True]}), "delete")
    msg, *_ = gui.done_do(pending)
    assert "folder deleted" in msg and not book.exists() and (tmp_path / "out" / "mybook.m4b").exists()


def test_books_can_be_read_from_markdown_html_and_docx(tmp_path):
    import zipfile
    from audiobook_gen import extract
    md = tmp_path / "my_book.md"
    md.write_text("---\ntitle: The Test Book\nauthor: A. Writer\n---\n\n## Chapter One\n\nIt was a **dark** night, see [this](http://x).\n\n- one\n- two\n\n"
                  "```\ncode()\n```\n\n## Chapter Two\n\nThe second chapter.\n")
    d = extract.extract(str(md), str(tmp_path / "w1"))
    assert d["title"] == "The Test Book" and d["author"] == "A. Writer" and [c["title"] for c in d["chapters"]] == ["Chapter One", "Chapter Two"]
    assert "dark night, see this." in d["chapters"][0]["text"] and "code()" not in d["chapters"][0]["text"] and "one\n\ntwo" in d["chapters"][0]["text"]
    h = tmp_path / "page.html"
    h.write_text("<html><head><title>Web Book</title><meta name='author' content='W. Eb'></head><body><nav>menu</nav><h1>First</h1><p>Alpha beta.</p>"
                 "<h1>Second</h1><p>Gamma.</p><script>x()</script></body></html>")
    d = extract.extract(str(h), str(tmp_path / "w2"))
    assert d["title"] == "Web Book" and d["author"] == "W. Eb" and [c["title"] for c in d["chapters"]] == ["First", "Second"] and "menu" not in d["chapters"][0]["text"]
    w = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
    para = lambda t, s=None: f"<w:p>{f'<w:pPr><w:pStyle w:val=\"{s}\"/></w:pPr>' if s else ''}<w:r><w:t>{t}</w:t></w:r></w:p>"
    doc = tmp_path / "doc.docx"
    with zipfile.ZipFile(doc, "w") as z:
        z.writestr("word/document.xml", f'<w:document xmlns:w="{w}"><w:body>{para("Doc Book", "Title")}{para("One", "Heading1")}{para("First text.")}'
                                        f'{para("Two", "Heading1")}{para("Second text.")}</w:body></w:document>')
    d = extract.extract(str(doc), str(tmp_path / "w3"))
    assert d["title"] == "Doc Book" and [(c["title"], c["text"]) for c in d["chapters"]] == [("One", "First text."), ("Two", "Second text.")]
    import pytest
    with pytest.raises(ValueError):
        extract.extract(str(tmp_path / "x.pdf"), str(tmp_path / "w4"))


def test_any_audio_format_can_be_a_clone_reference(tmp_path, monkeypatch):
    import subprocess
    import soundfile as sf
    from audiobook_gen import audio_io, voices
    monkeypatch.setattr(audio_io, "CACHE", tmp_path / "cache")
    monkeypatch.setattr(voices, "LIB", tmp_path / "lib")
    src = tmp_path / "tone.wav"
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-f", "lavfi", "-i", "sine=frequency=300:duration=3", "-ar", "44100", str(src)], check=True)
    for ext, codec in ((".m4a", "aac"), (".opus", "libopus"), (".wma", "wmav2"), (".mp3", "libmp3lame")):
        f = tmp_path / f"clip{ext}"
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(src), "-c:a", codec, str(f)], check=True)
        a, sr = audio_io.read_audio(f)
        assert len(a) > sr and a.ndim == 1                                         # decoded, mono
        voices.save_voice(f"v{ext[1:]}", str(f), "a tone", {})
        ref, rsr = sf.read(tmp_path / "lib" / f"v{ext[1:]}" / "ref.wav")
        assert rsr == 24000 and len(ref) > 24000
    import pytest
    bad = tmp_path / "notaudio.m4a"; bad.write_bytes(b"hello world" * 200)
    with pytest.raises(ValueError, match="could not be read"):
        audio_io.to_wav(bad)
