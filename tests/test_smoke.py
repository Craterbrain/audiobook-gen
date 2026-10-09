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
    for name in ("RUNNER_PID", "JOB_PID", "SUPERVISOR_PID", "STOPPED", "SPEECH_LIMIT", "SETTINGS", "HALTED"):   # never touch the real queue's files
        monkeypatch.setattr(jq, name, tmp_path / "queue" / name.lower())
    sent = []                                                                   # alerts are recorded, never sent
    monkeypatch.setattr(jq, "notify", lambda text, title="", **k: sent.append((title, text)) or True)
    jq.sent_alerts = sent
    work = tmp_path / "book"; work.mkdir()
    (work / "segments.json").write_text("[]")
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
    assert len(out) == 23 and out[0] == str(p) and out[2] == "Running book" and out[3] == "An Author"
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
    assert gui.gen_settings(str(p))[-1] is False                       # the toggle shows what is saved


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
    assert pick["value"] == [ids[2]]                                                  # the ticks survive a refresh
    assert [c[0].split(". ")[1].split(" —")[0] for c in pick["choices"]] == titles()
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


def test_a_finished_book_can_free_its_disk_space(tmp_path, monkeypatch):
    import sys
    jq, work = _queue_env(tmp_path, monkeypatch)
    (work / "clips").mkdir(); (work / "clips" / "a.wav").write_text("x")
    out = tmp_path / "x.m4b"
    jq.add(str(work), "Tidy book", out=str(out))
    jq.save_settings(clean_after_done=True)
    class R(jq.Runner):
        synth_cmd = lambda self, j: [sys.executable, "-c", "print('[progress] 1/1')"]
        assemble_cmd = lambda self, j: [sys.executable, "-c", f"open({str(out)!r}, 'w').write('x')"]
    R(poll=0.1, foreign=lambda: "").step()
    assert out.exists() and not (work / "clips").exists()


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
