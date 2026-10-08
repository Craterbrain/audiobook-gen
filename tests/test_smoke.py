import json, sys
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
    assert set(gui.LEX_MODES.values()) == {"verified", "rawipa", "respell", "plain"}


def test_placeholder_cover_has_large_readable_title(tmp_path):
    from PIL import Image
    from audiobook_gen.assemble import make_cover
    p = make_cover(tmp_path / "c.jpg", "Genesis", "NASB 1995")
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
