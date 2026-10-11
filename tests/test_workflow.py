import json
import os
import time

import yaml

from audiobook_gen import prep, workflow as wf


def make_book(tmp_path, prepared=True, **cfg):
    w = tmp_path / "book"; w.mkdir()
    (w / "chapters.json").write_text(json.dumps({"title": "T", "author": "A", "chapters": [{"index": 1, "title": "One", "text": "x"}, {"index": 2, "title": "Two", "text": "y"}]}))
    (w / "segments.json").write_text(json.dumps([{"id": "1-1", "speaker": "Narrator", "kind": "narration", "text": "Hi.", "chapter": 1},
                                                 {"id": "2-1", "speaker": "Bob", "kind": "dialogue", "text": "Yo.", "chapter": 2}]))
    (w / "config.yaml").write_text(yaml.safe_dump({"voices": {"Narrator": {"engine": "kokoro", "voice": "bm_george"}}, **cfg}))
    if prepared:
        prep.mark_done(w)
    return w


def state(stages, i):
    return {s.id: s for s in stages}[i]


def by(w, **k):
    k.setdefault("out", w.parent / "none.m4b")
    return wf.stage_status(w, **k)


def test_no_book_means_book_required_and_the_rest_blocked():
    s = wf.stage_status(None)
    assert s[0].state == "required" and all(x.blocked_by == "book" for x in s[1:])


def test_prepared_book_recommends_voices(tmp_path):
    s = by(make_book(tmp_path))
    assert state(s, "book").state == "done" and state(s, "voices").state == "recommended" and state(s, "make").state == "required"


def test_changed_chapters_make_the_book_stale_and_block_the_rest(tmp_path):
    w = make_book(tmp_path)
    (w / "chapters.json").write_text(json.dumps({"title": "T", "chapters": []}))
    s = by(w)
    assert state(s, "book").state == "stale" and state(s, "make").blocked_by == "book" and state(s, "voices").blocked_by == "book"


def test_ticks_that_differ_from_the_prepared_chapters_are_stale(tmp_path):
    w = make_book(tmp_path)
    assert state(by(w, ticks={1, 2}), "book").state == "done"
    assert state(by(w, ticks={1}), "book").state == "stale"


def test_voices_done_for_one_narrator_or_after_ack_and_stale_with_new_role(tmp_path):
    w = make_book(tmp_path, single_voice={"enabled": True, "voice": "bm_george"})
    assert state(by(w), "voices").state == "done"
    (tmp_path / "x").mkdir(); w2 = make_book(tmp_path / "x")
    wf.ack(w2, "voices")
    assert state(by(w2), "voices").state == "done"
    segs = json.loads((w2 / "segments.json").read_text()); segs.append({"id": "2-2", "speaker": "Carol", "kind": "dialogue", "text": "Hi", "chapter": 2})
    (w2 / "segments.json").write_text(json.dumps(segs))
    assert state(by(w2), "voices").state == "stale"


def test_names_recommended_then_done_after_ack_then_stale_with_new_name(tmp_path):
    w = make_book(tmp_path)
    lex = [{"term": "Zorbax", "kind": "name", "known": False, "count": 5, "ipa": "", "respell": ""},
           {"term": "Vlad", "kind": "name", "known": False, "count": 3, "ipa": "", "respell": ""},
           {"term": "Bob", "kind": "name", "known": True, "count": 9, "ipa": "", "respell": ""}]
    (w / "lexicon.json").write_text(json.dumps(lex))
    assert state(by(w), "names").state == "recommended" and "2 names" in state(by(w), "names").detail
    wf.ack(w, "names")
    assert state(by(w), "names").state == "done"
    (w / "lexicon.json").write_text(json.dumps(lex + [{"term": "Quux", "kind": "place", "known": False, "count": 2, "ipa": "", "respell": ""}]))
    assert state(by(w), "names").state == "stale"


def test_lost_spoken_title_makes_extras_stale(tmp_path):
    w = make_book(tmp_path)
    (w / "music.json").write_text(json.dumps({"enabled": True, "intro": {"track": "t", "title": "A Tale"}}))
    assert state(by(w), "extras").state == "stale"
    c = json.loads((w / "chapters.json").read_text()); c["chapters"].insert(0, {"index": 0, "title": "Opening", "text": "A Tale", "opening": True})
    (w / "chapters.json").write_text(json.dumps(c)); prep.restamp(w)
    assert state(by(w), "extras").state == "done"


def test_make_is_stale_with_the_cost_of_what_changed(tmp_path):
    w = make_book(tmp_path)
    out = tmp_path / "book.m4b"; out.write_bytes(b"x")
    wf.stamp_queued(w, "j1", "T", "A")
    assert state(by(w, out=out), "make").state == "done"
    cfg = yaml.safe_load((w / "config.yaml").read_text()); cfg["pacing_ms"] = {"sentence": 500}
    (w / "config.yaml").write_text(yaml.safe_dump(cfg))
    s = state(by(w, out=out), "make")
    assert s.state == "stale" and s.cost == "build"
    cfg["voices"]["Bob"] = {"engine": "kokoro", "voice": "am_adam"}
    (w / "config.yaml").write_text(yaml.safe_dump(cfg))
    assert state(by(w, out=out), "make").cost == "speech"


def test_a_running_job_makes_make_busy(tmp_path):
    w = make_book(tmp_path)
    job = {"work": str(w), "status": "running", "progress": "5/10", "note": "making the speech"}
    s = state(by(w, jobs=[job]), "make")
    assert s.state == "busy" and "5/10" in s.detail


def test_fix_states(tmp_path):
    w = make_book(tmp_path)
    out = tmp_path / "book.m4b"; out.write_bytes(b"x")
    assert state(by(w, out=out), "fix").blocked_by == "make"             # no clips yet
    (w / "clips_meta.json").write_text("{}")
    (w / "qc_report.json").write_text(json.dumps({"problems_left": [1, 2, 3]}))
    s = state(by(w, out=out), "fix")
    assert s.state == "recommended" and "3 clips flagged" in s.detail
    (w / "clip_overrides.json").write_text("{}")
    os.utime(out, (time.time() - 100, time.time() - 100))
    assert state(by(w, out=out), "fix").state == "stale"


def test_without_workflow_json_the_mtime_decides(tmp_path):
    w = make_book(tmp_path)
    out = tmp_path / "book.m4b"; out.write_bytes(b"x")
    os.utime(out, (time.time() - 100, time.time() - 100))
    s = state(by(w, out=out), "make")
    assert s.state == "stale" and s.cost == "unknown"


def test_next_step_points_at_the_earliest_hard_step_and_ignores_optional(tmp_path):
    w = make_book(tmp_path, prepared=False)
    s = by(w)
    assert wf.next_step(s, "make").id == "book" and wf.next_step(s, "book") is None
    prep.mark_done(w)
    s = by(w)
    assert wf.next_step(s, "make").id == "voices"                          # recommended comes after required/stale


def test_the_two_fingerprints_do_not_overlap(tmp_path):
    w = make_book(tmp_path)
    cfg = yaml.safe_load((w / "config.yaml").read_text())
    s0, b0 = wf.speech_fp(w, cfg), wf.build_fp(w, cfg)
    assert wf.speech_fp(w, {**cfg, "pacing_ms": {"sentence": 9}}) == s0 and wf.build_fp(w, {**cfg, "pacing_ms": {"sentence": 9}}) != b0
    assert wf.build_fp(w, {**cfg, "voices": {}}) == b0 and wf.speech_fp(w, {**cfg, "voices": {}}) != s0


def test_gui_strip_and_queueing_stamp_the_book(tmp_path, monkeypatch):
    from audiobook_gen import gui, jobqueue as jq
    monkeypatch.setattr(jq, "QUEUE", tmp_path / "queue"); monkeypatch.setattr(jq, "JOBS", tmp_path / "queue" / "jobs.json")
    monkeypatch.setattr(jq, "ensure_supervisor", lambda: False)
    w = make_book(tmp_path)
    assert "① Book" in gui.refresh_strip("") and "⑥ Fix" in gui.refresh_strip(str(w))
    gui._enqueue(w, "T", "A", "", None, "", "", "")
    assert json.loads((w / "workflow.json").read_text())["queued"]["speech"]
    assert "⟳" in gui.refresh_strip(str(w))                                  # its job is waiting: Make shows busy
