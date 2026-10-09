import pytest


@pytest.fixture(autouse=True)
def _isolated_run_stats(tmp_path, monkeypatch):
    """Tests must never write into the real work/run_stats.json."""
    from audiobook_gen import runstats
    monkeypatch.setattr(runstats, "PATH", tmp_path / "run_stats.json")


@pytest.fixture(autouse=True)
def _no_real_gpu_lease(tmp_path, monkeypatch):
    """Tests never take or wait for the real queue's GPU lease."""
    from audiobook_gen import jobqueue
    monkeypatch.setattr(jobqueue, "GUI_LOCK", tmp_path / "gui_lock")
    try:
        from audiobook_gen import gui
        monkeypatch.setattr(gui, "need_gpu", lambda: None)
    except Exception:
        pass
