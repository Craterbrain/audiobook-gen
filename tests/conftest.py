import pytest


@pytest.fixture(autouse=True)
def _isolated_run_stats(tmp_path, monkeypatch):
    """Tests must never write into the real work/run_stats.json."""
    from audiobook_gen import runstats
    monkeypatch.setattr(runstats, "PATH", tmp_path / "run_stats.json")
