"""Engines that live in their own virtualenv (their libraries pin a different transformers than the main one).
The model runs in a worker process that reads one JSON request per line on stdin and answers with the path of a
wav file on stdout; the main process keeps it loaded between clips."""
import json
import subprocess
import tempfile
import threading
from pathlib import Path

import numpy as np
import soundfile as sf

from .base import TTSEngine

ROOT = Path(__file__).resolve().parent.parent.parent


class RemoteEngine(TTSEngine):
    lexicon_mode = "respell"   # plain-text models
    venv: str
    worker: str

    @classmethod
    def available(cls) -> bool:
        return (ROOT / cls.venv / "bin" / "python").exists()

    def __init__(self, device_pref: str = "auto", **_):
        import os
        (ROOT / "work").mkdir(exist_ok=True)
        self._log = open(ROOT / "work" / f"{self.name}_worker.log", "ab")
        env = {**os.environ, "PYTHONPATH": str(ROOT), "PYTHONUNBUFFERED": "1"}
        self.proc = subprocess.Popen([str(ROOT / self.venv / "bin" / "python"), str(Path(__file__).with_name(self.worker))],
                                     stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=self._log, text=True, cwd=ROOT, env=env)
        self._lock = threading.Lock()
        ready = self._reply()
        if not ready.get("ready"):
            raise RuntimeError(f"{self.name} worker failed to start; see work/{self.name}_worker.log")
        self.sample_rate = ready["sr"]

    def _reply(self) -> dict:
        for line in iter(self.proc.stdout.readline, ""):
            if line.startswith("{"):
                return json.loads(line)
        raise RuntimeError(f"{self.name} worker died; see work/{self.name}_worker.log")

    def synth(self, text: str, voice: dict) -> np.ndarray:
        req = {"text": text, **{k: voice[k] for k in ("ref_audio", "ref_text", "exaggeration", "cfg_weight", "seed") if k in voice}}
        with self._lock, tempfile.TemporaryDirectory() as tmp:
            req["out"] = str(Path(tmp) / "out.wav")
            self.proc.stdin.write(json.dumps(req) + "\n")
            self.proc.stdin.flush()
            reply = self._reply()
            if "error" in reply:
                raise RuntimeError(f"{self.name}: {reply['error']}")
            audio, sr = sf.read(req["out"], dtype="float32")
        self.sample_rate = sr
        return audio

    def __del__(self):
        try:
            self.proc.stdin.close()
            self.proc.terminate()
        except Exception:
            pass


class Qwen3Engine(RemoteEngine):
    name, venv, worker = "qwen3", ".venv-tts2", "worker_qwen3.py"


class ChatterboxEngine(RemoteEngine):
    name, venv, worker = "chatterbox", ".venv-chatterbox", "worker_chatterbox.py"
