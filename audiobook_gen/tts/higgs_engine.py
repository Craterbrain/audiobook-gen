"""Higgs Audio v3 TTS (4B) zero-shot cloning on XPU, in the main virtualenv (transformers >= 5.5, bf16, about 9.4 GB of the card).

It reads plain text (no IPA, no respellings) and takes control tags at the start of a sentence. The voice dict supplies ref_audio + ref_text
and these settings: temperature (0.7), top_k (50), top_p (0.95), seed, tag (e.g. "<|prosody:expressive_high|>" put in front of every
clip) and pace (a pitch-preserving stretch afterwards: 1.0 = as made, 1.08 = a little faster). voices.HIGGS_DEFAULTS has the defaults.

The model code is the community port multimodalart/higgs-audio-v3-tts-4b-transformers (trust_remote_code); it is pinned to the commit that was
read before use, so an update of that repository is never run unseen. Licence of the weights: research and non-commercial."""
import subprocess

import numpy as np

from .base import TTSEngine
from .device import pick_device, sync

MODEL = "multimodalart/higgs-audio-v3-tts-4b-transformers"
REVISION = "30f01593ee6a12efa586c92455afe4b76e45095d"      # the commit whose code was read before it was first run


class HiggsEngine(TTSEngine):
    name = "higgs"
    lexicon_mode = "respell"
    sample_rate = 24000

    @classmethod
    def available(cls) -> bool:
        """The main environment can run it (transformers 5.5 or newer). The 9.3 GB of weights download the first time it is used."""
        try:
            import transformers
            major, minor = (int(x) for x in transformers.__version__.split(".")[:2])
            return (major, minor) >= (5, 5)
        except Exception:
            return False

    @classmethod
    def cached(cls) -> bool:
        """The weights are already on this machine."""
        try:
            from huggingface_hub import try_to_load_from_cache
            return isinstance(try_to_load_from_cache(MODEL, "model.safetensors", revision=REVISION), str)
        except Exception:
            return False

    def __init__(self, device_pref: str = "auto", **_):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.device = pick_device(device_pref)
        self.tok = AutoTokenizer.from_pretrained(MODEL, revision=REVISION)
        self.model = AutoModelForCausalLM.from_pretrained(MODEL, revision=REVISION, trust_remote_code=True, dtype=torch.bfloat16).to(self.device).eval()
        self._codes: dict[str, object] = {}

    def _reference(self, path: str):
        """The reference clip's codes, made once per clip."""
        if path not in self._codes:
            import soundfile as sf
            import torch
            y, sr = sf.read(path, dtype="float32")
            self._codes[path] = self.model._encode_reference(torch.from_numpy(y if y.ndim == 1 else y.mean(1)), sr).cpu()
        return self._codes[path]

    def synth(self, text: str, voice: dict) -> np.ndarray:
        import torch
        seed = voice.get("seed")
        if seed is not None and int(seed) >= 0:
            torch.manual_seed(int(seed))
        wav = self.model.generate_speech(
            (voice.get("tag") or "") + text, self.tok, reference_codes=self._reference(voice["ref_audio"]), reference_text=voice.get("ref_text"),
            temperature=float(voice.get("temperature", 0.7)), top_k=int(voice.get("top_k", 50)), top_p=float(voice.get("top_p", 0.95)),
            max_new_tokens=int(voice.get("max_new_tokens", 1500)))
        sync(self.device)
        audio = wav.numpy().astype(np.float32)
        pace = float(voice.get("pace", 1.08))
        return stretch(audio, self.sample_rate, pace) if abs(pace - 1.0) > 0.005 and len(audio) else audio


def stretch(audio: np.ndarray, sr: int, speed: float) -> np.ndarray:
    """Faster or slower without changing the pitch (ffmpeg's atempo: good quality for changes of a few percent)."""
    speed = min(2.0, max(0.5, speed))
    r = subprocess.run(["ffmpeg", "-v", "error", "-f", "f32le", "-ar", str(sr), "-ac", "1", "-i", "-", "-filter:a", f"atempo={speed:.4f}", "-f", "f32le", "-"],
                       input=audio.astype(np.float32).tobytes(), capture_output=True)
    if r.returncode != 0 or not r.stdout:
        return audio
    return np.frombuffer(r.stdout, dtype=np.float32).copy()
