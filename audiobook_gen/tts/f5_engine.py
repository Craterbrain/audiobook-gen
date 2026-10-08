"""F5-TTS zero-shot cloning on XPU. Roles supply ref_audio + ref_text."""
import numpy as np

from .base import TTSEngine
from .device import pick_device, sync


def _patch_torchaudio_load():
    """torchaudio>=2.9 loads via torchcodec, whose PyPI wheel links CUDA libs (absent on XPU).
    Route torchaudio.load through soundfile instead."""
    import soundfile as sf
    import torch
    import torchaudio

    def load(path, *a, **k):
        data, sr = sf.read(str(path), dtype="float32", always_2d=True)
        return torch.from_numpy(data.T.copy()), sr

    torchaudio.load = load


class F5Engine(TTSEngine):
    name = "f5"
    lexicon_mode = "respell"  # plain-text model: use phonetic respellings

    def __init__(self, device_pref: str = "auto", model: str = "F5TTS_v1_Base", precision: str = "float16"):
        _patch_torchaudio_load()
        import torch
        from f5_tts.api import F5TTS

        self.device = pick_device(device_pref)
        # F5TTS accepts an explicit device string; "xpu:0" routes model + Vocos vocoder there.
        self.tts = F5TTS(model=model, device=self.device)
        self.sample_rate = self.tts.target_sample_rate
        # F5 loads in float32 on Intel GPUs (its half-precision switch only checks for CUDA), which is ~4.8x slower
        # than the float16 the card handles natively. bfloat16 fails in the ODE solver; float16 sounds the same.
        self.half = precision == "float16" and self.device.startswith("xpu")
        if self.half:
            self.tts.ema_model.to(torch.float16)

    def synth(self, text: str, voice: dict) -> np.ndarray:
        out = self._infer(text, voice)
        if self.half and (not np.isfinite(out).all() or np.abs(out).max() < 1e-4):   # half precision misbehaved
            import torch
            self.tts.ema_model.to(torch.float32)
            self.half = False
            print("[f5] float16 produced invalid audio; falling back to float32")
            out = self._infer(text, voice)
        return out

    def _infer(self, text: str, voice: dict) -> np.ndarray:
        seed = voice.get("seed")
        wav, sr, _ = self.tts.infer(
            ref_file=voice["ref_audio"], ref_text=voice["ref_text"], gen_text=text,
            speed=voice.get("speed", 1.0), nfe_step=int(voice.get("nfe_step", 32)),
            cfg_strength=voice.get("cfg_strength", 2.0), sway_sampling_coef=voice.get("sway_sampling_coef", -1.0),
            cross_fade_duration=voice.get("cross_fade_duration", 0.15), target_rms=voice.get("target_rms", 0.1),
            seed=None if seed is None or seed < 0 else int(seed), remove_silence=False,
        )
        sync(self.device)
        assert sr == self.sample_rate
        return np.asarray(wav, dtype=np.float32)
