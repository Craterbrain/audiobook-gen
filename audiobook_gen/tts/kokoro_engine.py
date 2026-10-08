"""Kokoro-82M on Intel XPU. Pure PyTorch ops -> no CUDA dependency."""
import numpy as np

from .base import TTSEngine
from .device import pick_device, sync


class KokoroEngine(TTSEngine):
    name = "kokoro"
    lexicon_mode = "ipa"  # misaki understands [word](/IPA/)

    def __init__(self, device_pref: str = "auto"):
        from kokoro import KModel

        self.device = pick_device(device_pref)
        self.model = KModel().to(self.device).eval()
        self._pipes = {}

    def _pipe(self, voice_name: str):
        from kokoro import KPipeline

        lang = voice_name[0]  # a = American, b = British
        if lang not in self._pipes:
            self._pipes[lang] = KPipeline(lang_code=lang, model=self.model)
        return self._pipes[lang]

    def synth(self, text: str, voice: dict) -> np.ndarray:
        v = voice["voice"]
        if v.startswith("pack:"):   # a voicepack made from a recording: voices/kokoro/<name>.pt
            import json

            import torch

            from ..voicepack import PACKS
            name = v[5:]
            meta = json.loads((PACKS / f"{name}.json").read_text())
            lang = meta.get("lang", "a")
            eq, pack_speed = meta, meta.get("speed", 1.0)
            style, pipe = torch.load(PACKS / f"{name}.pt", weights_only=True), self._pipe(lang + "_")
        else:
            style, pipe, eq, pack_speed = v, self._pipe(v), None, 1.0
        parts = [a.detach().cpu().numpy() for _, _, a in
                 pipe(text, voice=style, speed=voice.get("speed", 1.0) * pack_speed) if a is not None]
        sync(self.device)
        out = np.concatenate(parts).astype(np.float32) if parts else np.zeros(0, np.float32)
        if eq:   # a voice pack made from a recording carries its EQ, pause tightening and pitch trim
            from ..voicepack import finish
            out = finish(out, self.sample_rate, eq)
        return out
