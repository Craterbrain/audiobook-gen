"""Chatterbox voice cloning worker. Run with .venv-chatterbox."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _worker_io import serve  # first: it takes over stdout
import soundfile as sf
import torch

_load = torch.load
torch.load = lambda *a, **k: _load(*a, **{**k, "map_location": "cpu"})   # its checkpoints were saved on CUDA
from chatterbox.tts import ChatterboxTTS

model = ChatterboxTTS.from_pretrained(device="xpu")
current = [None]


def handle(req):
    if req.get("seed") is not None and req["seed"] >= 0:
        torch.manual_seed(int(req["seed"]))
    ex = float(req.get("exaggeration", 0.5))
    key = (req["ref_audio"], ex)
    if current[0] != key:    # voice + emotion conditioning is cached between sentences
        model.prepare_conditionals(req["ref_audio"], exaggeration=ex)
        current[0] = key
    wav = model.generate(req["text"], exaggeration=ex, cfg_weight=float(req.get("cfg_weight", 0.4)))
    torch.xpu.synchronize()
    sf.write(req["out"], wav.squeeze().cpu().numpy(), model.sr)


serve(model.sr, handle)
