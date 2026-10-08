"""Qwen3-TTS (1.7B Base) voice cloning worker. Run with .venv-tts2."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _worker_io import serve  # first: it takes over stdout
import soundfile as sf
import torch
from qwen_tts import Qwen3TTSModel

model = Qwen3TTSModel.from_pretrained("Qwen/Qwen3-TTS-12Hz-1.7B-Base", device_map="xpu:0", dtype=torch.bfloat16)
prompts: dict = {}


def handle(req):
    key = (req["ref_audio"], req["ref_text"])
    if key not in prompts:   # the voice is encoded once, not for every sentence
        prompts[key] = model.create_voice_clone_prompt(ref_audio=req["ref_audio"], ref_text=req["ref_text"].replace("“", "").replace("”", ""))
    if req.get("seed") is not None and req["seed"] >= 0:
        torch.manual_seed(int(req["seed"]))
    wavs, sr = model.generate_voice_clone(text=req["text"], language="English", voice_clone_prompt=prompts[key])
    torch.xpu.synchronize()
    sf.write(req["out"], wavs[0], sr)


serve(24000, handle)
