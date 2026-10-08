"""Read an existing audiobook (.m4b): chapters, clean speech clips, and a plain-text Whisper transcript per clip."""
import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CACHE = ROOT / "work" / "_m4b_cache"
_ASR = {}


def chapters(path: str) -> list[dict]:
    """[{index, title, start, end}] in seconds; a file without chapter marks is one chapter."""
    probe = json.loads(subprocess.run(
        ["ffprobe", "-v", "error", "-print_format", "json", "-show_chapters", "-show_format", str(path)],
        check=True, capture_output=True, text=True).stdout)
    out = [{"index": i + 1, "title": (c.get("tags") or {}).get("title") or f"Chapter {i + 1}",
            "start": float(c["start_time"]), "end": float(c["end_time"])}
           for i, c in enumerate(probe.get("chapters", []))]
    return out or [{"index": 1, "title": Path(path).stem, "start": 0.0,
                    "end": float(probe["format"]["duration"])}]


def extract_wav(path: str, start: float, end: float, sr: int, out: Path) -> Path:
    out.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-ss", f"{start:.3f}", "-to", f"{end:.3f}",
                    "-i", str(path), "-vn", "-ac", "1", "-ar", str(sr), "-c:a", "pcm_s16le", str(out)], check=True)
    return out


def speech_windows(wav: Path, target: float = 9.0, lo: float = 6.0, hi: float = 12.0, top: int = 5,
                   min_pause: float = 0.3) -> list[tuple[float, float]]:
    """Clean clip spans (seconds within `wav`) found from the waveform alone: they start after a pause, end at
    a pause, last lo..hi seconds, and stay clear of the chapter's music/room tone. No transcript needed."""
    import numpy as np
    import soundfile as sf

    a, sr = sf.read(wav, dtype="float32")
    frame = int(sr * 0.02)
    n = len(a) // frame
    rms = np.sqrt((a[: n * frame].reshape(n, frame) ** 2).mean(1))
    voiced = rms > max(0.008, 0.1 * np.percentile(rms, 95))
    # pauses = runs of unvoiced frames at least min_pause long
    pauses, i = [], 0
    while i < n:
        if not voiced[i]:
            j = i
            while j < n and not voiced[j]:
                j += 1
            if (j - i) * 0.02 >= min_pause:
                pauses.append((i * 0.02, j * 0.02))
            i = j
        else:
            i += 1
    cands = []
    for x in range(len(pauses) - 1):
        for y in range(x + 1, len(pauses)):
            s, e = pauses[x][1], pauses[y][0]  # speech from end of one pause to start of a later one
            if e - s > hi:
                break
            if e - s >= lo:
                # longer closing pause = more likely a real sentence end
                cands.append((s, e, -abs((e - s) - target) + min(pauses[y][1] - pauses[y][0], 1.0)))
    picked = []
    for s, e, sc in sorted(cands, key=lambda c: -c[2]):
        if all(e <= p[0] or s >= p[1] for p in picked):
            picked.append((s, e))
        if len(picked) >= top:
            break
    return sorted(picked)


def cut(path: str, start: float, end: float, out: Path, pad: float = 0.08) -> Path:
    """Cut [start, end] as 24 kHz mono 16-bit with a little room at both ends."""
    return extract_wav(path, max(0.0, start - pad), end + pad, 24000, out)


def transcribe_clip(wav: Path, model: str = "openai/whisper-small.en", device_pref: str = "auto") -> str:
    """Plain-text speech to text for one short clip (no timestamps; clips this short don't loop)."""
    import soundfile as sf
    import torch
    from transformers import pipeline

    from .tts.device import pick_device

    if model not in _ASR:
        _ASR[model] = pipeline("automatic-speech-recognition", model=model,
                               device=pick_device(device_pref), dtype=torch.float32)
    audio, sr = sf.read(wav, dtype="float32")
    if sr != 16000:
        from math import gcd

        from scipy.signal import resample_poly
        g = gcd(16000, sr)
        audio = resample_poly(audio, 16000 // g, sr // g).astype("float32")
    return _ASR[model]({"raw": audio, "sampling_rate": 16000})["text"].strip()


def free_asr() -> None:
    import gc

    import torch
    _ASR.clear()
    gc.collect()
    if hasattr(torch, "xpu") and torch.xpu.is_available():
        torch.xpu.empty_cache()
