"""Read any audio file the user gives us (wav, mp3, flac, ogg, opus, m4a, m4b, aac, wma, a video's soundtrack ...).

soundfile reads the common ones directly; anything else is decoded by ffmpeg into a cached mono 16-bit WAV first.
"""
import hashlib
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CACHE = ROOT / "cache" / "audio_in"
AUDIO_EXTS = [".wav", ".mp3", ".flac", ".ogg", ".oga", ".opus", ".m4a", ".m4b", ".mp4", ".aac", ".wma", ".aif", ".aiff", ".webm", ".mkv",
              ".amr", ".ac3", ".caf", ".3gp", ".mov"]
MAX_SECONDS = 600           # a clone clip is a few seconds; never decode more than ten minutes of a long file


def to_wav(path) -> str:
    """The path itself if soundfile can open it, else a decoded WAV copy made by ffmpeg. Raises ValueError if it is not audio."""
    import soundfile as sf
    p = Path(path)
    if not p.exists():
        raise ValueError(f"{p.name} does not exist.")
    try:
        sf.info(str(p))
        return str(p)
    except Exception:
        pass
    st = p.stat()
    key = hashlib.sha1(f"{p.resolve()}|{st.st_mtime_ns}|{st.st_size}".encode()).hexdigest()[:16]
    out = CACHE / f"{p.stem[:40]}_{key}.wav"
    if not out.exists():
        CACHE.mkdir(parents=True, exist_ok=True)
        r = subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(p), "-vn", "-ac", "1", "-t", str(MAX_SECONDS),
                            "-c:a", "pcm_s16le", str(out)], capture_output=True, text=True)
        if r.returncode != 0 or not out.exists() or out.stat().st_size < 1000:
            out.unlink(missing_ok=True)
            raise ValueError(f"{p.name} could not be read as audio ({(r.stderr or 'no audio found').strip().splitlines()[-1][:120]}).")
    return str(out)


def read_audio(path, mono: bool = True):
    """(samples as float32, sample rate) from any audio file. Mono by default (channels averaged)."""
    import soundfile as sf
    a, sr = sf.read(to_wav(path), dtype="float32", always_2d=True)
    return (a.mean(1) if mono else a), sr
