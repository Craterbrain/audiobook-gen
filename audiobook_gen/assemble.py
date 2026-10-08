"""Module 4: crossfade/pace clips into chapters, then mux a tagged, chaptered .m4b."""
import json
import subprocess
from math import gcd
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly


def _load(path: Path, sr: int) -> np.ndarray:
    a, file_sr = sf.read(path, dtype="float32")
    if a.ndim > 1:
        a = a.mean(1)
    if file_sr != sr:
        g = gcd(sr, file_sr)
        a = resample_poly(a, sr // g, file_sr // g).astype(np.float32)
    return _trim(a, sr)


def _trim(a: np.ndarray, sr: int, thresh: float = 0.005, keep_ms: int = 25) -> np.ndarray:
    idx = np.flatnonzero(np.abs(a) > thresh)
    if not len(idx):
        return a
    k = int(sr * keep_ms / 1000)
    return a[max(0, idx[0] - k): idx[-1] + k]


def join(a: np.ndarray, b: np.ndarray, sr: int, pause_ms: int, xfade_ms: int) -> np.ndarray:
    """Append b to a. With a pause: fade a's tail / b's head around silence.
    Without: equal-power overlap crossfade."""
    n = int(sr * xfade_ms / 1000)
    n = min(n, len(a) // 2, len(b) // 2)
    if pause_ms > 0 or n == 0:
        a, b = a.copy(), b.copy()
        f = min(int(sr * 0.015), len(a) // 2, len(b) // 2)  # de-click
        if f:
            a[-f:] *= np.linspace(1, 0, f, dtype=np.float32)
            b[:f] *= np.linspace(0, 1, f, dtype=np.float32)
        return np.concatenate([a, np.zeros(int(sr * pause_ms / 1000), np.float32), b])
    t = np.linspace(0, np.pi / 2, n, dtype=np.float32)
    mid = a[-n:] * np.cos(t) + b[:n] * np.sin(t)
    return np.concatenate([a[:-n], mid, b[n:]])


def build_chapter(segs: list[dict], clips: dict, clip_dir: Path, cfg: dict) -> np.ndarray:
    """Chunks within a segment get a sentence pause; segments get paragraph / speaker-change pauses."""
    sr, p, xf = cfg["sample_rate"], cfg["pacing_ms"], cfg["crossfade_ms"]
    out, prev = None, None
    for seg in segs:
        for ci, f in enumerate(clips.get(seg["id"], [])):
            a = _load(clip_dir / f, sr)
            if out is None:
                out = np.concatenate([np.zeros(int(sr * p["chapter_start"] / 1000), np.float32), a])
                continue
            if ci:
                pause = p["sentence"]
            elif prev and prev["text"].rstrip().endswith((",", ";", ":", "—", "–", "-")):
                pause = p.get("continuation", 140)    # the sentence carries on: "Upon my word," / cried the old man,
            elif prev and prev.get("kind") == "dialogue" and seg.get("kind") == "narration" and len(seg["text"]) < 90:
                pause = p.get("tag", 120)             # a quote followed by its tag: "In an hour?" / inquired Danglars
            else:
                pause = p["paragraph"] if seg.get("para_start") else p["speaker_change"]
            out = join(out, a, sr, pause, xf)
        prev = seg
    return out if out is not None else np.zeros(sr, np.float32)


def ffmeta(chapters: list[tuple[str, float]], title: str, author: str) -> str:
    lines = [";FFMETADATA1", f"title={title}", f"artist={author}", f"album={title}",
             "genre=Audiobook"]
    t = 0
    for name, dur in chapters:
        end = t + round(dur * 1000)
        lines += ["[CHAPTER]", "TIMEBASE=1/1000", f"START={t}", f"END={end}", f"title={name}"]
        t = end
    return "\n".join(lines) + "\n"


def _serif(size: int):
    """A bold serif font: the system's own (fontconfig), a few common paths, or Pillow's built-in scalable font."""
    import subprocess

    from PIL import ImageFont
    cands = []
    try:
        cands.append(subprocess.run(["fc-match", "-f", "%{file}", "serif:bold"], capture_output=True, text=True, timeout=5).stdout.strip())
    except Exception:
        pass
    cands += ["/usr/share/fonts/TTF/DejaVuSerif-Bold.ttf", "/usr/share/fonts/noto/NotoSerif-Bold.ttf",
              "/usr/share/fonts/liberation/LiberationSerif-Bold.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSerif-Bold.ttf",
              "/System/Library/Fonts/Supplemental/Georgia Bold.ttf", "C:/Windows/Fonts/georgiab.ttf"]
    for c in cands:
        if c and Path(c).exists():
            try:
                return ImageFont.truetype(c, size)
            except OSError:
                pass
    return ImageFont.load_default(size=size)


def _wrap(d, text: str, font, max_w: int) -> list[str]:
    lines, cur = [], ""
    for w in text.split():
        t = f"{cur} {w}".strip()
        if cur and d.textlength(t, font=font) > max_w:
            lines.append(cur); cur = w
        else:
            cur = t
    return lines + ([cur] if cur else [])


def _tracked(d, xy, text, font, fill, track: int, anchor_center: bool = True):
    """Text with extra letter-spacing, centred on xy[0]."""
    widths = [d.textlength(c, font=font) for c in text]
    total = sum(widths) + track * (len(text) - 1)
    x = xy[0] - total / 2 if anchor_center else xy[0]
    for c, w in zip(text, widths):
        d.text((x, xy[1]), c, font=font, fill=fill, anchor="ls")
        x += w + track


def make_cover(path: Path, title: str, author: str) -> Path:
    """A 1400x1400 placeholder cover: navy gradient, double gold frame, auto-fitted serif title, divider, author."""
    from PIL import Image, ImageDraw
    S, N = 2, 1400                                    # drawn at 2x and scaled down, for smooth edges
    W = N * S
    gold, cream, mute = (212, 175, 90), (244, 237, 218), (160, 140, 96)
    img = Image.new("RGB", (W, W))
    px = img.load()
    for y in range(W):                                # vertical gradient, a little lighter in the middle
        t = abs(y / W - 0.5) * 2
        c = tuple(int(a + (b - a) * t) for a, b in zip((38, 50, 88), (18, 24, 44)))
        for x in range(W):
            px[x, y] = c
    d = ImageDraw.Draw(img)
    d.rectangle([60 * S, 60 * S, (N - 60) * S, (N - 60) * S], outline=gold, width=7 * S)
    d.rectangle([84 * S, 84 * S, (N - 84) * S, (N - 84) * S], outline=mute, width=2 * S)
    cx = W // 2
    box_w = (N - 2 * 170) * S                         # text column, clear of the frame

    # title: the biggest size at which it wraps to at most 4 lines and fits the column
    title = (title or "Untitled").strip()
    for size in range(230, 80, -6):
        f = _serif(size * S)
        lines = _wrap(d, title, f, box_w)
        if len(lines) <= 4 and max(d.textlength(l, font=f) for l in lines) <= box_w and len(lines) * size * 1.18 <= 640:
            break
    lh = int(size * S * 1.18)
    cap = int(size * S * 0.72)                        # height of a capital letter
    authors = _wrap(d, (author or "").strip().upper(), _serif(54 * S), box_w - 120 * S)[:2] if (author or "").strip() else []
    total = cap + (len(lines) - 1) * lh + ((80 + 82 * len(authors)) * S if authors else 0)
    y = (W - total) // 2 + cap - 10 * S               # first baseline: the whole block is centred, a touch high
    for l in lines:
        d.text((cx, y), l, font=f, fill=cream, anchor="ms")
        y += lh
    y -= lh                                           # back to the last baseline
    if authors:
        y += 80 * S                                   # divider: a rule with a diamond in the middle
        d.line([cx - 230 * S, y, cx - 22 * S, y], fill=gold, width=3 * S)
        d.line([cx + 22 * S, y, cx + 230 * S, y], fill=gold, width=3 * S)
        d.polygon([(cx, y - 12 * S), (cx + 12 * S, y), (cx, y + 12 * S), (cx - 12 * S, y)], fill=gold)
        af = _serif(54 * S)
        for line in authors:
            y += 82 * S
            _tracked(d, (cx, y), line, af, gold, 6 * S)
    _tracked(d, (cx, 190 * S), "AUDIOBOOK", _serif(30 * S), mute, 12 * S)
    img.resize((N, N), Image.LANCZOS).save(path, "JPEG", quality=92)
    return path


def assemble(work: Path, cfg: dict, out_path: Path, cover: str | None = None,
             title: str | None = None, author: str | None = None, only_chapters=None) -> Path:
    meta = json.loads((work / "chapters.json").read_text())
    from .synth import load_segments
    segs = load_segments(work, cfg)
    clips = json.loads((work / "clips.json").read_text())
    sr = cfg["sample_rate"]
    title, author = title or meta["title"], author or meta.get("author", "")
    (work / "chapters").mkdir(exist_ok=True)

    chap_info, listing = [], []
    for ch in meta["chapters"]:
        if only_chapters and ch["index"] not in only_chapters:
            continue
        cs = [s for s in segs if s["chapter"] == ch["index"]]
        audio = build_chapter(cs, clips, work / "clips", cfg)
        peak = np.abs(audio).max() or 1.0
        audio = audio * min(1.0, 0.7 / peak)  # peak-limit to ~ -3 dBFS
        wav = work / "chapters" / f"{ch['index']:03d}.wav"
        sf.write(wav, audio, sr)
        chap_info.append((ch["title"], len(audio) / sr))
        listing.append(f"file '{wav.resolve()}'")
        print(f"[assemble] {ch['title']}: {len(audio) / sr:.1f}s")

    (work / "list.txt").write_text("\n".join(listing))
    (work / "chapters.ffmeta").write_text(ffmeta(chap_info, title, author))
    cover_path = Path(cover or meta.get("cover") or "") if (cover or meta.get("cover")) else \
        make_cover(work / "cover.jpg", title, author)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = ["ffmpeg", "-y", "-loglevel", "error",
           "-f", "concat", "-safe", "0", "-i", str(work / "list.txt"),
           "-i", str(work / "chapters.ffmeta"), "-i", str(cover_path),
           "-map", "0:a", "-map", "2:v", "-map_metadata", "1", "-map_chapters", "1",
           "-af", "loudnorm=I=-18:TP=-2:LRA=11", "-ar", str(sr), "-ac", "1",
           "-c:a", "aac", "-b:a", "64k", "-c:v", "mjpeg", "-disposition:v:0", "attached_pic",
           "-movflags", "+faststart", "-f", "ipod", str(out_path)]
    subprocess.run(cmd, check=True)
    return out_path
