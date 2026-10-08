"""Turn a voice into a native Kokoro voicepack.

A Kokoro voice is one style tensor [510, 1, 256] (row = phoneme count). There is no released encoder that maps
audio to that tensor, so it is *searched*: start from the stock voice that already sounds most like the target,
then nudge a shared 256-number offset with an evolution strategy, keeping changes that make Kokoro's speech
score a higher speaker-similarity (WavLM x-vector cosine) to the target recordings. The result is an ordinary
.pt voicepack that works anywhere Kokoro does, takes IPA, and runs ~7x faster than F5.
It captures timbre and general delivery, not a perfect copy; compare with the sample it writes."""
import json
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
PACKS = ROOT / "voices" / "kokoro"
SV_MODEL = "microsoft/wavlm-base-plus-sv"
STOCK = ["af_heart", "af_bella", "af_nicole", "af_sarah", "af_sky", "am_adam", "am_echo", "am_eric", "am_liam",
         "am_michael", "am_onyx", "am_puck", "bf_emma", "bf_isabella", "bm_daniel", "bm_fable", "bm_george", "bm_lewis"]
EVAL_TEXT = ["In the beginning was the Word, and the Word was with God.",
             "Verily, verily, I say unto thee, except a man be born again, he cannot see the kingdom of God."]
HELD_OUT = ["And the light shineth in darkness; and the darkness comprehended it not.",
            "The next day he went down to the river, and there was no one there but the old man.",
            "She said nothing for a long time, and then she asked him what he wanted."]  # never used while searching


def hf_share(audio: np.ndarray, sr: int, above: float = 4000.0) -> float:
    """Fraction of the energy above `above` Hz. Natural speech has very little; hiss, buzz and codec-like
    smear raise it."""
    from scipy.signal import welch
    f, P = welch(audio, sr, nperseg=2048)
    return float(P[f >= above].sum() / (P.sum() + 1e-12))


THIRDS = [100, 125, 160, 200, 250, 315, 400, 500, 630, 800, 1000, 1250, 1600, 2000, 2500, 3150, 4000, 5000, 6300, 8000]


def third_octave_db(audio: np.ndarray, sr: int) -> np.ndarray:
    """Long-term spectrum in 1/3-octave bands (dB), 100 Hz - 8 kHz."""
    from scipy.signal import welch
    f, P = welch(audio, sr, nperseg=4096)
    return np.array([10 * np.log10(P[(f >= fc / 2 ** (1 / 6)) & (f < fc * 2 ** (1 / 6))].sum() + 1e-14) for fc in THIRDS])


def envelope_db(audio: np.ndarray, sr: int) -> np.ndarray:
    """Average spectral envelope of the voiced frames in 1/3-octave bands (dB). Cepstral smoothing removes the
    pitch harmonics, so voices with different pitch can be compared by tone alone."""
    from scipy.signal import stft
    f, _, Z = stft(audio, sr, nperseg=1024, noverlap=768)
    mag = np.abs(Z) + 1e-8
    frame_e = (mag ** 2).sum(0)
    keep = frame_e > 0.1 * np.percentile(frame_e, 90)                  # speech frames only
    logm = np.log(mag[:, keep])
    ceps = np.fft.irfft(logm, axis=0)
    ceps[40:-40 or None] = 0                                           # keep slowly varying structure only
    smooth = np.fft.rfft(ceps, axis=0).real                            # back to a smooth log spectrum
    env = 20 / np.log(10) * smooth.mean(1)                             # mean in dB over frames
    return np.array([env[(f >= fc / 2 ** (1 / 6)) & (f < fc * 2 ** (1 / 6))].mean() for fc in THIRDS])


def log_f0(audio: np.ndarray, sr: int) -> np.ndarray:
    """ln(F0) of the voiced frames (pyin). Pitch level and how much it moves are a big part of who a voice
    sounds like; the speaker embedding barely sees either."""
    import librosa
    y = audio.astype("float32")
    f, _, _ = librosa.pyin(y, fmin=55, fmax=260, sr=sr, frame_length=2048, hop_length=480)
    rms = librosa.feature.rms(y=y, frame_length=2048, hop_length=480)[0][: len(f)]
    loud = rms > 0.1 * np.percentile(rms, 95)      # the tracker finds 'pitch' in room noise and silence
    return np.log(f[~np.isnan(f) & loud])


def activity(audio: np.ndarray, sr: int) -> tuple[float, float]:
    """(seconds of speech, total seconds): frames well above the noise floor count as speech."""
    fr = int(sr * 0.02)
    n = len(audio) // fr
    rms = np.sqrt((audio[: n * fr].reshape(n, fr) ** 2).mean(1))
    return float((rms > max(0.01, 0.1 * np.percentile(rms, 95))).sum() * 0.02), n * 0.02


def pace_of(audios: list[np.ndarray], sr: int, words: int) -> tuple[float, float]:
    """(words per second of speech, share of time that is pause) over several clips."""
    act = sum(activity(a, sr)[0] for a in audios)
    tot = sum(activity(a, sr)[1] for a in audios)
    return words / max(act, 1e-6), 1 - act / max(tot, 1e-6)


def match_eq(generated: list[np.ndarray], target: list[np.ndarray], sr: int, max_boost: float = 6.0,
             passes: int = 2) -> dict:
    """EQ (1/3-octave gains in dB) that moves the voice's spectral envelope to the target's. Two passes: the EQ
    is applied, the remaining difference measured again and added. Limited to +-6 dB (+3 dB above 4 kHz, so
    codec hiss isn't chased) and smoothed."""
    t = envelope_db(np.concatenate(target), sr)
    mid = slice(THIRDS.index(400), THIRDS.index(1250) + 1)
    cap = np.where(np.array(THIRDS) >= 4000, 3.0, max_boost)
    gain = np.zeros(len(THIRDS))
    for _ in range(passes):
        eq = {"freqs": THIRDS, "gains_db": list(gain)}
        g = envelope_db(np.concatenate([apply_eq(a, sr, eq) for a in generated]), sr)
        step = (t - t[mid].mean()) - (g - g[mid].mean())
        step = np.convolve(np.pad(step, 1, mode="edge"), np.ones(3) / 3, mode="valid")
        gain = np.clip(gain + step, -max_boost, cap)
    return {"freqs": THIRDS, "gains_db": [round(float(x), 1) for x in gain]}


def apply_eq(audio: np.ndarray, sr: int, eq: dict | None) -> np.ndarray:
    if not eq or not len(audio):
        return audio
    from scipy.signal import istft, stft
    f, _, Z = stft(audio, sr, nperseg=1024, noverlap=768)
    gdb = np.interp(np.log(np.maximum(f, 1.0)), np.log(eq["freqs"]), eq["gains_db"])      # flat outside 100 Hz - 8 kHz
    gdb = np.where(f > 12000, 0.0, gdb)
    _, y = istft(Z * (10 ** (gdb / 20))[:, None], sr, nperseg=1024, noverlap=768)
    y = y[: len(audio)].astype("float32")
    peak = np.abs(y).max()
    return y * (0.98 / peak) if peak > 0.98 else y


def formants(audio: np.ndarray, sr: int) -> list[float]:
    """Median F1-F3 (Hz) over loud, voiced frames, from LPC roots. Vowel colour: a voice can match in pitch and
    broad tone and still sound 'shifted' if its formants sit higher or lower."""
    import librosa
    y = librosa.resample(audio.astype("float32"), orig_sr=sr, target_sr=10000)
    y = np.append(y[0], y[1:] - 0.97 * y[:-1])
    n, hop = 250, 125
    frames = [y[i:i + n] * np.hanning(n) for i in range(0, len(y) - n, hop)]
    rms = np.array([np.sqrt((f ** 2).mean()) for f in frames])
    found = []
    for f, r in zip(frames, rms):
        if r < 0.5 * np.percentile(rms, 90):
            continue
        z = [x for x in np.roots(librosa.lpc(f, order=12)) if x.imag > 0]
        fr = sorted((np.angle(x) * 10000 / (2 * np.pi), -np.log(abs(x)) * 10000 / np.pi) for x in z)
        fr = [h for h, bw in fr if h > 200 and bw < 400]
        if len(fr) >= 3:
            found.append(fr[:3])
    return [round(float(x)) for x in np.median(np.array(found), axis=0)] if found else []


def trim_pauses(audio: np.ndarray, sr: int, scale: float, keep: float = 0.12) -> np.ndarray:
    """Shorten every silent stretch longer than `keep` seconds: the part beyond `keep` is multiplied by `scale`."""
    if not len(audio):
        return audio
    fr = int(sr * 0.02)
    n = len(audio) // fr
    rms = np.sqrt((audio[: n * fr].reshape(n, fr) ** 2).mean(1))
    quiet = rms <= max(0.01, 0.1 * np.percentile(rms, 95))
    out, i = [], 0
    while i < n:
        j = i
        while j < n and quiet[j] == quiet[i]:
            j += 1
        seg = audio[i * fr:j * fr]
        if quiet[i] and len(seg) > keep * sr:
            if i == 0 or j == n:      # silence at either end of the clip: assembly adds its own pauses between clips
                seg = seg[len(seg) - int(keep * sr):] if i == 0 else seg[: int(keep * sr)]
            else:
                k = int(keep * sr + (len(seg) - keep * sr) * scale)
                seg = np.concatenate([seg[: k // 2], seg[len(seg) - (k - k // 2):]])
        out.append(seg)
        i = j
    tail = audio[n * fr:]
    return np.concatenate(out + [tail])


def shift_pitch(audio: np.ndarray, sr: int, ratio: float) -> np.ndarray:
    """Plain resampling: pitch and formants move together by `ratio` (<1 = lower, darker) and the clip gets
    longer by 1/ratio. No phase vocoder, so no echo; the voice's speed setting makes up the time."""
    if not ratio or abs(ratio - 1) < 0.003 or not len(audio):
        return audio
    from fractions import Fraction

    from scipy.signal import resample_poly
    fr = Fraction(1 / ratio).limit_denominator(200)
    return resample_poly(audio.astype("float32"), fr.numerator, fr.denominator).astype("float32")


def shift_formants(audio: np.ndarray, sr: int, ratio: float) -> np.ndarray:
    """Move the spectral envelope (vowel colour) by `ratio` (<1 = darker, bigger-sounding) and leave the harmonics,
    so the pitch stays: the smooth envelope is split off by cepstral liftering, warped along frequency, put back."""
    if not ratio or abs(ratio - 1) < 0.005 or not len(audio):
        return audio
    from scipy.signal import istft, stft
    f, _, Z = stft(audio, sr, nperseg=1024, noverlap=768)
    mag = np.abs(Z) + 1e-8
    ceps = np.fft.irfft(np.log(mag), axis=0)
    ceps[28:-28] = 0
    env = np.exp(np.fft.rfft(ceps, axis=0).real)
    warped = np.stack([np.interp(f / ratio, f, env[:, t]) for t in range(env.shape[1])], axis=1)   # env'(f) = env(f/ratio)
    _, y = istft(Z * warped / env, sr, nperseg=1024, noverlap=768)
    y = y[: len(audio)].astype("float32")
    peak = np.abs(y).max()
    return y * (0.98 / peak) if peak > 0.98 else y


def finish(audio: np.ndarray, sr: int, meta: dict) -> np.ndarray:
    """Everything a voicepack applies after Kokoro: tone EQ, tighter pauses, a small pitch trim."""
    out = apply_eq(audio, sr, meta.get("eq"))
    if "pause_scale" in meta:
        out = trim_pauses(out, sr, meta["pause_scale"])
    return shift_pitch(out, sr, meta.get("pitch_ratio", 1.0))


def refine(name: str, target_clips, device: str | None = None, log=print) -> dict:
    """Measure the finished voice against the real clips on unseen sentences and store corrections in its json:
    a pitch ratio (applied by resampling, which also lowers the formants), a speed that gives the real speaking
    rate afterwards, and silence at clip edges trimmed."""
    import json

    import torch
    meta = json.loads((PACKS / f"{name}.json").read_text())
    dev = device or ("xpu" if torch.xpu.is_available() else "cpu")
    sy, sr = Synth(dev), 24000
    pack, lang = torch.load(PACKS / f"{name}.pt", weights_only=True), meta.get("lang", "a")
    base = {k: v for k, v in meta.items() if k not in ("pitch_ratio", "pause_scale", "pitch_cents", "formant_ratio")}
    base["pause_scale"] = 1.0
    words = sum(len(t.split()) for t in HELD_OUT)
    lt = np.concatenate([log_f0(a, r) for a, r in target_clips])
    pace_t = (meta.get("pace_target") or [None])[0]

    def say(t, speed):
        return np.concatenate([x.detach().cpu().numpy() for _, _, x in sy.pipes[lang](t, voice=pack, speed=speed) if x is not None])
    speed = meta.get("speed", 1.0) if pace_t else 1.0
    ratio = 1.0
    for _ in range(4):
        gen = [finish(say(t, speed), sr, base) for t in HELD_OUT]
        ratio *= float(np.exp(lt.mean() - np.concatenate([log_f0(shift_pitch(a, sr, ratio), sr) for a in gen]).mean()))
        out = [shift_pitch(a, sr, ratio) for a in gen]
        if pace_t:
            speed = float(np.clip(speed * pace_t / pace_of(out, sr, words)[0], 0.7, 1.4))
    meta = {k: v for k, v in meta.items() if k not in ("pitch_cents", "formant_ratio")}
    ft = formants(np.concatenate([a for a, _ in target_clips]), target_clips[0][1])
    fg = formants(np.concatenate(out), sr)
    meta.update(pitch_ratio=round(ratio, 4), pause_scale=1.0, speed=round(speed, 3), formants_target=ft, formants_voice=fg)
    (PACKS / f"{name}.json").write_text(json.dumps(meta, indent=2))
    log(f"pitch ratio {ratio:.3f}, speed {speed:.3f}, formants real {ft} voice {fg}")
    return meta


def band_profile(audio: np.ndarray, sr: int) -> np.ndarray:
    """Energy of the 1-2, 2-4 and 4-8 kHz bands in dB relative to 500-1000 Hz: the shape of the upper mids.
    A voice that scores well but scoops 2-4 kHz sounds hollow ("crushed mids")."""
    from scipy.signal import welch
    f, P = welch(audio, sr, nperseg=4096)
    e = lambda lo, hi: P[(f >= lo) & (f < hi)].sum() + 1e-12
    ref = e(500, 1000)
    return np.array([10 * np.log10(e(lo, hi) / ref) for lo, hi in ((1000, 2000), (2000, 4000), (4000, 8000))])


def lowpass(audio: np.ndarray, sr: int, hz: float = 6000.0) -> np.ndarray:
    from scipy.signal import butter, sosfiltfilt
    return sosfiltfilt(butter(6, hz, btype="low", fs=sr, output="sos"), audio).astype("float32")


class Scorer:
    """Speaker-similarity of audio to a target, using WavLM's speaker-verification head."""

    def __init__(self, device: str):
        import torch
        from transformers import AutoFeatureExtractor, WavLMForXVector
        self.torch, self.device = torch, device
        self.fe = AutoFeatureExtractor.from_pretrained(SV_MODEL)
        self.model = WavLMForXVector.from_pretrained(SV_MODEL).to(device).eval()

    def embed(self, audio: np.ndarray, sr: int):
        torch = self.torch
        audio = lowpass(audio, sr)   # judge the voice (formants, pitch), not the top octave where codec artifacts live
        if sr != 16000:
            from math import gcd

            from scipy.signal import resample_poly
            g = gcd(16000, sr)
            audio = resample_poly(audio, 16000 // g, sr // g).astype("float32")
        x = self.fe(audio, sampling_rate=16000, return_tensors="pt", padding=True).to(self.device)
        with torch.no_grad():
            e = self.model(**x).embeddings
        return torch.nn.functional.normalize(e, dim=-1)[0]

    def embed_t(self, audio, sr: int = 24000):
        """Differentiable version of embed() for torch audio: band-limit, resample, normalise like the feature
        extractor, then the x-vector head. Gradients reach the audio (and so the voice)."""
        import torchaudio.functional as AF
        x = AF.lowpass_biquad(audio, sr, 6000.0)
        x = AF.resample(x, sr, 16000)
        x = (x - x.mean()) / (x.std() + 1e-7)
        e = self.model(input_values=x[None]).embeddings
        return self.torch.nn.functional.normalize(e, dim=-1)[0]

    def target(self, clips: list[tuple[np.ndarray, int]]):
        e = self.torch.stack([self.embed(a, sr) for a, sr in clips]).mean(0)
        return self.torch.nn.functional.normalize(e, dim=-1)

    def score(self, audios: list[np.ndarray], sr: int, target) -> float:
        return float(np.mean([float(self.embed(a, sr) @ target) for a in audios]))


class Synth:
    """Fast Kokoro calls for search: phonemes are computed once, the pack is swapped each time."""

    def __init__(self, device: str):
        from kokoro import KModel, KPipeline
        self.model = KModel().to(device).eval()
        self.pipes = {c: KPipeline(lang_code=c, model=self.model) for c in "ab"}
        self.ps = {}

    def say(self, pack, text: str, lang: str):
        import torch
        key = (lang, text)
        if key not in self.ps:
            self.ps[key] = self.pipes[lang].g2p(text)[0]
        ps = self.ps[key]
        with torch.no_grad():
            audio = self.model(ps, pack[len(ps) - 1].to(self.model.device if hasattr(self.model, "device") else "cpu"), 1.0)
        return audio.detach().cpu().numpy()

    def stock(self, name: str):
        import torch
        from huggingface_hub import hf_hub_download
        return torch.load(hf_hub_download("hexgrad/Kokoro-82M", f"voices/{name}.pt"), weights_only=True)


def _gradient_delta(sy, sc, base, lang, tgt_emb, target_audio, steps, lr, progress, dev):
    """Gradient descent on a shared 256-number offset: the loss compares the voice's speaker embedding AND its
    average mel spectrum (timbre) with the target's. Kokoro's rounded durations carry no gradient, so pace is
    handled separately; everything else (pitch, energy, timbre) is differentiable."""
    import torch
    import torchaudio
    from kokoro import KModel

    for m in (sy.model, sc.model):
        for q in m.parameters():
            q.requires_grad_(False)
    mel = torchaudio.transforms.MelSpectrogram(24000, n_fft=1024, hop_length=256, n_mels=80, f_min=50.0, f_max=8000.0).to(dev)

    def stats(a):
        m = torch.log(mel(a) + 1e-6)
        e = m.exp().sum(0)
        keep = e > 0.1 * torch.quantile(e, 0.9)
        mm = m[:, keep]
        return mm.mean(1), mm.std(1)

    with torch.no_grad():
        ts = [stats(torch.as_tensor(a, dtype=torch.float32, device=dev)) for a in target_audio]
        t_mu, t_sd = torch.stack([x[0] for x in ts]).mean(0), torch.stack([x[1] for x in ts]).mean(0)
    delta = torch.zeros(1, 1, 256, requires_grad=True)
    opt = torch.optim.Adam([delta], lr=lr)
    texts = EVAL_TEXT
    vocab = sy.model.vocab
    log = []
    for it in range(steps):
        text = texts[it % len(texts)]
        key = (lang, text)
        if key not in sy.ps:
            sy.ps[key] = sy.pipes[lang].g2p(text)[0]
        ps = sy.ps[key]
        ids = torch.LongTensor([[0, *[vocab[c] for c in ps if c in vocab], 0]]).to(dev)
        ref = (base + delta)[len(ps) - 1].to(dev)
        audio, _ = KModel.forward_with_tokens.__wrapped__(sy.model, ids, ref, 1.0)   # the no_grad wrapper removed
        audio = audio.squeeze()
        mu, sd = stats(audio)
        l_mel = (mu - t_mu).abs().mean() + 0.5 * (sd - t_sd).abs().mean()
        l_spk = 1 - sc.embed_t(audio) @ tgt_emb
        loss = 2.0 * l_spk + 0.3 * l_mel + 0.5 * (delta ** 2).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
        log.append((float(l_spk), float(l_mel)))
        if progress and it % 5 == 0:
            progress(it / steps, f"Gradient step {it + 1}/{steps}: speaker gap {l_spk.item():.3f}, timbre gap {l_mel.item():.3f}")
    return delta.detach(), log


def search(target_clips, name: str, steps: int = 150, seed: int = 0, progress=None, device: str | None = None,
           init_pack: str | Path | None = None, match_pace: bool = True, match_tone: bool = True, method: str = "es", lr: float = 0.01, match_pitch: bool = True) -> dict:
    """Find a voicepack for the target recordings; writes voices/kokoro/<name>.pt (+ .json). Returns a report.
    init_pack: a voicepack to start from (e.g. one made earlier); it is used only if it scores better on this
    target than every stock voice, so a warm start can save steps but never hurts."""
    import torch

    from .tts.device import pick_device
    dev = device or pick_device("auto")
    sc, sy = Scorer(dev), Synth(dev)
    Path(PACKS).mkdir(parents=True, exist_ok=True)
    tgt = sc.target(target_clips)
    sr = 24000

    target_hf = float(np.mean([hf_share(a, r) for a, r in target_clips]))
    target_audio = [a for a, _ in target_clips]
    target_sr = target_clips[0][1]
    pace_t = None
    if match_pace:   # the real speaker's rate and pause share, from Whisper's word count of each clip
        import tempfile

        import soundfile as sf

        from .m4b import free_asr, transcribe_clip
        words = 0
        for a, r in target_clips:
            tmp = Path(tempfile.mkdtemp()) / "c.wav"
            sf.write(tmp, a, r, subtype="PCM_16")
            words += len(transcribe_clip(tmp, "openai/whisper-small.en", "auto").split())
        free_asr()
        pace_t = pace_of(target_audio, target_sr, words)
        if progress:
            progress(0.015, f"Target pace: {pace_t[0]:.2f} words/s of speech, {pace_t[1]:.0%} pauses")
    f0_t = None
    if match_pitch:
        lf = np.concatenate([log_f0(a, r) for a, r in target_clips])
        f0_t = (float(lf.mean()), float(lf.std()))
        if progress:
            progress(0.016, f"Target pitch: {np.exp(f0_t[0]):.0f} Hz, varying by {100 * f0_t[1]:.0f}%")

    def f0_pen(audios):
        if not f0_t:
            return 0.0
        lf = np.concatenate([log_f0(a, sr) for a in audios])
        if len(lf) < 30:
            return 0.5
        return 1.5 * abs(float(lf.mean()) - f0_t[0]) + 1.5 * abs(float(lf.std()) - f0_t[1])

    def synth_all(pack, lang, texts=EVAL_TEXT):
        return [sy.say(pack, t, lang) for t in texts]

    # allowed distance from the starting voice: about the gap between two real stock voices (kept in the
    # same family of sounds, so the search can't wander into unnatural territory)
    means = torch.stack([sy.stock(v).mean(0)[0] for v in STOCK])
    radius = float(torch.cdist(means, means).median()) * 0.9

    def evaluate(pack, lang, delta_norm=0.0, hf_ref=None):
        audios = synth_all(pack, lang)
        score = sc.score(audios, sr, tgt) - f0_pen(audios)
        if hf_ref is None:
            return score
        hf = float(np.mean([hf_share(a, sr) for a in audios]))
        pen_hf = 0.12 * max(0.0, hf / hf_ref - 1.3)                  # no more than ~30% above natural
        dev = np.abs(np.mean([band_profile(a, sr) for a in audios], axis=0) - base_profile)
        pen_hf += 0.03 * float(np.maximum(0.0, dev - 3.0).sum() / 3.0)   # keep each upper-mid band within 3 dB of the start
        if pace_t:
            rate, pause = pace_of(audios, sr, sum(len(t.split()) for t in EVAL_TEXT))
            pen_hf += 0.25 * abs(float(np.log(rate / pace_t[0]))) + 0.15 * abs(pause - pace_t[1])   # pacing like the speaker
        pen_norm = 0.10 * max(0.0, delta_norm / radius - 1.0)
        return score - pen_hf - pen_norm

    # 1. which stock voice is already closest?
    ranked = []
    for i, v in enumerate(STOCK):
        if progress:
            progress(0.02 + 0.08 * i / len(STOCK), f"Scoring stock voice {v}")
        ranked.append((evaluate(sy.stock(v), v[0] if v[0] in "ab" else "a"), v))
    ranked.sort(reverse=True)
    best_stock = ranked[0]
    lang = best_stock[1][0]
    top2 = (sy.stock(ranked[0][1]) + sy.stock(ranked[1][1])) / 2 if ranked[1][1][0] == lang else None
    base = sy.stock(best_stock[1])
    base_score = best_stock[0]
    started_from = best_stock[1]
    if init_pack:
        warm = torch.load(init_pack, weights_only=True)
        ws = evaluate(warm, lang)
        if ws > base_score:
            base, base_score, started_from = warm, ws, f"warm start {Path(init_pack).name}"
        if progress:
            progress(0.11, f"Warm start scores {ws:.3f} vs best stock voice {best_stock[0]:.3f}")
    if top2 is not None:
        s2 = evaluate(top2, lang)
        if s2 > base_score:
            base, base_score = top2, s2

    # 2. (1+4)-ES with antithetic pairs on a shared offset added to every row of the pack, penalised for
    #    raising high-frequency energy above natural speech or straying far from the starting voice
    base_hf = float(np.mean([hf_share(a, sr) for a in synth_all(base, lang)]))
    hf_ref = max(target_hf, 0.004)   # the real speaker's level, even if the starting voice is brighter
    base_profile = np.mean([band_profile(a, sr) for a in synth_all(base, lang)], axis=0)
    g = torch.Generator().manual_seed(seed)
    delta = torch.zeros(1, 1, 256)
    cur, sigma, history = evaluate(base, lang, 0.0, hf_ref), 0.02, [base_score]
    t0 = time.time()
    used = "evolution strategy"
    if method == "gradient":
        try:
            delta, glog = _gradient_delta(sy, sc, base, lang, tgt, target_audio, steps, lr,
                                          lambda f, d: progress(0.10 + 0.85 * f, d) if progress else None, dev)
            cur = evaluate(base + delta, lang, float(delta.norm()), hf_ref)
            history = [base_score, cur]
            used, steps = "gradient descent", 0           # skip the ES loop below
        except Exception as ex:   # an op without a backward on this device: fall back to the slower search
            print(f"[voicepack] gradient search unavailable ({ex!r:.120}); using the evolution strategy")
            delta = torch.zeros(1, 1, 256)
    for it in range(steps):
        if progress:
            progress(0.10 + 0.85 * it / steps, f"Searching: step {it + 1}/{steps}, similarity {cur:.3f} (stock best {best_stock[0]:.3f})")
        eps = torch.randn(2, 1, 1, 256, generator=g) * sigma
        eps[..., 128:] *= 1.5          # the second half of the style vector drives pitch and rhythm
        cand = [delta + eps[0], delta - eps[0], delta + eps[1], delta - eps[1]]
        scores = [evaluate(base + c, lang, float(c.norm()), hf_ref) for c in cand]
        k = int(np.argmax(scores))
        if scores[k] > cur:
            delta, cur = cand[k], scores[k]
            sigma = min(sigma * 1.15, 0.08)
        else:
            sigma = max(sigma * 0.9, 0.003)
        history.append(cur)
    pack = (base + delta).clone()
    held = lambda p: sc.score(synth_all(p, lang, HELD_OUT), sr, tgt)   # sentences it was never tuned on
    held_stock, held_new = held(sy.stock(best_stock[1])), held(pack)
    hf_new = float(np.mean([hf_share(a, sr) for a in synth_all(pack, lang, HELD_OUT)]))
    extra = {}
    if pace_t:   # leftover rate difference becomes a speed setting stored with the voice
        rate, pause = pace_of(synth_all(pack, lang, HELD_OUT), sr, sum(len(t.split()) for t in HELD_OUT))
        extra.update(speed=round(float(np.clip(pace_t[0] / rate, 0.8, 1.25)), 3), pace_target=[round(pace_t[0], 2), round(pace_t[1], 3)],
                     pace_before_speed=[round(rate, 2), round(pause, 3)])
    if match_tone:   # tone: an EQ curve from the difference in long-term spectrum
        gen = synth_all(pack, lang, EVAL_TEXT + HELD_OUT)
        extra["eq"] = match_eq(gen, target_audio, sr)
        mid = slice(THIRDS.index(400), THIRDS.index(1250) + 1)
        rel = lambda e: e - e[mid].mean()
        t_env = rel(envelope_db(np.concatenate(target_audio), sr))
        extra["tone_error_db"] = {
            "before_eq": round(float(np.abs(rel(envelope_db(np.concatenate(gen), sr)) - t_env).mean()), 2),
            "after_eq": round(float(np.abs(rel(envelope_db(np.concatenate([apply_eq(a, sr, extra["eq"]) for a in gen]), sr)) - t_env).mean()), 2)}
    hf_stock = float(np.mean([hf_share(a, sr) for a in synth_all(sy.stock(best_stock[1]), lang, HELD_OUT)]))
    PACKS.mkdir(parents=True, exist_ok=True)
    torch.save(pack, PACKS / f"{name}.pt")
    report = {"name": name, "lang": lang, "start_voice": best_stock[1], "stock_similarity": round(best_stock[0], 4),
              "final_similarity": round(cur, 4), "started_from": started_from, "method": used, "steps": steps,
              "held_out_stock": round(held_stock, 4), "held_out_new": round(held_new, 4),
              **extra, "f0_target_hz": None if not f0_t else [round(float(np.exp(f0_t[0])), 1), round(f0_t[1], 3)], "band_db_start": [round(float(x), 1) for x in base_profile], "hf_share_target": round(target_hf, 4), "hf_share_stock": round(hf_stock, 4), "hf_share_new": round(hf_new, 4), "seconds": round(time.time() - t0),
              "top_stock": [(v, round(s, 3)) for s, v in ranked[:5]], "history": [round(h, 4) for h in history[::max(1, steps // 20)]]}
    (PACKS / f"{name}.json").write_text(json.dumps(report, indent=2))
    return report


def target_from_audiobook(path: str, around: float = 0.5, spread: float = 0.05, minutes: float = 4.0, clips: int = 6,
                          seed: int | None = None, outdir: Path | None = None, chapter: int | None = None):
    """A random snippet near `around` (0.5 = the middle) of the audiobook, as clean 8-12 s clips cut at pauses.
    Returns ([(audio, sr)...], info) ready for search()."""
    import random

    import soundfile as sf

    from . import m4b
    rng = random.Random(seed)
    chs = m4b.chapters(path)
    total = chs[-1]["end"]
    if chapter:   # a random stretch inside that chapter (30 s in from either end)
        ch = chs[int(chapter) - 1]
        lo, hi = ch["start"] + 30, ch["end"] - minutes * 60 - 30
        start = rng.uniform(lo, max(lo, hi))
    else:
        start = total * (around + rng.uniform(-spread, spread))
    window = m4b.extract_wav(path, start, start + minutes * 60, 24000, (outdir or m4b.CACHE) / "snippet.wav")
    spans = m4b.speech_windows(window, target=10.0, lo=7.0, hi=12.0, top=clips)
    out = []
    for k, (a, b) in enumerate(spans):
        wav = m4b.cut(path, start + a, start + b, (outdir or m4b.CACHE) / f"target_{k}.wav")
        audio, sr = sf.read(wav, dtype="float32")
        out.append((audio, sr))
    info = {"chapter": chapter, "start_s": round(start), "start_h": round(start / 3600, 2), "total_h": round(total / 3600, 2), "clips": len(out)}
    return out, info
