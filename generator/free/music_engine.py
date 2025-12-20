# generator/free/music_engine.py
"""
SoundFlow Music Engine – v5.0 (Production, Dynamic Genres)
High-quality synthesis, stereo-aware DSP, configurable binaural.
Supports:
- Dynamic per-request stem generation (render_stem)
- Many genre styles: Trance, House, Lounge, Techno, Deep, EDM, Chillout,
  Bass, Dance, Vocal-ish, Hard, Ambient, Synth, Classic, etc.
- Backward-compatible older exports used by remix_daily
- Safe filtering on short buffers (no SciPy filtfilt padlen crashes)
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Any, Optional, Tuple
import hashlib

import numpy as np
from scipy import signal
from scipy.io import wavfile

# =============================================================================
# CONSTANTS
# =============================================================================

SAMPLE_RATE = 44100
TWOPI = 2.0 * np.pi


# =============================================================================
# SEEDING (CRITICAL: fixes “everything sounds the same”)
# =============================================================================

def _stable_u32(s: str) -> int:
    h = hashlib.sha256(s.encode("utf-8")).digest()
    return int.from_bytes(h[:4], "big", signed=False)

def _seed_from(seed: Optional[str | int], *, salt: str = "", variant: int = 1) -> int:
    """
    Deterministic 32-bit seed used for all random decisions.
    If seed is None → still deterministic per salt+variant (but not recommended).
    """
    if seed is None:
        base = f"default:{salt}:v{variant}"
        return _stable_u32(base)

    if isinstance(seed, int):
        base = f"{seed}:{salt}:v{variant}"
        return _stable_u32(base)

    base = f"{seed}:{salt}:v{variant}"
    return _stable_u32(base)

def _rng(seed: Optional[str | int], *, salt: str = "", variant: int = 1) -> np.random.RandomState:
    return np.random.RandomState(_seed_from(seed, salt=salt, variant=variant))


# =============================================================================
# BASIC UTILS
# =============================================================================

def _as_float(audio: np.ndarray) -> np.ndarray:
    if not isinstance(audio, np.ndarray):
        audio = np.asarray(audio)
    if audio.dtype == np.int16:
        return audio.astype(np.float32) / 32768.0
    return audio.astype(np.float32, copy=False)

def normalize(audio: np.ndarray, target: float = 0.95) -> np.ndarray:
    x = _as_float(audio)
    peak = float(np.max(np.abs(x))) if x.size else 0.0
    return (x * (target / peak)).astype(np.float32) if peak > 1e-6 else x

def soft_clip(audio: np.ndarray, threshold: float = 0.9) -> np.ndarray:
    x = _as_float(audio)
    threshold = float(max(1e-6, threshold))
    return (np.tanh(x / threshold) * threshold).astype(np.float32)

def mono_sum(audio: np.ndarray) -> np.ndarray:
    x = _as_float(audio)
    if x.ndim == 2 and x.shape[1] == 2:
        return x.mean(axis=1).astype(np.float32)
    return x

def pan(audio: np.ndarray, amount: float) -> np.ndarray:
    x = mono_sum(audio)
    amount = float(np.clip(amount, -1.0, 1.0))
    left = x * np.cos((amount + 1.0) * np.pi / 4.0)
    right = x * np.sin((amount + 1.0) * np.pi / 4.0)
    return np.stack([left, right], axis=1).astype(np.float32)

def stereo_width(audio: np.ndarray, amount: float) -> np.ndarray:
    x = _as_float(audio)
    amount = float(np.clip(amount, 0.0, 2.0))
    if x.ndim != 2 or x.shape[1] != 2:
        x = np.stack([mono_sum(x), mono_sum(x)], axis=1)

    mid = x.mean(axis=1)
    side = (x[:, 0] - x[:, 1]) * 0.5
    side *= amount
    left = mid + side
    right = mid - side
    return np.stack([left, right], axis=1).astype(np.float32)

def save_wav(path: Path, audio: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    x = normalize(audio, 0.98)
    x = np.clip(x, -1.0, 1.0)
    wavfile.write(str(path), SAMPLE_RATE, (x * 32767.0).astype(np.int16))


# =============================================================================
# FILTERS + EQ (SAFE)
# =============================================================================

def _safe_filtfilt(b: np.ndarray, a: np.ndarray, x: np.ndarray) -> np.ndarray:
    """
    Safe filtfilt that falls back to lfilter if signal is too short.
    Works for mono (N,) and stereo (N,2).
    """
    x = _as_float(x)

    def _one(sig: np.ndarray) -> np.ndarray:
        sig = sig.reshape(-1).astype(np.float32, copy=False)
        padlen = 3 * (max(len(a), len(b)) - 1)
        if sig.size <= padlen + 1:
            return signal.lfilter(b, a, sig).astype(np.float32, copy=False)
        return signal.filtfilt(b, a, sig).astype(np.float32, copy=False)

    if x.ndim == 1:
        return _one(x)
    if x.ndim == 2 and x.shape[1] == 2:
        l = _one(x[:, 0])
        r = _one(x[:, 1])
        return np.stack([l, r], axis=1).astype(np.float32, copy=False)
    return _one(x.reshape(-1))

def apply_lowpass(audio: np.ndarray, cutoff: float, order: int = 4) -> np.ndarray:
    nyq = SAMPLE_RATE / 2.0
    cutoff = float(np.clip(cutoff, 10.0, nyq * 0.99))
    wn = cutoff / nyq
    b, a = signal.butter(int(order), wn, btype="low")
    return _safe_filtfilt(b, a, audio)

def apply_highpass(audio: np.ndarray, cutoff: float, order: int = 4) -> np.ndarray:
    nyq = SAMPLE_RATE / 2.0
    cutoff = float(np.clip(cutoff, 10.0, nyq * 0.99))
    wn = cutoff / nyq
    b, a = signal.butter(int(order), wn, btype="high")
    return _safe_filtfilt(b, a, audio)

def apply_parametric_eq(audio: np.ndarray, freq: float, gain_db: float, q: float = 1.0) -> np.ndarray:
    """
    Peaking EQ biquad (RBJ cookbook). Uses lfilter (safe for all lengths).
    """
    x = _as_float(audio)
    freq = float(np.clip(freq, 20.0, SAMPLE_RATE / 2.0 - 200.0))
    q = float(max(0.1, q))

    A = 10 ** (float(gain_db) / 40.0)
    w0 = TWOPI * freq / SAMPLE_RATE
    alpha = np.sin(w0) / (2.0 * q)

    b0 = 1.0 + alpha * A
    b1 = -2.0 * np.cos(w0)
    b2 = 1.0 - alpha * A
    a0 = 1.0 + alpha / A
    a1 = -2.0 * np.cos(w0)
    a2 = 1.0 - alpha / A

    b = np.array([b0, b1, b2], dtype=np.float64) / a0
    a = np.array([1.0, a1 / a0, a2 / a0], dtype=np.float64)

    if x.ndim == 1:
        return signal.lfilter(b, a, x).astype(np.float32, copy=False)
    if x.ndim == 2 and x.shape[1] == 2:
        l = signal.lfilter(b, a, x[:, 0]).astype(np.float32, copy=False)
        r = signal.lfilter(b, a, x[:, 1]).astype(np.float32, copy=False)
        return np.stack([l, r], axis=1).astype(np.float32, copy=False)
    return signal.lfilter(b, a, x.reshape(-1)).astype(np.float32, copy=False)

def apply_resonant_filter(audio: np.ndarray, cutoff: float, resonance: float = 0.3) -> np.ndarray:
    x = _as_float(audio)
    cutoff = float(np.clip(cutoff, 40.0, SAMPLE_RATE / 2.0 - 400.0))
    resonance = float(np.clip(resonance, 0.0, 0.95))
    peak_gain_db = 2.0 + 18.0 * resonance
    x = apply_parametric_eq(x, cutoff, peak_gain_db, q=1.0 + 6.0 * resonance)
    x = apply_lowpass(x, cutoff, order=4)
    return x.astype(np.float32)

def multiband_process(audio: np.ndarray, low_gain: float = 1.0, mid_gain: float = 1.0, high_gain: float = 1.0) -> np.ndarray:
    x = _as_float(audio)
    low = apply_lowpass(x, 250.0)
    mid = apply_lowpass(apply_highpass(x, 250.0), 4000.0)
    high = apply_highpass(x, 4000.0)
    return (low * float(low_gain) + mid * float(mid_gain) + high * float(high_gain)).astype(np.float32)


# =============================================================================
# SATURATION + REVERB + SIDECHAIN
# =============================================================================

def apply_overdrive(audio: np.ndarray, drive: float = 1.0) -> np.ndarray:
    x = _as_float(audio)
    drive = float(np.clip(drive, 0.0, 10.0))
    return (np.tanh(x * (1.0 + drive)) * 0.9).astype(np.float32)

def apply_algorithmic_reverb(audio: np.ndarray, room_size: float = 0.6, damping: float = 0.4, wet: float = 0.25) -> np.ndarray:
    x = _as_float(audio)
    wet = float(np.clip(wet, 0.0, 1.0))
    if wet <= 1e-6:
        return x

    def _verb(sig: np.ndarray) -> np.ndarray:
        sig = sig.reshape(-1).astype(np.float32, copy=False)
        comb_delays = [1557, 1617, 1491, 1422]
        comb_gains = np.array([0.805, 0.827, 0.783, 0.764], dtype=np.float32)
        comb_gains *= float(np.clip(room_size, 0.1, 1.2))
        comb_gains *= (1.0 - float(np.clip(damping, 0.0, 0.9)) * 0.35)

        y = np.zeros_like(sig, dtype=np.float32)
        for delay, gain in zip(comb_delays, comb_gains):
            buf = np.zeros_like(sig, dtype=np.float32)
            for i in range(delay, len(sig)):
                buf[i] = sig[i] + gain * buf[i - delay]
            y += buf * 0.25

        for ap_delay in [225, 556]:
            tmp = np.zeros_like(y, dtype=np.float32)
            for i in range(ap_delay, len(y)):
                tmp[i] = -y[i] + y[i - ap_delay] + 0.5 * tmp[i - ap_delay]
            y = tmp

        return (sig * (1.0 - wet) + y * wet).astype(np.float32)

    if x.ndim == 1:
        return _verb(x)
    if x.ndim == 2 and x.shape[1] == 2:
        l = _verb(x[:, 0])
        r = _verb(x[:, 1])
        return np.stack([l, r], axis=1).astype(np.float32)
    return _verb(x.reshape(-1))

def apply_sidechain_envelope(audio: np.ndarray, bpm: float, duck_amount: float = 0.7) -> np.ndarray:
    x = _as_float(audio)
    duck_amount = float(np.clip(duck_amount, 0.0, 1.0))
    bpm_f = float(max(1e-6, bpm))
    beat_samples = int((60.0 / bpm_f) * SAMPLE_RATE)

    env = np.ones(len(x) if x.ndim == 1 else x.shape[0], dtype=np.float32)
    duck_len = int(SAMPLE_RATE * 0.15)
    half = duck_len // 2
    rest = duck_len - half

    num_beats = env.size // beat_samples
    for i in range(num_beats):
        start = i * beat_samples
        end = start + duck_len
        if end >= env.size:
            break
        curve = np.concatenate(
            [
                np.linspace(1.0, duck_amount, half) ** 2,
                np.linspace(duck_amount, 1.0, rest) ** 2,
            ]
        ).astype(np.float32)
        env[start:end] = curve

    if x.ndim == 1:
        return (x * env).astype(np.float32)
    if x.ndim == 2 and x.shape[1] == 2:
        y = x.copy()
        y[:, 0] *= env
        y[:, 1] *= env
        return y.astype(np.float32)

    flat = x.reshape(-1).astype(np.float32, copy=False)
    return (flat * env[: flat.size]).astype(np.float32)


# =============================================================================
# MUSICAL HELPERS (KEYS / SCALES)
# =============================================================================

_NOTE_TO_SEMI = {
    "C": 0, "C#": 1, "DB": 1, "D": 2, "D#": 3, "EB": 3, "E": 4, "F": 5,
    "F#": 6, "GB": 6, "G": 7, "G#": 8, "AB": 8, "A": 9, "A#": 10, "BB": 10, "B": 11
}

def key_to_root_hz(key: str, octave: int = 3) -> float:
    """
    Rough mapping: A4=440, return key at chosen octave.
    octave=3 gives a musically useful root for chords/leads.
    """
    k = str(key).strip().upper()
    semi = _NOTE_TO_SEMI.get(k, 9)  # default A
    # C0 ~ 16.35 Hz; formula: f = 16.35 * 2^(n/12)
    c0 = 16.3516
    n = semi + 12 * int(octave)
    return float(c0 * (2 ** (n / 12.0)))

def semitone(freq: float, semi: float) -> float:
    return float(freq) * (2 ** (float(semi) / 12.0))


# =============================================================================
# OSCILLATORS + ENVELOPES
# =============================================================================

def _adsr(n: int, a: float, d: float, s: float, r: float) -> np.ndarray:
    aN = max(1, int(a * SAMPLE_RATE))
    dN = max(1, int(d * SAMPLE_RATE))
    rN = max(1, int(r * SAMPLE_RATE))
    if aN + dN + rN > n:
        scale = n / float(aN + dN + rN)
        aN = max(1, int(aN * scale))
        dN = max(1, int(dN * scale))
        rN = max(1, int(rN * scale))
    sN = max(0, n - aN - dN - rN)

    env = np.concatenate(
        [
            np.linspace(0, 1, aN) ** 0.5,
            np.linspace(1, s, dN) ** 2.0,
            np.full(sN, s, dtype=np.float32),
            np.linspace(s, 0, rN) ** 2.0,
        ]
    ).astype(np.float32)
    if env.size < n:
        env = np.pad(env, (0, n - env.size))
    return env[:n].astype(np.float32)

class Oscillator:
    def saw(self, freq: float, n: int, brightness: float = 1.0) -> np.ndarray:
        t = np.arange(n, dtype=np.float32) / SAMPLE_RATE
        brightness = float(np.clip(brightness, 0.2, 2.0))
        max_h = max(1, int(30 * brightness))
        out = np.zeros(n, dtype=np.float32)
        for k in range(1, max_h + 1):
            out += np.sin(TWOPI * float(freq) * k * t) / k
        return (out * (2.0 / np.pi)).astype(np.float32)

    def supersaw(self, freq: float, n: int, detune: float = 0.08, variation: float = 0.0) -> np.ndarray:
        voices = 7
        detune = float(np.clip(detune, 0.0, 0.2))
        variation = float(np.clip(variation, 0.0, 1.0))
        t = np.arange(n, dtype=np.float32) / SAMPLE_RATE
        det_lfo = np.sin(TWOPI * 0.05 * t) * (variation * 0.02)

        out = np.zeros(n, dtype=np.float32)
        offsets = np.linspace(-detune, detune, voices, dtype=np.float32)
        for dt in offsets:
            out += self.saw(float(freq) * (1.0 + dt + det_lfo), n, brightness=1.0)
        return (out / float(voices)).astype(np.float32)

osc = Oscillator()


# =============================================================================
# BASIC BUILDING BLOCKS
# =============================================================================

def _sine_kick(seconds: float = 0.45, f0: float = 160.0, f1: float = 45.0, drive: float = 1.4) -> np.ndarray:
    n = int(seconds * SAMPLE_RATE)
    t = np.arange(n, dtype=np.float32) / SAMPLE_RATE
    freq = float(f0) * np.exp(-t * 18.0) + float(f1)
    phase = np.cumsum(freq) * (TWOPI / SAMPLE_RATE)
    x = np.sin(phase) * np.exp(-t * 10.0)
    x = np.tanh(x * float(drive))
    return x.astype(np.float32)

def _noise_hit(n: int, hp: float, lp: float, decay: float, velocity: float) -> np.ndarray:
    noise = np.random.randn(int(n)).astype(np.float32)
    noise = apply_highpass(noise, float(hp))
    noise = apply_lowpass(noise, float(lp))
    env = np.exp(-np.linspace(0, float(decay), int(n))).astype(np.float32)
    return (noise * env * float(velocity)).astype(np.float32)

def _chord(freqs: list[float], n: int, brightness: float = 0.8) -> np.ndarray:
    x = np.zeros(int(n), dtype=np.float32)
    for f in freqs:
        x += osc.saw(float(f), int(n), brightness=float(brightness)) * 0.33
    return x.astype(np.float32)

def _bar_n(bpm: float, bars: float = 1.0) -> int:
    beat = 60.0 / float(max(1e-6, bpm))
    seconds = beat * 4.0 * float(bars)
    return int(seconds * SAMPLE_RATE)


# =============================================================================
# GENRE STYLES (ADD NEW GENRES HERE)
# =============================================================================

GENRE_STYLES: Dict[str, Dict[str, Any]] = {
    # energetic
    "techno": {"bpm": 130, "swing": 0.00, "brightness": 0.65, "reverb": 0.18, "duck": 0.55},
    "house": {"bpm": 124, "swing": 0.03, "brightness": 0.55, "reverb": 0.22, "duck": 0.45},
    "deep":  {"bpm": 122, "swing": 0.05, "brightness": 0.45, "reverb": 0.25, "duck": 0.40},
    "edm":   {"bpm": 128, "swing": 0.00, "brightness": 0.85, "reverb": 0.20, "duck": 0.60},
    "dance": {"bpm": 128, "swing": 0.01, "brightness": 0.70, "reverb": 0.18, "duck": 0.55},
    "hard":  {"bpm": 150, "swing": 0.00, "brightness": 0.75, "reverb": 0.12, "duck": 0.65},
    "bass":  {"bpm": 140, "swing": 0.00, "brightness": 0.70, "reverb": 0.15, "duck": 0.55},

    # melodic
    "trance": {"bpm": 138, "swing": 0.00, "brightness": 0.90, "reverb": 0.28, "duck": 0.55},
    "synth":  {"bpm": 105, "swing": 0.00, "brightness": 0.75, "reverb": 0.30, "duck": 0.40},

    # chill
    "chillout": {"bpm": 90, "swing": 0.06, "brightness": 0.35, "reverb": 0.35, "duck": 0.25},
    "lounge":   {"bpm": 100, "swing": 0.05, "brightness": 0.40, "reverb": 0.32, "duck": 0.25},
    "ambient":  {"bpm": 70, "swing": 0.00, "brightness": 0.25, "reverb": 0.45, "duck": 0.15},

    # special
    "classic": {"bpm": 90, "swing": 0.00, "brightness": 0.30, "reverb": 0.38, "duck": 0.10},
    "vocal":   {"bpm": 128, "swing": 0.02, "brightness": 0.65, "reverb": 0.28, "duck": 0.50},
}

def _norm_genre(genre: str) -> str:
    g = str(genre or "").strip().lower()
    g = g.replace("progressive house", "house")
    g = g.replace("deep house", "deep")
    g = g.replace("drum and bass", "bass")
    g = g.replace("dnb", "bass")
    g = g.replace("dubstep", "bass")
    g = g.replace("chill", "chillout")
    return g if g in GENRE_STYLES else "techno"


# =============================================================================
# DYNAMIC STEM GENERATORS (PRODUCTION)
# =============================================================================

def _humanize_steps(rnd: np.random.RandomState, n_steps: int, swing: float) -> np.ndarray:
    """
    Returns per-step timing offsets (in steps) small, deterministic.
    """
    swing = float(np.clip(swing, 0.0, 0.12))
    offs = rnd.uniform(-0.15, 0.15, size=n_steps).astype(np.float32) * swing
    # classic swing: delay odd 16ths a bit
    for i in range(n_steps):
        if i % 2 == 1:
            offs[i] += swing * 0.35
    return offs

def _render_kick(bpm: int, rnd: np.random.RandomState, *, variant: int, genre: str) -> np.ndarray:
    n = _bar_n(bpm, 1.0)
    beat = 60.0 / float(bpm)
    step = int((beat / 4.0) * SAMPLE_RATE)

    g = _norm_genre(genre)
    st = GENRE_STYLES[g]

    x = np.zeros(n, dtype=np.float32)
    # vary kick parameters
    f0 = 140 + rnd.randint(0, 60) + 8 * max(0, variant - 1)
    f1 = 38 + rnd.randint(0, 18)
    drive = 1.1 + (st["duck"] * 1.5) + rnd.uniform(0.0, 0.6)

    hit = _sine_kick(seconds=0.35 + rnd.uniform(0.0, 0.18), f0=f0, f1=f1, drive=drive)

    # 4-on-the-floor for most genres
    hits = [0, 4, 8, 12]
    if g in ("chillout", "ambient", "lounge", "classic"):
        hits = [0, 8]  # half-time
    if g in ("hard",):
        hits = [0, 4, 8, 12]  # still 4, but heavier

    for s in hits:
        pos = s * step
        end = min(n, pos + hit.size)
        x[pos:end] += hit[: end - pos]

    x = apply_highpass(x, 20.0)
    x = soft_clip(x * (1.05 + rnd.uniform(0.0, 0.25)), 0.88)
    return mono_sum(x)

def _render_drums(bpm: int, rnd: np.random.RandomState, *, variant: int, genre: str) -> np.ndarray:
    n = _bar_n(bpm, 1.0)
    beat = 60.0 / float(bpm)
    step = int((beat / 4.0) * SAMPLE_RATE)

    g = _norm_genre(genre)
    st = GENRE_STYLES[g]
    swing = float(st["swing"])

    x = np.zeros(n, dtype=np.float32)
    time_offs = _humanize_steps(rnd, 16, swing=swing)

    # components
    clap = _noise_hit(int((0.12 + rnd.uniform(0.0, 0.10)) * SAMPLE_RATE), 900.0, 4800.0, decay=18.0, velocity=0.65 + 0.2 * rnd.rand())
    hat  = _noise_hit(int((0.03 + rnd.uniform(0.0, 0.04)) * SAMPLE_RATE), 5500.0, 12000.0, decay=30.0, velocity=0.18 + 0.2 * rnd.rand())
    ohat = _noise_hit(int((0.06 + rnd.uniform(0.0, 0.05)) * SAMPLE_RATE), 4500.0, 11000.0, decay=12.0, velocity=0.22 + 0.2 * rnd.rand())

    # kick pattern (drums stem includes kick for house/dance/edm)
    kick = _render_kick(bpm, rnd, variant=variant, genre=genre)

    for i in range(16):
        # step timing
        pos = int((i + float(time_offs[i])) * step)
        if pos < 0:
            continue
        if pos >= n:
            break

        # kick
        if g not in ("ambient", "classic"):
            if i % 4 == 0:
                end = min(n, pos + kick.size // 16)  # short segment
                x[pos:end] += kick[: end - pos] * (0.9 + 0.2 * rnd.rand())

        # clap/snare on 2/4
        if i in (4, 12) and g not in ("ambient", "classic"):
            end = min(n, pos + clap.size)
            x[pos:end] += clap[: end - pos] * (0.7 + 0.2 * rnd.rand())

        # closed hats
        if g in ("techno", "house", "deep", "edm", "dance", "trance", "vocal", "bass", "hard"):
            if i % 2 == 1:
                end = min(n, pos + hat.size)
                x[pos:end] += hat[: end - pos] * (0.6 + 0.3 * rnd.rand())
            # open hat occasionally
            if i in (7, 15) and rnd.rand() < 0.6:
                end = min(n, pos + ohat.size)
                x[pos:end] += ohat[: end - pos] * (0.55 + 0.25 * rnd.rand())

        # chill hats
        if g in ("chillout", "lounge", "classic"):
            if i in (3, 11, 15) and rnd.rand() < 0.9:
                end = min(n, pos + hat.size)
                x[pos:end] += hat[: end - pos] * (0.35 + 0.15 * rnd.rand())

    # tone
    if g in ("lofi", "chillout", "lounge", "ambient", "classic"):
        x = apply_lowpass(x, 6500.0)
    else:
        x = apply_lowpass(x, 11000.0)

    x = soft_clip(x * (1.0 + 0.2 * rnd.rand()), 0.9)
    return x.astype(np.float32)

def _render_bass(bpm: int, key: str, rnd: np.random.RandomState, *, variant: int, genre: str) -> np.ndarray:
    n = _bar_n(bpm, 1.0)
    beat = 60.0 / float(bpm)
    step = int((beat / 4.0) * SAMPLE_RATE)
    note_len = int(step * (0.85 + 0.12 * rnd.rand()))

    g = _norm_genre(genre)
    st = GENRE_STYLES[g]
    root = key_to_root_hz(key, octave=2)

    # choose a scale-ish pool
    if g in ("trance", "edm", "dance", "vocal"):
        degrees = [0, 0, 7, 0, 5, 0, 7, 0, 0, 12, 0, 7, 0, 5, 0, 7]
    elif g in ("deep", "house", "lounge"):
        degrees = [0, 0, 0, 7, 0, 0, 5, 0, 0, 7, 0, 0, 3, 0, 0, 5]
    elif g in ("bass", "hard"):
        degrees = [0, 0, 7, 0, 10, 0, 7, 0, 0, 0, 12, 0, 10, 0, 7, 0]
    else:
        degrees = [0, 0, 7, 0, 0, 10, 0, 0, 7, 0, 0, 3, 0, 0, 10, 0]

    # mutate degrees with deterministic randomness
    if rnd.rand() < 0.75:
        for i in range(16):
            if rnd.rand() < (0.12 + 0.03 * variant):
                degrees[i] = rnd.choice([0, 3, 5, 7, 10, 12])

    x = np.zeros(n, dtype=np.float32)
    brightness = float(np.clip(st["brightness"], 0.25, 0.9))

    for i, semi in enumerate(degrees):
        pos = i * step
        end = min(n, pos + note_len)
        f = semitone(root, semi)
        tone = osc.saw(f, end - pos, brightness=brightness * 0.75)
        env = _adsr(end - pos, 0.005 + 0.01 * rnd.rand(), 0.04 + 0.06 * rnd.rand(), 0.25 + 0.25 * rnd.rand(), 0.03 + 0.05 * rnd.rand())
        x[pos:end] += tone * env

    # bass tone shaping
    if g in ("bass", "hard"):
        x = apply_overdrive(x, drive=1.2 + 1.0 * rnd.rand())
        x = apply_lowpass(x, 1800.0 + 800 * rnd.rand())
    elif g in ("deep", "house", "lounge"):
        x = apply_lowpass(x, 900.0 + 250 * rnd.rand())
    else:
        x = apply_resonant_filter(x, cutoff=450.0 + 800 * rnd.rand(), resonance=0.15 + 0.35 * rnd.rand())

    return mono_sum(x).astype(np.float32)

def _render_music_layer(bpm: int, key: str, rnd: np.random.RandomState, *, variant: int, genre: str) -> np.ndarray:
    """
    “Music” layer: arp/lead/chords depending on genre.
    """
    n = _bar_n(bpm, 1.0)
    beat = 60.0 / float(bpm)
    step16 = int((beat / 4.0) * SAMPLE_RATE)
    g = _norm_genre(genre)
    st = GENRE_STYLES[g]
    root = key_to_root_hz(key, octave=3)

    x = np.zeros(n, dtype=np.float32)

    # Progression (4 chords)
    if g in ("trance", "edm", "dance", "vocal"):
        # i - VI - III - VII (minor pop/trance feel)
        prog = [
            [0, 3, 7],
            [-3, 0, 4],
            [4, 7, 11],
            [7, 10, 14],
        ]
    elif g in ("deep", "house", "lounge"):
        # i - VI - iv - v
        prog = [
            [0, 3, 7],
            [-3, 0, 4],
            [-5, -2, 2],
            [-4, -1, 3],
        ]
    elif g in ("classic",):
        # more “diatonic-ish”
        prog = [
            [0, 4, 7],
            [5, 9, 12],
            [7, 11, 14],
            [0, 4, 7],
        ]
    else:
        prog = [
            [0, 3, 7],
            [0, 5, 10],
            [0, 3, 7],
            [0, 7, 10],
        ]

    # choose mode: arp vs chords vs pad
    if g in ("ambient", "chillout"):
        # long pad chords
        seg = n // 4
        for ci, chord in enumerate(prog):
            pos = ci * seg
            end = min(n, pos + seg)
            freqs = [semitone(root, s) for s in chord]
            tone = _chord(freqs, end - pos, brightness=0.35 + 0.15 * rnd.rand())
            env = _adsr(end - pos, 0.10, 0.15, 0.55, 0.25)
            x[pos:end] += tone * env * 0.5

        x = apply_lowpass(x, 5200.0)
        x = apply_algorithmic_reverb(x, room_size=0.85, wet=0.45)

    elif g in ("trance", "edm", "dance", "synth", "vocal"):
        # 16th arp/lead
        notes = [0, 3, 7, 10, 12, 10, 7, 3]
        note_len = int(step16 * (0.65 + 0.20 * rnd.rand()))
        detune = 0.08 + 0.06 * rnd.rand()

        for i in range(16):
            pos = i * step16
            end = min(n, pos + note_len)
            semi = notes[(i + rnd.randint(0, 2)) % len(notes)]
            f = semitone(root * (2 ** (rnd.choice([0, 0, 1]) )), semi)
            tone = osc.supersaw(f, end - pos, detune=detune, variation=0.2 + 0.5 * rnd.rand())
            env = _adsr(end - pos, 0.003, 0.02 + 0.03 * rnd.rand(), 0.22 + 0.25 * rnd.rand(), 0.04 + 0.06 * rnd.rand())
            x[pos:end] += tone * env * (0.55 + 0.20 * rnd.rand())

        # vocal-ish formants
        if g == "vocal":
            x = apply_parametric_eq(x, 800.0, 2.5, q=1.0)
            x = apply_parametric_eq(x, 1400.0, 2.0, q=1.0)
            x = apply_parametric_eq(x, 2600.0, 1.5, q=1.2)

        x = stereo_width(pan(x, 0.0), 1.6 + 0.3 * rnd.rand())
        x = apply_lowpass(x, 12000.0)
        x = apply_algorithmic_reverb(x, room_size=0.55 + 0.2 * rnd.rand(), wet=float(st["reverb"]))
        x = apply_sidechain_envelope(x, bpm, duck_amount=float(st["duck"]))

    else:
        # chord stabs / riffs
        chord_len = int((beat * (0.50 + 0.35 * rnd.rand())) * SAMPLE_RATE)
        gap = int(beat * SAMPLE_RATE)
        pos = 0
        for chord in prog:
            if pos + chord_len >= n:
                break
            freqs = [semitone(root, s) for s in chord]
            tone = _chord(freqs, chord_len, brightness=0.55 + 0.25 * rnd.rand())
            env = _adsr(chord_len, 0.005, 0.05 + 0.05 * rnd.rand(), 0.2, 0.08 + 0.10 * rnd.rand())
            x[pos:pos+chord_len] += tone * env * 0.6
            pos += gap

        x = stereo_width(pan(x, 0.0), 1.35 + 0.35 * rnd.rand())
        x = apply_algorithmic_reverb(x, room_size=0.60 + 0.25 * rnd.rand(), wet=float(st["reverb"]))
        x = apply_lowpass(x, 9000.0 + 4000.0 * rnd.rand())

    return x.astype(np.float32)

def _render_texture(seconds: float, rnd: np.random.RandomState, kind: str) -> np.ndarray:
    n = int(seconds * SAMPLE_RATE)
    kind = str(kind).lower().strip()

    if kind == "vinyl":
        x = rnd.randn(n).astype(np.float32) * 0.02
        x = apply_highpass(x, 80.0)
        x = apply_lowpass(x, 5500.0)
        # clicks
        for _ in range(10 + rnd.randint(0, 18)):
            pos = rnd.randint(0, max(1, n - 220))
            click = rnd.uniform(-1, 1, 220).astype(np.float32)
            click *= np.exp(-np.linspace(0, 10, 220)).astype(np.float32)
            x[pos:pos+220] += click * (0.04 + 0.07 * rnd.rand())
        return x.astype(np.float32)

    if kind == "rain":
        x = rnd.randn(n).astype(np.float32) * 0.05
        x = apply_highpass(x, 150.0)
        x = apply_lowpass(x, 6500.0)
        t = np.arange(n, dtype=np.float32) / SAMPLE_RATE
        x *= (0.70 + 0.30 * np.sin(TWOPI * (0.08 + 0.08 * rnd.rand()) * t)).astype(np.float32)
        return x.astype(np.float32)

    # fallback: air
    x = rnd.randn(n).astype(np.float32) * 0.015
    x = apply_highpass(x, 300.0)
    x = apply_lowpass(x, 8000.0)
    return x.astype(np.float32)


# =============================================================================
# MAIN PRODUCTION ENTRY: RENDER ANY STEM (THIS IS WHAT remix_daily SHOULD USE)
# =============================================================================

def render_stem(
    out_path: Path,
    *,
    stem: str,
    genre: str,
    bpm: int,
    key: str = "A",
    seed: Optional[str | int] = None,
    variant: int = 1,
    bars: int = 1,
    texture_type: str = "vinyl",
) -> None:
    """
    Render a fresh stem deterministically from (seed, genre, stem, variant, bpm, key).

    This is the function you should call from remix_daily.py to avoid static stem reuse.

    stem: "kick" | "drums" | "bass" | "music" | "pad" | "texture"
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    stem_l = str(stem).lower().strip()

    # RNG per stem call
    rnd = _rng(seed, salt=f"{genre}:{stem_l}:{bpm}:{key}:{bars}", variant=int(variant))

    # duration
    n = _bar_n(bpm, float(max(1, bars)))

    if stem_l == "kick":
        x = _render_kick(bpm, rnd, variant=variant, genre=genre)
        # extend to bars by repeating
        if x.size < n:
            reps = int(np.ceil(n / x.size))
            x = np.tile(x, reps)[:n]
        save_wav(out_path, x)
        return

    if stem_l == "drums":
        x = _render_drums(bpm, rnd, variant=variant, genre=genre)
        if x.size < n:
            reps = int(np.ceil(n / x.size))
            x = np.tile(x, reps)[:n]
        save_wav(out_path, x)
        return

    if stem_l == "bass":
        x = _render_bass(bpm, key, rnd, variant=variant, genre=genre)
        if x.size < n:
            reps = int(np.ceil(n / x.size))
            x = np.tile(x, reps)[:n]
        save_wav(out_path, x)
        return

    if stem_l in ("music", "pad", "synth"):
        x = _render_music_layer(bpm, key, rnd, variant=variant, genre=genre)
        if x.size < n:
            reps = int(np.ceil(n / x.size))
            x = np.tile(x, reps)[:n]
        save_wav(out_path, x)
        return

    if stem_l in ("texture", "ambience"):
        seconds = (60.0 / float(max(1e-6, bpm))) * 4.0 * float(max(1, bars))
        x = _render_texture(seconds, rnd, kind=texture_type)
        save_wav(out_path, x)
        return

    raise ValueError(f"Unknown stem type: {stem}")


# =============================================================================
# BACKWARD-COMPAT STEM LIBRARY API (kept for older remix_daily)
# These now accept optional seed to stop being identical forever.
# =============================================================================

def generate_techno_kick(out_path: Path, bpm: int = 130, variant: int = 1, seed: Optional[str | int] = None, **_ignored: Any) -> None:
    render_stem(out_path, stem="kick", genre="Techno", bpm=bpm, key="A", seed=seed, variant=variant, bars=1)

def generate_techno_bass(out_path: Path, bpm: int = 130, variant: int = 1, seed: Optional[str | int] = None, **_ignored: Any) -> None:
    render_stem(out_path, stem="bass", genre="Techno", bpm=bpm, key="A", seed=seed, variant=variant, bars=1)

def generate_techno_arp(out_path: Path, bpm: int = 130, variant: int = 1, seed: Optional[str | int] = None, **_ignored: Any) -> None:
    render_stem(out_path, stem="music", genre="Techno", bpm=bpm, key="A", seed=seed, variant=variant, bars=1)

def generate_house_drums(out_path: Path, bpm: int = 124, variant: int = 1, seed: Optional[str | int] = None, **_ignored: Any) -> None:
    render_stem(out_path, stem="drums", genre="House", bpm=bpm, key="A", seed=seed, variant=variant, bars=1)

def generate_deep_house_bass(out_path: Path, bpm: int = 124, variant: int = 1, seed: Optional[str | int] = None, **_ignored: Any) -> None:
    render_stem(out_path, stem="bass", genre="Deep", bpm=bpm, key="A", seed=seed, variant=variant, bars=1)

def generate_house_chords(out_path: Path, bpm: int = 124, seed: Optional[str | int] = None, **_ignored: Any) -> None:
    render_stem(out_path, stem="music", genre="House", bpm=bpm, key="A", seed=seed, variant=1, bars=1)

def generate_lofi_drums(out_path: Path, bpm: int = 85, variant: int = 1, seed: Optional[str | int] = None, **_ignored: Any) -> None:
    render_stem(out_path, stem="drums", genre="Chillout", bpm=bpm, key="C", seed=seed, variant=variant, bars=1)

def generate_lofi_keys(out_path: Path, bpm: int = 85, variant: int = 1, seed: Optional[str | int] = None, **_ignored: Any) -> None:
    render_stem(out_path, stem="music", genre="Chillout", bpm=bpm, key="C", seed=seed, variant=variant, bars=1)

def generate_wobble_bass(out_path: Path, bpm: int = 140, seed: Optional[str | int] = None, **_ignored: Any) -> None:
    render_stem(out_path, stem="bass", genre="Bass", bpm=bpm, key="A", seed=seed, variant=1, bars=1)

def generate_hard_kick(out_path: Path, bpm: int = 150, seed: Optional[str | int] = None, **_ignored: Any) -> None:
    render_stem(out_path, stem="kick", genre="Hard", bpm=bpm, key="A", seed=seed, variant=1, bars=1)

def generate_synth_bass(out_path: Path, bpm: int = 105, seed: Optional[str | int] = None, **_ignored: Any) -> None:
    render_stem(out_path, stem="bass", genre="Synth", bpm=bpm, key="A", seed=seed, variant=1, bars=1)

def generate_gated_snare(out_path: Path, bpm: int = 105, seed: Optional[str | int] = None, **_ignored: Any) -> None:
    # emulate as drums for synth style
    render_stem(out_path, stem="drums", genre="Synth", bpm=bpm, key="A", seed=seed, variant=2, bars=1)

def generate_rave_piano(out_path: Path, bpm: int = 140, seed: Optional[str | int] = None, **_ignored: Any) -> None:
    render_stem(out_path, stem="music", genre="Dance", bpm=bpm, key="C", seed=seed, variant=2, bars=1)

def generate_texture(out_path: Path, type: str = "vinyl", seed: Optional[str | int] = None, **_ignored: Any) -> None:
    render_stem(out_path, stem="texture", genre="Ambient", bpm=90, key="A", seed=seed, variant=1, bars=4, texture_type=type)


# =============================================================================
# FOCUS (BINAURAL) ENGINE - CONFIGURABLE + LEGACY-COMPATIBLE
# =============================================================================

FOCUS_PRESETS: Dict[str, Dict[str, float]] = {
    "deep_sleep": {"base_freq": 100.0, "beat_freq": 2.5, "noise_mix": 0.20},
    "meditation": {"base_freq": 150.0, "beat_freq": 6.0, "noise_mix": 0.25},
    "active_focus": {"base_freq": 250.0, "beat_freq": 20.0, "noise_mix": 0.30},
    "coding": {"base_freq": 300.0, "beat_freq": 40.0, "noise_mix": 0.25},
}

def _generate_focus_session_core(
    out_path: Path,
    duration_sec: float,
    base_freq: float = 250.0,
    beat_freq: float = 20.0,
    noise_mix: float = 0.30,
    rain: float = 0.0,
    vinyl: float = 0.0,
    white: float = 0.0,
    channels: int = 2,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    n = int(float(duration_sec) * SAMPLE_RATE)
    t = np.arange(n, dtype=np.float32) / SAMPLE_RATE

    nm = float(np.clip(noise_mix, 0.0, 1.0))
    noise = (np.random.randn(n).astype(np.float32) * nm) if nm > 0.0 else np.zeros(n, dtype=np.float32)

    bed = np.zeros(n, dtype=np.float32)
    w = float(np.clip(white, 0.0, 1.0))
    if w > 0.0:
        bed += np.random.randn(n).astype(np.float32) * (0.02 * w)

    r_amt = float(np.clip(rain, 0.0, 1.0))
    if r_amt > 0.0:
        r = np.random.randn(n).astype(np.float32) * 0.05
        r = apply_highpass(r, 150.0)
        r = apply_lowpass(r, 6500.0)
        bed += r * (0.35 * r_amt)

    v_amt = float(np.clip(vinyl, 0.0, 1.0))
    if v_amt > 0.0:
        v = np.random.randn(n).astype(np.float32) * 0.02
        v = apply_highpass(v, 80.0)
        v = apply_lowpass(v, 5500.0)
        bed += v * (0.25 * v_amt)

    channels = 1 if int(channels) == 1 else 2
    if channels == 1:
        tone = np.sin(TWOPI * float(base_freq) * t).astype(np.float32)
        mono = normalize(tone * 0.6 + noise + bed, 0.98)
        wavfile.write(str(out_path), SAMPLE_RATE, (mono * 32767.0).astype(np.int16))
        return

    left = np.sin(TWOPI * float(base_freq) * t).astype(np.float32)
    right = np.sin(TWOPI * float(base_freq + beat_freq) * t).astype(np.float32)

    stereo = np.stack([left, right], axis=1) * 0.55
    stereo[:, 0] += (noise + bed)
    stereo[:, 1] += (noise + bed)

    stereo = normalize(stereo, 0.98)
    wavfile.write(str(out_path), SAMPLE_RATE, (stereo * 32767.0).astype(np.int16))

def generate_focus_session(
    out_path: Path,
    duration_sec: float | None = None,
    *,
    preset_name: str | None = None,
    add_rain: bool | None = None,
    base_freq: float | None = None,
    beat_freq: float | None = None,
    noise_mix: float | None = None,
    rain: float = 0.0,
    vinyl: float = 0.0,
    white: float = 0.0,
    channels: int = 2,
    **_ignored: Any,
) -> None:
    if duration_sec is None:
        raise TypeError("generate_focus_session() missing required argument: 'duration_sec'")

    if preset_name is None:
        preset_name = "active_focus"
    preset = FOCUS_PRESETS.get(str(preset_name), FOCUS_PRESETS["active_focus"])

    bf = float(preset["base_freq"] if base_freq is None else base_freq)
    beat = float(preset["beat_freq"] if beat_freq is None else beat_freq)
    nm = float(preset["noise_mix"] if noise_mix is None else noise_mix)

    if add_rain:
        if rain <= 0.0:
            rain = 0.35

    _generate_focus_session_core(
        out_path=out_path,
        duration_sec=float(duration_sec),
        base_freq=bf,
        beat_freq=beat,
        noise_mix=nm,
        rain=float(rain),
        vinyl=float(vinyl),
        white=float(white),
        channels=int(channels),
    )


# =============================================================================
# LEGACY “INFINITE_*” EXPORTS (used by older remix_daily variants)
# =============================================================================

def generate_infinite_kick(out_path: Path, bpm: int, seed: int, energy: float = 0.8) -> None:
    render_stem(out_path, stem="kick", genre="Techno", bpm=int(bpm), key="A", seed=int(seed), variant=1, bars=1)

def generate_infinite_bass(out_path: Path, bpm: int, seed: int, genre: str = "techno") -> None:
    render_stem(out_path, stem="bass", genre=str(genre), bpm=int(bpm), key="A", seed=int(seed), variant=1, bars=1)

def generate_infinite_drums(out_path: Path, bpm: int, seed: int, type: str = "hats") -> None:
    render_stem(out_path, stem="drums", genre="House", bpm=int(bpm), key="A", seed=int(seed), variant=1, bars=1)

def generate_infinite_pad(out_path: Path, bpm: int, seed: int) -> None:
    render_stem(out_path, stem="music", genre="Ambient", bpm=int(bpm), key="A", seed=int(seed), variant=1, bars=1)
