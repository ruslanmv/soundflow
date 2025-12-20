# generator/free/music_engine.py
"""
SoundFlow Music Engine – v4.1 (Production)
High-quality synthesis, stereo-aware DSP, configurable binaural

Exports BOTH:
- legacy "infinite_*" API (used by older remix_daily versions)
- stem-library API used by remix_daily.ensure_procedural_library()

Key production fixes:
- Safe filtering on short buffers (no SciPy filtfilt padlen crashes)
- Stereo utilities (pan/width/mono) + mono rules for kick/bass
- Sidechain envelope utility
- Focus engine supports BOTH:
  - modern parameters (base_freq/beat_freq/noise_mix/rain/vinyl/white/channels)
  - legacy signature (preset_name/add_rain) to avoid TypeError
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Any

import numpy as np
from scipy import signal
from scipy.io import wavfile

# =============================================================================
# CONSTANTS
# =============================================================================

SAMPLE_RATE = 44100
TWOPI = 2.0 * np.pi

# =============================================================================
# BASIC UTILS
# =============================================================================


def _as_float(audio: np.ndarray) -> np.ndarray:
    """Convert int16/float64/etc to float32 in [-1, 1] if int16."""
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
    """
    Constant-power pan.
    amount: -1 (left) → 0 (center) → +1 (right)
    """
    x = mono_sum(audio)
    amount = float(np.clip(amount, -1.0, 1.0))
    left = x * np.cos((amount + 1.0) * np.pi / 4.0)
    right = x * np.sin((amount + 1.0) * np.pi / 4.0)
    return np.stack([left, right], axis=1).astype(np.float32)


def stereo_width(audio: np.ndarray, amount: float) -> np.ndarray:
    """
    Mid/Side width.
    amount: 0 (mono) → 1 (original) → 2 (wide)
    """
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
    Safe filtfilt that falls back to lfilter if the signal is too short.
    Works for mono (N,) and stereo (N,2).
    """
    x = _as_float(x)

    def _one(sig: np.ndarray) -> np.ndarray:
        sig = sig.reshape(-1).astype(np.float32, copy=False)

        # SciPy filtfilt padlen default:
        # padlen = 3 * (max(len(a), len(b)) - 1)
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
    Simple peaking EQ biquad (RBJ cookbook).
    Uses lfilter (stable for all lengths).
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
    """
    Musical resonant-ish lowpass:
    - peaking EQ near cutoff
    - then lowpass
    """
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
    """
    Small Schroeder-style reverb (CPU-light). Mono-safe.
    If stereo input, processes each channel independently.
    """
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
    """
    Simple 4-on-the-floor ducking envelope applied to any audio.
    duck_amount: 0..1, where lower = more duck (e.g., 0.4 heavy duck)
    """
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
# BASIC SYNTH BUILDING BLOCKS (KICK/NOISE/CHORDS)
# =============================================================================


def _sine_kick(seconds: float = 0.45, f0: float = 160.0, f1: float = 45.0, drive: float = 1.4) -> np.ndarray:
    n = int(seconds * SAMPLE_RATE)
    t = np.arange(n, dtype=np.float32) / SAMPLE_RATE
    freq = float(f0) * np.exp(-t * 18.0) + float(f1)  # exponential pitch drop
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


# =============================================================================
# STEM GENERATORS (STEM LIBRARY API)
# =============================================================================


def generate_techno_kick(out_path: Path, bpm: int = 130, variant: int = 1) -> None:
    np.random.seed(1000 + int(variant))
    beat = 60.0 / float(bpm)
    bar_sec = beat * 4.0
    n = int(bar_sec * SAMPLE_RATE)
    x = np.zeros(n, dtype=np.float32)

    hit = _sine_kick(seconds=0.45, f0=160.0 + 10.0 * variant, f1=45.0, drive=1.3 + 0.2 * variant)
    step = int((beat / 4.0) * SAMPLE_RATE)

    for s in range(0, 16, 4):  # 4-on-the-floor
        pos = s * step
        end = min(n, pos + hit.size)
        x[pos:end] += hit[: end - pos]

    x = apply_highpass(x, 20.0)
    x = soft_clip(x * 1.1, 0.9)
    x = mono_sum(x)  # kick must be mono
    save_wav(out_path, x)


def generate_techno_bass(out_path: Path, bpm: int = 130, variant: int = 1) -> None:
    np.random.seed(2000 + int(variant))
    beat = 60.0 / float(bpm)
    bar_sec = beat * 4.0
    n = int(bar_sec * SAMPLE_RATE)
    x = np.zeros(n, dtype=np.float32)

    base = 49.0  # G-ish low
    step = int((beat / 4.0) * SAMPLE_RATE)
    note_len = int(step * 0.95)

    pattern = [0, 0, 7, 0, 0, 10, 0, 0, 7, 0, 0, 3, 0, 0, 10, 0]
    for i, semis in enumerate(pattern):
        pos = i * step
        end = min(n, pos + note_len)
        freq = base * (2 ** (float(semis) / 12.0))
        tone = osc.saw(freq, end - pos, brightness=0.6)
        env = _adsr(end - pos, 0.005, 0.06, 0.35, 0.04)
        x[pos:end] += tone * env

    cutoff = 500.0 + 250.0 * float(variant)
    x = apply_resonant_filter(x, cutoff=cutoff, resonance=0.35)
    x = mono_sum(x)  # bass must be mono
    save_wav(out_path, x)


def generate_techno_arp(out_path: Path, bpm: int = 130, variant: int = 1) -> None:
    np.random.seed(3000 + int(variant))
    beat = 60.0 / float(bpm)
    bar_sec = beat * 4.0
    n = int(bar_sec * SAMPLE_RATE)
    x = np.zeros(n, dtype=np.float32)

    root = 220.0
    notes = [0, 3, 7, 10, 12, 10, 7, 3]  # minor-ish
    step = int((beat / 4.0) * SAMPLE_RATE)
    note_len = int(step * 0.75)

    for i in range(16):
        pos = i * step
        end = min(n, pos + note_len)
        semi = notes[i % len(notes)]
        freq = root * (2 ** (float(semi) / 12.0))
        tone = osc.supersaw(freq, end - pos, detune=0.09, variation=0.25)
        env = _adsr(end - pos, 0.005, 0.03, 0.3, 0.05)
        x[pos:end] += tone * env * 0.6

    # arp: wide + a bit of reverb, but keep mono-safe processing
    st = stereo_width(pan(x, 0.0), 1.4)
    st = apply_algorithmic_reverb(st, room_size=0.55, wet=0.25)
    save_wav(out_path, st)


def generate_house_drums(out_path: Path, bpm: int = 124, variant: int = 1) -> None:
    np.random.seed(4000 + int(variant))
    beat = 60.0 / float(bpm)
    bar_sec = beat * 4.0
    n = int(bar_sec * SAMPLE_RATE)
    x = np.zeros(n, dtype=np.float32)
    step = int((beat / 4.0) * SAMPLE_RATE)

    kick = _sine_kick(seconds=0.40, f0=150.0, f1=50.0, drive=1.2)
    clap = _noise_hit(int(0.18 * SAMPLE_RATE), 900.0, 4500.0, decay=18.0, velocity=0.8)
    hat = _noise_hit(int(0.06 * SAMPLE_RATE), 5000.0, 12000.0, decay=30.0, velocity=0.35)

    for i in range(16):
        pos = i * step
        if i % 4 == 0:
            end = min(n, pos + kick.size)
            x[pos:end] += kick[: end - pos]
        if i in (4, 12):
            end = min(n, pos + clap.size)
            x[pos:end] += clap[: end - pos] * 0.8
        if i % 2 == 1:
            end = min(n, pos + hat.size)
            x[pos:end] += hat[: end - pos]

    x = soft_clip(x * 1.1, 0.9)
    save_wav(out_path, x)


def generate_deep_house_bass(out_path: Path, bpm: int = 124, variant: int = 1) -> None:
    np.random.seed(5000 + int(variant))
    beat = 60.0 / float(bpm)
    bar_sec = beat * 4.0
    n = int(bar_sec * SAMPLE_RATE)
    x = np.zeros(n, dtype=np.float32)

    base = 55.0  # A1
    step = int((beat / 4.0) * SAMPLE_RATE)
    note_len = int(step * 0.98)

    pattern = [0, 0, 0, 7, 0, 0, 5, 0, 0, 7, 0, 0, 3, 0, 0, 5]
    for i, semis in enumerate(pattern):
        pos = i * step
        end = min(n, pos + note_len)
        freq = base * (2 ** (float(semis) / 12.0))
        tone = osc.saw(freq, end - pos, brightness=0.5)
        env = _adsr(end - pos, 0.01, 0.08, 0.4, 0.06)
        x[pos:end] += tone * env

    x = apply_lowpass(x, 900.0)
    x = mono_sum(x)  # bass must be mono
    save_wav(out_path, x)


def generate_house_chords(out_path: Path, bpm: int = 124) -> None:
    np.random.seed(6000)
    beat = 60.0 / float(bpm)
    bar_sec = beat * 4.0
    n = int(bar_sec * SAMPLE_RATE)
    x = np.zeros(n, dtype=np.float32)

    # i - VI - iv - v (A minor-ish around 220 root)
    prog = [
        [220.0, 261.63, 329.63],
        [174.61, 220.0, 261.63],
        [196.0, 246.94, 293.66],
        [207.65, 261.63, 311.13],
    ]

    chord_len = int(beat * SAMPLE_RATE)  # 1 beat stab
    gap = int(beat * SAMPLE_RATE)

    pos = 0
    for c in prog:
        if pos + chord_len >= n:
            break
        tone = _chord(c, chord_len, brightness=0.9)
        env = _adsr(chord_len, 0.005, 0.06, 0.2, 0.08)
        x[pos : pos + chord_len] += tone * env * 0.7
        pos += gap

    st = stereo_width(pan(x, 0.0), 1.6)
    st = apply_algorithmic_reverb(st, room_size=0.7, wet=0.30)
    save_wav(out_path, st)


def generate_lofi_drums(out_path: Path, bpm: int = 85, variant: int = 1) -> None:
    np.random.seed(7000 + int(variant))
    beat = 60.0 / float(bpm)
    bar_sec = beat * 4.0
    n = int(bar_sec * SAMPLE_RATE)
    x = np.zeros(n, dtype=np.float32)

    step = int((beat / 4.0) * SAMPLE_RATE)
    kick = _sine_kick(seconds=0.50, f0=120.0, f1=50.0, drive=1.0)
    sn = _noise_hit(int(0.14 * SAMPLE_RATE), 700.0, 3500.0, decay=12.0, velocity=0.6)
    hat = _noise_hit(int(0.05 * SAMPLE_RATE), 4500.0, 9000.0, decay=18.0, velocity=0.25)

    for i in range(16):
        pos = i * step
        if i % 8 == 0:
            end = min(n, pos + kick.size)
            x[pos:end] += kick[: end - pos] * 0.9
        if i in (4, 12):
            end = min(n, pos + sn.size)
            x[pos:end] += sn[: end - pos] * 0.7
        if i % 2 == 1:
            end = min(n, pos + hat.size)
            x[pos:end] += hat[: end - pos] * 0.5

    x = apply_lowpass(x, 6000.0)
    x = apply_overdrive(x, drive=0.5)
    save_wav(out_path, x)


def generate_lofi_keys(out_path: Path, bpm: int = 85, variant: int = 1) -> None:
    np.random.seed(8000 + int(variant))
    beat = 60.0 / float(bpm)
    bar_sec = beat * 4.0
    n = int(bar_sec * SAMPLE_RATE)
    x = np.zeros(n, dtype=np.float32)

    chords = [
        [220.0, 261.63, 329.63],
        [196.0, 246.94, 293.66],
        [174.61, 220.0, 261.63],
        [196.0, 246.94, 293.66],
    ]
    quarter = int((bar_sec / 4.0) * SAMPLE_RATE)

    for i in range(4):
        pos = i * quarter
        end = min(n, pos + quarter)
        tone = _chord(chords[i], end - pos, brightness=0.55)
        env = _adsr(end - pos, 0.03, 0.12, 0.55, 0.15)
        x[pos:end] += tone * env * 0.45

    x = apply_lowpass(x, 5500.0)
    x = apply_algorithmic_reverb(x, room_size=0.55, wet=0.25)
    save_wav(out_path, x)


def generate_wobble_bass(out_path: Path, bpm: int = 140) -> None:
    np.random.seed(9000)
    beat = 60.0 / float(bpm)
    bar_sec = beat * 4.0
    n = int(bar_sec * SAMPLE_RATE)
    t = np.arange(n, dtype=np.float32) / SAMPLE_RATE

    base = 49.0
    x = osc.saw(base, n, brightness=0.8) * 0.8

    wob = 0.5 + 0.5 * np.sin(TWOPI * 2.0 * t)
    cutoff_curve = 200.0 + wob * 1800.0

    y = np.zeros_like(x, dtype=np.float32)
    chunk = 4096
    for i in range(0, n, chunk):
        c = float(np.mean(cutoff_curve[i : i + chunk]))
        y[i : i + chunk] = apply_lowpass(x[i : i + chunk], c)

    y = apply_overdrive(y, drive=1.2)
    y = mono_sum(y)
    save_wav(out_path, y)


def generate_hard_kick(out_path: Path, bpm: int = 150) -> None:
    np.random.seed(10000)
    beat = 60.0 / float(bpm)
    bar_sec = beat * 4.0
    n = int(bar_sec * SAMPLE_RATE)
    x = np.zeros(n, dtype=np.float32)

    hit = _sine_kick(seconds=0.35, f0=190.0, f1=52.0, drive=2.2)
    step = int((beat / 4.0) * SAMPLE_RATE)
    for s in range(0, 16, 4):
        pos = s * step
        end = min(n, pos + hit.size)
        x[pos:end] += hit[: end - pos]

    x = apply_overdrive(x, drive=2.5)
    x = soft_clip(x * 1.2, 0.85)
    x = mono_sum(x)  # kick mono
    save_wav(out_path, x)


def generate_synth_bass(out_path: Path, bpm: int = 105) -> None:
    np.random.seed(11000)
    beat = 60.0 / float(bpm)
    bar_sec = beat * 4.0
    n = int(bar_sec * SAMPLE_RATE)
    x = np.zeros(n, dtype=np.float32)

    base = 55.0
    step = int((beat / 4.0) * SAMPLE_RATE)
    note_len = int(step * 0.95)
    pattern = [0, 0, 7, 0, 5, 0, 7, 0, 0, 0, 12, 0, 7, 0, 5, 0]

    for i, semis in enumerate(pattern):
        pos = i * step
        end = min(n, pos + note_len)
        freq = base * (2 ** (float(semis) / 12.0))
        tone = osc.saw(freq, end - pos, brightness=0.7)
        env = _adsr(end - pos, 0.01, 0.07, 0.45, 0.08)
        x[pos:end] += tone * env * 0.9

    x = apply_lowpass(x, 1200.0)
    x = mono_sum(x)
    save_wav(out_path, x)


def generate_gated_snare(out_path: Path, bpm: int = 105) -> None:
    np.random.seed(12000)
    beat = 60.0 / float(bpm)
    bar_sec = beat * 4.0
    n = int(bar_sec * SAMPLE_RATE)
    x = np.zeros(n, dtype=np.float32)

    step = int((beat / 4.0) * SAMPLE_RATE)
    sn = _noise_hit(int(0.20 * SAMPLE_RATE), 800.0, 6000.0, decay=10.0, velocity=0.9)

    gate_len = int(0.08 * SAMPLE_RATE)
    if sn.size > gate_len:
        sn[gate_len:] *= 0.0

    for i in (4, 12):
        pos = i * step
        end = min(n, pos + sn.size)
        x[pos:end] += sn[: end - pos]

    x = apply_algorithmic_reverb(x, room_size=0.7, wet=0.18)
    save_wav(out_path, x)


def generate_rave_piano(out_path: Path, bpm: int = 140) -> None:
    np.random.seed(13000)
    beat = 60.0 / float(bpm)
    bar_sec = beat * 4.0
    n = int(bar_sec * SAMPLE_RATE)
    x = np.zeros(n, dtype=np.float32)

    chords = [
        [261.63, 329.63, 392.0],
        [293.66, 369.99, 440.0],
        [246.94, 311.13, 369.99],
        [261.63, 329.63, 392.0],
    ]
    stab_len = int(0.35 * beat * SAMPLE_RATE)
    step = int(beat * SAMPLE_RATE)

    pos = 0
    for c in chords:
        if pos + stab_len >= n:
            break
        tone = _chord(c, stab_len, brightness=1.2)
        env = _adsr(stab_len, 0.002, 0.05, 0.2, 0.10)
        x[pos : pos + stab_len] += tone * env * 0.9
        pos += step

    st = stereo_width(pan(x, 0.0), 1.7)
    st = apply_algorithmic_reverb(st, room_size=0.8, wet=0.35)
    save_wav(out_path, st)


def generate_texture(out_path: Path, type: str = "vinyl") -> None:
    t = str(type).lower().strip()
    np.random.seed(14000 + (1 if t == "rain" else 0))

    seconds = 16.0
    n = int(seconds * SAMPLE_RATE)

    if t == "vinyl":
        x = np.random.randn(n).astype(np.float32) * 0.02
        x = apply_highpass(x, 80.0)
        x = apply_lowpass(x, 5500.0)

        for _ in range(14):
            pos = np.random.randint(0, n - 200)
            click = np.random.uniform(-1, 1, 200).astype(np.float32)
            click *= np.exp(-np.linspace(0, 10, 200)).astype(np.float32)
            x[pos : pos + 200] += click * 0.08

    else:  # rain
        x = np.random.randn(n).astype(np.float32) * 0.05
        x = apply_highpass(x, 150.0)
        x = apply_lowpass(x, 6500.0)
        tt = np.arange(n, dtype=np.float32) / SAMPLE_RATE
        x *= (0.75 + 0.25 * np.sin(TWOPI * 0.12 * tt)).astype(np.float32)

    save_wav(out_path, x)


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
    # ---- legacy args (used by remix_daily) ----
    preset_name: str | None = None,
    add_rain: bool | None = None,
    # ---- modern args ----
    base_freq: float | None = None,
    beat_freq: float | None = None,
    noise_mix: float | None = None,
    rain: float = 0.0,
    vinyl: float = 0.0,
    white: float = 0.0,
    channels: int = 2,
    # allow passing extra keys from older code without crashing
    **_ignored: Any,
) -> None:
    """
    Unified Focus API.

    Supports:
    - Legacy: generate_focus_session(out_path, preset_name="active_focus", duration_sec=..., add_rain=True)
    - Modern: generate_focus_session(out_path, duration_sec=..., base_freq=..., beat_freq=..., noise_mix=..., rain=..., vinyl=..., white=..., channels=...)

    If preset_name is provided, its parameters are used as defaults, unless overridden by explicit base_freq/beat_freq/noise_mix.
    """
    if duration_sec is None:
        raise TypeError("generate_focus_session() missing required argument: 'duration_sec'")

    if preset_name is None:
        preset_name = "active_focus"

    preset = FOCUS_PRESETS.get(str(preset_name), FOCUS_PRESETS["active_focus"])

    bf = float(preset["base_freq"] if base_freq is None else base_freq)
    beat = float(preset["beat_freq"] if beat_freq is None else beat_freq)
    nm = float(preset["noise_mix"] if noise_mix is None else noise_mix)

    # legacy behavior: add_rain enables a gentle rain bed if user didn't specify rain
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
# BACKWARD-COMPATIBLE "INFINITE" EXPORTS (USED BY OLDER CODE)
# =============================================================================

def generate_infinite_kick(out_path: Path, bpm: int, seed: int, energy: float = 0.8) -> None:
    np.random.seed(int(seed) & 0xFFFFFFFF)
    # energy currently not used in this simple stem, but kept for API compatibility
    generate_techno_kick(out_path, bpm=int(bpm), variant=1)


def generate_infinite_bass(out_path: Path, bpm: int, seed: int, genre: str = "techno") -> None:
    np.random.seed(int(seed) & 0xFFFFFFFF)
    if str(genre).lower() == "house":
        generate_deep_house_bass(out_path, bpm=int(bpm), variant=1)
    else:
        generate_techno_bass(out_path, bpm=int(bpm), variant=1)


def generate_infinite_drums(out_path: Path, bpm: int, seed: int, type: str = "hats") -> None:
    np.random.seed(int(seed) & 0xFFFFFFFF)
    # keep it simple: house drums are a good general drum bed
    generate_house_drums(out_path, bpm=int(bpm), variant=1)


def generate_infinite_pad(out_path: Path, bpm: int, seed: int) -> None:
    np.random.seed(int(seed) & 0xFFFFFFFF)
    # use house chords as a "pad-ish" layer for compatibility
    generate_house_chords(out_path, bpm=int(bpm))
