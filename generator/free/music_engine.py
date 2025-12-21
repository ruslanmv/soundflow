# generator/free/music_engine.py
"""
SoundFlow Music Engine – v6.0 (Professional DJ Quality)
High-fidelity synthesis, dynamic arrangements, energy curves.

Features:
- Long stems (1-2 minutes) with proper structure
- Energy dynamics (build/peak/drop/breakdown)
- Professional synthesis (supersaw, FM, wavetable-style)
- Advanced arrangement (intro/build/peak/breakdown/outro)
- Humanization and groove
- Broadcast-quality mixing
- All major genres with authentic sound
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List
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
# SEEDING (CRITICAL: fixes "everything sounds the same")
# =============================================================================

def _stable_u32(s: str) -> int:
    h = hashlib.sha256(s.encode("utf-8")).digest()
    return int.from_bytes(h[:4], "big", signed=False)

def _seed_from(seed: Optional[str | int], *, salt: str = "", variant: int = 1) -> int:
    """Deterministic 32-bit seed"""
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
    """Safe filtfilt with fallback"""
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
    """Peaking EQ biquad"""
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
        curve = np.concatenate([
            np.linspace(1.0, duck_amount, half) ** 2,
            np.linspace(duck_amount, 1.0, rest) ** 2,
        ]).astype(np.float32)
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
# MUSICAL HELPERS
# =============================================================================

_NOTE_TO_SEMI = {
    "C": 0, "C#": 1, "DB": 1, "D": 2, "D#": 3, "EB": 3, "E": 4, "F": 5,
    "F#": 6, "GB": 6, "G": 7, "G#": 8, "AB": 8, "A": 9, "A#": 10, "BB": 10, "B": 11
}

def key_to_root_hz(key: str, octave: int = 3) -> float:
    k = str(key).strip().upper()
    semi = _NOTE_TO_SEMI.get(k, 9)
    c0 = 16.3516
    n = semi + 12 * int(octave)
    return float(c0 * (2 ** (n / 12.0)))

def semitone(freq: float, semi: float) -> float:
    return float(freq) * (2 ** (float(semi) / 12.0))

# =============================================================================
# PROFESSIONAL OSCILLATORS
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
    
    env = np.concatenate([
        np.linspace(0, 1, aN) ** 0.5,
        np.linspace(1, s, dN) ** 2.0,
        np.full(sN, s, dtype=np.float32),
        np.linspace(s, 0, rN) ** 2.0,
    ]).astype(np.float32)
    if env.size < n:
        env = np.pad(env, (0, n - env.size))
    return env[:n].astype(np.float32)

class Oscillator:
    def saw(self, freq: float, n: int, brightness: float = 1.0) -> np.ndarray:
        """Anti-aliased sawtooth with brightness control"""
        t = np.arange(n, dtype=np.float32) / SAMPLE_RATE
        brightness = float(np.clip(brightness, 0.2, 2.0))
        max_h = max(1, int(30 * brightness))
        out = np.zeros(n, dtype=np.float32)
        for k in range(1, max_h + 1):
            out += np.sin(TWOPI * float(freq) * k * t) / k
        return (out * (2.0 / np.pi)).astype(np.float32)
    
    def supersaw(self, freq: float, n: int, detune: float = 0.08, variation: float = 0.0) -> np.ndarray:
        """Roland JP-8000 style supersaw"""
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
# ENERGY CURVE SYSTEM
# =============================================================================

def create_energy_curve(total_samples: int, curve_type: str = "peak", rnd: Optional[np.random.RandomState] = None) -> np.ndarray:
    """
    Create energy envelope for entire track.
    """
    if rnd is None:
        rnd = np.random.RandomState(0)
    
    curve_type = str(curve_type).lower().strip()
    n = int(total_samples)
    
    if curve_type == "peak":
        # Classic club track structure
        intro = np.linspace(0.3, 0.6, n // 8) ** 2
        build1 = np.linspace(0.6, 0.9, n // 8) ** 1.5
        peak = np.ones(n // 4) * (0.95 + 0.05 * rnd.rand(n // 4))
        breakdown = np.linspace(0.9, 0.4, n // 8) ** 2
        build2 = np.linspace(0.4, 1.0, n // 8) ** 1.2
        outro = np.linspace(1.0, 0.2, n // 4) ** 2.5
        
        curve = np.concatenate([intro, build1, peak, breakdown, build2, outro])
        
    elif curve_type == "drop":
        # Bass/dubstep style
        intro_high = np.ones(n // 16) * 0.9
        drop = np.linspace(0.9, 0.2, n // 16) ** 3
        low = np.ones(n // 8) * (0.25 + 0.1 * rnd.rand(n // 8))
        build = np.linspace(0.25, 1.0, n // 4) ** 1.3
        peak = np.ones(n // 4) * (0.95 + 0.05 * rnd.rand(n // 4))
        outro = np.linspace(0.95, 0.1, n // 4) ** 2
        
        curve = np.concatenate([intro_high, drop, low, build, peak, outro])
        
    elif curve_type == "build":
        # Progressive build
        curve = np.linspace(0.3, 1.0, n) ** 1.5
        # Add some variation
        curve += np.sin(np.linspace(0, 8 * np.pi, n)) * 0.08
        
    else:  # linear
        curve = np.linspace(0.5, 0.95, n)
    
    # Ensure length
    if curve.size < n:
        curve = np.pad(curve, (0, n - curve.size), constant_values=curve[-1])
    curve = curve[:n]
    
    return curve.astype(np.float32)

# =============================================================================
# ENERGY CURVE DEFINITIONS (EXPORTED)
# =============================================================================

ENERGY_CURVES: Dict[str, str] = {
    "peak": "Classic club track (intro → build → peak → breakdown → outro)",
    "drop": "Bass/dubstep style (high → drop → low → build → peak)",
    "build": "Progressive build (steady increase to climax)",
    "linear": "Constant energy (steady throughout)",
}

# =============================================================================
# GENRE STYLES
# =============================================================================

GENRE_STYLES: Dict[str, Dict[str, Any]] = {
    "techno": {"bpm": 130, "swing": 0.00, "brightness": 0.65, "reverb": 0.18, "duck": 0.55},
    "house": {"bpm": 124, "swing": 0.03, "brightness": 0.55, "reverb": 0.22, "duck": 0.45},
    "deep": {"bpm": 122, "swing": 0.05, "brightness": 0.45, "reverb": 0.25, "duck": 0.40},
    "trance": {"bpm": 138, "swing": 0.00, "brightness": 0.90, "reverb": 0.28, "duck": 0.55},
    "edm": {"bpm": 128, "swing": 0.00, "brightness": 0.85, "reverb": 0.20, "duck": 0.60},
    "dance": {"bpm": 128, "swing": 0.01, "brightness": 0.70, "reverb": 0.18, "duck": 0.55},
    "hard": {"bpm": 150, "swing": 0.00, "brightness": 0.75, "reverb": 0.12, "duck": 0.65},
    "bass": {"bpm": 140, "swing": 0.00, "brightness": 0.70, "reverb": 0.15, "duck": 0.55},
    "synth": {"bpm": 105, "swing": 0.00, "brightness": 0.75, "reverb": 0.30, "duck": 0.40},
    "chillout": {"bpm": 90, "swing": 0.06, "brightness": 0.35, "reverb": 0.35, "duck": 0.25},
    "lounge": {"bpm": 100, "swing": 0.05, "brightness": 0.40, "reverb": 0.32, "duck": 0.25},
    "ambient": {"bpm": 70, "swing": 0.00, "brightness": 0.25, "reverb": 0.45, "duck": 0.15},
    "classic": {"bpm": 90, "swing": 0.00, "brightness": 0.30, "reverb": 0.38, "duck": 0.10},
    "vocal": {"bpm": 128, "swing": 0.02, "brightness": 0.65, "reverb": 0.28, "duck": 0.50},
    "lofi": {"bpm": 85, "swing": 0.08, "brightness": 0.30, "reverb": 0.15, "duck": 0.20},
    # ✅ JAZZ NOW SUPPORTED
    "jazz": {"bpm": 120, "swing": 0.12, "brightness": 0.40, "reverb": 0.25, "duck": 0.10},
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
# PROFESSIONAL KICK DRUM (3-LAYER)
# =============================================================================

def _professional_kick(
    bpm: float,
    rnd: np.random.RandomState,
    *,
    energy: float = 0.8,
    style: str = "techno"
) -> np.ndarray:
    """
    Professional 3-layer kick:
    - Sub (20-60Hz): Deep punch
    - Body (60-200Hz): Fundamental
    - Click (2-8kHz): Attack transient
    """
    n = int(0.5 * SAMPLE_RATE)  # 500ms kick
    t = np.arange(n, dtype=np.float32) / SAMPLE_RATE
    
    energy = float(np.clip(energy, 0.3, 1.0))
    
    # Layer 1: Sub (20-60Hz)
    f0_sub = 70 + rnd.randint(-10, 20)
    f1_sub = 38 + rnd.randint(-5, 10)
    freq_sub = f0_sub * np.exp(-t * 18.0) + f1_sub
    phase_sub = np.cumsum(freq_sub) * (TWOPI / SAMPLE_RATE)
    sub = np.sin(phase_sub) * np.exp(-t * 8.0) * energy
    sub += np.sin(phase_sub * 0.5) * np.exp(-t * 6.0) * (energy * 0.4)
    
    # Layer 2: Body (60-200Hz)
    f0_body = 180 + rnd.randint(-20, 30)
    f1_body = 75 + rnd.randint(-10, 15)
    freq_body = f0_body * np.exp(-t * 20.0) + f1_body
    phase_body = np.cumsum(freq_body) * (TWOPI / SAMPLE_RATE)
    body = np.sin(phase_body) * np.exp(-t * 12.0) * energy
    body += np.sin(phase_body * 2.0) * np.exp(-t * 14.0) * (energy * 0.3)
    
    # Layer 3: Click (2-8kHz transient)
    click_n = int(0.008 * SAMPLE_RATE)  # 8ms
    click = rnd.randn(click_n).astype(np.float32)
    click = apply_highpass(click, 2000.0)
    click = apply_lowpass(click, 8000.0)
    click *= np.exp(-np.linspace(0, 50, click_n)).astype(np.float32)
    click *= energy * 1.2
    
    # Combine layers
    kick = np.zeros(n, dtype=np.float32)
    kick[:] = sub * 0.45 + body * 0.35
    kick[:click_n] += click * 0.20
    
    # Saturation for analog warmth
    kick = np.tanh(kick * (1.3 + 0.5 * energy))
    
    return kick.astype(np.float32)

# =============================================================================
# LONG STEM GENERATORS (1-2 MINUTES)
# =============================================================================

def _render_kick_long(
    bpm: int,
    rnd: np.random.RandomState,
    *,
    variant: int,
    genre: str,
    bars: int = 16,
    energy_curve: Optional[np.ndarray] = None
) -> np.ndarray:
    """Generate evolving kick pattern"""
    beat = 60.0 / float(bpm)
    bar_samples = int(beat * 4.0 * SAMPLE_RATE)
    n = bar_samples * int(bars)
    
    if energy_curve is None:
        energy_curve = np.ones(n, dtype=np.float32) * 0.8
    
    g = _norm_genre(genre)
    x = np.zeros(n, dtype=np.float32)
    
    kick_base = _professional_kick(bpm, rnd, energy=0.8, style=g)
    
    for bar_idx in range(bars):
        bar_start = bar_idx * bar_samples
        bar_end = min(n, bar_start + bar_samples)
        bar_energy = energy_curve[bar_start:bar_end].mean()
        
        if bar_idx % 4 == 0:
            kick = _professional_kick(bpm, rnd, energy=float(bar_energy), style=g)
        
        if g in ("ambient", "chillout", "classic"):
            hits = [0, 8]
        elif g == "jazz":
            hits = [0, 6, 8, 14] if rnd.rand() > 0.5 else [0, 8, 10]
        else:
            hits = [0, 4, 8, 12]
        
        step = int((beat / 4.0) * SAMPLE_RATE)
        for s in hits:
            pos = bar_start + s * step
            if pos + kick.size >= bar_end:
                break
            energy_mult = energy_curve[pos] if pos < len(energy_curve) else 1.0
            end = min(n, pos + kick.size)
            x[pos:end] += kick[:end-pos] * energy_mult
    
    x = apply_highpass(x, 20.0)
    x = soft_clip(x * 1.05, 0.88)
    return x.astype(np.float32)

def _render_drums_long(
    bpm: int,
    rnd: np.random.RandomState,
    *,
    variant: int,
    genre: str,
    bars: int = 16,
    energy_curve: Optional[np.ndarray] = None
) -> np.ndarray:
    """Generate full drum kit with fills and variation"""
    beat = 60.0 / float(bpm)
    bar_samples = int(beat * 4.0 * SAMPLE_RATE)
    n = bar_samples * int(bars)
    
    if energy_curve is None:
        energy_curve = np.ones(n, dtype=np.float32) * 0.8
    
    g = _norm_genre(genre)
    st = GENRE_STYLES[g]
    swing = float(st["swing"])
    
    x = np.zeros(n, dtype=np.float32)
    
    # Percussion
    clap = rnd.randn(int(0.15 * SAMPLE_RATE)).astype(np.float32)
    clap = apply_highpass(clap, 900.0)
    clap = apply_lowpass(clap, 4800.0)
    clap *= np.exp(-np.linspace(0, 18, clap.size)).astype(np.float32) * 0.7
    
    hat_closed = rnd.randn(int(0.04 * SAMPLE_RATE)).astype(np.float32)
    hat_closed = apply_highpass(hat_closed, 5500.0)
    hat_closed *= np.exp(-np.linspace(0, 30, hat_closed.size)).astype(np.float32) * 0.20
    
    step = int((beat / 4.0) * SAMPLE_RATE)
    
    for bar_idx in range(bars):
        bar_start = bar_idx * bar_samples
        is_fill_bar = (bar_idx % 8 == 7)
        is_breakdown = energy_curve[bar_start:min(n, bar_start + bar_samples)].mean() < 0.4
        
        for i in range(16):
            pos = bar_start + int((i + swing * (i % 2)) * step)
            if pos >= n:
                break
            
            energy_mult = energy_curve[pos] if pos < len(energy_curve) else 1.0
            
            # Clap/Snare
            if i in (4, 12) and not is_breakdown:
                end = min(n, pos + clap.size)
                x[pos:end] += clap[:end-pos] * energy_mult * 0.75
            
            # Hi-hats
            if i % 2 == 1:
                end = min(n, pos + hat_closed.size)
                x[pos:end] += hat_closed[:end-pos] * energy_mult
            
            # Jazz Swing Ride
            if g == "jazz":
                if i in (0, 2, 4, 6, 8, 10, 12, 14) or (i % 4 == 3):
                    end = min(n, pos + hat_closed.size)
                    x[pos:end] += hat_closed[:end-pos] * energy_mult * 0.8

    if g in ("lofi", "chillout", "lounge"):
        x = apply_lowpass(x, 6500.0)
    else:
        x = apply_lowpass(x, 11000.0)
    
    x = soft_clip(x * 1.05, 0.9)
    return x.astype(np.float32)

def _render_bass_long(
    bpm: int,
    key: str,
    rnd: np.random.RandomState,
    *,
    variant: int,
    genre: str,
    bars: int = 16,
    energy_curve: Optional[np.ndarray] = None
) -> np.ndarray:
    """Generate evolving bassline"""
    beat = 60.0 / float(bpm)
    bar_samples = int(beat * 4.0 * SAMPLE_RATE)
    n = bar_samples * int(bars)
    
    if energy_curve is None:
        energy_curve = np.ones(n, dtype=np.float32) * 0.8
    
    g = _norm_genre(genre)
    st = GENRE_STYLES[g]
    root = key_to_root_hz(key, octave=2)
    
    # Progressions
    if g in ("trance", "edm", "dance"):
        prog_semis = [0, 0, 7, 0, 5, 0, 7, 0, 0, 12, 0, 7, 0, 5, 0, 7]
    elif g == "jazz":
        prog_semis = [0, 4, 7, 11, 2, 5, 9, 0, -5, -2, 2, 5, 0, 4, 7, 11]
    elif g in ("deep", "house"):
        prog_semis = [0, 0, 0, 7, 0, 0, 5, 0, 0, 7, 0, 0, 3, 0, 0, 5]
    else:
        prog_semis = [0, 0, 7, 0, 0, 10, 0, 0, 7, 0, 0, 3, 0, 0, 10, 0]
    
    x = np.zeros(n, dtype=np.float32)
    step = int((beat / 4.0) * SAMPLE_RATE)
    note_len = int(step * 0.88)
    
    for i in range(min(len(prog_semis) * 8, (n // step))):
        pos = i * step
        if pos + note_len >= n:
            break
        
        semi = prog_semis[i % len(prog_semis)]
        f = semitone(root, semi)
        
        # Jazz Walking Bass
        if g == "jazz":
             if i % 4 != 0: # Add chromatic approach notes
                 f = semitone(root, semi + rnd.choice([-1, 1]))
        
        tone = osc.saw(f, note_len, brightness=st["brightness"] * 0.75)
        bar_energy = energy_curve[pos:min(n, pos + note_len)].mean()
        env = _adsr(note_len, 0.005, 0.06, 0.30 * bar_energy, 0.04)
        x[pos:pos+note_len] += tone * env
    
    x = apply_lowpass(x, 900.0)
    return mono_sum(x).astype(np.float32)

# =============================================================================
# VIRTUAL INSTRUMENTS (CLEANED / HOUSE-READY)
# =============================================================================

def _keytrack_lp(freq: float, base: float, amount: float, lo: float, hi: float) -> float:
    """Simple keytracking for cutoff."""
    f = float(max(20.0, freq))
    cutoff = base + amount * (f ** 0.35)
    return float(np.clip(cutoff, lo, hi))

def _tiny_transient(rnd: np.random.RandomState, n: int, amp: float) -> np.ndarray:
    """Short, band-limited transient for realism."""
    if n <= 0: return np.zeros(0, dtype=np.float32)
    x = rnd.randn(n).astype(np.float32) * float(amp)
    x *= np.linspace(1.0, 0.0, n, dtype=np.float32)
    x = apply_highpass(x, 1200.0)
    x = apply_lowpass(x, 6500.0)
    return x.astype(np.float32)

def _generate_tone_fm_rhodes(
    freq: float, duration_samples: int, velocity: float = 0.8, rnd: Optional[np.random.RandomState] = None
) -> np.ndarray:
    """Cleaner Deep House Rhodes/DX EP (No Christmas Bells)."""
    if rnd is None: rnd = np.random.RandomState(0)
    n = int(max(1, duration_samples))
    f = float(max(20.0, freq))
    vel = float(np.clip(velocity, 0.0, 1.2))
    t = np.arange(n, dtype=np.float32) / SAMPLE_RATE

    # Ratio 3.0 = Classic EP tone (Not 14.0 which is Bells)
    ratio = 3.0 + 3.0 * float(rnd.rand()) 
    mod_f = f * ratio
    
    # Mod Index
    idx = (1.3 + 1.6 * vel) * np.exp(-t * (7.0 + 4.0 * vel))
    mod = np.sin(TWOPI * mod_f * t) * idx
    car = np.sin(TWOPI * f * t + mod)

    # Body warmth
    car += 0.18 * np.sin(TWOPI * (f * 2.0) * t)
    
    # Envelope
    env = _adsr(n, 0.003, 0.22, 0.35 + 0.25 * vel, 0.14) * vel
    y = (car * env).astype(np.float32)

    # Cleanup
    y = apply_highpass(y, 90.0)
    return soft_clip(y * 1.05, threshold=0.92)

def _generate_tone_organ_m1(
    freq: float, duration_samples: int, velocity: float = 0.8, rnd: Optional[np.random.RandomState] = None
) -> np.ndarray:
    """Classic House Organ (Smoother, Lower Volume)."""
    if rnd is None: rnd = np.random.RandomState(0)
    n = int(max(1, duration_samples))
    f = float(max(20.0, freq))
    vel = float(np.clip(velocity, 0.0, 1.2))
    t = np.arange(n, dtype=np.float32) / SAMPLE_RATE

    # Classic Organ Partials
    harmonics = [(1.0, 0.70), (2.0, 0.34), (3.0, 0.14), (4.0, 0.06)]
    y = np.zeros(n, dtype=np.float32)
    for ratio, amp in harmonics:
        y += np.sin(TWOPI * (f * ratio) * t) * amp

    # Woody Attack (Seed stable)
    chiff_n = min(n, int(0.018 * SAMPLE_RATE))
    y[:chiff_n] += _tiny_transient(rnd, chiff_n, amp=0.04 + 0.05 * vel)

    # Envelope (Punchy)
    env = _adsr(n, 0.004, 0.08, 0.92, 0.10) * vel
    y *= env

    # Tone Shaping
    y = apply_highpass(y, 120.0)
    y = apply_lowpass(y, _keytrack_lp(f, base=4800.0, amount=45.0, lo=2200.0, hi=9800.0))
    y = soft_clip(y * 1.08, threshold=0.93)

    # ✅ VOLUME REDUCTION (Prevents overpowering the mix)
    return (y * 0.75).astype(np.float32)

def _generate_tone_house_piano(
    freq: float, duration_samples: int, velocity: float = 0.8, rnd: Optional[np.random.RandomState] = None
) -> np.ndarray:
    """Bright House Piano Stab (Korg M1 Piano style)."""
    if rnd is None: rnd = np.random.RandomState(0)
    n = int(max(1, duration_samples))
    f = float(max(20.0, freq))
    vel = float(np.clip(velocity, 0.0, 1.2))
    t = np.arange(n, dtype=np.float32) / SAMPLE_RATE

    # Synth Piano: Fundamental + Octave + Bite
    y = np.sin(TWOPI * f * t) * 0.65
    y += np.sin(TWOPI * (f * 2.0) * t) * 0.22
    y += np.sin(TWOPI * (f * 3.0) * t) * 0.10 # 3rd harmonic bite

    # Hammer Noise
    atk_n = min(n, int(0.010 * SAMPLE_RATE))
    y[:atk_n] += _tiny_transient(rnd, atk_n, amp=0.05 + 0.05 * vel)

    # Fast Decay Envelope
    env = np.exp(-t * (3.2 + 1.2 * (1.0 - vel))) * vel
    y *= env

    # Filter & Saturation
    y = apply_highpass(y, 120.0)
    return soft_clip(y * 1.1, threshold=0.92).astype(np.float32)

def _generate_tone_pluck_analog(
    freq: float, duration_samples: int, velocity: float = 0.8, rnd: Optional[np.random.RandomState] = None
) -> np.ndarray:
    """Analog Pluck with Filter Envelope."""
    if rnd is None: rnd = np.random.RandomState(0)
    n = int(max(1, duration_samples))
    f = float(max(20.0, freq))
    vel = float(np.clip(velocity, 0.0, 1.2))

    # Use Sawtooth source
    raw = osc.saw(f, n, brightness=0.85)
    t = np.arange(n, dtype=np.float32) / SAMPLE_RATE
    
    # Fake Filter Envelope (Crossfade Bright -> Warm)
    warm = apply_lowpass(raw, 2500.0)
    bright = apply_lowpass(raw, 8000.0)
    filt_env = np.exp(-t * 9.0)
    y = bright * filt_env + warm * (1.0 - filt_env)

    # Amp Envelope
    amp = np.exp(-t * 8.0) * vel
    y *= amp
    
    return apply_highpass(y, 120.0).astype(np.float32)

def _generate_tone_piano_simple(
    freq: float, duration_samples: int, velocity: float = 0.8, rnd: Optional[np.random.RandomState] = None
) -> np.ndarray:
    """Gentle Acoustic Piano."""
    if rnd is None: rnd = np.random.RandomState(0)
    n = int(max(1, duration_samples))
    f = float(max(20.0, freq))
    t = np.arange(n, dtype=np.float32) / SAMPLE_RATE

    y = np.sin(TWOPI * f * t) * 0.65
    y += np.sin(TWOPI * (f * 2.002) * t) * 0.20
    y += np.sin(TWOPI * (f * 3.005) * t) * 0.08

    env = np.exp(-t * 1.6) * velocity
    y *= env
    return apply_highpass(y, 80.0).astype(np.float32)

# =============================================================================
# MAIN MUSIC RENDERER
# =============================================================================

def _render_music_long(
    bpm: int,
    key: str,
    rnd: np.random.RandomState,
    *,
    variant: int,
    genre: str,
    bars: int = 16,
    energy_curve: Optional[np.ndarray] = None
) -> np.ndarray:
    """
    Generate professional melodic layers with adaptive instruments.
    """
    beat = 60.0 / float(bpm)
    bar_samples = int(beat * 4.0 * SAMPLE_RATE)
    n = bar_samples * int(bars)
    
    if energy_curve is None:
        energy_curve = np.ones(n, dtype=np.float32) * 0.8
    
    g = _norm_genre(genre)
    st = GENRE_STYLES[g]
    root = key_to_root_hz(key, octave=3)
    
    x = np.zeros(n, dtype=np.float32)

    # 1. INSTRUMENT SELECTION (Adaptive)
    # Don't use the same "Stupid Organ" for everything.
    
    instrument = "saw" # Fallback
    
    if g in ("house", "deep"):
        # House: Mix of Organ, Piano, and Rhodes
        # Weighted to avoid organ fatigue
        instrument = rnd.choice(["organ", "piano_house", "rhodes", "organ"], p=[0.3, 0.4, 0.2, 0.1])
    elif g in ("jazz", "soul"):
        instrument = rnd.choice(["rhodes", "piano_simple"])
    elif g in ("trance", "edm"):
        instrument = rnd.choice(["pluck", "supersaw"])
    elif g in ("chillout", "lounge"):
        instrument = "rhodes"
    elif g in ("classic",):
        instrument = "piano_simple"

    # 2. PROGRESSIONS (Vibe)
    prog = [[0, 3, 7], [0, 4, 7], [-2, 2, 5], [5, 9, 12]] # Default
    
    if g == "jazz":
        prog = [[2, 5, 9, 12], [7, 11, 14, 17], [0, 4, 7, 11], [0, 4, 7, 11]]
    elif g in ("house", "deep"):
        # Classic House Minor 7ths
        prog = [[0, 3, 7, 10], [7, 10, 14, 17], [-4, 0, 3, 7], [5, 8, 12, 15]]

    # 3. RHYTHM STYLE
    rhythm_style = rnd.choice(["sustained", "plucks", "offbeat"])
    if g == "house": rhythm_style = "offbeat"
    if g == "trance": rhythm_style = "plucks"
    if g == "ambient": rhythm_style = "sustained"

    chord_len = bar_samples
    
    for bar_idx in range(bars):
        chord = prog[bar_idx % len(prog)]
        bar_start = bar_idx * bar_samples
        bar_energy = energy_curve[bar_start:min(n, bar_start + bar_samples)].mean()
        
        # --- RHYTHM LOGIC ---
        
        if rhythm_style == "sustained":
            # PADS
            end = min(n, bar_start + chord_len)
            dur = end - bar_start
            
            freqs = [semitone(root, s) for s in chord]
            chord_mix = np.zeros(dur, dtype=np.float32)
            
            for i, f in enumerate(freqs):
                if i == 1: f *= 2.0 # Spread voicing
                
                # Render Tone
                if instrument == "supersaw":
                    tone = osc.supersaw(f, dur, detune=0.12)
                elif instrument == "rhodes":
                    tone = _generate_tone_fm_rhodes(f, dur, velocity=0.65, rnd=rnd)
                else:
                    tone = osc.saw(f, dur, brightness=0.4)
                
                chord_mix += tone * (0.8 / len(freqs))
            
            env = _adsr(dur, 0.2, 0.2, 0.7 * bar_energy, 0.5)
            x[bar_start:end] += chord_mix * env

        elif rhythm_style in ("plucks", "offbeat"):
            # PATTERNS
            step_size = int(beat * 0.25 * SAMPLE_RATE)
            steps = 16
            
            for i in range(steps):
                step_pos = bar_start + i * step_size
                if step_pos >= n: break
                
                should_play = False
                velocity = bar_energy
                
                # House Groove
                if g in ("house", "deep"):
                    if i % 4 == 2: should_play = True # The "And"
                    if rnd.rand() < 0.15 and i % 4 == 3: should_play = True # Swing
                elif g in ("trance", "edm"): 
                    should_play = True
                elif g == "jazz":
                    if i in [0, 3, 6, 9, 12]: should_play = True

                if should_play:
                    dur = int(step_size * 0.8)
                    freqs = [semitone(root, s) for s in chord]
                    chord_mix = np.zeros(dur, dtype=np.float32)
                    
                    for f in freqs:
                        # ✅ ORGAN RANGE SAFETY:
                        # If instrument is Organ and pitch is too high (>800Hz), drop octave
                        if instrument == "organ" and f > 800: f *= 0.5
                        
                        if instrument == "organ":
                            tone = _generate_tone_organ_m1(f, dur, velocity, rnd)
                        elif instrument == "piano_house":
                            tone = _generate_tone_house_piano(f, dur, velocity, rnd)
                        elif instrument == "rhodes":
                            tone = _generate_tone_fm_rhodes(f, dur, velocity, rnd)
                        elif instrument == "pluck":
                            tone = _generate_tone_pluck_analog(f, dur, velocity, rnd)
                        elif instrument == "piano_simple":
                            tone = _generate_tone_piano_simple(f, dur, velocity, rnd)
                        else:
                            tone = osc.saw(f, dur, brightness=0.7) * np.exp(-np.linspace(0,10,dur))
                            
                        chord_mix += tone * (1.0 / len(freqs))
                    
                    x[step_pos:step_pos+dur] += chord_mix * 0.9

    # --- MIXING ---
    x = stereo_width(pan(x, 0.0), 1.5)
    
    # FX per instrument
    if instrument == "rhodes":
        x = apply_algorithmic_reverb(x, room_size=0.5, wet=0.25)
    elif instrument == "organ":
        x = apply_highpass(x, 180.0) # Keep clean
        x = apply_algorithmic_reverb(x, room_size=0.4, wet=0.15)
    elif instrument == "piano_house":
        x = apply_highpass(x, 150.0)
        x = apply_algorithmic_reverb(x, room_size=0.6, wet=0.20)

    # Sidechain
    x = apply_sidechain_envelope(x, bpm, duck_amount=st["duck"])

    return x.astype(np.float32)

def _render_texture(seconds: float, rnd: np.random.RandomState, kind: str) -> np.ndarray:
    """Generate ambient texture (Optional)"""
    n = int(seconds * SAMPLE_RATE)
    kind = str(kind).lower().strip()
    
    if kind == "vinyl":
        x = rnd.randn(n).astype(np.float32) * 0.02
        x = apply_highpass(x, 80.0)
        x = apply_lowpass(x, 5500.0)
        for _ in range(10 + rnd.randint(0, 18)):
            pos = rnd.randint(0, max(1, n - 220))
            click = rnd.uniform(-1, 1, 220).astype(np.float32) * 0.05
            x[pos:pos+220] += click
        return x
    
    if kind == "rain":
        x = rnd.randn(n).astype(np.float32) * 0.05
        x = apply_highpass(x, 150.0)
        x = apply_lowpass(x, 6500.0)
        return x
    
    # Default: light air noise
    x = rnd.randn(n).astype(np.float32) * 0.015
    x = apply_highpass(x, 300.0)
    return x

# =============================================================================
# MAIN PRODUCTION ENTRY: RENDER ANY STEM (WITH ENERGY CURVES)
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
    bars: int = 16,
    duration_sec: Optional[float] = None,  # ✅ AUTO-CALC BARS
    texture_type: str = "vinyl",
    energy_curve: str = "peak",
) -> None:
    """
    Render professional DJ-quality stem.
    
    Args:
        stem: "kick" | "drums" | "bass" | "music" | "pad" | "texture"
        genre: Genre style
        bpm: Tempo
        key: Musical key
        seed: Deterministic seed
        variant: Variation number
        bars: Length in bars (default 16)
        duration_sec: IF SET, OVERRIDES 'bars' to fill this duration
        energy_curve: "peak" | "drop" | "linear" | "build"
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    stem_l = str(stem).lower().strip()
    
    # Calculate beat info
    beat = 60.0 / float(max(1e-6, bpm))
    bar_duration = beat * 4.0

    # ✅ AUTO-CALCULATE BARS IF DURATION IS PROVIDED
    if duration_sec is not None and duration_sec > 0:
        # Calculate how many bars fit in the requested duration
        # We use ceil to ensure we cover the full duration
        bars = int(np.ceil(duration_sec / bar_duration))
    
    # Ensure minimum length
    bars = max(1, bars)
    
    # Init RNG with the calculated bar length so structure matches
    rnd = _rng(seed, salt=f"{genre}:{stem_l}:{bpm}:{key}:{bars}", variant=int(variant))
    
    # Calculate exact sample length based on grid (bars)
    seconds = bar_duration * float(bars)
    n = int(seconds * SAMPLE_RATE)
    
    # Create energy curve
    energy_env = create_energy_curve(n, curve_type=energy_curve, rnd=rnd)
    
    if stem_l == "kick":
        x = _render_kick_long(bpm, rnd, variant=variant, genre=genre, bars=bars, energy_curve=energy_env)
        save_wav(out_path, x)
        return
    
    if stem_l == "drums":
        x = _render_drums_long(bpm, rnd, variant=variant, genre=genre, bars=bars, energy_curve=energy_env)
        save_wav(out_path, x)
        return
    
    if stem_l == "bass":
        x = _render_bass_long(bpm, key, rnd, variant=variant, genre=genre, bars=bars, energy_curve=energy_env)
        save_wav(out_path, x)
        return
    
    if stem_l in ("music", "pad", "synth", "chords", "melody"):
        x = _render_music_long(bpm, key, rnd, variant=variant, genre=genre, bars=bars, energy_curve=energy_env)
        save_wav(out_path, x)
        return
    
    if stem_l in ("texture", "ambience"):
        # For texture, we just use the raw seconds logic from the override or bars
        x = _render_texture(seconds, rnd, kind=texture_type)
        save_wav(out_path, x)
        return
    
    raise ValueError(f"Unknown stem type: {stem}")

# =============================================================================
# BACKWARD-COMPAT EXPORTS (NOW GENERATE LONGER STEMS)
# =============================================================================

def generate_techno_kick(out_path: Path, bpm: int = 130, variant: int = 1, seed: Optional[str | int] = None, **_ignored: Any) -> None:
    render_stem(out_path, stem="kick", genre="Techno", bpm=bpm, key="A", seed=seed, variant=variant, bars=16)

def generate_techno_bass(out_path: Path, bpm: int = 130, variant: int = 1, seed: Optional[str | int] = None, **_ignored: Any) -> None:
    render_stem(out_path, stem="bass", genre="Techno", bpm=bpm, key="A", seed=seed, variant=variant, bars=16)

def generate_techno_arp(out_path: Path, bpm: int = 130, variant: int = 1, seed: Optional[str | int] = None, **_ignored: Any) -> None:
    render_stem(out_path, stem="music", genre="Techno", bpm=bpm, key="A", seed=seed, variant=variant, bars=16)

def generate_house_drums(out_path: Path, bpm: int = 124, variant: int = 1, seed: Optional[str | int] = None, **_ignored: Any) -> None:
    render_stem(out_path, stem="drums", genre="House", bpm=bpm, key="A", seed=seed, variant=variant, bars=16)

def generate_deep_house_bass(out_path: Path, bpm: int = 124, variant: int = 1, seed: Optional[str | int] = None, **_ignored: Any) -> None:
    render_stem(out_path, stem="bass", genre="Deep", bpm=bpm, key="A", seed=seed, variant=variant, bars=16)

def generate_house_chords(out_path: Path, bpm: int = 124, seed: Optional[str | int] = None, **_ignored: Any) -> None:
    render_stem(out_path, stem="music", genre="House", bpm=bpm, key="A", seed=seed, variant=1, bars=16)

def generate_lofi_drums(out_path: Path, bpm: int = 85, variant: int = 1, seed: Optional[str | int] = None, **_ignored: Any) -> None:
    render_stem(out_path, stem="drums", genre="Chillout", bpm=bpm, key="C", seed=seed, variant=variant, bars=16)

def generate_lofi_keys(out_path: Path, bpm: int = 85, variant: int = 1, seed: Optional[str | int] = None, **_ignored: Any) -> None:
    render_stem(out_path, stem="music", genre="Chillout", bpm=bpm, key="C", seed=seed, variant=variant, bars=16)

def generate_wobble_bass(out_path: Path, bpm: int = 140, seed: Optional[str | int] = None, **_ignored: Any) -> None:
    render_stem(out_path, stem="bass", genre="Bass", bpm=bpm, key="A", seed=seed, variant=1, bars=16)

def generate_hard_kick(out_path: Path, bpm: int = 150, seed: Optional[str | int] = None, **_ignored: Any) -> None:
    render_stem(out_path, stem="kick", genre="Hard", bpm=bpm, key="A", seed=seed, variant=1, bars=16)

def generate_synth_bass(out_path: Path, bpm: int = 105, seed: Optional[str | int] = None, **_ignored: Any) -> None:
    render_stem(out_path, stem="bass", genre="Synth", bpm=bpm, key="A", seed=seed, variant=1, bars=16)

def generate_gated_snare(out_path: Path, bpm: int = 105, seed: Optional[str | int] = None, **_ignored: Any) -> None:
    render_stem(out_path, stem="drums", genre="Synth", bpm=bpm, key="A", seed=seed, variant=2, bars=16)

def generate_rave_piano(out_path: Path, bpm: int = 140, seed: Optional[str | int] = None, **_ignored: Any) -> None:
    render_stem(out_path, stem="music", genre="Dance", bpm=bpm, key="C", seed=seed, variant=2, bars=16)

def generate_texture(out_path: Path, type: str = "vinyl", seed: Optional[str | int] = None, **_ignored: Any) -> None:
    render_stem(out_path, stem="texture", genre="Ambient", bpm=90, key="A", seed=seed, variant=1, bars=16, texture_type=type)

# =============================================================================
# FOCUS ENGINE (BINAURAL)
# =============================================================================

FOCUS_PRESETS: Dict[str, Dict[str, float]] = {
    "deep_sleep": {"base_freq": 100.0, "beat_freq": 2.5, "noise_mix": 0.20},
    "meditation": {"base_freq": 150.0, "beat_freq": 6.0, "noise_mix": 0.25},
    "active_focus": {"base_freq": 250.0, "beat_freq": 20.0, "noise_mix": 0.30},
    "coding": {"base_freq": 300.0, "beat_freq": 40.0, "noise_mix": 0.25},
}

def generate_focus_session(
    out_path: Path,
    duration_sec: float,
    *,
    preset_name: str = "active_focus",
    add_rain: bool = False,
    base_freq: float | None = None,
    beat_freq: float | None = None,
    **_ignored: Any,
) -> None:
    """Generate binaural focus session"""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    
    preset = FOCUS_PRESETS.get(preset_name, FOCUS_PRESETS["active_focus"])
    bf = preset["base_freq"] if base_freq is None else base_freq
    beat = preset["beat_freq"] if beat_freq is None else beat_freq
    
    n = int(duration_sec * SAMPLE_RATE)
    t = np.arange(n, dtype=np.float32) / SAMPLE_RATE
    
    left = np.sin(TWOPI * bf * t).astype(np.float32)
    right = np.sin(TWOPI * (bf + beat) * t).astype(np.float32)
    
    if add_rain:
        rain = np.random.randn(n).astype(np.float32) * 0.03
        rain = apply_highpass(rain, 150.0)
        rain = apply_lowpass(rain, 6500.0)
        left += rain
        right += rain
    
    stereo = np.stack([left, right], axis=1) * 0.55
    stereo = normalize(stereo, 0.98)
    wavfile.write(str(out_path), SAMPLE_RATE, (stereo * 32767.0).astype(np.int16))

# =============================================================================
# LEGACY INFINITE EXPORTS
# =============================================================================

def generate_infinite_kick(out_path: Path, bpm: int, seed: int, energy: float = 0.8) -> None:
    render_stem(out_path, stem="kick", genre="Techno", bpm=bpm, key="A", seed=seed, variant=1, bars=16)

def generate_infinite_bass(out_path: Path, bpm: int, seed: int, genre: str = "techno") -> None:
    render_stem(out_path, stem="bass", genre=genre, bpm=bpm, key="A", seed=seed, variant=1, bars=16)

def generate_infinite_drums(out_path: Path, bpm: int, seed: int, type: str = "hats") -> None:
    render_stem(out_path, stem="drums", genre="House", bpm=bpm, key="A", seed=seed, variant=1, bars=16)

def generate_infinite_pad(out_path: Path, bpm: int, seed: int) -> None:
    render_stem(out_path, stem="music", genre="Ambient", bpm=bpm, key="A", seed=seed, variant=1, bars=16)