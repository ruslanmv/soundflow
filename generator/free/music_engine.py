# generator/free/music_engine.py
from __future__ import annotations

import numpy as np
import scipy.io.wavfile as wavfile
import random
from pathlib import Path
from scipy import signal

SAMPLE_RATE = 44100

# -----------------------------------------------------------------------------
# DSP UTILS
# -----------------------------------------------------------------------------

def normalize(audio: np.ndarray) -> np.ndarray:
    max_val = float(np.max(np.abs(audio))) if audio.size else 0.0
    return (audio / max_val) if max_val > 0 else audio

def apply_overdrive(audio: np.ndarray, drive: float = 0.5) -> np.ndarray:
    """Tube distortion simulation."""
    return np.tanh(audio * (1.0 + drive))

def apply_bitcrush(audio: np.ndarray, bits: int = 8) -> np.ndarray:
    """Digital degradation."""
    steps = 2 ** int(bits)
    return np.round(audio * steps) / steps

def apply_delay(audio: np.ndarray, bpm: float, feedback: float = 0.35, mix: float = 0.25, dotted: bool = False) -> np.ndarray:
    """
    Delay effect (safe).
    NOTE: kept simple (feed-forward) to avoid muddy build-up.
    """
    if mix <= 0.0:
        return audio
    mult = 0.75 if dotted else 1.0
    delay_samples = int((60.0 / float(bpm)) * mult * SAMPLE_RATE)
    if delay_samples <= 0 or delay_samples >= len(audio):
        return audio
    out = np.copy(audio)
    delayed = np.zeros_like(audio)
    delayed[delay_samples:] = audio[:-delay_samples]
    # light feedback (optional)
    if feedback > 0:
        out = out + delayed * (mix * (1.0 + feedback * 0.15))
    else:
        out = out * (1 - mix) + delayed * mix
    return out

def apply_reverb_ir(audio: np.ndarray, decay: float = 0.45, wet: float = 0.25) -> np.ndarray:
    """Simple convolution reverb simulation using a shaped noise IR."""
    if wet <= 0:
        return audio
    ir_len = max(32, int(SAMPLE_RATE * float(decay)))
    t = np.linspace(0, 1, ir_len, False)
    ir = np.random.uniform(-1, 1, ir_len).astype(np.float32)
    ir *= np.exp(-10.0 * t).astype(np.float32)
    ir = normalize(ir) * 0.5
    wet_sig = signal.fftconvolve(audio, ir, mode="same")
    return audio * (1.0 - wet) + wet_sig * wet

def apply_lowpass(audio: np.ndarray, cutoff: float) -> np.ndarray:
    nyquist = 0.5 * SAMPLE_RATE
    norm_cutoff = float(cutoff) / nyquist
    norm_cutoff = min(0.99, max(1e-6, norm_cutoff))
    b, a = signal.butter(1, norm_cutoff, btype="low", analog=False)
    return signal.lfilter(b, a, audio)

def apply_highpass(audio: np.ndarray, cutoff: float) -> np.ndarray:
    nyquist = 0.5 * SAMPLE_RATE
    norm_cutoff = float(cutoff) / nyquist
    norm_cutoff = min(0.99, max(1e-6, norm_cutoff))
    b, a = signal.butter(1, norm_cutoff, btype="high", analog=False)
    return signal.lfilter(b, a, audio)

def apply_resonant_filter(audio: np.ndarray, cutoff: float, resonance: float) -> np.ndarray:
    """
    Simulates a resonant low-pass filter (safe).
    `resonance` kept for API compatibility; simple 2nd order LPF here.
    """
    nyquist = 0.5 * SAMPLE_RATE
    norm_cutoff = float(cutoff) / nyquist
    norm_cutoff = min(0.99, max(1e-6, norm_cutoff))
    b, a = signal.butter(2, norm_cutoff, btype="low", analog=False)
    return signal.lfilter(b, a, audio)

def adsr(length: int, attack: float = 0.01, decay: float = 0.1, sustain: float = 0.7, release: float = 0.2) -> np.ndarray:
    total = int(length)
    a = int(attack * SAMPLE_RATE)
    d = int(decay * SAMPLE_RATE)
    r = int(release * SAMPLE_RATE)
    if a + d + r > total and (a + d + r) > 0:
        factor = total / float(a + d + r)
        a, d, r = int(a * factor), int(d * factor), int(r * factor)
    s_len = total - a - d - r
    if s_len < 0:
        s_len = 0
    env = np.concatenate([
        np.linspace(0, 1, a, endpoint=False) if a > 0 else np.array([], dtype=np.float32),
        np.linspace(1, sustain, d, endpoint=False) if d > 0 else np.array([], dtype=np.float32),
        np.full(s_len, sustain, dtype=np.float32),
        np.linspace(sustain, 0, r, endpoint=True) if r > 0 else np.array([], dtype=np.float32),
    ])
    if len(env) < total:
        env = np.pad(env, (0, total - len(env)))
    return env[:total].astype(np.float32)

# -----------------------------------------------------------------------------
# NOISE HELPERS (reduced + higher quality)
# -----------------------------------------------------------------------------

def shaped_noise(n: int, *, level: float = 0.03, lp: float = 6500.0, hp: float = 120.0) -> np.ndarray:
    """
    Controlled noise: band-limited + low level.
    This replaces harsh full-band noise that ruins mixes.
    """
    x = np.random.normal(0.0, 1.0, int(n)).astype(np.float32)
    x = apply_highpass(x, hp)
    x = apply_lowpass(x, lp)
    x = normalize(x) * float(level)
    return x

def click_noise(n: int, *, level: float = 0.12) -> np.ndarray:
    """Short clicky transient noise with safe level."""
    t = np.linspace(0, 1, int(n), False).astype(np.float32)
    x = np.random.uniform(-1, 1, int(n)).astype(np.float32)
    x *= np.exp(-80.0 * t).astype(np.float32)
    x = normalize(x) * float(level)
    return x

def vinyl_crackle(samples: int, *, bed_level: float = 0.006, clicks: int = 14, click_level: float = 0.08) -> np.ndarray:
    """
    Much quieter vinyl: tiny bed + fewer softer clicks.
    """
    bed = np.random.uniform(-1, 1, samples).astype(np.float32) * float(bed_level)
    for _ in range(int(clicks)):
        idx = random.randint(0, samples - 1)
        bed[idx] += float(click_level)
    bed = apply_lowpass(bed, 6000.0)
    bed = apply_highpass(bed, 80.0)
    return bed.astype(np.float32)

def rain_bed(samples: int, *, level: float = 0.012) -> np.ndarray:
    """
    Natural-ish rain: band-limited noise with slow amplitude drift.
    Much quieter than before.
    """
    x = shaped_noise(samples, level=level, lp=5200.0, hp=150.0)
    # slow drift (avoid static hiss feel)
    t = np.linspace(0, 1, samples, False).astype(np.float32)
    drift = 0.75 + 0.25 * np.sin(2 * np.pi * 0.12 * t).astype(np.float32)
    return (x * drift).astype(np.float32)

# -----------------------------------------------------------------------------
# GENRE 1: TECHNO / TRANCE (128-138 BPM)
# -----------------------------------------------------------------------------

def generate_techno_kick(out_path: Path, bpm: int = 130, variant: int = 1):
    """
    Modern Techno Kick with controlled rumble.
    Reduced noise + rumble gain so it doesn't crush the mix.
    """
    beat_dur = 60.0 / float(bpm)
    total_samples = int(SAMPLE_RATE * beat_dur * 4 * 4)
    audio = np.zeros(total_samples, dtype=np.float32)

    # 1. Main Kick
    k_len = 0.30
    t = np.linspace(0, k_len, int(SAMPLE_RATE * k_len), False).astype(np.float32)
    freq_sweep = 200.0 * np.exp(-20.0 * t) + 40.0
    kick_core = np.sin(2 * np.pi * freq_sweep * t) * np.exp(-5.0 * t)

    # ✅ Much quieter / band-limited click (was too loud and full-band)
    click = click_noise(len(t), level=0.07)
    main_kick = kick_core + click
    main_kick = apply_overdrive(main_kick, 1.7)
    main_kick = apply_lowpass(main_kick, 9000.0)

    # 2. Rumble (reverb tail) — controlled
    burst = shaped_noise(int(SAMPLE_RATE * 0.10), level=0.08, lp=2800.0, hp=120.0)
    rumble = apply_reverb_ir(burst, decay=0.35, wet=0.35)
    rumble = apply_overdrive(rumble, 2.2)
    rumble = apply_resonant_filter(rumble, 150.0, 0.0)
    rumble = apply_lowpass(rumble, 1200.0)

    # Ducking
    duck_len = int(SAMPLE_RATE * 0.15)
    if duck_len < len(rumble):
        rumble[:duck_len] *= (np.linspace(0, 1, duck_len, endpoint=False).astype(np.float32) ** 2)

    # Place hits
    for i in range(16):
        start = int(i * beat_dur * SAMPLE_RATE)
        if start + len(main_kick) < len(audio):
            audio[start:start + len(main_kick)] += main_kick * 0.85
        if start + len(rumble) < len(audio):
            # ✅ reduced rumble level (was 0.6)
            audio[start:start + len(rumble)] += rumble * 0.35

    save_wav(out_path, audio)

def generate_techno_bass(out_path: Path, bpm: int = 128, key_freq: float = 49.0, variant: int = 1):
    """
    Rolling Trance Bass or Acid Line depending on variant.
    """
    beat_dur = 60.0 / float(bpm)
    step_dur = beat_dur / 4.0
    total_samples = int(SAMPLE_RATE * beat_dur * 4 * 4)
    audio = np.zeros(total_samples, dtype=np.float32)

    if variant == 1:
        t = np.linspace(0, step_dur, int(SAMPLE_RATE * step_dur), False).astype(np.float32)
        saw = signal.sawtooth(2 * np.pi * float(key_freq) * t).astype(np.float32)
        bass_hit = saw * adsr(len(saw), 0.005, 0.10, 0.35, 0.05)
        bass_hit = apply_lowpass(bass_hit, 900.0)

        for i in range(64):
            if i % 4 != 0:
                start = int(i * step_dur * SAMPLE_RATE)
                if start + len(bass_hit) < len(audio):
                    audio[start:start + len(bass_hit)] += bass_hit * 0.75
    else:
        for i in range(64):
            pitch = float(key_freq) * (1.0 if i % 8 < 4 else 2.0)
            t = np.linspace(0, step_dur, int(SAMPLE_RATE * step_dur), False).astype(np.float32)
            raw = signal.sawtooth(2 * np.pi * pitch * t).astype(np.float32)

            lfo = 450.0 * np.sin(i * 0.2)
            filtered = apply_resonant_filter(raw, 800.0 + lfo, 0.8)
            distorted = apply_overdrive(filtered, 2.3)

            env = adsr(len(distorted), 0.01, 0.10, 0.0, 0.10)
            start = int(i * step_dur * SAMPLE_RATE)
            if start + len(distorted) < len(audio):
                audio[start:start + len(distorted)] += distorted * env * 0.60

    save_wav(out_path, audio)

def generate_techno_arp(out_path: Path, bpm: int = 138, key_freq: float = 196.0, variant: int = 1):
    """
    Detuned Supersaw Lead (Trance).
    """
    beat_dur = 60.0 / float(bpm)
    step_dur = beat_dur / 4.0
    total_samples = int(SAMPLE_RATE * beat_dur * 4 * 4)
    audio = np.zeros(total_samples, dtype=np.float32)

    def make_supersaw(freq: float, dur: float) -> np.ndarray:
        t = np.linspace(0, dur, int(SAMPLE_RATE * dur), False).astype(np.float32)
        detunes = [0.992, 0.996, 1.0, 1.004, 1.008]
        mix = np.zeros_like(t)
        for d in detunes:
            mix += signal.sawtooth(2 * np.pi * (freq * d) * t).astype(np.float32)
        return (mix / len(detunes)).astype(np.float32)

    base_midi = 12 * np.log2(float(key_freq) / 440.0) + 69
    scale = [0, 2, 4, 5, 7, 9, 11]

    for i in range(64):
        idx = scale[i % len(scale)] + (12 if i % 16 > 8 else 0)
        freq = 440.0 * 2 ** ((base_midi + idx - 69) / 12)

        note = make_supersaw(freq, step_dur)
        note *= adsr(len(note), 0.01, 0.10, 0.35, 0.25)

        start = int(i * step_dur * SAMPLE_RATE)
        if start + len(note) < len(audio):
            audio[start:start + len(note)] += note * 0.35

    audio = apply_delay(audio, bpm, dotted=True, mix=0.28, feedback=0.25)
    audio = apply_lowpass(audio, 7000.0)
    save_wav(out_path, audio)

# -----------------------------------------------------------------------------
# GENRE 2: HOUSE / DEEP (124 BPM)
# -----------------------------------------------------------------------------

def generate_house_drums(out_path: Path, bpm: int = 124, variant: int = 1):
    beat_dur = 60.0 / float(bpm)
    total_samples = int(SAMPLE_RATE * beat_dur * 4 * 4)
    audio = np.zeros(total_samples, dtype=np.float32)

    k_len = 0.20
    kt = np.linspace(0, k_len, int(SAMPLE_RATE * k_len), False).astype(np.float32)
    kick = np.sin(2 * np.pi * 100 * np.exp(-15 * kt) * kt) * np.exp(-5 * kt)
    kick = apply_overdrive(kick, 0.5)

    c_len = 0.15
    clap = shaped_noise(int(SAMPLE_RATE * c_len), level=0.10, lp=7000.0, hp=250.0)
    clap *= np.exp(-20 * np.linspace(0, c_len, int(SAMPLE_RATE * c_len), False)).astype(np.float32)

    h_len = 0.10
    hat = shaped_noise(int(SAMPLE_RATE * h_len), level=0.07, lp=9000.0, hp=3500.0)
    hat *= np.exp(-15 * np.linspace(0, h_len, int(SAMPLE_RATE * h_len), False)).astype(np.float32)

    for bar in range(4):
        for beat in range(4):
            t0 = int((bar * 4 + beat) * beat_dur * SAMPLE_RATE)
            if t0 + len(kick) < len(audio):
                audio[t0:t0 + len(kick)] += kick * 0.85

            if beat in [1, 3]:
                if t0 + len(clap) < len(audio):
                    audio[t0:t0 + len(clap)] += clap * 0.55

            t_off = int((bar * 4 + beat + 0.5) * beat_dur * SAMPLE_RATE)
            if t_off + len(hat) < len(audio):
                audio[t_off:t_off + len(hat)] += hat * 0.45

    save_wav(out_path, audio)

def generate_deep_house_bass(out_path: Path, bpm: int = 124, key_freq: float = 49.0, variant: int = 1):
    beat_dur = 60.0 / float(bpm)
    total_samples = int(SAMPLE_RATE * beat_dur * 4 * 4)
    audio = np.zeros(total_samples, dtype=np.float32)

    note_dur = beat_dur * 0.5
    t = np.linspace(0, note_dur, int(SAMPLE_RATE * note_dur), False).astype(np.float32)
    wave = np.sin(2 * np.pi * float(key_freq) * t).astype(np.float32)
    wave *= adsr(len(wave), 0.01, 0.20, 0.35, 0.10)

    pattern = [1, 0, 0, 1, 0, 1, 0, 0]
    for i in range(32):
        if pattern[i % 8]:
            start = int(i * note_dur * SAMPLE_RATE)
            if start + len(wave) < len(audio):
                audio[start:start + len(wave)] += wave * 0.75

    save_wav(out_path, audio)

def generate_house_chords(out_path: Path, bpm: int = 124, key_freq: float = 261.63):
    beat_dur = 60.0 / float(bpm)
    total_samples = int(SAMPLE_RATE * beat_dur * 4 * 4)
    audio = np.zeros(total_samples, dtype=np.float32)

    chord_len = 0.20
    t = np.linspace(0, chord_len, int(SAMPLE_RATE * chord_len), False).astype(np.float32)
    intervals = [1, 1.2, 1.5, 1.78]
    stab = np.zeros_like(t)
    for ratio in intervals:
        f = float(key_freq) * float(ratio)
        stab += np.sin(2 * np.pi * f * t).astype(np.float32)
        stab += 0.5 * np.sin(2 * np.pi * (f * 2.0) * t).astype(np.float32)
    stab *= np.exp(-10 * t).astype(np.float32)
    stab = apply_lowpass(stab, 6500.0)

    for bar in range(4):
        for beat in range(4):
            if beat in [1, 3]:
                start = int((bar * 4 + beat + 0.25) * beat_dur * SAMPLE_RATE)
                if start + len(stab) < len(audio):
                    audio[start:start + len(stab)] += stab * 0.42

    audio = apply_delay(audio, bpm, dotted=True, mix=0.22, feedback=0.20)
    save_wav(out_path, audio)

# -----------------------------------------------------------------------------
# GENRE 3: LO-FI / CHILL (85 BPM)
# -----------------------------------------------------------------------------

def generate_lofi_drums(out_path: Path, bpm: int = 85, variant: int = 1):
    beat_dur = 60.0 / float(bpm)
    total_samples = int(SAMPLE_RATE * beat_dur * 4 * 4)
    audio = np.zeros(total_samples, dtype=np.float32)

    k_len = 0.40
    kt = np.linspace(0, k_len, int(SAMPLE_RATE * k_len), False).astype(np.float32)
    kick = np.sin(2 * np.pi * 60 * kt).astype(np.float32) * np.exp(-10 * kt).astype(np.float32)

    s_len = 0.20
    snare = shaped_noise(int(SAMPLE_RATE * s_len), level=0.10, lp=2200.0, hp=200.0)
    snare *= np.exp(-15 * np.linspace(0, s_len, int(SAMPLE_RATE * s_len), False)).astype(np.float32)

    for i in range(4):
        offset = i * 4 * beat_dur
        for bt in [0, 2.5]:
            pos = int((offset + bt * beat_dur) * SAMPLE_RATE)
            if pos + len(kick) < len(audio):
                audio[pos:pos + len(kick)] += kick * 0.90
        for bt in [1, 3]:
            pos = int((offset + bt * beat_dur) * SAMPLE_RATE)
            if pos + len(snare) < len(audio):
                audio[pos:pos + len(snare)] += snare * 0.55

    save_wav(out_path, audio)

def generate_lofi_keys(out_path: Path, bpm: int = 85, key_freq: float = 261.63, variant: int = 1):
    duration = (60.0 / float(bpm)) * 4 * 4
    t = np.linspace(0, duration, int(SAMPLE_RATE * duration), False).astype(np.float32)

    vibrato = 1.0 + 0.003 * np.sin(2 * np.pi * 4.0 * t).astype(np.float32)
    audio = np.zeros_like(t)

    bar_len = int(SAMPLE_RATE * (60.0 / float(bpm)) * 4)
    freqs = [float(key_freq), float(key_freq) * 1.2, float(key_freq) * 1.5]

    for b in range(4):
        start = b * bar_len
        for i, f in enumerate(freqs):
            actual_start = start + i * 2000
            if actual_start < len(audio):
                rem_len = len(audio) - actual_start
                lt = t[:rem_len]
                val = np.sin(2 * np.pi * (f * vibrato[:rem_len]) * lt).astype(np.float32) * np.exp(-1.5 * lt).astype(np.float32)
                audio[actual_start:] += val * 0.28

    audio = apply_lowpass(audio, 1800.0)
    save_wav(out_path, audio)

def generate_texture(out_path: Path, type: str = "vinyl"):
    """
    Textures are now MUCH quieter by default.
    This prevents noise from destroying music quality.
    """
    sec = 16
    samples = int(SAMPLE_RATE * sec)

    if type == "vinyl":
        audio = vinyl_crackle(samples, bed_level=0.006, clicks=14, click_level=0.08)
    else:
        # rain
        audio = rain_bed(samples, level=0.012)

    save_wav(out_path, audio)

# -----------------------------------------------------------------------------
# GENRE 4: BASS / DUBSTEP (140 BPM)
# -----------------------------------------------------------------------------

def generate_wobble_bass(out_path: Path, bpm: int = 140, key_freq: float = 55.0):
    beat_dur = 60.0 / float(bpm)
    total_samples = int(SAMPLE_RATE * beat_dur * 4 * 4)
    audio = np.zeros(total_samples, dtype=np.float32)

    t = np.linspace(0, beat_dur * 16, total_samples, False).astype(np.float32)

    cursor = 0
    bar_samples = int(SAMPLE_RATE * beat_dur * 4)
    for speed in [3, 6, 12, 4]:
        if cursor + bar_samples > len(t):
            break
        tc = t[cursor:cursor + bar_samples]
        chunk = signal.sawtooth(2 * np.pi * float(key_freq) * tc).astype(np.float32)
        lfo = 0.5 * (1 + np.sin(2 * np.pi * speed * tc).astype(np.float32))
        chunk = apply_bitcrush(chunk * lfo, 6)
        audio[cursor:cursor + bar_samples] = chunk * 0.75
        cursor += bar_samples

    audio = apply_lowpass(audio, 9000.0)
    save_wav(out_path, audio)

# -----------------------------------------------------------------------------
# GENRE 5: HARDSTYLE / HARD (150 BPM)
# -----------------------------------------------------------------------------

def generate_hard_kick(out_path: Path, bpm: int = 150):
    beat_dur = 60.0 / float(bpm)
    total_samples = int(SAMPLE_RATE * beat_dur * 4 * 4)
    audio = np.zeros(total_samples, dtype=np.float32)

    k_len = 0.40
    t = np.linspace(0, k_len, int(SAMPLE_RATE * k_len), False).astype(np.float32)

    freq = 200.0 * np.exp(-10.0 * t) + 40.0
    raw = np.sin(2 * np.pi * freq * t).astype(np.float32)
    distorted = apply_overdrive(np.clip(raw * 5.0, -0.8, 0.8), 1.8)

    for i in range(16):
        start = int(i * beat_dur * SAMPLE_RATE)
        if start + len(distorted) < len(audio):
            audio[start:start + len(distorted)] += distorted * 0.85

    save_wav(out_path, audio)

# -----------------------------------------------------------------------------
# GENRE 6: SYNTHWAVE (105 BPM)
# -----------------------------------------------------------------------------

def generate_synth_bass(out_path: Path, bpm: int = 105, key_freq: float = 65.4):
    beat_dur = 60.0 / float(bpm)
    total_samples = int(SAMPLE_RATE * beat_dur * 4 * 4)
    audio = np.zeros(total_samples, dtype=np.float32)

    note_dur = beat_dur * 0.5
    t = np.linspace(0, note_dur, int(SAMPLE_RATE * note_dur), False).astype(np.float32)
    wave = signal.sawtooth(2 * np.pi * float(key_freq) * t).astype(np.float32)
    note = wave * adsr(len(wave), 0.01, 0.10, 0.45, 0.10) * np.exp(-10 * t).astype(np.float32)

    for i in range(32):
        start = int(i * note_dur * SAMPLE_RATE)
        if start + len(note) < len(audio):
            audio[start:start + len(note)] += note * 0.62

    save_wav(out_path, audio)

def generate_gated_snare(out_path: Path, bpm: int = 105):
    beat_dur = 60.0 / float(bpm)
    total_samples = int(SAMPLE_RATE * beat_dur * 4 * 4)
    audio = np.zeros(total_samples, dtype=np.float32)

    s_len = 0.30
    t = np.linspace(0, s_len, int(SAMPLE_RATE * s_len), False).astype(np.float32)

    # ✅ safer noise: band-limited and lower
    noise = shaped_noise(len(t), level=0.10, lp=6000.0, hp=250.0)
    tone = np.sin(2 * np.pi * 180.0 * t).astype(np.float32) * 0.25
    raw = (noise + tone).astype(np.float32)

    gate = np.ones_like(t)
    gate[-int(len(t) * 0.2):] = 0.0
    snare = raw * gate

    for i in range(4):
        bar_off = i * 4 * beat_dur
        for b in [1, 3]:
            start = int((bar_off + b) * SAMPLE_RATE)
            if start + len(snare) < len(audio):
                audio[start:start + len(snare)] += snare * 0.70

    save_wav(out_path, audio)

# -----------------------------------------------------------------------------
# GENRE 7: EURO / RAVE (140 BPM)
# -----------------------------------------------------------------------------

def generate_rave_piano(out_path: Path, bpm: int = 140, key_freq: float = 440.0):
    beat_dur = 60.0 / float(bpm)
    total_samples = int(SAMPLE_RATE * beat_dur * 4 * 4)
    audio = np.zeros(total_samples, dtype=np.float32)

    chord_len = 0.20
    t = np.linspace(0, chord_len, int(SAMPLE_RATE * chord_len), False).astype(np.float32)
    stab = np.zeros_like(t)

    for f in [float(key_freq), float(key_freq) * 1.25, float(key_freq) * 1.5]:
        stab += np.sin(2 * np.pi * f * t).astype(np.float32)

    stab *= np.exp(-8 * t).astype(np.float32)
    stab = apply_lowpass(stab, 6500.0)

    for i in range(16):
        if i % 2 != 0:
            start = int(i * beat_dur * SAMPLE_RATE)
            if start + len(stab) < len(audio):
                audio[start:start + len(stab)] += stab * 0.55

    save_wav(out_path, audio)

# -----------------------------------------------------------------------------
# IO UTILS
# -----------------------------------------------------------------------------

def save_wav(path: Path, data: np.ndarray):
    path.parent.mkdir(parents=True, exist_ok=True)

    data = np.asarray(data, dtype=np.float32)
    # gentle safety clip before normalize
    data = np.clip(data, -1.0, 1.0)

    max_val = float(np.max(np.abs(data))) if data.size else 0.0
    if max_val > 0:
        data = data / max_val

    data_int = (data * 32767.0).astype(np.int16)
    wavfile.write(str(path), SAMPLE_RATE, data_int)
    print(f"🎹 Generated stem: {path.name}")
