# generator/free/music_engine.py
from __future__ import annotations

import numpy as np
import scipy.io.wavfile as wavfile
import random
from pathlib import Path
from scipy import signal
from typing import Optional, Tuple

SAMPLE_RATE = 44100

# =============================================================================
# CORE DSP UTILITIES - PROFESSIONAL GRADE
# =============================================================================

def normalize(audio: np.ndarray, target: float = 0.95) -> np.ndarray:
    """Normalize audio to target peak level (default -0.4dB)"""
    max_val = float(np.max(np.abs(audio))) if audio.size else 0.0
    return (audio * target / max_val) if max_val > 1e-6 else audio

def soft_clip(audio: np.ndarray, threshold: float = 0.9) -> np.ndarray:
    """Soft clipping for analog-style saturation"""
    return np.tanh(audio / threshold) * threshold

def apply_overdrive(audio: np.ndarray, drive: float = 1.5) -> np.ndarray:
    """Tube-style overdrive with harmonic enhancement"""
    driven = audio * (1.0 + drive)
    return np.tanh(driven) * 0.9

def apply_bitcrush(audio: np.ndarray, bits: int = 8) -> np.ndarray:
    """Lo-fi digital degradation"""
    steps = 2 ** max(1, int(bits))
    return np.round(audio * steps) / steps

# =============================================================================
# ADVANCED FILTER SYSTEM - 24dB/OCTAVE RESONANT FILTERS
# =============================================================================

def apply_lowpass(audio: np.ndarray, cutoff: float, order: int = 4) -> np.ndarray:
    """Professional 24dB/oct lowpass filter"""
    nyquist = SAMPLE_RATE / 2
    norm_cutoff = np.clip(cutoff / nyquist, 0.001, 0.99)
    b, a = signal.butter(order, norm_cutoff, btype='low')
    return signal.filtfilt(b, a, audio)  # Zero-phase

def apply_highpass(audio: np.ndarray, cutoff: float, order: int = 4) -> np.ndarray:
    """Professional 24dB/oct highpass filter"""
    nyquist = SAMPLE_RATE / 2
    norm_cutoff = np.clip(cutoff / nyquist, 0.001, 0.99)
    b, a = signal.butter(order, norm_cutoff, btype='high')
    return signal.filtfilt(b, a, audio)

def apply_bandpass(audio: np.ndarray, low_cut: float, high_cut: float) -> np.ndarray:
    """Bandpass filter for frequency isolation"""
    nyquist = SAMPLE_RATE / 2
    low_norm = np.clip(low_cut / nyquist, 0.001, 0.98)
    high_norm = np.clip(high_cut / nyquist, low_norm + 0.01, 0.99)
    b, a = signal.butter(3, [low_norm, high_norm], btype='band')
    return signal.filtfilt(b, a, audio)

def apply_resonant_filter(audio: np.ndarray, cutoff: float, resonance: float = 0.7) -> np.ndarray:
    """
    Moog-style resonant lowpass filter with self-oscillation capability.
    Resonance: 0.0 = flat response, 0.95 = near self-oscillation
    """
    nyquist = SAMPLE_RATE / 2
    norm_cutoff = np.clip(cutoff / nyquist, 0.001, 0.99)
    resonance = np.clip(resonance, 0.0, 0.95)
    
    if resonance < 0.1:
        # No resonance - use standard Butterworth
        b, a = signal.butter(4, norm_cutoff, btype='low')
    else:
        # Resonant filter - use Chebyshev Type 1
        ripple_db = resonance * 15.0  # 0-15dB resonant peak
        try:
            b, a = signal.cheby1(4, ripple_db, norm_cutoff, btype='low')
        except:
            # Fallback if parameters are invalid
            b, a = signal.butter(4, norm_cutoff, btype='low')
    
    return signal.filtfilt(b, a, audio)

def apply_parametric_eq(audio: np.ndarray, freq: float, gain_db: float, q: float = 1.0) -> np.ndarray:
    """Parametric EQ band (cut or boost)"""
    nyquist = SAMPLE_RATE / 2
    w0 = np.clip(freq / nyquist, 0.001, 0.99)
    
    gain_lin = 10 ** (gain_db / 40)  # Convert dB to linear
    
    # Simple peak filter
    b, a = signal.iirpeak(w0, q, fs=SAMPLE_RATE)
    filtered = signal.lfilter(b, a, audio)
    
    # Blend with original based on gain
    return audio + (filtered - audio) * (gain_lin - 1.0)

def multiband_process(audio: np.ndarray, 
                      low_gain: float = 1.0, 
                      mid_gain: float = 1.0, 
                      high_gain: float = 1.0) -> np.ndarray:
    """
    3-band multiband processing for professional mixing.
    Splits audio into sub/mid/high and processes separately.
    """
    # Split into bands
    low = apply_lowpass(audio, 250)
    mid = apply_bandpass(audio, 250, 4000)
    high = apply_highpass(audio, 4000)
    
    # Process each band
    low = low * low_gain
    mid = mid * mid_gain
    high = high * high_gain
    
    # Recombine
    return low + mid + high

# =============================================================================
# ENVELOPE GENERATORS
# =============================================================================

def adsr(length: int, attack: float = 0.01, decay: float = 0.1, 
         sustain: float = 0.7, release: float = 0.2) -> np.ndarray:
    """Professional ADSR envelope generator"""
    total = int(length)
    a = max(1, int(attack * SAMPLE_RATE))
    d = max(1, int(decay * SAMPLE_RATE))
    r = max(1, int(release * SAMPLE_RATE))
    
    if a + d + r > total:
        # Scale down if envelope is too long
        factor = total / (a + d + r)
        a, d, r = max(1, int(a * factor)), max(1, int(d * factor)), max(1, int(r * factor))
    
    s_len = max(0, total - a - d - r)
    
    # Build envelope with smooth curves
    attack_curve = np.linspace(0, 1, a) ** 0.5  # Exponential rise
    decay_curve = np.linspace(1, sustain, d) ** 2  # Exponential fall
    sustain_curve = np.full(s_len, sustain)
    release_curve = (np.linspace(sustain, 0, r) ** 2)  # Exponential fall
    
    env = np.concatenate([attack_curve, decay_curve, sustain_curve, release_curve])
    
    # Ensure exact length
    if len(env) < total:
        env = np.pad(env, (0, total - len(env)))
    return env[:total].astype(np.float32)

# =============================================================================
# ANTI-ALIASED OSCILLATOR SYSTEM
# =============================================================================

class AntiAliasedOscillator:
    """Professional oscillator with anti-aliasing for pristine sound quality"""
    
    def __init__(self, sample_rate: int = SAMPLE_RATE):
        self.sr = sample_rate
        
    def generate_saw(self, freq: float, duration: float, num_harmonics: int = 50) -> np.ndarray:
        """
        Anti-aliased sawtooth using additive synthesis.
        Prevents digital harshness and aliasing artifacts.
        """
        samples = int(duration * self.sr)
        t = np.linspace(0, duration, samples, endpoint=False, dtype=np.float32)
        
        # Calculate max harmonics below Nyquist
        max_harmonic = min(num_harmonics, int((self.sr / 2) / freq))
        
        output = np.zeros(samples, dtype=np.float32)
        
        # Sum harmonics with proper amplitude scaling
        for n in range(1, max_harmonic + 1):
            amplitude = 1.0 / n
            output += amplitude * np.sin(2 * np.pi * freq * n * t)
        
        # Normalize to proper sawtooth amplitude
        return output * (2.0 / np.pi)
    
    def generate_supersaw(self, freq: float, duration: float, 
                          num_voices: int = 7, detune: float = 0.08) -> np.ndarray:
        """
        Roland JP-8000 style supersaw - the signature Trance sound.
        Multiple detuned saw waves for thick, wide sound.
        """
        samples = int(duration * self.sr)
        output = np.zeros(samples, dtype=np.float32)
        
        # Create symmetric detune spread
        detune_amounts = np.linspace(-detune, detune, num_voices)
        
        for i, dt in enumerate(detune_amounts):
            detuned_freq = freq * (1.0 + dt)
            
            # Generate saw with slight random phase for naturalness
            voice = self.generate_saw(detuned_freq, duration)
            phase_offset = np.random.randint(0, samples // 20)
            voice = np.roll(voice, phase_offset)
            
            output += voice
        
        return output / num_voices  # Normalize
    
    def generate_pulse(self, freq: float, duration: float, width: float = 0.5) -> np.ndarray:
        """
        Pulse wave with variable width (50% = square wave).
        Uses additive synthesis for anti-aliasing.
        """
        samples = int(duration * self.sr)
        t = np.linspace(0, duration, samples, endpoint=False, dtype=np.float32)
        
        max_harmonic = min(50, int((self.sr / 2) / freq))
        output = np.zeros(samples, dtype=np.float32)
        
        # Fourier series for pulse wave
        for n in range(1, max_harmonic + 1, 2):  # Only odd harmonics
            amplitude = (4.0 / (np.pi * n)) * np.sin(n * np.pi * width)
            output += amplitude * np.sin(2 * np.pi * freq * n * t)
        
        return output

# Global oscillator instance
osc = AntiAliasedOscillator()

# =============================================================================
# PROFESSIONAL EFFECTS PROCESSORS
# =============================================================================

def apply_algorithmic_reverb(audio: np.ndarray, room_size: float = 0.5, 
                             damping: float = 0.5, wet: float = 0.3) -> np.ndarray:
    """
    Freeverb-style algorithmic reverb (industry standard).
    Much better than noise-based convolution.
    """
    if wet <= 0:
        return audio
    
    # Schroeder reverb: 4 parallel comb filters + 2 series allpass
    comb_delays = [1557, 1617, 1491, 1422]  # In samples (tuned for 44.1kHz)
    comb_gains = np.array([0.805, 0.827, 0.783, 0.764]) * room_size
    
    # Apply damping
    comb_gains *= (1.0 - damping * 0.3)
    
    output = np.zeros_like(audio, dtype=np.float32)
    
    # Parallel comb filters
    for delay_samples, gain in zip(comb_delays, comb_gains):
        comb_out = np.zeros_like(audio)
        for i in range(delay_samples, len(audio)):
            comb_out[i] = audio[i] + gain * comb_out[i - delay_samples]
        output += comb_out * 0.25
    
    # Series allpass filters (diffusion)
    allpass_delays = [225, 556]
    for ap_delay in allpass_delays:
        temp = np.zeros_like(output)
        for i in range(ap_delay, len(output)):
            temp[i] = -output[i] + output[i - ap_delay] + 0.5 * temp[i - ap_delay]
        output = temp
    
    # Mix dry/wet
    return audio * (1.0 - wet) + output * wet

def apply_stereo_delay(audio: np.ndarray, bpm: float, 
                       feedback: float = 0.35, mix: float = 0.25, 
                       dotted: bool = True) -> np.ndarray:
    """
    Ping-pong stereo delay synchronized to BPM.
    Creates spacious, rhythmic delay effects.
    """
    if mix <= 0:
        return audio
    
    # Convert to stereo if mono
    if len(audio.shape) == 1:
        audio = np.stack([audio, audio], axis=1)
    
    # Calculate delay time
    note_duration = (60.0 / bpm)
    delay_time = note_duration * (0.75 if dotted else 1.0)  # Dotted 8th or quarter
    delay_samples = int(delay_time * SAMPLE_RATE)
    
    if delay_samples <= 0 or delay_samples >= len(audio):
        return audio
    
    output = np.copy(audio)
    temp_audio = np.copy(audio)
    
    # Create ping-pong effect (4 taps)
    for tap in range(4):
        # Left to right
        delayed_l = np.zeros_like(temp_audio[:, 0])
        delayed_l[delay_samples:] = temp_audio[:-delay_samples, 0] * (feedback ** (tap + 1))
        
        # Right to left
        delayed_r = np.zeros_like(temp_audio[:, 1])
        delayed_r[delay_samples:] = temp_audio[:-delay_samples, 1] * (feedback ** (tap + 1))
        
        # Cross feedback (ping-pong)
        output[:, 1] += delayed_l * mix
        output[:, 0] += delayed_r * mix
        
        temp_audio[:, 0] = delayed_r
        temp_audio[:, 1] = delayed_l
    
    return output

def apply_sidechain_envelope(audio: np.ndarray, bpm: float, duck_amount: float = 0.7) -> np.ndarray:
    """
    Creates sidechain compression envelope (for applying to bass/pads).
    This simulates the "pumping" effect in EDM.
    """
    beat_dur = 60.0 / bpm
    beat_samples = int(beat_dur * SAMPLE_RATE)
    
    # Create ducking envelope
    envelope = np.ones(len(audio), dtype=np.float32)
    duck_len = int(SAMPLE_RATE * 0.15)  # 150ms duck
    
    # Number of beats in the audio
    num_beats = len(audio) // beat_samples
    
    for i in range(num_beats):
        start = i * beat_samples
        if start + duck_len < len(envelope):
            # Duck curve: 1.0 → duck_amount → 1.0
            duck_curve = np.concatenate([
                np.linspace(1.0, duck_amount, duck_len // 2) ** 2,
                np.linspace(duck_amount, 1.0, duck_len // 2) ** 2
            ])
            envelope[start:start + duck_len] = duck_curve
    
    return audio * envelope



# =============================================================================
# BINAURAL BEAT & FOCUS ENGINE
# =============================================================================

# Standard brainwave frequency ranges for the dashboard
BRAINWAVE_PRESETS = {
    "deep_sleep": {"base": 100, "beat": 2.5, "type": "Delta"},    # 0.5-4Hz: Deep sleep, healing
    "meditation": {"base": 150, "beat": 6.0, "type": "Theta"},    # 4-8Hz: Meditation, creativity
    "relaxation": {"base": 200, "beat": 10.0, "type": "Alpha"},   # 8-14Hz: Relaxed focus, stress reduction
    "active_focus": {"base": 250, "beat": 20.0, "type": "Beta"},  # 14-30Hz: Active thinking, problem solving
    "high_performance": {"base": 300, "beat": 40.0, "type": "Gamma"} # 30Hz+: Peak concentration, coding
}

def generate_binaural_beat(duration: float, 
                           base_freq: float = 200.0, 
                           beat_freq: float = 10.0, 
                           sample_rate: int = SAMPLE_RATE) -> np.ndarray:
    """
    Generates a stereo binaural beat.
    
    How it works:
    - Left Ear hears: base_freq (e.g., 200Hz)
    - Right Ear hears: base_freq + beat_freq (e.g., 210Hz)
    - Brain perceives: The difference (10Hz Alpha wave)
    
    Returns:
        (N, 2) numpy array (Stereo Audio)
    """
    samples = int(duration * sample_rate)
    t = np.linspace(0, duration, samples, endpoint=False).astype(np.float32)
    
    # Generate left and right channels
    # We use a lower amplitude (0.5) so it can be mixed behind music
    left_channel = 0.5 * np.sin(2 * np.pi * base_freq * t)
    right_channel = 0.5 * np.sin(2 * np.pi * (base_freq + beat_freq) * t)
    
    # Soften the attack/release so it doesn't click
    envelope = adsr(samples, attack=1.0, decay=0.0, sustain=1.0, release=1.0)
    left_channel *= envelope
    right_channel *= envelope
    
    # Stack into stereo (samples, 2)
    stereo_audio = np.stack([left_channel, right_channel], axis=1)
    
    return stereo_audio

def generate_focus_session(out_path: Path, 
                           preset_name: str = "active_focus", 
                           duration_sec: float = 60.0,
                           add_rain: bool = True):
    """
    Generates a full 'SoundFlow' Focus Session:
    1. Binaural Beat Layer (Hidden entrainment)
    2. Pink Noise / Rain Layer (Sound masking)
    3. Drone Layer (Musical anchor)
    """
    # 1. Get Settings
    settings = BRAINWAVE_PRESETS.get(preset_name, BRAINWAVE_PRESETS["active_focus"])
    print(f"🧠 Generating Session: {settings['type']} ({settings['beat']}Hz)")
    
    # 2. Generate Binaural Layer
    binaural = generate_binaural_beat(duration_sec, 
                                      base_freq=settings['base'], 
                                      beat_freq=settings['beat'])
    
    # 3. Generate Pink Noise / Rain Texture (for background masking)
    # We reuse your existing 'shaped_noise' but make it stereo
    noise_len = len(binaural)
    noise_l = shaped_noise(noise_len, level=0.05, lp=3000, hp=100)
    noise_r = shaped_noise(noise_len, level=0.05, lp=3000, hp=100)
    
    if add_rain:
        # Modulate noise to sound like rain
        t = np.linspace(0, duration_sec, noise_len, endpoint=False)
        # Slow random volume drift
        drift_l = 0.8 + 0.2 * np.sin(2 * np.pi * 0.2 * t)
        drift_r = 0.8 + 0.2 * np.sin(2 * np.pi * 0.23 * t + 1) # Slight offset
        noise_l *= drift_l
        noise_r *= drift_r
        
    noise_stereo = np.stack([noise_l, noise_r], axis=1)

    # 4. Generate Drone Pad (Musical Layer)
    # A simple sine wave chord that holds the root note
    t = np.linspace(0, duration_sec, noise_len, endpoint=False)
    drone_freq = settings['base'] / 2  # Octave lower
    drone_l = 0.1 * np.sin(2 * np.pi * drone_freq * t)
    drone_r = 0.1 * np.sin(2 * np.pi * (drone_freq * 1.01) * t) # Detuned slightly for width
    drone_stereo = np.stack([drone_l, drone_r], axis=1)
    
    # 5. Mix All Layers
    # Weights: Binaural (40%), Noise (30%), Drone (30%)
    mixed = (binaural * 0.4) + (noise_stereo * 0.3) + (drone_stereo * 0.3)
    
    # 6. Save (Handle Stereo Save Logic)
    # Since your original save_wav might expect mono, we ensure it handles stereo
    # (Scipy wavfile write handles 2D arrays automatically)
    save_wav(out_path, mixed)

# =============================================================================
# NOISE GENERATORS (LOW LEVEL, HIGH QUALITY)
# =============================================================================

def shaped_noise(n: int, level: float = 0.03, lp: float = 6500, hp: float = 120) -> np.ndarray:
    """Band-limited shaped noise for percussion/textures"""
    noise = np.random.normal(0, 1, int(n)).astype(np.float32)
    noise = apply_highpass(noise, hp, order=2)
    noise = apply_lowpass(noise, lp, order=2)
    return normalize(noise) * level

def click_transient(n: int, level: float = 0.15) -> np.ndarray:
    """Ultra-short click for kick attack"""
    t = np.linspace(0, 1, int(n), endpoint=False).astype(np.float32)
    click = np.random.uniform(-1, 1, int(n)).astype(np.float32)
    click *= np.exp(-100 * t)  # Very fast decay
    return normalize(click) * level

# =============================================================================
# PROFESSIONAL KICK DRUM GENERATOR - 3-LAYER SYSTEM
# =============================================================================

def generate_professional_kick(bpm: int = 128, style: str = "techno") -> np.ndarray:
    """
    Club-ready kick drum with 3 layers:
    1. SUB (20-60Hz) - Deep sub-bass punch
    2. BODY (60-200Hz) - Fundamental tone
    3. CLICK (2-8kHz) - Attack transient
    """
    beat_dur = 60.0 / bpm
    
    # === LAYER 1: SUB (The Deep Punch) ===
    sub_len = 0.35 if style == "techno" else 0.25
    t_sub = np.linspace(0, sub_len, int(SAMPLE_RATE * sub_len), endpoint=False).astype(np.float32)
    
    # Pitch envelope: starts at 70Hz, drops to 38Hz
    freq_env_sub = 70 * np.exp(-14 * t_sub) + 38
    phase_sub = np.cumsum(freq_env_sub) * (2 * np.pi / SAMPLE_RATE)
    sub_layer = np.sin(phase_sub).astype(np.float32)
    
    # Add subharmonic for extra weight
    sub_layer += 0.35 * np.sin(phase_sub / 2)
    
    # Amplitude envelope - fast attack, controlled decay
    sub_layer *= np.exp(-9 * t_sub)
    
    # Soft saturation for analog character
    sub_layer = np.tanh(sub_layer * 1.3) * 0.9
    
    # === LAYER 2: BODY (The Fundamental) ===
    body_len = 0.22
    t_body = np.linspace(0, body_len, int(SAMPLE_RATE * body_len), endpoint=False).astype(np.float32)
    
    # Pitch envelope for body (higher fundamental)
    freq_env_body = 180 * np.exp(-18 * t_body) + 75
    phase_body = np.cumsum(freq_env_body) * (2 * np.pi / SAMPLE_RATE)
    body_layer = np.sin(phase_body).astype(np.float32)
    
    # Add 2nd harmonic for richness
    body_layer += 0.45 * np.sin(phase_body * 2)
    
    # Sharper envelope
    body_layer *= np.exp(-11 * t_body)
    
    # Gentle overdrive
    body_layer = np.tanh(body_layer * 1.8) * 0.85
    
    # === LAYER 3: CLICK (The Attack) ===
    click_len = 0.008  # 8ms
    click_layer = click_transient(int(SAMPLE_RATE * click_len), level=0.25)
    click_layer = apply_bandpass(click_layer, 2000, 8000)
    
    # === COMBINE LAYERS ===
    max_len = max(len(sub_layer), len(body_layer), len(click_layer))
    kick = np.zeros(max_len, dtype=np.float32)
    
    # Layer with proper gain staging
    kick[:len(sub_layer)] += sub_layer * 0.75      # Sub is dominant
    kick[:len(body_layer)] += body_layer * 0.55    # Body adds punch
    kick[:len(click_layer)] += click_layer * 0.20  # Click adds definition
    
    # Final processing
    kick = apply_lowpass(kick, 14000, order=2)  # Remove ultra-highs
    kick = soft_clip(kick)
    kick = normalize(kick, target=0.95)
    
    return kick

# =============================================================================
# GENRE-SPECIFIC GENERATORS (PROFESSIONAL QUALITY)
# =============================================================================

def generate_techno_kick(out_path: Path, bpm: int = 130, variant: int = 1):
    """Professional Techno Kick - Club Standard"""
    beat_dur = 60.0 / bpm
    total_samples = int(SAMPLE_RATE * beat_dur * 16)  # 4 bars
    audio = np.zeros(total_samples, dtype=np.float32)
    
    # Generate kick
    kick = generate_professional_kick(bpm, style="techno")
    
    # Place kicks on grid (4-on-the-floor)
    for i in range(16):
        start = int(i * beat_dur * SAMPLE_RATE)
        if start + len(kick) < len(audio):
            audio[start:start + len(kick)] += kick
    
    # Create sidechain envelope for export
    sidechain_env = np.ones(total_samples, dtype=np.float32)
    duck_len = int(SAMPLE_RATE * 0.15)
    for i in range(16):
        start = int(i * beat_dur * SAMPLE_RATE)
        if start + duck_len < len(sidechain_env):
            duck_curve = np.concatenate([
                np.linspace(1.0, 0.3, duck_len // 2) ** 2,
                np.linspace(0.3, 1.0, duck_len // 2) ** 2
            ])
            sidechain_env[start:start + duck_len] = duck_curve
    
    save_wav(out_path, audio)
    
    # Save sidechain envelope for mixing
    sc_path = out_path.parent / f"{out_path.stem}_sidechain.npy"
    np.save(sc_path, sidechain_env)

def generate_techno_bass(out_path: Path, bpm: int = 128, key_freq: float = 49.0, variant: int = 1):
    """Rolling Techno Bass with Anti-Aliased Synthesis"""
    beat_dur = 60.0 / bpm
    step_dur = beat_dur / 4.0
    total_samples = int(SAMPLE_RATE * beat_dur * 16)
    audio = np.zeros(total_samples, dtype=np.float32)
    
    if variant == 1:
        # Rolling bass pattern
        bass_note = osc.generate_saw(key_freq, step_dur, num_harmonics=40)
        bass_note *= adsr(len(bass_note), 0.005, 0.10, 0.40, 0.08)
        bass_note = apply_resonant_filter(bass_note, 900, resonance=0.3)
        bass_note = soft_clip(bass_note * 1.2)
        
        # Offbeat pattern (not on kick)
        for i in range(64):
            if i % 4 != 0:  # Skip kick positions
                start = int(i * step_dur * SAMPLE_RATE)
                if start + len(bass_note) < len(audio):
                    audio[start:start + len(bass_note)] += bass_note
    else:
        # Acid bassline variant
        for i in range(64):
            pitch = key_freq * (1.0 if i % 8 < 4 else 2.0)
            bass_note = osc.generate_saw(pitch, step_dur, num_harmonics=35)
            
            # Sweeping filter with LFO
            lfo = 500 * np.sin(i * 0.25)
            bass_note = apply_resonant_filter(bass_note, 700 + lfo, resonance=0.75)
            bass_note = apply_overdrive(bass_note, drive=2.0)
            
            env = adsr(len(bass_note), 0.01, 0.08, 0.0, 0.12)
            start = int(i * step_dur * SAMPLE_RATE)
            if start + len(bass_note) < len(audio):
                audio[start:start + len(bass_note)] += bass_note * env * 0.65
    
    # Apply sidechain if available
    sc_path = out_path.parent / "kick_techno_v1_sidechain.npy"
    if sc_path.exists():
        sidechain_env = np.load(sc_path)
        if len(sidechain_env) == len(audio):
            audio = audio * sidechain_env
    
    save_wav(out_path, audio)

def generate_techno_arp(out_path: Path, bpm: int = 138, key_freq: float = 196.0, variant: int = 1):
    """Supersaw Trance Lead - Wide and Lush"""
    beat_dur = 60.0 / bpm
    step_dur = beat_dur / 4.0
    total_samples = int(SAMPLE_RATE * beat_dur * 16)
    audio = np.zeros(total_samples, dtype=np.float32)
    
    # Musical scale (minor pentatonic)
    base_midi = 12 * np.log2(key_freq / 440.0) + 69
    scale = [0, 3, 5, 7, 10, 12, 15]  # Minor pentatonic extended
    
    for i in range(64):
        # Generate note from scale
        scale_idx = i % len(scale)
        octave_shift = 12 if i % 16 > 10 else 0
        midi_note = base_midi + scale[scale_idx] + octave_shift
        freq = 440.0 * (2 ** ((midi_note - 69) / 12))
        
        # Generate supersaw
        note = osc.generate_supersaw(freq, step_dur, num_voices=7, detune=0.09)
        note *= adsr(len(note), 0.01, 0.12, 0.40, 0.30)
        note = apply_lowpass(note, 7000, order=2)
        
        start = int(i * step_dur * SAMPLE_RATE)
        if start + len(note) < len(audio):
            audio[start:start + len(note)] += note * 0.40
    
    # Add effects
    audio = apply_algorithmic_reverb(audio, room_size=0.4, wet=0.25)
    
    # Convert to stereo for ping-pong delay
    audio_stereo = np.stack([audio, audio], axis=1)
    audio_stereo = apply_stereo_delay(audio_stereo, bpm, feedback=0.30, mix=0.25, dotted=True)
    
    # Convert back to mono for consistency
    audio = (audio_stereo[:, 0] + audio_stereo[:, 1]) / 2
    
    save_wav(out_path, audio)

def generate_house_drums(out_path: Path, bpm: int = 124, variant: int = 1):
    """Deep House Drums - Groovy and Warm"""
    beat_dur = 60.0 / bpm
    total_samples = int(SAMPLE_RATE * beat_dur * 16)
    audio = np.zeros(total_samples, dtype=np.float32)
    
    # Kick (warmer than techno)
    kick = generate_professional_kick(bpm, style="house")
    
    # Clap/Snare (shaped noise with tone)
    clap_len = 0.18
    clap_t = np.linspace(0, clap_len, int(SAMPLE_RATE * clap_len), endpoint=False)
    clap = shaped_noise(len(clap_t), level=0.15, lp=6000, hp=300)
    clap += 0.3 * np.sin(2 * np.pi * 200 * clap_t)  # Add tone
    clap *= np.exp(-18 * clap_t)
    clap = apply_overdrive(clap, drive=0.8)
    
    # Hi-hat (crisp but not harsh)
    hat_len = 0.08
    hat_t = np.linspace(0, hat_len, int(SAMPLE_RATE * hat_len), endpoint=False)
    hat = shaped_noise(len(hat_t), level=0.10, lp=9000, hp=4000)
    hat *= np.exp(-22 * hat_t)
    
    # Pattern (4 bars, 16 beats)
    for i in range(16):
        beat_pos = i % 4
        bar = i // 4
        
        # Kick on every beat
        start = int(i * beat_dur * SAMPLE_RATE)
        if start + len(kick) < len(audio):
            audio[start:start + len(kick)] += kick * 0.90
        
        # Clap on 2 and 4
        if beat_pos in [1, 3]:
            if start + len(clap) < len(audio):
                audio[start:start + len(clap)] += clap * 0.60
        
        # Hi-hats (8th notes)
        for h in [0.0, 0.5]:
            hat_start = int((i + h) * beat_dur * SAMPLE_RATE)
            if hat_start + len(hat) < len(audio):
                audio[hat_start:hat_start + len(hat)] += hat * 0.50
    
    save_wav(out_path, audio)

def generate_deep_house_bass(out_path: Path, bpm: int = 124, key_freq: float = 49.0, variant: int = 1):
    """Deep House Bass - Warm and Groovy"""
    beat_dur = 60.0 / bpm
    total_samples = int(SAMPLE_RATE * beat_dur * 16)
    audio = np.zeros(total_samples, dtype=np.float32)
    
    note_dur = beat_dur * 0.5
    
    # Sine wave bass (warm and deep)
    t = np.linspace(0, note_dur, int(SAMPLE_RATE * note_dur), endpoint=False)
    wave = np.sin(2 * np.pi * key_freq * t)
    wave += 0.2 * np.sin(2 * np.pi * key_freq * 2 * t)  # 2nd harmonic
    wave *= adsr(len(wave), 0.01, 0.20, 0.45, 0.15)
    wave = soft_clip(wave * 1.1)
    
    # Groovy pattern
    pattern = [1, 0, 0, 1, 0, 1, 0, 0]
    for i in range(32):
        if pattern[i % 8]:
            start = int(i * note_dur * SAMPLE_RATE)
            if start + len(wave) < len(audio):
                audio[start:start + len(wave)] += wave * 0.80
    
    save_wav(out_path, audio)

def generate_house_chords(out_path: Path, bpm: int = 124, key_freq: float = 261.63):
    """House Piano Stabs"""
    beat_dur = 60.0 / bpm
    total_samples = int(SAMPLE_RATE * beat_dur * 16)
    audio = np.zeros(total_samples, dtype=np.float32)
    
    chord_len = 0.25
    t = np.linspace(0, chord_len, int(SAMPLE_RATE * chord_len), endpoint=False)
    
    # Major 7th chord (1-3-5-7)
    intervals = [1.0, 1.25, 1.5, 1.875]
    stab = np.zeros_like(t)
    
    for ratio in intervals:
        freq = key_freq * ratio
        # Fundamental + harmonics for piano-like sound
        stab += np.sin(2 * np.pi * freq * t)
        stab += 0.5 * np.sin(2 * np.pi * freq * 2 * t)
        stab += 0.25 * np.sin(2 * np.pi * freq * 3 * t)
    
    stab *= np.exp(-8 * t)  # Piano-like decay
    stab = apply_lowpass(stab, 6500, order=2)
    stab = soft_clip(stab * 0.9)
    
    # Stab pattern (offbeat)
    for i in range(16):
        if i % 4 in [1, 3]:
            start = int((i + 0.25) * beat_dur * SAMPLE_RATE)
            if start + len(stab) < len(audio):
                audio[start:start + len(stab)] += stab * 0.45
    
    # Add reverb
    audio = apply_algorithmic_reverb(audio, room_size=0.5, wet=0.30)
    
    save_wav(out_path, audio)

def generate_lofi_drums(out_path: Path, bpm: int = 85, variant: int = 1):
    """Lo-Fi Hip Hop Drums"""
    beat_dur = 60.0 / bpm
    total_samples = int(SAMPLE_RATE * beat_dur * 16)
    audio = np.zeros(total_samples, dtype=np.float32)
    
    # Soft kick
    kick_t = np.linspace(0, 0.35, int(SAMPLE_RATE * 0.35), endpoint=False)
    kick = np.sin(2 * np.pi * 65 * kick_t) * np.exp(-9 * kick_t)
    kick = apply_lowpass(kick, 2000, order=1)  # Very muffled
    
    # Snare
    snare_len = 0.22
    snare = shaped_noise(int(SAMPLE_RATE * snare_len), level=0.12, lp=2500, hp=250)
    snare *= np.exp(-14 * np.linspace(0, snare_len, int(SAMPLE_RATE * snare_len), endpoint=False))
    
    # Pattern
    for i in range(16):
        beat_pos = i % 4
        
        # Kick on 1 and 3.5
        if beat_pos in [0, 2]:
            start = int((i + (0.5 if beat_pos == 2 else 0)) * beat_dur * SAMPLE_RATE)
            if start + len(kick) < len(audio):
                audio[start:start + len(kick)] += kick * 0.95
        
        # Snare on 2 and 4
        if beat_pos in [1, 3]:
            start = int(i * beat_dur * SAMPLE_RATE)
            if start + len(snare) < len(audio):
                audio[start:start + len(snare)] += snare * 0.65
    
    # Add vinyl texture
    vinyl = shaped_noise(total_samples, level=0.008, lp=5000, hp=100)
    audio = audio + vinyl
    
    save_wav(out_path, audio)

def generate_lofi_keys(out_path: Path, bpm: int = 85, key_freq: float = 261.63, variant: int = 1):
    """Lo-Fi Keys with Warmth and Imperfection"""
    duration = (60.0 / bpm) * 16
    t = np.linspace(0, duration, int(SAMPLE_RATE * duration), endpoint=False)
    
    # Vibrato (pitch wobble)
    vibrato = 1.0 + 0.004 * np.sin(2 * np.pi * 4.5 * t)
    
    audio = np.zeros_like(t)
    bar_len = int(SAMPLE_RATE * (60.0 / bpm) * 4)
    
    # Chord notes
    chord = [key_freq, key_freq * 1.2, key_freq * 1.5]
    
    for bar in range(4):
        start_sample = bar * bar_len
        for i, freq in enumerate(chord):
            note_start = start_sample + i * 3000
            if note_start < len(audio):
                remaining = len(audio) - note_start
                note_t = t[:remaining]
                # Sine + harmonics for warmth
                note = np.sin(2 * np.pi * (freq * vibrato[:remaining]) * note_t)
                note += 0.3 * np.sin(2 * np.pi * (freq * 2 * vibrato[:remaining]) * note_t)
                note *= np.exp(-1.2 * note_t)
                audio[note_start:] += note * 0.30
    
    # Warm filtering
    audio = apply_lowpass(audio, 2200, order=2)
    audio = apply_highpass(audio, 100, order=1)
    
    save_wav(out_path, audio)

def generate_wobble_bass(out_path: Path, bpm: int = 140, key_freq: float = 55.0):
    """Dubstep Wobble Bass"""
    beat_dur = 60.0 / bpm
    total_samples = int(SAMPLE_RATE * beat_dur * 16)
    
    t = np.linspace(0, beat_dur * 16, total_samples, endpoint=False)
    
    # Base sawtooth
    bass = osc.generate_saw(key_freq, beat_dur * 16, num_harmonics=60)
    
    # Wobble with changing speeds
    audio = np.zeros(total_samples, dtype=np.float32)
    bar_samples = int(SAMPLE_RATE * beat_dur * 4)
    
    wobble_speeds = [3, 6, 12, 4]
    
    for bar_idx, speed in enumerate(wobble_speeds):
        start = bar_idx * bar_samples
        end = min(start + bar_samples, total_samples)
        chunk_t = t[start:end]
        chunk = bass[start:end]
        
        # LFO for filter cutoff modulation
        lfo = 0.5 * (1 + np.sin(2 * np.pi * speed * chunk_t))
        
        # Modulate chunk with LFO (simplified)
        audio[start:end] = chunk * lfo
    
    # Heavy processing
    audio = apply_bitcrush(audio, bits=7)
    audio = apply_overdrive(audio, drive=2.5)
    audio = apply_lowpass(audio, 9000, order=2)
    
    save_wav(out_path, audio)

def generate_hard_kick(out_path: Path, bpm: int = 150):
    """Hardstyle Reverse Bass Kick"""
    beat_dur = 60.0 / bpm
    total_samples = int(SAMPLE_RATE * beat_dur * 16)
    audio = np.zeros(total_samples, dtype=np.float32)
    
    kick_len = 0.45
    t = np.linspace(0, kick_len, int(SAMPLE_RATE * kick_len), endpoint=False)
    
    # Pitch envelope with reverse section
    freq_env = 220 * np.exp(-12 * t) + 45
    phase = np.cumsum(freq_env) * (2 * np.pi / SAMPLE_RATE)
    kick = np.sin(phase)
    
    # Heavy distortion
    kick = np.clip(kick * 6.0, -0.85, 0.85)
    kick = apply_overdrive(kick, drive=2.2)
    kick *= np.exp(-5 * t)
    
    # Place kicks
    for i in range(16):
        start = int(i * beat_dur * SAMPLE_RATE)
        if start + len(kick) < len(audio):
            audio[start:start + len(kick)] += kick * 0.90
    
    save_wav(out_path, audio)

def generate_synth_bass(out_path: Path, bpm: int = 105, key_freq: float = 65.4):
    """Synthwave Bass"""
    beat_dur = 60.0 / bpm
    total_samples = int(SAMPLE_RATE * beat_dur * 16)
    audio = np.zeros(total_samples, dtype=np.float32)
    
    note_dur = beat_dur * 0.5
    
    # Sawtooth bass with envelope
    bass_note = osc.generate_saw(key_freq, note_dur, num_harmonics=45)
    bass_note *= adsr(len(bass_note), 0.01, 0.12, 0.50, 0.12)
    bass_note = apply_lowpass(bass_note, 3000, order=2)
    bass_note = soft_clip(bass_note * 1.1)
    
    for i in range(32):
        start = int(i * note_dur * SAMPLE_RATE)
        if start + len(bass_note) < len(audio):
            audio[start:start + len(bass_note)] += bass_note * 0.70
    
    save_wav(out_path, audio)

def generate_gated_snare(out_path: Path, bpm: int = 105):
    """80s Gated Reverb Snare"""
    beat_dur = 60.0 / bpm
    total_samples = int(SAMPLE_RATE * beat_dur * 16)
    audio = np.zeros(total_samples, dtype=np.float32)
    
    snare_len = 0.35
    t = np.linspace(0, snare_len, int(SAMPLE_RATE * snare_len), endpoint=False)
    
    # Noise + tone
    snare = shaped_noise(len(t), level=0.15, lp=5500, hp=280)
    snare += 0.3 * np.sin(2 * np.pi * 185 * t)
    
    # Gate (cut tail suddenly)
    gate = np.ones_like(t)
    gate[-int(len(t) * 0.25):] = 0.0
    snare = snare * gate
    
    for i in range(16):
        if i % 4 in [1, 3]:
            start = int(i * beat_dur * SAMPLE_RATE)
            if start + len(snare) < len(audio):
                audio[start:start + len(snare)] += snare * 0.75
    
    save_wav(out_path, audio)

def generate_rave_piano(out_path: Path, bpm: int = 140, key_freq: float = 440.0):
    """90s Rave Piano Stabs"""
    beat_dur = 60.0 / bpm
    total_samples = int(SAMPLE_RATE * beat_dur * 16)
    audio = np.zeros(total_samples, dtype=np.float32)
    
    chord_len = 0.22
    t = np.linspace(0, chord_len, int(SAMPLE_RATE * chord_len), endpoint=False)
    
    # Major triad
    stab = np.zeros_like(t)
    for freq in [key_freq, key_freq * 1.25, key_freq * 1.5]:
        stab += np.sin(2 * np.pi * freq * t)
        stab += 0.4 * np.sin(2 * np.pi * freq * 2 * t)
    
    stab *= np.exp(-9 * t)
    stab = apply_lowpass(stab, 6500, order=2)
    
    for i in range(16):
        if i % 2 == 1:
            start = int(i * beat_dur * SAMPLE_RATE)
            if start + len(stab) < len(audio):
                audio[start:start + len(stab)] += stab * 0.60
    
    save_wav(out_path, audio)

def generate_texture(out_path: Path, type: str = "vinyl"):
    """Ambient Textures"""
    sec = 16
    samples = int(SAMPLE_RATE * sec)
    
    if type == "vinyl":
        # Vinyl crackle
        audio = shaped_noise(samples, level=0.008, lp=5500, hp=90)
        # Add occasional clicks
        for _ in range(12):
            pos = random.randint(0, samples - 100)
            audio[pos:pos + 100] += click_transient(100, level=0.10)
    else:
        # Rain ambience
        audio = shaped_noise(samples, level=0.015, lp=4800, hp=180)
        # Slow modulation
        t = np.linspace(0, 1, samples, endpoint=False)
        drift = 0.7 + 0.3 * np.sin(2 * np.pi * 0.15 * t)
        audio = audio * drift
    
    save_wav(out_path, audio)

# =============================================================================
# I/O UTILITIES
# =============================================================================

def save_wav(path: Path, data: np.ndarray):
    """Save audio as 16-bit WAV file with proper normalization"""
    path.parent.mkdir(parents=True, exist_ok=True)
    
    data = np.asarray(data, dtype=np.float32)
    
    # Safety clip
    data = np.clip(data, -1.0, 1.0)
    
    # Normalize to -0.5dB peak
    max_val = np.max(np.abs(data))
    if max_val > 1e-6:
        data = data * (0.95 / max_val)
    
    # Convert to 16-bit
    data_int = (data * 32767.0).astype(np.int16)
    
    wavfile.write(str(path), SAMPLE_RATE, data_int)
    print(f"🎹 Generated: {path.name}")