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

def normalize(audio):
    max_val = np.max(np.abs(audio))
    return audio / max_val if max_val > 0 else audio

def apply_overdrive(audio, drive=0.5):
    """Tube distortion simulation."""
    return np.tanh(audio * (1 + drive))

def apply_lowpass(audio, cutoff=3000):
    nyquist = 0.5 * SAMPLE_RATE
    norm_cutoff = cutoff / nyquist
    b, a = signal.butter(1, norm_cutoff, btype='low', analog=False)
    return signal.lfilter(b, a, audio)

def apply_delay(audio, bpm, feedback=0.4, mix=0.3):
    delay_samples = int((60 / bpm) * 0.75 * SAMPLE_RATE) 
    if delay_samples >= len(audio): return audio
    out = np.copy(audio)
    delayed_signal = np.zeros_like(audio)
    delayed_signal[delay_samples:] = audio[:-delay_samples]
    return out * (1 - mix) + delayed_signal * mix

def adsr(length, attack=0.01, decay=0.1, sustain=0.7, release=0.2):
    total = length
    a, d, r = int(attack*SAMPLE_RATE), int(decay*SAMPLE_RATE), int(release*SAMPLE_RATE)
    if a+d+r > total:
        factor = total / (a+d+r)
        a, d, r = int(a*factor), int(d*factor), int(r*factor)
    s_len = total - a - d - r
    if s_len < 0: s_len = 0
    env = np.concatenate([np.linspace(0,1,a), np.linspace(1,sustain,d), np.full(s_len,sustain), np.linspace(sustain,0,r)])
    if len(env) < total: env = np.pad(env, (0, total - len(env)))
    return env[:total]

# -----------------------------------------------------------------------------
# TECHNO GENERATORS (High Energy)
# -----------------------------------------------------------------------------

def generate_techno_kick(out_path: Path, bpm=128, variant=1):
    """Generates varied Techno kicks (Hard, Deep, or Rumble)."""
    beat_dur = 60 / bpm
    total_samples = int(SAMPLE_RATE * beat_dur * 4 * 4) # 4 bars
    audio = np.zeros(total_samples)
    
    # Variant parameters
    start_freq = 150 if variant == 1 else (180 if variant == 2 else 120)
    decay_speed = 30 if variant == 1 else (40 if variant == 2 else 20)
    distortion = 3.0 if variant == 2 else 1.0
    
    kick_len = 0.35
    t = np.linspace(0, kick_len, int(SAMPLE_RATE * kick_len), False)
    
    # Sweep
    freq = start_freq * np.exp(-decay_speed * t) + 40
    wave = np.sin(2 * np.pi * freq * t)
    
    # Click/Transients
    click = np.random.uniform(-1, 1, len(t)) * np.exp(-80 * t)
    
    kick = wave * 0.9 + click * 0.2
    kick = apply_overdrive(kick, drive=distortion)
    kick *= np.exp(-5 * t)

    # 4/4 Pattern
    for i in range(16): 
        start = int(i * beat_dur * SAMPLE_RATE)
        if start + len(kick) < len(audio): audio[start:start+len(kick)] += kick * 0.9

    save_wav(out_path, audio)

def generate_techno_bass(out_path: Path, bpm=128, key_freq=49.0, variant=1):
    """Generates FM Bass (Growl) or Offbeat Rumble."""
    beat_dur = 60 / bpm
    total_samples = int(SAMPLE_RATE * beat_dur * 4 * 4)
    audio = np.zeros(total_samples)
    
    note_dur = beat_dur * 0.5 # 8th notes
    
    for i in range(32): # 8 notes * 4 bars
        # Offbeat pattern (1 & 2 & 3 & 4 &)
        if i % 2 != 0: 
            t = np.linspace(0, note_dur, int(SAMPLE_RATE*note_dur), False)
            
            if variant == 1: # FM Growl
                mod = 2.0 * np.sin(2*np.pi*key_freq*2*t) * np.exp(-5*t)
                wave = np.sin(2*np.pi*(key_freq*t + mod))
                wave = apply_overdrive(wave, 1.5)
            else: # Rolling Sub
                wave = np.sin(2*np.pi*key_freq*t)
                wave = apply_lowpass(wave, 400)

            env = adsr(len(wave), 0.01, 0.1, 0.6, 0.1)
            
            start = int(i * note_dur * SAMPLE_RATE)
            if start + len(wave) < len(audio): audio[start:start+len(wave)] += wave * env * 0.8

    save_wav(out_path, audio)

def generate_techno_arp(out_path: Path, bpm=128, key_freq=196.0, variant=1):
    """Generates Super-Saw or Pluck arps."""
    beat_dur = 60 / bpm
    step_dur = beat_dur / 4
    total_steps = 16 * 4
    audio = np.zeros(int(SAMPLE_RATE * step_dur * total_steps))
    
    base = 12 * np.log2(key_freq / 440.0) + 69
    scale = [0, 2, 3, 5, 7, 8, 10] if variant == 1 else [0, 3, 5, 7, 10]
    
    curr = 0
    for i in range(total_steps):
        # Rhythm variation
        play = (i % 2 == 0) if variant == 1 else (i % 4 != 0)
        
        if play:
            idx = scale[curr % len(scale)]
            freq = 440.0 * 2**((base + idx - 69) / 12)
            
            t = np.linspace(0, step_dur, int(SAMPLE_RATE*step_dur), False)
            
            # Super Saw logic
            saw = (signal.sawtooth(2*np.pi*freq*t) + 
                   signal.sawtooth(2*np.pi*freq*1.01*t)) * 0.5
            
            # Filter envelope
            cutoff_env = np.linspace(1.0, 0.0, len(t))
            saw *= cutoff_env
            
            start = int(i * step_dur * SAMPLE_RATE)
            if start + len(saw) < len(audio): audio[start:start+len(saw)] += saw * 0.4
            
            curr += random.choice([1, 2, -1])

    # Add delay for texture
    audio = apply_delay(audio, bpm)
    save_wav(out_path, audio)

# -----------------------------------------------------------------------------
# LO-FI GENERATORS (Chill / Study)
# -----------------------------------------------------------------------------

def generate_lofi_drums(out_path: Path, bpm=85, variant=1):
    """Soft, muffled beats with swing."""
    beat_dur = 60 / bpm
    total_samples = int(SAMPLE_RATE * beat_dur * 4 * 4)
    audio = np.zeros(total_samples)
    
    # Soft Kick
    k_len = 0.4
    kt = np.linspace(0, k_len, int(SAMPLE_RATE*k_len))
    kick = np.sin(2*np.pi*60*kt) * np.exp(-10*kt)
    
    # Snare (Noise)
    s_len = 0.2
    snare = np.random.uniform(-1, 1, int(SAMPLE_RATE*s_len)) * np.exp(-15*np.linspace(0, s_len, int(SAMPLE_RATE*s_len)))
    snare = apply_lowpass(snare, 1200)

    # HiHat (Tick)
    h_len = 0.05
    hat = np.random.uniform(-0.5, 0.5, int(SAMPLE_RATE*h_len)) * np.exp(-50*np.linspace(0, h_len, int(SAMPLE_RATE*h_len)))

    for i in range(4): # 4 bars
        offset = i * 4 * beat_dur
        
        # Kick pattern variations
        k_times = [0, 2.5] if variant == 1 else [0, 1.5, 3.5]
        for t in k_times:
            pos = int((offset + t * beat_dur) * SAMPLE_RATE)
            if pos+len(kick) < len(audio): audio[pos:pos+len(kick)] += kick
            
        # Snare always on 2 and 4
        for t in [1, 3]:
            pos = int((offset + t * beat_dur) * SAMPLE_RATE)
            if pos+len(snare) < len(audio): audio[pos:pos+len(snare)] += snare * 0.7

        # Hats (every 8th note with swing)
        for j in range(8):
            swing = 0.05 if j % 2 != 0 else 0
            t = j * 0.5 + swing
            pos = int((offset + t * beat_dur) * SAMPLE_RATE)
            if pos+len(hat) < len(audio): audio[pos:pos+len(hat)] += hat * 0.3

    save_wav(out_path, audio)

def generate_lofi_keys(out_path: Path, bpm=85, key_freq=261.63, variant=1):
    """Wobbly electric piano."""
    duration = (60 / bpm) * 4 * 4
    t = np.linspace(0, duration, int(SAMPLE_RATE*duration), False)
    
    # Vibrato
    vibrato = 1.0 + 0.003 * np.sin(2 * np.pi * 4.0 * t)
    
    # Progression
    # Variant 1: I - ii - V
    # Variant 2: I - vi - IV
    freqs_1 = [[key_freq*x for x in [1, 1.2, 1.5]], [key_freq*x for x in [1.125, 1.33, 1.66]], [key_freq*x for x in [1.5, 1.875, 2.25]]]
    
    audio = np.zeros_like(t)
    
    bar_len = int(SAMPLE_RATE * (60/bpm) * 4)
    
    for b in range(4):
        chord = freqs_1[b % len(freqs_1)]
        start = b * bar_len
        
        # Strumming effect
        for i, f in enumerate(chord):
            # Electric Piano Synthesis
            strum_delay = i * 2000 # samples
            actual_start = start + strum_delay
            
            if actual_start < len(audio):
                rem_len = len(audio) - actual_start
                # Synthesis
                lt = t[:rem_len]
                # Modulator
                mod = np.sin(2*np.pi*f*2*lt) * np.exp(-2*lt)
                # Carrier with Vibrato
                car = np.sin(2*np.pi*(f*vibrato[:rem_len])*lt + 1.5*mod)
                
                # Envelope
                env = np.exp(-1.5 * lt)
                
                audio[actual_start:] += car * env * 0.3

    audio = apply_lowpass(audio, 1800)
    save_wav(out_path, audio)

def generate_texture(out_path: Path, type='vinyl'):
    """Generates Vinyl Crackle or Rain."""
    sec = 16
    samples = int(SAMPLE_RATE * sec)
    
    if type == 'vinyl':
        audio = np.random.uniform(-0.02, 0.02, samples)
        # Pops
        for _ in range(30):
            idx = random.randint(0, samples-1)
            audio[idx] += 0.4
    else: # Rain
        audio = np.random.uniform(-0.05, 0.05, samples)
        audio = apply_lowpass(audio, 800)
        
    save_wav(out_path, audio)

# -----------------------------------------------------------------------------
# IO
# -----------------------------------------------------------------------------
def save_wav(path: Path, data: np.ndarray):
    path.parent.mkdir(parents=True, exist_ok=True)
    max_val = np.max(np.abs(data))
    if max_val > 0: data = data / max_val
    data_int = (data * 32767).astype(np.int16)
    wavfile.write(str(path), SAMPLE_RATE, data_int)
    print(f"🎹 Generated stem variant: {path.name}")