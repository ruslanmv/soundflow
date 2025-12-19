# generator/free/remix_daily.py
"""
SoundFlow Music Generation Engine - Remix Daily
Version 2.1.0

Supports:
- Electronic: Techno, Trance, House, Deep House, Bass, Hardstyle
- Chill: Lo-Fi, Ambient, Chillout, Downtempo
- Jazz: Jazz, Neo-Soul, Smooth Jazz
- Classical: Piano, Ambient Piano
- Focus: Deep Work, Coding, Study, Meditation
"""

from __future__ import annotations

import argparse
import os
import random
import shutil
import yaml
import json
import numpy as np
from pathlib import Path
from scipy.io import wavfile

# Import the specific generators and DSP effects
from free.music_engine import (
    # Electronic
    generate_techno_kick,
    generate_techno_bass,
    generate_techno_arp,
    generate_house_drums,
    generate_deep_house_bass,
    generate_house_chords,
    generate_wobble_bass,
    generate_hard_kick,
    generate_synth_bass,
    generate_gated_snare,
    generate_rave_piano,
    
    # Chill & Ambient
    generate_lofi_drums,
    generate_lofi_keys,
    generate_texture,
    
    # Focus Engine
    generate_focus_session,
    
    # DSP Effects (Smart Mixer)
    apply_overdrive,
    apply_algorithmic_reverb,
    apply_lowpass,
    apply_highpass,
    apply_resonant_filter,
    apply_parametric_eq,
    
    # Additional Effects
    multiband_process,
    soft_clip,
    normalize,
    
    SAMPLE_RATE,
)

from common.audio_utils import (
    require_ffmpeg,
    ffmpeg_loop_to_duration,
    ffmpeg_mix,
    ffmpeg_fade,
    ffmpeg_loudnorm,
    ffmpeg_encode_mp3,
    _run,
)
from common.r2_upload import upload_file

ROOT = Path(__file__).resolve().parent
ASSETS = ROOT / "assets"
STEMS_DIR = ASSETS / "stems"
TMP = Path(".soundflow_tmp/free")
OUT = Path(".soundflow_out/free")

# =============================================================================
# GENRE DEFINITIONS
# =============================================================================

GENRE_MAPPINGS = {
    # Electronic Dance
    "Techno": {"bpm": 130, "stems": ["kick_techno", "bass_techno", "arp_techno"]},
    "Trance": {"bpm": 138, "stems": ["kick_techno", "bass_techno", "arp_trance", "pad_trance"]},
    "House": {"bpm": 124, "stems": ["drums_house", "bass_deep", "chords_house"]},
    "Deep": {"bpm": 122, "stems": ["drums_deep", "bass_deep", "pad_deep", "texture_vinyl"]},
    "Bass": {"bpm": 140, "stems": ["kick_bass", "bass_wobble", "synth_bass"]},
    "Hard": {"bpm": 150, "stems": ["kick_hard", "bass_hard", "synth_hard", "snare_gated"]},
    
    # Chill & Ambient
    "Chillout": {"bpm": 90, "stems": ["drums_lofi", "keys_lofi", "pad_ambient", "texture_vinyl"]},
    "Ambient": {"bpm": 60, "stems": ["pad_ambient", "drone_deep", "texture_rain"]},
    "Lounge": {"bpm": 92, "stems": ["drums_lofi", "rhodes_jazz", "bass_upright", "texture_vinyl"]},
    
    # Jazz & Soul
    "Jazz": {"bpm": 120, "stems": ["drums_jazz", "bass_upright", "piano_jazz", "rhodes_jazz"]},
    "NeoSoul": {"bpm": 85, "stems": ["drums_soul", "bass_upright", "rhodes_neo", "pad_warm"]},
    
    # Piano & Classical
    "Piano": {"bpm": 72, "stems": ["piano_classical", "pad_strings", "texture_room"]},
    "Ambient Piano": {"bpm": 60, "stems": ["piano_ambient", "pad_ambient", "texture_rain"]},
    
    # Focus & Productivity
    "Deep Work": {"bpm": 100, "stems": ["drums_minimal", "bass_deep", "pad_ambient"]},
    "Coding": {"bpm": 128, "stems": ["kick_minimal", "bass_techno", "arp_minimal"]},
    "Study": {"bpm": 85, "stems": ["drums_lofi", "keys_lofi", "pad_warm", "texture_vinyl"]},
}

# =============================================================================
# PROFESSIONAL MIXING UTILITIES
# =============================================================================

def apply_stem_eq(audio: np.ndarray, stem_type: str) -> np.ndarray:
    """
    Apply EQ carving based on stem type.
    This prevents frequency masking and creates professional separation.
    """
    
    if stem_type == "kick":
        # Kick: Boost sub, cut mids
        audio = apply_highpass(audio, 25)
        audio = apply_parametric_eq(audio, 60, gain_db=2, q=1.2)
        audio = apply_parametric_eq(audio, 300, gain_db=-3, q=2.0)
    
    elif stem_type == "bass" or stem_type == "bass_upright":
        # Bass: Duck for kick, remove lows
        audio = apply_highpass(audio, 35)
        audio = apply_parametric_eq(audio, 55, gain_db=-5, q=1.5)
        audio = apply_lowpass(audio, 4500)
    
    elif stem_type in ["synth", "arp", "arp_trance", "arp_minimal"]:
        # Synths: Remove lows, boost presence
        audio = apply_highpass(audio, 180)
        audio = apply_parametric_eq(audio, 2800, gain_db=1.5, q=0.8)
    
    elif stem_type in ["pad", "pad_trance", "pad_ambient", "pad_warm", "pad_strings", "pad_deep"]:
        # Pads: Remove lows, make spacious
        audio = apply_highpass(audio, 220)
        audio = apply_lowpass(audio, 8000)
    
    elif stem_type in ["chords", "chords_house", "stab"]:
        # Chords: Clean mids
        audio = apply_highpass(audio, 150)
        audio = apply_parametric_eq(audio, 1200, gain_db=1.0, q=0.6)
    
    elif stem_type in ["piano", "piano_jazz", "piano_classical", "piano_ambient", "rhodes", "rhodes_jazz", "rhodes_neo"]:
        # Piano/Rhodes: Natural warmth
        audio = apply_highpass(audio, 80)
        audio = apply_parametric_eq(audio, 250, gain_db=1.2, q=0.8)
        audio = apply_parametric_eq(audio, 3500, gain_db=0.8, q=0.5)
    
    elif stem_type in ["drums", "drums_jazz", "drums_soul", "drums_minimal"]:
        # Full drum kit: Preserve transients
        audio = apply_highpass(audio, 40)
        audio = apply_parametric_eq(audio, 80, gain_db=1.5, q=1.0)
        audio = apply_parametric_eq(audio, 8000, gain_db=1.0, q=0.4)
    
    elif stem_type in ["texture", "texture_vinyl", "texture_rain", "texture_room"]:
        # Textures: High-pass aggressively
        audio = apply_highpass(audio, 450)
        audio = apply_lowpass(audio, 7500)
    
    elif stem_type in ["drone", "drone_deep"]:
        # Drones: Sub focus
        audio = apply_highpass(audio, 30)
        audio = apply_lowpass(audio, 200)
    
    return audio

def professional_mix(
    stem_paths: list[Path], 
    output: Path, 
    genre: str = "techno", 
    synth_params: dict = None
) -> None:
    """
    Professional mixing with proper gain staging, frequency carving,
    and Smart Mixer effects application.
    
    Args:
        stem_paths: List of WAV file paths to mix
        output: Output WAV path
        genre: Genre for genre-specific processing
        synth_params: Smart Mixer parameters (cutoff, resonance, drive, space)
    """
    
    # 1. Load all stems
    stems = []
    max_len = 0
    
    for stem_path in stem_paths:
        rate, data = wavfile.read(str(stem_path))
        if data.dtype == np.int16:
            data = data.astype(np.float32) / 32768.0
        if len(data.shape) > 1:
            data = data[:, 0]  # Convert to mono
        
        stems.append({
            'audio': data,
            'path': stem_path,
            'type': detect_stem_type(stem_path.name)
        })
        max_len = max(max_len, len(data))
    
    # 2. Gain staging and EQ per stem
    processed_stems = []
    
    for stem in stems:
        audio = stem['audio']
        stem_type = stem['type']
        
        # Pad to same length
        if len(audio) < max_len:
            audio = np.pad(audio, (0, max_len - len(audio)))
        
        # Apply EQ carving
        audio = apply_stem_eq(audio, stem_type)
        
        # Genre-specific gain staging
        if "jazz" in genre.lower() or "soul" in genre.lower():
            gain_map = {
                'drums': 0.65, 'drums_jazz': 0.65, 'drums_soul': 0.62,
                'bass': 0.68, 'bass_upright': 0.70,
                'piano': 0.72, 'piano_jazz': 0.75,
                'rhodes': 0.70, 'rhodes_jazz': 0.72, 'rhodes_neo': 0.70,
                'pad': 0.38, 'pad_warm': 0.40,
                'texture': 0.15,
            }
        elif "ambient" in genre.lower() or "piano" in genre.lower():
            gain_map = {
                'piano': 0.80, 'piano_classical': 0.82, 'piano_ambient': 0.78,
                'pad': 0.48, 'pad_ambient': 0.50, 'pad_strings': 0.45,
                'drone': 0.35, 'drone_deep': 0.32,
                'texture': 0.18,
            }
        elif "chill" in genre.lower() or "lofi" in genre.lower() or "lounge" in genre.lower():
            gain_map = {
                'drums': 0.58, 'drums_lofi': 0.60,
                'bass': 0.65, 'bass_upright': 0.68,
                'keys': 0.62, 'keys_lofi': 0.65,
                'rhodes': 0.68,
                'pad': 0.40, 'pad_warm': 0.42,
                'texture': 0.22,
            }
        else:
            # Electronic / Dance
            gain_map = {
                'kick': 0.88, 'kick_techno': 0.88, 'kick_hard': 0.90, 'kick_bass': 0.86, 'kick_minimal': 0.75,
                'bass': 0.72, 'bass_techno': 0.72, 'bass_deep': 0.68, 'bass_wobble': 0.75, 'bass_hard': 0.70,
                'drums': 0.68, 'drums_house': 0.70, 'drums_deep': 0.65, 'drums_minimal': 0.60,
                'synth': 0.52, 'synth_bass': 0.55, 'synth_hard': 0.58,
                'arp': 0.48, 'arp_techno': 0.50, 'arp_trance': 0.52, 'arp_minimal': 0.45,
                'pad': 0.42, 'pad_trance': 0.45, 'pad_deep': 0.48,
                'chords': 0.45, 'chords_house': 0.48,
                'piano': 0.50, 'rave_piano': 0.55,
                'snare': 0.60, 'snare_gated': 0.62,
                'texture': 0.18,
            }
        
        gain = gain_map.get(stem_type, 0.50)
        audio = audio * gain
        
        # Apply sidechain to bass/pads (if kick exists) - only for electronic genres
        if "jazz" not in genre.lower() and "piano" not in genre.lower() and "ambient" not in genre.lower():
            if stem_type in ['bass', 'bass_techno', 'bass_deep', 'bass_wobble', 'pad', 'pad_trance', 'pad_deep', 'synth', 'arp', 'chords']:
                sidechain_file = STEMS_DIR / "kick_techno_v1_sidechain.npy"
                if sidechain_file.exists():
                    sidechain_env = np.load(sidechain_file)
                    if len(sidechain_env) == len(audio):
                        audio = audio * sidechain_env
        
        processed_stems.append(audio)
    
    # 3. Sum all stems
    mixed = np.sum(processed_stems, axis=0).astype(np.float32)

    # =========================================================================
    # SMART MIXER EFFECTS (PRO PANEL)
    # =========================================================================
    if synth_params:
        # A. Drive / Grit (Saturation)
        drive_amt = synth_params.get("drive", 0.0)
        if drive_amt > 0.05:
            # Map 0.0-1.0 to meaningful drive values (0-5.0)
            drive_value = drive_amt * 4.0
            mixed = apply_overdrive(mixed, drive=drive_value)

        # B. Filter Cutoff / Resonance
        cutoff_amt = synth_params.get("cutoff", 1.0)
        resonance_amt = synth_params.get("resonance", 0.0)
        
        if cutoff_amt < 0.98:
            # Map 0.0-1.0 to Frequency (100Hz - 20kHz)
            # Logarithmic mapping for natural feel
            cutoff_freq = 100 + (20000 * (cutoff_amt ** 2))
            
            if resonance_amt > 0.1:
                # Apply resonant filter for "acid" sound
                mixed = apply_resonant_filter(
                    mixed, 
                    cutoff=cutoff_freq, 
                    resonance=resonance_amt * 0.9
                )
            else:
                # Simple lowpass
                mixed = apply_lowpass(mixed, cutoff=cutoff_freq)

        # C. Space / Reverb
        space_amt = synth_params.get("space", 0.0)
        if space_amt > 0.05:
            # Map 0.0-1.0 to Reverb size/wetness
            room_size = 0.3 + (space_amt * 0.6)
            wet_amount = space_amt * 0.5
            mixed = apply_algorithmic_reverb(
                mixed, 
                room_size=room_size, 
                wet=wet_amount
            )

    # 4. Master bus processing
    mixed = master_chain(mixed, genre=genre)
    
    # 5. Save
    output.parent.mkdir(parents=True, exist_ok=True)
    data_int = (mixed * 32767.0).astype(np.int16)
    wavfile.write(str(output), SAMPLE_RATE, data_int)
    print(f"✅ Mixed: {output.name}")

def master_chain(audio: np.ndarray, genre: str) -> np.ndarray:
    """
    Professional mastering chain for broadcast-ready output.
    Genre-specific processing for optimal results.
    """
    
    # 1. Subsonic filter
    audio = apply_highpass(audio, 20, order=4)
    
    # 2. Genre-specific EQ
    if "jazz" in genre.lower() or "soul" in genre.lower():
        # Jazz: Natural warmth, preserve dynamics
        audio = apply_parametric_eq(audio, 100, gain_db=0.5, q=0.8)   # Warmth
        audio = apply_parametric_eq(audio, 2500, gain_db=0.3, q=0.6)  # Clarity
        audio = apply_parametric_eq(audio, 8000, gain_db=0.2, q=0.5)  # Air
    
    elif "piano" in genre.lower() or "classical" in genre.lower():
        # Piano: Natural resonance
        audio = apply_parametric_eq(audio, 250, gain_db=0.6, q=0.7)
        audio = apply_parametric_eq(audio, 3000, gain_db=0.4, q=0.5)
        audio = apply_parametric_eq(audio, 10000, gain_db=0.3, q=0.4)
    
    elif "ambient" in genre.lower() or "chill" in genre.lower():
        # Ambient: Smooth, wide
        audio = apply_parametric_eq(audio, 60, gain_db=0.4, q=0.6)
        audio = apply_parametric_eq(audio, 5000, gain_db=0.2, q=0.4)
        audio = apply_parametric_eq(audio, 12000, gain_db=0.4, q=0.3)
    
    else:
        # Electronic: Punchy, bright
        audio = apply_parametric_eq(audio, 80, gain_db=0.8, q=0.7)    # Sub warmth
        audio = apply_parametric_eq(audio, 3000, gain_db=0.5, q=0.5)  # Presence
        audio = apply_parametric_eq(audio, 10000, gain_db=0.3, q=0.4) # Air
    
    # 3. Multiband gentle compression (simulated)
    if "ambient" in genre.lower() or "piano" in genre.lower():
        # Gentler compression for acoustic genres
        audio = multiband_process(audio, low_gain=1.02, mid_gain=0.99, high_gain=1.01)
    else:
        audio = multiband_process(audio, low_gain=1.05, mid_gain=0.98, high_gain=1.02)
    
    # 4. Soft clipping (analog-style limiting)
    if "jazz" in genre.lower() or "classical" in genre.lower():
        # Less aggressive clipping for acoustic
        audio = soft_clip(audio * 1.05, threshold=0.90)
    else:
        audio = soft_clip(audio * 1.1, threshold=0.85)
    
    # 5. Final normalization to target LUFS
    if "techno" in genre.lower() or "hard" in genre.lower() or "bass" in genre.lower():
        target_rms = 0.15  # Loud club sound
    elif "jazz" in genre.lower() or "classical" in genre.lower():
        target_rms = 0.10  # Preserve dynamics
    else:
        target_rms = 0.12  # Balanced
    
    rms = np.sqrt(np.mean(audio ** 2))
    if rms > 1e-6:
        audio = audio * (target_rms / rms)
    
    # 6. True peak limiting
    audio = np.clip(audio, -0.95, 0.95)
    
    # 7. Final normalize to -0.3dB
    audio = normalize(audio, target=0.97)
    
    return audio

def detect_stem_type(filename: str) -> str:
    """Detect stem type from filename for proper processing"""
    name = filename.lower()
    
    # Kicks
    if 'kick_techno' in name: return 'kick_techno'
    if 'kick_hard' in name: return 'kick_hard'
    if 'kick_bass' in name: return 'kick_bass'
    if 'kick_minimal' in name: return 'kick_minimal'
    if 'kick' in name: return 'kick'
    
    # Bass
    if 'bass_techno' in name: return 'bass_techno'
    if 'bass_deep' in name or 'bass_house' in name: return 'bass_deep'
    if 'bass_wobble' in name: return 'bass_wobble'
    if 'bass_upright' in name: return 'bass_upright'
    if 'bass_synth' in name: return 'bass_synth'
    if 'bass_hard' in name: return 'bass_hard'
    if 'bass' in name: return 'bass'
    
    # Drums
    if 'drums_house' in name: return 'drums_house'
    if 'drums_deep' in name: return 'drums_deep'
    if 'drums_lofi' in name: return 'drums_lofi'
    if 'drums_jazz' in name: return 'drums_jazz'
    if 'drums_soul' in name: return 'drums_soul'
    if 'drums_minimal' in name: return 'drums_minimal'
    if 'drums' in name or 'drum' in name: return 'drums'
    
    # Melodic
    if 'arp_trance' in name: return 'arp_trance'
    if 'arp_minimal' in name: return 'arp_minimal'
    if 'arp_techno' in name or 'arp' in name: return 'arp'
    
    if 'chords_house' in name or 'chord_house' in name: return 'chords_house'
    if 'chords' in name or 'chord' in name or 'stab' in name: return 'chords'
    
    if 'synth_bass' in name: return 'synth_bass'
    if 'synth_hard' in name: return 'synth_hard'
    if 'synth' in name or 'lead' in name: return 'synth'
    
    # Pads
    if 'pad_trance' in name: return 'pad_trance'
    if 'pad_deep' in name: return 'pad_deep'
    if 'pad_ambient' in name: return 'pad_ambient'
    if 'pad_warm' in name: return 'pad_warm'
    if 'pad_strings' in name: return 'pad_strings'
    if 'pad' in name: return 'pad'
    
    # Piano & Keys
    if 'piano_jazz' in name: return 'piano_jazz'
    if 'piano_classical' in name: return 'piano_classical'
    if 'piano_ambient' in name: return 'piano_ambient'
    if 'piano_rave' in name or 'rave_piano' in name: return 'rave_piano'
    if 'piano' in name: return 'piano'
    
    if 'rhodes_jazz' in name: return 'rhodes_jazz'
    if 'rhodes_neo' in name: return 'rhodes_neo'
    if 'rhodes' in name: return 'rhodes'
    
    if 'keys_lofi' in name: return 'keys_lofi'
    if 'keys' in name or 'key' in name: return 'keys'
    
    # Percussion
    if 'snare_gated' in name: return 'snare_gated'
    if 'snare' in name: return 'snare'
    
    # Textures & Drones
    if 'texture_vinyl' in name: return 'texture_vinyl'
    if 'texture_rain' in name: return 'texture_rain'
    if 'texture_room' in name: return 'texture_room'
    if 'texture' in name: return 'texture'
    
    if 'drone_deep' in name: return 'drone_deep'
    if 'drone' in name: return 'drone'
    
    return 'other'

# =============================================================================
# PRESET MANAGEMENT
# =============================================================================

def load_presets() -> dict:
    """Load preset configurations from YAML"""
    p = Path(__file__).resolve().parents[1] / "prompts" / "free_presets.yaml"
    if not p.exists():
        return {"presets": []}
    return yaml.safe_load(p.read_text(encoding="utf-8"))

def ensure_procedural_library(date_seed: str):
    """
    Generate comprehensive stem library for all genres.
    
    This creates all the procedural audio assets needed for:
    - Electronic (Techno, Trance, House, Bass, Hard)
    - Chill (Lo-Fi, Ambient)
    - Jazz & Soul
    - Piano & Classical
    """
    STEMS_DIR.mkdir(parents=True, exist_ok=True)
    
    print("🎛️  Generating stem library...")
    
    # =========================================================================
    # ELECTRONIC - TECHNO (130 BPM)
    # =========================================================================
    for v in [1, 2]:
        if not (STEMS_DIR / f"kick_techno_v{v}.wav").exists():
            generate_techno_kick(STEMS_DIR / f"kick_techno_v{v}.wav", bpm=130, variant=v)
        if not (STEMS_DIR / f"bass_techno_v{v}.wav").exists():
            generate_techno_bass(STEMS_DIR / f"bass_techno_v{v}.wav", bpm=130, variant=v)
        if not (STEMS_DIR / f"arp_techno_v{v}.wav").exists():
            generate_techno_arp(STEMS_DIR / f"arp_techno_v{v}.wav", bpm=130, variant=v)
    
    # =========================================================================
    # ELECTRONIC - TRANCE (138 BPM)
    # =========================================================================
    # Note: Trance uses techno kicks but with different arps/pads
    for v in [1, 2]:
        if not (STEMS_DIR / f"arp_trance_v{v}.wav").exists():
            # Generate brighter, more melodic arps for trance
            generate_techno_arp(STEMS_DIR / f"arp_trance_v{v}.wav", bpm=138, variant=v)
        if not (STEMS_DIR / f"pad_trance_v{v}.wav").exists():
            # Trance pads can be generated from keys with more reverb
            generate_lofi_keys(STEMS_DIR / f"pad_trance_v{v}.wav", bpm=138, variant=v)
    
    # =========================================================================
    # ELECTRONIC - HOUSE (124 BPM)
    # =========================================================================
    for v in [1, 2]:
        if not (STEMS_DIR / f"drums_house_v{v}.wav").exists():
            generate_house_drums(STEMS_DIR / f"drums_house_v{v}.wav", bpm=124, variant=v)
        if not (STEMS_DIR / f"bass_deep_v{v}.wav").exists():
            generate_deep_house_bass(STEMS_DIR / f"bass_deep_v{v}.wav", bpm=124, variant=v)
    
    if not (STEMS_DIR / "chords_house_stab.wav").exists():
        generate_house_chords(STEMS_DIR / "chords_house_stab.wav", bpm=124)
    
    # =========================================================================
    # ELECTRONIC - BASS MUSIC (140 BPM)
    # =========================================================================
    if not (STEMS_DIR / "bass_wobble_v1.wav").exists():
        generate_wobble_bass(STEMS_DIR / "bass_wobble_v1.wav", bpm=140)
    if not (STEMS_DIR / "kick_bass_heavy.wav").exists():
        generate_techno_kick(STEMS_DIR / "kick_bass_heavy.wav", bpm=140, variant=2)
    
    # =========================================================================
    # ELECTRONIC - HARD/HARDSTYLE (150 BPM)
    # =========================================================================
    if not (STEMS_DIR / "kick_hard_gong.wav").exists():
        generate_hard_kick(STEMS_DIR / "kick_hard_gong.wav", bpm=150)
    if not (STEMS_DIR / "bass_hard_scream.wav").exists():
        generate_techno_bass(STEMS_DIR / "bass_hard_scream.wav", bpm=150, variant=2)
    
    # =========================================================================
    # ELECTRONIC - SYNTHWAVE (105 BPM)
    # =========================================================================
    if not (STEMS_DIR / "bass_synth_roll.wav").exists():
        generate_synth_bass(STEMS_DIR / "bass_synth_roll.wav", bpm=105)
    if not (STEMS_DIR / "snare_gated_80s.wav").exists():
        generate_gated_snare(STEMS_DIR / "snare_gated_80s.wav", bpm=105)
    
    # =========================================================================
    # ELECTRONIC - EURO RAVE (140 BPM)
    # =========================================================================
    if not (STEMS_DIR / "piano_rave_m1.wav").exists():
        generate_rave_piano(STEMS_DIR / "piano_rave_m1.wav", bpm=140)
    
    # =========================================================================
    # CHILL - LO-FI (85 BPM)
    # =========================================================================
    for v in [1, 2]:
        if not (STEMS_DIR / f"drums_lofi_v{v}.wav").exists():
            generate_lofi_drums(STEMS_DIR / f"drums_lofi_v{v}.wav", bpm=85, variant=v)
        if not (STEMS_DIR / f"keys_lofi_v{v}.wav").exists():
            generate_lofi_keys(STEMS_DIR / f"keys_lofi_v{v}.wav", bpm=85, variant=v)
    
    # =========================================================================
    # CHILL - AMBIENT (60 BPM)
    # =========================================================================
    for v in [1, 2]:
        if not (STEMS_DIR / f"pad_ambient_v{v}.wav").exists():
            # Ambient pads from lofi keys with processing
            generate_lofi_keys(STEMS_DIR / f"pad_ambient_v{v}.wav", bpm=60, variant=v)
        if not (STEMS_DIR / f"drone_deep_v{v}.wav").exists():
            # Deep drone from bass generator
            generate_deep_house_bass(STEMS_DIR / f"drone_deep_v{v}.wav", bpm=60, variant=v)
    
    # =========================================================================
    # JAZZ & SOUL (120 BPM base, 85 BPM neo-soul)
    # =========================================================================
    # Note: Jazz elements can be generated from existing generators with modifications
    for v in [1, 2]:
        if not (STEMS_DIR / f"drums_jazz_v{v}.wav").exists():
            # Jazz drums - use house drums with swing
            generate_house_drums(STEMS_DIR / f"drums_jazz_v{v}.wav", bpm=120, variant=v)
        
        if not (STEMS_DIR / f"bass_upright_v{v}.wav").exists():
            # Upright bass - use deep house bass with different tone
            generate_deep_house_bass(STEMS_DIR / f"bass_upright_v{v}.wav", bpm=120, variant=v)
        
        if not (STEMS_DIR / f"rhodes_jazz_v{v}.wav").exists():
            # Rhodes - use lofi keys
            generate_lofi_keys(STEMS_DIR / f"rhodes_jazz_v{v}.wav", bpm=120, variant=v)
        
        if not (STEMS_DIR / f"piano_jazz_v{v}.wav").exists():
            # Jazz piano - use house chords
            generate_house_chords(STEMS_DIR / f"piano_jazz_v{v}.wav", bpm=120)
    
    # Neo-Soul (slower tempo)
    for v in [1, 2]:
        if not (STEMS_DIR / f"drums_soul_v{v}.wav").exists():
            generate_lofi_drums(STEMS_DIR / f"drums_soul_v{v}.wav", bpm=85, variant=v)
        if not (STEMS_DIR / f"rhodes_neo_v{v}.wav").exists():
            generate_lofi_keys(STEMS_DIR / f"rhodes_neo_v{v}.wav", bpm=85, variant=v)
    
    # =========================================================================
    # PIANO & CLASSICAL (72 BPM)
    # =========================================================================
    for v in [1, 2]:
        if not (STEMS_DIR / f"piano_classical_v{v}.wav").exists():
            # Classical piano - use rave piano generator
            generate_rave_piano(STEMS_DIR / f"piano_classical_v{v}.wav", bpm=72)
        
        if not (STEMS_DIR / f"piano_ambient_v{v}.wav").exists():
            # Ambient piano - slower rave piano
            generate_rave_piano(STEMS_DIR / f"piano_ambient_v{v}.wav", bpm=60)
        
        if not (STEMS_DIR / f"pad_strings_v{v}.wav").exists():
            # String pads - from lofi keys
            generate_lofi_keys(STEMS_DIR / f"pad_strings_v{v}.wav", bpm=72, variant=v)
    
    # =========================================================================
    # FOCUS & PRODUCTIVITY
    # =========================================================================
    if not (STEMS_DIR / "kick_minimal_subtle.wav").exists():
        generate_techno_kick(STEMS_DIR / "kick_minimal_subtle.wav", bpm=100, variant=1)
    
    if not (STEMS_DIR / "drums_minimal_flow.wav").exists():
        generate_lofi_drums(STEMS_DIR / "drums_minimal_flow.wav", bpm=100, variant=1)
    
    if not (STEMS_DIR / "arp_minimal_pulse.wav").exists():
        generate_techno_arp(STEMS_DIR / "arp_minimal_pulse.wav", bpm=100, variant=1)
    
    if not (STEMS_DIR / "pad_warm_comfort.wav").exists():
        generate_lofi_keys(STEMS_DIR / "pad_warm_comfort.wav", bpm=85, variant=2)
    
    # =========================================================================
    # TEXTURES & ATMOSPHERES
    # =========================================================================
    if not (STEMS_DIR / "texture_vinyl.wav").exists():
        generate_texture(STEMS_DIR / "texture_vinyl.wav", type="vinyl")
    
    if not (STEMS_DIR / "texture_rain.wav").exists():
        generate_texture(STEMS_DIR / "texture_rain.wav", type="rain")
    
    if not (STEMS_DIR / "texture_room.wav").exists():
        generate_texture(STEMS_DIR / "texture_room.wav", type="room")
    
    print("✅ Stem library ready")

def get_random_variant(prefix: str, rnd: random.Random) -> Path | None:
    """Select random variant of a stem by prefix"""
    candidates = list(STEMS_DIR.glob(f"{prefix}*.wav"))
    if not candidates:
        candidates = list(STEMS_DIR.glob(f"*{prefix}*.wav"))
    if not candidates:
        return None
    candidates.sort()
    return rnd.choice(candidates)

def soften_nature_bed(in_wav: Path, out_wav: Path, gain_db: float = -18.0) -> None:
    """Process nature sounds to be non-intrusive background"""
    require_ffmpeg()
    out_wav.parent.mkdir(parents=True, exist_ok=True)
    
    if shutil.which("ffmpeg") is None:
        out_wav.write_bytes(in_wav.read_bytes())
        return
    
    af = (
        f"volume={gain_db}dB,"
        f"highpass=f=140,"
        f"lowpass=f=5500,"
        f"tremolo=f=0.15:d=0.10"  # Slow modulation for natural feel
    )
    
    cmd = ["ffmpeg", "-y", "-i", str(in_wav), "-af", af, str(out_wav)]
    _run(cmd)

# =============================================================================
# TRACK BUILDER
# =============================================================================

def build_track(date: str, preset: dict, total_sec: int) -> tuple[Path, dict]:
    """
    Build complete track with professional mixing.
    
    Supports:
    - Focus Engine (binaural beats)
    - Smart Mixer (pro synthesis controls)
    - All genres (Electronic, Chill, Jazz, Piano)
    
    Args:
        date: Generation date for seeding
        preset: Preset configuration dictionary
        total_sec: Track duration in seconds
    
    Returns:
        Tuple of (mp3_path, metadata_dict)
    """
    preset_id = preset.get("id", "custom")
    rnd = random.Random(f"{date}:{preset_id}")
    
    require_ffmpeg()
    TMP.mkdir(parents=True, exist_ok=True)
    OUT.mkdir(parents=True, exist_ok=True)
    
    ensure_procedural_library(date)
    
    # =========================================================================
    # 1. CHECK FOR FOCUS ENGINE REQUEST
    # =========================================================================
    focus_settings = preset.get("focus", {})
    binaural_mode = focus_settings.get("binaural_mode", "off")
    
    if binaural_mode in ["focus", "relax"]:
        # Generate Focus Engine session (bypasses standard mixing)
        preset_name = "active_focus" if binaural_mode == "focus" else "meditation"
        
        ambience_settings = focus_settings.get("ambience", {})
        add_rain = ambience_settings.get("rain", 0) > 0.1
        
        out_wav = TMP / f"{date}_{preset_id}_focus.wav"
        generate_focus_session(
            out_path=out_wav,
            preset_name=preset_name,
            duration_sec=total_sec,
            add_rain=add_rain
        )
        
        mp3 = OUT / f"free-{date}-{preset_id}.mp3"
        ffmpeg_encode_mp3(out_wav, mp3, bitrate="192k")
        
        entry = {
            "id": f"free-{date}-{preset_id}",
            "title": f"Focus Session ({binaural_mode.title()})",
            "tier": "free",
            "date": date,
            "category": "focus",
            "durationSec": total_sec,
            "url": None,
        }
        return mp3, entry
    
    # =========================================================================
    # 2. STANDARD MUSIC GENERATION
    # =========================================================================
    
    selected: list[Path] = []
    genre = preset.get("genre", "Trance")
    
    # === STEM SELECTION ===
    if "layers" in preset and "enabled" in preset["layers"]:
        # UI-driven selection (from frontend)
        enabled_layers = preset["layers"]["enabled"]
        
        for layer in enabled_layers:
            if layer == "drums":
                # Select drums based on genre
                if "House" in genre or "Deep" in genre:
                    selected.append(get_random_variant("drums_house", rnd))
                elif "Jazz" in genre:
                    selected.append(get_random_variant("drums_jazz", rnd))
                elif "Soul" in genre:
                    selected.append(get_random_variant("drums_soul", rnd))
                elif "Lofi" in genre or "Chill" in genre or "Lounge" in genre:
                    selected.append(get_random_variant("drums_lofi", rnd))
                elif "Hard" in genre:
                    selected.append(get_random_variant("kick_hard", rnd))
                elif "Ambient" in genre or "Piano" in genre:
                    pass  # No drums for ambient/piano
                elif "Work" in genre or "Coding" in genre:
                    selected.append(get_random_variant("drums_minimal", rnd))
                else:
                    selected.append(get_random_variant("kick_techno", rnd))
            
            elif layer == "bass":
                if "House" in genre or "Deep" in genre or "Lounge" in genre:
                    selected.append(get_random_variant("bass_deep", rnd))
                elif "Jazz" in genre or "Soul" in genre:
                    selected.append(get_random_variant("bass_upright", rnd))
                elif "Bass" in genre or "Dubstep" in genre:
                    selected.append(get_random_variant("bass_wobble", rnd))
                elif "Hard" in genre:
                    selected.append(get_random_variant("bass_hard", rnd))
                elif "Synth" in genre:
                    selected.append(get_random_variant("bass_synth", rnd))
                elif "Ambient" in genre:
                    selected.append(get_random_variant("drone_deep", rnd))
                else:
                    selected.append(get_random_variant("bass_techno", rnd))
            
            elif layer in ["music", "pad", "synth"]:
                if "Trance" in genre:
                    selected.append(get_random_variant("arp_trance", rnd))
                    selected.append(get_random_variant("pad_trance", rnd))
                elif "House" in genre or "Deep" in genre:
                    selected.append(get_random_variant("chords_house", rnd))
                elif "Jazz" in genre:
                    selected.append(get_random_variant("piano_jazz", rnd))
                    selected.append(get_random_variant("rhodes_jazz", rnd))
                elif "Soul" in genre:
                    selected.append(get_random_variant("rhodes_neo", rnd))
                elif "Piano" in genre:
                    selected.append(get_random_variant("piano_classical", rnd))
                    selected.append(get_random_variant("pad_strings", rnd))
                elif "Lofi" in genre or "Chill" in genre or "Study" in genre:
                    selected.append(get_random_variant("keys_lofi", rnd))
                elif "Ambient" in genre:
                    selected.append(get_random_variant("piano_ambient", rnd))
                    selected.append(get_random_variant("pad_ambient", rnd))
                elif "Lounge" in genre:
                    selected.append(get_random_variant("rhodes_jazz", rnd))
                elif "Euro" in genre:
                    selected.append(get_random_variant("piano_rave", rnd))
                elif "Work" in genre or "Coding" in genre:
                    selected.append(get_random_variant("arp_minimal", rnd))
                else:
                    selected.append(get_random_variant("arp_techno", rnd))
            
            elif layer in ["texture", "ambience"]:
                # Check ambience settings from Focus Engine
                ambience_settings = preset.get("focus", {}).get("ambience", {})
                
                if ambience_settings.get("rain", 0) > 0.1:
                    rain = get_random_variant("texture_rain", rnd)
                    if rain:
                        softened = TMP / f"{date}_{preset_id}_rain_soft.wav"
                        soften_nature_bed(rain, softened, gain_db=-18.0)
                        selected.append(softened)
                
                elif ambience_settings.get("vinyl", 0) > 0.1:
                    selected.append(get_random_variant("texture_vinyl", rnd))
                
                else:
                    # Fallback texture based on genre
                    tex_type = preset.get("texture", "none")
                    if "rain" in tex_type.lower():
                        selected.append(get_random_variant("texture_rain", rnd))
                    elif "vinyl" in tex_type.lower():
                        selected.append(get_random_variant("texture_vinyl", rnd))
                    elif "room" in tex_type.lower():
                        selected.append(get_random_variant("texture_room", rnd))
    
    else:
        # Preset-driven selection (batch/legacy mode)
        genre_lower = genre.lower()
        
        if "techno" in genre_lower or "work" in genre_lower:
            selected.extend([
                get_random_variant("kick_techno", rnd),
                get_random_variant("bass_techno", rnd),
                get_random_variant("arp_techno", rnd)
            ])
        
        elif "trance" in genre_lower:
            selected.extend([
                get_random_variant("kick_techno", rnd),
                get_random_variant("bass_techno", rnd),
                get_random_variant("arp_trance", rnd),
                get_random_variant("pad_trance", rnd)
            ])
        
        elif "house" in genre_lower:
            selected.extend([
                get_random_variant("drums_house", rnd),
                get_random_variant("bass_deep", rnd),
                get_random_variant("chords_house", rnd)
            ])
        
        elif "jazz" in genre_lower:
            selected.extend([
                get_random_variant("drums_jazz", rnd),
                get_random_variant("bass_upright", rnd),
                get_random_variant("piano_jazz", rnd),
                get_random_variant("rhodes_jazz", rnd)
            ])
        
        elif "soul" in genre_lower:
            selected.extend([
                get_random_variant("drums_soul", rnd),
                get_random_variant("bass_upright", rnd),
                get_random_variant("rhodes_neo", rnd),
                get_random_variant("pad_warm", rnd)
            ])
        
        elif "piano" in genre_lower:
            selected.extend([
                get_random_variant("piano_classical", rnd),
                get_random_variant("pad_strings", rnd),
                get_random_variant("texture_room", rnd)
            ])
        
        elif "ambient" in genre_lower:
            selected.extend([
                get_random_variant("piano_ambient", rnd),
                get_random_variant("pad_ambient", rnd),
                get_random_variant("drone_deep", rnd),
                get_random_variant("texture_rain", rnd)
            ])
        
        elif "lofi" in genre_lower or "chill" in genre_lower or "study" in genre_lower:
            selected.extend([
                get_random_variant("drums_lofi", rnd),
                get_random_variant("keys_lofi", rnd),
                get_random_variant("pad_warm", rnd),
                get_random_variant("texture_vinyl", rnd)
            ])
        
        elif "lounge" in genre_lower:
            selected.extend([
                get_random_variant("drums_lofi", rnd),
                get_random_variant("bass_upright", rnd),
                get_random_variant("rhodes_jazz", rnd),
                get_random_variant("texture_vinyl", rnd)
            ])
        
        else:
            # Default fallback
            selected.append(get_random_variant("drums_lofi", rnd))
            selected.append(get_random_variant("keys_lofi", rnd))
    
    # Clean up None values
    selected = [s for s in selected if s is not None]
    
    if not selected:
        # Absolute fallback
        selected.append(get_random_variant("keys_lofi", rnd))
    
    # === LOOPING ===
    looped_paths: list[Path] = []
    for i, stem in enumerate(selected):
        out_wav = TMP / f"{date}_{preset_id}_{i}_loop.wav"
        ffmpeg_loop_to_duration(stem, out_wav, total_sec)
        looped_paths.append(out_wav)
    
    # === PROFESSIONAL MIXING (WITH SMART MIXER FX) ===
    mixed_wav = TMP / f"{date}_{preset_id}_mixed.wav"
    
    # Extract Smart Mixer parameters from server
    smart_mixer_config = preset.get("smart_mixer", {})
    synth_params = smart_mixer_config.get("synth", None)
    
    professional_mix(looped_paths, mixed_wav, genre=genre, synth_params=synth_params)
    
    # === FADES ===
    faded = TMP / f"{date}_{preset_id}_fade.wav"
    ffmpeg_fade(mixed_wav, faded, fade_in_ms=1500, fade_out_ms=3000, total_sec=total_sec)
    
    # === LOUDNESS NORMALIZATION ===
    target_lufs = preset.get("mix", {}).get("target_lufs", -14.0)
    normed = TMP / f"{date}_{preset_id}_norm.wav"
    ffmpeg_loudnorm(faded, normed, target_lufs=target_lufs)
    
    # === MP3 ENCODING ===
    mp3 = OUT / f"free-{date}-{preset_id}.mp3"
    ffmpeg_encode_mp3(normed, mp3, bitrate="192k")
    
    entry = {
        "id": f"free-{date}-{preset_id}",
        "title": preset.get("title", f"{genre} Session"),
        "tier": "free",
        "date": date,
        "category": preset.get("category", "focus"),
        "genre": genre,
        "durationSec": total_sec,
        "url": None,
    }
    
    return mp3, entry

# =============================================================================
# CLI INTERFACE
# =============================================================================

def main():
    ap = argparse.ArgumentParser(
        description="SoundFlow Music Generation Engine v2.1.0",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  Generate daily tracks:
    python remix_daily.py --date 2025-12-19 --duration-sec 180
  
  Generate from UI JSON:
    python remix_daily.py --date 2025-12-19 --json recipe.json
  
  Generate and upload:
    python remix_daily.py --date 2025-12-19 --upload
        """
    )
    ap.add_argument("--date", required=True, help="Generation date (YYYY-MM-DD)")
    ap.add_argument("--duration-sec", type=int, default=180, help="Track duration in seconds (default: 180)")
    ap.add_argument("--upload", action="store_true", help="Upload to R2 storage")
    ap.add_argument("--json", type=str, help="JSON recipe file from UI")
    
    args = ap.parse_args()
    
    bucket = os.environ.get("R2_BUCKET")
    
    print("=" * 70)
    print("🎵 SoundFlow Music Generation Engine v2.1.0")
    print("=" * 70)
    print()
    
    # JSON mode (UI-driven)
    if args.json:
        p = Path(args.json)
        if not p.exists():
            raise RuntimeError(f"JSON file not found: {args.json}")
        
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
            combos = data.get("combinations", [])
            print(f"📂 Processing {len(combos)} combinations from UI...")
            print()
            
            for i, combo in enumerate(combos, 1):
                try:
                    print(f"[{i}/{len(combos)}] Generating: {combo.get('title', combo.get('id'))}...")
                    mp3, entry = build_track(args.date, combo, args.duration_sec)
                    print(f"✅ {mp3.name}")
                    
                    if args.upload and bucket:
                        key = f"audio/free/{args.date}/{mp3.name}"
                        upload_file(mp3, bucket, key, public=True)
                        print(f"☁️  Uploaded to R2")
                    print()
                
                except Exception as e:
                    print(f"❌ Failed: {combo.get('id')}: {e}")
                    print()
        
        except json.JSONDecodeError as e:
            print(f"❌ Invalid JSON: {e}")
    
    # Preset mode (daily generation)
    else:
        data = load_presets()
        presets = data.get("presets", [])
        
        if not presets:
            print("⚠️  No presets found in free_presets.yaml")
            print("   Creating default preset...")
            presets = [{
                "id": "default-lofi",
                "title": "Lo-Fi Study Session",
                "genre": "Chillout",
                "category": "focus"
            }]
        
        print(f"🎼 Generating {len(presets)} tracks for {args.date}...")
        print(f"⏱️  Duration: {args.duration_sec}s per track")
        print()
        
        for i, preset in enumerate(presets, 1):
            try:
                print(f"[{i}/{len(presets)}] {preset.get('title', preset['id'])}...")
                mp3, entry = build_track(args.date, preset, args.duration_sec)
                print(f"✅ {mp3.name}")
                
                if args.upload and bucket:
                    key = f"audio/free/{args.date}/{mp3.name}"
                    upload_file(mp3, bucket, key, public=True)
                    print(f"☁️  Uploaded to R2")
                print()
            
            except Exception as e:
                print(f"❌ {preset['id']}: {e}")
                print()
    
    print("=" * 70)
    print("✅ Generation complete!")
    print("=" * 70)

if __name__ == "__main__":
    main()