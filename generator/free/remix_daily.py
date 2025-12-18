# generator/free/remix_daily.py
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

from free.music_engine import (
    generate_techno_kick,
    generate_techno_bass,
    generate_techno_arp,
    generate_lofi_drums,
    generate_lofi_keys,
    generate_texture,
    generate_house_drums,
    generate_deep_house_bass,
    generate_house_chords,
    generate_wobble_bass,
    generate_hard_kick,
    generate_synth_bass,
    generate_gated_snare,
    generate_rave_piano,
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
# PROFESSIONAL MIXING UTILITIES
# =============================================================================

def apply_stem_eq(audio: np.ndarray, stem_type: str) -> np.ndarray:
    """
    Apply EQ carving based on stem type.
    This prevents frequency masking and creates professional separation.
    """
    from free.music_engine import apply_highpass, apply_lowpass, apply_bandpass, apply_parametric_eq
    
    if stem_type == "kick":
        # Kick: Boost sub, cut mids
        audio = apply_highpass(audio, 25)
        audio = apply_parametric_eq(audio, 60, gain_db=2, q=1.2)
        audio = apply_parametric_eq(audio, 300, gain_db=-3, q=2.0)
    
    elif stem_type == "bass":
        # Bass: Duck for kick, remove lows
        audio = apply_highpass(audio, 35)
        audio = apply_parametric_eq(audio, 55, gain_db=-5, q=1.5)
        audio = apply_lowpass(audio, 4500)
    
    elif stem_type == "synth" or stem_type == "arp":
        # Synths: Remove lows, boost presence
        audio = apply_highpass(audio, 180)
        audio = apply_parametric_eq(audio, 2800, gain_db=1.5, q=0.8)
    
    elif stem_type == "pad" or stem_type == "chords":
        # Pads: Remove lows, make spacious
        audio = apply_highpass(audio, 220)
        audio = apply_lowpass(audio, 8000)
    
    elif stem_type == "texture":
        # Textures: High-pass aggressively
        audio = apply_highpass(audio, 450)
        audio = apply_lowpass(audio, 7500)
    
    return audio

def professional_mix(stem_paths: list[Path], output: Path, genre: str = "techno") -> None:
    """
    Professional mixing with proper gain staging and frequency carving.
    This is the SECRET SAUCE for broadcast-quality output.
    """
    from free.music_engine import (
        apply_lowpass, apply_highpass, multiband_process,
        apply_algorithmic_reverb, normalize
    )
    
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
        
        # Gain staging (relative levels)
        gain_map = {
            'kick': 0.88,
            'bass': 0.72,
            'drums': 0.68,
            'synth': 0.52,
            'arp': 0.48,
            'pad': 0.42,
            'chords': 0.45,
            'texture': 0.18,
        }
        gain = gain_map.get(stem_type, 0.50)
        audio = audio * gain
        
        # Apply sidechain to bass/pads (if kick exists)
        if stem_type in ['bass', 'pad', 'synth', 'arp']:
            sidechain_file = STEMS_DIR / "kick_techno_v1_sidechain.npy"
            if sidechain_file.exists():
                sidechain_env = np.load(sidechain_file)
                if len(sidechain_env) == len(audio):
                    audio = audio * sidechain_env
        
        processed_stems.append(audio)
    
    # 3. Sum all stems
    mixed = np.sum(processed_stems, axis=0).astype(np.float32)
    
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
    """
    from free.music_engine import (
        apply_highpass, apply_lowpass, multiband_process,
        apply_parametric_eq, soft_clip, normalize
    )
    
    # 1. Subsonic filter
    audio = apply_highpass(audio, 20, order=4)
    
    # 2. Gentle EQ adjustments
    audio = apply_parametric_eq(audio, 80, gain_db=0.8, q=0.7)   # Sub warmth
    audio = apply_parametric_eq(audio, 3000, gain_db=0.5, q=0.5) # Presence
    audio = apply_parametric_eq(audio, 10000, gain_db=0.3, q=0.4) # Air
    
    # 3. Multiband gentle compression (simulated)
    audio = multiband_process(audio, low_gain=1.05, mid_gain=0.98, high_gain=1.02)
    
    # 4. Soft clipping (analog-style limiting)
    audio = soft_clip(audio * 1.1, threshold=0.85)
    
    # 5. Final normalization to target LUFS
    # Approximate LUFS-based normalization
    rms = np.sqrt(np.mean(audio ** 2))
    target_rms = 0.15 if "techno" in genre or "hard" in genre else 0.12
    if rms > 1e-6:
        audio = audio * (target_rms / rms)
    
    # 6. True peak limiting
    audio = np.clip(audio, -0.95, 0.95)
    
    # 7. Final normalize to -0.3dB
    audio = normalize(audio, target=0.97)
    
    return audio

def detect_stem_type(filename: str) -> str:
    """Detect stem type from filename"""
    name = filename.lower()
    if 'kick' in name:
        return 'kick'
    elif 'bass' in name:
        return 'bass'
    elif 'drum' in name:
        return 'drums'
    elif 'arp' in name:
        return 'arp'
    elif 'synth' in name or 'lead' in name:
        return 'synth'
    elif 'chord' in name or 'stab' in name or 'piano' in name:
        return 'chords'
    elif 'pad' in name or 'key' in name:
        return 'pad'
    elif 'texture' in name or 'vinyl' in name or 'rain' in name:
        return 'texture'
    return 'other'

# =============================================================================
# PRESET MANAGEMENT
# =============================================================================

def load_presets() -> dict:
    """Load preset configurations"""
    p = Path(__file__).resolve().parents[1] / "prompts" / "free_presets.yaml"
    if not p.exists():
        return {"presets": []}
    return yaml.safe_load(p.read_text(encoding="utf-8"))

def ensure_procedural_library(date_seed: str):
    """Generate all stem assets"""
    STEMS_DIR.mkdir(parents=True, exist_ok=True)
    
    print("🎛️  Generating stem library...")
    
    # Techno (130 BPM)
    for v in [1, 2]:
        if not (STEMS_DIR / f"kick_techno_v{v}.wav").exists():
            generate_techno_kick(STEMS_DIR / f"kick_techno_v{v}.wav", bpm=130, variant=v)
        if not (STEMS_DIR / f"bass_techno_v{v}.wav").exists():
            generate_techno_bass(STEMS_DIR / f"bass_techno_v{v}.wav", bpm=130, variant=v)
        if not (STEMS_DIR / f"arp_techno_v{v}.wav").exists():
            generate_techno_arp(STEMS_DIR / f"arp_techno_v{v}.wav", bpm=130, variant=v)
    
    # House (124 BPM)
    for v in [1, 2]:
        if not (STEMS_DIR / f"drums_house_v{v}.wav").exists():
            generate_house_drums(STEMS_DIR / f"drums_house_v{v}.wav", bpm=124, variant=v)
        if not (STEMS_DIR / f"bass_deep_v{v}.wav").exists():
            generate_deep_house_bass(STEMS_DIR / f"bass_deep_v{v}.wav", bpm=124, variant=v)
    
    if not (STEMS_DIR / "chords_house_stab.wav").exists():
        generate_house_chords(STEMS_DIR / "chords_house_stab.wav", bpm=124)
    
    # Lo-Fi (85 BPM)
    for v in [1, 2]:
        if not (STEMS_DIR / f"drums_lofi_v{v}.wav").exists():
            generate_lofi_drums(STEMS_DIR / f"drums_lofi_v{v}.wav", bpm=85, variant=v)
        if not (STEMS_DIR / f"keys_lofi_v{v}.wav").exists():
            generate_lofi_keys(STEMS_DIR / f"keys_lofi_v{v}.wav", bpm=85, variant=v)
    
    # Bass Music (140 BPM)
    if not (STEMS_DIR / "bass_wobble_v1.wav").exists():
        generate_wobble_bass(STEMS_DIR / "bass_wobble_v1.wav", bpm=140)
    
    # Hard (150 BPM)
    if not (STEMS_DIR / "kick_hard_gong.wav").exists():
        generate_hard_kick(STEMS_DIR / "kick_hard_gong.wav", bpm=150)
    
    # Synthwave (105 BPM)
    if not (STEMS_DIR / "bass_synth_roll.wav").exists():
        generate_synth_bass(STEMS_DIR / "bass_synth_roll.wav", bpm=105)
    if not (STEMS_DIR / "snare_gated_80s.wav").exists():
        generate_gated_snare(STEMS_DIR / "snare_gated_80s.wav", bpm=105)
    
    # Euro (140 BPM)
    if not (STEMS_DIR / "piano_rave_m1.wav").exists():
        generate_rave_piano(STEMS_DIR / "piano_rave_m1.wav", bpm=140)
    
    # Textures
    if not (STEMS_DIR / "texture_vinyl.wav").exists():
        generate_texture(STEMS_DIR / "texture_vinyl.wav", type="vinyl")
    if not (STEMS_DIR / "texture_rain.wav").exists():
        generate_texture(STEMS_DIR / "texture_rain.wav", type="rain")
    
    print("✅ Stem library ready")

def get_random_variant(prefix: str, rnd: random.Random) -> Path | None:
    """Select random variant of a stem"""
    candidates = list(STEMS_DIR.glob(f"{prefix}*.wav"))
    if not candidates:
        candidates = list(STEMS_DIR.glob(f"*{prefix}*.wav"))
    if not candidates:
        return None
    candidates.sort()
    return rnd.choice(candidates)

def soften_nature_bed(in_wav: Path, out_wav: Path, gain_db: float = -18.0) -> None:
    """Process nature sounds to be non-intrusive"""
    require_ffmpeg()
    out_wav.parent.mkdir(parents=True, exist_ok=True)
    
    if shutil.which("ffmpeg") is None:
        out_wav.write_bytes(in_wav.read_bytes())
        return
    
    af = (
        f"volume={gain_db}dB,"
        f"highpass=f=140,"
        f"lowpass=f=5500,"
        f"tremolo=f=0.15:d=0.10"
    )
    
    cmd = ["ffmpeg", "-y", "-i", str(in_wav), "-af", af, str(out_wav)]
    _run(cmd)

# =============================================================================
# TRACK BUILDER
# =============================================================================

def build_track(date: str, preset: dict, total_sec: int) -> tuple[Path, dict]:
    """Build complete track with professional mixing"""
    preset_id = preset.get("id", "custom")
    rnd = random.Random(f"{date}:{preset_id}")
    
    require_ffmpeg()
    TMP.mkdir(parents=True, exist_ok=True)
    OUT.mkdir(parents=True, exist_ok=True)
    
    ensure_procedural_library(date)
    
    selected: list[Path] = []
    
    # === STEM SELECTION ===
    if "layers" in preset and "enabled" in preset["layers"]:
        # UI-driven selection
        enabled_layers = preset["layers"]["enabled"]
        genre = preset.get("genre", "Trance")
        
        for layer in enabled_layers:
            if layer == "drums":
                if "House" in genre:
                    selected.append(get_random_variant("drums_house", rnd))
                elif "Lofi" in genre or "Chill" in genre:
                    selected.append(get_random_variant("drums_lofi", rnd))
                elif "Hard" in genre:
                    selected.append(get_random_variant("kick_hard", rnd))
                else:
                    selected.append(get_random_variant("kick_techno", rnd))
            
            elif layer == "bass":
                if "House" in genre:
                    selected.append(get_random_variant("bass_deep", rnd))
                elif "Bass" in genre or "Dubstep" in genre:
                    selected.append(get_random_variant("bass_wobble", rnd))
                elif "Synth" in genre:
                    selected.append(get_random_variant("bass_synth", rnd))
                else:
                    selected.append(get_random_variant("bass_techno", rnd))
            
            elif layer in ["music", "pad", "synth"]:
                if "House" in genre:
                    selected.append(get_random_variant("chords_house", rnd))
                elif "Lofi" in genre:
                    selected.append(get_random_variant("keys_lofi", rnd))
                elif "Euro" in genre:
                    selected.append(get_random_variant("piano_rave", rnd))
                else:
                    selected.append(get_random_variant("arp_techno", rnd))
            
            elif layer in ["texture", "ambience"]:
                tex_type = preset.get("texture", "none")
                if "rain" in tex_type.lower():
                    rain = get_random_variant("texture_rain", rnd)
                    if rain:
                        softened = TMP / f"{date}_{preset_id}_rain_soft.wav"
                        soften_nature_bed(rain, softened, gain_db=-18.0)
                        selected.append(softened)
                elif "vinyl" in tex_type.lower():
                    selected.append(get_random_variant("texture_vinyl", rnd))
    
    else:
        # Preset-driven selection
        if "deep_work" in preset_id or "techno" in preset_id:
            selected.extend([
                get_random_variant("kick_techno", rnd),
                get_random_variant("bass_techno", rnd),
                get_random_variant("arp_techno", rnd)
            ])
        
        elif "study" in preset_id or "chill" in preset_id:
            selected.extend([
                get_random_variant("drums_lofi", rnd),
                get_random_variant("keys_lofi", rnd),
                get_random_variant("texture_vinyl", rnd)
            ])
        
        elif "relax" in preset_id or "nature" in preset_id:
            selected.append(get_random_variant("keys_lofi", rnd))
            rain = get_random_variant("texture_rain", rnd)
            if rain:
                softened = TMP / f"{date}_{preset_id}_rain_soft.wav"
                soften_nature_bed(rain, softened, gain_db=-18.0)
                selected.append(softened)
        
        elif "house" in preset_id:
            selected.extend([
                get_random_variant("drums_house", rnd),
                get_random_variant("bass_deep", rnd),
                get_random_variant("chords_house", rnd)
            ])
        
        elif "synth" in preset_id or "retro" in preset_id:
            selected.extend([
                get_random_variant("bass_synth", rnd),
                get_random_variant("snare_gated", rnd)
            ])
        
        else:
            selected.append(get_random_variant("texture_vinyl", rnd))
    
    # Clean up None values
    selected = [s for s in selected if s is not None]
    
    if not selected:
        raise RuntimeError(f"No stems found for {preset_id}")
    
    # === LOOPING ===
    looped_paths: list[Path] = []
    for i, stem in enumerate(selected):
        out_wav = TMP / f"{date}_{preset_id}_{i}_loop.wav"
        ffmpeg_loop_to_duration(stem, out_wav, total_sec)
        looped_paths.append(out_wav)
    
    # === PROFESSIONAL MIXING ===
    mixed_wav = TMP / f"{date}_{preset_id}_mixed.wav"
    genre = preset.get("genre", "techno")
    professional_mix(looped_paths, mixed_wav, genre=genre)
    
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
        "title": preset.get("title", preset_id),
        "tier": "free",
        "date": date,
        "category": preset.get("category", "focus"),
        "durationSec": total_sec,
        "url": None,
    }
    
    return mp3, entry

# =============================================================================
# CLI INTERFACE
# =============================================================================

def main():
    ap = argparse.ArgumentParser(description="SoundFlow Music Generation Engine")
    ap.add_argument("--date", required=True, help="Generation date (YYYY-MM-DD)")
    ap.add_argument("--duration-sec", type=int, default=120, help="Track duration in seconds")
    ap.add_argument("--upload", action="store_true", help="Upload to R2 storage")
    ap.add_argument("--json", type=str, help="JSON recipe file from UI")
    
    args = ap.parse_args()
    
    bucket = os.environ.get("R2_BUCKET")
    
    # JSON mode (UI-driven)
    if args.json:
        p = Path(args.json)
        if not p.exists():
            raise RuntimeError(f"JSON file not found: {args.json}")
        
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
            combos = data.get("combinations", [])
            print(f"📂 Processing {len(combos)} combinations...")
            
            for combo in combos:
                try:
                    mp3, entry = build_track(args.date, combo, args.duration_sec)
                    print(f"✅ {mp3.name}")
                    
                    if args.upload and bucket:
                        key = f"audio/free/{args.date}/{mp3.name}"
                        upload_file(mp3, bucket, key, public=True)
                
                except Exception as e:
                    print(f"❌ {combo.get('id')}: {e}")
        
        except json.JSONDecodeError as e:
            print(f"❌ Invalid JSON: {e}")
    
    # Preset mode (daily generation)
    else:
        data = load_presets()
        presets = data.get("presets", [])
        
        print(f"🎵 Generating {len(presets)} tracks for {args.date}...")
        
        for preset in presets:
            try:
                mp3, entry = build_track(args.date, preset, args.duration_sec)
                print(f"✅ {mp3.name}")
                
                if args.upload and bucket:
                    key = f"audio/free/{args.date}/{mp3.name}"
                    upload_file(mp3, bucket, key, public=True)
            
            except Exception as e:
                print(f"❌ {preset['id']}: {e}")

if __name__ == "__main__":
    main()