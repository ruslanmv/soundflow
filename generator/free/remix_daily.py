#!/usr/bin/env python3
# generator/free/remix_daily.py
"""
SoundFlow Music Generation Engine - Remix Daily v6.0 (PROFESSIONAL DJ QUALITY)

✅ NEW FEATURES:
- Long stems (16-32 bars / 1-2 minutes)
- Energy curves (peak, drop, linear, build)
- Dynamic arrangement (intro/build/peak/breakdown/outro)
- Professional 3-layer kicks
- Chord progressions with evolution
- Filter sweeps and automation
- Drum fills and variations
- Broadcast-quality mixing

Supports:
- All electronic genres (Techno, House, Trance, Deep, Bass, Hard)
- Chill/ambient (Lo-Fi, Ambient, Chillout, Lounge)
- Jazz & Soul (Jazz, Neo-Soul)
- Piano & Classical
- Focus sessions (Binaural beats)
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import yaml
from scipy.io import wavfile

from free.music_engine import (
    # PRIMARY API - Professional stem rendering
    render_stem,
    
    # FOCUS ENGINE
    generate_focus_session,
    
    # DSP for mixing
    apply_overdrive,
    apply_algorithmic_reverb,
    apply_lowpass,
    apply_highpass,
    apply_resonant_filter,
    apply_parametric_eq,
    multiband_process,
    soft_clip,
    normalize,
    SAMPLE_RATE,
)

from common.audio_utils import (
    require_ffmpeg,
    ffmpeg_loop_to_duration,
    ffmpeg_fade,
    ffmpeg_loudnorm,
    ffmpeg_encode_mp3,
    _run,
)

from common.r2_upload import upload_file

# =============================================================================
# PATHS
# =============================================================================

ROOT = Path(__file__).resolve().parent
ASSETS = ROOT / "assets"
STEMS_DIR = ASSETS / "stems"

TMP = Path(".soundflow_tmp/free")
OUT = Path(".soundflow_out/free")

# =============================================================================
# CONFIGURATION
# =============================================================================

# Professional stem lengths (in bars)
STEM_LENGTHS = {
    "short": 8,      # ~30s @ 128 BPM
    "medium": 16,    # ~1min @ 128 BPM (DEFAULT)
    "long": 32,      # ~2min @ 128 BPM
    "full": 64,      # ~4min @ 128 BPM
}

# Energy curve presets
ENERGY_CURVES = {
    "peak": "Classic club track (intro → build → peak → breakdown → outro)",
    "drop": "Bass/dubstep style (high → drop → low → build → peak)",
    "build": "Progressive build (steady increase to climax)",
    "linear": "Constant energy (steady throughout)",
}

# =============================================================================
# SAFETY
# =============================================================================

_SAFE_CHARS_RE = re.compile(r"[^a-zA-Z0-9._-]+")

def safe_slug(s: str, max_len: int = 120) -> str:
    s = (s or "").strip()
    s = _SAFE_CHARS_RE.sub("_", s)
    s = s.strip("._-")
    if not s:
        s = "track"
    return s[:max_len]

def assert_audio_not_empty(path: Path, min_bytes: int = 20_000) -> None:
    if not path.exists():
        raise RuntimeError(f"Output file missing: {path}")
    size = path.stat().st_size
    if size < min_bytes:
        raise RuntimeError(f"Output file too small ({size} bytes): {path}")

# =============================================================================
# PRESET LOADING
# =============================================================================

def load_presets() -> dict:
    p = Path(__file__).resolve().parents[1] / "prompts" / "free_presets.yaml"
    if not p.exists():
        return {"presets": []}
    return yaml.safe_load(p.read_text(encoding="utf-8"))

# =============================================================================
# OPTIONAL FALLBACK LIBRARY (Legacy support)
# =============================================================================

def ensure_procedural_library(date_seed: str) -> None:
    """Legacy fallback - not needed for render_stem"""
    STEMS_DIR.mkdir(parents=True, exist_ok=True)
    return

def get_random_variant(prefix: str, rnd: random.Random) -> Optional[Path]:
    candidates = list(STEMS_DIR.glob(f"{prefix}*.wav"))
    if not candidates:
        candidates = list(STEMS_DIR.glob(f"*{prefix}*.wav"))
    if not candidates:
        return None
    candidates.sort()
    return rnd.choice(candidates)

def soften_nature_bed(in_wav: Path, out_wav: Path, gain_db: float = -18.0) -> None:
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
# PROFESSIONAL MIXING (GENRE-AWARE)
# =============================================================================

def detect_stem_type(filename: str) -> str:
    """Detect stem type from filename"""
    name = filename.lower()
    
    # Kicks
    if "kick" in name:
        return "kick"
    
    # Bass
    if "bass" in name:
        return "bass"
    
    # Drums
    if "drum" in name or "snare" in name or "clap" in name or "hat" in name:
        return "drums"
    
    # Melodic
    if "arp" in name:
        return "arp"
    if "synth" in name or "lead" in name:
        return "synth"
    if "chord" in name or "stab" in name:
        return "chords"
    if "piano" in name or "rhodes" in name:
        return "piano"
    if "pad" in name or "string" in name:
        return "pad"
    
    # Textures
    if "texture" in name or "vinyl" in name or "rain" in name or "room" in name:
        return "texture"
    
    return "other"

def apply_stem_eq(audio: np.ndarray, stem_type: str, genre: str) -> np.ndarray:
    """
    Genre-aware EQ carving for professional separation.
    """
    
    if stem_type == "kick":
        # Kick: Boost sub, cut mids
        audio = apply_highpass(audio, 25)
        audio = apply_parametric_eq(audio, 60, gain_db=2.5, q=1.2)
        audio = apply_parametric_eq(audio, 300, gain_db=-3, q=2.0)
        if "hard" in genre.lower() or "bass" in genre.lower():
            audio = apply_parametric_eq(audio, 100, gain_db=3.0, q=1.0)
    
    elif stem_type == "bass":
        # Bass: Duck for kick, control lows
        audio = apply_highpass(audio, 35)
        audio = apply_parametric_eq(audio, 55, gain_db=-5, q=1.5)
        audio = apply_lowpass(audio, 4500)
        if "deep" in genre.lower() or "house" in genre.lower():
            audio = apply_parametric_eq(audio, 80, gain_db=1.5, q=0.8)
    
    elif stem_type in ("synth", "arp"):
        # Synths: Remove lows, boost presence
        audio = apply_highpass(audio, 180)
        audio = apply_parametric_eq(audio, 2800, gain_db=1.5, q=0.8)
        if "trance" in genre.lower():
            audio = apply_parametric_eq(audio, 5000, gain_db=1.0, q=0.5)
    
    elif stem_type in ("pad", "chords"):
        # Pads: Remove lows, make spacious
        audio = apply_highpass(audio, 220)
        audio = apply_lowpass(audio, 8000)
        audio = apply_parametric_eq(audio, 1200, gain_db=0.8, q=0.6)
    
    elif stem_type == "piano":
        # Piano: Natural warmth
        audio = apply_highpass(audio, 80)
        audio = apply_parametric_eq(audio, 250, gain_db=1.2, q=0.8)
        audio = apply_parametric_eq(audio, 3500, gain_db=0.8, q=0.5)
    
    elif stem_type == "drums":
        # Full drums: Preserve transients
        audio = apply_highpass(audio, 40)
        audio = apply_parametric_eq(audio, 80, gain_db=1.5, q=1.0)
        audio = apply_parametric_eq(audio, 8000, gain_db=1.0, q=0.4)
    
    elif stem_type == "texture":
        # Textures: High-pass aggressively
        audio = apply_highpass(audio, 450)
        audio = apply_lowpass(audio, 7500)
    
    return audio

def master_chain(audio: np.ndarray, genre: str) -> np.ndarray:
    """
    Professional mastering chain with genre-specific processing.
    """
    if audio.size < 64:
        return audio.astype(np.float32)
    
    # 1. Subsonic filter
    audio = apply_highpass(audio, 20, order=4)
    
    # 2. Genre-specific EQ
    if "jazz" in genre.lower() or "soul" in genre.lower():
        # Jazz: Natural warmth
        audio = apply_parametric_eq(audio, 100, gain_db=0.5, q=0.8)
        audio = apply_parametric_eq(audio, 2500, gain_db=0.3, q=0.6)
        audio = apply_parametric_eq(audio, 8000, gain_db=0.2, q=0.5)
    
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
        audio = apply_parametric_eq(audio, 80, gain_db=0.8, q=0.7)
        audio = apply_parametric_eq(audio, 3000, gain_db=0.5, q=0.5)
        audio = apply_parametric_eq(audio, 10000, gain_db=0.3, q=0.4)
    
    # 3. Multiband compression
    if "ambient" in genre.lower() or "piano" in genre.lower():
        audio = multiband_process(audio, low_gain=1.02, mid_gain=0.99, high_gain=1.01)
    else:
        audio = multiband_process(audio, low_gain=1.05, mid_gain=0.98, high_gain=1.02)
    
    # 4. Soft clipping
    if "jazz" in genre.lower() or "classical" in genre.lower():
        audio = soft_clip(audio * 1.05, threshold=0.90)
    else:
        audio = soft_clip(audio * 1.1, threshold=0.85)
    
    # 5. Final normalization
    if "techno" in genre.lower() or "hard" in genre.lower() or "bass" in genre.lower():
        target_rms = 0.15  # Loud
    elif "jazz" in genre.lower() or "classical" in genre.lower():
        target_rms = 0.10  # Dynamic
    else:
        target_rms = 0.12  # Balanced
    
    rms = float(np.sqrt(np.mean(audio ** 2))) if audio.size else 0.0
    if rms > 1e-6:
        audio = audio * (target_rms / rms)
    
    # 6. True peak limiting
    audio = np.clip(audio, -0.95, 0.95)
    
    # 7. Final normalize
    audio = normalize(audio, target=0.97)
    
    return audio.astype(np.float32)

def professional_mix(
    stem_paths: List[Path],
    output_wav: Path,
    genre: str = "techno",
    synth_params: Optional[dict] = None
) -> None:
    """
    Professional mixing with genre-aware processing.
    """
    stems: List[Dict[str, Any]] = []
    max_len = 0
    
    for stem_path in stem_paths:
        if stem_path is None:
            continue
        rate, data = wavfile.read(str(stem_path))
        if data.dtype == np.int16:
            data = data.astype(np.float32) / 32768.0
        if data.ndim > 1:
            data = data[:, 0]  # Mono for now
        
        stems.append({
            "audio": data.astype(np.float32, copy=False),
            "path": stem_path,
            "type": detect_stem_type(stem_path.name),
        })
        max_len = max(max_len, len(data))
    
    if not stems:
        raise RuntimeError("No stems to mix")
    
    # Process each stem
    processed: List[np.ndarray] = []
    
    for stem in stems:
        audio = stem["audio"]
        st = stem["type"]
        
        # Pad to same length
        if len(audio) < max_len:
            audio = np.pad(audio, (0, max_len - len(audio)))
        
        # Apply genre-aware EQ
        audio = apply_stem_eq(audio, st, genre)
        
        # Genre-specific gain staging
        if "jazz" in genre.lower() or "soul" in genre.lower():
            gain_map = {
                "drums": 0.65, "kick": 0.70,
                "bass": 0.68,
                "piano": 0.75,
                "pad": 0.38,
                "texture": 0.15,
            }
        elif "ambient" in genre.lower() or "piano" in genre.lower():
            gain_map = {
                "piano": 0.80,
                "pad": 0.48,
                "texture": 0.18,
            }
        elif "chill" in genre.lower() or "lofi" in genre.lower():
            gain_map = {
                "drums": 0.58, "kick": 0.60,
                "bass": 0.65,
                "synth": 0.62, "arp": 0.58,
                "pad": 0.40,
                "texture": 0.22,
            }
        else:
            # Electronic
            gain_map = {
                "kick": 0.88,
                "bass": 0.72,
                "drums": 0.68,
                "synth": 0.52, "arp": 0.48,
                "pad": 0.42, "chords": 0.45,
                "piano": 0.50,
                "texture": 0.18,
            }
        
        gain = gain_map.get(st, 0.50)
        audio = audio * float(gain)
        
        processed.append(audio.astype(np.float32, copy=False))
    
    # Sum stems
    mix = np.sum(processed, axis=0).astype(np.float32)
    
    # =========================================================================
    # SMART MIXER EFFECTS
    # =========================================================================
    if synth_params:
        # Drive/Saturation
        drive_amt = float(synth_params.get("drive", 0.0))
        if drive_amt > 0.05:
            mix = apply_overdrive(mix, drive=drive_amt * 4.0)
        
        # Filter Cutoff/Resonance
        cutoff_amt = float(synth_params.get("cutoff", 1.0))
        resonance_amt = float(synth_params.get("resonance", 0.0))
        
        if cutoff_amt < 0.98:
            cutoff_freq = 100.0 + (20000.0 * (cutoff_amt ** 2))
            
            if resonance_amt > 0.1:
                mix = apply_resonant_filter(
                    mix,
                    cutoff=cutoff_freq,
                    resonance=resonance_amt * 0.9
                )
            else:
                mix = apply_lowpass(mix, cutoff=cutoff_freq)
        
        # Space/Reverb
        space_amt = float(synth_params.get("space", 0.0))
        if space_amt > 0.05:
            room_size = 0.3 + (space_amt * 0.6)
            wet_amount = space_amt * 0.5
            mix = apply_algorithmic_reverb(
                mix,
                room_size=room_size,
                wet=wet_amount
            )
    
    # Master chain
    mix = master_chain(mix, genre=genre)
    
    # Save
    output_wav.parent.mkdir(parents=True, exist_ok=True)
    wavfile.write(str(output_wav), SAMPLE_RATE, (mix * 32767.0).astype(np.int16))
    print(f"✅ Mixed: {output_wav.name}")

# =============================================================================
# SCHEMA COMPATIBILITY
# =============================================================================

def _coerce_enabled_layers(preset: dict) -> List[str]:
    layers = preset.get("layers")
    if isinstance(layers, dict):
        enabled = layers.get("enabled", [])
        if isinstance(enabled, list):
            return [str(x) for x in enabled]
    
    if isinstance(layers, list):
        return [str(x) for x in layers]
    
    music_cfg = preset.get("music", {})
    ml = music_cfg.get("layers")
    if isinstance(ml, list):
        return [str(x) for x in ml]
    
    return ["drums", "bass", "music"]

def _get_genre(preset: dict) -> str:
    music_cfg = preset.get("music", {})
    genre = music_cfg.get("genre") or preset.get("genre") or "Techno"
    return str(genre)

def _get_bpm(preset: dict) -> int:
    music_cfg = preset.get("music", {})
    bpm = music_cfg.get("bpm") or preset.get("bpm") or 128
    try:
        return int(bpm)
    except Exception:
        return 128

def _get_key(preset: dict) -> str:
    music_cfg = preset.get("music", {})
    k = music_cfg.get("key") or preset.get("key") or "A"
    return str(k)

def _get_variation(preset: dict) -> float:
    v = preset.get("variation", 0.25)
    try:
        return float(v)
    except Exception:
        return 0.25

def _get_target_lufs(preset: dict) -> float:
    master_cfg = preset.get("master", {})
    if isinstance(master_cfg, dict) and "target_lufs" in master_cfg:
        return float(master_cfg["target_lufs"])
    mix_cfg = preset.get("mix", {})
    if isinstance(mix_cfg, dict) and "target_lufs" in mix_cfg:
        return float(mix_cfg["target_lufs"])
    return -14.0

def _get_synth_params(preset: dict) -> Optional[dict]:
    # API schema
    mixer_cfg = preset.get("mixer", {})
    if isinstance(mixer_cfg, dict):
        sp = mixer_cfg.get("synth")
        if isinstance(sp, dict):
            return {
                "cutoff": float(sp.get("cutoff", 100.0)) / 100.0,
                "resonance": float(sp.get("resonance", 0.0)) / 100.0,
                "drive": float(sp.get("drive", 0.0)) / 100.0,
                "space": float(sp.get("space", 0.0)) / 100.0,
            }
    
    # Recipe schema
    sm = preset.get("smart_mixer", {})
    if isinstance(sm, dict):
        sp = sm.get("synth")
        if isinstance(sp, dict):
            return sp
    
    return None

def _get_focus_mode(preset: dict) -> str:
    focus_cfg = preset.get("focus", {})
    if isinstance(focus_cfg, dict) and "binaural_mode" in focus_cfg:
        return str(focus_cfg.get("binaural_mode", "off"))
    
    if isinstance(focus_cfg, dict) and "mode" in focus_cfg:
        return str(focus_cfg.get("mode", "off"))
    
    return "off"

def _get_engine_mode(preset: dict) -> str:
    m = preset.get("mode")
    if isinstance(m, str) and m in ("music", "focus", "hybrid"):
        return m
    
    fm = _get_focus_mode(preset)
    if fm in ("focus", "relax"):
        return "focus"
    return "music"

def _get_focus_mix(preset: dict) -> float:
    focus_cfg = preset.get("focus", {})
    if isinstance(focus_cfg, dict) and "mix" in focus_cfg:
        try:
            return float(focus_cfg.get("mix", 30.0)) / 100.0
        except Exception:
            return 0.30
    return 0.30

def _get_ambience(preset: dict) -> Dict[str, float]:
    focus_cfg = preset.get("focus", {})
    amb = {}
    if isinstance(focus_cfg, dict):
        amb = focus_cfg.get("ambience", {}) or {}
    
    def _norm(x: Any) -> float:
        try:
            v = float(x)
        except Exception:
            return 0.0
        if v > 1.0:
            v = v / 100.0
        return float(np.clip(v, 0.0, 1.0))
    
    return {
        "rain": _norm(amb.get("rain", 0.0)),
        "vinyl": _norm(amb.get("vinyl", 0.0)),
        "white": _norm(amb.get("white", 0.0)),
    }

def _get_stem_length(preset: dict, default: str = "medium") -> int:
    """
    Get stem length in bars from preset.
    
    Options:
    - "short": 8 bars (~30s)
    - "medium": 16 bars (~1min) [DEFAULT]
    - "long": 32 bars (~2min)
    - "full": 64 bars (~4min)
    """
    length = preset.get("stem_length", default)
    if isinstance(length, int):
        return max(1, int(length))
    
    length_str = str(length).lower().strip()
    return STEM_LENGTHS.get(length_str, STEM_LENGTHS["medium"])

def _get_energy_curve(preset: dict, default: str = "peak") -> str:
    """
    Get energy curve type from preset.
    
    Options:
    - "peak": Classic club track
    - "drop": Bass/dubstep style
    - "build": Progressive build
    - "linear": Constant energy
    """
    curve = preset.get("energy_curve", default)
    curve_str = str(curve).lower().strip()
    
    if curve_str in ENERGY_CURVES:
        return curve_str
    
    return default

# =============================================================================
# PROFESSIONAL STEM RENDERING
# =============================================================================

def render_stem_for_request(
    *,
    out_dir: Path,
    date: str,
    safe_id: str,
    seed_str: str,
    genre: str,
    bpm: int,
    key: str,
    layer: str,
    variation: float,
    ambience: Dict[str, float],
    bars: int = 16,
    energy_curve: str = "peak",
) -> List[Path]:
    """
    Render professional-quality stems (1-2 minutes) with energy dynamics.
    
    Args:
        bars: Stem length in bars (8=short, 16=medium, 32=long, 64=full)
        energy_curve: "peak", "drop", "build", or "linear"
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    
    layer_l = str(layer).lower().strip()
    paths: List[Path] = []
    
    # Deterministic variant per layer
    v = float(np.clip(variation, 0.0, 1.0))
    base_variant = 1 + (abs(hash(layer_l)) % 8)
    variant = int(base_variant + int(v * 3.0))
    
    # Map UI/API layers to render_stem
    if layer_l in ("kick",):
        p = out_dir / f"{date}_{safe_id}_kick_v{variant}_b{bars}.wav"
        render_stem(
            out_path=p,
            stem="kick",
            genre=genre,
            bpm=bpm,
            key=key,
            seed=seed_str,
            variant=variant,
            bars=bars,
            energy_curve=energy_curve
        )
        paths.append(p)
        return paths
    
    if layer_l in ("drums",):
        p = out_dir / f"{date}_{safe_id}_drums_v{variant}_b{bars}.wav"
        render_stem(
            out_path=p,
            stem="drums",
            genre=genre,
            bpm=bpm,
            key=key,
            seed=seed_str,
            variant=variant,
            bars=bars,
            energy_curve=energy_curve
        )
        paths.append(p)
        return paths
    
    if layer_l in ("bass",):
        p = out_dir / f"{date}_{safe_id}_bass_v{variant}_b{bars}.wav"
        render_stem(
            out_path=p,
            stem="bass",
            genre=genre,
            bpm=bpm,
            key=key,
            seed=seed_str,
            variant=variant,
            bars=bars,
            energy_curve=energy_curve
        )
        paths.append(p)
        return paths
    
    if layer_l in ("music", "pad", "synth", "melody", "chords"):
        p = out_dir / f"{date}_{safe_id}_music_v{variant}_b{bars}.wav"
        render_stem(
            out_path=p,
            stem="music",
            genre=genre,
            bpm=bpm,
            key=key,
            seed=seed_str,
            variant=variant,
            bars=bars,
            energy_curve=energy_curve
        )
        paths.append(p)
        return paths
    
    if layer_l in ("texture", "ambience"):
        # Textures are longer for natural evolution
        texture_bars = max(bars, 16)
        
        if ambience.get("rain", 0.0) > 0.01:
            p_r = out_dir / f"{date}_{safe_id}_texture_rain_v{variant}.wav"
            render_stem(
                out_path=p_r,
                stem="texture",
                genre=genre,
                bpm=bpm,
                key=key,
                seed=seed_str,
                variant=variant,
                bars=texture_bars,
                texture_type="rain",
                energy_curve="linear"
            )
            softened = out_dir / f"{date}_{safe_id}_texture_rain_soft_v{variant}.wav"
            soften_nature_bed(p_r, softened, gain_db=-18.0)
            paths.append(softened)
        
        if ambience.get("vinyl", 0.0) > 0.01 or not paths:
            p_v = out_dir / f"{date}_{safe_id}_texture_vinyl_v{variant}.wav"
            render_stem(
                out_path=p_v,
                stem="texture",
                genre=genre,
                bpm=bpm,
                key=key,
                seed=seed_str,
                variant=variant,
                bars=texture_bars,
                texture_type="vinyl",
                energy_curve="linear"
            )
            paths.append(p_v)
        
        return paths
    
    return []

# =============================================================================
# FALLBACK STEM SELECTION (Legacy)
# =============================================================================

def _fallback_choose_stems_for_layer(
    layer: str,
    genre: str,
    rnd: random.Random
) -> List[Path]:
    """Legacy fallback - only used if render_stem fails"""
    g = genre.lower()
    
    if layer == "drums":
        if "house" in g:
            p = get_random_variant("drums_house", rnd)
            return [p] if p else []
        if "lofi" in g or "chill" in g:
            p = get_random_variant("drums_lofi", rnd)
            return [p] if p else []
        p = get_random_variant("kick_techno", rnd)
        return [p] if p else []
    
    if layer == "bass":
        if "house" in g:
            p = get_random_variant("bass_deep", rnd)
            return [p] if p else []
        if "bass" in g:
            p = get_random_variant("bass_wobble", rnd)
            return [p] if p else []
        p = get_random_variant("bass_techno", rnd)
        return [p] if p else []
    
    if layer in ("music", "pad", "synth"):
        if "house" in g:
            p = get_random_variant("chords_house", rnd)
            return [p] if p else []
        if "lofi" in g or "chill" in g:
            p = get_random_variant("keys_lofi", rnd)
            return [p] if p else []
        p = get_random_variant("arp_techno", rnd)
        return [p] if p else []
    
    if layer in ("texture", "ambience"):
        p = get_random_variant("texture_vinyl", rnd)
        return [p] if p else []
    
    return []

# =============================================================================
# TRACK BUILDER (PROFESSIONAL)
# =============================================================================

def build_track(date: str, preset: dict, total_sec: int) -> Tuple[Path, dict]:
    """
    Build professional DJ-quality track with long stems and energy dynamics.
    """
    require_ffmpeg()
    TMP.mkdir(parents=True, exist_ok=True)
    OUT.mkdir(parents=True, exist_ok=True)
    
    # Optional fallback library
    ensure_procedural_library(date)
    
    preset_id = str(preset.get("id", "custom"))
    safe_id = safe_slug(preset_id)
    
    # Deterministic seed
    seed_str = str(preset.get("seed") or f"{date}:{preset_id}")
    
    # Extract parameters
    genre = _get_genre(preset)
    bpm = _get_bpm(preset)
    key = _get_key(preset)
    variation = _get_variation(preset)
    target_lufs = _get_target_lufs(preset)
    synth_params = _get_synth_params(preset)
    
    engine_mode = _get_engine_mode(preset)
    focus_mode = _get_focus_mode(preset)
    focus_mix = _get_focus_mix(preset)
    ambience = _get_ambience(preset)
    
    # NEW: Professional features
    stem_length_bars = _get_stem_length(preset, default="medium")
    energy_curve = _get_energy_curve(preset, default="peak")
    
    channels = int(preset.get("channels", 2))
    channels = 2 if channels not in (1, 2) else channels
    
    enabled_layers = _coerce_enabled_layers(preset)
    
    # Fallback RNG
    rnd = random.Random(seed_str + ":fallback")
    
    rendered_dir = TMP / "rendered" / safe_id
    
    print("=" * 70)
    print(f"🎵 Building: {preset.get('title', preset_id)}")
    print(f"   Genre: {genre} | BPM: {bpm} | Key: {key}")
    print(f"   Stem Length: {stem_length_bars} bars")
    print(f"   Energy Curve: {energy_curve}")
    print(f"   Layers: {', '.join(enabled_layers)}")
    print("=" * 70)
    
    # -------------------------------------------------------------------------
    # 1) FOCUS RENDER
    # -------------------------------------------------------------------------
    focus_audio_path: Optional[Path] = None
    if engine_mode in ("focus", "hybrid") and focus_mode in ("focus", "relax"):
        if focus_mode == "focus":
            base_freq, beat_freq = 250.0, 20.0
        else:
            base_freq, beat_freq = 150.0, 6.0
        
        focus_wav = TMP / f"{date}_{safe_id}_focus.wav"
        generate_focus_session(
            out_path=focus_wav,
            duration_sec=float(total_sec),
            base_freq=base_freq,
            beat_freq=beat_freq,
            add_rain=(ambience["rain"] > 0.01),
        )
        focus_audio_path = focus_wav
        
        if engine_mode == "focus":
            mp3 = OUT / f"free-{date}-{safe_id}.mp3"
            normed = TMP / f"{date}_{safe_id}_focus_norm.wav"
            ffmpeg_loudnorm(focus_wav, normed, target_lufs=target_lufs)
            ffmpeg_encode_mp3(normed, mp3, bitrate="320k")
            assert_audio_not_empty(mp3)
            
            entry = {
                "id": f"free-{date}-{preset_id}",
                "title": preset.get("title", f"Focus Session ({focus_mode})"),
                "tier": "free",
                "date": date,
                "category": "focus",
                "durationSec": total_sec,
                "url": None,
                "genre": genre,
                "bpm": bpm,
                "mode": "focus",
                "focus_mode": focus_mode,
                "layers": [],
                "seed": seed_str,
            }
            return mp3, entry
    
    # -------------------------------------------------------------------------
    # 2) MUSIC STEM RENDER (PROFESSIONAL)
    # -------------------------------------------------------------------------
    selected: List[Path] = []
    
    # Auto-add texture if ambience enabled
    needs_texture = (ambience["rain"] > 0.01) or (ambience["vinyl"] > 0.01)
    layers_to_render = list(enabled_layers)
    if needs_texture and ("texture" not in [x.lower() for x in layers_to_render]):
        layers_to_render.append("texture")
    
    for layer in layers_to_render:
        try:
            print(f"   🎛️  Rendering {layer} ({stem_length_bars} bars, {energy_curve} curve)...")
            selected.extend(
                render_stem_for_request(
                    out_dir=rendered_dir,
                    date=date,
                    safe_id=safe_id,
                    seed_str=seed_str,
                    genre=genre,
                    bpm=bpm,
                    key=key,
                    layer=layer,
                    variation=variation,
                    ambience=ambience,
                    bars=stem_length_bars,
                    energy_curve=energy_curve,
                )
            )
        except Exception as e:
            print(f"   ⚠️  Render failed for {layer}: {e}")
            print(f"   ⚠️  Using fallback stems...")
            selected.extend(_fallback_choose_stems_for_layer(str(layer), genre, rnd))
    
    # Dedup
    dedup: List[Path] = []
    seen = set()
    for s in selected:
        if not s:
            continue
        try:
            keyp = str(s.resolve())
        except Exception:
            keyp = str(s)
        if keyp in seen:
            continue
        seen.add(keyp)
        dedup.append(s)
    selected = dedup
    
    if not selected:
        fb = get_random_variant("kick_techno", rnd) or get_random_variant("drums_lofi", rnd)
        if fb:
            selected = [fb]
        else:
            raise RuntimeError("No stems available")
    
    print(f"   ✅ Generated {len(selected)} stems")
    
    # -------------------------------------------------------------------------
    # 3) NO LOOPING NEEDED (Stems are already full length)
    # -------------------------------------------------------------------------
    # Since stems are now 16-64 bars (1-4 minutes), we don't need to loop them
    # We'll just trim/pad to exact duration if needed
    
    loops: List[Path] = []
    for i, stem in enumerate(selected):
        # Read stem
        rate, data = wavfile.read(str(stem))
        if data.dtype == np.int16:
            data = data.astype(np.float32) / 32768.0
        if data.ndim > 1:
            data = data[:, 0]
        
        target_samples = int(total_sec * SAMPLE_RATE)
        
        if len(data) < target_samples:
            # Pad if too short
            data = np.pad(data, (0, target_samples - len(data)))
        else:
            # Trim if too long
            data = data[:target_samples]
        
        # Save trimmed/padded stem
        out_wav = TMP / f"{date}_{safe_id}_{i}_fitted.wav"
        wavfile.write(str(out_wav), SAMPLE_RATE, (data * 32767.0).astype(np.int16))
        loops.append(out_wav)
    
    # -------------------------------------------------------------------------
    # 4) PROFESSIONAL MIXDOWN
    # -------------------------------------------------------------------------
    print(f"   🎚️  Mixing with professional chain...")
    mixed_wav = TMP / f"{date}_{safe_id}_mixed.wav"
    professional_mix(loops, mixed_wav, genre=genre, synth_params=synth_params)
    
    # -------------------------------------------------------------------------
    # 5) FADE + LOUDNORM
    # -------------------------------------------------------------------------
    print(f"   🎛️  Applying fades and loudness normalization...")
    faded = TMP / f"{date}_{safe_id}_fade.wav"
    ffmpeg_fade(mixed_wav, faded, fade_in_ms=1500, fade_out_ms=3000, total_sec=total_sec)
    
    normed = TMP / f"{date}_{safe_id}_norm.wav"
    ffmpeg_loudnorm(faded, normed, target_lufs=target_lufs)
    
    # -------------------------------------------------------------------------
    # 6) HYBRID BLEND (Optional)
    # -------------------------------------------------------------------------
    if engine_mode == "hybrid" and focus_audio_path is not None:
        print(f"   🔀 Blending with focus session ({int(focus_mix*100)}% focus)...")
        hybrid_wav = TMP / f"{date}_{safe_id}_hybrid.wav"
        
        fm = float(np.clip(focus_mix, 0.0, 1.0))
        mm = 1.0 - fm
        
        af = (
            f"[0:a]volume={mm}[m];"
            f"[1:a]volume={fm}[f];"
            f"[m][f]amix=inputs=2:normalize=0:dropout_transition=0"
        )
        cmd = [
            "ffmpeg", "-y",
            "-i", str(normed),
            "-i", str(focus_audio_path),
            "-filter_complex", af,
            str(hybrid_wav),
        ]
        _run(cmd)
        
        hybrid_norm = TMP / f"{date}_{safe_id}_hybrid_norm.wav"
        ffmpeg_loudnorm(hybrid_wav, hybrid_norm, target_lufs=target_lufs)
        final_wav = hybrid_norm
    else:
        final_wav = normed
    
    # -------------------------------------------------------------------------
    # 7) EXPORT MP3 (320kbps for professional quality)
    # -------------------------------------------------------------------------
    print(f"   💿 Encoding to MP3 (320kbps)...")
    mp3 = OUT / f"free-{date}-{safe_id}.mp3"
    ffmpeg_encode_mp3(final_wav, mp3, bitrate="320k")
    assert_audio_not_empty(mp3)
    
    print(f"   ✅ Complete: {mp3.name}")
    print("=" * 70)
    
    entry = {
        "id": f"free-{date}-{preset_id}",
        "title": preset.get("title", preset_id),
        "tier": "free",
        "date": date,
        "category": preset.get("category", "music"),
        "durationSec": total_sec,
        "url": None,
        "genre": genre,
        "bpm": bpm,
        "key": key,
        "mode": engine_mode,
        "focus_mode": focus_mode,
        "layers": enabled_layers,
        "stem_length_bars": stem_length_bars,
        "energy_curve": energy_curve,
        "seed": seed_str,
    }
    
    return mp3, entry

# =============================================================================
# CLI
# =============================================================================

def main() -> None:
    ap = argparse.ArgumentParser(
        description="SoundFlow Professional DJ Engine v6.0",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  Generate daily tracks:
    python remix_daily.py --date 2025-12-20 --duration-sec 180
  
  Generate from JSON recipe:
    python remix_daily.py --date 2025-12-20 --json recipe.json
  
  Generate and upload:
    python remix_daily.py --date 2025-12-20 --upload

Features:
  - Long stems (16-64 bars / 1-4 minutes)
  - Energy curves (peak/drop/build/linear)
  - Professional 3-layer kicks
  - Dynamic arrangement
  - Broadcast-quality mixing
        """
    )
    ap.add_argument("--date", required=True, help="Generation date (YYYY-MM-DD)")
    ap.add_argument("--duration-sec", type=int, default=180, help="Track duration in seconds (default: 180)")
    ap.add_argument("--upload", action="store_true", help="Upload to R2 storage")
    ap.add_argument("--json", type=str, help="JSON recipe file")
    
    args = ap.parse_args()
    
    bucket = os.environ.get("R2_BUCKET")
    
    print("=" * 70)
    print("🎵 SoundFlow Professional DJ Engine v6.0")
    print("=" * 70)
    print()
    
    if args.json:
        p = Path(args.json)
        if not p.exists():
            raise RuntimeError(f"JSON file not found: {args.json}")
        
        data = json.loads(p.read_text(encoding="utf-8"))
        combos = data.get("combinations", [])
        print(f"📂 Processing {len(combos)} combinations from recipe...")
        print()
        
        for i, combo in enumerate(combos, 1):
            print(f"[{i}/{len(combos)}]")
            mp3, _ = build_track(args.date, combo, args.duration_sec)
            
            if args.upload and bucket:
                key = f"audio/free/{args.date}/{mp3.name}"
                upload_file(mp3, bucket, key, public=True)
                print(f"   ☁️  Uploaded to R2: {key}")
            print()
    
    else:
        data = load_presets()
        presets = data.get("presets", [])
        
        if not presets:
            print("⚠️  No presets found in free_presets.yaml")
            print("   Using default preset...")
            presets = [{
                "id": "default-trance",
                "title": "Trance Session",
                "genre": "Trance",
                "category": "music",
                "stem_length": "medium",
                "energy_curve": "peak"
            }]
        
        print(f"🎼 Generating {len(presets)} tracks for {args.date}...")
        print(f"⏱️  Duration: {args.duration_sec}s per track")
        print()
        
        for i, preset in enumerate(presets, 1):
            print(f"[{i}/{len(presets)}]")
            mp3, _ = build_track(args.date, preset, args.duration_sec)
            
            if args.upload and bucket:
                key = f"audio/free/{args.date}/{mp3.name}"
                upload_file(mp3, bucket, key, public=True)
                print(f"   ☁️  Uploaded to R2: {key}")
            print()
    
    print("=" * 70)
    print("✅ Generation complete!")
    print("=" * 70)

if __name__ == "__main__":
    main()