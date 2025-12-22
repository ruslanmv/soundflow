#!/usr/bin/env python3
# generator/free/remix_daily.py
"""
SoundFlow Music Generation Engine - Remix Daily v6.1 (STEREO & CLUB MIX)

✅ UPGRADES v6.1:
- True Stereo Pipeline (No more mono collapse)
- Club-Standard Bass Management (Mono < 120Hz)
- Split-Band Filtering (Kick stays punchy while synths filter)
- Analog-style Saturation curves
- "Air" EQ boost for electronic genres
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
    render_stem,
    generate_focus_session,
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

STEM_LENGTHS = {
    "short": 8,
    "medium": 16,
    "long": 32,
    "full": 64,
}

ENERGY_CURVES = {
    "peak": "Classic club track (intro → build → peak → breakdown → outro)",
    "drop": "Bass/dubstep style (high → drop → low → build → peak)",
    "build": "Progressive build (steady increase to climax)",
    "linear": "Constant energy (steady throughout)",
}

# =============================================================================
# AUDIO UTILS (STEREO AWARE)
# =============================================================================

def _to_stereo(x: np.ndarray) -> np.ndarray:
    """
    Ensure audio is stereo float32.
    Accepts: mono (N,) or stereo (N,2).
    """
    if x is None:
        return np.zeros((0, 2), dtype=np.float32)
    x = x.astype(np.float32, copy=False)
    if x.ndim == 1:
        # Duplicate mono channel to L/R
        return np.stack([x, x], axis=1).astype(np.float32, copy=False)
    if x.ndim == 2 and x.shape[1] == 2:
        return x.astype(np.float32, copy=False)
    # Fallback: flatten to mono then duplicate
    m = x.reshape(-1).astype(np.float32, copy=False)
    return np.stack([m, m], axis=1).astype(np.float32, copy=False)

def _mono_below(audio: np.ndarray, cutoff_hz: float = 120.0) -> np.ndarray:
    """
    Make low frequencies mono (club standard), keep highs stereo.
    Works on stereo arrays (N,2).
    """
    a = _to_stereo(audio)
    # Split bands L/R independent
    low_l = apply_lowpass(a[:, 0], cutoff_hz)
    low_r = apply_lowpass(a[:, 1], cutoff_hz)
    
    # Mono Sum the low end
    low_m = 0.5 * (low_l + low_r)
    
    # Extract highs
    high_l = a[:, 0] - low_l
    high_r = a[:, 1] - low_r
    
    # Recombine: Mono Lows + Stereo Highs
    out_l = low_m + high_l
    out_r = low_m + high_r
    return np.stack([out_l, out_r], axis=1).astype(np.float32)

def _safe_read_wav(path: Path) -> np.ndarray:
    """
    Read WAV into float32 stereo. Supports int16/float WAV.
    """
    try:
        rate, data = wavfile.read(str(path))
    except Exception:
        return np.zeros((0, 2), dtype=np.float32)

    if data.dtype == np.int16:
        x = (data.astype(np.float32) / 32768.0)
    else:
        x = data.astype(np.float32, copy=False)
    return _to_stereo(x)

def _write_wav(path: Path, audio: np.ndarray) -> None:
    """
    Write float32 stereo/mono to int16 WAV.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    x = np.clip(audio.astype(np.float32, copy=False), -1.0, 1.0)
    if x.ndim == 2 and x.shape[1] == 2:
        wavfile.write(str(path), SAMPLE_RATE, (x * 32767.0).astype(np.int16))
    else:
        wavfile.write(str(path), SAMPLE_RATE, (x.reshape(-1) * 32767.0).astype(np.int16))

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


def load_daily_plan() -> dict:
    """Load daily generation plan mapping site categories to genres."""
    p = Path(__file__).resolve().parents[1] / "prompts" / "free_daily_plan.yaml"
    if not p.exists():
        raise RuntimeError(f"Daily plan not found: {p}")
    return yaml.safe_load(p.read_text(encoding="utf-8"))


def apply_genre_rotation(track_config: dict, date: str) -> dict:
    """
    Apply genre rotation based on day of week.

    Args:
        track_config: Track configuration from daily plan
        date: Date string (YYYY-MM-DD)

    Returns:
        Updated track config with rotated genre
    """
    from datetime import datetime

    # Parse date to get day of week (0=Monday, 6=Sunday)
    dt = datetime.strptime(date, "%Y-%m-%d")
    day_of_week = dt.weekday()

    # Load rotation schedule
    plan = load_daily_plan()
    rotation = plan.get("genre_rotation", {})

    # Get site category
    site_category = track_config.get("siteCategory", "").lower().replace(" ", "_")

    # Apply rotation if defined
    if site_category in rotation:
        day_rotation = rotation[site_category]
        if day_of_week in day_rotation:
            rotated_genre = day_rotation[day_of_week]
            track_config = dict(track_config)  # Copy
            track_config["genre"] = rotated_genre
            print(f"   🔄 Genre rotation ({site_category}): {rotated_genre} (day {day_of_week})")

    return track_config

def ensure_procedural_library(date_seed: str) -> None:
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
    name = filename.lower()
    if "kick" in name: return "kick"
    if "bass" in name: return "bass"
    if "drum" in name or "snare" in name or "clap" in name or "hat" in name: return "drums"
    if "arp" in name: return "arp"
    if "synth" in name or "lead" in name: return "synth"
    if "chord" in name or "stab" in name: return "chords"
    if "piano" in name or "rhodes" in name: return "piano"
    if "pad" in name or "string" in name: return "pad"
    if "texture" in name or "vinyl" in name or "rain" in name or "room" in name: return "texture"
    return "other"

def apply_stem_eq(audio: np.ndarray, stem_type: str, genre: str) -> np.ndarray:
    """Genre-aware EQ carving."""
    # Ensure stereo input for processing filters
    audio = _to_stereo(audio)
    
    # Process both channels symmetrically
    def _eq(ch):
        if stem_type == "kick":
            ch = apply_highpass(ch, 25)
            ch = apply_parametric_eq(ch, 60, gain_db=2.5, q=1.2)
            ch = apply_parametric_eq(ch, 300, gain_db=-3, q=2.0)
            if "hard" in genre.lower() or "bass" in genre.lower():
                ch = apply_parametric_eq(ch, 100, gain_db=3.0, q=1.0)
        elif stem_type == "bass":
            ch = apply_highpass(ch, 35)
            ch = apply_parametric_eq(ch, 55, gain_db=-5, q=1.5) # Slot for kick
            ch = apply_lowpass(ch, 4500)
        elif stem_type in ("synth", "arp"):
            ch = apply_highpass(ch, 180)
            ch = apply_parametric_eq(ch, 2800, gain_db=1.5, q=0.8)
        elif stem_type in ("pad", "chords"):
            ch = apply_highpass(ch, 220)
            ch = apply_lowpass(ch, 8000)
        elif stem_type == "piano":
            ch = apply_highpass(ch, 80)
            ch = apply_parametric_eq(ch, 250, gain_db=1.2, q=0.8)
        elif stem_type == "drums":
            ch = apply_highpass(ch, 40)
            ch = apply_parametric_eq(ch, 80, gain_db=1.5, q=1.0)
            ch = apply_parametric_eq(ch, 8000, gain_db=1.0, q=0.4)
        elif stem_type == "texture":
            ch = apply_highpass(ch, 450)
            ch = apply_lowpass(ch, 7500)
        return ch

    l = _eq(audio[:, 0])
    r = _eq(audio[:, 1])
    return np.stack([l, r], axis=1)

def master_chain(audio: np.ndarray, genre: str) -> np.ndarray:
    """Professional mastering chain."""
    if audio.size < 64: return audio
    
    audio = _to_stereo(audio)
    
    # 1. Subsonic filter
    audio = apply_highpass(audio, 20, order=4)
    
    # 2. Genre-specific EQ (Gentle master bus adjustments)
    if "jazz" in genre.lower() or "soul" in genre.lower():
        audio = apply_parametric_eq(audio, 100, gain_db=0.5, q=0.8)
        audio = apply_parametric_eq(audio, 8000, gain_db=0.3, q=0.5)
    elif "electronic" in genre.lower() or "house" in genre.lower() or "techno" in genre.lower():
        audio = apply_parametric_eq(audio, 70, gain_db=0.8, q=0.7) # Kick weight
        audio = apply_parametric_eq(audio, 10000, gain_db=0.5, q=0.4) # Air
    
    # 3. Glue Compression (Multiband)
    audio = multiband_process(audio, low_gain=1.03, mid_gain=0.99, high_gain=1.02)
    
    # 4. Soft Clip (Saturation before limiting)
    audio = soft_clip(audio * 1.1, threshold=0.90)
    
    # 5. True Peak Limit
    audio = np.clip(audio, -0.98, 0.98)
    
    # 6. Final Normalize
    audio = normalize(audio, target=0.98)
    
    return audio.astype(np.float32)

def professional_mix(
    stem_paths: List[Path],
    output_wav: Path,
    genre: str = "techno",
    synth_params: Optional[dict] = None
) -> None:
    """
    Professional mixing with STEREO processing.
    """
    stems: List[Dict[str, Any]] = []
    max_len = 0
    
    for stem_path in stem_paths:
        if stem_path is None: continue
        # ✅ LOAD AS STEREO
        data = _safe_read_wav(stem_path)
        
        stems.append({
            "audio": data,
            "path": stem_path,
            "type": detect_stem_type(stem_path.name),
        })
        max_len = max(max_len, data.shape[0])
    
    if not stems:
        raise RuntimeError("No stems to mix")
    
    processed: List[np.ndarray] = []
    
    for stem in stems:
        audio = stem["audio"]
        st = stem["type"]
        
        # Pad to same length
        if audio.shape[0] < max_len:
            pad_amt = max_len - audio.shape[0]
            audio = np.pad(audio, ((0, pad_amt), (0, 0)))
        
        # EQ
        audio = apply_stem_eq(audio, st, genre)
        
        # ✅ CLUB FIX: Kick and Bass < 120Hz = MONO
        if st in ("kick", "bass"):
            audio = _mono_below(audio, cutoff_hz=120.0)
        
        # Gain Staging
        gain_map = {
            "kick": 0.90,
            "bass": 0.75,
            "drums": 0.70,
            "synth": 0.55,
            "arp": 0.50,
            "pad": 0.45,
            "piano": 0.55,
            "texture": 0.15,
        }
        
        # Adjust gain map for lighter genres
        if "jazz" in genre.lower() or "chill" in genre.lower():
            gain_map["kick"] = 0.65
            gain_map["piano"] = 0.80
            
        gain = gain_map.get(st, 0.50)
        audio = audio * float(gain)
        processed.append(audio)
    
    # Sum stems
    mix = np.sum(processed, axis=0).astype(np.float32)
    
    # =========================================================================
    # SMART MIXER (TONE SHAPING)
    # =========================================================================
    if synth_params:
        # Drive (Saturation)
        drive_amt = float(synth_params.get("drive", 0.0))
        if drive_amt > 0.05:
            mix = apply_overdrive(mix, drive=drive_amt * 3.2)
        
        # Filter (Cutoff)
        cutoff_amt = float(synth_params.get("cutoff", 1.0))
        resonance_amt = float(synth_params.get("resonance", 0.0))
        
        # ✅ SMART FILTER: Don't kill the kick! 
        # Only filter mids/highs when cutoff is lowered.
        if cutoff_amt < 0.98:
            cutoff_freq = 600.0 + (18000.0 * (cutoff_amt ** 2))
            
            # Split mix
            low_end = apply_lowpass(mix, 200.0)
            mid_high = mix - low_end
            
            if resonance_amt > 0.1:
                mid_high = apply_resonant_filter(mid_high, cutoff=cutoff_freq, resonance=resonance_amt * 0.6)
            else:
                mid_high = apply_lowpass(mid_high, cutoff=cutoff_freq)
            
            mix = low_end + mid_high
        
        # Air Boost for electronic genres (Prevents dullness)
        gl = genre.lower()
        if any(x in gl for x in ["house", "techno", "trance", "edm", "dance"]):
            mix = apply_parametric_eq(mix, 9500, gain_db=0.6, q=0.7)

        # Reverb
        space_amt = float(synth_params.get("space", 0.0))
        if space_amt > 0.05:
            room_size = 0.3 + (space_amt * 0.6)
            wet_amount = space_amt * 0.5
            mix = apply_algorithmic_reverb(mix, room_size=room_size, wet=wet_amount)
    
    # ✅ Final Mono-Low Integrity
    mix = _mono_below(mix, cutoff_hz=120.0)
    
    # Master
    mix = master_chain(mix, genre=genre)
    
    # Save
    output_wav.parent.mkdir(parents=True, exist_ok=True)
    _write_wav(output_wav, mix)
    print(f"✅ Mixed: {output_wav.name}")

# =============================================================================
# SCHEMA COMPATIBILITY HELPERS
# =============================================================================

def _coerce_enabled_layers(preset: dict) -> List[str]:
    layers = preset.get("layers")
    if isinstance(layers, dict):
        enabled = layers.get("enabled", [])
        if isinstance(enabled, list): return [str(x) for x in enabled]
    if isinstance(layers, list): return [str(x) for x in layers]
    music_cfg = preset.get("music", {})
    ml = music_cfg.get("layers")
    if isinstance(ml, list): return [str(x) for x in ml]
    return ["drums", "bass", "music"]

def _get_genre(preset: dict) -> str:
    music_cfg = preset.get("music", {})
    return str(music_cfg.get("genre") or preset.get("genre") or "Techno")

def _get_bpm(preset: dict) -> int:
    music_cfg = preset.get("music", {})
    try: return int(music_cfg.get("bpm") or preset.get("bpm") or 128)
    except: return 128

def _get_key(preset: dict) -> str:
    music_cfg = preset.get("music", {})
    return str(music_cfg.get("key") or preset.get("key") or "A")

def _get_variation(preset: dict) -> float:
    try: return float(preset.get("variation", 0.25))
    except: return 0.25

def _get_target_lufs(preset: dict) -> float:
    master_cfg = preset.get("master", {})
    if isinstance(master_cfg, dict) and "target_lufs" in master_cfg:
        return float(master_cfg["target_lufs"])
    return -14.0

def _get_synth_params(preset: dict) -> Optional[dict]:
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
    return None

def _get_focus_mode(preset: dict) -> str:
    focus_cfg = preset.get("focus", {})
    if isinstance(focus_cfg, dict) and "mode" in focus_cfg:
        return str(focus_cfg.get("mode", "off"))
    return "off"

def _get_engine_mode(preset: dict) -> str:
    m = preset.get("mode")
    if isinstance(m, str) and m in ("music", "focus", "hybrid"): return m
    return "music"

def _get_focus_mix(preset: dict) -> float:
    focus_cfg = preset.get("focus", {})
    if isinstance(focus_cfg, dict) and "mix" in focus_cfg:
        try: return float(focus_cfg.get("mix", 30.0)) / 100.0
        except: return 0.30
    return 0.30

def _get_ambience(preset: dict) -> Dict[str, float]:
    focus_cfg = preset.get("focus", {})
    amb = {}
    if isinstance(focus_cfg, dict):
        amb = focus_cfg.get("ambience", {}) or {}
    
    def _norm(x: Any) -> float:
        try: v = float(x)
        except: return 0.0
        if v > 1.0: v = v / 100.0
        return float(np.clip(v, 0.0, 1.0))
    
    return {
        "rain": _norm(amb.get("rain", 0.0)),
        "vinyl": _norm(amb.get("vinyl", 0.0)),
        "white": _norm(amb.get("white", 0.0)),
    }

def _get_stem_length(preset: dict, default: str = "medium") -> int:
    length = preset.get("stem_length", default)
    if isinstance(length, int): return max(1, int(length))
    return STEM_LENGTHS.get(str(length).lower().strip(), STEM_LENGTHS["medium"])

def _get_energy_curve(preset: dict, default: str = "peak") -> str:
    curve = preset.get("energy_curve", default)
    curve_str = str(curve).lower().strip()
    return curve_str if curve_str in ENERGY_CURVES else default

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
    duration_sec: float,
) -> List[Path]:
    """
    Render professional-quality stems.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    layer_l = str(layer).lower().strip()
    paths: List[Path] = []
    
    # Deterministic variant per layer
    v = float(np.clip(variation, 0.0, 1.0))
    base_variant = 1 + (abs(hash(layer_l)) % 8)
    variant = int(base_variant + int(v * 3.0))
    
    stem_map = {
        "kick": "kick",
        "drums": "drums",
        "bass": "bass",
        "music": "music", "pad": "music", "synth": "music", "melody": "music", "chords": "music"
    }
    
    if layer_l in stem_map:
        stem_type = stem_map[layer_l]
        p = out_dir / f"{date}_{safe_id}_{stem_type}_v{variant}.wav"
        render_stem(
            out_path=p,
            stem=stem_type,
            genre=genre,
            bpm=bpm,
            key=key,
            seed=seed_str,
            variant=variant,
            bars=bars,
            energy_curve=energy_curve,
            duration_sec=duration_sec # ✅ PASS DURATION TO ENGINE
        )
        paths.append(p)
        return paths
    
    if layer_l in ("texture", "ambience"):
        # Auto-calculate texture needs
        if ambience.get("rain", 0.0) > 0.01:
            p_r = out_dir / f"{date}_{safe_id}_texture_rain.wav"
            render_stem(
                out_path=p_r,
                stem="texture",
                genre=genre,
                bpm=bpm,
                key=key,
                seed=seed_str,
                variant=variant,
                bars=bars,
                duration_sec=duration_sec,
                texture_type="rain",
                energy_curve="linear"
            )
            # Soften rain specifically
            softened = out_dir / f"{date}_{safe_id}_texture_rain_soft.wav"
            soften_nature_bed(p_r, softened, gain_db=-18.0)
            paths.append(softened)
        
        if ambience.get("vinyl", 0.0) > 0.01:
            p_v = out_dir / f"{date}_{safe_id}_texture_vinyl.wav"
            render_stem(
                out_path=p_v,
                stem="texture",
                genre=genre,
                bpm=bpm,
                key=key,
                seed=seed_str,
                variant=variant,
                bars=bars,
                duration_sec=duration_sec,
                texture_type="vinyl",
                energy_curve="linear"
            )
            paths.append(p_v)
            
        return paths
    
    return []

# =============================================================================
# FALLBACK STEM SELECTION (Legacy)
# =============================================================================

def _fallback_choose_stems_for_layer(layer: str, genre: str, rnd: random.Random) -> List[Path]:
    g = genre.lower()
    if layer == "drums":
        return [get_random_variant("drums_house" if "house" in g else "kick_techno", rnd)]
    if layer == "bass":
        return [get_random_variant("bass_deep" if "house" in g else "bass_techno", rnd)]
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
    
    ensure_procedural_library(date)
    
    preset_id = str(preset.get("id", "custom"))
    safe_id = safe_slug(preset_id)
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
    
    stem_length_bars = _get_stem_length(preset, default="medium")
    energy_curve = _get_energy_curve(preset, default="peak")
    enabled_layers = _coerce_enabled_layers(preset)
    
    rnd = random.Random(seed_str + ":fallback")
    rendered_dir = TMP / "rendered" / safe_id
    
    print("=" * 70)
    print(f"🎵 Building: {preset.get('title', preset_id)}")
    print(f"   Genre: {genre} | BPM: {bpm} | Key: {key} | Duration: {total_sec}s")
    print(f"   Energy Curve: {energy_curve}")
    print(f"   Layers: {', '.join(enabled_layers)}")
    print("=" * 70)
    
    # -------------------------------------------------------------------------
    # 1) FOCUS RENDER
    # -------------------------------------------------------------------------
    focus_audio_path: Optional[Path] = None
    if engine_mode in ("focus", "hybrid") and focus_mode in ("focus", "relax"):
        base_freq = 250.0 if focus_mode == "focus" else 150.0
        beat_freq = 20.0 if focus_mode == "focus" else 6.0
        
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
            return mp3, {
                "id": f"free-{date}-{preset_id}",
                "title": preset.get("title", f"Focus Session ({focus_mode})"),
                "durationSec": total_sec,
                "genre": genre,
                "mode": "focus",
            }
    
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
            print(f"   🎛️  Rendering {layer}...")
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
                    duration_sec=total_sec # ✅ Ensure stems match track duration
                )
            )
        except Exception as e:
            print(f"   ⚠️  Render failed for {layer}: {e}")
            selected.extend(_fallback_choose_stems_for_layer(str(layer), genre, rnd))
    
    # Dedup
    dedup: List[Path] = []
    seen = set()
    for s in selected:
        if not s: continue
        keyp = str(s)
        if keyp in seen: continue
        seen.add(keyp)
        dedup.append(s)
    selected = dedup
    
    if not selected:
        raise RuntimeError("No stems available")
    
    # -------------------------------------------------------------------------
    # 3) SMART LOOP STEMS (STEREO PRESERVED)
    # -------------------------------------------------------------------------
    print(f"   🔄 Looping stems to {total_sec}s with crossfades...")
    loops: List[Path] = []
    
    for i, stem in enumerate(selected):
        if not stem or not stem.exists(): continue
        out_wav = TMP / f"{date}_{safe_id}_{i}_loop.wav"
        
        try:
            data = _safe_read_wav(stem) # (N, 2)
            target_samples = int(total_sec * SAMPLE_RATE)
            
            if data.shape[0] >= target_samples:
                final = data[:target_samples, :]
            else:
                crossfade_samples = int(0.05 * SAMPLE_RATE)
                if data.shape[0] <= crossfade_samples * 2: crossfade_samples = 0
                num_loops = int(np.ceil(target_samples / data.shape[0])) + 1
                
                looped = []
                for loop_idx in range(num_loops):
                    if loop_idx == 0:
                        looped.append(data)
                    else:
                        if crossfade_samples > 0:
                            fade_out = np.linspace(1, 0, crossfade_samples)
                            fade_in = np.linspace(0, 1, crossfade_samples)
                            prev_end = looped[-1][-crossfade_samples:, :]
                            curr_start = data[:crossfade_samples, :]
                            # Stereo Crossfade
                            crossfaded = prev_end * fade_out[:, None] + curr_start * fade_in[:, None]
                            looped[-1] = looped[-1][:-crossfade_samples, :]
                            looped.append(np.concatenate([crossfaded, data[crossfade_samples:, :]], axis=0))
                        else:
                            looped.append(data)
                
                final = np.concatenate(looped, axis=0)[:target_samples, :]
            
            _write_wav(out_wav, final)
            loops.append(out_wav)
            
        except Exception as e:
            print(f"   ⚠️  Looping error: {e}, falling back to ffmpeg")
            ffmpeg_loop_to_duration(stem, out_wav, total_sec)
            loops.append(out_wav)
    
    # -------------------------------------------------------------------------
    # 4) PROFESSIONAL MIXDOWN (STEREO)
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
    
    final_wav = normed
    
    # -------------------------------------------------------------------------
    # 6) HYBRID BLEND (Optional)
    # -------------------------------------------------------------------------
    if engine_mode == "hybrid" and focus_audio_path is not None:
        hybrid_wav = TMP / f"{date}_{safe_id}_hybrid.wav"
        fm = float(np.clip(focus_mix, 0.0, 1.0))
        mm = 1.0 - fm
        af = f"[0:a]volume={mm}[m];[1:a]volume={fm}[f];[m][f]amix=inputs=2:normalize=0:dropout_transition=0"
        cmd = ["ffmpeg", "-y", "-i", str(normed), "-i", str(focus_audio_path), "-filter_complex", af, str(hybrid_wav)]
        _run(cmd)
        
        hybrid_norm = TMP / f"{date}_{safe_id}_hybrid_norm.wav"
        ffmpeg_loudnorm(hybrid_wav, hybrid_norm, target_lufs=target_lufs)
        final_wav = hybrid_norm
    
    # -------------------------------------------------------------------------
    # 7) EXPORT MP3 (320kbps)
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
        "layers": enabled_layers,
        "seed": seed_str,
    }
    
    return mp3, entry

# =============================================================================
# CLI
# =============================================================================

def main() -> None:
    ap = argparse.ArgumentParser(
        description="SoundFlow Professional DJ Engine v6.1 (Stereo)",
    )
    ap.add_argument("--date", required=True, help="Generation date (YYYY-MM-DD)")
    ap.add_argument("--duration-sec", type=int, default=180, help="Track duration in seconds")
    ap.add_argument("--upload", action="store_true", help="Upload to R2 storage and update catalog")
    ap.add_argument("--json", type=str, help="JSON recipe file (legacy mode)")
    ap.add_argument(
        "--use-daily-plan",
        action="store_true",
        default=True,
        help="Use daily plan (default)",
    )

    args = ap.parse_args()
    bucket = os.environ.get("R2_BUCKET")

    print("=" * 70)
    print("🎵 SoundFlow Professional DJ Engine v6.1")
    print("=" * 70)

    # -------------------------------------------------------------------------
    # LEGACY MODE: Custom JSON
    # -------------------------------------------------------------------------
    if args.json:
        p = Path(args.json)
        if not p.exists():
            raise RuntimeError(f"JSON file not found: {args.json}")
        data = json.loads(p.read_text(encoding="utf-8"))
        combos = data.get("combinations", [])

        for i, combo in enumerate(combos, 1):
            mp3, _ = build_track(args.date, combo, args.duration_sec)
            if args.upload and bucket:
                key = f"audio/free/{args.date}/{mp3.name}"
                upload_file(mp3, bucket, key, public=True)
                print(f"   ☁️  Uploaded to R2: {key}")
        return

    # -------------------------------------------------------------------------
    # DAILY PLAN MODE: Generate for all site categories
    # -------------------------------------------------------------------------
    if args.use_daily_plan:
        print("\n📋 Loading daily generation plan...")
        plan = load_daily_plan()
        tracks = plan.get("tracks", [])

        if not tracks:
            raise RuntimeError("No tracks defined in free_daily_plan.yaml")

        print(f"   📦 Found {len(tracks)} categories to generate\n")

        # Import catalog utilities
        from common.catalog_write import (
            get_catalog_paths,
            read_catalog,
            upsert_tracks,
            write_catalog,
        )

        new_entries = []

        for i, track_config in enumerate(tracks, 1):
            # Apply genre rotation based on day of week
            track_config = apply_genre_rotation(track_config, args.date)

            # Override duration if specified
            if args.duration_sec:
                track_config["duration_sec"] = args.duration_sec

            # Generate track
            mp3, metadata = build_track(
                args.date,
                track_config,
                track_config.get("duration_sec", args.duration_sec),
            )

            # Build catalog entry
            track_id = track_config.get("id", f"track_{i}")
            site_category = track_config.get("siteCategory", "General")

            entry = {
                "id": f"free-{args.date}-{track_id}",
                "title": f"{site_category} Daily - {args.date}",
                "tier": "free",
                "date": args.date,
                "category": site_category,
                "genre": track_config.get("genre", "electronic"),
                "bpm": track_config.get("bpm", 120),
                "key": track_config.get("key", "C"),
                "durationSec": metadata.get("duration_sec", args.duration_sec),
                "goalTags": track_config.get("goalTags", []),
                "natureTags": track_config.get("natureTags", []),
                "energyMin": int(track_config.get("energyMin", 0)),
                "energyMax": int(track_config.get("energyMax", 100)),
                "ambienceMin": int(track_config.get("ambienceMin", 0)),
                "ambienceMax": int(track_config.get("ambienceMax", 100)),
                "objectKey": f"audio/free/{args.date}/{mp3.name}",
            }

            # Upload to R2
            if args.upload and bucket:
                upload_file(mp3, bucket, entry["objectKey"], public=True)
                print(f"   ☁️  Uploaded to R2: {entry['objectKey']}")

            new_entries.append(entry)
            print(f"   ✅ Completed: {entry['id']}\n")

        # Update free catalog
        if args.upload and bucket:
            print("\n📚 Updating free catalog...")
            paths = get_catalog_paths()

            existing = read_catalog(bucket, paths.free_key)
            merged = upsert_tracks(existing, new_entries)
            write_catalog(bucket, paths.free_key, merged)

            print(f"   ✅ Updated: s3://{bucket}/{paths.free_key}")
            print(f"   📊 Total free tracks: {len(merged)}")
            print(f"   🆕 New tracks: {len(new_entries)}")

            print(
                "\nℹ️  Run 'python -m common.build_general_catalog --bucket "
                f"{bucket} --upload' to rebuild general index"
            )
        else:
            print(
                f"\nℹ️  Generated {len(new_entries)} tracks locally. "
                "Run with --upload to push to R2 and update catalog."
            )

        return

    # -------------------------------------------------------------------------
    # FALLBACK MODE: Use presets (legacy)
    # -------------------------------------------------------------------------
    print("\n📋 Loading presets (legacy mode)...")
    presets = load_presets().get("presets", [])
    if not presets:
        presets = [
            {
                "id": "default-house",
                "title": "House Session",
                "genre": "House",
                "category": "music",
                "stem_length": "medium",
                "energy_curve": "peak",
            }
        ]

    for i, preset in enumerate(presets, 1):
        mp3, _ = build_track(args.date, preset, args.duration_sec)
        if args.upload and bucket:
            key = f"audio/free/{args.date}/{mp3.name}"
            upload_file(mp3, bucket, key, public=True)
            print(f"   ☁️  Uploaded to R2: {key}")

if __name__ == "__main__":
    main()