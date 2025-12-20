#!/usr/bin/env python3
# generator/free/remix_daily.py
"""
SoundFlow Music Generation Engine - Remix Daily (FREE)
Version 4.2 (API + Recipe Compatible, Deterministic, High-Fidelity Master)

Goals:
- Works from BOTH:
  (A) FastAPI server preset schema (nested: music/mixer/master)
  (B) JSON recipe schema (flat: genre/bpm/layers.enabled/focus.binaural_mode)
- Generates DIFFERENT music per genre/seed/variation (no “all tracks identical”)
- Safe filenames on Windows/WSL (no ':' '/' etc)
- Production-grade export (WAV master -> LUFS -> MP3)
- Focus and Hybrid supported

Notes on “Spotify / DI.FM quality”:
- This pipeline produces clean, normalized masters suitable for distribution.
- Actual platform loudness targets vary; default is -14 LUFS which is common for streaming.
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
    # STEM LIBRARY GENERATORS
    generate_techno_kick,
    generate_techno_bass,
    generate_techno_arp,
    generate_house_drums,
    generate_deep_house_bass,
    generate_house_chords,
    generate_lofi_drums,
    generate_lofi_keys,
    generate_wobble_bass,
    generate_hard_kick,
    generate_synth_bass,
    generate_gated_snare,
    generate_rave_piano,
    generate_texture,
    # FOCUS ENGINE (NEW SIGNATURE)
    generate_focus_session,
    # DSP
    apply_overdrive,
    apply_algorithmic_reverb,
    apply_lowpass,
    apply_resonant_filter,
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
# SAFETY: filename sanitizer (CRITICAL for Windows mounts + ffmpeg)
# =============================================================================

_SAFE_CHARS_RE = re.compile(r"[^a-zA-Z0-9._-]+")


def safe_slug(s: str, max_len: int = 120) -> str:
    """
    Make a filesystem- and ffmpeg-safe filename fragment.
    Removes ':', '/', '\\', spaces, etc.
    """
    s = (s or "").strip()
    s = _SAFE_CHARS_RE.sub("_", s)
    s = s.strip("._-")
    if not s:
        s = "track"
    return s[:max_len]


def assert_audio_not_empty(path: Path, min_bytes: int = 20_000) -> None:
    """
    Fail fast if output is missing or suspiciously small.
    Helps catch ffmpeg failures and silent renders.
    """
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
# STEM LIBRARY
# =============================================================================

def ensure_procedural_library(date_seed: str) -> None:
    """
    Generates a small reusable stem library. Deterministic enough for repeatable
    smoke tests, but still produces different final tracks via:
    - random variant selection (seeded per track)
    - smart-mixer parameters
    - genre routing
    """
    STEMS_DIR.mkdir(parents=True, exist_ok=True)
    print("🎛️  Generating stem library…")

    # Techno (130 BPM)
    for v in (1, 2):
        if not (STEMS_DIR / f"kick_techno_v{v}.wav").exists():
            generate_techno_kick(STEMS_DIR / f"kick_techno_v{v}.wav", bpm=130, variant=v)
        if not (STEMS_DIR / f"bass_techno_v{v}.wav").exists():
            generate_techno_bass(STEMS_DIR / f"bass_techno_v{v}.wav", bpm=130, variant=v)
        if not (STEMS_DIR / f"arp_techno_v{v}.wav").exists():
            generate_techno_arp(STEMS_DIR / f"arp_techno_v{v}.wav", bpm=130, variant=v)

    # House (124 BPM)
    for v in (1, 2):
        if not (STEMS_DIR / f"drums_house_v{v}.wav").exists():
            generate_house_drums(STEMS_DIR / f"drums_house_v{v}.wav", bpm=124, variant=v)
        if not (STEMS_DIR / f"bass_deep_v{v}.wav").exists():
            generate_deep_house_bass(STEMS_DIR / f"bass_deep_v{v}.wav", bpm=124, variant=v)

    if not (STEMS_DIR / "chords_house_stab.wav").exists():
        generate_house_chords(STEMS_DIR / "chords_house_stab.wav", bpm=124)

    # Lo-Fi (85 BPM)
    for v in (1, 2):
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


def get_random_variant(prefix: str, rnd: random.Random) -> Optional[Path]:
    candidates = list(STEMS_DIR.glob(f"{prefix}*.wav"))
    if not candidates:
        candidates = list(STEMS_DIR.glob(f"*{prefix}*.wav"))
    if not candidates:
        return None
    candidates.sort()
    return rnd.choice(candidates)


def soften_nature_bed(in_wav: Path, out_wav: Path, gain_db: float = -18.0) -> None:
    """
    Keeps rain/ambience unobtrusive (DI.FM style).
    """
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
# MIXING / MASTERING
# =============================================================================

def detect_stem_type(filename: str) -> str:
    name = filename.lower()
    if "kick" in name:
        return "kick"
    if "bass" in name:
        return "bass"
    if "drum" in name:
        return "drums"
    if "arp" in name:
        return "arp"
    if "synth" in name or "lead" in name:
        return "synth"
    if "chord" in name or "stab" in name or "piano" in name:
        return "chords"
    if "pad" in name or "key" in name:
        return "pad"
    if "texture" in name or "vinyl" in name or "rain" in name:
        return "texture"
    return "other"


def apply_stem_eq(audio: np.ndarray, stem_type: str) -> np.ndarray:
    from free.music_engine import apply_highpass, apply_lowpass, apply_parametric_eq

    if stem_type == "kick":
        audio = apply_highpass(audio, 25)
        audio = apply_parametric_eq(audio, 60, gain_db=2, q=1.2)
        audio = apply_parametric_eq(audio, 300, gain_db=-3, q=2.0)
    elif stem_type == "bass":
        audio = apply_highpass(audio, 35)
        audio = apply_parametric_eq(audio, 55, gain_db=-5, q=1.5)
        audio = apply_lowpass(audio, 4500)
    elif stem_type in ("synth", "arp"):
        audio = apply_highpass(audio, 180)
        audio = apply_parametric_eq(audio, 2800, gain_db=1.5, q=0.8)
    elif stem_type in ("pad", "chords"):
        audio = apply_highpass(audio, 220)
        audio = apply_lowpass(audio, 8000)
    elif stem_type == "texture":
        audio = apply_highpass(audio, 450)
        audio = apply_lowpass(audio, 7500)

    return audio


def master_chain(audio: np.ndarray, genre: str) -> np.ndarray:
    """
    “Broadcast-ready” chain before LUFS normalization.
    LUFS normalization is done by ffmpeg_loudnorm afterwards.
    """
    from free.music_engine import (
        apply_highpass, multiband_process,
        apply_parametric_eq, soft_clip, normalize
    )

    if audio.size < 64:
        return audio.astype(np.float32)

    audio = apply_highpass(audio, 20, order=4)

    # gentle tone shaping
    audio = apply_parametric_eq(audio, 80, gain_db=0.8, q=0.7)
    audio = apply_parametric_eq(audio, 3000, gain_db=0.5, q=0.5)
    audio = apply_parametric_eq(audio, 10000, gain_db=0.3, q=0.4)

    audio = multiband_process(audio, low_gain=1.05, mid_gain=0.98, high_gain=1.02)
    audio = soft_clip(audio * 1.1, threshold=0.85)

    rms = float(np.sqrt(np.mean(audio ** 2))) if audio.size else 0.0
    target_rms = 0.15 if ("techno" in genre.lower() or "hard" in genre.lower()) else 0.12
    if rms > 1e-6:
        audio = audio * (target_rms / rms)

    audio = np.clip(audio, -0.95, 0.95)
    audio = normalize(audio, target=0.97)
    return audio.astype(np.float32)


def professional_mix(
    stem_paths: List[Path],
    output_wav: Path,
    genre: str = "techno",
    synth_params: Optional[dict] = None
) -> None:
    """
    In-Python summing + light DSP, then writes a WAV master.
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
            data = data[:, 0]  # mono

        stems.append({
            "audio": data.astype(np.float32, copy=False),
            "path": stem_path,
            "type": detect_stem_type(stem_path.name),
        })
        max_len = max(max_len, len(data))

    if not stems:
        raise RuntimeError("No stems to mix (stem_paths empty)")

    processed: List[np.ndarray] = []
    for stem in stems:
        audio = stem["audio"]
        st = stem["type"]

        if len(audio) < max_len:
            audio = np.pad(audio, (0, max_len - len(audio)))

        audio = apply_stem_eq(audio, st)

        # gain staging
        gain_map = {
            "kick": 0.90,
            "bass": 0.72,
            "drums": 0.68,
            "synth": 0.52,
            "arp": 0.48,
            "pad": 0.42,
            "chords": 0.45,
            "texture": 0.18,
            "other": 0.50,
        }
        audio = audio * float(gain_map.get(st, 0.50))

        processed.append(audio.astype(np.float32, copy=False))

    mix = np.sum(processed, axis=0).astype(np.float32)

    # smart mixer FX (API or recipe)
    if synth_params:
        drive_amt = float(synth_params.get("drive", 0.0))
        if drive_amt > 0.05:
            mix = apply_overdrive(mix, drive=drive_amt * 4.0)

        cutoff_amt = float(synth_params.get("cutoff", 1.0))
        resonance_amt = float(synth_params.get("resonance", 0.0))
        if cutoff_amt < 0.98:
            cutoff_freq = 100.0 + (20000.0 * (cutoff_amt ** 2))
            if resonance_amt > 0.1:
                mix = apply_resonant_filter(mix, cutoff=cutoff_freq, resonance=resonance_amt * 0.9)
            else:
                mix = apply_lowpass(mix, cutoff=cutoff_freq)

        space_amt = float(synth_params.get("space", 0.0))
        if space_amt > 0.05:
            mix = apply_algorithmic_reverb(mix, room_size=0.3 + (space_amt * 0.6), wet=space_amt * 0.5)

    # master bus
    mix = master_chain(mix, genre=genre)

    output_wav.parent.mkdir(parents=True, exist_ok=True)
    wavfile.write(str(output_wav), SAMPLE_RATE, (mix * 32767.0).astype(np.int16))
    print(f"✅ Mixed: {output_wav.name}")


# =============================================================================
# SCHEMA COMPAT LAYER (API preset vs recipe combo)
# =============================================================================

def _coerce_enabled_layers(preset: dict) -> List[str]:
    """
    Accepts BOTH:
      - API: preset["layers"] = ["drums","bass","music"]
      - recipe: preset["layers"] = {"enabled":[...]}
      - legacy: preset["music"]["layers"] = [...]
    """
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


def _get_variation(preset: dict) -> float:
    # API uses top-level variation; recipes may omit
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
    # API: preset["mixer"]["synth"] in 0..100 usually
    mixer_cfg = preset.get("mixer", {})
    if isinstance(mixer_cfg, dict):
        sp = mixer_cfg.get("synth")
        if isinstance(sp, dict):
            # server sends 0..100; map to 0..1 for our mixer
            return {
                "cutoff": float(sp.get("cutoff", 100.0)) / 100.0,
                "resonance": float(sp.get("resonance", 0.0)) / 100.0,
                "drive": float(sp.get("drive", 0.0)) / 100.0,
                "space": float(sp.get("space", 0.0)) / 100.0,
            }

    # recipe: preset["smart_mixer"]["synth"] already 0..1 sometimes
    sm = preset.get("smart_mixer", {})
    if isinstance(sm, dict):
        sp = sm.get("synth")
        if isinstance(sp, dict):
            return sp

    return None


def _get_focus_mode(preset: dict) -> str:
    """
    Accepts:
      - API: preset["mode"] in ("music","focus","hybrid") and preset["focus"]["mode"] in ("off","focus","relax")
      - recipe: preset["focus"]["binaural_mode"]
    """
    # Recipe first
    focus_cfg = preset.get("focus", {})
    if isinstance(focus_cfg, dict) and "binaural_mode" in focus_cfg:
        return str(focus_cfg.get("binaural_mode", "off"))

    # API
    if isinstance(focus_cfg, dict) and "mode" in focus_cfg:
        return str(focus_cfg.get("mode", "off"))

    return "off"


def _get_engine_mode(preset: dict) -> str:
    # API: preset["mode"] is authoritative
    m = preset.get("mode")
    if isinstance(m, str) and m in ("music", "focus", "hybrid"):
        return m
    # recipe may omit mode; infer from focus binaural_mode
    fm = _get_focus_mode(preset)
    if fm in ("focus", "relax"):
        return "focus"
    return "music"


def _get_focus_mix(preset: dict) -> float:
    # API uses preset["focus"]["mix"] (0..100)
    focus_cfg = preset.get("focus", {})
    if isinstance(focus_cfg, dict) and "mix" in focus_cfg:
        try:
            return float(focus_cfg.get("mix", 30.0)) / 100.0
        except Exception:
            return 0.30
    # recipe may not have this; default
    return 0.30


def _get_ambience(preset: dict) -> Dict[str, float]:
    focus_cfg = preset.get("focus", {})
    amb = {}
    if isinstance(focus_cfg, dict):
        amb = focus_cfg.get("ambience", {}) or {}
    # normalize to 0..1 floats
    def _norm(x: Any) -> float:
        try:
            v = float(x)
        except Exception:
            return 0.0
        # if 0..100
        if v > 1.0:
            v = v / 100.0
        return float(np.clip(v, 0.0, 1.0))

    return {
        "rain": _norm(amb.get("rain", 0.0)),
        "vinyl": _norm(amb.get("vinyl", 0.0)),
        "white": _norm(amb.get("white", 0.0)),
    }


# =============================================================================
# STEM SELECTION (GENRE ROUTING)
# =============================================================================

def _choose_stems_for_layer(layer: str, genre: str, rnd: random.Random) -> List[Path]:
    """
    Returns a list because some layers map to multiple stems for richer mixes.
    """
    g = genre.lower()

    # drums
    if layer == "drums":
        if "house" in g:
            p = get_random_variant("drums_house", rnd)
            return [p] if p else []
        if "lofi" in g or "chill" in g or "study" in g:
            p = get_random_variant("drums_lofi", rnd)
            return [p] if p else []
        if "hard" in g:
            p = get_random_variant("kick_hard", rnd)
            return [p] if p else []
        # default techno kick
        p = get_random_variant("kick_techno", rnd)
        return [p] if p else []

    # bass
    if layer == "bass":
        if "house" in g:
            p = get_random_variant("bass_deep", rnd)
            return [p] if p else []
        if "bass" in g or "dubstep" in g:
            p = get_random_variant("bass_wobble", rnd)
            return [p] if p else []
        if "synth" in g or "synthwave" in g or "retro" in g:
            p = get_random_variant("bass_synth", rnd)
            return [p] if p else []
        p = get_random_variant("bass_techno", rnd)
        return [p] if p else []

    # music/pads/synths
    if layer in ("music", "pad", "synth", "melody"):
        if "house" in g:
            p = get_random_variant("chords_house", rnd)
            return [p] if p else []
        if "lofi" in g or "chill" in g:
            p = get_random_variant("keys_lofi", rnd)
            return [p] if p else []
        if "euro" in g or "rave" in g:
            p = get_random_variant("piano_rave", rnd)
            return [p] if p else []
        # default arp for techno/trance
        p = get_random_variant("arp_techno", rnd)
        return [p] if p else []

    # texture / ambience
    if layer in ("texture", "ambience"):
        # prefer explicit ambience sliders if present
        # (handled earlier in build_track), so here we just provide a fallback:
        p = get_random_variant("texture_vinyl", rnd)
        return [p] if p else []

    return []


# =============================================================================
# TRACK BUILDER (API + Recipe compatible)
# =============================================================================

def build_track(date: str, preset: dict, total_sec: int) -> Tuple[Path, dict]:
    """
    Main entrypoint used by:
    - FastAPI server.py (API requests)
    - tests/test_generator.py (JSON recipe)
    - CLI daily generation
    """
    require_ffmpeg()
    TMP.mkdir(parents=True, exist_ok=True)
    OUT.mkdir(parents=True, exist_ok=True)

    ensure_procedural_library(date)

    preset_id = str(preset.get("id", "custom"))
    safe_id = safe_slug(preset_id)

    # Deterministic RNG:
    # - If preset has explicit seed, use it (API provides request.seed)
    # - Otherwise still stable based on date + id
    seed_str = str(preset.get("seed") or f"{date}:{preset_id}")
    rnd = random.Random(seed_str)

    genre = _get_genre(preset)
    bpm = _get_bpm(preset)
    variation = _get_variation(preset)
    target_lufs = _get_target_lufs(preset)
    synth_params = _get_synth_params(preset)

    engine_mode = _get_engine_mode(preset)          # music/focus/hybrid
    focus_mode = _get_focus_mode(preset)            # off/focus/relax
    focus_mix = _get_focus_mix(preset)              # 0..1
    ambience = _get_ambience(preset)                # 0..1

    channels = int(preset.get("channels", 2))
    channels = 2 if channels not in (1, 2) else channels

    enabled_layers = _coerce_enabled_layers(preset)

    # -------------------------------------------------------------------------
    # 1) FOCUS RENDER (focus OR hybrid)
    # -------------------------------------------------------------------------
    focus_audio_path: Optional[Path] = None
    if engine_mode in ("focus", "hybrid") and focus_mode in ("focus", "relax"):
        # preset mapping for binaural beat speeds
        if focus_mode == "focus":
            base_freq, beat_freq, noise_mix = 250.0, 20.0, 0.30
        else:
            base_freq, beat_freq, noise_mix = 150.0, 6.0, 0.25

        focus_wav = TMP / f"{date}_{safe_id}_focus.wav"

        # generate_focus_session uses NEW signature (no preset_name / add_rain)
        generate_focus_session(
            out_path=focus_wav,
            duration_sec=float(total_sec),
            base_freq=base_freq,
            beat_freq=beat_freq,
            noise_mix=noise_mix,
            rain=float(ambience["rain"]),
            vinyl=float(ambience["vinyl"]),
            white=float(ambience["white"]),
            channels=channels,
        )

        focus_audio_path = focus_wav

        # Focus-only shortcut
        if engine_mode == "focus":
            mp3 = OUT / f"free-{date}-{safe_id}.mp3"
            # normalize focus WAV to LUFS then MP3
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
            }
            return mp3, entry

    # -------------------------------------------------------------------------
    # 2) MUSIC STEM SELECTION (music OR hybrid)
    # -------------------------------------------------------------------------
    selected: List[Path] = []

    # Add textures if user requested via ambience sliders even in music mode
    if ambience["rain"] > 0.01:
        rain = get_random_variant("texture_rain", rnd)
        if rain:
            softened = TMP / f"{date}_{safe_id}_rain_soft.wav"
            soften_nature_bed(rain, softened, gain_db=-18.0)
            selected.append(softened)

    if ambience["vinyl"] > 0.01:
        v = get_random_variant("texture_vinyl", rnd)
        if v:
            selected.append(v)

    # choose per-layer stems
    for layer in enabled_layers:
        selected.extend(_choose_stems_for_layer(layer, genre, rnd))

    # remove None and duplicates while preserving order
    dedup: List[Path] = []
    seen = set()
    for s in selected:
        if not s:
            continue
        key = str(s.resolve())
        if key in seen:
            continue
        seen.add(key)
        dedup.append(s)
    selected = dedup

    # fallback if nothing selected
    if not selected:
        fallback = get_random_variant("kick_techno", rnd) or get_random_variant("drums_lofi", rnd)
        if fallback:
            selected = [fallback]

    # -------------------------------------------------------------------------
    # 3) LOOP STEMS TO DURATION
    # -------------------------------------------------------------------------
    loops: List[Path] = []
    for i, stem in enumerate(selected):
        out_wav = TMP / f"{date}_{safe_id}_{i}_loop.wav"
        ffmpeg_loop_to_duration(stem, out_wav, total_sec)
        loops.append(out_wav)

    # -------------------------------------------------------------------------
    # 4) MIXDOWN (MUSIC WAV MASTER)
    # -------------------------------------------------------------------------
    mixed_wav = TMP / f"{date}_{safe_id}_mixed.wav"
    professional_mix(loops, mixed_wav, genre=genre, synth_params=synth_params)

    # variation modulation: do subtle post-filter so variants differ clearly
    # (without needing more stems)
    if variation > 0.001:
        # deterministic per-track wobble
        v_rng = random.Random(seed_str + ":variation")
        # cutoff between ~5k..14k
        cutoff = 5000.0 + float(v_rng.random()) * (9000.0 * float(np.clip(variation, 0.0, 1.0)))
        sr, x = wavfile.read(str(mixed_wav))
        if x.dtype == np.int16:
            x = x.astype(np.float32) / 32768.0
        if x.ndim > 1:
            x = x[:, 0]
        x = apply_lowpass(x.astype(np.float32, copy=False), cutoff=cutoff)
        wavfile.write(str(mixed_wav), SAMPLE_RATE, (np.clip(x, -1, 1) * 32767).astype(np.int16))

    # -------------------------------------------------------------------------
    # 5) FADE + LOUDNORM
    # -------------------------------------------------------------------------
    faded = TMP / f"{date}_{safe_id}_fade.wav"
    ffmpeg_fade(mixed_wav, faded, fade_in_ms=1500, fade_out_ms=3000, total_sec=total_sec)

    normed = TMP / f"{date}_{safe_id}_norm.wav"
    ffmpeg_loudnorm(faded, normed, target_lufs=target_lufs)

    # -------------------------------------------------------------------------
    # 6) HYBRID BLEND (OPTIONAL)
    # -------------------------------------------------------------------------
    if engine_mode == "hybrid" and focus_audio_path is not None:
        # Mix normed music + focus bed in ffmpeg so it stays sample-accurate.
        hybrid_wav = TMP / f"{date}_{safe_id}_hybrid.wav"

        # ffmpeg filter: scale music and focus then add
        # focus_mix is 0..1 (portion of focus). keep music energy dominant.
        fm = float(np.clip(focus_mix, 0.0, 1.0))
        mm = 1.0 - fm

        # We use amerge+pan for consistency even if focus is stereo.
        af = (
            f"[0:a]volume={mm}[m];"
            f"[1:a]volume={fm}[f];"
            f"[m][f]amix=inputs=2:normalize=0:dropout_transition=0"
        )
        cmd = ["ffmpeg", "-y", "-i", str(normed), "-i", str(focus_audio_path), "-filter_complex", af, str(hybrid_wav)]
        _run(cmd)

        # re-normalize hybrid to target LUFS
        hybrid_norm = TMP / f"{date}_{safe_id}_hybrid_norm.wav"
        ffmpeg_loudnorm(hybrid_wav, hybrid_norm, target_lufs=target_lufs)
        final_wav = hybrid_norm
    else:
        final_wav = normed

    # -------------------------------------------------------------------------
    # 7) EXPORT MP3
    # -------------------------------------------------------------------------
    mp3 = OUT / f"free-{date}-{safe_id}.mp3"
    # Use 320k for higher fidelity (Spotify re-encodes; DI.FM typically prefers good masters)
    ffmpeg_encode_mp3(final_wav, mp3, bitrate="320k")
    assert_audio_not_empty(mp3)

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
        "mode": engine_mode,
        "focus_mode": focus_mode,
        "layers": enabled_layers,
        "seed": seed_str,
    }
    return mp3, entry


# =============================================================================
# CLI
# =============================================================================

def main() -> None:
    ap = argparse.ArgumentParser(description="SoundFlow FREE Remix Engine")
    ap.add_argument("--date", required=True, help="Generation date (YYYY-MM-DD)")
    ap.add_argument("--duration-sec", type=int, default=120, help="Track duration in seconds")
    ap.add_argument("--upload", action="store_true", help="Upload to R2 storage")
    ap.add_argument("--json", type=str, help="JSON recipe file from UI/tests")
    args = ap.parse_args()

    bucket = os.environ.get("R2_BUCKET")

    if args.json:
        p = Path(args.json)
        if not p.exists():
            raise RuntimeError(f"JSON file not found: {args.json}")

        data = json.loads(p.read_text(encoding="utf-8"))
        combos = data.get("combinations", [])
        print(f"📂 Processing {len(combos)} combinations...")

        for combo in combos:
            mp3, _ = build_track(args.date, combo, args.duration_sec)
            print(f"✅ {mp3.name}")

            if args.upload and bucket:
                key = f"audio/free/{args.date}/{mp3.name}"
                upload_file(mp3, bucket, key, public=True)
    else:
        data = load_presets()
        presets = data.get("presets", [])
        print(f"🎵 Generating {len(presets)} tracks for {args.date}...")

        for preset in presets:
            mp3, _ = build_track(args.date, preset, args.duration_sec)
            print(f"✅ {mp3.name}")

            if args.upload and bucket:
                key = f"audio/free/{args.date}/{mp3.name}"
                upload_file(mp3, bucket, key, public=True)


if __name__ == "__main__":
    main()
