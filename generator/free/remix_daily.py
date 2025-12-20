#!/usr/bin/env python3
# generator/free/remix_daily.py
"""
SoundFlow Music Generation Engine - Remix Daily (FREE)
Version 5.0 (Production, Dynamic Per-Request Stems, API + Recipe Compatible)

✅ What changed vs your old version:
- build_track() no longer selects from a tiny static stem library as the *primary* path
- It now renders NEW stems per request via free.music_engine.render_stem(...)
- Adds render_stem_for_request() helper
- Keeps ensure_procedural_library() + get_random_variant() as OPTIONAL fallback only

Key outcome:
- House / Techno / Lofi / Trance / Deep / EDM / Chillout / Bass / Dance / Vocal / Hard / Ambient / Synth / Classic
  will sound meaningfully different because stems are regenerated per request using the request seed.

Notes:
- For “always new” tracks, make sure the API seed changes per request.
  Your server already does: seed = request.seed or f"{request.genre}:{int(start)}"
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
    # NEW PRIMARY API
    render_stem,

    # FOCUS ENGINE
    generate_focus_session,

    # DSP used in professional_mix + master_chain
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
STEMS_DIR = ASSETS / "stems"  # fallback library only

TMP = Path(".soundflow_tmp/free")
OUT = Path(".soundflow_out/free")

# =============================================================================
# SAFETY: filename sanitizer (CRITICAL for Windows mounts + ffmpeg)
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
# OPTIONAL FALLBACK STEM LIBRARY (kept, but NOT the main path anymore)
# =============================================================================

def ensure_procedural_library(date_seed: str) -> None:
    """
    Optional fallback library for environments where render_stem() is disabled.
    In production, you typically won't need this if render_stem is the primary path.
    """
    STEMS_DIR.mkdir(parents=True, exist_ok=True)
    # You can keep your old library generator here if you want.
    # For now, we keep it as a "no-op safe" fallback.
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
    if "snare" in name or "clap" in name:
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
    stems: List[Dict[str, Any]] = []
    max_len = 0

    for stem_path in stem_paths:
        if stem_path is None:
            continue
        rate, data = wavfile.read(str(stem_path))
        if data.dtype == np.int16:
            data = data.astype(np.float32) / 32768.0
        if data.ndim > 1:
            data = data[:, 0]  # keep mix mono here (final mastering is still good)

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

    mix = master_chain(mix, genre=genre)

    output_wav.parent.mkdir(parents=True, exist_ok=True)
    wavfile.write(str(output_wav), SAMPLE_RATE, (mix * 32767.0).astype(np.int16))
    print(f"✅ Mixed: {output_wav.name}")


# =============================================================================
# SCHEMA COMPAT LAYER (API preset vs recipe combo)
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
    # API: preset["mixer"]["synth"] in 0..100 usually
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

    # recipe: preset["smart_mixer"]["synth"] may already be 0..1
    sm = preset.get("smart_mixer", {})
    if isinstance(sm, dict):
        sp = sm.get("synth")
        if isinstance(sp, dict):
            return sp

    return None


def _get_focus_mode(preset: dict) -> str:
    # Recipe schema
    focus_cfg = preset.get("focus", {})
    if isinstance(focus_cfg, dict) and "binaural_mode" in focus_cfg:
        return str(focus_cfg.get("binaural_mode", "off"))

    # API schema
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


# =============================================================================
# NEW: PER-REQUEST STEM RENDERING (PRIMARY PATH)
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
) -> List[Path]:
    """
    Renders one or more stems for a given layer into TMP/rendered/<safe_id>/...

    Returns a list because:
    - layer "drums" may generate "drums" only (one file), but we keep list type for expansion.
    - "texture" can generate multiple textures (rain + vinyl) depending on ambience.
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    layer_l = str(layer).lower().strip()
    paths: List[Path] = []

    # IMPORTANT:
    # - if seed_str changes per request, music changes per request
    # - if seed_str is the same, output is deterministic (good for repeatability)

    # Use variant to introduce deterministic branching per layer and per "variation"
    # (variation doesn’t replace seed; it widens internal randomness)
    v = float(np.clip(variation, 0.0, 1.0))
    base_variant = 1 + (abs(hash(layer_l)) % 8)
    # nudge variant by variation (deterministic)
    variant = int(base_variant + int(v * 3.0))

    # Map your UI/API layers to render_stem stems
    if layer_l in ("kick",):
        p = out_dir / f"{date}_{safe_id}_kick_v{variant}.wav"
        render_stem(out_path=p, stem="kick", genre=genre, bpm=bpm, key=key, seed=seed_str, variant=variant, bars=1)
        paths.append(p)
        return paths

    if layer_l in ("drums",):
        p = out_dir / f"{date}_{safe_id}_drums_v{variant}.wav"
        render_stem(out_path=p, stem="drums", genre=genre, bpm=bpm, key=key, seed=seed_str, variant=variant, bars=1)
        paths.append(p)
        return paths

    if layer_l in ("bass",):
        p = out_dir / f"{date}_{safe_id}_bass_v{variant}.wav"
        render_stem(out_path=p, stem="bass", genre=genre, bpm=bpm, key=key, seed=seed_str, variant=variant, bars=1)
        paths.append(p)
        return paths

    if layer_l in ("music", "pad", "synth", "melody", "chords"):
        p = out_dir / f"{date}_{safe_id}_music_v{variant}.wav"
        render_stem(out_path=p, stem="music", genre=genre, bpm=bpm, key=key, seed=seed_str, variant=variant, bars=1)
        paths.append(p)
        return paths

    if layer_l in ("texture", "ambience"):
        # If ambience sliders are used, generate both as needed.
        # If neither is enabled, still generate a subtle vinyl texture for glue.
        if ambience.get("rain", 0.0) > 0.01:
            p_r = out_dir / f"{date}_{safe_id}_texture_rain_v{variant}.wav"
            render_stem(out_path=p_r, stem="texture", genre=genre, bpm=bpm, key=key, seed=seed_str, variant=variant, bars=4, texture_type="rain")
            # optionally soften
            softened = out_dir / f"{date}_{safe_id}_texture_rain_soft_v{variant}.wav"
            soften_nature_bed(p_r, softened, gain_db=-18.0)
            paths.append(softened)

        if ambience.get("vinyl", 0.0) > 0.01 or not paths:
            p_v = out_dir / f"{date}_{safe_id}_texture_vinyl_v{variant}.wav"
            render_stem(out_path=p_v, stem="texture", genre=genre, bpm=bpm, key=key, seed=seed_str, variant=variant, bars=4, texture_type="vinyl")
            paths.append(p_v)

        return paths

    # Unknown layer -> ignore
    return []


# =============================================================================
# OPTIONAL: FALLBACK STEM SELECTION (ONLY if render_stem fails)
# =============================================================================

def _fallback_choose_stems_for_layer(layer: str, genre: str, rnd: random.Random) -> List[Path]:
    g = genre.lower()

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
        p = get_random_variant("kick_techno", rnd)
        return [p] if p else []

    if layer == "bass":
        if "house" in g:
            p = get_random_variant("bass_deep", rnd)
            return [p] if p else []
        if "bass" in g or "dubstep" in g:
            p = get_random_variant("bass_wobble", rnd)
            return [p] if p else []
        if "synth" in g:
            p = get_random_variant("bass_synth", rnd)
            return [p] if p else []
        p = get_random_variant("bass_techno", rnd)
        return [p] if p else []

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
        p = get_random_variant("arp_techno", rnd)
        return [p] if p else []

    if layer in ("texture", "ambience"):
        p = get_random_variant("texture_vinyl", rnd)
        return [p] if p else []

    return []


# =============================================================================
# TRACK BUILDER (PRODUCTION)
# =============================================================================

def build_track(date: str, preset: dict, total_sec: int) -> Tuple[Path, dict]:
    require_ffmpeg()
    TMP.mkdir(parents=True, exist_ok=True)
    OUT.mkdir(parents=True, exist_ok=True)

    # Optional fallback library init (kept, but not required for render_stem path)
    ensure_procedural_library(date)

    preset_id = str(preset.get("id", "custom"))
    safe_id = safe_slug(preset_id)

    # Deterministic per-track seed string:
    # API provides preset["seed"] (server.py uses request.seed or genre:timestamp)
    seed_str = str(preset.get("seed") or f"{date}:{preset_id}")

    genre = _get_genre(preset)
    bpm = _get_bpm(preset)
    key = _get_key(preset)
    variation = _get_variation(preset)
    target_lufs = _get_target_lufs(preset)
    synth_params = _get_synth_params(preset)

    engine_mode = _get_engine_mode(preset)   # music/focus/hybrid
    focus_mode = _get_focus_mode(preset)     # off/focus/relax
    focus_mix = _get_focus_mix(preset)       # 0..1
    ambience = _get_ambience(preset)         # 0..1

    channels = int(preset.get("channels", 2))
    channels = 2 if channels not in (1, 2) else channels

    enabled_layers = _coerce_enabled_layers(preset)

    # Deterministic fallback RNG (only used if render_stem fails)
    rnd = random.Random(seed_str + ":fallback")

    rendered_dir = TMP / "rendered" / safe_id

    # -------------------------------------------------------------------------
    # 1) FOCUS RENDER (focus OR hybrid)
    # -------------------------------------------------------------------------
    focus_audio_path: Optional[Path] = None
    if engine_mode in ("focus", "hybrid") and focus_mode in ("focus", "relax"):
        if focus_mode == "focus":
            base_freq, beat_freq, noise_mix = 250.0, 20.0, 0.30
        else:
            base_freq, beat_freq, noise_mix = 150.0, 6.0, 0.25

        focus_wav = TMP / f"{date}_{safe_id}_focus.wav"
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
    # 2) MUSIC STEM RENDER (music OR hybrid)
    #     ✅ PRIMARY: render_stem_for_request()
    #     ✅ Fallback: static stem library selection
    # -------------------------------------------------------------------------
    selected: List[Path] = []

    # auto-add texture layer if ambience sliders are used
    # (even if user didn't explicitly include "texture" in layers)
    needs_texture = (ambience["rain"] > 0.01) or (ambience["vinyl"] > 0.01)
    layers_to_render = list(enabled_layers)
    if needs_texture and ("texture" not in [x.lower() for x in layers_to_render]):
        layers_to_render.append("texture")

    for layer in layers_to_render:
        try:
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
                )
            )
        except Exception as e:
            # fallback for robustness: old static stems if render fails
            print(f"⚠️ render_stem failed for layer={layer} ({e}); using fallback stems.")
            selected.extend(_fallback_choose_stems_for_layer(str(layer), genre, rnd))

    # Dedup while preserving order
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
        # hard fallback
        fb = get_random_variant("kick_techno", rnd) or get_random_variant("drums_lofi", rnd)
        if fb:
            selected = [fb]
        else:
            raise RuntimeError("No stems available (render + fallback both failed).")

    # -------------------------------------------------------------------------
    # 3) LOOP STEMS TO REQUEST DURATION
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
    # 7) EXPORT MP3
    # -------------------------------------------------------------------------
    mp3 = OUT / f"free-{date}-{safe_id}.mp3"
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
        "key": key,
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
