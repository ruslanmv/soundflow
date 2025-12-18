# generator/free/remix_daily.py
from __future__ import annotations

import argparse
import os
import random
import yaml
from pathlib import Path

# Import Generators
from free.music_engine import (
    generate_techno_kick,
    generate_techno_bass,
    generate_techno_arp,
    generate_lofi_drums,
    generate_lofi_keys,
    generate_texture,
)

from common.audio_utils import (
    require_ffmpeg,
    ffmpeg_loop_to_duration,
    ffmpeg_mix,
    ffmpeg_fade,
    ffmpeg_loudnorm,
    ffmpeg_encode_mp3,
)
from common.r2_upload import upload_file
from common.catalog_write import read_catalog, write_catalog, upsert_tracks, get_catalog_paths

ROOT = Path(__file__).resolve().parent
ASSETS = ROOT / "assets"
STEMS_DIR = ASSETS / "stems"
TMP = Path(".soundflow_tmp/free")
OUT = Path(".soundflow_out/free")

def load_presets() -> dict:
    p = Path(__file__).resolve().parents[1] / "prompts" / "free_presets.yaml"
    return yaml.safe_load(p.read_text(encoding="utf-8"))

def ensure_procedural_library(date_seed: str):
    """
    Generates a library of variants for each sound type.
    We create _v1, _v2, etc. so the remixer has options.
    """
    STEMS_DIR.mkdir(parents=True, exist_ok=True)
    
    # -----------------------
    # TECHNO LIBRARY (128 BPM)
    # -----------------------
    # Generate 2 Variations of Kicks
    for v in [1, 2]:
        path = STEMS_DIR / f"kick_techno_v{v}.wav"
        if not path.exists(): generate_techno_kick(path, bpm=128, variant=v)

    # Generate 2 Variations of Bass
    for v in [1, 2]:
        path = STEMS_DIR / f"bass_techno_v{v}.wav"
        if not path.exists(): generate_techno_bass(path, bpm=128, key_freq=49.0, variant=v)
        
    # Generate 2 Variations of Arps (Different scales/patterns)
    for v in [1, 2]:
        path = STEMS_DIR / f"arp_techno_v{v}.wav"
        if not path.exists(): generate_techno_arp(path, bpm=128, key_freq=196.0, variant=v)

    # -----------------------
    # LO-FI LIBRARY (85 BPM)
    # -----------------------
    # Generate 2 Variations of Drums
    for v in [1, 2]:
        path = STEMS_DIR / f"drums_lofi_v{v}.wav"
        if not path.exists(): generate_lofi_drums(path, bpm=85, variant=v)
        
    # Generate 2 Variations of Keys
    for v in [1, 2]:
        path = STEMS_DIR / f"keys_lofi_v{v}.wav"
        if not path.exists(): generate_lofi_keys(path, bpm=85, key_freq=261.63, variant=v)
        
    # -----------------------
    # AMBIENT LIBRARY
    # -----------------------
    if not (STEMS_DIR / "texture_vinyl.wav").exists():
        generate_texture(STEMS_DIR / "texture_vinyl.wav", type='vinyl')
    if not (STEMS_DIR / "texture_rain.wav").exists():
        generate_texture(STEMS_DIR / "texture_rain.wav", type='rain')


def get_random_variant(prefix: str, rnd: random.Random) -> Path:
    """Finds all files matching prefix and picks one."""
    candidates = list(STEMS_DIR.glob(f"{prefix}*.wav"))
    if not candidates:
        raise RuntimeError(f"Missing stems for {prefix}. Run ensure_procedural_library first.")
    candidates.sort()
    return rnd.choice(candidates)

def build_track(date: str, preset: dict, total_sec: int) -> tuple[Path, dict]:
    preset_id = preset["id"]
    rnd = random.Random(f"{date}:{preset_id}") # Deterministic per day/preset

    require_ffmpeg()
    TMP.mkdir(parents=True, exist_ok=True)
    OUT.mkdir(parents=True, exist_ok=True)

    ensure_procedural_library(date)

    selected: list[Path] = []
    
    # ---------------------------------------------------------
    # DYNAMIC LOGIC: Pick Random Variants from Library
    # ---------------------------------------------------------
    
    if "deep_work" in preset_id:
        print(f"🎵 Building TECHNO track for {preset_id}")
        # Mix and Match: Maybe Kick v1 + Bass v2 this time?
        selected.append(get_random_variant("kick_techno", rnd))
        selected.append(get_random_variant("bass_techno", rnd))
        selected.append(get_random_variant("arp_techno", rnd))
        
    elif "study" in preset_id:
        print(f"🎵 Building LO-FI track for {preset_id}")
        selected.append(get_random_variant("drums_lofi", rnd))
        selected.append(get_random_variant("keys_lofi", rnd))
        selected.append(get_random_variant("texture_vinyl", rnd))
        
    elif "relax" in preset_id or "nature" in preset_id:
        print(f"🎵 Building AMBIENT track for {preset_id}")
        selected.append(get_random_variant("keys_lofi", rnd)) # Reuse keys for chill
        selected.append(get_random_variant("texture_rain", rnd))
        
    else:
        print(f"⚠️ Unknown preset type {preset_id}, using generic fallback.")
        selected.append(get_random_variant("texture_vinyl", rnd))

    # Mix
    looped_paths: list[Path] = []
    for i, stem in enumerate(selected):
        out_wav = TMP / f"{date}_{preset_id}_{i}_loop.wav"
        ffmpeg_loop_to_duration(stem, out_wav, total_sec)
        looped_paths.append(out_wav)

    mixed = TMP / f"{date}_{preset_id}_mix.wav"
    ffmpeg_mix(looped_paths, mixed)

    faded = TMP / f"{date}_{preset_id}_fade.wav"
    ffmpeg_fade(mixed, faded, fade_in_ms=1500, fade_out_ms=3000, total_sec=total_sec)

    normed = TMP / f"{date}_{preset_id}_norm.wav"
    ffmpeg_loudnorm(faded, normed, target_lufs=-14.0)

    mp3 = OUT / f"free-{date}-{preset_id}.mp3"
    ffmpeg_encode_mp3(normed, mp3, bitrate="192k")

    # Catalog Entry
    entry = {
        "id": f"free-{date}-{preset_id}",
        "title": preset["title"],
        "tier": "free",
        "date": date,
        "category": preset["category"],
        "durationSec": total_sec,
        "url": None,
    }
    return mp3, entry

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", required=True)
    ap.add_argument("--duration-sec", type=int, default=120)
    ap.add_argument("--upload", action="store_true")
    args = ap.parse_args()
    
    bucket = os.environ.get("R2_BUCKET")
    data = load_presets()
    presets = data.get("presets", [])
    
    for p in presets:
        try:
            mp3, entry = build_track(args.date, p, args.duration_sec)
            print(f"✅ Generated: {mp3}")
            if args.upload and bucket:
                key = f"audio/free/{args.date}/{mp3.name}"
                upload_file(mp3, bucket, key, public=True)
        except Exception as e:
            print(f"❌ Error {p['id']}: {e}")

if __name__ == "__main__":
    main()