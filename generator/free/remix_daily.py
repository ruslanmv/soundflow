# generator/free/remix_daily.py
from __future__ import annotations

import argparse
import os
import random
import shutil
import yaml
import json
from pathlib import Path

# NEW: Import ALL generators mapping to the high-fidelity ones
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
)

from common.audio_utils import (
    require_ffmpeg,
    ffmpeg_loop_to_duration,
    ffmpeg_mix,
    ffmpeg_fade,
    ffmpeg_loudnorm,
    ffmpeg_encode_mp3,
    _run,  # Used to call ffmpeg filter for nature bed softening
)
from common.r2_upload import upload_file

ROOT = Path(__file__).resolve().parent
ASSETS = ROOT / "assets"
STEMS_DIR = ASSETS / "stems"
TMP = Path(".soundflow_tmp/free")
OUT = Path(".soundflow_out/free")


def load_presets() -> dict:
    p = Path(__file__).resolve().parents[1] / "prompts" / "free_presets.yaml"
    return yaml.safe_load(p.read_text(encoding="utf-8"))


def ensure_procedural_library(date_seed: str):
    """Generates the base library of assets for all genres."""
    STEMS_DIR.mkdir(parents=True, exist_ok=True)

    # 1. TECHNO / TRANCE (128-138 BPM)
    for v in [1, 2]:
        if not (STEMS_DIR / f"kick_techno_v{v}.wav").exists():
            generate_techno_kick(STEMS_DIR / f"kick_techno_v{v}.wav", bpm=130, variant=v)
        if not (STEMS_DIR / f"bass_techno_v{v}.wav").exists():
            generate_techno_bass(STEMS_DIR / f"bass_techno_v{v}.wav", bpm=130, variant=v)
        if not (STEMS_DIR / f"arp_techno_v{v}.wav").exists():
            generate_techno_arp(STEMS_DIR / f"arp_techno_v{v}.wav", bpm=130, variant=v)

    # 2. HOUSE / DEEP (124 BPM)
    for v in [1, 2]:
        if not (STEMS_DIR / f"drums_house_v{v}.wav").exists():
            generate_house_drums(STEMS_DIR / f"drums_house_v{v}.wav", bpm=124, variant=v)
        if not (STEMS_DIR / f"bass_deep_v{v}.wav").exists():
            generate_deep_house_bass(STEMS_DIR / f"bass_deep_v{v}.wav", bpm=124, variant=v)
    if not (STEMS_DIR / "chords_house_stab.wav").exists():
        generate_house_chords(STEMS_DIR / "chords_house_stab.wav", bpm=124)

    # 3. LO-FI / CHILL (85 BPM)
    for v in [1, 2]:
        if not (STEMS_DIR / f"drums_lofi_v{v}.wav").exists():
            generate_lofi_drums(STEMS_DIR / f"drums_lofi_v{v}.wav", bpm=85, variant=v)
        if not (STEMS_DIR / f"keys_lofi_v{v}.wav").exists():
            generate_lofi_keys(STEMS_DIR / f"keys_lofi_v{v}.wav", bpm=85, variant=v)

    # 4. BASS / DUBSTEP (140 BPM)
    if not (STEMS_DIR / "bass_wobble_v1.wav").exists():
        generate_wobble_bass(STEMS_DIR / "bass_wobble_v1.wav", bpm=140)

    # 5. HARDSTYLE (150 BPM)
    if not (STEMS_DIR / "kick_hard_gong.wav").exists():
        generate_hard_kick(STEMS_DIR / "kick_hard_gong.wav", bpm=150)

    # 6. SYNTHWAVE (105 BPM)
    if not (STEMS_DIR / "bass_synth_roll.wav").exists():
        generate_synth_bass(STEMS_DIR / "bass_synth_roll.wav", bpm=105)
    if not (STEMS_DIR / "snare_gated_80s.wav").exists():
        generate_gated_snare(STEMS_DIR / "snare_gated_80s.wav", bpm=105)

    # 7. EURO / DANCE (140 BPM)
    if not (STEMS_DIR / "piano_rave_m1.wav").exists():
        generate_rave_piano(STEMS_DIR / "piano_rave_m1.wav", bpm=140)

    # TEXTURES
    if not (STEMS_DIR / "texture_vinyl.wav").exists():
        generate_texture(STEMS_DIR / "texture_vinyl.wav", type="vinyl")
    if not (STEMS_DIR / "texture_rain.wav").exists():
        generate_texture(STEMS_DIR / "texture_rain.wav", type="rain")


def get_random_variant(prefix: str, rnd: random.Random) -> Path | None:
    """Helper to pick v1 or v2 randomly."""
    candidates = list(STEMS_DIR.glob(f"{prefix}*.wav"))
    if not candidates:
        candidates = list(STEMS_DIR.glob(f"*{prefix}*.wav"))
    if not candidates:
        # Don't spam warning if it's just an optional layer missing, but nice to know
        # print(f"⚠️ Warning: No stems found for {prefix}")
        return None
    candidates.sort()
    return rnd.choice(candidates)


def soften_nature_bed(in_wav: Path, out_wav: Path, *, gain_db: float = -16.0) -> None:
    """
    Reduce volume + make nature bed more natural (less harsh/strong):
      - lower level (default -16 dB)
      - remove rumble (highpass)
      - remove harsh hiss (lowpass)
      - subtle slow drift (tremolo) so it's less static
    """
    require_ffmpeg()
    out_wav.parent.mkdir(parents=True, exist_ok=True)

    # If ffmpeg isn't available for any reason, fall back to copy
    if shutil.which("ffmpeg") is None:
        out_wav.write_bytes(in_wav.read_bytes())
        return

    tremolo_freq = 0.12   # Hz (slow)
    tremolo_depth = 0.12  # subtle

    af = (
        f"volume={gain_db}dB,"
        f"highpass=f=120,"
        f"lowpass=f=6000,"
        f"tremolo=f={tremolo_freq}:d={tremolo_depth}"
    )

    cmd = ["ffmpeg", "-y", "-i", str(in_wav), "-af", af, str(out_wav)]
    _run(cmd)


def build_track(date: str, preset: dict, total_sec: int) -> tuple[Path, dict]:
    preset_id = preset.get("id", "custom")
    rnd = random.Random(f"{date}:{preset_id}")

    require_ffmpeg()
    TMP.mkdir(parents=True, exist_ok=True)
    OUT.mkdir(parents=True, exist_ok=True)

    ensure_procedural_library(date)

    selected: list[Path] = []

    # ---------------------------------------------------------
    # SELECTION LOGIC
    # ---------------------------------------------------------
    
    # CASE 1: UI Payload (JSON) -> Explicit layers
    if "layers" in preset and "enabled" in preset["layers"]:
        enabled_layers = preset["layers"]["enabled"] # List like ["drums", "bass"]
        genre = preset.get("genre", "Trance")
        
        # Map generic UI layer names to specific Genre Stems
        for layer in enabled_layers:
            if layer == "drums":
                if "House" in genre or "Deep" in genre: selected.append(get_random_variant("drums_house", rnd))
                elif "Lofi" in genre or "Chill" in genre: selected.append(get_random_variant("drums_lofi", rnd))
                elif "Hard" in genre: selected.append(get_random_variant("kick_hard", rnd))
                else: selected.append(get_random_variant("kick_techno", rnd)) # Default Techno
            
            elif layer == "bass":
                if "House" in genre or "Deep" in genre: selected.append(get_random_variant("bass_deep", rnd))
                elif "Bass" in genre or "Dubstep" in genre: selected.append(get_random_variant("bass_wobble", rnd))
                elif "Synth" in genre: selected.append(get_random_variant("bass_synth", rnd))
                else: selected.append(get_random_variant("bass_techno", rnd))
            
            elif layer == "music" or layer == "pad":
                if "House" in genre: selected.append(get_random_variant("chords_house", rnd))
                elif "Lofi" in genre: selected.append(get_random_variant("keys_lofi", rnd))
                elif "Euro" in genre or "Dance" in genre: selected.append(get_random_variant("piano_rave", rnd))
                else: selected.append(get_random_variant("arp_techno", rnd))
            
            elif layer == "texture" or layer == "ambience":
                tex_type = preset.get("texture", "none")
                if "rain" in tex_type.lower():
                    rain = get_random_variant("texture_rain", rnd)
                    if rain:
                        softened = TMP / f"{date}_{preset_id}_rain_soft.wav"
                        soften_nature_bed(rain, softened, gain_db=-16.0)
                        selected.append(softened)
                elif "vinyl" in tex_type.lower():
                    selected.append(get_random_variant("texture_vinyl", rnd))

    # CASE 2: Standard Presets (Daily Runner)
    else:
        if "deep_work" in preset_id or "techno" in preset_id:
            selected.append(get_random_variant("kick_techno", rnd))
            selected.append(get_random_variant("bass_techno", rnd))
            selected.append(get_random_variant("arp_techno", rnd))

        elif "study" in preset_id or "chill" in preset_id:
            selected.append(get_random_variant("drums_lofi", rnd))
            selected.append(get_random_variant("keys_lofi", rnd))
            selected.append(get_random_variant("texture_vinyl", rnd))

        elif "relax" in preset_id or "nature" in preset_id or "ambient" in preset_id:
            selected.append(get_random_variant("keys_lofi", rnd))
            rain = get_random_variant("texture_rain", rnd)
            if rain is not None:
                softened = TMP / f"{date}_{preset_id}_rain_soft.wav"
                soften_nature_bed(rain, softened, gain_db=-16.0)
                selected.append(softened)

        elif "house" in preset_id or "deep" in preset_id:
            selected.append(get_random_variant("drums_house", rnd))
            selected.append(get_random_variant("bass_deep", rnd))
            selected.append(get_random_variant("chords_house", rnd))

        elif "synth" in preset_id or "retro" in preset_id:
            selected.append(get_random_variant("bass_synth", rnd))
            selected.append(get_random_variant("snare_gated", rnd))

        elif "hard" in preset_id:
            selected.append(get_random_variant("kick_hard", rnd))

        else:
            # Fallback
            selected.append(get_random_variant("texture_vinyl", rnd))

    # Clean up None values
    selected = [s for s in selected if s is not None]

    if not selected:
        raise RuntimeError(f"No stems found for {preset_id}")

    # Mix
    looped_paths: list[Path] = []
    for i, stem in enumerate(selected):
        out_wav = TMP / f"{date}_{preset_id}_{i}_loop.wav"
        ffmpeg_loop_to_duration(stem, out_wav, total_sec)
        looped_paths.append(out_wav)

    mixed = TMP / f"{date}_{preset_id}_mix.wav"
    ffmpeg_mix(looped_paths, mixed)

    # Determine mix params (UI can override)
    fade_in = 1500
    fade_out = 3000
    target_lufs = -14.0
    
    if "mix" in preset:
        target_lufs = float(preset["mix"].get("target_lufs", -14.0))

    faded = TMP / f"{date}_{preset_id}_fade.wav"
    ffmpeg_fade(mixed, faded, fade_in_ms=fade_in, fade_out_ms=fade_out, total_sec=total_sec)

    normed = TMP / f"{date}_{preset_id}_norm.wav"
    ffmpeg_loudnorm(faded, normed, target_lufs=target_lufs)

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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", required=True)
    ap.add_argument("--duration-sec", type=int, default=120)
    ap.add_argument("--upload", action="store_true")
    # NEW: Support for JSON recipes from React UI
    ap.add_argument("--json", type=str, help="Path to UI generated JSON file")
    
    args = ap.parse_args()

    bucket = os.environ.get("R2_BUCKET")
    
    # MODE A: JSON Input (React UI)
    if args.json:
        p = Path(args.json)
        if not p.exists():
            raise RuntimeError(f"JSON file not found: {args.json}")
        
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
            combos = data.get("combinations", [])
            print(f"📂 Processing {len(combos)} combinations from {args.json}...")
            
            for combo in combos:
                try:
                    mp3, entry = build_track(args.date, combo, args.duration_sec)
                    print(f"✅ Generated: {mp3}")
                except Exception as e:
                    print(f"❌ Error {combo.get('id')}: {e}")
        except json.JSONDecodeError:
            print("❌ Invalid JSON file.")

    # MODE B: Standard Daily Presets
    else:
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