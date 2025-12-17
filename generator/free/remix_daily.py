from __future__ import annotations

import argparse
import os
import random
from pathlib import Path
import yaml

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


def choose_stem(tag: str) -> Path | None:
    """
    Stems must be named like:
      drums_clean_01.wav
      pad_warm_02.wav
      ambience_rain_01.wav
    We'll match by prefix tag_*
    """
    candidates = list(STEMS_DIR.glob(f"{tag}_*.wav"))
    if not candidates:
        return None
    candidates.sort()
    return random.choice(candidates)


def build_track(date: str, preset: dict, total_sec: int) -> tuple[Path, dict]:
    """
    Creates a free track by selecting stems and mixing.
    Returns mp3 path + catalog entry.
    """
    preset_id = preset["id"]
    title = preset["title"]
    category = preset["category"]

    # deterministic-ish randomness per day+preset so daily is stable
    rnd = random.Random(f"{date}:{preset_id}")

    require_ffmpeg()
    TMP.mkdir(parents=True, exist_ok=True)
    OUT.mkdir(parents=True, exist_ok=True)

    stem_rules = preset.get("stem_rules", {})
    selected: list[Path] = []

    # Select stems
    for channel, tags in stem_rules.items():
        tags = tags or []
        # choose one tag from candidates
        if not tags:
            continue
        tag = rnd.choice(tags)
        stem = choose_stem(tag)
        if stem:
            selected.append(stem)

    if not selected:
        raise RuntimeError(f"No stems found for preset {preset_id}. Add files to {STEMS_DIR}")

    # Loop each stem to full duration (WAV)
    looped_paths: list[Path] = []
    for i, stem in enumerate(selected):
        out_wav = TMP / f"{date}_{preset_id}_{i}_loop.wav"
        ffmpeg_loop_to_duration(stem, out_wav, total_sec)
        looped_paths.append(out_wav)

    # Mixdown
    mixed = TMP / f"{date}_{preset_id}_mix.wav"
    ffmpeg_mix(looped_paths, mixed)

    # Fade
    fade_in_ms = int(preset.get("fade_in_ms", 1200))
    fade_out_ms = int(preset.get("fade_out_ms", 2500))
    faded = TMP / f"{date}_{preset_id}_fade.wav"
    ffmpeg_fade(mixed, faded, fade_in_ms=fade_in_ms, fade_out_ms=fade_out_ms, total_sec=total_sec)

    # Loudness normalize
    target_lufs = float(preset.get("target_lufs", -14))
    normed = TMP / f"{date}_{preset_id}_norm.wav"
    ffmpeg_loudnorm(faded, normed, target_lufs=target_lufs)

    # Encode MP3
    bitrate = preset.get("export_bitrate", "192k")
    mp3 = OUT / f"free-{date}-{preset_id}.mp3"
    ffmpeg_encode_mp3(normed, mp3, bitrate=bitrate)

    entry = {
        "id": f"free-{date}-{preset_id}",
        "title": title,
        "tier": "free",
        "date": date,
        "category": category,
        "durationSec": total_sec,
        "goalTags": preset.get("goalTags", []),
        "natureTags": preset.get("natureTags", []),
        "energyMin": int(preset.get("energyMin", 0)),
        "energyMax": int(preset.get("energyMax", 100)),
        "ambienceMin": int(preset.get("ambienceMin", 0)),
        "ambienceMax": int(preset.get("ambienceMax", 100)),
        # url is filled later after upload (or use public base URL)
        "url": None,
    }
    return mp3, entry


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", required=True, help="YYYY-MM-DD")
    ap.add_argument("--duration-sec", type=int, default=3600, help="Free track duration (sec)")
    ap.add_argument("--upload", action="store_true", help="Upload to R2 and update catalog")
    args = ap.parse_args()

    date = args.date
    total_sec = args.duration_sec

    data = load_presets()
    defaults = data.get("defaults", {})
    presets = data.get("presets", [])

    bucket = os.environ.get("R2_BUCKET")
    if args.upload and not bucket:
        raise RuntimeError("R2_BUCKET env var missing for upload mode.")

    free_base = os.getenv("FREE_PUBLIC_BASE_URL", "").rstrip("/")

    new_entries: list[dict] = []

    for p in presets:
        merged = dict(defaults)
        merged.update(p)
        mp3, entry = build_track(date, merged, total_sec=total_sec)

        if args.upload:
            key = f"audio/free/{date}/{mp3.name}"
            upload_file(mp3, bucket=bucket, key=key, public=True)

            if free_base:
                entry["url"] = f"{free_base}/{key}"
            else:
                # if you use R2 public bucket / custom domain, put it here
                entry["url"] = f"/{key}"

        new_entries.append(entry)
        print("✅ Built free track:", entry["id"], mp3)

    if args.upload:
        paths = get_catalog_paths()
        existing = read_catalog(bucket, paths.free_key)
        merged = upsert_tracks(existing, new_entries)
        write_catalog(bucket, paths.free_key, merged)
        print(f"✅ Updated catalog: s3://{bucket}/{paths.free_key} (count={len(merged)})")
    else:
        print("ℹ️ Run with --upload to push to R2 and update catalog.")


if __name__ == "__main__":
    main()
