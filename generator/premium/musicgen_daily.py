from __future__ import annotations

import argparse
import os
from pathlib import Path
import yaml
import numpy as np

import torch
from transformers import pipeline

from premium.postprocess import postprocess_wav_to_mp3
from common.r2_upload import upload_file
from common.catalog_write import read_catalog, write_catalog, upsert_tracks, get_catalog_paths
from common.audio_utils import require_ffmpeg
import scipy.io.wavfile  # requires scipy in environment


ROOT = Path(__file__).resolve().parent
OUT = Path(".soundflow_out/premium")
TMP = Path(".soundflow_tmp/premium")


def load_prompts() -> dict:
    p = Path(__file__).resolve().parents[1] / "prompts" / "premium_prompts.yaml"
    return yaml.safe_load(p.read_text(encoding="utf-8"))


def generate_chunk(pipe, prompt: str, duration_sec: int, device: int) -> tuple[int, np.ndarray]:
    """
    Generates one chunk as float waveform using transformers pipeline.
    """
    # MusicGen uses max_new_tokens, but we keep it simple with pipeline defaults.
    # For better control, use `MusicgenForConditionalGeneration` directly.
    out = pipe(prompt)
    audio = out["audio"][0]
    sr = out["sampling_rate"]
    # Normalize
    audio = audio / max(1e-8, np.max(np.abs(audio)))
    return sr, audio.astype(np.float32)


def stitch_repeat(sr: int, chunk: np.ndarray, target_total_sec: int) -> np.ndarray:
    """
    Simple stitching: repeat chunk until reaching target length.
    """
    target_len = sr * target_total_sec
    if len(chunk) >= target_len:
        return chunk[:target_len]

    reps = int(np.ceil(target_len / len(chunk)))
    long = np.tile(chunk, reps)[:target_len]
    return long


def save_wav(sr: int, audio: np.ndarray, out_wav: Path) -> None:
    out_wav.parent.mkdir(parents=True, exist_ok=True)
    pcm16 = (audio * 32767.0).astype(np.int16)
    scipy.io.wavfile.write(str(out_wav), sr, pcm16)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", required=True, help="YYYY-MM-DD")
    ap.add_argument("--model", default="facebook/musicgen-small")
    ap.add_argument("--upload", action="store_true")
    args = ap.parse_args()

    date = args.date
    require_ffmpeg()

    data = load_prompts()
    defaults = data.get("defaults", {})
    tracks = data.get("tracks", [])

    # GPU setup
    device = 0 if torch.cuda.is_available() else -1
    print("⚡ device:", "GPU" if device == 0 else "CPU")

    # MusicGen pipeline
    pipe = pipeline("text-to-audio", model=args.model, device=device)

    bucket = os.environ.get("R2_BUCKET")
    if args.upload and not bucket:
        raise RuntimeError("R2_BUCKET env var missing for upload mode.")

    new_entries: list[dict] = []

    for t in tracks:
        merged = dict(defaults)
        merged.update(t)

        prompt = merged["prompt"]
        duration_sec = int(merged.get("duration_sec", 30))
        target_total_sec = int(merged.get("target_total_sec", 600))

        print("🎵 Generating:", merged["title"], "prompt:", prompt)

        sr, chunk = generate_chunk(pipe, prompt, duration_sec=duration_sec, device=device)
        full = stitch_repeat(sr, chunk, target_total_sec=target_total_sec)

        wav = TMP / f"{date}_{merged['id']}.wav"
        mp3 = OUT / f"premium-{date}-{merged['id']}.mp3"
        save_wav(sr, full, wav)

        postprocess_wav_to_mp3(
            inp_wav=wav,
            out_mp3=mp3,
            total_sec=target_total_sec,
            target_lufs=float(merged.get("target_lufs", -14)),
            fade_in_ms=int(merged.get("fade_in_ms", 1200)),
            fade_out_ms=int(merged.get("fade_out_ms", 2500)),
            bitrate=str(merged.get("export_bitrate", "192k")),
        )

        entry = {
            "id": f"premium-{date}-{merged['id']}",
            "title": merged["title"],
            "tier": "premium",
            "date": date,
            "category": merged["category"],
            "durationSec": target_total_sec,
            "goalTags": merged.get("goalTags", []),
            "natureTags": merged.get("natureTags", []),
            "energyMin": int(merged.get("energyMin", 0)),
            "energyMax": int(merged.get("energyMax", 100)),
            "ambienceMin": int(merged.get("ambienceMin", 0)),
            "ambienceMax": int(merged.get("ambienceMax", 100)),
            "objectKey": f"audio/premium/{date}/{mp3.name}",
        }

        if args.upload:
            upload_file(mp3, bucket=bucket, key=entry["objectKey"], public=False)

        new_entries.append(entry)
        print("✅ Built premium track:", entry["id"], mp3)

    if args.upload:
        paths = get_catalog_paths()
        existing = read_catalog(bucket, paths.premium_key)
        merged = upsert_tracks(existing, new_entries)
        write_catalog(bucket, paths.premium_key, merged)
        print(f"✅ Updated premium catalog: s3://{bucket}/{paths.premium_key} (count={len(merged)})")
    else:
        print("ℹ️ Run with --upload to push to R2 and update premium catalog.")


if __name__ == "__main__":
    main()
