"""
Premium MusicGen Daily Track Generator
========================================
Enterprise-Grade AI Music Production using MusicGen Stereo Large

This module generates high-fidelity AI music using:
- facebook/musicgen-stereo-large (3.3B parameters)
- Native audiocraft library (NOT transformers pipeline)
- Advanced DSP post-processing
- Professional mastering chain

Hardware Requirements:
- GPU with 16GB+ VRAM (NVIDIA A100, V100, or RTX 3090/4090)
- Recommended platforms: Google Colab Pro, Kaggle, RunPod, AWS p3/p4

Author: SoundFlow Engineering Team
"""

from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import yaml

from common.audio_utils import require_ffmpeg
from common.catalog_write import get_catalog_paths, read_catalog, upsert_tracks, write_catalog
from common.r2_upload import upload_file
from premium.postprocess import postprocess_track

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("SoundFlow-Premium")

# ============================================================================
# CONFIGURATION
# ============================================================================

# Model: STEREO LARGE is the key to "Best Ever" quality
MODEL_SIZE = "facebook/musicgen-stereo-large"  # 3.3B params, stereo output
# Alternative: "facebook/musicgen-stereo-medium" (1.5B) for faster generation

# Paths
ROOT = Path(__file__).resolve().parent
OUT = Path(".soundflow_out/premium")
TMP = Path(".soundflow_tmp/premium")

# ============================================================================
# AUDIOCRAFT INTEGRATION
# ============================================================================


def load_musicgen_model(model_name: str = MODEL_SIZE, device: str = "auto"):
    """
    Loads MusicGen model using native audiocraft library.

    Args:
        model_name: HuggingFace model ID
        device: Device to load model on ("cuda", "cpu", or "auto")

    Returns:
        MusicGen model instance
    """
    try:
        from audiocraft.models import MusicGen
    except ImportError:
        raise ImportError(
            "❌ audiocraft library not found!\n"
            "Install with: pip install audiocraft\n"
            "Or: uv pip install audiocraft"
        )

    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    logger.info(f"⚡ Device: {device.upper()}")
    if device == "cuda":
        gpu_name = torch.cuda.get_device_name(0)
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
        logger.info(f"🎮 GPU: {gpu_name} ({gpu_mem:.1f}GB VRAM)")

    logger.info(f"📥 Loading Model: {model_name}")
    logger.info("⏳ This may take 2-5 minutes on first run (downloading ~13GB)...")

    model = MusicGen.get_pretrained(model_name, device=device)

    logger.info("✅ Model loaded successfully!")
    return model


def configure_generation_params(
    model,
    duration: int = 30,
    top_k: int = 250,
    top_p: float = 0.0,
    temperature: float = 1.0,
    cfg_coef: float = 3.5,
):
    """
    Configures generation parameters for high quality output.

    Args:
        model: MusicGen model instance
        duration: Generation duration in seconds (max ~30s per chunk)
        top_k: Top-K sampling (higher = more creative, 250 is recommended)
        top_p: Top-P/nucleus sampling (0.0 = disabled, use top_k instead)
        temperature: Sampling temperature (1.0 = default, >1.0 = more random)
        cfg_coef: Classifier-free guidance coefficient (higher = follow prompt more strictly)
                  Range: 1.0-10.0, recommended: 3.0-4.0
    """
    model.set_generation_params(
        duration=duration,
        top_k=top_k,
        top_p=top_p,
        temperature=temperature,
        cfg_coef=cfg_coef,
    )
    logger.info(
        f"⚙️ Generation Config: duration={duration}s, top_k={top_k}, "
        f"temp={temperature}, cfg={cfg_coef}"
    )


def generate_music(
    model,
    prompt: str,
    duration: int = 30,
) -> tuple[int, np.ndarray]:
    """
    Generates music from text prompt using MusicGen.

    Args:
        model: MusicGen model instance
        prompt: Text description of desired music
        duration: Target duration in seconds

    Returns:
        Tuple of (sample_rate, audio_array)
        - sample_rate: Always 32000 Hz for MusicGen
        - audio_array: Stereo float32 array, shape (2, samples)
    """
    logger.info(f"🎵 Generating: '{prompt}'")

    # Generate (returns tensor of shape [1, 2, samples])
    with torch.no_grad():
        wav = model.generate([prompt], progress=True)

    # Convert to numpy
    audio = wav[0].cpu().numpy()  # Shape: (2, samples)

    # Transpose to (samples, 2) for scipy compatibility
    audio = audio.T

    # Get sample rate
    sample_rate = model.sample_rate

    logger.info(f"✅ Generated: {audio.shape[0] / sample_rate:.1f}s @ {sample_rate}Hz")

    return sample_rate, audio


def save_wav(audio: np.ndarray, sr: int, path: Path) -> None:
    """
    Saves audio to WAV file using audiocraft's built-in utilities.

    Args:
        audio: Audio array (samples, channels)
        sr: Sample rate
        path: Output path (without extension)
    """
    try:
        from audiocraft.data.audio import audio_write
    except ImportError:
        # Fallback to scipy
        import scipy.io.wavfile as wavfile

        path.parent.mkdir(parents=True, exist_ok=True)
        audio_int16 = (audio * 32767.0).astype(np.int16)
        wavfile.write(path.with_suffix(".wav"), sr, audio_int16)
        return

    path.parent.mkdir(parents=True, exist_ok=True)

    # Convert to torch tensor (channels, samples)
    audio_tensor = torch.from_numpy(audio.T).unsqueeze(0)  # (1, 2, samples)

    # audio_write expects path without extension
    audio_write(
        path.with_suffix(""),
        audio_tensor[0],
        sr,
        strategy="loudness",
        loudness_headroom_db=14,  # Prevent clipping
    )


# ============================================================================
# TRACK GENERATION PIPELINE
# ============================================================================


def load_prompts() -> dict:
    """Loads premium prompts configuration."""
    p = Path(__file__).resolve().parents[1] / "prompts" / "premium_prompts.yaml"
    if not p.exists():
        raise FileNotFoundError(f"Prompts file not found: {p}")
    return yaml.safe_load(p.read_text(encoding="utf-8"))


def generate_track_extended(
    model,
    prompt: str,
    target_duration: int,
    chunk_duration: int = 30,
) -> tuple[int, np.ndarray]:
    """
    Generates extended tracks by stitching multiple chunks.

    For tracks longer than 30s, we generate multiple chunks and concatenate.
    Future improvement: Crossfade between chunks for seamless transitions.

    Args:
        model: MusicGen model instance
        prompt: Music description
        target_duration: Total desired duration in seconds
        chunk_duration: Duration of each generated chunk

    Returns:
        Tuple of (sample_rate, audio_array)
    """
    if target_duration <= chunk_duration:
        return generate_music(model, prompt, target_duration)

    # Generate multiple chunks
    chunks = []
    sr = None
    n_chunks = int(np.ceil(target_duration / chunk_duration))

    logger.info(f"🔗 Generating {n_chunks} chunks for {target_duration}s track...")

    for i in range(n_chunks):
        logger.info(f"  Chunk {i + 1}/{n_chunks}...")
        sr, chunk = generate_music(model, prompt, chunk_duration)
        chunks.append(chunk)

    # Concatenate chunks
    full_audio = np.concatenate(chunks, axis=0)

    # Trim to exact duration
    target_samples = int(sr * target_duration)
    full_audio = full_audio[:target_samples]

    logger.info(f"✅ Stitched {n_chunks} chunks → {len(full_audio) / sr:.1f}s")

    return sr, full_audio


def main():
    """Main generation pipeline."""
    parser = argparse.ArgumentParser(description="Premium MusicGen Daily Generator")
    parser.add_argument("--date", required=True, help="Generation date (YYYY-MM-DD)")
    parser.add_argument("--model", default=MODEL_SIZE, help="Model name/path")
    parser.add_argument("--upload", action="store_true", help="Upload to R2 and update catalog")
    parser.add_argument("--device", default="auto", help="Device (cuda/cpu/auto)")
    parser.add_argument("--duration", type=int, help="Override target duration (seconds)")
    args = parser.parse_args()

    date = args.date
    require_ffmpeg()

    # Load prompts
    data = load_prompts()
    defaults = data.get("defaults", {})
    tracks = data.get("tracks", [])

    if not tracks:
        logger.error("❌ No tracks defined in premium_prompts.yaml")
        return

    # Load model
    model = load_musicgen_model(args.model, args.device)

    # Configure generation parameters
    chunk_duration = defaults.get("duration_sec", 30)
    configure_generation_params(
        model,
        duration=chunk_duration,
        top_k=250,
        temperature=1.0,
        cfg_coef=3.5,
    )

    # Setup upload (if enabled)
    bucket = os.environ.get("R2_BUCKET")
    if args.upload and not bucket:
        raise RuntimeError("❌ R2_BUCKET environment variable required for --upload")

    new_entries = []

    # Generate tracks
    for track_cfg in tracks:
        # Merge defaults with track config
        cfg = {**defaults, **track_cfg}

        track_id = cfg["id"]
        title = cfg["title"]
        prompt = cfg["prompt"]
        category = cfg.get("category", "General")

        # Duration
        if args.duration:
            target_duration = args.duration
        else:
            target_duration = int(cfg.get("target_total_sec", 180))

        logger.info(f"\n{'=' * 70}")
        logger.info(f"🎼 Track: {title}")
        logger.info(f"📝 Prompt: {prompt}")
        logger.info(f"⏱️  Duration: {target_duration}s")
        logger.info(f"{'=' * 70}\n")

        # Generate
        sr, audio = generate_track_extended(model, prompt, target_duration, chunk_duration)

        # Save raw WAV
        wav_path = TMP / f"{date}_{track_id}"
        save_wav(audio, sr, wav_path)

        # Post-process with DSP
        mp3_path = OUT / f"premium-{date}-{track_id}.mp3"

        dsp_flags = cfg.get("dsp_flags", {})
        postprocess_track(
            inp_wav=wav_path.with_suffix(".wav"),
            out_mp3=mp3_path,
            target_lufs=float(cfg.get("target_lufs", -14.0)),
            bitrate=str(cfg.get("export_bitrate", "320k")),
            dsp_flags=dsp_flags,
        )

        # Catalog entry
        entry = {
            "id": f"premium-{date}-{track_id}",
            "title": title,
            "tier": "premium",
            "date": date,
            "category": category,
            "durationSec": target_duration,
            "goalTags": cfg.get("goalTags", []),
            "natureTags": cfg.get("natureTags", []),
            "energyMin": int(cfg.get("energyMin", 0)),
            "energyMax": int(cfg.get("energyMax", 100)),
            "ambienceMin": int(cfg.get("ambienceMin", 0)),
            "ambienceMax": int(cfg.get("ambienceMax", 100)),
            "objectKey": f"audio/premium/{date}/{mp3_path.name}",
        }

        # Upload
        if args.upload:
            logger.info(f"☁️  Uploading to R2: {entry['objectKey']}")
            upload_file(mp3_path, bucket=bucket, key=entry["objectKey"], public=False)

        new_entries.append(entry)
        logger.info(f"✅ Completed: {entry['id']} → {mp3_path}")

    # Update catalog
    if args.upload:
        paths = get_catalog_paths()
        existing = read_catalog(bucket, paths.premium_key)
        merged = upsert_tracks(existing, new_entries)
        write_catalog(bucket, paths.premium_key, merged)
        logger.info(
            f"\n✅ Updated premium catalog: s3://{bucket}/{paths.premium_key} "
            f"(total tracks: {len(merged)})"
        )
    else:
        logger.info(
            f"\nℹ️  Dry run complete. Generated {len(new_entries)} tracks."
        )
        logger.info("   Run with --upload to push to R2 and update catalog.")

    logger.info("\n🎉 Premium generation complete!")


if __name__ == "__main__":
    main()
