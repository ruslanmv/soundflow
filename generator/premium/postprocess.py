"""
Premium DSP Post-Processing Engine
===================================
Neuro-Symbolic Audio Enhancement for Enterprise Music Production

Features:
- Binaural Beat Injection (Gamma/Alpha/Theta waves for Focus/Meditation)
- Sidechain Compression (EDM/House pumping effect)
- Professional Mastering Chain (LUFS normalization, True Peak limiting)
- High-Fidelity Export (320kbps MP3 with proper fades)
"""

from __future__ import annotations

import logging
import shutil
from pathlib import Path
from typing import Optional

import numpy as np
import scipy.io.wavfile as wavfile
import ffmpeg

logger = logging.getLogger("SoundFlow-DSP")

# ============================================================================
# SYSTEM CHECKS
# ============================================================================


def require_ffmpeg():
    """Checks if FFmpeg is installed."""
    if shutil.which("ffmpeg") is None:
        raise RuntimeError(
            "❌ FFmpeg is missing! Install it via:\n"
            "  Ubuntu/Debian: apt-get install ffmpeg\n"
            "  macOS: brew install ffmpeg\n"
            "  Colab: !apt-get -qq install ffmpeg"
        )


# ============================================================================
# NEURO-SYMBOLIC DSP FUNCTIONS (The Mathematical Layer)
# ============================================================================


def inject_binaural_beats(
    audio: np.ndarray,
    sr: int,
    base_freq: float = 200.0,
    beat_freq: float = 40.0,
    mix: float = 0.12,
) -> np.ndarray:
    """
    Injects precise binaural beats into stereo audio.

    Theory: Binaural beats occur when two slightly different frequencies
    are played in each ear. The brain perceives the difference as a rhythmic pulse.

    Args:
        audio: Stereo audio (N, 2) or mono (N,)
        sr: Sample rate
        base_freq: Base carrier frequency (Hz) - typically 100-300 Hz
        beat_freq: Beat frequency (Hz) - the perceived rhythm:
            - Gamma (30-50 Hz): Focus, concentration, peak performance
            - Beta (13-30 Hz): Active thinking, alertness
            - Alpha (8-13 Hz): Relaxation, meditation
            - Theta (4-8 Hz): Deep meditation, creativity
            - Delta (0.5-4 Hz): Deep sleep, healing
        mix: Mix level (0.0-1.0), how much binaural to blend with original

    Returns:
        Enhanced stereo audio with binaural beats
    """
    n_samples = len(audio)
    t = np.arange(n_samples) / sr

    # Generate pure sine waves
    # Left ear: base frequency
    # Right ear: base + beat frequency
    sine_left = np.sin(2 * np.pi * base_freq * t)
    sine_right = np.sin(2 * np.pi * (base_freq + beat_freq) * t)

    # Convert mono to stereo if needed
    if audio.ndim == 1:
        audio = np.stack([audio, audio], axis=1)

    # Apply gentle mixing (preserve original audio character)
    out = audio.copy()
    out[:, 0] = (audio[:, 0] * (1.0 - mix)) + (sine_left * mix * 0.4)
    out[:, 1] = (audio[:, 1] * (1.0 - mix)) + (sine_right * mix * 0.4)

    logger.info(
        f"🧠 Injected Binaural: {beat_freq}Hz beat @ {base_freq}Hz carrier "
        f"(mix={mix:.1%})"
    )
    return out


def apply_sidechain_compression(
    audio: np.ndarray,
    sr: int,
    bpm: int = 128,
    strength: float = 0.7,
    attack_ratio: float = 0.3,
) -> np.ndarray:
    """
    Simulates sidechain compression (volume ducking) for EDM/House tracks.

    Theory: In dance music, the kick drum triggers a compressor on other elements,
    creating a "pumping" or "breathing" effect that adds energy and groove.

    Args:
        audio: Audio array (mono or stereo)
        sr: Sample rate
        bpm: Tempo in beats per minute
        strength: Duck depth (0.0-1.0), where 1.0 = maximum ducking
        attack_ratio: Portion of beat spent ducking (0.0-1.0)

    Returns:
        Audio with sidechain pumping effect
    """
    # Calculate samples per beat
    beat_samples = int((60.0 / bpm) * sr)
    duck_len = int(beat_samples * attack_ratio)

    # Create envelope curve (exponential recovery for natural sound)
    # Starts at (1 - strength), rises to 1.0
    curve = np.linspace(1.0 - strength, 1.0, duck_len) ** 0.5

    # Full beat envelope: duck + sustain
    full_env = np.ones(beat_samples, dtype=np.float32)
    full_env[:duck_len] = curve

    # Tile envelope across entire track
    n_samples = len(audio)
    repeats = int(np.ceil(n_samples / len(full_env)))
    master_env = np.tile(full_env, repeats)[:n_samples]

    # Apply envelope
    if audio.ndim == 1:
        result = audio * master_env
    else:
        result = audio * master_env[:, np.newaxis]

    logger.info(
        f"👟 Applied Sidechain: {bpm} BPM, strength={strength:.1%}, "
        f"attack={attack_ratio:.1%}"
    )
    return result


def apply_fade_in_out(
    audio: np.ndarray,
    sr: int,
    fade_in_ms: int = 1500,
    fade_out_ms: int = 3000,
) -> np.ndarray:
    """
    Applies smooth fade in/out to prevent clicks and pops.

    Args:
        audio: Audio array
        sr: Sample rate
        fade_in_ms: Fade-in duration in milliseconds
        fade_out_ms: Fade-out duration in milliseconds

    Returns:
        Audio with fades applied
    """
    n_samples = len(audio)
    fade_in_samples = int((fade_in_ms / 1000.0) * sr)
    fade_out_samples = int((fade_out_ms / 1000.0) * sr)

    # Create fade curves (sine-based for smooth transition)
    fade_in_curve = np.sin(np.linspace(0, np.pi / 2, fade_in_samples)) ** 2
    fade_out_curve = np.sin(np.linspace(np.pi / 2, 0, fade_out_samples)) ** 2

    # Build full envelope
    envelope = np.ones(n_samples, dtype=np.float32)
    envelope[:fade_in_samples] = fade_in_curve
    envelope[-fade_out_samples:] = fade_out_curve

    # Apply
    if audio.ndim == 1:
        return audio * envelope
    return audio * envelope[:, np.newaxis]


# ============================================================================
# MAIN PROCESSING PIPELINE
# ============================================================================


def postprocess_track(
    inp_wav: Path,
    out_mp3: Path,
    target_lufs: float = -14.0,
    bitrate: str = "320k",
    dsp_flags: Optional[dict] = None,
) -> None:
    """
    Complete premium post-processing pipeline.

    Pipeline:
    1. Load audio from WAV
    2. Apply DSP enhancements (binaural, sidechain)
    3. Apply fades
    4. Save intermediate WAV
    5. Master with FFmpeg (loudness normalization, true peak limiting)
    6. Export to high-quality MP3

    Args:
        inp_wav: Input WAV file path
        out_mp3: Output MP3 file path
        target_lufs: Target loudness in LUFS (broadcast standard is -14.0)
        bitrate: MP3 bitrate (320k for premium quality)
        dsp_flags: DSP configuration dict with keys:
            - binaural: bool, enable binaural beats
            - binaural_freq: float, beat frequency in Hz
            - binaural_base: float, carrier frequency in Hz
            - sidechain: bool, enable sidechain compression
            - bpm: int, tempo for sidechain
            - ducking_strength: float, sidechain intensity (0-1)
            - fade_in_ms: int, fade-in duration
            - fade_out_ms: int, fade-out duration
    """
    require_ffmpeg()
    dsp_flags = dsp_flags or {}

    logger.info(f"🎚️ Processing: {inp_wav.name}")

    # -------------------------------------------------------------------------
    # 1. Load Audio
    # -------------------------------------------------------------------------
    sr, audio = wavfile.read(inp_wav)

    # Convert int16 to float32 (-1.0 to 1.0)
    if audio.dtype == np.int16:
        audio = audio.astype(np.float32) / 32768.0
    elif audio.dtype == np.int32:
        audio = audio.astype(np.float32) / 2147483648.0

    logger.info(f"📊 Loaded: {sr}Hz, {audio.shape}, dtype={audio.dtype}")

    # -------------------------------------------------------------------------
    # 2. Apply Neuro-Symbolic Enhancements
    # -------------------------------------------------------------------------

    if dsp_flags.get("binaural", False):
        audio = inject_binaural_beats(
            audio,
            sr,
            base_freq=dsp_flags.get("binaural_base", 200.0),
            beat_freq=dsp_flags.get("binaural_freq", 40.0),
            mix=dsp_flags.get("binaural_mix", 0.12),
        )

    if dsp_flags.get("sidechain", False):
        audio = apply_sidechain_compression(
            audio,
            sr,
            bpm=dsp_flags.get("bpm", 128),
            strength=dsp_flags.get("ducking_strength", 0.7),
            attack_ratio=dsp_flags.get("sidechain_attack", 0.3),
        )

    # -------------------------------------------------------------------------
    # 3. Apply Fades
    # -------------------------------------------------------------------------

    fade_in = dsp_flags.get("fade_in_ms", 1500)
    fade_out = dsp_flags.get("fade_out_ms", 3000)

    if fade_in > 0 or fade_out > 0:
        audio = apply_fade_in_out(audio, sr, fade_in, fade_out)
        logger.info(f"🎬 Fades: in={fade_in}ms, out={fade_out}ms")

    # -------------------------------------------------------------------------
    # 4. Save Intermediate DSP WAV
    # -------------------------------------------------------------------------

    tmp_dir = Path(".soundflow_tmp/premium")
    tmp_dir.mkdir(parents=True, exist_ok=True)
    tmp_dsp = tmp_dir / (inp_wav.stem + "_dsp.wav")

    # Clip to prevent distortion
    audio = np.clip(audio, -1.0, 1.0)

    # Convert back to int16
    audio_int16 = (audio * 32767.0).astype(np.int16)
    wavfile.write(tmp_dsp, sr, audio_int16)

    logger.info(f"💾 Saved DSP intermediate: {tmp_dsp.name}")

    # -------------------------------------------------------------------------
    # 5. Final Mastering with FFmpeg
    # -------------------------------------------------------------------------

    out_mp3.parent.mkdir(parents=True, exist_ok=True)

    try:
        # Professional mastering chain:
        # - loudnorm: EBU R128 loudness normalization
        # - I: Integrated loudness target (LUFS)
        # - TP: True peak limit (dBTP)
        # - LRA: Loudness range target
        (
            ffmpeg
            .input(str(tmp_dsp))
            .filter('loudnorm', I=target_lufs, TP=-1.5, LRA=11)
            .output(
                str(out_mp3),
                audio_bitrate=bitrate,
                ar=44100,  # Standard CD quality sample rate
                ac=2,      # Stereo
            )
            .overwrite_output()
            .run(capture_stdout=True, capture_stderr=True, quiet=True)
        )

        logger.info(
            f"✅ Mastered & Exported: {out_mp3.name} "
            f"({target_lufs} LUFS, {bitrate})"
        )

    except ffmpeg.Error as e:
        logger.error(f"❌ FFmpeg Error: {e.stderr.decode() if e.stderr else str(e)}")
        raise


# ============================================================================
# LEGACY COMPATIBILITY
# ============================================================================


def postprocess_wav_to_mp3(
    inp_wav: Path,
    out_mp3: Path,
    total_sec: int,
    target_lufs: float = -14.0,
    fade_in_ms: int = 1200,
    fade_out_ms: int = 2500,
    bitrate: str = "320k",
) -> None:
    """
    Legacy function for backward compatibility.
    Redirects to the new postprocess_track() function.
    """
    dsp_flags = {
        "fade_in_ms": fade_in_ms,
        "fade_out_ms": fade_out_ms,
    }
    postprocess_track(inp_wav, out_mp3, target_lufs, bitrate, dsp_flags)
