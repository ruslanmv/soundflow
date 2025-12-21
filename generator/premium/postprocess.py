from __future__ import annotations

from pathlib import Path
from common.audio_utils import ffmpeg_fade, ffmpeg_loudnorm, ffmpeg_encode_mp3, require_ffmpeg


def postprocess_wav_to_mp3(
    inp_wav: Path,
    out_mp3: Path,
    total_sec: int,
    target_lufs: float = -14.0,
    fade_in_ms: int = 1200,
    fade_out_ms: int = 2500,
    bitrate: str = "192k",
) -> None:
    require_ffmpeg()

    tmp_dir = Path(".soundflow_tmp/premium")
    tmp_dir.mkdir(parents=True, exist_ok=True)

    faded = tmp_dir / (out_mp3.stem + "_fade.wav")
    normed = tmp_dir / (out_mp3.stem + "_norm.wav")

    ffmpeg_fade(inp_wav, faded, fade_in_ms=fade_in_ms, fade_out_ms=fade_out_ms, total_sec=total_sec)
    ffmpeg_loudnorm(faded, normed, target_lufs=target_lufs)
    ffmpeg_encode_mp3(normed, out_mp3, bitrate=bitrate)
