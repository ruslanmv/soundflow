from __future__ import annotations

import os
import subprocess
from pathlib import Path


class AudioError(RuntimeError):
    pass


def require_ffmpeg() -> None:
    try:
        subprocess.run(["ffmpeg", "-version"], check=True, capture_output=True)
    except Exception as e:
        raise AudioError("ffmpeg not found. Install ffmpeg to run audio pipeline.") from e


def ffmpeg_mix(inputs: list[Path], output: Path) -> None:
    """
    Simple mixdown: sum all inputs with `amix`.
    Assumes inputs are already roughly aligned/looped.
    """
    if not inputs:
        raise AudioError("No inputs provided for mix.")
    output.parent.mkdir(parents=True, exist_ok=True)

    cmd = ["ffmpeg", "-y"]
    for p in inputs:
        cmd += ["-i", str(p)]

    # amix all streams
    cmd += [
        "-filter_complex",
        f"amix=inputs={len(inputs)}:duration=longest:dropout_transition=2",
        "-c:a",
        "pcm_s16le",
        str(output),
    ]

    _run(cmd)


def ffmpeg_loop_to_duration(inp: Path, out: Path, duration_sec: int) -> None:
    """
    Loops input to requested duration.
    """
    out.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg",
        "-y",
        "-stream_loop",
        "-1",
        "-i",
        str(inp),
        "-t",
        str(duration_sec),
        "-c:a",
        "pcm_s16le",
        str(out),
    ]
    _run(cmd)


def ffmpeg_fade(inp: Path, out: Path, fade_in_ms: int, fade_out_ms: int, total_sec: int) -> None:
    """
    Applies fade in/out.
    """
    out.parent.mkdir(parents=True, exist_ok=True)
    fade_in = fade_in_ms / 1000.0
    fade_out = fade_out_ms / 1000.0
    fade_out_start = max(0.0, total_sec - fade_out)

    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        str(inp),
        "-af",
        f"afade=t=in:st=0:d={fade_in},afade=t=out:st={fade_out_start}:d={fade_out}",
        "-c:a",
        "pcm_s16le",
        str(out),
    ]
    _run(cmd)


def ffmpeg_loudnorm(inp: Path, out: Path, target_lufs: float = -14.0) -> None:
    """
    One-pass loudness normalization (simple).
    For higher accuracy you can do 2-pass; this is stable for production use.
    """
    out.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        str(inp),
        "-af",
        f"loudnorm=I={target_lufs}:TP=-1.5:LRA=11",
        "-c:a",
        "pcm_s16le",
        str(out),
    ]
    _run(cmd)


def ffmpeg_encode_mp3(inp: Path, out: Path, bitrate: str = "192k") -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        str(inp),
        "-codec:a",
        "libmp3lame",
        "-b:a",
        bitrate,
        "-ar",
        "44100",
        "-ac",
        "2",
        str(out),
    ]
    _run(cmd)


def _run(cmd: list[str]) -> None:
    p = subprocess.run(cmd, capture_output=True, text=True)
    if p.returncode != 0:
        raise AudioError(f"Command failed: {' '.join(cmd)}\nSTDERR:\n{p.stderr}")
