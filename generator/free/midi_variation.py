from __future__ import annotations

import argparse
import random
from pathlib import Path

from common.audio_utils import require_ffmpeg, _run  # noqa: SLF001


def generate_simple_midi(out_mid: Path, seed: str, bpm: int = 80) -> None:
    """
    Minimal deterministic MIDI generator without heavy deps.
    Writes a very simple melody using `midiutil` would be easier, but we keep deps minimal.
    For production, replace with a proper MIDI lib or a transformer.
    """
    # If you want real MIDI, add dependency: midiutil
    # This file is a placeholder for the concept.
    raise RuntimeError(
        "MIDI generation placeholder. Add `midiutil` and implement proper MIDI writing if needed."
    )


def render_soundfont(mid: Path, sf2: Path, out_wav: Path) -> None:
    """
    Render MIDI to audio using fluidsynth (needs system package).
    On GitHub Actions you can apt-get install fluidsynth + soundfonts.
    """
    require_ffmpeg()
    out_wav.parent.mkdir(parents=True, exist_ok=True)
    cmd = ["fluidsynth", "-ni", str(sf2), str(mid), "-F", str(out_wav), "-r", "44100"]
    _run(cmd)  # noqa: SLF001


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", required=True)
    ap.add_argument("--soundfont", required=True, help="Path to .sf2")
    args = ap.parse_args()

    date = args.date
    sf2 = Path(args.soundfont)

    rnd = random.Random(f"{date}:midi")
    bpm = rnd.choice([70, 80, 90])

    out_mid = Path(".soundflow_out/midi") / f"{date}.mid"
    out_wav = Path(".soundflow_out/midi") / f"{date}.wav"

    out_mid.parent.mkdir(parents=True, exist_ok=True)

    # Placeholder: implement real midi writing if needed
    generate_simple_midi(out_mid, seed=date, bpm=bpm)

    render_soundfont(out_mid, sf2, out_wav)
    print("✅ Rendered MIDI variation:", out_wav)


if __name__ == "__main__":
    main()
