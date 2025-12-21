#!/usr/bin/env python3
# generator/tests/test_music.py
"""
SoundFlow API – "Competition Pack" Generator (PRO ENERGY + AUTHENTIC CLUB FEEL)

This version is tuned to take advantage of your recent engine upgrades
(anti-8bit polyBLEP + oversampling + SVF + PRO_PRESETS in music_engine.py).

What this script does differently vs your previous one:
- Uses "music" genre strings that align with your GENRE_STYLES normalization:
  Techno / House / Deep / Trance / EDM / Dance / Chillout / Lounge / Ambient / Lofi / Synth
- Uses longer stems & better arrangement feel by asking for:
  - energy_curve: peak/build/linear
  - higher intensity for club, moderate for relax/focus
- Keeps ambience OFF for competition masters (clean, modern)
- Generates a "WIN pack": fewer tracks, but each is meant to be a “final candidate”
- Adds a simple "retry with new seed" if the server returns a weird result

Assumptions:
- POST /api/generate accepts your payload fields and returns {"url": "..."}
- Your backend understands layers: ["kick","drums","bass","music","pad","synth"] (extra layers ignored safely)

Usage:
  python3 generator/tests/test_music.py

Tip:
- Run this, listen, and then lock the best seeds by printing meta["seed"] from server,
  or by setting FORCE_SEED_OVERRIDE below.
"""

from __future__ import annotations

import json
import time
import urllib.request
import urllib.error
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional
import random

# =============================================================================
# CONFIG
# =============================================================================

API_BASE = "http://localhost:8000"

OUT_DIR = Path("competition_pack_outputs")
OUT_DIR.mkdir(parents=True, exist_ok=True)

DEFAULT_TIMEOUT_SEC = 900  # allow heavy renders
DATE = datetime.now().strftime("%Y-%m-%d")

PREFERRED_FORMAT = "mp3"  # set "wav" if your API returns wav urls

# If you want fully repeatable outputs, set this to a non-empty string and
# the script will derive deterministic seeds from it.
FORCE_SEED_OVERRIDE: Optional[str] = None  # e.g. "my_competition_run_01"

# If a render fails, retry with new seeds this many times
RETRIES_PER_TRACK = 2

# =============================================================================
# HTTP HELPERS
# =============================================================================

def _http_post_json(url: str, payload: dict) -> dict:
    body = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=DEFAULT_TIMEOUT_SEC) as resp:
        return json.loads(resp.read().decode("utf-8"))

def _download_file(url: str, out_path: Path) -> None:
    with urllib.request.urlopen(url, timeout=DEFAULT_TIMEOUT_SEC) as resp:
        out_path.write_bytes(resp.read())

# =============================================================================
# PAYLOAD BUILDER
# =============================================================================

def _seed(name: str, *, tag: str, salt: str) -> str:
    """
    Unique-ish seed per request, but optionally repeatable with FORCE_SEED_OVERRIDE.
    """
    base = FORCE_SEED_OVERRIDE or str(time.time())
    rnd = random.Random(f"{base}:{name}:{tag}:{salt}")
    return f"competition-{tag}-{name.replace(' ', '')}-{rnd.randint(1000, 9999)}"

def make_payload(
    *,
    name: str,
    genre: str,
    bpm: int,
    key: str,
    vibe: str,
    duration: int,
    intensity: int,
    target_lufs: float,
    synth_params: Optional[Dict[str, float]] = None,
    layers: Optional[List[str]] = None,
    variation: float = 0.50,
    force_new: bool = True,
) -> Dict[str, Any]:
    """
    Build a competition-ready request.
    - ambience hard-off by default (clean master)
    - use slightly lower variation than before to keep musical coherence,
      but still different each seed.
    """
    if synth_params is None:
        synth_params = {
            "cutoff": 78.0,
            "resonance": 22.0,
            "drive": 18.0,
            "space": 25.0,
        }

    if layers is None:
        layers = ["kick", "drums", "bass", "music", "pad", "synth"]

    seed = _seed(
        name,
        tag=genre.lower().replace(" ", "_"),
        salt=f"{bpm}:{key}:{vibe}:{duration}:{intensity}:{variation}:{target_lufs}",
    )

    return {
        "mode": "music",
        "channels": 2,
        "variation": float(variation),

        "genre": genre,
        "bpm": int(bpm),
        "key": key,
        "seed": seed,
        "layers": layers,

        "energy_curve": vibe,
        "duration": int(duration),

        # clean: no vinyl/rain/white noise for competition masters
        "focus_mode": "off",
        "ambience": {"rain": 0, "vinyl": 0, "white": 0},

        "intensity": int(intensity),
        "synth_params": synth_params,

        "target_lufs": float(target_lufs),
        "force_new": bool(force_new),
    }

# =============================================================================
# PRESET LIBRARY (WINNERS)
# =============================================================================

@dataclass(frozen=True)
class TrackCase:
    id: str
    payload: Dict[str, Any]
    notes: str

def build_winner_pack() -> List[TrackCase]:
    """
    “Less but better.” These are designed to sound like real club music,
    exploiting your improved synth quality in music_engine.py.

    LUFS targets:
      - club:  -9.5 to -10.0 (competitive)
      - relax: -14.0 to -14.5 (pleasant)
      - focus: -12.5 to -13.0 (steady)
    """
    tracks: List[TrackCase] = []

    # =========================
    # CLUB WINNERS
    # =========================

    # 1) House (festival/club-ready, warm groove, strong hook)
    tracks.append(TrackCase(
        id=f"{DATE}-WIN-house-golden-hour",
        notes="Big-room-friendly House: punchy low-end, bright but not harsh, peak curve.",
        payload=make_payload(
            name="Golden Hour House",
            genre="House",
            bpm=126,
            key="G#",
            vibe="peak",
            duration=200,
            intensity=96,
            target_lufs=-10.0,
            variation=0.48,
            layers=["kick", "drums", "bass", "music", "pad", "synth"],
            synth_params={
                "cutoff": 90.0,
                "resonance": 30.0,
                "drive": 28.0,
                "space": 18.0,
            },
        ),
    ))

    # 2) Techno (hypnotic, driving, proper tension/release)
    tracks.append(TrackCase(
        id=f"{DATE}-WIN-techno-hypnotic-pressure",
        notes="Hypnotic Techno: tight space, heavier drive, peak curve.",
        payload=make_payload(
            name="Hypnotic Pressure",
            genre="Techno",
            bpm=132,
            key="F",
            vibe="peak",
            duration=210,
            intensity=98,
            target_lufs=-9.5,
            variation=0.46,
            layers=["kick", "drums", "bass", "music", "pad"],
            synth_params={
                "cutoff": 78.0,
                "resonance": 26.0,
                "drive": 38.0,
                "space": 10.0,
            },
        ),
    ))

    # 3) Trance (anthem, euphoric, wide + emotional)
    tracks.append(TrackCase(
        id=f"{DATE}-WIN-trance-euphoria-anthem",
        notes="Euphoric Trance: wide pads, large space, cleaner drive, peak curve.",
        payload=make_payload(
            name="Euphoria Anthem",
            genre="Trance",
            bpm=138,
            key="A",
            vibe="peak",
            duration=230,
            intensity=94,
            target_lufs=-10.0,
            variation=0.50,
            layers=["kick", "drums", "bass", "music", "pad", "synth"],
            synth_params={
                "cutoff": 92.0,
                "resonance": 22.0,
                "drive": 20.0,
                "space": 58.0,
            },
        ),
    ))

    # 4) Melodic Techno-ish (build curve = competition “story” track)
    tracks.append(TrackCase(
        id=f"{DATE}-WIN-melodic-night-drive",
        notes="Melodic Techno build: emotional + modern, build curve to climax.",
        payload=make_payload(
            name="Night Drive",
            genre="Techno",
            bpm=128,
            key="G",
            vibe="build",
            duration=220,
            intensity=86,
            target_lufs=-11.0,
            variation=0.52,
            layers=["kick", "drums", "bass", "music", "pad", "synth"],
            synth_params={
                "cutoff": 82.0,
                "resonance": 24.0,
                "drive": 22.0,
                "space": 22.0,
            },
        ),
    ))

    # =========================
    # RELAX / LOUNGE (CLEAN)
    # =========================
    tracks.append(TrackCase(
        id=f"{DATE}-RELAX-lounge-silk-road",
        notes="Lounge: warm, smooth, not too loud, linear curve.",
        payload=make_payload(
            name="Silk Road Lounge",
            genre="Lounge",
            bpm=98,
            key="A#",
            vibe="linear",
            duration=260,
            intensity=40,
            target_lufs=-14.0,
            variation=0.45,
            layers=["drums", "bass", "music", "pad"],
            synth_params={
                "cutoff": 50.0,
                "resonance": 10.0,
                "drive": 6.0,
                "space": 45.0,
            },
        ),
    ))

    tracks.append(TrackCase(
        id=f"{DATE}-RELAX-chillout-ocean-air",
        notes="Chillout: airy pads, gentle groove, linear curve.",
        payload=make_payload(
            name="Ocean Air",
            genre="Chillout",
            bpm=90,
            key="E",
            vibe="linear",
            duration=260,
            intensity=36,
            target_lufs=-14.5,
            variation=0.47,
            layers=["drums", "bass", "music", "pad"],
            synth_params={
                "cutoff": 46.0,
                "resonance": 12.0,
                "drive": 4.0,
                "space": 58.0,
            },
        ),
    ))

    # =========================
    # FOCUS / CODING (STEADY)
    # =========================
    tracks.append(TrackCase(
        id=f"{DATE}-FOCUS-deep-flow-state",
        notes="Deep focus: steady, non-fatiguing, linear curve.",
        payload=make_payload(
            name="Flow State",
            genre="Deep",
            bpm=118,
            key="E",
            vibe="linear",
            duration=300,
            intensity=52,
            target_lufs=-12.6,
            variation=0.44,
            layers=["drums", "bass", "music", "pad"],
            synth_params={
                "cutoff": 62.0,
                "resonance": 18.0,
                "drive": 10.0,
                "space": 28.0,
            },
        ),
    ))

    tracks.append(TrackCase(
        id=f"{DATE}-FOCUS-synth-coding-drift",
        notes="Synth coding: gentle motion, clean, linear curve.",
        payload=make_payload(
            name="Coding Drift",
            genre="Synth",
            bpm=105,
            key="C",
            vibe="linear",
            duration=300,
            intensity=50,
            target_lufs=-12.9,
            variation=0.46,
            layers=["drums", "bass", "music", "pad", "synth"],
            synth_params={
                "cutoff": 58.0,
                "resonance": 16.0,
                "drive": 9.0,
                "space": 34.0,
            },
        ),
    ))

    return tracks

# =============================================================================
# RUNNER
# =============================================================================

def _pretty_payload_line(p: Dict[str, Any]) -> str:
    return f"{p['genre']} | {p['bpm']} BPM | key {p['key']} | curve {p['energy_curve']} | LUFS {p['target_lufs']} | int {p['intensity']}"

def main() -> None:
    print("\n🏆 SoundFlow WINNER PACK – PRO ENERGY GENERATOR 🏆")
    print(f"Date:   {DATE}")
    print(f"Server: {API_BASE}")
    print(f"Out:    {OUT_DIR.resolve()}")
    print(f"Format: {PREFERRED_FORMAT}")
    if FORCE_SEED_OVERRIDE:
        print(f"Seed:   FORCE_SEED_OVERRIDE={FORCE_SEED_OVERRIDE}")
    print()

    cases = build_winner_pack()
    total = len(cases)
    print(f"🚀 Generating {total} “winner candidates”…\n")

    manifest: List[Dict[str, Any]] = []

    for idx, case in enumerate(cases, 1):
        tid = case.id
        payload = dict(case.payload)  # copy so we can change seed on retries
        print(f"[{idx}/{total}] 🎼 {tid}")
        print(f"    {case.notes}")
        print(f"    { _pretty_payload_line(payload) }")

        last_err: Optional[str] = None
        meta: Optional[dict] = None

        for attempt in range(RETRIES_PER_TRACK + 1):
            try:
                start = time.time()
                meta = _http_post_json(f"{API_BASE}/api/generate", payload)
                dur = time.time() - start

                url = meta.get("url")
                if not url:
                    raise RuntimeError(f"API response missing 'url': {meta}")

                out_name = f"{tid}.{PREFERRED_FORMAT}"
                out_path = OUT_DIR / out_name
                _download_file(url, out_path)

                size_mb = out_path.stat().st_size / (1024 * 1024)
                print(f"    ✅ DONE in {dur:.1f}s | {size_mb:.2f} MB")
                print(f"       Saved: {out_path}\n")

                manifest.append({
                    "id": tid,
                    "notes": case.notes,
                    "payload": payload,
                    "meta": meta,
                    "file": str(out_path),
                })
                last_err = None
                break

            except urllib.error.HTTPError as e:
                last_err = f"HTTP {e.code} {e.reason}"
                try:
                    body = e.read().decode("utf-8", errors="replace")
                    last_err += f" | {body[:400]}"
                except Exception:
                    pass

            except Exception as e:
                last_err = str(e)

            # retry: new seed
            if attempt < RETRIES_PER_TRACK:
                payload["seed"] = _seed(
                    name=payload["seed"],
                    tag=payload["genre"].lower().replace(" ", "_"),
                    salt=f"retry{attempt}:{time.time()}",
                )
                payload["force_new"] = True
                print(f"    ⚠️  Attempt {attempt+1} failed: {last_err}")
                print(f"    🔁 Retrying with new seed...\n")
            else:
                print(f"    ❌ FAILED: {last_err}\n")

    # Write a manifest so you can keep/lock best seeds later
    manifest_path = OUT_DIR / f"{DATE}-winner_pack_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print("🎉 Finished.")
    print(f"📌 Manifest written: {manifest_path}")
    print("Next step: listen and keep the best seeds from manifest['payload']['seed'].\n")

if __name__ == "__main__":
    main()
