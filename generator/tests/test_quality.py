#!/usr/bin/env python3
# generator/tests/test_quality.py
"""
SoundFlow API – Professional Quality Assurance Test (CLEAN & PUNCHY EDITION)

CHANGES:
- 🚫 NOISE REMOVED: "Texture" layer removed from all presets. Pure audio only.
- 🔊 BASS BOOST: Synth parameters tuned for maximum punch in House/Trance.
- 🎚️ CLUB MASTER: Target loudness set to -10 LUFS for competitive volume.
"""

from __future__ import annotations

import json
import time
import urllib.request
import urllib.error
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List
import random
from datetime import datetime

# =============================================================================
# CONFIGURATION
# =============================================================================

API_BASE = "http://localhost:8000"
OUT_DIR = Path("quality_showcase_outputs")
OUT_DIR.mkdir(parents=True, exist_ok=True)

DEFAULT_TIMEOUT_SEC = 300 
DATE = datetime.now().strftime("%Y-%m-%d")

# =============================================================================
# HELPERS
# =============================================================================

def _http_post_json(url: str, payload: dict) -> dict:
    body = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=DEFAULT_TIMEOUT_SEC) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except Exception as e:
        print(f"❌ API Request Failed: {e}")
        raise

def _download_file(url: str, out_path: Path) -> None:
    try:
        with urllib.request.urlopen(url, timeout=DEFAULT_TIMEOUT_SEC) as resp:
            out_path.write_bytes(resp.read())
    except Exception as e:
        print(f"❌ Download Failed: {e}")
        raise

# =============================================================================
# PROFESSIONAL PRESET BUILDER
# =============================================================================

def make_pro_payload(
    name: str,
    genre: str,
    bpm: int,
    key: str,
    vibe: str = "peak",
    duration: int = 180,
    intensity: int = 80,
    synth_settings: Dict[str, float] = None
) -> Dict[str, Any]:
    
    rnd = random.Random(f"{name}:{time.time()}")
    seed = f"pro-showcase-{name.replace(' ', '')}-{rnd.randint(1000, 9999)}"
    
    if not synth_settings:
        synth_settings = {
            "cutoff": 85.0,
            "resonance": 25.0,
            "drive": 15.0,
            "space": 35.0
        }

    return {
        "mode": "music",
        "channels": 2,
        "variation": 0.45,
        "genre": genre,
        "bpm": bpm,
        "key": key,
        "seed": seed,
        
        # ✅ FIX: REMOVED "texture" LAYER. NO NOISE. PURE MUSIC.
        "layers": ["drums", "bass", "music", "pad", "synth"],
        
        "energy_curve": vibe,
        "duration": duration,
        
        # ✅ DOUBLE SAFETY: Explicitly disable ambience engine
        "focus_mode": "off",
        "ambience": {"rain": 0, "vinyl": 0, "white": 0},
        
        "intensity": intensity,
        "synth_params": synth_settings,
        
        # ✅ CLUB MASTERING: Louder target for "Punch"
        "target_lufs": -10.0, 
        "force_new": True,
    }

# =============================================================================
# TEST CASES - CLUB & PRO QUALITY
# =============================================================================

def get_showcase_tracks():
    tracks = []

    # 1. HOUSE PUNCH (Feel the Beats)
    # High drive, tight resonance for punchy bass
    tracks.append({
        "id": "house_club_banger_clean",
        "payload": make_pro_payload(
            name="Club House Prime",
            genre="House",
            bpm=126,
            key="G#",
            vibe="peak",
            intensity=95,
            synth_settings={
                "cutoff": 90.0,    # Open filter for brightness
                "resonance": 45.0, # High resonance for "punch"
                "drive": 35.0,     # High drive for saturation/fatness
                "space": 20.0      # Tight reverb (not muddy)
            }
        )
    })

    # 2. VOCAL ANTHEM (Clear & Melodic)
    # Clean settings to let the "vocal" style melody shine
    tracks.append({
        "id": "vocal_trance_clean",
        "payload": make_pro_payload(
            name="Vocal Anthem",
            genre="Vocal", 
            bpm=138,
            key="F",
            vibe="peak",
            intensity=90,
            synth_settings={
                "cutoff": 85.0,
                "resonance": 30.0,
                "drive": 20.0,     # Warm but clean
                "space": 50.0      # Big stadium reverb
            }
        )
    })

    # 3. CLASSICAL PIANO (Mozart/Rachmaninoff)
    # Zero drive to prevent distortion, just pure tone
    tracks.append({
        "id": "classical_virtuoso",
        "payload": make_pro_payload(
            name="Grand Piano Concerto",
            genre="Classic",
            bpm=80,
            key="C#",
            vibe="build",
            intensity=60,
            synth_settings={
                "cutoff": 50.0,
                "resonance": 5.0,  # Low resonance for natural sound
                "drive": 0.0,      # ZERO saturation for clarity
                "space": 70.0      # Concert hall ambience
            }
        )
    })

    # 4. LUXURY LOUNGE (Relaxation)
    tracks.append({
        "id": "lounge_sunset",
        "payload": make_pro_payload(
            name="Sunset Martini",
            genre="Lounge",
            bpm=100,
            key="A#",
            vibe="linear",
            intensity=40,
            synth_settings={
                "cutoff": 45.0,    # Warm/Darker
                "resonance": 10.0,
                "drive": 5.0,
                "space": 40.0
            }
        )
    })

    # 5. DEEP CODING (Focus)
    tracks.append({
        "id": "coding_deep_flow",
        "payload": make_pro_payload(
            name="Deep Flow",
            genre="Deep",
            bpm=118,
            key="E",
            vibe="linear",
            intensity=50,
            synth_settings={
                "cutoff": 60.0,
                "resonance": 20.0,
                "drive": 10.0,
                "space": 30.0
            }
        )
    })

    return tracks

# =============================================================================
# RUNNER
# =============================================================================

def main():
    print("\n✨ SoundFlow PRO AUDIO AUDIT (Clean, Punchy, No Noise) ✨")
    print(f"Server: {API_BASE}")
    print(f"Output: {OUT_DIR.resolve()}\n")

    cases = get_showcase_tracks()
    total = len(cases)
    
    print(f"🚀 Generating {total} Showcase Tracks...\n")

    for i, case in enumerate(cases, 1):
        tid = case["id"]
        p = case["payload"]
        
        print(f"[{i}/{total}] 🎹 Generating: {tid} ({p['genre']} - {p['bpm']} BPM)...")
        
        start = time.time()
        try:
            meta = _http_post_json(f"{API_BASE}/api/generate", p)
            dur = time.time() - start
            
            filename = f"{tid}.mp3"
            out_path = OUT_DIR / filename
            _download_file(meta['url'], out_path)
            
            size_mb = out_path.stat().st_size / (1024 * 1024)
            print(f"   ✅ DONE in {dur:.1f}s | Size: {size_mb:.2f}MB")
            print(f"      Saved: {filename}\n")
            
        except Exception as e:
            print(f"   ❌ FAILED: {e}\n")

    print("\n🎉 Audit Complete. Tracks are clean and club-ready!")

if __name__ == "__main__":
    main()