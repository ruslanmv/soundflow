#!/usr/bin/env python3
# generator/tests/test_api.py
"""
SoundFlow API – Full Integration Smoke Test

What this tests:
- FastAPI backend is reachable
- /api/health is OK
- /api/generate works across genres + modes
- Focus + hybrid engines work
- MP3s are non-empty and look like real MP3s
- Returned URL is fetchable and produces bytes

This is NOT an audio quality test.
"""

from __future__ import annotations

import json
import sys
import time
import urllib.request
import urllib.error
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List
from datetime import datetime


# =============================================================================
# CONFIG
# =============================================================================

API_BASE = "http://localhost:8000"
OUT_DIR = Path("api_test_outputs")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Guard against silent/empty outputs (tune if your mp3 bitrate changes a lot)
MIN_MP3_SIZE_BYTES = 30_000

DEFAULT_TIMEOUT_SEC = 180
DATE = datetime.now().strftime("%Y-%m-%d")


# =============================================================================
# HELPERS
# =============================================================================

def _http_get_json(url: str, timeout: int = DEFAULT_TIMEOUT_SEC) -> dict:
    req = urllib.request.Request(url, method="GET")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = resp.read()
            return json.loads(data.decode("utf-8"))
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", errors="replace") if e.fp else ""
        raise RuntimeError(f"GET {url} failed: {e.code} {e.reason}\n{body}") from e
    except Exception as e:
        raise RuntimeError(f"GET {url} failed: {e}") from e


def _http_post_json(url: str, payload: dict, timeout: int = DEFAULT_TIMEOUT_SEC) -> dict:
    body = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = resp.read()
            return json.loads(data.decode("utf-8"))
    except urllib.error.HTTPError as e:
        body_txt = e.read().decode("utf-8", errors="replace") if e.fp else ""
        raise RuntimeError(f"POST {url} failed: {e.code} {e.reason}\n{body_txt}") from e
    except Exception as e:
        raise RuntimeError(f"POST {url} failed: {e}") from e


def _download_file(url: str, out_path: Path, timeout: int = DEFAULT_TIMEOUT_SEC) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)

    req = urllib.request.Request(url, method="GET")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            chunk = resp.read()
            out_path.write_bytes(chunk)
    except urllib.error.HTTPError as e:
        body_txt = e.read().decode("utf-8", errors="replace") if e.fp else ""
        raise RuntimeError(f"Download failed: {e.code} {e.reason} for {url}\n{body_txt}") from e
    except Exception as e:
        raise RuntimeError(f"Download failed for {url}: {e}") from e


def _looks_like_mp3(path: Path) -> bool:
    """
    Lightweight validation:
    - MP3 often starts with "ID3" tag OR frame sync 0xFF 0xFB/0xF3/0xF2
    """
    data = path.read_bytes()
    if len(data) < 4:
        return False
    if data[:3] == b"ID3":
        return True
    # frame sync check
    b0, b1 = data[0], data[1]
    if b0 == 0xFF and (b1 & 0xE0) == 0xE0:
        return True
    return False


def _safe_filename(name: str) -> str:
    # remove characters that can confuse filesystems or URLs
    bad = ['/', '\\', ':', '*', '?', '"', '<', '>', '|']
    for ch in bad:
        name = name.replace(ch, "_")
    return name


@dataclass
class TestCase:
    id: str
    payload: Dict[str, Any]


def run_case(case: TestCase) -> Path:
    print(f"\n🎛️  API TEST: {case.id}")

    meta = _http_post_json(f"{API_BASE}/api/generate", case.payload)

    # server returns absolute URL now
    mp3_url = meta.get("url")
    if not mp3_url or not isinstance(mp3_url, str):
        raise RuntimeError(f"Invalid response: missing 'url'. Got: {meta}")

    filename = meta.get("filename") or f"{case.id}.mp3"
    filename = _safe_filename(str(filename))
    out_path = OUT_DIR / filename

    _download_file(mp3_url, out_path)

    size = out_path.stat().st_size
    if size < MIN_MP3_SIZE_BYTES:
        raise RuntimeError(f"MP3 too small ({size} bytes). File: {out_path}")

    if not _looks_like_mp3(out_path):
        raise RuntimeError(f"Downloaded file doesn't look like MP3: {out_path}")

    print(f"✅ OK: {out_path.name} ({size} bytes)")
    return out_path


# =============================================================================
# TEST MATRIX BUILDER
# =============================================================================

def make_music_payload(
    *,
    test_id: str,
    genre: str,
    bpm: int,
    key: str,
    layers: List[str],
    energy_curve: str,
    duration: int,
    variation: float,
    intensity: int,
    synth: Dict[str, float],
    channels: int = 2,
) -> Dict[str, Any]:
    return {
        "mode": "music",
        "channels": channels,
        "variation": float(variation),
        "genre": genre,
        "bpm": int(bpm),
        "key": key,
        "seed": f"api-{DATE}-{test_id}",
        "layers": layers,
        "energy_curve": energy_curve,
        "duration": int(duration),
        "focus_mode": "off",
        "ambience": {"rain": 0, "vinyl": 0, "white": 0},
        "intensity": int(intensity),
        "synth_params": {
            "cutoff": float(synth.get("cutoff", 75)),
            "resonance": float(synth.get("resonance", 30)),
            "drive": float(synth.get("drive", 10)),
            "space": float(synth.get("space", 20)),
        },
        "target_lufs": -14.0,
    }


def make_focus_payload(
    *,
    test_id: str,
    focus_mode: str,
    duration: int,
    ambience: Dict[str, int],
    channels: int = 2,
) -> Dict[str, Any]:
    return {
        "mode": "focus",
        "channels": channels,
        "variation": 0.0,
        "genre": "Ambient",
        "bpm": 60,
        "key": "A",
        "seed": f"api-{DATE}-{test_id}",
        "layers": [],
        "energy_curve": "linear",
        "duration": int(duration),
        "focus_mode": focus_mode,
        "ambience": {
            "rain": int(ambience.get("rain", 0)),
            "vinyl": int(ambience.get("vinyl", 0)),
            "white": int(ambience.get("white", 0)),
        },
        "intensity": 0,
        "synth_params": {"cutoff": 75, "resonance": 30, "drive": 0, "space": 0},
        "target_lufs": -14.0,
    }


def make_hybrid_payload(
    *,
    test_id: str,
    genre: str,
    bpm: int,
    layers: List[str],
    duration: int,
    focus_mode: str,
    ambience: Dict[str, int],
    focus_mix: int,
    variation: float,
    intensity: int,
    synth: Dict[str, float],
    channels: int = 2,
) -> Dict[str, Any]:
    return {
        "mode": "hybrid",
        "channels": channels,
        "focus_mix": int(focus_mix),
        "variation": float(variation),

        "genre": genre,
        "bpm": int(bpm),
        "key": "A",
        "seed": f"api-{DATE}-{test_id}",

        "layers": layers,
        "energy_curve": "linear",
        "duration": int(duration),

        "focus_mode": focus_mode,
        "ambience": {
            "rain": int(ambience.get("rain", 0)),
            "vinyl": int(ambience.get("vinyl", 0)),
            "white": int(ambience.get("white", 0)),
        },

        "intensity": int(intensity),
        "synth_params": {
            "cutoff": float(synth.get("cutoff", 70)),
            "resonance": float(synth.get("resonance", 20)),
            "drive": float(synth.get("drive", 10)),
            "space": float(synth.get("space", 30)),
        },
        "target_lufs": -14.0,
    }


def build_test_cases(duration_music: int = 30, duration_focus: int = 30) -> List[TestCase]:
    """
    Covers:
    - genres: Techno, House, Lofi, Jazz, Trance, Euro, Hard, Bass
    - variations: low/med/high
    - channels: stereo
    - focus and hybrid
    """
    cases: List[TestCase] = []

    # A) Music genres (use layer sets that your remix_daily mapping can actually fulfill)
    genres = [
        ("techno", "Techno", 130, "A", ["drums", "bass", "music"], "peak"),
        ("house",  "House",  124, "A", ["drums", "bass", "music"], "linear"),
        ("lofi",   "Lofi",    85, "C", ["drums", "music", "texture"], "linear"),
        ("jazz",   "Jazz",   120, "F", ["drums", "bass", "music"], "linear"),
        ("trance", "Trance", 138, "A", ["drums", "bass", "music"], "peak"),
        ("euro",   "Euro",   140, "A", ["drums", "music"], "peak"),
        ("hard",   "Hard",   150, "A", ["drums", "bass"], "peak"),
        ("bass",   "Bass",   140, "A", ["drums", "bass"], "drop"),
    ]

    variation_sweeps = [
        ("vlow", 0.05, 35, {"cutoff": 80, "resonance": 15, "drive": 5,  "space": 15}),
        ("vmed", 0.25, 60, {"cutoff": 85, "resonance": 25, "drive": 15, "space": 25}),
        ("vhi",  0.50, 80, {"cutoff": 65, "resonance": 40, "drive": 25, "space": 35}),
    ]

    for short_id, genre, bpm, key, layers, curve in genres:
        for v_id, variation, intensity, synth in variation_sweeps:
            test_id = f"{short_id}_{v_id}"
            cases.append(
                TestCase(
                    id=test_id,
                    payload=make_music_payload(
                        test_id=test_id,
                        genre=genre,
                        bpm=bpm,
                        key=key,
                        layers=layers,
                        energy_curve=curve,
                        duration=duration_music,
                        variation=variation,
                        intensity=intensity,
                        synth=synth,
                        channels=2,
                    ),
                )
            )

    # B) Focus modes
    cases.append(
        TestCase(
            id="focus_focus_rain",
            payload=make_focus_payload(
                test_id="focus_focus_rain",
                focus_mode="focus",
                duration=duration_focus,
                ambience={"rain": 70, "vinyl": 0, "white": 20},
                channels=2,
            ),
        )
    )
    cases.append(
        TestCase(
            id="focus_relax_vinyl",
            payload=make_focus_payload(
                test_id="focus_relax_vinyl",
                focus_mode="relax",
                duration=duration_focus,
                ambience={"rain": 0, "vinyl": 60, "white": 10},
                channels=2,
            ),
        )
    )

    # C) Hybrid
    cases.append(
        TestCase(
            id="hybrid_lofi_relax",
            payload=make_hybrid_payload(
                test_id="hybrid_lofi_relax",
                genre="Lofi",
                bpm=80,
                layers=["music", "texture"],
                duration=duration_music,
                focus_mode="relax",
                ambience={"rain": 40, "vinyl": 40, "white": 0},
                focus_mix=35,
                variation=0.25,
                intensity=40,
                synth={"cutoff": 55, "resonance": 15, "drive": 10, "space": 45},
                channels=2,
            ),
        )
    )

    return cases


# =============================================================================
# MAIN
# =============================================================================

def main() -> int:
    print("\n🎵 SoundFlow API – Full Integration Smoke Test")
    print(f"API: {API_BASE}")
    print(f"Output dir: {OUT_DIR.resolve()}")

    # 1) health check
    try:
        health = _http_get_json(f"{API_BASE}/api/health", timeout=30)
        print(f"✅ Health: status={health.get('status')} tracks={health.get('tracks_count')}")
    except Exception as e:
        print(f"❌ Health check failed: {e}")
        return 2

    # 2) run test cases
    cases = build_test_cases()
    failures = 0
    produced: List[Path] = []

    for i, case in enumerate(cases, start=1):
        try:
            print(f"\n[{i}/{len(cases)}]")
            out = run_case(case)
            produced.append(out)
        except Exception as e:
            failures += 1
            print(f"❌ FAILED [{case.id}]: {e}")

        # avoid hammering ffmpeg
        time.sleep(0.4)

    # 3) summary
    print("\n====================")
    print("📦 Test Summary")
    print("====================")
    print(f"Total: {len(cases)}")
    print(f"Passed: {len(cases) - failures}")
    print(f"Failed: {failures}")
    print(f"Output: {OUT_DIR.resolve()}")
    if produced:
        biggest = max(produced, key=lambda p: p.stat().st_size)
        print(f"Largest MP3: {biggest.name} ({biggest.stat().st_size} bytes)")

    if failures:
        print("\n❌ Some API tests failed.")
        return 1

    print("\n✅ ALL API TESTS PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
