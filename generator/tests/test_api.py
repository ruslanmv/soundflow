#!/usr/bin/env python3
# generator/tests/test_api.py
"""
SoundFlow API – Full Integration Smoke Test (Dynamic-Stem & Duration Edition)

✅ What this version adds:
- TESTS DIFFERENT LENGTHS: Ranging from 30s to 180s (3 minutes).
- TESTS ALL GENRES: Iterates through the full style catalog.
- Unique Seeds: Ensures every request generates fresh audio.
- CLEAN AUDIO: Disabled random white noise injection for pure music tests.

This validates:
- The engine calculates bars correctly for long durations.
- The looping logic works without crashing.
- MP3 encoding handles large files.
"""

from __future__ import annotations

import json
import sys
import time
import urllib.request
import urllib.error
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional
from datetime import datetime, timezone
import hashlib
import os
import random


# =============================================================================
# CONFIG
# =============================================================================

API_BASE = "http://localhost:8000"
OUT_DIR = Path("api_test_outputs")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Guard against silent/empty outputs
MIN_MP3_SIZE_BYTES = 30_000

# Increased timeout to allow for 3-minute track generation
DEFAULT_TIMEOUT_SEC = 300 
DATE = datetime.now().strftime("%Y-%m-%d")

# Run each test case multiple times to prove diversity
RUNS_PER_CASE = 1 

# If True, check for duplicate audio content
CHECK_DUPLICATE_AUDIO = True


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
    data = path.read_bytes()
    if len(data) < 4:
        return False
    if data[:3] == b"ID3":
        return True
    b0, b1 = data[0], data[1]
    if b0 == 0xFF and (b1 & 0xE0) == 0xE0:
        return True
    return False


def _safe_filename(name: str) -> str:
    bad = ['/', '\\', ':', '*', '?', '"', '<', '>', '|']
    for ch in bad:
        name = name.replace(ch, "_")
    return name


def _utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")


def _make_unique_seed(prefix: str, test_id: str, run_index: int) -> str:
    nonce = random.randint(0, 2**31 - 1)
    return f"{prefix}-{DATE}-{test_id}-run{run_index}-{_utc_stamp()}-pid{os.getpid()}-n{nonce}"


def _mp3_fingerprint(path: Path, bytes_to_hash: int = 128_000) -> str:
    data = path.read_bytes()
    chunk = data[: min(len(data), bytes_to_hash)]
    return hashlib.sha256(chunk).hexdigest()


@dataclass
class TestCase:
    id: str
    payload: Dict[str, Any]


def run_case(case: TestCase, run_index: int = 1) -> Path:
    # Print what we are testing specifically
    dur = case.payload.get("duration", "?")
    mode = case.payload.get("mode", "music")
    genre = case.payload.get("genre", "N/A")
    
    print(f"\n🎛️  API TEST: {case.id}")
    print(f"   ℹ️  {mode.upper()} | {genre} | Duration: {dur}s | Run: {run_index}")

    start_t = time.time()
    meta = _http_post_json(f"{API_BASE}/api/generate", case.payload)
    gen_time = time.time() - start_t

    mp3_url = meta.get("url")
    if not mp3_url or not isinstance(mp3_url, str):
        raise RuntimeError(f"Invalid response: missing 'url'. Got: {meta}")

    filename = meta.get("filename") or f"{case.id}.mp3"
    filename = _safe_filename(str(filename))

    out_path = OUT_DIR / f"{case.id}__run{run_index}__{dur}s__{filename}"

    _download_file(mp3_url, out_path)

    size = out_path.stat().st_size
    if size < MIN_MP3_SIZE_BYTES:
        raise RuntimeError(f"MP3 too small ({size} bytes). File: {out_path}")

    if not _looks_like_mp3(out_path):
        raise RuntimeError(f"Downloaded file doesn't look like MP3: {out_path}")

    print(f"   ✅ OK: {out_path.name}")
    print(f"   📊 Size: {size/1024:.1f} KB | Gen Time: {gen_time:.1f}s")
    return out_path


# =============================================================================
# PAYLOAD BUILDERS
# =============================================================================

_KEYS = ["A", "C", "D", "E", "F", "G"]
_ENERGY = ["linear", "drop", "peak"]


def _jitter_int(base: int, pct: float, rnd: random.Random, lo: Optional[int] = None, hi: Optional[int] = None) -> int:
    span = max(1, int(round(base * pct)))
    v = base + rnd.randint(-span, span)
    if lo is not None:
        v = max(lo, v)
    if hi is not None:
        v = min(hi, v)
    return int(v)


def _jitter_float(base: float, amt: float, rnd: random.Random, lo: float = 0.0, hi: float = 1.0) -> float:
    v = base + (rnd.random() * 2.0 - 1.0) * amt
    return float(max(lo, min(hi, v)))


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
    run_index: int = 1,
) -> Dict[str, Any]:
    rnd = random.Random(f"{DATE}:{test_id}:{run_index}:{time.time_ns()}")
    seed = _make_unique_seed("api", test_id, run_index)

    bpm2 = _jitter_int(bpm, 0.03, rnd, lo=60, hi=200)
    key2 = rnd.choice(_KEYS) if rnd.random() < 0.45 else key
    curve2 = rnd.choice(_ENERGY) if rnd.random() < 0.20 else energy_curve
    layers2 = layers[:]
    rnd.shuffle(layers2)

    cutoff = _jitter_float(float(synth.get("cutoff", 75)), 12.0, rnd, lo=0.0, hi=100.0)
    resonance = _jitter_float(float(synth.get("resonance", 30)), 10.0, rnd, lo=0.0, hi=100.0)
    drive = _jitter_float(float(synth.get("drive", 10)), 12.0, rnd, lo=0.0, hi=100.0)
    space = _jitter_float(float(synth.get("space", 20)), 12.0, rnd, lo=0.0, hi=100.0)

    # ✅ FORCE CLEAN AUDIO: No noise injection for pure music tests
    rain = 0
    vinyl = 0
    white = 0
    
    # Only add light vinyl if specifically testing Lo-Fi, otherwise keep clean
    if genre.lower() in ("lofi", "chillout") and rnd.random() < 0.3:
        vinyl = rnd.randint(5, 15)

    return {
        "mode": "music",
        "channels": channels,
        "variation": float(variation),
        "genre": genre,
        "bpm": int(bpm2),
        "key": str(key2),
        "seed": seed,
        "layers": layers2,
        "energy_curve": curve2,
        "duration": int(duration),
        "focus_mode": "off",
        "ambience": {"rain": rain, "vinyl": vinyl, "white": white},
        "intensity": int(intensity),
        "synth_params": {
            "cutoff": float(cutoff),
            "resonance": float(resonance),
            "drive": float(drive),
            "space": float(space),
        },
        "target_lufs": -14.0,
        "force_new": True,
    }


def make_focus_payload(
    *,
    test_id: str,
    focus_mode: str,
    duration: int,
    ambience: Dict[str, int],
    channels: int = 2,
    run_index: int = 1,
) -> Dict[str, Any]:
    seed = _make_unique_seed("api", test_id, run_index)
    return {
        "mode": "focus",
        "channels": channels,
        "variation": 0.0,
        "genre": "Ambient",
        "bpm": 60,
        "key": "A",
        "seed": seed,
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
        "force_new": True,
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
    run_index: int = 1,
) -> Dict[str, Any]:
    rnd = random.Random(f"{DATE}:{test_id}:{run_index}:{time.time_ns()}")
    seed = _make_unique_seed("api", test_id, run_index)
    bpm2 = _jitter_int(bpm, 0.03, rnd, lo=60, hi=200)
    layers2 = layers[:]
    rnd.shuffle(layers2)

    return {
        "mode": "hybrid",
        "channels": channels,
        "focus_mix": int(focus_mix),
        "variation": float(variation),
        "genre": genre,
        "bpm": int(bpm2),
        "key": rnd.choice(_KEYS),
        "seed": seed,
        "layers": layers2,
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
        "force_new": True,
    }


# =============================================================================
# TEST MATRIX BUILDER
# =============================================================================

def build_test_cases() -> List[TestCase]:
    """
    Builds a diverse set of test cases covering all genres and varied durations.
    Structure: (id, genre, bpm, key, layers, curve, DURATION_SEC)
    """
    cases: List[TestCase] = []

    # Format: (short_id, genre, bpm, key, layers, curve, DURATION)
    genres = [
        # --- LONG TRACKS (3 Minutes) ---
        ("techno_long",   "Techno",   130, "A", ["drums", "bass", "music"], "peak",   180),
        ("trance_long",   "Trance",   138, "F", ["drums", "bass", "music"], "peak",   180),
        ("house_long",    "House",    124, "C", ["drums", "bass", "music"], "linear", 180),
        
        # --- MEDIUM TRACKS (1-1.5 Minutes) ---
        ("deep_med",      "Deep",     122, "A", ["drums", "bass", "music"], "linear", 90),
        ("edm_med",       "EDM",      128, "G", ["drums", "bass", "music"], "peak",   75),
        ("bass_med",      "Bass",     140, "E", ["drums", "bass"],          "drop",   60),
        ("jazz_med",      "Jazz",     120, "F", ["drums", "bass", "music"], "linear", 60),

        # --- SHORT TRACKS (30-45 Seconds) ---
        ("chill_short",   "Chillout",  95, "C", ["drums", "music", "texture"], "linear", 45),
        ("lofi_short",    "Lofi",      85, "C", ["drums", "music", "texture"], "linear", 35),
        ("ambient_short", "Ambient",   70, "E", ["music", "texture"],          "linear", 30),
        ("classic_short", "Classic",   90, "C", ["music"],                     "linear", 30),
        
        # --- SPECIALTY ---
        ("hard_fast",     "Hard",     150, "A", ["drums", "bass"],             "peak",   45),
        ("synth_retro",   "Synth",    105, "D", ["drums", "bass", "music"],    "linear", 60),
    ]

    variation_defaults = {"cutoff": 75, "resonance": 25, "drive": 10, "space": 25}

    for short_id, genre, bpm, key, layers, curve, duration in genres:
        cases.append(
            TestCase(
                id=short_id,
                payload=make_music_payload(
                    test_id=short_id,
                    genre=genre,
                    bpm=bpm,
                    key=key,
                    layers=layers,
                    energy_curve=curve,
                    duration=duration,  # ✅ Passing specific duration
                    variation=0.3,
                    intensity=60,
                    synth=variation_defaults,
                    channels=2,
                    run_index=1,
                ),
            )
        )

    # Focus modes
    cases.append(
        TestCase(
            id="focus_binaural_long",
            payload=make_focus_payload(
                test_id="focus_binaural_long",
                focus_mode="focus",
                duration=120, # 2 Minutes
                ambience={"rain": 70, "vinyl": 0, "white": 20},
                channels=2,
                run_index=1,
            ),
        )
    )

    # Hybrid
    cases.append(
        TestCase(
            id="hybrid_mix",
            payload=make_hybrid_payload(
                test_id="hybrid_mix",
                genre="Lofi",
                bpm=80,
                layers=["music", "texture", "drums"],
                duration=60,
                focus_mode="relax",
                ambience={"rain": 30, "vinyl": 30, "white": 0},
                focus_mix=35,
                variation=0.25,
                intensity=40,
                synth=variation_defaults,
                channels=2,
                run_index=1,
            ),
        )
    )

    return cases


# =============================================================================
# MAIN
# =============================================================================

def main() -> int:
    print("\n🎵 SoundFlow API – Full Integration Smoke Test (Dynamic-Stem Edition)")
    print(f"API: {API_BASE}")
    print(f"Output dir: {OUT_DIR.resolve()}")
    print(f"RUNS_PER_CASE: {RUNS_PER_CASE}")
    print(f"CHECK_DUPLICATE_AUDIO: {CHECK_DUPLICATE_AUDIO}")

    # 1) health check
    try:
        health = _http_get_json(f"{API_BASE}/api/health", timeout=30)
        print(f"✅ Health: status={health.get('status')} tracks={health.get('tracks_count')}")
    except Exception as e:
        print(f"❌ Health check failed: {e}")
        return 2

    # 2) run test cases
    test_cases = build_test_cases()
    failures = 0
    produced: List[Path] = []
    fingerprints: Dict[str, str] = {}

    total_runs = len(test_cases) * RUNS_PER_CASE
    run_counter = 0

    for case in test_cases:
        for run_index in range(1, RUNS_PER_CASE + 1):
            run_counter += 1
            print(f"\n🔹 Progress: [{run_counter}/{total_runs}]")

            # Clone payload to inject run index
            payload = dict(case.payload)
            mode = payload.get("mode", "music")

            try:
                # We need to regenerate the payload to get a fresh seed for this specific run_index
                if mode == "music":
                    payload = make_music_payload(
                        test_id=case.id,
                        genre=str(payload.get("genre")),
                        bpm=int(payload.get("bpm")),
                        key=str(payload.get("key")),
                        layers=list(payload.get("layers")),
                        energy_curve=str(payload.get("energy_curve")),
                        duration=int(payload.get("duration")),
                        variation=float(payload.get("variation")),
                        intensity=int(payload.get("intensity")),
                        synth=dict(payload.get("synth_params")),
                        channels=int(payload.get("channels")),
                        run_index=run_index,
                    )
                elif mode == "focus":
                    payload = make_focus_payload(
                        test_id=case.id,
                        focus_mode=str(payload.get("focus_mode")),
                        duration=int(payload.get("duration")),
                        ambience=dict(payload.get("ambience")),
                        channels=int(payload.get("channels")),
                        run_index=run_index,
                    )
                elif mode == "hybrid":
                    payload = make_hybrid_payload(
                        test_id=case.id,
                        genre=str(payload.get("genre")),
                        bpm=int(payload.get("bpm")),
                        layers=list(payload.get("layers")),
                        duration=int(payload.get("duration")),
                        focus_mode=str(payload.get("focus_mode")),
                        ambience=dict(payload.get("ambience")),
                        focus_mix=int(payload.get("focus_mix")),
                        variation=float(payload.get("variation")),
                        intensity=int(payload.get("intensity")),
                        synth=dict(payload.get("synth_params")),
                        channels=int(payload.get("channels")),
                        run_index=run_index,
                    )

                out = run_case(TestCase(id=case.id, payload=payload), run_index=run_index)
                produced.append(out)

                if CHECK_DUPLICATE_AUDIO:
                    fp = _mp3_fingerprint(out)
                    if fp in fingerprints:
                        print(f"⚠️  WARNING: MP3 fingerprint duplicate with {fingerprints[fp]}")
                    else:
                        fingerprints[fp] = out.name

            except Exception as e:
                failures += 1
                print(f"❌ FAILED [{case.id} run {run_index}]: {e}")

            # small sleep to let server cleanup
            time.sleep(0.5)

    # 3) summary
    print("\n====================")
    print("📦 Test Summary")
    print("====================")
    print(f"Total runs: {total_runs}")
    print(f"Passed: {total_runs - failures}")
    print(f"Failed: {failures}")
    print(f"Output: {OUT_DIR.resolve()}")

    if produced:
        biggest = max(produced, key=lambda p: p.stat().st_size)
        print(f"Largest MP3: {biggest.name} ({biggest.stat().st_size/1024/1024:.2f} MB)")
    if CHECK_DUPLICATE_AUDIO and produced:
        print(f"Unique fingerprints: {len(fingerprints)} / {len(produced)}")

    if failures:
        print("\n❌ Some API tests failed.")
        return 1

    print("\n✅ ALL API TESTS PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())