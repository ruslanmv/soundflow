#!/usr/bin/env python3
# generator/tests/test_api.py
"""
SoundFlow API – Full Integration Smoke Test (Dynamic-Stem Edition)

✅ What this version adds (to force NEW music / NEW stems):
- Every request uses a *unique seed* (seed includes UTC timestamp + nonce)
- Every request varies:
  - key, bpm micro-variation, layers order
  - synth_params
  - ambience (even for music) to force texture regeneration
- Optional: "force_new": true (ignored by server if not implemented, harmless)
- Optional: can run each test multiple times (RUNS_PER_CASE) to prove diversity

This validates:
- FastAPI backend is reachable
- /api/health OK
- /api/generate works across genres + modes
- Focus + hybrid engines work
- MP3 URLs download and look like MP3
- MP3s are non-empty

Note:
- To truly get different music, your backend MUST pass request.seed into remix_daily.build_track()
  and remix_daily must call render_stem(... seed=seed ...). (Your updated remix_daily does.)
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

# Guard against silent/empty outputs (tune if your mp3 bitrate changes a lot)
MIN_MP3_SIZE_BYTES = 30_000

DEFAULT_TIMEOUT_SEC = 180
DATE = datetime.now().strftime("%Y-%m-%d")

# Run each test case multiple times to verify it really produces different music
RUNS_PER_CASE = 1  # set to 2 or 3 to prove diversity

# If True, we compute a quick content fingerprint of each MP3 and warn on duplicates
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
    """
    Unique per request, guaranteed.
    Also includes PID + a random nonce to avoid collisions in parallel runs.
    """
    nonce = random.randint(0, 2**31 - 1)
    return f"{prefix}-{DATE}-{test_id}-run{run_index}-{_utc_stamp()}-pid{os.getpid()}-n{nonce}"


def _mp3_fingerprint(path: Path, bytes_to_hash: int = 128_000) -> str:
    """
    Fast, cheap "are these files basically identical" check.
    Hashes the first N bytes (most encoders include deterministic headers but audio frames differ too).
    """
    data = path.read_bytes()
    chunk = data[: min(len(data), bytes_to_hash)]
    return hashlib.sha256(chunk).hexdigest()


@dataclass
class TestCase:
    id: str
    payload: Dict[str, Any]


def run_case(case: TestCase, run_index: int = 1) -> Path:
    print(f"\n🎛️  API TEST: {case.id} (run {run_index}/{RUNS_PER_CASE})")

    meta = _http_post_json(f"{API_BASE}/api/generate", case.payload)

    mp3_url = meta.get("url")
    if not mp3_url or not isinstance(mp3_url, str):
        raise RuntimeError(f"Invalid response: missing 'url'. Got: {meta}")

    filename = meta.get("filename") or f"{case.id}.mp3"
    filename = _safe_filename(str(filename))

    # Make output file name unique even if backend returns same filename
    out_path = OUT_DIR / f"{case.id}__run{run_index}__{filename}"

    _download_file(mp3_url, out_path)

    size = out_path.stat().st_size
    if size < MIN_MP3_SIZE_BYTES:
        raise RuntimeError(f"MP3 too small ({size} bytes). File: {out_path}")

    if not _looks_like_mp3(out_path):
        raise RuntimeError(f"Downloaded file doesn't look like MP3: {out_path}")

    print(f"✅ OK: {out_path.name} ({size} bytes)")
    return out_path


# =============================================================================
# PAYLOAD BUILDERS (DYNAMIC STEM FORCING)
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
    """
    This version intentionally injects per-request variability to force new stems.
    """
    rnd = random.Random(f"{DATE}:{test_id}:{run_index}:{time.time_ns()}")
    seed = _make_unique_seed("api", test_id, run_index)

    # micro-variation: bpm/key/curve/layers order
    bpm2 = _jitter_int(bpm, 0.03, rnd, lo=60, hi=200)
    key2 = rnd.choice(_KEYS) if rnd.random() < 0.45 else key
    curve2 = rnd.choice(_ENERGY) if rnd.random() < 0.20 else energy_curve
    layers2 = layers[:]
    rnd.shuffle(layers2)

    # synth params vary per request (0..100)
    cutoff = _jitter_float(float(synth.get("cutoff", 75)), 12.0, rnd, lo=0.0, hi=100.0)
    resonance = _jitter_float(float(synth.get("resonance", 30)), 10.0, rnd, lo=0.0, hi=100.0)
    drive = _jitter_float(float(synth.get("drive", 10)), 12.0, rnd, lo=0.0, hi=100.0)
    space = _jitter_float(float(synth.get("space", 20)), 12.0, rnd, lo=0.0, hi=100.0)

    # Add subtle ambience sometimes to force texture generation (even in music mode)
    rain = 0
    vinyl = 0
    white = 0
    if rnd.random() < 0.35:
        vinyl = rnd.randint(5, 35)
    if rnd.random() < 0.20:
        rain = rnd.randint(5, 35)
    if rnd.random() < 0.20:
        white = rnd.randint(5, 25)

    return {
        "mode": "music",
        "channels": channels,
        "variation": float(variation),
        "genre": genre,
        "bpm": int(bpm2),
        "key": str(key2),
        "seed": seed,  # ✅ UNIQUE PER REQUEST
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

        # Optional "hint" field (server will ignore unless you implement it)
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
        "duration": int(duration),  # must be >= 30 per your server model
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
        "seed": seed,  # ✅ UNIQUE PER REQUEST

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

def build_test_cases(duration_music: int = 30, duration_focus: int = 30) -> List[TestCase]:
    """
    Genres list expanded and aligned with your new routing (Trance etc).
    Note: Your backend currently accepts any genre string; the generator must implement differences.
    """
    cases: List[TestCase] = []

    genres = [
        ("techno",   "Techno",   130, "A", ["drums", "bass", "music"], "peak"),
        ("house",    "House",    124, "A", ["drums", "bass", "music"], "linear"),
        ("deep",     "Deep",     122, "A", ["drums", "bass", "music"], "linear"),
        ("edm",      "EDM",      128, "A", ["drums", "bass", "music"], "peak"),
        ("trance",   "Trance",   138, "A", ["drums", "bass", "music"], "peak"),
        ("lounge",   "Lounge",   110, "C", ["drums", "bass", "music", "texture"], "linear"),
        ("chillout", "Chillout",  95, "C", ["drums", "music", "texture"], "linear"),
        ("ambient",  "Ambient",   70, "C", ["music", "texture"], "linear"),
        ("bass",     "Bass",     140, "A", ["drums", "bass"], "drop"),
        ("dance",    "Dance",    128, "A", ["drums", "bass", "music"], "peak"),
        ("hard",     "Hard",     150, "A", ["drums", "bass"], "peak"),
        ("synth",    "Synth",    105, "A", ["drums", "bass", "music"], "linear"),
        ("classic",  "Classic",   90, "C", ["music"], "linear"),
        ("vocal",    "Vocal",   128, "A", ["drums", "bass", "music"], "peak"),
        ("lofi",     "Lofi",      85, "C", ["drums", "music", "texture"], "linear"),
        ("jazz",     "Jazz",     120, "F", ["drums", "bass", "music"], "linear"),
    ]

    variation_sweeps = [
        ("vlow", 0.05, 35, {"cutoff": 80, "resonance": 15, "drive": 5,  "space": 15}),
        ("vmed", 0.25, 60, {"cutoff": 85, "resonance": 25, "drive": 15, "space": 25}),
        ("vhi",  0.50, 80, {"cutoff": 65, "resonance": 40, "drive": 25, "space": 35}),
    ]

    for short_id, genre, bpm, key, layers, curve in genres:
        for v_id, variation, intensity, synth in variation_sweeps:
            test_id = f"{short_id}_{v_id}"
            # seed will be injected per-run in make_music_payload
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
                        run_index=1,  # overwritten per-run later
                    ),
                )
            )

    # Focus modes (duration must be >= 30 to avoid 422)
    cases.append(
        TestCase(
            id="focus_focus_rain",
            payload=make_focus_payload(
                test_id="focus_focus_rain",
                focus_mode="focus",
                duration=duration_focus,
                ambience={"rain": 70, "vinyl": 0, "white": 20},
                channels=2,
                run_index=1,
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
                run_index=1,
            ),
        )
    )

    # Hybrid
    cases.append(
        TestCase(
            id="hybrid_lofi_relax",
            payload=make_hybrid_payload(
                test_id="hybrid_lofi_relax",
                genre="Lofi",
                bpm=80,
                layers=["music", "texture", "drums"],
                duration=duration_music,
                focus_mode="relax",
                ambience={"rain": 40, "vinyl": 40, "white": 0},
                focus_mix=35,
                variation=0.25,
                intensity=40,
                synth={"cutoff": 55, "resonance": 15, "drive": 10, "space": 45},
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
    base_cases = build_test_cases(duration_music=30, duration_focus=30)
    failures = 0
    produced: List[Path] = []
    fingerprints: Dict[str, str] = {}  # fp -> filename

    total_runs = len(base_cases) * RUNS_PER_CASE
    run_counter = 0

    for case in base_cases:
        for run_index in range(1, RUNS_PER_CASE + 1):
            run_counter += 1
            print(f"\n[{run_counter}/{total_runs}]")

            # Rebuild payload with per-run unique seed + per-run jitter
            # We detect mode to call the right builder.
            payload = dict(case.payload)
            mode = payload.get("mode", "music")

            try:
                if mode == "music":
                    payload = make_music_payload(
                        test_id=case.id,
                        genre=str(payload.get("genre", "Techno")),
                        bpm=int(payload.get("bpm", 128)),
                        key=str(payload.get("key", "A")),
                        layers=list(payload.get("layers", ["drums", "bass", "music"])),
                        energy_curve=str(payload.get("energy_curve", "linear")),
                        duration=int(payload.get("duration", 30)),
                        variation=float(payload.get("variation", 0.25)),
                        intensity=int(payload.get("intensity", 50)),
                        synth=dict(payload.get("synth_params", {})),
                        channels=int(payload.get("channels", 2)),
                        run_index=run_index,
                    )
                elif mode == "focus":
                    payload = make_focus_payload(
                        test_id=case.id,
                        focus_mode=str(payload.get("focus_mode", "focus")),
                        duration=int(payload.get("duration", 30)),
                        ambience=dict(payload.get("ambience", {})),
                        channels=int(payload.get("channels", 2)),
                        run_index=run_index,
                    )
                elif mode == "hybrid":
                    payload = make_hybrid_payload(
                        test_id=case.id,
                        genre=str(payload.get("genre", "Lofi")),
                        bpm=int(payload.get("bpm", 80)),
                        layers=list(payload.get("layers", ["music", "texture"])),
                        duration=int(payload.get("duration", 30)),
                        focus_mode=str(payload.get("focus_mode", "relax")),
                        ambience=dict(payload.get("ambience", {})),
                        focus_mix=int(payload.get("focus_mix", 35)),
                        variation=float(payload.get("variation", 0.25)),
                        intensity=int(payload.get("intensity", 40)),
                        synth=dict(payload.get("synth_params", {})),
                        channels=int(payload.get("channels", 2)),
                        run_index=run_index,
                    )

                out = run_case(TestCase(id=case.id, payload=payload), run_index=run_index)
                produced.append(out)

                if CHECK_DUPLICATE_AUDIO:
                    fp = _mp3_fingerprint(out)
                    if fp in fingerprints:
                        print(f"⚠️  WARNING: MP3 fingerprint duplicate with {fingerprints[fp]} (possible identical output)")
                    else:
                        fingerprints[fp] = out.name

            except Exception as e:
                failures += 1
                print(f"❌ FAILED [{case.id} run {run_index}]: {e}")

            # avoid hammering ffmpeg
            time.sleep(0.35)

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
        print(f"Largest MP3: {biggest.name} ({biggest.stat().st_size} bytes)")
    if CHECK_DUPLICATE_AUDIO and produced:
        print(f"Unique fingerprints: {len(fingerprints)} / {len(produced)}")

    if failures:
        print("\n❌ Some API tests failed.")
        return 1

    print("\n✅ ALL API TESTS PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
