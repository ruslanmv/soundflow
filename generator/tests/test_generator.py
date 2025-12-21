#!/usr/bin/env python3
# generator/tests/test_generator.py
"""
SoundFlow Generator – Full Smoke Test (Music + Focus)

Goals:
- Generate REAL music for each genre
- Exercise arranger, mixer, DSP, and focus engine
- Catch missing imports, broken routing, crashes
- Keep duration short for fast iteration / CI

This is NOT an audio quality test.
It is a stability + coverage test.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import json
from datetime import datetime, timezone
from pathlib import Path

# =============================================================================
# Minimal .env loader (no deps)
# =============================================================================

def load_dotenv_file(path: Path) -> None:
    if not path.exists():
        return

    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        k = k.strip()
        v = v.strip().strip('"').strip("'")
        if k and k not in os.environ:
            os.environ[k] = v


def utc_today() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d")


# =============================================================================
# Subprocess runner (PYTHONPATH-safe)
# =============================================================================

def run(cmd: list[str], cwd: Path | None = None) -> None:
    print("\n$ " + " ".join(cmd))

    env = os.environ.copy()

    # Force PYTHONPATH so `free.*` imports work everywhere
    generator_root = Path(__file__).resolve().parents[1]
    existing = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = str(generator_root) + (
        os.pathsep + existing if existing else ""
    )

    proc = subprocess.run(
        cmd,
        cwd=str(cwd) if cwd else None,
        env=env,
        text=True,
    )

    if proc.returncode != 0:
        raise SystemExit(proc.returncode)


# =============================================================================
# Cleanup utilities
# =============================================================================

def clean_procedural_stems() -> None:
    """Delete generated procedural stems to force regeneration."""
    generator_root = Path(__file__).resolve().parents[1]
    stems_dir = generator_root / "free" / "assets" / "stems"

    if not stems_dir.exists():
        return

    print(f"\n🧹 Cleaning procedural stems: {stems_dir}")
    count = 0
    for f in stems_dir.glob("*.wav"):
        f.unlink()
        count += 1
    print(f"  Removed {count} stems.")


# =============================================================================
# Test recipe (FULL MUSIC COVERAGE)
# =============================================================================

def create_full_coverage_recipe(path: Path) -> None:
    """
    Forces the FREE engine to generate:
    - Real MUSIC for each genre
    - Different structures & layers
    - Focus engine
    """

    recipe = {
        "combinations": [

            # ======================
            # ELECTRONIC MUSIC
            # ======================

            {
                "id": "test_techno",
                "seed": "smoke-techno",
                "genre": "Techno",
                "bpm": 130,
                "layers": {"enabled": ["drums", "bass", "pad"]},
                "structure": "peak",
                "title": "Smoke Techno"
            },
            {
                "id": "test_house",
                "seed": "smoke-house",
                "genre": "House",
                "bpm": 124,
                "layers": {"enabled": ["drums", "bass", "pad"]},
                "structure": "linear",
                "title": "Smoke House"
            },

            # ======================
            # CHILL / STUDY
            # ======================

            {
                "id": "test_lofi",
                "seed": "smoke-lofi",
                "genre": "Lofi",
                "bpm": 85,
                "layers": {"enabled": ["drums", "pad"]},
                "structure": "linear",
                "title": "Smoke Lofi"
            },

            # ======================
            # JAZZ / MUSICALITY
            # ======================

            {
                "id": "test_jazz",
                "seed": "smoke-jazz",
                "genre": "Jazz",
                "bpm": 120,
                "layers": {"enabled": ["drums", "bass", "pad"]},
                "structure": "linear",
                "title": "Smoke Jazz"
            },

            # ======================
            # PURE FOCUS ENGINE
            # ======================

            {
                "id": "test_focus",
                "seed": "smoke-focus",
                "genre": "Ambient",
                "bpm": 60,
                "layers": {"enabled": []},
                "focus": {
                    "binaural_mode": "focus",
                    "ambience": {
                        "rain": 80,
                        "white": 30
                    }
                },
                "title": "Smoke Focus"
            }
        ]
    }

    path.write_text(json.dumps(recipe, indent=2), encoding="utf-8")
    print(f"📝 Created smoke test recipe: {path.name}")


# =============================================================================
# MAIN
# =============================================================================

def main() -> int:
    script_dir = Path(__file__).resolve().parent
    generator_dir = script_dir.parent

    # Load envs
    load_dotenv_file(generator_dir / ".env")
    load_dotenv_file(Path(".env"))

    ap = argparse.ArgumentParser(description="SoundFlow Generator Smoke Test")
    ap.add_argument("--date", default=utc_today(), help="YYYY-MM-DD")
    ap.add_argument("--free", action="store_true", help="Run free engine tests")
    ap.add_argument("--clean", action="store_true", help="Force stem regeneration")
    ap.add_argument(
        "--duration-sec",
        type=int,
        default=20,
        help="Short duration for fast tests",
    )

    args = ap.parse_args()

    if not args.free:
        args.free = True

    print("\n🎛️ SoundFlow Generator Smoke Test")
    print(f"Date: {args.date}")
    print(f"Free engine: {args.free}")
    print(f"Clean stems: {args.clean}")
    print(f"Duration: {args.duration_sec}s")

    if args.clean:
        clean_procedural_stems()

    # -------------------------------------------------------------------------
    # FREE ENGINE – FULL MUSIC COVERAGE
    # -------------------------------------------------------------------------

    if args.free:
        recipe_path = script_dir / "smoke_test_recipe.json"
        create_full_coverage_recipe(recipe_path)

        try:
            cmd = [
                sys.executable,
                "-m",
                "free.remix_daily",
                "--date", args.date,
                "--duration-sec", str(args.duration_sec),
                "--json", str(recipe_path),
            ]
            run(cmd)
        finally:
            if recipe_path.exists():
                recipe_path.unlink()

    print("\n✅ Smoke test completed successfully.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
