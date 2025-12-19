#!/usr/bin/env python3
# generator/tests/test_generator.py
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import json
from datetime import datetime, timezone
from pathlib import Path

# -------------------------
# Minimal .env loader (no deps)
# -------------------------
def load_dotenv_file(path: str) -> None:
    p = Path(path)
    if not p.exists():
        return

    for raw in p.read_text(encoding="utf-8").splitlines():
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


def run(cmd: list[str], cwd: str | None = None) -> None:
    print("\n$ " + " ".join(cmd))
    
    # [FIX] Force PYTHONPATH to include the 'generator' directory
    env = os.environ.copy()
    generator_root = Path(__file__).resolve().parents[1]
    
    current_path = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = str(generator_root) + (os.pathsep + current_path if current_path else "")

    p = subprocess.run(cmd, cwd=cwd, env=env, text=True)
    if p.returncode != 0:
        raise SystemExit(p.returncode)

def clean_stems():
    """Deletes generated procedural stems to force re-generation."""
    generator_root = Path(__file__).resolve().parents[1]
    stems_dir = generator_root / "free" / "assets" / "stems"
    
    if not stems_dir.exists():
        return

    print(f"\n🧹 Cleaning procedural stems in {stems_dir}...")
    targets = ["drums", "kick", "bass", "arp", "pad", "synth", "texture"]
    
    count = 0
    for file in stems_dir.glob("*.wav"):
        if any(t in file.name for t in targets):
            file.unlink()
            count += 1
    print(f"  Removed {count} stems to force regeneration.")

def create_full_coverage_recipe(path: Path):
    """
    Creates a JSON recipe that forces the generator to produce ONE track
    of every available style to ensure full coverage.
    """
    recipe = {
        "combinations": [
            # 1. TECHNO (Default logic)
            {
                "id": "test_techno", 
                "genre": "Techno", 
                "bpm": 130, 
                "layers": {"enabled": ["drums", "bass", "music"]},
                "title": "Smoke Test Techno"
            },
            # 2. HOUSE (Triggers: drums_house, bass_deep)
            {
                "id": "test_house", 
                "genre": "House", 
                "bpm": 124, 
                "layers": {"enabled": ["drums", "bass", "music"]},
                "title": "Smoke Test House"
            },
            # 3. LO-FI (Triggers: drums_lofi, keys_lofi, vinyl)
            {
                "id": "test_lofi", 
                "genre": "Lo-Fi", 
                "bpm": 85, 
                "layers": {"enabled": ["drums", "music", "texture"]},
                "texture": "vinyl",
                "title": "Smoke Test LoFi"
            },
            # 4. BASS / DUBSTEP (Triggers: bass_wobble)
            {
                "id": "test_bass", 
                "genre": "Bass", 
                "bpm": 140, 
                "layers": {"enabled": ["drums", "bass"]},
                "title": "Smoke Test Bass"
            },
            # 5. HARD STYLE (Triggers: kick_hard)
            {
                "id": "test_hard", 
                "genre": "Hard", 
                "bpm": 150, 
                "layers": {"enabled": ["drums", "bass"]},
                "title": "Smoke Test Hard"
            },
            # 6. SYNTHWAVE (Triggers: bass_synth, snare_gated)
            {
                "id": "test_synth", 
                "genre": "Synthwave", 
                "bpm": 105, 
                "layers": {"enabled": ["drums", "bass"]},
                "title": "Smoke Test Synth"
            },
            # 7. EURO / RAVE (Triggers: piano_rave)
            {
                "id": "test_euro", 
                "genre": "Euro", 
                "bpm": 140, 
                "layers": {"enabled": ["drums", "music"]},
                "title": "Smoke Test Rave"
            },
            # 8. FOCUS ENGINE (Triggers: binaural generator)
            {
                "id": "test_focus",
                "genre": "Ambient",
                "bpm": 60,
                "layers": {"enabled": ["texture"]},
                "focus": {
                    "binaural_mode": "focus", 
                    "ambience": {"rain": 100}
                },
                "title": "Smoke Test Focus"
            }
        ]
    }
    path.write_text(json.dumps(recipe, indent=2), encoding="utf-8")
    print(f"📝 Created comprehensive test recipe: {path.name}")

def main() -> int:
    # Resolve paths relative to this script
    script_dir = Path(__file__).resolve().parent
    gen_dir = script_dir.parent
    
    load_dotenv_file(str(gen_dir / ".env"))
    load_dotenv_file(".env")

    ap = argparse.ArgumentParser()
    ap.add_argument("--date", default=utc_today(), help="YYYY-MM-DD")
    ap.add_argument("--free", action="store_true", help="Run free remix generator (All Genres)")
    ap.add_argument("--premium", action="store_true", help="Run premium musicgen generator")
    ap.add_argument("--upload", action="store_true", help="Upload outputs")
    ap.add_argument("--buffer", action="store_true", help="Run buffer publisher")
    ap.add_argument("--validate-r2", action="store_true", help="Validate R2 catalogs")
    ap.add_argument("--validate-local", action="store_true", help="Validate local sample catalogs")
    ap.add_argument("--clean", action="store_true", help="Delete existing stems")
    ap.add_argument("--free-duration-sec", type=int, default=30, help="Test duration (short for speed)")
    ap.add_argument("--musicgen-model", default="facebook/musicgen-small", help="MusicGen model id")

    args = ap.parse_args()
    date = args.date

    # Default behavior
    if not (args.free or args.premium or args.validate_r2 or args.validate_local):
        args.free = True
        args.upload = False

    # Safety checks
    if args.upload:
        required = ["R2_ENDPOINT", "R2_BUCKET", "R2_ACCESS_KEY_ID", "R2_SECRET_ACCESS_KEY"]
        missing = [k for k in required if not os.getenv(k)]
        if missing:
            print("\n❌ Upload requested but missing env vars in generator/.env:")
            for m in missing:
                print("  -", m)
            return 2

    print("SoundFlow Generator Smoke Test")
    print(f"Date: {date}")
    print(f"Modes: Free={args.free} | Premium={args.premium} | Clean={args.clean}")
    
    # 0) CLEAN
    if args.clean:
        clean_stems()

    # 1) FREE ENGINE - FULL COVERAGE TEST
    if args.free:
        # Create temporary recipe to test ALL genres
        recipe_path = script_dir / "smoke_test_recipe.json"
        create_full_coverage_recipe(recipe_path)
        
        cmd = [
            sys.executable,
            "-m",
            "free.remix_daily",
            "--date", date,
            "--duration-sec", str(args.free_duration_sec),
            "--json", str(recipe_path)  # Pass the test recipe
        ]
        
        if args.upload:
            cmd.append("--upload")
            
        try:
            run(cmd)
        finally:
            # Clean up the test file
            if recipe_path.exists():
                recipe_path.unlink()

    # 2) PREMIUM
    if args.premium:
        cmd = [
            sys.executable,
            "-m",
            "premium.musicgen_daily",
            "--date", date,
            "--model", args.musicgen_model,
        ]
        if args.upload:
            cmd.append("--upload")
        run(cmd)

    # 3) BUFFER PUBLISHER
    if args.buffer:
        cmd = [sys.executable, "-m", "common.buffer_publisher", "--date", date]
        run(cmd)

    # 4) VALIDATION
    if args.validate_local:
        run([sys.executable, "-m", "common.validate_catalog", "--tier", "free", "--source", "local"])
        run([sys.executable, "-m", "common.validate_catalog", "--tier", "premium", "--source", "local"])

    if args.validate_r2:
        run([sys.executable, "-m", "common.validate_catalog", "--tier", "free", "--source", "r2"])
        run([sys.executable, "-m", "common.validate_catalog", "--tier", "premium", "--source", "r2"])

    print("\n✅ Generator smoke test completed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())