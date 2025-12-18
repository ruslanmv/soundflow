#!/usr/bin/env python3
# generator/tests/test_generator.py
from __future__ import annotations

import argparse
import os
import subprocess
import sys
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
    # This allows 'from common import ...' to work inside the subprocesses
    env = os.environ.copy()
    
    # Calculate the generator root directory
    # This file is in: generator/tests/test_generator.py
    # parents[0] = generator/tests
    # parents[1] = generator
    generator_root = Path(__file__).resolve().parents[1]
    
    # Prepend to PYTHONPATH
    current_path = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = str(generator_root) + (os.pathsep + current_path if current_path else "")

    p = subprocess.run(cmd, cwd=cwd, env=env, text=True)
    if p.returncode != 0:
        raise SystemExit(p.returncode)


def main() -> int:
    # Try load generator/.env (run from repo root OR generator/)
    # We resolve relative to this script to be safe
    script_dir = Path(__file__).resolve().parent
    gen_dir = script_dir.parent
    
    load_dotenv_file(str(gen_dir / ".env"))
    load_dotenv_file(".env")

    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--date",
        default=utc_today(),
        help="YYYY-MM-DD (default: today UTC)",
    )
    ap.add_argument("--free", action="store_true", help="Run free remix generator")
    ap.add_argument("--premium", action="store_true", help="Run premium musicgen generator")
    ap.add_argument("--upload", action="store_true", help="Upload outputs + update catalog in R2")
    ap.add_argument("--buffer", action="store_true", help="Ensure premium has track for today (buffer publish)")
    ap.add_argument("--validate-r2", action="store_true", help="Validate R2 catalogs after upload")
    ap.add_argument("--validate-local", action="store_true", help="Validate local sample catalogs")
    ap.add_argument("--free-duration-sec", type=int, default=120, help="Free track duration seconds (test default: 120)")
    ap.add_argument("--musicgen-model", default="facebook/musicgen-small", help="MusicGen model id")

    args = ap.parse_args()
    date = args.date

    # Default behavior: if user passed nothing, run a small FREE test locally (no upload)
    if not (args.free or args.premium or args.validate_r2 or args.validate_local):
        args.free = True
        args.upload = False

    # Safety: upload requires R2 creds
    if args.upload:
        required = ["R2_ENDPOINT", "R2_BUCKET", "R2_ACCESS_KEY_ID", "R2_SECRET_ACCESS_KEY"]
        missing = [k for k in required if not os.getenv(k)]
        if missing:
            print("\n❌ Upload requested but missing env vars in generator/.env:")
            for m in missing:
                print("  -", m)
            return 2

    print("SoundFlow Generator Smoke Test")
    print("date:", date)
    print("free:", args.free, "| premium:", args.premium, "| upload:", args.upload)
    print("buffer:", args.buffer, "| validate-r2:", args.validate_r2, "| validate-local:", args.validate_local)

    # 1) FREE
    if args.free:
        cmd = [
            sys.executable,
            "-m",
            "free.remix_daily",  # [FIX] Run as 'free.remix_daily', not 'generator.free...'
            "--date",
            date,
            "--duration-sec",
            str(args.free_duration_sec),
        ]
        if args.upload:
            cmd.append("--upload")
        run(cmd)

    # 2) PREMIUM (MusicGen)
    if args.premium:
        cmd = [
            sys.executable,
            "-m",
            "premium.musicgen_daily", # [FIX] Run as 'premium.musicgen_daily'
            "--date",
            date,
            "--model",
            args.musicgen_model,
        ]
        if args.upload:
            cmd.append("--upload")
        run(cmd)

    # 3) BUFFER PUBLISHER (premium safety net)
    if args.buffer:
        cmd = [sys.executable, "-m", "common.buffer_publisher", "--date", date] # [FIX] 'common.buffer_publisher'
        run(cmd)

    # 4) VALIDATION (LOCAL samples)
    if args.validate_local:
        run([sys.executable, "-m", "common.validate_catalog", "--tier", "free", "--source", "local"])
        run([sys.executable, "-m", "common.validate_catalog", "--tier", "premium", "--source", "local"])

    # 5) VALIDATION (R2 catalogs)
    if args.validate_r2:
        run([sys.executable, "-m", "common.validate_catalog", "--tier", "free", "--source", "r2"])
        run([sys.executable, "-m", "common.validate_catalog", "--tier", "premium", "--source", "r2"])

    print("\n✅ Generator smoke test completed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())