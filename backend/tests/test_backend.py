#!/usr/bin/env python3
# backend/tests/test_backend.py
#
# Smoke-test for SoundFlow backend + local audio generation.
#
# Usage (from backend/):
#   python ./tests/test_backend.py

from __future__ import annotations

import json
import math
import os
import time
import urllib.error
import urllib.request
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional


# -------------------------
# .env loader (no deps)
# -------------------------
def load_dotenv_file(path: str) -> None:
    p = Path(path)
    if not p.exists():
        return

    for raw in p.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            continue

        k, v = line.split("=", 1)
        k = k.strip()
        v = v.strip().strip('"').strip("'")

        if not k:
            continue

        os.environ.setdefault(k, v)


# Load backend/.env when running from backend/
load_dotenv_file(".env")


@dataclass
class HttpResult:
    status: int
    headers: Dict[str, str]
    body_text: str


def http_json(
    method: str,
    url: str,
    payload: Optional[Dict[str, Any]] = None,
    headers: Optional[Dict[str, str]] = None,
    timeout: int = 20,
) -> HttpResult:
    data = None
    req_headers = {"Accept": "application/json"}
    if headers:
        req_headers.update(headers)

    if payload is not None:
        data = json.dumps(payload).encode("utf-8")
        req_headers["Content-Type"] = "application/json"

    req = urllib.request.Request(url, data=data, method=method, headers=req_headers)

    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            body = resp.read().decode("utf-8", errors="replace")
            hdrs = {k: v for (k, v) in resp.headers.items()}
            return HttpResult(status=resp.status, headers=hdrs, body_text=body)
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", errors="replace") if e.fp else str(e)
        hdrs = {k: v for (k, v) in (e.headers.items() if e.headers else [])}
        return HttpResult(status=e.code, headers=hdrs, body_text=body)
    except Exception as e:
        return HttpResult(status=0, headers={}, body_text=f"ERROR: {type(e).__name__}: {e}")


def pretty_print_result(title: str, r: HttpResult) -> None:
    print(f"\n=== {title} ===")
    print(f"Status: {r.status}")
    if r.status == 0:
        print(r.body_text)
        return

    # Try JSON pretty print; fallback to raw
    try:
        obj = json.loads(r.body_text)
        print(json.dumps(obj, indent=2, ensure_ascii=False))
    except Exception:
        print(r.body_text[:2000])


def generate_test_wav(
    out_path: Path,
    duration_sec: float = 2.0,
    sample_rate: int = 44100,
    freq_hz: float = 440.0,
    volume: float = 0.2,
) -> Path:
    out_path.parent.mkdir(parents=True, exist_ok=True)

    n_samples = int(duration_sec * sample_rate)
    amp = int(32767 * max(0.0, min(1.0, volume)))

    with wave.open(str(out_path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)  # 16-bit
        wf.setframerate(sample_rate)

        frames = bytearray()
        for i in range(n_samples):
            t = i / sample_rate
            s = int(amp * math.sin(2.0 * math.pi * freq_hz * t))
            frames += int(s).to_bytes(2, byteorder="little", signed=True)

        wf.writeframes(frames)

    return out_path


def main() -> int:
    base = os.getenv("SOUNDFLOW_API_URL", "http://localhost:8000").rstrip("/")
    premium_api_key = os.getenv("PREMIUM_API_KEY") or os.getenv("premium_api_key", "")
    admin_api_key = os.getenv("ADMIN_API_KEY") or os.getenv("admin_api_key", "")

    print("SoundFlow Backend Smoke Test")
    print(f"API Base: {base}")
    print(f"PREMIUM_API_KEY set: {'yes' if bool(premium_api_key) else 'no'}")
    print(f"ADMIN_API_KEY set: {'yes' if bool(admin_api_key) else 'no'}")

    # 1) Local audio creation test
    tmp_dir = Path("_tmp")
    wav_path = tmp_dir / "test-tone.wav"
    t0 = time.time()
    generate_test_wav(wav_path)
    dt = (time.time() - t0) * 1000
    print(f"\n✅ Created test audio: {wav_path} ({wav_path.stat().st_size} bytes) in {dt:.1f}ms")

    # 2) Backend health
    r = http_json("GET", f"{base}/health")
    pretty_print_result("GET /health", r)
    if r.status == 0:
        print("\n❌ Backend is not reachable. Is uvicorn running?")
        return 2

    # 3) Free tracks catalog
    r = http_json("GET", f"{base}/tracks/free")
    pretty_print_result("GET /tracks/free", r)

    # 4) Free session routing
    free_payload = {
        "goal": "Deep Work",
        "durationMin": 50,
        "energy": 60,
        "ambience": 30,
        "nature": "Rain",
    }
    r = http_json("POST", f"{base}/session/free", payload=free_payload)
    pretty_print_result("POST /session/free", r)

    # 5) Premium daily + signed URL (optional)
    if premium_api_key:
        headers = {"X-Premium-Key": premium_api_key}

        r = http_json("GET", f"{base}/tracks/premium/daily", headers=headers)
        pretty_print_result("GET /tracks/premium/daily", r)

        # Parse logic to handle LIST or OBJECT response
        try:
            obj = json.loads(r.body_text) if r.status and r.status < 400 else None
        except Exception:
            obj = None

        track_id = None
        
        # Case 1: List of tracks (what your API currently returns)
        if isinstance(obj, list) and len(obj) > 0:
            item = obj[0]
            if isinstance(item, dict) and "id" in item:
                track_id = item["id"]
        
        # Case 2: Single object (fallback logic)
        elif isinstance(obj, dict):
            if "id" in obj:
                track_id = obj["id"]
            elif "track" in obj and isinstance(obj["track"], dict):
                track_id = obj["track"].get("id")

        if track_id:
            print(f"\n--> Found track ID: {track_id}, fetching signed URL...")
            r2 = http_json("GET", f"{base}/tracks/premium/{track_id}/signed", headers=headers)
            pretty_print_result(f"GET /tracks/premium/{track_id}/signed", r2)
        else:
            print("\nℹ️ Premium daily did not return any recognized track ID.")
    else:
        print("\nℹ️ Skipping premium tests (set PREMIUM_API_KEY in backend/.env to enable).")

    print("\n✅ Smoke test completed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())