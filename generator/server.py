# generator/server.py
"""
SoundFlow Professional DJ Engine - Backend API Server
Version 3.2 (Deterministic + Hybrid Ready)

Responsibilities:
- Validate & normalize generation requests
- Build explicit engine presets
- Call a single engine entry point
- Return truthful metadata about what was generated
"""

from __future__ import annotations

import os
import sys
import uvicorn
import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Literal
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Body, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

# -----------------------------------------------------------------------------
# PATH FIX
# -----------------------------------------------------------------------------
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# =============================================================================
# STATE / PATHS (MUST EXIST BEFORE app.mount)
# =============================================================================

OUTPUT_DIR = Path(".soundflow_out/free")
TMP_DIR = Path(".soundflow_tmp/free")
library_initialized = False

# ✅ IMPORTANT: create dirs BEFORE StaticFiles mounts
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
TMP_DIR.mkdir(parents=True, exist_ok=True)

# -----------------------------------------------------------------------------
# ENGINE IMPORTS
# -----------------------------------------------------------------------------
try:
    from free.remix_daily import build_track, ensure_procedural_library
    from free.music_engine import SAMPLE_RATE
except ImportError as e:
    print(f"❌ Import Error: {e}")
    sys.exit(1)

# -----------------------------------------------------------------------------
# LOGGING
# -----------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("SoundFlowDJ")

# =============================================================================
# DATA MODELS
# =============================================================================

class SynthParams(BaseModel):
    cutoff: float = Field(75, ge=0, le=100)
    resonance: float = Field(30, ge=0, le=100)
    drive: float = Field(10, ge=0, le=100)
    space: float = Field(20, ge=0, le=100)

class AmbienceParams(BaseModel):
    rain: float = Field(0, ge=0, le=100)
    vinyl: float = Field(0, ge=0, le=100)
    white: float = Field(0, ge=0, le=100)

class GenerateRequest(BaseModel):
    # --- intent ---
    mode: Literal["music", "focus", "hybrid"] = "music"
    channels: Literal[1, 2] = 2
    focus_mix: float = Field(30, ge=0, le=100)
    variation: float = Field(0.25, ge=0.0, le=1.0)

    # --- music ---
    genre: str = "Techno"
    bpm: int = Field(128, ge=60, le=200)
    key: str = "A"
    seed: Optional[str] = None

    layers: List[str] = ["drums", "bass", "music", "pad"]
    energy_curve: Literal["linear", "drop", "peak"] = "peak"
    duration: int = Field(180, ge=30, le=900)

    # --- focus ---
    focus_mode: Literal["off", "focus", "relax"] = "off"
    ambience: AmbienceParams = AmbienceParams()

    # --- mixer ---
    intensity: float = Field(50, ge=0, le=100)
    synth_params: SynthParams = SynthParams()

    target_lufs: float = -14.0

class TrackMetadata(BaseModel):
    id: str
    name: str
    filename: str
    url: str

    mode: str
    channels: int
    genre: str
    bpm_used: int
    key: str
    duration: int
    seed: str

    layers_active: List[str]
    energy_curve: str
    focus_mode: str
    focus_mix: float
    variation: float

    generated_at: str

class HealthResponse(BaseModel):
    status: str
    engine: str
    version: str
    sample_rate: int
    tracks_count: int
    timestamp: str

# =============================================================================
# LIFESPAN
# =============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    global library_initialized

    logger.info("🎵 SoundFlow Engine starting...")

    # dirs are already created, but keep safe
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    TMP_DIR.mkdir(parents=True, exist_ok=True)

    try:
        ensure_procedural_library(datetime.now().strftime("%Y-%m-%d"))
        library_initialized = True
        logger.info("✅ Procedural library ready")
    except Exception:
        logger.exception("❌ ensure_procedural_library failed")
        library_initialized = False

    yield

    logger.info("🛑 SoundFlow Engine stopped")

# =============================================================================
# APP
# =============================================================================

app = FastAPI(
    title="SoundFlow Infinite DJ Engine",
    version="3.2.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ✅ do not crash if dir missing (we also mkdir above; this is extra safety)
app.mount("/audio", StaticFiles(directory=str(OUTPUT_DIR), check_dir=False), name="audio")

# =============================================================================
# ENDPOINTS
# =============================================================================

@app.get("/api/health", response_model=HealthResponse)
async def health():
    return HealthResponse(
        status="online" if library_initialized else "initializing",
        engine="Infinite Music + Focus Hybrid Engine",
        version="3.2.0",
        sample_rate=SAMPLE_RATE,
        tracks_count=len(list(OUTPUT_DIR.glob("*.mp3"))),
        timestamp=datetime.now().isoformat(),
    )

@app.post("/api/generate", response_model=TrackMetadata)
async def generate(http_request: Request, request: GenerateRequest = Body(...)):
    """
    Core generation endpoint.

    IMPORTANT:
    - Returns an absolute URL for the generated MP3 so the frontend (Next.js :3000)
      doesn't try to fetch /audio/... from itself.
    """
    if not library_initialized:
        raise HTTPException(503, "Engine initializing")

    start = time.time()

    # ---- deterministic seed ----
    seed = request.seed or f"{request.genre}:{int(start)}"

    # ---- build engine preset (NO GUESSING) ----
    preset = {
        "id": f"{request.genre.lower()}-{seed}",
        "title": f"Infinite {request.genre}",
        "seed": seed,

        "mode": request.mode,
        "channels": request.channels,
        "variation": request.variation,

        "music": {
            "genre": request.genre,
            "bpm": request.bpm,
            "key": request.key,
            "layers": request.layers,
            "energy_curve": request.energy_curve,
        },

        "focus": {
            "mode": request.focus_mode,
            "mix": request.focus_mix,
            "ambience": request.ambience.model_dump(),
        },

        "mixer": {
            "intensity": request.intensity,
            "synth": request.synth_params.model_dump(),
        },

        "master": {
            "target_lufs": request.target_lufs
        }
    }

    logger.info(
        f"🎛️ Generate | mode={request.mode} | bpm={request.bpm} | "
        f"channels={request.channels} | variation={request.variation}"
    )

    try:
        mp3_path, entry = build_track(
            date=datetime.now().strftime("%Y-%m-%d"),
            preset=preset,
            total_sec=request.duration,
        )
    except Exception as e:
        logger.exception("Generation failed")
        raise HTTPException(500, str(e))

    # ✅ absolute audio URL (prevents Next.js from trying localhost:3000/audio/..)
    audio_url = str(http_request.base_url) + f"audio/{mp3_path.name}"

    return TrackMetadata(
        id=entry.get("id", "unknown"),
        name=entry.get("title", "Untitled"),
        filename=mp3_path.name,
        url=audio_url,

        mode=request.mode,
        channels=request.channels,
        genre=request.genre,
        bpm_used=request.bpm,
        key=request.key,
        duration=request.duration,
        seed=seed,

        layers_active=request.layers,
        energy_curve=request.energy_curve,
        focus_mode=request.focus_mode,
        focus_mix=request.focus_mix,
        variation=request.variation,

        generated_at=datetime.now().isoformat(),
    )

# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
