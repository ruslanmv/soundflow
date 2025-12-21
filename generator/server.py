# generator/server.py
"""
SoundFlow Professional DJ Engine - Backend API Server
Version 6.0 (Professional DJ Quality)

✅ NEW FEATURES v6.0:
- Long stems (8-64 bars / 30s-4min)
- Energy curves (peak, drop, build, linear)
- All genres (Electronic, Chill, Jazz, Piano)
- Professional 3-layer synthesis
- Dynamic arrangement
- Broadcast-quality mixing
- Comprehensive metadata
- Track management (CRUD)
- Batch generation
- Statistics & analytics

Ready for professional DJ dashboard integration.
"""

from __future__ import annotations

import os
import sys
import uvicorn
import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Literal, Dict, Any
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Body, Request, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse, FileResponse
from pydantic import BaseModel, Field

# =============================================================================
# PATH SETUP
# =============================================================================

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# =============================================================================
# GLOBAL STATE
# =============================================================================

OUTPUT_DIR = Path(".soundflow_out/free")
TMP_DIR = Path(".soundflow_tmp/free")

# Create directories before imports
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
TMP_DIR.mkdir(parents=True, exist_ok=True)

library_initialized = False
generation_stats = {
    "total_generated": 0,
    "last_generation_time": 0.0,
    "average_generation_time": 0.0,
    "genres_generated": {},
    "energy_curves_used": {},
}

# =============================================================================
# ENGINE IMPORTS
# =============================================================================

try:
    from free.remix_daily import build_track, ensure_procedural_library
    from free.music_engine import SAMPLE_RATE, GENRE_STYLES, ENERGY_CURVES as ENGINE_CURVES
except ImportError as e:
    print(f"❌ Import Error: {e}")
    print("Make sure you're running from the generator/ directory")
    sys.exit(1)

# =============================================================================
# LOGGING
# =============================================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger("SoundFlowDJ")

# =============================================================================
# CONSTANTS & CONFIGURATION
# =============================================================================

# Available genres (from music_engine.py)
AVAILABLE_GENRES = list(GENRE_STYLES.keys())

# Stem length presets
STEM_LENGTHS = {
    "short": {"bars": 8, "duration_approx": "~30s @ 128 BPM"},
    "medium": {"bars": 16, "duration_approx": "~1min @ 128 BPM"},
    "long": {"bars": 32, "duration_approx": "~2min @ 128 BPM"},
    "full": {"bars": 64, "duration_approx": "~4min @ 128 BPM"},
}

# Energy curve types
ENERGY_CURVES = {
    "peak": "Classic club track (intro → build → peak → breakdown → outro)",
    "drop": "Bass/dubstep style (high → drop → low → build → peak)",
    "build": "Progressive build (steady increase to climax)",
    "linear": "Constant energy (steady throughout)",
}

# Musical keys
MUSICAL_KEYS = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]

# Layer types
AVAILABLE_LAYERS = ["kick", "drums", "bass", "music", "pad", "synth", "texture", "ambience"]

# =============================================================================
# DATA MODELS
# =============================================================================

class SynthParams(BaseModel):
    """Smart Mixer synthesis parameters (0-100)"""
    cutoff: float = Field(75.0, ge=0, le=100, description="Filter cutoff frequency")
    resonance: float = Field(30.0, ge=0, le=100, description="Filter resonance/Q")
    drive: float = Field(10.0, ge=0, le=100, description="Saturation/distortion")
    space: float = Field(20.0, ge=0, le=100, description="Reverb amount")

class AmbienceParams(BaseModel):
    """Ambience texture levels (0-100)"""
    rain: float = Field(0.0, ge=0, le=100, description="Rain ambience")
    vinyl: float = Field(0.0, ge=0, le=100, description="Vinyl crackle")
    white: float = Field(0.0, ge=0, le=100, description="White noise")

class GenerateRequest(BaseModel):
    """Complete generation request with all professional features"""
    
    # ========== CORE SETTINGS ==========
    mode: Literal["music", "focus", "hybrid"] = Field(
        "music",
        description="Generation mode: music only, focus only, or hybrid blend"
    )
    
    channels: Literal[1, 2] = Field(
        2,
        description="Audio channels: 1=mono, 2=stereo"
    )
    
    duration: int = Field(
        180,
        ge=30,
        le=900,
        description="Track duration in seconds (30s-15min)"
    )
    
    seed: Optional[str] = Field(
        None,
        description="Deterministic seed (leave empty for auto-generation)"
    )
    
    variation: float = Field(
        0.25,
        ge=0.0,
        le=1.0,
        description="Variation amount (0=identical, 1=maximum diversity)"
    )
    
    # ========== MUSIC SETTINGS ==========
    genre: str = Field(
        "Techno",
        description=f"Music genre. Available: {', '.join(AVAILABLE_GENRES)}"
    )
    
    bpm: int = Field(
        128,
        ge=60,
        le=200,
        description="Beats per minute (60-200)"
    )
    
    key: str = Field(
        "A",
        description=f"Musical key. Available: {', '.join(MUSICAL_KEYS)}"
    )
    
    layers: List[str] = Field(
        ["drums", "bass", "music"],
        description=f"Active layers. Available: {', '.join(AVAILABLE_LAYERS)}"
    )
    
    # ========== PROFESSIONAL FEATURES (NEW) ==========
    stem_length: Literal["short", "medium", "long", "full"] = Field(
        "medium",
        description="Stem length preset (short=8bars, medium=16bars, long=32bars, full=64bars)"
    )
    
    energy_curve: Literal["peak", "drop", "build", "linear"] = Field(
        "peak",
        description="Energy dynamics throughout the track"
    )
    
    # ========== FOCUS ENGINE ==========
    focus_mode: Literal["off", "focus", "relax"] = Field(
        "off",
        description="Binaural beats mode (off=disabled, focus=Beta 20Hz, relax=Alpha 10Hz)"
    )
    
    focus_mix: float = Field(
        30.0,
        ge=0,
        le=100,
        description="Focus/music blend in hybrid mode (0=music only, 100=focus only)"
    )
    
    ambience: AmbienceParams = Field(
        default_factory=AmbienceParams,
        description="Ambient texture levels"
    )
    
    # ========== SMART MIXER ==========
    intensity: float = Field(
        50.0,
        ge=0,
        le=100,
        description="Session intensity (0=ambient, 100=peak energy)"
    )
    
    synth_params: SynthParams = Field(
        default_factory=SynthParams,
        description="Advanced synthesis controls"
    )
    
    # ========== MASTERING ==========
    target_lufs: float = Field(
        -14.0,
        ge=-24.0,
        le=-6.0,
        description="Target loudness in LUFS (-14 = streaming standard)"
    )

class TrackMetadata(BaseModel):
    """Complete track metadata response"""
    
    # Basic info
    id: str
    name: str
    filename: str
    url: str
    
    # Generation info
    mode: str
    channels: int
    duration: int
    seed: str
    variation: float
    generated_at: str
    generation_time: float
    
    # Music settings
    genre: str
    bpm: int
    key: str
    layers_active: List[str]
    
    # Professional features
    stem_length: str
    stem_length_bars: int
    energy_curve: str
    
    # Focus settings
    focus_mode: str
    focus_mix: float
    has_binaural: bool
    
    # Mixer settings
    intensity: float
    intensity_label: str
    
    # Technical info
    sample_rate: int
    target_lufs: float
    file_size_mb: float

class BatchGenerateRequest(BaseModel):
    """Batch generation request"""
    requests: List[GenerateRequest] = Field(
        ...,
        max_items=10,
        description="List of generation requests (max 10)"
    )

class HealthResponse(BaseModel):
    """System health response"""
    status: str
    engine: str
    version: str
    sample_rate: int
    tracks_count: int
    total_size_mb: float
    timestamp: str
    
    # Available options
    available_genres: List[str]
    available_keys: List[str]
    available_layers: List[str]
    stem_lengths: Dict[str, Any]
    energy_curves: Dict[str, str]

class StatsResponse(BaseModel):
    """Generation statistics"""
    total_generated: int
    last_generation_time: float
    average_generation_time: float
    library_size: int
    genres_generated: Dict[str, int]
    energy_curves_used: Dict[str, int]
    uptime_hours: float

class ErrorResponse(BaseModel):
    """Error response"""
    status: str
    error: str
    detail: Optional[str] = None
    timestamp: str

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def get_intensity_label(intensity: float) -> str:
    """Convert intensity to human-readable label"""
    if intensity < 30:
        return "ambient"
    elif intensity < 70:
        return "balanced"
    else:
        return "peak"

def validate_genre(genre: str) -> str:
    """Validate and normalize genre"""
    genre_lower = genre.lower().strip()
    
    # Normalize aliases
    aliases = {
        "progressive house": "house",
        "deep house": "deep",
        "drum and bass": "bass",
        "dnb": "bass",
        "dubstep": "bass",
        "chill": "chillout",
        "lofi": "chillout",
    }
    
    genre_normalized = aliases.get(genre_lower, genre_lower)
    
    if genre_normalized not in [g.lower() for g in AVAILABLE_GENRES]:
        raise ValueError(
            f"Invalid genre '{genre}'. Available: {', '.join(AVAILABLE_GENRES)}"
        )
    
    # Return proper case
    for g in AVAILABLE_GENRES:
        if g.lower() == genre_normalized:
            return g
    
    return genre

def validate_layers(layers: List[str]) -> List[str]:
    """Validate layer names"""
    valid_layers_lower = [l.lower() for l in AVAILABLE_LAYERS]
    
    for layer in layers:
        if layer.lower() not in valid_layers_lower:
            raise ValueError(
                f"Invalid layer '{layer}'. Available: {', '.join(AVAILABLE_LAYERS)}"
            )
    
    return layers

# =============================================================================
# LIFESPAN MANAGEMENT
# =============================================================================

app_start_time = time.time()

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager"""
    global library_initialized
    
    logger.info("=" * 70)
    logger.info("🎵 SoundFlow Professional DJ Engine v6.0")
    logger.info("=" * 70)
    logger.info(f"📁 Output Directory: {OUTPUT_DIR.absolute()}")
    logger.info(f"🎼 Sample Rate: {SAMPLE_RATE} Hz")
    logger.info(f"🌐 API Docs: http://localhost:8000/docs")
    
    # Initialize library
    try:
        logger.info("🎛️  Initializing procedural library...")
        ensure_procedural_library(datetime.now().strftime("%Y-%m-%d"))
        library_initialized = True
        logger.info("✅ Procedural library ready")
    except Exception as e:
        logger.error(f"❌ Library initialization failed: {e}")
        logger.error("Engine will continue but generation may be affected")
        library_initialized = False
    
    logger.info("🚀 SoundFlow DJ Engine Ready!")
    logger.info("=" * 70)
    
    yield
    
    # Shutdown
    logger.info("🛑 SoundFlow DJ Engine Shutting Down...")
    logger.info(f"📊 Total tracks generated: {generation_stats['total_generated']}")
    logger.info(f"⏱️  Average generation time: {generation_stats['average_generation_time']:.1f}s")

# =============================================================================
# FASTAPI APP
# =============================================================================

app = FastAPI(
    title="SoundFlow Professional DJ Engine",
    description="Real-time AI music generation with professional DJ quality",
    version="6.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production: ["https://yourdomain.com"]
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Static files
app.mount("/audio", StaticFiles(directory=str(OUTPUT_DIR), check_dir=False), name="audio")

# =============================================================================
# API ENDPOINTS
# =============================================================================

@app.get("/", response_model=Dict[str, str])
async def root():
    """Root endpoint"""
    return {
        "service": "SoundFlow Professional DJ Engine",
        "version": "6.0.0",
        "status": "online",
        "docs": "/docs",
        "health": "/api/health"
    }

@app.get("/api/health", response_model=HealthResponse)
async def health_check():
    """
    System health check with comprehensive information.
    """
    try:
        tracks = list(OUTPUT_DIR.glob("*.mp3"))
        total_size = sum(t.stat().st_size for t in tracks) / (1024 * 1024)
        
        return HealthResponse(
            status="online" if library_initialized else "initializing",
            engine="Professional DJ Quality + Focus Engine",
            version="6.0.0",
            sample_rate=SAMPLE_RATE,
            tracks_count=len(tracks),
            total_size_mb=round(total_size, 2),
            timestamp=datetime.now().isoformat(),
            
            # Available options
            available_genres=AVAILABLE_GENRES,
            available_keys=MUSICAL_KEYS,
            available_layers=AVAILABLE_LAYERS,
            stem_lengths=STEM_LENGTHS,
            energy_curves=ENERGY_CURVES,
        )
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/library", response_model=Dict[str, Any])
async def get_library():
    """
    Get list of all generated tracks with metadata.
    """
    try:
        tracks = []
        
        for mp3_file in sorted(OUTPUT_DIR.glob("*.mp3"), key=lambda x: x.stat().st_mtime, reverse=True):
            file_stat = mp3_file.stat()
            
            tracks.append({
                "id": mp3_file.stem,
                "name": mp3_file.stem.replace("-", " ").replace("_", " ").title(),
                "filename": mp3_file.name,
                "url": f"/audio/{mp3_file.name}",
                "size_mb": round(file_stat.st_size / (1024 * 1024), 2),
                "created_at": datetime.fromtimestamp(file_stat.st_mtime).isoformat()
            })
        
        return {
            "status": "success",
            "count": len(tracks),
            "total_size_mb": round(sum(t["size_mb"] for t in tracks), 2),
            "tracks": tracks
        }
    
    except Exception as e:
        logger.error(f"Library fetch failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/generate", response_model=TrackMetadata)
async def generate_track(
    http_request: Request,
    request: GenerateRequest = Body(...)
):
    """
    Generate a professional DJ-quality track.
    
    Features:
    - Long stems (8-64 bars)
    - Energy curves (peak/drop/build/linear)
    - All genres (Electronic/Chill/Jazz/Piano)
    - Binaural beats for focus
    - Smart mixer controls
    - Broadcast-quality mixing
    
    Returns:
    - Complete track metadata
    - Absolute audio URL
    - Generation statistics
    """
    
    if not library_initialized:
        raise HTTPException(
            status_code=503,
            detail="Engine is still initializing. Please wait a moment."
        )
    
    # Validate inputs
    try:
        genre = validate_genre(request.genre)
        validate_layers(request.layers)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    
    # Generate deterministic seed
    start_time = time.time()
    seed = request.seed or f"{genre}:{int(start_time)}"
    
    # Log request
    logger.info("=" * 70)
    logger.info(f"🎵 Generation Request:")
    logger.info(f"   Genre: {genre} | BPM: {request.bpm} | Key: {request.key}")
    logger.info(f"   Layers: {', '.join(request.layers)}")
    logger.info(f"   Stem Length: {request.stem_length}")
    logger.info(f"   Energy Curve: {request.energy_curve}")
    logger.info(f"   Mode: {request.mode}")
    logger.info(f"   Duration: {request.duration}s")
    logger.info(f"   Seed: {seed}")
    
    # Build engine preset
    preset = {
        "id": f"{genre.lower()}-{seed}",
        "title": f"AI {genre} Session",
        "seed": seed,
        
        # Mode
        "mode": request.mode,
        "channels": request.channels,
        "variation": request.variation,
        
        # Music settings
        "music": {
            "genre": genre,
            "bpm": request.bpm,
            "key": request.key,
            "layers": request.layers,
        },
        
        # Professional features
        "stem_length": request.stem_length,
        "energy_curve": request.energy_curve,
        
        # Focus engine
        "focus": {
            "mode": request.focus_mode,
            "mix": request.focus_mix,
            "ambience": request.ambience.model_dump(),
        },
        
        # Smart mixer
        "mixer": {
            "intensity": request.intensity,
            "synth": request.synth_params.model_dump(),
        },
        
        # Master
        "master": {
            "target_lufs": request.target_lufs
        }
    }
    
    # Generate track
    try:
        mp3_path, entry = build_track(
            date=datetime.now().strftime("%Y-%m-%d"),
            preset=preset,
            total_sec=request.duration,
        )
    except Exception as e:
        logger.error(f"❌ Generation failed: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Track generation failed: {str(e)}"
        )
    
    generation_time = time.time() - start_time
    
    # Update statistics
    generation_stats["total_generated"] += 1
    generation_stats["last_generation_time"] = generation_time
    
    if generation_stats["average_generation_time"] == 0:
        generation_stats["average_generation_time"] = generation_time
    else:
        generation_stats["average_generation_time"] = (
            generation_stats["average_generation_time"] * 0.9 + generation_time * 0.1
        )
    
    # Track genre usage
    generation_stats["genres_generated"][genre] = \
        generation_stats["genres_generated"].get(genre, 0) + 1
    
    # Track energy curve usage
    generation_stats["energy_curves_used"][request.energy_curve] = \
        generation_stats["energy_curves_used"].get(request.energy_curve, 0) + 1
    
    logger.info(f"   ✅ Generated in {generation_time:.1f}s")
    logger.info("=" * 70)
    
    # Build absolute URL
    base_url = str(http_request.base_url).rstrip("/")
    audio_url = f"{base_url}/audio/{mp3_path.name}"
    
    # Get file size
    file_size_mb = mp3_path.stat().st_size / (1024 * 1024)
    
    # Get stem length in bars
    stem_length_bars = STEM_LENGTHS[request.stem_length]["bars"]
    
    return TrackMetadata(
        # Basic info
        id=entry.get("id", "unknown"),
        name=entry.get("title", "Untitled"),
        filename=mp3_path.name,
        url=audio_url,
        
        # Generation info
        mode=request.mode,
        channels=request.channels,
        duration=request.duration,
        seed=seed,
        variation=request.variation,
        generated_at=datetime.now().isoformat(),
        generation_time=round(generation_time, 2),
        
        # Music settings
        genre=genre,
        bpm=request.bpm,
        key=request.key,
        layers_active=request.layers,
        
        # Professional features
        stem_length=request.stem_length,
        stem_length_bars=stem_length_bars,
        energy_curve=request.energy_curve,
        
        # Focus settings
        focus_mode=request.focus_mode,
        focus_mix=request.focus_mix,
        has_binaural=request.focus_mode != "off",
        
        # Mixer settings
        intensity=request.intensity,
        intensity_label=get_intensity_label(request.intensity),
        
        # Technical info
        sample_rate=SAMPLE_RATE,
        target_lufs=request.target_lufs,
        file_size_mb=round(file_size_mb, 2),
    )

@app.post("/api/generate/batch")
async def generate_batch(
    http_request: Request,
    batch_request: BatchGenerateRequest,
    background_tasks: BackgroundTasks = None
):
    """
    Generate multiple tracks in batch.
    
    Maximum 10 tracks per batch.
    Useful for pre-generating a DJ set.
    """
    if not library_initialized:
        raise HTTPException(status_code=503, detail="Engine is initializing")
    
    requests = batch_request.requests
    
    if len(requests) > 10:
        raise HTTPException(
            status_code=400,
            detail="Maximum 10 tracks per batch"
        )
    
    logger.info(f"🎼 Batch generation: {len(requests)} tracks")
    
    results = []
    
    for i, req in enumerate(requests, 1):
        try:
            logger.info(f"📀 Generating track {i}/{len(requests)}")
            track = await generate_track(http_request, req)
            results.append({
                "status": "success",
                "track": track.model_dump()
            })
        except Exception as e:
            logger.error(f"❌ Track {i} failed: {e}")
            results.append({
                "status": "error",
                "error": str(e),
                "request": req.model_dump()
            })
    
    successful = len([r for r in results if r["status"] == "success"])
    
    return {
        "status": "completed",
        "total": len(requests),
        "successful": successful,
        "failed": len(requests) - successful,
        "results": results
    }

@app.delete("/api/tracks/{track_id}")
async def delete_track(track_id: str):
    """
    Delete a specific track by ID.
    """
    try:
        track_files = list(OUTPUT_DIR.glob(f"*{track_id}*.mp3"))
        
        if not track_files:
            raise HTTPException(
                status_code=404,
                detail=f"Track '{track_id}' not found"
            )
        
        for track_file in track_files:
            track_file.unlink()
            logger.info(f"🗑️  Deleted: {track_file.name}")
        
        return {
            "status": "success",
            "message": f"Deleted {len(track_files)} file(s)",
            "deleted_files": [f.name for f in track_files]
        }
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Delete failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/tracks/clear")
async def clear_all_tracks():
    """
    Clear all generated tracks.
    
    WARNING: This deletes all MP3 files!
    """
    try:
        tracks = list(OUTPUT_DIR.glob("*.mp3"))
        count = len(tracks)
        total_size = sum(t.stat().st_size for t in tracks)
        
        for track in tracks:
            track.unlink()
        
        logger.warning(f"🗑️  Cleared {count} tracks ({total_size / (1024*1024):.1f} MB)")
        
        return {
            "status": "success",
            "message": f"Cleared {count} tracks",
            "freed_space_mb": round(total_size / (1024 * 1024), 2)
        }
    
    except Exception as e:
        logger.error(f"Clear failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/stats", response_model=StatsResponse)
async def get_stats():
    """
    Get generation statistics and analytics.
    """
    uptime = (time.time() - app_start_time) / 3600  # hours
    
    return StatsResponse(
        total_generated=generation_stats["total_generated"],
        last_generation_time=round(generation_stats["last_generation_time"], 2),
        average_generation_time=round(generation_stats["average_generation_time"], 2),
        library_size=len(list(OUTPUT_DIR.glob("*.mp3"))),
        genres_generated=generation_stats["genres_generated"],
        energy_curves_used=generation_stats["energy_curves_used"],
        uptime_hours=round(uptime, 2),
    )

@app.get("/api/genres", response_model=Dict[str, Any])
async def get_genres():
    """
    Get list of available genres with their default settings.
    """
    genres_info = {}
    
    for genre, settings in GENRE_STYLES.items():
        genres_info[genre] = {
            "name": genre.title(),
            "default_bpm": settings["bpm"],
            "swing": settings["swing"],
            "brightness": settings["brightness"],
            "reverb": settings["reverb"],
            "duck": settings["duck"],
            "description": f"{genre.title()} style with {settings['bpm']} BPM default"
        }
    
    return {
        "status": "success",
        "count": len(genres_info),
        "genres": genres_info
    }

# =============================================================================
# ERROR HANDLERS
# =============================================================================

@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    """Custom HTTP exception handler"""
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "status": "error",
            "error": exc.detail,
            "timestamp": datetime.now().isoformat()
        }
    )

@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    """Catch-all exception handler"""
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={
            "status": "error",
            "error": "Internal server error",
            "detail": str(exc),
            "timestamp": datetime.now().isoformat()
        }
    )

# =============================================================================
# MAIN ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("🎵 SoundFlow Professional DJ Engine v6.0")
    print("=" * 70)
    print()
    print("🚀 Starting server...")
    print("🌐 Frontend:  http://localhost:3000")
    print("🔌 Backend:   http://localhost:8000")
    print("📡 API Docs:  http://localhost:8000/docs")
    print()
    print("✨ NEW FEATURES v6.0:")
    print("   ✓ Long stems (8-64 bars / 30s-4min)")
    print("   ✓ Energy curves (peak/drop/build/linear)")
    print("   ✓ Professional 3-layer synthesis")
    print("   ✓ All genres (Electronic/Chill/Jazz/Piano)")
    print("   ✓ Dynamic arrangement")
    print("   ✓ Broadcast-quality mixing")
    print("   ✓ Comprehensive metadata")
    print("   ✓ Track management (CRUD)")
    print("   ✓ Batch generation")
    print("   ✓ Statistics & analytics")
    print()
    print("Press CTRL+C to stop")
    print("=" * 70)
    print()
    
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info",
        access_log=True
    )
