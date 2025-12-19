# generator/server.py
"""
SoundFlow Professional DJ Engine - Backend API Server
Version 2.1.0

Features:
- Real-time AI music generation
- Focus Engine (Binaural beats)
- Smart Mixer (Pro synthesis controls)
- Energy Curve arrangement
- Track library management
- Batch generation support
"""

from __future__ import annotations

import os
import sys
import uvicorn
import logging
import time
import asyncio
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any, List

from fastapi import FastAPI, HTTPException, Body, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse, FileResponse
from pydantic import BaseModel, Field

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import our professional music engine
try:
    from free.remix_daily import build_track, ensure_procedural_library
    from free.music_engine import SAMPLE_RATE
except ImportError as e:
    print(f"❌ Import Error: {e}")
    print("Make sure you're running from the generator/ directory")
    print("Required: free/remix_daily.py and free/music_engine.py")
    sys.exit(1)

# Setup logging with detailed formatting
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger("SoundFlowDJ")

# ============================================================================
# FASTAPI APP SETUP
# ============================================================================

app = FastAPI(
    title="SoundFlow Professional DJ Engine",
    description="Real-time AI music generation with Focus Engine & Smart Mixer",
    version="2.1.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS Configuration - Allow Frontend Access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production: ["https://yourdomain.com"]
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Storage paths
OUTPUT_DIR = Path(".soundflow_out/free")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TMP_DIR = Path(".soundflow_tmp/free")
TMP_DIR.mkdir(parents=True, exist_ok=True)

# Serve generated audio files
app.mount("/audio", StaticFiles(directory=OUTPUT_DIR), name="audio")

# ============================================================================
# DATA MODELS
# ============================================================================

class SynthParams(BaseModel):
    """
    Pro synthesis controls for the Smart Mixer.
    
    - cutoff: Filter frequency (0-100)
    - resonance: Filter resonance/Q (0-100)
    - drive: Distortion/saturation (0-100)
    - space: Reverb amount (0-100)
    """
    cutoff: float = Field(default=75.0, ge=0, le=100, description="Filter cutoff frequency")
    resonance: float = Field(default=30.0, ge=0, le=100, description="Filter resonance")
    drive: float = Field(default=10.0, ge=0, le=100, description="Saturation/distortion")
    space: float = Field(default=20.0, ge=0, le=100, description="Reverb amount")

class AmbienceParams(BaseModel):
    """
    Background ambience texture levels for Focus Engine.
    
    - rain: Rain sound intensity (0-100)
    - vinyl: Vinyl crackle intensity (0-100)
    - white: White noise intensity (0-100)
    """
    rain: float = Field(default=0.0, ge=0, le=100, description="Rain ambience level")
    vinyl: float = Field(default=0.0, ge=0, le=100, description="Vinyl crackle level")
    white: float = Field(default=0.0, ge=0, le=100, description="White noise level")

class GenerateRequest(BaseModel):
    """
    Complete request model for track generation.
    
    Supports both basic and advanced parameters for professional DJ use.
    """
    # Core parameters
    genre: str = Field(default="Trance", description="Music genre")
    bpm: int = Field(default=128, ge=60, le=200, description="Beats per minute")
    key: str = Field(default="A", description="Musical key (C, D, E, F, G, A, B)")
    layers: List[str] = Field(
        default=["drums", "bass", "music"], 
        description="Active layers: drums, bass, music, pad, texture, ambience"
    )
    duration: int = Field(default=180, ge=30, le=600, description="Duration in seconds")
    
    # Focus Engine parameters
    binaural: Optional[str] = Field(
        default="off", 
        description="Binaural mode: off, focus (Beta waves), or relax (Alpha waves)"
    )
    ambience: Optional[AmbienceParams] = Field(
        default=None, 
        description="Background ambience textures"
    )
    
    # Smart Mixer parameters
    intensity: Optional[float] = Field(
        default=50.0, 
        ge=0, 
        le=100, 
        description="Session intensity (0=ambient, 100=peak energy)"
    )
    synth_params: Optional[SynthParams] = Field(
        default=None, 
        description="Advanced synthesis controls"
    )
    energy_curve: Optional[str] = Field(
        default="peak", 
        description="Energy arrangement: linear, drop, or peak"
    )
    
    # Legacy/internal parameters
    scale: Optional[str] = Field(default="minor", description="Musical scale")
    instrument: Optional[str] = Field(default="hybrid", description="Instrument mode")
    texture: Optional[str] = Field(default="none", description="Texture type")
    target_lufs: Optional[float] = Field(default=-14.0, description="Target loudness in LUFS")

class TrackMetadata(BaseModel):
    """Response model with track information"""
    id: str
    name: str
    filename: str
    url: str
    genre: str
    bpm: int
    key: str
    duration: int
    generated_at: str
    # Optional extended metadata
    has_binaural: Optional[bool] = False
    intensity_level: Optional[str] = "balanced"
    energy_curve: Optional[str] = "peak"

class HealthResponse(BaseModel):
    """System health check response"""
    status: str
    engine: str
    version: str
    sample_rate: int
    output_dir: str
    tracks_count: int
    timestamp: str

class ErrorResponse(BaseModel):
    """Standardized error response"""
    status: str
    error: str
    detail: Optional[str] = None
    timestamp: str

# ============================================================================
# GLOBAL STATE
# ============================================================================

library_initialized = False
generation_stats = {
    "total_generated": 0,
    "last_generation_time": 0.0,
    "average_generation_time": 0.0
}

# ============================================================================
# STARTUP / SHUTDOWN
# ============================================================================

@app.on_event("startup")
async def startup_event():
    """Initialize the music engine on startup"""
    global library_initialized
    
    logger.info("=" * 70)
    logger.info("🎵 SoundFlow Professional DJ Engine v2.1.0")
    logger.info("=" * 70)
    logger.info(f"📁 Output Directory: {OUTPUT_DIR.absolute()}")
    logger.info(f"🎼 Sample Rate: {SAMPLE_RATE} Hz")
    logger.info(f"🌐 API Docs: http://localhost:8000/docs")
    
    # Initialize procedural library in background
    async def init_library():
        global library_initialized
        try:
            logger.info("🎛️  Initializing procedural library...")
            ensure_procedural_library(datetime.now().strftime("%Y-%m-%d"))
            library_initialized = True
            logger.info("✅ Procedural library ready")
        except Exception as e:
            logger.error(f"❌ Library initialization failed: {e}")
            logger.error("Engine will continue but generation may fail")
    
    asyncio.create_task(init_library())
    
    logger.info("🚀 SoundFlow DJ Engine Ready!")
    logger.info("=" * 70)

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    logger.info("🛑 SoundFlow DJ Engine Shutting Down...")
    logger.info(f"📊 Total tracks generated: {generation_stats['total_generated']}")
    logger.info(f"⏱️  Average generation time: {generation_stats['average_generation_time']:.1f}s")

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def get_intensity_label(intensity: float) -> str:
    """Convert intensity value to human-readable label"""
    if intensity < 30:
        return "ambient"
    elif intensity < 70:
        return "balanced"
    else:
        return "peak"

def validate_layers(layers: List[str]) -> bool:
    """Validate that layer names are correct"""
    valid_layers = {"drums", "bass", "music", "pad", "texture", "ambience"}
    return all(layer in valid_layers for layer in layers)

# ============================================================================
# API ENDPOINTS
# ============================================================================

@app.get("/", response_model=Dict[str, str])
async def root():
    """Root endpoint with service information"""
    return {
        "service": "SoundFlow Professional DJ Engine",
        "version": "2.1.0",
        "status": "online",
        "docs": "/docs",
        "health": "/api/health"
    }

@app.get("/api/health", response_model=HealthResponse)
async def health_check():
    """
    System health check endpoint.
    
    Returns:
    - System status
    - Engine version
    - Configuration details
    - Track count
    """
    try:
        track_count = len(list(OUTPUT_DIR.glob("*.mp3")))
        
        return HealthResponse(
            status="online" if library_initialized else "initializing",
            engine="High-Fidelity Professional + Focus Engine",
            version="2.1.0",
            sample_rate=SAMPLE_RATE,
            output_dir=str(OUTPUT_DIR.absolute()),
            tracks_count=track_count,
            timestamp=datetime.now().isoformat()
        )
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/library", response_model=Dict[str, Any])
async def get_library():
    """
    Get list of all generated tracks.
    
    Returns tracks sorted by creation time (newest first).
    """
    try:
        tracks = []
        
        for mp3_file in sorted(OUTPUT_DIR.glob("*.mp3"), key=os.path.getmtime, reverse=True):
            file_stat = mp3_file.stat()
            
            tracks.append({
                "id": mp3_file.stem,
                "name": mp3_file.stem.replace("-", " ").replace("_", " ").title(),
                "filename": mp3_file.name,
                "url": f"http://localhost:8000/audio/{mp3_file.name}",
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
async def generate_track(request: GenerateRequest = Body(...)):
    """
    Generate a new music track with Focus Engine & Smart Mixer.
    
    This endpoint creates a professional-quality track based on the provided
    parameters, including binaural beats for focus and advanced synthesis controls.
    
    Parameters:
    - genre: Music style (Trance, House, Techno, etc.)
    - bpm: Tempo (60-200)
    - key: Musical key
    - layers: Active instrument layers
    - binaural: Focus mode (off/focus/relax)
    - intensity: Energy level (0-100)
    - synth_params: Advanced controls
    - energy_curve: Arrangement style
    
    Returns:
    - Track metadata with playback URL
    """
    
    if not library_initialized:
        raise HTTPException(
            status_code=503, 
            detail="Engine is still initializing. Please wait a moment."
        )
    
    # Validate layers
    if not validate_layers(request.layers):
        raise HTTPException(
            status_code=400,
            detail="Invalid layer names. Valid: drums, bass, music, pad, texture, ambience"
        )
    
    try:
        # Generate unique ID
        timestamp = int(time.time())
        preset_id = f"{request.genre.lower()}-{timestamp}"
        
        logger.info("=" * 70)
        logger.info(f"🎵 Generation Request:")
        logger.info(f"   Genre: {request.genre} | BPM: {request.bpm} | Key: {request.key}")
        logger.info(f"   Layers: {', '.join(request.layers)}")
        logger.info(f"   Duration: {request.duration}s")
        logger.info(f"   Focus Mode: {request.binaural}")
        logger.info(f"   Intensity: {request.intensity}/100")
        logger.info(f"   Energy Curve: {request.energy_curve}")
        
        # Set defaults for optional parameters
        if request.ambience is None:
            request.ambience = AmbienceParams()
        
        if request.synth_params is None:
            request.synth_params = SynthParams()
        
        # === CONSTRUCT ENGINE PRESET ===
        # Map API request to internal engine format
        preset = {
            "id": preset_id,
            "title": f"AI {request.genre} {request.bpm}BPM",
            "genre": request.genre,
            "bpm": request.bpm,
            "key": request.key,
            "key_freq": 440.0,  # A440 standard
            "scale": request.scale or "minor",
            "instrument": request.instrument or "hybrid",
            "texture": request.texture or "none",
            
            # Layer Management
            "layers": {
                "enabled": request.layers
            },
            
            # Focus Engine Parameters
            "focus": {
                "binaural_mode": request.binaural,  # off, focus, relax
                "base_freq": 200.0 if request.binaural == "focus" else 150.0,
                "beat_freq": 20.0 if request.binaural == "focus" else 10.0,
                "ambience": {
                    "rain": request.ambience.rain / 100.0,
                    "vinyl": request.ambience.vinyl / 100.0,
                    "white": request.ambience.white / 100.0
                }
            },
            
            # Smart Mixer Parameters
            "smart_mixer": {
                "intensity": request.intensity,
                "intensity_label": get_intensity_label(request.intensity),
                "energy_curve": request.energy_curve,
                "synth": {
                    "cutoff": request.synth_params.cutoff / 100.0,
                    "resonance": request.synth_params.resonance / 100.0,
                    "drive": request.synth_params.drive / 100.0,
                    "space": request.synth_params.space / 100.0
                }
            },
            
            # Mix Parameters
            "mix": {
                "duration_sec": request.duration,
                "humanize_ms": 10,
                "target_lufs": request.target_lufs or -14.0,
                "noise": {
                    "enabled": any([
                        request.ambience.rain > 0,
                        request.ambience.vinyl > 0,
                        request.ambience.white > 0
                    ]),
                    "gain_db": -18.0,
                    "lowpass_hz": 6000,
                    "highpass_hz": 120
                }
            },
            
            # Metadata
            "metadata": {
                "created_at": datetime.now().isoformat(),
                "engine_version": "2.1.0",
                "has_focus_engine": request.binaural != "off",
                "intensity_level": get_intensity_label(request.intensity)
            }
        }
        
        # Generate the track
        logger.info(f"🔧 Building track: {preset_id}")
        start_time = time.time()
        
        mp3_path, entry = build_track(
            date=datetime.now().strftime("%Y-%m-%d"),
            preset=preset,
            total_sec=request.duration
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
        
        logger.info(f"✅ Generated in {generation_time:.1f}s: {mp3_path.name}")
        logger.info("=" * 70)
        
        # Build response
        track_metadata = TrackMetadata(
            id=entry.get("id", preset_id),
            name=entry.get("title", preset["title"]),
            filename=mp3_path.name,
            url=f"http://localhost:8000/audio/{mp3_path.name}",
            genre=request.genre,
            bpm=request.bpm,
            key=request.key,
            duration=request.duration,
            generated_at=datetime.now().isoformat(),
            has_binaural=request.binaural != "off",
            intensity_level=get_intensity_label(request.intensity),
            energy_curve=request.energy_curve
        )
        
        return track_metadata
    
    except Exception as e:
        logger.error(f"❌ Generation failed: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Track generation failed: {str(e)}"
        )

@app.post("/api/generate/batch")
async def generate_batch(
    requests: List[GenerateRequest],
    background_tasks: BackgroundTasks = None
):
    """
    Generate multiple tracks in batch.
    
    Useful for pre-generating a set list for a DJ session.
    Maximum 10 tracks per batch.
    """
    if not library_initialized:
        raise HTTPException(status_code=503, detail="Engine is initializing")
    
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
            track = await generate_track(req)
            results.append({"status": "success", "track": track})
        except Exception as e:
            logger.error(f"❌ Track {i} failed: {e}")
            results.append({
                "status": "error",
                "error": str(e),
                "request": req.dict()
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
        # Find matching files
        track_files = list(OUTPUT_DIR.glob(f"*{track_id}*.mp3"))
        
        if not track_files:
            raise HTTPException(
                status_code=404,
                detail=f"Track '{track_id}' not found"
            )
        
        # Delete files
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
    
    WARNING: This deletes all MP3 files in the output directory!
    Use with caution.
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

@app.get("/api/stats")
async def get_stats():
    """
    Get generation statistics.
    """
    return {
        "status": "success",
        "statistics": {
            "total_generated": generation_stats["total_generated"],
            "last_generation_time": round(generation_stats["last_generation_time"], 2),
            "average_generation_time": round(generation_stats["average_generation_time"], 2),
            "library_size": len(list(OUTPUT_DIR.glob("*.mp3"))),
            "uptime": "N/A"  # Could add uptime tracking
        }
    }

# ============================================================================
# ERROR HANDLERS
# ============================================================================

@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    """Custom HTTP exception handler with detailed errors"""
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
    """Catch-all exception handler for unexpected errors"""
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

# ============================================================================
# MAIN ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("🎵 SoundFlow Professional DJ Engine v2.1.0")
    print("=" * 70)
    print()
    print("🚀 Starting server...")
    print("🌐 Frontend:  http://localhost:3000")
    print("🔌 Backend:   http://localhost:8000")
    print("📡 API Docs:  http://localhost:8000/docs")
    print()
    print("💡 New Features:")
    print("   ✓ Focus Engine (Binaural beats)")
    print("   ✓ Smart Mixer (Pro synthesis)")
    print("   ✓ Energy Curve arrangement")
    print("   ✓ Ambience textures")
    print("   ✓ Batch generation")
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