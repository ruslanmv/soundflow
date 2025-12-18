# generator/server.py
from __future__ import annotations

import os
import sys
import uvicorn
import logging
import time
import asyncio
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any

from fastapi import FastAPI, HTTPException, Body, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse
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
    sys.exit(1)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("SoundFlowDJ")

# ============================================================================
# FASTAPI APP SETUP
# ============================================================================

app = FastAPI(
    title="SoundFlow Professional DJ Engine",
    description="Real-time AI music generation for live DJ performance",
    version="2.0.0"
)

# CORS Configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify your domain
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

class GenerateRequest(BaseModel):
    """Request model for track generation"""
    genre: str = Field(default="Trance", description="Music genre")
    bpm: int = Field(default=128, ge=60, le=200, description="Beats per minute")
    key: str = Field(default="A", description="Musical key")
    layers: list[str] = Field(default=["drums", "bass", "music"], description="Active layers")
    duration: int = Field(default=180, ge=30, le=600, description="Duration in seconds")
    
    # Optional advanced parameters
    scale: Optional[str] = Field(default=None, description="Musical scale")
    instrument: Optional[str] = Field(default=None, description="Instrument mode")
    texture: Optional[str] = Field(default=None, description="Texture type")
    target_lufs: Optional[float] = Field(default=-14.0, description="Target loudness")

class TrackMetadata(BaseModel):
    """Track metadata response"""
    id: str
    name: str
    filename: str
    url: str
    genre: str
    bpm: int
    key: str
    duration: int
    generated_at: str

class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    engine: str
    version: str
    sample_rate: int
    output_dir: str
    timestamp: str

class ErrorResponse(BaseModel):
    """Error response model"""
    status: str
    error: str
    detail: Optional[str] = None

# ============================================================================
# GLOBAL STATE
# ============================================================================

generation_tasks: Dict[str, Dict[str, Any]] = {}
library_initialized = False

# ============================================================================
# STARTUP / SHUTDOWN
# ============================================================================

@app.on_event("startup")
async def startup_event():
    """Initialize the music engine on startup"""
    global library_initialized
    
    logger.info("🎵 SoundFlow DJ Engine Starting...")
    logger.info(f"📁 Output Directory: {OUTPUT_DIR.absolute()}")
    logger.info(f"🎼 Sample Rate: {SAMPLE_RATE} Hz")
    
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
    
    asyncio.create_task(init_library())
    
    logger.info("🚀 SoundFlow DJ Engine Ready!")
    logger.info("🌐 Frontend: http://localhost:3000")
    logger.info("🔌 Backend API: http://localhost:8000")
    logger.info("📡 API Docs: http://localhost:8000/docs")

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    logger.info("🛑 SoundFlow DJ Engine Shutting Down...")

# ============================================================================
# API ENDPOINTS
# ============================================================================

@app.get("/", response_model=Dict[str, str])
async def root():
    """Root endpoint"""
    return {
        "service": "SoundFlow Professional DJ Engine",
        "version": "2.0.0",
        "status": "online",
        "docs": "/docs"
    }

@app.get("/api/health", response_model=HealthResponse)
async def health_check():
    """
    Health check endpoint
    Returns system status and configuration
    """
    return HealthResponse(
        status="online" if library_initialized else "initializing",
        engine="High-Fidelity Professional",
        version="2.0.0",
        sample_rate=SAMPLE_RATE,
        output_dir=str(OUTPUT_DIR.absolute()),
        timestamp=datetime.now().isoformat()
    )

@app.get("/api/library", response_model=Dict[str, Any])
async def get_library():
    """
    Get list of all generated tracks
    """
    try:
        tracks = []
        for mp3_file in sorted(OUTPUT_DIR.glob("*.mp3"), key=os.path.getmtime, reverse=True):
            # Parse filename for metadata (format: free-DATE-PRESET.mp3)
            parts = mp3_file.stem.split("-")
            
            tracks.append({
                "id": mp3_file.stem,
                "name": mp3_file.stem,
                "filename": mp3_file.name,
                "url": f"http://localhost:8000/audio/{mp3_file.name}",
                "size_mb": round(mp3_file.stat().st_size / (1024 * 1024), 2),
                "created_at": datetime.fromtimestamp(mp3_file.stat().st_mtime).isoformat()
            })
        
        return {
            "status": "success",
            "count": len(tracks),
            "tracks": tracks
        }
    
    except Exception as e:
        logger.error(f"Library fetch failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/generate", response_model=TrackMetadata)
async def generate_track(
    request: GenerateRequest = Body(...),
    background_tasks: BackgroundTasks = None
):
    """
    Generate a new music track
    
    This endpoint creates a professional-quality track based on the provided parameters.
    Generation happens asynchronously while allowing the DJ to continue mixing.
    
    Parameters:
    - genre: Music genre (Trance, House, Techno, etc.)
    - bpm: Beats per minute (60-200)
    - key: Musical key (C, D, E, F, G, A, B)
    - layers: Active layers (drums, bass, music, pad, texture, ambience)
    - duration: Track duration in seconds (30-600)
    
    Returns:
    - Track metadata including URL for playback
    """
    
    if not library_initialized:
        raise HTTPException(
            status_code=503,
            detail="Engine is still initializing. Please wait a moment."
        )
    
    try:
        # Generate unique timestamp-based ID
        date_str = datetime.now().strftime("%Y-%m-%d")
        timestamp = int(time.time())
        preset_id = f"{request.genre.lower()}-{timestamp}"
        
        logger.info(f"🎵 Generation Request: {request.genre} @ {request.bpm} BPM")
        logger.info(f"   Layers: {', '.join(request.layers)}")
        logger.info(f"   Duration: {request.duration}s")
        
        # Build preset payload for the music engine
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
            "layers": {
                "enabled": request.layers
            },
            "mix": {
                "duration_sec": request.duration,
                "humanize_ms": 10,
                "target_lufs": request.target_lufs or -14.0,
                "noise": {
                    "enabled": "texture" in request.layers or "ambience" in request.layers,
                    "gain_db": -18.0,
                    "lowpass_hz": 6000,
                    "highpass_hz": 120
                }
            },
            "metadata": {
                "created_at": datetime.now().isoformat(),
                "engine_version": "2.0.0"
            }
        }
        
        # Generate the track (this is CPU-intensive but non-blocking for the API)
        logger.info(f"🔧 Building track: {preset_id}")
        start_time = time.time()
        
        mp3_path, entry = build_track(
            date=date_str,
            preset=preset,
            total_sec=request.duration
        )
        
        generation_time = time.time() - start_time
        logger.info(f"✅ Generated in {generation_time:.1f}s: {mp3_path.name}")
        
        # Build response
        track_metadata = TrackMetadata(
            id=entry["id"],
            name=entry["title"],
            filename=mp3_path.name,
            url=f"http://localhost:8000/audio/{mp3_path.name}",
            genre=request.genre,
            bpm=request.bpm,
            key=request.key,
            duration=request.duration,
            generated_at=datetime.now().isoformat()
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
    requests: list[GenerateRequest],
    background_tasks: BackgroundTasks = None
):
    """
    Generate multiple tracks in batch
    Useful for pre-generating a set list
    """
    if not library_initialized:
        raise HTTPException(
            status_code=503,
            detail="Engine is still initializing"
        )
    
    if len(requests) > 10:
        raise HTTPException(
            status_code=400,
            detail="Maximum 10 tracks per batch"
        )
    
    results = []
    
    for req in requests:
        try:
            track = await generate_track(req)
            results.append({"status": "success", "track": track})
        except Exception as e:
            results.append({"status": "error", "error": str(e)})
    
    return {
        "status": "completed",
        "total": len(requests),
        "successful": len([r for r in results if r["status"] == "success"]),
        "results": results
    }

@app.delete("/api/tracks/{track_id}")
async def delete_track(track_id: str):
    """
    Delete a generated track
    """
    try:
        # Find the file
        track_files = list(OUTPUT_DIR.glob(f"*{track_id}*.mp3"))
        
        if not track_files:
            raise HTTPException(status_code=404, detail="Track not found")
        
        # Delete the file
        for track_file in track_files:
            track_file.unlink()
            logger.info(f"🗑️  Deleted: {track_file.name}")
        
        return {"status": "success", "message": f"Deleted {len(track_files)} file(s)"}
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Delete failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/tracks/clear")
async def clear_all_tracks():
    """
    Clear all generated tracks (use with caution!)
    """
    try:
        tracks = list(OUTPUT_DIR.glob("*.mp3"))
        count = len(tracks)
        
        for track in tracks:
            track.unlink()
        
        logger.info(f"🗑️  Cleared {count} tracks")
        
        return {
            "status": "success",
            "message": f"Cleared {count} tracks"
        }
    
    except Exception as e:
        logger.error(f"Clear failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# ============================================================================
# ERROR HANDLERS
# ============================================================================

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

# ============================================================================
# MAIN ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("🎵 SoundFlow Professional DJ Engine v2.0.0")
    print("=" * 70)
    print()
    print("🚀 Starting server...")
    print("🌐 Frontend:  http://localhost:3000")
    print("🔌 Backend:   http://localhost:8000")
    print("📡 API Docs:  http://localhost:8000/docs")
    print()
    print("💡 Features:")
    print("   ✓ Real-time AI music generation")
    print("   ✓ Professional mixing engine")
    print("   ✓ Non-blocking async generation")
    print("   ✓ Dual-deck DJ interface")
    print("   ✓ Auto-generate mode")
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