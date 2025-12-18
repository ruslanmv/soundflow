# generator/server.py
from __future__ import annotations
import os
import uvicorn
import shutil
import subprocess
import logging
from fastapi import FastAPI, HTTPException, Body
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from pathlib import Path
from typing import List, Optional

# Import your existing logic
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from free.remix_daily import build_track, get_random_variant
from free.music_engine import save_wav, SAMPLE_RATE
from common.audio_utils import ffmpeg_loudnorm, ffmpeg_encode_mp3

# Setup Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("SoundFlowDJ")

app = FastAPI(title="SoundFlow Enterprise DJ Engine")

# 1. CORS: Allow React (port 3000) to talk to Python (port 8000)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 2. Storage Setup
OUTPUT_DIR = Path(".soundflow_out/free")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
TMP_DIR = Path(".soundflow_tmp/dj_mix")
TMP_DIR.mkdir(parents=True, exist_ok=True)

# Serve audio files statically
app.mount("/audio", StaticFiles(directory=OUTPUT_DIR), name="audio")

# --- DATA MODELS ---

class GenerateRequest(BaseModel):
    date: str = "2025-01-01"
    preset: dict

class MergeRequest(BaseModel):
    track_a: str  # Filename of first track
    track_b: str  # Filename of second track
    operation: str = "overlay" # overlay (mix), append (sequence)
    balance: float = 0.5 # 0.0 = A only, 1.0 = B only, 0.5 = Mix

class LayerRequest(BaseModel):
    base_track: str # Filename of existing track
    layer_type: str # e.g., "kick_techno", "texture_rain"
    variant: int = 1

# --- HELPER FUNCTIONS ---

def ffmpeg_merge(file_a: Path, file_b: Path, out_path: Path):
    """
    High-fidelity mixing of two audio streams using FFmpeg filter_complex.
    Normalizes the mix to prevent clipping.
    """
    cmd = [
        "ffmpeg", "-y",
        "-i", str(file_a),
        "-i", str(file_b),
        "-filter_complex",
        "amix=inputs=2:duration=longest:dropout_transition=2,loudnorm=I=-14:TP=-1.5:LRA=11",
        str(out_path)
    ]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

# --- API ENDPOINTS ---

@app.get("/api/health")
async def health_check():
    return {"status": "online", "mode": "Enterprise DJ"}

@app.get("/api/list")
async def list_tracks():
    """Returns a list of all generated tracks available in the 'crate'."""
    files = sorted(list(OUTPUT_DIR.glob("*.mp3")), key=os.path.getmtime, reverse=True)
    return {
        "count": len(files),
        "tracks": [f.name for f in files]
    }

@app.post("/api/generate")
async def generate_track(payload: dict = Body(...)):
    """
    Core Generation: Creates a track from a recipe.
    """
    try:
        logger.info(f"Received generation request")
        
        # Normalize payload structure (handle array vs single object)
        if "combinations" in payload:
            data = payload["combinations"][0]
        else:
            data = payload

        # Run the generator
        date_str = "2025-12-18" # Fixed date for deterministic seed, or use current
        duration = int(data.get("mix", {}).get("duration_sec", 120))
        
        mp3_path, entry = build_track(date_str, data, duration)
        
        return {
            "status": "success", 
            "url": f"http://localhost:8000/audio/{mp3_path.name}",
            "filename": mp3_path.name,
            "meta": entry
        }

    except Exception as e:
        logger.error(f"Generation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/dj/merge")
async def mix_tracks(req: MergeRequest):
    """
    DJ Mixer: Merges two existing tracks into a new master mix.
    Useful for transitions or mashing up two different styles.
    """
    try:
        path_a = OUTPUT_DIR / req.track_a
        path_b = OUTPUT_DIR / req.track_b
        
        if not path_a.exists() or not path_b.exists():
            raise HTTPException(status_code=404, detail="One or more tracks not found")

        # Generate unique name for the mix
        mix_id = f"mix_{req.track_a.split('.')[0]}_{req.track_b.split('.')[0]}"[:50]
        output_wav = TMP_DIR / f"{mix_id}.wav"
        output_mp3 = OUTPUT_DIR / f"DJ_MIX_{mix_id}.mp3"

        # Perform the Mix
        ffmpeg_merge(path_a, path_b, output_wav)
        
        # Convert to MP3 for streaming
        ffmpeg_encode_mp3(output_wav, output_mp3, bitrate="320k") # High quality for DJ
        
        return {
            "status": "success",
            "message": "Tracks mixed successfully",
            "url": f"http://localhost:8000/audio/{output_mp3.name}",
            "filename": output_mp3.name
        }

    except subprocess.CalledProcessError:
        raise HTTPException(status_code=500, detail="FFmpeg mixing failed")
    except Exception as e:
        logger.error(e)
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/dj/layer")
async def layer_stem(req: LayerRequest):
    """
    Live Overdub: Generates a NEW stem (e.g., extra drums) and layers it 
    on top of an existing track.
    """
    try:
        base_path = OUTPUT_DIR / req.base_track
        if not base_path.exists():
            raise HTTPException(status_code=404, detail="Base track not found")

        # 1. Generate the isolated stem
        stem_name = f"live_{req.layer_type}_{req.variant}.wav"
        stem_path = TMP_DIR / stem_name
        
        # We need to hook into the library generator helper
        # Since logic is inside remix_daily, we cheat slightly by using the engine directly
        # or we rely on pre-existing stems if speed is critical.
        # For true generative power, we look for the file in assets:
        
        import random
        # Try to find a source stem from assets matching the request
        # This simulates "Generating" the stem logic
        source_stem = get_random_variant(req.layer_type, random.Random())
        
        if not source_stem:
             raise HTTPException(status_code=400, detail=f"Generator for {req.layer_type} not found")

        # 2. Loop the stem to match the base track duration
        # We need the duration of the base track first.
        # For simplicity in this demo, we assume 120s or mix simply.
        
        output_mp3 = OUTPUT_DIR / f"REMIX_{req.layer_type}_{req.base_track}"
        
        # Mix base + new stem
        # Note: We are mixing an MP3 (base) with a WAV (stem)
        ffmpeg_merge(base_path, source_stem, TMP_DIR / "temp_layer.wav")
        ffmpeg_encode_mp3(TMP_DIR / "temp_layer.wav", output_mp3, bitrate="192k")

        return {
            "status": "success",
            "message": f"Layered {req.layer_type} onto track",
            "url": f"http://localhost:8000/audio/{output_mp3.name}",
            "filename": output_mp3.name
        }

    except Exception as e:
        logger.error(e)
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    print("🚀 Starting Enterprise DJ Server on http://localhost:8000")
    print("🎛️  Ready for real-time mixing...")
    uvicorn.run(app, host="0.0.0.0", port=8000)