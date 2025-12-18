# generator/server.py
from __future__ import annotations
import os
import uvicorn
from fastapi import FastAPI, HTTPException, Body
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from pathlib import Path
import logging

# Import your existing logic
# We need to make sure python can find the modules
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from free.remix_daily import build_track

# Setup Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("SoundFlowAPI")

app = FastAPI()

# 1. Allow React (port 3000) to talk to Python (port 8000)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # For local dev, allow all
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 2. Serve the generated MP3s so the browser can play them
OUTPUT_DIR = Path(".soundflow_out/free")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
app.mount("/audio", StaticFiles(directory=OUTPUT_DIR), name="audio")

class GenerateRequest(BaseModel):
    date: str = "2025-01-01"
    preset: dict

@app.post("/api/generate")
async def generate_track(payload: dict = Body(...)):
    """
    Receives a single combo JSON from the UI and generates audio.
    """
    try:
        logger.info(f"Received request: {payload.get('name', 'Unknown')}")
        
        # Extract parameters
        # The UI sends a "combinations" array usually, or a single object.
        # We handle single object generation here for the "Play" button.
        
        # If UI sends the whole schema wrapper, grab the first combo
        if "combinations" in payload:
            data = payload["combinations"][0]
        else:
            data = payload

        # Run the generator (reusing your existing high-quality logic)
        # We use a fixed date for testing or the one provided
        date_str = "2025-12-18" 
        duration = int(data.get("mix", {}).get("duration_sec", 120))
        
        mp3_path, entry = build_track(date_str, data, duration)
        
        # Return the URL for the frontend to play
        filename = mp3_path.name
        return {
            "status": "success", 
            "url": f"http://localhost:8000/audio/{filename}",
            "filename": filename
        }

    except Exception as e:
        logger.error(f"Generation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    print("🚀 Starting API Backend on http://localhost:8000")
    uvicorn.run(app, host="0.0.0.0", port=8000)