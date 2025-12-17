from __future__ import annotations

import random
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

app = FastAPI(title="SoundFlow AI API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Replace with your Vercel domain in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class SessionRequest(BaseModel):
    goal: str = Field(min_length=1)
    durationMin: int = Field(ge=1, le=12 * 60)
    energy: int = Field(ge=0, le=100)
    ambience: int = Field(ge=0, le=100)
    nature: str = Field(min_length=1)

@app.get("/")
def read_root():
    return {"status": "SoundFlow Backend Active"}

@app.post("/generate-session")
def generate_session(req: SessionRequest):
    bpm = 70 + (req.energy / 100 * 50)
    return {
        "id": f"sess_{random.randint(1000, 9999)}",
        "plan": {
            "tempoBpm": int(bpm),
            "key": "C Minor",
            "instrumentation": "Synths, Piano",
            "ambiencePrompt": f"{req.nature} sounds",
            "durationSec": req.durationMin * 60,
        },
        "musicUrl": "https://cdn.pixabay.com/download/audio/2022/03/15/audio_2b8b2b51aa.mp3?filename=ambient-11090.mp3",
        "ambienceUrl": "https://cdn.pixabay.com/download/audio/2022/02/23/audio_0f1c2f3c28.mp3?filename=rain-ambient-11060.mp3",
        "durationSec": req.durationMin * 60,
    }
