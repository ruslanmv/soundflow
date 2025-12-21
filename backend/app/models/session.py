from __future__ import annotations

from pydantic import BaseModel, Field
from app.models.track import TrackPublic


class FreeSessionRequest(BaseModel):
    goal: str = Field(min_length=1)
    durationMin: int = Field(ge=1, le=360)
    energy: int = Field(ge=0, le=100)
    ambience: int = Field(ge=0, le=100)
    nature: str = Field(min_length=1)


class FreeSessionResponse(BaseModel):
    track: TrackPublic
