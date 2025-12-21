from __future__ import annotations

from typing import Literal, Optional
from pydantic import BaseModel, Field


Tier = Literal["free", "premium"]


class TrackBase(BaseModel):
    id: str
    title: str
    tier: Tier
    date: str  # YYYY-MM-DD
    category: str  # Deep Work / Study / Relax / Nature / Flow State
    durationSec: int = Field(ge=1)

    # Matching tags for deterministic routing
    goalTags: list[str] = Field(default_factory=list)
    natureTags: list[str] = Field(default_factory=list)
    energyMin: int = 0
    energyMax: int = 100
    ambienceMin: int = 0
    ambienceMax: int = 100


class TrackPublic(TrackBase):
    """
    Public representation:
    - for free tracks, `url` is public
    - for premium tracks, `url` can be omitted or point to a landing route (NOT the audio object)
    """
    url: Optional[str] = None


class TrackPremiumPrivate(TrackBase):
    """
    Premium private catalog entry stored in premium-index.json.
    The object key should point to a private object in R2.
    """
    objectKey: str


class TrackPremiumSigned(TrackBase):
    """
    Response model when client requests a signed URL.
    """
    signedUrl: str
    expiresInSec: int
