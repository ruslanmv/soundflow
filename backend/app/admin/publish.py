from __future__ import annotations

from fastapi import APIRouter
from pydantic import BaseModel, Field

from app.core.auth import require_admin
from app.services.catalog import CatalogService

router = APIRouter()


class PublishPayload(BaseModel):
    """
    Payload should be a list of track dicts matching:
    - Free: TrackPublic schema (with url)
    - Premium: TrackPremiumPrivate schema (with objectKey)
    """
    tracks: list[dict] = Field(default_factory=list)


@router.post("/publish/free")
def publish_free(payload: PublishPayload, _=require_admin):
    CatalogService().publish_free_catalog(payload.tracks)
    return {"ok": True, "published": "free", "count": len(payload.tracks)}


@router.post("/publish/premium")
def publish_premium(payload: PublishPayload, _=require_admin):
    CatalogService().publish_premium_catalog(payload.tracks)
    return {"ok": True, "published": "premium", "count": len(payload.tracks)}
