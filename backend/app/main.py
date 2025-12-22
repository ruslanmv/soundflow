from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.core.config import settings
from app.admin.publish import router as admin_router
from app.services.catalog import CatalogService
from app.services.router import pick_best_track
from app.core.auth import require_premium
from app.models.session import FreeSessionRequest, FreeSessionResponse
from app.models.track import TrackPublic, TrackPremiumSigned

app = FastAPI(title="SoundFlow API", version="1.0.0")

# CORS
origins = settings.cors_origins_list
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Routers
app.include_router(admin_router, prefix="/admin", tags=["admin"])


@app.get("/health")
def health():
    return {"ok": True, "service": "soundflow-backend"}


@app.get("/tracks/free", response_model=list[TrackPublic])
def tracks_free():
    """
    Returns the Free catalog (public URLs or public object paths).
    """
    catalog = CatalogService()
    return catalog.get_free_catalog()


@app.get("/tracks/premium/daily", response_model=list[TrackPublic])
def tracks_premium_daily():
    """
    Returns today's premium track metadata (NO signed URLs here).
    Client must request signed URL per track.
    """
    catalog = CatalogService()
    return catalog.get_premium_today_public()


@app.get("/tracks/premium/{track_id}/signed", response_model=TrackPremiumSigned)
def track_premium_signed(track_id: str, _=require_premium):
    """
    Premium-only. Returns a short-lived signed URL to the private audio object.
    """
    catalog = CatalogService()
    return catalog.get_premium_signed(track_id)


@app.post("/session/free", response_model=FreeSessionResponse)
def session_free(req: FreeSessionRequest):
    """
    Deterministic routing for free users:
    Picks best track from free catalog based on goal/duration/energy/ambience/nature.
    """
    catalog = CatalogService()
    tracks = catalog.get_free_catalog()

    chosen = pick_best_track(
        tracks=tracks,
        goal=req.goal,
        duration_min=req.durationMin,
        energy=req.energy,
        ambience=req.ambience,
        nature=req.nature,
    )

    return FreeSessionResponse(track=chosen)


# ==============================================================================
# CATALOG ENDPOINTS (For Frontend Playlist/Browse)
# ==============================================================================


@app.get("/catalog/all", response_model=list[TrackPublic])
def catalog_all():
    """
    Returns the general catalog (all tracks: free + premium).

    This is the main index for browsing all available tracks.
    Premium tracks will have objectKey but not signed URLs.
    """
    catalog = CatalogService()
    return catalog.get_general_catalog()


@app.get("/catalog/by-category/{category}", response_model=list[TrackPublic])
def catalog_by_category(category: str):
    """
    Returns tracks for a specific site category.

    Categories:
    - deep-work
    - study
    - relax
    - nature
    - flow-state
    """
    catalog = CatalogService()
    return catalog.get_category_catalog(category)


@app.get("/catalog/premium", response_model=list[TrackPublic])
def catalog_premium():
    """
    Returns all premium tracks (metadata only, no signed URLs).

    For playback, client must request signed URL per track via:
    GET /tracks/premium/{track_id}/signed
    """
    catalog = CatalogService()
    return catalog.get_premium_catalog()
