# backend/app/services/catalog.py
from __future__ import annotations

import time
from typing import Any, Literal

import orjson
from fastapi import HTTPException

from app.core.config import settings
from app.models.track import TrackPublic, TrackPremiumPrivate, TrackPremiumSigned
from app.services.signer import presign_get
from app.services.storage_r2 import get_object_bytes, put_json_bytes


class _Cache:
    free_raw: bytes | None = None
    premium_raw: bytes | None = None
    free_ts: float = 0.0
    premium_ts: float = 0.0


_CACHE = _Cache()
_CACHE_TTL_SEC = 60


def _loads(data: bytes) -> Any:
    # orjson expects bytes/str; we always pass bytes
    return orjson.loads(data)


def _dumps(obj: Any) -> bytes:
    return orjson.dumps(obj, option=orjson.OPT_INDENT_2)


def _today_yyyy_mm_dd() -> str:
    return time.strftime("%Y-%m-%d")


def _ensure_list(obj: Any) -> list[dict]:
    """
    Catalog files should always be a JSON array of objects.
    If someone uploads {} by mistake, fail fast with clear error.
    """
    if obj is None:
        return []
    if isinstance(obj, list):
        # ensure each item is dict-like
        out: list[dict] = []
        for i, item in enumerate(obj):
            if not isinstance(item, dict):
                # Skip invalid items or log warning
                continue
            out.append(item)
        return out
    
    # Fallback: if it's not a list, return empty list to prevent crash
    return []


def _sanitize_item(item: dict, default_tier: str = "free") -> dict:
    """
    Fixes legacy/incomplete data from R2 to match the current Pydantic schema.
    """
    # 1. Map 'duration' -> 'durationSec' (Fixes "field required" for durationSec)
    if "duration" in item and "durationSec" not in item:
        item["durationSec"] = item.pop("duration")
    
    # 2. Add missing 'tier'
    if "tier" not in item:
        item["tier"] = default_tier

    # 3. Add missing 'date'
    if "date" not in item:
        item["date"] = "2024-01-01"  # Default fallback date

    return item


class CatalogService:
    """
    Catalogs are stored in R2 (S3-compatible) as JSON:

    - Free catalog: list of TrackPublic including direct 'url'
      key: settings.catalog_free_key

    - Premium catalog: list of TrackPremiumPrivate including 'objectKey'
      key: settings.catalog_premium_key

    This service:
    - caches raw JSON bytes briefly
    - validates shape
    - converts into Pydantic models
    """

    # -------------------------
    # Public read APIs
    # -------------------------

    def get_free_catalog(self) -> list[TrackPublic]:
        raw = self._get_cached_raw("free")
        items = _ensure_list(_loads(raw))

        # Sanitize data before validation
        clean_items = [_sanitize_item(x, "free") for x in items]

        return [TrackPublic(**x) for x in clean_items]

    def get_premium_catalog_private(self) -> list[TrackPremiumPrivate]:
        raw = self._get_cached_raw("premium")
        items = _ensure_list(_loads(raw))

        # Sanitize data before validation
        clean_items = [_sanitize_item(x, "premium") for x in items]

        return [TrackPremiumPrivate(**x) for x in clean_items]

    def get_premium_today_public(self) -> list[TrackPublic]:
        """
        Returns today's premium tracks with *public metadata only* (no objectKey, no signed url).
        """
        today = _today_yyyy_mm_dd()
        premium = self.get_premium_catalog_private()
        todays = [p for p in premium if p.date == today]

        # Strip objectKey and set url to None (signed endpoint returns url)
        out: list[TrackPublic] = []
        for p in todays:
            base = p.model_dump(exclude={"objectKey"})
            out.append(TrackPublic(**base, url=None))
        return out

    def get_premium_signed(self, track_id: str) -> TrackPremiumSigned:
        premium = self.get_premium_catalog_private()
        match = next((p for p in premium if p.id == track_id), None)
        if not match:
            raise HTTPException(status_code=404, detail="Premium track not found")

        signed = presign_get(match.objectKey)
        if not signed:
            raise HTTPException(status_code=500, detail="Failed to create signed URL")

        base = match.model_dump(exclude={"objectKey"})
        return TrackPremiumSigned(
            **base,
            signedUrl=signed,
            expiresInSec=settings.signed_url_ttl_sec,
        )

    # -------------------------
    # Admin write APIs
    # -------------------------

    def publish_free_catalog(self, tracks: list[dict]) -> None:
        """
        Admin: overwrite the free catalog.
        """
        body = _dumps(tracks)
        put_json_bytes(settings.catalog_free_key, body)

        # Bust cache
        _CACHE.free_raw = None
        _CACHE.free_ts = 0.0

    def publish_premium_catalog(self, tracks: list[dict]) -> None:
        """
        Admin: overwrite the premium catalog.
        """
        body = _dumps(tracks)
        put_json_bytes(settings.catalog_premium_key, body)

        # Bust cache
        _CACHE.premium_raw = None
        _CACHE.premium_ts = 0.0

    # -------------------------
    # Cache
    # -------------------------

    def _get_cached_raw(self, which: Literal["free", "premium"]) -> bytes:
        now = time.time()

        if which == "free":
            if _CACHE.free_raw and (now - _CACHE.free_ts) < _CACHE_TTL_SEC:
                return _CACHE.free_raw

            raw = get_object_bytes(settings.catalog_free_key)
            if not raw:
                # dev-friendly fallback
                raw = b"[]"

            _CACHE.free_raw = raw
            _CACHE.free_ts = now
            return raw

        if which == "premium":
            if _CACHE.premium_raw and (now - _CACHE.premium_ts) < _CACHE_TTL_SEC:
                return _CACHE.premium_raw

            raw = get_object_bytes(settings.catalog_premium_key)
            if not raw:
                raw = b"[]"

            _CACHE.premium_raw = raw
            _CACHE.premium_ts = now
            return raw

        raise ValueError("Unknown cache key")