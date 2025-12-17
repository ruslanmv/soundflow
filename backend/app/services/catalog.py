from __future__ import annotations

import time
import orjson
from fastapi import HTTPException

from app.core.config import settings
from app.models.track import TrackPublic, TrackPremiumPrivate, TrackPremiumSigned
from app.services.storage_r2 import get_json_object, put_json_object
from app.services.signer import presign_get


class _Cache:
    free_json: bytes | None = None
    premium_json: bytes | None = None
    free_ts: float = 0
    premium_ts: float = 0


_CACHE = _Cache()
_CACHE_TTL_SEC = 60


def _loads(data: bytes):
    return orjson.loads(data)


def _dumps(obj) -> bytes:
    return orjson.dumps(obj, option=orjson.OPT_INDENT_2)


class CatalogService:
    """
    Catalogs are stored as JSON in R2:
    - free-index.json: public URLs or public paths
    - premium-index.json: private objectKey, signed at request time
    """

    def get_free_catalog(self) -> list[TrackPublic]:
        raw = self._get_cached("free")
        items = _loads(raw)
        return [TrackPublic(**x) for x in items]

    def get_premium_catalog_private(self) -> list[TrackPremiumPrivate]:
        raw = self._get_cached("premium")
        items = _loads(raw)
        return [TrackPremiumPrivate(**x) for x in items]

    def get_premium_today_public(self) -> list[TrackPublic]:
        """
        Returns today's premium tracks as public metadata only.
        """
        today = time.strftime("%Y-%m-%d")
        premium = self.get_premium_catalog_private()
        todays = [p for p in premium if p.date == today]
        return [
            TrackPublic(**p.model_dump(exclude={"objectKey"}), url=None)
            for p in todays
        ]

    def get_premium_signed(self, track_id: str) -> TrackPremiumSigned:
        premium = self.get_premium_catalog_private()
        match = next((p for p in premium if p.id == track_id), None)
        if not match:
            raise HTTPException(status_code=404, detail="Premium track not found")

        signed = presign_get(match.objectKey)
        base = match.model_dump(exclude={"objectKey"})
        return TrackPremiumSigned(**base, signedUrl=signed, expiresInSec=settings.signed_url_ttl_sec)

    def publish_free_catalog(self, tracks: list[dict]) -> None:
        """
        Admin: overwrite the free catalog.
        """
        put_json_object(settings.catalog_free_key, _dumps(tracks))

        # Bust cache
        _CACHE.free_json = None
        _CACHE.free_ts = 0

    def publish_premium_catalog(self, tracks: list[dict]) -> None:
        """
        Admin: overwrite the premium catalog.
        """
        put_json_object(settings.catalog_premium_key, _dumps(tracks))

        # Bust cache
        _CACHE.premium_json = None
        _CACHE.premium_ts = 0

    def _get_cached(self, which: str) -> bytes:
        now = time.time()

        if which == "free":
            if _CACHE.free_json and (now - _CACHE.free_ts) < _CACHE_TTL_SEC:
                return _CACHE.free_json
            data = get_json_object(settings.catalog_free_key)
            _CACHE.free_json = data
            _CACHE.free_ts = now
            return data

        if which == "premium":
            if _CACHE.premium_json and (now - _CACHE.premium_ts) < _CACHE_TTL_SEC:
                return _CACHE.premium_json
            data = get_json_object(settings.catalog_premium_key)
            _CACHE.premium_json = data
            _CACHE.premium_ts = now
            return data

        raise ValueError("Unknown cache key")
