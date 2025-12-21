from __future__ import annotations

from app.core.config import settings
from app.services.storage_r2 import generate_presigned_url


def presign_get(object_key: str, expires_in: int | None = None) -> str:
    """
    Generates a short-lived signed URL for a private audio object.
    """
    ttl = expires_in or settings.signed_url_ttl_sec
    
    # We now delegate directly to the function in storage_r2.py
    return generate_presigned_url(object_key, expiration=ttl)