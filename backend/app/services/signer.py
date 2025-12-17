from __future__ import annotations

from app.core.config import settings
from app.services.storage_r2 import r2_client


def presign_get(object_key: str, expires_in: int | None = None) -> str:
    """
    Generates a short-lived signed URL for a private audio object.
    """
    ttl = expires_in or settings.signed_url_ttl_sec
    s3 = r2_client()
    return s3.generate_presigned_url(
        ClientMethod="get_object",
        Params={"Bucket": settings.r2_bucket, "Key": object_key},
        ExpiresIn=ttl,
    )
