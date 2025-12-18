# backend/app/services/storage_r2.py
from __future__ import annotations

import boto3
from botocore.exceptions import ClientError
from app.core.config import settings


def get_s3_client():
    return boto3.client(
        "s3",
        endpoint_url=settings.r2_endpoint,
        aws_access_key_id=settings.r2_access_key_id,
        aws_secret_access_key=settings.r2_secret_access_key,
        region_name=settings.r2_region,
    )


def _clean_key(key: str) -> str:
    return key.strip().lstrip("/").strip("'").strip('"')


def get_object_bytes(key: str) -> bytes | None:
    """
    Return raw bytes from R2. None if missing (dev-friendly).
    """
    s3 = get_s3_client()
    clean_key = _clean_key(key)
    print(f"🔍 R2 Fetch: Bucket='{settings.r2_bucket}' Key='{clean_key}'")

    try:
        obj = s3.get_object(Bucket=settings.r2_bucket, Key=clean_key)
        body = obj["Body"].read()  # <-- bytes
        if not body:
            return b"[]"  # dev fallback
        return body
    except ClientError as e:
        if e.response["Error"]["Code"] == "NoSuchKey":
            print(f"❌ R2 Error: Key not found: {clean_key}")
            return b"[]"  # dev fallback
        raise


def put_json_bytes(key: str, body: bytes) -> None:
    """
    Upload pre-serialized JSON bytes.
    """
    s3 = get_s3_client()
    clean_key = _clean_key(key)
    print(f"⬆️ R2 Upload: Bucket='{settings.r2_bucket}' Key='{clean_key}'")

    s3.put_object(
        Bucket=settings.r2_bucket,
        Key=clean_key,
        Body=body,
        ContentType="application/json",
    )


def generate_presigned_url(object_key: str, expiration: int = 3600) -> str:
    s3 = get_s3_client()
    clean_key = _clean_key(object_key)
    try:
        return s3.generate_presigned_url(
            "get_object",
            Params={"Bucket": settings.r2_bucket, "Key": clean_key},
            ExpiresIn=expiration,
        )
    except ClientError as e:
        print(f"❌ R2 Sign Error: {e}")
        return ""
