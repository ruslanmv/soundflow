from __future__ import annotations

import boto3
from botocore.config import Config
from app.core.config import settings


def r2_client():
    """
    Returns an S3-compatible client configured for Cloudflare R2.
    Works for AWS S3 as well (change endpoint).
    """
    return boto3.client(
        "s3",
        endpoint_url=settings.r2_endpoint,
        aws_access_key_id=settings.r2_access_key_id,
        aws_secret_access_key=settings.r2_secret_access_key,
        region_name=settings.r2_region,
        config=Config(signature_version="s3v4"),
    )


def get_json_object(key: str) -> bytes:
    s3 = r2_client()
    obj = s3.get_object(Bucket=settings.r2_bucket, Key=key)
    return obj["Body"].read()


def put_json_object(key: str, data: bytes, content_type: str = "application/json") -> None:
    s3 = r2_client()
    s3.put_object(
        Bucket=settings.r2_bucket,
        Key=key,
        Body=data,
        ContentType=content_type,
        CacheControl="no-cache",
    )
