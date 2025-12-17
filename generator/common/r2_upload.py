from __future__ import annotations

import os
from pathlib import Path
import boto3
from botocore.config import Config


def r2_client():
    endpoint = os.environ["R2_ENDPOINT"]
    access = os.environ["R2_ACCESS_KEY_ID"]
    secret = os.environ["R2_SECRET_ACCESS_KEY"]
    region = os.getenv("R2_REGION", "auto")
    return boto3.client(
        "s3",
        endpoint_url=endpoint,
        aws_access_key_id=access,
        aws_secret_access_key=secret,
        region_name=region,
        config=Config(signature_version="s3v4"),
    )


def upload_file(local_path: Path, bucket: str, key: str, public: bool) -> None:
    s3 = r2_client()
    extra = {
        "ContentType": "audio/mpeg" if str(local_path).endswith(".mp3") else "audio/wav",
        "CacheControl": "public, max-age=31536000" if public else "private, max-age=0",
    }
    # R2 supports ACL in limited ways; safest is to control public/private by bucket policy.
    # We still set a conservative cache-control.
    s3.upload_file(str(local_path), bucket, key, ExtraArgs=extra)


def upload_json_bytes(data: bytes, bucket: str, key: str) -> None:
    s3 = r2_client()
    s3.put_object(
        Bucket=bucket,
        Key=key,
        Body=data,
        ContentType="application/json",
        CacheControl="no-cache",
    )
