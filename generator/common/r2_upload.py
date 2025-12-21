from __future__ import annotations

import mimetypes
import os
from pathlib import Path
from typing import Optional

import boto3
from botocore.config import Config


def _clean_key(key: str) -> str:
    return key.strip().lstrip("/").strip("'").strip('"')


def normalize_key(key: str) -> str:
    """
    Canonical key normalization used by *both* reads and writes.
    Applies R2_PREFIX consistently.
    """
    prefix = os.getenv("R2_PREFIX", "").strip().strip("/")
    clean = _clean_key(key)
    return f"{prefix}/{clean}" if prefix else clean


def _guess_content_type(path: Path) -> str:
    ext = path.suffix.lower()
    if ext == ".mp3":
        return "audio/mpeg"
    if ext == ".wav":
        return "audio/wav"
    if ext == ".ogg":
        return "audio/ogg"
    if ext == ".m4a":
        return "audio/mp4"
    if ext == ".aac":
        return "audio/aac"
    if ext == ".flac":
        return "audio/flac"

    guess, _ = mimetypes.guess_type(str(path))
    return guess or "application/octet-stream"


def _env(name: str) -> str:
    v = os.getenv(name)
    if not v:
        raise RuntimeError(f"Missing required env var: {name}")
    return v


def r2_client():
    endpoint = _env("R2_ENDPOINT")
    access = _env("R2_ACCESS_KEY_ID")
    secret = _env("R2_SECRET_ACCESS_KEY")
    region = os.getenv("R2_REGION", "auto")

    return boto3.client(
        "s3",
        endpoint_url=endpoint,
        aws_access_key_id=access,
        aws_secret_access_key=secret,
        region_name=region,
        config=Config(signature_version="s3v4"),
    )


def get_object_bytes(bucket: str, key: str) -> bytes:
    """
    Reads object bytes from R2 using normalized key (prefix-safe).
    """
    s3 = r2_client()
    final_key = normalize_key(key)
    print(f"📥 R2 Read: bucket='{bucket}' key='{final_key}'")
    obj = s3.get_object(Bucket=bucket, Key=final_key)
    return obj["Body"].read()


def upload_file(
    local_path: Path,
    bucket: str,
    key: str,
    public: bool = False,
    cache_seconds: Optional[int] = None,
) -> str:
    s3 = r2_client()

    local_path = Path(local_path)
    if not local_path.exists():
        raise FileNotFoundError(f"Local file not found: {local_path}")

    final_key = normalize_key(key)
    content_type = _guess_content_type(local_path)

    if cache_seconds is None:
        cache_seconds = 31536000 if public else 0

    cache_control = (
        f"public, max-age={cache_seconds}, immutable"
        if public and cache_seconds > 0
        else "private, no-store"
    )

    extra = {"ContentType": content_type, "CacheControl": cache_control}

    print(f"⬆️ R2 Upload: bucket='{bucket}' key='{final_key}' type='{content_type}' public={public}")
    s3.upload_file(str(local_path), bucket, final_key, ExtraArgs=extra)
    return final_key


def upload_json_bytes(data: bytes, bucket: str, key: str) -> str:
    s3 = r2_client()
    final_key = normalize_key(key)

    print(f"⬆️ R2 Upload JSON: bucket='{bucket}' key='{final_key}'")
    s3.put_object(
        Bucket=bucket,
        Key=final_key,
        Body=data,
        ContentType="application/json",
        CacheControl="no-cache",
    )
    return final_key


def upload_json_obj(obj: object, bucket: str, key: str) -> str:
    import json
    data = json.dumps(obj, indent=2, ensure_ascii=False).encode("utf-8")
    return upload_json_bytes(data, bucket=bucket, key=key)
