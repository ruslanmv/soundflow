from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

import orjson

from common.r2_upload import upload_json_bytes, get_object_bytes


@dataclass
class CatalogPaths:
    free_key: str
    premium_key: str


def get_catalog_paths() -> CatalogPaths:
    return CatalogPaths(
        free_key=os.getenv("CATALOG_FREE_KEY", "catalog/free-index.json"),
        premium_key=os.getenv("CATALOG_PREMIUM_KEY", "catalog/premium-index.json"),
    )


def _loads(b: bytes) -> Any:
    return orjson.loads(b)


def _dumps(obj: Any) -> bytes:
    return orjson.dumps(obj, option=orjson.OPT_INDENT_2)


def read_catalog(bucket: str, key: str) -> list[dict]:
    try:
        raw = get_object_bytes(bucket=bucket, key=key)   # <-- prefix-safe read
        data = _loads(raw)
        return data if isinstance(data, list) else []
    except Exception:
        return []


def upsert_tracks(existing: list[dict], new_tracks: list[dict]) -> list[dict]:
    by_id = {t["id"]: t for t in existing if "id" in t}
    for t in new_tracks:
        by_id[t["id"]] = t

    merged = list(by_id.values())
    merged.sort(key=lambda x: (x.get("date", ""), x.get("title", "")), reverse=True)
    return merged


def write_catalog(bucket: str, key: str, tracks: list[dict]) -> None:
    upload_json_bytes(_dumps(tracks), bucket=bucket, key=key)
