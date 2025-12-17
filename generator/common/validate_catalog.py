from __future__ import annotations

import argparse
import os
import re
from datetime import datetime
from typing import Any

import orjson

from common.catalog_write import get_catalog_paths
from common.r2_upload import r2_client


DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")


class ValidationError(RuntimeError):
    pass


def _loads(b: bytes) -> Any:
    return orjson.loads(b)


def _read_local(path: str) -> list[dict]:
    data = _loads(open(path, "rb").read())
    if not isinstance(data, list):
        raise ValidationError("Catalog must be a JSON array.")
    return data


def _read_r2(bucket: str, key: str) -> list[dict]:
    s3 = r2_client()
    obj = s3.get_object(Bucket=bucket, Key=key)
    data = _loads(obj["Body"].read())
    if not isinstance(data, list):
        raise ValidationError("Catalog must be a JSON array.")
    return data


def _require_fields(t: dict, fields: list[str]) -> None:
    for f in fields:
        if f not in t:
            raise ValidationError(f"Missing required field '{f}' in track: {t.get('id')}")


def _validate_date(s: str) -> None:
    if not DATE_RE.match(s):
        raise ValidationError(f"Invalid date format '{s}', expected YYYY-MM-DD")
    # also validate real date
    datetime.strptime(s, "%Y-%m-%d")


def validate_track(t: dict, tier: str) -> None:
    if not isinstance(t, dict):
        raise ValidationError("Each track must be an object.")

    base_fields = [
        "id",
        "title",
        "tier",
        "date",
        "category",
        "durationSec",
        "goalTags",
        "natureTags",
        "energyMin",
        "energyMax",
        "ambienceMin",
        "ambienceMax",
    ]
    _require_fields(t, base_fields)

    if t["tier"] != tier:
        raise ValidationError(f"Track tier mismatch: expected '{tier}', got '{t['tier']}' for {t['id']}")

    if not isinstance(t["id"], str) or len(t["id"]) < 5:
        raise ValidationError(f"Invalid id: {t.get('id')}")

    if not isinstance(t["title"], str) or not t["title"].strip():
        raise ValidationError(f"Invalid title: {t.get('id')}")

    _validate_date(t["date"])

    if not isinstance(t["durationSec"], int) or t["durationSec"] <= 0:
        raise ValidationError(f"Invalid durationSec for {t['id']}")

    if not isinstance(t["goalTags"], list) or not all(isinstance(x, str) for x in t["goalTags"]):
        raise ValidationError(f"Invalid goalTags for {t['id']}")

    if not isinstance(t["natureTags"], list) or not all(isinstance(x, str) for x in t["natureTags"]):
        raise ValidationError(f"Invalid natureTags for {t['id']}")

    for k in ["energyMin", "energyMax", "ambienceMin", "ambienceMax"]:
        if not isinstance(t[k], int) or not (0 <= t[k] <= 100):
            raise ValidationError(f"Invalid {k} for {t['id']}")

    if t["energyMin"] > t["energyMax"]:
        raise ValidationError(f"energyMin > energyMax for {t['id']}")
    if t["ambienceMin"] > t["ambienceMax"]:
        raise ValidationError(f"ambienceMin > ambienceMax for {t['id']}")

    # tier-specific
    if tier == "free":
        _require_fields(t, ["url"])
        if not isinstance(t["url"], str) or not t["url"].strip():
            raise ValidationError(f"Free track must have non-empty url: {t['id']}")
    else:
        _require_fields(t, ["objectKey"])
        if not isinstance(t["objectKey"], str) or not t["objectKey"].startswith("audio/"):
            raise ValidationError(f"Premium track must have objectKey starting with 'audio/': {t['id']}")


def validate_catalog(tracks: list[dict], tier: str) -> None:
    if not tracks:
        raise ValidationError("Catalog is empty (this is allowed only if you intend it).")

    ids = set()
    for t in tracks:
        validate_track(t, tier=tier)
        if t["id"] in ids:
            raise ValidationError(f"Duplicate track id found: {t['id']}")
        ids.add(t["id"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tier", choices=["free", "premium"], required=True)
    ap.add_argument("--source", choices=["local", "r2"], default="local")
    ap.add_argument("--path", help="Local path to catalog JSON (when --source local)")
    args = ap.parse_args()

    paths = get_catalog_paths()

    if args.source == "local":
        path = args.path or ("catalog/free-index.sample.json" if args.tier == "free" else "catalog/premium-index.sample.json")
        tracks = _read_local(path)
    else:
        bucket = os.environ["R2_BUCKET"]
        key = paths.free_key if args.tier == "free" else paths.premium_key
        tracks = _read_r2(bucket=bucket, key=key)

    validate_catalog(tracks, tier=args.tier)
    print(f"✅ Catalog OK: tier={args.tier} count={len(tracks)} source={args.source}")


if __name__ == "__main__":
    main()
