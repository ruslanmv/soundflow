from __future__ import annotations

import argparse
import os
from datetime import datetime, timezone
from typing import Optional

from common.catalog_write import read_catalog, write_catalog, upsert_tracks, get_catalog_paths


def utc_today() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d")


def find_for_date(tracks: list[dict], date: str) -> list[dict]:
    return [t for t in tracks if t.get("date") == date]


def newest_track(tracks: list[dict], category: Optional[str] = None) -> Optional[dict]:
    candidates = tracks
    if category:
        candidates = [t for t in tracks if t.get("category") == category]
    if not candidates:
        return None
    candidates = sorted(candidates, key=lambda x: (x.get("date", ""), x.get("id", "")), reverse=True)
    return candidates[0]


def duplicate_as_today(src: dict, date: str) -> dict:
    # Keep objectKey same, just create new metadata record for today
    new_id = f"premium-{date}-buffer-{src['id']}"
    title = src.get("title", "AI Premium Daily")
    if "Daily" not in title:
        title = f"{title} (Daily)"

    out = dict(src)
    out["id"] = new_id
    out["date"] = date
    out["title"] = title
    out["bufferFallback"] = True
    out["sourceId"] = src.get("id")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", default=None, help="YYYY-MM-DD (default: today UTC)")
    ap.add_argument("--category", default=None, help="Optional: buffer only for this category")
    args = ap.parse_args()

    date = args.date or utc_today()
    bucket = os.environ["R2_BUCKET"]
    paths = get_catalog_paths()

    premium = read_catalog(bucket=bucket, key=paths.premium_key)
    if not premium:
        raise RuntimeError("Premium catalog is empty; cannot buffer-publish.")

    existing_today = find_for_date(premium, date)
    if args.category:
        existing_today = [t for t in existing_today if t.get("category") == args.category]

    if existing_today:
        print(f"✅ Premium already has tracks for {date} (count={len(existing_today)})")
        return

    src = newest_track(premium, category=args.category)
    if not src:
        raise RuntimeError(f"No premium tracks found to buffer from (category={args.category}).")

    buffered = duplicate_as_today(src, date)
    merged = upsert_tracks(premium, [buffered])
    write_catalog(bucket=bucket, key=paths.premium_key, tracks=merged)

    print("✅ Buffer published premium track for today:")
    print("   date:", date)
    print("   newId:", buffered["id"])
    print("   objectKey:", buffered["objectKey"])
    print("   sourceId:", buffered.get("sourceId"))


if __name__ == "__main__":
    main()
