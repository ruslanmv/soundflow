"""
General Catalog Builder for SoundFlow
======================================
Merges free + premium catalogs and creates derived indexes:
- catalog/all.json (general index)
- catalog/by-category/<slug>.json (per-category indexes)

Usage:
    python -m common.build_general_catalog --bucket my-bucket
    python -m common.build_general_catalog --bucket my-bucket --upload
"""

from __future__ import annotations

import argparse
import logging
from collections import defaultdict
from pathlib import Path
from typing import Any

import orjson

from common.catalog_write import get_catalog_paths, read_catalog, write_catalog

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("CatalogBuilder")

# Site categories (from frontend/lib/presets.ts)
SITE_CATEGORIES = [
    "Deep Work",
    "Study",
    "Relax",
    "Nature",
    "Flow State",
]


def slugify(text: str) -> str:
    """Convert category name to URL-safe slug."""
    return text.lower().replace(" ", "-")


def build_all_catalog(free_tracks: list[dict], premium_tracks: list[dict]) -> list[dict]:
    """
    Merge free and premium tracks into a single catalog.

    Tracks are sorted by:
    1. Tier (premium first)
    2. Date (newest first)
    3. ID
    """
    all_tracks = []

    # Add premium tracks with tier marker
    for track in premium_tracks:
        track["tier"] = "premium"
        all_tracks.append(track)

    # Add free tracks with tier marker
    for track in free_tracks:
        track["tier"] = "free"
        all_tracks.append(track)

    # Sort: premium first, then by date descending
    all_tracks.sort(
        key=lambda t: (
            0 if t.get("tier") == "premium" else 1,
            t.get("date", ""),
            t.get("id", ""),
        ),
        reverse=True,
    )

    logger.info(
        f"📦 Built general catalog: {len(all_tracks)} tracks "
        f"({len(premium_tracks)} premium, {len(free_tracks)} free)"
    )

    return all_tracks


def build_category_catalogs(all_tracks: list[dict]) -> dict[str, list[dict]]:
    """
    Group tracks by site category.

    Returns:
        Dict mapping category slug to track list
    """
    by_category = defaultdict(list)

    for track in all_tracks:
        category = track.get("category", "").strip()

        if not category:
            continue

        # Normalize category name
        # Handle both "Deep Work" and "deep_work" style
        category_normalized = category.replace("_", " ").title()

        if category_normalized in SITE_CATEGORIES:
            slug = slugify(category_normalized)
            by_category[slug].append(track)

    # Sort each category by date descending
    for slug, tracks in by_category.items():
        tracks.sort(key=lambda t: (t.get("date", ""), t.get("id", "")), reverse=True)

    logger.info(
        f"📂 Built category indexes: {len(by_category)} categories "
        f"({', '.join(f'{slug}={len(tracks)}' for slug, tracks in sorted(by_category.items()))})"
    )

    return dict(by_category)


def save_local_catalog(data: Any, path: Path) -> None:
    """Save catalog to local JSON file."""
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("wb") as f:
        f.write(orjson.dumps(data, option=orjson.OPT_INDENT_2))

    logger.info(f"💾 Saved local: {path} ({len(data)} items)")


def main():
    parser = argparse.ArgumentParser(description="Build general catalog and indexes")
    parser.add_argument("--bucket", required=True, help="R2 bucket name")
    parser.add_argument(
        "--upload",
        action="store_true",
        help="Upload catalogs to R2 (default: save locally only)",
    )
    parser.add_argument(
        "--local-dir",
        type=Path,
        default=Path(".soundflow_catalogs"),
        help="Local directory for catalog cache",
    )
    args = parser.parse_args()

    bucket = args.bucket
    upload = args.upload
    local_dir: Path = args.local_dir

    logger.info("=" * 70)
    logger.info("🏗️  Building General Catalog")
    logger.info("=" * 70)

    # Get catalog paths
    paths = get_catalog_paths()

    # -------------------------------------------------------------------------
    # 1. Read free and premium catalogs from R2
    # -------------------------------------------------------------------------

    logger.info(f"\n📥 Reading catalogs from R2 bucket: {bucket}")

    free_tracks = read_catalog(bucket, paths.free_key)
    premium_tracks = read_catalog(bucket, paths.premium_key)

    logger.info(f"  Free tracks: {len(free_tracks)}")
    logger.info(f"  Premium tracks: {len(premium_tracks)}")

    if not free_tracks and not premium_tracks:
        logger.warning("⚠️  No tracks found in either catalog!")
        return

    # -------------------------------------------------------------------------
    # 2. Build general catalog (all.json)
    # -------------------------------------------------------------------------

    logger.info(f"\n🔨 Building general catalog...")

    all_tracks = build_all_catalog(free_tracks, premium_tracks)

    # -------------------------------------------------------------------------
    # 3. Build per-category catalogs
    # -------------------------------------------------------------------------

    logger.info(f"\n🔨 Building category catalogs...")

    by_category = build_category_catalogs(all_tracks)

    # -------------------------------------------------------------------------
    # 4. Save locally (always)
    # -------------------------------------------------------------------------

    logger.info(f"\n💾 Saving catalogs locally to: {local_dir}")

    # Save general catalog
    save_local_catalog(all_tracks, local_dir / "all.json")

    # Save category catalogs
    category_dir = local_dir / "by-category"
    for slug, tracks in by_category.items():
        save_local_catalog(tracks, category_dir / f"{slug}.json")

    # -------------------------------------------------------------------------
    # 5. Upload to R2 (if --upload)
    # -------------------------------------------------------------------------

    if upload:
        logger.info(f"\n☁️  Uploading catalogs to R2...")

        # Upload general catalog
        write_catalog(bucket, "catalog/all.json", all_tracks)
        logger.info(f"  ✅ Uploaded: catalog/all.json")

        # Upload category catalogs
        for slug, tracks in by_category.items():
            category_key = f"catalog/by-category/{slug}.json"
            write_catalog(bucket, category_key, tracks)
            logger.info(f"  ✅ Uploaded: {category_key}")

    else:
        logger.info(
            f"\nℹ️  Dry run complete. Run with --upload to push to R2."
        )

    # -------------------------------------------------------------------------
    # 6. Summary
    # -------------------------------------------------------------------------

    logger.info("\n" + "=" * 70)
    logger.info("✅ Catalog build complete!")
    logger.info("=" * 70)
    logger.info(f"\n📊 Summary:")
    logger.info(f"  Total tracks: {len(all_tracks)}")
    logger.info(f"  Premium: {len(premium_tracks)}")
    logger.info(f"  Free: {len(free_tracks)}")
    logger.info(f"  Categories: {len(by_category)}")

    for slug, tracks in sorted(by_category.items()):
        premium_count = len([t for t in tracks if t.get("tier") == "premium"])
        free_count = len([t for t in tracks if t.get("tier") == "free"])
        logger.info(
            f"    {slug}: {len(tracks)} tracks ({premium_count} premium, {free_count} free)"
        )

    if upload:
        logger.info(f"\n📦 Catalogs available at:")
        logger.info(f"  s3://{bucket}/catalog/all.json")
        for slug in sorted(by_category.keys()):
            logger.info(f"  s3://{bucket}/catalog/by-category/{slug}.json")


if __name__ == "__main__":
    main()
