import sys
import os
import json
import time
import boto3

# Ensure we can import from app
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from app.core.config import settings

def get_s3_client():
    return boto3.client(
        "s3",
        endpoint_url=settings.r2_endpoint,
        aws_access_key_id=settings.r2_access_key_id,
        aws_secret_access_key=settings.r2_secret_access_key,
        region_name=settings.r2_region,
    )

def put_json(key: str, data: list):
    s3 = get_s3_client()
    clean_key = key.strip().lstrip("/").strip("'").strip('"')
    print(f"⬆️ Uploading to: {clean_key}...")
    
    s3.put_object(
        Bucket=settings.r2_bucket,
        Key=clean_key,
        Body=json.dumps(data, indent=2),
        ContentType="application/json"
    )
    print("✅ Success!")

def main():
    print(f"🌱 Seeding R2 Bucket: {settings.r2_bucket}")
    
    # Use today's date so the "Daily" endpoints work immediately
    today = time.strftime("%Y-%m-%d")

    # ---------------------------------------------------------
    # AUDIO LINKS (Royalty Free / Creative Commons)
    # ---------------------------------------------------------
    # The smooth beat you provided
    TRACK_LOFI = "https://cdn.pixabay.com/download/audio/2022/05/27/audio_1808fbf07a.mp3"
    
    # Nature sounds (Rain)
    TRACK_RAIN = "https://cdn.pixabay.com/download/audio/2022/07/04/audio_3d1627f12e.mp3"
    
    # Ambient Forest
    TRACK_FOREST = "https://cdn.pixabay.com/download/audio/2022/02/12/audio_96489886b4.mp3"

    # ---------------------------------------------------------
    # 1. PREMIUM CATALOG DATA
    # ---------------------------------------------------------
    premium_data = [
        {
            "id": f"premium-{today}-deep_work",
            "title": "AI Deep Work Daily",
            "tier": "premium",
            "date": today,
            "category": "Deep Work / Focus",
            "durationSec": 900,
            "goalTags": ["deep_work", "flow", "high_performance"],
            "natureTags": ["rain"],
            "energyMin": 45,
            "energyMax": 95,
            "ambienceMin": 0,
            "ambienceMax": 60,
            # In a real app, 'objectKey' points to a PRIVATE file in R2. 
            # For this seed, we point to a dummy path, but the 'previewUrl' works.
            "objectKey": f"audio/premium/{today}/deep_work.mp3", 
            "previewUrl": TRACK_LOFI 
        },
        {
            "id": f"premium-{today}-study",
            "title": "AI Study Daily",
            "tier": "premium",
            "date": today,
            "category": "Study / Reading",
            "durationSec": 900,
            "goalTags": ["study", "reading", "calm_focus"],
            "natureTags": ["forest"],
            "energyMin": 10,
            "energyMax": 75,
            "ambienceMin": 0,
            "ambienceMax": 75,
            "objectKey": f"audio/premium/{today}/study.mp3",
            "previewUrl": TRACK_FOREST
        }
    ]

    # ---------------------------------------------------------
    # 2. FREE CATALOG DATA
    # ---------------------------------------------------------
    free_data = [
        {
            "id": "free-001",
            "title": "Morning Flow (Free)",
            "tier": "free",
            "date": "2024-01-01",
            "category": "General",
            "durationSec": 300,
            "goalTags": ["morning", "wakeup"],
            "natureTags": ["birds"],
            "energyMin": 50,
            "energyMax": 80,
            "ambienceMin": 0,
            "ambienceMax": 50,
            "url": TRACK_LOFI
        },
        {
            "id": "free-002",
            "title": "Rainy Focus (Free)",
            "tier": "free",
            "date": "2024-01-01",
            "category": "Focus",
            "durationSec": 600,
            "goalTags": ["focus", "work"],
            "natureTags": ["rain"],
            "energyMin": 30,
            "energyMax": 60,
            "ambienceMin": 0,
            "ambienceMax": 40,
            "url": TRACK_RAIN
        },
         {
            "id": "free-003",
            "title": "Forest Ambience (Free)",
            "tier": "free",
            "date": "2024-01-01",
            "category": "Relax",
            "durationSec": 450,
            "goalTags": ["relax", "sleep"],
            "natureTags": ["forest"],
            "energyMin": 10,
            "energyMax": 40,
            "ambienceMin": 50,
            "ambienceMax": 90,
            "url": TRACK_FOREST
        }
    ]

    # ---------------------------------------------------------
    # 3. UPLOAD
    # ---------------------------------------------------------
    # Upload Premium
    put_json(settings.catalog_premium_key, premium_data)
    
    # Upload Free
    put_json(settings.catalog_free_key, free_data)

    print("\n🎉 Seeding complete! Try the /tracks/free or /tracks/premium/daily endpoints now.")

if __name__ == "__main__":
    main()