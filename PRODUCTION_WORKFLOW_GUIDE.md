# 🎵 SoundFlow Production Workflow Guide

**Complete workflow for FREE + PREMIUM music generation with R2 catalog indexing**

This document describes the production-ready workflow that generates music daily for all site categories, maintains R2 catalogs, and serves content via the backend API.

---

## 📊 System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    SOUNDFLOW PRODUCTION SYSTEM                  │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  FREE TIER                          PREMIUM TIER                │
│  ├─ Daily Generation (GitHub)       ├─ On-Demand (Manual/GPU)  │
│  ├─ 5 Tracks/Day (All Categories)   ├─ MusicGen Stereo Large   │
│  ├─ Procedural Synthesis             ├─ Advanced DSP Processing│
│  └─ 192kbps MP3                      └─ 320kbps MP3            │
│                                                                 │
│  STORAGE (Cloudflare R2)                                       │
│  ├─ audio/free/YYYY-MM-DD/*.mp3                                │
│  ├─ audio/premium/YYYY-MM-DD/*.mp3                             │
│  ├─ catalog/free.json                                          │
│  ├─ catalog/premium.json                                       │
│  ├─ catalog/all.json                   ← GENERAL INDEX         │
│  └─ catalog/by-category/*.json         ← PER-CATEGORY INDEXES  │
│                                                                 │
│  BACKEND (FastAPI)                                             │
│  ├─ GET /catalog/all                  │  All tracks            │
│  ├─ GET /catalog/by-category/{slug}   │  Category-specific     │
│  ├─ GET /catalog/free                 │  Free tracks           │
│  ├─ GET /catalog/premium              │  Premium tracks        │
│  └─ GET /tracks/premium/{id}/signed   │  Signed URLs          │
│                                                                 │
│  FRONTEND (Next.js)                                            │
│  ├─ Browse by Category (Deep Work, Study, etc.)               │
│  ├─ Filter by Tier (Free/Premium)                             │
│  └─ Play with inline player                                   │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## 🎯 Site Categories → Musical Genres Mapping

The system generates **ONE track per site category** daily:

| Site Category | Musical Genres (Rotating) | Use Case |
|---------------|---------------------------|----------|
| **Deep Work** | deep, house, techno, synth | Coding, focused tasks |
| **Study** | lofi, chillout, lounge, jazz | Reading, learning |
| **Relax** | ambient, classic, lounge | Meditation, unwinding |
| **Nature** | ambient (+ heavy nature sounds) | Natural environments |
| **Flow State** | trance, edm, dance, bass | High energy, workouts |

### Genre Rotation

Genres rotate by day of week for variety:
- **Monday**: Deep Work=deep, Study=lofi, Flow=trance, etc.
- **Tuesday**: Deep Work=house, Study=chillout, Flow=edm, etc.

See `generator/prompts/free_daily_plan.yaml` for complete configuration.

---

## ⚙️ Daily FREE Generation Workflow

**Trigger**: Daily at 02:10 UTC (GitHub Actions: `free-daily.yml`)

### Steps

1. **Generate Tracks**
   ```bash
   python -m free.remix_daily \
     --date 2025-01-15 \
     --use-daily-plan \
     --upload
   ```

   Output:
   - 5 MP3 files (one per site category)
   - Uploaded to `audio/free/2025-01-15/*.mp3`
   - Free catalog updated: `catalog/free.json`

2. **Build General Catalog**
   ```bash
   python -m common.build_general_catalog \
     --bucket $R2_BUCKET \
     --upload
   ```

   Output:
   - `catalog/all.json` (merged free + premium)
   - `catalog/by-category/deep-work.json`
   - `catalog/by-category/study.json`
   - `catalog/by-category/relax.json`
   - `catalog/by-category/nature.json`
   - `catalog/by-category/flow-state.json`

3. **Validate**
   ```bash
   python -m common.validate_catalog --tier free --source r2
   ```

### Configuration File

**`generator/prompts/free_daily_plan.yaml`**

```yaml
tracks:
  - id: deep_work_daily
    siteCategory: "Deep Work"
    genre: "deep"
    bpm: 124
    key: "Dm"
    layers: {drums: 70, bass: 80, music: 60, pad: 50}
    goalTags: ["deep_work", "coding", "focus"]
    # ... more config
```

### Manual Run

```bash
cd generator

# Install dependencies
uv sync

# Generate for today
TODAY=$(date +"%Y-%m-%d")
uv run python -m free.remix_daily \
  --date "$TODAY" \
  --use-daily-plan \
  --upload

# Rebuild catalogs
uv run python -m common.build_general_catalog \
  --bucket $R2_BUCKET \
  --upload
```

---

## 💎 On-Demand PREMIUM Generation Workflow

**Trigger**: Manual (GitHub Actions: `premium-on-demand.yml`)

### Workflow Dispatch Inputs

- **category**: Deep Work | Study | Relax | Nature | Flow State
- **track_id**: Custom identifier (e.g., `custom_focus_1`)
- **duration**: Track length in seconds (default: 180)
- **model**: musicgen-stereo-large | musicgen-stereo-medium
- **gpu_provider**: manual | modal | replicate | runpod

### GPU Providers

#### Option 1: Manual (Google Colab)

1. Open `generator/notebooks/SoundFlow_Premium_Colab.ipynb`
2. Enable GPU (Runtime > Change runtime type > GPU)
3. Run all cells
4. Download generated track
5. Upload to R2:
   ```bash
   aws s3 cp track.mp3 s3://$R2_BUCKET/audio/premium/2025-01-15/track_id.mp3
   ```
6. Rebuild catalogs:
   ```bash
   python -m common.build_general_catalog --bucket $R2_BUCKET --upload
   ```

#### Option 2: Modal (Future)

Serverless GPU execution via Modal.com. Integration not yet implemented.

#### Option 3: Replicate (Future)

API-based generation via Replicate.com. Integration not yet implemented.

#### Option 4: RunPod (Future)

Serverless GPU via RunPod.io. Integration not yet implemented.

### Manual Local Generation (if you have GPU)

```bash
cd generator

# Generate with premium generator
python -m premium.musicgen_daily \
  --date 2025-01-15 \
  --model facebook/musicgen-stereo-large \
  --device cuda \
  --upload

# Rebuild catalogs
python -m common.build_general_catalog \
  --bucket $R2_BUCKET \
  --upload
```

---

## 📦 R2 Catalog Structure

### Catalog Files

```
s3://soundflow-bucket/
├── audio/
│   ├── free/
│   │   └── 2025-01-15/
│   │       ├── free-2025-01-15-deep_work_daily.mp3
│   │       ├── free-2025-01-15-study_daily.mp3
│   │       ├── free-2025-01-15-relax_daily.mp3
│   │       ├── free-2025-01-15-nature_daily.mp3
│   │       └── free-2025-01-15-flow_state_daily.mp3
│   └── premium/
│       └── 2025-01-15/
│           └── premium-2025-01-15-*.mp3
│
└── catalog/
    ├── free.json               # All free tracks
    ├── premium.json            # All premium tracks
    ├── all.json                # GENERAL INDEX (free + premium)
    └── by-category/
        ├── deep-work.json      # Deep Work tracks only
        ├── study.json          # Study tracks only
        ├── relax.json          # Relax tracks only
        ├── nature.json         # Nature tracks only
        └── flow-state.json     # Flow State tracks only
```

### Catalog Entry Schema

```json
{
  "id": "free-2025-01-15-deep_work_daily",
  "title": "Deep Work Daily - 2025-01-15",
  "tier": "free",
  "date": "2025-01-15",
  "category": "Deep Work",
  "genre": "deep",
  "bpm": 124,
  "key": "Dm",
  "durationSec": 180,
  "goalTags": ["deep_work", "coding", "focus"],
  "natureTags": ["minimal"],
  "energyMin": 50,
  "energyMax": 75,
  "ambienceMin": 5,
  "ambienceMax": 30,
  "objectKey": "audio/free/2025-01-15/free-2025-01-15-deep_work_daily.mp3",
  "url": "https://r2.soundflow.app/audio/free/2025-01-15/free-2025-01-15-deep_work_daily.mp3"
}
```

---

## 🌐 Backend API Endpoints

### Catalog Browse Endpoints

```http
GET /catalog/all
Returns: General catalog (all tracks, free + premium)
```

```http
GET /catalog/by-category/{category}
Parameters: category = deep-work | study | relax | nature | flow-state
Returns: Tracks for specific category
```

```http
GET /catalog/free
Returns: All free tracks (public URLs)
```

```http
GET /catalog/premium
Returns: All premium tracks (metadata only, no URLs)
```

### Playback Endpoints

```http
GET /tracks/premium/{track_id}/signed
Headers: Authorization: Bearer {premium_token}
Returns: Signed URL for premium track (expires in 1 hour)
```

### Example: Frontend Integration

```typescript
// Fetch all tracks
const response = await fetch('https://api.soundflow.app/catalog/all');
const allTracks = await response.json();

// Fetch Deep Work category
const response = await fetch('https://api.soundflow.app/catalog/by-category/deep-work');
const deepWorkTracks = await response.json();

// Play premium track (requires auth)
const response = await fetch(
  `https://api.soundflow.app/tracks/premium/${trackId}/signed`,
  { headers: { 'Authorization': `Bearer ${token}` } }
);
const { signedUrl } = await response.json();
audioElement.src = signedUrl;
```

---

## 🎨 Frontend Integration

### Playlist Page (Recommended)

Create `frontend/app/playlists/page.tsx`:

```tsx
'use client';

import { useState, useEffect } from 'react';

export default function PlaylistsPage() {
  const [category, setCategory] = useState('all');
  const [tracks, setTracks] = useState([]);

  useEffect(() => {
    const url = category === 'all'
      ? '/api/catalog/all'
      : `/api/catalog/by-category/${category}`;

    fetch(url)
      .then(res => res.json())
      .then(data => setTracks(data));
  }, [category]);

  return (
    <div>
      <h1>Music Library</h1>

      <select value={category} onChange={(e) => setCategory(e.target.value)}>
        <option value="all">All Categories</option>
        <option value="deep-work">Deep Work</option>
        <option value="study">Study</option>
        <option value="relax">Relax</option>
        <option value="nature">Nature</option>
        <option value="flow-state">Flow State</option>
      </select>

      <div>
        {tracks.map(track => (
          <TrackCard key={track.id} track={track} />
        ))}
      </div>
    </div>
  );
}
```

---

## 🚀 Deployment Checklist

### 1. Environment Variables

Set in GitHub Secrets:

```
R2_ENDPOINT
R2_BUCKET
R2_ACCESS_KEY_ID
R2_SECRET_ACCESS_KEY
R2_REGION
CATALOG_FREE_KEY
FREE_PUBLIC_BASE_URL
```

### 2. Initial Setup

```bash
# Generate first batch of free tracks
python -m free.remix_daily --date 2025-01-15 --use-daily-plan --upload

# Build initial catalogs
python -m common.build_general_catalog --bucket $R2_BUCKET --upload
```

### 3. Enable GitHub Workflows

- ✅ `free-daily.yml` - Generates daily free tracks
- ✅ `premium-on-demand.yml` - Manual premium generation

### 4. Backend Deployment

```bash
cd backend
uv sync
uv run uvicorn app.main:app --host 0.0.0.0 --port 8000
```

### 5. Frontend Deployment

```bash
cd frontend
npm install
npm run build
npm start
```

---

## 📊 Monitoring & Validation

### Check Workflow Runs

```bash
# View GitHub Actions
https://github.com/YOUR_USERNAME/soundflow/actions
```

### Validate Catalogs

```bash
cd generator

# Validate free catalog
uv run python -m common.validate_catalog --tier free --source r2

# Validate premium catalog
uv run python -m common.validate_catalog --tier premium --source r2
```

### Check R2 Contents

```bash
# List catalogs
aws s3 ls s3://$R2_BUCKET/catalog/

# Check catalog content
aws s3 cp s3://$R2_BUCKET/catalog/all.json - | jq .
```

---

## 🔧 Troubleshooting

### Daily workflow not running

- Check GitHub Actions is enabled
- Verify cron schedule (02:10 UTC)
- Check R2 credentials in secrets

### Catalog not updating

- Check workflow logs
- Verify R2 bucket permissions
- Run manual catalog build:
  ```bash
  python -m common.build_general_catalog --bucket $R2_BUCKET --upload
  ```

### Frontend not showing tracks

- Check backend API is running
- Verify CORS settings
- Check browser console for errors
- Test API endpoint directly:
  ```bash
  curl https://api.soundflow.app/catalog/all
  ```

---

## 📚 Additional Resources

- **Free Daily Plan**: `generator/prompts/free_daily_plan.yaml`
- **Premium Prompts**: `generator/prompts/premium_prompts.yaml`
- **Catalog Builder**: `generator/common/build_general_catalog.py`
- **Free Generator**: `generator/free/remix_daily.py`
- **Premium Generator**: `generator/premium/musicgen_daily.py`
- **Backend API**: `backend/app/main.py`
- **Catalog Service**: `backend/app/services/catalog.py`

---

## ✅ Summary

This production workflow provides:

✅ **Daily FREE generation** - 5 tracks covering all site categories
✅ **On-demand PREMIUM** - GPU-based high-quality generation
✅ **R2 catalog system** - General index + per-category indexes
✅ **Backend API** - RESTful endpoints for catalog browsing
✅ **Frontend ready** - Easy integration for playlist/browse pages

**Result**: A complete, production-ready AI music service with automated daily updates and flexible premium generation!
