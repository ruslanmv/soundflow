# SoundFlow Generator

Daily generation pipeline for:
- Free catalog: loop/stem remix (CPU)
- Optional: MIDI variations (CPU)
- Premium catalog: MusicGen/StableAudio (GPU)

## Install
Use uv (recommended) or pip:

\`\`\`bash
cd generator
uv sync
\`\`\`

## Environment Variables (R2/S3)
Required:
- R2_ENDPOINT
- R2_BUCKET
- R2_ACCESS_KEY_ID
- R2_SECRET_ACCESS_KEY
- R2_REGION (default: auto)

Catalog paths:
- CATALOG_FREE_KEY (default: catalog/free-index.json)
- CATALOG_PREMIUM_KEY (default: catalog/premium-index.json)

Public base URL for free tracks (if you expose via CDN):
- FREE_PUBLIC_BASE_URL (example: https://cdn.yoursite.com/audio/free)

## Run Free Daily (CPU)
\`\`\`bash
uv run python -m free.remix_daily --date 2025-12-17
\`\`\`

## Run Premium Daily (GPU)
\`\`\`bash
uv run python -m premium.musicgen_daily --date 2025-12-17 --model facebook/musicgen-small
\`\`\`

## Notes
- Premium outputs are uploaded as private objects (objectKey only in catalog).
- Free outputs can be public or private; catalog uses a url field for free.
- Keep a buffer: generate 7–14 days ahead for premium to avoid downtime.
