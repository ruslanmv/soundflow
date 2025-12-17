# SoundFlow Backend (FastAPI + uv)

This backend provides:
- Free catalog + deterministic track routing
- Premium catalog + signed URLs (private objects)
- Admin endpoints to publish/update catalogs in R2/S3

## Requirements
- Python 3.10+
- uv installed: https://docs.astral.sh/uv/

## Setup (local)
\`\`\`bash
cd backend
uv sync
uv run uvicorn app.main:app --reload --port 8000
\`\`\`

## Env vars
Copy to .env (or set in your hosting provider):

### R2/S3:
- R2_ENDPOINT (e.g. https://<accountid>.r2.cloudflarestorage.com)
- R2_BUCKET
- R2_ACCESS_KEY_ID
- R2_SECRET_ACCESS_KEY
- R2_REGION (default: auto)

### Catalog keys in bucket:
- CATALOG_FREE_KEY (default: catalog/free-index.json)
- CATALOG_PREMIUM_KEY (default: catalog/premium-index.json)

### Auth:
- ADMIN_API_KEY (required for publish endpoints)
- PREMIUM_API_KEY (optional simple premium gating)

### CORS
- CORS_ORIGINS (comma-separated; default "*")

## Run in production
Use a process manager (systemd) or container.
Example:
\`\`\`bash
uv run uvicorn app.main:app --host 0.0.0.0 --port 8000
\`\`\`
