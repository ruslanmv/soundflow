# SoundFlow AI Monorepo

## Structure
- **/frontend**: Next.js App Router (Deploy this folder to Vercel)
- **/backend**: FastAPI (uv + pyproject.toml) — deploy later to Render/Railway/Cloud Run/VPS

## Quickstart

### Install everything
```bash
make install
```

### Run locally
```bash
# Terminal 1
make serve-backend

# Terminal 2
make serve-frontend
```

- Frontend: http://localhost:3000
- Backend:   http://localhost:8000

## Deploy

### Frontend (Vercel)
1. Push this repo to GitHub.
2. Import repo in Vercel.
3. **Root Directory**: set to `frontend`.
4. Deploy.

### Backend (later)
Deploy `backend/` then set this env var in Vercel:
- `PYTHON_API_URL=https://your-backend-host`

The Vercel route `/api/session` will proxy to the backend.
