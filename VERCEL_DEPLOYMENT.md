# Vercel Deployment Configuration

## Setup Instructions

This project is structured as a monorepo with the Next.js application located in the `frontend/` directory.

### Required Vercel Configuration

To deploy this project successfully on Vercel, you **must** configure the following setting in your Vercel project dashboard:

1. Go to your project settings on Vercel
2. Navigate to **General** → **Build & Development Settings**
3. Set **Root Directory** to: `frontend`
4. Save the changes

### Why This is Needed

Vercel needs to know that the Next.js application lives in the `frontend/` subdirectory rather than the repository root. Setting the Root Directory tells Vercel to:
- Look for `package.json` in `frontend/`
- Run `npm install` in `frontend/`
- Run `npm run build` in `frontend/`
- Detect the Next.js framework correctly

### Project Structure

```
soundflow/
├── frontend/          ← Next.js application (Set as Root Directory in Vercel)
│   ├── app/
│   ├── components/
│   ├── lib/
│   ├── package.json   ← Contains Next.js dependency
│   └── ...
├── backend/           ← Python FastAPI backend
├── generator/         ← Audio generation scripts
└── package.json       ← Monorepo root
```

### Alternative: Vercel CLI Deployment

If you prefer to deploy via CLI with the correct settings:

```bash
cd frontend
vercel --prod
```

This automatically uses the `frontend/` directory as the build root.
