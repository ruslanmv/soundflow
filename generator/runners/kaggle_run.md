# Run Premium Generation on Kaggle

## Why Kaggle
- Free GPU quota (weekly)
- Better than Colab for repeatable notebooks
- Still not guaranteed daily (use buffer strategy)

## Steps
1. Create a Kaggle Notebook (GPU enabled)
2. Add secrets (Environment Variables):
   - R2_ENDPOINT
   - R2_BUCKET
   - R2_ACCESS_KEY_ID
   - R2_SECRET_ACCESS_KEY
   - R2_REGION (auto)
   - CATALOG_PREMIUM_KEY (optional)

3. Install system deps:
\`\`\`bash
apt-get update && apt-get install -y ffmpeg
\`\`\`

4. Install python deps:
\`\`\`bash
pip install -U "uv>=0.4"
cd /kaggle/working
# copy generator/ into working directory or mount from dataset
uv sync
\`\`\`

5. Run:
\`\`\`bash
uv run python -m premium.musicgen_daily --date 2025-12-17 --upload
\`\`\`

## Buffer strategy
Generate 7–14 days ahead (run with different --date) so premium never breaks.
