# Run Premium Generation on Google Colab (Free)

## Best use
- Seeding premium catalog
- R&D on prompts/settings
- Backup generation

## Steps
1. Runtime → Change runtime type → GPU
2. Install deps:
\`\`\`bash
!apt-get update && apt-get install -y ffmpeg
!pip install -U uv
\`\`\`

3. Upload the generator/ folder or clone repo.

4. Set env vars (in Colab):
\`\`\`python
import os
os.environ["R2_ENDPOINT"] = "..."
os.environ["R2_BUCKET"] = "..."
os.environ["R2_ACCESS_KEY_ID"] = "..."
os.environ["R2_SECRET_ACCESS_KEY"] = "..."
os.environ["R2_REGION"] = "auto"
\`\`\`

5. Run:
\`\`\`bash
!cd generator && uv sync
!cd generator && uv run python -m premium.musicgen_daily --date 2025-12-17 --upload
\`\`\`

## Recommendation
Use Colab to generate a 7–14 day buffer in advance.
