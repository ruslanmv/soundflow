.DEFAULT_GOAL := help

PROJECT_NAME := soundflow-monorepo

.PHONY: help install install-frontend install-backend install-generator lock-backend lock-generator serve serve-frontend serve-backend test-generator clean gpu-premium gpu-test gpu-colab gpu-kaggle

help: ## Show available commands
	@echo "$(PROJECT_NAME)"
	@echo ""
	@echo "Usage:"
	@echo "  make install            Install frontend + backend + generator dependencies"
	@echo "  make lock-backend       Regenerate uv.lock (backend)"
	@echo "  make lock-generator     Regenerate uv.lock (generator)"
	@echo "  make serve              Print how to run services"
	@echo "  make serve-frontend     Run Next.js dev server"
	@echo "  make serve-backend      Run FastAPI dev server"
	@echo "  make test-generator     Run the generator smoke test"
	@echo "  make clean              Remove build artifacts"

install: install-frontend install-backend install-generator ## Install all dependencies

install-frontend: ## Install frontend dependencies
	@cd frontend && ( [ -f package-lock.json ] && npm ci || npm install )

lock-backend: ## Regenerate backend uv.lock using Python 3.11
	@cd backend && \
	UV_LINK_MODE=copy uv python install 3.11 && \
	UV_LINK_MODE=copy uv python pin 3.11 && \
	rm -f uv.lock && \
	UV_LINK_MODE=copy uv lock

install-backend: ## Install backend dependencies using uv + Python 3.11
	@cd backend && \
	UV_LINK_MODE=copy uv python install 3.11 && \
	UV_LINK_MODE=copy uv python pin 3.11 && \
	( UV_LINK_MODE=copy uv sync || (echo "uv.lock invalid -> regenerating..." && rm -f uv.lock && UV_LINK_MODE=copy uv lock && UV_LINK_MODE=copy uv sync) )

lock-generator: ## Regenerate generator uv.lock using Python 3.11
	@cd generator && \
	UV_LINK_MODE=copy uv python install 3.11 && \
	UV_LINK_MODE=copy uv python pin 3.11 && \
	rm -f uv.lock && \
	UV_LINK_MODE=copy uv lock

install-generator: ## Install generator dependencies using uv + Python 3.11
	@cd generator && \
	UV_LINK_MODE=copy uv python install 3.11 && \
	UV_LINK_MODE=copy uv python pin 3.11 && \
	( UV_LINK_MODE=copy uv sync || (echo "uv.lock invalid -> regenerating..." && rm -f uv.lock && UV_LINK_MODE=copy uv lock && UV_LINK_MODE=copy uv sync) )

serve: ## Run both services (prints commands)
	@echo "Run these in separate terminals:"
	@echo "  make serve-backend"
	@echo "  make serve-frontend"

serve-frontend: ## Run Next.js dev server
	@cd frontend && npm run dev

serve-backend: ## Run FastAPI dev server (Python 3.11 via uv)
	@cd backend && uv run uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

test-generator: ## Run generator smoke test
	@cd generator && uv run python tests/test_generator.py

clean: ## Remove build artifacts
	@rm -rf frontend/.next frontend/out backend/.venv generator/.venv

# ============================================================================
# GPU/PREMIUM TARGETS
# ============================================================================

gpu-premium: ## Generate premium tracks (requires GPU)
	@echo "🎵 Generating Premium Tracks with MusicGen Stereo Large..."
	@echo "⚠️  Requires: CUDA GPU with 16GB+ VRAM"
	@cd generator && uv run python -m premium.musicgen_daily --date $(shell date +%Y-%m-%d) --device cuda

gpu-test: ## Test GPU availability and model loading
	@echo "🔍 Testing GPU Setup..."
	@cd generator && uv run python -c "import torch; print(f'CUDA Available: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}'); from audiocraft.models import MusicGen; print('✅ Audiocraft installed')"

gpu-colab: ## Setup instructions for Google Colab
	@echo "📋 Google Colab Setup Instructions:"
	@echo ""
	@echo "1. Clone repository:"
	@echo "   !git clone https://github.com/YOUR_USERNAME/soundflow.git"
	@echo "   %cd soundflow/generator"
	@echo ""
	@echo "2. Install system dependencies:"
	@echo "   !apt-get -qq update && apt-get -qq install -y ffmpeg"
	@echo ""
	@echo "3. Install Python dependencies:"
	@echo "   !pip install -q -r requirements-gpu.txt"
	@echo ""
	@echo "4. Generate tracks:"
	@echo "   !python -m premium.musicgen_daily --date 2025-01-15 --device cuda"
	@echo ""
	@echo "5. Download results:"
	@echo "   from google.colab import files"
	@echo "   files.download('.soundflow_out/premium/premium-2025-01-15-deep_focus_gamma.mp3')"

gpu-kaggle: ## Setup instructions for Kaggle
	@echo "📋 Kaggle Setup Instructions:"
	@echo ""
	@echo "1. Create new notebook with GPU accelerator (P100 or T4)"
	@echo ""
	@echo "2. Add dataset: Upload soundflow generator folder"
	@echo ""
	@echo "3. Install dependencies:"
	@echo "   !pip install -q -r /kaggle/input/soundflow/requirements-gpu.txt"
	@echo ""
	@echo "4. Generate tracks:"
	@echo "   !python /kaggle/input/soundflow/premium/musicgen_daily.py --date 2025-01-15"
	@echo ""
	@echo "5. Output available in: /kaggle/working/.soundflow_out/premium/"