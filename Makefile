.DEFAULT_GOAL := help

PROJECT_NAME := soundflow-monorepo

.PHONY: help install install-frontend install-backend lock-backend serve serve-frontend serve-backend clean

help: ## Show available commands
	@echo "$(PROJECT_NAME)"
	@echo ""
	@echo "Usage:"
	@echo "  make install         Install frontend + backend dependencies"
	@echo "  make lock-backend    Regenerate uv.lock (backend)"
	@echo "  make serve           Print how to run both services"
	@echo "  make serve-frontend  Run Next.js dev server"
	@echo "  make serve-backend   Run FastAPI dev server"
	@echo "  make clean           Remove build artifacts"

install: install-frontend install-backend ## Install all dependencies

install-frontend: ## Install frontend dependencies (prefer reproducible install)
	@cd frontend && ( [ -f package-lock.json ] && npm ci || npm install )

lock-backend: ## Regenerate backend uv.lock using Python 3.11
	@cd backend && \
	UV_LINK_MODE=copy uv python install 3.11 && \
	UV_LINK_MODE=copy uv python pin 3.11 && \
	rm -f uv.lock && \
	UV_LINK_MODE=copy uv lock

install-backend: ## Install backend dependencies using uv + Python 3.11 (auto-heals broken uv.lock)
	@cd backend && \
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

clean: ## Remove build artifacts (keeps uv.lock for reproducible installs)
	@rm -rf frontend/.next frontend/out backend/.venv
