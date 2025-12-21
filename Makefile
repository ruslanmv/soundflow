.DEFAULT_GOAL := help

PROJECT_NAME := soundflow-monorepo

.PHONY: help install install-frontend install-backend install-generator lock-backend lock-generator serve serve-frontend serve-backend test-generator clean

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