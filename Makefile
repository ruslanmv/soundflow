.DEFAULT_GOAL := help

PROJECT_NAME := soundflow-monorepo

.PHONY: help install install-frontend install-backend serve serve-frontend serve-backend clean

help: ## Show available commands
	@echo "$(PROJECT_NAME)"
	@echo ""
	@echo "Usage:"
	@echo "  make install        Install frontend + backend dependencies"
	@echo "  make serve          Print how to run both services"
	@echo "  make serve-frontend Run Next.js dev server"
	@echo "  make serve-backend  Run FastAPI dev server"
	@echo "  make clean          Remove build artifacts"

install: install-frontend install-backend ## Install all dependencies

install-frontend: ## Install frontend dependencies
	cd frontend && npm install

install-backend: ## Install backend dependencies using uv
	cd backend && uv sync

serve: ## Run both services (prints commands)
	@echo "Run these in separate terminals:"
	@echo "  make serve-backend"
	@echo "  make serve-frontend"

serve-frontend: ## Run Next.js dev server
	cd frontend && npm run dev

serve-backend: ## Run FastAPI dev server
	cd backend && uv run uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

clean: ## Remove build artifacts
	rm -rf frontend/.next frontend/out backend/.venv
