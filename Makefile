# Makefile pour Jarvis MCP
.PHONY: help dev build test clean docker-dev docker-build docker-clean

# Variables
COMPOSE_FILE = frontend/docker-compose.yml
SERVICE_FRONTEND = frontend-dev

help: ## Affiche l'aide
	@echo "Jarvis MCP - Commandes disponibles:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

dev: ## Lance l'environnement de développement
	@echo "🚀 Lancement de l'environnement de développement..."
	docker-compose -f $(COMPOSE_FILE) --profile dev up --build

build: ## Build l'application en mode production
	@echo "🏗️ Build de l'application..."
	docker-compose -f $(COMPOSE_FILE) build frontend

start: ## Démarre l'application en mode production
	@echo "▶️ Démarrage en mode production..."
	docker-compose -f $(COMPOSE_FILE) up frontend -d

stop: ## Arrête tous les services
	@echo "⏹️ Arrêt des services..."
	docker-compose -f $(COMPOSE_FILE) down

logs: ## Affiche les logs du frontend
	docker-compose -f $(COMPOSE_FILE) logs -f $(SERVICE_FRONTEND)

clean: ## Nettoie les containers et volumes
	@echo "🧹 Nettoyage des containers..."
	docker-compose -f $(COMPOSE_FILE) down -v --remove-orphans
	docker system prune -f

install-pnpm: ## Installe pnpm localement
	@if ! command -v pnpm &> /dev/null; then \
		echo "📦 Installation de pnpm..."; \
		npm install -g pnpm@8; \
	else \
		echo "✅ pnpm déjà installé"; \
	fi

setup: install-pnpm ## Setup initial du projet
	@echo "🛠️ Setup du projet..."
	cd frontend && pnpm install

test: ## Lance les tests
	@echo "🧪 Lancement des tests..."
	docker-compose -f $(COMPOSE_FILE) exec $(SERVICE_FRONTEND) pnpm test

lint: ## Lance le linting
	@echo "🔍 Linting du code..."
	docker-compose -f $(COMPOSE_FILE) exec $(SERVICE_FRONTEND) pnpm lint

type-check: ## Vérification TypeScript
	@echo "📝 Vérification TypeScript..."
	docker-compose -f $(COMPOSE_FILE) exec $(SERVICE_FRONTEND) pnpm type-check

health: ## Vérifie la santé des services
	@echo "❤️ Vérification de la santé des services..."
	@curl -f http://localhost:3000 > /dev/null 2>&1 && echo "✅ Frontend OK" || echo "❌ Frontend DOWN"
	@curl -f http://localhost:8000/health > /dev/null 2>&1 && echo "✅ Backend OK" || echo "❌ Backend DOWN"

restart: stop start ## Redémarre les services

fresh: clean setup dev ## Nettoyage complet et redémarrage