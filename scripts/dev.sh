#!/bin/bash
# Script de développement pour Jarvis MCP

set -e

echo "🚀 Jarvis MCP Development Setup"
echo "================================"

# Vérification des dépendances
echo "📋 Vérification des dépendances..."

if ! command -v docker &> /dev/null; then
    echo "❌ Docker n'est pas installé"
    exit 1
fi

if ! command -v docker-compose &> /dev/null; then
    echo "❌ Docker Compose n'est pas installé"
    exit 1
fi

echo "✅ Docker et Docker Compose sont installés"

# Nettoyage des processus conflictuels
echo "🧹 Nettoyage des processus conflictuels..."
pkill -f "next dev" 2>/dev/null || true
pkill -f "node.*3000" 2>/dev/null || true

# Vérification des ports
echo "🔍 Vérification des ports..."
if lsof -i :3000 &> /dev/null; then
    echo "⚠️  Port 3000 occupé, nettoyage..."
    lsof -ti :3000 | xargs kill -9 2>/dev/null || true
fi

if lsof -i :8000 &> /dev/null; then
    echo "✅ Backend sur port 8000 détecté"
else
    echo "⚠️  Backend non détecté sur port 8000"
fi

# Démarrage des services
echo "🚀 Démarrage des services..."
cd "$(dirname "$0")/.."

# Build et start
docker-compose -f frontend/docker-compose.yml --profile dev up --build -d

echo "⏳ Attente du démarrage des services..."
sleep 10

# Health check
echo "❤️  Vérification de la santé..."
if curl -f http://localhost:3000 &> /dev/null; then
    echo "✅ Frontend accessible sur http://localhost:3000"
else
    echo "❌ Frontend non accessible"
    echo "📋 Logs du frontend:"
    docker-compose -f frontend/docker-compose.yml logs frontend-dev
fi

echo ""
echo "🎉 Setup terminé!"
echo "Frontend: http://localhost:3000"
echo "Backend:  http://localhost:8000"
echo ""
echo "Commandes utiles:"
echo "make logs     - Voir les logs"
echo "make stop     - Arrêter les services"
echo "make restart  - Redémarrer"