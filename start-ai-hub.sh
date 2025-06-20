#!/bin/bash
# Script de démarrage AI Memory Hub

echo "🧠 AI Memory Hub - Démarrage"
echo "============================"

# Vérifications
if ! command -v docker &> /dev/null; then
    echo "❌ Docker requis"
    exit 1
fi

if ! curl -s http://localhost:11434 &> /dev/null; then
    echo "⚠️  Ollama non détecté sur port 11434"
    echo "💡 Assure-toi qu'Ollama est démarré: ollama serve"
fi

# Variables d'environnement
export PERPLEXITY_API_KEY=${PERPLEXITY_API_KEY:-""}

# Nettoyage des ports
echo "🧹 Nettoyage des ports..."
lsof -ti :3000 | xargs kill -9 2>/dev/null || true
lsof -ti :8001 | xargs kill -9 2>/dev/null || true

# Démarrage
echo "🚀 Démarrage des services..."
docker-compose -f docker-compose.simple.yml up --build -d

# Attente
echo "⏳ Attente démarrage..."
sleep 10

# Vérifications
echo "❤️  Vérifications:"
if curl -f http://localhost:3000 &> /dev/null; then
    echo "✅ Frontend: http://localhost:3000"
else
    echo "❌ Frontend inaccessible"
fi

if curl -f http://localhost:8001 &> /dev/null; then
    echo "✅ API: http://localhost:8001"
else
    echo "❌ API inaccessible"
fi

echo ""
echo "🎉 AI Memory Hub prêt!"
echo "Interface: http://localhost:3000"
echo "API: http://localhost:8001/docs"
echo ""
echo "Commandes utiles:"
echo "docker-compose -f docker-compose.simple.yml logs -f  # Voir logs"
echo "docker-compose -f docker-compose.simple.yml down     # Arrêter"