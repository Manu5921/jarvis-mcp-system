#!/bin/bash

# Jarvis MCP - Startup Script
echo "🤖 Démarrage de Jarvis MCP System..."

# Check if Docker is running
if ! docker info >/dev/null 2>&1; then
    echo "❌ Docker n'est pas démarré. Veuillez démarrer Docker et réessayer."
    exit 1
fi

# Check if Ollama is running (optional but recommended)
if ! curl -s http://localhost:11434/api/version >/dev/null 2>&1; then
    echo "⚠️  Ollama ne semble pas être démarré sur le port 11434"
    echo "   Pour une expérience complète, démarrez Ollama avec: ollama serve"
    echo "   Ou installez-le depuis: https://ollama.ai"
    echo ""
fi

echo "🏗️  Construction et démarrage des conteneurs..."

# Build and start the containers
docker-compose -f docker-compose-simple.yml up --build -d

# Wait for services to be ready
echo "⏳ Attente du démarrage des services..."
sleep 5

# Check if services are running
if docker-compose -f docker-compose-simple.yml ps | grep -q "Up"; then
    echo "✅ Services démarrés avec succès!"
    echo ""
    echo "🌐 Interface Web: http://localhost:4002"
    echo "🔧 API Backend: http://localhost:4000"
    echo "📊 Status API: http://localhost:4000/mcp/status"
    echo ""
    echo "📝 Commandes utiles:"
    echo "   - Voir les logs: docker-compose -f docker-compose-simple.yml logs -f"
    echo "   - Arrêter: docker-compose -f docker-compose-simple.yml down"
    echo "   - Redémarrer: docker-compose -f docker-compose-simple.yml restart"
    echo ""
    echo "🤖 Jarvis MCP est prêt!"
else
    echo "❌ Erreur lors du démarrage des services"
    echo "📋 Vérifiez les logs: docker-compose -f docker-compose-simple.yml logs"
    exit 1
fi