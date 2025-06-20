#!/bin/bash
# Script d'arrêt MCP Hub - Écosystème IA

echo "🛑 MCP Hub - Arrêt écosystème IA"
echo "================================="

# Arrêter Docker Compose
echo "📦 Arrêt des containers MCP..."
docker-compose -f docker-compose.mcp.yml down

# Nettoyage des ports (optionnel)
echo "🧹 Nettoyage des ports MCP..."
for port in {4000..4010}; do
    lsof -ti :$port | xargs kill -9 2>/dev/null || true
done

echo ""
echo "✅ Écosystème MCP arrêté avec succès!"
echo ""
echo "🔄 Pour redémarrer:"
echo "./start-mcp-hub.sh"
echo ""
echo "🗑️  Pour nettoyer complètement (volumes + images):"
echo "docker-compose -f docker-compose.mcp.yml down -v"
echo "docker system prune -f"