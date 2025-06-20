#!/bin/bash
# Script de démarrage MCP Hub - Écosystème IA avec communications MCP

echo "🧠 MCP Hub - Démarrage écosystème IA"
echo "===================================="

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
export CLAUDE_API_KEY=${CLAUDE_API_KEY:-""}

if [ -z "$PERPLEXITY_API_KEY" ]; then
    echo "⚠️  Variable PERPLEXITY_API_KEY non définie"
    echo "💡 Export PERPLEXITY_API_KEY=your_key"
fi

# Nettoyage des ports 4000-4010
echo "🧹 Nettoyage des ports MCP..."
for port in {4000..4010}; do
    lsof -ti :$port | xargs kill -9 2>/dev/null || true
done

# Démarrage écosystème MCP
echo "🚀 Démarrage écosystème MCP..."
docker-compose -f docker-compose.mcp.yml up --build -d

# Attente démarrage
echo "⏳ Attente démarrage services MCP..."
sleep 15

# Vérifications
echo "❤️  Vérifications MCP:"

services=(
    "4000:MCP Hub Central"
    "4002:Frontend MCP"
    "4003:Ollama MCP"
    "4004:Perplexity MCP" 
    "4005:Memory MCP"
    "4006:Tools MCP"
)

all_good=true

for service in "${services[@]}"; do
    port=$(echo $service | cut -d: -f1)
    name=$(echo $service | cut -d: -f2)
    
    if curl -f http://localhost:$port &> /dev/null; then
        echo "✅ $name: http://localhost:$port"
    else
        echo "❌ $name inaccessible"
        all_good=false
    fi
done

# Test communication MCP
echo ""
echo "🔗 Test communications MCP..."

# Test MCP Hub
if curl -f http://localhost:4000/mcp/status &> /dev/null; then
    echo "✅ MCP Hub responsive"
    
    # Afficher le statut des serveurs MCP
    echo "📊 Statut serveurs MCP:"
    curl -s http://localhost:4000/mcp/status | python3 -m json.tool 2>/dev/null || echo "Format JSON invalide"
else
    echo "❌ MCP Hub non responsive"
    all_good=false
fi

echo ""
if [ "$all_good" = true ]; then
    echo "🎉 Écosystème MCP opérationnel!"
    echo ""
    echo "🔗 Accès services:"
    echo "Frontend MCP:     http://localhost:4002"
    echo "MCP Hub API:      http://localhost:4000"
    echo "MCP Hub Status:   http://localhost:4000/mcp/status"
    echo "MCP Hub Docs:     http://localhost:4000/docs"
    echo ""
    echo "🧠 Communication MCP active entre:"
    echo "- Ollama (local dev)"
    echo "- Perplexity (recherche)"
    echo "- Memory (contexte/historique)"
    echo "- Tools (fichiers/utilitaires)"
    echo ""
    echo "💡 Features MCP:"
    echo "- Orchestration intelligente"
    echo "- Contexte partagé entre AIs"
    echo "- Optimisation de prompts"
    echo "- Mémoire persistante"
    echo "- Communication temps réel"
else
    echo "⚠️  Certains services ont des problèmes"
    echo "📋 Logs pour debug:"
    echo "docker-compose -f docker-compose.mcp.yml logs"
fi

echo ""
echo "🛠️  Commandes utiles:"
echo "docker-compose -f docker-compose.mcp.yml logs -f           # Voir logs"
echo "docker-compose -f docker-compose.mcp.yml down             # Arrêter"
echo "docker-compose -f docker-compose.mcp.yml restart          # Redémarrer"