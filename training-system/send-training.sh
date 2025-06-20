#!/bin/bash

# Script pour envoyer les problèmes de formation à Jarvis
# Évite les validations répétées de Claude Code

MCP_HUB_URL="http://localhost:4000/mcp/chat"

# Fonction pour envoyer un message
send_training_problem() {
    local message="$1"
    local session_id="$2"
    
    echo "📤 Envoi: $message"
    
    curl -X POST "$MCP_HUB_URL" \
         -H "Content-Type: application/json" \
         -d "{\"message\": \"$message\", \"session_id\": \"$session_id\"}" \
         -s | jq '.response' 2>/dev/null || echo "Réponse reçue"
    
    echo ""
}

# Problème 1 - Formation de base
echo "🎓 LANCEMENT FORMATION JARVIS MCP"
echo "================================="

send_training_problem \
    "FORMATION PROBLÈME 1: Analyse restaurant-app structure et framework. WORKFLOW: LS puis Read package.json puis analyse factuelle." \
    "training-01"

echo "✅ Problème 1 envoyé - Vérifiez la réponse de Jarvis"