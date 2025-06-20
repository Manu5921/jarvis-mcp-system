#!/bin/bash

# Script de lancement de la formation Jarvis MCP
# Force l'utilisation des outils et valide automatiquement les réponses

echo "🎓 FORMATION JARVIS MCP - DÉMARRAGE"
echo "===================================="

# Vérification que le système MCP est démarré
echo "🔍 Vérification du système MCP..."
if ! curl -s http://localhost:4000/health > /dev/null; then
    echo "❌ Hub MCP non accessible. Démarrage requis..."
    echo "   Exécutez: docker-compose -f docker-compose-mcp-advanced.yml up -d"
    exit 1
fi

echo "✅ Hub MCP accessible"

# Installation des dépendances si nécessaire
echo "📦 Vérification des dépendances..."
if [ ! -d "node_modules" ]; then
    echo "📥 Installation des dépendances..."
    npm init -y 2>/dev/null
    npm install axios 2>/dev/null
fi

# Lancement du système de formation
echo "🚀 Lancement de la formation automatique..."
node training-orchestrator.js

echo ""
echo "🎯 FORMATION TERMINÉE"
echo "📊 Consultez les logs pour les résultats détaillés"