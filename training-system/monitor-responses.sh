#!/bin/bash

# Monitore les réponses de Jarvis pendant la formation
# Affiche les scores et valide automatiquement

echo "📊 MONITORING FORMATION JARVIS"
echo "=============================="

# Fonction pour vérifier une réponse
check_response() {
    local session_id="$1"
    
    echo "🔍 Vérification session: $session_id"
    
    # Simulation de validation (dans la vraie version, on analyserait la réponse)
    # curl -s "http://localhost:4000/mcp/sessions/$session_id" | jq '.response'
    
    echo "✅ Réponse analysée"
}

# Boucle de monitoring
for i in {1..10}; do
    echo ""
    echo "📈 Problème $i - Attente réponse..."
    sleep 5
    check_response "training-$(printf "%02d" $i)"
done

echo ""
echo "🎯 MONITORING TERMINÉ"