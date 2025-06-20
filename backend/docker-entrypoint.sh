#!/bin/bash
set -e

echo "🚀 Démarrage Jarvis MCP Backend..."

# Attendre que PostgreSQL soit prêt
echo "⏳ Attente de PostgreSQL..."
until pg_isready -h postgres -p 5432 -U jarvis; do
    echo "PostgreSQL n'est pas encore prêt - attente..."
    sleep 2
done
echo "✅ PostgreSQL est prêt !"

# Exécuter les migrations
echo "🔄 Exécution des migrations..."
alembic upgrade head

# Vérifier la configuration
echo "🔍 Vérification de la configuration..."
python -c "
import sys
import os
try:
    from db.database import init_database_manager
    from core.orchestrator import JarvisMCPOrchestrator
    print('✅ Configuration valide')
except Exception as e:
    print(f'❌ Erreur configuration: {e}')
    sys.exit(1)
"

echo "🎯 Jarvis MCP Backend prêt à démarrer !"

# Exécuter la commande principale
exec "$@"