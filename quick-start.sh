#!/bin/bash

# 🚀 Quick Start Script pour Jarvis MCP
# Script de démarrage rapide avec vérifications automatiques

set -e

echo "🤖 JARVIS MCP - ASSISTANT IA ORCHESTRATEUR"
echo "=========================================="
echo ""

# Couleurs pour les messages
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Fonctions utilitaires
print_success() {
    echo -e "${GREEN}✅ $1${NC}"
}

print_error() {
    echo -e "${RED}❌ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠️ $1${NC}"
}

print_info() {
    echo -e "${BLUE}ℹ️ $1${NC}"
}

# Vérifier les prérequis
check_prerequisites() {
    print_info "Vérification des prérequis..."
    
    # Python 3.11+
    if command -v python3 &> /dev/null; then
        PYTHON_VERSION=$(python3 --version | grep -oE '[0-9]+\.[0-9]+')
        if [[ $(echo "$PYTHON_VERSION >= 3.11" | bc -l) -eq 1 ]]; then
            print_success "Python $PYTHON_VERSION détecté"
        else
            print_error "Python 3.11+ requis (détecté: $PYTHON_VERSION)"
            exit 1
        fi
    else
        print_error "Python 3 non trouvé"
        exit 1
    fi
    
    # Node.js 18+
    if command -v node &> /dev/null; then
        NODE_VERSION=$(node --version | grep -oE '[0-9]+' | head -1)
        if [[ $NODE_VERSION -ge 18 ]]; then
            print_success "Node.js v$NODE_VERSION détecté"
        else
            print_error "Node.js 18+ requis (détecté: v$NODE_VERSION)"
            exit 1
        fi
    else
        print_error "Node.js non trouvé"
        exit 1
    fi
    
    # PostgreSQL
    if command -v psql &> /dev/null; then
        print_success "PostgreSQL détecté"
    else
        print_warning "PostgreSQL non détecté - installation recommandée"
        read -p "Continuer sans PostgreSQL? (y/N): " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            exit 1
        fi
    fi
    
    # Ollama
    if command -v ollama &> /dev/null; then
        print_success "Ollama détecté"
        
        # Vérifier si Ollama est en cours d'exécution
        if ollama list &> /dev/null; then
            print_success "Ollama service actif"
            
            # Vérifier le modèle llama3.2:8b
            if ollama list | grep -q "llama3.2:8b"; then
                print_success "Modèle llama3.2:8b disponible"
            else
                print_warning "Modèle llama3.2:8b non trouvé"
                read -p "Télécharger maintenant? (Y/n): " -n 1 -r
                echo
                if [[ ! $REPLY =~ ^[Nn]$ ]]; then
                    print_info "Téléchargement du modèle llama3.2:8b..."
                    ollama pull llama3.2:8b
                    print_success "Modèle téléchargé avec succès"
                fi
            fi
        else
            print_warning "Ollama service non actif"
            print_info "Démarrez Ollama avec: ollama serve"
        fi
    else
        print_warning "Ollama non détecté"
        print_info "Installez Ollama depuis: https://ollama.com"
        read -p "Continuer sans Ollama? (y/N): " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            exit 1
        fi
    fi
}

# Setup environnement backend
setup_backend() {
    print_info "Configuration du backend..."
    
    cd backend
    
    # Créer environnement virtuel
    if [ ! -d "venv" ]; then
        print_info "Création de l'environnement virtuel..."
        python3 -m venv venv
    fi
    
    # Activer environnement virtuel
    source venv/bin/activate
    
    # Installer dépendances
    print_info "Installation des dépendances Python..."
    pip install --upgrade pip
    pip install -r requirements.txt
    
    # Créer fichier .env si inexistant
    if [ ! -f ".env" ]; then
        print_info "Création du fichier .env..."
        cat > .env << EOF
DATABASE_URL=postgresql+asyncpg://jarvis:jarvis_password@localhost:5432/jarvis_mcp
SECRET_KEY=$(openssl rand -hex 32)
ANTHROPIC_API_KEY=
OPENAI_API_KEY=
ENVIRONMENT=development
EOF
        print_warning "Configurez vos clés API dans backend/.env"
    fi
    
    cd ..
    print_success "Backend configuré"
}

# Setup environnement frontend  
setup_frontend() {
    print_info "Configuration du frontend..."
    
    cd frontend
    
    # Installer dépendances
    print_info "Installation des dépendances Node.js..."
    npm install
    
    # Créer fichier .env.local si inexistant
    if [ ! -f ".env.local" ]; then
        print_info "Création du fichier .env.local..."
        cat > .env.local << EOF
NEXT_PUBLIC_API_URL=http://localhost:8000
NEXT_PUBLIC_WS_URL=ws://localhost:8000
EOF
    fi
    
    cd ..
    print_success "Frontend configuré"
}

# Configurer base de données
setup_database() {
    print_info "Configuration de la base de données..."
    
    # Vérifier si PostgreSQL est installé
    if ! command -v psql &> /dev/null; then
        print_warning "PostgreSQL non installé - passage à SQLite pour le développement"
        return
    fi
    
    # Créer base de données si elle n'existe pas
    if ! psql -lqt | cut -d \| -f 1 | grep -qw jarvis_mcp; then
        print_info "Création de la base de données jarvis_mcp..."
        
        # Commandes de création
        sudo -u postgres createdb jarvis_mcp 2>/dev/null || createdb jarvis_mcp
        sudo -u postgres psql -c "CREATE USER jarvis WITH PASSWORD 'jarvis_password';" 2>/dev/null || \
            psql -c "CREATE USER jarvis WITH PASSWORD 'jarvis_password';"
        sudo -u postgres psql -c "GRANT ALL PRIVILEGES ON DATABASE jarvis_mcp TO jarvis;" 2>/dev/null || \
            psql -c "GRANT ALL PRIVILEGES ON DATABASE jarvis_mcp TO jarvis;"
        
        # Activer extension pgvector
        sudo -u postgres psql jarvis_mcp -c "CREATE EXTENSION IF NOT EXISTS vector;" 2>/dev/null || \
            psql jarvis_mcp -c "CREATE EXTENSION IF NOT EXISTS vector;"
        
        print_success "Base de données créée"
    else
        print_success "Base de données jarvis_mcp existe déjà"
    fi
    
    # Exécuter migrations
    cd backend
    source venv/bin/activate
    print_info "Exécution des migrations..."
    alembic upgrade head
    cd ..
    
    print_success "Base de données configurée"
}

# Démarrer les services
start_services() {
    print_info "Démarrage des services..."
    
    # Créer script de démarrage
    cat > start_jarvis.sh << 'EOF'
#!/bin/bash

# Couleurs
GREEN='\033[0;32m'
BLUE='\033[0;34m'
NC='\033[0m'

echo -e "${BLUE}🚀 Démarrage Jarvis MCP...${NC}"

# Fonction pour tuer les processus en arrière-plan à la sortie
cleanup() {
    echo -e "\n${BLUE}🛑 Arrêt des services Jarvis MCP...${NC}"
    kill $BACKEND_PID $FRONTEND_PID 2>/dev/null
    exit 0
}

trap cleanup SIGINT SIGTERM

# Démarrer backend
echo -e "${GREEN}▶️ Démarrage du backend...${NC}"
cd backend
source venv/bin/activate
python main.py &
BACKEND_PID=$!
cd ..

# Attendre que le backend soit prêt
sleep 5

# Démarrer frontend
echo -e "${GREEN}▶️ Démarrage du frontend...${NC}"
cd frontend
npm run dev &
FRONTEND_PID=$!
cd ..

echo -e "${GREEN}✅ Jarvis MCP démarré !${NC}"
echo ""
echo "🌐 Frontend: http://localhost:3000"
echo "🔧 API Backend: http://localhost:8000"
echo "📚 Documentation API: http://localhost:8000/docs"
echo ""
echo "Appuyez sur Ctrl+C pour arrêter les services"

# Attendre les processus
wait $BACKEND_PID $FRONTEND_PID
EOF
    
    chmod +x start_jarvis.sh
    print_success "Script de démarrage créé"
    
    print_info "Pour démarrer Jarvis MCP:"
    echo ""
    echo -e "${BLUE}  ./start_jarvis.sh${NC}"
    echo ""
}

# Menu principal
main_menu() {
    echo "Que souhaitez-vous faire ?"
    echo ""
    echo "1) Installation complète"
    echo "2) Vérifier les prérequis seulement"
    echo "3) Configuration backend seulement"
    echo "4) Configuration frontend seulement"
    echo "5) Configuration base de données seulement"
    echo "6) Créer scripts de démarrage seulement"
    echo "7) Quitter"
    echo ""
    read -p "Choisissez une option (1-7): " choice
    
    case $choice in
        1)
            check_prerequisites
            setup_backend
            setup_frontend
            setup_database
            start_services
            print_success "Installation complète terminée !"
            ;;
        2)
            check_prerequisites
            print_success "Vérification des prérequis terminée"
            ;;
        3)
            setup_backend
            ;;
        4)
            setup_frontend
            ;;
        5)
            setup_database
            ;;
        6)
            start_services
            ;;
        7)
            print_info "Au revoir !"
            exit 0
            ;;
        *)
            print_error "Option invalide"
            main_menu
            ;;
    esac
}

# Vérifier si on est dans le bon répertoire
if [ ! -f "README.md" ] || [ ! -d "backend" ] || [ ! -d "frontend" ]; then
    print_error "Exécutez ce script depuis la racine du projet Jarvis MCP"
    exit 1
fi

# Démarrer le menu principal
main_menu