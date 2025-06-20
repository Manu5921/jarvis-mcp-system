# 🧠 Jarvis MCP - Écosystème IA Intelligent

> **Orchestration d'agents IA via Model Context Protocol (MCP)**
> 
> Un hub central pour coordonner Ollama (local), Perplexity (recherche), Memory (contexte) et Tools (utilitaires) via le protocole MCP standardisé.

## 🏗️ Architecture MCP

```
┌─────────────────────────────────────────────────────────┐
│                    MCP HUB CENTRAL                       │
│                   (Port 4000)                           │
│  ┌─────────────────────────────────────────────────────┐ │
│  │ • Routage intelligent des requêtes                 │ │
│  │ • Orchestration multi-agents                       │ │
│  │ • Optimisation de prompts contextuels             │ │
│  │ • WebSocket temps réel                             │ │
│  └─────────────────────────────────────────────────────┘ │
└─────────────────────┬───────────────────────────────────┘
                      │
        ┌─────────────┼─────────────┐
        │             │             │
   ┌────▼───┐    ┌───▼────┐    ┌───▼────┐    ┌────▼───┐
   │Ollama  │    │Perplex.│    │Memory  │    │Tools   │
   │MCP     │    │MCP     │    │MCP     │    │MCP     │
   │:4003   │    │:4004   │    │:4005   │    │:4006   │
   │        │    │        │    │        │    │        │
   │🦙 Local│    │🔍 Search│    │💾 SQLite│    │🛠️ Utils│
   │AI      │    │API     │    │Context │    │Files   │
   └────────┘    └────────┘    └────────┘    └────────┘
```

## 🚀 Démarrage Rapide

### Prérequis
```bash
# Ollama doit être démarré en local
ollama serve

# Variables d'environnement (optionnelles)
export PERPLEXITY_API_KEY="your_perplexity_api_key"
```

### Démarrage de l'écosystème
```bash
# Donner les permissions d'exécution
chmod +x start-mcp-hub.sh

# Démarrer tous les services MCP
./start-mcp-hub.sh
```

### Vérification du statut
```bash
# Vérifier que tous les services sont actifs
curl http://localhost:4000/mcp/status

# Interface web
open http://localhost:4002
```

## 🔗 Services MCP

### 🏗️ MCP Hub Central (Port 4000)
**Orchestrateur principal pour tous les agents IA**

- **Endpoint principal:** `http://localhost:4000`
- **Documentation:** `http://localhost:4000/docs`
- **Status:** `http://localhost:4000/mcp/status`

**Fonctionnalités:**
- Routage intelligent selon la complexité des requêtes
- Optimisation automatique des prompts
- Gestion de la mémoire vectorielle
- Communication WebSocket temps réel
- Load balancing entre agents

### 🦙 Ollama MCP Server (Port 4003) 
**Assistant IA local pour développement**

**Capacités MCP:**
- `ollama_chat`: Chat avec contexte projet
- `ollama_analyze_code`: Analyse spécialisée de code
- `ollama_optimize_prompt`: Optimisation pour autres AIs

**Modèles supportés:**
- `llama3.2:3b` (rapide, requêtes simples)
- `llama3.2:8b` (complexe, analyse de code)

### 🔍 Perplexity MCP Server (Port 4004)
**Recherche en temps réel et actualités tech**

**Capacités MCP:**
- `perplexity_search`: Recherche contextuelle
- `perplexity_research`: Recherche multi-angles
- `perplexity_tech_news`: Actualités technologiques
- `perplexity_compare`: Comparaison de technologies

**Modèles:**
- `llama-3.1-sonar-small-128k-online` (rapide)
- `llama-3.1-sonar-large-128k-online` (approfondi)

### 💾 Memory MCP Server (Port 4005)
**Gestion intelligente du contexte et historique**

**Capacités MCP:**
- `memory_store`: Stockage conversations
- `memory_get_context`: Récupération contexte pertinent
- `memory_optimize_prompt`: Optimisation basée sur l'historique
- `memory_search`: Recherche dans l'historique
- `memory_stats`: Statistiques d'usage

**Base de données:** SQLite avec indexation sémantique

### 🛠️ Tools MCP Server (Port 4006)
**Utilitaires système et manipulation de fichiers**

**Capacités MCP:**
- `file_read/write/list`: Gestion fichiers sécurisée
- `web_fetch`: Récupération de contenu web
- `command_run`: Exécution commandes (whitelist)
- `code_analyze`: Analyse statique de code
- `backup_create`: Sauvegarde du workspace

## 🌐 Interface Web (Port 4002)

**Interface simple et moderne pour interagir avec l'écosystème MCP**

### Fonctionnalités:
- **Chat Multi-Agents:** Sélection automatique ou manuelle de l'agent
- **Optimisation de Prompts:** Génération de prompts optimisés pour Claude/ChatGPT
- **Historique Intelligent:** Recherche et réutilisation du contexte
- **Statistiques en Temps Réel:** Monitoring des performances
- **Interface Responsive:** Compatible mobile/desktop

### Agents disponibles:
- 🦙 **Ollama Local:** Rapide, ideal pour développement
- 🔍 **Perplexity Pro:** Recherche en temps réel, actualités
- 🤖 **Claude Code:** Génération prompts optimisés
- 💭 **ChatGPT:** Génération prompts optimisés

## 📡 API MCP Hub

### Endpoints principaux

```bash
# Chat avec agent automatique
POST /mcp/chat
{
  "message": "Comment optimiser une base de données ?",
  "agent": "auto|ollama|perplexity",
  "project": "mon-projet",
  "context": "FastAPI + SQLite"
}

# Optimisation de prompt
POST /mcp/optimize-prompt  
{
  "original_prompt": "Explain this code",
  "target_ai": "claude|chatgpt|perplexity",
  "context": "React component",
  "project": "frontend-app"
}

# Statut de l'écosystème
GET /mcp/status

# Historique conversations
GET /mcp/conversations?limit=10&project=my-project
```

### Réponses types
```json
{
  "response": "...",
  "agent_used": "ollama",
  "optimized_prompt": "...",
  "context_applied": "...",
  "timestamp": "2024-12-20T...",
  "conversation_id": 123
}
```

## ⚙️ Configuration Avancée

### Variables d'environnement
```bash
# Perplexity API (optionnel)
PERPLEXITY_API_KEY=your_key

# Ollama (local)
OLLAMA_HOST=localhost:11434

# Ports personnalisés (défaut: 4000-4006)
MCP_HUB_PORT=4000
OLLAMA_MCP_PORT=4003
PERPLEXITY_MCP_PORT=4004
MEMORY_MCP_PORT=4005
TOOLS_MCP_PORT=4006
FRONTEND_PORT=4002
```

### Configuration Ollama
```bash
# Installer des modèles supplémentaires
ollama pull llama3.2:8b
ollama pull codellama:7b
ollama pull mistral:7b

# Vérifier les modèles disponibles
ollama list
```

## 🧪 Tests et Debugging

### Tests de santé des services
```bash
# Test individuel des services MCP
curl http://localhost:4003/health  # Ollama MCP
curl http://localhost:4004/health  # Perplexity MCP  
curl http://localhost:4005/health  # Memory MCP
curl http://localhost:4006/health  # Tools MCP

# Test du hub central
curl http://localhost:4000/mcp/status
```

### Tests de chat
```bash
# Test Ollama via MCP Hub
curl -X POST http://localhost:4000/mcp/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "Bonjour!", "agent": "ollama"}'

# Test Perplexity via MCP Hub  
curl -X POST http://localhost:4000/mcp/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "Latest AI news", "agent": "perplexity"}'
```

### Logs et monitoring
```bash
# Voir les logs en temps réel
docker-compose -f docker-compose.mcp.yml logs -f

# Logs d'un service spécifique
docker-compose -f docker-compose.mcp.yml logs -f mcp-hub
docker-compose -f docker-compose.mcp.yml logs -f ollama-mcp
```

## 🔧 Maintenance

### Arrêt et redémarrage
```bash
# Arrêter l'écosystème
docker-compose -f docker-compose.mcp.yml down

# Redémarrer un service spécifique  
docker-compose -f docker-compose.mcp.yml restart mcp-hub

# Reconstruction complète
docker-compose -f docker-compose.mcp.yml up --build -d
```

### Nettoyage
```bash
# Nettoyer les containers et volumes
docker-compose -f docker-compose.mcp.yml down -v
docker system prune -f

# Nettoyer les ports 4000-4010
./start-mcp-hub.sh  # Le script nettoie automatiquement
```

## 🎯 Cas d'Usage

### 1. Développement Local
- **Ollama** pour l'assistance code rapide
- **Memory** pour maintenir le contexte projet
- **Tools** pour l'analyse de fichiers

### 2. Recherche et Veille Tech
- **Perplexity** pour les dernières actualités
- **Memory** pour stocker les insights
- **Hub** pour synthétiser les informations

### 3. Optimisation de Prompts
- **Ollama** analyse votre contexte  
- **Memory** récupère l'historique pertinent
- **Hub** génère des prompts optimisés pour Claude/ChatGPT

### 4. Orchestration Multi-Agents
- **Routage automatique** selon la complexité
- **Contexte partagé** entre tous les agents
- **Apprentissage** des préférences utilisateur

## 🔍 Dépannage

### Problèmes courants

**Ollama non accessible:**
```bash
# Vérifier qu'Ollama est démarré
ollama serve
curl http://localhost:11434/api/tags
```

**Ports occupés:**
```bash
# Le script start-mcp-hub.sh nettoie automatiquement
# Ou manuellement:
lsof -ti :4000 | xargs kill -9
```

**Perplexity API limitée:**
```bash
# L'écosystème fonctionne sans clé Perplexity
# Seuls les tests Perplexity échoueront
```

**Problèmes CORS Frontend:**
```bash
# Le frontend est configuré pour CORS
# Vérifier que MCP Hub autorise les CORS
curl -H "Origin: http://localhost:4002" http://localhost:4000/mcp/status
```

## 🚀 Roadmap

### Prochaines fonctionnalités
- [ ] Support Claude API direct (en plus des prompts)
- [ ] Interface vocale avec reconnaissance/synthèse
- [ ] Plugins personnalisés pour nouveaux agents
- [ ] Dashboard d'analytics avancé
- [ ] API REST complète pour intégrations externes
- [ ] Support multi-utilisateurs avec authentification

---

**🎉 L'écosystème Jarvis MCP est maintenant opérationnel !**

Accédez à l'interface sur `http://localhost:4002` et commencez à orchestrer vos agents IA de manière intelligente.