# 🤖 Jarvis MCP - Multi-Channel Processor

**Assistant IA local "Jarvis" avec orchestration intelligente multi-agents**

Jarvis MCP est un orchestrateur IA avancé qui coordonne plusieurs agents IA (Ollama local, Claude, OpenAI) pour fournir des réponses optimisées selon le contexte et les préférences utilisateur.

## 🎯 Fonctionnalités Principales

### 🧠 **Orchestration Intelligente**
- **Routage automatique** des requêtes vers l'agent IA le plus adapté
- **Traitement parallèle** ou séquentiel selon la complexité
- **Fusion intelligente** des réponses multiples
- **Apprentissage continu** basé sur le feedback utilisateur

### 🔌 **Agents IA Supportés**
- **Ollama Local** (llama3.2:8b) - Confidentialité et performance
- **Claude (Anthropic)** - Expertise code et analyse
- **OpenAI GPT** - Créativité et tâches générales
- **Architecture modulaire** pour ajouter facilement de nouveaux agents

### 💾 **Mémoire Vectorielle**
- **Base PostgreSQL + pgvector** pour mémoire sémantique
- **Embeddings** pour contexte pertinent
- **Historique intelligent** des conversations
- **Profils utilisateur** personnalisés

### 🌐 **Interface Moderne**
- **Dashboard React/Next.js 15** avec Tailwind CSS
- **WebSocket temps réel** pour communication fluide
- **Visualisation** des sessions et métriques
- **Configuration** des préférences avancées

## 🏗️ Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Frontend      │    │   Backend MCP    │    │   Agents IA     │
│   React/Next.js │◄──►│   FastAPI        │◄──►│   Ollama        │
│   + WebSocket   │    │   + WebSocket    │    │   Claude        │
│   + Dashboard   │    │   + Orchestrator │    │   OpenAI        │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                              │
                              ▼
                       ┌──────────────────┐
                       │   PostgreSQL     │
                       │   + pgvector     │
                       │   + Embeddings   │
                       └──────────────────┘
```

## 🚀 Installation Rapide

### Prérequis
- **Python 3.11+**
- **Node.js 18+**
- **PostgreSQL 15+** avec extension pgvector
- **Ollama** installé et démarré
- **Clés API** Claude et/ou OpenAI (optionnel)

### 1. Clone et Setup Backend

```bash
git clone <votre-repo>
cd jarvis-mcp/backend

# Créer environnement virtuel
python -m venv venv
source venv/bin/activate  # Linux/Mac
# ou venv\\Scripts\\activate  # Windows

# Installer dépendances
pip install -r requirements.txt
```

### 2. Configuration Base de Données

```bash
# Installer PostgreSQL et pgvector
sudo apt install postgresql postgresql-contrib  # Ubuntu
brew install postgresql               # macOS

# Créer base de données
sudo -u postgres createdb jarvis_mcp
sudo -u postgres psql -c "CREATE USER jarvis WITH PASSWORD 'jarvis_password';"
sudo -u postgres psql -c "GRANT ALL PRIVILEGES ON DATABASE jarvis_mcp TO jarvis;"

# Activer pgvector
sudo -u postgres psql jarvis_mcp -c "CREATE EXTENSION vector;"
```

### 3. Configuration Environnement

```bash
# Créer fichier .env
cat > .env << EOF
DATABASE_URL=postgresql+asyncpg://jarvis:jarvis_password@localhost:5432/jarvis_mcp
SECRET_KEY=your-super-secret-key-change-in-production
ANTHROPIC_API_KEY=your-claude-api-key-optional
OPENAI_API_KEY=your-openai-api-key-optional
EOF
```

### 4. Migrations Base de Données

```bash
# Initialiser Alembic et créer tables
alembic upgrade head
```

### 5. Setup Frontend

```bash
cd ../frontend

# Installer dépendances
npm install

# Configuration environnement
cat > .env.local << EOF
NEXT_PUBLIC_API_URL=http://localhost:8000
NEXT_PUBLIC_WS_URL=ws://localhost:8000
EOF
```

### 6. Démarrage

**Terminal 1 - Backend :**
```bash
cd backend
python main.py
# Serveur disponible sur http://localhost:8000
```

**Terminal 2 - Frontend :**
```bash
cd frontend  
npm run dev
# Interface disponible sur http://localhost:3000
```

**Terminal 3 - Ollama :**
```bash
ollama serve
ollama pull llama3.2:8b  # Télécharger le modèle
```

## 📖 Guide Utilisation

### Interface Web

1. **Accédez** à http://localhost:3000
2. **Créez** un compte utilisateur
3. **Configurez** vos préférences (agents favoris, ton, langue)
4. **Commencez** à discuter avec Jarvis !

### API REST

```bash
# Health check
curl http://localhost:8000/health

# Chat (avec authentification)
curl -X POST http://localhost:8000/chat \\
  -H "Authorization: Bearer <token>" \\
  -H "Content-Type: application/json" \\
  -d '{"content": "Explique-moi Next.js", "message_type": "text"}'
```

### WebSocket

```javascript
const ws = new WebSocket('ws://localhost:8000/ws/client-123');

ws.onopen = () => {
  ws.send(JSON.stringify({
    type: 'chat',
    content: 'Bonjour Jarvis !',
    user_id: 'user-uuid'
  }));
};

ws.onmessage = (event) => {
  const response = JSON.parse(event.data);
  console.log('Réponse Jarvis:', response);
};
```

## ⚙️ Configuration Avancée

### Routage Intelligent

Modifiez `config/jarvis_config.json` :

```json
{
  "routing": {
    "rules": {
      "code": ["claude", "ollama"],
      "creative": ["openai", "ollama"],
      "analysis": ["claude", "ollama"],
      "general": ["ollama", "openai"]
    },
    "parallel_processing": true,
    "max_parallel_agents": 2
  }
}
```

### Agents IA

```json
{
  "agents": {
    "ollama": {
      "enabled": true,
      "model": "llama3.2:8b",
      "capabilities": ["general", "code", "analysis"]
    },
    "claude": {
      "enabled": true,
      "model": "claude-3-sonnet-20240229",
      "capabilities": ["code", "analysis", "writing"]
    }
  }
}
```

## 🧪 Tests et Développement

### Tests Backend

```bash
cd backend
pytest tests/ -v
```

### Tests Frontend

```bash
cd frontend
npm test
npm run test:watch
```

### Linting et Formatage

```bash
# Backend
black backend/
isort backend/
mypy backend/

# Frontend  
npm run lint
npm run type-check
```

## 📊 Métriques et Monitoring

### Dashboard Métriques

- **Agents Status** : État temps réel des agents IA
- **Performance** : Temps de réponse, taux de succès
- **Usage** : Statistiques utilisateur et sessions
- **Qualité** : Scores de confiance et feedback

### API Métriques

```bash
curl http://localhost:8000/metrics
```

### Logs Structurés

```bash
tail -f logs/jarvis.log | jq .
```

## 🔌 Extension et Personnalisation

### Ajouter un Nouvel Agent IA

1. **Créer** `backend/agents/new_agent.py` :

```python
from .base import BaseAgent, AgentResponse

class NewAgent(BaseAgent):
    async def process_request(self, content: str, **kwargs) -> AgentResponse:
        # Implémentation de votre agent
        return AgentResponse(...)
```

2. **Enregistrer** dans `core/orchestrator.py`
3. **Configurer** dans `config/jarvis_config.json`

### Personnaliser le Frontend

```bash
# Ajouter composants dans frontend/components/
# Modifier styles dans frontend/app/globals.css
# Étendre hooks dans frontend/hooks/
```

## 🔒 Sécurité et Production

### Variables d'Environnement Production

```bash
export SECRET_KEY="$(openssl rand -hex 32)"
export DATABASE_URL="postgresql+asyncpg://user:pass@host:5432/db"
export ANTHROPIC_API_KEY="your-production-key"
export OPENAI_API_KEY="your-production-key"
```

### Déploiement Docker

```bash
# TODO: Ajouter Dockerfile et docker-compose.yml
docker-compose up -d
```

### Reverse Proxy

```nginx
# Configuration Nginx
location /api {
    proxy_pass http://localhost:8000;
    proxy_set_header Host $host;
    proxy_set_header X-Real-IP $remote_addr;
}

location /ws {
    proxy_pass http://localhost:8000;
    proxy_http_version 1.1;
    proxy_set_header Upgrade $http_upgrade;
    proxy_set_header Connection "upgrade";
}
```

## 🤝 Contribution

1. **Fork** le projet
2. **Créez** votre branche (`git checkout -b feature/AmazingFeature`)
3. **Committez** (`git commit -m 'Add AmazingFeature'`)
4. **Push** (`git push origin feature/AmazingFeature`)
5. **Ouvrez** une Pull Request

## 📜 Licence

Distribué sous licence MIT. Voir `LICENSE` pour plus d'informations.

## 🙏 Remerciements

- **Ollama** pour l'IA locale
- **Anthropic Claude** pour l'expertise IA
- **OpenAI** pour l'innovation IA
- **FastAPI** pour l'API moderne
- **Next.js** pour le frontend réactif
- **PostgreSQL + pgvector** pour la mémoire vectorielle

## 📞 Support

- **Documentation** : Voir `/docs`
- **Issues** : GitHub Issues
- **Discussions** : GitHub Discussions

---

**🚀 Jarvis MCP - Votre assistant IA personnel intelligent et local !**