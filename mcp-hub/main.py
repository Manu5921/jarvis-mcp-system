"""
MCP Hub Central - Orchestrateur intelligent pour écosystème IA
Gère les communications MCP entre AIs et outils
"""

import asyncio
import json
import logging
import sqlite3
import time
from contextlib import asynccontextmanager
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Union
from pathlib import Path

from fastapi import FastAPI, HTTPException, status, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import uvicorn
import httpx

# Configuration
DATABASE_PATH = "/app/data/mcp_hub.db"
OLLAMA_HOST = "host.docker.internal:11434"
MCP_HUB_PORT = 4000

# Global state
mcp_clients = {}
active_connections = []

# Initialize database
def init_database():
    """Initialize SQLite database for MCP Hub"""
    Path("/app/data").mkdir(parents=True, exist_ok=True)
    
    conn = sqlite3.connect(DATABASE_PATH)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS conversations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            message TEXT NOT NULL,
            response TEXT NOT NULL,
            agent_used TEXT NOT NULL,
            project TEXT,
            context TEXT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            response_time_ms INTEGER,
            success BOOLEAN DEFAULT TRUE
        )
    """)
    
    conn.execute("""
        CREATE TABLE IF NOT EXISTS agent_stats (
            agent_name TEXT PRIMARY KEY,
            total_requests INTEGER DEFAULT 0,
            total_success INTEGER DEFAULT 0,
            avg_response_time_ms REAL DEFAULT 0,
            last_used DATETIME
        )
    """)
    
    conn.commit()
    conn.close()

# Lifespan management
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifecycle"""
    # Startup
    init_database()
    logging.info("🧠 MCP Hub Central démarré")
    yield
    # Shutdown
    logging.info("🛑 MCP Hub Central arrêté")

# FastAPI app with modern patterns
app = FastAPI(
    title="MCP Hub Central",
    version="1.0.0",
    description="Orchestrateur intelligent pour écosystème IA via MCP",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic Models with modern typing
class ChatRequest(BaseModel):
    message: str = Field(..., description="Message à envoyer à l'agent IA")
    agent: str = Field("auto", description="Agent IA: 'auto', 'ollama', 'perplexity'")
    project: Optional[str] = Field(None, description="Nom du projet pour contexte")
    context: Optional[str] = Field(None, description="Contexte additionnel")

class OptimizePromptRequest(BaseModel):
    original_prompt: str = Field(..., description="Prompt original à optimiser")
    target_ai: str = Field(..., description="IA cible: 'claude', 'chatgpt', 'perplexity'")
    context: Optional[str] = Field(None, description="Contexte du projet")
    project: Optional[str] = Field(None, description="Nom du projet")

class ChatResponse(BaseModel):
    response: str
    agent_used: str
    optimized_prompt: Optional[str] = None
    timestamp: str
    response_time_ms: int
    conversation_id: int

class ConversationItem(BaseModel):
    id: int
    message: str
    response: str
    agent_used: str
    project: Optional[str]
    timestamp: str

class StatsResponse(BaseModel):
    total_conversations: int
    recent_activity: int
    ai_usage: Dict[str, int]
    avg_response_times: Dict[str, float]

# Agent managers
class AgentManager:
    """Gestionnaire des agents IA"""
    
    @staticmethod
    async def call_ollama(message: str, context: str = None) -> str:
        """Appel à Ollama local"""
        try:
            # Préparer le prompt avec contexte
            prompt = message
            if context:
                prompt = f"Contexte: {context}\n\nQuestion: {message}"
            
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.post(
                    f"http://{OLLAMA_HOST}/api/generate",
                    json={
                        "model": "llama3.2:3b",
                        "prompt": prompt,
                        "stream": False,
                        "options": {
                            "temperature": 0.7,
                            "top_p": 0.9
                        }
                    }
                )
                
                if response.status_code == 200:
                    data = response.json()
                    return data.get("response", "Erreur: pas de réponse d'Ollama")
                else:
                    return f"Erreur Ollama: HTTP {response.status_code}"
                    
        except Exception as e:
            return f"Erreur connexion Ollama: {str(e)}"
    
    @staticmethod
    async def call_perplexity(message: str, context: str = None) -> str:
        """Appel à Perplexity (simulé pour l'instant)"""
        # Pour l'instant, on simule Perplexity
        enhanced_query = f"Recherche: {message}"
        if context:
            enhanced_query += f" (Contexte: {context})"
        
        return f"🔍 Résultats Perplexity pour: {enhanced_query}\n\n[Simulation] Cette fonctionnalité sera connectée au vrai Perplexity Pro quand la clé API sera fournie."
    
    @staticmethod
    async def optimize_prompt(original: str, target_ai: str, context: str = None, project: str = None) -> str:
        """Optimisation de prompt pour différentes IA"""
        
        optimizations = {
            "claude": f"""Tu es Claude Code, assistant expert en développement.

Contexte du projet: {project or "Projet générique"}
Contexte technique: {context or "Non spécifié"}

Demande originale: {original}

Réponds de manière structurée avec:
1. Analyse technique précise
2. Solutions pratiques avec code si pertinent
3. Recommandations d'amélioration
4. Considérations de sécurité/performance si applicable""",

            "chatgpt": f"""Tu es un assistant développeur expert.

Projet: {project or "Développement"}
Contexte: {context or "Non spécifié"}

Demande: {original}

Fournis une réponse équilibrée avec:
- Explication claire du problème/besoin
- Solution étape par étape
- Code d'exemple si nécessaire
- Alternatives possibles""",

            "perplexity": f"""Recherche approfondie sur: {original}

Projet: {project or ""}
Contexte: {context or ""}

Focus sur:
- Informations récentes (2024-2025)
- Meilleures pratiques actuelles
- Documentation officielle
- Exemples concrets et tutoriels"""
        }
        
        return optimizations.get(target_ai, original)

# Database helpers
class DatabaseManager:
    """Gestionnaire base de données SQLite"""
    
    @staticmethod
    def store_conversation(message: str, response: str, agent: str, project: str = None, 
                          context: str = None, response_time_ms: int = 0) -> int:
        """Stocker une conversation"""
        conn = sqlite3.connect(DATABASE_PATH)
        cursor = conn.execute("""
            INSERT INTO conversations (message, response, agent_used, project, context, response_time_ms)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (message, response, agent, project, context, response_time_ms))
        
        conversation_id = cursor.lastrowid
        conn.commit()
        conn.close()
        
        # Mettre à jour les stats
        DatabaseManager.update_agent_stats(agent, response_time_ms, True)
        
        return conversation_id
    
    @staticmethod
    def update_agent_stats(agent: str, response_time_ms: int, success: bool):
        """Mettre à jour les statistiques d'un agent"""
        conn = sqlite3.connect(DATABASE_PATH)
        
        # Vérifier si l'agent existe
        cursor = conn.execute("SELECT * FROM agent_stats WHERE agent_name = ?", (agent,))
        if cursor.fetchone():
            # Mettre à jour
            conn.execute("""
                UPDATE agent_stats 
                SET total_requests = total_requests + 1,
                    total_success = total_success + ?,
                    avg_response_time_ms = (avg_response_time_ms * total_requests + ?) / (total_requests + 1),
                    last_used = CURRENT_TIMESTAMP
                WHERE agent_name = ?
            """, (1 if success else 0, response_time_ms, agent))
        else:
            # Créer
            conn.execute("""
                INSERT INTO agent_stats (agent_name, total_requests, total_success, avg_response_time_ms, last_used)
                VALUES (?, 1, ?, ?, CURRENT_TIMESTAMP)
            """, (agent, 1 if success else 0, response_time_ms))
        
        conn.commit()
        conn.close()
    
    @staticmethod
    def get_conversations(limit: int = 10, project: str = None) -> List[Dict]:
        """Récupérer les conversations récentes"""
        conn = sqlite3.connect(DATABASE_PATH)
        
        query = "SELECT id, message, response, agent_used, project, timestamp FROM conversations"
        params = []
        
        if project:
            query += " WHERE project = ?"
            params.append(project)
        
        query += " ORDER BY timestamp DESC LIMIT ?"
        params.append(limit)
        
        cursor = conn.execute(query, params)
        conversations = []
        
        for row in cursor.fetchall():
            conversations.append({
                "id": row[0],
                "message": row[1],
                "response": row[2][:200] + "..." if len(row[2]) > 200 else row[2],
                "agent_used": row[3],
                "project": row[4],
                "timestamp": row[5]
            })
        
        conn.close()
        return conversations
    
    @staticmethod
    def get_stats() -> Dict:
        """Récupérer les statistiques globales"""
        conn = sqlite3.connect(DATABASE_PATH)
        
        # Total conversations
        cursor = conn.execute("SELECT COUNT(*) FROM conversations")
        total_conversations = cursor.fetchone()[0]
        
        # Activité récente (7 derniers jours)
        cursor = conn.execute("""
            SELECT COUNT(*) FROM conversations 
            WHERE timestamp > datetime('now', '-7 days')
        """)
        recent_activity = cursor.fetchone()[0]
        
        # Usage par IA
        cursor = conn.execute("""
            SELECT agent_used, COUNT(*) 
            FROM conversations 
            GROUP BY agent_used
        """)
        ai_usage = dict(cursor.fetchall())
        
        # Temps de réponse moyens
        cursor = conn.execute("""
            SELECT agent_name, avg_response_time_ms 
            FROM agent_stats
        """)
        avg_response_times = dict(cursor.fetchall())
        
        conn.close()
        
        return {
            "total_conversations": total_conversations,
            "recent_activity": recent_activity,
            "ai_usage": ai_usage,
            "avg_response_times": avg_response_times
        }

# HTTP Client for MCP servers
class MCPClient:
    """HTTP client for MCP servers"""
    
    def __init__(self, base_url: str):
        self.base_url = base_url
        self.client = httpx.AsyncClient(timeout=30.0)
    
    async def call_tool(self, tool_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Call a tool on the MCP server"""
        try:
            response = await self.client.post(
                f"{self.base_url}/tools/{tool_name}",
                json=arguments
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logging.error(f"Error calling {tool_name}: {e}")
            raise
    
    async def list_tools(self) -> List[Dict[str, Any]]:
        """List available tools"""
        try:
            response = await self.client.get(f"{self.base_url}/tools")
            response.raise_for_status()
            return response.json().get("tools", [])
        except Exception as e:
            logging.error(f"Error listing tools: {e}")
            return []
    
    async def health_check(self) -> bool:
        """Check if server is healthy"""
        try:
            response = await self.client.get(f"{self.base_url}/health", timeout=5.0)
            return response.status_code == 200
        except:
            return False

async def initialize_mcp_ecosystem():
    """Initialize HTTP connections to MCP servers"""
    servers = {
        "ollama": "http://mcp-ollama:4003",
        "perplexity": "http://mcp-perplexity:4004", 
        "memory": "http://mcp-memory:4005",
        "tools": "http://mcp-tools:4006"
    }
    
    for name, url in servers.items():
        try:
            client = MCPClient(url)
            if await client.health_check():
                mcp_clients[name] = client
                logging.info(f"✅ MCP Server connecté: {name} at {url}")
            else:
                logging.warning(f"⚠️ MCP Server {name} non disponible at {url}")
        except Exception as e:
            logging.error(f"❌ Erreur connexion MCP {name}: {e}")

# AI Routing Logic
class AIRouter:
    def __init__(self):
        self.routing_rules = {
            "code": ["ollama", "tools"],
            "search": ["perplexity", "memory"],
            "analysis": ["ollama", "memory", "tools"],
            "creative": ["ollama"],
            "debug": ["ollama", "tools"],
            "research": ["perplexity", "memory"]
        }
    
    def detect_intent(self, query: str) -> str:
        """Simple intent detection"""
        query_lower = query.lower()
        
        if any(word in query_lower for word in ["bug", "error", "debug", "fix"]):
            return "debug"
        elif any(word in query_lower for word in ["search", "find", "research", "latest"]):
            return "search"
        elif any(word in query_lower for word in ["code", "function", "class", "implement"]):
            return "code"
        elif any(word in query_lower for word in ["analyze", "explain", "understand"]):
            return "analysis"
        elif any(word in query_lower for word in ["create", "write", "generate"]):
            return "creative"
        else:
            return "code"  # Default for development focus

router = AIRouter()


# Main chat endpoint for frontend
@app.post("/mcp/chat")
async def handle_chat(request: ChatRequest) -> ChatResponse:
    """Handle chat requests from frontend"""
    start_time = time.time()
    
    try:
        if request.agent == "auto":
            # Simple auto-routing based on query
            intent = router.detect_intent(request.message)
            agent = "ollama" if intent in ["code", "analysis", "creative"] else "perplexity"
        else:
            agent = request.agent
        
        response_text = ""
        
        if agent == "ollama":
            response_text = await AgentManager.call_ollama(request.message, request.context)
        elif agent == "perplexity":
            response_text = await AgentManager.call_perplexity(request.message, request.context)
        else:
            response_text = "Agent non supporté"
        
        response_time_ms = int((time.time() - start_time) * 1000)
        
        # Store conversation
        conversation_id = DatabaseManager.store_conversation(
            request.message, response_text, agent, 
            request.project, request.context, response_time_ms
        )
        
        return ChatResponse(
            response=response_text,
            agent_used=agent,
            timestamp=datetime.now().isoformat(),
            response_time_ms=response_time_ms,
            conversation_id=conversation_id
        )
        
    except Exception as e:
        logging.error(f"Erreur chat: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/mcp/optimize-prompt")
async def handle_optimize_prompt(request: OptimizePromptRequest) -> ChatResponse:
    """Handle prompt optimization requests"""
    start_time = time.time()
    
    try:
        optimized_prompt = await AgentManager.optimize_prompt(
            request.original_prompt, request.target_ai, 
            request.context, request.project
        )
        
        response_time_ms = int((time.time() - start_time) * 1000)
        
        # Store optimization
        conversation_id = DatabaseManager.store_conversation(
            request.original_prompt, optimized_prompt, 
            f"optimize-{request.target_ai}", 
            request.project, request.context, response_time_ms
        )
        
        return ChatResponse(
            response="Prompt optimisé généré avec succès",
            agent_used=f"optimize-{request.target_ai}",
            optimized_prompt=optimized_prompt,
            timestamp=datetime.now().isoformat(),
            response_time_ms=response_time_ms,
            conversation_id=conversation_id
        )
        
    except Exception as e:
        logging.error(f"Erreur optimisation prompt: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/mcp/conversations")
async def get_conversations(limit: int = 10, project: str = None) -> List[ConversationItem]:
    """Get conversation history"""
    try:
        conversations = DatabaseManager.get_conversations(limit, project)
        return [
            ConversationItem(
                id=conv["id"],
                message=conv["message"],
                response=conv["response"],
                agent_used=conv["agent_used"],
                project=conv["project"],
                timestamp=conv["timestamp"]
            )
            for conv in conversations
        ]
    except Exception as e:
        logging.error(f"Erreur récupération conversations: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/mcp/stats")
async def get_stats() -> StatsResponse:
    """Get usage statistics"""
    try:
        stats = DatabaseManager.get_stats()
        return StatsResponse(**stats)
    except Exception as e:
        logging.error(f"Erreur récupération stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/mcp/status")
async def get_mcp_status():
    """Get status of all MCP servers"""
    status = {}
    
    for name, client in mcp_clients.items():
        try:
            health = await client.health_check()
            status[name] = {
                "connected": health,
                "last_ping": datetime.now().isoformat()
            }
        except Exception as e:
            status[name] = {
                "connected": False,
                "error": str(e)
            }
    
    # Get stats for compatibility with frontend
    try:
        stats = DatabaseManager.get_stats()
        return {
            "mcp_servers": status,
            "active_connections": len(active_connections),
            "hub_status": "running",
            **stats
        }
    except:
        return {
            "mcp_servers": status,
            "active_connections": len(active_connections),
            "hub_status": "running"
        }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    uvicorn.run(app, host="0.0.0.0", port=MCP_HUB_PORT)