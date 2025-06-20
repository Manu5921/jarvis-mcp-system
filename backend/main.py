"""
Jarvis MCP FastAPI Server
Multi-Channel Processor for AI orchestration
"""

import json
import logging
import asyncio
from datetime import datetime, timezone
from typing import Dict, List, Optional, Any
from pathlib import Path
from contextlib import asynccontextmanager

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, Depends, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
import uvicorn
from pydantic import BaseModel, Field
from uuid import UUID, uuid4

from core.orchestrator import JarvisMCPOrchestrator
from db.database import init_database_manager, get_database_session, health_check
from db.models import User, Session, Message, UserPreference
from utils.websocket_manager import WebSocketManager
from utils.auth import get_current_user, create_access_token
from utils.logging_config import setup_logging

# Configuration logging
setup_logging()
logger = logging.getLogger(__name__)

# Variables globales
orchestrator: Optional[JarvisMCPOrchestrator] = None
websocket_manager: WebSocketManager = WebSocketManager()
config: Dict[str, Any] = {}

# Modèles Pydantic pour l'API
class ChatRequest(BaseModel):
    content: str = Field(..., min_length=1, max_length=8000)
    session_id: Optional[UUID] = None
    message_type: str = Field(default="text", pattern="^(text|code|image)$")
    
class ChatResponse(BaseModel):
    status: str
    response: Optional[Dict[str, Any]] = None
    session_id: UUID
    message_id: UUID
    execution_time: float
    agents_used: List[str]
    
class HealthResponse(BaseModel):
    status: str
    database: Dict[str, Any]
    agents: Dict[str, Any]
    websocket_connections: int
    uptime_seconds: float

class AgentStatusResponse(BaseModel):
    agents: Dict[str, Any]
    orchestrator_initialized: bool
    
class UserPreferencesRequest(BaseModel):
    preferred_agent: Optional[str] = "ollama"
    preferred_tone: Optional[str] = "neutral"
    preferred_language: Optional[str] = "fr"
    auto_switch_enabled: Optional[bool] = True
    parallel_processing: Optional[bool] = False

# Cycle de vie de l'application
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Gestion du cycle de vie de l'application"""
    global orchestrator, config
    
    # Startup
    logger.info("🚀 Démarrage Jarvis MCP Server")
    
    try:
        # Charger la configuration
        config_path = Path(__file__).parent.parent / "config" / "jarvis_config.json"
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
        
        # TODO: Initialiser la base de données
        # db_manager = init_database_manager(
        #     config["database"]["url"], 
        #     config["database"]["echo"]
        # )
        # await db_manager.init_database()
        
        # TODO: Initialiser l'orchestrateur
        # orchestrator = JarvisMCPOrchestrator(config)
        # await orchestrator.initialize()
        
        # Démarrer les tâches de fond
        asyncio.create_task(background_tasks())
        
        # Démarrer le heartbeat WebSocket
        websocket_manager._start_heartbeat()
        
        logger.info("✅ Jarvis MCP Server démarré avec succès")
        
    except Exception as e:
        logger.error(f"❌ Erreur démarrage: {e}")
        raise
    
    yield
    
    # Shutdown
    logger.info("🛑 Arrêt Jarvis MCP Server")
    
    if orchestrator:
        # Nettoyer l'orchestrateur et les agents
        for agent in orchestrator.agent_pool.agents.values():
            await agent.cleanup()
    
    # Fermer les connexions WebSocket
    await websocket_manager.disconnect_all()
    
    logger.info("✅ Jarvis MCP Server arrêté proprement")

# Application FastAPI
app = FastAPI(
    title="Jarvis MCP Server",
    description="Multi-Channel Processor for AI orchestration",
    version="1.0.0",
    lifespan=lifespan
)

# Configuration CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],  # Frontend React
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Routes principales
@app.get("/", response_model=Dict[str, str])
async def root():
    """Point d'entrée de l'API"""
    return {
        "service": "Jarvis MCP Server",
        "version": "1.0.0",
        "status": "running",
        "docs": "/docs",
        "health": "/health"
    }

@app.get("/")
async def root():
    """Point d'entrée de l'API"""
    return {
        "service": "Jarvis MCP",
        "version": "1.0.0",
        "status": "running",
        "description": "Multi-Channel Processor for AI orchestration",
        "endpoints": {
            "health": "/health",
            "docs": "/docs",
            "chat": "/chat",
            "websocket": "/ws",
            "agents": "/agents/status",
            "metrics": "/metrics"
        }
    }

@app.get("/health", response_model=HealthResponse)
async def health_endpoint(db = Depends(get_database_session)):
    """Vérification de santé complète du système"""
    try:
        # Santé base de données
        db_health = await health_check(db)
        
        # Santé agents IA
        agents_health = {}
        if orchestrator:
            agents_health = await orchestrator.get_agent_status()
        
        # Uptime (approximatif)
        uptime = 0.0  # TODO: implémenter le tracking d'uptime
        
        status = "healthy"
        if db_health["status"] != "healthy":
            status = "degraded"
        elif not agents_health or all(agent["status"] != "healthy" for agent in agents_health.values()):
            status = "degraded"
        
        return HealthResponse(
            status=status,
            database=db_health,
            agents=agents_health,
            websocket_connections=websocket_manager.get_connection_count(),
            uptime_seconds=uptime
        )
        
    except Exception as e:
        logger.error(f"Erreur health check: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Health check failed"
        )

@app.get("/agents/status", response_model=AgentStatusResponse)
async def get_agents_status():
    """Statut des agents IA"""
    if not orchestrator:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Orchestrator not initialized"
        )
    
    agents_status = await orchestrator.get_agent_status()
    
    return AgentStatusResponse(
        agents=agents_status,
        orchestrator_initialized=orchestrator.is_initialized
    )

@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(
    request: ChatRequest,
    current_user: User = Depends(get_current_user),
    db = Depends(get_database_session)
):
    """Endpoint principal pour les conversations"""
    if not orchestrator or not orchestrator.is_initialized:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="AI orchestrator not available"
        )
    
    try:
        # Créer ou récupérer la session
        session_id = request.session_id or uuid4()
        
        # Récupérer les préférences utilisateur
        user_preferences = await db.query(UserPreference).filter(
            UserPreference.user_id == current_user.id
        ).first()
        
        # Traiter la requête avec l'orchestrateur
        result = await orchestrator.process_request(
            user_id=current_user.id,
            session_id=session_id,
            content=request.content,
            message_type=request.message_type,
            preferences=user_preferences
        )
        
        if result["status"] == "success":
            return ChatResponse(
                status="success",
                response=result["response"],
                session_id=session_id,
                message_id=uuid4(),  # TODO: utiliser l'ID réel du message
                execution_time=result["execution_time"],
                agents_used=result["agents_used"]
            )
        else:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=result.get("message", "Request processing failed")
            )
    
    except Exception as e:
        logger.error(f"Erreur chat endpoint: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error"
        )

@app.websocket("/ws/{client_id}")
async def websocket_endpoint(websocket: WebSocket, client_id: str):
    """WebSocket pour communication temps réel"""
    await websocket_manager.connect(websocket, client_id)
    
    try:
        while True:
            # Recevoir les messages WebSocket
            data = await websocket.receive_json()
            
            # Traiter le message
            await handle_websocket_message(websocket, client_id, data)
            
    except WebSocketDisconnect:
        websocket_manager.disconnect(client_id)
        logger.info(f"Client WebSocket {client_id} déconnecté")
    except Exception as e:
        logger.error(f"Erreur WebSocket {client_id}: {e}")
        websocket_manager.disconnect(client_id)

async def handle_websocket_message(websocket: WebSocket, client_id: str, data: Dict[str, Any]):
    """Traite les messages WebSocket"""
    try:
        message_type = data.get("type")
        
        if message_type == "chat":
            # Message de chat via WebSocket
            content = data.get("content", "")
            user_id = data.get("user_id")  # TODO: authentification WebSocket
            
            if not content or not user_id:
                await websocket.send_json({
                    "type": "error",
                    "message": "Content and user_id required"
                })
                return
            
            # Traiter avec l'orchestrateur (version simplifiée)
            if orchestrator and orchestrator.is_initialized:
                result = await orchestrator.process_request(
                    user_id=UUID(user_id),
                    session_id=uuid4(),
                    content=content,
                    message_type="text"
                )
                
                await websocket.send_json({
                    "type": "chat_response",
                    "data": result
                })
            else:
                await websocket.send_json({
                    "type": "error",
                    "message": "AI orchestrator not available"
                })
                
        elif message_type == "ping":
            # Répondre au ping pour garder la connexion vivante
            await websocket.send_json({
                "type": "pong",
                "timestamp": datetime.now(timezone.utc).isoformat()
            })
        
        else:
            await websocket.send_json({
                "type": "error",
                "message": f"Unknown message type: {message_type}"
            })
    
    except Exception as e:
        logger.error(f"Erreur traitement message WebSocket: {e}")
        await websocket.send_json({
            "type": "error",
            "message": "Message processing failed"
        })

@app.post("/users/preferences")
async def update_user_preferences(
    preferences: UserPreferencesRequest,
    current_user: User = Depends(get_current_user),
    db = Depends(get_database_session)
):
    """Met à jour les préférences utilisateur"""
    try:
        # Chercher les préférences existantes
        user_pref = await db.query(UserPreference).filter(
            UserPreference.user_id == current_user.id
        ).first()
        
        if not user_pref:
            # Créer nouvelles préférences
            user_pref = UserPreference(
                user_id=current_user.id,
                preferred_agent=preferences.preferred_agent,
                preferred_tone=preferences.preferred_tone,
                preferred_language=preferences.preferred_language,
                auto_switch_enabled=preferences.auto_switch_enabled,
                parallel_processing=preferences.parallel_processing
            )
            db.add(user_pref)
        else:
            # Mettre à jour les préférences existantes
            for field, value in preferences.dict(exclude_unset=True).items():
                setattr(user_pref, field, value)
        
        await db.commit()
        
        return {"status": "success", "message": "Preferences updated"}
        
    except Exception as e:
        logger.error(f"Erreur mise à jour préférences: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to update preferences"
        )

@app.get("/metrics")
async def get_metrics():
    """Métriques du système"""
    if not orchestrator:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Orchestrator not available"
        )
    
    try:
        metrics = await orchestrator.get_metrics()
        return metrics
    except Exception as e:
        logger.error(f"Erreur récupération métriques: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve metrics"
        )

# Tâches de fond
async def background_tasks():
    """Tâches de fond périodiques"""
    while True:
        try:
            # Health check périodique des agents
            if orchestrator:
                await orchestrator.agent_pool.health_check()
            
            # Nettoyage des connexions WebSocket inactives
            websocket_manager.cleanup_inactive_connections()
            
            # Attendre 30 secondes avant le prochain cycle
            await asyncio.sleep(30)
            
        except Exception as e:
            logger.error(f"Erreur tâche de fond: {e}")
            await asyncio.sleep(5)  # Attendre moins longtemps en cas d'erreur

# Point d'entrée pour le développement
if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )