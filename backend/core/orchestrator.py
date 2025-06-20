"""
Jarvis MCP Core Orchestrator
Central intelligence for multi-agent AI coordination
"""

import asyncio
import json
import logging
from datetime import datetime, timezone
from typing import Dict, List, Optional, Any, Tuple
from uuid import UUID, uuid4

from sentence_transformers import SentenceTransformer
import numpy as np

from agents.base import BaseAgent, AgentResponse, AgentStatus
from agents.ollama_agent import OllamaAgent
from agents.claude_agent import ClaudeAgent
from agents.openai_agent import OpenAIAgent
from db.models import Session, Message, UserPreference, MemoryVector
from db.database import get_database_session
from utils.embeddings import EmbeddingManager
from utils.metrics import MetricsCollector

logger = logging.getLogger(__name__)

class RequestContext:
    """Contexte d'une requête utilisateur"""
    
    def __init__(
        self, 
        user_id: UUID,
        session_id: UUID,
        content: str,
        message_type: str = "text",
        preferences: Optional[UserPreference] = None
    ):
        self.id = uuid4()
        self.user_id = user_id
        self.session_id = session_id
        self.content = content
        self.message_type = message_type
        self.preferences = preferences
        self.created_at = datetime.now(timezone.utc)
        
        # Analyse sémantique
        self.category: Optional[str] = None
        self.confidence: float = 0.0
        self.priority: int = 1
        self.embedding: Optional[List[float]] = None
        
        # Routage
        self.selected_agents: List[str] = []
        self.parallel_execution: bool = False
        
    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": str(self.id),
            "user_id": str(self.user_id),
            "session_id": str(self.session_id),
            "content": self.content,
            "category": self.category,
            "confidence": self.confidence,
            "selected_agents": self.selected_agents,
            "created_at": self.created_at.isoformat()
        }

class AgentPool:
    """Pool d'agents IA disponibles"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.agents: Dict[str, BaseAgent] = {}
        self.agent_status: Dict[str, AgentStatus] = {}
        self.metrics = MetricsCollector()
        
    async def initialize_agents(self):
        """Initialise tous les agents configurés"""
        agents_config = self.config.get("agents", {})
        
        for agent_id, agent_config in agents_config.items():
            if not agent_config.get("enabled", False):
                continue
                
            try:
                agent = await self._create_agent(agent_id, agent_config)
                if agent:
                    self.agents[agent_id] = agent
                    self.agent_status[agent_id] = AgentStatus.READY
                    logger.info(f"Agent {agent_id} initialisé avec succès")
                    
            except Exception as e:
                logger.error(f"Erreur initialisation agent {agent_id}: {e}")
                self.agent_status[agent_id] = AgentStatus.ERROR
    
    async def _create_agent(self, agent_id: str, config: Dict[str, Any]) -> Optional[BaseAgent]:
        """Factory pour créer un agent spécifique"""
        if agent_id == "ollama":
            return OllamaAgent(config)
        elif agent_id == "claude":
            return ClaudeAgent(config)
        elif agent_id == "openai":
            return OpenAIAgent(config)
        else:
            logger.warning(f"Type d'agent inconnu: {agent_id}")
            return None
    
    def get_available_agents(self) -> List[str]:
        """Retourne la liste des agents disponibles"""
        return [
            agent_id for agent_id, status in self.agent_status.items()
            if status == AgentStatus.READY
        ]
    
    def get_agent_capabilities(self, agent_id: str) -> List[str]:
        """Retourne les capacités d'un agent"""
        if agent_id in self.agents:
            return self.agents[agent_id].capabilities
        return []
    
    async def health_check(self) -> Dict[str, Any]:
        """Vérifie l'état de santé de tous les agents"""
        health_status = {}
        
        for agent_id, agent in self.agents.items():
            try:
                status = await agent.health_check()
                health_status[agent_id] = status
                self.agent_status[agent_id] = (
                    AgentStatus.READY if status["status"] == "healthy" 
                    else AgentStatus.ERROR
                )
            except Exception as e:
                health_status[agent_id] = {"status": "error", "error": str(e)}
                self.agent_status[agent_id] = AgentStatus.ERROR
        
        return health_status

class IntelligentRouter:
    """Routeur intelligent pour sélection d'agents"""
    
    def __init__(self, config: Dict[str, Any], agent_pool: AgentPool):
        self.config = config.get("routing", {})
        self.agent_pool = agent_pool
        self.routing_rules = self.config.get("rules", {})
        self.default_agent = self.config.get("default_agent", "ollama")
        self.auto_switch_threshold = self.config.get("auto_switch_threshold", 0.7)
        
    async def analyze_request(self, context: RequestContext) -> RequestContext:
        """Analyse et catégorise une requête"""
        content_lower = context.content.lower()
        
        # Catégorisation simple basée sur mots-clés
        categories = {
            "code": ["code", "function", "class", "import", "def", "var", "const", "async", "api", "endpoint"],
            "creative": ["écris", "crée", "imagine", "story", "poème", "article", "blog"],
            "analysis": ["analyse", "explique", "compare", "évalue", "résume"],
            "general": ["bonjour", "hello", "comment", "what", "how", "why"],
            "translation": ["traduis", "translate", "en français", "in english"]
        }
        
        max_score = 0
        best_category = "general"
        
        for category, keywords in categories.items():
            score = sum(1 for keyword in keywords if keyword in content_lower)
            if score > max_score:
                max_score = score
                best_category = category
        
        context.category = best_category
        context.confidence = min(max_score / 3.0, 1.0)  # Normalisation
        
        return context
    
    async def select_agents(self, context: RequestContext) -> List[str]:
        """Sélectionne les agents optimaux pour une requête"""
        available_agents = self.agent_pool.get_available_agents()
        
        if not available_agents:
            return []
        
        # Règles de routage par catégorie
        preferred_agents = self.routing_rules.get(context.category, [self.default_agent])
        
        # Filtrer les agents disponibles
        selected_agents = [
            agent for agent in preferred_agents 
            if agent in available_agents
        ]
        
        # Fallback sur agent par défaut
        if not selected_agents and self.default_agent in available_agents:
            selected_agents = [self.default_agent]
        
        # Gestion du traitement parallèle
        if (context.confidence >= self.auto_switch_threshold and 
            len(selected_agents) > 1 and 
            self.config.get("parallel_processing", False)):
            context.parallel_execution = True
            # Limiter le nombre d'agents en parallèle
            max_parallel = self.config.get("max_parallel_agents", 2)
            selected_agents = selected_agents[:max_parallel]
        else:
            # Sélectionner le meilleur agent uniquement
            selected_agents = selected_agents[:1]
        
        context.selected_agents = selected_agents
        return selected_agents

class JarvisMCPOrchestrator:
    """Orchestrateur central Jarvis MCP"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.agent_pool = AgentPool(config)
        self.router = IntelligentRouter(config, self.agent_pool)
        self.embedding_manager = EmbeddingManager(config.get("memory", {}))
        self.metrics = MetricsCollector()
        
        # État de l'orchestrateur
        self.is_initialized = False
        self.active_sessions: Dict[UUID, RequestContext] = {}
        
    async def initialize(self):
        """Initialise l'orchestrateur et tous ses composants"""
        try:
            # Initialiser les agents
            await self.agent_pool.initialize_agents()
            
            # Initialiser le gestionnaire d'embeddings
            await self.embedding_manager.initialize()
            
            self.is_initialized = True
            logger.info("Jarvis MCP Orchestrator initialisé avec succès")
            
        except Exception as e:
            logger.error(f"Erreur initialisation orchestrateur: {e}")
            raise
    
    async def process_request(
        self, 
        user_id: UUID,
        session_id: UUID,
        content: str,
        message_type: str = "text",
        preferences: Optional[UserPreference] = None
    ) -> Dict[str, Any]:
        """Traite une requête utilisateur complète"""
        
        if not self.is_initialized:
            raise RuntimeError("Orchestrateur non initialisé")
        
        # Créer le contexte de requête
        context = RequestContext(
            user_id=user_id,
            session_id=session_id,
            content=content,
            message_type=message_type,
            preferences=preferences
        )
        
        try:
            # 1. Analyser la requête
            context = await self.router.analyze_request(context)
            
            # 2. Générer l'embedding pour la mémoire
            context.embedding = await self.embedding_manager.create_embedding(content)
            
            # 3. Récupérer le contexte pertinent de la mémoire
            memory_context = await self._get_memory_context(user_id, context.embedding)
            
            # 4. Sélectionner les agents
            selected_agents = await self.router.select_agents(context)
            
            if not selected_agents:
                return {
                    "status": "error",
                    "message": "Aucun agent disponible",
                    "context": context.to_dict()
                }
            
            # 5. Exécuter les agents
            if context.parallel_execution:
                responses = await self._execute_parallel(context, selected_agents, memory_context)
            else:
                responses = await self._execute_sequential(context, selected_agents, memory_context)
            
            # 6. Synthétiser la réponse finale
            final_response = await self._synthesize_response(context, responses)
            
            # 7. Sauvegarder dans la base de données
            await self._save_interaction(context, responses, final_response)
            
            # 8. Mettre à jour la mémoire
            await self._update_memory(user_id, context, final_response)
            
            return {
                "status": "success",
                "response": final_response,
                "context": context.to_dict(),
                "agents_used": selected_agents,
                "execution_time": (datetime.now(timezone.utc) - context.created_at).total_seconds()
            }
            
        except Exception as e:
            logger.error(f"Erreur traitement requête {context.id}: {e}")
            return {
                "status": "error",
                "message": f"Erreur interne: {str(e)}",
                "context": context.to_dict()
            }
    
    async def _execute_parallel(
        self, 
        context: RequestContext, 
        agent_ids: List[str],
        memory_context: str
    ) -> List[AgentResponse]:
        """Exécute plusieurs agents en parallèle"""
        tasks = []
        
        for agent_id in agent_ids:
            if agent_id in self.agent_pool.agents:
                agent = self.agent_pool.agents[agent_id]
                task = agent.process_request(
                    content=context.content,
                    context=memory_context,
                    user_preferences=context.preferences
                )
                tasks.append(task)
        
        # Exécuter en parallèle avec timeout
        try:
            responses = await asyncio.wait_for(
                asyncio.gather(*tasks, return_exceptions=True),
                timeout=120  # 2 minutes timeout
            )
            
            # Filtrer les exceptions
            valid_responses = [
                response for response in responses 
                if isinstance(response, AgentResponse)
            ]
            
            return valid_responses
            
        except asyncio.TimeoutError:
            logger.warning(f"Timeout execution parallèle pour requête {context.id}")
            return []
    
    async def _execute_sequential(
        self, 
        context: RequestContext, 
        agent_ids: List[str],
        memory_context: str
    ) -> List[AgentResponse]:
        """Exécute les agents de manière séquentielle"""
        responses = []
        
        for agent_id in agent_ids:
            if agent_id in self.agent_pool.agents:
                try:
                    agent = self.agent_pool.agents[agent_id]
                    response = await agent.process_request(
                        content=context.content,
                        context=memory_context,
                        user_preferences=context.preferences
                    )
                    responses.append(response)
                    
                    # Si la réponse est satisfaisante, arrêter
                    if response.confidence >= 0.8:
                        break
                        
                except Exception as e:
                    logger.error(f"Erreur agent {agent_id}: {e}")
                    continue
        
        return responses
    
    async def _synthesize_response(
        self, 
        context: RequestContext, 
        responses: List[AgentResponse]
    ) -> Dict[str, Any]:
        """Synthétise les réponses multiples en une réponse finale"""
        
        if not responses:
            return {
                "content": "Désolé, je n'ai pas pu traiter votre demande.",
                "confidence": 0.0,
                "agent_used": None
            }
        
        # Si une seule réponse, la retourner directement
        if len(responses) == 1:
            response = responses[0]
            return {
                "content": response.content,
                "confidence": response.confidence,
                "agent_used": response.agent_id,
                "response_time": response.response_time,
                "metadata": response.metadata
            }
        
        # Fusion intelligente de multiples réponses
        best_response = max(responses, key=lambda r: r.confidence)
        
        # Agrégation des métadonnées
        combined_metadata = {
            "agents_consulted": [r.agent_id for r in responses],
            "average_confidence": sum(r.confidence for r in responses) / len(responses),
            "best_agent": best_response.agent_id,
            "total_response_time": sum(r.response_time for r in responses)
        }
        
        return {
            "content": best_response.content,
            "confidence": best_response.confidence,
            "agent_used": best_response.agent_id,
            "response_time": best_response.response_time,
            "metadata": combined_metadata
        }
    
    async def _get_memory_context(self, user_id: UUID, query_embedding: List[float]) -> str:
        """Récupère le contexte pertinent de la mémoire utilisateur"""
        try:
            # Cette fonction serait implémentée avec la base de données
            # Pour l'instant, retourner un contexte vide
            return ""
        except Exception as e:
            logger.error(f"Erreur récupération mémoire: {e}")
            return ""
    
    async def _save_interaction(
        self, 
        context: RequestContext, 
        responses: List[AgentResponse],
        final_response: Dict[str, Any]
    ):
        """Sauvegarde l'interaction en base de données"""
        # Implémentation de la sauvegarde
        pass
    
    async def _update_memory(
        self, 
        user_id: UUID, 
        context: RequestContext, 
        final_response: Dict[str, Any]
    ):
        """Met à jour la mémoire vectorielle"""
        # Implémentation de la mise à jour mémoire
        pass
    
    async def get_agent_status(self) -> Dict[str, Any]:
        """Retourne l'état de tous les agents"""
        return await self.agent_pool.health_check()
    
    async def get_metrics(self) -> Dict[str, Any]:
        """Retourne les métriques de l'orchestrateur"""
        return await self.metrics.get_summary()