"""
Base classes for AI agents in Jarvis MCP
Abstract interface for all AI agent implementations
"""

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from typing import Dict, List, Optional, Any
from uuid import uuid4

class AgentStatus(Enum):
    """Statut d'un agent IA"""
    INITIALIZING = "initializing"
    READY = "ready"
    BUSY = "busy"
    ERROR = "error"
    OFFLINE = "offline"

@dataclass
class AgentResponse:
    """Réponse standardisée d'un agent IA"""
    agent_id: str
    content: str
    confidence: float  # 0.0 - 1.0
    response_time: float  # en secondes
    
    # Métadonnées optionnelles
    model_used: Optional[str] = None
    tokens_used: Optional[int] = None
    temperature: Optional[float] = None
    metadata: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    
    def __post_init__(self):
        self.id = str(uuid4())
        self.timestamp = datetime.now(timezone.utc)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "agent_id": self.agent_id,
            "content": self.content,
            "confidence": self.confidence,
            "response_time": self.response_time,
            "model_used": self.model_used,
            "tokens_used": self.tokens_used,
            "temperature": self.temperature,
            "metadata": self.metadata,
            "error": self.error,
            "timestamp": self.timestamp.isoformat()
        }

@dataclass
class AgentCapabilities:
    """Capacités d'un agent IA"""
    supported_tasks: List[str]  # ["code", "creative", "analysis"]
    max_context_length: int
    supports_streaming: bool
    supports_function_calling: bool
    languages: List[str]  # ["fr", "en", "es"]
    
class BaseAgent(ABC):
    """Classe de base pour tous les agents IA"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.agent_id = config.get("agent_id", self.__class__.__name__.lower())
        self.capabilities = config.get("capabilities", [])
        self.status = AgentStatus.INITIALIZING
        self.model_name = config.get("model", "unknown")
        self.temperature = config.get("temperature", 0.7)
        self.max_tokens = config.get("max_tokens", 2048)
        
        # Statistiques
        self.total_requests = 0
        self.successful_requests = 0
        self.failed_requests = 0
        self.total_response_time = 0.0
        
    @abstractmethod
    async def initialize(self) -> bool:
        """Initialise l'agent (connexion, modèles, etc.)"""
        pass
    
    @abstractmethod
    async def process_request(
        self, 
        content: str, 
        context: Optional[str] = None,
        user_preferences: Optional[Dict[str, Any]] = None
    ) -> AgentResponse:
        """Traite une requête utilisateur"""
        pass
    
    @abstractmethod
    async def health_check(self) -> Dict[str, Any]:
        """Vérifie l'état de santé de l'agent"""
        pass
    
    async def cleanup(self):
        """Nettoie les ressources de l'agent"""
        self.status = AgentStatus.OFFLINE
    
    def _format_prompt(
        self, 
        content: str, 
        context: Optional[str] = None,
        user_preferences: Optional[Dict[str, Any]] = None
    ) -> str:
        """Formate le prompt selon les préférences"""
        
        # Prompt de base
        prompt_parts = []
        
        # Contexte système
        if context:
            prompt_parts.append(f"Contexte: {context}")
        
        # Préférences utilisateur
        if user_preferences:
            tone = user_preferences.get("preferred_tone", "neutral")
            language = user_preferences.get("preferred_language", "fr")
            
            if tone == "casual":
                prompt_parts.append("Réponds de manière décontractée et amicale.")
            elif tone == "formal":
                prompt_parts.append("Réponds de manière formelle et professionnelle.")
            
            if language == "fr":
                prompt_parts.append("Réponds en français.")
            elif language == "en":
                prompt_parts.append("Respond in English.")
        
        # Contenu principal
        prompt_parts.append(f"Requête: {content}")
        
        return "\n\n".join(prompt_parts)
    
    def _calculate_confidence(self, response_data: Dict[str, Any]) -> float:
        """Calcule un score de confiance pour la réponse"""
        # Implémentation basique - peut être surchargée
        
        content_length = len(response_data.get("content", ""))
        
        # Facteurs de confiance
        factors = []
        
        # Longueur de réponse (réponses trop courtes = moins fiables)
        if content_length > 100:
            factors.append(0.8)
        elif content_length > 50:
            factors.append(0.6)
        else:
            factors.append(0.3)
        
        # Présence d'erreurs
        if response_data.get("error"):
            factors.append(0.1)
        else:
            factors.append(0.9)
        
        # Temps de réponse (très rapide ou très lent = suspect)
        response_time = response_data.get("response_time", 0)
        if 1 <= response_time <= 30:
            factors.append(0.8)
        else:
            factors.append(0.5)
        
        # Moyenne pondérée
        return sum(factors) / len(factors) if factors else 0.5
    
    def update_statistics(self, success: bool, response_time: float):
        """Met à jour les statistiques de l'agent"""
        self.total_requests += 1
        self.total_response_time += response_time
        
        if success:
            self.successful_requests += 1
        else:
            self.failed_requests += 1
    
    def get_statistics(self) -> Dict[str, Any]:
        """Retourne les statistiques de l'agent"""
        avg_response_time = (
            self.total_response_time / self.total_requests 
            if self.total_requests > 0 else 0
        )
        
        success_rate = (
            self.successful_requests / self.total_requests 
            if self.total_requests > 0 else 0
        )
        
        return {
            "agent_id": self.agent_id,
            "status": self.status.value,
            "model_name": self.model_name,
            "capabilities": self.capabilities,
            "statistics": {
                "total_requests": self.total_requests,
                "successful_requests": self.successful_requests,
                "failed_requests": self.failed_requests,
                "success_rate": success_rate,
                "average_response_time": avg_response_time
            }
        }

class MockAgent(BaseAgent):
    """Agent mock pour tests et développement"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.agent_id = "mock"
        self.capabilities = ["general", "test"]
        self.model_name = "mock-model-v1"
    
    async def initialize(self) -> bool:
        """Initialise l'agent mock"""
        self.status = AgentStatus.READY
        return True
    
    async def process_request(
        self, 
        content: str, 
        context: Optional[str] = None,
        user_preferences: Optional[Dict[str, Any]] = None
    ) -> AgentResponse:
        """Simule le traitement d'une requête"""
        
        start_time = time.time()
        
        # Simuler du traitement
        import asyncio
        await asyncio.sleep(0.5)  # 500ms de traitement simulé
        
        response_time = time.time() - start_time
        
        # Générer une réponse mock
        mock_content = f"Réponse mock pour: '{content[:50]}...'"
        
        if "erreur" in content.lower():
            # Simuler une erreur
            response = AgentResponse(
                agent_id=self.agent_id,
                content="",
                confidence=0.0,
                response_time=response_time,
                model_used=self.model_name,
                error="Erreur simulée"
            )
            self.update_statistics(False, response_time)
        else:
            # Réponse normale
            response = AgentResponse(
                agent_id=self.agent_id,
                content=mock_content,
                confidence=0.75,
                response_time=response_time,
                model_used=self.model_name,
                tokens_used=len(mock_content.split()) * 2,
                temperature=self.temperature
            )
            self.update_statistics(True, response_time)
        
        return response
    
    async def health_check(self) -> Dict[str, Any]:
        """Health check pour l'agent mock"""
        return {
            "status": "healthy",
            "agent_id": self.agent_id,
            "model": self.model_name,
            "latency_ms": 100,
            "last_check": datetime.now(timezone.utc).isoformat()
        }