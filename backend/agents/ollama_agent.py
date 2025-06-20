"""
Ollama Agent for Jarvis MCP
Local AI agent using Ollama API
"""

import time
import httpx
import logging
from typing import Dict, List, Optional, Any

from .base import BaseAgent, AgentResponse, AgentStatus

logger = logging.getLogger(__name__)

class OllamaAgent(BaseAgent):
    """Agent local Ollama pour traitement IA privé"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.agent_id = "ollama"
        self.base_url = config.get("base_url", "http://localhost:11434")
        self.model_name = config.get("model", "llama3.2:8b")
        self.timeout = config.get("timeout", 60)
        self.capabilities = config.get("capabilities", ["general", "code", "analysis"])
        
        # Client HTTP pour Ollama
        self.client = httpx.AsyncClient(
            base_url=self.base_url,
            timeout=httpx.Timeout(self.timeout)
        )
        
    async def initialize(self) -> bool:
        """Initialise la connexion Ollama"""
        try:
            # Vérifier que Ollama est accessible
            response = await self.client.get("/api/tags")
            
            if response.status_code == 200:
                models = response.json().get("models", [])
                model_names = [model["name"] for model in models]
                
                if self.model_name in model_names:
                    self.status = AgentStatus.READY
                    logger.info(f"Ollama agent initialisé avec modèle {self.model_name}")
                    return True
                else:
                    logger.error(f"Modèle {self.model_name} non trouvé dans Ollama")
                    self.status = AgentStatus.ERROR
                    return False
            else:
                logger.error(f"Ollama non accessible: {response.status_code}")
                self.status = AgentStatus.ERROR
                return False
                
        except Exception as e:
            logger.error(f"Erreur initialisation Ollama: {e}")
            self.status = AgentStatus.ERROR
            return False
    
    async def process_request(
        self, 
        content: str, 
        context: Optional[str] = None,
        user_preferences: Optional[Dict[str, Any]] = None
    ) -> AgentResponse:
        """Traite une requête avec Ollama"""
        
        if self.status != AgentStatus.READY:
            return AgentResponse(
                agent_id=self.agent_id,
                content="",
                confidence=0.0,
                response_time=0.0,
                error="Agent Ollama non disponible"
            )
        
        start_time = time.time()
        self.status = AgentStatus.BUSY
        
        try:
            # Formater le prompt
            formatted_prompt = self._format_prompt(content, context, user_preferences)
            
            # Enrichir avec des instructions spécifiques Ollama
            system_prompt = self._build_system_prompt(user_preferences)
            full_prompt = f"{system_prompt}\n\n{formatted_prompt}"
            
            # Préparer la requête Ollama
            request_data = {
                "model": self.model_name,
                "prompt": full_prompt,
                "stream": False,
                "options": {
                    "temperature": self.temperature,
                    "num_predict": self.max_tokens,
                    "top_p": 0.9,
                    "repeat_penalty": 1.1
                }
            }
            
            # Appel à l'API Ollama
            response = await self.client.post("/api/generate", json=request_data)
            response_time = time.time() - start_time
            
            if response.status_code == 200:
                result = response.json()
                content_response = result.get("response", "").strip()
                
                # Calculer les métriques
                tokens_used = self._estimate_tokens(full_prompt + content_response)
                confidence = self._calculate_confidence({
                    "content": content_response,
                    "response_time": response_time,
                    "tokens": tokens_used
                })
                
                # Créer la réponse
                agent_response = AgentResponse(
                    agent_id=self.agent_id,
                    content=content_response,
                    confidence=confidence,
                    response_time=response_time,
                    model_used=self.model_name,
                    tokens_used=tokens_used,
                    temperature=self.temperature,
                    metadata={
                        "eval_count": result.get("eval_count"),
                        "eval_duration": result.get("eval_duration"),
                        "prompt_eval_count": result.get("prompt_eval_count"),
                        "total_duration": result.get("total_duration")
                    }
                )
                
                self.update_statistics(True, response_time)
                logger.info(f"Ollama requête traitée en {response_time:.2f}s")
                
            else:
                # Erreur HTTP
                error_msg = f"Erreur Ollama HTTP {response.status_code}: {response.text}"
                agent_response = AgentResponse(
                    agent_id=self.agent_id,
                    content="",
                    confidence=0.0,
                    response_time=response_time,
                    error=error_msg
                )
                
                self.update_statistics(False, response_time)
                logger.error(error_msg)
                
        except httpx.TimeoutException:
            response_time = time.time() - start_time
            error_msg = f"Timeout Ollama après {self.timeout}s"
            agent_response = AgentResponse(
                agent_id=self.agent_id,
                content="",
                confidence=0.0,
                response_time=response_time,
                error=error_msg
            )
            self.update_statistics(False, response_time)
            logger.error(error_msg)
            
        except Exception as e:
            response_time = time.time() - start_time
            error_msg = f"Erreur Ollama: {str(e)}"
            agent_response = AgentResponse(
                agent_id=self.agent_id,
                content="",
                confidence=0.0,
                response_time=response_time,
                error=error_msg
            )
            self.update_statistics(False, response_time)
            logger.error(error_msg)
        
        finally:
            self.status = AgentStatus.READY
        
        return agent_response
    
    def _build_system_prompt(self, user_preferences: Optional[Dict[str, Any]] = None) -> str:
        """Construit le prompt système pour Ollama"""
        
        base_prompt = """Tu es Jarvis, un assistant IA intelligent et polyvalent. Tu es intégré dans un système MCP (Multi-Channel Processor) qui coordonne plusieurs agents IA.

Tes capacités principales:
- Répondre à des questions générales
- Analyser et expliquer du code
- Aider avec la programmation
- Effectuer des analyses et synthèses

Instructions importantes:
- Sois précis et structuré dans tes réponses
- Si tu n'es pas sûr, indique-le clairement
- Adapte ton ton selon les préférences utilisateur
- Pour le code, utilise des exemples concrets"""
        
        if user_preferences:
            tone = user_preferences.get("preferred_tone", "neutral")
            language = user_preferences.get("preferred_language", "fr")
            
            if tone == "casual":
                base_prompt += "\n- Utilise un ton décontracté et amical"
            elif tone == "formal":
                base_prompt += "\n- Utilise un ton formel et professionnel"
            
            if language == "en":
                base_prompt = base_prompt.replace("Tu es", "You are").replace("Tes capacités", "Your capabilities")
        
        return base_prompt
    
    def _estimate_tokens(self, text: str) -> int:
        """Estime le nombre de tokens approximatif"""
        # Estimation basique : ~4 caractères par token pour le français
        return len(text) // 4
    
    def _calculate_confidence(self, response_data: Dict[str, Any]) -> float:
        """Calcule la confiance spécifique à Ollama"""
        base_confidence = super()._calculate_confidence(response_data)
        
        # Facteurs spécifiques à Ollama
        content = response_data.get("content", "")
        
        # Réponses avec du code sont généralement plus fiables
        if "```" in content or "def " in content or "function" in content:
            base_confidence = min(base_confidence + 0.1, 1.0)
        
        # Réponses très courtes moins fiables
        if len(content) < 20:
            base_confidence *= 0.5
        
        # Réponses qui semblent incomplètes
        if content.endswith("...") or "je ne peux pas" in content.lower():
            base_confidence *= 0.7
        
        return round(base_confidence, 3)
    
    async def health_check(self) -> Dict[str, Any]:
        """Vérifie la santé d'Ollama"""
        try:
            start_time = time.time()
            
            # Test simple de génération
            test_data = {
                "model": self.model_name,
                "prompt": "Test de santé",
                "stream": False,
                "options": {"num_predict": 10}
            }
            
            response = await self.client.post("/api/generate", json=test_data)
            latency = (time.time() - start_time) * 1000  # en ms
            
            if response.status_code == 200:
                # Vérifier les modèles disponibles
                models_response = await self.client.get("/api/tags")
                models = models_response.json().get("models", []) if models_response.status_code == 200 else []
                
                return {
                    "status": "healthy",
                    "agent_id": self.agent_id,
                    "model": self.model_name,
                    "latency_ms": round(latency, 2),
                    "models_available": len(models),
                    "last_check": time.time(),
                    "base_url": self.base_url
                }
            else:
                return {
                    "status": "unhealthy",
                    "agent_id": self.agent_id,
                    "error": f"HTTP {response.status_code}",
                    "latency_ms": round(latency, 2)
                }
                
        except Exception as e:
            return {
                "status": "unhealthy",
                "agent_id": self.agent_id,
                "error": str(e),
                "base_url": self.base_url
            }
    
    async def cleanup(self):
        """Nettoie les ressources Ollama"""
        await self.client.aclose()
        await super().cleanup()
        logger.info("Ollama agent nettoyé")
    
    async def get_available_models(self) -> List[str]:
        """Retourne la liste des modèles Ollama disponibles"""
        try:
            response = await self.client.get("/api/tags")
            if response.status_code == 200:
                models = response.json().get("models", [])
                return [model["name"] for model in models]
            return []
        except Exception as e:
            logger.error(f"Erreur récupération modèles Ollama: {e}")
            return []
    
    async def pull_model(self, model_name: str) -> bool:
        """Télécharge un modèle Ollama"""
        try:
            response = await self.client.post("/api/pull", json={"name": model_name})
            return response.status_code == 200
        except Exception as e:
            logger.error(f"Erreur téléchargement modèle {model_name}: {e}")
            return False