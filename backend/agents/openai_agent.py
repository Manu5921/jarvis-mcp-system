"""
OpenAI Agent for Jarvis MCP
Integration with OpenAI's GPT models
"""

import os
import time
import logging
from typing import Dict, List, Optional, Any

import openai

from .base import BaseAgent, AgentResponse, AgentStatus

logger = logging.getLogger(__name__)

class OpenAIAgent(BaseAgent):
    """Agent OpenAI pour créativité et tâches générales"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.agent_id = "openai"
        self.model_name = config.get("model", "gpt-4-turbo-preview")
        self.capabilities = config.get("capabilities", ["creative", "general", "coding"])
        
        # Configuration API
        api_key_env = config.get("api_key_env", "OPENAI_API_KEY")
        self.api_key = os.getenv(api_key_env)
        
        # Client OpenAI
        self.client = None
        
    async def initialize(self) -> bool:
        """Initialise le client OpenAI"""
        try:
            if not self.api_key:
                logger.error("Clé API OpenAI manquante")
                self.status = AgentStatus.ERROR
                return False
            
            # Initialiser le client OpenAI
            self.client = openai.AsyncOpenAI(api_key=self.api_key)
            
            # Test de connexion
            test_response = await self.client.chat.completions.create(
                model=self.model_name,
                max_tokens=5,
                messages=[{"role": "user", "content": "Test"}]
            )
            
            if test_response.choices:
                self.status = AgentStatus.READY
                logger.info(f"OpenAI agent initialisé avec modèle {self.model_name}")
                return True
            else:
                logger.error("Test OpenAI échoué")
                self.status = AgentStatus.ERROR
                return False
                
        except Exception as e:
            logger.error(f"Erreur initialisation OpenAI: {e}")
            self.status = AgentStatus.ERROR
            return False
    
    async def process_request(
        self, 
        content: str, 
        context: Optional[str] = None,
        user_preferences: Optional[Dict[str, Any]] = None
    ) -> AgentResponse:
        """Traite une requête avec OpenAI"""
        
        if self.status != AgentStatus.READY or not self.client:
            return AgentResponse(
                agent_id=self.agent_id,
                content="",
                confidence=0.0,
                response_time=0.0,
                error="Agent OpenAI non disponible"
            )
        
        start_time = time.time()
        self.status = AgentStatus.BUSY
        
        try:
            # Construire les messages pour OpenAI
            messages = self._build_messages(content, context, user_preferences)
            
            # Appel à l'API OpenAI
            response = await self.client.chat.completions.create(
                model=self.model_name,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                messages=messages,
                stream=False
            )
            
            response_time = time.time() - start_time
            
            # Extraire le contenu
            content_response = ""
            if response.choices and len(response.choices) > 0:
                content_response = response.choices[0].message.content or ""
            
            # Calculer les métriques
            prompt_tokens = response.usage.prompt_tokens if response.usage else 0
            completion_tokens = response.usage.completion_tokens if response.usage else 0
            total_tokens = response.usage.total_tokens if response.usage else 0
            
            confidence = self._calculate_confidence({
                "content": content_response,
                "response_time": response_time,
                "tokens": total_tokens,
                "finish_reason": response.choices[0].finish_reason if response.choices else None
            })
            
            # Créer la réponse
            agent_response = AgentResponse(
                agent_id=self.agent_id,
                content=content_response,
                confidence=confidence,
                response_time=response_time,
                model_used=self.model_name,
                tokens_used=total_tokens,
                temperature=self.temperature,
                metadata={
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "finish_reason": response.choices[0].finish_reason if response.choices else None,
                    "model": response.model,
                    "system_fingerprint": response.system_fingerprint
                }
            )
            
            self.update_statistics(True, response_time)
            logger.info(f"OpenAI requête traitée en {response_time:.2f}s, {total_tokens} tokens")
            
        except openai.APIError as e:
            response_time = time.time() - start_time
            error_msg = f"Erreur API OpenAI: {e}"
            agent_response = AgentResponse(
                agent_id=self.agent_id,
                content="",
                confidence=0.0,
                response_time=response_time,
                error=error_msg
            )
            self.update_statistics(False, response_time)
            logger.error(error_msg)
            
        except openai.RateLimitError as e:
            response_time = time.time() - start_time
            error_msg = f"Rate limit OpenAI dépassé: {e}"
            agent_response = AgentResponse(
                agent_id=self.agent_id,
                content="",
                confidence=0.0,
                response_time=response_time,
                error=error_msg
            )
            self.update_statistics(False, response_time)
            logger.warning(error_msg)
            
        except Exception as e:
            response_time = time.time() - start_time
            error_msg = f"Erreur OpenAI: {str(e)}"
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
    
    def _build_messages(
        self, 
        content: str, 
        context: Optional[str] = None,
        user_preferences: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, str]]:
        """Construit les messages pour l'API OpenAI"""
        
        messages = []
        
        # Message système
        system_prompt = self._build_system_prompt(user_preferences)
        messages.append({
            "role": "system",
            "content": system_prompt
        })
        
        # Ajouter le contexte si disponible
        if context:
            messages.append({
                "role": "system", 
                "content": f"Contexte disponible: {context}"
            })
        
        # Message utilisateur
        messages.append({
            "role": "user",
            "content": content
        })
        
        return messages
    
    def _build_system_prompt(self, user_preferences: Optional[Dict[str, Any]] = None) -> str:
        """Construit le prompt système pour OpenAI"""
        
        base_prompt = """Tu es un agent IA intégré dans le système Jarvis MCP, spécialisé dans les tâches créatives et générales. Tes domaines d'expertise sont:

🎨 **Créativité & Contenu**:
- Rédaction créative (articles, histoires, poésies)
- Brainstorming et génération d'idées
- Adaptation de contenu pour différents publics
- Création de concepts et scénarios

💬 **Communication & Langues**:
- Rédaction professionnelle et marketing
- Traduction et localisation
- Reformulation et synthèse
- Communication interculturelle

🧠 **Tâches Générales**:
- Résolution de problèmes créatifs
- Conseil et recommandations
- Planification et organisation
- Support conversationnel

Instructions importantes:
- Sois créatif et original dans tes réponses
- Adapte le style selon le contexte et l'audience
- Propose plusieurs alternatives quand pertinent
- Utilise des exemples vivants et engageants
- Maintiens un équilibre entre créativité et utilité"""
        
        if user_preferences:
            tone = user_preferences.get("preferred_tone", "neutral")
            language = user_preferences.get("preferred_language", "fr")
            
            if tone == "casual":
                base_prompt += "\n- Adopte un style décontracté, personnel et amical"
            elif tone == "formal":
                base_prompt += "\n- Maintiens un style formel et professionnel"
            
            if language == "en":
                base_prompt = """You are an AI agent integrated into the Jarvis MCP system, specialized in creative and general tasks. Your areas of expertise are:

🎨 **Creativity & Content**:
- Creative writing (articles, stories, poetry)
- Brainstorming and idea generation
- Content adaptation for different audiences
- Concept and scenario creation

💬 **Communication & Languages**:
- Professional and marketing writing
- Translation and localization
- Reformulation and synthesis
- Cross-cultural communication

🧠 **General Tasks**:
- Creative problem solving
- Advice and recommendations
- Planning and organization
- Conversational support

Important instructions:
- Be creative and original in your responses
- Adapt style according to context and audience
- Propose multiple alternatives when relevant
- Use vivid and engaging examples
- Maintain balance between creativity and utility"""
        
        return base_prompt
    
    def _calculate_confidence(self, response_data: Dict[str, Any]) -> float:
        """Calcule la confiance spécifique à OpenAI"""
        base_confidence = super()._calculate_confidence(response_data)
        
        content = response_data.get("content", "")
        finish_reason = response_data.get("finish_reason")
        
        # Réponses complètes sont plus fiables
        if finish_reason == "stop":
            base_confidence = min(base_confidence + 0.1, 1.0)
        elif finish_reason == "length":
            base_confidence *= 0.8  # Tronquée
        
        # Contenu créatif avec structure
        if any(marker in content for marker in ["1.", "2.", "3.", "-", "•", "**"]):
            base_confidence = min(base_confidence + 0.05, 1.0)
        
        # Réponses très créatives (questions, variations)
        creative_indicators = ["imagine", "créons", "exemple", "idée", "pourrait être"]
        if any(indicator in content.lower() for indicator in creative_indicators):
            base_confidence = min(base_confidence + 0.1, 1.0)
        
        # Longueur optimale pour le contenu créatif
        if 200 <= len(content) <= 2000:
            base_confidence = min(base_confidence + 0.05, 1.0)
        
        return round(base_confidence, 3)
    
    async def health_check(self) -> Dict[str, Any]:
        """Vérifie la santé de l'API OpenAI"""
        try:
            if not self.client:
                return {
                    "status": "unhealthy",
                    "agent_id": self.agent_id,
                    "error": "Client non initialisé"
                }
            
            start_time = time.time()
            
            # Test rapide avec OpenAI
            response = await self.client.chat.completions.create(
                model=self.model_name,
                max_tokens=5,
                messages=[{"role": "user", "content": "Ping"}]
            )
            
            latency = (time.time() - start_time) * 1000  # en ms
            
            return {
                "status": "healthy",
                "agent_id": self.agent_id,
                "model": self.model_name,
                "latency_ms": round(latency, 2),
                "last_check": time.time(),
                "api_version": "v1"
            }
            
        except openai.APIError as e:
            return {
                "status": "unhealthy",
                "agent_id": self.agent_id,
                "error": f"API Error: {e}",
                "error_type": "api_error"
            }
        except openai.RateLimitError as e:
            return {
                "status": "rate_limited",
                "agent_id": self.agent_id,
                "error": f"Rate limit: {e}",
                "error_type": "rate_limit"
            }
        except Exception as e:
            return {
                "status": "unhealthy",
                "agent_id": self.agent_id,
                "error": str(e),
                "error_type": "unknown"
            }
    
    async def cleanup(self):
        """Nettoie les ressources OpenAI"""
        if self.client:
            await self.client.close()
        await super().cleanup()
        logger.info("OpenAI agent nettoyé")
    
    def get_supported_models(self) -> List[str]:
        """Retourne la liste des modèles OpenAI supportés"""
        return [
            "gpt-4-turbo-preview",
            "gpt-4-1106-preview",
            "gpt-4",
            "gpt-3.5-turbo",
            "gpt-3.5-turbo-16k"
        ]
    
    async def get_models(self) -> List[Dict[str, Any]]:
        """Récupère la liste des modèles disponibles depuis l'API"""
        try:
            if not self.client:
                return []
                
            models = await self.client.models.list()
            return [
                {
                    "id": model.id,
                    "created": model.created,
                    "owned_by": model.owned_by
                }
                for model in models.data
                if "gpt" in model.id
            ]
        except Exception as e:
            logger.error(f"Erreur récupération modèles OpenAI: {e}")
            return []