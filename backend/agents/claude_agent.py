"""
Claude Agent for Jarvis MCP
Integration with Anthropic's Claude API
"""

import os
import time
import logging
from typing import Dict, List, Optional, Any

import anthropic

from .base import BaseAgent, AgentResponse, AgentStatus

logger = logging.getLogger(__name__)

class ClaudeAgent(BaseAgent):
    """Agent Claude pour expertise en code et analyse"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.agent_id = "claude"
        self.model_name = config.get("model", "claude-3-sonnet-20240229")
        self.capabilities = config.get("capabilities", ["code", "analysis", "writing"])
        
        # Configuration API
        api_key_env = config.get("api_key_env", "ANTHROPIC_API_KEY")
        self.api_key = os.getenv(api_key_env)
        
        # Client Anthropic
        self.client = None
        
    async def initialize(self) -> bool:
        """Initialise le client Claude"""
        try:
            if not self.api_key:
                logger.error("Clé API Anthropic manquante")
                self.status = AgentStatus.ERROR
                return False
            
            # Initialiser le client Anthropic
            self.client = anthropic.AsyncAnthropic(api_key=self.api_key)
            
            # Test de connexion
            test_response = await self.client.messages.create(
                model=self.model_name,
                max_tokens=10,
                messages=[{"role": "user", "content": "Test"}]
            )
            
            if test_response.content:
                self.status = AgentStatus.READY
                logger.info(f"Claude agent initialisé avec modèle {self.model_name}")
                return True
            else:
                logger.error("Test Claude échoué")
                self.status = AgentStatus.ERROR
                return False
                
        except Exception as e:
            logger.error(f"Erreur initialisation Claude: {e}")
            self.status = AgentStatus.ERROR
            return False
    
    async def process_request(
        self, 
        content: str, 
        context: Optional[str] = None,
        user_preferences: Optional[Dict[str, Any]] = None
    ) -> AgentResponse:
        """Traite une requête avec Claude"""
        
        if self.status != AgentStatus.READY or not self.client:
            return AgentResponse(
                agent_id=self.agent_id,
                content="",
                confidence=0.0,
                response_time=0.0,
                error="Agent Claude non disponible"
            )
        
        start_time = time.time()
        self.status = AgentStatus.BUSY
        
        try:
            # Construire les messages pour Claude
            messages = self._build_messages(content, context, user_preferences)
            
            # Appel à l'API Claude
            response = await self.client.messages.create(
                model=self.model_name,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                messages=messages
            )
            
            response_time = time.time() - start_time
            
            # Extraire le contenu
            content_response = ""
            if response.content and len(response.content) > 0:
                content_response = response.content[0].text if hasattr(response.content[0], 'text') else str(response.content[0])
            
            # Calculer les métriques
            input_tokens = response.usage.input_tokens if response.usage else 0
            output_tokens = response.usage.output_tokens if response.usage else 0
            total_tokens = input_tokens + output_tokens
            
            confidence = self._calculate_confidence({
                "content": content_response,
                "response_time": response_time,
                "tokens": total_tokens,
                "model": self.model_name
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
                    "input_tokens": input_tokens,
                    "output_tokens": output_tokens,
                    "stop_reason": response.stop_reason,
                    "model": response.model
                }
            )
            
            self.update_statistics(True, response_time)
            logger.info(f"Claude requête traitée en {response_time:.2f}s, {total_tokens} tokens")
            
        except anthropic.APIError as e:
            response_time = time.time() - start_time
            error_msg = f"Erreur API Claude: {e}"
            agent_response = AgentResponse(
                agent_id=self.agent_id,
                content="",
                confidence=0.0,
                response_time=response_time,
                error=error_msg
            )
            self.update_statistics(False, response_time)
            logger.error(error_msg)
            
        except anthropic.RateLimitError as e:
            response_time = time.time() - start_time
            error_msg = f"Rate limit Claude dépassé: {e}"
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
            error_msg = f"Erreur Claude: {str(e)}"
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
        """Construit les messages pour l'API Claude"""
        
        messages = []
        
        # Message système via le premier message utilisateur (Claude n'a pas de system role)
        system_prompt = self._build_system_prompt(user_preferences)
        
        if context:
            system_prompt += f"\n\nContexte disponible:\n{context}"
        
        # Combiner le prompt système avec la requête utilisateur
        full_content = f"{system_prompt}\n\nRequête utilisateur: {content}"
        
        messages.append({
            "role": "user",
            "content": full_content
        })
        
        return messages
    
    def _build_system_prompt(self, user_preferences: Optional[Dict[str, Any]] = None) -> str:
        """Construit le prompt système pour Claude"""
        
        base_prompt = """Tu es Claude, intégré dans le système Jarvis MCP en tant qu'agent spécialisé. Tes domaines d'expertise sont:

🔧 **Programmation & Code**:
- Analyse, débogage et optimisation de code
- Architecture logicielle et bonnes pratiques
- Revue de code et suggestions d'amélioration
- Génération de code robuste et documenté

📊 **Analyse & Recherche**:
- Analyse de données et insights
- Synthèse de documents complexes
- Comparaisons techniques détaillées
- Évaluation de solutions

✍️ **Rédaction Technique**:
- Documentation technique claire
- Spécifications et cahiers des charges
- Rapports d'analyse structurés

Instructions importantes:
- Fournis des réponses précises et actionnables
- Utilise des exemples concrets quand approprié
- Structure tes réponses avec des sections claires
- Pour le code, inclus des commentaires explicatifs
- Indique clairement tes limites ou incertitudes"""
        
        if user_preferences:
            tone = user_preferences.get("preferred_tone", "neutral")
            language = user_preferences.get("preferred_language", "fr")
            
            if tone == "casual":
                base_prompt += "\n- Adopte un style accessible et décontracté"
            elif tone == "formal":
                base_prompt += "\n- Maintiens un style professionnel et précis"
            
            if language == "en":
                base_prompt = """You are Claude, integrated into the Jarvis MCP system as a specialized agent. Your areas of expertise are:

🔧 **Programming & Code**:
- Code analysis, debugging and optimization
- Software architecture and best practices
- Code review and improvement suggestions
- Robust and documented code generation

📊 **Analysis & Research**:
- Data analysis and insights
- Complex document synthesis
- Detailed technical comparisons
- Solution evaluation

✍️ **Technical Writing**:
- Clear technical documentation
- Specifications and requirements
- Structured analysis reports

Important instructions:
- Provide precise and actionable responses
- Use concrete examples when appropriate
- Structure your responses with clear sections
- For code, include explanatory comments
- Clearly indicate your limitations or uncertainties"""
        
        return base_prompt
    
    def _calculate_confidence(self, response_data: Dict[str, Any]) -> float:
        """Calcule la confiance spécifique à Claude"""
        base_confidence = super()._calculate_confidence(response_data)
        
        content = response_data.get("content", "")
        model = response_data.get("model", "")
        
        # Claude est généralement plus fiable pour le code et l'analyse
        if any(keyword in content.lower() for keyword in ["```", "function", "class", "import", "def"]):
            base_confidence = min(base_confidence + 0.15, 1.0)
        
        # Les modèles plus récents sont plus fiables
        if "claude-3" in model:
            base_confidence = min(base_confidence + 0.1, 1.0)
        
        # Réponses structurées avec sections
        if any(marker in content for marker in ["##", "**", "1.", "2.", "3."]):
            base_confidence = min(base_confidence + 0.05, 1.0)
        
        # Réponses avec disclaimers (signe de prudence)
        if any(phrase in content.lower() for phrase in ["je ne suis pas sûr", "il se pourrait", "probablement"]):
            base_confidence *= 0.9
        
        return round(base_confidence, 3)
    
    async def health_check(self) -> Dict[str, Any]:
        """Vérifie la santé de l'API Claude"""
        try:
            if not self.client:
                return {
                    "status": "unhealthy",
                    "agent_id": self.agent_id,
                    "error": "Client non initialisé"
                }
            
            start_time = time.time()
            
            # Test rapide avec Claude
            response = await self.client.messages.create(
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
                "api_version": "2023-06-01"
            }
            
        except anthropic.APIError as e:
            return {
                "status": "unhealthy",
                "agent_id": self.agent_id,
                "error": f"API Error: {e}",
                "error_type": "api_error"
            }
        except anthropic.RateLimitError as e:
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
        """Nettoie les ressources Claude"""
        if self.client:
            await self.client.close()
        await super().cleanup()
        logger.info("Claude agent nettoyé")
    
    def get_supported_models(self) -> List[str]:
        """Retourne la liste des modèles Claude supportés"""
        return [
            "claude-3-opus-20240229",
            "claude-3-sonnet-20240229", 
            "claude-3-haiku-20240307",
            "claude-3-5-sonnet-20241022"
        ]