"""
🛡️ COUCHE DE RÉSILIENCE MCP 
Système avancé de récupération, fallbacks et stratégies adaptatives
"""

import asyncio
import logging
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
from enum import Enum
import json

logger = logging.getLogger(__name__)

class ErrorType(Enum):
    ENCODING_ERROR = "encoding_error"
    TIMEOUT_ERROR = "timeout_error"
    ACCESS_DENIED = "access_denied"
    FILE_NOT_FOUND = "file_not_found"
    NETWORK_ERROR = "network_error"
    VALIDATION_FAILED = "validation_failed"
    UNKNOWN_ERROR = "unknown_error"

class RecoveryStrategy(Enum):
    RETRY_WITH_FALLBACK = "retry_with_fallback"
    ALTERNATIVE_TOOLS = "alternative_tools"
    PARTIAL_ANALYSIS = "partial_analysis"
    CONTEXT_INFERENCE = "context_inference"
    GRACEFUL_DEGRADATION = "graceful_degradation"

class ResilienceMetrics:
    """Métriques de performance du système de résilience"""
    
    def __init__(self):
        self.recovery_attempts = 0
        self.successful_recoveries = 0
        self.failed_recoveries = 0
        self.error_patterns = {}
        self.strategy_effectiveness = {}
        self.performance_history = []
    
    def record_recovery_attempt(self, error_type: ErrorType, strategy: RecoveryStrategy, success: bool):
        """Enregistre une tentative de récupération"""
        self.recovery_attempts += 1
        
        if success:
            self.successful_recoveries += 1
        else:
            self.failed_recoveries += 1
        
        # Patterns d'erreurs
        error_key = error_type.value
        if error_key not in self.error_patterns:
            self.error_patterns[error_key] = {"count": 0, "recoveries": 0}
        
        self.error_patterns[error_key]["count"] += 1
        if success:
            self.error_patterns[error_key]["recoveries"] += 1
        
        # Efficacité des stratégies
        strategy_key = strategy.value
        if strategy_key not in self.strategy_effectiveness:
            self.strategy_effectiveness[strategy_key] = {"attempts": 0, "successes": 0}
        
        self.strategy_effectiveness[strategy_key]["attempts"] += 1
        if success:
            self.strategy_effectiveness[strategy_key]["successes"] += 1
    
    def get_success_rate(self) -> float:
        """Taux de succès global"""
        if self.recovery_attempts == 0:
            return 1.0
        return self.successful_recoveries / self.recovery_attempts
    
    def get_best_strategy_for_error(self, error_type: ErrorType) -> Optional[RecoveryStrategy]:
        """Retourne la meilleure stratégie pour un type d'erreur"""
        # Logique simple pour MVP - peut être améliorée avec ML
        if error_type == ErrorType.ENCODING_ERROR:
            return RecoveryStrategy.RETRY_WITH_FALLBACK
        elif error_type == ErrorType.TIMEOUT_ERROR:
            return RecoveryStrategy.ALTERNATIVE_TOOLS
        elif error_type == ErrorType.ACCESS_DENIED:
            return RecoveryStrategy.CONTEXT_INFERENCE
        elif error_type == ErrorType.FILE_NOT_FOUND:
            return RecoveryStrategy.ALTERNATIVE_TOOLS
        else:
            return RecoveryStrategy.GRACEFUL_DEGRADATION

class MCPResilienceLayer:
    """Couche de résilience principale pour le système MCP"""
    
    def __init__(self):
        self.metrics = ResilienceMetrics()
        self.error_cache = {}  # Cache des erreurs récentes
        self.recovery_strategies = self._init_recovery_strategies()
    
    def _init_recovery_strategies(self) -> Dict[RecoveryStrategy, callable]:
        """Initialise les stratégies de récupération"""
        return {
            RecoveryStrategy.RETRY_WITH_FALLBACK: self._strategy_retry_with_fallback,
            RecoveryStrategy.ALTERNATIVE_TOOLS: self._strategy_alternative_tools,
            RecoveryStrategy.PARTIAL_ANALYSIS: self._strategy_partial_analysis,
            RecoveryStrategy.CONTEXT_INFERENCE: self._strategy_context_inference,
            RecoveryStrategy.GRACEFUL_DEGRADATION: self._strategy_graceful_degradation
        }
    
    async def handle_tool_failure(self, 
                                 tool_name: str, 
                                 tool_args: Dict[str, Any], 
                                 error: Exception,
                                 context: Dict[str, Any],
                                 mcp_clients: Dict[str, Any]) -> Dict[str, Any]:
        """
        Point d'entrée principal pour gérer les échecs d'outils
        """
        logger.warning(f"🛡️ RÉSILIENCE: Gestion échec {tool_name} - {str(error)[:100]}")
        
        # 1. Classification de l'erreur
        error_type = self._classify_error(error, tool_name, tool_args)
        
        # 2. Sélection de la stratégie
        strategy = self.metrics.get_best_strategy_for_error(error_type)
        
        # 3. Exécution de la récupération
        recovery_result = await self._execute_recovery_strategy(
            strategy, tool_name, tool_args, error, context, mcp_clients
        )
        
        # 4. Enregistrement des métriques
        self.metrics.record_recovery_attempt(error_type, strategy, recovery_result['success'])
        
        return recovery_result
    
    def _classify_error(self, error: Exception, tool_name: str, tool_args: Dict[str, Any]) -> ErrorType:
        """Classifie le type d'erreur pour choisir la stratégie appropriée"""
        error_str = str(error).lower()
        
        if "encodage" in error_str or "decode" in error_str or "unicode" in error_str:
            return ErrorType.ENCODING_ERROR
        elif "timeout" in error_str or "timed out" in error_str:
            return ErrorType.TIMEOUT_ERROR
        elif "access" in error_str or "permission" in error_str:
            return ErrorType.ACCESS_DENIED
        elif "not found" in error_str or "no such file" in error_str:
            return ErrorType.FILE_NOT_FOUND
        elif "connection" in error_str or "network" in error_str:
            return ErrorType.NETWORK_ERROR
        else:
            return ErrorType.UNKNOWN_ERROR
    
    async def _execute_recovery_strategy(self, 
                                       strategy: RecoveryStrategy,
                                       tool_name: str,
                                       tool_args: Dict[str, Any],
                                       error: Exception,
                                       context: Dict[str, Any],
                                       mcp_clients: Dict[str, Any]) -> Dict[str, Any]:
        """Exécute la stratégie de récupération sélectionnée"""
        
        strategy_func = self.recovery_strategies.get(strategy)
        if not strategy_func:
            return {"success": False, "data": None, "message": f"Stratégie {strategy} non implémentée"}
        
        try:
            result = await strategy_func(tool_name, tool_args, error, context, mcp_clients)
            return result
        except Exception as strategy_error:
            logger.error(f"🛡️ RÉSILIENCE: Échec stratégie {strategy}: {strategy_error}")
            return {"success": False, "data": None, "message": f"Stratégie échouée: {strategy_error}"}
    
    async def _strategy_retry_with_fallback(self, tool_name: str, tool_args: Dict[str, Any], 
                                          error: Exception, context: Dict[str, Any], 
                                          mcp_clients: Dict[str, Any]) -> Dict[str, Any]:
        """Stratégie: Retry avec paramètres modifiés"""
        logger.info(f"🛡️ RÉSILIENCE: Retry avec fallback pour {tool_name}")
        
        if tool_name == "Read":
            # Pour Read, on a déjà implémenté le multi-encodage
            # Essayons avec des paramètres plus permissifs
            modified_args = tool_args.copy()
            modified_args["encoding"] = "latin-1"  # Encodage plus permissif
            
            try:
                result = await mcp_clients["ollama"].call_tool(tool_name, modified_args)
                if result and not result.get("is_error", False):
                    return {
                        "success": True, 
                        "data": result,
                        "message": "🛡️ Récupération réussie avec encodage fallback"
                    }
            except Exception as retry_error:
                logger.warning(f"🛡️ RÉSILIENCE: Retry échoué: {retry_error}")
        
        return {"success": False, "data": None, "message": "Retry avec fallback échoué"}
    
    async def _strategy_alternative_tools(self, tool_name: str, tool_args: Dict[str, Any], 
                                        error: Exception, context: Dict[str, Any], 
                                        mcp_clients: Dict[str, Any]) -> Dict[str, Any]:
        """Stratégie: Utiliser des outils alternatifs"""
        logger.info(f"🛡️ RÉSILIENCE: Outils alternatifs pour {tool_name}")
        
        if tool_name == "Read":
            # Si Read échoue, essayer LS du parent + inférence
            file_path = tool_args.get("file_path", "")
            parent_path = "/".join(file_path.split("/")[:-1])
            
            try:
                # LS du répertoire parent
                ls_result = await mcp_clients["ollama"].call_tool("LS", {"path": parent_path})
                if ls_result and not ls_result.get("is_error", False):
                    
                    # Essayer de lire des fichiers similaires
                    filename = file_path.split("/")[-1]
                    file_ext = filename.split(".")[-1] if "." in filename else ""
                    
                    # Inférer le contenu basé sur la structure
                    inference = self._infer_content_from_structure(ls_result, file_ext, filename)
                    
                    return {
                        "success": True,
                        "data": {"content": [{"type": "text", "text": inference}]},
                        "message": "🛡️ Récupération par inférence structurelle"
                    }
            except Exception as alt_error:
                logger.warning(f"🛡️ RÉSILIENCE: Outils alternatifs échoués: {alt_error}")
        
        elif tool_name == "LS":
            # Si LS échoue, essayer Glob sur le parent
            path = tool_args.get("path", "")
            parent_path = "/".join(path.split("/")[:-1])
            
            try:
                glob_result = await mcp_clients["ollama"].call_tool("Glob", {
                    "pattern": "*",
                    "path": parent_path
                })
                if glob_result and not glob_result.get("is_error", False):
                    return {
                        "success": True,
                        "data": glob_result,
                        "message": "🛡️ Récupération via Glob alternatif"
                    }
            except Exception as glob_error:
                logger.warning(f"🛡️ RÉSILIENCE: Glob alternatif échoué: {glob_error}")
        
        return {"success": False, "data": None, "message": "Aucun outil alternatif disponible"}
    
    async def _strategy_partial_analysis(self, tool_name: str, tool_args: Dict[str, Any], 
                                       error: Exception, context: Dict[str, Any], 
                                       mcp_clients: Dict[str, Any]) -> Dict[str, Any]:
        """Stratégie: Analyse partielle avec les données disponibles"""
        logger.info(f"🛡️ RÉSILIENCE: Analyse partielle malgré échec {tool_name}")
        
        # Rassembler toutes les données collectées jusqu'ici
        partial_data = context.get("collected_data", [])
        
        if partial_data:
            summary = "🛡️ ANALYSE PARTIELLE - Données disponibles:\n\n"
            for i, data in enumerate(partial_data, 1):
                summary += f"{i}. {data['tool']} : {data['summary']}\n"
            
            summary += f"\n⚠️ {tool_name} échoué: {str(error)[:100]}\n"
            summary += "\n🔍 RECOMMANDATION: Analyse basée sur les données partielles collectées"
            
            return {
                "success": True,
                "data": {"content": [{"type": "text", "text": summary}]},
                "message": "🛡️ Analyse partielle générée"
            }
        
        return {"success": False, "data": None, "message": "Pas de données partielles disponibles"}
    
    async def _strategy_context_inference(self, tool_name: str, tool_args: Dict[str, Any], 
                                        error: Exception, context: Dict[str, Any], 
                                        mcp_clients: Dict[str, Any]) -> Dict[str, Any]:
        """Stratégie: Inférence basée sur le contexte"""
        logger.info(f"🛡️ RÉSILIENCE: Inférence contextuelle pour {tool_name}")
        
        # Inférer basé sur les informations disponibles
        project_path = context.get("target_path", "")
        analysis_type = context.get("analysis_type", "")
        
        inference = f"🛡️ INFÉRENCE CONTEXTUELLE:\n\n"
        inference += f"📁 Chemin analysé: {project_path}\n"
        inference += f"🎯 Type d'analyse: {analysis_type}\n"
        inference += f"❌ Outil échoué: {tool_name} - {str(error)[:100]}\n\n"
        
        # Inférences basées sur le chemin
        if "restaurant-app" in project_path:
            inference += "🍽️ INFÉRENCE: Projet restaurant (Next.js probable)\n"
        elif "design-agent" in project_path:
            inference += "🎨 INFÉRENCE: Agent de design (TypeScript + Templates)\n"
        elif "orchestrator" in project_path:
            inference += "🎭 INFÉRENCE: Orchestrateur (Coordination multi-agents)\n"
        
        inference += "\n⚠️ Cette analyse est basée sur l'inférence contextuelle uniquement"
        
        return {
            "success": True,
            "data": {"content": [{"type": "text", "text": inference}]},
            "message": "🛡️ Inférence contextuelle générée"
        }
    
    async def _strategy_graceful_degradation(self, tool_name: str, tool_args: Dict[str, Any], 
                                           error: Exception, context: Dict[str, Any], 
                                           mcp_clients: Dict[str, Any]) -> Dict[str, Any]:
        """Stratégie: Dégradation gracieuse avec feedback instructif"""
        logger.info(f"🛡️ RÉSILIENCE: Dégradation gracieuse pour {tool_name}")
        
        graceful_message = f"🛡️ DÉGRADATION GRACIEUSE:\n\n"
        graceful_message += f"❌ L'outil {tool_name} a échoué: {str(error)[:100]}\n\n"
        graceful_message += f"🔧 ACTIONS RECOMMANDÉES:\n"
        
        if tool_name == "Read":
            graceful_message += "• Vérifier l'encodage du fichier\n"
            graceful_message += "• Essayer avec un éditeur de texte externe\n"
            graceful_message += "• Vérifier les permissions d'accès\n"
        elif tool_name == "LS":
            graceful_message += "• Vérifier l'existence du répertoire\n" 
            graceful_message += "• Vérifier les permissions d'accès\n"
            graceful_message += "• Essayer avec un chemin parent\n"
        elif tool_name == "Glob":
            graceful_message += "• Simplifier le pattern de recherche\n"
            graceful_message += "• Vérifier la syntaxe du pattern\n"
            graceful_message += "• Essayer avec LS alternatif\n"
        
        graceful_message += f"\n💡 Le système continue avec les données disponibles"
        
        return {
            "success": True,
            "data": {"content": [{"type": "text", "text": graceful_message}]},
            "message": "🛡️ Dégradation gracieuse appliquée"
        }
    
    def _infer_content_from_structure(self, ls_result: Dict[str, Any], file_ext: str, filename: str) -> str:
        """Infère le contenu d'un fichier basé sur la structure du répertoire"""
        
        inference = f"🛡️ INFÉRENCE STRUCTURELLE pour {filename}:\n\n"
        
        # Analyser le contenu LS
        ls_content = ls_result.get("content", [{}])[0].get("text", "")
        
        if file_ext == "json":
            if "package.json" in filename:
                inference += "📦 INFÉRENCE: package.json - Configuration npm/yarn\n"
                inference += "🔍 Contenu probable: dependencies, scripts, metadata\n"
            elif "config" in filename:
                inference += "⚙️ INFÉRENCE: Fichier de configuration\n"
                inference += "🔍 Contenu probable: paramètres, clés API, environnement\n"
        
        elif file_ext in ["ts", "js"]:
            if "index" in filename:
                inference += "📚 INFÉRENCE: Point d'entrée principal\n"
                inference += "🔍 Contenu probable: exports, imports, initialisation\n"
            elif "orchestrator" in filename:
                inference += "🎭 INFÉRENCE: Orchestrateur de coordination\n"
                inference += "🔍 Contenu probable: coordination agents, workflows\n"
        
        # Analyser la structure pour plus d'indices
        if "coordination/" in ls_content:
            inference += "\n📁 Répertoire 'coordination/' détecté\n"
            inference += "🎯 Suggestion: Système de coordination multi-agents\n"
        
        if "workflows/" in ls_content:
            inference += "\n📁 Répertoire 'workflows/' détecté\n"
            inference += "🎯 Suggestion: Processus automatisés et pipelines\n"
        
        inference += f"\n⚠️ Cette inférence est basée sur la structure observée"
        
        return inference
    
    def get_resilience_report(self) -> Dict[str, Any]:
        """Génère un rapport de performance de la couche de résilience"""
        return {
            "success_rate": self.metrics.get_success_rate(),
            "total_recoveries": self.metrics.recovery_attempts,
            "successful_recoveries": self.metrics.successful_recoveries,
            "error_patterns": self.metrics.error_patterns,
            "strategy_effectiveness": self.metrics.strategy_effectiveness,
            "timestamp": datetime.now().isoformat()
        }

# Instance globale
resilience_layer = MCPResilienceLayer()