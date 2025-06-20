"""
Middleware de validation MCP - Force l'utilisation des outils
Intercepte les requêtes et ajoute des contraintes obligatoires
"""

import re
from typing import Dict, List, Any, Optional
import logging

logger = logging.getLogger(__name__)

class MCPValidationMiddleware:
    """Middleware qui force l'utilisation correcte des outils MCP"""
    
    def __init__(self):
        self.forbidden_phrases = [
            'probablement', 'semble', 'peut-être', 'il se peut que',
            'on peut supposer', 'il est possible que', 'vraisemblablement'
        ]
        
        self.analysis_keywords = [
            'analyse', 'structure', 'projet', 'framework', 'technologies'
        ]
        
        self.required_tools_map = {
            'structure': ['LS'],
            'projet': ['LS', 'Read'],
            'framework': ['Read'],
            'technologies': ['Glob', 'Read'],
            'analyse': ['LS', 'Read']
        }
    
    def preprocess_request(self, message: str, session_id: str) -> Dict[str, Any]:
        """Prétraite la requête et ajoute des contraintes obligatoires"""
        
        # Détection du type d'analyse demandée
        analysis_type = self._detect_analysis_type(message.lower())
        
        if analysis_type:
            # Force l'ajout d'instructions obligatoires
            enhanced_message = self._enhance_message_with_requirements(message, analysis_type)
            
            # Ajoute un préambule de contraintes
            constraint_prefix = self._generate_constraint_prefix(analysis_type)
            
            final_message = f"{constraint_prefix}\n\n{enhanced_message}"
            
            logger.info(f"🔧 Middleware: Enhanced message for session {session_id}")
            logger.info(f"🎯 Analysis type: {analysis_type}")
            logger.info(f"🛠️ Required tools: {self.required_tools_map.get(analysis_type, [])}")
            
            return {
                'enhanced_message': final_message,
                'required_tools': self.required_tools_map.get(analysis_type, []),
                'analysis_type': analysis_type,
                'validation_active': True
            }
        
        return {
            'enhanced_message': message,
            'required_tools': [],
            'analysis_type': None,
            'validation_active': False
        }
    
    def validate_response(self, response: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Valide la réponse et génère un score"""
        
        if not context.get('validation_active'):
            return {'score': 10, 'passed': True, 'feedback': []}
        
        score = 5  # Score de base
        feedback = []
        errors = []
        
        # 1. Vérification des phrases interdites
        forbidden_found = []
        for phrase in self.forbidden_phrases:
            if phrase.lower() in response.lower():
                forbidden_found.append(phrase)
        
        if forbidden_found:
            score -= 3
            errors.append(f"Phrases interdites: {', '.join(forbidden_found)}")
        else:
            score += 1
            feedback.append("✅ Pas de suppositions détectées")
        
        # 2. Vérification utilisation des outils requis
        required_tools = context.get('required_tools', [])
        tools_evidence = self._detect_tool_usage(response)
        
        missing_tools = []
        for tool in required_tools:
            if tool not in tools_evidence:
                missing_tools.append(tool)
        
        if missing_tools:
            score -= 4
            errors.append(f"Outils manquants: {', '.join(missing_tools)}")
        else:
            score += 3
            feedback.append(f"✅ Outils utilisés: {', '.join(tools_evidence)}")
        
        # 3. Vérification contenu factuel
        factual_indicators = [
            r'version\s+\d+\.\d+',  # Versions
            r'package\.json',        # Fichiers config
            r'next\.js|react|typescript',  # Technologies courantes
            r'/[\w\-/]+/',          # Chemins de fichiers
        ]
        
        factual_score = 0
        for pattern in factual_indicators:
            if re.search(pattern, response, re.IGNORECASE):
                factual_score += 1
        
        if factual_score >= 2:
            score += 2
            feedback.append(f"✅ Contenu factuel détecté ({factual_score} indicateurs)")
        else:
            score -= 1
            errors.append("Contenu insuffisamment factuel")
        
        # Score final
        final_score = max(0, min(10, score))
        passed = final_score >= 7
        
        return {
            'score': final_score,
            'passed': passed,
            'feedback': feedback,
            'errors': errors,
            'tools_evidence': tools_evidence,
            'factual_score': factual_score
        }
    
    def generate_correction_feedback(self, validation: Dict[str, Any], context: Dict[str, Any]) -> str:
        """Génère un feedback de correction détaillé"""
        
        analysis_type = context.get('analysis_type', 'analyse')
        required_tools = context.get('required_tools', [])
        
        feedback = f"""
🚫 ÉCHEC VALIDATION MCP - Score: {validation['score']}/10

❌ ERREURS DÉTECTÉES:
{chr(10).join(f"• {error}" for error in validation['errors'])}

🔧 CORRECTION OBLIGATOIRE:

1. EXPLORATION OBLIGATOIRE:
   - Utilisez LS pour explorer la structure du répertoire
   - Lisez les fichiers de configuration réels (package.json, etc.)
   - Utilisez Glob pour identifier les patterns de code

2. INTERDICTIONS ABSOLUES:
   - STOP aux suppositions ("probablement", "semble", "peut-être")
   - STOP aux inventions de technologies sans preuve
   - STOP aux réponses génériques

3. WORKFLOW OBLIGATOIRE POUR {analysis_type.upper()}:
   {chr(10).join(f"   - {tool}: OBLIGATOIRE" for tool in required_tools)}

4. RECOMMENCER IMMÉDIATEMENT:
   Refaites l'analyse en utilisant UNIQUEMENT les outils MCP et les faits découverts.

⚠️ ATTENTION: Cette validation est automatique. Votre prochaine réponse sera re-scorée.
"""
        
        return feedback
    
    def _detect_analysis_type(self, message: str) -> Optional[str]:
        """Détecte le type d'analyse demandée"""
        for keyword in self.analysis_keywords:
            if keyword in message:
                return keyword
        return None
    
    def _enhance_message_with_requirements(self, message: str, analysis_type: str) -> str:
        """Enrichit le message avec des exigences spécifiques"""
        
        enhancements = {
            'structure': "Explorez d'abord la structure avec LS, puis analysez les fichiers trouvés.",
            'projet': "Lisez les fichiers de configuration (package.json, README.md) après exploration LS.",
            'framework': "Identifiez le framework en lisant package.json et les fichiers de config réels.",
            'technologies': "Utilisez Glob pour identifier les patterns de fichiers, puis Read pour confirmer.",
            'analyse': "Commencez par LS pour explorer, puis Read les fichiers clés pour une analyse factuelle."
        }
        
        enhancement = enhancements.get(analysis_type, enhancements['analyse'])
        
        return f"{message}\n\nEXIGENCE: {enhancement}"
    
    def _generate_constraint_prefix(self, analysis_type: str) -> str:
        """Génère un préfixe de contraintes obligatoires"""
        
        required_tools = self.required_tools_map.get(analysis_type, ['LS', 'Read'])
        
        return f"""
🔒 CONTRAINTES VALIDATION MCP ACTIVE:
• OBLIGATION: Utilisez {', '.join(required_tools)} avant toute réponse
• INTERDICTION: Suppositions, inventions, réponses génériques
• VALIDATION: Score minimum 7/10 requis
• ÉCHEC: Feedback correctif automatique
"""
    
    def _detect_tool_usage(self, response: str) -> List[str]:
        """Détecte les outils utilisés dans la réponse"""
        tools_found = []
        
        # Patterns de détection d'utilisation des outils
        tool_patterns = {
            'LS': [r'ls\s+/', r'listing', r'structure.*répertoire', r'fichiers.*trouvés'],
            'Read': [r'lecture.*package\.json', r'contenu.*fichier', r'read.*config'],
            'Glob': [r'glob', r'patterns.*fichiers', r'recherche.*\*'],
            'Grep': [r'grep', r'recherche.*contenu', r'filtrage']
        }
        
        for tool, patterns in tool_patterns.items():
            for pattern in patterns:
                if re.search(pattern, response, re.IGNORECASE):
                    if tool not in tools_found:
                        tools_found.append(tool)
                    break
        
        return tools_found

# Instance globale du middleware
validation_middleware = MCPValidationMiddleware()