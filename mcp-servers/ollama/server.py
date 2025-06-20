"""
Ollama MCP Server - Interface MCP pour Ollama local
Standardise les communications avec Ollama via protocole MCP
"""

import asyncio
import logging
import os
from typing import Any, Sequence
import requests

from mcp.server.models import InitializationOptions
from mcp.server import NotificationOptions, Server
from mcp.types import Resource, Tool, TextContent, ImageContent, EmbeddedResource
import mcp.types as types

# Configuration
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "localhost:11434")
server = Server("ollama-mcp")

# Ollama Client
class OllamaClient:
    def __init__(self, host: str):
        self.base_url = f"http://{host}"
    
    async def generate(self, model: str, prompt: str, context: str = None) -> str:
        """Generate response with Ollama"""
        try:
            # Add context if provided
            if context:
                prompt = f"Contexte: {context}\n\nQuestion: {prompt}"
            
            response = requests.post(
                f"{self.base_url}/api/generate",
                json={
                    "model": model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": 0.7,
                        "top_p": 0.9
                    }
                },
                timeout=120
            )
            
            if response.status_code == 200:
                return response.json().get("response", "Erreur de génération")
            else:
                return f"Erreur Ollama: {response.status_code}"
                
        except Exception as e:
            return f"Erreur connexion Ollama: {str(e)}"
    
    async def get_models(self) -> list:
        """Get available models"""
        try:
            response = requests.get(f"{self.base_url}/api/tags")
            if response.status_code == 200:
                models = response.json().get("models", [])
                return [model["name"] for model in models]
            return []
        except:
            return ["llama3.2:3b", "llama3.2:8b"]  # Fallback

ollama = OllamaClient(OLLAMA_HOST)

# MCP Tools
@server.list_tools()
async def handle_list_tools() -> list[Tool]:
    """List available Ollama tools"""
    return [
        Tool(
            name="ollama_chat",
            description="Chat avec Ollama local pour développement et code",
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Question ou demande pour Ollama"
                    },
                    "model": {
                        "type": "string", 
                        "description": "Modèle Ollama à utiliser",
                        "enum": ["llama3.2:3b", "llama3.2:8b", "auto"],
                        "default": "auto"
                    },
                    "context": {
                        "type": "string",
                        "description": "Contexte additionnel",
                        "default": ""
                    },
                    "project": {
                        "type": "string",
                        "description": "Nom du projet",
                        "default": ""
                    }
                },
                "required": ["query"]
            }
        ),
        Tool(
            name="ollama_analyze_code",
            description="Analyse de code spécialisée avec Ollama",
            inputSchema={
                "type": "object",
                "properties": {
                    "code": {
                        "type": "string",
                        "description": "Code à analyser"
                    },
                    "language": {
                        "type": "string",
                        "description": "Langage de programmation"
                    },
                    "analysis_type": {
                        "type": "string",
                        "enum": ["review", "debug", "optimize", "explain"],
                        "description": "Type d'analyse"
                    }
                },
                "required": ["code", "analysis_type"]
            }
        ),
        Tool(
            name="ollama_optimize_prompt",
            description="Optimise un prompt pour une autre IA",
            inputSchema={
                "type": "object", 
                "properties": {
                    "original_prompt": {
                        "type": "string",
                        "description": "Prompt original"
                    },
                    "target_ai": {
                        "type": "string",
                        "enum": ["claude", "chatgpt", "perplexity"],
                        "description": "IA cible"
                    },
                    "context": {
                        "type": "string",
                        "description": "Contexte du projet"
                    }
                },
                "required": ["original_prompt", "target_ai"]
            }
        )
    ]

@server.call_tool()
async def handle_call_tool(name: str, arguments: dict) -> list[types.TextContent]:
    """Handle tool calls"""
    
    if name == "ollama_chat":
        query = arguments.get("query", "")
        model = arguments.get("model", "auto")
        context = arguments.get("context", "")
        project = arguments.get("project", "")
        
        # Auto-select model based on query complexity
        if model == "auto":
            model = "llama3.2:3b" if len(query) < 150 else "llama3.2:8b"
        
        # Build enhanced context
        enhanced_context = []
        if project:
            enhanced_context.append(f"Projet: {project}")
        if context:
            enhanced_context.append(f"Contexte: {context}")
        
        full_context = " | ".join(enhanced_context) if enhanced_context else ""
        
        # Generate response
        response = await ollama.generate(model, query, full_context)
        
        return [types.TextContent(
            type="text",
            text=json.dumps({
                "response": response,
                "model_used": model,
                "context_applied": full_context,
                "timestamp": datetime.now().isoformat()
            })
        )]
    
    elif name == "ollama_analyze_code":
        code = arguments.get("code", "")
        language = arguments.get("language", "")
        analysis_type = arguments.get("analysis_type", "review")
        
        # Specialized prompts for code analysis
        prompts = {
            "review": f"Analyse ce code {language} et donne des suggestions d'amélioration:\n```{language}\n{code}\n```",
            "debug": f"Trouve et explique les bugs potentiels dans ce code {language}:\n```{language}\n{code}\n```", 
            "optimize": f"Optimise ce code {language} pour de meilleures performances:\n```{language}\n{code}\n```",
            "explain": f"Explique ligne par ligne ce que fait ce code {language}:\n```{language}\n{code}\n```"
        }
        
        prompt = prompts.get(analysis_type, prompts["review"])
        model = "llama3.2:8b"  # Use larger model for code analysis
        
        response = await ollama.generate(model, prompt)
        
        return [types.TextContent(
            type="text", 
            text=json.dumps({
                "analysis": response,
                "analysis_type": analysis_type,
                "language": language,
                "model_used": model
            })
        )]
    
    elif name == "ollama_optimize_prompt":
        original_prompt = arguments.get("original_prompt", "")
        target_ai = arguments.get("target_ai", "claude")
        context = arguments.get("context", "")
        
        # AI-specific optimization prompts
        optimization_prompts = {
            "claude": f"""Optimise ce prompt pour Claude Code (assistant de développement):
            
Prompt original: {original_prompt}
Contexte: {context}

Crée un prompt structuré et précis pour Claude Code, en incluant:
- Le contexte technique nécessaire
- Des instructions claires et spécifiques  
- Le format de réponse attendu""",

            "chatgpt": f"""Optimise ce prompt pour ChatGPT:
            
Prompt original: {original_prompt}
Contexte: {context}

Crée un prompt équilibré pour ChatGPT, en incluant:
- Le rôle à jouer
- Le contexte et les contraintes
- Un exemple si pertinent""",

            "perplexity": f"""Optimise cette recherche pour Perplexity:
            
Recherche originale: {original_prompt}
Contexte: {context}

Reformule en requête de recherche optimale pour Perplexity:
- Mots-clés techniques précis
- Critères de fraîcheur si nécessaire
- Spécialisation du domaine"""
        }
        
        optimization_prompt = optimization_prompts.get(target_ai, optimization_prompts["claude"])
        response = await ollama.generate("llama3.2:8b", optimization_prompt)
        
        return [types.TextContent(
            type="text",
            text=json.dumps({
                "optimized_prompt": response,
                "target_ai": target_ai,
                "original": original_prompt
            })
        )]
    
    else:
        raise ValueError(f"Outil inconnu: {name}")

# Resources (optionnel - pour partager des infos sur Ollama)
@server.list_resources()
async def handle_list_resources() -> list[Resource]:
    """List Ollama resources"""
    models = await ollama.get_models()
    
    return [
        Resource(
            uri="ollama://models",
            name="Modèles Ollama disponibles", 
            description="Liste des modèles Ollama installés",
            mimeType="application/json"
        ),
        Resource(
            uri="ollama://status",
            name="Statut Ollama",
            description="Statut de connexion Ollama",
            mimeType="application/json"
        )
    ]

@server.read_resource()
async def handle_read_resource(uri: str) -> str:
    """Read Ollama resources"""
    if uri == "ollama://models":
        models = await ollama.get_models()
        return json.dumps({"models": models})
    elif uri == "ollama://status":
        try:
            response = requests.get(f"http://{OLLAMA_HOST}/api/tags", timeout=5)
            status = "online" if response.status_code == 200 else "offline"
        except:
            status = "offline"
        return json.dumps({"status": status, "host": OLLAMA_HOST})
    else:
        raise ValueError(f"Ressource inconnue: {uri}")

# Run server
async def main():
    # Import here to avoid issues with asyncio
    from mcp.server.stdio import stdio_server
    
    async with stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream,
            write_stream,
            InitializationOptions(
                server_name="ollama-mcp",
                server_version="1.0.0",
                capabilities=server.get_capabilities(
                    notification_options=NotificationOptions(),
                    experimental_capabilities={},
                ),
            ),
        )

if __name__ == "__main__":
    import json
    from datetime import datetime
    asyncio.run(main())