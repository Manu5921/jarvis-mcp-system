"""
Perplexity MCP Server - Interface MCP pour Perplexity Pro
Recherche et analyse en temps réel via protocole MCP
"""

import asyncio
import json
import logging
import os
from datetime import datetime
from typing import Any

import requests
from mcp.server.models import InitializationOptions
from mcp.server import NotificationOptions, Server
from mcp.types import Resource, Tool, TextContent
import mcp.types as types

# Configuration
PERPLEXITY_API_KEY = os.getenv("PERPLEXITY_API_KEY", "")
PERPLEXITY_BASE_URL = "https://api.perplexity.ai/chat/completions"

server = Server("perplexity-mcp")

# Perplexity Client
class PerplexityClient:
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = PERPLEXITY_BASE_URL
    
    async def search(self, query: str, model: str = "llama-3.1-sonar-small-128k-online") -> dict:
        """Search with Perplexity"""
        if not self.api_key:
            return {"error": "Clé API Perplexity manquante"}
        
        try:
            response = requests.post(
                self.base_url,
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json"
                },
                json={
                    "model": model,
                    "messages": [{"role": "user", "content": query}],
                    "max_tokens": 2000,
                    "temperature": 0.2,
                    "top_p": 0.9,
                    "return_citations": True,
                    "search_domain_filter": ["github.com", "stackoverflow.com", "docs.python.org", "developer.mozilla.org"],
                    "return_images": False
                },
                timeout=30
            )
            
            if response.status_code == 200:
                data = response.json()
                choice = data["choices"][0]
                
                return {
                    "content": choice["message"]["content"],
                    "citations": choice.get("citations", []),
                    "model": model,
                    "usage": data.get("usage", {}),
                    "timestamp": datetime.now().isoformat()
                }
            else:
                return {"error": f"Erreur Perplexity: {response.status_code} - {response.text}"}
                
        except Exception as e:
            return {"error": f"Erreur connexion Perplexity: {str(e)}"}
    
    async def research_deep(self, topic: str, focus_areas: list = None) -> dict:
        """Deep research on a topic"""
        focus_areas = focus_areas or ["latest developments", "best practices", "examples"]
        
        results = {}
        
        for area in focus_areas:
            query = f"{topic} {area} 2024 2025"
            result = await self.search(query, "llama-3.1-sonar-large-128k-online")
            results[area] = result
        
        return {
            "topic": topic,
            "research_areas": results,
            "summary_query": f"Summarize latest insights about {topic}",
            "timestamp": datetime.now().isoformat()
        }

perplexity = PerplexityClient(PERPLEXITY_API_KEY)

# MCP Tools
@server.list_tools()
async def handle_list_tools() -> list[Tool]:
    """List available Perplexity tools"""
    return [
        Tool(
            name="perplexity_search",
            description="Recherche en temps réel avec Perplexity Pro",
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Requête de recherche"
                    },
                    "model": {
                        "type": "string",
                        "enum": ["llama-3.1-sonar-small-128k-online", "llama-3.1-sonar-large-128k-online"],
                        "default": "llama-3.1-sonar-small-128k-online",
                        "description": "Modèle Perplexity"
                    },
                    "focus": {
                        "type": "string",
                        "enum": ["development", "latest", "documentation", "examples", "general"],
                        "default": "development",
                        "description": "Focus de la recherche"
                    }
                },
                "required": ["query"]
            }
        ),
        Tool(
            name="perplexity_research",
            description="Recherche approfondie multi-angles",
            inputSchema={
                "type": "object",
                "properties": {
                    "topic": {
                        "type": "string",
                        "description": "Sujet de recherche"
                    },
                    "aspects": {
                        "type": "array",
                        "items": {"type": "string"},
                        "default": ["latest updates", "best practices", "examples", "troubleshooting"],
                        "description": "Aspects à explorer"
                    }
                },
                "required": ["topic"]
            }
        ),
        Tool(
            name="perplexity_tech_news",
            description="Actualités tech récentes sur un sujet",
            inputSchema={
                "type": "object",
                "properties": {
                    "technology": {
                        "type": "string",
                        "description": "Technologie ou framework"
                    },
                    "timeframe": {
                        "type": "string",
                        "enum": ["last week", "last month", "last quarter", "2024"],
                        "default": "last month",
                        "description": "Période de recherche"
                    }
                },
                "required": ["technology"]
            }
        ),
        Tool(
            name="perplexity_compare",
            description="Compare plusieurs technologies/solutions",
            inputSchema={
                "type": "object",
                "properties": {
                    "options": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Technologies/solutions à comparer"
                    },
                    "criteria": {
                        "type": "array", 
                        "items": {"type": "string"},
                        "default": ["performance", "ease of use", "community", "documentation"],
                        "description": "Critères de comparaison"
                    }
                },
                "required": ["options"]
            }
        )
    ]

@server.call_tool()
async def handle_call_tool(name: str, arguments: dict) -> list[types.TextContent]:
    """Handle tool calls"""
    
    if name == "perplexity_search":
        query = arguments.get("query", "")
        model = arguments.get("model", "llama-3.1-sonar-small-128k-online")
        focus = arguments.get("focus", "development")
        
        # Enhance query based on focus
        enhanced_queries = {
            "development": f"{query} programming development code examples tutorial",
            "latest": f"{query} latest 2024 2025 recent updates news",
            "documentation": f"{query} official documentation API reference guide",
            "examples": f"{query} examples code samples implementation tutorial",
            "general": query
        }
        
        enhanced_query = enhanced_queries.get(focus, query)
        result = await perplexity.search(enhanced_query, model)
        
        return [types.TextContent(
            type="text",
            text=json.dumps({
                "search_result": result,
                "original_query": query,
                "enhanced_query": enhanced_query,
                "focus": focus
            })
        )]
    
    elif name == "perplexity_research":
        topic = arguments.get("topic", "")
        aspects = arguments.get("aspects", ["latest updates", "best practices", "examples"])
        
        result = await perplexity.research_deep(topic, aspects)
        
        return [types.TextContent(
            type="text",
            text=json.dumps({
                "research_result": result,
                "topic": topic,
                "aspects_covered": aspects
            })
        )]
    
    elif name == "perplexity_tech_news":
        technology = arguments.get("technology", "")
        timeframe = arguments.get("timeframe", "last month")
        
        query = f"{technology} news updates {timeframe} 2024 development release"
        result = await perplexity.search(query, "llama-3.1-sonar-large-128k-online")
        
        return [types.TextContent(
            type="text",
            text=json.dumps({
                "tech_news": result,
                "technology": technology,
                "timeframe": timeframe
            })
        )]
    
    elif name == "perplexity_compare":
        options = arguments.get("options", [])
        criteria = arguments.get("criteria", ["performance", "ease of use"])
        
        if len(options) < 2:
            return [types.TextContent(
                type="text",
                text=json.dumps({"error": "Au moins 2 options requises pour comparaison"})
            )]
        
        comparison_query = f"""Compare {' vs '.join(options)} for:
        {', '.join(criteria)}
        
        Provide detailed comparison with pros/cons, use cases, and recommendations."""
        
        result = await perplexity.search(comparison_query, "llama-3.1-sonar-large-128k-online")
        
        return [types.TextContent(
            type="text",
            text=json.dumps({
                "comparison": result,
                "options_compared": options,
                "criteria": criteria
            })
        )]
    
    else:
        raise ValueError(f"Outil inconnu: {name}")

# Resources
@server.list_resources()
async def handle_list_resources() -> list[Resource]:
    """List Perplexity resources"""
    return [
        Resource(
            uri="perplexity://status",
            name="Statut Perplexity",
            description="Statut de connexion Perplexity API",
            mimeType="application/json"
        ),
        Resource(
            uri="perplexity://models",
            name="Modèles Perplexity",
            description="Modèles disponibles sur Perplexity",
            mimeType="application/json"
        )
    ]

@server.read_resource()
async def handle_read_resource(uri: str) -> str:
    """Read Perplexity resources"""
    if uri == "perplexity://status":
        status = "configured" if PERPLEXITY_API_KEY else "missing_api_key"
        return json.dumps({
            "status": status,
            "api_key_configured": bool(PERPLEXITY_API_KEY),
            "base_url": PERPLEXITY_BASE_URL
        })
    elif uri == "perplexity://models":
        return json.dumps({
            "models": [
                "llama-3.1-sonar-small-128k-online",
                "llama-3.1-sonar-large-128k-online",
                "llama-3.1-sonar-huge-128k-online"
            ]
        })
    else:
        raise ValueError(f"Ressource inconnue: {uri}")

# Run server
async def main():
    from mcp.server.stdio import stdio_server
    
    async with stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream,
            write_stream,
            InitializationOptions(
                server_name="perplexity-mcp",
                server_version="1.0.0",
                capabilities=server.get_capabilities(
                    notification_options=NotificationOptions(),
                    experimental_capabilities={},
                ),
            ),
        )

if __name__ == "__main__":
    asyncio.run(main())