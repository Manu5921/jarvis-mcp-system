"""
Memory MCP Server - Gestion intelligente du contexte et historique
Mémoire vectorielle et optimisation de prompts via MCP
"""

import asyncio
import json
import logging
import os
import sqlite3
from datetime import datetime, timedelta
from typing import Any, List, Dict
from pathlib import Path

from mcp.server.models import InitializationOptions
from mcp.server import NotificationOptions, Server
from mcp.types import Resource, Tool, TextContent
import mcp.types as types

# Configuration
DATABASE_PATH = "/app/data/memory.db"
server = Server("memory-mcp")

# Database setup
def init_database():
    """Initialize SQLite database for memory"""
    Path("/app/data").mkdir(parents=True, exist_ok=True)
    
    conn = sqlite3.connect(DATABASE_PATH)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS conversations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            query TEXT NOT NULL,
            response TEXT NOT NULL,
            ai_used TEXT NOT NULL,
            project TEXT,
            context TEXT,
            optimized_prompt TEXT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            embedding_summary TEXT,
            tags TEXT
        )
    """)
    
    conn.execute("""
        CREATE TABLE IF NOT EXISTS context_patterns (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            pattern_type TEXT NOT NULL,
            pattern_data TEXT NOT NULL,
            frequency INTEGER DEFAULT 1,
            last_used DATETIME DEFAULT CURRENT_TIMESTAMP,
            effectiveness_score REAL DEFAULT 0.5
        )
    """)
    
    conn.execute("""
        CREATE TABLE IF NOT EXISTS prompt_optimizations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            original_prompt TEXT NOT NULL,
            optimized_prompt TEXT NOT NULL,
            target_ai TEXT NOT NULL,
            context_used TEXT,
            success_rate REAL DEFAULT 0.0,
            usage_count INTEGER DEFAULT 0,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    """)
    
    conn.commit()
    conn.close()

init_database()

# Memory Manager
class MemoryManager:
    def __init__(self):
        self.db_path = DATABASE_PATH
    
    def get_connection(self):
        return sqlite3.connect(self.db_path)
    
    async def store_conversation(self, query: str, response: str, ai_used: str, 
                                project: str = None, context: str = None, 
                                optimized_prompt: str = None) -> int:
        """Store conversation in memory"""
        conn = self.get_connection()
        
        # Generate summary and tags
        summary = self.generate_summary(query, response)
        tags = self.extract_tags(query, response, ai_used)
        
        cursor = conn.execute("""
            INSERT INTO conversations 
            (query, response, ai_used, project, context, optimized_prompt, embedding_summary, tags)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (query, response, ai_used, project, context, optimized_prompt, summary, tags))
        
        conversation_id = cursor.lastrowid
        conn.commit()
        conn.close()
        
        return conversation_id
    
    async def get_relevant_context(self, query: str, project: str = None, limit: int = 5) -> List[Dict]:
        """Get relevant conversations for context"""
        conn = self.get_connection()
        
        # Simple keyword-based relevance for now
        keywords = self.extract_keywords(query)
        
        conditions = []
        params = []
        
        # Keyword matching
        if keywords:
            keyword_conditions = []
            for keyword in keywords[:3]:  # Limit to top 3 keywords
                keyword_conditions.append("(query LIKE ? OR response LIKE ? OR tags LIKE ?)")
                params.extend([f"%{keyword}%", f"%{keyword}%", f"%{keyword}%"])
            conditions.append(f"({' OR '.join(keyword_conditions)})")
        
        # Project filtering
        if project:
            conditions.append("project = ?")
            params.append(project)
        
        # Recent conversations (last 30 days)
        conditions.append("timestamp > ?")
        params.append((datetime.now() - timedelta(days=30)).isoformat())
        
        where_clause = " AND ".join(conditions) if conditions else "1=1"
        
        cursor = conn.execute(f"""
            SELECT query, response, ai_used, project, context, timestamp
            FROM conversations 
            WHERE {where_clause}
            ORDER BY timestamp DESC 
            LIMIT ?
        """, params + [limit])
        
        results = []
        for row in cursor.fetchall():
            results.append({
                "query": row[0],
                "response": row[1][:200] + "..." if len(row[1]) > 200 else row[1],
                "ai_used": row[2],
                "project": row[3],
                "context": row[4],
                "timestamp": row[5]
            })
        
        conn.close()
        return results
    
    async def optimize_prompt(self, original_prompt: str, target_ai: str, 
                             context: str = None, project: str = None) -> str:
        """Generate optimized prompt based on patterns and context"""
        conn = self.get_connection()
        
        # Get successful patterns for this AI
        cursor = conn.execute("""
            SELECT optimized_prompt, success_rate, usage_count
            FROM prompt_optimizations 
            WHERE target_ai = ? AND success_rate > 0.7
            ORDER BY success_rate DESC, usage_count DESC
            LIMIT 3
        """, (target_ai,))
        
        successful_patterns = cursor.fetchall()
        
        # Get relevant context
        relevant_context = await self.get_relevant_context(original_prompt, project, 3)
        
        conn.close()
        
        # Build optimized prompt
        optimized_parts = []
        
        # Add project context if available
        if project:
            optimized_parts.append(f"Projet: {project}")
        
        # Add relevant historical context
        if relevant_context:
            context_summary = self.build_context_summary(relevant_context)
            optimized_parts.append(f"Contexte historique: {context_summary}")
        
        # Add current context
        if context:
            optimized_parts.append(f"Contexte actuel: {context}")
        
        # AI-specific optimizations
        ai_specific_prompt = self.get_ai_specific_optimization(original_prompt, target_ai)
        optimized_parts.append(ai_specific_prompt)
        
        optimized_prompt = "\n\n".join(optimized_parts)
        
        # Store optimization for learning
        await self.store_optimization(original_prompt, optimized_prompt, target_ai, context)
        
        return optimized_prompt
    
    async def store_optimization(self, original: str, optimized: str, target_ai: str, context: str):
        """Store prompt optimization for learning"""
        conn = self.get_connection()
        
        conn.execute("""
            INSERT INTO prompt_optimizations 
            (original_prompt, optimized_prompt, target_ai, context_used)
            VALUES (?, ?, ?, ?)
        """, (original, optimized, target_ai, context))
        
        conn.commit()
        conn.close()
    
    def generate_summary(self, query: str, response: str) -> str:
        """Generate summary for embedding/search"""
        # Simple summary - first 100 chars of query + response type
        summary = query[:100]
        if "error" in response.lower():
            summary += " [ERROR]"
        elif "code" in response.lower() or "function" in response.lower():
            summary += " [CODE]"
        elif "explain" in query.lower():
            summary += " [EXPLAIN]"
        return summary
    
    def extract_tags(self, query: str, response: str, ai_used: str) -> str:
        """Extract tags for categorization"""
        tags = [ai_used]
        
        # Technology tags
        tech_keywords = ["python", "javascript", "react", "nextjs", "docker", "sql", "api", "fastapi"]
        for keyword in tech_keywords:
            if keyword in query.lower() or keyword in response.lower():
                tags.append(keyword)
        
        # Intent tags
        if any(word in query.lower() for word in ["error", "bug", "fix", "debug"]):
            tags.append("debug")
        elif any(word in query.lower() for word in ["how", "explain", "what"]):
            tags.append("explanation")
        elif any(word in query.lower() for word in ["optimize", "improve", "better"]):
            tags.append("optimization")
        
        return ",".join(tags[:5])  # Limit to 5 tags
    
    def extract_keywords(self, text: str) -> List[str]:
        """Extract keywords for relevance matching"""
        # Simple keyword extraction
        words = text.lower().split()
        
        # Filter common words
        stop_words = {"the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for", "of", "with", "by", "from", "up", "about", "into", "through", "during", "before", "after", "above", "below", "between", "among", "across", "against", "within"}
        
        keywords = [word for word in words if len(word) > 3 and word not in stop_words]
        return keywords[:5]  # Top 5 keywords
    
    def build_context_summary(self, relevant_context: List[Dict]) -> str:
        """Build context summary from relevant conversations"""
        if not relevant_context:
            return ""
        
        summaries = []
        for ctx in relevant_context[:3]:  # Limit to 3 most relevant
            summary = f"- {ctx['ai_used']}: {ctx['query'][:50]}... → {ctx['response'][:50]}..."
            summaries.append(summary)
        
        return "\n".join(summaries)
    
    def get_ai_specific_optimization(self, prompt: str, target_ai: str) -> str:
        """Get AI-specific prompt optimization"""
        optimizations = {
            "claude": f"""Tu es Claude Code, assistant expert en développement.
Contexte: {prompt}

Réponds de manière structurée avec:
1. Analyse technique précise
2. Solutions pratiques avec code si pertinent  
3. Recommandations d'amélioration
4. Considérations de sécurité/performance si applicable""",
            
            "chatgpt": f"""Tu es un assistant développeur expert. 

Demande: {prompt}

Fournis une réponse équilibrée avec:
- Explication claire du problème/besoin
- Solution étape par étape
- Code d'exemple si nécessaire
- Alternatives possibles""",
            
            "perplexity": f"""Recherche approfondie sur: {prompt}

Focus sur:
- Informations récentes (2024-2025)
- Meilleures pratiques actuelles
- Documentation officielle
- Exemples concrets et tutoriels""",
            
            "ollama": f"""Question technique: {prompt}

Contexte: Assistant local pour développement
Attentes: Réponse directe et pratique avec exemples de code si pertinent."""
        }
        
        return optimizations.get(target_ai, prompt)

memory_manager = MemoryManager()

# MCP Tools
@server.list_tools()
async def handle_list_tools() -> list[Tool]:
    """List available Memory tools"""
    return [
        Tool(
            name="memory_store",
            description="Stocker une conversation dans la mémoire",
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Question originale"},
                    "response": {"type": "string", "description": "Réponse reçue"},
                    "ai_used": {"type": "string", "description": "IA utilisée"},
                    "project": {"type": "string", "description": "Nom du projet"},
                    "context": {"type": "string", "description": "Contexte additionnel"},
                    "optimized_prompt": {"type": "string", "description": "Prompt optimisé utilisé"}
                },
                "required": ["query", "response", "ai_used"]
            }
        ),
        Tool(
            name="memory_get_context",
            description="Récupérer le contexte pertinent pour une requête",
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Requête actuelle"},
                    "project": {"type": "string", "description": "Projet concerné"},
                    "limit": {"type": "integer", "default": 5, "description": "Nombre max de résultats"}
                },
                "required": ["query"]
            }
        ),
        Tool(
            name="memory_optimize_prompt",
            description="Optimiser un prompt pour une IA cible",
            inputSchema={
                "type": "object",
                "properties": {
                    "original_prompt": {"type": "string", "description": "Prompt original"},
                    "target_ai": {"type": "string", "enum": ["claude", "chatgpt", "perplexity", "ollama"], "description": "IA cible"},
                    "context": {"type": "string", "description": "Contexte actuel"},
                    "project": {"type": "string", "description": "Projet concerné"}
                },
                "required": ["original_prompt", "target_ai"]
            }
        ),
        Tool(
            name="memory_search",
            description="Rechercher dans l'historique des conversations",
            inputSchema={
                "type": "object",
                "properties": {
                    "keywords": {"type": "array", "items": {"type": "string"}, "description": "Mots-clés de recherche"},
                    "ai_filter": {"type": "string", "description": "Filtrer par IA"},
                    "project_filter": {"type": "string", "description": "Filtrer par projet"},
                    "limit": {"type": "integer", "default": 10, "description": "Nombre max de résultats"}
                },
                "required": ["keywords"]
            }
        ),
        Tool(
            name="memory_stats",
            description="Statistiques d'usage de la mémoire",
            inputSchema={
                "type": "object",
                "properties": {
                    "period": {"type": "string", "enum": ["week", "month", "all"], "default": "month", "description": "Période d'analyse"}
                }
            }
        )
    ]

@server.call_tool()
async def handle_call_tool(name: str, arguments: dict) -> list[types.TextContent]:
    """Handle tool calls"""
    
    if name == "memory_store":
        conversation_id = await memory_manager.store_conversation(
            query=arguments.get("query", ""),
            response=arguments.get("response", ""),
            ai_used=arguments.get("ai_used", ""),
            project=arguments.get("project"),
            context=arguments.get("context"),
            optimized_prompt=arguments.get("optimized_prompt")
        )
        
        return [types.TextContent(
            type="text",
            text=json.dumps({
                "stored": True,
                "conversation_id": conversation_id,
                "timestamp": datetime.now().isoformat()
            })
        )]
    
    elif name == "memory_get_context":
        context = await memory_manager.get_relevant_context(
            query=arguments.get("query", ""),
            project=arguments.get("project"),
            limit=arguments.get("limit", 5)
        )
        
        return [types.TextContent(
            type="text",
            text=json.dumps({
                "context": context,
                "query": arguments.get("query", ""),
                "context_count": len(context)
            })
        )]
    
    elif name == "memory_optimize_prompt":
        optimized = await memory_manager.optimize_prompt(
            original_prompt=arguments.get("original_prompt", ""),
            target_ai=arguments.get("target_ai", ""),
            context=arguments.get("context"),
            project=arguments.get("project")
        )
        
        return [types.TextContent(
            type="text",
            text=json.dumps({
                "optimized_prompt": optimized,
                "original_prompt": arguments.get("original_prompt", ""),
                "target_ai": arguments.get("target_ai", "")
            })
        )]
    
    elif name == "memory_search":
        keywords = arguments.get("keywords", [])
        ai_filter = arguments.get("ai_filter")
        project_filter = arguments.get("project_filter")
        limit = arguments.get("limit", 10)
        
        conn = memory_manager.get_connection()
        
        conditions = []
        params = []
        
        # Keyword search
        if keywords:
            keyword_conditions = []
            for keyword in keywords:
                keyword_conditions.append("(query LIKE ? OR response LIKE ? OR tags LIKE ?)")
                params.extend([f"%{keyword}%", f"%{keyword}%", f"%{keyword}%"])
            conditions.append(f"({' OR '.join(keyword_conditions)})")
        
        # Filters
        if ai_filter:
            conditions.append("ai_used = ?")
            params.append(ai_filter)
        
        if project_filter:
            conditions.append("project = ?")
            params.append(project_filter)
        
        where_clause = " AND ".join(conditions) if conditions else "1=1"
        
        cursor = conn.execute(f"""
            SELECT query, response, ai_used, project, timestamp
            FROM conversations 
            WHERE {where_clause}
            ORDER BY timestamp DESC 
            LIMIT ?
        """, params + [limit])
        
        results = []
        for row in cursor.fetchall():
            results.append({
                "query": row[0],
                "response": row[1][:300] + "..." if len(row[1]) > 300 else row[1],
                "ai_used": row[2],
                "project": row[3],
                "timestamp": row[4]
            })
        
        conn.close()
        
        return [types.TextContent(
            type="text",
            text=json.dumps({
                "search_results": results,
                "keywords": keywords,
                "total_found": len(results)
            })
        )]
    
    elif name == "memory_stats":
        period = arguments.get("period", "month")
        
        conn = memory_manager.get_connection()
        
        # Date filter
        if period == "week":
            date_filter = (datetime.now() - timedelta(days=7)).isoformat()
        elif period == "month":
            date_filter = (datetime.now() - timedelta(days=30)).isoformat()
        else:
            date_filter = "1970-01-01"
        
        # Total conversations
        cursor = conn.execute("SELECT COUNT(*) FROM conversations WHERE timestamp > ?", (date_filter,))
        total_conversations = cursor.fetchone()[0]
        
        # By AI
        cursor = conn.execute("""
            SELECT ai_used, COUNT(*) 
            FROM conversations 
            WHERE timestamp > ? 
            GROUP BY ai_used
        """, (date_filter,))
        by_ai = dict(cursor.fetchall())
        
        # By project
        cursor = conn.execute("""
            SELECT project, COUNT(*) 
            FROM conversations 
            WHERE timestamp > ? AND project IS NOT NULL
            GROUP BY project
        """, (date_filter,))
        by_project = dict(cursor.fetchall())
        
        conn.close()
        
        return [types.TextContent(
            type="text",
            text=json.dumps({
                "period": period,
                "total_conversations": total_conversations,
                "by_ai": by_ai,
                "by_project": by_project,
                "timestamp": datetime.now().isoformat()
            })
        )]
    
    else:
        raise ValueError(f"Outil inconnu: {name}")

# Resources
@server.list_resources()
async def handle_list_resources() -> list[Resource]:
    """List Memory resources"""
    return [
        Resource(
            uri="memory://stats",
            name="Statistiques mémoire",
            description="Statistiques d'usage de la mémoire",
            mimeType="application/json"
        ),
        Resource(
            uri="memory://recent",
            name="Conversations récentes",
            description="Dernières conversations stockées",
            mimeType="application/json"
        )
    ]

@server.read_resource()
async def handle_read_resource(uri: str) -> str:
    """Read Memory resources"""
    if uri == "memory://stats":
        conn = memory_manager.get_connection()
        cursor = conn.execute("SELECT COUNT(*) FROM conversations")
        total = cursor.fetchone()[0]
        
        cursor = conn.execute("SELECT ai_used, COUNT(*) FROM conversations GROUP BY ai_used")
        by_ai = dict(cursor.fetchall())
        
        conn.close()
        
        return json.dumps({
            "total_conversations": total,
            "by_ai": by_ai,
            "database_path": DATABASE_PATH
        })
    elif uri == "memory://recent":
        recent = await memory_manager.get_relevant_context("", None, 10)
        return json.dumps({"recent_conversations": recent})
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
                server_name="memory-mcp",
                server_version="1.0.0",
                capabilities=server.get_capabilities(
                    notification_options=NotificationOptions(),
                    experimental_capabilities={},
                ),
            ),
        )

if __name__ == "__main__":
    asyncio.run(main())