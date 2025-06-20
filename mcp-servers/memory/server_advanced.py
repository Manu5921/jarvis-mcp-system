"""
Advanced MCP Memory Server with Intelligent Context Management
Implements persistent memory with semantic search and context optimization
"""

import asyncio
import json
import logging
import sqlite3
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
import hashlib
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import uvicorn

# Configuration
MEMORY_SERVER_PORT = 4005
DATABASE_PATH = "/app/data/memory.db"
MAX_CONTEXT_SIZE = 8000  # characters
MAX_MEMORIES_PER_QUERY = 50

# Logging setup
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# MCP Protocol Models
class MCPInitRequest(BaseModel):
    client_info: Dict[str, str]

class MCPInitResponse(BaseModel):
    session_id: str
    server_info: Dict[str, str]
    capabilities: Dict[str, Any]

class MCPToolCallRequest(BaseModel):
    arguments: Dict[str, Any]
    session_id: Optional[str] = None

class MCPToolResponse(BaseModel):
    content: List[Dict[str, Any]]
    is_error: bool = False

class MCPTool(BaseModel):
    name: str
    description: str
    input_schema: Dict[str, Any]

class MCPResource(BaseModel):
    uri: str
    name: str
    description: str
    mime_type: str

# Session management
active_sessions: Dict[str, Dict] = {}

# Database initialization
def init_memory_database():
    """Initialize SQLite database for memory storage"""
    Path("/app/data").mkdir(parents=True, exist_ok=True)
    
    conn = sqlite3.connect(DATABASE_PATH)
    
    # Conversations table
    conn.execute("""
        CREATE TABLE IF NOT EXISTS conversations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT NOT NULL,
            message TEXT NOT NULL,
            response TEXT NOT NULL,
            context TEXT,
            project TEXT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            message_hash TEXT UNIQUE,
            importance_score REAL DEFAULT 0.5,
            access_count INTEGER DEFAULT 0,
            last_accessed DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    """)
    
    # Context patterns table
    conn.execute("""
        CREATE TABLE IF NOT EXISTS context_patterns (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            pattern_type TEXT NOT NULL,
            pattern_value TEXT NOT NULL,
            context_data TEXT,
            frequency INTEGER DEFAULT 1,
            effectiveness_score REAL DEFAULT 0.5,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
            updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    """)
    
    # Memory associations table
    conn.execute("""
        CREATE TABLE IF NOT EXISTS memory_associations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            memory_id INTEGER,
            associated_memory_id INTEGER,
            association_type TEXT,
            strength REAL DEFAULT 0.5,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (memory_id) REFERENCES conversations (id),
            FOREIGN KEY (associated_memory_id) REFERENCES conversations (id)
        )
    """)
    
    # User preferences table
    conn.execute("""
        CREATE TABLE IF NOT EXISTS user_preferences (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT NOT NULL,
            preference_key TEXT NOT NULL,
            preference_value TEXT NOT NULL,
            confidence REAL DEFAULT 0.5,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
            updated_at DATETIME DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(session_id, preference_key)
        )
    """)
    
    # Create indexes for performance (after table creation)
    try:
        conn.execute("CREATE INDEX IF NOT EXISTS idx_conversations_session ON conversations(session_id)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_conversations_project ON conversations(project)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_conversations_timestamp ON conversations(timestamp)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_conversations_hash ON conversations(message_hash)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_patterns_type ON context_patterns(pattern_type)")
    except sqlite3.OperationalError as e:
        logger.warning(f"Index creation warning: {e}")
    
    conn.commit()
    conn.close()
    logger.info("✅ Memory database initialized")

# Initialize database on startup (moved to app startup event)

# FastAPI app
app = FastAPI(
    title="Advanced MCP Memory Server",
    version="2.0.0",
    description="Intelligent memory management with context optimization"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.on_event("startup")
async def startup_event():
    """Initialize database on app startup"""
    init_memory_database()

# MCP Protocol Endpoints
@app.post("/mcp/initialize")
async def initialize_mcp_session(request: MCPInitRequest) -> MCPInitResponse:
    """Initialize MCP session with memory capabilities"""
    session_id = str(uuid.uuid4())
    
    active_sessions[session_id] = {
        "created_at": datetime.now(),
        "client_info": request.client_info,
        "memory_access_count": 0,
        "context_optimizations": 0,
        "last_activity": datetime.now()
    }
    
    logger.info(f"✅ Memory MCP Session initialized: {session_id}")
    
    return MCPInitResponse(
        session_id=session_id,
        server_info={
            "name": "jarvis-memory-server",
            "version": "2.0.0",
            "description": "Intelligent memory management with context optimization"
        },
        capabilities={
            "tools": {
                "list_changed": True,
                "supports_progress": True
            },
            "resources": {
                "list_changed": True,
                "supports_templates": True
            },
            "experimental": {
                "semantic_search": True,
                "context_optimization": True,
                "pattern_learning": True
            }
        }
    )

@app.get("/mcp/tools")
async def list_mcp_tools():
    """List all available memory tools"""
    tools = [
        MCPTool(
            name="store_conversation",
            description="Store a conversation in memory with intelligent context extraction",
            input_schema={
                "type": "object",
                "properties": {
                    "session_id": {"type": "string", "description": "Session identifier"},
                    "message": {"type": "string", "description": "User message"},
                    "response": {"type": "string", "description": "AI response"},
                    "context": {"type": "object", "description": "Additional context"},
                    "project": {"type": "string", "description": "Project name"},
                    "importance": {"type": "number", "default": 0.5, "description": "Importance score (0-1)"}
                },
                "required": ["session_id", "message", "response"]
            }
        ),
        MCPTool(
            name="get_context",
            description="Retrieve relevant context for a query using semantic matching",
            input_schema={
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Query to find context for"},
                    "session_id": {"type": "string", "description": "Session identifier"},
                    "project": {"type": "string", "description": "Project filter"},
                    "max_results": {"type": "integer", "default": 10, "description": "Maximum results"},
                    "time_range_days": {"type": "integer", "default": 30, "description": "Time range in days"}
                },
                "required": ["query"]
            }
        ),
        MCPTool(
            name="search_memories",
            description="Search through stored memories with advanced filtering",
            input_schema={
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search query"},
                    "session_id": {"type": "string", "description": "Session filter"},
                    "project": {"type": "string", "description": "Project filter"},
                    "date_from": {"type": "string", "description": "Start date (ISO format)"},
                    "date_to": {"type": "string", "description": "End date (ISO format)"},
                    "min_importance": {"type": "number", "default": 0.0, "description": "Minimum importance score"},
                    "max_results": {"type": "integer", "default": 20, "description": "Maximum results"}
                },
                "required": ["query"]
            }
        ),
        MCPTool(
            name="update_memory_importance",
            description="Update the importance score of a memory",
            input_schema={
                "type": "object",
                "properties": {
                    "memory_id": {"type": "integer", "description": "Memory ID"},
                    "importance_score": {"type": "number", "description": "New importance score (0-1)"},
                    "reason": {"type": "string", "description": "Reason for update"}
                },
                "required": ["memory_id", "importance_score"]
            }
        ),
        MCPTool(
            name="learn_pattern",
            description="Learn a new context pattern from successful interactions",
            input_schema={
                "type": "object",
                "properties": {
                    "pattern_type": {"type": "string", "description": "Type of pattern (e.g., 'coding', 'debug', 'research')"},
                    "pattern_value": {"type": "string", "description": "Pattern identifier or key"},
                    "context_data": {"type": "object", "description": "Context data that worked well"},
                    "effectiveness_score": {"type": "number", "default": 0.7, "description": "How effective this pattern was"}
                },
                "required": ["pattern_type", "pattern_value", "context_data"]
            }
        ),
        MCPTool(
            name="get_conversation_history",
            description="Get conversation history for a session or project",
            input_schema={
                "type": "object",
                "properties": {
                    "session_id": {"type": "string", "description": "Session identifier"},
                    "project": {"type": "string", "description": "Project name"},
                    "limit": {"type": "integer", "default": 50, "description": "Maximum conversations"},
                    "include_context": {"type": "boolean", "default": True, "description": "Include context data"}
                },
                "required": []
            }
        ),
        MCPTool(
            name="optimize_context",
            description="Optimize context for better AI responses based on learned patterns",
            input_schema={
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Current query"},
                    "current_context": {"type": "object", "description": "Current context"},
                    "target_ai": {"type": "string", "description": "Target AI system"},
                    "session_id": {"type": "string", "description": "Session identifier"}
                },
                "required": ["query", "current_context"]
            }
        ),
        MCPTool(
            name="get_memory_stats",
            description="Get statistics about stored memories and patterns",
            input_schema={
                "type": "object",
                "properties": {
                    "session_id": {"type": "string", "description": "Session filter"},
                    "project": {"type": "string", "description": "Project filter"},
                    "days_back": {"type": "integer", "default": 30, "description": "Days to analyze"}
                },
                "required": []
            }
        ),
        MCPTool(
            name="create_memory_association",
            description="Create associations between related memories",
            input_schema={
                "type": "object",
                "properties": {
                    "memory_id": {"type": "integer", "description": "Primary memory ID"},
                    "associated_memory_id": {"type": "integer", "description": "Associated memory ID"},
                    "association_type": {"type": "string", "description": "Type of association"},
                    "strength": {"type": "number", "default": 0.5, "description": "Association strength (0-1)"}
                },
                "required": ["memory_id", "associated_memory_id", "association_type"]
            }
        ),
        MCPTool(
            name="cleanup_old_memories",
            description="Clean up old, low-importance memories to maintain performance",
            input_schema={
                "type": "object",
                "properties": {
                    "days_old": {"type": "integer", "default": 90, "description": "Days old threshold"},
                    "max_importance": {"type": "number", "default": 0.3, "description": "Maximum importance to delete"},
                    "dry_run": {"type": "boolean", "default": True, "description": "Only show what would be deleted"}
                },
                "required": []
            }
        )
    ]
    
    return {"tools": [tool.dict() for tool in tools]}

@app.get("/mcp/resources")
async def list_mcp_resources():
    """List available memory resources"""
    resources = [
        MCPResource(
            uri="docs://memory-server",
            name="Memory Server Documentation",
            description="Complete documentation for memory management",
            mime_type="text/markdown"
        ),
        MCPResource(
            uri="stats://memory-usage",
            name="Memory Usage Statistics",
            description="Current memory usage and optimization stats",
            mime_type="application/json"
        ),
        MCPResource(
            uri="patterns://learned",
            name="Learned Patterns",
            description="Patterns learned from successful interactions",
            mime_type="application/json"
        )
    ]
    
    return {"resources": [resource.dict() for resource in resources]}

# Tool Implementation Endpoints
@app.post("/mcp/tools/store_conversation/call")
async def tool_store_conversation(request: MCPToolCallRequest) -> MCPToolResponse:
    """Store conversation with intelligent context extraction"""
    try:
        session_id = request.arguments["session_id"]
        message = request.arguments["message"]
        response = request.arguments["response"]
        context = request.arguments.get("context", {})
        project = request.arguments.get("project")
        importance = request.arguments.get("importance", 0.5)
        
        # Generate hash for deduplication
        message_hash = hashlib.md5(f"{message}:{response}".encode()).hexdigest()
        
        conn = sqlite3.connect(DATABASE_PATH)
        
        try:
            # Store conversation
            cursor = conn.execute("""
                INSERT INTO conversations 
                (session_id, message, response, context, project, message_hash, importance_score)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (session_id, message, response, json.dumps(context), project, message_hash, importance))
            
            memory_id = cursor.lastrowid
            
            # Update session stats
            if session_id in active_sessions:
                active_sessions[session_id]["memory_access_count"] += 1
                active_sessions[session_id]["last_activity"] = datetime.now()
            
            conn.commit()
            
            # Learn patterns from this interaction
            await _learn_patterns_from_conversation(message, response, context, project)
            
            return MCPToolResponse(
                content=[{
                    "type": "text",
                    "text": json.dumps({
                        "status": "stored",
                        "memory_id": memory_id,
                        "session_id": session_id,
                        "importance_score": importance,
                        "timestamp": datetime.now().isoformat()
                    }, indent=2)
                }]
            )
            
        except sqlite3.IntegrityError:
            # Duplicate found, update instead
            conn.execute("""
                UPDATE conversations 
                SET access_count = access_count + 1, 
                    last_accessed = CURRENT_TIMESTAMP,
                    importance_score = MAX(importance_score, ?)
                WHERE message_hash = ?
            """, (importance, message_hash))
            conn.commit()
            
            return MCPToolResponse(
                content=[{
                    "type": "text",
                    "text": json.dumps({
                        "status": "updated",
                        "message": "Conversation already exists, updated access count and importance",
                        "session_id": session_id
                    }, indent=2)
                }]
            )
        
        finally:
            conn.close()
            
    except Exception as e:
        logger.error(f"Error storing conversation: {e}")
        return MCPToolResponse(
            content=[{
                "type": "text",
                "text": f"Error storing conversation: {str(e)}"
            }],
            is_error=True
        )

@app.post("/mcp/tools/get_context/call")
async def tool_get_context(request: MCPToolCallRequest) -> MCPToolResponse:
    """Get relevant context using semantic matching"""
    try:
        query = request.arguments["query"]
        session_id = request.arguments.get("session_id")
        project = request.arguments.get("project")
        max_results = request.arguments.get("max_results", 10)
        time_range_days = request.arguments.get("time_range_days", 30)
        
        conn = sqlite3.connect(DATABASE_PATH)
        
        # Build query with filters
        sql_query = """
            SELECT id, message, response, context, project, timestamp, importance_score, access_count
            FROM conversations 
            WHERE timestamp > datetime('now', '-{} days')
        """.format(time_range_days)
        
        params = []
        
        if session_id:
            sql_query += " AND session_id = ?"
            params.append(session_id)
        
        if project:
            sql_query += " AND project = ?"
            params.append(project)
        
        # Simple text matching for now (can be enhanced with embeddings)
        query_words = query.lower().split()
        if query_words:
            word_conditions = []
            for word in query_words:
                word_conditions.append("(LOWER(message) LIKE ? OR LOWER(response) LIKE ?)")
                params.extend([f"%{word}%", f"%{word}%"])
            
            sql_query += " AND (" + " OR ".join(word_conditions) + ")"
        
        # Order by relevance (importance + recency + access count)
        sql_query += """ 
            ORDER BY (importance_score * 0.4 + 
                     (julianday('now') - julianday(timestamp)) / -365.0 * 0.3 + 
                     access_count / 100.0 * 0.3) DESC
            LIMIT ?
        """
        params.append(max_results)
        
        cursor = conn.execute(sql_query, params)
        results = cursor.fetchall()
        
        # Format results
        context_items = []
        for row in results:
            context_data = json.loads(row[3]) if row[3] else {}
            context_items.append({
                "memory_id": row[0],
                "message": row[1][:200] + "..." if len(row[1]) > 200 else row[1],
                "response": row[2][:300] + "..." if len(row[2]) > 300 else row[2],
                "context": context_data,
                "project": row[4],
                "timestamp": row[5],
                "importance_score": row[6],
                "access_count": row[7]
            })
            
            # Update access count
            conn.execute("""
                UPDATE conversations 
                SET access_count = access_count + 1, last_accessed = CURRENT_TIMESTAMP
                WHERE id = ?
            """, (row[0],))
        
        conn.commit()
        conn.close()
        
        # Build optimized context
        optimized_context = _build_optimized_context(context_items, query)
        
        return MCPToolResponse(
            content=[{
                "type": "text",
                "text": json.dumps({
                    "query": query,
                    "context_items": context_items,
                    "optimized_context": optimized_context,
                    "total_found": len(context_items),
                    "timestamp": datetime.now().isoformat()
                }, indent=2)
            }]
        )
        
    except Exception as e:
        logger.error(f"Error getting context: {e}")
        return MCPToolResponse(
            content=[{
                "type": "text",
                "text": f"Error getting context: {str(e)}"
            }],
            is_error=True
        )

@app.post("/mcp/tools/search_memories/call")
async def tool_search_memories(request: MCPToolCallRequest) -> MCPToolResponse:
    """Search through stored memories with advanced filtering"""
    try:
        query = request.arguments["query"]
        session_id = request.arguments.get("session_id")
        project = request.arguments.get("project")
        date_from = request.arguments.get("date_from")
        date_to = request.arguments.get("date_to")
        min_importance = request.arguments.get("min_importance", 0.0)
        max_results = request.arguments.get("max_results", 20)
        
        conn = sqlite3.connect(DATABASE_PATH)
        
        # Build advanced search query
        sql_query = "SELECT * FROM conversations WHERE 1=1"
        params = []
        
        # Text search
        if query:
            sql_query += " AND (LOWER(message) LIKE ? OR LOWER(response) LIKE ?)"
            params.extend([f"%{query.lower()}%", f"%{query.lower()}%"])
        
        # Filters
        if session_id:
            sql_query += " AND session_id = ?"
            params.append(session_id)
        
        if project:
            sql_query += " AND project = ?"
            params.append(project)
        
        if date_from:
            sql_query += " AND timestamp >= ?"
            params.append(date_from)
        
        if date_to:
            sql_query += " AND timestamp <= ?"
            params.append(date_to)
        
        if min_importance > 0:
            sql_query += " AND importance_score >= ?"
            params.append(min_importance)
        
        sql_query += " ORDER BY importance_score DESC, timestamp DESC LIMIT ?"
        params.append(max_results)
        
        cursor = conn.execute(sql_query, params)
        columns = [description[0] for description in cursor.description]
        results = []
        
        for row in cursor.fetchall():
            memory = dict(zip(columns, row))
            if memory['context']:
                memory['context'] = json.loads(memory['context'])
            results.append(memory)
        
        conn.close()
        
        return MCPToolResponse(
            content=[{
                "type": "text",
                "text": json.dumps({
                    "query": query,
                    "results": results,
                    "total_found": len(results),
                    "search_params": {
                        "session_id": session_id,
                        "project": project,
                        "date_from": date_from,
                        "date_to": date_to,
                        "min_importance": min_importance
                    }
                }, indent=2)
            }]
        )
        
    except Exception as e:
        logger.error(f"Error searching memories: {e}")
        return MCPToolResponse(
            content=[{
                "type": "text",
                "text": f"Error searching memories: {str(e)}"
            }],
            is_error=True
        )

@app.post("/mcp/tools/optimize_context/call")
async def tool_optimize_context(request: MCPToolCallRequest) -> MCPToolResponse:
    """Optimize context based on learned patterns"""
    try:
        query = request.arguments["query"]
        current_context = request.arguments["current_context"]
        target_ai = request.arguments.get("target_ai", "general")
        session_id = request.arguments.get("session_id")
        
        conn = sqlite3.connect(DATABASE_PATH)
        
        # Get learned patterns for this type of query
        query_type = _classify_query_type(query)
        
        cursor = conn.execute("""
            SELECT pattern_value, context_data, effectiveness_score, frequency
            FROM context_patterns 
            WHERE pattern_type = ? 
            ORDER BY effectiveness_score DESC, frequency DESC
            LIMIT 5
        """, (query_type,))
        
        patterns = cursor.fetchall()
        
        # Build optimized context
        optimized_context = dict(current_context)
        
        # Apply learned patterns
        optimizations_applied = []
        for pattern in patterns:
            pattern_data = json.loads(pattern[1])
            effectiveness = pattern[2]
            
            if effectiveness > 0.6:  # Only apply high-effectiveness patterns
                for key, value in pattern_data.items():
                    if key not in optimized_context or effectiveness > 0.8:
                        optimized_context[key] = value
                        optimizations_applied.append({
                            "pattern": pattern[0],
                            "key": key,
                            "effectiveness": effectiveness
                        })
        
        # Add session-specific optimizations
        if session_id in active_sessions:
            active_sessions[session_id]["context_optimizations"] += 1
        
        conn.close()
        
        return MCPToolResponse(
            content=[{
                "type": "text",
                "text": json.dumps({
                    "original_context": current_context,
                    "optimized_context": optimized_context,
                    "query_type": query_type,
                    "optimizations_applied": optimizations_applied,
                    "target_ai": target_ai,
                    "confidence": _calculate_optimization_confidence(optimizations_applied)
                }, indent=2)
            }]
        )
        
    except Exception as e:
        logger.error(f"Error optimizing context: {e}")
        return MCPToolResponse(
            content=[{
                "type": "text",
                "text": f"Error optimizing context: {str(e)}"
            }],
            is_error=True
        )

@app.post("/mcp/tools/get_memory_stats/call")
async def tool_get_memory_stats(request: MCPToolCallRequest) -> MCPToolResponse:
    """Get comprehensive memory statistics"""
    try:
        session_id = request.arguments.get("session_id")
        project = request.arguments.get("project")
        days_back = request.arguments.get("days_back", 30)
        
        conn = sqlite3.connect(DATABASE_PATH)
        
        # Basic stats
        stats = {
            "total_memories": 0,
            "recent_memories": 0,
            "average_importance": 0.0,
            "most_accessed": [],
            "project_breakdown": {},
            "pattern_stats": {},
            "session_activity": {}
        }
        
        # Total memories
        cursor = conn.execute("SELECT COUNT(*) FROM conversations")
        stats["total_memories"] = cursor.fetchone()[0]
        
        # Recent memories
        cursor = conn.execute("""
            SELECT COUNT(*) FROM conversations 
            WHERE timestamp > datetime('now', '-{} days')
        """.format(days_back))
        stats["recent_memories"] = cursor.fetchone()[0]
        
        # Average importance
        cursor = conn.execute("SELECT AVG(importance_score) FROM conversations")
        result = cursor.fetchone()[0]
        stats["average_importance"] = result if result else 0.0
        
        # Most accessed memories
        cursor = conn.execute("""
            SELECT message, access_count, importance_score 
            FROM conversations 
            ORDER BY access_count DESC 
            LIMIT 5
        """)
        stats["most_accessed"] = [
            {"message": row[0][:100] + "...", "access_count": row[1], "importance": row[2]}
            for row in cursor.fetchall()
        ]
        
        # Project breakdown
        cursor = conn.execute("""
            SELECT project, COUNT(*), AVG(importance_score) 
            FROM conversations 
            WHERE project IS NOT NULL 
            GROUP BY project
        """)
        stats["project_breakdown"] = {
            row[0]: {"count": row[1], "avg_importance": row[2]}
            for row in cursor.fetchall()
        }
        
        # Pattern stats
        cursor = conn.execute("""
            SELECT pattern_type, COUNT(*), AVG(effectiveness_score) 
            FROM context_patterns 
            GROUP BY pattern_type
        """)
        stats["pattern_stats"] = {
            row[0]: {"count": row[1], "avg_effectiveness": row[2]}
            for row in cursor.fetchall()
        }
        
        conn.close()
        
        return MCPToolResponse(
            content=[{
                "type": "text",
                "text": json.dumps(stats, indent=2)
            }]
        )
        
    except Exception as e:
        logger.error(f"Error getting memory stats: {e}")
        return MCPToolResponse(
            content=[{
                "type": "text",
                "text": f"Error getting memory stats: {str(e)}"
            }],
            is_error=True
        )

# Helper functions
async def _learn_patterns_from_conversation(message: str, response: str, context: Dict, project: str):
    """Learn patterns from successful conversations"""
    try:
        query_type = _classify_query_type(message)
        
        # Extract useful context patterns
        if context and len(response) > 50:  # Assume good response if reasonably long
            conn = sqlite3.connect(DATABASE_PATH)
            
            # Check if this pattern already exists
            cursor = conn.execute("""
                SELECT id, frequency, effectiveness_score 
                FROM context_patterns 
                WHERE pattern_type = ? AND pattern_value = ?
            """, (query_type, project or "general"))
            
            existing = cursor.fetchone()
            
            if existing:
                # Update existing pattern
                new_frequency = existing[1] + 1
                new_effectiveness = min(1.0, existing[2] + 0.1)  # Slowly increase effectiveness
                
                conn.execute("""
                    UPDATE context_patterns 
                    SET frequency = ?, effectiveness_score = ?, updated_at = CURRENT_TIMESTAMP
                    WHERE id = ?
                """, (new_frequency, new_effectiveness, existing[0]))
            else:
                # Create new pattern
                conn.execute("""
                    INSERT INTO context_patterns 
                    (pattern_type, pattern_value, context_data, effectiveness_score)
                    VALUES (?, ?, ?, ?)
                """, (query_type, project or "general", json.dumps(context), 0.6))
            
            conn.commit()
            conn.close()
            
    except Exception as e:
        logger.error(f"Error learning patterns: {e}")

def _classify_query_type(query: str) -> str:
    """Classify query type for pattern learning"""
    query_lower = query.lower()
    
    if any(word in query_lower for word in ["bug", "error", "debug", "fix", "broken"]):
        return "debug"
    elif any(word in query_lower for word in ["how to", "tutorial", "guide", "explain"]):
        return "learning"
    elif any(word in query_lower for word in ["code", "function", "class", "implement", "create"]):
        return "coding"
    elif any(word in query_lower for word in ["optimize", "improve", "performance", "better"]):
        return "optimization"
    elif any(word in query_lower for word in ["research", "find", "search", "information"]):
        return "research"
    else:
        return "general"

def _build_optimized_context(context_items: List[Dict], query: str) -> str:
    """Build optimized context string from memory items"""
    if not context_items:
        return ""
    
    # Prioritize items and build context
    context_parts = ["=== Relevant Context ==="]
    
    for item in context_items[:5]:  # Top 5 most relevant
        context_parts.append(f"• {item['message'][:100]}...")
        if item.get('project'):
            context_parts.append(f"  Project: {item['project']}")
    
    context_str = "\n".join(context_parts)
    
    # Truncate if too long
    if len(context_str) > MAX_CONTEXT_SIZE:
        context_str = context_str[:MAX_CONTEXT_SIZE] + "...\n[Context truncated]"
    
    return context_str

def _calculate_optimization_confidence(optimizations: List[Dict]) -> float:
    """Calculate confidence in context optimization"""
    if not optimizations:
        return 0.0
    
    total_effectiveness = sum(opt["effectiveness"] for opt in optimizations)
    return min(1.0, total_effectiveness / len(optimizations))

@app.post("/mcp/close")
async def close_mcp_session(request: Dict[str, str]):
    """Close MCP session"""
    session_id = request.get("session_id")
    if session_id in active_sessions:
        session_stats = active_sessions[session_id]
        logger.info(f"🔌 Memory Session closed: {session_id} "
                   f"(accessed: {session_stats['memory_access_count']}, "
                   f"optimizations: {session_stats['context_optimizations']})")
        del active_sessions[session_id]
    
    return {"status": "closed", "session_id": session_id}

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    conn = sqlite3.connect(DATABASE_PATH)
    cursor = conn.execute("SELECT COUNT(*) FROM conversations")
    memory_count = cursor.fetchone()[0]
    conn.close()
    
    return {
        "status": "healthy",
        "server": "advanced-mcp-memory",
        "version": "2.0.0",
        "active_sessions": len(active_sessions),
        "total_memories": memory_count,
        "timestamp": datetime.now().isoformat()
    }

if __name__ == "__main__":
    logger.info("🧠 Starting Advanced MCP Memory Server...")
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=MEMORY_SERVER_PORT,
        log_level="info"
    )