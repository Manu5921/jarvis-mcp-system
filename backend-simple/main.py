"""
AI Memory Hub - Backend simple pour usage personnel
Gère l'historique et optimise les prompts entre AIs
"""

import sqlite3
import json
import requests
from datetime import datetime
from typing import List, Dict, Optional
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os

# Configuration
app = FastAPI(title="AI Memory Hub", version="1.0.0")
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "localhost:11434")
PERPLEXITY_API_KEY = os.getenv("PERPLEXITY_API_KEY", "")
DB_PATH = "/app/data/ai_memory.db"

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Models
class ConversationCreate(BaseModel):
    query: str
    ai_target: str  # "ollama", "claude", "perplexity", "chatgpt"
    project: Optional[str] = None
    context: Optional[str] = None

class ConversationResponse(BaseModel):
    id: int
    query: str
    response: str
    ai_used: str
    project: Optional[str]
    timestamp: str
    optimized_prompt: Optional[str] = None

# Database setup
def init_db():
    """Initialize SQLite database"""
    Path("/app/data").mkdir(exist_ok=True)
    
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS conversations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            query TEXT NOT NULL,
            response TEXT NOT NULL,
            ai_used TEXT NOT NULL,
            project TEXT,
            context TEXT,
            optimized_prompt TEXT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.commit()
    conn.close()

init_db()

# AI Clients
class OllamaClient:
    def __init__(self):
        self.base_url = f"http://{OLLAMA_HOST}"
    
    def generate(self, prompt: str, model: str = "llama3.2:8b") -> str:
        try:
            response = requests.post(
                f"{self.base_url}/api/generate",
                json={
                    "model": model,
                    "prompt": prompt,
                    "stream": False
                },
                timeout=60
            )
            return response.json().get("response", "Erreur Ollama")
        except Exception as e:
            return f"Erreur Ollama: {str(e)}"

class PerplexityClient:
    def __init__(self):
        self.api_key = PERPLEXITY_API_KEY
        self.base_url = "https://api.perplexity.ai/chat/completions"
    
    def search(self, query: str) -> str:
        if not self.api_key:
            return "Clé API Perplexity manquante"
        
        try:
            response = requests.post(
                self.base_url,
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json"
                },
                json={
                    "model": "llama-3.1-sonar-small-128k-online",
                    "messages": [{"role": "user", "content": query}],
                    "max_tokens": 1000
                },
                timeout=30
            )
            return response.json()["choices"][0]["message"]["content"]
        except Exception as e:
            return f"Erreur Perplexity: {str(e)}"

# Clients
ollama = OllamaClient()
perplexity = PerplexityClient()

# Prompt optimization logic
def get_optimized_prompt(query: str, ai_target: str, context: str = None) -> str:
    """Generate optimized prompts based on AI target and context"""
    
    base_context = ""
    if context:
        base_context = f"Contexte du projet: {context}\n\n"
    
    # Get recent relevant conversations for context
    recent_context = get_recent_context(query)
    if recent_context:
        base_context += f"Historique pertinent:\n{recent_context}\n\n"
    
    optimizations = {
        "claude": f"""{base_context}Tu es Claude Code, assistant de développement. 
Réponds de manière concise et structurée.
Instructions spécifiques: {query}""",
        
        "ollama": f"""{base_context}Tu es un assistant de développement local expert.
Concentre-toi sur du code pratique et des solutions directes.
Question: {query}""",
        
        "perplexity": f"""Recherche et analyse des informations récentes sur: {query}
Fournis des sources fiables et des insights techniques actualisés.""",
        
        "chatgpt": f"""{base_context}Tu es un assistant IA polyvalent. 
Fournis une réponse équilibrée entre théorie et pratique.
Demande: {query}"""
    }
    
    return optimizations.get(ai_target, query)

def get_recent_context(query: str, limit: int = 3) -> str:
    """Get recent relevant conversations for context"""
    conn = sqlite3.connect(DB_PATH)
    
    # Simple keyword matching for relevance
    keywords = query.lower().split()[:3]  # Take first 3 words
    keyword_conditions = " OR ".join([f"query LIKE '%{kw}%'" for kw in keywords])
    
    cursor = conn.execute(f"""
        SELECT query, response, ai_used 
        FROM conversations 
        WHERE {keyword_conditions}
        ORDER BY timestamp DESC 
        LIMIT ?
    """, (limit,))
    
    results = cursor.fetchall()
    conn.close()
    
    if not results:
        return ""
    
    context_parts = []
    for query_prev, response_prev, ai_used in results:
        context_parts.append(f"- {ai_used}: {query_prev[:100]}... → {response_prev[:150]}...")
    
    return "\n".join(context_parts)

# API Endpoints
@app.get("/")
async def root():
    return {"message": "AI Memory Hub API", "status": "running"}

@app.get("/conversations", response_model=List[ConversationResponse])
async def get_conversations(limit: int = 50, project: Optional[str] = None):
    """Get conversation history"""
    conn = sqlite3.connect(DB_PATH)
    
    if project:
        cursor = conn.execute("""
            SELECT * FROM conversations 
            WHERE project = ? 
            ORDER BY timestamp DESC 
            LIMIT ?
        """, (project, limit))
    else:
        cursor = conn.execute("""
            SELECT * FROM conversations 
            ORDER BY timestamp DESC 
            LIMIT ?
        """, (limit,))
    
    results = cursor.fetchall()
    conn.close()
    
    conversations = []
    for row in results:
        conversations.append(ConversationResponse(
            id=row[0],
            query=row[1],
            response=row[2],
            ai_used=row[3],
            project=row[4],
            timestamp=row[7],
            optimized_prompt=row[6]
        ))
    
    return conversations

@app.post("/chat/ollama")
async def chat_ollama(conv: ConversationCreate):
    """Chat with Ollama local models"""
    optimized_prompt = get_optimized_prompt(conv.query, "ollama", conv.context)
    
    # Use 3b for quick responses, 8b for complex tasks
    model = "llama3.2:3b" if len(conv.query) < 100 else "llama3.2:8b"
    response = ollama.generate(optimized_prompt, model)
    
    # Save to database
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.execute("""
        INSERT INTO conversations (query, response, ai_used, project, context, optimized_prompt)
        VALUES (?, ?, ?, ?, ?, ?)
    """, (conv.query, response, f"ollama-{model}", conv.project, conv.context, optimized_prompt))
    
    conversation_id = cursor.lastrowid
    conn.commit()
    conn.close()
    
    return {
        "id": conversation_id,
        "response": response,
        "model_used": model,
        "optimized_prompt": optimized_prompt
    }

@app.post("/search/perplexity")
async def search_perplexity(conv: ConversationCreate):
    """Search with Perplexity Pro"""
    optimized_prompt = get_optimized_prompt(conv.query, "perplexity", conv.context)
    response = perplexity.search(optimized_prompt)
    
    # Save to database
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.execute("""
        INSERT INTO conversations (query, response, ai_used, project, context, optimized_prompt)
        VALUES (?, ?, ?, ?, ?, ?)
    """, (conv.query, response, "perplexity-pro", conv.project, conv.context, optimized_prompt))
    
    conversation_id = cursor.lastrowid
    conn.commit()
    conn.close()
    
    return {
        "id": conversation_id,
        "response": response,
        "optimized_prompt": optimized_prompt
    }

@app.post("/prompt/optimize")
async def optimize_prompt(conv: ConversationCreate):
    """Generate optimized prompt for external AI (Claude Code, ChatGPT, etc.)"""
    optimized_prompt = get_optimized_prompt(conv.query, conv.ai_target, conv.context)
    
    # Save the optimization request
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.execute("""
        INSERT INTO conversations (query, response, ai_used, project, context, optimized_prompt)
        VALUES (?, ?, ?, ?, ?, ?)
    """, (conv.query, "Prompt généré", f"optimize-{conv.ai_target}", conv.project, conv.context, optimized_prompt))
    
    conversation_id = cursor.lastrowid
    conn.commit()
    conn.close()
    
    return {
        "id": conversation_id,
        "optimized_prompt": optimized_prompt,
        "copy_ready": True
    }

@app.get("/stats")
async def get_stats():
    """Get usage statistics"""
    conn = sqlite3.connect(DB_PATH)
    
    # Count by AI
    cursor = conn.execute("""
        SELECT ai_used, COUNT(*) as count 
        FROM conversations 
        GROUP BY ai_used 
        ORDER BY count DESC
    """)
    ai_usage = dict(cursor.fetchall())
    
    # Recent activity
    cursor = conn.execute("""
        SELECT COUNT(*) 
        FROM conversations 
        WHERE timestamp > datetime('now', '-7 days')
    """)
    recent_count = cursor.fetchone()[0]
    
    # Top projects
    cursor = conn.execute("""
        SELECT project, COUNT(*) as count 
        FROM conversations 
        WHERE project IS NOT NULL 
        GROUP BY project 
        ORDER BY count DESC 
        LIMIT 5
    """)
    top_projects = dict(cursor.fetchall())
    
    conn.close()
    
    return {
        "ai_usage": ai_usage,
        "recent_activity": recent_count,
        "top_projects": top_projects
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)