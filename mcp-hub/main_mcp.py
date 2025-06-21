"""
Jarvis MCP Hub - Advanced MCP Orchestrator with Real-time Communication
Manages intelligent MCP communications between AIs and tools with WebSocket support
Enhanced with validation middleware for forced tool usage
"""

import asyncio
import json
import logging
import sqlite3
import time
import uuid
import os
from contextlib import asynccontextmanager
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Union, Set, Tuple
from pathlib import Path

from fastapi import FastAPI, HTTPException, status, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
import uvicorn
import httpx

# Import validation middleware et couche de résilience
from validation_middleware import validation_middleware
from resilience_layer import resilience_layer

# Configuration
DATABASE_PATH = "/app/data/mcp_hub.db"
OLLAMA_HOST = "host.docker.internal:11434"
MCP_HUB_PORT = 4000

# Prompt interpreter for automatic tool detection
def interpret_prompt(prompt: str) -> Dict[str, Any]:
    """Automatically detect required tools from prompt text"""
    tools = []
    metadata = {}
    
    prompt_lower = prompt.lower()
    
    # 🕵️ DEBUG: Log interpret_prompt call
    logger.error(f"🕵️ INTERPRET_PROMPT CALLED → prompt_length: {len(prompt)} | contains_exigence: {'EXIGENCE:' in prompt}")
    
    # Detect LS tool need
    if any(keyword in prompt_lower for keyword in ["ls", "structure", "explore", "list", "directory"]):
        logger.error(f"🕵️ INTERPRET_PROMPT ADDING LS → path: /data/digital-agency-ai")
        tools.append({
            "tool_name": "LS",
            "arguments": {"path": "/data/digital-agency-ai"}
        })
    
    # Detect Read tool need
    if any(keyword in prompt_lower for keyword in ["read", "package.json", "config", "file"]):
        if "package.json" in prompt_lower:
            tools.append({
                "tool_name": "Read", 
                "arguments": {"file_path": "/data/digital-agency-ai/package.json"}
            })
    
    # Detect Glob tool need
    if any(keyword in prompt_lower for keyword in ["glob", "pattern", "search", "find"]):
        tools.append({
            "tool_name": "Glob",
            "arguments": {"pattern": "**/*.{js,ts,json}"}
        })
    
    return {
        "required_tools": tools,
        "metadata": {
            "analysis_type": "project_exploration" if tools else "general",
            "auto_detected": True
        }
    } if tools else {}

# Security utility to sanitize paths
def sanitize_path(path: str) -> str:
    """Remove dangerous characters that could corrupt paths"""
    if not isinstance(path, str):
        return str(path)
    return path.strip().replace("\n", "").replace("\r", "").replace(":", "")

# Tool arguments normalization utility
def normalize_tool_args(tool_name: str, tool_args: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize tool arguments to fix path duplicates and sanitize"""
    normalized = tool_args.copy()
    
    if "path" in normalized:
        original = normalized["path"]
        logger.warning(f"🕵️ PRE-SANITIZE (path): TYPE='{type(normalized['path'])}', REPR='{repr(normalized['path'])}', VALUE='{normalized['path']}'")
        # Apply sanitization BEFORE normalization
        sanitized = sanitize_path(normalized["path"])
        normalized["path"] = normalize_path(sanitized)
        logger.warning(f"🔧 TOOL PATH FIXED → {tool_name}: '{original}' → '{normalized['path']}'")
    
    if "file_path" in normalized:
        original = normalized["file_path"]
        logger.warning(f"🕵️ PRE-SANITIZE (file_path): TYPE='{type(normalized['file_path'])}', REPR='{repr(normalized['file_path'])}', VALUE='{normalized['file_path']}'")
        # Apply sanitization BEFORE normalization
        sanitized = sanitize_path(normalized["file_path"])
        normalized["file_path"] = normalize_path(sanitized)
        logger.warning(f"🔧 TOOL FILE_PATH FIXED → {tool_name}: '{original}' → '{normalized['file_path']}'")
    
    return normalized

# Path normalization utility
def normalize_path(path: str, root: str = "/data") -> str:
    """
    Normalize paths to prevent duplicates like /data/digital-agency-ai/digital-agency-ai
    - Handles absolute vs relative paths correctly
    - Removes consecutive duplicate segments
    - Ensures result stays under root
    """
    if not path:
        return path
    
    import os
    
    # 1) If path is already absolute, don't re-prefix
    if os.path.isabs(path):
        candidate = os.path.normpath(path)
    else:
        candidate = os.path.normpath(os.path.join(root, path))
    
    # 2) Remove consecutive duplicates like /data/foo/foo
    parts = []
    for part in candidate.split(os.sep):
        if part and (not parts or parts[-1] != part):
            parts.append(part)
    
    result = os.sep.join(parts)
    if candidate.startswith('/'):
        result = '/' + result
    
    logger.info(f"🔧 PATH NORMALIZE: '{path}' → '{result}'")
    return result

import os # Ensure os is imported at the top of the file if not already

def extract_clean_path(message: str, keyword_to_find: str) -> Optional[str]:
    '''
    Extracts a path segment that starts with keyword_to_find from a message string.
    The path is assumed to follow the keyword and is terminated by the earliest of space, newline, or colon.
    Returns the normalized path segment including the keyword_to_find if found.
    e.g., message="...ls digital-agency-ai/src ...", keyword_to_find="digital-agency-ai" -> "digital-agency-ai/src"
    '''
    start_idx = message.find(keyword_to_find)
    if start_idx == -1:
        return None

    # Determine the end of the path string from start_idx
    space_end = message.find(" ", start_idx)
    newline_end = message.find("\n", start_idx) # Use actual newline character
    colon_end = message.find(":", start_idx)

    possible_ends = [e for e in [space_end, newline_end, colon_end] if e != -1]

    end_idx = len(message) # Default to end of message if no delimiters found
    if possible_ends:
        end_idx = min(possible_ends)

    # Extract the raw path segment
    raw_path_segment = message[start_idx:end_idx].strip()

    # Normalize the extracted path segment.
    # os.path.normpath removes trailing slashes and resolves ., .. if present.
    if raw_path_segment:
        # If keyword_to_find was absolute, raw_path_segment will be absolute.
        # If keyword_to_find was relative, raw_path_segment will be relative.
        # normpath is fine with both.
        return os.path.normpath(raw_path_segment)
    return None

# HTTP Client configuration for long AI responses
def get_resilient_http_client(read_timeout: float = 180.0) -> httpx.AsyncClient:
    """Get HTTP client optimized for long AI responses"""
    timeout = httpx.Timeout(
        connect=30.0,      # Connection timeout
        read=read_timeout, # Read timeout (long AI responses)
        write=90.0,        # Write timeout
        pool=90.0          # Pool timeout
    )
    return httpx.AsyncClient(timeout=timeout)

def calculate_mcp_validation_score(tools_used: List[str], tool_results: Dict[str, Any]) -> int:
    score = 0
    # Ensure tool_results is structured like {"LS": {"success": True, ...}}
    # Tools_used here refers to tools that were *attempted* and have an entry in tool_results.

    # Score for individual tools succeeding
    if tool_results.get("LS", {}).get("success"):
        score += 3
        # LS is in tools_used if it was attempted; success is checked via tool_results

    if tool_results.get("Read", {}).get("success"):
        score += 3

    if tool_results.get("Glob", {}).get("success"):
        score += 2

    # Bonus for multiple successful tools
    successful_tool_executions = 0
    # Iterate over actual keys in tool_results to see what was run
    for tool_name in tool_results.keys():
        if tool_results[tool_name].get("success"):
            # Consider only main tools for this specific bonus, or adjust as needed
            if tool_name in ["LS", "Read", "Glob"]:
                successful_tool_executions +=1

    if successful_tool_executions >= 2: # If at least two of the main tools succeeded
        score += 2

    return min(score, 10)

# Logging setup
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Global state
mcp_clients: Dict[str, 'MCPClient'] = {}
active_connections: List[WebSocket] = []
active_mcp_sessions: Dict[str, Dict] = {}
streaming_responses: Dict[str, asyncio.Queue] = {}

# Pydantic Models
class ChatRequest(BaseModel):
    message: str = Field(..., description="Message à envoyer à l'agent IA")
    agent: str = Field("auto", description="Agent IA: 'auto', 'ollama', 'perplexity'")
    project: Optional[str] = Field(None, description="Nom du projet pour contexte")
    context: Optional[str] = Field(None, description="Contexte additionnel")
    session_id: Optional[str] = Field(None, description="ID de session pour WebSocket")
    stream: bool = Field(False, description="Activer le streaming des réponses")
    tools_needed: List[str] = Field(default_factory=list, description="Outils MCP requis")

class ChatResponse(BaseModel):
    response: str
    agent_used: str
    optimized_prompt: Optional[str] = None
    timestamp: str
    response_time_ms: int
    conversation_id: int
    session_id: Optional[str] = None
    tools_used: List[str] = Field(default_factory=list)
    mcp_results: Dict[str, Any] = Field(default_factory=dict)
    streaming: bool = False
    # 🔧 Validation fields
    validation_score: Optional[int] = None
    validation_passed: Optional[bool] = None

class StreamingChunk(BaseModel):
    chunk: str
    session_id: str
    agent_used: str
    final: bool = False
    tools_used: List[str] = Field(default_factory=list)
    timestamp: str

class MCPToolCall(BaseModel):
    tool_name: str
    arguments: Dict[str, Any]
    server_name: str

class MCPRequest(BaseModel):
    message: str
    session_id: str
    tools: List[MCPToolCall] = Field(default_factory=list)
    context: Dict[str, Any] = Field(default_factory=dict)
    
class MCPResponse(BaseModel):
    session_id: str
    response: str
    tool_results: Dict[str, Any] = Field(default_factory=dict)
    agents_used: List[str] = Field(default_factory=list)
    processing_time_ms: int

class OptimizePromptRequest(BaseModel):
    original_prompt: str = Field(..., description="Prompt original à optimiser")
    target_ai: str = Field(..., description="IA cible: 'claude', 'chatgpt', 'perplexity'")
    context: Optional[str] = Field(None, description="Contexte du projet")
    project: Optional[str] = Field(None, description="Nom du projet")

class ConversationItem(BaseModel):
    id: int
    message: str
    response: str
    agent_used: str
    project: Optional[str]
    timestamp: str

class StatsResponse(BaseModel):
    total_conversations: int
    recent_activity: int
    ai_usage: Dict[str, int]
    avg_response_times: Dict[str, float]

# Database initialization
def init_database():
    """Initialize SQLite database for MCP Hub"""
    Path("/app/data").mkdir(parents=True, exist_ok=True)
    
    conn = sqlite3.connect(DATABASE_PATH)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS conversations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            message TEXT NOT NULL,
            response TEXT NOT NULL,
            agent_used TEXT NOT NULL,
            project TEXT,
            context TEXT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            response_time_ms INTEGER,
            success BOOLEAN DEFAULT TRUE
        )
    """)
    
    conn.execute("""
        CREATE TABLE IF NOT EXISTS agent_stats (
            agent_name TEXT PRIMARY KEY,
            total_requests INTEGER DEFAULT 0,
            total_success INTEGER DEFAULT 0,
            avg_response_time_ms REAL DEFAULT 0,
            last_used DATETIME
        )
    """)
    
    conn.commit()
    conn.close()

# MCP Client with full protocol support
class MCPClient:
    """Advanced MCP Client with full protocol communication"""
    
    def __init__(self, server_name: str, base_url: str):
        self.server_name = server_name
        self.base_url = base_url
        self.client = get_resilient_http_client(300.0)  # 5 minutes pour les analyses complexes
        self.session_id = None
        self.tools_cache = {}
        self.resources_cache = {}
        
    async def initialize(self) -> bool:
        """Initialize MCP session with full handshake"""
        try:
            response = await self.client.post(
                f"{self.base_url}/mcp/initialize",
                json={"client_info": {"name": "jarvis-mcp-hub", "version": "1.0.0"}}
            )
            response.raise_for_status()
            data = response.json()
            self.session_id = data.get("session_id")
            
            # Cache available tools and resources
            await self._refresh_capabilities()
            
            logger.info(f"✅ MCP Client initialized for {self.server_name}")
            return True
        except Exception as e:
            logger.error(f"❌ MCP Client init failed for {self.server_name}: {e}")
            return False
    
    async def _refresh_capabilities(self):
        """Refresh tools and resources cache"""
        try:
            # Get tools
            tools_response = await self.client.get(f"{self.base_url}/mcp/tools")
            if tools_response.status_code == 200:
                self.tools_cache = tools_response.json().get("tools", {})
            
            # Get resources
            resources_response = await self.client.get(f"{self.base_url}/mcp/resources")
            if resources_response.status_code == 200:
                self.resources_cache = resources_response.json().get("resources", {})
                
        except Exception as e:
            logger.warning(f"Failed to refresh capabilities for {self.server_name}: {e}")
    
    async def call_tool(self, tool_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Call a tool via MCP protocol"""
        try:
            # 🔧 NORMALISATION DÉFENSIVE : S'applique à TOUS les flux d'exécution
            logger.info(f"🧭 MCPClient.call_tool() → TOOL: {tool_name} → ARGS: {arguments}")
            normalized_arguments = normalize_tool_args(tool_name, arguments)
            
            response = await self.client.post(
                f"{self.base_url}/mcp/tools/{tool_name}/call",
                json={
                    "arguments": normalized_arguments,
                    "session_id": self.session_id
                }
            )
            response.raise_for_status()
            return response.json()
        except httpx.TimeoutException as e:
            logger.error(f"TIMEOUT calling {tool_name} on {self.server_name}: {e}")
            raise
        except httpx.RequestError as e:
            logger.error(f"REQUEST ERROR calling {tool_name} on {self.server_name}: {e}")
            raise
        except Exception as e:
            logger.error(f"UNKNOWN ERROR calling {tool_name} on {self.server_name}: {type(e).__name__}: {e}")
            raise
    
    async def get_resource(self, resource_uri: str) -> Dict[str, Any]:
        """Get a resource via MCP protocol"""
        try:
            response = await self.client.get(
                f"{self.base_url}/mcp/resources/{resource_uri}",
                params={"session_id": self.session_id}
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f"Error getting resource {resource_uri} from {self.server_name}: {e}")
            raise
    
    async def list_tools(self) -> List[Dict[str, Any]]:
        """List available tools"""
        if isinstance(self.tools_cache, dict):
            return list(self.tools_cache.values())
        elif isinstance(self.tools_cache, list):
            return self.tools_cache
        else:
            return []
    
    async def list_resources(self) -> List[Dict[str, Any]]:
        """List available resources"""
        if isinstance(self.resources_cache, dict):
            return list(self.resources_cache.values())
        elif isinstance(self.resources_cache, list):
            return self.resources_cache
        else:
            return []
    
    async def health_check(self) -> bool:
        """Check if server is healthy"""
        try:
            response = await self.client.get(f"{self.base_url}/health", timeout=5.0)
            return response.status_code == 200
        except:
            return False
    
    async def close(self):
        """Close MCP session"""
        if self.session_id:
            try:
                await self.client.post(
                    f"{self.base_url}/mcp/close",
                    json={"session_id": self.session_id}
                )
            except:
                pass
        await self.client.aclose()

# Agent managers
class AgentManager:
    """Enhanced Agent Manager with MCP integration"""
    
    @staticmethod
    async def call_ollama(message: str, context: str = None, session_id: str = None) -> Tuple[str, List[str], Dict[str, Any]]:
        """Enhanced Ollama call with MCP integration and file tools"""
        # Initialize these at the function scope to ensure they are always defined for return
        tools_successfully_used = []
        tool_execution_details = {}
        response_text = ""

        try:
            # 🔥 FORÇAGE ABSOLU - FORMATION JARVIS MCP AVEC OUTILS OBLIGATOIRES
            if "ollama" in mcp_clients:
                # tools_successfully_used and tool_execution_details are already function-scoped
                # and initialized at the start of the 'try' block.
                # No need for 'current_run_...' versions here.
                message_lower = message.lower()
                tools_to_call = []
                
                # 🎯 DÉTECTION FORCÉE - TOUTE ANALYSE = EXPLORATION OBLIGATOIRE
                analysis_triggers = [
                    "analyse", "structure", "projet", "framework", "technologies", 
                    "identifie", "quelles", "comment", "workflows", "configuration",
                    "monitoring", "integrations", "utilisé", "fonctionne",
                    "fichiers", "dossier", "racine", "principaux", "digital-agency-ai",
                    "explore", "list", "contenu", "répertoire"
                ]
                
                is_analysis_request = any(trigger in message_lower for trigger in analysis_triggers)
                logger.info(f"🔍 FORMATION: Analyse détectée: {is_analysis_request}")
                
                # Initialize variables for proper scope
                tools_to_call = []
                target_path = "/data/digital-agency-ai"  # Default target path
                
                if is_analysis_request:
                    # 💥 FORÇAGE SYSTÉMATIQUE DES OUTILS MCP
                    logger.info("🚀 FORMATION: Forçage exploration MCP obligatoire")
                    
                    # 📁 ÉTAPE 1: DÉTECTION INTELLIGENTE DU CHEMIN
                    target_path = "/data/digital-agency-ai"  # Défaut
                    
                    # Patterns de détection de chemins spécifiques
                    path_mappings = {
                        "restaurant-app": "/data/digital-agency-ai/restaurant-app",
                        "design-agent": "/data/digital-agency-ai/agents/01-design-agent", 
                        "webdev-agent": "/data/digital-agency-ai/agents/02-webdev-agent",
                        "orchestrator": "/data/digital-agency-ai/agents/00-orchestrator",
                        "monitoring": "/data/digital-agency-ai/monitoring",
                        "integrations": "/data/digital-agency-ai/integrations",
                        "agents/": "/data/digital-agency-ai/agents"
                    }
                    
                    # Recherche du chemin spécifique dans le message
                    for keyword, path in path_mappings.items():
                        if keyword in message:
                            target_path = path
                            logger.info(f"🎯 FORMATION: Chemin ciblé détecté: {target_path}")
                            break
                    
                    # 🔧 FORÇAGE ÉTAPE 1: LS OBLIGATOIRE
                    if "\n" in target_path or "EXIGENCE" in target_path or "EXIGENCE:" in target_path:
                        logger.error(f"🚫 INVALID FORCED LS PATH DETECTED: '{repr(target_path)}'. Skipping LS tool.")
                    else:
                        tools_to_call.append(("LS", {"path": target_path}))
                        logger.warning(f"🕵️ FORCED TOOL ADD: LS | TARGET_PATH: {target_path} | MSG_START: {repr(message[:100])}")
                        logger.error(f"🕵️ APPEL DIRECT OUTIL #1: LS | PATH: {target_path} | MESSAGE_LENGTH: {len(message)}")
                        logger.info(f"📁 FORMATION: LS forcé sur {target_path}")
                    
                    # 🔧 FORÇAGE ÉTAPE 2: READ package.json SI PERTINENT
                    if any(word in message_lower for word in ["framework", "technologies", "package", "dependencies"]):
                        package_path = f"{target_path}/package.json"
                        tools_to_call.append(("Read", {"file_path": package_path}))
                        logger.warning(f"🕵️ FORCED TOOL ADD: Read | PACKAGE_PATH: {package_path} | MSG_START: {repr(message[:100])}")
                        logger.info(f"📄 FORMATION: Read forcé sur {package_path}")
                    
                    # 🔧 FORÇAGE ÉTAPE 3: GLOB SI WORKFLOWS/FICHIERS MENTIONNÉS
                    if any(word in message_lower for word in ["workflows", "fichiers", "types", "patterns"]):
                        if "workflows" in target_path or "workflows" in message_lower:
                            glob_path = f"{target_path}/workflows" if not target_path.endswith("/workflows") else target_path
                            tools_to_call.append(("Glob", {"pattern": "**/*.{js,ts}", "path": glob_path}))
                        else:
                            tools_to_call.append(("Glob", {"pattern": "**/*.{js,ts,json}", "path": target_path}))
                        logger.warning(f"🕵️ FORCED TOOL ADD: Glob | PATTERN: {tools_to_call[-1][1].get('pattern')} | PATH: {tools_to_call[-1][1].get('path')} | MSG_START: {repr(message[:100])}")
                        logger.info(f"🔍 FORMATION: Glob forcé sur patterns")
                
                else:
                    # Fallback: détection basique pour compatibilité (seulement si forçage n'a rien ajouté)
                    if any(word in message_lower for word in ["ls", "list", "structure", "répertoire", "dossier"]):
                        # Extract path from message and convert to container path
                        # Order matters: more specific keywords (longer, more absolute) first.
                        path_keywords_map = {
                            "/Users/manu/Documents/DEV/digital-agency-ai": "/data/digital-agency-ai", # Full local path example
                            "agents/": "/data/digital-agency-ai/agents/", # Agents folder (ensure keyword ends with / if it's a dir)
                            "restaurant-app": "/data/digital-agency-ai/restaurant-app", # Specific module
                            "digital-agency-ai": "/data/digital-agency-ai" # Project root keyword (least specific)
                        }
                        detected_path = None
                        extracted_segment_for_log = None

                        for keyword_to_search, container_root_for_keyword in path_keywords_map.items():
                            path_segment_found = extract_clean_path(message, keyword_to_search)

                            if path_segment_found:
                                logger.warning(f"🕵️ EXTRACT_CLEAN_PATH LOGIC: keyword='{keyword_to_search}', input_message_snippet='{repr(message[:150])}', extracted_segment='{repr(path_segment_found)}'")
                                extracted_segment_for_log = path_segment_found
                                
                                if path_segment_found.startswith("/data/"): # Already an absolute /data path
                                    detected_path = os.path.normpath(path_segment_found)
                                elif keyword_to_search.startswith("/Users/"): # Keyword was a full local path
                                    # path_segment_found is like "/Users/manu/Documents/DEV/digital-agency-ai/src"
                                    # We need the part relative to keyword_to_search: "src"
                                    relative_to_keyword = path_segment_found[len(keyword_to_search):].lstrip('/')
                                    if not relative_to_keyword and path_segment_found == keyword_to_search: # Extracted exactly the keyword
                                        detected_path = os.path.normpath(container_root_for_keyword)
                                    else:
                                        detected_path = os.path.normpath(os.path.join(container_root_for_keyword, relative_to_keyword))
                                else: # Keyword is a simple token like "digital-agency-ai" or "agents/"
                                    # path_segment_found is like "digital-agency-ai/src" or "agents/01-design-agent"
                                    # container_root_for_keyword is like "/data/digital-agency-ai" or "/data/digital-agency-ai/agents/"
                                    
                                    # Get the part of path_segment_found that is "after" the keyword_to_search token
                                    path_relative_to_keyword_token = path_segment_found
                                    if path_segment_found.startswith(keyword_to_search):
                                         path_relative_to_keyword_token = path_segment_found[len(keyword_to_search):].lstrip('/')

                                    # If keyword_to_search ended with '/' (e.g. "agents/"), container_root_for_keyword also ends with '/'
                                    # os.path.join handles this correctly.
                                    # e.g. join("/data/digital-agency-ai/agents/", "01-design-agent")
                                    if not path_relative_to_keyword_token and path_segment_found == keyword_to_search: # Extracted exactly the keyword
                                        detected_path = os.path.normpath(container_root_for_keyword)
                                    else:
                                        detected_path = os.path.normpath(os.path.join(container_root_for_keyword, path_relative_to_keyword_token))
                                logger.warning(f"🕵️ DETECTED_PATH CONSTRUCTION (Fallback): final_detected_path='{repr(detected_path)}' from keyword='{keyword_to_search}' and segment='{repr(path_segment_found)}'")
                                break # Found a keyword and processed, stop searching

                        if detected_path: # Only proceed if a path was detected
                            if "\n" in detected_path or "EXIGENCE" in detected_path or "EXIGENCE:" in detected_path:
                                logger.error(f"🚫 INVALID FALLBACK LS PATH DETECTED: '{repr(detected_path)}'. Skipping LS tool.")
                            else:
                                # The original tools_to_call.append and its associated logger.warning follow here
                                logger.warning(f"🕵️ FALLBACK TOOL ADD: LS | DETECTED_PATH: {detected_path} | RAW_EXTRACTED: {extracted_segment_for_log}")
                                if not any(t[0] == "LS" and t[1].get("path") == detected_path for t in tools_to_call):
                                    tools_to_call.append(("LS", {"path": detected_path}))
                    
                    # Fallback Read
                    if any(word in message_lower for word in ["read", "contenu", "package.json", "fichier"]):
                        # Extract file path from message
                        if "package.json" in message:
                            # Find the base path and append package.json
                            path_keywords = ["/Users/", "digital-agency-ai"]
                            for keyword in path_keywords:
                                if keyword in message:
                                    start = message.find(keyword)
                                    end = message.find(" ", start)
                                    if end == -1:
                                        end = len(message)
                                    original_base_path = message[start:end]
                                
                                    # 🔧 CONVERSION CHEMIN CONTAINER
                                    if "/Users/manu/Documents/DEV/digital-agency-ai" in original_base_path:
                                        container_base_path = original_base_path.replace("/Users/manu/Documents/DEV/digital-agency-ai", "/data/digital-agency-ai")
                                    elif "digital-agency-ai" in original_base_path and not original_base_path.startswith("/"):
                                        if original_base_path.startswith("digital-agency-ai"):
                                            container_base_path = f"/data/{original_base_path}"
                                        else:
                                            container_base_path = f"/data/digital-agency-ai/{original_base_path}"
                                    else:
                                        container_base_path = original_base_path
                                    
                                    tools_to_call.append(("Read", {"file_path": f"{container_base_path}/package.json"}))
                                    break
                
                    # Fallback Glob
                    if any(word in message_lower for word in ["glob", "search", "pattern", "*.js", "*.ts"]):
                        # Extract pattern and path
                        patterns = ["**/*.js", "**/*.ts", "**/*.json"]
                        for pattern in patterns:
                            if pattern.replace("**", "") in message:
                                # 🔧 UTILISATION CHEMIN CONTAINER POUR GLOB
                                tools_to_call.append(("Glob", {"pattern": pattern, "path": "/data/digital-agency-ai"}))
                                break
                
                # 🛡️ EXÉCUTION AVEC COUCHE DE RÉSILIENCE
                tool_results = []
                collected_data = []  # Pour la couche de résilience
                
                logger.info(f"🛠️ PRE-NORMALIZATION tools_to_call (AgentManager loop): {tools_to_call}")
                # 🧹 NORMALISATION GLOBALE: S'applique à TOUS les tools_to_call
                logger.warning(f"🧹 NORMALISATION: Starting with {len(tools_to_call)} tools: {[t[0] for t in tools_to_call]}")
                normalized_tools = []
                for tool_name, tool_args in tools_to_call:
                    # Traçage avant normalisation
                    logger.warning(f"🛠️ TOOL INPUT → {tool_name} args: {tool_args}")
                    
                    # Normalisation des chemins
                    normalized_args = tool_args.copy()
                    if "path" in normalized_args:
                        normalized_args["path"] = normalize_path(normalized_args["path"])
                    if "file_path" in normalized_args:
                        normalized_args["file_path"] = normalize_path(normalized_args["file_path"])
                    
                    logger.warning(f"🔧 TOOL NORMALIZED → {tool_name} args: {normalized_args}")
                    normalized_tools.append((tool_name, normalized_args))
                
                for tool_name, tool_args in normalized_tools:
                    try:
                        logger.info(f"🛡️ RÉSILIENCE: Exécution {tool_name} avec fallbacks")
                        # 🔧 NORMALISATION GLOBALE avant TOUT appel d'outil
                        normalized_args = normalize_tool_args(tool_name, tool_args)
                        result = await mcp_clients["ollama"].call_tool(tool_name, normalized_args)
                        
                        # STEP 1: Log tool success immediately after call_tool (as requested by user)
                        logger.error(f'✅ TOOL SUCCESS → {tool_name} → result: {result}')
                        
                        if result and "content" in result and not result.get("is_error", False):
                            # Succès normal
                            content_text = result['content'][0]['text']
                            tool_results.append(f"\n🔧 {tool_name} résultat:\n{content_text}\n")
                            collected_data.append({
                                "tool": tool_name,
                                "summary": content_text[:200] + "..." if len(content_text) > 200 else content_text,
                                "success": True
                            })
                            logger.info(f"🛡️ RÉSILIENCE: {tool_name} réussi")
                            
                        else:
                            # Échec détecté - activation couche de résilience
                            error_msg = result.get('content', [{}])[0].get('text', 'Erreur inconnue') if result else "Pas de réponse"
                            error = Exception(error_msg)
                            
                            logger.warning(f"🛡️ RÉSILIENCE: {tool_name} échoué, activation récupération")
                            
                            # Contexte pour la récupération
                            recovery_context = {
                                "target_path": target_path,
                                "analysis_type": "analyse", 
                                "collected_data": collected_data,
                                "session_id": session_id
                            }
                            
                            # Tentative de récupération
                            recovery_result = await resilience_layer.handle_tool_failure(
                                tool_name, tool_args, error, recovery_context, mcp_clients
                            )
                            
                            if recovery_result["success"]:
                                # Récupération réussie
                                recovery_data = recovery_result["data"]
                                recovery_content = recovery_data['content'][0]['text'] if recovery_data else recovery_result["message"]
                                tool_results.append(f"\n🛡️ {tool_name} récupéré:\n{recovery_content}\n")
                                collected_data.append({
                                    "tool": tool_name,
                                    "summary": recovery_result["message"],
                                    "success": True,
                                    "recovered": True
                                })
                                logger.info(f"🛡️ RÉSILIENCE: {tool_name} récupéré avec succès")
                            else:
                                # Récupération échouée
                                tool_results.append(f"\n❌ {tool_name} échec critique:\n{recovery_result['message']}\n")
                                collected_data.append({
                                    "tool": tool_name,
                                    "summary": f"Échec: {recovery_result['message']}",
                                    "success": False
                                })
                                logger.error(f"🛡️ RÉSILIENCE: {tool_name} récupération échouée")
                                
                    except Exception as e:
                        # Erreur inattendue - activation couche de résilience
                        logger.error(f"🛡️ RÉSILIENCE: Erreur critique {tool_name}: {e}")
                        
                        recovery_context = {
                            "target_path": target_path,
                            "analysis_type": "analyse",
                            "collected_data": collected_data,
                            "session_id": session_id
                        }
                        
                        recovery_result = await resilience_layer.handle_tool_failure(
                            tool_name, tool_args, e, recovery_context, mcp_clients
                        )
                        
                        if recovery_result["success"]:
                            recovery_data = recovery_result["data"]
                            recovery_content = recovery_data['content'][0]['text'] if recovery_data else recovery_result["message"]
                            tool_results.append(f"\n🛡️ {tool_name} récupéré (exception):\n{recovery_content}\n")
                        else:
                            tool_results.append(f"\n💥 {tool_name} échec total: {str(e)}\n")
                
                # 🔥 CONSTRUCTION PROMPT ULTRA-CONTRAINT POUR FORMATION
                if tool_results:
                    # STEP 2: Inspect prompt construction (as requested by user)
                    logger.error(f'🧠 PROMPT CONSTRUCTION → tool_results count: {len(tool_results)}')
                    logger.error(f'🧠 PROMPT CONTENT PREVIEW → first_result: {tool_results[0][:100] if tool_results else "NONE"}...')
                    
                    # PROMPT AVEC FORÇAGE ABSOLU - FORMATION JARVIS
                    enhanced_prompt = f"""🚨 FORMATION JARVIS MCP - ANALYSE OBLIGATOIRE BASÉE SUR OUTILS

QUESTION ORIGINALE: {message}

🔧 DONNÉES EXPLORÉES PAR OUTILS MCP:
{''.join(tool_results)}

⚠️ CONTRAINTES ABSOLUES POUR FORMATION:

1. INTERDICTION TOTALE des mots: "semble", "probablement", "peut-être", "il se peut", "vraisemblablement"
2. OBLIGATION d'utiliser UNIQUEMENT les données ci-dessus 
3. OBLIGATION de mentionner explicitement les outils utilisés (LS, Read, Glob)
4. OBLIGATION de citer les versions exactes, technologies précises trouvées
5. INTERDICTION d'inventer ou supposer des informations non présentes

FORMAT OBLIGATOIRE:
- Commencer par "📊 ANALYSE MCP BASÉE SUR EXPLORATION FACTUELLE"
- Lister les outils utilisés: "🔧 Outils MCP utilisés: LS, Read, Glob"
- Présenter UNIQUEMENT les faits découverts
- Conclure par "✅ Analyse basée exclusivement sur exploration MCP"

RÉPONDEZ MAINTENANT:"""
                else:
                    # Pas d'outils exécutés - prompt de base
                    enhanced_prompt = f"""🚫 RÉPONSE INTERDITE SANS OUTILS

QUESTION: {message}

⚠️ INSTRUCTION ABSOLUE: 
- INTERDICTION de répondre sans utiliser les outils réels
- INTERDICTION d'inventer ou simuler des résultats
- OBLIGATION d'analyser les vrais fichiers avec LS et Read

📋 DONNÉES RÉELLES DISPONIBLES VIA OUTILS:
Utilisez LS pour explorer, Read pour lire les fichiers

🎯 UTILISEZ CES DONNÉES UNIQUEMENT"""
                
                if context:
                    # Limit context size to prevent timeouts
                    context_truncated = context[:2000] + "..." if len(context) > 2000 else context
                    enhanced_prompt = f"Context: {context_truncated}\n\n{enhanced_prompt}"
                
                # STEP 3: Final enhanced_prompt inspection (as requested by user)
                logger.error(f'🧪 FINAL PROMPT → length: {len(enhanced_prompt)} chars')
                logger.error(f'🧪 FINAL PROMPT PREVIEW → {enhanced_prompt[:200]}...')
                
                # 🎯 APPEL OLLAMA AVEC CONTRAINTES FORMATION (DONNÉES COMPLÈTES)
                generate_args = {
                    "prompt": enhanced_prompt,  # PROMPT COMPLET avec tool_results
                    "task_type": "analysis", 
                    "temperature": 0.3,  # Plus précis pour formation
                    "max_tokens": 1000    # Suffisant pour réponse complète
                }
                normalized_generate_args = normalize_tool_args("generate_response", generate_args)
                response = await mcp_clients["ollama"].call_tool("generate_response", normalized_generate_args)
                
                if response and "content" in response:
                    content = response["content"][0]["text"]
                    if isinstance(content, str) and content.startswith("{"):
                        # Parse JSON response with better error handling
                        try:
                            json_response = json.loads(content)
                            return json_response.get("response", content)
                        except json.JSONDecodeError as e:
                            logger.warning(f"Failed to parse Ollama JSON response: {e}")
                            logger.warning(f"Raw content: {content[:200]}...")
                            # Try to extract response from malformed JSON
                            if '"response":' in content:
                                import re
                                match = re.search(r'"response":\s*"([^"]*(?:\\.[^"]*)*)"', content)
                                if match:
                                    return match.group(1).replace('\n', '\\n').replace('\\"', '"')
                            return content
                    response_text = content
                else:
                    response_text = "Erreur: Réponse MCP Ollama invalide"

            else: # This else corresponds to 'if "ollama" in mcp_clients:'
                # Fallback to direct Ollama if MCP not available
                prompt = message
            if context:
                prompt = f"Context: {context}\n\nQuestion: {message}"

            async with get_resilient_http_client(180.0) as client:
                response = await client.post(
                    f"http://{OLLAMA_HOST}/api/generate",
                    json={
                        "model": "llama3.2:3b",
                        "prompt": prompt,
                        "stream": False,
                        "options": {
                            "temperature": 0.7,
                            "top_p": 0.9
                        }
                    }
                )
                
                if response.status_code == 200:
                    data = response.json()
                    response_text = data.get("response", "Erreur: pas de réponse d'Ollama")
                else:
                    response_text = f"Erreur Ollama: HTTP {response.status_code}"

            # Populate tool execution details from collected_data

            # Process collected_data if tools were run in the MCP path
            # This block is INSIDE the 'if "ollama" in mcp_clients:' block
            if 'collected_data' in locals() and collected_data: # locals() check is robust
                logger.info(f"🔄 AGENT_MANAGER: Processing collected_data for return: {collected_data}")
                # tools_successfully_used and tool_execution_details are already initialized (function-scope)
                for item in collected_data:
                    tool_name = item.get("tool")
                    if not tool_name:
                        continue
                    success_status = item.get("success", False)
                    tool_execution_details[tool_name] = { # Populate function-scoped dict
                        "success": success_status,
                        "summary": item.get("summary", ""),
                        "recovered": item.get("recovered", False)
                    }
                    if success_status:
                        if tool_name not in tools_successfully_used: # Populate function-scoped list
                            tools_successfully_used.append(tool_name)

                logger.info(f"🔄 AGENT_MANAGER: Processed tools_successfully_used: {tools_successfully_used}")
                logger.info(f"🔄 AGENT_MANAGER: Processed tool_execution_details: {repr(tool_execution_details)[:500]}")
            else:
                # If collected_data was not populated (e.g. no tools in tools_to_call)
                # tools_successfully_used and tool_execution_details will remain [] and {} as initialized.
                logger.info("🔄 AGENT_MANAGER: No collected_data to process for tool metadata return (tools_successfully_used and tool_execution_details will remain empty).")

            # ... existing logic to build 'enhanced_prompt' using 'tool_results' string list ...
            # ... existing call to Ollama's generate_response to get the main 'response_text' ...
            # Ensure 'response_text' (initialized at the start of try block)
            # is assigned the text from Ollama's generate_response call.
            # This part of the code (Ollama generate call and response_text assignment) is assumed to be correct from previous steps.
            # Example:
            # ollama_gen_response_obj = await mcp_clients["ollama"].call_tool("generate_response", normalized_generate_args)
            # if ollama_gen_response_obj and "content" in ollama_gen_response_obj:
            #    # ... (extraction logic) ...
            #    response_text = extracted_text_from_ollama_generate
            # else:
            #    response_text = "Erreur: Réponse MCP Ollama invalide après génération."

            logger.info(f"🔄 AGENT_MANAGER: Returning from MCP tool path. Tools Used: {tools_successfully_used}, Details: {repr(tool_execution_details)[:200]}")
            return response_text, tools_successfully_used, tool_execution_details
            
        # This 'else' handles the case where 'ollama' is NOT in mcp_clients (direct Ollama fallback)
        else:
            # response_text is already initialized to "" at the start of the try block.
            # tools_successfully_used is [], tool_execution_details is {} (function-scoped).
            prompt = message
            if context:
                prompt = f"Context: {context}\n\nQuestion: {message}"
            
            async with get_resilient_http_client(180.0) as client:
                response = await client.post(
                    f"http://{OLLAMA_HOST}/api/generate",
                    json={
                        "model": "llama3.2:3b",
                        "prompt": prompt,
                        "stream": False,
                        "options": {
                            "temperature": 0.7,
                            "top_p": 0.9
                        }
                    }
                )
                if response.status_code == 200:
                    data = response.json()
                    response_text = data.get("response", "Erreur: pas de réponse d'Ollama")
                else:
                    response_text = f"Erreur Ollama: HTTP {response.status_code}"

            logger.info("🔄 AGENT_MANAGER: Returning from direct Ollama fallback path. No MCP tool metadata.")
            return response_text, [], {} # Return empty list/dict for tools
                    
        except Exception as e:
            logger.error(f"Error in enhanced call_ollama: {e}")
            return f"Erreur connexion Ollama: {str(e)}", [], {}
    
    @staticmethod
    async def call_perplexity(message: str, context: str = None) -> str:
        """Enhanced Perplexity integration"""
        enhanced_query = f"🔍 Recherche: {message}"
        if context:
            enhanced_query += f" (Contexte: {context})"
        
        # TODO: Real Perplexity Pro integration when API key is provided
        return f"{enhanced_query}\n\n[Simulation] Cette fonctionnalité sera connectée au vrai Perplexity Pro quand la clé API sera fournie."
    
    @staticmethod
    async def optimize_prompt(original: str, target_ai: str, context: str = None, project: str = None) -> str:
        """Enhanced prompt optimization"""
        
        optimizations = {
            "claude": f"""Tu es Claude Code, assistant expert en développement.

Contexte du projet: {project or "Projet générique"}
Contexte technique: {context or "Non spécifié"}

Demande originale: {original}

Réponds de manière structurée avec:
1. Analyse technique précise
2. Solutions pratiques avec code si pertinent
3. Recommandations d'amélioration
4. Considérations de sécurité/performance si applicable""",

            "chatgpt": f"""Tu es un assistant développeur expert.

Projet: {project or "Développement"}
Contexte: {context or "Non spécifié"}

Demande: {original}

Fournis une réponse équilibrée avec:
- Explication claire du problème/besoin
- Solution étape par étape
- Code d'exemple si nécessaire
- Alternatives possibles""",

            "perplexity": f"""Recherche approfondie sur: {original}

Projet: {project or ""}
Contexte: {context or ""}

Focus sur:
- Informations récentes (2024-2025)
- Meilleures pratiques actuelles
- Documentation officielle
- Exemples concrets et tutoriels"""
        }
        
        return optimizations.get(target_ai, original)

# Database helpers
class DatabaseManager:
    """Enhanced Database Manager with MCP support"""
    
    @staticmethod
    def store_conversation(message: str, response: str, agent: str, project: str = None, 
                          context: str = None, response_time_ms: int = 0) -> int:
        """Store conversation with enhanced metadata"""
        conn = sqlite3.connect(DATABASE_PATH)
        cursor = conn.execute("""
            INSERT INTO conversations (message, response, agent_used, project, context, response_time_ms)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (message, response, agent, project, context, response_time_ms))
        
        conversation_id = cursor.lastrowid
        conn.commit()
        conn.close()
        
        # Update agent stats
        DatabaseManager.update_agent_stats(agent, response_time_ms, True)
        
        return conversation_id
    
    @staticmethod
    def update_agent_stats(agent: str, response_time_ms: int, success: bool):
        """Update agent statistics"""
        conn = sqlite3.connect(DATABASE_PATH)
        
        cursor = conn.execute("SELECT * FROM agent_stats WHERE agent_name = ?", (agent,))
        if cursor.fetchone():
            conn.execute("""
                UPDATE agent_stats 
                SET total_requests = total_requests + 1,
                    total_success = total_success + ?,
                    avg_response_time_ms = (avg_response_time_ms * total_requests + ?) / (total_requests + 1),
                    last_used = CURRENT_TIMESTAMP
                WHERE agent_name = ?
            """, (1 if success else 0, response_time_ms, agent))
        else:
            conn.execute("""
                INSERT INTO agent_stats (agent_name, total_requests, total_success, avg_response_time_ms, last_used)
                VALUES (?, 1, ?, ?, CURRENT_TIMESTAMP)
            """, (agent, 1 if success else 0, response_time_ms))
        
        conn.commit()
        conn.close()
    
    @staticmethod
    def get_conversations(limit: int = 10, project: str = None) -> List[Dict]:
        """Get conversation history"""
        conn = sqlite3.connect(DATABASE_PATH)
        
        query = "SELECT id, message, response, agent_used, project, timestamp FROM conversations"
        params = []
        
        if project:
            query += " WHERE project = ?"
            params.append(project)
        
        query += " ORDER BY timestamp DESC LIMIT ?"
        params.append(limit)
        
        cursor = conn.execute(query, params)
        conversations = []
        
        for row in cursor.fetchall():
            conversations.append({
                "id": row[0],
                "message": row[1],
                "response": row[2][:200] + "..." if len(row[2]) > 200 else row[2],
                "agent_used": row[3],
                "project": row[4],
                "timestamp": row[5]
            })
        
        conn.close()
        return conversations
    
    @staticmethod
    def get_stats() -> Dict:
        """Get comprehensive statistics"""
        conn = sqlite3.connect(DATABASE_PATH)
        
        # Total conversations
        cursor = conn.execute("SELECT COUNT(*) FROM conversations")
        total_conversations = cursor.fetchone()[0]
        
        # Recent activity (7 days)
        cursor = conn.execute("""
            SELECT COUNT(*) FROM conversations 
            WHERE timestamp > datetime('now', '-7 days')
        """)
        recent_activity = cursor.fetchone()[0]
        
        # Usage by AI
        cursor = conn.execute("""
            SELECT agent_used, COUNT(*) 
            FROM conversations 
            GROUP BY agent_used
        """)
        ai_usage = dict(cursor.fetchall())
        
        # Average response times
        cursor = conn.execute("""
            SELECT agent_name, avg_response_time_ms 
            FROM agent_stats
        """)
        avg_response_times = dict(cursor.fetchall())
        
        conn.close()
        
        return {
            "total_conversations": total_conversations,
            "recent_activity": recent_activity,
            "ai_usage": ai_usage,
            "avg_response_times": avg_response_times
        }

# AI Routing Logic
class AIRouter:
    def __init__(self):
        self.routing_rules = {
            "code": ["ollama", "tools"],
            "search": ["perplexity", "memory"],
            "analysis": ["ollama", "memory", "tools"],
            "creative": ["ollama"],
            "debug": ["ollama", "tools"],
            "research": ["perplexity", "memory"]
        }
    
    def detect_intent(self, query: str) -> str:
        """Advanced intent detection"""
        query_lower = query.lower()
        
        if any(word in query_lower for word in ["bug", "error", "debug", "fix"]):
            return "debug"
        elif any(word in query_lower for word in ["search", "find", "research", "latest"]):
            return "search"
        elif any(word in query_lower for word in ["code", "function", "class", "implement"]):
            return "code"
        elif any(word in query_lower for word in ["analyze", "explain", "understand"]):
            return "analysis"
        elif any(word in query_lower for word in ["create", "write", "generate"]):
            return "creative"
        else:
            return "code"  # Default for development focus

# MCP Orchestrator for intelligent coordination
class MCPOrchestrator:
    """Advanced MCP Orchestrator with inter-agent communication"""
    
    def __init__(self):
        self.router = AIRouter()
        self.active_sessions: Dict[str, Dict] = {}
        
    async def process_mcp_request(self, request: MCPRequest) -> MCPResponse:
        """Process request through advanced MCP ecosystem"""
        start_time = time.time()
        session_id = request.session_id
        
        # Initialize session context
        if session_id not in self.active_sessions:
            self.active_sessions[session_id] = {
                "created_at": datetime.now(),
                "context": {},
                "conversation_history": []
            }
        
        session = self.active_sessions[session_id]
        session["conversation_history"].append({
            "timestamp": datetime.now().isoformat(),
            "message": request.message,
            "context": request.context
        })
        
        # Determine which agents/tools to use
        intent = self.router.detect_intent(request.message)
        required_servers = self._get_required_servers(intent, request.tools)
        
        tool_results = {}
        agents_used = []
        final_response = ""
        
        # Execute MCP tool calls if specified
        if request.tools:
            for tool_call in request.tools:
                if tool_call.server_name in mcp_clients:
                    try:
                        result = await mcp_clients[tool_call.server_name].call_tool(
                            tool_call.tool_name,
                            tool_call.arguments
                        )
                        tool_results[f"{tool_call.server_name}:{tool_call.tool_name}"] = result
                        agents_used.append(tool_call.server_name)
                    except Exception as e:
                        tool_results[f"{tool_call.server_name}:{tool_call.tool_name}"] = {
                            "error": str(e)
                        }
        
        # Process with appropriate AI based on intent
        if "ollama" in required_servers and "ollama" in mcp_clients:
            # Use Ollama for main processing
            context_str = self._build_context_string(session, tool_results)
            final_response = await AgentManager.call_ollama(request.message, context_str, session_id)
            agents_used.append("ollama")
            
        elif "perplexity" in required_servers and "perplexity" in mcp_clients:
            # Use Perplexity for research
            final_response = await AgentManager.call_perplexity(request.message, 
                                                               request.context.get("project"))
            agents_used.append("perplexity")
        else:
            # Fallback to simple processing
            final_response = f"Processed: {request.message}\n\nTool Results: {json.dumps(tool_results, indent=2)}"
        
        # Update memory if available
        if "memory" in mcp_clients:
            try:
                await self._update_memory(session_id, request.message, final_response, tool_results)
            except Exception as e:
                logger.warning(f"Memory update failed: {e}")
        
        processing_time = int((time.time() - start_time) * 1000)
        
        return MCPResponse(
            session_id=session_id,
            response=final_response,
            tool_results=tool_results,
            agents_used=agents_used,
            processing_time_ms=processing_time
        )
    
    def _get_required_servers(self, intent: str, explicit_tools: List[MCPToolCall]) -> List[str]:
        """Determine required MCP servers based on intent and explicit tools"""
        servers = set()
        
        # Add servers from explicit tool calls
        for tool_call in explicit_tools:
            servers.add(tool_call.server_name)
        
        # Add servers based on intent
        intent_mapping = {
            "code": ["ollama", "tools", "memory"],
            "search": ["perplexity", "tools", "memory"],
            "analysis": ["ollama", "memory", "tools"],
            "creative": ["ollama", "memory"],
            "debug": ["ollama", "tools", "memory"],
            "research": ["perplexity", "memory", "tools"]
        }
        
        for server in intent_mapping.get(intent, ["ollama", "memory"]):
            servers.add(server)
        
        return list(servers)
    
    def _build_context_string(self, session: Dict, tool_results: Dict) -> str:
        """Build context string from session history and tool results"""
        context_parts = []
        
        # Add recent conversation history
        recent_history = session["conversation_history"][-3:]
        if recent_history:
            context_parts.append("Conversation History:")
            for item in recent_history:
                context_parts.append(f"- {item['timestamp']}: {item['message'][:100]}...")
        
        # Add tool results
        if tool_results:
            context_parts.append("\nTool Results:")
            for tool_name, result in tool_results.items():
                if isinstance(result, dict) and "error" not in result:
                    context_parts.append(f"- {tool_name}: {str(result)[:200]}...")
                elif "error" in result:
                    context_parts.append(f"- {tool_name}: ERROR - {result['error']}")
        
        return "\n".join(context_parts)
    
    async def _update_memory(self, session_id: str, message: str, response: str, tool_results: Dict):
        """Update memory with conversation data"""
        try:
            memory_data = {
                "session_id": session_id,
                "message": message,
                "response": response,
                "tool_results": tool_results,
                "timestamp": datetime.now().isoformat()
            }
            
            await mcp_clients["memory"].call_tool("store_conversation", memory_data)
        except Exception as e:
            logger.error(f"Memory update failed: {e}")

# Initialize MCP ecosystem
async def initialize_mcp_ecosystem():
    """Initialize advanced MCP ecosystem with real protocol communication"""
    servers = {
        "ollama": "http://mcp-ollama:4003",
        "perplexity": "http://mcp-perplexity:4004", 
        "memory": "http://mcp-memory:4005",
        "tools": "http://mcp-tools:4006"
    }
    
    initialization_results = []
    
    for name, url in servers.items():
        try:
            client = MCPClient(name, url)
            
            # Health check first
            if await client.health_check():
                # Initialize MCP session
                if await client.initialize():
                    mcp_clients[name] = client
                    initialization_results.append(f"✅ {name}")
                    logger.info(f"✅ MCP Server fully initialized: {name} at {url}")
                else:
                    initialization_results.append(f"❌ {name} (init failed)")
                    logger.error(f"❌ MCP Server init failed: {name}")
            else:
                initialization_results.append(f"⚠️ {name} (unreachable)")
                logger.warning(f"⚠️ MCP Server unreachable: {name} at {url}")
                
        except Exception as e:
            initialization_results.append(f"❌ {name} (error: {e})")
            logger.error(f"❌ Erreur connexion MCP {name}: {e}")
    
    logger.info(f"MCP Ecosystem Status: {', '.join(initialization_results)}")
    return len(mcp_clients)

# Lifespan management
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifecycle with MCP initialization"""
    # Startup
    init_database()
    connected_servers = await initialize_mcp_ecosystem()
    logger.info(f"🧠 Jarvis MCP Hub started with {connected_servers} servers")
    yield
    # Shutdown
    for name, client in mcp_clients.items():
        try:
            await client.close()
            logger.info(f"Closed MCP client: {name}")
        except Exception as e:
            logger.error(f"Error closing MCP client {name}: {e}")
    logger.info("🛑 Jarvis MCP Hub stopped")

# FastAPI app with lifespan management
app = FastAPI(
    title="Jarvis MCP Hub",
    version="2.0.0",
    description="Advanced MCP Orchestrator with Real-time Communication",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize orchestrator
orchestrator = MCPOrchestrator()

# WebSocket endpoint for real-time MCP communication
@app.websocket("/mcp/ws/{session_id}")
async def websocket_mcp_endpoint(websocket: WebSocket, session_id: str):
    """WebSocket endpoint for real-time MCP communication"""
    await websocket.accept()
    active_connections.append(websocket)
    
    logger.info(f"🔗 WebSocket connected: {session_id}")
    
    # Send welcome message
    await websocket.send_json({
        "type": "connection",
        "session_id": session_id,
        "status": "connected",
        "available_servers": list(mcp_clients.keys()),
        "timestamp": datetime.now().isoformat()
    })
    
    try:
        while True:
            data = await websocket.receive_json()
            logger.debug(f"🧾 Message WebSocket reçu : {data}")
            
            if data.get("type") == "mcp_request":
                # Auto-detect tools from prompt if not provided
                message = data["message"]
                provided_tools = data.get("tools", [])
                
                # If no tools provided, try to auto-detect
                if not provided_tools:
                    interpreted = interpret_prompt(message)
                    if interpreted:
                        logger.info(f"🤖 Auto-detected tools for: {message[:50]}...")
                        logger.info(f"🛠️ Tools detected: {[t['tool_name'] for t in interpreted['required_tools']]}")
                        provided_tools = interpreted["required_tools"]
                
                # Create MCP request
                mcp_request = MCPRequest(
                    message=message,
                    session_id=session_id,
                    tools=provided_tools,
                    context=data.get("context", {})
                )
                
                # Process with orchestrator
                response = await orchestrator.process_mcp_request(mcp_request)
                
                # Send response
                await websocket.send_json({
                    "type": "mcp_response",
                    "session_id": session_id,
                    "response": response.response,
                    "tool_results": response.tool_results,
                    "agents_used": response.agents_used,
                    "processing_time_ms": response.processing_time_ms,
                    "timestamp": datetime.now().isoformat()
                })
                
                # Store in database
                DatabaseManager.store_conversation(
                    mcp_request.message, 
                    response.response, 
                    ",".join(response.agents_used),
                    mcp_request.context.get("project"), 
                    json.dumps(mcp_request.context), 
                    response.processing_time_ms
                )
                
            elif data.get("type") == "streaming_request":
                # Handle streaming request
                await handle_streaming_request(websocket, session_id, data)
                
    except WebSocketDisconnect:
        active_connections.remove(websocket)
        logger.info(f"🔌 WebSocket disconnected: {session_id}")
    except Exception as e:
        logger.error(f"WebSocket error for {session_id}: {e}")
        try:
            await websocket.send_json({
                "type": "error",
                "session_id": session_id,
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            })
        except:
            pass
        active_connections.remove(websocket)

async def handle_streaming_request(websocket: WebSocket, session_id: str, data: Dict):
    """Handle streaming chat request with real-time chunks"""
    try:
        message = data["message"]
        agent = data.get("agent", "auto")
        context = data.get("context", {})
        
        # Determine agent
        if agent == "auto":
            intent = orchestrator.router.detect_intent(message)
            agent = "ollama" if intent in ["code", "analysis", "creative"] else "perplexity"
        
        # Send start streaming signal
        await websocket.send_json({
            "type": "stream_start",
            "session_id": session_id,
            "agent_used": agent,
            "timestamp": datetime.now().isoformat()
        })
        
        # Stream response chunks
        full_response = ""
        chunk_count = 0
        
        if agent == "ollama":
            # Simulate streaming from Ollama
            response = await AgentManager.call_ollama(message, json.dumps(context), session_id)
            words = response.split()
            
            for i in range(0, len(words), 3):  # Send in chunks of 3 words
                chunk = " ".join(words[i:i+3])
                full_response += chunk + " "
                chunk_count += 1
                
                await websocket.send_json({
                    "type": "stream_chunk",
                    "session_id": session_id,
                    "chunk": chunk + " ",
                    "chunk_number": chunk_count,
                    "agent_used": agent,
                    "timestamp": datetime.now().isoformat()
                })
                
                # Small delay for streaming effect
                await asyncio.sleep(0.1)
        
        # Send end streaming signal
        await websocket.send_json({
            "type": "stream_end",
            "session_id": session_id,
            "final_response": full_response.strip(),
            "total_chunks": chunk_count,
            "agent_used": agent,
            "timestamp": datetime.now().isoformat()
        })
        
    except Exception as e:
        await websocket.send_json({
            "type": "stream_error",
            "session_id": session_id,
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        })

# Enhanced chat endpoint with MCP support
@app.post("/mcp/chat")
async def handle_mcp_chat(request: ChatRequest) -> ChatResponse:
    """Enhanced chat with full MCP support and validation middleware"""
    start_time = time.time()
    session_id = request.session_id or str(uuid.uuid4())
    
    # 🔧 VALIDATION MIDDLEWARE - Prétraitement obligatoire
    preprocessing_result = validation_middleware.preprocess_request(request.message, session_id)
    enhanced_message = preprocessing_result['enhanced_message']
    validation_context = preprocessing_result
    
    logger.info(f"🔧 Middleware: Message enhanced for session {session_id}")
    if preprocessing_result['validation_active']:
        logger.info(f"🎯 Validation active - Required tools: {preprocessing_result['required_tools']}")
    
    try:
        # Convert to MCP request if tools are needed
        if request.tools_needed:
            tools = []
            for tool_spec in request.tools_needed:
                if ":" in tool_spec:
                    server_name, tool_name = tool_spec.split(":", 1)
                    tools.append(MCPToolCall(
                        tool_name=tool_name,
                        arguments={},
                        server_name=server_name
                    ))
            
            mcp_request = MCPRequest(
                message=enhanced_message,  # 🔧 Use enhanced message
                session_id=session_id,
                tools=tools,
                context={
                    "project": request.project,
                    "context": request.context,
                    "validation_context": validation_context  # 🔧 Add validation context
                }
            )
            
            mcp_response = await orchestrator.process_mcp_request(mcp_request)
            
            response_time_ms = int((time.time() - start_time) * 1000)
            conversation_id = DatabaseManager.store_conversation(
                request.message, mcp_response.response, 
                ",".join(mcp_response.agents_used),
                request.project, request.context, response_time_ms
            )
            
            return ChatResponse(
                response=mcp_response.response,
                agent_used=",".join(mcp_response.agents_used),
                timestamp=datetime.now().isoformat(),
                response_time_ms=response_time_ms,
                conversation_id=conversation_id,
                session_id=session_id,
                tools_used=mcp_response.agents_used,
                mcp_results=mcp_response.tool_results
            )
        
        # Fallback to simple agent processing
        if request.agent == "auto":
            intent = orchestrator.router.detect_intent(request.message)
            agent = "ollama" if intent in ["code", "analysis", "creative"] else "perplexity"
        else:
            agent = request.agent
        
        # Initialize variables at a scope visible to the final ChatResponse return
        response_text = ""
        tools_executed_list_for_response = []
        actual_tool_outcomes_dict_for_response = {}
        tool_based_score = 0
        
        if agent == "ollama":
            ollama_specific_response_text, tools_from_ollama, outcomes_from_ollama = \
                await AgentManager.call_ollama(enhanced_message, request.context, request.session_id)

            response_text = ollama_specific_response_text

            # CRITICAL: Ensure these assignments happen to the variables used in ChatResponse
            tools_executed_list_for_response = tools_from_ollama
            actual_tool_outcomes_dict_for_response = outcomes_from_ollama

            tool_based_score = calculate_mcp_validation_score(tools_executed_list_for_response, actual_tool_outcomes_dict_for_response)
            logger.info(f"🔧 MCTOOL_SCORE: Actual tool execution score: {tool_based_score}/10 for session {session_id}. Tools executed: {tools_executed_list_for_response}, Outcomes: {repr(actual_tool_outcomes_dict_for_response)[:500]}")
        elif agent == "perplexity":
            response_text = await AgentManager.call_perplexity(enhanced_message, request.context)
            # tools_executed_list_for_response and actual_tool_outcomes_dict_for_response remain empty
            # tool_based_score remains 0
        else:
            response_text = "Agent non supporté"
            # tools_executed_list_for_response and actual_tool_outcomes_dict_for_response remain empty
            # tool_based_score remains 0
        
        response_time_ms = int((time.time() - start_time) * 1000)

        # text_validation_result and final_score calculation will use 'response_text' and 'tool_based_score'
        # which are now correctly populated based on the agent path.

        text_validation_result = validation_middleware.validate_response(response_text, validation_context)
        logger.info(f"🔧 TEXT_VALIDATION_SCORE: Middleware text analysis score: {text_validation_result.get('score')}/10 for session {session_id}")

        final_score = tool_based_score # Start with the tool-based score
        final_errors = []
        final_feedback = [f"Tool execution score (actual): {tool_based_score}/10."] # Start feedback with tool score

        # Add points/errors from text validation (excluding its tool scoring part)
        if text_validation_result.get('errors'):
            for err_msg in text_validation_result['errors']:
                if "Outils manquants" not in err_msg: # Avoid double penalty from text-based tool check
                    final_errors.append(err_msg)
                    # Simplified score adjustment from text validation errors
                    if "Phrases interdites" in err_msg: final_score = max(0, final_score - 3)
                    if "Contenu insuffisamment factuel" in err_msg: final_score = max(0, final_score - 2) # Slightly higher penalty

        if text_validation_result.get('feedback'):
             for fb_msg in text_validation_result['feedback']:
                if "Outils utilisés" not in fb_msg : final_feedback.append(fb_msg)
        
        # Ensure score is within bounds 0-10
        final_score = max(0, min(10, final_score))
        final_passed = final_score >= 7
        
        response_to_send_to_user = response_text # Use response_text (which is ollama_response_text or perplexity response)
        agent_status_for_db = agent # 'agent' variable should be defined from earlier logic in handle_mcp_chat

        if validation_context.get('validation_active') and not final_passed:
            # Construct a temporary context for generating feedback message
            # to reflect the new consolidated list of errors.
            current_issue_summary_for_feedback = {
                'score': final_score,
                'errors': final_errors
            }
            # Add a generic tool error if tool score is low and not already covered by other specific text errors
            if tool_based_score < 7 and not any("Outils manquants" in e for e in final_errors):
                # Check if required_tools were actually missing based on tool_based_score logic
                required_by_middleware = validation_context.get('required_tools', [])
                actually_missing = []
                for req_tool in required_by_middleware:
                    if not actual_tool_outcomes_dict.get(req_tool, {}).get("success"):
                        actually_missing.append(req_tool)
                if actually_missing:
                    current_issue_summary_for_feedback['errors'].append(f"Outils requis ({', '.join(actually_missing)}) non exécutés avec succès.")

            correction_feedback = validation_middleware.generate_correction_feedback(
                current_issue_summary_for_feedback, validation_context
            )
            logger.warning(f"🚫 MCP_VALIDATION_FINAL: Combined Validation failed for session {session_id}: {final_score}/10. Errors: {final_errors}")
            response_to_send_to_user = f"{response_text}\n\n{correction_feedback}"
            agent_status_for_db = f"{agent}_validation_failed"
        elif validation_context.get('validation_active'):
             logger.info(f"✅ MCP_VALIDATION_FINAL: Combined Validation passed for session {session_id}: {final_score}/10.")
             agent_status_for_db = agent
        
        # Store conversation using the potentially modified response and agent status
        conversation_id = DatabaseManager.store_conversation(
            request.message, response_to_send_to_user, agent_status_for_db,
            request.project, request.context, response_time_ms # response_time_ms should be defined earlier in the function
        )
        
        # These logs should be placed just before the final 'return ChatResponse(...)'
        # Ensure that 'tools_executed_list' and 'actual_tool_outcomes_dict' here are the
        # variables intended to be passed to ChatResponse.

        log_prefix = "🔧 FINAL_CHAT_RESPONSE_PREP"
        if agent == "ollama": # This 'agent' variable is from the outer scope, correctly reflecting the chosen agent
            log_prefix += " (Ollama Path)"

        logger.info(f"{log_prefix}: tools_executed_list='{repr(tools_executed_list_for_response)}'")
        logger.info(f"{log_prefix}: actual_tool_outcomes_dict='{repr(actual_tool_outcomes_dict_for_response)}'")
        logger.info(f"{log_prefix}: final_score='{final_score}', final_passed='{final_passed}'")
        logger.info(f"{log_prefix}: response_to_send_to_user (length)='{len(response_to_send_to_user)}'")
        logger.info(f"{log_prefix}: agent_status_for_db='{agent_status_for_db}'")

        return ChatResponse(
            response=response_to_send_to_user,
            agent_used=agent_status_for_db, # Use the status reflecting validation outcome
            timestamp=datetime.now().isoformat(),
            response_time_ms=response_time_ms, # This should be defined from start_time at the beginning of handle_mcp_chat
            conversation_id=conversation_id,
            session_id=session_id,
            tools_used=tools_executed_list_for_response,      # Populate with actual tools executed
            mcp_results=actual_tool_outcomes_dict_for_response, # Populate with actual tool outcomes
            validation_score=final_score,
            validation_passed=final_passed
        )
        
    except Exception as e:
        logger.error(f"Erreur chat MCP: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/mcp/optimize-prompt")
async def handle_optimize_prompt(request: OptimizePromptRequest) -> ChatResponse:
    """Handle prompt optimization requests"""
    start_time = time.time()
    
    try:
        optimized_prompt = await AgentManager.optimize_prompt(
            request.original_prompt, request.target_ai, 
            request.context, request.project
        )
        
        response_time_ms = int((time.time() - start_time) * 1000)
        
        # Store optimization
        conversation_id = DatabaseManager.store_conversation(
            request.original_prompt, optimized_prompt, 
            f"optimize-{request.target_ai}", 
            request.project, request.context, response_time_ms
        )
        
        return ChatResponse(
            response="Prompt optimisé généré avec succès",
            agent_used=f"optimize-{request.target_ai}",
            optimized_prompt=optimized_prompt,
            timestamp=datetime.now().isoformat(),
            response_time_ms=response_time_ms,
            conversation_id=conversation_id
        )
        
    except Exception as e:
        logger.error(f"Erreur optimisation prompt: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/mcp/conversations")
async def get_conversations(limit: int = 10, project: str = None) -> List[ConversationItem]:
    """Get conversation history"""
    try:
        conversations = DatabaseManager.get_conversations(limit, project)
        return [
            ConversationItem(
                id=conv["id"],
                message=conv["message"],
                response=conv["response"],
                agent_used=conv["agent_used"],
                project=conv["project"],
                timestamp=conv["timestamp"]
            )
            for conv in conversations
        ]
    except Exception as e:
        logger.error(f"Erreur récupération conversations: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/mcp/stats")
async def get_stats() -> StatsResponse:
    """Get usage statistics"""
    try:
        stats = DatabaseManager.get_stats()
        return StatsResponse(**stats)
    except Exception as e:
        logger.error(f"Erreur récupération stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/mcp/status")
async def get_mcp_status():
    """Get comprehensive MCP ecosystem status"""
    status = {}
    capabilities = {}
    
    for name, client in mcp_clients.items():
        try:
            health = await client.health_check()
            tools = await client.list_tools() if health else []
            resources = await client.list_resources() if health else []
            
            status[name] = {
                "connected": health,
                "session_id": client.session_id,
                "last_ping": datetime.now().isoformat(),
                "tools_count": len(tools),
                "resources_count": len(resources)
            }
            
            capabilities[name] = {
                "tools": [tool.get("name", "unknown") for tool in tools],
                "resources": [res.get("uri", "unknown") for res in resources]
            }
            
        except Exception as e:
            status[name] = {
                "connected": False,
                "error": str(e),
                "session_id": None,
                "tools_count": 0,
                "resources_count": 0
            }
            capabilities[name] = {"tools": [], "resources": []}
    
    # Get stats for compatibility
    try:
        stats = DatabaseManager.get_stats()
        return {
            "mcp_servers": status,
            "mcp_capabilities": capabilities,
            "active_websockets": len(active_connections),
            "active_sessions": len(orchestrator.active_sessions),
            "hub_status": "running",
            "ecosystem_health": {
                "total_servers": len(status),
                "connected_servers": len([s for s in status.values() if s["connected"]]),
                "total_tools": sum(s["tools_count"] for s in status.values()),
                "total_resources": sum(s["resources_count"] for s in status.values())
            },
            **stats
        }
    except Exception as e:
        logger.error(f"Error getting stats: {e}")
        return {
            "mcp_servers": status,
            "mcp_capabilities": capabilities,
            "active_websockets": len(active_connections),
            "active_sessions": len(orchestrator.active_sessions),
            "hub_status": "running",
            "ecosystem_health": {
                "total_servers": len(status),
                "connected_servers": len([s for s in status.values() if s["connected"]]),
                "total_tools": sum(s["tools_count"] for s in status.values()),
                "total_resources": sum(s["resources_count"] for s in status.values())
            }
        }

# New MCP-specific endpoints
@app.get("/mcp/tools")
async def list_all_mcp_tools():
    """List all available MCP tools across servers"""
    all_tools = {}
    
    for server_name, client in mcp_clients.items():
        try:
            tools = await client.list_tools()
            all_tools[server_name] = tools
        except Exception as e:
            all_tools[server_name] = {"error": str(e)}
    
    return {"tools_by_server": all_tools}

@app.get("/mcp/resources")
async def list_all_mcp_resources():
    """List all available MCP resources across servers"""
    all_resources = {}
    
    for server_name, client in mcp_clients.items():
        try:
            resources = await client.list_resources()
            all_resources[server_name] = resources
        except Exception as e:
            all_resources[server_name] = {"error": str(e)}
    
    return {"resources_by_server": all_resources}

class ToolExecutionError(Exception):
    """Exception personnalisée pour erreurs d'outils MCP"""
    def __init__(self, tool: str, error_type: str, detail: str, server: str = "unknown"):
        self.tool = tool
        self.error_type = error_type
        self.detail = detail
        self.server = server
        super().__init__(f"{server}:{tool} - {error_type}: {detail}")

@app.post("/mcp/tool/{server_name}/{tool_name}")
async def call_mcp_tool(server_name: str, tool_name: str, arguments: Dict[str, Any]):
    """Call specific MCP tool with robust error handling"""
    if server_name not in mcp_clients:
        raise HTTPException(status_code=404, detail=f"Server {server_name} not found")
    
    # ✅ 3. VALIDATION PRÉALABLE DES ARGUMENTS
    if not isinstance(arguments, dict):
        raise HTTPException(
            status_code=400, 
            detail=f"MCP Tool call doit contenir un champ `arguments` de type dict, reçu: {type(arguments)}"
        )
    
    try:
        result = await mcp_clients[server_name].call_tool(tool_name, arguments)
        
        # ✅ GESTION ROBUSTE DES ERREURS D'OUTILS
        if isinstance(result, dict) and result.get("is_error", False):
            content = result.get("content", [{}])
            if content and isinstance(content[0], dict):
                error_data = content[0]
                tool_name_error = error_data.get("tool", tool_name)
                error_type = error_data.get("type", "Exception")
                error_detail = error_data.get("error", "Unknown error")
                
                logger.error(f"🚨 Tool execution error: {server_name}:{tool_name_error} - {error_type}: {error_detail}")
                raise ToolExecutionError(tool_name_error, error_type, error_detail, server_name)
        
        return {"result": result, "server": server_name, "tool": tool_name}
        
    except ToolExecutionError as tool_error:
        # Re-raise tool execution errors avec le bon status code
        raise HTTPException(status_code=422, detail=str(tool_error))
    except Exception as e:
        logger.error(f"💥 Hub error calling {server_name}:{tool_name}: {type(e).__name__}: {e}")
        raise HTTPException(status_code=500, detail=f"Hub error: {str(e)}")

@app.get("/mcp/resource/{server_name}/{resource_uri:path}")
async def get_mcp_resource(server_name: str, resource_uri: str):
    """Get specific MCP resource"""
    if server_name not in mcp_clients:
        raise HTTPException(status_code=404, detail=f"Server {server_name} not found")
    
    try:
        result = await mcp_clients[server_name].get_resource(resource_uri)
        return {"result": result, "server": server_name, "resource": resource_uri}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy", 
        "timestamp": datetime.now().isoformat(),
        "mcp_servers_connected": len(mcp_clients),
        "active_websockets": len(active_connections)
    }

@app.post("/mcp/init")
async def manual_init_mcp():
    """Manual MCP initialization endpoint"""
    try:
        connected_servers = await initialize_mcp_ecosystem()
        return {
            "status": "initialized",
            "connected_servers": connected_servers,
            "servers": list(mcp_clients.keys()),
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Manual MCP init failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/mcp/resilience")
async def get_resilience_metrics():
    """Endpoint pour surveiller les métriques de résilience"""
    try:
        report = resilience_layer.get_resilience_report()
        return report
    except Exception as e:
        logger.error(f"Error getting resilience metrics: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(
        app, 
        host="0.0.0.0", 
        port=MCP_HUB_PORT,
        log_level="info",
        access_log=True
    )