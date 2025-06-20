"""
Advanced MCP Ollama Server with Intelligent Model Management
Implements smart model selection, context optimization, and advanced AI capabilities
"""

import asyncio
import json
import logging
import os
import uuid
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional, AsyncGenerator
import httpx
import aiofiles
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import uvicorn
import glob
from fastapi import Request
from fastapi.responses import JSONResponse
from pydantic import ValidationError

# Configuration
OLLAMA_SERVER_PORT = 4003
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "host.docker.internal:11434")
MAX_CONTEXT_LENGTH = 8192
DEFAULT_MODEL = "llama3.2:3b"

# Available models with their capabilities
AVAILABLE_MODELS = {
    "llama3.2:3b": {
        "description": "Fast general-purpose model",
        "strengths": ["general", "coding", "reasoning"],
        "context_length": 8192,
        "speed": "fast"
    },
    "llama3.1:8b": {
        "description": "Balanced model for complex tasks",
        "strengths": ["coding", "analysis", "creative"],
        "context_length": 8192,
        "speed": "medium"
    },
    "qwen2.5-coder:7b": {
        "description": "Specialized coding model",
        "strengths": ["coding", "debugging", "code-review"],
        "context_length": 16384,
        "speed": "medium"
    }
}

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
model_performance_stats: Dict[str, Dict] = {}

# FastAPI app
app = FastAPI(
    title="Advanced MCP Ollama Server",
    version="2.0.0",
    description="Intelligent Ollama integration with model management and optimization"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 🛡️ Handler d'exception Pydantic pour debug
@app.exception_handler(ValidationError)
async def validation_exception_handler(request: Request, exc: ValidationError):
    raw_body = await request.body()
    logger.error(f"💥 Erreur de validation Pydantic: {exc}")
    logger.error(f"🧾 Payload brut reçu: {raw_body}")
    logger.error(f"🔍 Erreurs détaillées: {exc.errors()}")
    return JSONResponse(
        status_code=422,
        content={
            "detail": "Validation error",
            "errors": exc.errors(),
            "raw_body": raw_body.decode('utf-8') if raw_body else None
        }
    )

# Initialize model stats
for model in AVAILABLE_MODELS:
    model_performance_stats[model] = {
        "total_requests": 0,
        "total_tokens": 0,
        "avg_response_time": 0.0,
        "success_rate": 1.0,
        "last_used": None
    }

# MCP Protocol Endpoints
@app.post("/mcp/initialize")
async def initialize_mcp_session(request: MCPInitRequest) -> MCPInitResponse:
    """Initialize MCP session with Ollama capabilities"""
    session_id = str(uuid.uuid4())
    
    # Check Ollama availability
    ollama_available = await _check_ollama_health()
    available_models = await _get_available_models() if ollama_available else []
    
    active_sessions[session_id] = {
        "created_at": datetime.now(),
        "client_info": request.client_info,
        "preferred_model": DEFAULT_MODEL,
        "conversation_history": [],
        "context_length_used": 0,
        "total_requests": 0,
        "ollama_available": ollama_available
    }
    
    logger.info(f"✅ Ollama MCP Session initialized: {session_id} "
                f"(Ollama: {'✅' if ollama_available else '❌'})")
    
    return MCPInitResponse(
        session_id=session_id,
        server_info={
            "name": "jarvis-ollama-server",
            "version": "2.0.0",
            "description": "Advanced Ollama integration with intelligent model selection"
        },
        capabilities={
            "tools": {
                "list_changed": True,
                "supports_progress": True,
                "supports_streaming": True
            },
            "resources": {
                "list_changed": True,
                "supports_templates": True
            },
            "experimental": {
                "model_selection": True,
                "context_optimization": True,
                "performance_tracking": True,
                "streaming_responses": True
            },
            "ollama": {
                "available": ollama_available,
                "models": available_models,
                "host": OLLAMA_HOST
            }
        }
    )

@app.get("/mcp/tools")
async def list_mcp_tools():
    """List all available Ollama tools"""
    tools = [
        MCPTool(
            name="generate_response",
            description="Generate AI response using optimal model selection",
            input_schema={
                "type": "object",
                "properties": {
                    "prompt": {"type": "string", "description": "Input prompt for generation"},
                    "model": {"type": "string", "description": "Specific model to use (optional)"},
                    "task_type": {"type": "string", "description": "Type of task: general, coding, analysis, creative"},
                    "max_tokens": {"type": "integer", "default": 1000, "description": "Maximum tokens to generate"},
                    "temperature": {"type": "number", "default": 0.7, "description": "Temperature for generation"},
                    "context": {"type": "string", "description": "Additional context"},
                    "stream": {"type": "boolean", "default": False, "description": "Enable streaming response"}
                },
                "required": ["prompt"]
            }
        ),
        MCPTool(
            name="chat_conversation",
            description="Multi-turn conversation with context management",
            input_schema={
                "type": "object",
                "properties": {
                    "message": {"type": "string", "description": "User message"},
                    "session_id": {"type": "string", "description": "Session identifier"},
                    "system_prompt": {"type": "string", "description": "System prompt for context"},
                    "maintain_context": {"type": "boolean", "default": True, "description": "Maintain conversation context"},
                    "model": {"type": "string", "description": "Specific model to use"},
                    "task_type": {"type": "string", "description": "Type of conversation task"}
                },
                "required": ["message"]
            }
        ),
        MCPTool(
            name="analyze_code",
            description="Analyze code using specialized coding models",
            input_schema={
                "type": "object",
                "properties": {
                    "code": {"type": "string", "description": "Code to analyze"},
                    "language": {"type": "string", "description": "Programming language"},
                    "analysis_type": {"type": "string", "description": "Type: review, debug, optimize, explain"},
                    "specific_focus": {"type": "string", "description": "Specific area to focus on"},
                    "include_suggestions": {"type": "boolean", "default": True, "description": "Include improvement suggestions"}
                },
                "required": ["code", "analysis_type"]
            }
        ),
        MCPTool(
            name="optimize_prompt",
            description="Optimize prompts for better AI responses",
            input_schema={
                "type": "object",
                "properties": {
                    "original_prompt": {"type": "string", "description": "Original prompt to optimize"},
                    "target_task": {"type": "string", "description": "Target task type"},
                    "target_model": {"type": "string", "description": "Target model for optimization"},
                    "context": {"type": "string", "description": "Additional context"},
                    "optimization_goals": {"type": "array", "items": {"type": "string"}, "description": "Optimization objectives"}
                },
                "required": ["original_prompt", "target_task"]
            }
        ),
        MCPTool(
            name="generate_embeddings",
            description="Generate text embeddings for semantic analysis",
            input_schema={
                "type": "object",
                "properties": {
                    "text": {"type": "string", "description": "Text to generate embeddings for"},
                    "model": {"type": "string", "default": "nomic-embed-text", "description": "Embedding model"},
                    "normalize": {"type": "boolean", "default": True, "description": "Normalize embeddings"}
                },
                "required": ["text"]
            }
        ),
        MCPTool(
            name="compare_models",
            description="Compare responses from different models",
            input_schema={
                "type": "object",
                "properties": {
                    "prompt": {"type": "string", "description": "Prompt to test"},
                    "models": {"type": "array", "items": {"type": "string"}, "description": "Models to compare"},
                    "evaluation_criteria": {"type": "array", "items": {"type": "string"}, "description": "Criteria for comparison"},
                    "include_performance": {"type": "boolean", "default": True, "description": "Include performance metrics"}
                },
                "required": ["prompt"]
            }
        ),
        MCPTool(
            name="get_model_info",
            description="Get detailed information about available models",
            input_schema={
                "type": "object",
                "properties": {
                    "model": {"type": "string", "description": "Specific model (optional)"},
                    "include_performance": {"type": "boolean", "default": True, "description": "Include performance stats"},
                    "include_capabilities": {"type": "boolean", "default": True, "description": "Include capability info"}
                },
                "required": []
            }
        ),
        MCPTool(
            name="manage_context",
            description="Optimize and manage conversation context",
            input_schema={
                "type": "object",
                "properties": {
                    "session_id": {"type": "string", "description": "Session identifier"},
                    "action": {"type": "string", "description": "Action: summarize, trim, optimize, clear"},
                    "max_length": {"type": "integer", "description": "Maximum context length"},
                    "preserve_important": {"type": "boolean", "default": True, "description": "Preserve important context"}
                },
                "required": ["session_id", "action"]
            }
        ),
        MCPTool(
            name="stream_generate",
            description="Generate streaming response with real-time updates",
            input_schema={
                "type": "object",
                "properties": {
                    "prompt": {"type": "string", "description": "Input prompt"},
                    "model": {"type": "string", "description": "Model to use"},
                    "chunk_size": {"type": "integer", "default": 50, "description": "Characters per chunk"},
                    "include_metadata": {"type": "boolean", "default": True, "description": "Include generation metadata"}
                },
                "required": ["prompt"]
            }
        ),
        # 🔧 FILE OPERATIONS TOOLS INTÉGRÉS
        MCPTool(
            name="LS",
            description="List directory contents (equivalent to ls command)",
            input_schema={
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Directory path to list"},
                    "recursive": {"type": "boolean", "default": False, "description": "List recursively"},
                    "include_hidden": {"type": "boolean", "default": False, "description": "Include hidden files"}
                },
                "required": ["path"]
            }
        ),
        MCPTool(
            name="Read",
            description="Read file contents (equivalent to cat/read)",
            input_schema={
                "type": "object",
                "properties": {
                    "file_path": {"type": "string", "description": "File path to read"},
                    "encoding": {"type": "string", "default": "utf-8", "description": "File encoding"},
                    "max_lines": {"type": "integer", "default": 1000, "description": "Maximum lines to read"}
                },
                "required": ["file_path"]
            }
        ),
        MCPTool(
            name="Glob",
            description="Search files by pattern (equivalent to find/glob)",
            input_schema={
                "type": "object",
                "properties": {
                    "pattern": {"type": "string", "description": "Glob pattern (e.g., **/*.js)"},
                    "path": {"type": "string", "default": ".", "description": "Base directory to search"},
                    "max_results": {"type": "integer", "default": 100, "description": "Maximum results"}
                },
                "required": ["pattern"]
            }
        ),
        MCPTool(
            name="Grep",
            description="Search content within files",
            input_schema={
                "type": "object",
                "properties": {
                    "pattern": {"type": "string", "description": "Search pattern"},
                    "path": {"type": "string", "default": ".", "description": "Directory to search"},
                    "file_pattern": {"type": "string", "default": "*", "description": "File pattern to include"}
                },
                "required": ["pattern"]
            }
        )
    ]
    
    return {"tools": [tool.dict() for tool in tools]}

@app.get("/mcp/resources")
async def list_mcp_resources():
    """List available Ollama resources"""
    resources = [
        MCPResource(
            uri="docs://ollama-server",
            name="Ollama Server Documentation",
            description="Complete documentation for Ollama integration",
            mime_type="text/markdown"
        ),
        MCPResource(
            uri="models://available",
            name="Available Models",
            description="List of available Ollama models with capabilities",
            mime_type="application/json"
        ),
        MCPResource(
            uri="stats://performance",
            name="Model Performance Statistics",
            description="Performance statistics for all models",
            mime_type="application/json"
        ),
        MCPResource(
            uri="config://optimization",
            name="Optimization Configuration",
            description="Current optimization settings and strategies",
            mime_type="application/json"
        )
    ]
    
    return {"resources": [resource.dict() for resource in resources]}

# Tool Implementation Endpoints
@app.post("/mcp/tools/generate_response/call")
async def tool_generate_response(raw_request: Request) -> MCPToolResponse:
    """Generate AI response with intelligent model selection"""
    try:
        # Parse JSON payload
        raw_body = await raw_request.body()
        request_data = json.loads(raw_body.decode('utf-8'))
        
        # Handle double-encapsulation from hub
        arguments = request_data.get("arguments", {})
        if isinstance(arguments, dict) and "arguments" in arguments and len(arguments) == 1:
            arguments = arguments["arguments"]
        
        prompt = arguments["prompt"]
        model = arguments.get("model")
        task_type = arguments.get("task_type", "general")
        max_tokens = arguments.get("max_tokens", 1000)
        temperature = arguments.get("temperature", 0.7)
        context = arguments.get("context", "")
        stream = arguments.get("stream", False)
        
        # Select optimal model if not specified
        if not model:
            model = _select_optimal_model(task_type, prompt)
        
        # Build full prompt with context
        full_prompt = _build_prompt_with_context(prompt, context, task_type)
        
        # Generate response
        if stream:
            # For streaming, we'll simulate chunks (real streaming would need WebSocket)
            response_text = await _generate_ollama_response(
                full_prompt, model, max_tokens, temperature
            )
            
            # Simulate streaming by breaking into chunks
            chunks = _create_response_chunks(response_text)
            
            return MCPToolResponse(
                content=[{
                    "type": "text",
                    "text": json.dumps({
                        "response": response_text,
                        "model_used": model,
                        "task_type": task_type,
                        "streaming": True,
                        "chunks": chunks,
                        "metadata": {
                            "prompt_length": len(full_prompt),
                            "response_length": len(response_text),
                            "temperature": temperature,
                            "max_tokens": max_tokens
                        }
                    }, indent=2)
                }]
            )
        else:
            response_text = await _generate_ollama_response(
                full_prompt, model, max_tokens, temperature
            )
            
            return MCPToolResponse(
                content=[{
                    "type": "text",
                    "text": json.dumps({
                        "response": response_text,
                        "model_used": model,
                        "task_type": task_type,
                        "streaming": False,
                        "metadata": {
                            "prompt_length": len(full_prompt),
                            "response_length": len(response_text),
                            "temperature": temperature,
                            "max_tokens": max_tokens
                        }
                    }, indent=2)
                }]
            )
        
    except Exception as e:
        logger.error(f"Error generating response: {e}")
        return MCPToolResponse(
            content=[{
                "type": "text",
                "text": f"Error generating response: {str(e)}"
            }],
            is_error=True
        )

@app.post("/mcp/tools/chat_conversation/call")
async def tool_chat_conversation(raw_request: Request) -> MCPToolResponse:
    """Handle multi-turn conversation with context management"""
    try:
        # Parse JSON payload
        raw_body = await raw_request.body()
        request_data = json.loads(raw_body.decode('utf-8'))
        
        # Handle double-encapsulation from hub
        arguments = request_data.get("arguments", {})
        if isinstance(arguments, dict) and "arguments" in arguments and len(arguments) == 1:
            arguments = arguments["arguments"]
        
        message = arguments["message"]
        session_id = arguments.get("session_id", request_data.get("session_id"))
        system_prompt = arguments.get("system_prompt", "")
        maintain_context = arguments.get("maintain_context", True)
        model = arguments.get("model")
        task_type = arguments.get("task_type", "general")
        
        if not session_id or session_id not in active_sessions:
            raise HTTPException(status_code=404, detail="Session not found")
        
        session = active_sessions[session_id]
        
        # Select model if not specified
        if not model:
            model = _select_optimal_model(task_type, message)
            session["preferred_model"] = model
        
        # Build conversation context
        if maintain_context:
            conversation_context = _build_conversation_context(session, system_prompt)
            full_prompt = f"{conversation_context}\n\nUser: {message}\nAssistant:"
        else:
            full_prompt = f"{system_prompt}\n\nUser: {message}\nAssistant:" if system_prompt else message
        
        # Generate response
        response_text = await _generate_ollama_response(full_prompt, model)
        
        # Update session history
        if maintain_context:
            session["conversation_history"].append({
                "user": message,
                "assistant": response_text,
                "timestamp": datetime.now().isoformat(),
                "model": model
            })
            
            # Trim context if too long
            session = _trim_context_if_needed(session)
        
        session["total_requests"] += 1
        
        return MCPToolResponse(
            content=[{
                "type": "text",
                "text": json.dumps({
                    "response": response_text,
                    "model_used": model,
                    "session_id": session_id,
                    "context_length": session["context_length_used"],
                    "conversation_turns": len(session["conversation_history"]),
                    "metadata": {
                        "maintain_context": maintain_context,
                        "task_type": task_type,
                        "timestamp": datetime.now().isoformat()
                    }
                }, indent=2)
            }]
        )
        
    except Exception as e:
        logger.error(f"Error in chat conversation: {e}")
        return MCPToolResponse(
            content=[{
                "type": "text",
                "text": f"Error in chat conversation: {str(e)}"
            }],
            is_error=True
        )

@app.post("/mcp/tools/analyze_code/call")
async def tool_analyze_code(raw_request: Request) -> MCPToolResponse:
    """Analyze code using specialized models"""
    try:
        # Parse JSON payload
        raw_body = await raw_request.body()
        request_data = json.loads(raw_body.decode('utf-8'))
        
        # Handle double-encapsulation from hub
        arguments = request_data.get("arguments", {})
        if isinstance(arguments, dict) and "arguments" in arguments and len(arguments) == 1:
            arguments = arguments["arguments"]
        
        code = arguments["code"]
        language = arguments.get("language", "unknown")
        analysis_type = arguments["analysis_type"]
        specific_focus = arguments.get("specific_focus", "")
        include_suggestions = arguments.get("include_suggestions", True)
        
        # Use coding-specialized model
        model = "codellama:7b" if "codellama:7b" in AVAILABLE_MODELS else DEFAULT_MODEL
        
        # Build analysis prompt
        analysis_prompt = _build_code_analysis_prompt(
            code, language, analysis_type, specific_focus, include_suggestions
        )
        
        # Generate analysis
        analysis_result = await _generate_ollama_response(analysis_prompt, model)
        
        return MCPToolResponse(
            content=[{
                "type": "text",
                "text": json.dumps({
                    "analysis": analysis_result,
                    "code_snippet": code[:200] + "..." if len(code) > 200 else code,
                    "language": language,
                    "analysis_type": analysis_type,
                    "model_used": model,
                    "specific_focus": specific_focus,
                    "included_suggestions": include_suggestions,
                    "metadata": {
                        "code_length": len(code),
                        "analysis_length": len(analysis_result),
                        "timestamp": datetime.now().isoformat()
                    }
                }, indent=2)
            }]
        )
        
    except Exception as e:
        logger.error(f"Error analyzing code: {e}")
        return MCPToolResponse(
            content=[{
                "type": "text",
                "text": f"Error analyzing code: {str(e)}"
            }],
            is_error=True
        )

# 🔧 FILE OPERATIONS TOOLS - INTÉGRATION DANS OLLAMA
@app.post("/mcp/tools/LS/call")
async def tool_ls(raw_request: Request) -> MCPToolResponse:
    """List directory contents (LS tool for Jarvis training)"""
    logger.info(f"🚀 LS Tool called - Starting execution")
    try:
        # 🛡️ DEBUG PAYLOAD BRUT
        raw_body = await raw_request.body()
        logger.info(f"🧾 LS Tool - Payload brut: {raw_body}")
        
        # Parse JSON manuellement
        import json
        try:
            request_data = json.loads(raw_body.decode('utf-8'))
            logger.info(f"🔍 LS Tool - JSON parsé: {request_data}")
            logger.info(f"🔍 LS Tool - Type: {type(request_data)}")
        except json.JSONDecodeError as e:
            logger.error(f"💥 LS Tool - Erreur JSON parsing: {e}")
            return MCPToolResponse(
                content=[{"type": "text", "text": f"❌ ERREUR LS: JSON invalide - {str(e)}"}],
                is_error=True
            )
        
        # ✅ 1. VALIDATION ROBUSTE DES ARGUMENTS
        def validate_ls_arguments(arguments: any) -> dict:
            """Validation robuste des arguments LS avec erreurs explicites"""
            if not isinstance(arguments, dict):
                raise ValueError(f"`arguments` doit être un dictionnaire, reçu: {type(arguments)}")
            if "path" not in arguments:
                raise KeyError("Champ `path` manquant dans arguments")
            if not isinstance(arguments["path"], str):
                raise TypeError(f"Le champ `path` doit être une chaîne, reçu: {type(arguments['path'])}")
            return arguments
        
        # Extraire et valider les arguments 
        # 🔧 FIX: Le hub double-encapsule les arguments
        raw_arguments = request_data.get("arguments", {})
        logger.info(f"🔍 LS Tool - Arguments bruts: {raw_arguments}")
        
        # Si les arguments sont double-encapsulés (via hub), extraire le niveau interne
        if isinstance(raw_arguments, dict) and "arguments" in raw_arguments and len(raw_arguments) == 1:
            logger.info(f"🔧 LS Tool - Double-encapsulation détectée, extraction niveau interne")
            raw_arguments = raw_arguments["arguments"]
            logger.info(f"🔧 LS Tool - Arguments extraits: {raw_arguments}")
        
        try:
            arguments = validate_ls_arguments(raw_arguments)
            path = arguments["path"]
            recursive = arguments.get("recursive", False)
            include_hidden = arguments.get("include_hidden", False)
            logger.info(f"✅ LS Tool - Arguments validés: path={path}, recursive={recursive}, hidden={include_hidden}")
        except (ValueError, KeyError, TypeError) as validation_error:
            logger.error(f"💥 LS Tool - Erreur de validation: {validation_error}")
            raise validation_error  # Re-raise pour que l'exception handler standardisé l'attrape
        
        path_obj = Path(path)
        if not path_obj.exists():
            return MCPToolResponse(
                content=[{
                    "type": "text",
                    "text": f"❌ ERREUR LS: Le répertoire '{path}' n'existe pas."
                }],
                is_error=True
            )
        
        if not path_obj.is_dir():
            return MCPToolResponse(
                content=[{
                    "type": "text",
                    "text": f"❌ ERREUR LS: '{path}' n'est pas un répertoire."
                }],
                is_error=True
            )
        
        results = []
        
        if recursive:
            for item in path_obj.rglob("*"):
                if not include_hidden and item.name.startswith("."):
                    continue
                results.append({
                    "name": item.name,
                    "path": str(item),
                    "type": "directory" if item.is_dir() else "file",
                    "size": item.stat().st_size if item.is_file() else None
                })
        else:
            for item in path_obj.iterdir():
                if not include_hidden and item.name.startswith("."):
                    continue
                results.append({
                    "name": item.name,
                    "path": str(item),
                    "type": "directory" if item.is_dir() else "file",
                    "size": item.stat().st_size if item.is_file() else None
                })
        
        # Format for Jarvis training - clear structure display
        formatted_output = f"📁 LS - Contenu de {path}:\n\n"
        
        directories = [r for r in results if r["type"] == "directory"]
        files = [r for r in results if r["type"] == "file"]
        
        if directories:
            formatted_output += "📂 RÉPERTOIRES:\n"
            for d in directories:
                formatted_output += f"  📁 {d['name']}/\n"
            formatted_output += "\n"
        
        if files:
            formatted_output += "📄 FICHIERS:\n"
            for f in files:
                size_str = f" ({f['size']} bytes)" if f['size'] else ""
                formatted_output += f"  📄 {f['name']}{size_str}\n"
        
        if not results:
            formatted_output += "📭 Répertoire vide\n"
            
        formatted_output += f"\n✅ LS TERMINÉ - {len(results)} éléments trouvés"
        
        return MCPToolResponse(
            content=[{
                "type": "text",
                "text": formatted_output
            }]
        )
        
    except Exception as e:
        logger.error(f"💥 LS Tool - Exception: {type(e).__name__}: {e}")
        # ✅ 2. RÉPONSE D'ERREUR STANDARDISÉE
        return MCPToolResponse(
            content=[{
                "error": str(e),
                "type": type(e).__name__,
                "tool": "LS",
                "details": {
                    "function": "tool_ls",
                    "server": "ollama",
                    "timestamp": datetime.now().isoformat()
                }
            }],
            is_error=True
        )

@app.post("/mcp/tools/Read/call")
async def tool_read(raw_request: Request) -> MCPToolResponse:
    """Read file contents (Read tool for Jarvis training)"""
    try:
        # Parse JSON payload
        raw_body = await raw_request.body()
        request_data = json.loads(raw_body.decode('utf-8'))
        
        # Handle double-encapsulation from hub
        arguments = request_data.get("arguments", {})
        if isinstance(arguments, dict) and "arguments" in arguments and len(arguments) == 1:
            arguments = arguments["arguments"]
        
        # 🛡️ VALIDATION ARGUMENTS
        if not arguments:
            return MCPToolResponse(
                content=[{"type": "text", "text": "❌ ERREUR READ: Arguments manquants"}],
                is_error=True
            )
        
        if "file_path" not in arguments:
            return MCPToolResponse(
                content=[{"type": "text", "text": "❌ ERREUR READ: Paramètre 'file_path' requis"}],
                is_error=True
            )
            
        file_path = arguments["file_path"]
        encoding = arguments.get("encoding", "utf-8")
        max_lines = arguments.get("max_lines", 1000)
        
        path_obj = Path(file_path)
        if not path_obj.exists():
            return MCPToolResponse(
                content=[{
                    "type": "text",
                    "text": f"❌ ERREUR READ: Le fichier '{file_path}' n'existe pas."
                }],
                is_error=True
            )
        
        if not path_obj.is_file():
            return MCPToolResponse(
                content=[{
                    "type": "text",
                    "text": f"❌ ERREUR READ: '{file_path}' n'est pas un fichier."
                }],
                is_error=True
            )
        
        # 🛡️ COUCHE DE RÉSILIENCE - Multi-encodage avec fallbacks
        encodings_to_try = [
            encoding,  # Encodage demandé en premier
            'utf-8',
            'latin-1', 
            'cp1252',
            'iso-8859-1',
            'utf-16'
        ]
        
        # Supprimer les doublons tout en gardant l'ordre
        seen = set()
        unique_encodings = []
        for enc in encodings_to_try:
            if enc not in seen:
                unique_encodings.append(enc)
                seen.add(enc)
        
        last_error = None
        
        # Tentative avec chaque encodage
        for attempt_encoding in unique_encodings:
            try:
                async with aiofiles.open(file_path, mode='r', encoding=attempt_encoding) as f:
                    lines = []
                    line_count = 0
                    async for line in f:
                        if line_count >= max_lines:
                            break
                        lines.append(f"{line_count + 1:4d}→{line.rstrip()}")
                        line_count += 1
                    
                    content = "\n".join(lines)
                    
                    # Format for Jarvis training avec info résilience
                    formatted_output = f"📖 READ - Contenu de {file_path}:\n"
                    if attempt_encoding != encoding:
                        formatted_output += f"🛡️ RÉSILIENCE: Encodage {attempt_encoding} utilisé (demandé: {encoding})\n"
                    formatted_output += f"\n{content}"
                    if line_count >= max_lines:
                        formatted_output += f"\n\n⚠️ Fichier tronqué à {max_lines} lignes"
                    formatted_output += f"\n\n✅ READ TERMINÉ - {line_count} lignes lues"
                    
                    return MCPToolResponse(
                        content=[{
                            "type": "text",
                            "text": formatted_output
                        }]
                    )
                    
            except UnicodeDecodeError as e:
                last_error = e
                logger.warning(f"🛡️ RÉSILIENCE: Échec encodage {attempt_encoding} pour {file_path}")
                continue
            except Exception as e:
                last_error = e
                logger.error(f"🛡️ RÉSILIENCE: Erreur inattendue avec {attempt_encoding}: {e}")
                continue
        
        # 🔄 FALLBACK FINAL - Lecture binaire si tout échoue
        try:
            logger.info(f"🛡️ RÉSILIENCE: Tentative lecture binaire pour {file_path}")
            with open(file_path, 'rb') as f:
                binary_content = f.read(min(1024 * 10, max_lines * 100))  # Max 10KB
                
            # Essai de détection automatique d'encodage
            try:
                import chardet
                detected = chardet.detect(binary_content)
                if detected['confidence'] > 0.7:
                    content = binary_content.decode(detected['encoding'])
                    lines = content.split('\n')[:max_lines]
                    formatted_lines = [f"{i+1:4d}→{line}" for i, line in enumerate(lines)]
                    
                    formatted_output = f"📖 READ - Contenu de {file_path}:\n"
                    formatted_output += f"🛡️ RÉSILIENCE: Encodage auto-détecté {detected['encoding']} (confiance: {detected['confidence']:.2f})\n"
                    formatted_output += f"\n" + "\n".join(formatted_lines)
                    formatted_output += f"\n\n✅ READ TERMINÉ - {len(lines)} lignes lues (fallback binaire)"
                    
                    return MCPToolResponse(
                        content=[{
                            "type": "text", 
                            "text": formatted_output
                        }]
                    )
            except ImportError:
                pass  # chardet pas disponible
            
            # Dernier recours - affichage hexadécimal partiel
            hex_content = binary_content[:500].hex()
            formatted_output = f"📖 READ - Contenu de {file_path}:\n"
            formatted_output += f"🛡️ RÉSILIENCE: Fichier binaire non-textuel\n"
            formatted_output += f"📊 Taille: {len(binary_content)} bytes\n"
            formatted_output += f"🔍 Début (hex): {hex_content[:100]}...\n"
            formatted_output += f"\n⚠️ READ PARTIEL - Fichier non-textuel détecté"
            
            return MCPToolResponse(
                content=[{
                    "type": "text",
                    "text": formatted_output
                }],
                is_error=False  # Pas d'erreur - juste un avertissement
            )
            
        except Exception as final_error:
            return MCPToolResponse(
                content=[{
                    "type": "text",
                    "text": f"❌ ERREUR READ CRITIQUE: Tous les fallbacks ont échoué\n🛡️ Encodages testés: {', '.join(unique_encodings)}\n💥 Erreur finale: {str(final_error)}"
                }],
                is_error=True
            )
        
    except Exception as e:
        logger.error(f"Error in Read tool: {e}")
        return MCPToolResponse(
            content=[{
                "type": "text",
                "text": f"❌ ERREUR READ: {str(e)}"
            }],
            is_error=True
        )

@app.post("/mcp/tools/Glob/call")
async def tool_glob(raw_request: Request) -> MCPToolResponse:
    """Search files by pattern (Glob tool for Jarvis training)"""
    try:
        # Parse JSON payload
        raw_body = await raw_request.body()
        request_data = json.loads(raw_body.decode('utf-8'))
        
        # Handle double-encapsulation from hub
        arguments = request_data.get("arguments", {})
        if isinstance(arguments, dict) and "arguments" in arguments and len(arguments) == 1:
            arguments = arguments["arguments"]
        
        pattern = arguments["pattern"]
        base_path = arguments.get("path", ".")
        max_results = arguments.get("max_results", 100)
        
        # Convert to absolute path
        base_path_obj = Path(base_path).resolve()
        
        if not base_path_obj.exists():
            return MCPToolResponse(
                content=[{
                    "type": "text",
                    "text": f"❌ ERREUR GLOB: Le répertoire '{base_path}' n'existe pas."
                }],
                is_error=True
            )
        
        # Use glob to find matching files
        search_pattern = str(base_path_obj / pattern)
        matches = glob.glob(search_pattern, recursive=True)
        
        # Limit results
        matches = matches[:max_results]
        
        # Sort for consistent output
        matches.sort()
        
        # Format for Jarvis training
        formatted_output = f"🔍 GLOB - Recherche pattern '{pattern}' dans {base_path}:\n\n"
        
        if matches:
            formatted_output += "📋 RÉSULTATS TROUVÉS:\n"
            for match in matches:
                path_obj = Path(match)
                file_type = "📁" if path_obj.is_dir() else "📄"
                formatted_output += f"  {file_type} {match}\n"
        else:
            formatted_output += "📭 Aucun fichier trouvé correspondant au pattern\n"
        
        formatted_output += f"\n✅ GLOB TERMINÉ - {len(matches)} fichiers trouvés"
        
        return MCPToolResponse(
            content=[{
                "type": "text",
                "text": formatted_output
            }]
        )
        
    except Exception as e:
        logger.error(f"Error in Glob tool: {e}")
        return MCPToolResponse(
            content=[{
                "type": "text",
                "text": f"❌ ERREUR GLOB: {str(e)}"
            }],
            is_error=True
        )

@app.post("/mcp/tools/Grep/call")
async def tool_grep(raw_request: Request) -> MCPToolResponse:
    """Search content within files (Grep tool for Jarvis training)"""
    try:
        # Parse JSON payload
        raw_body = await raw_request.body()
        request_data = json.loads(raw_body.decode('utf-8'))
        
        # Handle double-encapsulation from hub
        arguments = request_data.get("arguments", {})
        if isinstance(arguments, dict) and "arguments" in arguments and len(arguments) == 1:
            arguments = arguments["arguments"]
        
        pattern = arguments["pattern"]
        base_path = arguments.get("path", ".")
        file_pattern = arguments.get("file_pattern", "*")
        
        base_path_obj = Path(base_path)
        if not base_path_obj.exists():
            return MCPToolResponse(
                content=[{
                    "type": "text",
                    "text": f"❌ ERREUR GREP: Le répertoire '{base_path}' n'existe pas."
                }],
                is_error=True
            )
        
        matches = []
        search_files = glob.glob(str(base_path_obj / "**" / file_pattern), recursive=True)
        
        for file_path in search_files:
            if Path(file_path).is_file():
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        for line_num, line in enumerate(f, 1):
                            if pattern.lower() in line.lower():
                                matches.append({
                                    "file": file_path,
                                    "line_number": line_num,
                                    "line_content": line.strip()
                                })
                                if len(matches) >= 50:  # Limit matches
                                    break
                except (UnicodeDecodeError, PermissionError):
                    continue
        
        # Format for Jarvis training
        formatted_output = f"🔎 GREP - Recherche '{pattern}' dans {base_path}:\n\n"
        
        if matches:
            formatted_output += "📋 CORRESPONDANCES TROUVÉES:\n"
            for match in matches[:20]:  # Show first 20 matches
                formatted_output += f"  📄 {match['file']}:{match['line_number']} → {match['line_content']}\n"
            
            if len(matches) > 20:
                formatted_output += f"  ... et {len(matches) - 20} autres correspondances\n"
        else:
            formatted_output += "📭 Aucune correspondance trouvée\n"
        
        formatted_output += f"\n✅ GREP TERMINÉ - {len(matches)} correspondances trouvées"
        
        return MCPToolResponse(
            content=[{
                "type": "text",
                "text": formatted_output
            }]
        )
        
    except Exception as e:
        logger.error(f"Error in Grep tool: {e}")
        return MCPToolResponse(
            content=[{
                "type": "text",
                "text": f"❌ ERREUR GREP: {str(e)}"
            }],
            is_error=True
        )

@app.post("/mcp/tools/get_model_info/call")
async def tool_get_model_info(raw_request: Request) -> MCPToolResponse:
    """Get information about available models"""
    try:
        # Parse JSON payload
        raw_body = await raw_request.body()
        request_data = json.loads(raw_body.decode('utf-8'))
        
        # Handle double-encapsulation from hub
        arguments = request_data.get("arguments", {})
        if isinstance(arguments, dict) and "arguments" in arguments and len(arguments) == 1:
            arguments = arguments["arguments"]
        
        specific_model = arguments.get("model")
        include_performance = arguments.get("include_performance", True)
        include_capabilities = arguments.get("include_capabilities", True)
        
        if specific_model:
            if specific_model not in AVAILABLE_MODELS:
                raise HTTPException(status_code=404, detail="Model not found")
            
            model_info = dict(AVAILABLE_MODELS[specific_model])
            
            if include_performance and specific_model in model_performance_stats:
                model_info["performance"] = model_performance_stats[specific_model]
            
            result = {specific_model: model_info}
        else:
            result = {}
            for model_name, model_data in AVAILABLE_MODELS.items():
                model_info = dict(model_data)
                
                if include_performance and model_name in model_performance_stats:
                    model_info["performance"] = model_performance_stats[model_name]
                
                result[model_name] = model_info
        
        # Add general info
        info = {
            "models": result,
            "default_model": DEFAULT_MODEL,
            "ollama_host": OLLAMA_HOST,
            "available": await _check_ollama_health(),
            "total_models": len(AVAILABLE_MODELS)
        }
        
        if include_capabilities:
            info["capabilities"] = {
                "supported_tasks": ["general", "coding", "analysis", "creative", "reasoning"],
                "max_context_length": MAX_CONTEXT_LENGTH,
                "streaming_support": True,
                "embedding_support": True
            }
        
        return MCPToolResponse(
            content=[{
                "type": "text",
                "text": json.dumps(info, indent=2)
            }]
        )
        
    except Exception as e:
        logger.error(f"Error getting model info: {e}")
        return MCPToolResponse(
            content=[{
                "type": "text",
                "text": f"Error getting model info: {str(e)}"
            }],
            is_error=True
        )

# Helper functions
async def _check_ollama_health() -> bool:
    """Check if Ollama is available"""
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.get(f"http://{OLLAMA_HOST}/api/version")
            return response.status_code == 200
    except:
        return False

async def _get_available_models() -> List[str]:
    """Get list of available models from Ollama"""
    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.get(f"http://{OLLAMA_HOST}/api/tags")
            if response.status_code == 200:
                data = response.json()
                return [model["name"] for model in data.get("models", [])]
    except:
        pass
    return list(AVAILABLE_MODELS.keys())

def _select_optimal_model(task_type: str, prompt: str) -> str:
    """Select optimal model based on task type and prompt analysis"""
    prompt_lower = prompt.lower()
    
    # Analyze prompt for coding indicators
    coding_keywords = ["code", "function", "class", "debug", "programming", "algorithm"]
    if task_type == "coding" or any(keyword in prompt_lower for keyword in coding_keywords):
        if "qwen2.5-coder:7b" in AVAILABLE_MODELS:
            return "qwen2.5-coder:7b"
    
    # Analysis tasks
    if task_type == "analysis" or "analyze" in prompt_lower:
        if "llama3.1:8b" in AVAILABLE_MODELS:
            return "llama3.1:8b"
    
    # Creative tasks
    if task_type == "creative" or any(word in prompt_lower for word in ["creative", "story", "write"]):
        if "llama3.1:8b" in AVAILABLE_MODELS:
            return "llama3.1:8b"
    
    # Default to fastest general model
    return DEFAULT_MODEL

def _build_prompt_with_context(prompt: str, context: str, task_type: str) -> str:
    """Build optimized prompt with context"""
    parts = []
    
    # Add task-specific system prompt
    if task_type == "coding":
        parts.append("You are an expert software developer. Provide clear, efficient, and well-documented code solutions.")
    elif task_type == "analysis":
        parts.append("You are an analytical expert. Provide thorough, structured analysis with clear reasoning.")
    elif task_type == "creative":
        parts.append("You are a creative writing assistant. Generate engaging, original, and well-crafted content.")
    
    # Add context if provided
    if context:
        parts.append(f"Context: {context}")
    
    # Add main prompt
    parts.append(f"Request: {prompt}")
    
    return "\n\n".join(parts)

def _build_conversation_context(session: Dict, system_prompt: str) -> str:
    """Build conversation context from session history"""
    context_parts = []
    
    if system_prompt:
        context_parts.append(f"System: {system_prompt}")
    
    # Add recent conversation history
    history = session["conversation_history"][-5:]  # Last 5 exchanges
    for exchange in history:
        context_parts.append(f"User: {exchange['user']}")
        context_parts.append(f"Assistant: {exchange['assistant']}")
    
    context = "\n".join(context_parts)
    session["context_length_used"] = len(context)
    
    return context

def _trim_context_if_needed(session: Dict) -> Dict:
    """Trim context if it exceeds maximum length"""
    if session["context_length_used"] > MAX_CONTEXT_LENGTH:
        # Keep only the most recent exchanges
        while len(session["conversation_history"]) > 3 and session["context_length_used"] > MAX_CONTEXT_LENGTH:
            session["conversation_history"].pop(0)
            # Recalculate context length
            context = _build_conversation_context(session, "")
            session["context_length_used"] = len(context)
    
    return session

def _build_code_analysis_prompt(code: str, language: str, analysis_type: str, 
                               specific_focus: str, include_suggestions: bool) -> str:
    """Build specialized prompt for code analysis"""
    prompt_parts = [
        f"Analyze the following {language} code with focus on {analysis_type}:"
    ]
    
    if specific_focus:
        prompt_parts.append(f"Pay special attention to: {specific_focus}")
    
    prompt_parts.extend([
        "",
        "```" + language,
        code,
        "```",
        "",
        f"Please provide a {analysis_type} analysis including:"
    ])
    
    if analysis_type == "review":
        prompt_parts.extend([
            "- Code quality assessment",
            "- Best practices compliance",
            "- Potential issues or bugs",
            "- Readability and maintainability"
        ])
    elif analysis_type == "debug":
        prompt_parts.extend([
            "- Identify potential bugs",
            "- Logic errors",
            "- Runtime issues",
            "- Edge cases"
        ])
    elif analysis_type == "optimize":
        prompt_parts.extend([
            "- Performance bottlenecks",
            "- Memory usage optimization",
            "- Algorithm improvements",
            "- Efficiency enhancements"
        ])
    elif analysis_type == "explain":
        prompt_parts.extend([
            "- Code functionality explanation",
            "- Algorithm description",
            "- Data flow analysis",
            "- Purpose and design patterns"
        ])
    
    if include_suggestions:
        prompt_parts.append("- Specific improvement suggestions with examples")
    
    return "\n".join(prompt_parts)

def _create_response_chunks(response: str, chunk_size: int = 50) -> List[Dict]:
    """Create response chunks for streaming simulation"""
    chunks = []
    words = response.split()
    
    for i in range(0, len(words), chunk_size):
        chunk_words = words[i:i + chunk_size]
        chunk_text = " ".join(chunk_words)
        
        chunks.append({
            "chunk_id": i // chunk_size,
            "text": chunk_text,
            "is_final": i + chunk_size >= len(words),
            "timestamp": datetime.now().isoformat()
        })
    
    return chunks

async def _generate_ollama_response(prompt: str, model: str, max_tokens: int = 1000, 
                                   temperature: float = 0.7) -> str:
    """Generate response from Ollama with performance tracking"""
    start_time = datetime.now()
    
    try:
        async with httpx.AsyncClient(timeout=180.0) as client:
            response = await client.post(
                f"http://{OLLAMA_HOST}/api/generate",
                json={
                    "model": model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": temperature,
                        "num_predict": min(max_tokens, 500),  # Limit response length
                        "top_p": 0.9,
                        "top_k": 20,  # Reduce choices for speed
                        "repeat_penalty": 1.1,
                        "num_gpu": 1  # Use GPU if available
                    }
                }
            )
            
            if response.status_code == 200:
                data = response.json()
                response_text = data.get("response", "")
                
                # Update performance stats
                end_time = datetime.now()
                response_time = (end_time - start_time).total_seconds()
                
                _update_model_performance(model, len(prompt + response_text), response_time, True)
                
                return response_text
            else:
                error_msg = f"Ollama error: HTTP {response.status_code}"
                _update_model_performance(model, len(prompt), 0, False)
                return error_msg
                
    except Exception as e:
        _update_model_performance(model, len(prompt), 0, False)
        return f"Error connecting to Ollama: {str(e)}"

def _update_model_performance(model: str, tokens: int, response_time: float, success: bool):
    """Update model performance statistics"""
    if model not in model_performance_stats:
        model_performance_stats[model] = {
            "total_requests": 0,
            "total_tokens": 0,
            "avg_response_time": 0.0,
            "success_rate": 1.0,
            "last_used": None
        }
    
    stats = model_performance_stats[model]
    stats["total_requests"] += 1
    stats["total_tokens"] += tokens
    stats["last_used"] = datetime.now().isoformat()
    
    # Update average response time
    if success:
        old_avg = stats["avg_response_time"]
        stats["avg_response_time"] = (old_avg * (stats["total_requests"] - 1) + response_time) / stats["total_requests"]
    
    # Update success rate
    old_success_count = int(stats["success_rate"] * (stats["total_requests"] - 1))
    new_success_count = old_success_count + (1 if success else 0)
    stats["success_rate"] = new_success_count / stats["total_requests"]

@app.post("/mcp/close")
async def close_mcp_session(request: Dict[str, str]):
    """Close MCP session"""
    session_id = request.get("session_id")
    if session_id in active_sessions:
        session_stats = active_sessions[session_id]
        logger.info(f"🔌 Ollama Session closed: {session_id} "
                   f"(requests: {session_stats['total_requests']}, "
                   f"model: {session_stats.get('preferred_model', 'unknown')})")
        del active_sessions[session_id]
    
    return {"status": "closed", "session_id": session_id}

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    ollama_health = await _check_ollama_health()
    
    return {
        "status": "healthy",
        "server": "advanced-mcp-ollama",
        "version": "2.0.0",
        "active_sessions": len(active_sessions),
        "ollama_available": ollama_health,
        "ollama_host": OLLAMA_HOST,
        "available_models": len(AVAILABLE_MODELS),
        "timestamp": datetime.now().isoformat()
    }

if __name__ == "__main__":
    logger.info("🦙 Starting Advanced MCP Ollama Server...")
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=OLLAMA_SERVER_PORT,
        log_level="info"
    )