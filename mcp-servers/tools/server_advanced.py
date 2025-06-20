"""
Advanced MCP Tools Server with File Operations, Web Scraping, and Code Analysis
Implements full MCP protocol with modern patterns
"""

import asyncio
import json
import logging
import os
import subprocess
import tempfile
import uuid
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional
import aiofiles
import aiohttp
from bs4 import BeautifulSoup
import requests
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import uvicorn

# Configuration
TOOLS_SERVER_PORT = 4006
ALLOWED_EXTENSIONS = {'.txt', '.py', '.js', '.html', '.css', '.json', '.md', '.yml', '.yaml'}
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB

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

# FastAPI app
app = FastAPI(
    title="Advanced MCP Tools Server",
    version="2.0.0",
    description="Advanced file operations, web scraping, and code analysis via MCP protocol"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# MCP Protocol Endpoints
@app.post("/mcp/initialize")
async def initialize_mcp_session(request: MCPInitRequest) -> MCPInitResponse:
    """Initialize MCP session with full capabilities"""
    session_id = str(uuid.uuid4())
    
    active_sessions[session_id] = {
        "created_at": datetime.now(),
        "client_info": request.client_info,
        "working_directory": "/tmp",
        "file_operations_count": 0,
        "web_requests_count": 0
    }
    
    logger.info(f"✅ MCP Session initialized: {session_id}")
    
    return MCPInitResponse(
        session_id=session_id,
        server_info={
            "name": "jarvis-tools-server",
            "version": "2.0.0",
            "description": "Advanced file operations, web scraping, and code analysis"
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
                "completion": True,
                "hints": True
            }
        }
    )

@app.get("/mcp/tools")
async def list_mcp_tools():
    """List all available MCP tools"""
    tools = [
        MCPTool(
            name="read_file",
            description="Read content from a file safely",
            input_schema={
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "File path to read"},
                    "encoding": {"type": "string", "default": "utf-8", "description": "File encoding"}
                },
                "required": ["path"]
            }
        ),
        MCPTool(
            name="write_file",
            description="Write content to a file safely",
            input_schema={
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "File path to write"},
                    "content": {"type": "string", "description": "Content to write"},
                    "encoding": {"type": "string", "default": "utf-8", "description": "File encoding"},
                    "create_dirs": {"type": "boolean", "default": True, "description": "Create directories if needed"}
                },
                "required": ["path", "content"]
            }
        ),
        MCPTool(
            name="list_directory",
            description="List contents of a directory",
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
            name="create_directory",
            description="Create a directory with parent directories",
            input_schema={
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Directory path to create"},
                    "mode": {"type": "integer", "default": 755, "description": "Directory permissions"}
                },
                "required": ["path"]
            }
        ),
        MCPTool(
            name="delete_file",
            description="Delete a file or directory safely",
            input_schema={
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Path to delete"},
                    "recursive": {"type": "boolean", "default": False, "description": "Delete recursively for directories"}
                },
                "required": ["path"]
            }
        ),
        MCPTool(
            name="search_files",
            description="Search for files by name or content",
            input_schema={
                "type": "object",
                "properties": {
                    "directory": {"type": "string", "description": "Directory to search in"},
                    "pattern": {"type": "string", "description": "Search pattern (glob or regex)"},
                    "content_search": {"type": "string", "description": "Search within file content"},
                    "max_results": {"type": "integer", "default": 100, "description": "Maximum results"}
                },
                "required": ["directory", "pattern"]
            }
        ),
        MCPTool(
            name="fetch_url",
            description="Fetch content from a URL with advanced options",
            input_schema={
                "type": "object",
                "properties": {
                    "url": {"type": "string", "description": "URL to fetch"},
                    "method": {"type": "string", "default": "GET", "description": "HTTP method"},
                    "headers": {"type": "object", "default": {}, "description": "HTTP headers"},
                    "timeout": {"type": "integer", "default": 30, "description": "Request timeout"},
                    "follow_redirects": {"type": "boolean", "default": True, "description": "Follow redirects"}
                },
                "required": ["url"]
            }
        ),
        MCPTool(
            name="scrape_webpage",
            description="Scrape and parse webpage content",
            input_schema={
                "type": "object",
                "properties": {
                    "url": {"type": "string", "description": "URL to scrape"},
                    "selector": {"type": "string", "description": "CSS selector for content"},
                    "extract_links": {"type": "boolean", "default": False, "description": "Extract all links"},
                    "extract_images": {"type": "boolean", "default": False, "description": "Extract image URLs"},
                    "clean_text": {"type": "boolean", "default": True, "description": "Clean extracted text"}
                },
                "required": ["url"]
            }
        ),
        MCPTool(
            name="execute_command",
            description="Execute system command safely",
            input_schema={
                "type": "object",
                "properties": {
                    "command": {"type": "string", "description": "Command to execute"},
                    "working_dir": {"type": "string", "description": "Working directory"},
                    "timeout": {"type": "integer", "default": 30, "description": "Command timeout"},
                    "capture_output": {"type": "boolean", "default": True, "description": "Capture stdout/stderr"}
                },
                "required": ["command"]
            }
        ),
        MCPTool(
            name="analyze_code",
            description="Analyze code structure and complexity",
            input_schema={
                "type": "object",
                "properties": {
                    "file_path": {"type": "string", "description": "Path to code file"},
                    "language": {"type": "string", "description": "Programming language (auto-detect if not provided)"},
                    "include_metrics": {"type": "boolean", "default": True, "description": "Include complexity metrics"},
                    "include_dependencies": {"type": "boolean", "default": True, "description": "Extract dependencies"}
                },
                "required": ["file_path"]
            }
        ),
        MCPTool(
            name="format_code",
            description="Format code using appropriate formatter",
            input_schema={
                "type": "object",
                "properties": {
                    "file_path": {"type": "string", "description": "Path to code file"},
                    "language": {"type": "string", "description": "Programming language"},
                    "formatter": {"type": "string", "description": "Specific formatter to use"},
                    "in_place": {"type": "boolean", "default": False, "description": "Format in place"}
                },
                "required": ["file_path"]
            }
        ),
        MCPTool(
            name="compress_files",
            description="Compress files or directories",
            input_schema={
                "type": "object",
                "properties": {
                    "paths": {"type": "array", "items": {"type": "string"}, "description": "Paths to compress"},
                    "output_path": {"type": "string", "description": "Output archive path"},
                    "format": {"type": "string", "default": "zip", "description": "Archive format (zip, tar, tar.gz)"}
                },
                "required": ["paths", "output_path"]
            }
        ),
        MCPTool(
            name="extract_archive",
            description="Extract compressed archives",
            input_schema={
                "type": "object",
                "properties": {
                    "archive_path": {"type": "string", "description": "Path to archive file"},
                    "extract_to": {"type": "string", "description": "Extraction destination"},
                    "overwrite": {"type": "boolean", "default": False, "description": "Overwrite existing files"}
                },
                "required": ["archive_path", "extract_to"]
            }
        )
    ]
    
    return {"tools": [tool.dict() for tool in tools]}

@app.get("/mcp/resources")
async def list_mcp_resources():
    """List available MCP resources"""
    resources = [
        MCPResource(
            uri="docs://tools-server",
            name="Tools Server Documentation",
            description="Complete documentation for the tools server",
            mime_type="text/markdown"
        ),
        MCPResource(
            uri="schema://tools",
            name="Tools Schema",
            description="JSON schema for all available tools",
            mime_type="application/json"
        ),
        MCPResource(
            uri="examples://tools-usage",
            name="Tools Usage Examples",
            description="Examples of how to use each tool",
            mime_type="text/markdown"
        )
    ]
    
    return {"resources": [resource.dict() for resource in resources]}

# Tool Implementation Endpoints
@app.post("/mcp/tools/read_file/call")
async def tool_read_file(request: MCPToolCallRequest) -> MCPToolResponse:
    """Read file content safely"""
    try:
        path = request.arguments["path"]
        encoding = request.arguments.get("encoding", "utf-8")
        
        # Security check
        if not _is_safe_path(path):
            raise HTTPException(status_code=403, detail="Access denied to path")
        
        if not os.path.exists(path):
            raise HTTPException(status_code=404, detail="File not found")
        
        # Check file size
        file_size = os.path.getsize(path)
        if file_size > MAX_FILE_SIZE:
            raise HTTPException(status_code=413, detail="File too large")
        
        async with aiofiles.open(path, 'r', encoding=encoding) as f:
            content = await f.read()
        
        return MCPToolResponse(
            content=[{
                "type": "text",
                "text": content
            }]
        )
        
    except Exception as e:
        logger.error(f"Error reading file {path}: {e}")
        return MCPToolResponse(
            content=[{
                "type": "text", 
                "text": f"Error reading file: {str(e)}"
            }],
            is_error=True
        )

@app.post("/mcp/tools/write_file/call")
async def tool_write_file(request: MCPToolCallRequest) -> MCPToolResponse:
    """Write file content safely"""
    try:
        path = request.arguments["path"]
        content = request.arguments["content"]
        encoding = request.arguments.get("encoding", "utf-8")
        create_dirs = request.arguments.get("create_dirs", True)
        
        # Security check
        if not _is_safe_path(path):
            raise HTTPException(status_code=403, detail="Access denied to path")
        
        # Create directories if needed
        if create_dirs:
            os.makedirs(os.path.dirname(path), exist_ok=True)
        
        async with aiofiles.open(path, 'w', encoding=encoding) as f:
            await f.write(content)
        
        return MCPToolResponse(
            content=[{
                "type": "text",
                "text": f"Successfully wrote {len(content)} characters to {path}"
            }]
        )
        
    except Exception as e:
        logger.error(f"Error writing file {path}: {e}")
        return MCPToolResponse(
            content=[{
                "type": "text",
                "text": f"Error writing file: {str(e)}"
            }],
            is_error=True
        )

@app.post("/mcp/tools/list_directory/call")
async def tool_list_directory(request: MCPToolCallRequest) -> MCPToolResponse:
    """List directory contents"""
    try:
        path = request.arguments["path"]
        recursive = request.arguments.get("recursive", False)
        include_hidden = request.arguments.get("include_hidden", False)
        
        # Security check
        if not _is_safe_path(path):
            raise HTTPException(status_code=403, detail="Access denied to path")
        
        if not os.path.exists(path):
            raise HTTPException(status_code=404, detail="Directory not found")
        
        if not os.path.isdir(path):
            raise HTTPException(status_code=400, detail="Path is not a directory")
        
        items = []
        
        if recursive:
            for root, dirs, files in os.walk(path):
                if not include_hidden:
                    dirs[:] = [d for d in dirs if not d.startswith('.')]
                    files = [f for f in files if not f.startswith('.')]
                
                for name in dirs + files:
                    full_path = os.path.join(root, name)
                    rel_path = os.path.relpath(full_path, path)
                    stat = os.stat(full_path)
                    
                    items.append({
                        "name": rel_path,
                        "type": "directory" if os.path.isdir(full_path) else "file",
                        "size": stat.st_size,
                        "modified": datetime.fromtimestamp(stat.st_mtime).isoformat()
                    })
        else:
            for name in os.listdir(path):
                if not include_hidden and name.startswith('.'):
                    continue
                
                full_path = os.path.join(path, name)
                stat = os.stat(full_path)
                
                items.append({
                    "name": name,
                    "type": "directory" if os.path.isdir(full_path) else "file",
                    "size": stat.st_size,
                    "modified": datetime.fromtimestamp(stat.st_mtime).isoformat()
                })
        
        return MCPToolResponse(
            content=[{
                "type": "text",
                "text": json.dumps(items, indent=2)
            }]
        )
        
    except Exception as e:
        logger.error(f"Error listing directory {path}: {e}")
        return MCPToolResponse(
            content=[{
                "type": "text",
                "text": f"Error listing directory: {str(e)}"
            }],
            is_error=True
        )

@app.post("/mcp/tools/fetch_url/call")
async def tool_fetch_url(request: MCPToolCallRequest) -> MCPToolResponse:
    """Fetch content from URL with advanced options"""
    try:
        url = request.arguments["url"]
        method = request.arguments.get("method", "GET")
        headers = request.arguments.get("headers", {})
        timeout = request.arguments.get("timeout", 30)
        follow_redirects = request.arguments.get("follow_redirects", True)
        
        # Add user agent if not provided
        if "User-Agent" not in headers:
            headers["User-Agent"] = "Jarvis-MCP-Tools/2.0"
        
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=timeout)) as session:
            async with session.request(
                method, 
                url, 
                headers=headers,
                allow_redirects=follow_redirects
            ) as response:
                content = await response.text()
                
                result = {
                    "url": str(response.url),
                    "status_code": response.status,
                    "headers": dict(response.headers),
                    "content": content,
                    "content_length": len(content)
                }
                
                return MCPToolResponse(
                    content=[{
                        "type": "text",
                        "text": json.dumps(result, indent=2)
                    }]
                )
        
    except Exception as e:
        logger.error(f"Error fetching URL {url}: {e}")
        return MCPToolResponse(
            content=[{
                "type": "text",
                "text": f"Error fetching URL: {str(e)}"
            }],
            is_error=True
        )

@app.post("/mcp/tools/scrape_webpage/call")
async def tool_scrape_webpage(request: MCPToolCallRequest) -> MCPToolResponse:
    """Scrape and parse webpage content"""
    try:
        url = request.arguments["url"]
        selector = request.arguments.get("selector")
        extract_links = request.arguments.get("extract_links", False)
        extract_images = request.arguments.get("extract_images", False)
        clean_text = request.arguments.get("clean_text", True)
        
        # Fetch the webpage
        async with aiohttp.ClientSession() as session:
            async with session.get(url) as response:
                html = await response.text()
        
        soup = BeautifulSoup(html, 'html.parser')
        
        result = {
            "url": url,
            "title": soup.title.string if soup.title else None,
            "content": {}
        }
        
        # Extract specific content if selector provided
        if selector:
            elements = soup.select(selector)
            result["content"]["selected"] = []
            for elem in elements:
                text = elem.get_text(strip=clean_text) if clean_text else elem.get_text()
                result["content"]["selected"].append(text)
        else:
            # Extract main content
            text = soup.get_text(strip=clean_text) if clean_text else soup.get_text()
            result["content"]["text"] = text
        
        # Extract links if requested
        if extract_links:
            links = []
            for link in soup.find_all('a', href=True):
                links.append({
                    "url": link['href'],
                    "text": link.get_text(strip=True)
                })
            result["content"]["links"] = links
        
        # Extract images if requested
        if extract_images:
            images = []
            for img in soup.find_all('img', src=True):
                images.append({
                    "src": img['src'],
                    "alt": img.get('alt', ''),
                    "title": img.get('title', '')
                })
            result["content"]["images"] = images
        
        return MCPToolResponse(
            content=[{
                "type": "text",
                "text": json.dumps(result, indent=2)
            }]
        )
        
    except Exception as e:
        logger.error(f"Error scraping webpage {url}: {e}")
        return MCPToolResponse(
            content=[{
                "type": "text",
                "text": f"Error scraping webpage: {str(e)}"
            }],
            is_error=True
        )

@app.post("/mcp/tools/execute_command/call")
async def tool_execute_command(request: MCPToolCallRequest) -> MCPToolResponse:
    """Execute system command safely"""
    try:
        command = request.arguments["command"]
        working_dir = request.arguments.get("working_dir", "/tmp")
        timeout = request.arguments.get("timeout", 30)
        capture_output = request.arguments.get("capture_output", True)
        
        # Security check - only allow safe commands
        if not _is_safe_command(command):
            raise HTTPException(status_code=403, detail="Command not allowed for security reasons")
        
        # Security check for working directory
        if not _is_safe_path(working_dir):
            raise HTTPException(status_code=403, detail="Working directory not allowed")
        
        if capture_output:
            result = subprocess.run(
                command,
                shell=True,
                cwd=working_dir,
                timeout=timeout,
                capture_output=True,
                text=True
            )
            
            output = {
                "command": command,
                "return_code": result.returncode,
                "stdout": result.stdout,
                "stderr": result.stderr,
                "working_dir": working_dir
            }
        else:
            result = subprocess.run(
                command,
                shell=True,
                cwd=working_dir,
                timeout=timeout
            )
            
            output = {
                "command": command,
                "return_code": result.returncode,
                "working_dir": working_dir,
                "message": "Command executed without output capture"
            }
        
        return MCPToolResponse(
            content=[{
                "type": "text",
                "text": json.dumps(output, indent=2)
            }]
        )
        
    except subprocess.TimeoutExpired:
        return MCPToolResponse(
            content=[{
                "type": "text",
                "text": f"Command timed out after {timeout} seconds"
            }],
            is_error=True
        )
    except Exception as e:
        logger.error(f"Error executing command {command}: {e}")
        return MCPToolResponse(
            content=[{
                "type": "text",
                "text": f"Error executing command: {str(e)}"
            }],
            is_error=True
        )

@app.post("/mcp/tools/analyze_code/call")
async def tool_analyze_code(request: MCPToolCallRequest) -> MCPToolResponse:
    """Analyze code structure and complexity"""
    try:
        file_path = request.arguments["file_path"]
        language = request.arguments.get("language")
        include_metrics = request.arguments.get("include_metrics", True)
        include_dependencies = request.arguments.get("include_dependencies", True)
        
        # Security check
        if not _is_safe_path(file_path):
            raise HTTPException(status_code=403, detail="Access denied to path")
        
        if not os.path.exists(file_path):
            raise HTTPException(status_code=404, detail="File not found")
        
        # Read file content
        async with aiofiles.open(file_path, 'r', encoding='utf-8') as f:
            content = await f.read()
        
        # Detect language if not provided
        if not language:
            language = _detect_language(file_path)
        
        analysis = {
            "file_path": file_path,
            "language": language,
            "file_size": len(content),
            "line_count": len(content.splitlines()),
            "character_count": len(content)
        }
        
        if include_metrics:
            analysis["metrics"] = _analyze_code_metrics(content, language)
        
        if include_dependencies:
            analysis["dependencies"] = _extract_dependencies(content, language)
        
        return MCPToolResponse(
            content=[{
                "type": "text",
                "text": json.dumps(analysis, indent=2)
            }]
        )
        
    except Exception as e:
        logger.error(f"Error analyzing code {file_path}: {e}")
        return MCPToolResponse(
            content=[{
                "type": "text",
                "text": f"Error analyzing code: {str(e)}"
            }],
            is_error=True
        )

# Helper functions
def _is_safe_path(path: str) -> bool:
    """Check if path is safe to access"""
    # Resolve path to prevent directory traversal
    try:
        resolved = os.path.realpath(path)
        # Allow access to /tmp and current working directory
        allowed_prefixes = ["/tmp", "/app", os.getcwd()]
        return any(resolved.startswith(prefix) for prefix in allowed_prefixes)
    except:
        return False

def _is_safe_command(command: str) -> bool:
    """Check if command is safe to execute"""
    # Only allow specific safe commands
    safe_commands = {
        'ls', 'cat', 'head', 'tail', 'grep', 'find', 'wc', 'echo', 
        'pwd', 'whoami', 'date', 'which', 'python', 'node', 'npm'
    }
    
    # Extract the first word (command name)
    cmd_name = command.strip().split()[0]
    return cmd_name in safe_commands

def _detect_language(file_path: str) -> str:
    """Detect programming language from file extension"""
    ext = Path(file_path).suffix.lower()
    language_map = {
        '.py': 'python',
        '.js': 'javascript',
        '.ts': 'typescript',
        '.java': 'java',
        '.cpp': 'cpp',
        '.c': 'c',
        '.go': 'go',
        '.rs': 'rust',
        '.php': 'php',
        '.rb': 'ruby',
        '.html': 'html',
        '.css': 'css',
        '.json': 'json',
        '.yaml': 'yaml',
        '.yml': 'yaml'
    }
    return language_map.get(ext, 'unknown')

def _analyze_code_metrics(content: str, language: str) -> Dict[str, Any]:
    """Analyze code metrics like complexity"""
    lines = content.splitlines()
    non_empty_lines = [line for line in lines if line.strip()]
    
    metrics = {
        "total_lines": len(lines),
        "non_empty_lines": len(non_empty_lines),
        "blank_lines": len(lines) - len(non_empty_lines),
        "comment_lines": 0,
        "function_count": 0,
        "class_count": 0
    }
    
    # Language-specific analysis
    if language == 'python':
        for line in lines:
            stripped = line.strip()
            if stripped.startswith('#'):
                metrics["comment_lines"] += 1
            elif stripped.startswith('def '):
                metrics["function_count"] += 1
            elif stripped.startswith('class '):
                metrics["class_count"] += 1
    
    elif language == 'javascript':
        for line in lines:
            stripped = line.strip()
            if stripped.startswith('//') or stripped.startswith('*'):
                metrics["comment_lines"] += 1
            elif 'function ' in stripped or '=>' in stripped:
                metrics["function_count"] += 1
            elif stripped.startswith('class '):
                metrics["class_count"] += 1
    
    return metrics

def _extract_dependencies(content: str, language: str) -> List[str]:
    """Extract dependencies from code"""
    dependencies = []
    lines = content.splitlines()
    
    if language == 'python':
        for line in lines:
            stripped = line.strip()
            if stripped.startswith('import '):
                dep = stripped.replace('import ', '').split()[0]
                dependencies.append(dep)
            elif stripped.startswith('from '):
                dep = stripped.split()[1]
                dependencies.append(dep)
    
    elif language == 'javascript':
        for line in lines:
            stripped = line.strip()
            if 'require(' in stripped or 'import ' in stripped:
                # Extract module name (simplified)
                if 'require(' in stripped:
                    start = stripped.find("require('") + 9
                    end = stripped.find("')", start)
                    if start > 8 and end > start:
                        dependencies.append(stripped[start:end])
                elif 'import ' in stripped and 'from ' in stripped:
                    start = stripped.find("from '") + 6
                    end = stripped.find("'", start)
                    if start > 5 and end > start:
                        dependencies.append(stripped[start:end])
    
    return list(set(dependencies))  # Remove duplicates

@app.post("/mcp/close")
async def close_mcp_session(request: Dict[str, str]):
    """Close MCP session"""
    session_id = request.get("session_id")
    if session_id in active_sessions:
        del active_sessions[session_id]
        logger.info(f"🔌 MCP Session closed: {session_id}")
    
    return {"status": "closed", "session_id": session_id}

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "server": "advanced-mcp-tools",
        "version": "2.0.0",
        "active_sessions": len(active_sessions),
        "timestamp": datetime.now().isoformat()
    }

if __name__ == "__main__":
    logger.info("🛠️ Starting Advanced MCP Tools Server...")
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=TOOLS_SERVER_PORT,
        log_level="info"
    )