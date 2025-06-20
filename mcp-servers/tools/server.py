"""
Tools MCP Server - Utilitaires et outils système via MCP
Fichiers, web scraping, code analysis, etc.
"""

import asyncio
import json
import logging
import os
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any, List, Dict
import tempfile
import requests
import aiofiles

from mcp.server.models import InitializationOptions
from mcp.server import NotificationOptions, Server
from mcp.types import Resource, Tool, TextContent
import mcp.types as types

server = Server("tools-mcp")

# Configuration
WORKSPACE_PATH = "/app/workspace"
DATA_PATH = "/app/data"

# Ensure directories exist
Path(WORKSPACE_PATH).mkdir(parents=True, exist_ok=True)
Path(DATA_PATH).mkdir(parents=True, exist_ok=True)

# Tools Manager
class ToolsManager:
    def __init__(self):
        self.workspace = Path(WORKSPACE_PATH)
        self.data_path = Path(DATA_PATH)
    
    async def read_file(self, file_path: str) -> str:
        """Read file content safely"""
        try:
            full_path = self.workspace / file_path
            
            # Security check - ensure path is within workspace
            if not str(full_path.resolve()).startswith(str(self.workspace.resolve())):
                return "❌ Accès refusé : chemin en dehors du workspace"
            
            if not full_path.exists():
                return f"❌ Fichier non trouvé : {file_path}"
            
            async with aiofiles.open(full_path, 'r', encoding='utf-8') as f:
                content = await f.read()
                return content
                
        except Exception as e:
            return f"❌ Erreur lecture fichier : {str(e)}"
    
    async def write_file(self, file_path: str, content: str) -> str:
        """Write file content safely"""
        try:
            full_path = self.workspace / file_path
            
            # Security check
            if not str(full_path.resolve()).startswith(str(self.workspace.resolve())):
                return "❌ Accès refusé : chemin en dehors du workspace"
            
            # Create parent directories if needed
            full_path.parent.mkdir(parents=True, exist_ok=True)
            
            async with aiofiles.open(full_path, 'w', encoding='utf-8') as f:
                await f.write(content)
            
            return f"✅ Fichier écrit : {file_path} ({len(content)} caractères)"
            
        except Exception as e:
            return f"❌ Erreur écriture fichier : {str(e)}"
    
    async def list_files(self, directory: str = ".", pattern: str = "*") -> List[Dict[str, Any]]:
        """List files in directory"""
        try:
            full_path = self.workspace / directory
            
            # Security check
            if not str(full_path.resolve()).startswith(str(self.workspace.resolve())):
                return [{"error": "Accès refusé : chemin en dehors du workspace"}]
            
            if not full_path.exists():
                return [{"error": f"Répertoire non trouvé : {directory}"}]
            
            files = []
            for item in full_path.glob(pattern):
                stat = item.stat()
                files.append({
                    "name": item.name,
                    "path": str(item.relative_to(self.workspace)),
                    "type": "directory" if item.is_dir() else "file",
                    "size": stat.st_size if item.is_file() else 0,
                    "modified": datetime.fromtimestamp(stat.st_mtime).isoformat()
                })
            
            return sorted(files, key=lambda x: (x["type"] == "file", x["name"]))
            
        except Exception as e:
            return [{"error": f"Erreur listage : {str(e)}"}]
    
    async def fetch_url(self, url: str, method: str = "GET", headers: Dict = None) -> Dict[str, Any]:
        """Fetch URL content"""
        try:
            headers = headers or {"User-Agent": "Jarvis-MCP-Tools/1.0"}
            
            response = requests.request(
                method=method.upper(),
                url=url,
                headers=headers,
                timeout=30,
                allow_redirects=True
            )
            
            return {
                "status_code": response.status_code,
                "headers": dict(response.headers),
                "content": response.text[:10000],  # Limit to 10k chars
                "content_type": response.headers.get("content-type", ""),
                "url": response.url,
                "encoding": response.encoding
            }
            
        except Exception as e:
            return {"error": f"Erreur fetch URL : {str(e)}"}
    
    async def run_command(self, command: str, working_dir: str = None) -> Dict[str, Any]:
        """Run system command safely"""
        try:
            # Security: whitelist of allowed commands
            allowed_commands = [
                "ls", "cat", "grep", "find", "wc", "head", "tail",
                "git", "node", "npm", "python", "pip", "docker",
                "curl", "wget", "ping"
            ]
            
            command_parts = command.split()
            if not command_parts or command_parts[0] not in allowed_commands:
                return {"error": f"Commande non autorisée : {command_parts[0] if command_parts else 'vide'}"}
            
            work_dir = self.workspace / (working_dir or ".")
            if not str(work_dir.resolve()).startswith(str(self.workspace.resolve())):
                work_dir = self.workspace
            
            # Execute command
            result = subprocess.run(
                command,
                shell=True,
                cwd=work_dir,
                capture_output=True,
                text=True,
                timeout=30
            )
            
            return {
                "command": command,
                "exit_code": result.returncode,
                "stdout": result.stdout[:5000],  # Limit output
                "stderr": result.stderr[:5000],
                "working_dir": str(work_dir),
                "success": result.returncode == 0
            }
            
        except subprocess.TimeoutExpired:
            return {"error": "Commande timeout (30s)"}
        except Exception as e:
            return {"error": f"Erreur exécution : {str(e)}"}
    
    async def analyze_code(self, file_path: str, language: str = "auto") -> Dict[str, Any]:
        """Analyze code file"""
        try:
            content = await self.read_file(file_path)
            if content.startswith("❌"):
                return {"error": content}
            
            # Detect language if auto
            if language == "auto":
                ext = Path(file_path).suffix.lower()
                language_map = {
                    ".py": "python", ".js": "javascript", ".ts": "typescript",
                    ".jsx": "javascript", ".tsx": "typescript", ".html": "html",
                    ".css": "css", ".json": "json", ".md": "markdown",
                    ".yml": "yaml", ".yaml": "yaml", ".toml": "toml"
                }
                language = language_map.get(ext, "text")
            
            # Basic analysis
            lines = content.splitlines()
            
            analysis = {
                "file_path": file_path,
                "language": language,
                "lines_count": len(lines),
                "chars_count": len(content),
                "empty_lines": sum(1 for line in lines if not line.strip()),
                "functions": 0,
                "classes": 0,
                "imports": 0,
                "comments": 0
            }
            
            # Language-specific analysis
            if language == "python":
                analysis["functions"] = len([l for l in lines if l.strip().startswith("def ")])
                analysis["classes"] = len([l for l in lines if l.strip().startswith("class ")])
                analysis["imports"] = len([l for l in lines if l.strip().startswith(("import ", "from "))])
                analysis["comments"] = len([l for l in lines if l.strip().startswith("#")])
            
            elif language in ["javascript", "typescript"]:
                analysis["functions"] = len([l for l in lines if "function" in l or "=>" in l])
                analysis["classes"] = len([l for l in lines if l.strip().startswith("class ")])
                analysis["imports"] = len([l for l in lines if l.strip().startswith(("import ", "export "))])
                analysis["comments"] = len([l for l in lines if l.strip().startswith("//")])
            
            return analysis
            
        except Exception as e:
            return {"error": f"Erreur analyse code : {str(e)}"}
    
    async def create_backup(self, file_pattern: str = "*") -> str:
        """Create backup of workspace files"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_name = f"backup_{timestamp}.tar.gz"
            backup_path = self.data_path / backup_name
            
            # Create tar archive
            result = subprocess.run([
                "tar", "-czf", str(backup_path),
                "-C", str(self.workspace),
                "."
            ], capture_output=True, text=True, timeout=60)
            
            if result.returncode == 0:
                size = backup_path.stat().st_size
                return f"✅ Backup créé : {backup_name} ({size} bytes)"
            else:
                return f"❌ Erreur backup : {result.stderr}"
                
        except Exception as e:
            return f"❌ Erreur backup : {str(e)}"

tools_manager = ToolsManager()

# MCP Tools
@server.list_tools()
async def handle_list_tools() -> list[Tool]:
    """List available Tools"""
    return [
        Tool(
            name="file_read",
            description="Lire le contenu d'un fichier",
            inputSchema={
                "type": "object",
                "properties": {
                    "file_path": {
                        "type": "string",
                        "description": "Chemin relatif du fichier dans le workspace"
                    }
                },
                "required": ["file_path"]
            }
        ),
        Tool(
            name="file_write",
            description="Écrire du contenu dans un fichier",
            inputSchema={
                "type": "object",
                "properties": {
                    "file_path": {
                        "type": "string",
                        "description": "Chemin relatif du fichier dans le workspace"
                    },
                    "content": {
                        "type": "string",
                        "description": "Contenu à écrire"
                    }
                },
                "required": ["file_path", "content"]
            }
        ),
        Tool(
            name="file_list",
            description="Lister les fichiers d'un répertoire",
            inputSchema={
                "type": "object",
                "properties": {
                    "directory": {
                        "type": "string",
                        "default": ".",
                        "description": "Répertoire à lister"
                    },
                    "pattern": {
                        "type": "string",
                        "default": "*",
                        "description": "Pattern de fichiers (glob)"
                    }
                }
            }
        ),
        Tool(
            name="web_fetch",
            description="Récupérer le contenu d'une URL",
            inputSchema={
                "type": "object",
                "properties": {
                    "url": {
                        "type": "string",
                        "description": "URL à récupérer"
                    },
                    "method": {
                        "type": "string",
                        "enum": ["GET", "POST", "HEAD"],
                        "default": "GET",
                        "description": "Méthode HTTP"
                    },
                    "headers": {
                        "type": "object",
                        "description": "Headers HTTP additionnels"
                    }
                },
                "required": ["url"]
            }
        ),
        Tool(
            name="command_run",
            description="Exécuter une commande système",
            inputSchema={
                "type": "object",
                "properties": {
                    "command": {
                        "type": "string",
                        "description": "Commande à exécuter"
                    },
                    "working_dir": {
                        "type": "string",
                        "description": "Répertoire de travail relatif"
                    }
                },
                "required": ["command"]
            }
        ),
        Tool(
            name="code_analyze",
            description="Analyser un fichier de code",
            inputSchema={
                "type": "object",
                "properties": {
                    "file_path": {
                        "type": "string",
                        "description": "Chemin du fichier à analyser"
                    },
                    "language": {
                        "type": "string",
                        "enum": ["auto", "python", "javascript", "typescript", "html", "css"],
                        "default": "auto",
                        "description": "Langage du code"
                    }
                },
                "required": ["file_path"]
            }
        ),
        Tool(
            name="backup_create",
            description="Créer un backup du workspace",
            inputSchema={
                "type": "object",
                "properties": {
                    "pattern": {
                        "type": "string",
                        "default": "*",
                        "description": "Pattern de fichiers à inclure"
                    }
                }
            }
        )
    ]

@server.call_tool()
async def handle_call_tool(name: str, arguments: dict) -> list[types.TextContent]:
    """Handle tool calls"""
    
    if name == "file_read":
        file_path = arguments.get("file_path", "")
        result = await tools_manager.read_file(file_path)
        
        return [types.TextContent(
            type="text",
            text=json.dumps({
                "tool": "file_read",
                "file_path": file_path,
                "content": result,
                "success": not result.startswith("❌")
            })
        )]
    
    elif name == "file_write":
        file_path = arguments.get("file_path", "")
        content = arguments.get("content", "")
        result = await tools_manager.write_file(file_path, content)
        
        return [types.TextContent(
            type="text",
            text=json.dumps({
                "tool": "file_write",
                "file_path": file_path,
                "result": result,
                "success": result.startswith("✅")
            })
        )]
    
    elif name == "file_list":
        directory = arguments.get("directory", ".")
        pattern = arguments.get("pattern", "*")
        files = await tools_manager.list_files(directory, pattern)
        
        return [types.TextContent(
            type="text",
            text=json.dumps({
                "tool": "file_list",
                "directory": directory,
                "pattern": pattern,
                "files": files,
                "count": len(files)
            })
        )]
    
    elif name == "web_fetch":
        url = arguments.get("url", "")
        method = arguments.get("method", "GET")
        headers = arguments.get("headers", {})
        result = await tools_manager.fetch_url(url, method, headers)
        
        return [types.TextContent(
            type="text",
            text=json.dumps({
                "tool": "web_fetch",
                "url": url,
                "method": method,
                "result": result
            })
        )]
    
    elif name == "command_run":
        command = arguments.get("command", "")
        working_dir = arguments.get("working_dir")
        result = await tools_manager.run_command(command, working_dir)
        
        return [types.TextContent(
            type="text",
            text=json.dumps({
                "tool": "command_run",
                "command": command,
                "result": result
            })
        )]
    
    elif name == "code_analyze":
        file_path = arguments.get("file_path", "")
        language = arguments.get("language", "auto")
        result = await tools_manager.analyze_code(file_path, language)
        
        return [types.TextContent(
            type="text",
            text=json.dumps({
                "tool": "code_analyze",
                "file_path": file_path,
                "language": language,
                "analysis": result
            })
        )]
    
    elif name == "backup_create":
        pattern = arguments.get("pattern", "*")
        result = await tools_manager.create_backup(pattern)
        
        return [types.TextContent(
            type="text",
            text=json.dumps({
                "tool": "backup_create",
                "pattern": pattern,
                "result": result,
                "success": result.startswith("✅")
            })
        )]
    
    else:
        raise ValueError(f"Outil inconnu: {name}")

# Resources
@server.list_resources()
async def handle_list_resources() -> list[Resource]:
    """List Tools resources"""
    return [
        Resource(
            uri="tools://workspace",
            name="Workspace Files",
            description="Liste des fichiers du workspace",
            mimeType="application/json"
        ),
        Resource(
            uri="tools://system",
            name="System Info",
            description="Informations système",
            mimeType="application/json"
        )
    ]

@server.read_resource()
async def handle_read_resource(uri: str) -> str:
    """Read Tools resources"""
    if uri == "tools://workspace":
        files = await tools_manager.list_files(".", "**/*")
        return json.dumps({
            "workspace_path": WORKSPACE_PATH,
            "total_files": len(files),
            "files": files[:50]  # Limit to 50 files
        })
    elif uri == "tools://system":
        return json.dumps({
            "workspace_path": WORKSPACE_PATH,
            "data_path": DATA_PATH,
            "allowed_commands": ["ls", "cat", "grep", "find", "wc", "head", "tail", "git", "node", "npm", "python", "pip", "docker"],
            "python_version": subprocess.run(["python", "--version"], capture_output=True, text=True).stdout.strip()
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
                server_name="tools-mcp",
                server_version="1.0.0",
                capabilities=server.get_capabilities(
                    notification_options=NotificationOptions(),
                    experimental_capabilities={},
                ),
            ),
        )

if __name__ == "__main__":
    asyncio.run(main())