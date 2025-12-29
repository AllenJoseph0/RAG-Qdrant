
import os
import sys
import logging
import asyncio
import json
import uuid
import base64
from typing import Any, Dict, Optional
from flask import Blueprint, request, Response, stream_with_context, jsonify

# Ensure we can import from parent
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

try:
    from mcp.server import Server
    from mcp.types import Tool, TextContent, ImageContent, EmbeddedResource
    import mcp.types as types
except ImportError:
    pass

# Configure Logging
logger = logging.getLogger(__name__)

# --- MCP Server & Blueprint Definition ---

file_mcp_bp = Blueprint('file_mcp', __name__)

# Global Server Instance
file_mcp_server = Server("file-upload-agent")

@file_mcp_server.list_tools()
async def list_tools() -> list[Tool]:
    return [
        Tool(
            name="upload_file",
            description="Upload a file to the RAG knowledge base and optionally update the index.",
            inputSchema={
                "type": "object",
                "properties": {
                    "firm_id": {
                        "type": "string",
                        "description": "The ID of the firm/organization."
                    },
                    "category": {
                        "type": "string",
                        "description": "The RAG category/knowledge base to upload to."
                    },
                    "file_name": {
                        "type": "string",
                        "description": "Name of the file (with extension)."
                    },
                    "file_content": {
                        "type": "string",
                        "description": "Base64 encoded file content."
                    },
                    "description": {
                        "type": "string",
                        "description": "Optional description/metadata for the file."
                    },
                    "update_index": {
                        "type": "boolean",
                        "description": "Whether to automatically update the RAG index after upload (default: false)."
                    }
                },
                "required": ["firm_id", "category", "file_name", "file_content"]
            }
        )
    ]

@file_mcp_server.call_tool()
async def call_tool(name: str, arguments: Any) -> list[types.TextContent | types.ImageContent | types.EmbeddedResource]:
    if name != "upload_file":
        raise ValueError(f"Unknown tool: {name}")

    if not isinstance(arguments, dict):
        raise ValueError("Arguments must be a dictionary")

    firm_id = arguments.get("firm_id")
    category = arguments.get("category")
    file_name = arguments.get("file_name")
    file_content_b64 = arguments.get("file_content")
    description = arguments.get("description", "")
    update_index = arguments.get("update_index", False)
    
    if not all([firm_id, category, file_name, file_content_b64]):
        return [TextContent(type="text", text="Error: Missing required parameters (firm_id, category, file_name, file_content)")]

    try:
        # Decode base64 content
        file_content = base64.b64decode(file_content_b64)
        
        # Define upload path
        upload_dir = os.path.join(parent_dir, "data", "uploads", str(firm_id), category)
        os.makedirs(upload_dir, exist_ok=True)
        
        file_path = os.path.join(upload_dir, file_name)
        
        # Write file
        with open(file_path, 'wb') as f:
            f.write(file_content)
        
        logger.info(f"File uploaded: {file_path}")
        
        # Save description if provided (via backend API)
        if description:
            try:
                import requests
                backend_url = os.getenv("RAG_BACKEND_URL", "http://localhost:8351")
                
                # Save description via API
                desc_resp = requests.post(f"{backend_url}/api/rag/files/description", json={
                    "username": firm_id,
                    "category": category,
                    "filename": file_name,
                    "description": description
                })
                
                if desc_resp.status_code != 200:
                    logger.warning(f"Failed to save description: {desc_resp.text}")
            except Exception as e:
                logger.warning(f"Could not save description: {e}")
        
        result_text = f"✅ File '{file_name}' uploaded successfully to category '{category}' for firm {firm_id}."
        
        # Trigger index update if requested
        if update_index:
            try:
                import requests
                # Backend.js runs on port 8351, it will proxy to ai_server.py
                backend_url = os.getenv("RAG_BACKEND_URL", "http://localhost:8351")
                
                # Trigger update-index
                resp = requests.post(f"{backend_url}/api/rag/update-index", json={
                    "username": firm_id,
                    "category": category,
                    "firm_id": firm_id
                })
                
                if resp.status_code == 200:
                    result_text += "\n✅ Index update initiated."
                else:
                    result_text += f"\n⚠️ Index update failed: {resp.text}"
            except Exception as e:
                result_text += f"\n⚠️ Index update error: {str(e)}"
        
        return [TextContent(type="text", text=result_text)]
        
    except Exception as e:
        logger.error(f"File upload failed: {e}")
        return [TextContent(type="text", text=f"❌ Upload failed: {str(e)}")]


# --- SSE Transport Bridge for Flask ---

import queue as sync_queue
active_file_queues: Dict[str, sync_queue.Queue] = {}

@file_mcp_bp.route("/sse", methods=["GET"])
def handle_sse_connect():
    """
    Handle incoming SSE connection for file upload MCP.
    """
    session_id = str(uuid.uuid4())
    logger.info(f"New File MCP SSE connection: {session_id}")
    
    # Create a thread-safe queue for this session
    queue = sync_queue.Queue()
    active_file_queues[session_id] = queue
    
    def generate():
        # First event: endpoint for messages
        # The endpoint will be /mcp/file/messages due to blueprint prefix
        endpoint_url = f"/mcp/file/messages?session_id={session_id}"
        yield f"event: endpoint\ndata: {endpoint_url}\n\n"
        
        try:
            while True:
                try:
                    # Blocking get with timeout to allow heartbeats
                    message = queue.get(timeout=5)
                    yield f"event: message\ndata: {json.dumps(message)}\n\n"
                except sync_queue.Empty:
                    # Send keepalive
                    yield ": keepalive\n\n"
        except GeneratorExit:
            logger.info(f"File MCP SSE connection closed: {session_id}")
            active_file_queues.pop(session_id, None)

    return Response(stream_with_context(generate()), content_type="text/event-stream")

@file_mcp_bp.route("/messages", methods=["POST"])
async def handle_messages():
    """
    Handle client messages (JSON-RPC).
    """
    session_id = request.args.get("session_id")
    if not session_id or session_id not in active_file_queues:
        return jsonify({"error": "Session not found"}), 404
        
    message = request.json
    
    try:
        if not message:
            return jsonify("No content"), 400
            
        method = message.get("method")
        msg_id = message.get("id")
        
        response = None
        
        if method == "initialize":
            response = {
                "jsonrpc": "2.0",
                "id": msg_id,
                "result": {
                    "protocolVersion": "2024-11-05",
                    "capabilities": {
                        "tools": {}
                    },
                    "serverInfo": {
                        "name": "file-upload-agent",
                        "version": "1.0.0"
                    }
                }
            }
            
        elif method == "notifications/initialized":
            # No response needed
            pass
            
        elif method == "tools/list":
            tools = await list_tools()
            # Convert tools to dicts
            tools_data = [
                {
                    "name": t.name,
                    "description": t.description,
                    "inputSchema": t.inputSchema
                }
                for t in tools
            ]
            response = {
                "jsonrpc": "2.0",
                "id": msg_id,
                "result": {
                    "tools": tools_data
                }
            }
            
        elif method == "tools/call":
            params = message.get("params", {})
            name = params.get("name")
            args = params.get("arguments")
            
            result_content = await call_tool(name, args)
            
            # Serialize content
            content_list = []
            for item in result_content:
                if item.type == "text":
                    content_list.append({"type": "text", "text": item.text})
            
            response = {
                "jsonrpc": "2.0",
                "id": msg_id,
                "result": {
                    "content": content_list,
                    "isError": False
                }
            }
            
        else:
            # Unknown method
            pass
            
        if response:
            # Push to SSE Queue
            sq = active_file_queues[session_id]
            sq.put(response)
            
        return jsonify({"status": "ok"})
        
    except Exception as e:
        logger.error(f"Error handling message: {e}")
        return jsonify({"error": str(e)}), 500
