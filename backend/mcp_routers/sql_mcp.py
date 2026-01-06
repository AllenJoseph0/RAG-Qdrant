
import os
import sys
import logging
import asyncio
import json
import uuid
from typing import Any, Dict, Optional
from flask import Blueprint, request, Response, stream_with_context, jsonify

# Ensure we can import sql_service
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

try:
    from mcp.server import Server
    from mcp.types import Tool, TextContent, ImageContent, EmbeddedResource
    import mcp.types as types
except ImportError:
    # This might fail if mcp is not installed, but dependencies listed it
    pass

# Import service functions
try:
    from mcp_routers.sql_service import (
        generate_sql_query, 
        execute_generated_sql, 
        get_all_tables,
        get_db_connection_config
    )
except ImportError:
    try:
        from sql_service import (
            generate_sql_query, 
            execute_generated_sql,
            get_all_tables,
            get_db_connection_config
        )
    except ImportError as e:
        # Define mock/error functions if imports fail (e.g. strict environment)
        def generate_sql_query(*args, **kwargs): return {"success": False, "error": str(e)}
        def execute_generated_sql(*args, **kwargs): return {"success": False, "error": str(e)}
        def get_all_tables(*args, **kwargs): return []
        def get_db_connection_config(*args, **kwargs): return None


# Configure Logging
logger = logging.getLogger(__name__)

# --- MCP Server & Blueprint Definition ---

mcp_bp = Blueprint('mcp_server', __name__)

# Global Server Instance
# We use the standard low-level Server, not FastMCP, to have control over transport
mcp_server = Server("sql-agent")

@mcp_server.list_tools()
async def list_tools() -> list[Tool]:
    return [
        Tool(
            name="query_db",
            description="Query the database using natural language.",
            inputSchema={
                "type": "object",
                "properties": {
                    "question": {
                        "type": "string", 
                        "description": "The user's natural language question (e.g. 'Who is the top sales person?')."
                    },
                    "firm_id": {
                        "type": "string", 
                        "description": "The ID of the firm to query against."
                    },
                    "user_id": {
                        "type": "string", 
                        "description": "The ID of the user executing the query."
                    },
                    "db_name": {
                        "type": "string",
                        "description": "Optional: Specific database name to query (if firm has multiple databases)."
                    }
                },
                "required": ["question", "firm_id"]
            }
        ),
        Tool(
            name="list_tables",
            description="Get a list of all available table names in the database for the specified firm.",
            inputSchema={
                "type": "object",
                "properties": {
                    "firm_id": {
                        "type": "string",
                        "description": "The ID of the firm whose database tables to list."
                    },
                    "db_name": {
                        "type": "string",
                        "description": "Optional: Specific database name (if firm has multiple databases)."
                    }
                },
                "required": ["firm_id"]
            }
        )
    ]

@mcp_server.call_tool()
async def call_tool(name: str, arguments: Any) -> list[types.TextContent | types.ImageContent | types.EmbeddedResource]:
    if name == "query_db":
        if not isinstance(arguments, dict):
            raise ValueError("Arguments must be a dictionary")

        question = arguments.get("question")
        firm_id = arguments.get("firm_id")
        user_id = arguments.get("user_id")
        db_name = arguments.get("db_name")  # Optional
        
        if not question or not firm_id:
            return [TextContent(type="text", text="Error: Missing 'question' or 'firm_id'")]

        # Normalize user_id
        try:
            if user_id:
                user_id = int(str(user_id))
        except ValueError:
            pass

        logger.info(f"Executing query_db: {question} (Firm: {firm_id}, DB: {db_name or 'default'})")
        
        # 1. Generate SQL
        gen_result = generate_sql_query(question, firm_id, user_id=user_id, db_name=db_name)
        if not gen_result['success']:
            return [TextContent(type="text", text=f"Error Generating SQL: {gen_result.get('error')}")]
        
        sql = gen_result['sql']
        print("--------------------------------------------------")
        print(f"Generated SQL: {sql}")
        print("--------------------------------------------------")
        
        # Return ONLY the SQL as requested
        output_text = f"Generated SQL:\n{sql}"
        return [TextContent(type="text", text=output_text)]
    
    elif name == "list_tables":
        if not isinstance(arguments, dict):
            raise ValueError("Arguments must be a dictionary")
        
        firm_id = arguments.get("firm_id")
        db_name = arguments.get("db_name")  # Optional
        
        if not firm_id:
            return [TextContent(type="text", text="Error: Missing 'firm_id'")]
        
        logger.info(f"Executing list_tables for Firm: {firm_id}, DB: {db_name or 'default'}")
        
        # Get database configuration
        db_config = get_db_connection_config(firm_id, db_name=db_name)
        if not db_config:
            return [TextContent(type="text", text=f"Error: No database configured for firm {firm_id}")]
        
        # Get all tables
        try:
            tables_info = get_all_tables(db_config)
            table_names = [t['table_name'] for t in tables_info if t.get('table_name')]
            
            if not table_names:
                return [TextContent(type="text", text="No tables found in the database.")]
            
            # Format output
            output_text = f"Available tables ({len(table_names)}):\n" + "\n".join(f"- {name}" for name in table_names)
            return [TextContent(type="text", text=output_text)]
        except Exception as e:
            logger.error(f"Error fetching tables: {e}")
            return [TextContent(type="text", text=f"Error fetching tables: {str(e)}")]
    
    else:
        raise ValueError(f"Unknown tool: {name}")

# --- SSE Transport Bridge for Flask ---

# Store active streams: session_id -> queue
from asyncio import Queue
active_queues: Dict[str, Queue] = {}
# Store active initialization objects to run the server loop
active_transports: Dict[str, Any] = {}

import threading

@mcp_bp.route("/sse", methods=["GET"])
def handle_sse_connect():
    """
    Handle incoming SSE connection.
    This implementation mimics the MCP SSE transport.
    """
    session_id = str(uuid.uuid4())
    logger.info(f"New SSE connection: {session_id}")
    
    # Create a thread-safe queue for this session
    # Since Flask is sync, we might need a way to bridge async/sync.
    # We'll use a simple blocking queue or similar if running sync,
    # but here we are trying to stream.
    
    # IMPORTANT: Real production implementation requires proper Async support (Quart or Hypercorn).
    # With standard Flask, this blocks a worker.
    
    queue = asyncio.Queue()
    active_queues[session_id] = queue
    
    def generate():
        # First event: endpoint for messages
        # The client expects an 'endpoint' event telling it where to POST messages
        # Include the blueprint prefix in the URL
        endpoint_url = f"/mcp/sql/messages?session_id={session_id}"
        yield f"event: endpoint\ndata: {endpoint_url}\n\n"
        
        # Keep connection open and yield messages
        # We need an event loop to read from the asyncio Queue if we are populating it from async code
        # BUT Flask run is likely not in an async loop.
        
        # Workaround: Use a thread-safe Queue (queue.Queue) if Flask is threaded
        # Re-implement using standard queue for compatibility
        import queue as sync_queue
        sq = sync_queue.Queue()
        
        # We replace the async queue with a sync queue for this session in our global map
        # This is a bit hacky but necessary for Flask standard WSGI
        active_queues[session_id] = sq
        
        try:
            while True:
                try:
                    # Blocking get with timeout to allow heartbeats
                    message = sq.get(timeout=5)
                    yield f"event: message\ndata: {json.dumps(message)}\n\n"
                except sync_queue.Empty:
                    # Send keepalive or just pass
                    yield ": keepalive\n\n"
        except GeneratorExit:
            logger.info(f"SSE connection closed: {session_id}")
            active_queues.pop(session_id, None)

    return Response(stream_with_context(generate()), content_type="text/event-stream")

@mcp_bp.route("/messages", methods=["POST"])
async def handle_messages():
    """
    Handle client messages (JSON-RPC).
    """
    session_id = request.args.get("session_id")
    if not session_id or session_id not in active_queues:
        return jsonify({"error": "Session not found"}), 404
        
    message = request.json
    
    # We need to process this message using the mcp_server instance.
    # Since mcp_server.process_request is async, we need to await it.
    # Flask 2.0+ supports async views.
    
    # Issue: The output of process_request needs to go into the SSE stream (the Queue).
    # But mcp_server doesn't know about our Queue.
    
    # We need to create a custom Transport that writes to our Queue.
    
    from mcp.server.sse import SseServerTransport
    # Actually, we can just use the memory transport pattern or manual processing.
    
    # Manual Processing of JSON-RPC to keep it simple and robust within Flask:
    # 1. Parse JSONAL-RPC
    # 2. Invoke mcp_server.process_request ??
    # mcp_server.create_initialization_options() ...
    
    # To truly use the mcp library's logic, we should use 'run_with_transport'.
    # But that blocks.
    
    # SIMPLIFIED APPROACH:
    # We will just implement the Tools Logic since that's what we essentially need.
    # The MCP protocol is:
    # -> { jsonrpc, method, params, id }
    # <- { jsonrpc, result, id }
    
    # We can handle 'initialize', 'notifications/initialized', 'tools/list', 'tools/call' manually
    # to guarantee it works in this mixed sync/async environment.
    
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
                        "name": "sql-agent",
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
                # Add image support if needed
            
            response = {
                "jsonrpc": "2.0",
                "id": msg_id,
                "result": {
                    "content": content_list,
                    "isError": False
                }
            }
            
        else:
            # Ping or unknown
            pass
            
        if response:
            # Push to SSE Queue
            sq = active_queues[session_id]
            sq.put(response)
            
        return jsonify({"status": "ok"})
        
    except Exception as e:
        logger.error(f"Error handling message: {e}")
        return jsonify({"error": str(e)}), 500

