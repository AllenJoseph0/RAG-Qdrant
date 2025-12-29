import streamlit as st
import asyncio
import json
from mcp.client.sse import sse_client
from mcp.client.session import ClientSession
import requests

API_BASE_URL = "http://localhost:8352" # Backend API Base URL

# Page Configuration
st.set_page_config(
    page_title="SQL Agents MCP Tester",
    page_icon="🤖",
    layout="wide"
)

st.title("🛠️ SQL Agents MCP Tool Tester")
st.markdown("""
This utility connects to your running **MCP Server** via SSE (Server-Sent Events) 
to list available agents and test tool execution.
""")

# --- SIDEBAR CONFIGURATION ---
with st.sidebar:
    st.header("🔌 Connection Settings")
    server_url = st.text_input("MCP Server URL", value="http://localhost:8352/sse")
    
    st.header("👤 Context Defaults")
    st.info("Default values for common parameters.")
    default_user_id = st.text_input("Default User ID", value="1490")
    default_firm_id = st.number_input("Default Firm ID", value=5, min_value=1)

    st.divider()
    
    st.header("🗄️ Database Configuration")
    
    # Check if config exists for this firm
    fetched_config = {}
    if default_firm_id:
        try:
            # We use a reduced timeout so it doesn't hang UI if backend is down
            check_resp = requests.get(f"{API_BASE_URL}/api/sql_agent/config", params={"firm_id": default_firm_id}, timeout=2)
            if check_resp.status_code == 200:
                data = check_resp.json()
                if data.get("success"):
                   fetched_config = data.get("config", {})
                   st.caption("✅ Loaded existing config for this Firm ID")
        except:
             pass

    with st.expander("Configure Connection", expanded=False):
        db_host = st.text_input("Host", value=fetched_config.get("host", "localhost"))
        db_user = st.text_input("User", value=fetched_config.get("user", "root"))
        db_pass = st.text_input("Password", value=fetched_config.get("password", ""), type="password")
        db_name = st.text_input("Database", value=fetched_config.get("database", ""))
        db_port = st.number_input("Port", value=int(fetched_config.get("port", 3306)))
        
        if st.button("💾 Save & Connect DB"):
            try:
                payload = {
                    "firm_id": str(default_firm_id),
                    "user_id": str(default_user_id),
                    "db_config": {
                        "host": db_host,
                        "user": db_user,
                        "password": db_pass,
                        "database": db_name,
                        "port": db_port
                    }
                }
                resp = requests.post(f"{API_BASE_URL}/api/sql_agent/connect", json=payload)
                if resp.status_code == 200:
                    st.success("Database connected and saved!")
                else:
                    st.error(f"Failed: {resp.text}")
            except Exception as e:
                st.error(f"Request failed: {e}")
        
    if st.button("🔄 Sync Schema for RAG"):
        with st.spinner("Indexing schema tables..."):
            try:
                payload = {"firm_id": str(default_firm_id)}
                resp = requests.post(f"{API_BASE_URL}/api/sql_agent_rag/sync", json=payload)
                if resp.status_code == 200:
                    st.success(f"Sync Complete: {resp.json().get('message')}")
                else:
                    st.error(f"Sync Failed: {resp.text}")
            except Exception as e:
                st.error(f"Sync Request failed: {e}")

# --- ASYNC HELPERS ---
async def list_available_tools(url):
    """Connects to the server and fetches the list of available tools."""
    try:
        async with sse_client(url) as streams:
            async with ClientSession(streams[0], streams[1]) as session:
                await session.initialize()
                tools_result = await session.list_tools()
                return tools_result.tools
    except Exception as e:
        return f"Error: {str(e)}"

async def call_agent_tool(url, tool_name, arguments):
    """Calls a specific tool with the provided arguments."""
    try:
        async with sse_client(url) as streams:
            async with ClientSession(streams[0], streams[1]) as session:
                await session.initialize()
                
                result = await session.call_tool(tool_name, arguments)
                return result
    except Exception as e:
        return f"Error: {str(e)}"

# --- MAIN INTERFACE ---

# 1. Tool Discovery Section
st.subheader("1. Tool Discovery")
col1, col2 = st.columns([1, 3])
with col1:
    if st.button("🔄 Connect & List Tools", type="primary"):
        with st.spinner("Connecting to MCP Server..."):
            tools = asyncio.run(list_available_tools(server_url))
            st.session_state['tools'] = tools

if 'tools' in st.session_state:
    tools = st.session_state['tools']
    if isinstance(tools, str) and tools.startswith("Error"):
        st.error(f"Failed to connect: {tools}")
        st.warning("Make sure your MCP server (e.g., 'mcp_server.py') is running and accessible.")
    else:
        st.success(f"Found {len(tools)} Tool(s)")
        
        # Create a cleaner list for display
        tool_options = {t.name: t for t in tools}
        selected_tool_name = st.selectbox("Select Tool to Test", list(tool_options.keys()))
        
        # Display Tool Details
        if selected_tool_name:
            tool_info = tool_options[selected_tool_name]
            
            with st.expander("View Tool Schema", expanded=False):
                st.json(tool_info.inputSchema)
            
            st.markdown(f"**Description:** {tool_info.description}")

            # 2. Testing Section (Dynamic Form)
            st.divider()
            st.subheader(f"2. Test '{selected_tool_name}' - Inputs")
            
            # Dynamic Argument Generation
            arguments = {}
            schema = tool_info.inputSchema
            properties = schema.get('properties', {})
            required = schema.get('required', [])

            if not properties:
                st.info("This tool requires no arguments.")
            else:
                for arg_name, arg_schema in properties.items():
                    arg_type = arg_schema.get('type', 'string')
                    arg_desc = arg_schema.get('description', '')
                    label = f"{arg_name} ({arg_type})"
                    if arg_name in required:
                        label += " *"
                    
                    # Determine default value
                    val = None
                    if arg_name == 'user_id':
                        val = default_user_id
                    elif arg_name == 'firm_id':
                        val = str(default_firm_id)
                    
                    help_text = arg_desc
                    
                    if arg_type == 'integer' or arg_type == 'number':
                        # Use text input and cast later to allow flexible defaults, or use number_input
                        # Using text input for flexibility with empty states
                        user_input = st.text_input(label, value=val if val else "", help=help_text)
                        if user_input:
                            try:
                                arguments[arg_name] = int(user_input) if arg_type == 'integer' else float(user_input)
                            except:
                                st.warning(f"Invalid number for {arg_name}")
                    else:
                        # String / specific text
                        if 'query' in arg_name or 'question' in arg_name or 'prompt' in arg_name:
                            user_input = st.text_area(label, value="Show me the top users", help=help_text, height=100)
                        else:
                            user_input = st.text_input(label, value=val if val else "", help=help_text)
                        
                        arguments[arg_name] = user_input

            st.markdown("---")
            if st.button("🚀 Run Tool", type="primary"):
                # Basic validation for required
                missing = [req for req in required if req not in arguments or not str(arguments[req]).strip()]
                if missing:
                    st.error(f"Missing required fields: {', '.join(missing)}")
                else:
                    with st.spinner(f"Running query on {selected_tool_name}..."):
                        # Execute the tool call
                        result = asyncio.run(call_agent_tool(
                            server_url, 
                            selected_tool_name, 
                            arguments
                        ))
                        
                        st.subheader("Results")
                        
                        if isinstance(result, str) and result.startswith("Error"):
                            st.error(result)
                        else:
                            # Parse MCP Result content
                            if hasattr(result, 'content') and isinstance(result.content, list):
                                for content in result.content:
                                    if content.type == 'text':
                                        st.markdown(content.text)
                                    elif content.type == 'image':
                                        st.image(content.data)
                                    elif content.type == 'resource':
                                         st.code(content.text)
                            else:
                                st.json(result)

else:
    st.info("Click 'Connect & List Tools' to discover your Agents.")

# Footer
st.markdown("---")
st.caption("V-Agents MCP Debugger | Running on Streamlit")