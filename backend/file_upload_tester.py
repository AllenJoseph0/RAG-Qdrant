import streamlit as st
import asyncio
import json
import base64
from mcp.client.sse import sse_client
from mcp.client.session import ClientSession
import requests

API_BASE_URL = "http://localhost:8351"  # Backend.js (Node.js proxy)

# Page Configuration
st.set_page_config(
    page_title="File Upload MCP Tester",
    page_icon="📁",
    layout="wide"
)

st.title("📁 File Upload MCP Tool Tester")
st.markdown("""
This utility connects to the **File Upload MCP Server** via SSE to test file uploads to RAG knowledge bases.
""")

# --- SIDEBAR CONFIGURATION ---
with st.sidebar:
    st.header("🔌 Connection Settings")
    server_url = st.text_input("MCP Server URL", value="http://localhost:8352/mcp/file/sse")
    
    st.header("👤 Context Defaults")
    st.info("Default values for file uploads.")
    default_firm_id = st.number_input("Firm ID", value=5, min_value=1)
    default_user_id = st.number_input("User ID", value=1490, min_value=1)
    default_user_role = st.selectbox("User Role", ["admin", "business", "basic"], index=0, help="User role for permissions")
    
    st.divider()
    
    st.header("📂 RAG Categories")
    st.caption("Available knowledge bases for this firm")
    
    # Fetch available categories using /api/rag/structure
    if 'available_categories' not in st.session_state or st.button("🔄 Refresh Categories", key="refresh_cats"):
        available_categories = []
        try:
            resp = requests.get(f"{API_BASE_URL}/api/rag/structure", params={
                "username": str(default_firm_id)
            }, timeout=3)
            
            if resp.status_code == 200:
                structure_data = resp.json()
                # Response format: {firmId: [categories]}
                # Extract categories for this firm
                firm_key = str(default_firm_id)
                if isinstance(structure_data, dict) and firm_key in structure_data:
                    categories_list = structure_data[firm_key]
                    if isinstance(categories_list, list):
                        # Each item might be a dict with 'name' or just a string
                        available_categories = [
                            item.get('name') if isinstance(item, dict) else item 
                            for item in categories_list
                        ]
                    else:
                        available_categories = []
                
                st.session_state['available_categories'] = available_categories
                
                if available_categories:
                    st.success(f"✅ Found {len(available_categories)} categories")
                else:
                    st.info("No categories found. You can create a new one.")
            else:
                st.warning(f"Could not fetch categories (Status: {resp.status_code})")
                st.session_state['available_categories'] = []
        except Exception as e:
            st.warning(f"Could not fetch categories: {e}")
            st.session_state['available_categories'] = []
    
    available_categories = st.session_state.get('available_categories', [])
    
    # Display available categories
    if available_categories:
        with st.expander("📋 Available Categories", expanded=False):
            for cat in available_categories:
                st.caption(f"• {cat}")

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

async def call_upload_tool(url, arguments):
    """Calls the upload_file tool with the provided arguments."""
    try:
        async with sse_client(url) as streams:
            async with ClientSession(streams[0], streams[1]) as session:
                await session.initialize()
                
                result = await session.call_tool("upload_file", arguments)
                return result
    except Exception as e:
        return f"Error: {str(e)}"

# --- MAIN INTERFACE ---

# Context Summary
st.info(f"""
**Current Context:** 
🏢 Firm ID: `{default_firm_id}` | 👤 User ID: `{default_user_id}` | 🔑 Role: `{default_user_role}` | 📂 Categories: `{len(st.session_state.get('available_categories', []))}`
""")

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
        st.warning("Make sure the File Upload MCP server is running on the AI server.")
    else:
        st.success(f"Found {len(tools)} Tool(s)")
        
        # Display Tool Details
        if tools:
            tool_info = tools[0]  # Should be upload_file
            
            with st.expander("View Tool Schema", expanded=False):
                st.json(tool_info.inputSchema)
            
            st.markdown(f"**Tool Name:** `{tool_info.name}`")
            st.markdown(f"**Description:** {tool_info.description}")

            # 2. File Upload Section
            st.divider()
            st.subheader("2. Upload File to RAG")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("### 1. Choose a Category")
                
                # Build category options
                category_options = []
                if available_categories:
                    category_options = available_categories.copy()
                category_options.append("-- Create a new category --")
                
                # Category selector
                selected_option = st.selectbox(
                    "Select an existing category or create new",
                    category_options,
                    help="Choose from existing RAG knowledge bases or create a new one"
                )
                
                # If creating new, show text input
                if selected_option == "-- Create a new category --":
                    selected_category = st.text_input(
                        "New Category Name",
                        placeholder="e.g., company_docs, legal, hr_policies",
                        help="Enter a name for the new knowledge base category"
                    )
                else:
                    selected_category = selected_option
                
                # File upload
                uploaded_file = st.file_uploader("Choose a file", type=None)
                
                # Optional description
                file_description = st.text_area("File Description (optional)", 
                    placeholder="Brief description of the file content...",
                    height=100)
                
                # Update index option
                update_index = st.checkbox("Auto-update RAG index after upload", value=True)
            
            with col2:
                st.markdown("### Upload Preview")
                if uploaded_file:
                    st.info(f"**File:** {uploaded_file.name}")
                    st.info(f"**Size:** {uploaded_file.size:,} bytes")
                    st.info(f"**Type:** {uploaded_file.type or 'Unknown'}")
                    
                    if selected_category:
                        st.success(f"**Target:** Firm {default_firm_id} → {selected_category}")
                    
                    if file_description:
                        st.markdown("**Description:**")
                        st.caption(file_description)
                else:
                    st.warning("No file selected")
            
            st.markdown("---")
            
            # Upload button
            if st.button("🚀 Upload to RAG", type="primary", disabled=not uploaded_file or not selected_category):
                if not uploaded_file:
                    st.error("Please select a file to upload")
                elif not selected_category:
                    st.error("Please specify a category")
                else:
                    with st.spinner(f"Uploading {uploaded_file.name}..."):
                        try:
                            # Read and encode file
                            file_bytes = uploaded_file.read()
                            file_b64 = base64.b64encode(file_bytes).decode('utf-8')
                            
                            # Prepare arguments
                            arguments = {
                                "firm_id": str(default_firm_id),
                                "category": selected_category,
                                "file_name": uploaded_file.name,
                                "file_content": file_b64,
                                "update_index": update_index
                            }
                            
                            if file_description.strip():
                                arguments["description"] = file_description.strip()
                            
                            # Call MCP tool
                            result = asyncio.run(call_upload_tool(server_url, arguments))
                            
                            st.subheader("📊 Upload Result")
                            
                            if isinstance(result, str) and result.startswith("Error"):
                                st.error(result)
                            else:
                                # Parse MCP Result content
                                if hasattr(result, 'content') and isinstance(result.content, list):
                                    for content in result.content:
                                        if content.type == 'text':
                                            # Check for success/error indicators
                                            if "✅" in content.text or "successfully" in content.text.lower():
                                                st.success(content.text)
                                                
                                                # Show additional info box
                                                st.info(f"""
                                                **Upload Summary:**
                                                - 📁 File: `{uploaded_file.name}`
                                                - 📂 Category: `{selected_category}`
                                                - 🏢 Firm ID: `{default_firm_id}`
                                                - 📍 Path: `backend/data/uploads/{default_firm_id}/{selected_category}/{uploaded_file.name}`
                                                - 🔄 Index Updated: `{update_index}`
                                                """)
                                            elif "❌" in content.text or "failed" in content.text.lower():
                                                st.error(content.text)
                                            else:
                                                st.info(content.text)
                                        elif content.type == 'image':
                                            st.image(content.data)
                                        elif content.type == 'resource':
                                            st.code(content.text)
                                else:
                                    st.json(result)
                                    
                        except Exception as e:
                            st.error(f"Upload failed: {str(e)}")
                            with st.expander("Error Details"):
                                st.exception(e)

else:
    st.info("Click 'Connect & List Tools' to discover the File Upload tool.")

# Footer
st.markdown("---")
st.caption("File Upload MCP Tester | Upload files to RAG Knowledge Bases via MCP")

# Add some helpful tips
with st.expander("💡 Tips & Information"):
    st.markdown("""
    ### Supported File Types
    - **Documents:** PDF, DOCX, TXT, MD, etc.
    - **Images:** PNG, JPG, JPEG, etc.
    - **Audio:** MP3, WAV, M4A, etc.
    - **Other:** Any file type supported by your RAG system
    
    ### How It Works
    1. Select or create a RAG category (knowledge base)
    2. Choose a file from your computer
    3. Optionally add a description for better searchability
    4. Enable "Auto-update index" to make the file immediately searchable
    5. Click upload - the file is sent via MCP to the AI server
    
    ### What Happens
    - File is saved to: `backend/data/uploads/{firm_id}/{category}/{filename}`
    - Description is stored in: `.descriptions.json`
    - If "update index" is enabled, the RAG index is automatically rebuilt
    
    ### Testing Multiple Files
    You can upload multiple files by repeating the process. Each upload is independent.
    """)
