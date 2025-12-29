import pymysql
import os
import json
import logging
from typing import Dict, Any, List, Optional
from groq import Groq
from langchain_qdrant import QdrantVectorStore
from langchain_core.documents import Document

# Configure logging
logger = logging.getLogger(__name__)

# DB Config File
DB_PROFILES_FILE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "db", "sql_connections.json")
VECTOR_STORE_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "vectorstores")

# Global RAG Components 
_EMBEDDINGS = None
_QDRANT_CLIENT = None

def initialize_rag_components(embeddings_model, qdrant_client):
    """
    Called by main application (ai_server.py) to inject shared models.
    """
    global _EMBEDDINGS, _QDRANT_CLIENT
    _EMBEDDINGS = embeddings_model
    _QDRANT_CLIENT = qdrant_client
    logger.info("SQL Service RAG components initialized from main server.")

def get_embeddings():
    global _EMBEDDINGS
    if _EMBEDDINGS is None:
        raise RuntimeError("Embeddings model not initialized. Call initialize_rag_components() first.")
    return _EMBEDDINGS

def get_qdrant_client():
    global _QDRANT_CLIENT
    if _QDRANT_CLIENT is None:
        raise RuntimeError("Qdrant client not initialized. Call initialize_rag_components() first.")
    return _QDRANT_CLIENT


def _load_profiles() -> Dict[str, Any]:
    if not os.path.exists(DB_PROFILES_FILE):
        return {}
    try:
        with open(DB_PROFILES_FILE, 'r') as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Failed to load DB profiles: {e}")
        return {}

def _save_profiles(profiles: Dict[str, Any]):
    os.makedirs(os.path.dirname(DB_PROFILES_FILE), exist_ok=True)
    with open(DB_PROFILES_FILE, 'w') as f:
        json.dump(profiles, f, indent=4)

def save_db_connection(firm_id: str, db_config: Dict[str, Any]):
    """Save database connection details for a firm."""
    profiles = _load_profiles()
    # Basic validation
    required = ['host', 'user', 'password', 'database', 'port']
    if not all(k in db_config for k in required):
        raise ValueError(f"Missing required DB fields. Need: {required}")
    
    # Store securely (in a real app, encrypt the password)
    profiles[str(firm_id)] = db_config
    _save_profiles(profiles)
    return True

def get_db_connection_config(firm_id: str) -> Optional[Dict[str, Any]]:
    """Retrieve database connection details for a firm."""
    profiles = _load_profiles()
    return profiles.get(str(firm_id))

def delete_db_connection(firm_id: str) -> bool:
    """Delete database connection details for a firm."""
    profiles = _load_profiles()
    str_firm_id = str(firm_id)
    if str_firm_id in profiles:
        del profiles[str_firm_id]
        _save_profiles(profiles)
        return True
    return False

def get_db_connection(db_config: Dict[str, Any]):
    """Create a PyMySQL connection from config."""
    return pymysql.connect(
        host=db_config['host'],
        user=db_config['user'],
        password=db_config['password'],
        database=db_config['database'],
        port=int(db_config.get('port', 3306)),
        cursorclass=pymysql.cursors.DictCursor,
        connect_timeout=5
    )

def get_all_tables(db_config: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Fetches all tables and their column counts using a single optimized query."""
    tables_info = []
    try:
        conn = get_db_connection(db_config)
        with conn.cursor() as cursor:
             # Optimized query: fetch table name and column count in one go
            sql = """
                SELECT TABLE_NAME, COUNT(*) as COLUMN_COUNT 
                FROM information_schema.COLUMNS 
                WHERE TABLE_SCHEMA = %s 
                GROUP BY TABLE_NAME
            """
            cursor.execute(sql, (db_config['database'],))
            results = cursor.fetchall()
            
            for row in results:
                # row is a dict like {'TABLE_NAME': '...', 'COLUMN_COUNT': ...}
                # keys might be lowercase depending on cursor configuration effectively, but let's be safe
                t_name = row.get('TABLE_NAME') or row.get('table_name')
                c_count = row.get('COLUMN_COUNT') or row.get('column_count')
                
                tables_info.append({
                    "table_name": t_name,
                    "column_count": c_count
                })
                
        conn.close()
    except Exception as e:
        logger.error(f"Error fetching tables: {e}")
        # Fallback to SHOW TABLES if information_schema fails (permissions issues)
        try:
             logger.info("Falling back to SHOW TABLES iteration due to initial failure.")
             conn = get_db_connection(db_config)
             with conn.cursor() as cursor:
                cursor.execute("SHOW TABLES")
                tables = [list(row.values())[0] for row in cursor.fetchall()]
                # Just return names with 0 count to avoid hanging on massive DBs in fallback mode
                tables_info = [{"table_name": t, "column_count": 0} for t in tables]
             conn.close()
        except Exception as e2:
             logger.error(f"Fallback failed: {e2}")
             
    return tables_info

def test_connection(db_config: Dict[str, Any]) -> Dict[str, Any]:
    """Test the database connection."""
    try:
        conn = get_db_connection(db_config)
        with conn.cursor() as cursor:
            cursor.execute("SELECT 1")
        conn.close()
        
        # Connection successful, fetch tables
        tables = get_all_tables(db_config)
        
        return {
            "success": True, 
            "message": "Connection successful",
            "tables": tables
        }
    except Exception as e:
        return {"success": False, "error": str(e)}

def get_schema_summary(db_config: Dict[str, Any]) -> str:
    """Extracts a rich schema summary including table names, columns, and types."""
    try:
        conn = get_db_connection(db_config)
        schema_text = ""
        
        with conn.cursor() as cursor:
            # Get tables
            cursor.execute("SHOW TABLES")
            tables = [list(row.values())[0] for row in cursor.fetchall()]
            
            for table in tables:
                cursor.execute(f"DESCRIBE `{table}`")
                columns = cursor.fetchall()
                
                # Format: TableName (col1 type, col2 type...)
                # col keys: Field, Type, Null, Key, Default, Extra
                col_defs = []
                for col in columns:
                    col_name = col['Field']
                    col_type = col['Type']
                    key_info = "PK" if col['Key'] == 'PRI' else ""
                    col_defs.append(f"{col_name} ({col_type}) {key_info}".strip())
                
                # Get sample data (Limited to 1 row and minimal size)
                try:
                    cursor.execute(f"SELECT * FROM `{table}` LIMIT 1")
                    samples = cursor.fetchall()
                    sample_text = ""
                    if samples:
                        # Convert to string and truncate if too long
                        samples_str = str(samples)
                        if len(samples_str) > 500:
                            samples_str = samples_str[:500] + "... (truncated)"
                        sample_text = f"\nSample Data:\n{samples_str}"
                except:
                    sample_text = ""

                schema_text += f"\nTable: {table}\nColumns: {', '.join(col_defs)}{sample_text}\n"
                
                # Safety break if context gets too massive (e.g. > 15000 chars)
                if len(schema_text) > 20000:
                    schema_text += "\n... (Schema truncated due to size limit) ..."
                    break

                
        conn.close()
        return schema_text
    except Exception as e:
        logger.error(f"Schema extraction failed: {e}")
        # Return partial schema or error, but don't fail completely if possible
        return f"Error loading schema: {e}"

def extract_schema_documents(db_config: Dict[str, Any]) -> List[Document]:
    """
    Extracts schema information as a list of document chunks for vectorization.
    Each table is a document.
    """
    documents = []
    try:
        conn = get_db_connection(db_config)
        with conn.cursor() as cursor:
            cursor.execute("SHOW TABLES")
            tables = [list(row.values())[0] for row in cursor.fetchall()]
            
            for table in tables:
                cursor.execute(f"DESCRIBE `{table}`")
                columns = cursor.fetchall()
                
                col_details = []
                for col in columns:
                    col_text = f"- {col['Field']} ({col['Type']})"
                    if col['Key'] == 'PRI':
                        col_text += " [Primary Key]"
                    col_details.append(col_text)
                
                # Construct a semantic description
                content = f"Table Name: {table}\n"
                content += f"Columns:\n" + "\n".join(col_details)
                
                # Metadata for filtering
                metadata = {
                    "table_name": table,
                    "type": "table_schema"
                }
                
                documents.append(Document(page_content=content, metadata=metadata))
        conn.close()
    except Exception as e:
        logger.error(f"Schema document extraction failed: {e}")
    return documents

def sync_schema_vector_store(firm_id: str, db_config: Dict[str, Any]) -> bool:
    """
    Syncs the database schema to the vector store.
    Creates complete documents for each table and indexes them.
    """
    try:
        logger.info(f"Syncing schema for firm {firm_id}...")
        documents = extract_schema_documents(db_config)
        if not documents:
            logger.warning("No schema documents extracted.")
            return False
            
        embeddings = get_embeddings()
        client = get_qdrant_client()
        # Use consistent naming convention with ai_server.py
        collection_name = f"sql-schema-{firm_id}"
        
        
        client.recreate_collection(
            collection_name=collection_name, 
            vectors_config=client.get_collection(collection_name).config.params.vectors if False else None, # We let it default or handle elsewhere
            # Actually easier to let langchain handle or just use add_documents on a new instance
        )
        
        # Simpler fix:
        vector_store = QdrantVectorStore(
            client=client,
            collection_name=collection_name,
            embedding=embeddings
        )
        # Force recreation via client directly to be sure, then add
        from qdrant_client.http.models import VectorParams, Distance
        try:
             client.recreate_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(size=len(embeddings.embed_query("test")), distance=Distance.COSINE)
             )
        except Exception as e:
             logger.error(f"Collection creation error: {e}")
             
        vector_store.add_documents(documents)
        logger.info(f"Schema synced successfully for firm {firm_id}. Indexed {len(documents)} tables.")
        return True
    except Exception as e:
        logger.error(f"Vector store sync failed: {e}")
        return False

def get_relevant_schema(firm_id: str, query: str, db_config: Dict[str, Any] = None) -> str:
    """
    Retrieves relevant schema context using RAG.
    Selects top relevant tables based on query similarity.
    """
    try:
        # Use consistent naming convention with ai_server.py
        collection_name = f"sql-schema-{firm_id}"
        embeddings = get_embeddings()
        client = get_qdrant_client()
        
        # Check if collection exists and is green (ready)
        collection_ready = False
        try:
            status = client.get_collection(collection_name).status
            from qdrant_client.http.models import CollectionStatus
            if status == CollectionStatus.GREEN:
                collection_ready = True
        except:
            collection_ready = False
        
        if not collection_ready:
            if db_config:
                logger.info(f"Schema index {collection_name} not found or not ready. Syncing now...")
                success = sync_schema_vector_store(firm_id, db_config)
                if not success:
                    raise Exception("Failed to sync schema.")
            else:
                return "" # Can't sync without config
        else:
             logger.info(f"Using existing schema index: {collection_name}")
        
        vector_store = QdrantVectorStore(
            client=client, 
            collection_name=collection_name, 
            embedding=embeddings
        )
        
        # Search for relevant tables
        # k=7 allows for a good breadth of tables without overwhelming context
        results = vector_store.similarity_search(query, k=7)
        
        if not results:
            logger.warning("No relevant tables found via RAG.")
            if db_config:
                 pass        
        # Format results
        schema_parts = []
        for doc in results:
            schema_parts.append(doc.page_content)
            
        return "\n\n".join(schema_parts)
        
    except Exception as e:
        logger.error(f"RAG Retrieval failed: {e}")
        # Fallback to full schema extraction if RAG fails (e.g. library error)
        if db_config:
            logger.info("Falling back to full schema extraction.")
            return get_schema_summary(db_config)
        return "Error loading schema context."


from dotenv import load_dotenv

load_dotenv()

def fetch_llm_api_key(firm_id: str, user_id: int, provider: str = 'GROQ') -> Optional[str]:
    """Fetch the LLM API key from the main application database."""
    try:
        # Main App DB Connection (where LLM_DETAILS tables are stored)
        conn = pymysql.connect(
            host=os.getenv("DB_HOST"),
            user=os.getenv("DB_USER"),
            password=os.getenv("DB_PASSWORD"),
            database=os.getenv("DB_DATABASE"),
            port=int(os.getenv("DB_PORT", 3306)),
            cursorclass=pymysql.cursors.DictCursor,
            connect_timeout=5
        )
        with conn.cursor() as cursor:
            # Modified query to check both FIRMID and USERID or just FIRMID as a fallback if needed?
            # User explicit request: "userid and firmid is passed use both to get api key"
            sql = """
                SELECT API_KEY FROM LLM_DETAILS
                WHERE FIRMID = %s AND USERID = %s AND LLM_PROVIDER = %s AND STATUS = 'ACTIVE'
                ORDER BY UPD_DTM DESC LIMIT 1
            """
            cursor.execute(sql, (firm_id, user_id, provider))
            result = cursor.fetchone()
        conn.close()
        return result['API_KEY'] if result else None
    except Exception as e:
        logger.error(f"Failed to fetch API key from DB: {e}")
        return None

def generate_sql_query(user_query: str, firm_id: str, user_id: int = None, groq_api_key: str = None) -> Dict[str, Any]:
    """Generates SQL query using LLM based on dynamic schema."""
    
    # 1. Get DB Config
    db_config = get_db_connection_config(firm_id)
    if not db_config:
        return {"success": False, "error": "Database not configured for this firm."}
    
    # 2. Extract Schema (RAG approach)
    # Check if vector store needs update? We assume it's synced. 
    # But for robustness, passing db_config allows lazy sync if missing.
    schema_context = get_relevant_schema(firm_id, user_query, db_config)
    
    # If schema is empty (RAG failed totally), fallback to dynamic summary logic
    if not schema_context or "Error" in schema_context:
         logger.warning("RAG schema empty, falling back to legacy get_schema_summary")
         schema_context = get_schema_summary(db_config)

    # Add specific instruction about the provided context
    schema_context = f"Relevant Schema Tables (Selected by RAG):\n{schema_context}"

    
    # 3. Call LLM (Groq)
    if not groq_api_key and user_id:
         # Fetch from DB using both firm_id and user_id
         groq_api_key = fetch_llm_api_key(firm_id, user_id, 'GROQ')
    
    if not groq_api_key:
        error_msg = "LLM API Key missing."
        if not user_id:
            error_msg += " user_id is required to fetch API key."
        else:
            error_msg += " Please configure LLM settings for this user/firm."
            
        return {"success": False, "error": error_msg}

    try:
        client = Groq(api_key=groq_api_key)
        
        prompt = f"""
You are an expert MySQL Data Analyst. 
Given the following database schema and sample data, write a SQL query to answer the user's question.

Schema Context:
{schema_context}

User Question: {user_query}

Instructions:
1. Return ONLY the SQL query. No markdown formatting, no explanations.
2. If the user asks for "all today", use the CURDATE() function.
3. Handle complex joins if tables are related by naming conventions (e.g. vendor_id).
4. Do not limit results unless specified.
5. Ensure valid MySQL syntax.

SQL:
"""
        # Truncate prompt context if extremely long (Safety net)
        # llama-3.1-8b has 128k context, but let's be safe for global limits
        if len(prompt) > 50000:
            logger.warning(f"Prompt too long ({len(prompt)}), truncating schema context.")
            prompt = prompt[:50000] + "\n...[truncated]..."

        completion = client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model="llama-3.3-70b-versatile", # Or user preferred model
            temperature=0.1,
        )
        
        generated_sql = completion.choices[0].message.content.strip()
        # Clean cleanup
        generated_sql = generated_sql.replace("```sql", "").replace("```", "").strip()
        
        return {"success": True, "sql": generated_sql}
        
    except Exception as e:
        logger.error(f"LLM Generation failed: {e}")
        return {"success": False, "error": str(e)}

def execute_generated_sql(sql: str, firm_id: str) -> Dict[str, Any]:
    """Executes the given SQL against the firm's database."""
    db_config = get_db_connection_config(firm_id)
    if not db_config:
        return {"success": False, "error": "Database not configured."}
    
    try:
        conn = get_db_connection(db_config)
        with conn.cursor() as cursor:
            cursor.execute(sql)
            results = cursor.fetchall()
            row_count = len(results)
        conn.close()
        return {"success": True, "results": results, "count": row_count}
    except Exception as e:
        logger.error(f"SQL execution failed: {e}")
        return {"success": False, "error": str(e)}
