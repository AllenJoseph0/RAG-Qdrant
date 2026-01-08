import os
import time
import logging
import pymysql
from dotenv import load_dotenv

load_dotenv()

# Configure logging
logger = logging.getLogger(__name__)

DB_CONFIG = {
    'host': os.getenv("DB_HOST", "localhost"),
    'user': os.getenv("DB_USER", "root"),
    'password': os.getenv("DB_PASSWORD", ""),
    'database': os.getenv("DB_DATABASE", "rag_db"),
    'port': int(os.getenv("DB_PORT", 3306)),
    'cursorclass': pymysql.cursors.DictCursor
}

def log_rag_metric(firm_id, user_id=None, query_text="", query_type=None, 
                   response_text=None, latency_ms=0, success=False, 
                   error_message=None, context_docs_count=0, sql_executed=None):
    """
    Logs RAG/SQL metrics to the rag_metrics table.
    """
    try:
        conn = pymysql.connect(**DB_CONFIG)
        with conn.cursor() as cursor:
            sql = """
                INSERT INTO rag_metrics 
                (firm_id, user_id, query_text, query_type, response_text, 
                 latency_ms, success, error_message, context_docs_count, sql_executed)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            """
            cursor.execute(sql, (
                firm_id, 
                user_id, 
                query_text, 
                query_type, 
                response_text, 
                latency_ms, 
                int(success), 
                error_message, 
                context_docs_count, 
                sql_executed
            ))
        conn.commit()
        conn.close()
        logger.info(f"Logged metric for firm {firm_id}: success={success}, latency={latency_ms}ms")
    except Exception as e:
        logger.error(f"Failed to log RAG metric: {e}")
