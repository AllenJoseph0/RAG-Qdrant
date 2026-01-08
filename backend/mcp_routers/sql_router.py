from flask import Blueprint, request, jsonify
from .sql_service import save_db_connection, test_connection, generate_sql_query, execute_generated_sql, get_db_connection_config

sql_bp = Blueprint('sql_agent', __name__)

import time

# Simple in-memory cache: {firm_id: {'timestamp': float, 'result': dict}}
TEST_CACHE = {}
CACHE_DURATION = 60  # seconds

@sql_bp.route('/connect', methods=['POST', 'DELETE'])
def connect_db():
    from .metrics import log_rag_metric
    if request.method == 'DELETE':
        firm_id = request.args.get('firm_id') or request.json.get('firm_id')
        if not firm_id:
            return jsonify({"error": "firm_id required"}), 400
        
        # Clear cache on delete
        TEST_CACHE.pop(firm_id, None)

        from .sql_service import delete_db_connection
        if delete_db_connection(firm_id):
            return jsonify({"success": True, "message": "Database configuration deleted."})
        else:
            return jsonify({"error": "Configuration not found"}), 404

    # POST logic
    data = request.json
    firm_id = data.get('firm_id')
    user_id = data.get('user_id')
    db_config = data.get('db_config')
    
    if not firm_id or not db_config:
        return jsonify({"error": "firm_id and db_config required"}), 400
        
    # Test first
    test_result = test_connection(db_config)
    if not test_result['success']:
        return jsonify({"error": "Connection failed", "details": test_result['error']}), 400
        
    # Save
    try:
        # We invoke save with just firm_id as per current design, but can log user_id
        save_db_connection(firm_id, db_config)
        
        # Cache the successful test result so immediate subsequent checks don't hit DB
        TEST_CACHE[firm_id] = {
            'timestamp': time.time(),
            'result': test_result
        }
        
        return jsonify({"success": True, "message": "Database connected and saved."})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@sql_bp.route('/config', methods=['GET'])
def get_config():
    firm_id = request.args.get('firm_id')
    if not firm_id:
        return jsonify({"error": "firm_id required"}), 400
        
    config = get_db_connection_config(firm_id)
    if config:
        return jsonify({"success": True, "config": config})
    return jsonify({"success": False, "message": "No configuration found"}), 404

@sql_bp.route('/test', methods=['GET'])
def test_db_connection_route():
    firm_id = request.args.get('firm_id')
    user_id = request.args.get('user_id')
    refresh = request.args.get('refresh', 'false').lower() == 'true'
    db_name = request.args.get('db_name')
    
    if not firm_id:
        return jsonify({"error": "firm_id required"}), 400

    # Return cached result if valid and not forcing refresh
    if not refresh and firm_id in TEST_CACHE:
        entry = TEST_CACHE[firm_id]
        if time.time() - entry['timestamp'] < CACHE_DURATION:
            return jsonify(entry['result'])
        
    config = get_db_connection_config(firm_id, db_name=db_name)
    if not config:
        return jsonify({"success": False, "message": "Database not configured. Please add details below."}), 200
        
    result = test_connection(config)
    
    # Update cache
    TEST_CACHE[firm_id] = {
        'timestamp': time.time(),
        'result': result
    }
    
    return jsonify(result)

@sql_bp.route('/generate', methods=['POST'])
def generate_sql():
    data = request.json
    firm_id = data.get('firm_id')
    user_id = data.get('user_id') # Optional but recommended
    user_query = data.get('query')
    
    if not firm_id or not user_query:
        return jsonify({"error": "firm_id and query required"}), 400
        
    result = generate_sql_query(user_query, firm_id, user_id=user_id)
    if not result['success']:
        return jsonify(result), 500
        
    return jsonify(result)



@sql_bp.route('/ask', methods=['POST'])
def ask_question():
    """End-to-end endpoint: Generate + Execute"""
    from .metrics import log_rag_metric
    import time
    start_time = time.time()

    data = request.json
    firm_id = data.get('firm_id')
    user_id = data.get('user_id') # Optional but recommended
    user_query = data.get('query')
    
    if not firm_id or not user_query:
        return jsonify({"error": "firm_id and query required"}), 400
        
    # 1. Generate
    gen_result = generate_sql_query(user_query, firm_id, user_id=user_id)
    if not gen_result['success']:
        # Log failure
        latency_ms = int((time.time() - start_time) * 1000)
        try:
            log_rag_metric(
                firm_id=firm_id,
                user_id=user_id,
                query_text=user_query,
                query_type="sql_agent",
                response_text=gen_result.get("error"),
                latency_ms=latency_ms,
                success=False,
                error_message=gen_result.get("error"),
                sql_executed=None
            )
        except Exception as e:
            print(f"Metrics log failed: {e}")

        # Return generic error instead of strictly 500 to handle known config issues gracefully
        return jsonify({"step": "generation", "error": gen_result.get("error")}), 400 
        
    sql = gen_result['sql']
    
    # 2. Execute
    exec_result = execute_generated_sql(sql, firm_id)
    latency_ms = int((time.time() - start_time) * 1000)
    
    success = exec_result['success']
    results = exec_result.get("results")
    row_count = exec_result.get("count", 0)
    error_msg = exec_result.get("error")
    
    # Format response text for logging
    response_text = f"SQL Executed. Rows: {row_count}" if success else f"Execution Failed: {error_msg}"

    try:
        log_rag_metric(
            firm_id=firm_id,
            user_id=user_id,
            query_text=user_query,
            query_type="sql_agent",
            response_text=response_text,
            latency_ms=latency_ms,
            success=success,
            error_message=error_msg,
            sql_executed=sql,
            context_docs_count=0 # No docs context in this flow explicitly known here without digging into generate_sql_query
        )
    except Exception as e:
        print(f"Metrics log failed: {e}")

    if not success:
        return jsonify({
            "success": False,
            "sql": sql,
            "error": error_msg,
            "message": "SQL generated but execution failed."
        }), 500

    return jsonify({
        "success": True,
        "sql": sql,
        "results": results,
        "count": row_count,
        "message": "SQL generated and executed successfully."
    })

@sql_bp.route('/sync', methods=['POST'])
def sync_schema_endpoint():
    data = request.json
    firm_id = data.get('firm_id')
    
    if not firm_id:
        return jsonify({"error": "firm_id required"}), 400

    db_name = data.get('db_name')
    # Get saved config
    db_config = get_db_connection_config(firm_id, db_name=db_name)
    if not db_config:
        return jsonify({"error": "Database not configured for this firm"}), 400

    from .sql_service import sync_schema_vector_store
    success = sync_schema_vector_store(firm_id, db_config)
    
    if success:
        return jsonify({"success": True, "message": "Schema synced to vector store."})
    else:
        return jsonify({"success": False, "error": "Failed to sync schema."}), 500

@sql_bp.route('/tables', methods=['GET'])
def get_tables():
    """Get all table names for autocomplete"""
    firm_id = request.args.get('firm_id')
    
    if not firm_id:
        return jsonify({"error": "firm_id required"}), 400
    
    db_name = request.args.get('db_name')
    db_config = get_db_connection_config(firm_id, db_name=db_name)
    if not db_config:
        return jsonify({"success": False, "tables": [], "message": "Database not configured"}), 200
    
    from .sql_service import get_all_tables
    tables_info = get_all_tables(db_config)
    
    # Extract just the table names for autocomplete
    table_names = [t['table_name'] for t in tables_info if t.get('table_name')]
    
    return jsonify({"success": True, "tables": table_names})
