"""
LLMSQL2 Web Frontend
Simple web interface for database exploration and SQL queries.
"""

from flask import Flask, render_template, request, jsonify
import sqlite3
from pathlib import Path
import subprocess
import json
import re

app = Flask(__name__, template_folder='../templates', static_folder='../static')

# Base paths
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data" / "text2sql-data" / "data"
TRAINING_RESULTS_DIR = BASE_DIR / "training_results"

DATABASES = {
    "geography": {
        "path": DATA_DIR / "geography-db.added-in-2020.sqlite",
        "description": "US Geography - states, cities, rivers, mountains"
    },
    "atis": {
        "path": DATA_DIR / "atis-db.added-in-2020.sqlite",
        "description": "Flight booking - airlines, airports, flights, fares"
    },
    "advising": {
        "path": DATA_DIR / "advising-db.added-in-2020.sqlite",
        "description": "University advising - courses, students, instructors"
    },
    "restaurants": {
        "path": DATA_DIR / "restaurants-db.added-in-2020.sqlite",
        "description": "Restaurant locations and ratings"
    }
}


def get_db_connection(db_name):
    """Get SQLite connection for a database."""
    if db_name not in DATABASES:
        return None
    db_path = DATABASES[db_name]["path"]
    if not db_path.exists():
        return None
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    return conn


def get_tables(db_name):
    """Get list of tables in a database."""
    conn = get_db_connection(db_name)
    if not conn:
        return []
    cursor = conn.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")
    tables = [row[0] for row in cursor.fetchall()]
    conn.close()
    return tables


def get_table_schema(db_name, table_name):
    """Get schema for a specific table."""
    conn = get_db_connection(db_name)
    if not conn:
        return []
    cursor = conn.cursor()
    cursor.execute(f"PRAGMA table_info({table_name})")
    columns = [{"name": row[1], "type": row[2], "nullable": not row[3], "pk": row[5]} 
               for row in cursor.fetchall()]
    conn.close()
    return columns


def get_table_data(db_name, table_name, limit=100):
    """Get data from a table."""
    conn = get_db_connection(db_name)
    if not conn:
        return [], []
    cursor = conn.cursor()
    cursor.execute(f"SELECT * FROM {table_name} LIMIT {limit}")
    columns = [description[0] for description in cursor.description]
    rows = [dict(row) for row in cursor.fetchall()]
    conn.close()
    return columns, rows


def execute_query(db_name, query):
    """Execute a SQL query and return results."""
    conn = get_db_connection(db_name)
    if not conn:
        return None, "Database not found"
    
    try:
        cursor = conn.cursor()
        cursor.execute(query)
        
        # Check if it's a SELECT query
        if query.strip().upper().startswith("SELECT"):
            columns = [description[0] for description in cursor.description]
            rows = [dict(row) for row in cursor.fetchall()]
            conn.close()
            return {"columns": columns, "rows": rows, "count": len(rows)}, None
        else:
            # INSERT, UPDATE, DELETE
            conn.commit()
            affected = cursor.rowcount
            conn.close()
            return {"affected_rows": affected}, None
    except Exception as e:
        conn.close()
        return None, str(e)


# Routes
@app.route('/')
def index():
    """Home page."""
    return render_template('index.html', databases=DATABASES)


@app.route('/database/<db_name>')
def database_view(db_name):
    """View a specific database."""
    if db_name not in DATABASES:
        return "Database not found", 404
    tables = get_tables(db_name)
    return render_template('database.html', 
                          db_name=db_name, 
                          db_info=DATABASES[db_name],
                          tables=tables)


@app.route('/database/<db_name>/table/<table_name>')
def table_view(db_name, table_name):
    """View a specific table."""
    if db_name not in DATABASES:
        return "Database not found", 404
    schema = get_table_schema(db_name, table_name)
    columns, rows = get_table_data(db_name, table_name)
    tables = get_tables(db_name)
    return render_template('table.html',
                          db_name=db_name,
                          table_name=table_name,
                          schema=schema,
                          columns=columns,
                          rows=rows,
                          tables=tables)


@app.route('/query')
def query_page():
    """SQL query page."""
    return render_template('query.html', databases=DATABASES)


# API Routes
@app.route('/api/tables/<db_name>')
def api_tables(db_name):
    """API: Get tables for a database."""
    tables = get_tables(db_name)
    return jsonify({"tables": tables})


@app.route('/api/schema/<db_name>/<table_name>')
def api_schema(db_name, table_name):
    """API: Get schema for a table."""
    schema = get_table_schema(db_name, table_name)
    return jsonify({"schema": schema})


@app.route('/api/data/<db_name>/<table_name>')
def api_data(db_name, table_name):
    """API: Get data from a table."""
    limit = request.args.get('limit', 100, type=int)
    columns, rows = get_table_data(db_name, table_name, limit)
    return jsonify({"columns": columns, "rows": rows, "count": len(rows)})


@app.route('/api/query', methods=['POST'])
def api_query():
    """API: Execute a SQL query."""
    data = request.get_json()
    db_name = data.get('database')
    query = data.get('query')
    
    if not db_name or not query:
        return jsonify({"error": "Missing database or query"}), 400
    
    result, error = execute_query(db_name, query)
    if error:
        return jsonify({"error": error}), 400
    return jsonify(result)


@app.route('/api/insert/<db_name>/<table_name>', methods=['POST'])
def api_insert(db_name, table_name):
    """API: Insert a row into a table."""
    data = request.get_json()
    
    columns = ", ".join(data.keys())
    placeholders = ", ".join(["?" for _ in data])
    values = list(data.values())
    
    query = f"INSERT INTO {table_name} ({columns}) VALUES ({placeholders})"
    
    conn = get_db_connection(db_name)
    if not conn:
        return jsonify({"error": "Database not found"}), 404
    
    try:
        cursor = conn.cursor()
        cursor.execute(query, values)
        conn.commit()
        conn.close()
        return jsonify({"success": True, "id": cursor.lastrowid})
    except Exception as e:
        conn.close()
        return jsonify({"error": str(e)}), 400


@app.route('/api/delete/<db_name>/<table_name>', methods=['POST'])
def api_delete(db_name, table_name):
    """API: Delete a row from a table."""
    data = request.get_json()
    
    # Build WHERE clause from primary key or all columns
    conditions = " AND ".join([f"{k} = ?" for k in data.keys()])
    values = list(data.values())
    
    query = f"DELETE FROM {table_name} WHERE {conditions}"
    
    conn = get_db_connection(db_name)
    if not conn:
        return jsonify({"error": "Database not found"}), 404
    
    try:
        cursor = conn.cursor()
        cursor.execute(query, values)
        conn.commit()
        affected = cursor.rowcount
        conn.close()
        return jsonify({"success": True, "deleted": affected})
    except Exception as e:
        conn.close()
        return jsonify({"error": str(e)}), 400


# ============== TEXT-TO-SQL ROUTES ==============

# Schemas for each database (short version)
SCHEMAS = {
    "geography": "state(state_name, population, area, capital, density) | city(city_name, population, country_name, state_name) | river(river_name, length, country_name, traverse) | lake(lake_name, area, country_name, state_name) | mountain(mountain_name, mountain_altitude, country_name, state_name) | border_info(state_name, border) | highlow(state_name, highest_elevation, lowest_point, highest_point, lowest_elevation)",
    "atis": "flight(flight_id, flight_number, airline_code, from_airport, to_airport, departure_time, arrival_time) | airline(airline_code, airline_name) | airport(airport_code, airport_name, airport_location, state_code) | aircraft(aircraft_code, aircraft_description, manufacturer) | city(city_code, city_name, state_code, country_name)",
    "advising": "course(course_id, name, department, number, credits, description) | student(student_id, lastname, firstname, program_id, total_credit, total_gpa) | instructor(instructor_id, name) | semester(semester_id, semester, year) | student_record(student_id, course_id, semester, grade)",
    "restaurants": "restaurant(restaurant_id, name, food_type, city_name, rating) | location(restaurant_id, house_number, street_name, city_name) | geographic(city_name, county, region)"
}

# Available models
AVAILABLE_MODELS = {
    "gpt2-geography": {"type": "gpt2", "database": "geography"},
    "gpt2-atis": {"type": "gpt2", "database": "atis"},
    "gpt2-advising": {"type": "gpt2", "database": "advising"},
    "tinyllama-geography": {"type": "tinyllama", "database": "geography"},
    "tinyllama-atis": {"type": "tinyllama", "database": "atis"},
    "tinyllama-advising": {"type": "tinyllama", "database": "advising"},
}

# Model cache
_model_cache = {}

def load_model(model_name):
    """Load a model (with caching)."""
    global _model_cache
    
    if model_name in _model_cache:
        return _model_cache[model_name]
    
    try:
        import torch
        from transformers import AutoTokenizer, AutoModelForCausalLM
        from peft import PeftModel
    except ImportError:
        return None, None, "PyTorch/Transformers not installed. Run in Docker."
    
    model_info = AVAILABLE_MODELS.get(model_name)
    if not model_info:
        return None, None, f"Unknown model: {model_name}"
    
    model_path = TRAINING_RESULTS_DIR / model_name / "final"
    if not model_path.exists():
        return None, None, f"Model not found at {model_path}"
    
    try:
        if model_info["type"] == "gpt2":
            tokenizer = AutoTokenizer.from_pretrained(str(model_path))
            model = AutoModelForCausalLM.from_pretrained(str(model_path))
        else:  # tinyllama
            base_model = AutoModelForCausalLM.from_pretrained(
                'ManthanKulakarni/TinyLlama-1.1B-Text2SQL',
                torch_dtype=torch.float32,
                low_cpu_mem_usage=True
            )
            tokenizer = AutoTokenizer.from_pretrained('ManthanKulakarni/TinyLlama-1.1B-Text2SQL')
            model = PeftModel.from_pretrained(base_model, str(model_path))
        
        model.eval()
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        _model_cache[model_name] = (model, tokenizer, None)
        return model, tokenizer, None
    except Exception as e:
        return None, None, str(e)


def generate_sql_from_model(model_name, question, database):
    """Generate SQL using a trained model."""
    model, tokenizer, error = load_model(model_name)
    
    if error:
        return None, error
    
    model_info = AVAILABLE_MODELS[model_name]
    schema = SCHEMAS.get(database, "")
    
    try:
        import torch
        
        # Build prompt based on model type
        if model_info["type"] == "gpt2":
            prompt = f"Table: {schema}\nQuestion: {question}\nSQL:"
            sql_marker = "SQL:"
        else:  # tinyllama
            prompt = f"### Schema:\n{schema}\n\n### Question:\n{question}\n\n### SQL:\n"
            sql_marker = "### SQL:"
        
        inputs = tokenizer(prompt, return_tensors='pt', padding=True)
        
        with torch.no_grad():
            if model_info["type"] == "tinyllama":
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=256,
                    num_beams=3,
                    do_sample=False,
                    pad_token_id=tokenizer.eos_token_id
                )
            else:
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=256,
                    do_sample=False,
                    pad_token_id=tokenizer.eos_token_id
                )
        
        generated = tokenizer.decode(outputs[0], skip_special_tokens=True)
        raw_sql = generated.split(sql_marker)[-1].strip()
        
        # Post-process SQL
        sql = raw_sql.split('\n')[0].strip()
        sql = sql.split(';')[0].strip()
        for marker in ['###', '```', 'Question:', 'Table:', 'Schema:']:
            if marker in sql:
                sql = sql.split(marker)[0].strip()
        
        return sql, None
    except Exception as e:
        return None, str(e)


@app.route('/text2sql')
def text2sql_page():
    """Text-to-SQL page."""
    # Get available models (only those with trained weights)
    available = {}
    for name, info in AVAILABLE_MODELS.items():
        model_path = TRAINING_RESULTS_DIR / name / "final"
        if model_path.exists():
            available[name] = info
    
    return render_template('text2sql.html', 
                          databases=DATABASES, 
                          models=available,
                          schemas=SCHEMAS)


@app.route('/api/text2sql', methods=['POST'])
def api_text2sql():
    """API: Generate SQL from natural language."""
    data = request.get_json()
    question = data.get('question', '').strip()
    model_name = data.get('model', '')
    database = data.get('database', '')
    execute = data.get('execute', False)
    
    if not question:
        return jsonify({"error": "Please enter a question"}), 400
    if not model_name:
        return jsonify({"error": "Please select a model"}), 400
    if not database:
        return jsonify({"error": "Please select a database"}), 400
    
    # Generate SQL
    generated_sql, error = generate_sql_from_model(model_name, question, database)
    
    if error:
        return jsonify({"error": f"Model error: {error}"}), 500
    
    result = {
        "question": question,
        "generated_sql": generated_sql,
        "model": model_name,
        "database": database
    }
    
    # Optionally execute the SQL
    if execute and generated_sql:
        exec_result, exec_error = execute_query(database, generated_sql)
        if exec_error:
            result["execution_error"] = exec_error
        else:
            result["execution_result"] = exec_result
    
    return jsonify(result)


if __name__ == '__main__':
    print("Starting LLMSQL2 Web App...")
    print("Open http://localhost:5000 in your browser")
    app.run(debug=True, host='0.0.0.0', port=5000)
