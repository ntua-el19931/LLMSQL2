"""
Flask Web Demo for LLMSQL2 Text-to-SQL System.

Provides a simple web interface to:
1. Enter natural language questions
2. Select database and model
3. Generate SQL queries
4. Execute and view results

Usage:
    python -m src.web_demo
    # Then open http://localhost:5000 in browser
"""

import json
import time
import traceback
from pathlib import Path
from flask import Flask, request, jsonify, render_template_string
from typing import Dict, Optional

from .inference_api import Text2SQLInference, ModelType, SCHEMAS
from .database import DatabaseConnection, DATABASES
from .utils import logger

app = Flask(__name__)

# Global model cache
_models: Dict[str, Text2SQLInference] = {}

# HTML Template for the web interface
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>LLMSQL2 - Text to SQL Demo</title>
    <style>
        * { box-sizing: border-box; margin: 0; padding: 0; }
        body { font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; background: #1a1a2e; color: #eee; min-height: 100vh; padding: 20px; }
        .container { max-width: 1200px; margin: 0 auto; }
        h1 { text-align: center; margin-bottom: 30px; color: #00d4ff; }
        .subtitle { text-align: center; color: #888; margin-bottom: 30px; }
        .grid { display: grid; grid-template-columns: 1fr 1fr; gap: 20px; }
        .card { background: #16213e; border-radius: 12px; padding: 20px; box-shadow: 0 4px 6px rgba(0,0,0,0.3); }
        .card h2 { color: #00d4ff; margin-bottom: 15px; font-size: 1.2em; }
        label { display: block; margin-bottom: 5px; color: #aaa; }
        select, textarea, input { width: 100%; padding: 12px; border: 1px solid #333; border-radius: 8px; background: #0f0f23; color: #eee; font-size: 14px; margin-bottom: 15px; }
        textarea { min-height: 100px; resize: vertical; }
        button { background: linear-gradient(135deg, #00d4ff, #0099cc); color: #fff; border: none; padding: 12px 24px; border-radius: 8px; cursor: pointer; font-size: 16px; font-weight: bold; width: 100%; transition: transform 0.2s; }
        button:hover { transform: translateY(-2px); }
        button:disabled { background: #555; cursor: not-allowed; transform: none; }
        .result-box { background: #0f0f23; border-radius: 8px; padding: 15px; margin-top: 15px; }
        .sql-output { font-family: 'Courier New', monospace; white-space: pre-wrap; word-break: break-all; color: #00ff88; }
        .error { color: #ff6b6b; }
        .success { color: #00ff88; }
        .metrics { display: grid; grid-template-columns: repeat(3, 1fr); gap: 10px; margin-top: 15px; }
        .metric { background: #1a1a2e; padding: 10px; border-radius: 8px; text-align: center; }
        .metric-value { font-size: 1.5em; font-weight: bold; color: #00d4ff; }
        .metric-label { font-size: 0.8em; color: #888; }
        table { width: 100%; border-collapse: collapse; margin-top: 10px; }
        th, td { padding: 8px 12px; text-align: left; border-bottom: 1px solid #333; }
        th { background: #1a1a2e; color: #00d4ff; }
        .loading { display: none; text-align: center; padding: 20px; }
        .spinner { border: 3px solid #333; border-top: 3px solid #00d4ff; border-radius: 50%; width: 30px; height: 30px; animation: spin 1s linear infinite; margin: 0 auto; }
        @keyframes spin { 0% { transform: rotate(0deg); } 100% { transform: rotate(360deg); } }
        .examples { margin-top: 15px; }
        .example { background: #1a1a2e; padding: 8px 12px; border-radius: 4px; margin: 5px 0; cursor: pointer; transition: background 0.2s; }
        .example:hover { background: #2a2a4e; }
        .schema-box { font-family: monospace; font-size: 12px; color: #888; background: #0f0f23; padding: 10px; border-radius: 8px; max-height: 150px; overflow-y: auto; }
        .full-width { grid-column: span 2; }
    </style>
</head>
<body>
    <div class="container">
        <h1>🔮 LLMSQL2 Text-to-SQL Demo</h1>
        <p class="subtitle">Convert natural language questions to SQL queries using GPT-2 and TinyLlama</p>
        
        <div class="grid">
            <!-- Input Section -->
            <div class="card">
                <h2>📝 Input</h2>
                
                <label>Model</label>
                <select id="model">
                    <option value="gpt2">GPT-2 (Fast, ~80MB)</option>
                    <option value="tinyllama">TinyLlama (Better quality, ~2GB)</option>
                </select>
                
                <label>Database</label>
                <select id="database" onchange="updateSchema()">
                    <option value="geography">Geography (US states, cities, rivers)</option>
                    <option value="advising">Advising (University courses)</option>
                    <option value="atis">ATIS (Airline travel)</option>
                    <option value="restaurants">Restaurants (Locations, ratings)</option>
                </select>
                
                <label>Question</label>
                <textarea id="question" placeholder="Enter your question in natural language...">What is the capital of Texas?</textarea>
                
                <div style="display: flex; gap: 10px;">
                    <button onclick="generateSQL()" id="generateBtn">🚀 Generate SQL</button>
                    <button onclick="generateAndExecute()" id="executeBtn" style="background: linear-gradient(135deg, #00ff88, #00cc66);">⚡ Generate & Execute</button>
                </div>
                
                <div class="examples">
                    <h3 style="color: #888; font-size: 0.9em; margin-bottom: 10px;">Example Questions:</h3>
                    <div class="example" onclick="setQuestion('What is the capital of Texas?')">What is the capital of Texas?</div>
                    <div class="example" onclick="setQuestion('Which states border California?')">Which states border California?</div>
                    <div class="example" onclick="setQuestion('What is the largest city in the US?')">What is the largest city in the US?</div>
                    <div class="example" onclick="setQuestion('How many rivers are longer than 500 miles?')">How many rivers are longer than 500 miles?</div>
                </div>
            </div>
            
            <!-- Schema Section -->
            <div class="card">
                <h2>📊 Database Schema</h2>
                <div id="schemaBox" class="schema-box"></div>
                
                <div class="loading" id="loading">
                    <div class="spinner"></div>
                    <p style="margin-top: 10px;">Generating SQL...</p>
                </div>
                
                <div id="sqlResult" class="result-box" style="display: none;">
                    <h3 style="color: #00d4ff; margin-bottom: 10px;">Generated SQL:</h3>
                    <pre class="sql-output" id="sqlOutput"></pre>
                </div>
                
                <div id="metrics" class="metrics" style="display: none;">
                    <div class="metric">
                        <div class="metric-value" id="inferenceTime">-</div>
                        <div class="metric-label">Inference Time</div>
                    </div>
                    <div class="metric">
                        <div class="metric-value" id="modelUsed">-</div>
                        <div class="metric-label">Model</div>
                    </div>
                    <div class="metric">
                        <div class="metric-value" id="dbUsed">-</div>
                        <div class="metric-label">Database</div>
                    </div>
                </div>
            </div>
            
            <!-- Results Section -->
            <div class="card full-width" id="resultsCard" style="display: none;">
                <h2>📋 Query Results</h2>
                <div id="resultsContainer"></div>
            </div>
        </div>
    </div>
    
    <script>
        const schemas = {{ schemas | tojson }};
        
        function updateSchema() {
            const db = document.getElementById('database').value;
            document.getElementById('schemaBox').textContent = schemas[db]?.full || '';
        }
        
        function setQuestion(q) {
            document.getElementById('question').value = q;
        }
        
        async function generateSQL() {
            const model = document.getElementById('model').value;
            const database = document.getElementById('database').value;
            const question = document.getElementById('question').value;
            
            if (!question.trim()) {
                alert('Please enter a question');
                return;
            }
            
            document.getElementById('loading').style.display = 'block';
            document.getElementById('sqlResult').style.display = 'none';
            document.getElementById('metrics').style.display = 'none';
            document.getElementById('generateBtn').disabled = true;
            document.getElementById('executeBtn').disabled = true;
            
            try {
                const response = await fetch('/api/generate', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ model, database, question })
                });
                
                const data = await response.json();
                
                if (data.error) {
                    document.getElementById('sqlOutput').innerHTML = '<span class="error">' + data.error + '</span>';
                } else {
                    document.getElementById('sqlOutput').textContent = data.sql;
                    document.getElementById('inferenceTime').textContent = data.inference_time.toFixed(2) + 's';
                    document.getElementById('modelUsed').textContent = data.model.toUpperCase();
                    document.getElementById('dbUsed').textContent = data.database;
                    document.getElementById('metrics').style.display = 'grid';
                }
                
                document.getElementById('sqlResult').style.display = 'block';
            } catch (err) {
                document.getElementById('sqlOutput').innerHTML = '<span class="error">Error: ' + err.message + '</span>';
                document.getElementById('sqlResult').style.display = 'block';
            } finally {
                document.getElementById('loading').style.display = 'none';
                document.getElementById('generateBtn').disabled = false;
                document.getElementById('executeBtn').disabled = false;
            }
        }
        
        async function generateAndExecute() {
            const model = document.getElementById('model').value;
            const database = document.getElementById('database').value;
            const question = document.getElementById('question').value;
            
            if (!question.trim()) {
                alert('Please enter a question');
                return;
            }
            
            document.getElementById('loading').style.display = 'block';
            document.getElementById('sqlResult').style.display = 'none';
            document.getElementById('metrics').style.display = 'none';
            document.getElementById('resultsCard').style.display = 'none';
            document.getElementById('generateBtn').disabled = true;
            document.getElementById('executeBtn').disabled = true;
            
            try {
                const response = await fetch('/api/generate_and_execute', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ model, database, question })
                });
                
                const data = await response.json();
                
                if (data.error) {
                    document.getElementById('sqlOutput').innerHTML = '<span class="error">' + data.error + '</span>';
                    document.getElementById('sqlResult').style.display = 'block';
                } else {
                    document.getElementById('sqlOutput').textContent = data.sql;
                    document.getElementById('inferenceTime').textContent = data.inference_time.toFixed(2) + 's';
                    document.getElementById('modelUsed').textContent = data.model.toUpperCase();
                    document.getElementById('dbUsed').textContent = data.database;
                    document.getElementById('metrics').style.display = 'grid';
                    document.getElementById('sqlResult').style.display = 'block';
                    
                    // Show execution results
                    const resultsContainer = document.getElementById('resultsContainer');
                    if (data.execution_error) {
                        resultsContainer.innerHTML = '<p class="error">Execution Error: ' + data.execution_error + '</p>';
                    } else if (data.results && data.results.length > 0) {
                        let html = '<p class="success">Returned ' + data.results.length + ' rows (exec time: ' + data.execution_time.toFixed(3) + 's)</p>';
                        html += '<table><thead><tr>';
                        
                        // Header
                        for (let i = 0; i < data.results[0].length; i++) {
                            html += '<th>Column ' + (i+1) + '</th>';
                        }
                        html += '</tr></thead><tbody>';
                        
                        // Rows (limit to 20)
                        const maxRows = Math.min(data.results.length, 20);
                        for (let i = 0; i < maxRows; i++) {
                            html += '<tr>';
                            for (const val of data.results[i]) {
                                html += '<td>' + (val ?? 'NULL') + '</td>';
                            }
                            html += '</tr>';
                        }
                        
                        if (data.results.length > 20) {
                            html += '<tr><td colspan="100" style="text-align:center;color:#888;">... and ' + (data.results.length - 20) + ' more rows</td></tr>';
                        }
                        
                        html += '</tbody></table>';
                        resultsContainer.innerHTML = html;
                    } else {
                        resultsContainer.innerHTML = '<p class="success">Query executed successfully. No results returned.</p>';
                    }
                    document.getElementById('resultsCard').style.display = 'block';
                }
            } catch (err) {
                document.getElementById('sqlOutput').innerHTML = '<span class="error">Error: ' + err.message + '</span>';
                document.getElementById('sqlResult').style.display = 'block';
            } finally {
                document.getElementById('loading').style.display = 'none';
                document.getElementById('generateBtn').disabled = false;
                document.getElementById('executeBtn').disabled = false;
            }
        }
        
        // Initialize
        updateSchema();
    </script>
</body>
</html>
"""


def get_model(model_type: str, database: str) -> Text2SQLInference:
    """Get or load a model from cache."""
    cache_key = f"{model_type}-{database}"
    
    if cache_key not in _models:
        inference = Text2SQLInference()
        
        # Determine checkpoint path
        if model_type == "gpt2":
            checkpoint = f"/app/results/gpt2-{database}/final"
        else:
            checkpoint = f"/app/results/tinyllama-{database}/final"
        
        # Check if checkpoint exists
        if not Path(checkpoint).exists():
            raise FileNotFoundError(f"Model checkpoint not found: {checkpoint}")
        
        inference.load_model(model_type, checkpoint)
        _models[cache_key] = inference
    
    return _models[cache_key]


@app.route('/')
def index():
    """Render the main demo page."""
    return render_template_string(HTML_TEMPLATE, schemas=SCHEMAS)


@app.route('/api/generate', methods=['POST'])
def api_generate():
    """Generate SQL from a question."""
    try:
        data = request.json
        model_type = data.get('model', 'gpt2')
        database = data.get('database', 'geography')
        question = data.get('question', '')
        
        if not question:
            return jsonify({'error': 'Question is required'}), 400
        
        inference = get_model(model_type, database)
        result = inference.generate(question, database)
        
        return jsonify({
            'sql': result.sql,
            'raw_sql': result.raw_sql,
            'model': result.model,
            'database': result.database,
            'inference_time': result.inference_time
        })
    
    except FileNotFoundError as e:
        return jsonify({'error': str(e)}), 404
    except Exception as e:
        logger.error(f"Generation error: {e}")
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


@app.route('/api/generate_and_execute', methods=['POST'])
def api_generate_and_execute():
    """Generate SQL and execute it against the database."""
    try:
        data = request.json
        model_type = data.get('model', 'gpt2')
        database = data.get('database', 'geography')
        question = data.get('question', '')
        db_backend = data.get('backend', 'sqlite')  # 'sqlite' or 'postgresql'
        
        if not question:
            return jsonify({'error': 'Question is required'}), 400
        
        # Generate SQL
        inference = get_model(model_type, database)
        result = inference.generate(question, database)
        
        response = {
            'sql': result.sql,
            'raw_sql': result.raw_sql,
            'model': result.model,
            'database': result.database,
            'inference_time': result.inference_time,
            'results': None,
            'execution_error': None,
            'execution_time': 0
        }
        
        # Execute SQL
        db_config_key = f"{database}_{db_backend}"
        if db_config_key not in DATABASES:
            db_config_key = f"{database}_sqlite"  # Fallback to SQLite
        
        try:
            conn = DatabaseConnection(DATABASES[db_config_key])
            conn.connect()
            
            start_time = time.time()
            results = conn.execute(result.sql)
            exec_time = time.time() - start_time
            
            conn.close()
            
            response['results'] = [list(row) for row in results[:100]]  # Limit to 100 rows
            response['execution_time'] = exec_time
            
        except Exception as exec_error:
            response['execution_error'] = str(exec_error)
        
        return jsonify(response)
    
    except FileNotFoundError as e:
        return jsonify({'error': str(e)}), 404
    except Exception as e:
        logger.error(f"Generation/execution error: {e}")
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


@app.route('/api/models', methods=['GET'])
def api_list_models():
    """List available model checkpoints."""
    models = []
    results_dir = Path('/app/results')
    
    for model_type in ['gpt2', 'tinyllama']:
        for db in ['geography', 'advising', 'atis', 'restaurants']:
            checkpoint = results_dir / f"{model_type}-{db}" / "final"
            if checkpoint.exists():
                models.append({
                    'model': model_type,
                    'database': db,
                    'path': str(checkpoint)
                })
    
    return jsonify({'models': models})


@app.route('/api/databases', methods=['GET'])
def api_list_databases():
    """List available databases."""
    return jsonify({
        'databases': list(SCHEMAS.keys()),
        'schemas': SCHEMAS
    })


@app.route('/api/health', methods=['GET'])
def api_health():
    """Health check endpoint."""
    return jsonify({
        'status': 'ok',
        'timestamp': time.time()
    })


def main():
    """Run the Flask development server."""
    print("\n" + "="*60)
    print("LLMSQL2 Web Demo")
    print("="*60)
    print("Starting server at http://0.0.0.0:5000")
    print("Press Ctrl+C to stop")
    print("="*60 + "\n")
    
    app.run(host='0.0.0.0', port=5000, debug=False)


if __name__ == '__main__':
    main()
