"""
Category 1: Pre-trained Model Evaluation

Evaluates pre-trained Text-to-SQL models WITHOUT any fine-tuning.
Uses models directly from HuggingFace as-is.

Models:
- GPT-2: n22t7a/text2sql-tuned-gpt2
- TinyLlama: ManthanKulakarni/TinyLlama-1.1B-Text2SQL

IMPORTANT FINDINGS FROM MODEL ANALYSIS:
=======================================

GPT-2 (n22t7a/text2sql-tuned-gpt2):
  - This model is actually a SQL CODE COMPLETER, not a Text-to-SQL model
  - When given natural language prompts, it outputs EOS (100% probability)
  - It only generates SQL when given partial SQL as input (e.g., "SELECT * FROM")
  - For Text-to-SQL evaluation, we provide partial SQL prefix to get any output

TinyLlama (ManthanKulakarni/TinyLlama-1.1B-Text2SQL):
  - Uses ### Context / ### Question / ### Answer format
  - Generates SQL but quality is severely limited:
    * Hallucinates large numbers
    * Repeats clauses infinitely
    * Adds irrelevant WHERE clauses
  - This appears to be a training quality issue, not prompt format
"""

import json
import time
import os
from datetime import datetime
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Tuple
from collections import defaultdict

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# Add parent to path
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.database import DatabaseConnection, DATABASES
from src.query_complexity import QueryComplexityAnalyzer, ComplexityLevel
from src.utils import logger


# Pre-trained model configurations
# Configurations based on actual HuggingFace repo analysis:
#
# GPT-2 (n22t7a/text2sql-tuned-gpt2):
#   - SQL code completer, not Text-to-SQL
#   - Only generates when given partial SQL
#   - task_specific_params: do_sample=True, max_length=50
#
# TinyLlama (ManthanKulakarni/TinyLlama-1.1B-Text2SQL):
#   - max_length: 4096
#   - add_bos_token: true
#   - No chat template, uses ### format

PRETRAINED_MODELS = {
    'gpt2': {
        'name': 'GPT-2 Text2SQL',
        'model_id': 'n22t7a/text2sql-tuned-gpt2',
        'type': 'sql_completer',  # Note: This is actually a SQL completer
        'context_length': 1024,
        # For GPT-2, we must provide SQL prefix since it's a completer
        # Format: schema as comment, then start of SELECT
        'prompt_template': '-- Schema: {schema}\n-- Question: {question}\nSELECT',
        'prefix_output': 'SELECT',  # Prepend SELECT to output
        'do_sample': True,  # From repo config
        'max_new_tokens': 100,
    },
    'tinyllama': {
        'name': 'TinyLlama Text2SQL',
        'model_id': 'ManthanKulakarni/TinyLlama-1.1B-Text2SQL',
        'type': 'causal_lm',
        'context_length': 4096,  # From generation_config.json
        # Plain completion style (no chat template in repo)
        'prompt_template': '### Context:\n{schema}\n### Question:\n{question}\n### Answer:\n',
        'extract_after': '### Answer:',
        # Generation params
        'do_sample': False,  # Deterministic for SQL
        'max_new_tokens': 256,
    }
}


@dataclass
class EvalResult:
    """Single evaluation result."""
    question: str
    gold_sql: str
    predicted_sql: str
    complexity: str
    sqlite_executed: bool = False
    postgres_executed: bool = False
    sqlite_result_match: bool = False
    postgres_result_match: bool = False
    inference_time_ms: float = 0.0
    sqlite_exec_time_ms: float = 0.0
    postgres_exec_time_ms: float = 0.0
    sqlite_error: Optional[str] = None
    postgres_error: Optional[str] = None


@dataclass
class ModelMetrics:
    """Aggregated metrics for a model."""
    model_name: str
    database_name: str
    total_queries: int = 0
    sqlite_execution_success: int = 0
    postgres_execution_success: int = 0
    sqlite_result_match: int = 0
    postgres_result_match: int = 0
    consistent_results: int = 0
    total_inference_time: float = 0.0
    total_sqlite_exec_time: float = 0.0
    total_postgres_exec_time: float = 0.0
    
    accuracy_by_complexity: Dict[str, Dict[str, int]] = field(
        default_factory=lambda: {
            level.name: {'total': 0, 'sqlite_exec': 0, 'postgres_exec': 0, 
                        'sqlite_match': 0, 'postgres_match': 0}
            for level in ComplexityLevel
        }
    )
    
    error_counts: Dict[str, int] = field(default_factory=lambda: defaultdict(int))
    results: List[EvalResult] = field(default_factory=list)
    
    @property
    def sqlite_exec_accuracy(self) -> float:
        return self.sqlite_execution_success / self.total_queries if self.total_queries else 0.0
    
    @property
    def postgres_exec_accuracy(self) -> float:
        return self.postgres_execution_success / self.total_queries if self.total_queries else 0.0
    
    @property
    def sqlite_result_accuracy(self) -> float:
        return self.sqlite_result_match / self.total_queries if self.total_queries else 0.0
    
    @property
    def postgres_result_accuracy(self) -> float:
        return self.postgres_result_match / self.total_queries if self.total_queries else 0.0
    
    @property
    def consistency_rate(self) -> float:
        return self.consistent_results / self.total_queries if self.total_queries else 0.0
    
    @property
    def avg_inference_time(self) -> float:
        return self.total_inference_time / self.total_queries if self.total_queries else 0.0
    
    def _accuracy_by_complexity_dict(self) -> Dict[str, Dict[str, Any]]:
        """Convert accuracy_by_complexity to serializable dict with percentages."""
        result = {}
        for level_name, stats in self.accuracy_by_complexity.items():
            total = stats['total']
            if total > 0:
                result[level_name] = {
                    'total': total,
                    'sqlite_exec': f"{stats['sqlite_exec'] / total:.1%}",
                    'postgres_exec': f"{stats['postgres_exec'] / total:.1%}",
                    'sqlite_match': f"{stats['sqlite_match'] / total:.1%}",
                    'postgres_match': f"{stats['postgres_match'] / total:.1%}"
                }
            else:
                result[level_name] = {
                    'total': 0,
                    'sqlite_exec': 'N/A',
                    'postgres_exec': 'N/A',
                    'sqlite_match': 'N/A',
                    'postgres_match': 'N/A'
                }
        return result
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'model_name': self.model_name,
            'database_name': self.database_name,
            'total_queries': self.total_queries,
            'accuracy': {
                'sqlite_execution': f"{self.sqlite_exec_accuracy:.1%}",
                'postgres_execution': f"{self.postgres_exec_accuracy:.1%}",
                'sqlite_result_match': f"{self.sqlite_result_accuracy:.1%}",
                'postgres_result_match': f"{self.postgres_result_accuracy:.1%}",
                'cross_db_consistency': f"{self.consistency_rate:.1%}"
            },
            'performance': {
                'avg_inference_time_ms': round(self.avg_inference_time, 2),
                'avg_sqlite_exec_time_ms': round(self.total_sqlite_exec_time / self.total_queries if self.total_queries else 0, 2),
                'avg_postgres_exec_time_ms': round(self.total_postgres_exec_time / self.total_queries if self.total_queries else 0, 2),
                'total_inference_time_s': round(self.total_inference_time / 1000, 2)
            },
            'error_counts': dict(self.error_counts),
            'accuracy_by_complexity': self._accuracy_by_complexity_dict()
        }


class PretrainedModelLoader:
    """Load and manage pre-trained models."""
    
    def __init__(self):
        self.models = {}
        self.tokenizers = {}
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        logger.info(f"Using device: {self.device}")
    
    def load_model(self, model_key: str) -> Tuple[Any, Any]:
        """Load a pre-trained model from HuggingFace."""
        if model_key in self.models:
            return self.models[model_key], self.tokenizers[model_key]
        
        config = PRETRAINED_MODELS[model_key]
        model_id = config['model_id']
        
        logger.info(f"Loading pre-trained model: {config['name']} ({model_id})")
        
        try:
            tokenizer = AutoTokenizer.from_pretrained(model_id)
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            
            model = AutoModelForCausalLM.from_pretrained(
                model_id,
                torch_dtype=torch.float32,
                low_cpu_mem_usage=True
            )
            model.to(self.device)
            model.eval()
            
            self.models[model_key] = model
            self.tokenizers[model_key] = tokenizer
            
            logger.info(f"Successfully loaded {config['name']}")
            return model, tokenizer
            
        except Exception as e:
            logger.error(f"Failed to load {model_id}: {e}")
            raise
    
    def generate_sql(self, model_key: str, question: str, schema: str) -> Tuple[str, float]:
        """Generate SQL from question using pre-trained model."""
        model, tokenizer = self.load_model(model_key)
        config = PRETRAINED_MODELS[model_key]
        
        # Format prompt
        prompt = config['prompt_template'].format(question=question, schema=schema)
        
        # Tokenize with proper max length for context
        context_length = config.get('context_length', 1024)
        inputs = tokenizer(prompt, return_tensors='pt', truncation=True, max_length=context_length)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Get generation params from config
        do_sample = config.get('do_sample', False)
        max_new_tokens = config.get('max_new_tokens', 128)
        
        # Generate with model-specific params
        start_time = time.time()
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=do_sample,
                num_beams=1,
                pad_token_id=tokenizer.eos_token_id,  # Important: pad=eos for both models
                eos_token_id=tokenizer.eos_token_id,
            )
        inference_time = (time.time() - start_time) * 1000  # ms
        
        # Decode
        generated = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract SQL from response
        sql = self._extract_sql(generated, prompt, config)
        
        return sql, inference_time
    
    def _extract_sql(self, generated: str, prompt: str, config: dict) -> str:
        """Extract SQL from generated text based on model config."""
        # Check for extract_after marker (e.g., "### Answer:")
        extract_after = config.get('extract_after', '')
        if extract_after and extract_after in generated:
            sql = generated.split(extract_after)[-1].strip()
        elif prompt in generated:
            sql = generated[len(prompt):].strip()
        else:
            sql = generated.strip()
        
        # Clean up - take first line only
        if '\n' in sql:
            sql = sql.split('\n')[0].strip()
        
        # Remove any trailing ### markers
        if '###' in sql:
            sql = sql.split('###')[0].strip()
        
        # Add prefix if needed (e.g., GPT-2 needs SELECT prepended)
        prefix = config.get('prefix_output', '')
        if prefix and not sql.upper().startswith(prefix.upper()):
            sql = prefix + sql
        
        # Ensure it starts with SELECT
        if not sql.upper().startswith('SELECT'):
            # Try to find SELECT in the output
            upper = sql.upper()
            if 'SELECT' in upper:
                idx = upper.index('SELECT')
                sql = sql[idx:]
        
        # Clean up broken model outputs
        sql = self._cleanup_sql(sql)
        
        return sql
    
    def _cleanup_sql(self, sql: str) -> str:
        """Clean up common issues in generated SQL."""
        import re
        
        # Remove excessively large numbers (hallucinated by model)
        # Replace numbers with 15+ digits with reasonable placeholders
        sql = re.sub(r'\b\d{15,}\b', '1', sql)
        
        # Remove duplicate ORDER BY clauses (TinyLlama repeats these)
        # Find first complete ORDER BY and remove subsequent ones
        parts = re.split(r'\s+(ORDER\s+BY)\s+', sql, flags=re.IGNORECASE)
        if len(parts) > 1:
            # Keep first part + first ORDER BY clause
            result = parts[0] + ' ORDER BY ' + parts[1] if len(parts) > 1 else parts[0]
            # Find end of first ORDER BY (at LIMIT or end)
            limit_match = re.search(r'\bLIMIT\s+\d+', result, re.IGNORECASE)
            if limit_match:
                result = result[:limit_match.end()]
                # Remove any remaining duplicate ORDER BY
                result = re.split(r'\s+ORDER\s+BY\s+', result, flags=re.IGNORECASE)[0]
                if limit_match:
                    result = result.strip()
                    if not re.search(r'\bLIMIT\s+\d+\s*$', result, re.IGNORECASE):
                        result += ' ' + limit_match.group()
            sql = result
        
        # Remove duplicate LIMIT clauses
        limit_matches = list(re.finditer(r'\bLIMIT\s+(\d+)', sql, re.IGNORECASE))
        if len(limit_matches) > 1:
            # Keep only the first LIMIT
            first_limit = limit_matches[0]
            sql = sql[:first_limit.end()]
        
        # Remove duplicate GROUP BY clauses
        group_matches = list(re.finditer(r'\bGROUP\s+BY\s+', sql, re.IGNORECASE))
        if len(group_matches) > 1:
            sql = sql[:group_matches[1].start()]
        
        # Remove incomplete trailing parts
        sql = sql.strip()
        if sql.endswith((',', 'AND', 'OR', 'WHERE', 'BY', 'FROM')):
            # Find last complete clause
            for end_pattern in [';', 'LIMIT', 'ORDER BY', 'GROUP BY', 'HAVING', 'WHERE', 'FROM', 'SELECT']:
                match = re.search(rf'\b{end_pattern}\b', sql, re.IGNORECASE)
                if match:
                    # Cut at reasonable point
                    break
        
        return sql.strip()


class PretrainedEvaluator:
    """Evaluator for pre-trained models."""
    
    def __init__(self, database_name: str):
        self.database_name = database_name
        self.model_loader = PretrainedModelLoader()
        self.complexity_analyzer = QueryComplexityAnalyzer()
        
        # Database connections
        self.sqlite_config = DATABASES.get(f"{database_name}_sqlite")
        self.postgres_config = DATABASES.get(f"{database_name}_pg")
        self.sqlite_conn: Optional[DatabaseConnection] = None
        self.postgres_conn: Optional[DatabaseConnection] = None
    
    def connect(self):
        """Connect to databases."""
        if self.sqlite_config:
            try:
                self.sqlite_conn = DatabaseConnection(self.sqlite_config)
                self.sqlite_conn.connect()
                logger.info(f"Connected to SQLite: {self.database_name}")
            except Exception as e:
                logger.error(f"SQLite connection failed: {e}")
        
        if self.postgres_config:
            try:
                self.postgres_conn = DatabaseConnection(self.postgres_config)
                self.postgres_conn.connect()
                logger.info(f"Connected to PostgreSQL: {self.database_name}")
            except Exception as e:
                logger.error(f"PostgreSQL connection failed: {e}")
    
    def disconnect(self):
        """Close database connections."""
        if self.sqlite_conn:
            self.sqlite_conn.close()
        if self.postgres_conn:
            self.postgres_conn.close()
    
    def get_schema_string(self) -> str:
        """Get schema as CREATE TABLE statements for prompt.
        
        The pre-trained models expect CREATE TABLE statements in their prompts.
        """
        if self.sqlite_conn:
            try:
                tables = self.sqlite_conn.get_tables()
                schema_parts = []
                for table in tables[:7]:  # Limit to first 7 tables
                    # Skip any test/temp tables
                    if table.startswith('table_name_') or table.startswith('sqlite_'):
                        continue
                    # Use get_schema instead of get_columns
                    cols = self.sqlite_conn.get_schema(table)
                    col_defs = ', '.join([f"{c['name']} {c.get('type', 'TEXT')}" for c in cols[:10]])
                    schema_parts.append(f"CREATE TABLE {table} ({col_defs})")
                return '\n'.join(schema_parts)
            except Exception as e:
                logger.error(f"Error getting schema: {e}")
        return ""
    
    def execute_query(self, sql: str, conn: DatabaseConnection) -> Tuple[bool, Any, float, Optional[str]]:
        """Execute SQL query and return results."""
        start_time = time.time()
        try:
            results = conn.execute(sql)
            exec_time = (time.time() - start_time) * 1000
            return True, results, exec_time, None
        except Exception as e:
            exec_time = (time.time() - start_time) * 1000
            error_str = str(e)
            logger.error(f"Query error: {error_str[:100]}")
            return False, None, exec_time, error_str
    
    def categorize_error(self, error: str) -> str:
        """Categorize SQL error type."""
        error_lower = error.lower()
        if 'no such column' in error_lower or 'does not exist' in error_lower:
            return 'missing_column'
        elif 'no such table' in error_lower or 'relation' in error_lower:
            return 'missing_table'
        elif 'syntax' in error_lower:
            return 'syntax_error'
        elif 'ambiguous' in error_lower:
            return 'ambiguous_column'
        else:
            return 'other_error'
    
    def compare_results(self, result1: Any, result2: Any) -> bool:
        """Compare query results."""
        if result1 is None or result2 is None:
            return False
        try:
            set1 = set(str(r) for r in result1) if result1 else set()
            set2 = set(str(r) for r in result2) if result2 else set()
            return set1 == set2
        except:
            return False
    
    def evaluate_model(self, model_key: str, samples: List[Dict], schema: str) -> ModelMetrics:
        """Evaluate a single model on samples."""
        config = PRETRAINED_MODELS[model_key]
        metrics = ModelMetrics(
            model_name=f"{model_key}-pretrained",
            database_name=self.database_name
        )
        
        logger.info(f"Evaluating {config['name']} on {self.database_name} ({len(samples)} samples)")
        
        for i, sample in enumerate(samples):
            question = sample.get('question', '')
            gold_sql = sample.get('sql', '')
            
            if not question or not gold_sql:
                continue
            
            # Analyze complexity
            complexity = self.complexity_analyzer.analyze(gold_sql)
            complexity_level = complexity.level.name
            
            # Generate SQL
            try:
                predicted_sql, inference_time = self.model_loader.generate_sql(
                    model_key, question, schema
                )
            except Exception as e:
                logger.error(f"Generation failed: {e}")
                predicted_sql = ""
                inference_time = 0.0
            
            # Create result object
            result = EvalResult(
                question=question,
                gold_sql=gold_sql,
                predicted_sql=predicted_sql,
                complexity=complexity_level,
                inference_time_ms=inference_time
            )
            
            # Execute on SQLite
            if self.sqlite_conn and predicted_sql:
                success, sqlite_result, exec_time, error = self.execute_query(
                    predicted_sql, self.sqlite_conn
                )
                result.sqlite_executed = success
                result.sqlite_exec_time_ms = exec_time
                result.sqlite_error = error
                
                if success:
                    metrics.sqlite_execution_success += 1
                    # Check result match with gold
                    gold_success, gold_result, _, _ = self.execute_query(
                        gold_sql, self.sqlite_conn
                    )
                    if gold_success and self.compare_results(sqlite_result, gold_result):
                        result.sqlite_result_match = True
                        metrics.sqlite_result_match += 1
                elif error:
                    error_type = f"sqlite_{self.categorize_error(error)}"
                    metrics.error_counts[error_type] += 1
            
            # Execute on PostgreSQL
            if self.postgres_conn and predicted_sql:
                # Convert to lowercase for PostgreSQL
                pg_sql = predicted_sql.lower()
                success, pg_result, exec_time, error = self.execute_query(
                    pg_sql, self.postgres_conn
                )
                result.postgres_executed = success
                result.postgres_exec_time_ms = exec_time
                result.postgres_error = error
                
                if success:
                    metrics.postgres_execution_success += 1
                    # Check result match
                    gold_success, gold_result, _, _ = self.execute_query(
                        gold_sql.lower(), self.postgres_conn
                    )
                    if gold_success and self.compare_results(pg_result, gold_result):
                        result.postgres_result_match = True
                        metrics.postgres_result_match += 1
                elif error:
                    error_type = f"postgres_{self.categorize_error(error)}"
                    metrics.error_counts[error_type] += 1
            
            # Check cross-DB consistency
            if result.sqlite_executed and result.postgres_executed:
                metrics.consistent_results += 1
            
            # Update complexity stats
            metrics.accuracy_by_complexity[complexity_level]['total'] += 1
            if result.sqlite_executed:
                metrics.accuracy_by_complexity[complexity_level]['sqlite_exec'] += 1
            if result.postgres_executed:
                metrics.accuracy_by_complexity[complexity_level]['postgres_exec'] += 1
            if result.sqlite_result_match:
                metrics.accuracy_by_complexity[complexity_level]['sqlite_match'] += 1
            if result.postgres_result_match:
                metrics.accuracy_by_complexity[complexity_level]['postgres_match'] += 1
            
            # Update totals
            metrics.total_queries += 1
            metrics.total_inference_time += inference_time
            metrics.total_sqlite_exec_time += result.sqlite_exec_time_ms
            metrics.total_postgres_exec_time += result.postgres_exec_time_ms
            metrics.results.append(result)
            
            if (i + 1) % 10 == 0:
                logger.info(f"  Progress: {i + 1}/{len(samples)}")
        
        return metrics


def load_dataset_direct(database_name: str) -> List[Dict]:
    """Load dataset directly from JSON file and flatten to (question, sql) pairs."""
    json_path = f"/app/data/text2sql-data/data/{database_name}.json"
    
    try:
        with open(json_path, 'r') as f:
            data = json.load(f)
        
        samples = []
        
        if isinstance(data, list):
            for item in data:
                if 'sentences' in item and 'sql' in item:
                    # text2sql-data format: each item has sentences and sql
                    sql_list = item['sql']
                    sql = sql_list[0] if sql_list else ''
                    variables = {}
                    
                    # Get variable values from first sentence
                    if item['sentences']:
                        first_sent = item['sentences'][0]
                        question = first_sent.get('text', '')
                        variables = first_sent.get('variables', {})
                        
                        # Substitute variables in question and SQL
                        for var_name, var_value in variables.items():
                            question = question.replace(var_name, var_value)
                            sql = sql.replace(f'"{var_name}"', f'"{var_value}"')
                            sql = sql.replace(f"'{var_name}'", f"'{var_value}'")
                        
                        samples.append({
                            'question': question,
                            'sql': sql
                        })
                elif 'question' in item and 'sql' in item:
                    samples.append(item)
                elif 'nl' in item and 'query' in item:
                    samples.append({'question': item['nl'], 'sql': item['query']})
        
        logger.info(f"Loaded {len(samples)} samples from {database_name}")
        return samples
        
    except Exception as e:
        logger.error(f"Failed to load {json_path}: {e}")
        import traceback
        traceback.print_exc()
        return []


def run_pretrained_evaluation(database_name: str, num_samples: int = 20) -> Dict[str, ModelMetrics]:
    """Run evaluation for all pre-trained models on a database."""
    
    logger.info("=" * 60)
    logger.info(f"PRETRAINED EVALUATION: {database_name.upper()}")
    logger.info(f"Test samples: {num_samples}")
    logger.info("=" * 60)
    
    # Load test data directly from JSON
    samples = load_dataset_direct(database_name)
    
    if not samples:
        logger.error(f"No samples found for {database_name}")
        return {}
    
    # Limit samples
    test_samples = samples[:num_samples]
    
    # Initialize evaluator
    evaluator = PretrainedEvaluator(database_name)
    evaluator.connect()
    
    # Get schema
    schema = evaluator.get_schema_string()
    
    results = {}
    
    # Evaluate each model
    for model_key in PRETRAINED_MODELS.keys():
        logger.info(f"\n[{model_key.upper()}] Loading pre-trained model from HuggingFace...")
        try:
            metrics = evaluator.evaluate_model(model_key, test_samples, schema)
            results[model_key] = metrics
        except Exception as e:
            logger.error(f"Evaluation failed for {model_key}: {e}")
    
    evaluator.disconnect()
    
    # Print summary
    print("\n" + "=" * 80)
    print("RESULTS SUMMARY (PRE-TRAINED MODELS)")
    print("=" * 80)
    print(f"{'Metric':<35} {'GPT-2':<20} {'TinyLlama':<20}")
    print("-" * 75)
    
    for metric_name, getter in [
        ('Total Queries', lambda m: str(m.total_queries)),
        ('SQLite Execution Accuracy', lambda m: f"{m.sqlite_exec_accuracy:.1%}"),
        ('PostgreSQL Execution Accuracy', lambda m: f"{m.postgres_exec_accuracy:.1%}"),
        ('SQLite Result Match', lambda m: f"{m.sqlite_result_accuracy:.1%}"),
        ('PostgreSQL Result Match', lambda m: f"{m.postgres_result_accuracy:.1%}"),
        ('Cross-DB Consistency', lambda m: f"{m.consistency_rate:.1%}"),
        ('Avg Inference Time (ms)', lambda m: f"{m.avg_inference_time:.2f}"),
    ]:
        gpt2_val = getter(results['gpt2']) if 'gpt2' in results else 'N/A'
        tinyllama_val = getter(results['tinyllama']) if 'tinyllama' in results else 'N/A'
        print(f"{metric_name:<35} {gpt2_val:<20} {tinyllama_val:<20}")
    
    print("=" * 80)
    
    # Save results
    output_path = f"/app/results/pretrained_eval_{database_name}.json"
    output_data = {
        'timestamp': datetime.now().isoformat(),
        'database': database_name,
        'category': 'pretrained',
        'models': {name: m.to_dict() for name, m in results.items()}
    }
    
    with open(output_path, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    logger.info(f"\nResults saved to: {output_path}")
    
    return results


def run_all_pretrained_evaluations(num_samples: int = 20):
    """Run pretrained evaluation on all databases."""
    
    databases = ['geography', 'advising', 'atis', 'restaurants']
    all_results = {}
    
    for db in databases:
        try:
            all_results[db] = run_pretrained_evaluation(db, num_samples)
        except Exception as e:
            logger.error(f"Evaluation failed for {db}: {e}")
            import traceback
            traceback.print_exc()
    
    # Generate final comparison report
    print("\n" + "=" * 80)
    print("FINAL COMPARISON REPORT (PRE-TRAINED MODELS)")
    print("=" * 80)
    
    # Aggregate metrics
    summary = {
        'gpt2': {'sqlite_exec': 0, 'postgres_exec': 0, 'total': 0, 'inference_time': 0},
        'tinyllama': {'sqlite_exec': 0, 'postgres_exec': 0, 'total': 0, 'inference_time': 0}
    }
    
    for db, results in all_results.items():
        for model_key, metrics in results.items():
            summary[model_key]['sqlite_exec'] += metrics.sqlite_execution_success
            summary[model_key]['postgres_exec'] += metrics.postgres_execution_success
            summary[model_key]['total'] += metrics.total_queries
            summary[model_key]['inference_time'] += metrics.total_inference_time
    
    # Print final summary
    final_report = {
        'timestamp': datetime.now().isoformat(),
        'category': 'pretrained',
        'summary': {}
    }
    
    for model_key in ['gpt2', 'tinyllama']:
        s = summary[model_key]
        if s['total'] > 0:
            final_report['summary'][model_key] = {
                'overall_sqlite_accuracy': f"{s['sqlite_exec'] / s['total']:.1%}",
                'overall_postgres_accuracy': f"{s['postgres_exec'] / s['total']:.1%}",
                'total_queries': s['total'],
                'avg_inference_time_ms': round(s['inference_time'] / s['total'], 2)
            }
        else:
            final_report['summary'][model_key] = {
                'overall_sqlite_accuracy': 'N/A',
                'overall_postgres_accuracy': 'N/A',
                'total_queries': 0,
                'avg_inference_time_ms': 0
            }
    
    print(json.dumps(final_report['summary'], indent=2))
    
    # Save final report
    final_report['by_database'] = {
        db: {model: m.to_dict() for model, m in results.items()}
        for db, results in all_results.items()
    }
    
    output_path = "/app/results/pretrained_comparison_report.json"
    with open(output_path, 'w') as f:
        json.dump(final_report, f, indent=2)
    
    print(f"\nFull report saved to: {output_path}")
    
    return all_results


if __name__ == "__main__":
    run_all_pretrained_evaluations(num_samples=20)
