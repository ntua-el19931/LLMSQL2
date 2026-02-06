"""
Comprehensive Evaluation System for LLMSQL2.

This module provides:
1. Execution-based evaluation against both SQLite and PostgreSQL
2. Performance metrics (inference time, execution time, memory)
3. Cross-database comparison (same query on SQLite vs PostgreSQL)
4. Formal comparison report generation

Project Requirements:
- Compare GPT-2 vs TinyLlama
- Compare SQLite vs PostgreSQL
- Measure accuracy AND computational efficiency
"""

import json
import time
import traceback
import psutil
import os
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field, asdict
from collections import defaultdict

from .database import DatabaseConnection, DatabaseConfig, DATABASES
from .utils import logger, postprocess_sql, save_json, get_results_dir
from .query_complexity import QueryComplexityAnalyzer, ComplexityLevel


@dataclass
class QueryResult:
    """Result of a single query evaluation."""
    question: str
    gold_sql: str
    predicted_sql: str
    
    # Execution results
    sqlite_result: Optional[List[Tuple]] = None
    postgres_result: Optional[List[Tuple]] = None
    sqlite_error: Optional[str] = None
    postgres_error: Optional[str] = None
    
    # Metrics
    inference_time_ms: float = 0.0
    sqlite_exec_time_ms: float = 0.0
    postgres_exec_time_ms: float = 0.0
    
    # Complexity
    complexity_level: str = "UNKNOWN"
    complexity_score: float = 0.0
    
    # Accuracy flags
    sqlite_executes: bool = False
    postgres_executes: bool = False
    sqlite_matches_gold: bool = False
    postgres_matches_gold: bool = False
    results_consistent: bool = False  # Same result on both DBs
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'question': self.question,
            'gold_sql': self.gold_sql[:100] + '...' if len(self.gold_sql) > 100 else self.gold_sql,
            'predicted_sql': self.predicted_sql[:100] + '...' if len(self.predicted_sql) > 100 else self.predicted_sql,
            'complexity_level': self.complexity_level,
            'complexity_score': round(self.complexity_score, 2),
            'inference_time_ms': round(self.inference_time_ms, 2),
            'sqlite_exec_time_ms': round(self.sqlite_exec_time_ms, 2),
            'postgres_exec_time_ms': round(self.postgres_exec_time_ms, 2),
            'sqlite_executes': self.sqlite_executes,
            'postgres_executes': self.postgres_executes,
            'sqlite_matches_gold': self.sqlite_matches_gold,
            'postgres_matches_gold': self.postgres_matches_gold,
            'results_consistent': self.results_consistent,
            'sqlite_error': self.sqlite_error,
            'postgres_error': self.postgres_error
        }


@dataclass
class ModelMetrics:
    """Aggregated metrics for a single model on a single database."""
    model_name: str
    database_name: str
    total_queries: int = 0
    
    # Execution accuracy
    sqlite_execution_success: int = 0
    postgres_execution_success: int = 0
    
    # Result accuracy (matches gold)
    sqlite_result_match: int = 0
    postgres_result_match: int = 0
    
    # Cross-DB consistency
    consistent_results: int = 0
    
    # Timing (ms)
    total_inference_time: float = 0.0
    total_sqlite_exec_time: float = 0.0
    total_postgres_exec_time: float = 0.0
    
    # Accuracy by complexity level
    accuracy_by_complexity: Dict[str, Dict[str, int]] = field(
        default_factory=lambda: {
            level.name: {'total': 0, 'sqlite_exec': 0, 'postgres_exec': 0, 
                        'sqlite_match': 0, 'postgres_match': 0}
            for level in ComplexityLevel
        }
    )
    
    # Error tracking
    error_counts: Dict[str, int] = field(default_factory=lambda: defaultdict(int))
    
    # Individual results
    results: List[QueryResult] = field(default_factory=list)
    
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
    
    @property
    def avg_inference_time(self) -> float:
        return self.total_inference_time / self.total_queries if self.total_queries else 0.0
    
    @property
    def avg_sqlite_exec_time(self) -> float:
        return self.total_sqlite_exec_time / self.total_queries if self.total_queries else 0.0
    
    @property
    def avg_postgres_exec_time(self) -> float:
        return self.total_postgres_exec_time / self.total_queries if self.total_queries else 0.0
    
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
                'avg_sqlite_exec_time_ms': round(self.avg_sqlite_exec_time, 2),
                'avg_postgres_exec_time_ms': round(self.avg_postgres_exec_time, 2),
                'total_inference_time_s': round(self.total_inference_time / 1000, 2)
            },
            'error_counts': dict(self.error_counts),
            'accuracy_by_complexity': self._accuracy_by_complexity_dict()
        }


class ComprehensiveEvaluator:
    """
    Comprehensive evaluator that tests models against both SQLite and PostgreSQL.
    """
    
    def __init__(self, database_name: str):
        """
        Initialize evaluator for a specific database domain.
        
        Args:
            database_name: One of 'geography', 'advising', 'atis', 'restaurants'
        """
        self.database_name = database_name
        self.sqlite_config = DATABASES.get(f"{database_name}_sqlite")
        self.postgres_config = DATABASES.get(f"{database_name}_pg")
        
        self.sqlite_conn: Optional[DatabaseConnection] = None
        self.postgres_conn: Optional[DatabaseConnection] = None
    
    def connect(self):
        """Establish connections to both databases."""
        if self.sqlite_config:
            try:
                self.sqlite_conn = DatabaseConnection(self.sqlite_config)
                self.sqlite_conn.connect()
                logger.info(f"Connected to SQLite: {self.database_name}")
            except Exception as e:
                logger.warning(f"SQLite connection failed: {e}")
        
        if self.postgres_config:
            try:
                self.postgres_conn = DatabaseConnection(self.postgres_config)
                self.postgres_conn.connect()
                logger.info(f"Connected to PostgreSQL: {self.database_name}")
            except Exception as e:
                logger.warning(f"PostgreSQL connection failed: {e}")
    
    def close(self):
        """Close all database connections."""
        if self.sqlite_conn:
            self.sqlite_conn.close()
        if self.postgres_conn:
            self.postgres_conn.close()
    
    def _execute_with_timing(
        self, 
        conn: DatabaseConnection, 
        sql: str, 
        timeout: float = 5.0
    ) -> Tuple[Optional[List[Tuple]], Optional[str], float]:
        """
        Execute SQL and return (results, error, execution_time_ms).
        """
        start_time = time.perf_counter()
        try:
            results = conn.execute(sql)
            exec_time = (time.perf_counter() - start_time) * 1000
            return results, None, exec_time
        except Exception as e:
            exec_time = (time.perf_counter() - start_time) * 1000
            # Rollback transaction for PostgreSQL to recover from error state
            if conn.config.db_type == "postgresql":
                try:
                    conn.connection.rollback()
                except:
                    pass
            return None, str(e), exec_time
    
    def _compare_results(
        self, 
        results1: Optional[List[Tuple]], 
        results2: Optional[List[Tuple]]
    ) -> bool:
        """Compare two result sets (order-independent)."""
        if results1 is None or results2 is None:
            return False
        
        # Convert to comparable format
        try:
            set1 = set(tuple(str(x) for x in row) for row in results1)
            set2 = set(tuple(str(x) for x in row) for row in results2)
            return set1 == set2
        except:
            # Fallback to sorted string comparison
            return sorted(str(results1)) == sorted(str(results2))
    
    def _categorize_error(self, error: Optional[str]) -> str:
        """Categorize SQL error into types."""
        if error is None:
            return "none"
        
        error_lower = error.lower()
        
        if "syntax" in error_lower or "near" in error_lower:
            return "syntax_error"
        elif "no such table" in error_lower or "relation" in error_lower and "does not exist" in error_lower:
            return "missing_table"
        elif "no such column" in error_lower or "column" in error_lower and "does not exist" in error_lower:
            return "missing_column"
        elif "ambiguous" in error_lower:
            return "ambiguous_column"
        elif "timeout" in error_lower:
            return "timeout"
        else:
            return "other_error"
    
    def _adapt_sql_for_postgres(self, sql: str) -> str:
        """
        Adapt SQL for PostgreSQL compatibility.
        - Convert identifiers to lowercase
        - Handle quote differences
        """
        import re
        
        if not sql:
            return sql
        
        # PostgreSQL uses lowercase for unquoted identifiers
        # The model generates uppercase aliases like STATEalias0
        # We need to convert them to lowercase for PostgreSQL
        
        # Convert alias patterns like STATEalias0, CITYalias1 to lowercase
        sql_pg = re.sub(r'\b([A-Z]+alias\d+)\b', lambda m: m.group(1).lower(), sql)
        
        # Convert table names and column names to lowercase
        # This is a simple approach - more sophisticated would parse SQL
        sql_pg = re.sub(r'\b(STATE|CITY|RIVER|LAKE|MOUNTAIN|BORDER_INFO|HIGHLOW|STUDENT|COURSE|INSTRUCTOR|OFFERING|REGISTRATION|FLIGHT|AIRCRAFT|AIRLINE|AIRPORT|FARE|RESTAURANT|LOCATION|GEOGRAPHIC)\b', 
                        lambda m: m.group(1).lower(), sql_pg, flags=re.IGNORECASE)
        
        # Convert column names
        sql_pg = re.sub(r'\b(STATE_NAME|CITY_NAME|POPULATION|AREA|CAPITAL|DENSITY|COUNTRY_NAME|RIVER_NAME|LENGTH|TRAVERSE|LAKE_NAME|MOUNTAIN_NAME|MOUNTAIN_ALTITUDE|BORDER|HIGHEST_ELEVATION|LOWEST_POINT|HIGHEST_POINT|LOWEST_ELEVATION)\b',
                        lambda m: m.group(1).lower(), sql_pg, flags=re.IGNORECASE)
        
        # Replace double quotes with single quotes for string literals
        # Note: This is a simplistic approach
        sql_pg = sql_pg.replace('""', "'")
        
        return sql_pg
    
    def evaluate_query(
        self,
        question: str,
        gold_sql: str,
        predicted_sql: str,
        inference_time_ms: float = 0.0
    ) -> QueryResult:
        """
        Evaluate a single predicted query against both databases.
        """
        result = QueryResult(
            question=question,
            gold_sql=gold_sql,
            predicted_sql=predicted_sql,
            inference_time_ms=inference_time_ms
        )
        
        gold_sqlite_result = None
        gold_postgres_result = None
        
        # Execute on SQLite
        if self.sqlite_conn:
            # Execute predicted SQL
            result.sqlite_result, result.sqlite_error, result.sqlite_exec_time_ms = \
                self._execute_with_timing(self.sqlite_conn, predicted_sql)
            result.sqlite_executes = result.sqlite_error is None
            
            # Execute gold SQL for comparison
            gold_sqlite_result, gold_err, _ = self._execute_with_timing(self.sqlite_conn, gold_sql)
            
            # Compare results
            if result.sqlite_result is not None and gold_sqlite_result is not None:
                result.sqlite_matches_gold = self._compare_results(
                    result.sqlite_result, gold_sqlite_result
                )
        
        # Execute on PostgreSQL
        if self.postgres_conn:
            # Adapt SQL for PostgreSQL compatibility (lowercase identifiers)
            predicted_sql_pg = self._adapt_sql_for_postgres(predicted_sql)
            gold_sql_pg = self._adapt_sql_for_postgres(gold_sql)
            
            # Execute predicted SQL
            result.postgres_result, result.postgres_error, result.postgres_exec_time_ms = \
                self._execute_with_timing(self.postgres_conn, predicted_sql_pg)
            result.postgres_executes = result.postgres_error is None
            
            # Execute gold SQL for comparison
            gold_postgres_result, gold_err, _ = self._execute_with_timing(self.postgres_conn, gold_sql_pg)
            
            # Compare results
            if result.postgres_result is not None and gold_postgres_result is not None:
                result.postgres_matches_gold = self._compare_results(
                    result.postgres_result, gold_postgres_result
                )
        
        # Check cross-database consistency
        if result.sqlite_result is not None and result.postgres_result is not None:
            result.results_consistent = self._compare_results(
                result.sqlite_result, result.postgres_result
            )
        
        # Analyze complexity of gold SQL
        analyzer = getattr(self, '_complexity_analyzer', None)
        if analyzer is None:
            analyzer = QueryComplexityAnalyzer()
            self._complexity_analyzer = analyzer
        comp = analyzer.analyze(gold_sql)
        result.complexity_level = comp.level.name
        result.complexity_score = comp.score
        
        return result
    
    def evaluate_model(
        self,
        model_name: str,
        model,
        tokenizer,
        test_examples: List[Dict],
        schema: str,
        prompt_template: str = "gpt2",
        num_samples: int = 50
    ) -> ModelMetrics:
        """
        Evaluate a model on test examples.
        
        Args:
            model_name: Name for logging
            model: Loaded model
            tokenizer: Loaded tokenizer
            test_examples: List of {'question': str, 'gold_sql': str}
            schema: Database schema string
            prompt_template: "gpt2" or "tinyllama"
            num_samples: Number of samples to evaluate
        
        Returns:
            ModelMetrics with comprehensive results
        """
        import torch
        
        metrics = ModelMetrics(
            model_name=model_name,
            database_name=self.database_name
        )
        
        examples_subset = test_examples[:num_samples]
        
        logger.info(f"Evaluating {model_name} on {self.database_name} ({len(examples_subset)} samples)")
        
        # Track memory usage
        process = psutil.Process(os.getpid())
        start_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        for i, ex in enumerate(examples_subset):
            # Build prompt based on template
            if prompt_template == "gpt2":
                prompt = f"Table: {schema}\nQuestion: {ex['question']}\nSQL:"
                sql_marker = "SQL:"
            else:  # tinyllama
                prompt = f"### Schema:\n{schema}\n\n### Question:\n{ex['question']}\n\n### SQL:\n"
                sql_marker = "### SQL:"
            
            # Generate prediction
            inputs = tokenizer(prompt, return_tensors='pt', padding=True, truncation=True, max_length=512)
            
            start_time = time.perf_counter()
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=150,
                    do_sample=False,
                    pad_token_id=tokenizer.eos_token_id,
                    num_beams=1  # Greedy for speed
                )
            inference_time_ms = (time.perf_counter() - start_time) * 1000
            
            # Extract and clean SQL
            generated = tokenizer.decode(outputs[0], skip_special_tokens=True)
            raw_sql = generated.split(sql_marker)[-1].strip()
            pred_sql = postprocess_sql(raw_sql)
            
            # Evaluate against databases
            query_result = self.evaluate_query(
                question=ex['question'],
                gold_sql=ex['gold_sql'],
                predicted_sql=pred_sql,
                inference_time_ms=inference_time_ms
            )
            
            # Update metrics
            metrics.total_queries += 1
            metrics.results.append(query_result)
            
            metrics.total_inference_time += query_result.inference_time_ms
            metrics.total_sqlite_exec_time += query_result.sqlite_exec_time_ms
            metrics.total_postgres_exec_time += query_result.postgres_exec_time_ms
            
            if query_result.sqlite_executes:
                metrics.sqlite_execution_success += 1
            else:
                metrics.error_counts[f"sqlite_{self._categorize_error(query_result.sqlite_error)}"] += 1
            
            if query_result.postgres_executes:
                metrics.postgres_execution_success += 1
            else:
                metrics.error_counts[f"postgres_{self._categorize_error(query_result.postgres_error)}"] += 1
            
            if query_result.sqlite_matches_gold:
                metrics.sqlite_result_match += 1
            
            if query_result.postgres_matches_gold:
                metrics.postgres_result_match += 1
            
            if query_result.results_consistent:
                metrics.consistent_results += 1
            
            # Update complexity-based metrics
            lvl = query_result.complexity_level
            stats = metrics.accuracy_by_complexity[lvl]
            stats['total'] += 1
            if query_result.sqlite_executes:
                stats['sqlite_exec'] += 1
            if query_result.postgres_executes:
                stats['postgres_exec'] += 1
            if query_result.sqlite_matches_gold:
                stats['sqlite_match'] += 1
            if query_result.postgres_matches_gold:
                stats['postgres_match'] += 1
            
            # Progress logging
            if (i + 1) % 10 == 0:
                logger.info(f"  Progress: {i+1}/{len(examples_subset)}")
        
        end_memory = process.memory_info().rss / 1024 / 1024  # MB
        logger.info(f"Memory usage: {end_memory - start_memory:.1f} MB delta")
        
        return metrics


def load_test_data(database_name: str, num_samples: int = 50) -> List[Dict]:
    """Load test examples from JSON data files."""
    data_path = Path(f'/app/data/text2sql-data/data/{database_name}.json')
    
    if not data_path.exists():
        raise FileNotFoundError(f"Data file not found: {data_path}")
    
    with open(data_path, 'r') as f:
        data = json.load(f)
    
    test_examples = []
    
    # Use last 20% as test data
    test_start = int(len(data) * 0.8)
    test_data = data[test_start:]
    
    for item in test_data:
        if 'sentences' in item:
            for sent in item.get('sentences', []):
                question = sent.get('text', '')
                sql = item.get('sql', [])
                if isinstance(sql, list):
                    sql = sql[0] if sql else ''
                if question and sql:
                    test_examples.append({
                        'question': question,
                        'gold_sql': sql
                    })
                    
                    if len(test_examples) >= num_samples:
                        return test_examples
    
    return test_examples


def get_schema(database_name: str) -> str:
    """Get schema string for a database."""
    schemas = {
        'geography': (
            'state(state_name, population, area, capital, density) | '
            'city(city_name, population, country_name, state_name) | '
            'river(river_name, length, country_name, traverse) | '
            'lake(lake_name, area, country_name, state_name) | '
            'mountain(mountain_name, mountain_altitude, country_name, state_name) | '
            'border_info(state_name, border) | '
            'highlow(state_name, highest_elevation, lowest_point, highest_point, lowest_elevation)'
        ),
        'advising': (
            'student(student_id, name, email, gpa) | '
            'course(course_id, name, department, credits, description) | '
            'instructor(instructor_id, name, department) | '
            'offering(offering_id, course_id, semester, year, instructor_id) | '
            'registration(student_id, offering_id, grade)'
        ),
        'atis': (
            'flight(flight_id, airline, origin, destination, departure_time, arrival_time) | '
            'aircraft(aircraft_code, aircraft_type, capacity) | '
            'airline(airline_code, airline_name) | '
            'airport(airport_code, airport_name, city, state) | '
            'fare(fare_id, flight_id, fare_class, price)'
        ),
        'restaurants': (
            'restaurant(restaurant_id, name, food_type, rating) | '
            'location(restaurant_id, city, region) | '
            'geographic(city, region, county)'
        )
    }
    return schemas.get(database_name, '')


def run_comprehensive_evaluation(
    database_name: str,
    num_samples: int = 30,
    save_results: bool = True
) -> Dict[str, ModelMetrics]:
    """
    Run comprehensive evaluation on both models for a database.
    
    Returns:
        Dict mapping model name to ModelMetrics
    """
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from peft import PeftModel
    
    results = {}
    
    # Load test data and schema
    test_examples = load_test_data(database_name, num_samples)
    schema = get_schema(database_name)
    
    logger.info(f"\n{'='*60}")
    logger.info(f"COMPREHENSIVE EVALUATION: {database_name.upper()}")
    logger.info(f"Test samples: {len(test_examples)}")
    logger.info(f"{'='*60}\n")
    
    # Initialize evaluator
    evaluator = ComprehensiveEvaluator(database_name)
    evaluator.connect()
    
    try:
        # ===== GPT-2 Evaluation =====
        gpt2_checkpoint = f'/app/results/gpt2-{database_name}/final'
        if Path(gpt2_checkpoint).exists():
            logger.info(f"\n[1/2] Loading GPT-2 from {gpt2_checkpoint}")
            gpt2_tokenizer = AutoTokenizer.from_pretrained(gpt2_checkpoint)
            gpt2_model = AutoModelForCausalLM.from_pretrained(gpt2_checkpoint)
            gpt2_model.eval()
            if gpt2_tokenizer.pad_token is None:
                gpt2_tokenizer.pad_token = gpt2_tokenizer.eos_token
            
            gpt2_metrics = evaluator.evaluate_model(
                model_name=f"gpt2-{database_name}",
                model=gpt2_model,
                tokenizer=gpt2_tokenizer,
                test_examples=test_examples,
                schema=schema,
                prompt_template="gpt2",
                num_samples=num_samples
            )
            results['gpt2'] = gpt2_metrics
            
            # Free memory
            del gpt2_model, gpt2_tokenizer
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
        else:
            logger.warning(f"GPT-2 checkpoint not found: {gpt2_checkpoint}")
        
        # ===== TinyLlama Evaluation =====
        tinyllama_checkpoint = f'/app/results/tinyllama-{database_name}/final'
        if Path(tinyllama_checkpoint).exists():
            logger.info(f"\n[2/2] Loading TinyLlama from {tinyllama_checkpoint}")
            
            base_model = AutoModelForCausalLM.from_pretrained(
                'ManthanKulakarni/TinyLlama-1.1B-Text2SQL',
                torch_dtype=torch.float32,
                low_cpu_mem_usage=True
            )
            tl_tokenizer = AutoTokenizer.from_pretrained(tinyllama_checkpoint)
            tl_model = PeftModel.from_pretrained(base_model, tinyllama_checkpoint)
            tl_model.eval()
            if tl_tokenizer.pad_token is None:
                tl_tokenizer.pad_token = tl_tokenizer.eos_token
            
            tl_metrics = evaluator.evaluate_model(
                model_name=f"tinyllama-{database_name}",
                model=tl_model,
                tokenizer=tl_tokenizer,
                test_examples=test_examples,
                schema=schema,
                prompt_template="tinyllama",
                num_samples=num_samples
            )
            results['tinyllama'] = tl_metrics
            
            del tl_model, tl_tokenizer, base_model
        else:
            logger.warning(f"TinyLlama checkpoint not found: {tinyllama_checkpoint}")
        
    finally:
        evaluator.close()
    
    # Save results
    if save_results and results:
        output = {
            'timestamp': datetime.now().isoformat(),
            'database': database_name,
            'num_samples': num_samples,
            'models': {name: m.to_dict() for name, m in results.items()}
        }
        
        results_file = get_results_dir() / f"comprehensive_eval_{database_name}.json"
        save_json(output, results_file)
        logger.info(f"\nResults saved to: {results_file}")
    
    # Print summary
    print_comparison_table(results)
    
    return results


def print_comparison_table(results: Dict[str, ModelMetrics]):
    """Print a formatted comparison table."""
    print("\n" + "=" * 80)
    print("RESULTS SUMMARY")
    print("=" * 80)
    
    header = f"{'Metric':<30} {'GPT-2':<20} {'TinyLlama':<20}"
    print(header)
    print("-" * 70)
    
    gpt2 = results.get('gpt2')
    tl = results.get('tinyllama')
    
    def get_val(m, attr, fmt=".1%"):
        if m is None:
            return "N/A"
        val = getattr(m, attr)
        if fmt == ".1%":
            return f"{val:.1%}"
        elif fmt == ".2f":
            return f"{val:.2f}"
        return str(val)
    
    rows = [
        ("Total Queries", "total_queries", "d"),
        ("SQLite Execution Accuracy", "sqlite_exec_accuracy", ".1%"),
        ("PostgreSQL Execution Accuracy", "postgres_exec_accuracy", ".1%"),
        ("SQLite Result Match", "sqlite_result_accuracy", ".1%"),
        ("PostgreSQL Result Match", "postgres_result_accuracy", ".1%"),
        ("Cross-DB Consistency", "consistency_rate", ".1%"),
        ("Avg Inference Time (ms)", "avg_inference_time", ".2f"),
        ("Avg SQLite Exec Time (ms)", "avg_sqlite_exec_time", ".2f"),
        ("Avg PostgreSQL Exec Time (ms)", "avg_postgres_exec_time", ".2f"),
    ]
    
    for label, attr, fmt in rows:
        gpt2_val = get_val(gpt2, attr, fmt) if gpt2 else "N/A"
        tl_val = get_val(tl, attr, fmt) if tl else "N/A"
        print(f"{label:<30} {gpt2_val:<20} {tl_val:<20}")
    
    print("=" * 80)


def run_all_evaluations(num_samples: int = 30) -> Dict[str, Dict[str, ModelMetrics]]:
    """Run evaluation on all databases."""
    databases = ['geography', 'advising', 'atis', 'restaurants']
    all_results = {}
    
    for db in databases:
        try:
            all_results[db] = run_comprehensive_evaluation(db, num_samples)
        except Exception as e:
            logger.error(f"Evaluation failed for {db}: {e}")
            traceback.print_exc()
    
    # Generate final report
    generate_final_report(all_results)
    
    return all_results


def generate_final_report(all_results: Dict[str, Dict[str, ModelMetrics]]):
    """Generate final comparison report across all databases."""
    report = {
        'timestamp': datetime.now().isoformat(),
        'summary': {},
        'by_database': {}
    }
    
    # Aggregate totals
    gpt2_totals = {'sqlite_match': 0, 'postgres_match': 0, 'total': 0, 'inference_time': 0}
    tl_totals = {'sqlite_match': 0, 'postgres_match': 0, 'total': 0, 'inference_time': 0}
    
    for db_name, db_results in all_results.items():
        report['by_database'][db_name] = {}
        
        if 'gpt2' in db_results:
            m = db_results['gpt2']
            report['by_database'][db_name]['gpt2'] = m.to_dict()
            gpt2_totals['sqlite_match'] += m.sqlite_result_match
            gpt2_totals['postgres_match'] += m.postgres_result_match
            gpt2_totals['total'] += m.total_queries
            gpt2_totals['inference_time'] += m.total_inference_time
        
        if 'tinyllama' in db_results:
            m = db_results['tinyllama']
            report['by_database'][db_name]['tinyllama'] = m.to_dict()
            tl_totals['sqlite_match'] += m.sqlite_result_match
            tl_totals['postgres_match'] += m.postgres_result_match
            tl_totals['total'] += m.total_queries
            tl_totals['inference_time'] += m.total_inference_time
    
    # Calculate overall summary
    report['summary'] = {
        'gpt2': {
            'overall_sqlite_accuracy': f"{gpt2_totals['sqlite_match']/gpt2_totals['total']:.1%}" if gpt2_totals['total'] else "N/A",
            'overall_postgres_accuracy': f"{gpt2_totals['postgres_match']/gpt2_totals['total']:.1%}" if gpt2_totals['total'] else "N/A",
            'total_queries': gpt2_totals['total'],
            'avg_inference_time_ms': round(gpt2_totals['inference_time']/gpt2_totals['total'], 2) if gpt2_totals['total'] else 0
        },
        'tinyllama': {
            'overall_sqlite_accuracy': f"{tl_totals['sqlite_match']/tl_totals['total']:.1%}" if tl_totals['total'] else "N/A",
            'overall_postgres_accuracy': f"{tl_totals['postgres_match']/tl_totals['total']:.1%}" if tl_totals['total'] else "N/A",
            'total_queries': tl_totals['total'],
            'avg_inference_time_ms': round(tl_totals['inference_time']/tl_totals['total'], 2) if tl_totals['total'] else 0
        }
    }
    
    # Save report
    report_file = get_results_dir() / "final_comparison_report.json"
    save_json(report, report_file)
    
    # Print final summary
    print("\n" + "=" * 80)
    print("FINAL COMPARISON REPORT")
    print("=" * 80)
    print(json.dumps(report['summary'], indent=2))
    print(f"\nFull report saved to: {report_file}")


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Comprehensive evaluation of Text-to-SQL models')
    parser.add_argument('--database', '-d', type=str, default=None,
                        help='Database to evaluate (geography, advising, atis, restaurants). Default: all')
    parser.add_argument('--samples', '-n', type=int, default=30,
                        help='Number of samples per database (default: 30)')
    
    args = parser.parse_args()
    
    if args.database:
        run_comprehensive_evaluation(args.database, args.samples)
    else:
        run_all_evaluations(args.samples)
