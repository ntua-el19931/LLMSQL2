"""
Execution Accuracy Evaluation for Text-to-SQL models.

This module tests if generated SQL queries actually execute correctly
and return the same results as gold (expected) SQL queries.
"""

import sqlite3
import json
import time
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
from collections import defaultdict

from .utils import logger, postprocess_sql, save_json, get_results_dir


@dataclass
class ExecutionResult:
    """Result of executing a single SQL query."""
    question: str
    gold_sql: str
    predicted_sql: str
    gold_result: Optional[List[Tuple]] = None
    pred_result: Optional[List[Tuple]] = None
    gold_error: Optional[str] = None
    pred_error: Optional[str] = None
    results_match: bool = False
    pred_executes: bool = False
    inference_time: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'question': self.question,
            'gold_sql': self.gold_sql,
            'predicted_sql': self.predicted_sql,
            'gold_result_count': len(self.gold_result) if self.gold_result else 0,
            'pred_result_count': len(self.pred_result) if self.pred_result else 0,
            'gold_error': self.gold_error,
            'pred_error': self.pred_error,
            'results_match': self.results_match,
            'pred_executes': self.pred_executes,
            'inference_time': self.inference_time
        }


@dataclass
class ExecutionMetrics:
    """Aggregated execution metrics."""
    total: int = 0
    execution_success: int = 0  # Predicted SQL runs without error
    result_match: int = 0       # Results match gold SQL
    gold_errors: int = 0        # Gold SQL had errors (data issue)
    
    results: List[ExecutionResult] = field(default_factory=list)
    error_types: Dict[str, int] = field(default_factory=lambda: defaultdict(int))
    
    @property
    def execution_accuracy(self) -> float:
        """Percentage of predictions that execute without error."""
        if self.total == 0:
            return 0.0
        return self.execution_success / self.total
    
    @property
    def result_accuracy(self) -> float:
        """Percentage of predictions that return correct results."""
        valid_total = self.total - self.gold_errors
        if valid_total == 0:
            return 0.0
        return self.result_match / valid_total
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'total': self.total,
            'execution_success': self.execution_success,
            'execution_accuracy': f"{self.execution_accuracy:.1%}",
            'result_match': self.result_match,
            'result_accuracy': f"{self.result_accuracy:.1%}",
            'gold_errors': self.gold_errors,
            'error_types': dict(self.error_types)
        }
    
    def __repr__(self) -> str:
        return (
            f"ExecutionMetrics(\n"
            f"  total={self.total},\n"
            f"  execution_accuracy={self.execution_accuracy:.1%},\n"
            f"  result_accuracy={self.result_accuracy:.1%},\n"
            f"  gold_errors={self.gold_errors}\n"
            f")"
        )


class SQLExecutor:
    """Execute and compare SQL queries against a SQLite database."""
    
    def __init__(self, db_path: str):
        """
        Initialize with path to SQLite database.
        
        Args:
            db_path: Path to .sqlite file
        """
        self.db_path = db_path
        self.connection = None
        self._connect()
    
    def _connect(self):
        """Establish database connection."""
        try:
            self.connection = sqlite3.connect(self.db_path)
            self.connection.row_factory = sqlite3.Row
            logger.info(f"Connected to database: {self.db_path}")
        except Exception as e:
            logger.error(f"Failed to connect to {self.db_path}: {e}")
            raise
    
    def execute_query(self, sql: str, timeout: float = 5.0) -> Tuple[Optional[List[Tuple]], Optional[str]]:
        """
        Execute a SQL query and return results.
        
        Args:
            sql: SQL query string
            timeout: Maximum execution time in seconds
            
        Returns:
            Tuple of (results, error_message)
        """
        try:
            cursor = self.connection.cursor()
            cursor.execute(sql)
            results = cursor.fetchall()
            # Convert Row objects to tuples for comparison
            results = [tuple(row) for row in results]
            return results, None
        except Exception as e:
            return None, str(e)
    
    def compare_results(
        self, 
        gold_results: List[Tuple], 
        pred_results: List[Tuple],
        order_matters: bool = False
    ) -> bool:
        """
        Compare two result sets.
        
        Args:
            gold_results: Expected results
            pred_results: Predicted results
            order_matters: If False, compare as sets
            
        Returns:
            True if results match
        """
        if gold_results is None or pred_results is None:
            return False
        
        if order_matters:
            return gold_results == pred_results
        else:
            # Compare as multisets (order doesn't matter)
            return sorted(map(str, gold_results)) == sorted(map(str, pred_results))
    
    def evaluate_single(
        self, 
        question: str,
        gold_sql: str, 
        predicted_sql: str,
        inference_time: float = 0.0
    ) -> ExecutionResult:
        """
        Evaluate a single prediction.
        
        Args:
            question: Natural language question
            gold_sql: Ground truth SQL
            predicted_sql: Model-generated SQL
            inference_time: Time taken for inference
            
        Returns:
            ExecutionResult with detailed comparison
        """
        result = ExecutionResult(
            question=question,
            gold_sql=gold_sql,
            predicted_sql=predicted_sql,
            inference_time=inference_time
        )
        
        # Execute gold SQL
        result.gold_result, result.gold_error = self.execute_query(gold_sql)
        
        # Execute predicted SQL
        result.pred_result, result.pred_error = self.execute_query(predicted_sql)
        result.pred_executes = result.pred_error is None
        
        # Compare results if both executed successfully
        if result.gold_result is not None and result.pred_result is not None:
            result.results_match = self.compare_results(
                result.gold_result, 
                result.pred_result
            )
        
        return result
    
    def evaluate_batch(
        self,
        examples: List[Dict[str, str]],
        predictions: List[str]
    ) -> ExecutionMetrics:
        """
        Evaluate a batch of predictions.
        
        Args:
            examples: List of dicts with 'question' and 'gold_sql'
            predictions: List of predicted SQL strings
            
        Returns:
            ExecutionMetrics with aggregated results
        """
        metrics = ExecutionMetrics()
        
        for ex, pred_sql in zip(examples, predictions):
            result = self.evaluate_single(
                question=ex['question'],
                gold_sql=ex['gold_sql'],
                predicted_sql=pred_sql
            )
            
            metrics.total += 1
            metrics.results.append(result)
            
            if result.gold_error:
                metrics.gold_errors += 1
                metrics.error_types['gold_error'] += 1
            
            if result.pred_executes:
                metrics.execution_success += 1
            else:
                # Categorize error
                error_type = self._categorize_error(result.pred_error)
                metrics.error_types[error_type] += 1
            
            if result.results_match:
                metrics.result_match += 1
        
        return metrics
    
    def _categorize_error(self, error: str) -> str:
        """Categorize SQL error into types."""
        if error is None:
            return "none"
        
        error_lower = error.lower()
        
        if "syntax" in error_lower:
            return "syntax_error"
        elif "no such table" in error_lower:
            return "missing_table"
        elif "no such column" in error_lower:
            return "missing_column"
        elif "ambiguous" in error_lower:
            return "ambiguous_column"
        elif "near" in error_lower:
            return "syntax_error"
        else:
            return "other_error"
    
    def close(self):
        """Close database connection."""
        if self.connection:
            self.connection.close()


def run_execution_evaluation(
    model_name: str,
    model,
    tokenizer,
    db_path: str,
    test_examples: List[Dict],
    schema: str,
    prompt_template: str = "gpt2",
    num_samples: int = 20,
    save_results: bool = True
) -> ExecutionMetrics:
    """
    Run full execution accuracy evaluation for a model.
    
    Args:
        model_name: Name for logging
        model: Loaded model
        tokenizer: Loaded tokenizer
        db_path: Path to SQLite database
        test_examples: List of test examples with 'question' and 'gold_sql'
        schema: Database schema string
        prompt_template: "gpt2" or "tinyllama"
        num_samples: Number of samples to evaluate
        save_results: Whether to save results to file
        
    Returns:
        ExecutionMetrics
    """
    import torch
    
    logger.info(f"Running execution evaluation for {model_name}")
    logger.info(f"Database: {db_path}")
    logger.info(f"Samples: {num_samples}")
    
    executor = SQLExecutor(db_path)
    predictions = []
    examples_subset = test_examples[:num_samples]
    
    # Generate predictions
    for i, ex in enumerate(examples_subset):
        if prompt_template == "gpt2":
            prompt = f"Table: {schema}\nQuestion: {ex['question']}\nSQL:"
            sql_marker = "SQL:"
        else:  # tinyllama
            prompt = f"### Schema:\n{schema}\n\n### Question:\n{ex['question']}\n\n### SQL:\n"
            sql_marker = "### SQL:"
        
        inputs = tokenizer(prompt, return_tensors='pt', padding=True)
        
        start_time = time.time()
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=100,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id
            )
        inference_time = time.time() - start_time
        
        generated = tokenizer.decode(outputs[0], skip_special_tokens=True)
        raw_sql = generated.split(sql_marker)[-1].strip()
        pred_sql = postprocess_sql(raw_sql)
        
        predictions.append(pred_sql)
        
        if (i + 1) % 5 == 0:
            logger.info(f"  Progress: {i+1}/{num_samples}")
    
    # Evaluate against database
    metrics = executor.evaluate_batch(examples_subset, predictions)
    executor.close()
    
    # Save results
    if save_results:
        results_file = get_results_dir() / f"execution_eval_{model_name}.json"
        save_json({
            'model': model_name,
            'database': db_path,
            'metrics': metrics.to_dict(),
            'samples': [r.to_dict() for r in metrics.results[:10]]  # Save first 10
        }, results_file)
        logger.info(f"Results saved to {results_file}")
    
    return metrics


def main():
    """Run execution evaluation for both models on geography."""
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from peft import PeftModel
    
    # Load test data
    with open('/app/data/text2sql-data/data/geography.json', 'r') as f:
        data = json.load(f)
    
    test_examples = []
    for item in data[-50:]:
        if 'sentences' in item:
            for sent in item.get('sentences', []):
                question = sent.get('text', '')
                sql = item.get('sql', [])
                if isinstance(sql, list):
                    sql = sql[0] if sql else ''
                if question and sql:
                    test_examples.append({'question': question, 'gold_sql': sql})
    
    # Database path - use the actual database file
    db_path = '/app/data/text2sql-data/data/geography-db.added-in-2020.sqlite'
    
    # Schemas
    schema_short = 'state(state_name, population, area, capital, density) | city(city_name, population, country_name, state_name)'
    schema_full = '''state(state_name, population, area, capital, density) | city(city_name, population, country_name, state_name) | river(river_name, length, country_name, traverse) | lake(lake_name, area, country_name, state_name) | mountain(mountain_name, mountain_altitude, country_name, state_name) | border_info(state_name, border) | highlow(state_name, highest_elevation, lowest_point, highest_point, lowest_elevation)'''
    
    print("="*60)
    print("EXECUTION ACCURACY EVALUATION")
    print("="*60)
    
    # ========== GPT-2 ==========
    print("\n[1/2] Evaluating GPT-2...")
    gpt2_tokenizer = AutoTokenizer.from_pretrained('/app/results/gpt2-geography/final')
    gpt2_model = AutoModelForCausalLM.from_pretrained('/app/results/gpt2-geography/final')
    gpt2_model.eval()
    if gpt2_tokenizer.pad_token is None:
        gpt2_tokenizer.pad_token = gpt2_tokenizer.eos_token
    
    gpt2_metrics = run_execution_evaluation(
        model_name="gpt2-geography",
        model=gpt2_model,
        tokenizer=gpt2_tokenizer,
        db_path=db_path,
        test_examples=test_examples,
        schema=schema_short,
        prompt_template="gpt2",
        num_samples=20
    )
    
    print(f"\nGPT-2 Results:")
    print(f"  Execution Accuracy: {gpt2_metrics.execution_accuracy:.1%}")
    print(f"  Result Accuracy: {gpt2_metrics.result_accuracy:.1%}")
    print(f"  Error Types: {dict(gpt2_metrics.error_types)}")
    
    # Free memory
    del gpt2_model, gpt2_tokenizer
    
    # ========== TinyLlama ==========
    print("\n[2/2] Evaluating TinyLlama...")
    base_model = AutoModelForCausalLM.from_pretrained(
        'ManthanKulakarni/TinyLlama-1.1B-Text2SQL',
        torch_dtype=torch.float32,
        low_cpu_mem_usage=True
    )
    tl_tokenizer = AutoTokenizer.from_pretrained('/app/results/tinyllama-geography/final')
    tl_model = PeftModel.from_pretrained(base_model, '/app/results/tinyllama-geography/final')
    tl_model.eval()
    if tl_tokenizer.pad_token is None:
        tl_tokenizer.pad_token = tl_tokenizer.eos_token
    
    tl_metrics = run_execution_evaluation(
        model_name="tinyllama-geography",
        model=tl_model,
        tokenizer=tl_tokenizer,
        db_path=db_path,
        test_examples=test_examples,
        schema=schema_full,
        prompt_template="tinyllama",
        num_samples=20
    )
    
    print(f"\nTinyLlama Results:")
    print(f"  Execution Accuracy: {tl_metrics.execution_accuracy:.1%}")
    print(f"  Result Accuracy: {tl_metrics.result_accuracy:.1%}")
    print(f"  Error Types: {dict(tl_metrics.error_types)}")
    
    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"{'Model':<20} {'Exec Acc':<15} {'Result Acc':<15}")
    print("-"*50)
    print(f"{'GPT-2':<20} {gpt2_metrics.execution_accuracy:<15.1%} {gpt2_metrics.result_accuracy:<15.1%}")
    print(f"{'TinyLlama':<20} {tl_metrics.execution_accuracy:<15.1%} {tl_metrics.result_accuracy:<15.1%}")


if __name__ == '__main__':
    main()
