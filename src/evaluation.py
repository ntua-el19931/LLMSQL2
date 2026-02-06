"""
Evaluation metrics for Text-to-SQL models.
"""

import sqlite3
import re
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from collections import defaultdict

from .utils import normalize_sql, logger, save_json, get_results_dir
from .data_loader import Text2SQLExample


@dataclass
class EvaluationResult:
    """Result of evaluating a single example."""
    question: str
    gold_sql: str
    predicted_sql: str
    exact_match: bool
    execution_match: Optional[bool] = None
    error: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'question': self.question,
            'gold_sql': self.gold_sql,
            'predicted_sql': self.predicted_sql,
            'exact_match': self.exact_match,
            'execution_match': self.execution_match,
            'error': self.error
        }


@dataclass
class EvaluationMetrics:
    """Aggregated evaluation metrics."""
    total_examples: int = 0
    exact_match_count: int = 0
    execution_match_count: int = 0
    error_count: int = 0
    
    # Detailed breakdowns
    results: List[EvaluationResult] = field(default_factory=list)
    errors_by_type: Dict[str, int] = field(default_factory=lambda: defaultdict(int))
    
    @property
    def exact_match_accuracy(self) -> float:
        if self.total_examples == 0:
            return 0.0
        return self.exact_match_count / self.total_examples
    
    @property
    def execution_accuracy(self) -> float:
        if self.total_examples == 0:
            return 0.0
        return self.execution_match_count / self.total_examples
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'total_examples': self.total_examples,
            'exact_match_count': self.exact_match_count,
            'exact_match_accuracy': self.exact_match_accuracy,
            'execution_match_count': self.execution_match_count,
            'execution_accuracy': self.execution_accuracy,
            'error_count': self.error_count,
            'errors_by_type': dict(self.errors_by_type)
        }
    
    def __repr__(self) -> str:
        return (
            f"EvaluationMetrics(\n"
            f"  total_examples={self.total_examples},\n"
            f"  exact_match_accuracy={self.exact_match_accuracy:.2%},\n"
            f"  execution_accuracy={self.execution_accuracy:.2%},\n"
            f"  error_count={self.error_count}\n"
            f")"
        )


class SQLEvaluator:
    """Evaluator for SQL generation tasks."""
    
    def __init__(self, db_connection: Optional[sqlite3.Connection] = None):
        self.db_connection = db_connection
    
    def exact_match(self, gold_sql: str, predicted_sql: str) -> bool:
        """
        Check if predicted SQL exactly matches gold SQL after normalization.
        """
        gold_normalized = normalize_sql(gold_sql)
        pred_normalized = normalize_sql(predicted_sql)
        return gold_normalized == pred_normalized
    
    def execution_match(
        self, 
        gold_sql: str, 
        predicted_sql: str,
        db_connection: Optional[sqlite3.Connection] = None
    ) -> Tuple[bool, Optional[str]]:
        """
        Check if predicted SQL produces the same result as gold SQL.
        
        Returns:
            Tuple of (match_result, error_message)
        """
        conn = db_connection or self.db_connection
        if conn is None:
            return False, "No database connection available"
        
        try:
            # Execute gold SQL
            gold_cursor = conn.cursor()
            gold_cursor.execute(gold_sql)
            gold_result = set(gold_cursor.fetchall())
            
            # Execute predicted SQL
            pred_cursor = conn.cursor()
            pred_cursor.execute(predicted_sql)
            pred_result = set(pred_cursor.fetchall())
            
            return gold_result == pred_result, None
            
        except sqlite3.Error as e:
            return False, str(e)
        except Exception as e:
            return False, f"Unexpected error: {str(e)}"
    
    def evaluate_single(
        self,
        example: Text2SQLExample,
        predicted_sql: str,
        db_connection: Optional[sqlite3.Connection] = None
    ) -> EvaluationResult:
        """Evaluate a single example."""
        # Exact match
        exact = self.exact_match(example.sql, predicted_sql)
        
        # Execution match (if database available)
        conn = db_connection or self.db_connection
        exec_match = None
        error = None
        
        if conn:
            exec_match, error = self.execution_match(
                example.sql, predicted_sql, conn
            )
        
        return EvaluationResult(
            question=example.question,
            gold_sql=example.sql,
            predicted_sql=predicted_sql,
            exact_match=exact,
            execution_match=exec_match,
            error=error
        )
    
    def evaluate_batch(
        self,
        examples: List[Text2SQLExample],
        predictions: List[str],
        db_connections: Optional[Dict[str, sqlite3.Connection]] = None
    ) -> EvaluationMetrics:
        """
        Evaluate a batch of examples.
        
        Args:
            examples: List of Text2SQL examples with gold SQL
            predictions: List of predicted SQL queries
            db_connections: Optional dict mapping db_id to connections
            
        Returns:
            Aggregated evaluation metrics
        """
        metrics = EvaluationMetrics()
        
        for example, pred_sql in zip(examples, predictions):
            # Get database connection if available
            db_conn = None
            if db_connections and example.db_id in db_connections:
                db_conn = db_connections[example.db_id]
            
            result = self.evaluate_single(example, pred_sql, db_conn)
            metrics.results.append(result)
            metrics.total_examples += 1
            
            if result.exact_match:
                metrics.exact_match_count += 1
            
            if result.execution_match:
                metrics.execution_match_count += 1
            
            if result.error:
                metrics.error_count += 1
                # Categorize error
                error_type = self._categorize_error(result.error)
                metrics.errors_by_type[error_type] += 1
        
        return metrics
    
    def _categorize_error(self, error: str) -> str:
        """Categorize SQL error into types."""
        error_lower = error.lower()
        
        if "syntax" in error_lower:
            return "syntax_error"
        elif "no such table" in error_lower:
            return "missing_table"
        elif "no such column" in error_lower:
            return "missing_column"
        elif "ambiguous" in error_lower:
            return "ambiguous_reference"
        else:
            return "other"


def evaluate_model(
    model,
    examples: List[Text2SQLExample],
    db_connections: Optional[Dict[str, sqlite3.Connection]] = None,
    batch_size: int = 8,
    save_results: bool = True,
    results_name: str = "evaluation_results"
) -> EvaluationMetrics:
    """
    Evaluate a Text-to-SQL model on a dataset.
    
    Args:
        model: Text2SQL model with generate_sql_batch method
        examples: List of examples to evaluate
        db_connections: Optional database connections for execution accuracy
        batch_size: Batch size for generation
        save_results: Whether to save results to file
        results_name: Name for results file
        
    Returns:
        Evaluation metrics
    """
    from tqdm import tqdm
    from .utils import chunk_list
    
    logger.info(f"Evaluating model on {len(examples)} examples")
    
    all_predictions = []
    
    # Generate predictions in batches
    batches = chunk_list(examples, batch_size)
    for batch in tqdm(batches, desc="Generating SQL"):
        questions = [ex.question for ex in batch]
        schemas = [ex.schema for ex in batch]
        
        predictions = model.generate_sql_batch(questions, schemas)
        all_predictions.extend(predictions)
    
    # Evaluate
    evaluator = SQLEvaluator()
    metrics = evaluator.evaluate_batch(examples, all_predictions, db_connections)
    
    # Save results
    if save_results:
        results_file = get_results_dir() / f"{results_name}.json"
        save_json({
            'metrics': metrics.to_dict(),
            'results': [r.to_dict() for r in metrics.results]
        }, results_file)
        logger.info(f"Results saved to {results_file}")
    
    return metrics


def compare_models(
    models: Dict[str, Any],
    examples: List[Text2SQLExample],
    **kwargs
) -> Dict[str, EvaluationMetrics]:
    """
    Compare multiple models on the same dataset.
    
    Args:
        models: Dict mapping model names to model instances
        examples: Dataset examples
        
    Returns:
        Dict mapping model names to their metrics
    """
    results = {}
    
    for model_name, model in models.items():
        logger.info(f"Evaluating {model_name}...")
        metrics = evaluate_model(
            model, 
            examples, 
            results_name=f"eval_{model_name}",
            **kwargs
        )
        results[model_name] = metrics
        logger.info(f"{model_name}: {metrics}")
    
    return results
