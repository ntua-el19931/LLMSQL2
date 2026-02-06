# LLMSQL2 Source Package
"""
LLMSQL2: Text-to-SQL Evaluation Framework
Supports local models: TinyLlama-1.1B-Text2SQL and GPT-2-Text2SQL
"""

from .data_loader import Text2SQLDataLoader
from .model_inference import (
    Text2SQLModel, 
    GPT2Text2SQLModel,
    load_model, 
    load_gpt2_model,
    list_available_models
)
from .evaluation import SQLEvaluator
from .database import (
    DatabaseManager,
    DatabaseConnection,
    get_database,
    list_databases,
    execute_query,
    DATABASES
)
from .utils import logger, normalize_sql, extract_sql_from_response
from .comprehensive_eval import (
    ComprehensiveEvaluator,
    run_comprehensive_evaluation,
    run_all_evaluations,
    generate_final_report
)
from .inference_api import Text2SQLInference, SQLResult, SCHEMAS
from .report_generator import generate_report
from .query_complexity import (
    QueryComplexityAnalyzer,
    ComplexityLevel,
    ComplexityResult,
    QueryFeatures,
    categorize_dataset_queries
)

__all__ = [
    "Text2SQLDataLoader",
    "Text2SQLModel",
    "GPT2Text2SQLModel",
    "load_model",
    "load_gpt2_model",
    "list_available_models",
    "SQLEvaluator",
    "DatabaseManager",
    "DatabaseConnection",
    "get_database",
    "list_databases",
    "execute_query",
    "DATABASES",
    "logger",
    "normalize_sql",
    "extract_sql_from_response",
    # Comprehensive Evaluation
    "ComprehensiveEvaluator",
    "run_comprehensive_evaluation",
    "run_all_evaluations",
    "generate_final_report",
    # Inference API
    "Text2SQLInference",
    "SQLResult",
    "SCHEMAS",
    # Report Generator
    "generate_report",
    # Query Complexity Analyzer
    "QueryComplexityAnalyzer",
    "ComplexityLevel",
    "ComplexityResult",
    "QueryFeatures",
    "categorize_dataset_queries",
]