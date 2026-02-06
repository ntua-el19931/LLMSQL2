"""
Utility functions for LLMSQL2 project.
"""

import os
import json
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional, Union

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def get_project_root() -> Path:
    """Get the project root directory."""
    return Path(__file__).parent.parent


def get_data_dir() -> Path:
    """Get the data directory."""
    return get_project_root() / "data"


def get_results_dir() -> Path:
    """Get the results directory."""
    results_dir = get_project_root() / "results"
    results_dir.mkdir(exist_ok=True)
    return results_dir


def load_json(file_path: str | Path) -> Dict | List:
    """Load a JSON file."""
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def save_json(data: Dict | List, file_path: str | Path, indent: int = 2) -> None:
    """Save data to a JSON file."""
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=indent, ensure_ascii=False)


def normalize_sql(sql: str) -> str:
    """
    Normalize SQL query for comparison.
    - Lowercase
    - Remove extra whitespace
    - Standardize quotes
    """
    import sqlparse
    
    # Parse and format SQL
    formatted = sqlparse.format(
        sql,
        keyword_case='lower',
        identifier_case='lower',
        strip_comments=True,
        reindent=False
    )
    
    # Remove extra whitespace
    normalized = ' '.join(formatted.split())
    
    return normalized


def postprocess_sql(sql: str) -> str:
    """
    Post-process generated SQL to fix common issues.
    
    Fixes:
    - Double "SELECT SELECT" -> "SELECT"
    - Truncates at first semicolon (removes multiple queries)
    - Removes leading/trailing whitespace
    - Fixes other common duplications
    """
    import re
    
    if not sql:
        return sql
    
    # Truncate at first semicolon to remove multiple queries
    if ';' in sql:
        sql = sql.split(';')[0].strip() + ';'
    
    # Fix double SELECT (case-insensitive)
    sql = re.sub(r'\bSELECT\s+SELECT\b', 'SELECT', sql, flags=re.IGNORECASE)
    
    # Fix other potential double keywords
    double_keywords = ['FROM', 'WHERE', 'AND', 'OR', 'ORDER BY', 'GROUP BY', 'HAVING', 'JOIN', 'LEFT', 'RIGHT', 'INNER', 'OUTER']
    for keyword in double_keywords:
        pattern = rf'\b{keyword}\s+{keyword}\b'
        sql = re.sub(pattern, keyword, sql, flags=re.IGNORECASE)
    
    # Clean up extra whitespace
    sql = ' '.join(sql.split())
    
    return sql.strip()


def extract_sql_from_response(response: str, apply_postprocess: bool = True) -> str:
    """
    Extract SQL query from model response.
    Handles various output formats.
    
    Args:
        response: Raw model output
        apply_postprocess: Whether to apply post-processing fixes (default: True)
    
    Returns:
        Cleaned SQL query
    """
    # Try to find SQL in code blocks
    if "```sql" in response.lower():
        start = response.lower().find("```sql") + 6
        end = response.find("```", start)
        if end != -1:
            sql = response[start:end].strip()
            return postprocess_sql(sql) if apply_postprocess else sql
    
    if "```" in response:
        start = response.find("```") + 3
        end = response.find("```", start)
        if end != -1:
            sql = response[start:end].strip()
            return postprocess_sql(sql) if apply_postprocess else sql
    
    # Look for SELECT, INSERT, UPDATE, DELETE statements
    sql_keywords = ['select', 'insert', 'update', 'delete', 'create', 'drop', 'alter']
    response_lower = response.lower()
    
    for keyword in sql_keywords:
        if keyword in response_lower:
            start_idx = response_lower.find(keyword)
            # Find the end (semicolon or end of string)
            end_idx = response.find(';', start_idx)
            if end_idx != -1:
                sql = response[start_idx:end_idx + 1].strip()
            else:
                sql = response[start_idx:].strip()
            return postprocess_sql(sql) if apply_postprocess else sql
    
    sql = response.strip()
    return postprocess_sql(sql) if apply_postprocess else sql


def format_schema_for_prompt(schema: Union[Dict[str, Any], str]) -> str:
    """
    Format database schema for inclusion in prompts.
    Accepts either a dict or a string schema.
    """
    # If already a string, return as-is
    if isinstance(schema, str):
        return schema
    
    lines = []
    
    for table_name, columns in schema.items():
        col_strs = []
        for col in columns:
            col_str = f"{col['name']} {col['type']}"
            if col.get('primary_key'):
                col_str += " PRIMARY KEY"
            if col.get('foreign_key'):
                col_str += f" REFERENCES {col['foreign_key']}"
            col_strs.append(col_str)
        
        lines.append(f"CREATE TABLE {table_name} (")
        lines.append("  " + ",\n  ".join(col_strs))
        lines.append(");")
        lines.append("")
    
    return "\n".join(lines)


def chunk_list(lst: List, chunk_size: int) -> List[List]:
    """Split a list into chunks of specified size."""
    return [lst[i:i + chunk_size] for i in range(0, len(lst), chunk_size)]
