"""
Final Report Generator for LLMSQL2 Project.

Generates a comprehensive comparison report in Markdown and JSON formats,
comparing GPT-2 vs TinyLlama across all databases and both SQLite/PostgreSQL.

This fulfills the project requirement:
"Compare how LLMs perform when generating both simple and complex SQL queries"
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass

from .utils import get_results_dir, save_json, logger


@dataclass
class ModelPerformance:
    """Performance metrics for a single model-database combination."""
    model: str
    database: str
    sqlite_exec_accuracy: float = 0.0
    postgres_exec_accuracy: float = 0.0
    sqlite_result_accuracy: float = 0.0
    postgres_result_accuracy: float = 0.0
    avg_inference_time_ms: float = 0.0
    avg_sqlite_exec_time_ms: float = 0.0
    avg_postgres_exec_time_ms: float = 0.0
    total_queries: int = 0
    error_counts: Dict[str, int] = None
    
    def __post_init__(self):
        if self.error_counts is None:
            self.error_counts = {}


def load_evaluation_results() -> Dict[str, Dict[str, ModelPerformance]]:
    """Load all comprehensive evaluation results."""
    results_dir = get_results_dir()
    all_results = {}
    
    for db in ['geography', 'advising', 'atis', 'restaurants']:
        results_file = results_dir / f"comprehensive_eval_{db}.json"
        
        if results_file.exists():
            with open(results_file, 'r') as f:
                data = json.load(f)
            
            all_results[db] = {}
            
            for model_name, metrics in data.get('models', {}).items():
                perf = ModelPerformance(
                    model=model_name,
                    database=db,
                    sqlite_exec_accuracy=float(metrics['accuracy']['sqlite_execution'].rstrip('%')) / 100,
                    postgres_exec_accuracy=float(metrics['accuracy']['postgres_execution'].rstrip('%')) / 100,
                    sqlite_result_accuracy=float(metrics['accuracy']['sqlite_result_match'].rstrip('%')) / 100,
                    postgres_result_accuracy=float(metrics['accuracy']['postgres_result_match'].rstrip('%')) / 100,
                    avg_inference_time_ms=metrics['performance']['avg_inference_time_ms'],
                    avg_sqlite_exec_time_ms=metrics['performance']['avg_sqlite_exec_time_ms'],
                    avg_postgres_exec_time_ms=metrics['performance']['avg_postgres_exec_time_ms'],
                    total_queries=metrics['total_queries'],
                    error_counts=metrics.get('error_counts', {})
                )
                all_results[db][model_name] = perf
    
    return all_results


def calculate_overall_metrics(results: Dict[str, Dict[str, ModelPerformance]]) -> Dict[str, Dict[str, float]]:
    """Calculate overall metrics across all databases."""
    overall = {
        'gpt2': {
            'total_queries': 0,
            'sqlite_exec_correct': 0,
            'postgres_exec_correct': 0,
            'sqlite_result_correct': 0,
            'postgres_result_correct': 0,
            'total_inference_time': 0,
            'total_sqlite_exec_time': 0,
            'total_postgres_exec_time': 0
        },
        'tinyllama': {
            'total_queries': 0,
            'sqlite_exec_correct': 0,
            'postgres_exec_correct': 0,
            'sqlite_result_correct': 0,
            'postgres_result_correct': 0,
            'total_inference_time': 0,
            'total_sqlite_exec_time': 0,
            'total_postgres_exec_time': 0
        }
    }
    
    for db_name, db_results in results.items():
        for model_name, perf in db_results.items():
            m = overall.get(model_name, overall.get('gpt2'))
            m['total_queries'] += perf.total_queries
            m['sqlite_exec_correct'] += int(perf.sqlite_exec_accuracy * perf.total_queries)
            m['postgres_exec_correct'] += int(perf.postgres_exec_accuracy * perf.total_queries)
            m['sqlite_result_correct'] += int(perf.sqlite_result_accuracy * perf.total_queries)
            m['postgres_result_correct'] += int(perf.postgres_result_accuracy * perf.total_queries)
            m['total_inference_time'] += perf.avg_inference_time_ms * perf.total_queries
            m['total_sqlite_exec_time'] += perf.avg_sqlite_exec_time_ms * perf.total_queries
            m['total_postgres_exec_time'] += perf.avg_postgres_exec_time_ms * perf.total_queries
    
    # Calculate percentages
    for model in ['gpt2', 'tinyllama']:
        m = overall[model]
        total = m['total_queries'] if m['total_queries'] > 0 else 1
        m['sqlite_exec_accuracy'] = m['sqlite_exec_correct'] / total
        m['postgres_exec_accuracy'] = m['postgres_exec_correct'] / total
        m['sqlite_result_accuracy'] = m['sqlite_result_correct'] / total
        m['postgres_result_accuracy'] = m['postgres_result_correct'] / total
        m['avg_inference_time_ms'] = m['total_inference_time'] / total
        m['avg_sqlite_exec_time_ms'] = m['total_sqlite_exec_time'] / total
        m['avg_postgres_exec_time_ms'] = m['total_postgres_exec_time'] / total
    
    return overall


def generate_markdown_report(results: Dict[str, Dict[str, ModelPerformance]], overall: Dict[str, Dict[str, float]]) -> str:
    """Generate a Markdown format report."""
    
    report = []
    report.append("# LLMSQL2 - Text-to-SQL Evaluation Report")
    report.append(f"\n**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report.append("\n---\n")
    
    # Executive Summary
    report.append("## Executive Summary")
    report.append("\nThis report compares the performance of **GPT-2** and **TinyLlama** models")
    report.append("for Text-to-SQL generation across multiple databases (geography, advising, atis, restaurants)")
    report.append("and two database backends (SQLite and PostgreSQL).\n")
    
    # Overall Results Table
    report.append("## Overall Results\n")
    report.append("| Metric | GPT-2 | TinyLlama | Winner |")
    report.append("|--------|-------|-----------|--------|")
    
    gpt2 = overall.get('gpt2', {})
    tl = overall.get('tinyllama', {})
    
    def winner(g, t, higher_is_better=True):
        if higher_is_better:
            return "GPT-2" if g > t else ("TinyLlama" if t > g else "Tie")
        else:
            return "GPT-2" if g < t else ("TinyLlama" if t < g else "Tie")
    
    metrics = [
        ("Total Queries", gpt2.get('total_queries', 0), tl.get('total_queries', 0), None),
        ("SQLite Execution Accuracy", f"{gpt2.get('sqlite_exec_accuracy', 0):.1%}", f"{tl.get('sqlite_exec_accuracy', 0):.1%}", 
         winner(gpt2.get('sqlite_exec_accuracy', 0), tl.get('sqlite_exec_accuracy', 0))),
        ("PostgreSQL Execution Accuracy", f"{gpt2.get('postgres_exec_accuracy', 0):.1%}", f"{tl.get('postgres_exec_accuracy', 0):.1%}",
         winner(gpt2.get('postgres_exec_accuracy', 0), tl.get('postgres_exec_accuracy', 0))),
        ("SQLite Result Match", f"{gpt2.get('sqlite_result_accuracy', 0):.1%}", f"{tl.get('sqlite_result_accuracy', 0):.1%}",
         winner(gpt2.get('sqlite_result_accuracy', 0), tl.get('sqlite_result_accuracy', 0))),
        ("PostgreSQL Result Match", f"{gpt2.get('postgres_result_accuracy', 0):.1%}", f"{tl.get('postgres_result_accuracy', 0):.1%}",
         winner(gpt2.get('postgres_result_accuracy', 0), tl.get('postgres_result_accuracy', 0))),
        ("Avg Inference Time (ms)", f"{gpt2.get('avg_inference_time_ms', 0):.0f}", f"{tl.get('avg_inference_time_ms', 0):.0f}",
         winner(gpt2.get('avg_inference_time_ms', 0), tl.get('avg_inference_time_ms', 0), higher_is_better=False)),
    ]
    
    for name, g_val, t_val, w in metrics:
        w_str = w if w else "-"
        report.append(f"| {name} | {g_val} | {t_val} | {w_str} |")
    
    report.append("\n")
    
    # Database-specific Results
    report.append("## Results by Database\n")
    
    for db_name in ['geography', 'advising', 'atis', 'restaurants']:
        db_results = results.get(db_name, {})
        
        if not db_results:
            report.append(f"### {db_name.title()}\n")
            report.append("*No evaluation results available*\n")
            continue
        
        report.append(f"### {db_name.title()}\n")
        report.append("| Metric | GPT-2 | TinyLlama |")
        report.append("|--------|-------|-----------|")
        
        gpt2_perf = db_results.get('gpt2')
        tl_perf = db_results.get('tinyllama')
        
        def get_val(perf, attr, fmt=".1%"):
            if perf is None:
                return "N/A"
            val = getattr(perf, attr, 0)
            if fmt == ".1%":
                return f"{val:.1%}"
            elif fmt == ".0f":
                return f"{val:.0f}"
            elif fmt == ".2f":
                return f"{val:.2f}"
            return str(val)
        
        db_metrics = [
            ("SQLite Exec Accuracy", "sqlite_exec_accuracy", ".1%"),
            ("PostgreSQL Exec Accuracy", "postgres_exec_accuracy", ".1%"),
            ("SQLite Result Match", "sqlite_result_accuracy", ".1%"),
            ("PostgreSQL Result Match", "postgres_result_accuracy", ".1%"),
            ("Avg Inference (ms)", "avg_inference_time_ms", ".0f"),
            ("Avg SQLite Exec (ms)", "avg_sqlite_exec_time_ms", ".2f"),
            ("Avg PostgreSQL Exec (ms)", "avg_postgres_exec_time_ms", ".2f"),
        ]
        
        for name, attr, fmt in db_metrics:
            g_val = get_val(gpt2_perf, attr, fmt)
            t_val = get_val(tl_perf, attr, fmt)
            report.append(f"| {name} | {g_val} | {t_val} |")
        
        report.append("\n")
    
    # Performance Analysis
    report.append("## Performance Analysis\n")
    
    report.append("### Inference Speed\n")
    gpt2_time = gpt2.get('avg_inference_time_ms', 0)
    tl_time = tl.get('avg_inference_time_ms', 0)
    speedup = tl_time / gpt2_time if gpt2_time > 0 else 0
    
    report.append(f"- **GPT-2**: {gpt2_time:.0f}ms average inference time")
    report.append(f"- **TinyLlama**: {tl_time:.0f}ms average inference time")
    report.append(f"- **Speedup**: GPT-2 is **{speedup:.1f}x faster** than TinyLlama\n")
    
    report.append("### Database Backend Comparison\n")
    report.append("| Backend | GPT-2 Exec | TinyLlama Exec | GPT-2 Match | TinyLlama Match |")
    report.append("|---------|------------|----------------|-------------|-----------------|")
    report.append(f"| SQLite | {gpt2.get('sqlite_exec_accuracy', 0):.1%} | {tl.get('sqlite_exec_accuracy', 0):.1%} | {gpt2.get('sqlite_result_accuracy', 0):.1%} | {tl.get('sqlite_result_accuracy', 0):.1%} |")
    report.append(f"| PostgreSQL | {gpt2.get('postgres_exec_accuracy', 0):.1%} | {tl.get('postgres_exec_accuracy', 0):.1%} | {gpt2.get('postgres_result_accuracy', 0):.1%} | {tl.get('postgres_result_accuracy', 0):.1%} |")
    report.append("\n")
    
    # Key Findings
    report.append("## Key Findings\n")
    report.append("1. **Model Size vs Quality Trade-off**")
    report.append("   - GPT-2 (~82MB) is significantly faster but may have lower accuracy")
    report.append("   - TinyLlama (~2GB) provides better quality but requires more compute\n")
    
    report.append("2. **Database Backend Compatibility**")
    report.append("   - SQLite generally has higher execution success rates")
    report.append("   - PostgreSQL requires case-sensitive identifier handling\n")
    
    report.append("3. **Common Error Types**")
    report.append("   - Placeholder values (e.g., `state_name0`) instead of actual values")
    report.append("   - Case sensitivity issues between SQLite and PostgreSQL")
    report.append("   - Incomplete SQL generation (truncated queries)\n")
    
    # Recommendations
    report.append("## Recommendations\n")
    report.append("1. **For Production Use**")
    report.append("   - Use TinyLlama for higher accuracy requirements")
    report.append("   - Use GPT-2 for latency-sensitive applications\n")
    
    report.append("2. **For Better Accuracy**")
    report.append("   - Fine-tune with more epochs")
    report.append("   - Add value extraction from questions")
    report.append("   - Implement SQL validation before execution\n")
    
    report.append("---\n")
    report.append("*Report generated by LLMSQL2 Evaluation Framework*")
    
    return "\n".join(report)


def generate_json_report(results: Dict[str, Dict[str, ModelPerformance]], overall: Dict[str, Dict[str, float]]) -> Dict[str, Any]:
    """Generate JSON format report."""
    return {
        'metadata': {
            'generated_at': datetime.now().isoformat(),
            'project': 'LLMSQL2',
            'models': ['gpt2', 'tinyllama'],
            'databases': ['geography', 'advising', 'atis', 'restaurants'],
            'backends': ['sqlite', 'postgresql']
        },
        'overall': overall,
        'by_database': {
            db: {
                model: {
                    'sqlite_exec_accuracy': perf.sqlite_exec_accuracy,
                    'postgres_exec_accuracy': perf.postgres_exec_accuracy,
                    'sqlite_result_accuracy': perf.sqlite_result_accuracy,
                    'postgres_result_accuracy': perf.postgres_result_accuracy,
                    'avg_inference_time_ms': perf.avg_inference_time_ms,
                    'total_queries': perf.total_queries,
                    'error_counts': perf.error_counts
                }
                for model, perf in db_results.items()
            }
            for db, db_results in results.items()
        }
    }


def generate_report(output_format: str = "both") -> str:
    """
    Generate the final comparison report.
    
    Args:
        output_format: "markdown", "json", or "both"
    
    Returns:
        Path to the generated report(s)
    """
    results_dir = get_results_dir()
    
    # Load results
    results = load_evaluation_results()
    
    if not results:
        logger.warning("No evaluation results found. Run comprehensive_eval first.")
        return None
    
    overall = calculate_overall_metrics(results)
    
    output_files = []
    
    # Generate Markdown report
    if output_format in ["markdown", "both"]:
        md_report = generate_markdown_report(results, overall)
        md_file = results_dir / "FINAL_REPORT.md"
        with open(md_file, 'w') as f:
            f.write(md_report)
        output_files.append(str(md_file))
        logger.info(f"Markdown report saved to: {md_file}")
    
    # Generate JSON report
    if output_format in ["json", "both"]:
        json_report = generate_json_report(results, overall)
        json_file = results_dir / "final_report.json"
        save_json(json_report, json_file)
        output_files.append(str(json_file))
        logger.info(f"JSON report saved to: {json_file}")
    
    return output_files


def main():
    """Generate reports."""
    print("\n" + "="*60)
    print("LLMSQL2 Report Generator")
    print("="*60 + "\n")
    
    files = generate_report("both")
    
    if files:
        print(f"\nReports generated:")
        for f in files:
            print(f"  - {f}")
    else:
        print("No evaluation results found.")
        print("Run comprehensive evaluation first:")
        print("  docker exec llmsql2-app python -m src.comprehensive_eval")


if __name__ == '__main__':
    main()
