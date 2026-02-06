"""
Analyze dataset queries and show complexity distribution.
"""
import json
import sys
import os
import importlib.util

# Load query_complexity directly without going through __init__.py
module_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'src', 'query_complexity.py')
spec = importlib.util.spec_from_file_location("query_complexity", module_path)
query_complexity = importlib.util.module_from_spec(spec)
spec.loader.exec_module(query_complexity)

QueryComplexityAnalyzer = query_complexity.QueryComplexityAnalyzer

def analyze_dataset(name: str, path: str):
    """Analyze a dataset and print complexity breakdown."""
    print('=' * 70)
    print(f'{name.upper()} DATASET ANALYSIS')
    print('=' * 70)
    
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    analyzer = QueryComplexityAnalyzer()
    
    # Extract and analyze queries
    queries = []
    for item in data:
        if 'sql' in item:
            sql_list = item['sql'] if isinstance(item['sql'], list) else [item['sql']]
            queries.extend(sql_list)
    
    print(f'Total queries in dataset: {len(queries)}')
    
    # Get statistics
    stats = analyzer.get_statistics(queries)
    print(f"\nComplexity Distribution:")
    for level, info in stats['level_distribution'].items():
        bar = '#' * int(info['percentage'] / 2)
        print(f"  {level:10s}: {info['count']:4d} ({info['percentage']:5.1f}%) {bar}")
    
    print(f"\nScore Statistics:")
    print(f"  Min: {stats['score_statistics']['min']}")
    print(f"  Max: {stats['score_statistics']['max']}")
    print(f"  Mean: {stats['score_statistics']['mean']}")
    print(f"  Median: {stats['score_statistics']['median']}")
    
    # Show examples from each level
    print(f"\nSample queries by complexity:")
    grouped = analyzer.analyze_batch(queries)
    
    for level in ['SIMPLE', 'MEDIUM', 'COMPLEX', 'ADVANCED']:
        if grouped[level]:
            example = grouped[level][0]
            query_short = example.query[:70].replace('\n', ' ')
            print(f"\n{level}:")
            print(f"  Query: {query_short}...")
            print(f"  Score: {example.score}, Features: JOINs={example.features.num_joins}, "
                  f"Aggs={example.features.num_aggregations}, "
                  f"Subqueries={example.features.num_subqueries}")
    
    return stats

if __name__ == '__main__':
    datasets = [
        ('geography', 'data/text2sql-data/data/geography.json'),
        ('advising', 'data/text2sql-data/data/advising.json'),
        ('atis', 'data/text2sql-data/data/atis.json'),
        ('restaurants', 'data/text2sql-data/data/restaurants.json'),
    ]
    
    all_stats = {}
    for name, path in datasets:
        try:
            all_stats[name] = analyze_dataset(name, path)
            print('\n')
        except Exception as e:
            print(f"Error analyzing {name}: {e}\n")
    
    # Summary comparison
    print('=' * 70)
    print('SUMMARY: COMPLEXITY COMPARISON ACROSS DATASETS')
    print('=' * 70)
    print(f"\n{'Dataset':<12} {'SIMPLE':>10} {'MEDIUM':>10} {'COMPLEX':>10} {'ADVANCED':>10} {'Mean Score':>12}")
    print('-' * 70)
    
    for name, stats in all_stats.items():
        dist = stats['level_distribution']
        print(f"{name:<12} "
              f"{dist['SIMPLE']['percentage']:>9.1f}% "
              f"{dist['MEDIUM']['percentage']:>9.1f}% "
              f"{dist['COMPLEX']['percentage']:>9.1f}% "
              f"{dist['ADVANCED']['percentage']:>9.1f}% "
              f"{stats['score_statistics']['mean']:>12.2f}")
