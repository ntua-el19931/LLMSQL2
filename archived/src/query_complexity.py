"""
Query Complexity Analyzer for LLMSQL2
Categorizes SQL queries into complexity levels based on structural features.

Complexity Levels:
- SIMPLE: Basic SELECT with simple WHERE conditions
- MEDIUM: Queries with JOINs, aggregations, or GROUP BY
- COMPLEX: Subqueries, multiple JOINs, HAVING, nested conditions
- ADVANCED: Deeply nested subqueries, CTEs, complex aggregations
"""

import re
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional
from enum import Enum


class ComplexityLevel(Enum):
    SIMPLE = 1
    MEDIUM = 2
    COMPLEX = 3
    ADVANCED = 4
    
    def __str__(self):
        return self.name


@dataclass
class QueryFeatures:
    """Extracted features from a SQL query."""
    # Basic structure
    num_tables: int = 0
    num_columns: int = 0
    
    # JOINs
    num_joins: int = 0
    join_types: List[str] = field(default_factory=list)
    
    # Aggregations
    num_aggregations: int = 0
    aggregation_functions: List[str] = field(default_factory=list)
    
    # Grouping
    has_group_by: bool = False
    has_having: bool = False
    num_group_columns: int = 0
    
    # Subqueries
    num_subqueries: int = 0
    subquery_depth: int = 0
    
    # Conditions
    num_conditions: int = 0
    has_or_conditions: bool = False
    has_in_clause: bool = False
    has_between: bool = False
    has_like: bool = False
    has_not: bool = False
    
    # Other features
    has_distinct: bool = False
    has_order_by: bool = False
    has_limit: bool = False
    has_union: bool = False
    has_intersect: bool = False
    has_except: bool = False
    has_case_when: bool = False
    has_cte: bool = False  # Common Table Expressions (WITH clause)
    
    # Calculated scores
    raw_score: float = 0.0
    
    def to_dict(self) -> Dict:
        """Convert features to dictionary."""
        return {
            'num_tables': self.num_tables,
            'num_columns': self.num_columns,
            'num_joins': self.num_joins,
            'join_types': self.join_types,
            'num_aggregations': self.num_aggregations,
            'aggregation_functions': self.aggregation_functions,
            'has_group_by': self.has_group_by,
            'has_having': self.has_having,
            'num_group_columns': self.num_group_columns,
            'num_subqueries': self.num_subqueries,
            'subquery_depth': self.subquery_depth,
            'num_conditions': self.num_conditions,
            'has_or_conditions': self.has_or_conditions,
            'has_in_clause': self.has_in_clause,
            'has_between': self.has_between,
            'has_like': self.has_like,
            'has_not': self.has_not,
            'has_distinct': self.has_distinct,
            'has_order_by': self.has_order_by,
            'has_limit': self.has_limit,
            'has_union': self.has_union,
            'has_intersect': self.has_intersect,
            'has_except': self.has_except,
            'has_case_when': self.has_case_when,
            'has_cte': self.has_cte,
            'raw_score': self.raw_score
        }


@dataclass
class ComplexityResult:
    """Result of complexity analysis."""
    query: str
    level: ComplexityLevel
    score: float
    features: QueryFeatures
    explanation: str
    
    def to_dict(self) -> Dict:
        """Convert result to dictionary."""
        return {
            'query': self.query,
            'level': str(self.level),
            'level_value': self.level.value,
            'score': self.score,
            'features': self.features.to_dict(),
            'explanation': self.explanation
        }


class QueryComplexityAnalyzer:
    """
    Analyzes SQL queries and assigns complexity levels.
    
    Scoring weights (configurable):
    - JOIN: 2 points each
    - Aggregation: 1.5 points each
    - GROUP BY: 2 points
    - HAVING: 3 points
    - Subquery: 3 points each + depth bonus
    - OR conditions: 1 point
    - DISTINCT: 1 point
    - ORDER BY: 0.5 points
    - UNION/INTERSECT/EXCEPT: 3 points
    - CASE WHEN: 2 points
    - CTE: 4 points
    """
    
    # Complexity thresholds
    SIMPLE_MAX = 2.0
    MEDIUM_MAX = 6.0
    COMPLEX_MAX = 12.0
    
    # Scoring weights
    WEIGHTS = {
        'join': 2.0,
        'aggregation': 1.5,
        'group_by': 2.0,
        'having': 3.0,
        'subquery': 3.0,
        'subquery_depth_bonus': 1.5,
        'or_condition': 1.0,
        'in_clause': 0.5,
        'between': 0.5,
        'like': 0.5,
        'not': 0.5,
        'distinct': 1.0,
        'order_by': 0.5,
        'limit': 0.0,
        'union': 3.0,
        'intersect': 3.0,
        'except': 3.0,
        'case_when': 2.0,
        'cte': 4.0,
        'multi_table': 1.0,  # per additional table beyond 1
        'condition': 0.25,   # per condition beyond 1
    }
    
    # Aggregation function patterns
    AGG_FUNCTIONS = ['COUNT', 'SUM', 'AVG', 'MIN', 'MAX', 'GROUP_CONCAT', 'STRING_AGG']
    
    # JOIN patterns
    JOIN_PATTERN = re.compile(
        r'\b(INNER\s+JOIN|LEFT\s+(?:OUTER\s+)?JOIN|RIGHT\s+(?:OUTER\s+)?JOIN|'
        r'FULL\s+(?:OUTER\s+)?JOIN|CROSS\s+JOIN|NATURAL\s+JOIN|JOIN)\b',
        re.IGNORECASE
    )
    
    def __init__(self, weights: Optional[Dict[str, float]] = None):
        """
        Initialize analyzer with optional custom weights.
        
        Args:
            weights: Custom scoring weights (merged with defaults)
        """
        self.weights = self.WEIGHTS.copy()
        if weights:
            self.weights.update(weights)
    
    def analyze(self, sql: str) -> ComplexityResult:
        """
        Analyze a SQL query and return complexity result.
        
        Args:
            sql: SQL query string
            
        Returns:
            ComplexityResult with level, score, and features
        """
        # Normalize query
        sql_normalized = self._normalize_sql(sql)
        
        # Extract features
        features = self._extract_features(sql_normalized)
        
        # Calculate score
        score = self._calculate_score(features)
        features.raw_score = score
        
        # Determine level
        level = self._determine_level(score, features)
        
        # Generate explanation
        explanation = self._generate_explanation(features, level)
        
        return ComplexityResult(
            query=sql,
            level=level,
            score=round(score, 2),
            features=features,
            explanation=explanation
        )
    
    def _normalize_sql(self, sql: str) -> str:
        """Normalize SQL for analysis."""
        # Remove extra whitespace
        sql = ' '.join(sql.split())
        # Remove string literals (replace with placeholder)
        sql = re.sub(r"'[^']*'", "'STRING'", sql)
        sql = re.sub(r'"[^"]*"', '"STRING"', sql)
        return sql
    
    def _extract_features(self, sql: str) -> QueryFeatures:
        """Extract structural features from SQL query."""
        features = QueryFeatures()
        sql_upper = sql.upper()
        
        # Count tables (FROM and JOIN clauses)
        features.num_tables = self._count_tables(sql)
        
        # Count selected columns (approximate)
        features.num_columns = self._count_columns(sql)
        
        # Extract JOIN information
        joins = self.JOIN_PATTERN.findall(sql_upper)
        features.num_joins = len(joins)
        features.join_types = [j.strip().upper() for j in joins]
        
        # Extract aggregation functions
        for agg in self.AGG_FUNCTIONS:
            pattern = rf'\b{agg}\s*\('
            matches = re.findall(pattern, sql_upper)
            if matches:
                features.num_aggregations += len(matches)
                features.aggregation_functions.extend([agg] * len(matches))
        
        # GROUP BY
        group_match = re.search(r'\bGROUP\s+BY\b(.+?)(?:HAVING|ORDER|LIMIT|UNION|INTERSECT|EXCEPT|$)', 
                                sql_upper, re.DOTALL)
        if group_match:
            features.has_group_by = True
            # Count columns in GROUP BY
            group_cols = group_match.group(1)
            features.num_group_columns = len([c for c in group_cols.split(',') if c.strip()])
        
        # HAVING
        features.has_having = bool(re.search(r'\bHAVING\b', sql_upper))
        
        # Subqueries - count nested SELECT statements
        subquery_count, max_depth = self._count_subqueries(sql_upper)
        features.num_subqueries = subquery_count
        features.subquery_depth = max_depth
        
        # Conditions
        features.num_conditions = self._count_conditions(sql_upper)
        features.has_or_conditions = bool(re.search(r'\bOR\b', sql_upper))
        features.has_in_clause = bool(re.search(r'\bIN\s*\(', sql_upper))
        features.has_between = bool(re.search(r'\bBETWEEN\b', sql_upper))
        features.has_like = bool(re.search(r'\bLIKE\b', sql_upper))
        features.has_not = bool(re.search(r'\bNOT\b', sql_upper))
        
        # Other features
        features.has_distinct = bool(re.search(r'\bDISTINCT\b', sql_upper))
        features.has_order_by = bool(re.search(r'\bORDER\s+BY\b', sql_upper))
        features.has_limit = bool(re.search(r'\bLIMIT\b', sql_upper))
        features.has_union = bool(re.search(r'\bUNION\b', sql_upper))
        features.has_intersect = bool(re.search(r'\bINTERSECT\b', sql_upper))
        features.has_except = bool(re.search(r'\bEXCEPT\b', sql_upper))
        features.has_case_when = bool(re.search(r'\bCASE\s+WHEN\b', sql_upper))
        features.has_cte = bool(re.search(r'^\s*WITH\b', sql_upper))
        
        return features
    
    def _count_tables(self, sql: str) -> int:
        """Count number of tables referenced in query."""
        sql_upper = sql.upper()
        count = 0
        
        # Count FROM clause tables
        from_match = re.search(r'\bFROM\s+(\w+)', sql_upper)
        if from_match:
            count = 1
        
        # Add JOIN tables
        count += len(self.JOIN_PATTERN.findall(sql_upper))
        
        # Check for comma-separated tables in FROM
        from_clause = re.search(r'\bFROM\s+(.+?)(?:WHERE|GROUP|ORDER|LIMIT|HAVING|$)', 
                                sql_upper, re.DOTALL)
        if from_clause:
            from_text = from_clause.group(1)
            # Remove subqueries before counting commas
            from_text = re.sub(r'\([^)]*\)', '', from_text)
            # Remove JOIN clauses
            from_text = self.JOIN_PATTERN.sub('', from_text)
            # Count remaining commas (additional tables)
            comma_count = from_text.count(',')
            if comma_count > 0:
                count = max(count, comma_count + 1)
        
        return max(count, 1)
    
    def _count_columns(self, sql: str) -> int:
        """Approximate count of selected columns."""
        # Find SELECT clause
        select_match = re.search(r'\bSELECT\s+(.+?)\s+FROM\b', sql, re.IGNORECASE | re.DOTALL)
        if not select_match:
            return 1
        
        select_clause = select_match.group(1)
        
        # Handle SELECT *
        if '*' in select_clause:
            return 0  # Unknown, treat as minimal
        
        # Remove subqueries and functions before counting
        select_clause = re.sub(r'\([^)]*\)', 'FUNC', select_clause)
        
        # Count commas + 1
        return select_clause.count(',') + 1
    
    def _count_subqueries(self, sql: str) -> Tuple[int, int]:
        """
        Count subqueries and maximum nesting depth.
        
        Returns:
            Tuple of (count, max_depth)
        """
        # Count SELECT keywords (minus 1 for main query)
        select_count = len(re.findall(r'\bSELECT\b', sql, re.IGNORECASE))
        subquery_count = max(0, select_count - 1)
        
        # Calculate depth by counting nested parentheses with SELECT
        max_depth = 0
        current_depth = 0
        in_select = False
        
        i = 0
        while i < len(sql):
            if sql[i:i+6].upper() == 'SELECT':
                if in_select:
                    current_depth += 1
                    max_depth = max(max_depth, current_depth)
                in_select = True
                i += 6
            elif sql[i] == '(':
                i += 1
            elif sql[i] == ')':
                if current_depth > 0:
                    current_depth -= 1
                i += 1
            else:
                i += 1
        
        return subquery_count, max_depth
    
    def _count_conditions(self, sql: str) -> int:
        """Count WHERE/HAVING conditions."""
        count = 0
        
        # Count comparison operators
        count += len(re.findall(r'[<>=!]+', sql))
        
        # Count AND/OR
        count += len(re.findall(r'\b(AND|OR)\b', sql, re.IGNORECASE))
        
        # Count LIKE, IN, BETWEEN
        count += len(re.findall(r'\b(LIKE|IN|BETWEEN|IS\s+NULL|IS\s+NOT\s+NULL)\b', 
                                sql, re.IGNORECASE))
        
        return count
    
    def _calculate_score(self, features: QueryFeatures) -> float:
        """Calculate complexity score from features."""
        score = 0.0
        
        # JOINs
        score += features.num_joins * self.weights['join']
        
        # Aggregations
        score += features.num_aggregations * self.weights['aggregation']
        
        # GROUP BY / HAVING
        if features.has_group_by:
            score += self.weights['group_by']
        if features.has_having:
            score += self.weights['having']
        
        # Subqueries (with depth bonus)
        score += features.num_subqueries * self.weights['subquery']
        score += features.subquery_depth * self.weights['subquery_depth_bonus']
        
        # Conditions
        if features.has_or_conditions:
            score += self.weights['or_condition']
        if features.has_in_clause:
            score += self.weights['in_clause']
        if features.has_between:
            score += self.weights['between']
        if features.has_like:
            score += self.weights['like']
        if features.has_not:
            score += self.weights['not']
        
        # Extra conditions beyond basic
        if features.num_conditions > 1:
            score += (features.num_conditions - 1) * self.weights['condition']
        
        # Other features
        if features.has_distinct:
            score += self.weights['distinct']
        if features.has_order_by:
            score += self.weights['order_by']
        if features.has_union:
            score += self.weights['union']
        if features.has_intersect:
            score += self.weights['intersect']
        if features.has_except:
            score += self.weights['except']
        if features.has_case_when:
            score += self.weights['case_when']
        if features.has_cte:
            score += self.weights['cte']
        
        # Multiple tables penalty
        if features.num_tables > 1:
            score += (features.num_tables - 1) * self.weights['multi_table']
        
        return score
    
    def _determine_level(self, score: float, features: QueryFeatures) -> ComplexityLevel:
        """Determine complexity level from score and features."""
        # Override based on specific features
        if features.has_cte or features.subquery_depth >= 2:
            return ComplexityLevel.ADVANCED
        
        if features.num_subqueries >= 2 or (features.has_having and features.num_joins >= 2):
            return ComplexityLevel.COMPLEX
        
        # Score-based determination
        if score <= self.SIMPLE_MAX:
            return ComplexityLevel.SIMPLE
        elif score <= self.MEDIUM_MAX:
            return ComplexityLevel.MEDIUM
        elif score <= self.COMPLEX_MAX:
            return ComplexityLevel.COMPLEX
        else:
            return ComplexityLevel.ADVANCED
    
    def _generate_explanation(self, features: QueryFeatures, level: ComplexityLevel) -> str:
        """Generate human-readable explanation of complexity."""
        factors = []
        
        if features.num_joins > 0:
            join_str = ', '.join(set(features.join_types)) if features.join_types else 'JOIN'
            factors.append(f"{features.num_joins} {join_str}(s)")
        
        if features.num_aggregations > 0:
            agg_str = ', '.join(set(features.aggregation_functions))
            factors.append(f"aggregations ({agg_str})")
        
        if features.has_group_by:
            factors.append(f"GROUP BY ({features.num_group_columns} columns)")
        
        if features.has_having:
            factors.append("HAVING clause")
        
        if features.num_subqueries > 0:
            depth_str = f" (depth {features.subquery_depth})" if features.subquery_depth > 0 else ""
            factors.append(f"{features.num_subqueries} subquery(ies){depth_str}")
        
        if features.has_union:
            factors.append("UNION")
        if features.has_intersect:
            factors.append("INTERSECT")
        if features.has_except:
            factors.append("EXCEPT")
        
        if features.has_case_when:
            factors.append("CASE WHEN")
        
        if features.has_cte:
            factors.append("CTE (WITH clause)")
        
        if features.has_distinct:
            factors.append("DISTINCT")
        
        if features.has_or_conditions:
            factors.append("OR conditions")
        
        if not factors:
            factors.append("basic SELECT with simple conditions")
        
        return f"{level.name}: {', '.join(factors)}"
    
    def analyze_batch(self, queries: List[str]) -> Dict[str, List[ComplexityResult]]:
        """
        Analyze multiple queries and group by complexity level.
        
        Args:
            queries: List of SQL queries
            
        Returns:
            Dictionary mapping level names to lists of results
        """
        results = {level.name: [] for level in ComplexityLevel}
        
        for query in queries:
            try:
                result = self.analyze(query)
                results[result.level.name].append(result)
            except Exception as e:
                # Handle malformed queries gracefully
                print(f"Warning: Could not analyze query: {e}")
                continue
        
        return results
    
    def get_statistics(self, queries: List[str]) -> Dict:
        """
        Get complexity statistics for a set of queries.
        
        Args:
            queries: List of SQL queries
            
        Returns:
            Dictionary with statistics
        """
        results = [self.analyze(q) for q in queries]
        
        level_counts = {level.name: 0 for level in ComplexityLevel}
        scores = []
        
        for result in results:
            level_counts[result.level.name] += 1
            scores.append(result.score)
        
        total = len(results)
        
        return {
            'total_queries': total,
            'level_distribution': {
                level: {
                    'count': count,
                    'percentage': round(count / total * 100, 1) if total > 0 else 0
                }
                for level, count in level_counts.items()
            },
            'score_statistics': {
                'min': round(min(scores), 2) if scores else 0,
                'max': round(max(scores), 2) if scores else 0,
                'mean': round(sum(scores) / len(scores), 2) if scores else 0,
                'median': round(sorted(scores)[len(scores) // 2], 2) if scores else 0
            }
        }


def categorize_dataset_queries(dataset_path: str) -> Dict:
    """
    Load a dataset and categorize all queries by complexity.
    
    Args:
        dataset_path: Path to JSON dataset file
        
    Returns:
        Dictionary with categorized queries and statistics
    """
    import json
    
    with open(dataset_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    analyzer = QueryComplexityAnalyzer()
    
    # Extract SQL queries from dataset
    queries = []
    for item in data:
        if 'sql' in item:
            sql_list = item['sql'] if isinstance(item['sql'], list) else [item['sql']]
            queries.extend(sql_list)
    
    # Analyze all queries
    grouped = analyzer.analyze_batch(queries)
    stats = analyzer.get_statistics(queries)
    
    return {
        'statistics': stats,
        'queries_by_level': {
            level: [r.to_dict() for r in results]
            for level, results in grouped.items()
        }
    }


# Example usage and testing
if __name__ == '__main__':
    analyzer = QueryComplexityAnalyzer()
    
    # Test queries of varying complexity
    test_queries = [
        # SIMPLE
        "SELECT name FROM states",
        "SELECT * FROM city WHERE population > 1000",
        "SELECT state_name FROM state WHERE capital = 'Austin'",
        
        # MEDIUM
        "SELECT COUNT(*) FROM city WHERE state_name = 'Texas'",
        "SELECT state_name, AVG(population) FROM city GROUP BY state_name",
        "SELECT c.name, s.name FROM city c JOIN state s ON c.state_name = s.name",
        
        # COMPLEX
        """SELECT state_name, COUNT(*) as city_count 
           FROM city 
           GROUP BY state_name 
           HAVING COUNT(*) > 5 
           ORDER BY city_count DESC""",
        """SELECT name FROM state 
           WHERE population > (SELECT AVG(population) FROM state)""",
        """SELECT c.name, s.name, r.name 
           FROM city c 
           JOIN state s ON c.state_name = s.name 
           JOIN region r ON s.region = r.name 
           WHERE c.population > 100000""",
        
        # ADVANCED
        """WITH large_states AS (
               SELECT name, population FROM state WHERE population > 5000000
           )
           SELECT ls.name, COUNT(c.name) 
           FROM large_states ls 
           JOIN city c ON c.state_name = ls.name 
           GROUP BY ls.name""",
        """SELECT name FROM state 
           WHERE population > (
               SELECT AVG(population) FROM state WHERE region IN (
                   SELECT name FROM region WHERE country = 'USA'
               )
           )""",
    ]
    
    print("=" * 80)
    print("QUERY COMPLEXITY ANALYSIS")
    print("=" * 80)
    
    for query in test_queries:
        result = analyzer.analyze(query)
        print(f"\nQuery: {query[:60]}...")
        print(f"  Level: {result.level.name} (score: {result.score})")
        print(f"  Explanation: {result.explanation}")
        print(f"  Features: JOINs={result.features.num_joins}, "
              f"Aggs={result.features.num_aggregations}, "
              f"Subqueries={result.features.num_subqueries}")
    
    print("\n" + "=" * 80)
    print("BATCH STATISTICS")
    print("=" * 80)
    
    stats = analyzer.get_statistics(test_queries)
    print(f"\nTotal queries: {stats['total_queries']}")
    print("\nDistribution:")
    for level, info in stats['level_distribution'].items():
        print(f"  {level}: {info['count']} ({info['percentage']}%)")
    
    print(f"\nScore range: {stats['score_statistics']['min']} - {stats['score_statistics']['max']}")
    print(f"Mean score: {stats['score_statistics']['mean']}")
