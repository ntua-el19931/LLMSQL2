"""
SQL Enhancement Module - Improves model output quality.

Key features:
1. Value extraction: Replaces placeholders with actual values from questions
2. Schema-aware validation: Ensures column/table names are valid
3. Query optimization: Fixes common SQL issues
"""

import re
from typing import Dict, List, Optional, Set, Tuple
from dataclasses import dataclass


# Known entities for each database
KNOWN_ENTITIES = {
    "geography": {
        "states": [
            "alabama", "alaska", "arizona", "arkansas", "california", "colorado",
            "connecticut", "delaware", "florida", "georgia", "hawaii", "idaho",
            "illinois", "indiana", "iowa", "kansas", "kentucky", "louisiana",
            "maine", "maryland", "massachusetts", "michigan", "minnesota",
            "mississippi", "missouri", "montana", "nebraska", "nevada",
            "new hampshire", "new jersey", "new mexico", "new york",
            "north carolina", "north dakota", "ohio", "oklahoma", "oregon",
            "pennsylvania", "rhode island", "south carolina", "south dakota",
            "tennessee", "texas", "utah", "vermont", "virginia", "washington",
            "west virginia", "wisconsin", "wyoming"
        ],
        "cities": [
            "new york", "los angeles", "chicago", "houston", "phoenix",
            "philadelphia", "san antonio", "san diego", "dallas", "san jose",
            "austin", "jacksonville", "fort worth", "columbus", "charlotte",
            "seattle", "denver", "boston", "detroit", "portland", "miami",
            "las vegas", "atlanta", "minneapolis", "sacramento", "tucson"
        ],
        "rivers": [
            "mississippi", "missouri", "colorado", "rio grande", "columbia",
            "ohio", "arkansas", "red", "snake", "tennessee", "hudson",
            "potomac", "delaware", "susquehanna", "connecticut"
        ],
        "mountains": [
            "mount mckinley", "mount whitney", "mount rainier", "mount elbert",
            "mount hood", "mount shasta", "pikes peak", "mount washington"
        ]
    },
    "advising": {
        "semesters": ["fall", "spring", "summer", "winter"],
        "departments": ["cs", "math", "physics", "chemistry", "biology", "english"],
        "grades": ["a", "b", "c", "d", "f", "a+", "a-", "b+", "b-", "c+", "c-"]
    },
    "atis": {
        "airlines": [
            "american", "united", "delta", "southwest", "jetblue", "alaska",
            "spirit", "frontier", "hawaiian"
        ],
        "cities": [
            "new york", "los angeles", "chicago", "dallas", "denver", "atlanta",
            "san francisco", "seattle", "boston", "miami", "phoenix", "las vegas",
            "detroit", "minneapolis", "philadelphia", "houston", "orlando"
        ],
        "times": ["morning", "afternoon", "evening", "night"]
    },
    "restaurants": {
        "food_types": [
            "italian", "chinese", "mexican", "japanese", "indian", "thai",
            "french", "american", "greek", "korean", "vietnamese", "spanish"
        ],
        "cities": [
            "san francisco", "los angeles", "new york", "chicago", "boston",
            "seattle", "portland", "denver", "austin", "miami"
        ]
    }
}


@dataclass
class ExtractionResult:
    """Result of entity extraction from a question."""
    original_question: str
    extracted_values: Dict[str, List[str]]
    confidence: float


def extract_entities(question: str, database: str = "geography") -> ExtractionResult:
    """
    Extract named entities from a natural language question.
    
    Args:
        question: Natural language question
        database: Database context for entity types
        
    Returns:
        ExtractionResult with extracted values
    """
    question_lower = question.lower()
    entities = KNOWN_ENTITIES.get(database, {})
    extracted = {}
    
    for entity_type, values in entities.items():
        found = []
        for value in values:
            # Check for exact match or word boundary match
            pattern = r'\b' + re.escape(value) + r'\b'
            if re.search(pattern, question_lower):
                found.append(value)
        if found:
            extracted[entity_type] = found
    
    # Calculate confidence based on extraction success
    confidence = min(1.0, len(extracted) / 2) if extracted else 0.0
    
    return ExtractionResult(
        original_question=question,
        extracted_values=extracted,
        confidence=confidence
    )


def extract_numbers(question: str) -> List[str]:
    """Extract numeric values from question."""
    # Match integers and decimals
    numbers = re.findall(r'\b\d+(?:\.\d+)?\b', question)
    return numbers


def extract_quoted_values(question: str) -> List[str]:
    """Extract values in quotes from question."""
    quoted = re.findall(r'["\']([^"\']+)["\']', question)
    return quoted


def replace_placeholders(
    sql: str, 
    question: str, 
    database: str = "geography"
) -> str:
    """
    Replace placeholder values in SQL with actual values from the question.
    
    Args:
        sql: Generated SQL with placeholders like "state_name0"
        question: Original natural language question
        database: Database context
        
    Returns:
        SQL with placeholders replaced by actual values
    """
    # Extract entities from question
    extraction = extract_entities(question, database)
    
    # Define placeholder patterns and their corresponding entity types
    placeholder_mappings = {
        "geography": {
            r'state_name\d+': 'states',
            r'city_name\d+': 'cities', 
            r'river_name\d+': 'rivers',
            r'mountain_name\d+': 'mountains',
            r'lake_name\d+': 'states',  # Lakes are often referenced by state
        },
        "advising": {
            r'semester\d+': 'semesters',
            r'department\d+': 'departments',
            r'grade\d+': 'grades',
        },
        "atis": {
            r'airline\d+': 'airlines',
            r'city_name\d+': 'cities',
            r'from_city\d+': 'cities',
            r'to_city\d+': 'cities',
        },
        "restaurants": {
            r'food_type\d+': 'food_types',
            r'city_name\d+': 'cities',
        }
    }
    
    mappings = placeholder_mappings.get(database, {})
    result_sql = sql
    
    for pattern, entity_type in mappings.items():
        # Find all placeholders matching this pattern
        placeholders = re.findall(f'"{pattern}"', result_sql, re.IGNORECASE)
        
        if placeholders and entity_type in extraction.extracted_values:
            values = extraction.extracted_values[entity_type]
            for i, placeholder in enumerate(placeholders):
                if i < len(values):
                    # Replace with actual value
                    result_sql = result_sql.replace(
                        placeholder, 
                        f'"{values[i]}"',
                        1  # Replace only first occurrence
                    )
    
    return result_sql


def fix_table_case(sql: str, database: str = "geography") -> str:
    """
    Fix table and column name casing to match actual database schema.
    SQLite is case-insensitive but consistency helps.
    
    Args:
        sql: SQL query
        database: Database context
        
    Returns:
        SQL with corrected casing
    """
    # Schema definitions (lowercase as in actual DB)
    schemas = {
        "geography": {
            "tables": ["state", "city", "river", "lake", "mountain", "border_info", "highlow"],
            "columns": ["state_name", "city_name", "river_name", "lake_name", "mountain_name",
                       "population", "area", "capital", "density", "country_name", "length",
                       "traverse", "mountain_altitude", "border", "highest_elevation",
                       "lowest_elevation", "highest_point", "lowest_point"]
        }
    }
    
    schema = schemas.get(database, {})
    result = sql
    
    # Fix table names (convert UPPERCASE to lowercase)
    for table in schema.get("tables", []):
        # Match table name with word boundaries
        result = re.sub(
            rf'\b{table.upper()}\b',
            table,
            result
        )
        # Also handle mixed case like State, City
        result = re.sub(
            rf'\b{table.capitalize()}\b',
            table,
            result
        )
    
    return result


def fix_alias_references(sql: str) -> str:
    """
    Fix common alias issues like STATEalias0 when table is lowercase.
    
    Args:
        sql: SQL query
        
    Returns:
        SQL with fixed alias references
    """
    # Pattern: UPPERCASE_TABLE AS TABLEalias0 -> lowercase_table AS t0
    # For now, just ensure consistency
    
    # Remove redundant aliases for simple queries
    # e.g., "FROM STATE AS STATEalias0 WHERE STATEalias0.col" 
    # -> "FROM state WHERE state.col"
    
    result = sql
    
    # Common simplifications
    alias_patterns = [
        (r'\bSTATE\s+AS\s+STATEalias(\d+)', r'state'),
        (r'\bCITY\s+AS\s+CITYalias(\d+)', r'city'),
        (r'\bRIVER\s+AS\s+RIVERalias(\d+)', r'river'),
        (r'\bSTATEalias(\d+)\.', r'state.'),
        (r'\bCITYalias(\d+)\.', r'city.'),
        (r'\bRIVERalias(\d+)\.', r'river.'),
    ]
    
    for pattern, replacement in alias_patterns:
        result = re.sub(pattern, replacement, result, flags=re.IGNORECASE)
    
    return result


def enhance_sql(
    sql: str,
    question: str,
    database: str = "geography",
    fix_placeholders: bool = True,
    fix_casing: bool = True,
    simplify_aliases: bool = False
) -> str:
    """
    Apply all enhancements to generated SQL.
    
    Args:
        sql: Raw generated SQL
        question: Original question
        database: Database context
        fix_placeholders: Replace placeholders with extracted values
        fix_casing: Fix table/column casing
        simplify_aliases: Simplify alias references
        
    Returns:
        Enhanced SQL query
    """
    result = sql
    
    if fix_placeholders:
        result = replace_placeholders(result, question, database)
    
    if fix_casing:
        result = fix_table_case(result, database)
    
    if simplify_aliases:
        result = fix_alias_references(result)
    
    return result


def optimize_generation_params(model_type: str = "gpt2") -> Dict:
    """
    Get optimized generation parameters for each model type.
    
    Args:
        model_type: "gpt2" or "tinyllama"
        
    Returns:
        Dict of generation parameters
    """
    if model_type == "gpt2":
        return {
            "max_new_tokens": 80,
            "num_beams": 3,  # Beam search for better quality
            "do_sample": False,
            "early_stopping": True,
            "no_repeat_ngram_size": 3,  # Prevent repetition
            "length_penalty": 1.0,
        }
    else:  # tinyllama
        return {
            "max_new_tokens": 100,
            "num_beams": 2,
            "do_sample": False,
            "early_stopping": True,
            "no_repeat_ngram_size": 2,
            "repetition_penalty": 1.1,  # Slight penalty for repetition
        }


# Test function
def test_enhancement():
    """Test the enhancement functions."""
    test_cases = [
        {
            "question": "What is the capital of Texas?",
            "sql": 'SELECT state.capital FROM state WHERE state.state_name = "state_name0";',
            "database": "geography"
        },
        {
            "question": "How many people live in California?",
            "sql": 'SELECT state.population FROM state WHERE state.state_name = "state_name0";',
            "database": "geography"
        },
        {
            "question": "What rivers flow through Colorado?",
            "sql": 'SELECT river.river_name FROM river WHERE river.traverse = "state_name0";',
            "database": "geography"
        }
    ]
    
    print("SQL Enhancement Test")
    print("=" * 60)
    
    for tc in test_cases:
        print(f"\nQuestion: {tc['question']}")
        print(f"Original: {tc['sql']}")
        enhanced = enhance_sql(tc['sql'], tc['question'], tc['database'])
        print(f"Enhanced: {enhanced}")


if __name__ == "__main__":
    test_enhancement()
