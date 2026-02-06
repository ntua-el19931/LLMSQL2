"""
Comprehensive Evaluation Script for Fine-tuned Text-to-SQL Models.

Evaluates GPT-2 and TinyLlama models trained on geography and atis datasets.
Tests both string matching and execution accuracy against SQLite databases.
"""

import json
import torch
import sqlite3
import time
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from collections import defaultdict

# Base paths
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data" / "text2sql-data" / "data"
TRAINING_RESULTS_DIR = BASE_DIR / "training_results"
RESULTS_DIR = BASE_DIR / "results"

# Ensure results directory exists
RESULTS_DIR.mkdir(exist_ok=True)

# Database configurations
DATABASES = {
    "geography": {
        "json_file": DATA_DIR / "geography.json",
        "db_file": DATA_DIR / "geography-db.added-in-2020.sqlite",
        "schema_short": "state(state_name, population, area, capital, density) | city(city_name, population, country_name, state_name)",
        "schema_full": """state(state_name, population, area, capital, density) | city(city_name, population, country_name, state_name) | river(river_name, length, country_name, traverse) | lake(lake_name, area, country_name, state_name) | mountain(mountain_name, mountain_altitude, country_name, state_name) | border_info(state_name, border) | highlow(state_name, highest_elevation, lowest_point, highest_point, lowest_elevation)"""
    },
    "atis": {
        "json_file": DATA_DIR / "atis.json",
        "db_file": DATA_DIR / "atis-db.added-in-2020.sqlite",
        "schema_short": "flight(flight_id, from_airport, to_airport, airline, departure, arrival) | airline(airline_code, airline_name) | airport(airport_code, airport_name, city, state)",
        "schema_full": """flight(flight_id, flight_number, airline_code, from_airport, to_airport, departure_time, arrival_time, stops, connections, meal_code, aircraft_code, fare_basis_code) | airline(airline_code, airline_name) | airport(airport_code, airport_name, airport_location, state_code, country_name, time_zone_code) | aircraft(aircraft_code, aircraft_description, manufacturer, basic_type, propulsion, wide_body, pressurized) | city(city_code, city_name, state_code, country_name, time_zone_code) | fare(fare_id, from_airport, to_airport, fare_basis_code, fare_airline, restriction_code, one_direction_cost, round_trip_cost, round_trip_required) | food_service(meal_code, meal_number, compartment, meal_description) | ground_service(city_code, airport_code, transport_type, ground_fare)"""
    },
    "advising": {
        "json_file": DATA_DIR / "advising.json",
        "db_file": DATA_DIR / "advising-db.added-in-2020.sqlite",
        "schema_short": "course(course_id, name, department, credits) | student(student_id, name, email) | instructor(instructor_id, name)",
        "schema_full": """course(course_id, name, department, number, credits, advisory_requirement, enforced_requirement, description, num_semesters, num_enrolled, has_discussion, has_lab, has_projects, has_exams, num_reviews, clarity_score, easiness_score, helpfulness_score) | student(student_id, lastname, firstname, program_id, declare_major, total_credit, total_gpa, entered_as, admit_term, predicted_graduation_semester, degree, minor, internship) | instructor(instructor_id, name) | offering_instructor(offering_instructor_id, offering_id, instructor_id) | program_course(program_course_id, program_id, course_id, workload, category) | course_offering(offering_id, course_id, semester, section_number, start_time, end_time, monday, tuesday, wednesday, thursday, friday, saturday, sunday, has_final_project, has_final_exam, textbook_isbn, textbook_price, textbook_used_price) | student_record(student_id, course_id, semester, grade, how, transfer_source, earn_credit, repeat_term, test_id) | semester(semester_id, semester, year)"""
    },
    "restaurants": {
        "json_file": DATA_DIR / "restaurants.json",
        "db_file": DATA_DIR / "restaurants-db.added-in-2020.sqlite",
        "schema_short": "restaurant(restaurant_id, name, food_type, rating) | location(restaurant_id, city, county, region) | geographic(city, county, region)",
        "schema_full": """restaurant(restaurant_id, name, food_type, city_name, rating) | location(restaurant_id, house_number, street_name, city_name) | geographic(city_name, county, region)"""
    }
}

# Available trained models
TRAINED_MODELS = {
    "gpt2-geography": {
        "type": "gpt2",
        "path": TRAINING_RESULTS_DIR / "gpt2-geography" / "final",
        "database": "geography"
    },
    "gpt2-atis": {
        "type": "gpt2",
        "path": TRAINING_RESULTS_DIR / "gpt2-atis" / "final",
        "database": "atis"
    },
    "tinyllama-geography": {
        "type": "tinyllama",
        "path": TRAINING_RESULTS_DIR / "tinyllama-geography" / "final",
        "database": "geography"
    },
    "tinyllama-atis": {
        "type": "tinyllama",
        "path": TRAINING_RESULTS_DIR / "tinyllama-atis" / "final",
        "database": "atis"
    }
}


def postprocess_sql(sql: str) -> str:
    """Clean up generated SQL."""
    # Take only first line/statement
    sql = sql.split('\n')[0].strip()
    sql = sql.split(';')[0].strip()
    
    # Remove common artifacts
    for marker in ['###', '```', 'Question:', 'Table:', 'Schema:']:
        if marker in sql:
            sql = sql.split(marker)[0].strip()
    
    return sql


def normalize_sql(sql: str) -> str:
    """Normalize SQL for comparison."""
    import re
    sql = sql.lower().strip()
    sql = re.sub(r'\s+', ' ', sql)
    sql = re.sub(r'\s*,\s*', ', ', sql)
    sql = re.sub(r'\s*\(\s*', '(', sql)
    sql = re.sub(r'\s*\)\s*', ')', sql)
    return sql


def load_test_data(json_file: Path, num_test: int = 50) -> List[Dict]:
    """Load test examples from JSON file."""
    with open(json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    test_examples = []
    # Use last N examples as test set
    for item in data[-num_test:]:
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
    
    return test_examples


def execute_sql(db_path: Path, sql: str) -> Tuple[Optional[List], Optional[str]]:
    """Execute SQL and return results or error."""
    try:
        conn = sqlite3.connect(str(db_path))
        cursor = conn.cursor()
        cursor.execute(sql)
        results = cursor.fetchall()
        conn.close()
        return results, None
    except Exception as e:
        return None, str(e)


def load_gpt2_model(model_path: Path):
    """Load fine-tuned GPT-2 model."""
    print(f"  Loading GPT-2 from {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(str(model_path))
    model = AutoModelForCausalLM.from_pretrained(str(model_path))
    model.eval()
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer


def load_tinyllama_model(model_path: Path):
    """Load fine-tuned TinyLlama model with LoRA adapters."""
    print(f"  Loading TinyLlama base + adapters from {model_path}")
    base_model = AutoModelForCausalLM.from_pretrained(
        'ManthanKulakarni/TinyLlama-1.1B-Text2SQL',
        torch_dtype=torch.float32,
        low_cpu_mem_usage=True
    )
    tokenizer = AutoTokenizer.from_pretrained(str(model_path))
    model = PeftModel.from_pretrained(base_model, str(model_path))
    model.eval()
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer


def generate_sql(model, tokenizer, question: str, schema: str, model_type: str) -> str:
    """Generate SQL for a question."""
    if model_type == "gpt2":
        prompt = f"Table: {schema}\nQuestion: {question}\nSQL:"
        sql_marker = "SQL:"
    else:  # tinyllama
        prompt = f"### Schema:\n{schema}\n\n### Question:\n{question}\n\n### SQL:\n"
        sql_marker = "### SQL:"
    
    inputs = tokenizer(prompt, return_tensors='pt', padding=True)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=100,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id
        )
    
    generated = tokenizer.decode(outputs[0], skip_special_tokens=True)
    raw_sql = generated.split(sql_marker)[-1].strip()
    return postprocess_sql(raw_sql)


def evaluate_model(
    model_name: str,
    model,
    tokenizer,
    model_type: str,
    test_examples: List[Dict],
    db_config: Dict,
    num_samples: int = 20
) -> Dict[str, Any]:
    """Evaluate a single model."""
    print(f"\n{'='*60}")
    print(f"Evaluating: {model_name}")
    print(f"{'='*60}")
    
    # Use full schema for TinyLlama, short for GPT-2
    schema = db_config["schema_full"] if model_type == "tinyllama" else db_config["schema_short"]
    db_path = db_config["db_file"]
    
    results = {
        "model_name": model_name,
        "model_type": model_type,
        "num_samples": min(num_samples, len(test_examples)),
        "timestamp": str(datetime.now()),
        "metrics": {
            "exact_match": 0,
            "partial_match": 0,
            "execution_success": 0,
            "result_match": 0
        },
        "samples": []
    }
    
    samples = test_examples[:num_samples]
    
    for i, ex in enumerate(samples):
        start_time = time.time()
        pred_sql = generate_sql(model, tokenizer, ex['question'], schema, model_type)
        inference_time = time.time() - start_time
        
        # String comparison
        exact_match = normalize_sql(ex['gold_sql']) == normalize_sql(pred_sql)
        
        # Keyword overlap for partial match
        keywords = ['select', 'from', 'where', 'max', 'min', 'count', 'avg', 'sum', 'group', 'order', 'join']
        gold_kw = set(k for k in keywords if k in ex['gold_sql'].lower())
        pred_kw = set(k for k in keywords if k in pred_sql.lower())
        kw_overlap = len(gold_kw & pred_kw) / max(len(gold_kw), 1) if gold_kw else 0
        partial_match = not exact_match and kw_overlap > 0.5
        
        # Execution comparison
        gold_result, gold_error = execute_sql(db_path, ex['gold_sql'])
        pred_result, pred_error = execute_sql(db_path, pred_sql)
        
        execution_success = pred_error is None
        result_match = False
        if gold_result is not None and pred_result is not None:
            # Compare as sets (order independent)
            result_match = sorted(map(str, gold_result)) == sorted(map(str, pred_result))
        
        # Update metrics
        if exact_match:
            results["metrics"]["exact_match"] += 1
        if partial_match:
            results["metrics"]["partial_match"] += 1
        if execution_success:
            results["metrics"]["execution_success"] += 1
        if result_match:
            results["metrics"]["result_match"] += 1
        
        # Store sample
        sample = {
            "question": ex['question'],
            "gold_sql": ex['gold_sql'][:200],
            "pred_sql": pred_sql[:200],
            "exact_match": exact_match,
            "execution_success": execution_success,
            "result_match": result_match,
            "pred_error": pred_error[:100] if pred_error else None,
            "inference_time": round(inference_time, 3)
        }
        results["samples"].append(sample)
        
        # Progress
        if (i + 1) % 5 == 0:
            print(f"  Progress: {i+1}/{num_samples}")
    
    # Calculate percentages
    n = results["num_samples"]
    results["metrics"]["exact_match_pct"] = round(results["metrics"]["exact_match"] / n * 100, 1)
    results["metrics"]["execution_success_pct"] = round(results["metrics"]["execution_success"] / n * 100, 1)
    results["metrics"]["result_match_pct"] = round(results["metrics"]["result_match"] / n * 100, 1)
    
    print(f"\nResults for {model_name}:")
    print(f"  Exact Match:       {results['metrics']['exact_match']}/{n} ({results['metrics']['exact_match_pct']}%)")
    print(f"  Execution Success: {results['metrics']['execution_success']}/{n} ({results['metrics']['execution_success_pct']}%)")
    print(f"  Result Match:      {results['metrics']['result_match']}/{n} ({results['metrics']['result_match_pct']}%)")
    
    return results


def run_all_evaluations(num_samples: int = 20):
    """Run evaluations for all available trained models."""
    print("="*70)
    print("COMPREHENSIVE EVALUATION OF FINE-TUNED TEXT-TO-SQL MODELS")
    print("="*70)
    print(f"\nTimestamp: {datetime.now()}")
    print(f"Samples per model: {num_samples}")
    
    # Check which models are available
    available_models = {}
    for name, config in TRAINED_MODELS.items():
        if config["path"].exists():
            available_models[name] = config
            print(f"✓ Found: {name}")
        else:
            print(f"✗ Missing: {name} (expected at {config['path']})")
    
    if not available_models:
        print("\nNo trained models found! Please run training first.")
        return
    
    all_results = {
        "timestamp": str(datetime.now()),
        "num_samples": num_samples,
        "models": {}
    }
    
    # Evaluate each available model
    for model_name, model_config in available_models.items():
        db_name = model_config["database"]
        db_config = DATABASES[db_name]
        
        # Load test data
        test_examples = load_test_data(db_config["json_file"])
        print(f"\nLoaded {len(test_examples)} test examples for {db_name}")
        
        # Load model
        if model_config["type"] == "gpt2":
            model, tokenizer = load_gpt2_model(model_config["path"])
        else:
            model, tokenizer = load_tinyllama_model(model_config["path"])
        
        # Evaluate
        results = evaluate_model(
            model_name=model_name,
            model=model,
            tokenizer=tokenizer,
            model_type=model_config["type"],
            test_examples=test_examples,
            db_config=db_config,
            num_samples=num_samples
        )
        
        all_results["models"][model_name] = results
        
        # Free memory
        del model, tokenizer
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
    
    # Print summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print(f"\n{'Model':<25} {'Exact Match':<15} {'Exec Success':<15} {'Result Match':<15}")
    print("-"*70)
    
    for model_name, results in all_results["models"].items():
        m = results["metrics"]
        print(f"{model_name:<25} {m['exact_match_pct']:>10.1f}%    {m['execution_success_pct']:>10.1f}%    {m['result_match_pct']:>10.1f}%")
    
    # Save results
    output_file = RESULTS_DIR / f"evaluation_all_models_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to: {output_file}")
    
    return all_results


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Evaluate fine-tuned Text-to-SQL models")
    parser.add_argument("--samples", type=int, default=20, help="Number of test samples per model")
    parser.add_argument("--model", type=str, help="Specific model to evaluate (e.g., gpt2-geography)")
    
    args = parser.parse_args()
    
    if args.model:
        # Evaluate specific model
        if args.model not in TRAINED_MODELS:
            print(f"Unknown model: {args.model}")
            print(f"Available: {list(TRAINED_MODELS.keys())}")
        else:
            config = TRAINED_MODELS[args.model]
            if not config["path"].exists():
                print(f"Model not found at: {config['path']}")
            else:
                db_config = DATABASES[config["database"]]
                test_examples = load_test_data(db_config["json_file"])
                
                if config["type"] == "gpt2":
                    model, tokenizer = load_gpt2_model(config["path"])
                else:
                    model, tokenizer = load_tinyllama_model(config["path"])
                
                results = evaluate_model(
                    model_name=args.model,
                    model=model,
                    tokenizer=tokenizer,
                    model_type=config["type"],
                    test_examples=test_examples,
                    db_config=db_config,
                    num_samples=args.samples
                )
                
                output_file = RESULTS_DIR / f"evaluation_{args.model}.json"
                with open(output_file, 'w') as f:
                    json.dump(results, f, indent=2)
                print(f"\nResults saved to: {output_file}")
    else:
        # Evaluate all models
        run_all_evaluations(num_samples=args.samples)
