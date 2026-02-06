"""
TinyLlama Inference Tuning - Test different configurations to improve results.
"""

import json
import torch
import sqlite3
import time
from pathlib import Path
from datetime import datetime
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

# Docker paths
DATA_DIR = Path("/app/data/text2sql-data/data")
TRAINING_RESULTS_DIR = Path("/app/training_results")
RESULTS_DIR = Path("/app/results")

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
        "schema_short": "flight(flight_id, from_airport, to_airport, airline_code, departure_time, arrival_time) | airline(airline_code, airline_name) | airport(airport_code, airport_name, city) | city(city_code, city_name, state_code)",
        "schema_full": """flight(flight_id, flight_number, airline_code, from_airport, to_airport, departure_time, arrival_time, stops, connections, meal_code, aircraft_code) | airline(airline_code, airline_name) | airport(airport_code, airport_name, airport_location, state_code, country_name) | aircraft(aircraft_code, aircraft_description, manufacturer) | city(city_code, city_name, state_code, country_name) | fare(fare_id, from_airport, to_airport, fare_basis_code, fare_airline, one_direction_cost, round_trip_cost)"""
    }
}

# Few-shot examples for each database
FEW_SHOT_EXAMPLES = {
    "geography": [
        ("what is the capital of texas", "SELECT state.capital FROM state WHERE state.state_name = 'texas'"),
        ("how many rivers are there", "SELECT COUNT(river.river_name) FROM river"),
        ("what is the largest city", "SELECT city.city_name FROM city ORDER BY city.population DESC LIMIT 1"),
    ],
    "atis": [
        ("show me flights from boston to denver", "SELECT flight.flight_id FROM flight, airport AS a1, airport AS a2, city AS c1, city AS c2 WHERE flight.from_airport = a1.airport_code AND flight.to_airport = a2.airport_code AND a1.city_code = c1.city_code AND a2.city_code = c2.city_code AND c1.city_name = 'boston' AND c2.city_name = 'denver'"),
        ("what airlines fly to new york", "SELECT DISTINCT airline.airline_name FROM airline, flight, airport, city WHERE flight.airline_code = airline.airline_code AND flight.to_airport = airport.airport_code AND airport.city_code = city.city_code AND city.city_name = 'new york'"),
    ]
}


def postprocess_sql(sql):
    """Clean up generated SQL."""
    sql = sql.split('\n')[0].strip()
    sql = sql.split(';')[0].strip()
    for marker in ['###', '```', 'Question:', 'Table:', 'Schema:', 'Example']:
        if marker in sql:
            sql = sql.split(marker)[0].strip()
    # Remove trailing incomplete parts
    if sql.count('(') > sql.count(')'):
        # Try to balance parentheses
        diff = sql.count('(') - sql.count(')')
        sql = sql + ')' * diff
    return sql


def normalize_sql(sql):
    import re
    sql = sql.lower().strip()
    sql = re.sub(r'\s+', ' ', sql)
    return sql


def load_test_data(json_file, num_test=50):
    with open(json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    test_examples = []
    for item in data[-num_test:]:
        if 'sentences' in item:
            for sent in item.get('sentences', []):
                question = sent.get('text', '')
                sql = item.get('sql', [])
                if isinstance(sql, list):
                    sql = sql[0] if sql else ''
                if question and sql:
                    test_examples.append({'question': question, 'gold_sql': sql})
    return test_examples


def execute_sql(db_path, sql):
    try:
        conn = sqlite3.connect(str(db_path))
        cursor = conn.cursor()
        cursor.execute(sql)
        results = cursor.fetchall()
        conn.close()
        return results, None
    except Exception as e:
        return None, str(e)


def load_tinyllama_model(model_path):
    print(f"  Loading TinyLlama from {model_path}")
    base_model = AutoModelForCausalLM.from_pretrained(
        'ManthanKulakarni/TinyLlama-1.1B-Text2SQL',
        torch_dtype=torch.float32,
        low_cpu_mem_usage=True
    )
    tokenizer = AutoTokenizer.from_pretrained('ManthanKulakarni/TinyLlama-1.1B-Text2SQL')
    model = PeftModel.from_pretrained(base_model, str(model_path))
    model.eval()
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer


# Different prompt templates to test
PROMPT_TEMPLATES = {
    "original": lambda schema, question, examples: f"### Schema:\n{schema}\n\n### Question:\n{question}\n\n### SQL:\n",
    
    "simple": lambda schema, question, examples: f"Table: {schema}\nQuestion: {question}\nSQL:",
    
    "concise": lambda schema, question, examples: f"Schema: {schema}\nQ: {question}\nSQL: SELECT",
    
    "few_shot": lambda schema, question, examples: (
        f"### Schema:\n{schema}\n\n" +
        "\n".join([f"### Question:\n{q}\n### SQL:\n{s}" for q, s in examples[:2]]) +
        f"\n\n### Question:\n{question}\n\n### SQL:\n"
    ),
    
    "instruction": lambda schema, question, examples: (
        f"Convert the following question to SQL using this schema:\n{schema}\n\n"
        f"Question: {question}\nSQL:"
    ),
}

# Different generation configs to test
GENERATION_CONFIGS = {
    "greedy": {
        "max_new_tokens": 256,
        "do_sample": False,
    },
    "greedy_rep_penalty": {
        "max_new_tokens": 256,
        "do_sample": False,
        "repetition_penalty": 1.2,
    },
    "beam_search": {
        "max_new_tokens": 256,
        "num_beams": 3,
        "early_stopping": True,
    },
    "sampling": {
        "max_new_tokens": 256,
        "do_sample": True,
        "temperature": 0.7,
        "top_p": 0.9,
    },
    "sampling_low_temp": {
        "max_new_tokens": 256,
        "do_sample": True,
        "temperature": 0.3,
        "top_k": 50,
    },
}


def generate_sql(model, tokenizer, question, schema, prompt_template, gen_config, examples=None):
    """Generate SQL with specified prompt and generation config."""
    
    prompt_fn = PROMPT_TEMPLATES[prompt_template]
    prompt = prompt_fn(schema, question, examples or [])
    
    # Determine SQL marker based on template
    if "SQL: SELECT" in prompt:
        sql_marker = "SQL: SELECT"
        prefix = "SELECT "
    elif "SQL:" in prompt:
        sql_marker = "SQL:"
        prefix = ""
    else:
        sql_marker = "### SQL:"
        prefix = ""
    
    inputs = tokenizer(prompt, return_tensors='pt', padding=True)
    
    config = GENERATION_CONFIGS[gen_config].copy()
    config["pad_token_id"] = tokenizer.eos_token_id
    
    with torch.no_grad():
        outputs = model.generate(**inputs, **config)
    
    generated = tokenizer.decode(outputs[0], skip_special_tokens=True)
    raw_sql = generated.split(sql_marker)[-1].strip()
    return prefix + postprocess_sql(raw_sql)


def evaluate_config(model, tokenizer, test_examples, db_config, prompt_template, gen_config, num_samples=10):
    """Evaluate a specific configuration."""
    
    schema = db_config["schema_short"]  # Always use short schema
    db_path = db_config["db_file"]
    db_name = "geography" if "geography" in str(db_path) else "atis"
    examples = FEW_SHOT_EXAMPLES.get(db_name, [])
    
    exec_success = 0
    result_match = 0
    
    for ex in test_examples[:num_samples]:
        pred_sql = generate_sql(model, tokenizer, ex['question'], schema, prompt_template, gen_config, examples)
        
        gold_result, gold_error = execute_sql(db_path, ex['gold_sql'])
        pred_result, pred_error = execute_sql(db_path, pred_sql)
        
        if pred_error is None:
            exec_success += 1
        if gold_result is not None and pred_result is not None:
            if sorted(map(str, gold_result)) == sorted(map(str, pred_result)):
                result_match += 1
    
    return {
        "exec_success": exec_success,
        "exec_pct": round(exec_success / num_samples * 100, 1),
        "result_match": result_match,
        "result_pct": round(result_match / num_samples * 100, 1)
    }


def run_tuning_experiments(database="geography", num_samples=10):
    """Run experiments with different configurations."""
    
    print("="*70)
    print(f"TINYLLAMA TUNING EXPERIMENTS - {database.upper()}")
    print("="*70)
    
    model_path = TRAINING_RESULTS_DIR / f"tinyllama-{database}" / "final"
    if not model_path.exists():
        print(f"Model not found: {model_path}")
        return
    
    model, tokenizer = load_tinyllama_model(model_path)
    
    db_config = DATABASES[database]
    test_examples = load_test_data(db_config["json_file"])
    
    print(f"\nTesting {len(PROMPT_TEMPLATES)} prompt templates x {len(GENERATION_CONFIGS)} generation configs")
    print(f"Samples per config: {num_samples}\n")
    
    results = []
    
    for prompt_name in PROMPT_TEMPLATES:
        for gen_name in GENERATION_CONFIGS:
            print(f"Testing: {prompt_name} + {gen_name}...", end=" ", flush=True)
            
            metrics = evaluate_config(
                model, tokenizer, test_examples, db_config,
                prompt_name, gen_name, num_samples
            )
            
            print(f"Exec: {metrics['exec_pct']}%, Result: {metrics['result_pct']}%")
            
            results.append({
                "prompt": prompt_name,
                "generation": gen_name,
                **metrics
            })
    
    # Sort by result match
    results.sort(key=lambda x: (-x["result_pct"], -x["exec_pct"]))
    
    print("\n" + "="*70)
    print("RESULTS RANKED BY RESULT MATCH")
    print("="*70)
    print(f"{'Prompt':<15} {'Generation':<20} {'Exec %':<10} {'Result %':<10}")
    print("-"*55)
    
    for r in results[:10]:
        print(f"{r['prompt']:<15} {r['generation']:<20} {r['exec_pct']:<10} {r['result_pct']:<10}")
    
    # Save results
    output = {
        "database": database,
        "timestamp": str(datetime.now()),
        "num_samples": num_samples,
        "results": results
    }
    
    output_file = RESULTS_DIR / f"tinyllama_tuning_{database}.json"
    with open(output_file, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to: {output_file}")
    
    return results


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--database", type=str, default="geography", choices=["geography", "atis"])
    parser.add_argument("--samples", type=int, default=10)
    
    args = parser.parse_args()
    run_tuning_experiments(args.database, args.samples)
