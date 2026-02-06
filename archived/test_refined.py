#!/usr/bin/env python3
"""Refine TinyLlama prompts to get better results."""
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import sqlite3

model_id = 'ManthanKulakarni/TinyLlama-1.1B-Text2SQL'
print(f"Loading {model_id}...")
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float32)

db_path = '/app/data/text2sql-data/data/geography-db.added-in-2020.sqlite'

# Get simple schema with clear descriptions
conn = sqlite3.connect(db_path)
cursor = conn.cursor()

# Get state table schema
cursor.execute("PRAGMA table_info(state)")
cols = cursor.fetchall()
state_cols = ", ".join([f"{c[1]} ({c[2]})" for c in cols])
print(f"State columns: {state_cols}")

# Sample data
cursor.execute("SELECT * FROM state LIMIT 2")
sample = cursor.fetchall()
print(f"Sample data: {sample}")
conn.close()

# Test with very explicit prompts
test_questions = [
    ("what is the capital of texas", "SELECT capital FROM state WHERE state_name = 'texas'"),
    ("what is the population of california", "SELECT population FROM state WHERE state_name = 'california'"),
    ("list all state names", "SELECT state_name FROM state"),
]

print("\n=== Testing with explicit column descriptions ===\n")

for question, expected in test_questions:
    # Very explicit schema
    schema = """CREATE TABLE state (
  state_name TEXT,  -- the name of the state like 'texas', 'california'
  population INTEGER,  -- the population of the state
  area INTEGER,  -- the area in square miles  
  country_name TEXT,  -- always 'usa'
  capital TEXT,  -- the capital city of the state
  density REAL  -- population density
)"""

    prompt = f"""### Context:
{schema}

### Question:
{question}

### Answer:
"""

    inputs = tokenizer(prompt, return_tensors='pt')
    
    # Use temperature for variety
    outputs = model.generate(
        **inputs,
        max_new_tokens=50,
        do_sample=True,
        temperature=0.3,
        pad_token_id=tokenizer.eos_token_id
    )
    
    result = tokenizer.decode(outputs[0], skip_special_tokens=True)
    sql_part = result.split('### Answer:')[-1].strip()
    if '\n' in sql_part:
        sql_part = sql_part.split('\n')[0]
    if '###' in sql_part:
        sql_part = sql_part.split('###')[0].strip()
    
    print(f"Q: {question}")
    print(f"Expected: {expected}")
    print(f"Got:      {sql_part}")
    
    # Execute both
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    try:
        cursor.execute(sql_part)
        got_result = cursor.fetchall()[:3]
        cursor.execute(expected)
        exp_result = cursor.fetchall()[:3]
        print(f"Got result:      {got_result}")
        print(f"Expected result: {exp_result}")
        print(f"Match: {'✓' if got_result == exp_result else '✗'}")
    except Exception as e:
        print(f"Error: {str(e)[:60]}")
    conn.close()
    print()
