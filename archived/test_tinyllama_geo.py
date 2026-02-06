#!/usr/bin/env python3
"""Deep test of TinyLlama Text2SQL with geography database."""
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import sqlite3

model_id = 'ManthanKulakarni/TinyLlama-1.1B-Text2SQL'
print(f"Loading {model_id}...")
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float32)

# Get actual schema from geography database
db_path = '/app/data/text2sql-data/data/geography-db.added-in-2020.sqlite'
print(f"\n=== Getting schema from {db_path} ===")
conn = sqlite3.connect(db_path)
cursor = conn.cursor()

# Get all tables
cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
tables = [row[0] for row in cursor.fetchall()]
print(f"Tables: {tables}")

# Build CREATE TABLE statements
schema_parts = []
for table in tables[:8]:  # Get several tables
    cursor.execute(f"PRAGMA table_info({table})")
    cols = cursor.fetchall()
    col_defs = ", ".join([f"{c[1]} {c[2] or 'TEXT'}" for c in cols])
    create_stmt = f"CREATE TABLE {table} ({col_defs})"
    schema_parts.append(create_stmt)
    print(f"  {table}: {len(cols)} columns")

conn.close()

# Full schema for context
full_schema = "\n".join(schema_parts)
print(f"\nSchema length: {len(full_schema)} chars")
print(f"First 300 chars: {full_schema[:300]}...")

# Test questions from geography dataset
test_cases = [
    "what is the highest point in the state of oregon",
    "what states border texas",  
    "what is the population of california",
    "what rivers flow through colorado",
    "what is the capital of new york"
]

print("\n=== Testing TinyLlama with actual geography schema ===\n")

for question in test_cases:
    prompt = f"""### Context:
{full_schema}

### Question:
{question}

### Answer:
"""
    
    inputs = tokenizer(prompt, return_tensors='pt', truncation=True, max_length=2048)
    outputs = model.generate(
        **inputs, 
        max_new_tokens=80, 
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id
    )
    
    result = tokenizer.decode(outputs[0], skip_special_tokens=True)
    sql_part = result.split('### Answer:')[-1].strip()
    
    # Clean up
    if '###' in sql_part:
        sql_part = sql_part.split('###')[0].strip()
    sql_part = sql_part.split('\n')[0].strip()
    
    print(f"Q: {question}")
    print(f"SQL: {sql_part}")
    
    # Try to execute
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute(sql_part)
        results = cursor.fetchall()[:3]
        print(f"Result: {results}")
        conn.close()
    except Exception as e:
        print(f"Error: {str(e)[:80]}")
    print()
