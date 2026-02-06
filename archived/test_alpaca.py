#!/usr/bin/env python3
"""Test TinyLlama with refined Alpaca format."""
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import sqlite3

model_id = 'ManthanKulakarni/TinyLlama-1.1B-Text2SQL'
print(f"Loading {model_id}...")
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float32)

# Get schema
db_path = '/app/data/text2sql-data/data/geography-db.added-in-2020.sqlite'
conn = sqlite3.connect(db_path)
cursor = conn.cursor()

cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'table_name%'")
tables = [row[0] for row in cursor.fetchall()]

schema_parts = []
for table in tables:
    cursor.execute(f"PRAGMA table_info({table})")
    cols = cursor.fetchall()
    col_list = ", ".join([c[1] for c in cols])
    schema_parts.append(f"{table}({col_list})")
conn.close()

schema = "\n".join(schema_parts)
print(f"Schema:\n{schema}\n")

# Test cases
questions = [
    "what is the capital of texas",
    "what is the population of california", 
    "how many states are there",
    "which state has the largest population",
    "what rivers are in colorado"
]

print("=== Testing with Alpaca format ===\n")

for question in questions:
    prompt = f"""Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
Given the database schema:
{schema}

Write a SQL query to answer: {question}

### Response:
SELECT"""

    inputs = tokenizer(prompt, return_tensors='pt', truncation=True, max_length=1500)
    outputs = model.generate(
        **inputs, 
        max_new_tokens=60, 
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id
    )
    
    result = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Extract SQL after "### Response:"
    response = result.split('### Response:')[-1].strip()
    sql = 'SELECT' + response.split('SELECT')[-1] if 'SELECT' in response else response
    
    # Clean - stop at newline or ###
    if '\n' in sql:
        sql = sql.split('\n')[0]
    if '###' in sql:
        sql = sql.split('###')[0].strip()
    
    print(f"Q: {question}")
    print(f"SQL: {sql}")
    
    # Try executing
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute(sql)
        results = cursor.fetchall()[:3]
        print(f"✓ Result: {results}")
        conn.close()
    except Exception as e:
        print(f"✗ Error: {str(e)[:60]}")
    print()
