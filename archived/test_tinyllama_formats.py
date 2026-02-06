#!/usr/bin/env python3
"""Test TinyLlama with Alpaca/Chat instruction format."""
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

# Simple schema string
cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'table_name%'")
tables = [row[0] for row in cursor.fetchall()]

schema_parts = []
for table in tables:
    cursor.execute(f"PRAGMA table_info({table})")
    cols = cursor.fetchall()
    col_list = ", ".join([c[1] for c in cols])
    schema_parts.append(f"{table}({col_list})")
conn.close()

schema = "; ".join(schema_parts)
print(f"Schema: {schema[:200]}...")

question = "what is the capital of texas"

# Different instruction formats
formats = {
    "Alpaca": f"""Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
Given the following database schema:
{schema}

Convert this question to SQL: {question}

### Response:
""",

    "ChatML": f"""<|system|>
You are a SQL expert. Convert natural language to SQL.
<|user|>
Schema: {schema}
Question: {question}
<|assistant|>
""",

    "Vicuna": f"""A chat between a user and an assistant.

USER: Given this schema: {schema}
Write SQL for: {question}
ASSISTANT: """,

    "Simple": f"""Schema: {schema}

Question: {question}

SQL: """,

    "Direct": f"""Convert to SQL.
Tables: {schema}
Question: {question}
Answer: SELECT"""
}

print("\n=== Testing instruction formats ===\n")

for name, prompt in formats.items():
    inputs = tokenizer(prompt, return_tensors='pt', truncation=True, max_length=1500)
    outputs = model.generate(
        **inputs, 
        max_new_tokens=50, 
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id
    )
    
    result = tokenizer.decode(outputs[0], skip_special_tokens=True)
    generated = result[len(prompt):].strip()
    
    # Clean up
    if '\n' in generated:
        generated = generated.split('\n')[0]
    
    print(f"=== {name} ===")
    print(f"Generated: {generated[:100]}")
    print()
