#!/usr/bin/env python3
"""Test TinyLlama with its actual chat template."""
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

questions = [
    "what is the capital of texas",
    "what is the population of california", 
    "how many states are there",
]

print("=== Testing with chat template format ===\n")

for question in questions:
    # Use the chat template format from tokenizer config
    messages = [
        {"role": "system", "content": f"You are a SQL expert. Given this database schema:\n{schema}\n\nConvert questions to SQL queries."},
        {"role": "user", "content": question}
    ]
    
    # Apply chat template
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    print(f"Prompt preview: {prompt[:200]}...")
    
    inputs = tokenizer(prompt, return_tensors='pt', truncation=True, max_length=1500)
    outputs = model.generate(
        **inputs, 
        max_new_tokens=60, 
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id
    )
    
    result = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Extract after <|assistant|>
    if '<|assistant|>' in result:
        generated = result.split('<|assistant|>')[-1].strip()
    else:
        generated = result[len(prompt):].strip()
    
    print(f"Q: {question}")
    print(f"Generated: {generated[:100]}")
    
    # Try executing if it looks like SQL
    if 'SELECT' in generated.upper():
        sql = generated.split('\n')[0]
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

# Also try raw format without chat template
print("\n=== Testing raw format ===\n")
for question in questions[:2]:
    # Simple format
    prompt = f"<|user|>\nSchema: {schema}\nQuestion: {question}</s>\n<|assistant|>\n"
    
    inputs = tokenizer(prompt, return_tensors='pt', truncation=True, max_length=1500)
    outputs = model.generate(
        **inputs, 
        max_new_tokens=60, 
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id
    )
    
    result = tokenizer.decode(outputs[0], skip_special_tokens=True)
    generated = result.split('<|assistant|>')[-1].strip() if '<|assistant|>' in result else result[len(prompt):]
    
    print(f"Q: {question}")
    print(f"SQL: {generated[:100]}")
    print()
