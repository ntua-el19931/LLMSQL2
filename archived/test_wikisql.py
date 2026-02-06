#!/usr/bin/env python3
"""Test if TinyLlama works when given EXACTLY its expected format."""
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

model_id = 'ManthanKulakarni/TinyLlama-1.1B-Text2SQL'
print(f"Loading {model_id}...")
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float32)

# Test with EXACTLY the format the model seems to expect
# Based on its outputs, it expects: ### Context: CREATE TABLE table_name_XX...

test_cases = [
    # Simple table with clear column names
    {
        "table": "CREATE TABLE table_name_1 (name VARCHAR, capital VARCHAR, population INTEGER)",
        "question": "What is the capital of Texas?",
        "expected": "SELECT capital FROM table_name_1 WHERE name = 'Texas'"
    },
    {
        "table": "CREATE TABLE table_name_2 (id INTEGER, name VARCHAR, price REAL)",  
        "question": "What is the price of Apple?",
        "expected": "SELECT price FROM table_name_2 WHERE name = 'Apple'"
    },
    {
        "table": "CREATE TABLE table_name_3 (year INTEGER, winner VARCHAR, score INTEGER)",
        "question": "Who won in 2020?",
        "expected": "SELECT winner FROM table_name_3 WHERE year = 2020"
    }
]

print("\n=== Testing with WikiSQL-style format ===\n")

for test in test_cases:
    prompt = f"""### Context:
{test['table']}

### Question:
{test['question']}

### Answer:
"""

    inputs = tokenizer(prompt, return_tensors='pt')
    outputs = model.generate(
        **inputs, 
        max_new_tokens=50, 
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id
    )
    
    result = tokenizer.decode(outputs[0], skip_special_tokens=True)
    sql_part = result.split('### Answer:')[-1].strip()
    
    # Clean up
    if '###' in sql_part:
        sql_part = sql_part.split('###')[0].strip()
    sql_part = sql_part.split('\n')[0].strip()
    
    print(f"Q: {test['question']}")
    print(f"Expected: {test['expected']}")
    print(f"Got:      {sql_part}")
    print(f"Match: {'✓' if sql_part.lower().strip() == test['expected'].lower().strip() else '✗'}")
    print()
