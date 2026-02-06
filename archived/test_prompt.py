#!/usr/bin/env python3
"""Test TinyLlama prompt format."""
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

model_id = 'ManthanKulakarni/TinyLlama-1.1B-Text2SQL'
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float32)

# Test with simpler questions and schemas
tests = [
    {
        'schema': 'CREATE TABLE employees (id INT, name TEXT, salary INT, department TEXT)',
        'question': 'Show all employee names'
    },
    {
        'schema': 'CREATE TABLE products (id INT, name TEXT, price DECIMAL)',
        'question': 'What is the price of product Apple?'
    },
    {
        'schema': 'CREATE TABLE city (city_name TEXT, state TEXT, population INT)',
        'question': 'What cities are in California?'
    }
]

for test in tests:
    prompt = f"""### Context:
{test['schema']}

### Question:
{test['question']}

### Answer:
"""
    inputs = tokenizer(prompt, return_tensors='pt')
    outputs = model.generate(**inputs, max_new_tokens=50, do_sample=False, pad_token_id=tokenizer.eos_token_id)
    result = tokenizer.decode(outputs[0], skip_special_tokens=True)
    sql_part = result.split('### Answer:')[-1].strip()
    if '###' in sql_part:
        sql_part = sql_part.split('###')[0].strip()
    # Stop at newline
    sql_part = sql_part.split('\n')[0].strip()
    
    q = test['question']
    print(f'Q: {q}')
    print(f'SQL: {sql_part}')
    print()
