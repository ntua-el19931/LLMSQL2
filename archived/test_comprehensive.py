#!/usr/bin/env python3
"""Comprehensive test of TinyLlama with ### format."""
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import sqlite3

model_id = 'ManthanKulakarni/TinyLlama-1.1B-Text2SQL'
print(f"Loading {model_id}...")
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float32)

# Test cases with simple schemas
test_cases = [
    {
        'table': 'CREATE TABLE state (state_name TEXT, population INT, capital TEXT)',
        'query': 'What is the capital of Texas?',
        'expected': "SELECT capital FROM state WHERE state_name = 'Texas'"
    },
    {
        'table': 'CREATE TABLE products (id INT, name TEXT, price REAL)',
        'query': 'What is the price of Apple?',
        'expected': "SELECT price FROM products WHERE name = 'Apple'"
    },
    {
        'table': 'CREATE TABLE employees (id INT, name TEXT, salary INT, department TEXT)',
        'query': 'Show all employee names',
        'expected': "SELECT name FROM employees"
    },
    {
        'table': 'CREATE TABLE students (id INT, name TEXT, age INT, grade TEXT)',
        'query': 'How many students are there?',
        'expected': "SELECT COUNT(*) FROM students"
    },
    {
        'table': 'CREATE TABLE orders (order_id INT, customer TEXT, amount REAL, date TEXT)',
        'query': 'What is the total amount of all orders?',
        'expected': "SELECT SUM(amount) FROM orders"
    },
]

print("\n=== Testing with ### Context/Question/Answer format ===\n")

correct = 0
for i, test in enumerate(test_cases, 1):
    prompt = f"""### Context:
{test['table']}
### Question:
{test['query']}
### Answer:
"""
    
    inputs = tokenizer(prompt, return_tensors='pt')
    outputs = model.generate(
        **inputs,
        max_new_tokens=80,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id
    )
    
    result = tokenizer.decode(outputs[0], skip_special_tokens=True)
    generated = result.split('### Answer:')[-1].strip()
    
    # Clean up
    if '\n' in generated:
        generated = generated.split('\n')[0]
    if '###' in generated:
        generated = generated.split('###')[0].strip()
    
    # Check if syntactically valid by trying to parse SELECT
    has_select = 'SELECT' in generated.upper()
    
    print(f"Test {i}: {test['query'][:40]}...")
    print(f"  Expected: {test['expected']}")
    print(f"  Got:      {generated[:80]}")
    print(f"  Has SELECT: {'✓' if has_select else '✗'}")
    
    # Normalize and compare
    exp_norm = test['expected'].lower().replace('"', "'").replace(' ', '')
    got_norm = generated.lower().replace('"', "'").replace(' ', '')
    
    if exp_norm == got_norm:
        print("  Exact match: ✓")
        correct += 1
    elif has_select and 'FROM' in generated.upper():
        print("  Partial match (valid SQL structure)")
    else:
        print("  Match: ✗")
    print()

print(f"Exact matches: {correct}/{len(test_cases)}")
