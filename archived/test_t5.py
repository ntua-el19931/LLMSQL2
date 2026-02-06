#!/usr/bin/env python3
"""Test T5 text2sql model."""
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

model_id = 'n22t7a/text2sql-tuned-T5'
print(f'Loading {model_id}...')
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForSeq2SeqLM.from_pretrained(model_id)

# Test with cleaner CREATE TABLE format
tests = [
    ('CREATE TABLE state (state_name TEXT, population INT, capital TEXT) What is the capital of Texas?',
     'SELECT capital FROM state WHERE state_name = "texas"'),
    
    ('CREATE TABLE employees (id INT, name TEXT, salary INT) Show all employee names',
     'SELECT name FROM employees'),
    
    ('CREATE TABLE products (id INT, name TEXT, price REAL) What is the price of Apple?',
     'SELECT price FROM products WHERE name = "Apple"'),
]

print('Testing T5 model:')
print()
for prompt, expected in tests:
    inputs = tokenizer(prompt, return_tensors='pt')
    outputs = model.generate(**inputs, max_new_tokens=60)
    result = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    print(f'Input: {prompt[:70]}...')
    print(f'Expected: {expected}')
    print(f'Got: {result}')
    print()
