#!/usr/bin/env python3
"""Deep dive into GPT-2 Text2SQL model."""
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

model_id = 'n22t7a/text2sql-tuned-gpt2'
print(f"Loading {model_id}...")
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float32)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# The model seems to need SELECT as a starter
# Let's try with proper schema context

tests = [
    ("state(state_name, population, area)", "What is the population of Texas?"),
    ("employees(id, name, salary)", "Show all employee names"),
    ("products(id, name, price)", "What products cost more than 100?"),
]

print("\n=== Testing with SELECT prefix ===\n")
for schema, question in tests:
    # Format that worked: question + SELECT
    prompt = f"{schema} | {question} SELECT"
    
    inputs = tokenizer(prompt, return_tensors='pt')
    outputs = model.generate(
        **inputs, 
        max_new_tokens=50, 
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id
    )
    
    result = tokenizer.decode(outputs[0], skip_special_tokens=True)
    generated = result[len(prompt):].strip()
    
    print(f"Q: {question}")
    print(f"SQL: SELECT{generated}")
    print()

# Try Spider format: table | question
print("\n=== Testing Spider dataset format ===\n")
for schema, question in tests:
    prompt = f"{schema} | {question}"
    
    inputs = tokenizer(prompt, return_tensors='pt')
    outputs = model.generate(
        **inputs, 
        max_new_tokens=50, 
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id
    )
    
    result = tokenizer.decode(outputs[0], skip_special_tokens=True)
    generated = result[len(prompt):].strip()
    
    print(f"Q: {question}")
    print(f"Generated: {generated}")
    print()

# Try with special tokens
print("\n=== Check special tokens ===")
print(f"BOS: {tokenizer.bos_token}")
print(f"EOS: {tokenizer.eos_token}")
print(f"PAD: {tokenizer.pad_token}")
print(f"Vocab size: {len(tokenizer)}")

# Check if there are any special tokens in the vocab for SQL
print("\n=== SQL-related tokens ===")
sql_keywords = ['SELECT', 'FROM', 'WHERE', 'TABLE', 'INSERT', 'UPDATE', 'SQL', '<sql>', '[SQL]']
for kw in sql_keywords:
    tokens = tokenizer.encode(kw, add_special_tokens=False)
    print(f"{kw}: tokens={tokens}")
