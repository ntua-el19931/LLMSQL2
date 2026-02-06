#!/usr/bin/env python3
"""Test TinyLlama with the EXACT training data format."""
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

model_id = 'ManthanKulakarni/TinyLlama-1.1B-Text2SQL'
print(f"Loading {model_id}...")
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float32)

# Training data format:
# - table: CREATE TABLE statements
# - query: natural language question
# - sql: expected output

# Test with the EXACT format from training data
table = '''CREATE TABLE table_47666 (
    "Res." text,
    "Record" text,
    "Opponent" text,
    "Method" text,
    "Round" text
)'''
query = "Which opponent has 1-1 as the record?"
expected = """SELECT "Opponent" FROM table_47666 WHERE "Record" = '1-1'"""

# Try different possible prompt templates used during training
templates = {
    # Template 1: Just concatenation
    "concat": f"{table}\n{query}",
    
    # Template 2: With labels
    "labeled": f"Table: {table}\nQuestion: {query}\nSQL:",
    
    # Template 3: Instruction style
    "instruction": f"Given the table:\n{table}\n\nWrite SQL for: {query}\n\nSQL:",
    
    # Template 4: The ### format we saw the model outputting
    "hash": f"### Context:\n{table}\n\n### Question:\n{query}\n\n### Answer:\n",
    
    # Template 5: Simple with SELECT
    "select": f"{table}\n\n{query}\n\nSELECT",
    
    # Template 6: TRL/SFT common format
    "sft": f"<s>[INST] {table}\n\n{query} [/INST]",
    
    # Template 7: Just table and query separated
    "simple": f"{table}\n\nQuery: {query}\n\nAnswer:",
}

print(f"\nExpected SQL: {expected}\n")
print("="*60)

for name, prompt in templates.items():
    inputs = tokenizer(prompt, return_tensors='pt')
    outputs = model.generate(
        **inputs,
        max_new_tokens=60,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id
    )
    
    result = tokenizer.decode(outputs[0], skip_special_tokens=True)
    generated = result[len(prompt):].strip()
    
    # Clean up
    if '\n' in generated:
        generated = generated.split('\n')[0]
    
    # Check if it matches
    match = "✓" if expected.lower() in generated.lower() or generated.lower() in expected.lower() else "✗"
    
    print(f"\n{name}:")
    print(f"  Generated: {generated[:80]}")
    print(f"  Match: {match}")
