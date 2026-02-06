#!/usr/bin/env python3
"""Test TinyLlama with training examples from the dataset."""
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
import torch

model_id = 'ManthanKulakarni/TinyLlama-1.1B-Text2SQL'
print(f"Loading {model_id}...")
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float32)

# Load training examples
print("Loading training data...")
ds = load_dataset('ManthanKulakarni/Text2SQL', split='train', streaming=True)

print("\n=== Testing with actual training examples ===\n")

for i, example in enumerate(ds):
    if i >= 5:
        break
    
    table = example['table']
    query = example['query']
    expected_sql = example['sql']
    
    # Use the ### format
    prompt = f"""### Context:
{table}

### Question:
{query}

### Answer:
"""
    
    inputs = tokenizer(prompt, return_tensors='pt', truncation=True, max_length=1024)
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
    
    print(f"Example {i+1}:")
    print(f"  Query:    {query[:60]}...")
    print(f"  Expected: {expected_sql[:60]}...")
    print(f"  Got:      {generated[:60]}...")
    
    # Check if similar
    match = "✓" if (
        generated.lower().replace('"', "'").replace(' ', '') == 
        expected_sql.lower().replace('"', "'").replace(' ', '')
    ) else "✗"
    print(f"  Match: {match}")
    print()
