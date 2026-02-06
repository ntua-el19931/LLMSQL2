#!/usr/bin/env python3
"""Test GPT-2 as SQL completion model."""
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

model_id = 'n22t7a/text2sql-tuned-gpt2'
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id)

tokenizer.pad_token = tokenizer.eos_token

# Test as SQL completion
prompts = [
    'SELECT population FROM state WHERE',
    'SELECT * FROM state WHERE state_name =',
    'SELECT COUNT(*) FROM',
    'SELECT capital FROM state WHERE state_name = "',
]

print('Testing GPT-2 as SQL completion model:')
for prompt in prompts:
    inputs = tokenizer(prompt, return_tensors='pt')
    outputs = model.generate(
        **inputs,
        max_new_tokens=30,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id
    )
    result = tokenizer.decode(outputs[0], skip_special_tokens=True)
    generated = result[len(prompt):]
    print(f'Prompt: {prompt}')
    print(f'Completed: {generated}')
    print()
