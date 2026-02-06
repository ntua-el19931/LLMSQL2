#!/usr/bin/env python3
"""Test various prompt formats for GPT-2 Text2SQL model."""
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

model_id = 'n22t7a/text2sql-tuned-gpt2'
print(f"Loading {model_id}...")
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float32)

# Set pad token
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

question = "What is the population of Texas?"
schema = "state(state_name, population, area, capital, density)"

# Try many different prompt formats
prompts = [
    # Format 1: Simple question
    f"{question}",
    
    # Format 2: SQL prefix
    f"SQL: {question}",
    
    # Format 3: Question -> SQL
    f"Question: {question}\nSQL:",
    
    # Format 4: Translate style
    f"Translate to SQL: {question}\nSQL:",
    
    # Format 5: With schema
    f"Schema: {schema}\nQuestion: {question}\nSQL:",
    
    # Format 6: NL to SQL
    f"NL: {question}\nSQL:",
    
    # Format 7: Tables prefix
    f"Tables: {schema}\n{question}\nSELECT",
    
    # Format 8: Just SELECT start
    f"{question}\nSELECT",
    
    # Format 9: Spider style
    f"{schema} | {question}",
    
    # Format 10: Instruction style
    f"Convert the following natural language question to SQL:\n{question}\nSQL:",
]

print("\nTesting different prompt formats:\n")
for i, prompt in enumerate(prompts, 1):
    inputs = tokenizer(prompt, return_tensors='pt')
    
    # Try with different generation settings
    outputs = model.generate(
        **inputs, 
        max_new_tokens=50, 
        do_sample=False,
        num_beams=1,
        pad_token_id=tokenizer.eos_token_id
    )
    
    result = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Get only the generated part (after prompt)
    generated = result[len(prompt):].strip()
    
    print(f"Format {i}:")
    print(f"  Prompt ends with: ...{prompt[-30:]}")
    print(f"  Generated: {generated[:100]}")
    print()
