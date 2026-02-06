#!/usr/bin/env python3
"""Force GPT-2 to generate past EOS."""
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

model_id = 'n22t7a/text2sql-tuned-gpt2'
print(f"Loading {model_id}...")

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float32)

# Use a different pad token to prevent early stopping
tokenizer.pad_token = tokenizer.eos_token

question = "What is the population of Texas?"

# Try different prompt formats
prompts = [
    f"Question: {question}\nSQL: SELECT",
    f"{question} | SELECT",
    f"translate to sql: {question} SELECT",
    f"question: {question} answer: SELECT",
]

print("\n=== Testing with forced SELECT continuation ===\n")

for prompt in prompts:
    inputs = tokenizer(prompt, return_tensors='pt')
    
    # Try forcing generation with min_new_tokens
    outputs = model.generate(
        **inputs, 
        max_new_tokens=40,
        min_new_tokens=10,  # Force at least 10 tokens
        do_sample=True,
        temperature=0.5,
        pad_token_id=tokenizer.eos_token_id,
        eos_token_id=None  # Disable EOS stopping
    )
    
    result = tokenizer.decode(outputs[0], skip_special_tokens=True)
    generated = result[len(prompt):]
    
    print(f"Prompt: ...{prompt[-40:]}")
    print(f"Generated: SELECT{generated}")
    print()

# Now try the Spider-style format that's common for text2sql
print("\n=== Testing Spider-style format ===\n")
schema = "state : state_name , population , area , capital , density"
question = "What is the population of Texas?"

spider_prompts = [
    f"{schema} | {question}",
    f"Table: {schema}\nQuestion: {question}\nSQL:",
    f"{schema}\n{question}\n",
]

for prompt in spider_prompts:
    inputs = tokenizer(prompt, return_tensors='pt')
    outputs = model.generate(
        **inputs, 
        max_new_tokens=40,
        min_new_tokens=5,
        do_sample=True,
        temperature=0.3,
        pad_token_id=tokenizer.eos_token_id,
        eos_token_id=None
    )
    
    result = tokenizer.decode(outputs[0], skip_special_tokens=True)
    generated = result[len(prompt):]
    
    print(f"Prompt: {prompt[:60]}...")
    print(f"Generated: {generated}")
    print()
