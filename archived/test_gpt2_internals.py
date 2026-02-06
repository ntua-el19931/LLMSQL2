#!/usr/bin/env python3
"""Check GPT-2 model internals."""
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
import torch

model_id = 'n22t7a/text2sql-tuned-gpt2'
print(f"Loading {model_id}...")

# Check config
config = AutoConfig.from_pretrained(model_id)
print(f"\nModel Config:")
print(f"  Architecture: {config.architectures}")
print(f"  Vocab size: {config.vocab_size}")
print(f"  Hidden size: {config.n_embd}")
print(f"  Layers: {config.n_layer}")
print(f"  Heads: {config.n_head}")

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float32)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# Try with sampling enabled
question = "What is the population of Texas?"
prompt = f"Question: {question}\nSQL:"

print(f"\nPrompt: {prompt}")
print("\n=== Trying different generation strategies ===\n")

# Strategy 1: Greedy (no sampling)
inputs = tokenizer(prompt, return_tensors='pt')
outputs = model.generate(**inputs, max_new_tokens=30, do_sample=False, pad_token_id=tokenizer.eos_token_id)
result = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(f"Greedy: {result}")

# Strategy 2: With temperature
outputs = model.generate(**inputs, max_new_tokens=30, do_sample=True, temperature=0.7, pad_token_id=tokenizer.eos_token_id)
result = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(f"Temp=0.7: {result}")

# Strategy 3: Top-k
outputs = model.generate(**inputs, max_new_tokens=30, do_sample=True, top_k=50, pad_token_id=tokenizer.eos_token_id)
result = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(f"Top-k=50: {result}")

# Strategy 4: Top-p
outputs = model.generate(**inputs, max_new_tokens=30, do_sample=True, top_p=0.9, pad_token_id=tokenizer.eos_token_id)
result = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(f"Top-p=0.9: {result}")

# Let's also check logits directly
print("\n=== Checking raw logits ===")
with torch.no_grad():
    outputs = model(**inputs)
    logits = outputs.logits
    
    # Get the logits for the last token
    last_logits = logits[0, -1, :]
    
    # Get top 10 predicted tokens
    top_k = torch.topk(last_logits, 10)
    print("Top 10 predicted next tokens:")
    for score, idx in zip(top_k.values, top_k.indices):
        token = tokenizer.decode([idx.item()])
        print(f"  {token!r}: {score.item():.2f}")
