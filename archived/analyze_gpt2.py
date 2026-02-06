#!/usr/bin/env python3
"""Check what the GPT-2 Text2SQL model was trained on."""
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
import torch
import json

model_id = 'n22t7a/text2sql-tuned-gpt2'
print(f"Analyzing {model_id}...")

# Check if there's a training config or readme
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id)

# Look at the tokenizer's special tokens
print("\n=== Tokenizer Analysis ===")
print(f"Special tokens: {tokenizer.special_tokens_map}")
print(f"Additional special tokens: {tokenizer.additional_special_tokens if hasattr(tokenizer, 'additional_special_tokens') else 'None'}")

# Check if there are any special SQL-related tokens added
vocab = tokenizer.get_vocab()
sql_related = {k: v for k, v in vocab.items() if 'sql' in k.lower() or 'query' in k.lower() or 'table' in k.lower()}
print(f"\nSQL-related tokens in vocab: {len(sql_related)}")
if sql_related:
    print(f"Examples: {list(sql_related.items())[:10]}")

# Try to find what format was used in training by analyzing generation patterns
print("\n=== Generation Pattern Analysis ===")

# Common text2sql dataset formats to try
formats = {
    "Spider": "Tables: singer(singer_id, name, country) | Question: How many singers? | SQL:",
    "WikiSQL": "col : singer_id, name, country | Question: How many singers? | SELECT",
    "BIRD": "CREATE TABLE singer (singer_id INT, name TEXT, country TEXT);\n-- How many singers?\nSELECT",
    "CoSQL": "[TABLE] singer (singer_id, name, country) [NL] How many singers? [SQL]",
    "SEDE": "### Question: How many singers?\n### SQL:",
    "raw": "How many singers? SELECT",
}

for name, prompt in formats.items():
    inputs = tokenizer(prompt, return_tensors='pt')
    outputs = model.generate(
        **inputs, 
        max_new_tokens=20,
        min_new_tokens=5,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id,
        eos_token_id=None
    )
    result = tokenizer.decode(outputs[0], skip_special_tokens=True)
    generated = result[len(prompt):]
    print(f"{name}: {generated[:50]}")

# Check model embeddings for patterns
print("\n=== Model Architecture ===")
print(f"Vocab size: {model.config.vocab_size}")
print(f"Embedding dim: {model.config.n_embd}")
print(f"Layers: {model.config.n_layer}")
print(f"Model type: {model.config.model_type}")
