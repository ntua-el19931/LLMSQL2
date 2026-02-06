#!/usr/bin/env python3
"""Final attempt: try different generation strategies."""
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# Test both models with various strategies
models_to_test = [
    'n22t7a/text2sql-tuned-gpt2',
    'ManthanKulakarni/TinyLlama-1.1B-Text2SQL'
]

question = "What is the capital of Texas?"
table = "CREATE TABLE states (name VARCHAR, capital VARCHAR, population INTEGER)"

for model_id in models_to_test:
    print(f"\n{'='*60}")
    print(f"Model: {model_id}")
    print(f"{'='*60}")
    
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float32)
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # For GPT-2: try forcing SELECT continuation
    if 'gpt2' in model_id.lower():
        prompt = f"{table}\nQuestion: {question}\nSQL: SELECT"
    else:
        # For TinyLlama: use its expected format
        prompt = f"### Context:\n{table}\n\n### Question:\n{question}\n\n### Answer:\n"
    
    print(f"Prompt: {prompt[:100]}...")
    
    inputs = tokenizer(prompt, return_tensors='pt')
    
    # Try multiple generation strategies
    strategies = [
        {"do_sample": False, "num_beams": 1, "name": "Greedy"},
        {"do_sample": False, "num_beams": 4, "name": "Beam=4"},
        {"do_sample": True, "temperature": 0.3, "name": "Temp=0.3"},
        {"do_sample": True, "temperature": 0.1, "name": "Temp=0.1"},
        {"do_sample": True, "top_p": 0.95, "name": "Top-p=0.95"},
        {"do_sample": True, "top_k": 10, "name": "Top-k=10"},
    ]
    
    for strat in strategies:
        name = strat.pop("name")
        try:
            outputs = model.generate(
                **inputs,
                max_new_tokens=40,
                pad_token_id=tokenizer.eos_token_id,
                **strat
            )
            result = tokenizer.decode(outputs[0], skip_special_tokens=True)
            generated = result[len(prompt):].strip()
            if '\n' in generated:
                generated = generated.split('\n')[0]
            print(f"  {name}: {generated[:60]}")
        except Exception as e:
            print(f"  {name}: Error - {e}")
        strat["name"] = name  # restore
    
    del model
    del tokenizer
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
