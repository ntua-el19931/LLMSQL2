#!/usr/bin/env python3
"""Test TinyLlama with the exact internal format."""
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

model_id = 'ManthanKulakarni/TinyLlama-1.1B-Text2SQL'
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float32)

# The model seems to expect ### Context / Question / Answer format
# wrapped inside the chat template

table = 'CREATE TABLE state (state_name TEXT, population INT, capital TEXT)'
query = 'What is the capital of Texas?'

# Try the internal format directly without chat wrapper
# Based on what the model is generating, it expects:
# ### Context:
# CREATE TABLE...
# ### Question:
# ...
# ### Answer:

prompt = f"""### Context:
{table}
### Question:
{query}
### Answer:
"""

print("Testing direct ### format:")
print(f"Prompt: {prompt}")

inputs = tokenizer(prompt, return_tensors='pt')
outputs = model.generate(
    **inputs,
    max_new_tokens=60,
    do_sample=False,
    pad_token_id=tokenizer.eos_token_id
)

result = tokenizer.decode(outputs[0], skip_special_tokens=True)
answer = result.split('### Answer:')[-1].strip()
if '\n' in answer:
    answer = answer.split('\n')[0]
    
print(f"Generated: {answer}")
print()

# Try with chat wrapper containing the ### format
print("Testing chat wrapper with ### format inside:")
internal_content = f"""### Context:
{table}
### Question:
{query}
### Answer:"""

messages = [
    {'role': 'user', 'content': internal_content}
]

prompt2 = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
print(f"Prompt: {prompt2[:200]}...")

inputs = tokenizer(prompt2, return_tensors='pt')
outputs = model.generate(
    **inputs,
    max_new_tokens=60,
    do_sample=False,
    pad_token_id=tokenizer.eos_token_id
)

result = tokenizer.decode(outputs[0], skip_special_tokens=True)
if '<|assistant|>' in result:
    answer = result.split('<|assistant|>')[-1].strip()
else:
    answer = result.split('### Answer:')[-1].strip()
    
if '\n' in answer:
    answer = answer.split('\n')[0]

print(f"Generated: {answer}")
