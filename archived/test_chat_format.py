#!/usr/bin/env python3
"""Understand the training format used for TinyLlama Text2SQL."""
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

tokenizer = AutoTokenizer.from_pretrained('ManthanKulakarni/TinyLlama-1.1B-Text2SQL')

# The chat template from the tokenizer config
print('Chat template:')
print(tokenizer.chat_template)
print()

# Format a sample training example using the chat template
table = 'CREATE TABLE state (state_name TEXT, population INT, capital TEXT)'
query = 'What is the capital of Texas?'
sql = "SELECT capital FROM state WHERE state_name = 'texas'"

# Using chat format for SFT
messages = [
    {'role': 'user', 'content': f'{table}\n\n{query}'},
    {'role': 'assistant', 'content': sql}
]

formatted = tokenizer.apply_chat_template(messages, tokenize=False)
print('Formatted training example:')
print(formatted)
print()

# Now test inference using this format
print('='*60)
print('Testing inference with this format:')
print('='*60)

model = AutoModelForCausalLM.from_pretrained('ManthanKulakarni/TinyLlama-1.1B-Text2SQL', torch_dtype=torch.float32)

# For inference, we only provide the user message and let model generate
inference_messages = [
    {'role': 'user', 'content': f'{table}\n\n{query}'}
]

prompt = tokenizer.apply_chat_template(inference_messages, tokenize=False, add_generation_prompt=True)
print('Inference prompt:')
print(prompt)
print()

inputs = tokenizer(prompt, return_tensors='pt')
outputs = model.generate(
    **inputs,
    max_new_tokens=60,
    do_sample=False,
    pad_token_id=tokenizer.eos_token_id
)

result = tokenizer.decode(outputs[0], skip_special_tokens=True)
print('Full output:')
print(result)
print()

# Extract assistant response
if '<|assistant|>' in result:
    assistant_response = result.split('<|assistant|>')[-1].strip()
    print('Assistant response:')
    print(assistant_response)
