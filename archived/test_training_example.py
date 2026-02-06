"""Test TinyLlama with exact training examples."""
import sys
sys.path.insert(0, '/app')
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

model_name = 'ManthanKulakarni/TinyLlama-1.1B-Text2SQL'
print(f"Loading {model_name}...")
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16)

# EXACT training example 1
table1 = """CREATE TABLE table_47666 (
    "Res." text,
    "Record" text,
    "Opponent" text,
    "Method" text,
    "Round" text
)"""
query1 = 'Which opponent has 1-1 as the record?'
expected1 = """SELECT "Opponent" FROM table_47666 WHERE "Record" = '1-1'"""

prompt = f'### Context:\n{table1}\n### Question:\n{query1}\n### Answer:\n'
print("PROMPT:")
print(prompt)
print("="*60)

inputs = tokenizer(prompt, return_tensors='pt')
with torch.no_grad():
    outputs = model.generate(
        inputs.input_ids, 
        max_new_tokens=60, 
        do_sample=False, 
        pad_token_id=tokenizer.eos_token_id
    )
response = tokenizer.decode(outputs[0], skip_special_tokens=True)
sql = response.split('### Answer:')[-1].strip().split('\n')[0]

print('=== TEST WITH EXACT TRAINING EXAMPLE ===')
print('Query:', query1)
print('Expected:', expected1)
print('Got:', sql)
print()

# Test example 2 from training data
table2 = """CREATE TABLE schedule (
    Cinema_ID int,
    Film_ID int,
    Date text,
    Show_times_per_day int,
    Price float
)"""
query2 = 'What is the average price?'

prompt2 = f'### Context:\n{table2}\n### Question:\n{query2}\n### Answer:\n'
inputs2 = tokenizer(prompt2, return_tensors='pt')
with torch.no_grad():
    outputs2 = model.generate(
        inputs2.input_ids, 
        max_new_tokens=60, 
        do_sample=False, 
        pad_token_id=tokenizer.eos_token_id
    )
response2 = tokenizer.decode(outputs2[0], skip_special_tokens=True)
sql2 = response2.split('### Answer:')[-1].strip().split('\n')[0]

print('=== TEST 2 ===')
print('Query:', query2)
print('Got:', sql2)
