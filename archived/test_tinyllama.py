"""Test TinyLlama Text2SQL model with various queries."""
import sys
sys.path.insert(0, '/app')

from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

model_name = 'ManthanKulakarni/TinyLlama-1.1B-Text2SQL'
print(f"Loading {model_name}...")
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16)
model.eval()

# Test with simple geography-like schema
schema = '''CREATE TABLE state (state_name TEXT, capital TEXT, population INT)'''

tests = [
    'what is the capital of texas',
    'which state has the largest population', 
    'how many states are there',
    'list all states',
]

print('Testing TinyLlama with simplified schema...')
print(f'Schema: {schema}')
print()

for question in tests:
    prompt = f'''### Context:
{schema}
### Question:
{question}
### Answer:
'''
    inputs = tokenizer(prompt, return_tensors='pt')
    with torch.no_grad():
        outputs = model.generate(
            inputs.input_ids,
            max_new_tokens=80,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id
        )
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    sql = response.split('### Answer:')[-1].strip().split('\n')[0]
    
    print(f'Q: {question}')
    print(f'SQL: {sql}')
    print()
