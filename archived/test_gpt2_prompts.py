"""Test different prompt formats for GPT-2 Text2SQL model."""
import sys
sys.path.insert(0, '/app')

from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

model_name = 'n22t7a/text2sql-tuned-gpt2'
print(f"Loading {model_name}...")
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)
model.eval()

def get_eos_prob(prompt):
    """Get probability that model outputs EOS after prompt."""
    inputs = tokenizer(prompt, return_tensors='pt')
    with torch.no_grad():
        logits = model(**inputs).logits
    probs = torch.softmax(logits[0, -1, :], dim=-1)
    eos_prob = probs[tokenizer.eos_token_id].item()
    return eos_prob

def test_generate(prompt, max_tokens=50):
    """Test generation with a prompt."""
    inputs = tokenizer(prompt, return_tensors='pt')
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            do_sample=True,
            temperature=1.5,  # High temperature to escape EOS
            top_p=0.99,
            max_new_tokens=max_tokens,
            pad_token_id=tokenizer.eos_token_id,
        )
    new_tokens = outputs.shape[1] - inputs.input_ids.shape[1]
    generated = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return new_tokens, generated

# Test various prompt formats
prompts = [
    "SELECT",
    "SELECT capital FROM",
    "select * from",
    "Question: what is the capital Answer:",
    "table: state columns: name, capital question: capital of texas query:",
    "CREATE TABLE state (name TEXT, capital TEXT); SELECT",
    "-- SQL query\nSELECT",
    "/*SQL*/ SELECT",
]

print("\n" + "="*60)
print("EOS Probability for different prompts:")
print("="*60)

for prompt in prompts:
    eos_prob = get_eos_prob(prompt)
    print(f"EOS: {eos_prob:.1%} | {prompt[:50]!r}...")

# Test generation with lowest EOS prompt
print("\n" + "="*60)
print("Trying generation with high temperature:")
print("="*60)

test_prompts = [
    "SELECT capital FROM state WHERE",
    "SELECT * FROM",
]

for prompt in test_prompts:
    new_tokens, generated = test_generate(prompt)
    print(f"\nPrompt: {prompt!r}")
    print(f"New tokens: {new_tokens}")
    print(f"Generated: {generated!r}")
