"""
Formal evaluation script for both GPT-2 and TinyLlama models on geography dataset.
"""

import json
import torch
from datetime import datetime
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from .utils import postprocess_sql, normalize_sql, logger

def run_evaluation():
    # Load test data
    with open('/app/data/text2sql-data/data/geography.json', 'r') as f:
        data = json.load(f)

    # Extract test examples (use last 50 examples as test set)
    test_examples = []
    for item in data[-50:]:
        if 'sentences' in item:
            for sent in item.get('sentences', []):
                question = sent.get('text', '')
                sql = item.get('sql', [])
                if isinstance(sql, list):
                    sql = sql[0] if sql else ''
                if question and sql:
                    test_examples.append({'question': question, 'gold_sql': sql})

    print(f'Loaded {len(test_examples)} test examples')

    # Schema
    schema_short = 'state(state_name, population, area, capital, density) | city(city_name, population, country_name, state_name)'
    schema_full = '''state(state_name, population, area, capital, density) | city(city_name, population, country_name, state_name) | river(river_name, length, country_name, traverse) | lake(lake_name, area, country_name, state_name) | mountain(mountain_name, mountain_altitude, country_name, state_name) | border_info(state_name, border) | highlow(state_name, highest_elevation, lowest_point, highest_point, lowest_elevation)'''

    results = {'timestamp': str(datetime.now()), 'models': {}}

    # ========== GPT-2 Evaluation ==========
    print('\n' + '='*60)
    print('EVALUATING FINE-TUNED GPT-2')
    print('='*60)

    gpt2_path = '/app/results/gpt2-geography/final'
    gpt2_tokenizer = AutoTokenizer.from_pretrained(gpt2_path)
    gpt2_model = AutoModelForCausalLM.from_pretrained(gpt2_path)
    gpt2_model.eval()
    if gpt2_tokenizer.pad_token is None:
        gpt2_tokenizer.pad_token = gpt2_tokenizer.eos_token

    gpt2_correct = 0
    gpt2_partial = 0
    gpt2_results = []

    for i, ex in enumerate(test_examples[:20]):  # Limit to 20 for speed
        prompt = f"Table: {schema_short}\nQuestion: {ex['question']}\nSQL:"
        inputs = gpt2_tokenizer(prompt, return_tensors='pt', padding=True)
        with torch.no_grad():
            outputs = gpt2_model.generate(**inputs, max_new_tokens=80, do_sample=False, pad_token_id=gpt2_tokenizer.eos_token_id)
        generated = gpt2_tokenizer.decode(outputs[0], skip_special_tokens=True)
        raw_sql = generated.split('SQL:')[-1].strip()
        pred_sql = postprocess_sql(raw_sql)
        
        # Simple keyword matching for partial credit
        gold_lower = ex['gold_sql'].lower()
        pred_lower = pred_sql.lower()
        
        # Check if main keywords match
        keywords = ['select', 'from', 'where', 'max', 'min', 'count', 'avg', 'sum']
        gold_kw = set(k for k in keywords if k in gold_lower)
        pred_kw = set(k for k in keywords if k in pred_lower)
        keyword_overlap = len(gold_kw & pred_kw) / max(len(gold_kw), 1)
        
        exact_match = normalize_sql(ex['gold_sql']) == normalize_sql(pred_sql)
        if exact_match:
            gpt2_correct += 1
        elif keyword_overlap > 0.5:
            gpt2_partial += 1
        
        gpt2_results.append({
            'question': ex['question'],
            'gold': ex['gold_sql'][:100],
            'pred': pred_sql[:100],
            'exact': exact_match,
            'kw_overlap': keyword_overlap
        })
        
        if (i + 1) % 5 == 0:
            print(f'  GPT-2 progress: {i+1}/20')

    results['models']['gpt2'] = {
        'exact_match': gpt2_correct,
        'partial_match': gpt2_partial,
        'total': len(gpt2_results),
        'accuracy': gpt2_correct / len(gpt2_results) * 100,
        'samples': gpt2_results[:5]  # Save first 5 samples
    }
    print(f'GPT-2: {gpt2_correct}/{len(gpt2_results)} exact match ({gpt2_correct/len(gpt2_results)*100:.1f}%)')
    print(f'GPT-2: {gpt2_partial}/{len(gpt2_results)} partial match')

    # Free memory
    del gpt2_model, gpt2_tokenizer

    # ========== TinyLlama Evaluation ==========
    print('\n' + '='*60)
    print('EVALUATING FINE-TUNED TINYLLAMA')
    print('='*60)

    base_model = AutoModelForCausalLM.from_pretrained(
        'ManthanKulakarni/TinyLlama-1.1B-Text2SQL', 
        torch_dtype=torch.float32, 
        low_cpu_mem_usage=True
    )
    tinyllama_tokenizer = AutoTokenizer.from_pretrained('/app/results/tinyllama-geography/final')
    tinyllama_model = PeftModel.from_pretrained(base_model, '/app/results/tinyllama-geography/final')
    tinyllama_model.eval()
    if tinyllama_tokenizer.pad_token is None:
        tinyllama_tokenizer.pad_token = tinyllama_tokenizer.eos_token

    tl_correct = 0
    tl_partial = 0
    tl_results = []

    for i, ex in enumerate(test_examples[:20]):
        prompt = f'### Schema:\n{schema_full}\n\n### Question:\n{ex["question"]}\n\n### SQL:\n'
        inputs = tinyllama_tokenizer(prompt, return_tensors='pt', padding=True)
        with torch.no_grad():
            outputs = tinyllama_model.generate(**inputs, max_new_tokens=100, do_sample=False, pad_token_id=tinyllama_tokenizer.eos_token_id)
        generated = tinyllama_tokenizer.decode(outputs[0], skip_special_tokens=True)
        raw_sql = generated.split('### SQL:')[-1].strip()
        pred_sql = postprocess_sql(raw_sql)
        
        gold_lower = ex['gold_sql'].lower()
        pred_lower = pred_sql.lower()
        
        keywords = ['select', 'from', 'where', 'max', 'min', 'count', 'avg', 'sum']
        gold_kw = set(k for k in keywords if k in gold_lower)
        pred_kw = set(k for k in keywords if k in pred_lower)
        keyword_overlap = len(gold_kw & pred_kw) / max(len(gold_kw), 1)
        
        exact_match = normalize_sql(ex['gold_sql']) == normalize_sql(pred_sql)
        if exact_match:
            tl_correct += 1
        elif keyword_overlap > 0.5:
            tl_partial += 1
        
        tl_results.append({
            'question': ex['question'],
            'gold': ex['gold_sql'][:100],
            'pred': pred_sql[:100],
            'exact': exact_match,
            'kw_overlap': keyword_overlap
        })
        
        if (i + 1) % 5 == 0:
            print(f'  TinyLlama progress: {i+1}/20')

    results['models']['tinyllama'] = {
        'exact_match': tl_correct,
        'partial_match': tl_partial,
        'total': len(tl_results),
        'accuracy': tl_correct / len(tl_results) * 100,
        'samples': tl_results[:5]
    }
    print(f'TinyLlama: {tl_correct}/{len(tl_results)} exact match ({tl_correct/len(tl_results)*100:.1f}%)')
    print(f'TinyLlama: {tl_partial}/{len(tl_results)} partial match')

    # Save results
    with open('/app/results/formal_evaluation.json', 'w') as f:
        json.dump(results, f, indent=2)

    print('\n' + '='*60)
    print('SUMMARY')
    print('='*60)
    print(f"GPT-2:     {results['models']['gpt2']['accuracy']:.1f}% exact match")
    print(f"TinyLlama: {results['models']['tinyllama']['accuracy']:.1f}% exact match")
    print('Results saved to /app/results/formal_evaluation.json')
    
    return results


if __name__ == '__main__':
    run_evaluation()
