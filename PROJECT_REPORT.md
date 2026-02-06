# LLMSQL2: Text-to-SQL Model Evaluation Project
## Final Project Report

**Date:** February 6, 2026  
**Project:** Evaluation and Fine-tuning of Small Language Models for Text-to-SQL Tasks

---

## 1. Executive Summary

This project evaluated the capability of small, open-source language models to perform Text-to-SQL translation tasks. We investigated two pre-trained models from HuggingFace, discovered they required fine-tuning to be functional, trained them on benchmark datasets, and evaluated their performance using execution-based accuracy metrics.

### Key Findings
- **Pre-trained models do not work out-of-the-box** - Both selected models required fine-tuning to produce valid SQL
- **GPT-2 outperforms TinyLlama** on these tasks after fine-tuning
- **Simpler schemas yield better results** - Geography database (7 tables) achieved significantly better accuracy than ATIS (10+ tables)
- **Execution accuracy is the meaningful metric** - String matching is too strict; execution-based evaluation better reflects real-world utility

---

## 2. Models Evaluated

### 2.1 GPT-2 Text2SQL
- **Model:** `n22t7a/text2sql-tuned-gpt2`
- **Architecture:** GPT-2 (124M parameters)
- **Type:** Full fine-tuning
- **Prompt Format:** `Table: {schema}\nQuestion: {question}\nSQL:`

### 2.2 TinyLlama Text2SQL
- **Model:** `ManthanKulakarni/TinyLlama-1.1B-Text2SQL`
- **Architecture:** TinyLlama 1.1B with LoRA adapters
- **Type:** Parameter-efficient fine-tuning (LoRA)
- **Prompt Format:** `### Schema:\n{schema}\n\n### Question:\n{question}\n\n### SQL:\n`

---

## 3. Datasets

We used benchmark datasets from the [text2sql-data](https://github.com/jkkummerfeld/text2sql-data) repository:

| Dataset | Tables | Complexity | Training Examples |
|---------|--------|------------|-------------------|
| **Geography** | 7 | Low | ~600 |
| **ATIS** | 10+ | High | ~4,400 |
| Advising | 8 | Medium | ~3,800 |
| Restaurants | 3 | Low | ~300 |

**Note:** Training was completed on Geography and ATIS datasets. Advising and Restaurants were not trained due to time constraints.

---

## 4. Methodology

### 4.1 Initial Discovery: Pre-trained Models Don't Work

Our initial hypothesis was that the pre-trained HuggingFace models would work directly for Text-to-SQL tasks. **This proved incorrect:**

- **GPT-2 model** (`n22t7a/text2sql-tuned-gpt2`): Discovered to be a SQL code completer, not a Text-to-SQL translator. It outputs EOS token immediately after natural language input.
  
- **TinyLlama model** (`ManthanKulakarni/TinyLlama-1.1B-Text2SQL`): Produces malformed SQL with hallucinated columns and infinite repetitions without fine-tuning.

### 4.2 Fine-tuning Approach

Given the pre-trained models' limitations, we fine-tuned both models:

- **GPT-2:** Full model fine-tuning using HuggingFace Trainer
- **TinyLlama:** LoRA (Low-Rank Adaptation) fine-tuning for memory efficiency
- **Training Environment:** Google Colab with GPU (T4/A100)
- **Training Epochs:** 3-5 epochs per dataset

### 4.3 Evaluation Metrics

We employed three evaluation metrics:

1. **Exact Match:** String comparison of predicted vs. gold SQL (normalized)
2. **Execution Success:** Whether the predicted SQL executes without errors
3. **Result Match:** Whether the predicted SQL returns the same results as gold SQL

---

## 5. Results

### 5.1 Final Evaluation Results (256 max tokens)

| Model | Database | Exec Success | Result Match |
|-------|----------|--------------|--------------|
| **GPT-2** | Geography | **85%** | **35%** |
| **GPT-2** | ATIS | **30%** | **20%** |
| **TinyLlama** | Geography | **75%** | **30%** |
| **TinyLlama** | ATIS | 0% | 0% |

### 5.2 Key Observations

#### GPT-2 Performance
- Achieved **85% execution success** on Geography dataset
- **35% of queries return correct results**
- Performs reasonably on ATIS (30% execution) despite complexity
- Benefits from simpler prompt format

#### TinyLlama Performance
- Achieved **75% execution success** on Geography (with optimized settings)
- Completely fails on ATIS (0%) - queries too complex for the model
- **Requires beam search** for better results
- **Shorter schemas improve performance** significantly

### 5.3 Impact of Generation Parameters

For TinyLlama, we tested various configurations:

| Configuration | Exec Success | Result Match |
|--------------|--------------|--------------|
| Full schema + greedy | 40% | 15% |
| **Concise schema + beam search** | **75%** | **30%** |

---

## 6. Technical Insights

### 6.1 Why Pre-trained Models Failed

1. **Training Data Mismatch:** The pre-trained models were trained on different datasets/formats than our evaluation set
2. **Prompt Format Sensitivity:** Models are extremely sensitive to exact prompt formatting
3. **Domain Specificity:** Text-to-SQL requires domain-specific fine-tuning

### 6.2 Why ATIS is Harder

1. **Schema Complexity:** 10+ tables with many foreign key relationships
2. **Query Complexity:** Requires multiple JOINs, subqueries, and date handling
3. **Token Length:** Complex queries exceed model's effective generation capacity
4. **Ambiguity:** Flight booking queries have many valid SQL representations

### 6.3 Optimal Settings Discovered

**For GPT-2:**
- Prompt: `Table: {short_schema}\nQuestion: {question}\nSQL:`
- Generation: Greedy decoding, max_new_tokens=256
- Schema: Short/concise format

**For TinyLlama:**
- Prompt: `### Schema:\n{schema}\n\n### Question:\n{question}\n\n### SQL:\n`
- Generation: **Beam search (num_beams=3)**, max_new_tokens=256
- Schema: **Short/concise format** (not full schema)

---

## 7. Project Architecture

### 7.1 Directory Structure

```
LLMSQL2/
├── src/
│   ├── train_gpt2.py          # GPT-2 fine-tuning script
│   ├── train_tinyllama.py     # TinyLlama LoRA fine-tuning
│   ├── evaluate_docker.py     # Main evaluation script
│   ├── data_loader.py         # Dataset loading utilities
│   ├── database.py            # Database connections
│   ├── inference_api.py       # Unified inference API
│   └── utils.py               # Helper functions
├── training_results/          # Fine-tuned model checkpoints
│   ├── gpt2-geography/
│   ├── gpt2-atis/
│   ├── tinyllama-geography/
│   └── tinyllama-atis/
├── data/text2sql-data/        # Benchmark datasets
├── results/                   # Evaluation results (JSON)
├── docker-compose.yml         # Docker configuration
└── Dockerfile                 # Container definition
```

### 7.2 Docker Environment

The project is fully containerized:
- **PostgreSQL 15:** Production database
- **Python 3.10:** With PyTorch, Transformers, PEFT
- **Jupyter Lab:** For interactive development

---

## 8. Conclusions

### 8.1 Main Conclusions

1. **Small LLMs can perform Text-to-SQL** but require task-specific fine-tuning
2. **GPT-2 (124M params) outperforms TinyLlama (1.1B params)** on this task, suggesting model architecture matters more than size
3. **Schema complexity is the primary difficulty factor** - simple schemas (Geography) work well; complex schemas (ATIS) are challenging
4. **35% result accuracy is achievable** with small models on simple domains
5. **Generation parameters significantly impact results** - beam search improved TinyLlama by 35 percentage points

### 8.2 Limitations

- Only tested on 2 of 4 planned databases
- Limited training epochs due to compute constraints
- No comparison with larger models (GPT-3.5, GPT-4)
- Evaluation on 20 samples per model (statistical significance limited)

### 8.3 Future Work

1. **Train on remaining datasets:** Advising and Restaurants
2. **Increase training epochs** for ATIS to improve complex query handling
3. **Try larger models:** Llama-2-7B, Mistral-7B
4. **Schema linking:** Pre-identify relevant tables before generation
5. **SQL correction:** Post-process generated SQL to fix common errors

---

## 9. How to Reproduce

### 9.1 Setup
```bash
# Clone repository
git clone <repo-url>
cd LLMSQL2

# Get datasets
cd data
git clone https://github.com/jkkummerfeld/text2sql-data.git
cd ..

# Start Docker
docker-compose up -d

# Install dependencies
docker exec llmsql2-app pip install peft --quiet
```

### 9.2 Run Evaluation
```bash
docker exec llmsql2-app python /app/src/evaluate_docker.py --samples 20
```

### 9.3 Train New Models (on Colab)
Use the provided notebook: `colab_training_with_dataset_clone (1).ipynb`

---

## 10. References

1. text2sql-data repository: https://github.com/jkkummerfeld/text2sql-data
2. TinyLlama-1.1B-Text2SQL: https://huggingface.co/ManthanKulakarni/TinyLlama-1.1B-Text2SQL
3. text2sql-tuned-gpt2: https://huggingface.co/n22t7a/text2sql-tuned-gpt2
4. PEFT (LoRA): https://github.com/huggingface/peft

---

*Report generated: February 6, 2026*
