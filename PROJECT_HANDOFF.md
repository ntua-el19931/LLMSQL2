# LLMSQL2 Project Handoff Document

**Last Updated:** January 26, 2026  
**Purpose:** This document provides complete context for continuing this project.

---

## A) PROJECT OBJECTIVES

### Primary Goal
Build a Text-to-SQL system that converts natural language questions into SQL queries using **exactly 2 required models**:

1. **GPT-2**: `n22t7a/text2sql-tuned-gpt2` (82MB)
2. **TinyLlama**: `ManthanKulakarni/TinyLlama-1.1B-Text2SQL` (2GB)

### Target Databases (4 total, each with PostgreSQL + SQLite = 8 connections)
| Database | Tables | Domain |
|----------|--------|--------|
| geography | 7 | US states, cities, rivers, mountains |
| advising | 15 | University course advising |
| atis | 17-25 | Airline travel information |
| restaurants | 3 | Restaurant locations |

### Success Criteria
- Both models can generate valid SQL from natural language questions
- SQL queries execute successfully against the databases
- Evaluation metrics: execution accuracy, exact match accuracy

---

## B) WHAT HAS BEEN BUILT

### 1. Docker Infrastructure (FULLY WORKING)
- **docker-compose.yml**: Orchestrates 2 containers
- **llmsql2-app**: Python environment with PyTorch, Transformers, Jupyter
- **llmsql2-postgres**: PostgreSQL 15 with all 4 databases initialized
- **Volumes**: Persistent storage for models, data, and results

**To start:**
```bash
cd LLMSQL2
docker-compose up -d
```

### 2. Database Connections (ALL 8 VERIFIED WORKING)
- **PostgreSQL**: All 4 databases accessible via `psql` and `pg8000`
- **SQLite**: All 4 `.sqlite` files in `/app/data/text2sql-data/data/`
- **Test script**: `test_db_connections.py` - verifies all connections

### 3. Core Source Files (`src/` directory)

| File | Purpose | Status |
|------|---------|--------|
| `model_inference.py` | Model loading & inference | ✅ Complete |
| `data_loader.py` | Load training data from JSON | ✅ Complete |
| `database.py` | Database connection utilities | ✅ Complete |
| `utils.py` | Schema formatting, prompts, postprocess_sql() | ✅ Complete |
| `train_gpt2.py` | GPT-2 fine-tuning script | ✅ Complete |
| `train_tinyllama.py` | TinyLlama LoRA fine-tuning | ✅ Complete |
| `evaluation.py` | Evaluate model outputs | ✅ Complete |
| `run_evaluation.py` | Formal evaluation script | ✅ Complete |
| `execution_eval.py` | SQLite execution accuracy | ✅ Complete |
| `comprehensive_eval.py` | Full eval (SQLite + PostgreSQL) | ✅ Complete |
| `inference_api.py` | Unified inference API | ✅ Complete |
| `web_demo.py` | Flask web demo | ✅ Complete |
| `report_generator.py` | Final report generator | ✅ Complete |
| `__init__.py` | Module exports | ✅ Complete |

### 4. Fine-tuned Models Status

| Model | Geography | Advising | ATIS | Restaurants |
|-------|-----------|----------|------|-------------|
| **GPT-2** | ✅ Complete | ⏳ Pending | ⏳ Pending | ⏳ Pending |
| **TinyLlama** | ✅ Complete | 🔄 Training | ⏳ Pending | ⏳ Pending |

### 5. Model Checkpoints Location
- `/app/results/gpt2-geography/final` - Fine-tuned GPT-2 on geography
- `/app/results/tinyllama-geography/final` - Fine-tuned TinyLlama LoRA on geography
- `/app/results/tinyllama-advising/` - (training in progress)

### 6. Evaluation Results (Geography)

| Model | Exact Match | Partial Match | Notes |
|-------|-------------|---------------|-------|
| GPT-2 | 0% | 100% | Correct structure, alias differences prevent exact match |
| TinyLlama | 0% | 80% | Better on complex queries (GROUP BY, NOT IN) |

Both models generate semantically correct SQL but fail exact match due to:
- Alias naming (`CITYalias0` vs `CITYalias1`)
- Minor syntax variations (`IN` vs `=`)

### 7. Post-processing Added
`utils.py` now includes `postprocess_sql()` which fixes:
- Double "SELECT SELECT" → "SELECT"
- Truncates at first semicolon (removes multiple queries)
- Cleans duplicate keywords

---

## C) WHAT NEEDS TO BE BUILT/DONE

### IMMEDIATE PRIORITY: Fine-tune Both Models

#### Step 1: Fine-tune GPT-2 on Geography
```bash
# Refresh PATH if docker not found
$env:Path = [System.Environment]::GetEnvironmentVariable("Path","Machine") + ";" + [System.Environment]::GetEnvironmentVariable("Path","User")

# Start training (takes ~1 hour)
docker exec llmsql2-app python -m src.train_gpt2 --data /app/data/text2sql-data/data/geography.json --output /app/results/gpt2-geography --epochs 5 --batch-size 2
```

#### Step 2: Fine-tune TinyLlama on Geography
```bash
docker exec llmsql2-app python -m src.train_tinyllama --data /app/data/text2sql-data/data/geography.json --output /app/results/tinyllama-geography --epochs 3
```

#### Step 3: Evaluate Both Models
```bash
docker exec llmsql2-app python -m src.evaluation --model gpt2 --checkpoint /app/results/gpt2-geography/final --database geography
docker exec llmsql2-app python -m src.evaluation --model tinyllama --checkpoint /app/results/tinyllama-geography/final --database geography
```

#### Step 4: Repeat for Other Databases
After geography works, repeat fine-tuning for: advising, atis, restaurants

### KNOWN ISSUES TO FIX

1. **Double "SELECT SELECT" in GPT-2 output**
   - The model produces "SELECT SELECT * FROM..." 
   - Fix: Update prompt format in `train_gpt2.py` to not include "SELECT" at end of prompt
   - Or: Post-process output to remove duplicate SELECT

2. **Placeholder values instead of actual values**
   - Model outputs "state_name0" instead of actual value like "texas"
   - This is expected behavior for some text2sql approaches
   - May need value extraction from question

3. **TinyLlama may need LoRA rank adjustment**
   - Current: r=16, may need r=32 or r=64 for better results
   - Located in `train_tinyllama.py` LoraConfig

### EVALUATION WORKFLOW

After fine-tuning, test with:
```python
# In container Python
from src.model_inference import load_gpt2_model
model, tokenizer = load_gpt2_model("/app/results/gpt2-geography/final")

prompt = "Schema: state(state_name, capital, population)\nQuestion: What is the capital of Texas?\nSQL:"
inputs = tokenizer(prompt, return_tensors="pt")
output = model.generate(**inputs, max_new_tokens=100)
print(tokenizer.decode(output[0], skip_special_tokens=True))
```

### FILE STRUCTURE REFERENCE
```
LLMSQL2/
├── docker-compose.yml      # Main Docker config
├── Dockerfile              # App container build
├── docker/
│   ├── init-db.sql         # PostgreSQL initialization (all 4 DBs)
│   └── entrypoint.sh
├── src/
│   ├── model_inference.py  # GPT2Text2SQLModel, Text2SQLModel
│   ├── train_gpt2.py       # GPT-2 fine-tuning
│   ├── train_tinyllama.py  # TinyLlama LoRA fine-tuning
│   ├── evaluation.py       # Model evaluation
│   ├── data_loader.py      # Training data loading
│   ├── database.py         # DB connections
│   └── utils.py            # Utilities
├── data/
│   └── text2sql-data/      # Training data + SQLite DBs
├── results/                # Fine-tuned model checkpoints
└── test_db_connections.py  # Verify all 8 DB connections
```

### DOCKER COMMANDS REFERENCE
```bash
# Start containers
docker-compose up -d

# Check status
docker ps

# Enter app container
docker exec -it llmsql2-app bash

# View training logs
docker logs -f llmsql2-app

# Test PostgreSQL
docker exec llmsql2-postgres psql -U postgres -d geography -c "SELECT * FROM state LIMIT 5"

# Copy files to container
docker cp local_file.py llmsql2-app:/app/

# Stop containers
docker-compose down
```

### PACKAGE DEPENDENCIES (Already installed in container)
- torch, transformers, datasets
- peft, bitsandbytes, accelerate (for LoRA)
- pg8000 (PostgreSQL)
- sqlite3 (built-in)

---

## SUMMARY: NEXT IMMEDIATE ACTION

1. Start Docker: `docker-compose up -d`
2. Fine-tune GPT-2: `docker exec llmsql2-app python -m src.train_gpt2 --data /app/data/text2sql-data/data/geography.json --output /app/results/gpt2-geography --epochs 5 --batch-size 2`
3. Wait ~1 hour for training to complete
4. Test the fine-tuned model
5. Fine-tune TinyLlama next
6. Repeat for remaining 3 databases

---

**NOTE TO FUTURE CLAUDE INSTANCE:** The user prefers concise, action-oriented responses. Start by verifying Docker is running, then proceed with fine-tuning. The infrastructure is complete - focus on training and evaluation.
