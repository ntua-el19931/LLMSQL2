# LLMSQL2 Technical Documentation

## Architecture, Implementation & Development Guide

**Version:** 1.0  
**Last Updated:** January 31, 2026  
**Project:** LLMSQL2 - Comparison between Text-to-SQL Methods

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [System Architecture](#2-system-architecture)
3. [Technology Stack](#3-technology-stack)
4. [Directory Structure](#4-directory-structure)
5. [Core Modules Deep Dive](#5-core-modules-deep-dive)
6. [Data Pipeline](#6-data-pipeline)
7. [Model Training](#7-model-training)
8. [Evaluation System](#8-evaluation-system)
9. [Web Application](#9-web-application)
10. [Docker Infrastructure](#10-docker-infrastructure)
11. [Database Design](#11-database-design)
12. [Key Algorithms](#12-key-algorithms)
13. [Design Decisions](#13-design-decisions)
14. [Lessons Learned](#14-lessons-learned)

---

## 1. Project Overview

### 1.1 Project Requirements

From the course assignment (LLMSQL2 variant):

| Requirement | Our Implementation |
|-------------|-------------------|
| LLM #1 | GPT-2 (`n22t7a/text2sql-tuned-gpt2`) |
| LLM #2 | TinyLlama (`ManthanKulakarni/TinyLlama-1.1B-Text2SQL`) |
| RDBMS #1 | SQLite (in-memory and file-based) |
| RDBMS #2 | PostgreSQL 15 (Docker container) |
| Datasets | 4: geography, advising, atis, restaurants |
| Metrics | Accuracy (execution, result match) + Efficiency (time) |

### 1.2 What We Built

A complete Text-to-SQL evaluation framework:

```
┌─────────────────────────────────────────────────────────────┐
│                      LLMSQL2 Framework                       │
├─────────────────────────────────────────────────────────────┤
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────┐ │
│  │   Training   │  │  Inference  │  │     Evaluation      │ │
│  │   Pipeline   │  │     API     │  │      System         │ │
│  └─────────────┘  └─────────────┘  └─────────────────────┘ │
│         │                │                    │             │
│         ▼                ▼                    ▼             │
│  ┌─────────────────────────────────────────────────────────┐│
│  │              Database Layer (SQLite + PostgreSQL)       ││
│  └─────────────────────────────────────────────────────────┘│
│         │                │                    │             │
│         ▼                ▼                    ▼             │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────┐ │
│  │  Web Demo   │  │   Reports   │  │    Batch Training   │ │
│  │   (Flask)   │  │  Generator  │  │      Scripts        │ │
│  └─────────────┘  └─────────────┘  └─────────────────────┘ │
└─────────────────────────────────────────────────────────────┘
```

---

## 2. System Architecture

### 2.1 High-Level Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│                          Host Machine                             │
│  ┌────────────────────────────────────────────────────────────┐  │
│  │                    Docker Environment                       │  │
│  │  ┌─────────────────────┐    ┌─────────────────────────┐   │  │
│  │  │   llmsql2-app       │    │   llmsql2-postgres      │   │  │
│  │  │   ─────────────     │    │   ─────────────────     │   │  │
│  │  │   Python 3.10       │◄──►│   PostgreSQL 15         │   │  │
│  │  │   PyTorch           │    │   ───────────────       │   │  │
│  │  │   Transformers      │    │   • geography DB        │   │  │
│  │  │   Flask             │    │   • advising DB         │   │  │
│  │  │   Jupyter           │    │   • atis DB             │   │  │
│  │  │   ───────────       │    │   • restaurants DB      │   │  │
│  │  │   Port: 8888, 5000  │    │   Port: 5432            │   │  │
│  │  └─────────────────────┘    └─────────────────────────┘   │  │
│  │            │                           │                    │  │
│  │            │         Volumes           │                    │  │
│  │            ▼                           ▼                    │  │
│  │  ┌─────────────────────────────────────────────────────┐   │  │
│  │  │  ./src → /app/src                                    │   │  │
│  │  │  ./data → /app/data                                  │   │  │
│  │  │  ./results → /app/results                            │   │  │
│  │  │  ./notebooks → /app/notebooks                        │   │  │
│  │  └─────────────────────────────────────────────────────┘   │  │
│  └────────────────────────────────────────────────────────────┘  │
└──────────────────────────────────────────────────────────────────┘
```

### 2.2 Data Flow

```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│   Natural   │     │   Model     │     │   SQL       │
│  Language   │────►│  Inference  │────►│   Query     │
│  Question   │     │  (GPT2/TL)  │     │             │
└─────────────┘     └─────────────┘     └──────┬──────┘
                                               │
                         ┌─────────────────────┴─────────────────────┐
                         │                                           │
                         ▼                                           ▼
                  ┌─────────────┐                             ┌─────────────┐
                  │   SQLite    │                             │ PostgreSQL  │
                  │  Execution  │                             │  Execution  │
                  └──────┬──────┘                             └──────┬──────┘
                         │                                           │
                         └─────────────────────┬─────────────────────┘
                                               │
                                               ▼
                                        ┌─────────────┐
                                        │   Results   │
                                        │  Comparison │
                                        └─────────────┘
```

---

## 3. Technology Stack

### 3.1 Core Technologies

| Category | Technology | Version | Purpose |
|----------|------------|---------|---------|
| **Language** | Python | 3.10 | Primary development language |
| **Containerization** | Docker | 24.x | Environment isolation |
| **Orchestration** | Docker Compose | 2.x | Multi-container management |

### 3.2 Machine Learning Stack

| Library | Version | Purpose |
|---------|---------|---------|
| PyTorch | ≥2.0.0 | Deep learning framework |
| Transformers | ≥4.36.0 | Hugging Face model hub |
| PEFT | Latest | Parameter-efficient fine-tuning (LoRA) |
| Accelerate | ≥0.25.0 | Training optimization |
| Datasets | ≥2.16.0 | Data loading and processing |

### 3.3 Database Stack

| Technology | Version | Purpose |
|------------|---------|---------|
| PostgreSQL | 15 | Production-grade RDBMS |
| SQLite | 3.x | Lightweight embedded database |
| pg8000 | ≥1.31.0 | Pure Python PostgreSQL driver |
| sqlparse | ≥0.4.4 | SQL parsing and formatting |

### 3.4 Web Stack

| Library | Version | Purpose |
|---------|---------|---------|
| Flask | ≥3.0.0 | Web framework for demo |
| Jinja2 | Built-in | HTML templating |

### 3.5 Utilities

| Library | Purpose |
|---------|---------|
| pandas | Data manipulation |
| numpy | Numerical operations |
| tqdm | Progress bars |
| psutil | System monitoring |
| matplotlib/seaborn | Visualization |

---

## 4. Directory Structure

```
LLMSQL2/
├── docker-compose.yml          # Multi-container orchestration
├── docker-compose.gpu.yml      # GPU variant
├── Dockerfile                  # Main app container
├── Dockerfile.gpu              # GPU-enabled container
├── requirements.txt            # Python dependencies
├── train_all_models.py         # Batch training script
├── test_db_connections.py      # Database connectivity test
│
├── docker/
│   ├── init-db.sql             # PostgreSQL initialization
│   └── entrypoint.sh           # Container startup script
│
├── src/                        # Main source code
│   ├── __init__.py             # Module exports
│   ├── data_loader.py          # Dataset loading
│   ├── model_inference.py      # Model loading & inference
│   ├── train_gpt2.py           # GPT-2 fine-tuning
│   ├── train_tinyllama.py      # TinyLlama LoRA fine-tuning
│   ├── database.py             # Database connections
│   ├── utils.py                # Utility functions
│   ├── evaluation.py           # Basic evaluation
│   ├── execution_eval.py       # Execution-based evaluation
│   ├── comprehensive_eval.py   # Full evaluation system
│   ├── inference_api.py        # Unified inference API
│   ├── web_demo.py             # Flask web interface
│   └── report_generator.py     # Report generation
│
├── data/
│   └── text2sql-data/          # Dataset repository
│       └── data/
│           ├── geography.json  # Training data
│           ├── geography-db.added-in-2020.sqlite
│           ├── advising.json
│           ├── advising-db.added-in-2020.sqlite
│           ├── atis.json
│           ├── atis-db.added-in-2020.sqlite
│           ├── restaurants.json
│           └── restaurants-db.added-in-2020.sqlite
│
├── results/                    # Output directory
│   ├── gpt2-geography/final/   # Trained model checkpoints
│   ├── tinyllama-geography/final/
│   ├── comprehensive_eval_*.json
│   ├── FINAL_REPORT.md
│   └── final_report.json
│
├── notebooks/                  # Jupyter notebooks
│   └── 01_explore_data.ipynb
│
└── docs/                       # Documentation
    ├── USER_GUIDE.md
    └── TECHNICAL_DOCS.md
```

---

## 5. Core Modules Deep Dive

### 5.1 data_loader.py

**Purpose:** Load and preprocess text-to-SQL datasets.

**Key Classes:**

```python
@dataclass
class Text2SQLExample:
    """Single training/test example."""
    question: str        # Natural language question
    sql: str             # Gold SQL query
    schema: str          # Database schema
    database: str        # Database name
```

```python
class Text2SQLDataLoader:
    """Load data from JSON files."""
    
    def load_dataset(self, path: str) -> List[Text2SQLExample]:
        """Load and parse JSON dataset."""
        
    def create_train_test_split(self, examples, test_ratio=0.2):
        """Split into train/test sets."""
```

**Data Format (JSON):**
```json
{
  "sql": ["SELECT state_name FROM state WHERE capital = 'austin'"],
  "sentences": [
    {"text": "What state has Austin as its capital?"}
  ],
  "variables": [...]
}
```

### 5.2 model_inference.py

**Purpose:** Unified interface for loading and using models.

**Key Classes:**

```python
class GPT2Text2SQLModel:
    """GPT-2 based text-to-SQL model."""
    
    def __init__(self, model_path: str):
        self.model = AutoModelForCausalLM.from_pretrained(model_path)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    def generate(self, question: str, schema: str) -> str:
        """Generate SQL from question."""
        prompt = f"Table: {schema}\nQuestion: {question}\nSQL:"
        inputs = self.tokenizer(prompt, return_tensors='pt')
        outputs = self.model.generate(**inputs, max_new_tokens=100)
        return self.tokenizer.decode(outputs[0])
```

```python
class Text2SQLModel:
    """TinyLlama with LoRA adapters."""
    
    def __init__(self, base_model: str, adapter_path: str):
        self.base = AutoModelForCausalLM.from_pretrained(base_model)
        self.model = PeftModel.from_pretrained(self.base, adapter_path)
```

### 5.3 database.py

**Purpose:** Unified database connection handling for SQLite and PostgreSQL.

**Key Design:**

```python
@dataclass
class DatabaseConfig:
    """Database configuration."""
    name: str
    db_type: str  # 'postgresql' or 'sqlite'
    # PostgreSQL settings
    host: Optional[str] = None
    port: Optional[int] = None
    user: Optional[str] = None
    password: Optional[str] = None
    database: Optional[str] = None
    # SQLite settings
    filepath: Optional[str] = None
```

```python
class DatabaseConnection:
    """Unified database connection handler."""
    
    def connect(self):
        if self.config.db_type == "postgresql":
            self.connection = pg8000.connect(...)
        else:  # sqlite
            self.connection = sqlite3.connect(self.config.filepath)
    
    def execute(self, query: str) -> List[Tuple]:
        """Execute query and return results."""
```

**Pre-configured databases:**
```python
DATABASES = {
    "geography_pg": DatabaseConfig(name="geography_pg", db_type="postgresql", ...),
    "geography_sqlite": DatabaseConfig(name="geography_sqlite", db_type="sqlite", ...),
    # ... 8 total configurations
}
```

### 5.4 utils.py

**Purpose:** Shared utility functions.

**Key Functions:**

```python
def normalize_sql(sql: str) -> str:
    """Normalize SQL for comparison."""
    # Uses sqlparse for formatting
    formatted = sqlparse.format(sql, keyword_case='lower', ...)
    return ' '.join(formatted.split())
```

```python
def postprocess_sql(sql: str) -> str:
    """Fix common generation issues."""
    # Fix double SELECT
    sql = re.sub(r'\bSELECT\s+SELECT\b', 'SELECT', sql)
    # Truncate at semicolon
    if ';' in sql:
        sql = sql.split(';')[0].strip() + ';'
    return sql
```

```python
def extract_sql_from_response(response: str) -> str:
    """Extract SQL from model output."""
    # Handle code blocks, markers, etc.
```

### 5.5 comprehensive_eval.py

**Purpose:** Full evaluation against both SQLite and PostgreSQL.

**Key Classes:**

```python
@dataclass
class QueryResult:
    """Result of a single query evaluation."""
    question: str
    gold_sql: str
    predicted_sql: str
    
    # Execution on both backends
    sqlite_result: Optional[List[Tuple]]
    postgres_result: Optional[List[Tuple]]
    sqlite_error: Optional[str]
    postgres_error: Optional[str]
    
    # Timing metrics
    inference_time_ms: float
    sqlite_exec_time_ms: float
    postgres_exec_time_ms: float
    
    # Accuracy flags
    sqlite_executes: bool
    postgres_executes: bool
    sqlite_matches_gold: bool
    postgres_matches_gold: bool
    results_consistent: bool  # Same result on both DBs
```

```python
class ComprehensiveEvaluator:
    """Evaluates against both SQLite and PostgreSQL."""
    
    def evaluate_query(self, question, gold_sql, predicted_sql) -> QueryResult:
        """Evaluate a single prediction."""
        # Execute on SQLite
        sqlite_result = self._execute_on_sqlite(predicted_sql)
        # Execute on PostgreSQL (with case adaptation)
        postgres_sql = self._adapt_sql_for_postgres(predicted_sql)
        postgres_result = self._execute_on_postgres(postgres_sql)
        # Compare results
        ...
```

**PostgreSQL Adaptation:**
```python
def _adapt_sql_for_postgres(self, sql: str) -> str:
    """Convert SQL for PostgreSQL compatibility."""
    # Models generate uppercase identifiers: STATEalias0
    # PostgreSQL uses lowercase: statealias0
    sql_pg = re.sub(r'\b([A-Z]+alias\d+)\b', 
                    lambda m: m.group(1).lower(), sql)
    return sql_pg
```

---

## 6. Data Pipeline

### 6.1 Dataset Source

We use the **text2sql-data** repository:
- Repository: https://github.com/jkkummerfeld/text2sql-data
- Location: `/app/data/text2sql-data/data/`

### 6.2 Data Format

Each dataset consists of:
1. **JSON file** (`geography.json`): Training examples
2. **SQLite file** (`geography-db.added-in-2020.sqlite`): Database for execution

JSON structure:
```json
{
  "sql": ["SELECT river_name FROM river WHERE length > 500"],
  "sentences": [
    {"text": "Which rivers are longer than 500 miles?", "variables": {...}}
  ]
}
```

### 6.3 Preprocessing Pipeline

```
Raw JSON → Parse → Extract Q/SQL pairs → Format prompts → Tokenize → Training
```

**Prompt Templates:**

GPT-2:
```
Table: state(state_name, population, area, capital)
Question: What is the capital of Texas?
SQL:
```

TinyLlama:
```
### Schema:
state(state_name, population, area, capital)

### Question:
What is the capital of Texas?

### SQL:
```

---

## 7. Model Training

### 7.1 GPT-2 Training (train_gpt2.py)

**Base Model:** `n22t7a/text2sql-tuned-gpt2` (82MB)

**Training Approach:** Full fine-tuning (all parameters updated)

**Key Configuration:**
```python
@dataclass
class TrainingConfig:
    model_name: str = 'n22t7a/text2sql-tuned-gpt2'
    num_epochs: int = 5
    batch_size: int = 2
    learning_rate: float = 5e-5
    max_length: int = 256
    warmup_steps: int = 100
    save_steps: int = 500
```

**Training Loop:**
```python
trainer = Trainer(
    model=model,
    args=TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        learning_rate=learning_rate,
        warmup_steps=warmup_steps,
        logging_steps=50,
        save_steps=500,
        save_total_limit=2,
    ),
    train_dataset=train_dataset,
    data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False)
)
trainer.train()
```

### 7.2 TinyLlama Training (train_tinyllama.py)

**Base Model:** `ManthanKulakarni/TinyLlama-1.1B-Text2SQL` (2GB)

**Training Approach:** LoRA (Low-Rank Adaptation) - only adapters updated

**Why LoRA?**
- TinyLlama has 1.1B parameters
- Full fine-tuning would require significant GPU memory
- LoRA only trains ~0.1% of parameters
- Faster training, lower memory usage

**LoRA Configuration:**
```python
from peft import LoraConfig, get_peft_model

lora_config = LoraConfig(
    r=16,                      # Rank of decomposition
    lora_alpha=32,             # Scaling factor
    target_modules=["q_proj", "v_proj"],  # Which layers to adapt
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

model = get_peft_model(base_model, lora_config)
```

**Memory Comparison:**
| Approach | Memory Required | Trainable Params |
|----------|-----------------|------------------|
| Full fine-tuning | ~8GB | 1.1B (100%) |
| LoRA (r=16) | ~4GB | ~1M (0.1%) |

### 7.3 Batch Training (train_all_models.py)

**Design Goals:**
1. Fair comparison: Same epochs for each model-database pair
2. Sequential execution: One training at a time (memory constraints)
3. Logging: Track progress and capture errors

```python
# Training schedule
DATABASES = ["geography", "advising", "atis", "restaurants"]

for database in DATABASES:
    # Train GPT-2 first
    train_gpt2(database)
    # Then TinyLlama
    train_tinyllama(database)
```

---

## 8. Evaluation System

### 8.1 Evaluation Metrics

| Metric | Formula | What It Measures |
|--------|---------|------------------|
| **Execution Accuracy** | (Queries that run) / Total | SQL syntax correctness |
| **Result Accuracy** | (Correct results) / Total | Semantic correctness |
| **Exact Match** | (Identical SQL) / Total | String matching (low) |
| **Inference Time** | End - Start | Computational efficiency |

### 8.2 Evaluation Levels

**Level 1: String Match** (Basic)
```python
def exact_match(gold_sql, predicted_sql):
    return normalize_sql(gold_sql) == normalize_sql(predicted_sql)
```
Problem: Fails on equivalent but differently written SQL.

**Level 2: Execution Success** (Intermediate)
```python
def execution_success(predicted_sql, db_connection):
    try:
        db_connection.execute(predicted_sql)
        return True
    except:
        return False
```
Problem: SQL may run but return wrong results.

**Level 3: Result Match** (Best)
```python
def result_match(gold_sql, predicted_sql, db_connection):
    gold_results = db_connection.execute(gold_sql)
    pred_results = db_connection.execute(predicted_sql)
    return set(gold_results) == set(pred_results)
```
Measures true semantic correctness.

### 8.3 Cross-Database Evaluation

We test on **both** SQLite and PostgreSQL to:
1. Verify SQL portability
2. Compare database performance
3. Handle syntax differences

**Key Challenge: Case Sensitivity**
- SQLite: Case-insensitive identifiers
- PostgreSQL: Lowercase for unquoted identifiers

**Solution:**
```python
def _adapt_sql_for_postgres(self, sql):
    # Convert STATEalias0 → statealias0
    return re.sub(r'\b([A-Z]+alias\d+)\b', 
                  lambda m: m.group(1).lower(), sql)
```

---

## 9. Web Application

### 9.1 Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    Flask Application                     │
├─────────────────────────────────────────────────────────┤
│  Routes:                                                 │
│    GET  /              → Render HTML template           │
│    POST /api/generate  → Generate SQL only              │
│    POST /api/generate_and_execute → Generate + Execute  │
│    GET  /api/models    → List available models          │
│    GET  /api/databases → List databases + schemas       │
│    GET  /api/health    → Health check                   │
└─────────────────────────────────────────────────────────┘
```

### 9.2 Frontend Design

Single-page application with:
- Model/database selection dropdowns
- Question input textarea
- Example questions (clickable)
- SQL output display
- Query results table
- Performance metrics

**Tech:** Pure HTML/CSS/JavaScript (no frontend framework)

### 9.3 Model Caching

Models are cached in memory after first load:

```python
_models: Dict[str, Text2SQLInference] = {}

def get_model(model_type: str, database: str) -> Text2SQLInference:
    cache_key = f"{model_type}-{database}"
    
    if cache_key not in _models:
        inference = Text2SQLInference()
        inference.load_model(model_type, checkpoint_path)
        _models[cache_key] = inference
    
    return _models[cache_key]
```

---

## 10. Docker Infrastructure

### 10.1 Container Design

**llmsql2-app Container:**
```dockerfile
FROM python:3.10-slim

# Install dependencies
COPY requirements.txt .
RUN pip install -r requirements.txt

# Copy source code
COPY src/ /app/src/
COPY data/ /app/data/

WORKDIR /app
```

**llmsql2-postgres Container:**
```yaml
image: postgres:15
environment:
  POSTGRES_USER: postgres
  POSTGRES_PASSWORD: postgres
volumes:
  - ./docker/init-db.sql:/docker-entrypoint-initdb.d/init-db.sql
```

### 10.2 Database Initialization

`docker/init-db.sql` creates all 4 databases:

```sql
-- Create databases
CREATE DATABASE geography;
CREATE DATABASE advising;
CREATE DATABASE atis;
CREATE DATABASE restaurants;

-- Connect and create tables
\c geography
CREATE TABLE state (
    state_name VARCHAR(255) PRIMARY KEY,
    population INTEGER,
    area FLOAT,
    capital VARCHAR(255),
    density FLOAT
);
-- ... more tables
```

### 10.3 Volume Mounts

```yaml
volumes:
  - ./src:/app/src           # Source code (live reload)
  - ./data:/app/data         # Datasets
  - ./results:/app/results   # Persistent model checkpoints
  - ./notebooks:/app/notebooks
```

---

## 11. Database Design

### 11.1 Geography Database

```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│    state    │     │    city     │     │    river    │
├─────────────┤     ├─────────────┤     ├─────────────┤
│ state_name  │◄────│ state_name  │     │ river_name  │
│ population  │     │ city_name   │     │ length      │
│ area        │     │ population  │     │ traverse    │
│ capital     │     │ country_name│     │ country_name│
│ density     │     └─────────────┘     └─────────────┘
└─────────────┘

┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│    lake     │     │  mountain   │     │ border_info │
├─────────────┤     ├─────────────┤     ├─────────────┤
│ lake_name   │     │mountain_name│     │ state_name  │
│ area        │     │ altitude    │     │ border      │
│ state_name  │     │ state_name  │     └─────────────┘
│ country_name│     │ country_name│
└─────────────┘     └─────────────┘

┌─────────────┐
│   highlow   │
├─────────────┤
│ state_name  │
│ highest_elev│
│ lowest_elev │
│ highest_pt  │
│ lowest_pt   │
└─────────────┘
```

### 11.2 Table Sizes

| Database | Tables | Rows (approx) |
|----------|--------|---------------|
| geography | 7 | ~500 |
| advising | 15 | ~5,000 |
| atis | 17-25 | ~10,000 |
| restaurants | 3 | ~200 |

---

## 12. Key Algorithms

### 12.1 SQL Post-Processing

```python
def postprocess_sql(sql: str) -> str:
    """Fix common model output issues."""
    
    # Problem 1: Model outputs "SELECT SELECT * FROM..."
    # Solution: Remove duplicate keyword
    sql = re.sub(r'\bSELECT\s+SELECT\b', 'SELECT', sql, flags=re.IGNORECASE)
    
    # Problem 2: Model generates multiple queries
    # Solution: Keep only first query
    if ';' in sql:
        sql = sql.split(';')[0].strip() + ';'
    
    # Problem 3: Extra whitespace
    sql = ' '.join(sql.split())
    
    return sql
```

### 12.2 Result Comparison

```python
def compare_results(gold_results, pred_results):
    """Compare query results (order-independent)."""
    
    # Convert to sets of string tuples for comparison
    gold_set = set(tuple(str(x) for x in row) for row in gold_results)
    pred_set = set(tuple(str(x) for x in row) for row in pred_results)
    
    return gold_set == pred_set
```

### 12.3 PostgreSQL Transaction Recovery

```python
def _execute_with_timing(self, conn, sql):
    try:
        results = conn.execute(sql)
        return results, None
    except Exception as e:
        # PostgreSQL enters "aborted transaction" state after error
        # Must rollback before next query
        if conn.config.db_type == "postgresql":
            conn.connection.rollback()
        return None, str(e)
```

---

## 13. Design Decisions

### 13.1 Why Docker?

| Alternative | Problems | Docker Solution |
|-------------|----------|-----------------|
| Local Python | Dependency conflicts | Isolated environment |
| Local PostgreSQL | Installation complexity | Pre-configured container |
| Cloud GPUs | Cost, complexity | Runs on CPU (slower but free) |

### 13.2 Why GPT-2 + TinyLlama?

| Model | Size | Pros | Cons |
|-------|------|------|------|
| GPT-2 | 82MB | Fast, low memory | Lower quality |
| TinyLlama | 2GB | Better quality | Slower, more memory |

This gives us a clear **speed vs quality tradeoff** to analyze.

### 13.3 Why LoRA for TinyLlama?

Full fine-tuning 1.1B parameters requires:
- ~8GB GPU memory
- Hours of training time
- Risk of catastrophic forgetting

LoRA trains only ~1M parameters:
- ~4GB memory (CPU feasible)
- Faster convergence
- Preserves base model knowledge

### 13.4 Why Both SQLite and PostgreSQL?

Course requirement, but also:
- SQLite: Fast, no setup, good for testing
- PostgreSQL: Production-realistic, SQL dialect differences
- Comparison: Reveals model SQL generation quality

---

## 14. Lessons Learned

### 14.1 Technical Challenges

**1. Case Sensitivity**
- Problem: Models trained on SQLite syntax fail on PostgreSQL
- Solution: SQL adaptation layer that lowercases identifiers

**2. Transaction Handling**
- Problem: PostgreSQL stays in "aborted" state after error
- Solution: Automatic rollback after each error

**3. Placeholder Values**
- Problem: Models generate `state_name0` instead of `texas`
- Solution: This is expected; would need value extraction from questions

**4. Memory Management**
- Problem: Loading both models exhausts memory
- Solution: Load one, evaluate, delete, load next

### 14.2 What Worked Well

1. **Docker Compose**: Easy reproducibility
2. **Unified Database Interface**: Same code for SQLite/PostgreSQL
3. **Modular Design**: Each module is independent
4. **Comprehensive Metrics**: Capture both accuracy and timing

### 14.3 Future Improvements

1. **GPU Support**: Add CUDA containers for faster training
2. **Value Extraction**: Extract actual values from questions
3. **SQL Validation**: Validate SQL before execution
4. **More Models**: Add GPT-3.5, Llama-3, Qwen
5. **Query Complexity**: Categorize simple vs complex queries

---

## Appendix A: Code Examples

### A.1 Complete Training Example

```python
from src.train_gpt2 import train_gpt2

# Train GPT-2 on geography
train_gpt2(
    data_path="/app/data/text2sql-data/data/geography.json",
    output_dir="/app/results/gpt2-geography",
    epochs=5,
    batch_size=2,
    learning_rate=5e-5
)
```

### A.2 Complete Evaluation Example

```python
from src.comprehensive_eval import run_comprehensive_evaluation

# Evaluate on geography
results = run_comprehensive_evaluation(
    database_name="geography",
    num_samples=50,
    save_results=True
)

print(f"GPT-2 SQLite accuracy: {results['gpt2'].sqlite_result_accuracy:.1%}")
print(f"TinyLlama PostgreSQL accuracy: {results['tinyllama'].postgres_result_accuracy:.1%}")
```

### A.3 Complete Inference Example

```python
from src.inference_api import Text2SQLInference

# Load model
inference = Text2SQLInference()
inference.load_model("gpt2", "/app/results/gpt2-geography/final")

# Generate SQL
result = inference.generate(
    question="What is the capital of Texas?",
    database="geography"
)

print(f"SQL: {result.sql}")
print(f"Time: {result.inference_time:.2f}s")
```

---

## Appendix B: Configuration Reference

### B.1 Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| POSTGRES_HOST | localhost | PostgreSQL host |
| POSTGRES_PORT | 5432 | PostgreSQL port |
| POSTGRES_USER | postgres | Database user |
| POSTGRES_PASSWORD | postgres | Database password |

### B.2 Training Parameters

| Parameter | GPT-2 Default | TinyLlama Default |
|-----------|---------------|-------------------|
| Epochs | 5 | 3 |
| Batch Size | 2 | 2 |
| Learning Rate | 5e-5 | 2e-4 |
| Max Length | 256 | 512 |
| LoRA Rank | N/A | 16 |

---

*End of Technical Documentation*
