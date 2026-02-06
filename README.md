# LLMSQL2 - Text-to-SQL Model Evaluation Project

## Overview
This project evaluates and compares Text-to-SQL language models on benchmark datasets.
Fully containerized with Docker for easy deployment.

## Components
- **TinyLlama-1.1B-Text2SQL**: Fine-tuned TinyLlama for Text-to-SQL
- **GPT (OpenAI)**: Optional GPT-3.5/GPT-4 integration
- **SQLite**: In-memory database for quick testing
- **PostgreSQL**: Production-ready database (Docker)

## Datasets
Using datasets from [text2sql-data](https://github.com/jkkummerfeld/text2sql-data):
- Spider (cross-domain)
- Geography
- ATIS (flight booking)
- Academic
- And more...

## Project Structure
```
LLMSQL2/
├── docker/                  # Docker configuration
│   ├── init-db.sql          # PostgreSQL initialization
│   └── entrypoint.sh        # Container entrypoint
├── data/                    # Datasets
├── src/
│   ├── data_loader.py       # Load and preprocess datasets
│   ├── model_inference.py   # Model loading and inference
│   ├── evaluation.py        # Evaluation metrics
│   └── utils.py             # Helper functions
├── notebooks/               # Jupyter notebooks
├── results/                 # Evaluation results
├── Dockerfile               # CPU Docker image
├── Dockerfile.gpu           # GPU Docker image (NVIDIA)
├── docker-compose.yml       # Main compose file
├── docker-compose.gpu.yml   # GPU compose file
├── run.bat                  # Windows helper script
├── requirements.txt         # Dependencies
└── README.md               # This file
```

## Quick Start with Docker (Recommended)

### Prerequisites
- Docker Desktop installed

### Start the Application
```bash
# Windows
run.bat up

# Or manually
docker-compose up -d
```

### Access
- **Jupyter Lab**: http://localhost:8888
- **PostgreSQL**: localhost:5432 (user: postgres, password: postgres)

### Stop
```bash
run.bat down
# or
docker-compose down
```

### GPU Support (NVIDIA)
```bash
run.bat gpu
# or
docker-compose -f docker-compose.gpu.yml up -d
```

## Manual Setup (Without Docker)

### 1. Create Virtual Environment
```bash
python -m venv venv
venv\Scripts\activate  # Windows
source venv/bin/activate  # Linux/Mac
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Clone Dataset Repository
```bash
cd data
git clone https://github.com/jkkummerfeld/text2sql-data.git
```

### 4. Download Models
Models will be downloaded automatically from HuggingFace on first use.

## Usage

### Run Evaluation
```python
from src.model_inference import Text2SQLModel
from src.evaluation import evaluate_model

model = Text2SQLModel("ManthanKulakarni/TinyLlama-1.1B-Text2SQL")
results = evaluate_model(model, dataset="spider")
```

## References
- [Improving Text-to-SQL Evaluation Methodology (ACL 2018)](http://aclweb.org/anthology/P18-1033)
- [TinyLlama-1.1B-Text2SQL](https://huggingface.co/ManthanKulakarni/TinyLlama-1.1B-Text2SQL)
- [qwen2.5-coder-text2nosql](https://huggingface.co/frtcek95/qwen2.5-coder-text2nosql)

## Contributors
- [@ntua-el19931](https://github.com/ntua-el19931)
- [@Sskarm](https://github.com/Sskarm)
- [@dimtze03](https://github.com/dimtze03)
