# LLMSQL2 User Guide

## Complete How-To Guide for Text-to-SQL Evaluation

**Version:** 1.0  
**Last Updated:** January 31, 2026  
**Project:** LLMSQL2 - Comparison between Text-to-SQL Methods

---

## Table of Contents

1. [Introduction](#1-introduction)
2. [Prerequisites](#2-prerequisites)
3. [Installation & Setup](#3-installation--setup)
4. [Training Models](#4-training-models)
5. [Running Evaluations](#5-running-evaluations)
6. [Using the Web Demo](#6-using-the-web-demo)
7. [Generating Reports](#7-generating-reports)
8. [Database Operations](#8-database-operations)
9. [Troubleshooting](#9-troubleshooting)
10. [Quick Reference](#10-quick-reference)

---

## 1. Introduction

### What is LLMSQL2?

LLMSQL2 is a comprehensive framework for evaluating and comparing Text-to-SQL language models. It converts natural language questions into SQL queries using fine-tuned models.

### Project Objectives

According to the course requirements (LLMSQL2 variant):
- **Models:** GPT-2 and TinyLlama (Tiny-LLaMA)
- **Databases:** SQLite and PostgreSQL
- **Datasets:** At least 3 from text2sql-data repository
- **Metrics:** Accuracy (correctness) and computational efficiency (time/resources)

### What You Can Do

1. **Train** GPT-2 and TinyLlama models on text-to-SQL datasets
2. **Evaluate** model performance on both SQLite and PostgreSQL
3. **Compare** models across accuracy and speed metrics
4. **Demo** the system via a web interface
5. **Generate** formal comparison reports

---

## 2. Prerequisites

### Required Software

| Software | Version | Purpose |
|----------|---------|---------|
| Docker Desktop | Latest | Container runtime |
| Git | Latest | Version control |
| Python | 3.10+ | Local scripts (optional) |

### Hardware Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| RAM | 8 GB | 16 GB |
| Storage | 20 GB | 50 GB |
| GPU | Not required | NVIDIA (for faster training) |

### Verify Docker Installation

```powershell
# Check Docker is installed and running
docker --version
docker ps
```

Expected output:
```
Docker version 24.x.x, build xxxxx
CONTAINER ID   IMAGE   COMMAND   CREATED   STATUS   PORTS   NAMES
```

---

## 3. Installation & Setup

### Step 1: Clone or Navigate to Project

```powershell
cd D:\ASPS\LLMSQL2
```

### Step 2: Start Docker Containers

```powershell
# Start all containers (PostgreSQL + Python app)
docker-compose up -d
```

This starts two containers:
- `llmsql2-app`: Python environment with PyTorch, Transformers, Jupyter
- `llmsql2-postgres`: PostgreSQL 15 with all 4 databases initialized

### Step 3: Verify Containers Are Running

```powershell
docker ps --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}"
```

Expected output:
```
NAMES              STATUS           PORTS
llmsql2-app        Up X minutes     0.0.0.0:8888->8888/tcp
llmsql2-postgres   Up X minutes     0.0.0.0:5432->5432/tcp
```

### Step 4: Verify Database Connections

```powershell
# Test all database connections
docker exec llmsql2-app python test_db_connections.py
```

### Step 5: Access Jupyter Lab (Optional)

Open browser to: http://localhost:8888

---

## 4. Training Models

### 4.1 Understanding the Training Process

We fine-tune two models:
- **GPT-2** (`n22t7a/text2sql-tuned-gpt2`): 82MB, fast inference
- **TinyLlama** (`ManthanKulakarni/TinyLlama-1.1B-Text2SQL`): 2GB, better quality

On four databases:
- `geography`: US states, cities, rivers (7 tables)
- `advising`: University course advising (15 tables)
- `atis`: Airline travel information (17-25 tables)
- `restaurants`: Restaurant locations (3 tables)

### 4.2 Train a Single Model

#### Train GPT-2 on Geography

```powershell
docker exec llmsql2-app python -m src.train_gpt2 `
    --data /app/data/text2sql-data/data/geography.json `
    --output /app/results/gpt2-geography `
    --epochs 5 `
    --batch-size 2
```

**Parameters:**
- `--data`: Path to training data JSON file
- `--output`: Where to save model checkpoints
- `--epochs`: Number of training epochs (5 recommended)
- `--batch-size`: Batch size (2 for limited memory)

**Expected time:** ~1 hour on CPU

#### Train TinyLlama on Geography

```powershell
docker exec llmsql2-app python -m src.train_tinyllama `
    --data /app/data/text2sql-data/data/geography.json `
    --output /app/results/tinyllama-geography `
    --epochs 3
```

**Expected time:** ~2-3 hours on CPU

### 4.3 Train All Models (Batch)

To train both models on all 4 databases sequentially:

```powershell
# Start batch training in background window
Start-Process powershell -ArgumentList "-NoExit", "-Command", "cd D:\ASPS\LLMSQL2; python train_all_models.py 2>&1 | Tee-Object -FilePath training_batch.log"
```

**Monitor progress:**
```powershell
Get-Content D:\ASPS\LLMSQL2\training_batch.log -Tail 10 -Wait
```

**Total time:** 8-10 hours for all 8 training runs

### 4.4 Check Training Status

```powershell
# View last 10 lines of training log
Get-Content D:\ASPS\LLMSQL2\training_batch.log -Tail 10

# List trained model checkpoints
docker exec llmsql2-app ls -la /app/results/
```

### 4.5 Training Output

After training, you'll have:
```
/app/results/
├── gpt2-geography/final/          # GPT-2 fine-tuned on geography
├── gpt2-advising/final/           # GPT-2 fine-tuned on advising
├── gpt2-atis/final/               # GPT-2 fine-tuned on atis
├── gpt2-restaurants/final/        # GPT-2 fine-tuned on restaurants
├── tinyllama-geography/final/     # TinyLlama fine-tuned on geography
├── tinyllama-advising/final/      # ...
├── tinyllama-atis/final/
└── tinyllama-restaurants/final/
```

---

## 5. Running Evaluations

### 5.1 Types of Evaluation

| Type | What It Measures | Output |
|------|------------------|--------|
| **String Match** | Exact SQL text match | Low accuracy (alias differences) |
| **Execution** | SQL runs without error | Execution success rate |
| **Result Match** | Query returns correct data | True accuracy metric |

### 5.2 Run Comprehensive Evaluation (Recommended)

Evaluates both models on both SQLite AND PostgreSQL:

```powershell
# Evaluate on a single database
docker exec llmsql2-app python -m src.comprehensive_eval --database geography --samples 50

# Evaluate on ALL databases
docker exec llmsql2-app python -m src.comprehensive_eval --samples 30
```

**Parameters:**
- `--database` / `-d`: Specific database (geography, advising, atis, restaurants)
- `--samples` / `-n`: Number of test samples per database (default: 30)

**Output:**
```
============================================================
RESULTS SUMMARY
============================================================
Metric                         GPT-2                TinyLlama
----------------------------------------------------------------------
Total Queries                  30                   30
SQLite Execution Accuracy      70.0%                65.0%
PostgreSQL Execution Accuracy  55.0%                60.0%
SQLite Result Match            45.0%                50.0%
PostgreSQL Result Match        40.0%                48.0%
Avg Inference Time (ms)        150                  1200
============================================================
```

### 5.3 View Evaluation Results

```powershell
# View JSON results
docker exec llmsql2-app cat /app/results/comprehensive_eval_geography.json

# Copy results to local machine
docker cp llmsql2-app:/app/results/comprehensive_eval_geography.json ./results/
```

### 5.4 Run Quick Execution Test

```powershell
# Quick test with 5 samples
docker exec llmsql2-app python -m src.execution_eval
```

---

## 6. Using the Web Demo

### 6.1 Start the Web Server

```powershell
docker exec -d llmsql2-app python -m src.web_demo
```

Or in foreground (see logs):
```powershell
docker exec llmsql2-app python -m src.web_demo
```

### 6.2 Access the Demo

Open browser to: **http://localhost:5000**

### 6.3 Using the Interface

1. **Select Model**: Choose GPT-2 (fast) or TinyLlama (better quality)
2. **Select Database**: Choose the domain (geography, advising, etc.)
3. **Enter Question**: Type natural language question
4. **Generate SQL**: Click to generate SQL only
5. **Generate & Execute**: Generate SQL and run against database

### 6.4 Example Questions

**Geography:**
- "What is the capital of Texas?"
- "Which states border California?"
- "What is the largest city in the US?"
- "How many rivers are longer than 500 miles?"

**Advising:**
- "What courses are offered in Fall 2024?"
- "Which instructors teach computer science?"

**ATIS:**
- "Show me flights from Boston to Denver"
- "What airlines fly to Los Angeles?"

### 6.5 Stop the Web Server

```powershell
# Find and stop the process
docker exec llmsql2-app pkill -f "python -m src.web_demo"
```

---

## 7. Generating Reports

### 7.1 Generate Final Comparison Report

```powershell
docker exec llmsql2-app python -m src.report_generator
```

This creates two files:
- `/app/results/FINAL_REPORT.md` - Markdown format
- `/app/results/final_report.json` - JSON format

### 7.2 View the Report

```powershell
# View in terminal
docker exec llmsql2-app cat /app/results/FINAL_REPORT.md

# Copy to local machine
docker cp llmsql2-app:/app/results/FINAL_REPORT.md ./
```

### 7.3 Report Contents

The report includes:
- Executive summary
- Overall results comparison table
- Per-database breakdown
- Performance analysis (inference speed)
- Database backend comparison (SQLite vs PostgreSQL)
- Key findings and recommendations

---

## 8. Database Operations

### 8.1 Connect to PostgreSQL

```powershell
# Interactive psql session
docker exec -it llmsql2-postgres psql -U postgres -d geography

# Run a query directly
docker exec llmsql2-postgres psql -U postgres -d geography -c "SELECT * FROM state LIMIT 5"
```

### 8.2 Available Databases

| Database | Tables | Domain |
|----------|--------|--------|
| geography | 7 | US states, cities, rivers, mountains |
| advising | 15 | University course advising |
| atis | 17-25 | Airline travel information |
| restaurants | 3 | Restaurant locations |

### 8.3 List Tables in a Database

```powershell
# PostgreSQL
docker exec llmsql2-postgres psql -U postgres -d geography -c "\dt"

# SQLite
docker exec llmsql2-app sqlite3 /app/data/text2sql-data/data/geography-db.added-in-2020.sqlite ".tables"
```

### 8.4 View Table Schema

```powershell
# PostgreSQL
docker exec llmsql2-postgres psql -U postgres -d geography -c "\d state"

# SQLite
docker exec llmsql2-app sqlite3 /app/data/text2sql-data/data/geography-db.added-in-2020.sqlite ".schema state"
```

---

## 9. Troubleshooting

### 9.1 Docker Issues

**Problem:** "Cannot connect to Docker daemon"
```powershell
# Solution: Start Docker Desktop
Start-Process "C:\Program Files\Docker\Docker\Docker Desktop.exe"
Start-Sleep -Seconds 30
```

**Problem:** Containers not starting
```powershell
# Solution: Rebuild containers
docker-compose down
docker-compose up -d --build
```

### 9.2 Training Issues

**Problem:** Out of memory
```powershell
# Solution: Reduce batch size
docker exec llmsql2-app python -m src.train_gpt2 --batch-size 1 ...
```

**Problem:** Training stuck
```powershell
# Solution: Check logs
docker logs llmsql2-app --tail 50
```

### 9.3 Evaluation Issues

**Problem:** "Model checkpoint not found"
```powershell
# Solution: Verify checkpoint exists
docker exec llmsql2-app ls -la /app/results/gpt2-geography/final/
```

**Problem:** PostgreSQL "transaction aborted" errors
- This is handled automatically by the comprehensive_eval module
- The code includes automatic rollback after errors

### 9.4 Database Issues

**Problem:** "Connection refused" to PostgreSQL
```powershell
# Solution: Check container is running
docker ps | Select-String "postgres"

# Restart if needed
docker restart llmsql2-postgres
```

---

## 10. Quick Reference

### Essential Commands

```powershell
# Start everything
docker-compose up -d

# Stop everything
docker-compose down

# Check status
docker ps

# View logs
docker logs llmsql2-app --tail 50

# Enter container shell
docker exec -it llmsql2-app bash

# Train GPT-2
docker exec llmsql2-app python -m src.train_gpt2 --data /app/data/text2sql-data/data/geography.json --output /app/results/gpt2-geography --epochs 5

# Train TinyLlama
docker exec llmsql2-app python -m src.train_tinyllama --data /app/data/text2sql-data/data/geography.json --output /app/results/tinyllama-geography --epochs 3

# Run evaluation
docker exec llmsql2-app python -m src.comprehensive_eval

# Generate report
docker exec llmsql2-app python -m src.report_generator

# Start web demo
docker exec llmsql2-app python -m src.web_demo
```

### File Locations (Inside Container)

| Path | Description |
|------|-------------|
| `/app/src/` | Source code |
| `/app/data/text2sql-data/data/` | Datasets and SQLite files |
| `/app/results/` | Model checkpoints and evaluation results |
| `/app/notebooks/` | Jupyter notebooks |

### Ports

| Port | Service |
|------|---------|
| 5000 | Web Demo |
| 5432 | PostgreSQL |
| 8888 | Jupyter Lab |

---

## Appendix A: Complete Workflow Example

```powershell
# 1. Start Docker
docker-compose up -d

# 2. Wait for containers
Start-Sleep -Seconds 30

# 3. Verify everything is running
docker ps

# 4. Train GPT-2 on geography (1 hour)
docker exec llmsql2-app python -m src.train_gpt2 --data /app/data/text2sql-data/data/geography.json --output /app/results/gpt2-geography --epochs 5 --batch-size 2

# 5. Train TinyLlama on geography (2-3 hours)
docker exec llmsql2-app python -m src.train_tinyllama --data /app/data/text2sql-data/data/geography.json --output /app/results/tinyllama-geography --epochs 3

# 6. Run comprehensive evaluation
docker exec llmsql2-app python -m src.comprehensive_eval --database geography --samples 50

# 7. Generate report
docker exec llmsql2-app python -m src.report_generator

# 8. View results
docker exec llmsql2-app cat /app/results/FINAL_REPORT.md

# 9. (Optional) Start web demo
docker exec -d llmsql2-app python -m src.web_demo
# Open http://localhost:5000

# 10. Stop when done
docker-compose down
```

---

*End of User Guide*
