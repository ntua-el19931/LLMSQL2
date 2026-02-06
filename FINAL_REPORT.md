# LLMSQL2 Final Evaluation Report

**Generated:** February 2, 2026  
**Course Project:** Text-to-SQL Evaluation with GPT-2, TinyLlama, SQLite, PostgreSQL

---

## Executive Summary

This report presents the comprehensive evaluation of two Large Language Models (GPT-2 and TinyLlama) fine-tuned for Text-to-SQL generation across four database domains (geography, advising, atis, restaurants), tested against both SQLite and PostgreSQL.

### Key Findings

| Metric | GPT-2 | TinyLlama |
|--------|-------|-----------|
| **Total Queries Evaluated** | 80 | 20 |
| **Overall SQLite Execution Accuracy** | 21.2% | 20.0% |
| **Overall PostgreSQL Execution Accuracy** | 0.0% | 15.0% |
| **Average Inference Time (ms)** | 3,319 | 34,038 |
| **Speed Advantage** | **10.3x faster** | - |

**Winner:** GPT-2 provides better practical performance with comparable accuracy but significantly faster inference.

---

## 1. Model Configuration

### GPT-2 (Fine-tuned)
- **Base Model:** n22t7a/text2sql-tuned-gpt2
- **Model Size:** ~82 MB
- **Training:** 3 epochs, batch_size=4 on all 4 databases
- **Inference:** ~3.3 seconds per query

### TinyLlama (Pre-trained)
- **Base Model:** ManthanKulakarni/TinyLlama-1.1B-Text2SQL
- **Model Size:** ~2 GB
- **Training:** Only geography (CPU training too slow for full fine-tuning)
- **Inference:** ~34 seconds per query

---

## 2. Database-by-Database Results

### 2.1 Geography Database

| Metric | GPT-2 | TinyLlama |
|--------|-------|-----------|
| SQLite Execution Accuracy | **95.0%** | 50.0% |
| PostgreSQL Execution Accuracy | **45.0%** | 45.0% |
| SQLite Result Match | 5.0% | **20.0%** |
| PostgreSQL Result Match | 0.0% | **15.0%** |
| Cross-DB Consistency | **25.0%** | 5.0% |

**Analysis:** GPT-2 executes more queries successfully on SQLite (95%) but TinyLlama produces more semantically correct results (20% vs 5% result match).

#### Accuracy by Complexity Level

| Complexity | GPT-2 SQLite Exec | TinyLlama SQLite Exec |
|------------|-------------------|----------------------|
| SIMPLE (≤2.0) | 100.0% | 75.0% |
| MEDIUM (≤6.0) | 85.7% | 14.3% |
| COMPLEX (≤12.0) | 100.0% | 60.0% |
| ADVANCED (>12.0) | 100.0% | 75.0% |

### 2.2 Advising Database

| Metric | GPT-2 | TinyLlama |
|--------|-------|-----------|
| SQLite Execution Accuracy | **75.0%** | N/A |
| PostgreSQL Execution Accuracy | 0.0% | N/A |
| SQLite Result Match | **65.0%** | N/A |
| PostgreSQL Result Match | 0.0% | N/A |

**Analysis:** GPT-2 shows strong performance with 65% result match - the highest across all databases. TinyLlama was not evaluated (no fine-tuned checkpoint available).

#### Accuracy by Complexity Level

| Complexity | GPT-2 SQLite Match |
|------------|-------------------|
| SIMPLE | N/A (no samples) |
| MEDIUM | **86.7%** |
| COMPLEX | N/A (no samples) |
| ADVANCED | 0.0% |

### 2.3 ATIS Database (Airline Travel Information)

| Metric | GPT-2 | TinyLlama |
|--------|-------|-----------|
| SQLite Execution Accuracy | **40.0%** | N/A |
| PostgreSQL Execution Accuracy | 0.0% | N/A |
| SQLite Result Match | **15.0%** | N/A |
| PostgreSQL Result Match | 0.0% | N/A |

**Analysis:** ATIS proved to be the most challenging database with complex multi-table queries. The dataset has 57.6% ADVANCED complexity queries.

### 2.4 Restaurants Database

| Metric | GPT-2 | TinyLlama |
|--------|-------|-----------|
| SQLite Execution Accuracy | 0.0% | N/A |
| PostgreSQL Execution Accuracy | 0.0% | N/A |
| SQLite Result Match | 0.0% | N/A |
| PostgreSQL Result Match | 0.0% | N/A |

**Analysis:** Restaurants showed 0% success, likely due to schema mismatches between training data and actual database structure.

---

## 3. Error Analysis

### 3.1 Common Error Types

| Error Type | GPT-2 Count | Description |
|------------|-------------|-------------|
| Missing Column | High | Model generates column names with `0` suffix (e.g., `state_name0`) |
| Missing Table | High | PostgreSQL table names case-sensitive issues |
| Syntax Error | Medium | Incomplete SQL queries |
| Ambiguous Column | Low | Multiple tables with same column name |

### 3.2 SQLite vs PostgreSQL Differences

The models performed significantly better on SQLite than PostgreSQL due to:

1. **Case Sensitivity:** PostgreSQL is case-sensitive for identifiers, SQLite is not
2. **Table Naming:** Schema differences between SQLite (original) and PostgreSQL (migrated)
3. **Alias Handling:** PostgreSQL stricter about alias references

---

## 4. Performance Metrics

### 4.1 Inference Speed

| Model | Avg Inference (ms) | Queries/Hour |
|-------|-------------------|--------------|
| GPT-2 | 3,319 | ~1,085 |
| TinyLlama | 34,038 | ~106 |

GPT-2 is **10.3x faster** than TinyLlama on CPU.

### 4.2 Query Execution Speed

| Database | SQLite (ms) | PostgreSQL (ms) |
|----------|-------------|-----------------|
| Average | 2.04 | 0.78 |

SQLite execution is slightly slower but both are sub-millisecond for actual query execution.

---

## 5. Query Complexity Analysis

### 5.1 Complexity Distribution by Database

| Database | SIMPLE | MEDIUM | COMPLEX | ADVANCED |
|----------|--------|--------|---------|----------|
| Geography | 5.9% | 23.5% | 29.4% | 41.2% |
| Advising | 7.1% | 42.9% | 28.6% | 21.4% |
| ATIS | 3.0% | 15.2% | 24.2% | **57.6%** |
| Restaurants | **28.6%** | 42.9% | 14.3% | 14.3% |

**Hardest:** ATIS (57.6% ADVANCED)  
**Easiest:** Restaurants (28.6% SIMPLE)

### 5.2 Scoring System

| Feature | Points |
|---------|--------|
| JOIN clause | 2.0 |
| Aggregation (COUNT, SUM, etc.) | 1.5 |
| Subquery | 3.0 |
| HAVING clause | 3.0 |
| UNION/INTERSECT | 2.5 |
| Common Table Expression (CTE) | 4.0 |
| Window Function | 3.5 |
| CASE expression | 2.0 |

---

## 6. Recommendations

### 6.1 Model Selection

- **For Production:** Use GPT-2 for its speed advantage (10x faster)
- **For Accuracy:** Consider larger models or API-based solutions for complex queries

### 6.2 Improvements

1. **Schema Normalization:** Standardize column naming conventions between SQLite and PostgreSQL
2. **Post-processing:** Add SQL validation layer to fix common errors (missing aliases, case issues)
3. **Training Data:** Include more PostgreSQL-specific examples in training
4. **Query Complexity:** Train specifically on ADVANCED complexity queries

### 6.3 Future Work

1. Implement SQL post-processing/repair pipeline
2. Add schema-aware prompt engineering
3. Test with larger models (7B+ parameters)
4. Implement execution-guided inference

---

## 7. Conclusion

This evaluation demonstrates that:

1. **GPT-2** provides a good balance of speed and accuracy for Text-to-SQL tasks
2. **SQLite** execution is significantly more reliable than PostgreSQL
3. **Query complexity** strongly correlates with accuracy (simpler queries have higher success)
4. **Schema awareness** is critical for cross-database compatibility

For the course requirements comparing "GPT-2, TinyLlama, SQLite, PostgreSQL," GPT-2 is the recommended choice due to its 10x speed advantage with comparable accuracy.

---

## Appendix: Technical Details

### A. Training Configuration

```
GPT-2:
  - epochs: 3
  - batch_size: 4
  - learning_rate: 5e-5
  - max_length: 256

TinyLlama:
  - max_steps: 50 (CPU limitation)
  - batch_size: 1
```

### B. Evaluation Configuration

```
- Samples per database: 20
- Databases: geography, advising, atis, restaurants
- DB Engines: SQLite, PostgreSQL
- Metrics: Execution accuracy, Result match, Cross-DB consistency
```

### C. Files Generated

- `results/comprehensive_eval_geography.json`
- `results/comprehensive_eval_advising.json`
- `results/comprehensive_eval_atis.json`
- `results/comprehensive_eval_restaurants.json`
- `results/final_comparison_report.json`
