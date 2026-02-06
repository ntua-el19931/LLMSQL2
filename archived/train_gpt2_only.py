#!/usr/bin/env python3
"""
GPT-2 Only Training Script for LLMSQL2
Trains GPT-2 on all 4 databases (TinyLlama is too slow on CPU).
"""

import subprocess
import sys
import time
from datetime import datetime

# Training configuration
GPT2_EPOCHS = 3
GPT2_BATCH_SIZE = 4

# Databases to train on (geography already done)
DATABASES = ["advising", "atis", "restaurants"]

# Base paths (inside Docker container)
DATA_BASE = "/app/data/text2sql-data/data"
RESULTS_BASE = "/app/results"

def log(message: str):
    """Log message with timestamp"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] {message}", flush=True)

def train_gpt2(database: str) -> bool:
    """Train GPT-2 on a specific database"""
    data_path = f"{DATA_BASE}/{database}.json"
    output_path = f"{RESULTS_BASE}/gpt2-{database}"
    
    cmd = (
        f"docker exec llmsql2-app python -m src.train_gpt2 "
        f"--data {data_path} "
        f"--output {output_path} "
        f"--epochs {GPT2_EPOCHS} "
        f"--batch-size {GPT2_BATCH_SIZE}"
    )
    
    log(f"Starting: GPT-2 training on {database}")
    
    try:
        result = subprocess.run(cmd, shell=True)
        if result.returncode == 0:
            log(f"Completed: GPT-2 training on {database}")
            return True
        else:
            log(f"Failed: GPT-2 training on {database} (exit code {result.returncode})")
            return False
    except Exception as e:
        log(f"Error: GPT-2 training on {database} - {e}")
        return False

def main():
    log("=" * 60)
    log("LLMSQL2 GPT-2 Training (TinyLlama skipped - too slow on CPU)")
    log(f"Epochs: {GPT2_EPOCHS}, Batch size: {GPT2_BATCH_SIZE}")
    log(f"Databases: {', '.join(DATABASES)}")
    log("=" * 60)
    
    # Check Docker
    result = subprocess.run("docker ps | findstr llmsql2-app", shell=True, capture_output=True)
    if result.returncode != 0:
        log("Docker container not running. Starting...")
        subprocess.run("docker-compose up -d", shell=True, cwd="D:\\ASPS\\LLMSQL2")
        time.sleep(10)
    
    results = {}
    
    for database in DATABASES:
        log("")
        log("=" * 60)
        log(f"Training: {database.upper()}")
        log("=" * 60)
        results[database] = train_gpt2(database)
    
    # Summary
    log("")
    log("=" * 60)
    log("TRAINING COMPLETE - SUMMARY")
    log("=" * 60)
    for db, success in results.items():
        status = "SUCCESS" if success else "FAILED"
        log(f"  {db}: {status}")

if __name__ == "__main__":
    main()
