#!/usr/bin/env python3
"""
TinyLlama Training Script for LLMSQL2
Limited to 50 steps per database on CPU (~1.5-2 hours each).
Run this AFTER GPT-2 training completes.
"""

import subprocess
import time
from datetime import datetime

# Databases to train on
DATABASES = ["geography", "advising", "atis", "restaurants"]

# Base paths (inside Docker container)
DATA_BASE = "/app/data/text2sql-data/data"
RESULTS_BASE = "/app/results"

def log(message: str):
    """Log message with timestamp"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] {message}", flush=True)

def train_tinyllama(database: str) -> bool:
    """Train TinyLlama on a specific database (50 steps only)"""
    data_path = f"{DATA_BASE}/{database}.json"
    output_path = f"{RESULTS_BASE}/tinyllama-{database}"
    
    cmd = (
        f"docker exec llmsql2-app python -m src.train_tinyllama "
        f"--data {data_path} "
        f"--output {output_path} "
        f"--epochs 1"
    )
    
    log(f"Starting: TinyLlama training on {database} (50 steps max)")
    
    try:
        result = subprocess.run(cmd, shell=True)
        if result.returncode == 0:
            log(f"Completed: TinyLlama training on {database}")
            return True
        else:
            log(f"Failed: TinyLlama training on {database} (exit code {result.returncode})")
            return False
    except Exception as e:
        log(f"Error: TinyLlama training on {database} - {e}")
        return False

def main():
    log("=" * 60)
    log("LLMSQL2 TinyLlama Training (50 steps per database)")
    log("Estimated time: ~1.5-2 hours per database")
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
        results[database] = train_tinyllama(database)
    
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
