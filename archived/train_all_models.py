#!/usr/bin/env python3
"""
Batch Training Script for LLMSQL2
Trains both GPT-2 and TinyLlama on all 4 databases sequentially.
Ensures fair training: same epochs/parameters for each model-database combination.
"""

import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

# Training configuration - SAME for all to ensure fairness
GPT2_EPOCHS = 3       # Reduced from 5
GPT2_BATCH_SIZE = 4   # Increased from 2 for speed
TINYLLAMA_EPOCHS = 1  # Reduced from 3 - TinyLlama is very slow on CPU

# Databases to train on
DATABASES = ["geography", "advising", "atis", "restaurants"]

# Base paths (inside Docker container)
DATA_BASE = "/app/data/text2sql-data/data"
RESULTS_BASE = "/app/results"
LOG_FILE = "training_output.log"  # Different name to avoid conflicts

def log(message: str):
    """Log message with timestamp"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] {message}", flush=True)
    # Also append to log file with retry logic
    for attempt in range(3):
        try:
            with open(LOG_FILE, "a", encoding="utf-8") as f:
                f.write(f"[{timestamp}] {message}\n")
            break
        except PermissionError:
            time.sleep(0.5)
        except Exception:
            break

def run_docker_command(cmd: str, description: str) -> bool:
    """Run a command inside the Docker container"""
    log(f"Starting: {description}")
    full_cmd = f'docker exec llmsql2-app {cmd}'
    
    try:
        result = subprocess.run(
            full_cmd,
            shell=True,
            capture_output=False,  # Show output in real-time
            text=True
        )
        if result.returncode == 0:
            log(f"Completed: {description}")
            return True
        else:
            log(f"Failed: {description} (exit code {result.returncode})")
            return False
    except Exception as e:
        log(f"Error: {description} - {e}")
        return False

def train_gpt2(database: str) -> bool:
    """Train GPT-2 on a specific database"""
    data_path = f"{DATA_BASE}/{database}.json"
    output_path = f"{RESULTS_BASE}/gpt2-{database}"
    
    cmd = (
        f"python -m src.train_gpt2 "
        f"--data {data_path} "
        f"--output {output_path} "
        f"--epochs {GPT2_EPOCHS} "
        f"--batch-size {GPT2_BATCH_SIZE}"
    )
    
    return run_docker_command(cmd, f"GPT-2 training on {database}")

def train_tinyllama(database: str) -> bool:
    """Train TinyLlama on a specific database"""
    data_path = f"{DATA_BASE}/{database}.json"
    output_path = f"{RESULTS_BASE}/tinyllama-{database}"
    
    cmd = (
        f"python -m src.train_tinyllama "
        f"--data {data_path} "
        f"--output {output_path} "
        f"--epochs {TINYLLAMA_EPOCHS}"
    )
    
    return run_docker_command(cmd, f"TinyLlama training on {database}")

def check_docker_running() -> bool:
    """Check if Docker containers are running"""
    result = subprocess.run(
        "docker ps --format '{{.Names}}' | findstr llmsql2-app",
        shell=True,
        capture_output=True,
        text=True
    )
    return result.returncode == 0

def start_docker():
    """Start Docker containers if not running"""
    log("Starting Docker containers...")
    subprocess.run("docker-compose up -d", shell=True, cwd="D:\\ASPS\\LLMSQL2")
    time.sleep(30)  # Wait for containers to be ready

def main():
    log("=" * 60)
    log("LLMSQL2 Batch Training Started")
    log(f"Models: GPT-2 ({GPT2_EPOCHS} epochs), TinyLlama ({TINYLLAMA_EPOCHS} epochs)")
    log(f"Databases: {', '.join(DATABASES)}")
    log("=" * 60)
    
    # Check Docker
    if not check_docker_running():
        start_docker()
        if not check_docker_running():
            log("ERROR: Docker containers not running. Please start them manually.")
            sys.exit(1)
    
    # Training schedule: Alternate between models to ensure fairness
    # Round 1: GPT2-geo, TinyLlama-geo
    # Round 2: GPT2-advising, TinyLlama-advising
    # etc.
    
    results = {"gpt2": {}, "tinyllama": {}}
    
    for database in DATABASES:
        log(f"\n{'='*60}")
        log(f"Training Round: {database.upper()}")
        log(f"{'='*60}\n")
        
        # Train GPT-2
        results["gpt2"][database] = train_gpt2(database)
        
        # Train TinyLlama
        results["tinyllama"][database] = train_tinyllama(database)
    
    # Print summary
    log("\n" + "=" * 60)
    log("TRAINING SUMMARY")
    log("=" * 60)
    
    for model in ["gpt2", "tinyllama"]:
        for db in DATABASES:
            status = "✓ Success" if results[model][db] else "✗ Failed"
            log(f"{model:12} | {db:12} | {status}")
    
    # Count successes
    total = len(DATABASES) * 2
    successes = sum(1 for m in results.values() for r in m.values() if r)
    log(f"\nTotal: {successes}/{total} training runs completed successfully")
    log("=" * 60)

if __name__ == "__main__":
    main()
