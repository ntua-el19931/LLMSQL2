"""
Fine-tune TinyLlama on Text-to-SQL data.
Trains ManthanKulakarni/TinyLlama-1.1B-Text2SQL on database examples.
Uses LoRA for efficient fine-tuning of the 1.1B model.
"""

import json
import os
import torch
from pathlib import Path
from typing import List, Dict, Optional
from dataclasses import dataclass

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    BitsAndBytesConfig
)
from datasets import Dataset

try:
    from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training
    PEFT_AVAILABLE = True
except ImportError:
    PEFT_AVAILABLE = False
    print("WARNING: PEFT not available. Will do full fine-tuning (slower).")

from .utils import logger


# Schema definitions for different databases
SCHEMAS = {
    "geography": (
        "state(state_name, population, area, capital, density) | "
        "city(city_name, population, country_name, state_name) | "
        "river(river_name, length, country_name, traverse) | "
        "lake(lake_name, area, country_name, state_name) | "
        "mountain(mountain_name, mountain_altitude, country_name, state_name) | "
        "border_info(state_name, border) | "
        "highlow(state_name, highest_elevation, lowest_point, highest_point, lowest_elevation)"
    ),
    "advising": (
        "course(course_id, name, department, credits, description) | "
        "student(student_id, name, email, gpa) | "
        "instructor(instructor_id, name, department) | "
        "offering(offering_id, course_id, semester, year, instructor_id) | "
        "takes(student_id, offering_id, grade)"
    ),
    "atis": (
        "flight(flight_id, airline, from_airport, to_airport, departure_time, arrival_time) | "
        "airline(airline_code, airline_name) | "
        "airport(airport_code, airport_name, city, state) | "
        "aircraft(aircraft_code, aircraft_description, capacity)"
    ),
    "restaurants": (
        "restaurant(restaurant_id, name, food_type, city, rating) | "
        "location(restaurant_id, street_name, city) | "
        "geographic(city, county, region)"
    )
}


@dataclass
class TrainingConfig:
    """Configuration for training."""
    model_name: str = "ManthanKulakarni/TinyLlama-1.1B-Text2SQL"
    output_dir: str = "./results/tinyllama-finetuned"
    num_epochs: int = 3
    batch_size: int = 2
    learning_rate: float = 2e-4  # Higher LR for LoRA
    max_length: int = 384
    warmup_steps: int = 50
    save_steps: int = 200
    logging_steps: int = 25
    use_lora: bool = True
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05


def load_data(data_path: str) -> List[Dict]:
    """Load dataset from JSON file."""
    with open(data_path, 'r') as f:
        data = json.load(f)
    
    logger.info(f"Loaded {len(data)} examples from {data_path}")
    return data


def prepare_training_examples(
    data: List[Dict],
    database_name: str = "geography"
) -> List[str]:
    """
    Convert question-SQL pairs into training format for TinyLlama.
    
    Format: "### Schema:\n{schema}\n\n### Question:\n{question}\n\n### SQL:\n{sql}"
    """
    examples = []
    schema = SCHEMAS.get(database_name, SCHEMAS["geography"])
    
    for item in data:
        # Handle different data formats
        if 'sentences' in item:
            # Standard text2sql format
            for sent in item.get('sentences', []):
                question = sent.get('text', '')
                sql = item.get('sql', [])
                if isinstance(sql, list):
                    sql = sql[0] if sql else ''
                
                if question and sql:
                    # Clean SQL
                    sql_clean = sql.strip()
                    if not sql_clean.upper().startswith("SELECT"):
                        sql_clean = f"SELECT {sql_clean}"
                    
                    text = f"### Schema:\n{schema}\n\n### Question:\n{question}\n\n### SQL:\n{sql_clean}"
                    examples.append(text)
        elif 'question' in item and 'sql' in item:
            # Simple format
            question = item['question']
            sql = item['sql']
            sql_clean = sql.strip()
            if not sql_clean.upper().startswith("SELECT"):
                sql_clean = f"SELECT {sql_clean}"
            text = f"### Schema:\n{schema}\n\n### Question:\n{question}\n\n### SQL:\n{sql_clean}"
            examples.append(text)
    
    logger.info(f"Prepared {len(examples)} training examples")
    return examples


def create_dataset(
    examples: List[str],
    tokenizer,
    max_length: int = 384
) -> Dataset:
    """Create HuggingFace Dataset from examples."""
    
    def tokenize_function(example):
        return tokenizer(
            example['text'],
            truncation=True,
            max_length=max_length,
            padding='max_length'
        )
    
    dataset = Dataset.from_dict({'text': examples})
    tokenized = dataset.map(tokenize_function, remove_columns=['text'])
    
    return tokenized


def train_model(
    config: TrainingConfig,
    train_data_path: str,
    database_name: str = "geography",
    eval_data_path: Optional[str] = None
):
    """
    Fine-tune TinyLlama on text-to-SQL data.
    
    Args:
        config: Training configuration
        train_data_path: Path to training JSON file
        database_name: Name of database for schema selection
        eval_data_path: Optional path to evaluation JSON file
    """
    logger.info(f"Starting TinyLlama training with config: {config}")
    logger.info(f"Database: {database_name}")
    
    # Load tokenizer
    logger.info(f"Loading model: {config.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    
    # Set pad token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load model - use quantization only if CUDA is available
    if config.use_lora and PEFT_AVAILABLE and torch.cuda.is_available():
        logger.info("Loading model with 8-bit quantization for LoRA training (GPU detected)...")
        try:
            quantization_config = BitsAndBytesConfig(
                load_in_8bit=True,
                llm_int8_threshold=6.0
            )
            model = AutoModelForCausalLM.from_pretrained(
                config.model_name,
                quantization_config=quantization_config,
                device_map="auto",
                torch_dtype=torch.float16
            )
            model = prepare_model_for_kbit_training(model)
        except Exception as e:
            logger.warning(f"8-bit loading failed: {e}. Loading in float16...")
            model = AutoModelForCausalLM.from_pretrained(
                config.model_name,
                torch_dtype=torch.float16,
                device_map="auto"
            )
    elif config.use_lora and PEFT_AVAILABLE:
        # CPU mode - load in float32 for LoRA training
        logger.info("Loading model in float32 for LoRA training (CPU mode)...")
        model = AutoModelForCausalLM.from_pretrained(
            config.model_name,
            torch_dtype=torch.float32,
            low_cpu_mem_usage=True
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            config.model_name,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
        )
    
    model.config.pad_token_id = tokenizer.pad_token_id
    
    # Apply LoRA if available
    if config.use_lora and PEFT_AVAILABLE:
        logger.info("Applying LoRA configuration...")
        lora_config = LoraConfig(
            r=config.lora_r,
            lora_alpha=config.lora_alpha,
            lora_dropout=config.lora_dropout,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
            bias="none",
            task_type=TaskType.CAUSAL_LM
        )
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()
    
    # Load and prepare data
    train_data = load_data(train_data_path)
    train_examples = prepare_training_examples(train_data, database_name)
    train_dataset = create_dataset(train_examples, tokenizer, config.max_length)
    
    eval_dataset = None
    if eval_data_path:
        eval_data = load_data(eval_data_path)
        eval_examples = prepare_training_examples(eval_data, database_name)
        eval_dataset = create_dataset(eval_examples, tokenizer, config.max_length)
    
    # Data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False
    )
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=config.output_dir,
        overwrite_output_dir=True,
        num_train_epochs=config.num_epochs,
        per_device_train_batch_size=config.batch_size,
        per_device_eval_batch_size=config.batch_size,
        learning_rate=config.learning_rate,
        warmup_steps=config.warmup_steps,
        save_steps=config.save_steps,
        logging_steps=config.logging_steps,
        save_total_limit=2,
        prediction_loss_only=True,
        report_to="none",
        fp16=torch.cuda.is_available(),
        gradient_checkpointing=torch.cuda.is_available(),  # Only use on GPU - saves memory but slower on CPU
        gradient_accumulation_steps=4 if torch.cuda.is_available() else 8,  # Higher accumulation on CPU to reduce steps
        max_steps=50 if not torch.cuda.is_available() else -1,  # Reduced to 50 steps on CPU (~1.5 hours per DB)
    )
    
    # Create trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
    )
    
    # Train
    logger.info("Starting training...")
    trainer.train()
    
    # Save final model
    final_path = os.path.join(config.output_dir, "final")
    
    if config.use_lora and PEFT_AVAILABLE:
        # Save LoRA adapters
        model.save_pretrained(final_path)
        tokenizer.save_pretrained(final_path)
        logger.info(f"LoRA adapters saved to {final_path}")
    else:
        trainer.save_model(final_path)
        tokenizer.save_pretrained(final_path)
        logger.info(f"Model saved to {final_path}")
    
    return trainer


def main():
    """Main training function."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Fine-tune TinyLlama on Text-to-SQL data")
    parser.add_argument("--data", type=str, required=True,
                        help="Path to training data JSON")
    parser.add_argument("--database", type=str, default="geography",
                        choices=["geography", "advising", "atis", "restaurants"],
                        help="Database name for schema selection")
    parser.add_argument("--output", type=str, default="./results/tinyllama-finetuned",
                        help="Output directory for trained model")
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=2, help="Training batch size")
    parser.add_argument("--lr", type=float, default=2e-4, help="Learning rate")
    parser.add_argument("--no-lora", action="store_true", help="Disable LoRA (full fine-tuning)")
    
    args = parser.parse_args()
    
    config = TrainingConfig(
        output_dir=args.output,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        use_lora=not args.no_lora
    )
    
    train_model(config, args.data, args.database)


if __name__ == "__main__":
    main()
