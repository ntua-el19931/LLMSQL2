"""
Fine-tune GPT-2 on Text-to-SQL data.
Trains n22t7a/text2sql-tuned-gpt2 on geography database examples.
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
    DataCollatorForLanguageModeling
)
from datasets import Dataset

from .utils import logger


@dataclass
class TrainingConfig:
    """Configuration for training."""
    model_name: str = "n22t7a/text2sql-tuned-gpt2"
    output_dir: str = "./results/gpt2-finetuned"
    num_epochs: int = 3
    batch_size: int = 4
    learning_rate: float = 5e-5
    max_length: int = 256
    warmup_steps: int = 100
    save_steps: int = 500
    logging_steps: int = 50


def load_geography_data(data_path: str) -> List[Dict]:
    """Load geography dataset from JSON file."""
    with open(data_path, 'r') as f:
        data = json.load(f)
    
    logger.info(f"Loaded {len(data)} examples from {data_path}")
    return data


def prepare_training_examples(
    data: List[Dict],
    schema: str = None
) -> List[str]:
    """
    Convert question-SQL pairs into training format.
    
    Format: "Table: {schema}\nQuestion: {question}\nSQL: {sql}"
    """
    examples = []
    
    # Default geography schema
    if schema is None:
        schema = (
            "state(state_name, population, area, capital, density) | "
            "city(city_name, population, country_name, state_name) | "
            "river(river_name, length, country_name, traverse) | "
            "lake(lake_name, area, country_name, state_name) | "
            "mountain(mountain_name, mountain_altitude, country_name, state_name) | "
            "border_info(state_name, border) | "
            "highlow(state_name, highest_elevation, lowest_point, highest_point, lowest_elevation)"
        )
    
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
                    # Ensure SQL starts with SELECT properly
                    sql_clean = sql.strip()
                    if sql_clean.upper().startswith("SELECT"):
                        text = f"Table: {schema}\nQuestion: {question}\nSQL: {sql_clean}"
                    else:
                        text = f"Table: {schema}\nQuestion: {question}\nSQL: SELECT {sql_clean}"
                    examples.append(text)
        elif 'question' in item and 'sql' in item:
            # Simple format
            question = item['question']
            sql = item['sql']
            text = f"Table: {schema}\nQuestion: {question}\nSQL: {sql}"
            examples.append(text)
    
    logger.info(f"Prepared {len(examples)} training examples")
    return examples


def create_dataset(
    examples: List[str],
    tokenizer,
    max_length: int = 256
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
    eval_data_path: Optional[str] = None
):
    """
    Fine-tune GPT-2 on text-to-SQL data.
    
    Args:
        config: Training configuration
        train_data_path: Path to training JSON file
        eval_data_path: Optional path to evaluation JSON file
    """
    logger.info(f"Starting training with config: {config}")
    
    # Load tokenizer and model
    logger.info(f"Loading model: {config.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    model = AutoModelForCausalLM.from_pretrained(config.model_name)
    
    # Set pad token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = tokenizer.eos_token_id
    
    # Load and prepare data
    train_data = load_geography_data(train_data_path)
    train_examples = prepare_training_examples(train_data)
    train_dataset = create_dataset(train_examples, tokenizer, config.max_length)
    
    eval_dataset = None
    if eval_data_path:
        eval_data = load_geography_data(eval_data_path)
        eval_examples = prepare_training_examples(eval_data)
        eval_dataset = create_dataset(eval_examples, tokenizer, config.max_length)
    
    # Data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False  # GPT-2 is causal LM, not masked LM
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
        report_to="none",  # Disable wandb/tensorboard
        fp16=torch.cuda.is_available(),
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
    trainer.save_model(final_path)
    tokenizer.save_pretrained(final_path)
    logger.info(f"Model saved to {final_path}")
    
    return trainer


def main():
    """Main training function."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Fine-tune GPT-2 on Text-to-SQL data")
    parser.add_argument("--data", type=str, default="/app/data/text2sql-data/data/geography.json",
                        help="Path to training data JSON")
    parser.add_argument("--output", type=str, default="./results/gpt2-finetuned",
                        help="Output directory for trained model")
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=4, help="Training batch size")
    parser.add_argument("--lr", type=float, default=5e-5, help="Learning rate")
    
    args = parser.parse_args()
    
    config = TrainingConfig(
        output_dir=args.output,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr
    )
    
    train_model(config, args.data)


if __name__ == "__main__":
    main()
