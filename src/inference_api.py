"""
Unified Text-to-SQL Inference API.

Provides a clean interface to load and use any fine-tuned model
for text-to-SQL generation.
"""

import torch
from pathlib import Path
from typing import Dict, List, Optional, Union
from dataclasses import dataclass
from enum import Enum

from transformers import AutoTokenizer, AutoModelForCausalLM

try:
    from peft import PeftModel
    PEFT_AVAILABLE = True
except ImportError:
    PEFT_AVAILABLE = False

from .utils import postprocess_sql, logger


class ModelType(Enum):
    GPT2 = "gpt2"
    TINYLLAMA = "tinyllama"


# Database schemas
SCHEMAS = {
    "geography": {
        "short": "state(state_name, population, area, capital, density) | city(city_name, population, country_name, state_name)",
        "full": """state(state_name, population, area, capital, density) | city(city_name, population, country_name, state_name) | river(river_name, length, country_name, traverse) | lake(lake_name, area, country_name, state_name) | mountain(mountain_name, mountain_altitude, country_name, state_name) | border_info(state_name, border) | highlow(state_name, highest_elevation, lowest_point, highest_point, lowest_elevation)"""
    },
    "advising": {
        "short": "course(course_id, name, department, credits) | student(student_id, name, email, gpa) | instructor(instructor_id, name, department)",
        "full": """course(course_id, name, department, credits, description) | student(student_id, name, email, gpa) | instructor(instructor_id, name, department) | offering(offering_id, course_id, semester, year, instructor_id) | takes(student_id, offering_id, grade)"""
    },
    "atis": {
        "short": "flight(flight_id, airline, from_airport, to_airport, departure_time) | airline(airline_code, airline_name) | airport(airport_code, airport_name, city)",
        "full": """flight(flight_id, airline, from_airport, to_airport, departure_time, arrival_time) | airline(airline_code, airline_name) | airport(airport_code, airport_name, city, state) | aircraft(aircraft_code, aircraft_description, capacity)"""
    },
    "restaurants": {
        "short": "restaurant(restaurant_id, name, food_type, city, rating) | location(restaurant_id, street_name, city)",
        "full": """restaurant(restaurant_id, name, food_type, city, rating) | location(restaurant_id, street_name, city) | geographic(city, county, region)"""
    }
}


@dataclass
class SQLResult:
    """Result of SQL generation."""
    question: str
    sql: str
    raw_sql: str
    model: str
    database: str
    inference_time: float


class Text2SQLInference:
    """
    Unified interface for Text-to-SQL inference.
    
    Example usage:
        >>> inference = Text2SQLInference()
        >>> inference.load_model("gpt2", "/app/results/gpt2-geography/final")
        >>> result = inference.generate("What is the capital of Texas?", "geography")
        >>> print(result.sql)
    """
    
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.model_type: Optional[ModelType] = None
        self.model_path: Optional[str] = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
    
    def load_model(
        self, 
        model_type: Union[str, ModelType],
        checkpoint_path: str,
        base_model_name: Optional[str] = None
    ) -> None:
        """
        Load a fine-tuned model.
        
        Args:
            model_type: "gpt2" or "tinyllama"
            checkpoint_path: Path to fine-tuned model/adapter
            base_model_name: Base model name (required for TinyLlama LoRA)
        """
        if isinstance(model_type, str):
            model_type = ModelType(model_type.lower())
        
        self.model_type = model_type
        self.model_path = checkpoint_path
        
        logger.info(f"Loading {model_type.value} model from {checkpoint_path}")
        
        if model_type == ModelType.GPT2:
            self._load_gpt2(checkpoint_path)
        elif model_type == ModelType.TINYLLAMA:
            if base_model_name is None:
                base_model_name = "ManthanKulakarni/TinyLlama-1.1B-Text2SQL"
            self._load_tinyllama(checkpoint_path, base_model_name)
        
        logger.info(f"Model loaded successfully on {self.device}")
    
    def _load_gpt2(self, checkpoint_path: str) -> None:
        """Load GPT-2 model."""
        self.tokenizer = AutoTokenizer.from_pretrained(checkpoint_path)
        self.model = AutoModelForCausalLM.from_pretrained(checkpoint_path)
        self.model.eval()
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
    
    def _load_tinyllama(self, adapter_path: str, base_model_name: str) -> None:
        """Load TinyLlama with LoRA adapters."""
        if not PEFT_AVAILABLE:
            raise ImportError("PEFT is required for TinyLlama LoRA models")
        
        self.tokenizer = AutoTokenizer.from_pretrained(adapter_path)
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            torch_dtype=torch.float32,
            low_cpu_mem_usage=True
        )
        self.model = PeftModel.from_pretrained(base_model, adapter_path)
        self.model.eval()
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
    
    def generate(
        self,
        question: str,
        database: str = "geography",
        use_full_schema: bool = True,
        max_new_tokens: int = 100,
        **generation_kwargs
    ) -> SQLResult:
        """
        Generate SQL from a natural language question.
        
        Args:
            question: Natural language question
            database: Database name for schema selection
            use_full_schema: Use full or short schema
            max_new_tokens: Maximum tokens to generate
            **generation_kwargs: Additional generation parameters
            
        Returns:
            SQLResult with generated SQL
        """
        import time
        
        if self.model is None:
            raise RuntimeError("No model loaded. Call load_model() first.")
        
        # Get schema
        schema_type = "full" if use_full_schema else "short"
        schema = SCHEMAS.get(database, SCHEMAS["geography"])[schema_type]
        
        # Build prompt based on model type
        if self.model_type == ModelType.GPT2:
            prompt = f"Table: {schema}\nQuestion: {question}\nSQL:"
            sql_marker = "SQL:"
        else:  # TinyLlama
            prompt = f"### Schema:\n{schema}\n\n### Question:\n{question}\n\n### SQL:\n"
            sql_marker = "### SQL:"
        
        # Tokenize
        inputs = self.tokenizer(prompt, return_tensors='pt', padding=True)
        
        # Generate
        start_time = time.time()
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id,
                **generation_kwargs
            )
        inference_time = time.time() - start_time
        
        # Decode and extract SQL
        generated = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        raw_sql = generated.split(sql_marker)[-1].strip()
        clean_sql = postprocess_sql(raw_sql)
        
        return SQLResult(
            question=question,
            sql=clean_sql,
            raw_sql=raw_sql,
            model=self.model_type.value,
            database=database,
            inference_time=inference_time
        )
    
    def generate_batch(
        self,
        questions: List[str],
        database: str = "geography",
        **kwargs
    ) -> List[SQLResult]:
        """
        Generate SQL for multiple questions.
        
        Args:
            questions: List of natural language questions
            database: Database name
            **kwargs: Passed to generate()
            
        Returns:
            List of SQLResult
        """
        return [self.generate(q, database, **kwargs) for q in questions]
    
    def interactive(self, database: str = "geography"):
        """
        Start an interactive session.
        
        Args:
            database: Default database to use
        """
        print(f"\n{'='*60}")
        print("Text-to-SQL Interactive Mode")
        print(f"Model: {self.model_type.value}")
        print(f"Database: {database}")
        print("Type 'quit' to exit, 'db <name>' to change database")
        print(f"{'='*60}\n")
        
        current_db = database
        
        while True:
            try:
                question = input("Question: ").strip()
                
                if not question:
                    continue
                
                if question.lower() == 'quit':
                    print("Goodbye!")
                    break
                
                if question.lower().startswith('db '):
                    new_db = question[3:].strip()
                    if new_db in SCHEMAS:
                        current_db = new_db
                        print(f"Switched to database: {current_db}")
                    else:
                        print(f"Unknown database. Available: {list(SCHEMAS.keys())}")
                    continue
                
                result = self.generate(question, current_db)
                print(f"SQL: {result.sql}")
                print(f"(Time: {result.inference_time:.2f}s)\n")
                
            except KeyboardInterrupt:
                print("\nGoodbye!")
                break
            except Exception as e:
                print(f"Error: {e}\n")


def create_inference(
    model_type: str = "gpt2",
    database: str = "geography"
) -> Text2SQLInference:
    """
    Factory function to create and load an inference instance.
    
    Args:
        model_type: "gpt2" or "tinyllama"
        database: Database the model was trained on
        
    Returns:
        Loaded Text2SQLInference instance
    """
    inference = Text2SQLInference()
    
    if model_type.lower() == "gpt2":
        checkpoint = f"/app/results/gpt2-{database}/final"
        inference.load_model("gpt2", checkpoint)
    elif model_type.lower() == "tinyllama":
        checkpoint = f"/app/results/tinyllama-{database}/final"
        inference.load_model("tinyllama", checkpoint)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    return inference


def main():
    """Demo of the inference API."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Text-to-SQL Inference")
    parser.add_argument("--model", choices=["gpt2", "tinyllama"], default="gpt2")
    parser.add_argument("--database", default="geography")
    parser.add_argument("--interactive", "-i", action="store_true")
    parser.add_argument("--question", "-q", type=str, help="Single question to answer")
    
    args = parser.parse_args()
    
    # Create inference instance
    inference = create_inference(args.model, args.database)
    
    if args.interactive:
        inference.interactive(args.database)
    elif args.question:
        result = inference.generate(args.question, args.database)
        print(f"Question: {result.question}")
        print(f"SQL: {result.sql}")
        print(f"Time: {result.inference_time:.2f}s")
    else:
        # Demo with sample questions
        questions = [
            "What is the capital of Texas?",
            "How many people live in California?",
            "What state has the largest population?",
        ]
        
        print(f"\n{'='*60}")
        print(f"Model: {args.model} | Database: {args.database}")
        print(f"{'='*60}\n")
        
        for q in questions:
            result = inference.generate(q, args.database)
            print(f"Q: {q}")
            print(f"SQL: {result.sql}\n")


if __name__ == '__main__':
    main()
