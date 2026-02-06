"""
Model inference module for Text-to-SQL models.
Supports local models: TinyLlama, GPT-2 Text2SQL, and more.
"""

import os
import torch
from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass
from abc import ABC, abstractmethod

from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer, 
    BitsAndBytesConfig,
    pipeline
)

from .utils import logger, extract_sql_from_response, format_schema_for_prompt


@dataclass
class ModelConfig:
    """Configuration for model loading."""
    model_name: str
    device: str = "auto"
    load_in_8bit: bool = False
    load_in_4bit: bool = False
    max_length: int = 2048
    temperature: float = 0.1
    do_sample: bool = False
    torch_dtype: str = "auto"


class BaseText2SQLModel(ABC):
    """Abstract base class for Text-to-SQL models."""
    
    @abstractmethod
    def generate_sql(self, question: str, schema: Optional[Dict] = None) -> str:
        """Generate SQL from natural language question."""
        pass
    
    @abstractmethod
    def generate_sql_batch(self, questions: List[str], schemas: Optional[List[Dict]] = None) -> List[str]:
        """Generate SQL for a batch of questions."""
        pass


class Text2SQLModel(BaseText2SQLModel):
    """
    Text-to-SQL model wrapper for HuggingFace models.
    """
    
    # Known model configurations
    MODEL_CONFIGS = {
        "ManthanKulakarni/TinyLlama-1.1B-Text2SQL": {
            "prompt_template": """<|system|>
You are a helpful assistant that converts natural language questions to SQL queries.</s>
<|user|>
{schema_text}

Question: {question}

Generate the SQL query:</s>
<|assistant|>
""",
            "chat_template": True
        },
        "frtcek95/qwen2.5-coder-text2nosql": {
            "prompt_template": """You are an expert database assistant. Convert the following question to a database query.

Schema:
{schema_text}

Question: {question}

Query:""",
            "chat_template": True
        }
    }
    
    def __init__(self, config: Union[str, ModelConfig]):
        """
        Initialize the model.
        
        Args:
            config: Model name string or ModelConfig object
        """
        if isinstance(config, str):
            config = ModelConfig(model_name=config)
        
        self.config = config
        self.model = None
        self.tokenizer = None
        self.pipeline = None
        
        self._load_model()
    
    def _load_model(self):
        """Load the model and tokenizer."""
        logger.info(f"Loading model: {self.config.model_name}")
        
        # Configure quantization if requested
        quantization_config = None
        if self.config.load_in_4bit:
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4"
            )
        elif self.config.load_in_8bit:
            quantization_config = BitsAndBytesConfig(load_in_8bit=True)
        
        # Determine torch dtype
        if self.config.torch_dtype == "auto":
            torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        else:
            torch_dtype = getattr(torch, self.config.torch_dtype)
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.model_name,
            trust_remote_code=True
        )
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Load model
        self.model = AutoModelForCausalLM.from_pretrained(
            self.config.model_name,
            quantization_config=quantization_config,
            torch_dtype=torch_dtype,
            device_map=self.config.device,
            trust_remote_code=True
        )
        
        # Create pipeline for easier inference
        self.pipeline = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            device_map=self.config.device
        )
        
        logger.info(f"Model loaded successfully on device: {self.model.device}")
    
    def _get_prompt_template(self) -> str:
        """Get the prompt template for this model."""
        if self.config.model_name in self.MODEL_CONFIGS:
            return self.MODEL_CONFIGS[self.config.model_name]["prompt_template"]
        
        # Default template
        return """Given the following database schema:
{schema_text}

Convert this question to SQL:
Question: {question}

SQL:"""
    
    def _format_prompt(self, question: str, schema: Optional[Dict] = None) -> str:
        """Format the prompt for the model."""
        template = self._get_prompt_template()
        
        schema_text = ""
        if schema:
            schema_text = format_schema_for_prompt(schema)
        
        return template.format(
            question=question,
            schema_text=schema_text
        )
    
    def generate_sql(self, question: str, schema: Optional[Dict] = None) -> str:
        """
        Generate SQL from a natural language question.
        
        Args:
            question: Natural language question
            schema: Optional database schema
            
        Returns:
            Generated SQL query
        """
        prompt = self._format_prompt(question, schema)
        
        # Generate
        outputs = self.pipeline(
            prompt,
            max_new_tokens=256,
            temperature=self.config.temperature,
            do_sample=self.config.do_sample,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            return_full_text=False
        )
        
        response = outputs[0]['generated_text']
        sql = extract_sql_from_response(response)
        
        return sql
    
    def generate_sql_batch(
        self, 
        questions: List[str], 
        schemas: Optional[List[Dict]] = None
    ) -> List[str]:
        """
        Generate SQL for a batch of questions.
        
        Args:
            questions: List of natural language questions
            schemas: Optional list of database schemas
            
        Returns:
            List of generated SQL queries
        """
        if schemas is None:
            schemas = [None] * len(questions)
        
        prompts = [
            self._format_prompt(q, s) 
            for q, s in zip(questions, schemas)
        ]
        
        # Batch generation
        outputs = self.pipeline(
            prompts,
            max_new_tokens=256,
            temperature=self.config.temperature,
            do_sample=self.config.do_sample,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            return_full_text=False,
            batch_size=4
        )
        
        results = []
        for output in outputs:
            response = output[0]['generated_text']
            sql = extract_sql_from_response(response)
            results.append(sql)
        
        return results
    
    def __repr__(self) -> str:
        return f"Text2SQLModel(model_name='{self.config.model_name}')"


def load_model(
    model_name: str,
    load_in_4bit: bool = False,
    load_in_8bit: bool = False,
    **kwargs
) -> Text2SQLModel:
    """
    Factory function to load a Text-to-SQL model.
    
    Args:
        model_name: HuggingFace model name or path
        load_in_4bit: Whether to use 4-bit quantization
        load_in_8bit: Whether to use 8-bit quantization
        
    Returns:
        Loaded Text2SQLModel
    """
    config = ModelConfig(
        model_name=model_name,
        load_in_4bit=load_in_4bit,
        load_in_8bit=load_in_8bit,
        **kwargs
    )
    return Text2SQLModel(config)


# Available models
AVAILABLE_MODELS = {
    "tinyllama-text2sql": "ManthanKulakarni/TinyLlama-1.1B-Text2SQL",
    "gpt2-text2sql": "n22t7a/text2sql-tuned-gpt2",
    "t5-text2sql": "juierror/flan-t5-text2sql-with-schema",
    "qwen-text2nosql": "frtcek95/qwen2.5-coder-text2nosql",
}


def list_available_models() -> Dict[str, str]:
    """List available pre-configured models."""
    return AVAILABLE_MODELS.copy()


class GPT2Text2SQLModel(BaseText2SQLModel):
    """
    GPT-2 based Text-to-SQL model from HuggingFace.
    Uses the model: n22t7a/text2sql-tuned-gpt2
    
    Note: This model has limited capabilities and may produce suboptimal SQL.
    It's included for project requirements compatibility.
    """
    
    DEFAULT_MODEL = "n22t7a/text2sql-tuned-gpt2"
    
    def __init__(
        self, 
        model_name: str = None,
        device: str = "auto",
        temperature: float = 0.7,
        max_new_tokens: int = 100
    ):
        """
        Initialize the GPT-2 Text2SQL model.
        
        Args:
            model_name: HuggingFace model name (defaults to n22t7a/text2sql-tuned-gpt2)
            device: Device to run on ("auto", "cpu", "cuda")
            temperature: Sampling temperature
            max_new_tokens: Maximum tokens to generate
        """
        self.model_name = model_name or self.DEFAULT_MODEL
        self.device = device
        self.temperature = temperature
        self.max_new_tokens = max_new_tokens
        
        self.model = None
        self.tokenizer = None
        
        self._load_model()
    
    def _load_model(self):
        """Load the GPT-2 model and tokenizer."""
        logger.info(f"Loading GPT-2 Text2SQL model: {self.model_name}")
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            trust_remote_code=True
        )
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Load model
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            device_map=self.device,
            trust_remote_code=True
        )
        
        logger.info(f"GPT-2 Text2SQL model loaded on device: {self.model.device}")
    
    def _format_prompt(self, question: str, schema: Optional[Union[Dict, str]] = None) -> str:
        """Format the prompt for GPT-2 Text2SQL model."""
        schema_text = ""
        if schema:
            if isinstance(schema, str):
                schema_text = schema
            else:
                schema_text = format_schema_for_prompt(schema)
        
        # Use SQL-start format to encourage SQL generation
        if schema_text:
            prompt = f"Table: {schema_text}\nQuestion: {question}\nSQL: SELECT"
        else:
            prompt = f"Question: {question}\nSQL: SELECT"
        return prompt
    
    def generate_sql(self, question: str, schema: Optional[Union[Dict, str]] = None) -> str:
        """
        Generate SQL from a natural language question using GPT-2 Text2SQL.
        
        Args:
            question: Natural language question
            schema: Optional database schema (string or dict)
            
        Returns:
            Generated SQL query
        """
        prompt = self._format_prompt(question, schema)
        
        try:
            inputs = self.tokenizer(
                prompt, 
                return_tensors="pt"
            ).to(self.model.device)
            
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=True,
                temperature=self.temperature,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id
            )
            
            full_output = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Extract SQL part (after "SQL: ")
            if "SQL:" in full_output:
                sql = full_output.split("SQL:")[-1].strip()
            else:
                sql = full_output[len(prompt):].strip()
            
            # Clean up - take first line/statement
            sql = sql.split('\n')[0].strip()
            if not sql.upper().startswith('SELECT'):
                sql = 'SELECT ' + sql
            
            return sql
            
        except Exception as e:
            logger.error(f"GPT-2 generation error: {e}")
            return f"ERROR: {e}"
    
    def generate_sql_batch(
        self, 
        questions: List[str], 
        schemas: Optional[List[Union[Dict, str]]] = None
    ) -> List[str]:
        """
        Generate SQL for multiple questions.
        
        Args:
            questions: List of natural language questions
            schemas: Optional list of schemas (one per question)
            
        Returns:
            List of generated SQL queries
        """
        if schemas is None:
            schemas = [None] * len(questions)
        
        results = []
        for question, schema in zip(questions, schemas):
            sql = self.generate_sql(question, schema)
            results.append(sql)
        
        return results
    
    def __repr__(self) -> str:
        return f"GPT2Text2SQLModel(model='{self.model_name}')"


def load_gpt2_model(
    model_name: str = None,
    **kwargs
) -> GPT2Text2SQLModel:
    """
    Factory function to load the GPT-2 Text2SQL model.
    
    Args:
        model_name: HuggingFace model name (defaults to n22t7a/text2sql-tuned-gpt2)
        **kwargs: Additional arguments passed to GPT2Text2SQLModel
        
    Returns:
        GPT2Text2SQLModel instance
    """
    return GPT2Text2SQLModel(model_name=model_name, **kwargs)
