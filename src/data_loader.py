"""
Data loader for Text-to-SQL datasets.
Supports loading from text2sql-data repository.
"""

import json
import sqlite3
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from abc import ABC, abstractmethod

from .utils import get_data_dir, load_json, logger


@dataclass
class Text2SQLExample:
    """A single Text-to-SQL example."""
    question: str
    sql: str
    db_id: str
    schema: Optional[Dict[str, Any]] = None
    variables: Optional[Dict[str, str]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'question': self.question,
            'sql': self.sql,
            'db_id': self.db_id,
            'schema': self.schema,
            'variables': self.variables
        }


class BaseDataLoader(ABC):
    """Abstract base class for data loaders."""
    
    def __init__(self, data_dir: Optional[Path] = None):
        self.data_dir = data_dir or get_data_dir()
        self.text2sql_data_dir = self.data_dir / "text2sql-data" / "data"
    
    @abstractmethod
    def load(self) -> List[Text2SQLExample]:
        """Load the dataset."""
        pass
    
    @abstractmethod
    def get_schema(self, db_id: str) -> Dict[str, Any]:
        """Get schema for a specific database."""
        pass


class Text2SQLDataLoader(BaseDataLoader):
    """
    Loader for datasets from the text2sql-data repository.
    Supports: Academic, Advising, ATIS, Geography, IMDB, Restaurants, Scholar, Spider, Yelp, WikiSQL
    """
    
    AVAILABLE_DATASETS = [
        'academic', 'advising', 'atis', 'geography', 
        'imdb', 'restaurants', 'scholar', 'spider', 'yelp'
    ]
    
    def __init__(self, dataset_name: str, split: str = 'train', data_dir: Optional[Path] = None):
        super().__init__(data_dir)
        self.dataset_name = dataset_name.lower()
        self.split = split
        
        if self.dataset_name not in self.AVAILABLE_DATASETS:
            raise ValueError(f"Dataset '{dataset_name}' not available. Choose from: {self.AVAILABLE_DATASETS}")
        
        self.dataset_dir = self.text2sql_data_dir / self.dataset_name
        self._schema_cache: Dict[str, Dict[str, Any]] = {}
    
    def load(self) -> List[Text2SQLExample]:
        """Load the dataset from JSON files."""
        examples = []
        
        # Find the data file
        data_file = self._find_data_file()
        if not data_file:
            logger.error(f"Could not find data file for {self.dataset_name}")
            return examples
        
        logger.info(f"Loading {self.dataset_name} from {data_file}")
        data = load_json(data_file)
        
        # Parse based on format
        if isinstance(data, list):
            for item in data:
                example = self._parse_example(item)
                if example:
                    examples.append(example)
        elif isinstance(data, dict):
            # Some datasets have nested structure
            for key, items in data.items():
                if isinstance(items, list):
                    for item in items:
                        example = self._parse_example(item)
                        if example:
                            examples.append(example)
        
        logger.info(f"Loaded {len(examples)} examples from {self.dataset_name}")
        return examples
    
    def _find_data_file(self) -> Optional[Path]:
        """Find the appropriate data file for the dataset."""
        possible_files = [
            self.dataset_dir / f"{self.dataset_name}.json",
            self.dataset_dir / f"{self.split}.json",
            self.dataset_dir / f"{self.dataset_name}_{self.split}.json",
        ]
        
        # Also check for any JSON file in the directory
        if self.dataset_dir.exists():
            json_files = list(self.dataset_dir.glob("*.json"))
            possible_files.extend(json_files)
        
        for file_path in possible_files:
            if file_path.exists():
                return file_path
        
        return None
    
    def _parse_example(self, item: Dict[str, Any]) -> Optional[Text2SQLExample]:
        """Parse a single example from the dataset."""
        try:
            # Handle different field names across datasets
            question = item.get('question') or item.get('sentence') or item.get('nl') or ''
            sql = item.get('sql') or item.get('query') or ''
            db_id = item.get('db_id') or item.get('database') or self.dataset_name
            variables = item.get('variables') or item.get('sql_variables')
            
            if not question or not sql:
                return None
            
            # Handle SQL that might be a list
            if isinstance(sql, list):
                sql = sql[0] if sql else ''
            
            return Text2SQLExample(
                question=question,
                sql=sql,
                db_id=db_id,
                schema=self.get_schema(db_id),
                variables=variables
            )
        except Exception as e:
            logger.warning(f"Failed to parse example: {e}")
            return None
    
    def get_schema(self, db_id: str) -> Dict[str, Any]:
        """Get schema for a specific database."""
        if db_id in self._schema_cache:
            return self._schema_cache[db_id]
        
        schema = {}
        
        # Try to load schema from various sources
        schema_files = [
            self.dataset_dir / "schema.json",
            self.dataset_dir / f"{db_id}_schema.json",
            self.dataset_dir / "tables.json",
        ]
        
        for schema_file in schema_files:
            if schema_file.exists():
                try:
                    schema = load_json(schema_file)
                    break
                except Exception as e:
                    logger.warning(f"Failed to load schema from {schema_file}: {e}")
        
        self._schema_cache[db_id] = schema
        return schema
    
    def get_database_connection(self, db_id: str) -> Optional[sqlite3.Connection]:
        """Get a SQLite connection for executing queries."""
        db_files = [
            self.dataset_dir / f"{db_id}.sqlite",
            self.dataset_dir / f"{db_id}.db",
            self.dataset_dir / "database" / f"{db_id}" / f"{db_id}.sqlite",
        ]
        
        for db_file in db_files:
            if db_file.exists():
                return sqlite3.connect(str(db_file))
        
        return None


class SpiderDataLoader(Text2SQLDataLoader):
    """Specialized loader for Spider dataset."""
    
    def __init__(self, split: str = 'train', data_dir: Optional[Path] = None):
        super().__init__('spider', split, data_dir)
    
    def load(self) -> List[Text2SQLExample]:
        """Load Spider dataset with proper handling."""
        examples = []
        
        # Spider has specific file structure
        data_file = self.dataset_dir / f"{self.split}.json"
        tables_file = self.dataset_dir / "tables.json"
        
        if not data_file.exists():
            logger.error(f"Spider {self.split}.json not found at {data_file}")
            return examples
        
        # Load tables/schemas
        if tables_file.exists():
            tables_data = load_json(tables_file)
            for table_info in tables_data:
                db_id = table_info.get('db_id', '')
                self._schema_cache[db_id] = table_info
        
        # Load examples
        data = load_json(data_file)
        for item in data:
            example = Text2SQLExample(
                question=item.get('question', ''),
                sql=item.get('query', ''),
                db_id=item.get('db_id', ''),
                schema=self.get_schema(item.get('db_id', ''))
            )
            examples.append(example)
        
        logger.info(f"Loaded {len(examples)} Spider examples ({self.split})")
        return examples


def get_data_loader(dataset_name: str, split: str = 'train', **kwargs) -> BaseDataLoader:
    """Factory function to get appropriate data loader."""
    dataset_name = dataset_name.lower()
    
    if dataset_name == 'spider':
        return SpiderDataLoader(split=split, **kwargs)
    else:
        return Text2SQLDataLoader(dataset_name, split=split, **kwargs)


def list_available_datasets() -> List[str]:
    """List all available datasets."""
    return Text2SQLDataLoader.AVAILABLE_DATASETS
