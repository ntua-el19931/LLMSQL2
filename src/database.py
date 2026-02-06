"""
Database connection module for LLMSQL2.
Supports both PostgreSQL and SQLite databases.
"""

import os
import sqlite3
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass
from pathlib import Path

try:
    import pg8000
    PG_AVAILABLE = True
except ImportError:
    PG_AVAILABLE = False

from .utils import logger


@dataclass
class DatabaseConfig:
    """Database configuration."""
    name: str
    db_type: str  # 'postgresql' or 'sqlite'
    # PostgreSQL settings
    host: Optional[str] = None
    port: Optional[int] = None
    user: Optional[str] = None
    password: Optional[str] = None
    database: Optional[str] = None
    # SQLite settings
    filepath: Optional[str] = None
    # Schema info
    tables: Optional[List[str]] = None
    description: Optional[str] = None


# Available databases configuration
DATABASES: Dict[str, DatabaseConfig] = {
    # PostgreSQL databases
    "geography_pg": DatabaseConfig(
        name="geography_pg",
        db_type="postgresql",
        host=os.environ.get("POSTGRES_HOST", "localhost"),
        port=int(os.environ.get("POSTGRES_PORT", 5432)),
        user=os.environ.get("POSTGRES_USER", "postgres"),
        password=os.environ.get("POSTGRES_PASSWORD", "postgres"),
        database="geography",
        tables=["state", "city", "river", "lake", "mountain", "border_info", "highlow"],
        description="US Geography database (states, cities, rivers, etc.)"
    ),
    "advising_pg": DatabaseConfig(
        name="advising_pg",
        db_type="postgresql",
        host=os.environ.get("POSTGRES_HOST", "localhost"),
        port=int(os.environ.get("POSTGRES_PORT", 5432)),
        user=os.environ.get("POSTGRES_USER", "postgres"),
        password=os.environ.get("POSTGRES_PASSWORD", "postgres"),
        database="advising",
        tables=["student", "course", "instructor", "offering", "registration"],
        description="Academic advising database"
    ),
    "atis_pg": DatabaseConfig(
        name="atis_pg",
        db_type="postgresql",
        host=os.environ.get("POSTGRES_HOST", "localhost"),
        port=int(os.environ.get("POSTGRES_PORT", 5432)),
        user=os.environ.get("POSTGRES_USER", "postgres"),
        password=os.environ.get("POSTGRES_PASSWORD", "postgres"),
        database="atis",
        tables=["flight", "aircraft", "airline", "airport", "city", "fare"],
        description="Airline Travel Information System"
    ),
    "restaurants_pg": DatabaseConfig(
        name="restaurants_pg",
        db_type="postgresql",
        host=os.environ.get("POSTGRES_HOST", "localhost"),
        port=int(os.environ.get("POSTGRES_PORT", 5432)),
        user=os.environ.get("POSTGRES_USER", "postgres"),
        password=os.environ.get("POSTGRES_PASSWORD", "postgres"),
        database="restaurants",
        tables=["restaurant", "location", "geographic"],
        description="Restaurant locations database"
    ),
    
    # SQLite databases
    "geography_sqlite": DatabaseConfig(
        name="geography_sqlite",
        db_type="sqlite",
        filepath="/app/data/text2sql-data/data/geography-db.added-in-2020.sqlite",
        tables=["state", "city", "river", "lake", "mountain", "border_info", "highlow"],
        description="US Geography database (SQLite)"
    ),
    "advising_sqlite": DatabaseConfig(
        name="advising_sqlite",
        db_type="sqlite",
        filepath="/app/data/text2sql-data/data/advising-db.added-in-2020.sqlite",
        description="Academic advising database (SQLite)"
    ),
    "atis_sqlite": DatabaseConfig(
        name="atis_sqlite",
        db_type="sqlite",
        filepath="/app/data/text2sql-data/data/atis-db.added-in-2020.sqlite",
        description="Airline Travel Information System (SQLite)"
    ),
    "restaurants_sqlite": DatabaseConfig(
        name="restaurants_sqlite",
        db_type="sqlite",
        filepath="/app/data/text2sql-data/data/restaurants-db.added-in-2020.sqlite",
        description="Restaurant locations database (SQLite)"
    ),
}


class DatabaseConnection:
    """Unified database connection handler."""
    
    def __init__(self, config: DatabaseConfig):
        self.config = config
        self.connection = None
        
    def connect(self):
        """Establish database connection."""
        if self.config.db_type == "postgresql":
            if not PG_AVAILABLE:
                raise ImportError("pg8000 not installed")
            self.connection = pg8000.connect(
                host=self.config.host,
                port=self.config.port,
                user=self.config.user,
                password=self.config.password,
                database=self.config.database
            )
            logger.info(f"Connected to PostgreSQL: {self.config.database}")
        elif self.config.db_type == "sqlite":
            self.connection = sqlite3.connect(self.config.filepath)
            logger.info(f"Connected to SQLite: {self.config.filepath}")
        else:
            raise ValueError(f"Unknown database type: {self.config.db_type}")
        
        return self
    
    def execute(self, query: str, params: tuple = None) -> List[Tuple]:
        """Execute a query and return results."""
        cursor = self.connection.cursor()
        try:
            if params:
                cursor.execute(query, params)
            else:
                cursor.execute(query)
            
            # Check if it's a SELECT query
            if query.strip().upper().startswith("SELECT"):
                return cursor.fetchall()
            else:
                self.connection.commit()
                return []
        except Exception as e:
            logger.error(f"Query error: {e}")
            raise
        finally:
            cursor.close()
    
    def get_tables(self) -> List[str]:
        """Get list of tables in database."""
        if self.config.db_type == "postgresql":
            query = """
                SELECT table_name 
                FROM information_schema.tables 
                WHERE table_schema = 'public'
            """
        else:  # sqlite
            query = "SELECT name FROM sqlite_master WHERE type='table'"
        
        results = self.execute(query)
        return [row[0] for row in results]
    
    def get_schema(self, table_name: str) -> List[Dict]:
        """Get column information for a table."""
        if self.config.db_type == "postgresql":
            query = """
                SELECT column_name, data_type, is_nullable
                FROM information_schema.columns
                WHERE table_name = %s
                ORDER BY ordinal_position
            """
            results = self.execute(query, (table_name,))
        else:  # sqlite
            cursor = self.connection.cursor()
            cursor.execute(f"PRAGMA table_info({table_name})")
            pragma_results = cursor.fetchall()
            cursor.close()
            results = [(row[1], row[2], 'YES' if row[3] == 0 else 'NO') for row in pragma_results]
        
        return [
            {"name": row[0], "type": row[1], "nullable": row[2]}
            for row in results
        ]
    
    def get_schema_string(self) -> str:
        """Get formatted schema string for all tables."""
        tables = self.get_tables()
        schema_parts = []
        
        for table in tables:
            columns = self.get_schema(table)
            col_str = ", ".join([f"{c['name']} {c['type']}" for c in columns])
            schema_parts.append(f"{table}({col_str})")
        
        return " | ".join(schema_parts)
    
    def close(self):
        """Close the connection."""
        if self.connection:
            self.connection.close()
            logger.info(f"Closed connection to {self.config.name}")
    
    def __enter__(self):
        self.connect()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


class DatabaseManager:
    """Manager for multiple database connections."""
    
    def __init__(self):
        self.connections: Dict[str, DatabaseConnection] = {}
    
    def get_connection(self, db_name: str) -> DatabaseConnection:
        """Get or create a database connection."""
        if db_name not in DATABASES:
            raise ValueError(f"Unknown database: {db_name}. Available: {list(DATABASES.keys())}")
        
        if db_name not in self.connections:
            config = DATABASES[db_name]
            conn = DatabaseConnection(config)
            conn.connect()
            self.connections[db_name] = conn
        
        return self.connections[db_name]
    
    def list_databases(self) -> List[Dict]:
        """List all available databases."""
        return [
            {
                "name": name,
                "type": config.db_type,
                "description": config.description,
                "tables": config.tables
            }
            for name, config in DATABASES.items()
        ]
    
    def test_connection(self, db_name: str) -> Dict:
        """Test a database connection."""
        try:
            conn = self.get_connection(db_name)
            tables = conn.get_tables()
            return {
                "status": "connected",
                "database": db_name,
                "tables": tables,
                "table_count": len(tables)
            }
        except Exception as e:
            return {
                "status": "error",
                "database": db_name,
                "error": str(e)
            }
    
    def execute_sql(self, db_name: str, query: str) -> Dict:
        """Execute SQL on a specific database."""
        try:
            conn = self.get_connection(db_name)
            results = conn.execute(query)
            return {
                "status": "success",
                "database": db_name,
                "query": query,
                "results": results,
                "row_count": len(results)
            }
        except Exception as e:
            return {
                "status": "error",
                "database": db_name,
                "query": query,
                "error": str(e)
            }
    
    def close_all(self):
        """Close all connections."""
        for name, conn in self.connections.items():
            conn.close()
        self.connections.clear()


# Global database manager instance
db_manager = DatabaseManager()


def get_database(name: str) -> DatabaseConnection:
    """Get a database connection by name."""
    return db_manager.get_connection(name)


def list_databases() -> List[Dict]:
    """List all available databases."""
    return db_manager.list_databases()


def execute_query(db_name: str, query: str) -> Dict:
    """Execute a query on a database."""
    return db_manager.execute_sql(db_name, query)
