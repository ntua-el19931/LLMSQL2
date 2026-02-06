import sqlite3

# Use the actual database file
db_path = '/app/data/text2sql-data/data/geography-db.added-in-2020.sqlite'
conn = sqlite3.connect(db_path)
cursor = conn.cursor()

# Get all tables
cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
tables = cursor.fetchall()

print(f"Database: {db_path}")
print(f"\nTables found: {len(tables)}")
print("="*50)

for (table_name,) in tables:
    print(f"\nTable: {table_name}")
    cursor.execute(f"PRAGMA table_info({table_name})")
    columns = cursor.fetchall()
    for col in columns:
        print(f"  - {col[1]} ({col[2]})")
    
    # Show row count
    cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
    count = cursor.fetchone()[0]
    print(f"  Rows: {count}")

conn.close()
