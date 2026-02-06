"""Test all database connections via command line."""
import sqlite3

print("=" * 60)
print("SQLITE DATABASE TESTS")
print("=" * 60)

sqlite_dbs = [
    ("geography", "/app/data/text2sql-data/data/geography-db.added-in-2020.sqlite"),
    ("advising", "/app/data/text2sql-data/data/advising-db.added-in-2020.sqlite"),
    ("atis", "/app/data/text2sql-data/data/atis-db.added-in-2020.sqlite"),
    ("restaurants", "/app/data/text2sql-data/data/restaurants-db.added-in-2020.sqlite"),
]

for name, path in sqlite_dbs:
    try:
        conn = sqlite3.connect(path)
        cur = conn.cursor()
        cur.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = [r[0] for r in cur.fetchall()]
        print(f"\n{name}_sqlite: {len(tables)} tables")
        print(f"  Tables: {tables[:5]}{'...' if len(tables) > 5 else ''}")
        
        # Sample query
        if tables:
            cur.execute(f"SELECT * FROM {tables[0]} LIMIT 2")
            rows = cur.fetchall()
            print(f"  Sample from {tables[0]}: {len(rows)} rows")
        conn.close()
        print(f"  ✓ Connection OK")
    except Exception as e:
        print(f"\n{name}_sqlite: ERROR - {e}")

print("\n" + "=" * 60)
print("POSTGRESQL DATABASE TESTS")
print("=" * 60)

import pg8000

pg_dbs = ["geography", "advising", "atis", "restaurants"]

for db_name in pg_dbs:
    try:
        conn = pg8000.connect(
            host="postgres",
            port=5432,
            user="postgres",
            password="postgres",
            database=db_name
        )
        cur = conn.cursor()
        cur.execute("SELECT table_name FROM information_schema.tables WHERE table_schema = 'public'")
        tables = [r[0] for r in cur.fetchall()]
        print(f"\n{db_name}_pg: {len(tables)} tables")
        print(f"  Tables: {tables[:5]}{'...' if len(tables) > 5 else ''}")
        
        # Sample query
        if tables:
            cur.execute(f"SELECT * FROM {tables[0]} LIMIT 2")
            rows = cur.fetchall()
            print(f"  Sample from {tables[0]}: {len(rows)} rows")
        conn.close()
        print(f"  ✓ Connection OK")
    except Exception as e:
        print(f"\n{db_name}_pg: ERROR - {e}")

print("\n" + "=" * 60)
print("ALL DATABASE CONNECTION TESTS COMPLETE!")
print("=" * 60)
