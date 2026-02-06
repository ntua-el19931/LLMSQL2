"""Test database connections."""
from src.database import list_databases, get_database, DatabaseManager, DATABASES

# List all available databases
print('=' * 60)
print('AVAILABLE DATABASES')
print('=' * 60)
for db in list_databases():
    print(f"  - {db['name']}: {db['type'].upper()}")

# Test SQLite connections
print('\n' + '=' * 60)
print('TESTING SQLITE DATABASES')
print('=' * 60)
for db_name in ['geography_sqlite', 'advising_sqlite', 'atis_sqlite', 'restaurants_sqlite']:
    try:
        with get_database(db_name) as db:
            tables = db.get_tables()
            print(f'\n{db_name}:')
            print(f'  Tables ({len(tables)}): {tables[:5]}{"..." if len(tables) > 5 else ""}')
            # Sample query
            if tables:
                results = db.execute(f"SELECT * FROM {tables[0]} LIMIT 3")
                print(f'  Sample from {tables[0]}: {len(results)} rows')
    except Exception as e:
        print(f'{db_name}: ERROR - {e}')

# Test PostgreSQL connections
print('\n' + '=' * 60)
print('TESTING POSTGRESQL DATABASES')
print('=' * 60)
for db_name in ['geography_pg', 'advising_pg', 'atis_pg', 'restaurants_pg']:
    try:
        with get_database(db_name) as db:
            tables = db.get_tables()
            print(f'\n{db_name}:')
            print(f'  Tables ({len(tables)}): {tables}')
            # Sample query
            if tables:
                results = db.execute(f"SELECT * FROM {tables[0]} LIMIT 3")
                print(f'  Sample from {tables[0]}: {len(results)} rows')
    except Exception as e:
        print(f'{db_name}: ERROR - {e}')

print('\n' + '=' * 60)
print('ALL DATABASE TESTS COMPLETED!')
print('=' * 60)
