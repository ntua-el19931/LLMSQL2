"""Test text2sql models with real database execution."""
from src.database import get_database, list_databases, execute_query
from src.model_inference import load_gpt2_model, load_model

print("=" * 70)
print("TEXT2SQL MODEL + DATABASE INTEGRATION TEST")
print("=" * 70)

# Test questions for each database - with proper schema format
test_cases = [
    {
        "database": "geography_sqlite",
        "question": "Select the capital from state where state_name equals texas",
        "schema": "state: state_name, population, area, country_name, capital, density"
    },
    {
        "database": "geography_sqlite", 
        "question": "Select city_name from city where state_name equals california",
        "schema": "city: city_name, population, country_name, state_name"
    },
    {
        "database": "atis_sqlite",
        "question": "Select all airline_name from airline",
        "schema": "airline: airline_code, airline_name, note"
    },
    {
        "database": "restaurants_sqlite",
        "question": "Select max of RATING from RESTAURANT",
        "schema": "RESTAURANT: ID, NAME, FOOD_TYPE, CITY_NAME, RATING"
    },
]

# ============================================================
# TEST 1: GPT-2 Model (n22t7a/text2sql-tuned-gpt2)
# ============================================================
print("\n" + "=" * 70)
print("MODEL 1: GPT-2 Text2SQL (n22t7a/text2sql-tuned-gpt2)")
print("=" * 70)

try:
    print("\nLoading GPT-2 model...")
    gpt2_model = load_gpt2_model()
    print("GPT-2 model loaded!")
    
    gpt2_success = 0
    for i, test in enumerate(test_cases, 1):
        print(f"\n--- GPT-2 Test {i}: {test['database']} ---")
        print(f"Question: {test['question']}")
        
        sql = gpt2_model.generate_sql(test['question'], test['schema'])
        print(f"Generated SQL: {sql}")
        
        try:
            result = execute_query(test['database'], sql)
            if result['status'] == 'success':
                rows = result['results'][:3]
                print(f"✓ Results ({result['row_count']} rows): {rows}")
                gpt2_success += 1
            else:
                print(f"✗ Error: {result['error']}")
        except Exception as e:
            print(f"✗ Error: {e}")
    
    print(f"\nGPT-2 Results: {gpt2_success}/{len(test_cases)} successful")
except Exception as e:
    print(f"GPT-2 model error: {e}")

# ============================================================
# TEST 2: TinyLlama Model (ManthanKulakarni/TinyLlama-1.1B-Text2SQL)
# ============================================================
print("\n" + "=" * 70)
print("MODEL 2: TinyLlama Text2SQL (ManthanKulakarni/TinyLlama-1.1B-Text2SQL)")
print("=" * 70)

try:
    print("\nLoading TinyLlama model...")
    tinyllama_model = load_model("ManthanKulakarni/TinyLlama-1.1B-Text2SQL")
    print("TinyLlama model loaded!")
    
    tinyllama_success = 0
    for i, test in enumerate(test_cases, 1):
        print(f"\n--- TinyLlama Test {i}: {test['database']} ---")
        print(f"Question: {test['question']}")
        
        sql = tinyllama_model.generate_sql(test['question'], test['schema'])
        print(f"Generated SQL: {sql}")
        
        try:
            result = execute_query(test['database'], sql)
            if result['status'] == 'success':
                rows = result['results'][:3]
                print(f"✓ Results ({result['row_count']} rows): {rows}")
                tinyllama_success += 1
            else:
                print(f"✗ Error: {result['error']}")
        except Exception as e:
            print(f"✗ Error: {e}")
    
    print(f"\nTinyLlama Results: {tinyllama_success}/{len(test_cases)} successful")
except Exception as e:
    print(f"TinyLlama model error: {e}")

print("\n" + "=" * 70)
print("INTEGRATION TEST COMPLETE!")
print("=" * 70)
