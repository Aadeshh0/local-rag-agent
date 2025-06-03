import os
import pandas as pd

def check_file_structure():
    """Debug script to check file structure and paths"""
    print("🔍 DEBUG: Checking file structure...")
    print("=" * 50)
    
    # Current working directory
    cwd = os.getcwd()
    print(f"📁 Current working directory: {cwd}")
    
    # List contents of current directory
    print(f"\n📂 Contents of '{cwd}':")
    try:
        for item in os.listdir(cwd):
            item_path = os.path.join(cwd, item)
            item_type = "📁" if os.path.isdir(item_path) else "📄"
            print(f"  {item_type} {item}")
    except Exception as e:
        print(f"  ❌ Error listing directory: {e}")
    
    # Check for data directory
    data_path = os.path.join(cwd, 'data')
    print(f"\n📁 Checking data directory: {data_path}")
    if os.path.exists(data_path):
        print("  ✅ Data directory exists!")
        print(f"  📂 Contents of 'data':")
        try:
            for item in os.listdir(data_path):
                item_path = os.path.join(data_path, item)
                item_type = "📁" if os.path.isdir(item_path) else "📄"
                size = os.path.getsize(item_path) if os.path.isfile(item_path) else "N/A"
                print(f"    {item_type} {item} ({size} bytes)" if size != "N/A" else f"    {item_type} {item}")
        except Exception as e:
            print(f"    ❌ Error listing data directory: {e}")
    else:
        print("  ❌ Data directory NOT found!")
    
    # Check for CSV file specifically
    csv_file = os.path.join(data_path, 'realistic_restaurant_reviews.csv')
    print(f"\n📄 Checking CSV file: {csv_file}")
    if os.path.exists(csv_file):
        print("  ✅ CSV file exists!")
        try:
            # Check CSV contents
            df = pd.read_csv(csv_file)
            print(f"  📊 CSV has {len(df)} rows and {len(df.columns)} columns")
            print(f"  📋 Columns: {list(df.columns)}")
            print(f"  📝 Sample data:")
            print(df.head(2).to_string(index=False))
        except Exception as e:
            print(f"  ❌ Error reading CSV: {e}")
    else:
        print("  ❌ CSV file NOT found!")
        
        # Suggest alternative paths
        print("\n🔍 Looking for CSV in alternative locations...")
        possible_paths = [
            os.path.join(cwd, 'realistic_restaurant_reviews.csv'),
            os.path.join(cwd, 'src', 'data', 'realistic_restaurant_reviews.csv'),
            os.path.join(cwd, '..', 'data', 'realistic_restaurant_reviews.csv')
        ]
        
        for path in possible_paths:
            if os.path.exists(path):
                print(f"  ✅ Found CSV at: {path}")
            else:
                print(f"  ❌ Not found at: {path}")
    
    print("\n" + "=" * 50)
    print("🚀 If CSV is found, your config should use the correct path.")
    print("🚀 If not found, make sure to copy your CSV to the data/ directory.")

if __name__ == "__main__":
    check_file_structure()