"""Convert halflife_frequency.csv to final_halflifes.json format."""
import pandas as pd
import json
from pathlib import Path

# Paths
REPO_ROOT = Path(__file__).parent.parent
CSV_PATH = REPO_ROOT / "artifacts" / "ewma_halflife_selection" / "aggregation" / "halflife_frequency.csv"
JSON_PATH = REPO_ROOT / "artifacts" / "ewma_halflife_selection" / "aggregation" / "final_halflifes.json"

# Alternative CSV path (in case it's named differently)
CSV_PATH_ALT = REPO_ROOT / "artifacts" / "feature_scale" / "halflife_frequency.csv"

print("=" * 80)
print("HALFLIFE CSV TO JSON CONVERTER")
print("=" * 80)

# Try to find the CSV file
csv_path = None
if CSV_PATH.exists():
    csv_path = CSV_PATH
    print(f"Found CSV at: {CSV_PATH}")
elif CSV_PATH_ALT.exists():
    csv_path = CSV_PATH_ALT
    print(f"Found CSV at: {CSV_PATH_ALT}")
else:
    print(f"ERROR: CSV file not found at either:")
    print(f"  - {CSV_PATH}")
    print(f"  - {CSV_PATH_ALT}")
    print("\nPlease provide the correct path to your halflife_frequency.csv file")
    exit(1)

# Read CSV
print("\nReading CSV file...")
df = pd.read_csv(csv_path)
print(f"  Loaded {len(df)} rows")
print(f"  Columns: {list(df.columns)}")

# Show first few rows
print("\nFirst 5 rows:")
print(df.head())

# Expected format: feature,half_life or similar
# Try to detect column names
if 'feature' in df.columns and 'half_life' in df.columns:
    feature_col = 'feature'
    halflife_col = 'half_life'
elif 'feature' in df.columns and 'halflife' in df.columns:
    feature_col = 'feature'
    halflife_col = 'halflife'
elif 'feature_name' in df.columns and 'half_life' in df.columns:
    feature_col = 'feature_name'
    halflife_col = 'half_life'
else:
    # Assume first column is feature, second is half_life
    feature_col = df.columns[0]
    halflife_col = df.columns[1]
    print(f"\nAssuming columns: feature='{feature_col}', half_life='{halflife_col}'")

# Convert to dictionary
halflife_dict = {}
for _, row in df.iterrows():
    feature = row[feature_col]
    halflife = row[halflife_col]

    # Convert halflife to int if it's a number
    try:
        halflife = int(halflife)
    except:
        pass

    halflife_dict[feature] = halflife

print(f"\nConverted {len(halflife_dict)} feature half-lives")

# Sample output
print("\nSample half-lives:")
for i, (feat, hl) in enumerate(list(halflife_dict.items())[:5]):
    print(f"  {feat}: {hl}")
if len(halflife_dict) > 5:
    print(f"  ... and {len(halflife_dict) - 5} more")

# Create output directory if needed
JSON_PATH.parent.mkdir(parents=True, exist_ok=True)

# Write JSON
print(f"\nWriting to: {JSON_PATH}")
with open(JSON_PATH, 'w') as f:
    json.dump(halflife_dict, f, indent=2)

print("\n✓ Conversion complete!")
print("=" * 80)
