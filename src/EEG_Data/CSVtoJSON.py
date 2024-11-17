import pandas as pd

# Load the CSV file
csv_file = 'output.csv'
data = pd.read_csv(csv_file)

# Convert to JSON
json_file = 'output.json'
data.to_json(json_file, orient='records', lines=True)

print(f"CSV converted to JSON: {json_file}")
