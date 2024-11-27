import pandas as pd

# Convert to JSON
json_file = "output.json"
data = pd.read_json(json_file, orient="records", lines=True)

# Load the CSV file
csv_file = "output.csv"
data.to_csv(csv_file)
# Randomize the rows order
data = data.sample(frac=1)

# Save the CSV file with randomized rows
randomized_csv_file = "randomized_output.csv"
data.to_csv(randomized_csv_file, index=False)

print(f"JSON converted to CSV with randomized rows: {randomized_csv_file}")
print(f"JSON converted to CSV: {csv_file}")
