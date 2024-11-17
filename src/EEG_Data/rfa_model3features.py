import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Load your data into a DataFrame
data = pd.read_json('output.json', lines=True)

# Separate features and labels
X = data.drop(columns=['filename', 'direction'])
y = data['direction']

# Verify dataset shape
print(f"Dataset shape: {X.shape}")

# Validate indices
selected_features = [0, 82]  # Example indices
if any(idx >= X.shape[1] for idx in selected_features):
    print(f"One or more indices in {selected_features} are out of bounds!")
    selected_features = [0,9,26]  # Use valid indices

# Scaler for normalization
scaler = StandardScaler()

# Parameters
n_iterations = 50  # Number of random seeds to test
random_seeds = np.random.randint(0, 10000, size=n_iterations)  # Generate random seeds

# Store accuracy results
accuracies = []

# Loop through random seeds
for seed in random_seeds:
    print(f"Testing with random seed: {seed}")
    
    # Select specific features
    X_selected = X.iloc[:, selected_features]
    
    # Train-test split with the current seed
    X_train, X_test, y_train, y_test = train_test_split(
        X_selected, y, test_size=0.2, random_state=seed
    )
    
    # Normalize
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Train Random Forest Classifier
    clf = RandomForestClassifier(n_estimators=100, random_state=seed)
    clf.fit(X_train_scaled, y_train)

    # Predict and evaluate
    y_pred = clf.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred)
    accuracies.append(accuracy)
    print(f"Accuracy for seed {seed}: {accuracy}")

# Calculate average accuracy
average_accuracy = np.mean(accuracies)
print(f"\nAverage Accuracy over {n_iterations} seeds: {average_accuracy}")

print(X.columns[0])  # Get the column name for feature index 26
print(X.columns[9])  # Get the column name for feature index 26
print(X.columns[26])  # Get the column name for feature index 26

