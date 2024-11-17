import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import ParameterGrid, train_test_split
from sklearn.preprocessing import StandardScaler

# Load your data into a DataFrame
data = pd.read_json("../output.json", lines=True)

# Separate features and labels
X = data.drop(columns=["filename", "direction"])
y = data["direction"]

# Verify dataset shape
print(f"Dataset shape: {X.shape}")

# Validate indices
selected_features = [0, 9, 26]  # Valid feature indices
X_selected = X.iloc[:, selected_features]

# Normalize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_selected)

# Define the hyperparameter grid
param_grid = {
    "n_estimators": [50, 100, 200],
    "max_depth": [10, 20, None],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf": [1, 2, 4],
}

# Generate all combinations of hyperparameters
grid = list(ParameterGrid(param_grid))

# Initialize to store results
results = []

# Evaluate each hyperparameter combination over 50 runs
for params in grid:
    print(f"\nEvaluating hyperparameters: {params}")
    accuracies = []

    for seed in range(50):  # 50 runs with different random seeds
        # Train-test split with varying random seeds
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, train_size=0.8, test_size=0.2, random_state=seed
        )

        # Initialize the classifier with current hyperparameters
        clf = RandomForestClassifier(random_state=seed, **params)

        # Train and evaluate the model
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        accuracies.append(accuracy)

    # Calculate average accuracy for this hyperparameter combination
    avg_accuracy = np.mean(accuracies)
    results.append({"params": params, "average_accuracy": avg_accuracy})
    print(f"Average Accuracy over 50 runs: {avg_accuracy:.4f}")

# Find the best hyperparameters based on average accuracy
best_result = max(results, key=lambda x: x["average_accuracy"])
print(f"\nBest Hyperparameters: {best_result['params']}")
print(f"Best Average Accuracy: {best_result['average_accuracy']:.4f}")
