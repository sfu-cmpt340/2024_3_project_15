import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import ParameterGrid, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

# Load your data into a DataFrame
data = pd.read_csv("../output/randomized_output.csv")

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

# Define the hyperparameter grid for SVM
param_grid = {
    "C": [3, 4, 5, 6],  # Regularization parameter                         0.5945
    "kernel": ["linear", "rbf", "poly"],  # Kernel type
    "gamma": ["scale", "auto"],  # Kernel coefficient (ignored for linear kernel)
    "degree": [1, 2, 3],  # Degree for polynomial kernel
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

        # Initialize the SVM classifier with current hyperparameters
        clf = SVC(random_state=seed, **params)

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

import joblib

# Extract the best hyperparameters
best_params = best_result["params"]
print(f"Training the best SVM model with hyperparameters: {best_params}")

# Train the final SVM model with the best hyperparameters
best_svm_model = SVC(random_state=42, **best_params)
best_svm_model.fit(X_scaled, y)

# Save the scaler
scaler_path = "scaler_SVM.pkl"
joblib.dump(scaler, scaler_path)
print(f"Scaler saved as '{scaler_path}'")

# Save the best SVM model
svm_model_path = "best_SVM_model.pkl"
joblib.dump(best_svm_model, svm_model_path)
print(f"Model saved as '{svm_model_path}'")
