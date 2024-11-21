import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

# Load your data into a DataFrame
data = pd.read_json("../output/output.json", lines=True)

# Separate features and labels
X = data.drop(columns=["filename", "direction"])
y = data["direction"]

# Verify dataset shape
print(f"Dataset shape: {X.shape}")

# Feature selection using Mutual Information
k_best_features = 10  # Number of features to select
selector = SelectKBest(mutual_info_classif, k=k_best_features)
X_selected = selector.fit_transform(X, y)

selected_feature_indices = selector.get_support(indices=True)
selected_feature_names = X.columns[selected_feature_indices]
print(f"Selected features: {selected_feature_names}")

# Scaler for normalization
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_selected)

# Cross-validation setup
k_folds = 5
skf = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=42)

# Hyperparameter grid for SVM
svm_param_grid = {
    "C": [0.1, 1, 10, 100],
    "kernel": ["linear", "poly", "rbf", "sigmoid"],
    "gamma": ["scale", "auto"],
}

# Initialize SVM classifier
svm = SVC()

# GridSearchCV for SVM
print("\nTuning hyperparameters for SVM...")
svm_grid_search = GridSearchCV(
    estimator=svm,
    param_grid=svm_param_grid,
    cv=skf,
    scoring="accuracy",
    n_jobs=-1,
    verbose=2,
)

# Fit the model
svm_grid_search.fit(X_scaled, y)

# Extract best model and results
svm_best_model = svm_grid_search.best_estimator_
svm_best_params = svm_grid_search.best_params_
svm_best_score = svm_grid_search.best_score_

print(f"Best Parameters for SVM: {svm_best_params}")
print(f"Best Cross-Validation Accuracy for SVM: {svm_best_score:.4f}")
