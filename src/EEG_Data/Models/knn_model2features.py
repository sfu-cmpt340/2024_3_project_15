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
data = pd.read_json("../output.json", lines=True)

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

# Hyperparameter grids for GridSearchCV
param_grids = {
    "KNN": {
        "n_neighbors": [3, 5, 7, 9],
        "weights": ["uniform", "distance"],
        "metric": ["euclidean", "manhattan"],
    }
}

# Classifiers to evaluate
classifiers = {"KNN": KNeighborsClassifier()}

# Store results
results = []

# Evaluate each classifier with GridSearchCV
for clf_name, clf in classifiers.items():
    print(f"\nTuning hyperparameters for {clf_name}...")
    grid_search = GridSearchCV(
        estimator=clf,
        param_grid=param_grids[clf_name],
        cv=skf,
        scoring="accuracy",
        n_jobs=-1,
        verbose=2,
    )
    grid_search.fit(X_scaled, y)

    # Best model and score
    best_model = grid_search.best_estimator_
    best_params = grid_search.best_params_
    best_score = grid_search.best_score_

    results.append((clf_name, best_model, best_params, best_score))
    print(f"Best Parameters for {clf_name}: {best_params}")
    print(f"Best Cross-Validation Accuracy for {clf_name}: {best_score:.4f}")

# Find the best classifier
best_classifier = max(results, key=lambda x: x[3])
print(
    f"\nBest Classifier: {best_classifier[0]} with Accuracy: {best_classifier[3]:.4f}"
)
print(f"Best Hyperparameters: {best_classifier[2]}")
