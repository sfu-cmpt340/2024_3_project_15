import joblib
import pandas as pd
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import accuracy_score

# Load the pre-trained models and scalers
knn_model = joblib.load("./pkl_models/best_knn_model.pkl")
rf_model = joblib.load("./pkl_models/best_random_forest_model.pkl")
svm_model = joblib.load("./pkl_models/best_svm_model.pkl")

knn_scaler = joblib.load("./pkl_models/scaler_KNN.pkl")
rf_scaler = joblib.load("./pkl_models/scaler_RF.pkl")
svm_scaler = joblib.load("./pkl_models/scaler_SVM.pkl")

# Feature lists (ensure these match the training process)
knn_features = [
    "eeg_1_mean",
    "eeg_2_mean",
    "eeg_3_freq_std",
]  # Replace with actual features used for KNN
rf_features = [
    "eeg_1_mean",
    "eeg_2_mean",
    "eeg_3_freq_std",
]  # Replace with actual features used for RF
svm_features = [
    "eeg_1_mean",
    "eeg_2_mean",
    "eeg_3_freq_std",
]  # Replace with actual features used for SVM

# Create a voting ensemble model
voting_ensemble = VotingClassifier(
    estimators=[
        ("knn", knn_model),
        ("random_forest", rf_model),
        ("svm", svm_model),
    ],
    voting="hard",  # 'hard' for majority voting
)


# Function to classify test data and compute accuracy
def classify_with_ensemble(
    test_csv_path, ensemble_model, scalers, models, feature_sets
):
    # Load test data
    test_data = pd.read_csv(test_csv_path)

    # Extract labels
    y_test = test_data["direction"]

    # Normalize test data for each model
    X_test_scaled = {}
    for model_name, features in feature_sets.items():
        X_test = test_data[features]
        X_test_scaled[model_name] = scalers[model_name].transform(X_test)

    # Use individual models for predictions
    predictions_knn = models["knn"].predict(X_test_scaled["knn"])
    predictions_rf = models["rf"].predict(X_test_scaled["rf"])
    predictions_svm = models["svm"].predict(X_test_scaled["svm"])

    # Combine predictions using the ensemble
    # Assuming ensemble works on the first normalized set (all sets should be consistent in size)
    predictions_ensemble = ensemble_model.predict(X_test_scaled["svm"])

    # Compute accuracy for each model and ensemble
    accuracy_knn = accuracy_score(y_test, predictions_knn)
    accuracy_rf = accuracy_score(y_test, predictions_rf)
    accuracy_svm = accuracy_score(y_test, predictions_svm)
    accuracy_ensemble = accuracy_score(y_test, predictions_ensemble)

    # Print accuracy results
    print(f"KNN Model Accuracy: {accuracy_knn:.4f}")
    print(f"Random Forest Model Accuracy: {accuracy_rf:.4f}")
    print(f"SVM Model Accuracy: {accuracy_svm:.4f}")
    print(f"Voting Ensemble Accuracy: {accuracy_ensemble:.4f}")


# Specify test data file path
test_csv_path = "../testFiles/test_data.csv"  # Path to the CSV file with test data

# Map scalers, models, and feature sets for convenience
scalers = {"knn": knn_scaler, "rf": rf_scaler, "svm": svm_scaler}
models = {"knn": knn_model, "rf": rf_model, "svm": svm_model}
feature_sets = {"knn": knn_features, "rf": rf_features, "svm": svm_features}

# Classify test data and print results
classify_with_ensemble(test_csv_path, voting_ensemble, scalers, models, feature_sets)
