import joblib
import pandas as pd
from sklearn.metrics import accuracy_score

# Load the pre-trained models and scalers
knn_model = joblib.load("./pkl_models/best_knn_model.pkl")
rf_model = joblib.load("./pkl_models/best_random_forest_model.pkl")

knn_scaler = joblib.load("./pkl_models/scaler_KNN.pkl")
rf_scaler = joblib.load("./pkl_models/scaler_RF.pkl")

# Feature lists (ensure these match the training process)
knn_features = ["eeg_1_mean", "eeg_2_mean", "eeg_3_freq_std"]
rf_features = ["eeg_1_mean", "eeg_2_mean", "eeg_3_freq_std"]


# Function to classify test data and compute accuracy
def classify_with_models(test_csv_path, scalers, models, feature_sets):
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

    # Combine predictions using majority voting
    predictions_ensemble = []
    for i in range(len(y_test)):
        votes = [
            predictions_knn[i],
            predictions_rf[i],
        ]
        # Majority vote
        predictions_ensemble.append(max(set(votes), key=votes.count))

    # Compute accuracy for each model and ensemble
    accuracy_knn = accuracy_score(y_test, predictions_knn)
    accuracy_rf = accuracy_score(y_test, predictions_rf)
    accuracy_ensemble = accuracy_score(y_test, predictions_ensemble)

    # Print accuracy results
    print(f"KNN Model Accuracy: {accuracy_knn:.4f}")
    print(f"Random Forest Model Accuracy: {accuracy_rf:.4f}")
    print(f"Voting Ensemble Accuracy: {accuracy_ensemble:.4f}")


# Specify test data file path
test_csv_path = "../testFiles/test_data.csv"

# Map scalers, models, and feature sets for convenience
scalers = {"knn": knn_scaler, "rf": rf_scaler}
models = {"knn": knn_model, "rf": rf_model}
feature_sets = {"knn": knn_features, "rf": rf_features}

# Classify test data and print results
classify_with_models(test_csv_path, scalers, models, feature_sets)
