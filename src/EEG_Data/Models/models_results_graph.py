import joblib
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, roc_curve, auc, confusion_matrix, ConfusionMatrixDisplay, classification_report, precision_recall_curve, average_precision_score

# Load the pre-trained models and scalers
knn_model = joblib.load("pkl_models/best_knn_model.pkl")
rf_model = joblib.load("pkl_models/best_random_forest_model.pkl")
svm_model = joblib.load("pkl_models/best_svm_model.pkl")

knn_scaler = joblib.load("pkl_models/scaler_KNN.pkl")
rf_scaler = joblib.load("pkl_models/scaler_RF.pkl")
svm_scaler = joblib.load("pkl_models/scaler_SVM.pkl")

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

# Function to classify test data, compute accuracy, and plot results
def classify_with_models(
    test_csv_path, scalers, models, feature_sets
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

    # Combine predictions using majority voting
    predictions_ensemble = []
    for i in range(len(y_test)):
        votes = [
            predictions_knn[i],
            predictions_rf[i],
            predictions_svm[i],
        ]
        # Majority vote
        predictions_ensemble.append(max(set(votes), key=votes.count))

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

    # Plot metrics for individual models
    for model_name, predictions in zip(["KNN", "Random Forest", "SVM"], [predictions_knn, predictions_rf, predictions_svm]):
        # Confusion Matrix
        conf_matrix = confusion_matrix(y_test, predictions, labels=[0, 1])
        disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix, display_labels=["False", "True"])
        disp.plot(cmap=plt.cm.Blues)
        plt.title(f'Confusion Matrix for {model_name}')
        plt.show()

        # Classification Report
        print(f"\nClassification Report for {model_name}:")
        print(classification_report(y_test, predictions))

        # ROC Curve (if binary classification)
        if len(y_test.unique()) == 2:
            fpr, tpr, _ = roc_curve(y_test, predictions)
            roc_auc = auc(fpr, tpr)

            plt.figure()
            plt.plot(fpr, tpr, lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
            plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title(f'Receiver Operating Characteristic (ROC) Curve for {model_name}')
            plt.legend(loc='lower right')
            plt.show()

        # Precision-Recall Curve (if binary classification)
        if len(y_test.unique()) == 2:
            precision, recall, _ = precision_recall_curve(y_test, predictions)
            average_precision = average_precision_score(y_test, predictions)

            plt.figure()
            plt.step(recall, precision, color='b', alpha=0.2, where='post')
            plt.fill_between(recall, precision, step='post', alpha=0.2, color='b')
            plt.xlabel('Recall')
            plt.ylabel('Precision')
            plt.title(f'Precision-Recall Curve for {model_name}: AP={average_precision:.2f}')
            plt.show()

    # Confusion Matrix for Ensemble
    conf_matrix = confusion_matrix(y_test, predictions_ensemble, labels=[0, 1])
    disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix, display_labels=["False", "True"])
    disp.plot(cmap=plt.cm.Blues)
    plt.title('Confusion Matrix for Voting Ensemble')
    plt.show()

    # Classification Report for Ensemble
    print("\nClassification Report for Voting Ensemble:")
    print(classification_report(y_test, predictions_ensemble))

    # ROC Curve for Ensemble (if binary classification)
    if len(y_test.unique()) == 2:
        fpr, tpr, _ = roc_curve(y_test, predictions_ensemble)
        roc_auc = auc(fpr, tpr)

        plt.figure()
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) Curve for Voting Ensemble')
        plt.legend(loc='lower right')
        plt.show()
    else:
        print("ROC curve is not applicable for multi-class classification.")

    # Precision-Recall Curve for Ensemble (if binary classification)
    if len(y_test.unique()) == 2:
        precision, recall, _ = precision_recall_curve(y_test, predictions_ensemble)
        average_precision = average_precision_score(y_test, predictions_ensemble)

        plt.figure()
        plt.step(recall, precision, color='b', alpha=0.2, where='post')
        plt.fill_between(recall, precision, step='post', alpha=0.2, color='b')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title(f'Precision-Recall Curve for Voting Ensemble: AP={average_precision:.2f}')
        plt.show()
    else:
        print("Precision-Recall curve is not applicable for multi-class classification.")

# Specify test data file path
test_csv_path = "../testFiles/test_data.csv"

# Map scalers, models, and feature sets for convenience
scalers = {"knn": knn_scaler, "rf": rf_scaler, "svm": svm_scaler}
models = {"knn": knn_model, "rf": rf_model, "svm": svm_model}
feature_sets = {"knn": knn_features, "rf": rf_features, "svm": svm_features}

# Classify test data and plot results
classify_with_models(test_csv_path, scalers, models, feature_sets)
