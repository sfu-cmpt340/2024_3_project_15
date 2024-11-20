import joblib
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    auc,
    average_precision_score,
    classification_report,
    confusion_matrix,
    precision_recall_curve,
    roc_curve,
)

# Load the pre-trained models and scalers
KNN_MODEL = joblib.load("models/best_knn_model.pkl")
RF_MODEL = joblib.load("models/best_random_forest_model.pkl")
SVM_MODEL = joblib.load("models/best_svm_model.pkl")
KNN_SCALER = joblib.load("models/scaler_KNN.pkl")
RF_SCALER = joblib.load("models/scaler_RF.pkl")
SVM_SCALER = joblib.load("models/scaler_SVM.pkl")

# Feature lists
KNN_FEATURES = [
    "eeg_1_mean",
    "eeg_2_mean",
    "eeg_3_freq_std",
]

RF_FEATURES = [
    "eeg_1_mean",
    "eeg_2_mean",
    "eeg_3_freq_std",
]

SVM_FEATURES = [
    "eeg_1_mean",
    "eeg_2_mean",
    "eeg_3_freq_std",
]


def classify_with_models(test_csv_path, knn=1, rf=1, svm=1):
    """
    Classify test data using multiple models and combine predictions using a voting ensemble.
    Parameters:
    test_csv_path (str): Path to the CSV file containing the test data.
    knn, rf, svm (int): Binary flags (1 or 0). If 0, the corresponding model is skipped.
    """

    # Map scalers, models, and feature sets for convenience
    scalers = {"knn": KNN_SCALER, "rf": RF_SCALER, "svm": SVM_SCALER}
    models = {"knn": KNN_MODEL, "rf": RF_MODEL, "svm": SVM_MODEL}
    feature_sets = {"knn": KNN_FEATURES, "rf": RF_FEATURES, "svm": SVM_FEATURES}
    active_models = {"knn": knn, "rf": rf, "svm": svm}

    # Load test data
    test_data = pd.read_csv(test_csv_path)

    # Extract labels
    y_test = test_data["direction"]

    # Initialize predictions dictionary
    predictions = {}
    accuracies = {}

    # Normalize test data and predict for active models
    for model_name, is_active in active_models.items():
        if is_active:
            features = feature_sets[model_name]
            X_test = test_data[features]
            X_test_scaled = scalers[model_name].transform(X_test)
            predictions[model_name] = models[model_name].predict(X_test_scaled)
            accuracies[model_name] = accuracy_score(y_test, predictions[model_name])
            print(f"{model_name.upper()} Model Accuracy: {accuracies[model_name]:.4f}")

    # Combine predictions using majority voting if multiple models are active
    if len(predictions) > 0:
        predictions_ensemble = []
        for i in range(len(y_test)):
            votes = [pred[i] for pred in predictions.values()]
            predictions_ensemble.append(max(set(votes), key=votes.count))

        # Compute and print ensemble accuracy
        accuracy_ensemble = accuracy_score(y_test, predictions_ensemble)
        print(f"Voting Ensemble Accuracy: {accuracy_ensemble:.4f}")

        # Confusion Matrix for Ensemble
        conf_matrix = confusion_matrix(y_test, predictions_ensemble, labels=[0, 1])
        disp = ConfusionMatrixDisplay(
            confusion_matrix=conf_matrix, display_labels=["False", "True"]
        )
        disp.plot(cmap=plt.cm.Blues)
        plt.title("Confusion Matrix for Voting Ensemble")
        plt.show()

        # Classification Report for Ensemble
        print("\nClassification Report for Voting Ensemble:")
        print(classification_report(y_test, predictions_ensemble))

        # ROC and Precision-Recall Curves for Ensemble (if binary classification)
        if len(y_test.unique()) == 2:
            fpr, tpr, _ = roc_curve(y_test, predictions_ensemble)
            roc_auc = auc(fpr, tpr)

            plt.figure()
            plt.plot(
                fpr,
                tpr,
                color="darkorange",
                lw=2,
                label=f"ROC curve (area = {roc_auc:.2f})",
            )
            plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
            plt.xlabel("False Positive Rate")
            plt.ylabel("True Positive Rate")
            plt.title(
                "Receiver Operating Characteristic (ROC) Curve for Voting Ensemble"
            )
            plt.legend(loc="lower right")
            plt.show()

            precision, recall, _ = precision_recall_curve(y_test, predictions_ensemble)
            average_precision = average_precision_score(y_test, predictions_ensemble)

            plt.figure()
            plt.step(recall, precision, color="b", alpha=0.2, where="post")
            plt.fill_between(recall, precision, step="post", alpha=0.2, color="b")
            plt.xlabel("Recall")
            plt.ylabel("Precision")
            plt.title(
                f"Precision-Recall Curve for Voting Ensemble: AP={average_precision:.2f}"
            )
            plt.show()
        else:
            print(
                "ROC and Precision-Recall curves are not applicable for multi-class classification."
            )

    # Plot metrics for individual models
    for model_name, is_active in active_models.items():
        if is_active:
            # Confusion Matrix
            conf_matrix = confusion_matrix(
                y_test, predictions[model_name], labels=[0, 1]
            )
            disp = ConfusionMatrixDisplay(
                confusion_matrix=conf_matrix, display_labels=["False", "True"]
            )
            disp.plot(cmap=plt.cm.Blues)
            plt.title(f"Confusion Matrix for {model_name.upper()}")
            plt.show()

            # Classification Report
            print(f"\nClassification Report for {model_name.upper()}:")
            print(classification_report(y_test, predictions[model_name]))

            # ROC and Precision-Recall Curves (if binary classification)
            if len(y_test.unique()) == 2:
                fpr, tpr, _ = roc_curve(y_test, predictions[model_name])
                roc_auc = auc(fpr, tpr)

                plt.figure()
                plt.plot(fpr, tpr, lw=2, label=f"ROC curve (area = {roc_auc:.2f})")
                plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
                plt.xlabel("False Positive Rate")
                plt.ylabel("True Positive Rate")
                plt.title(f"ROC Curve for {model_name.upper()}")
                plt.legend(loc="lower right")
                plt.show()

                precision, recall, _ = precision_recall_curve(
                    y_test, predictions[model_name]
                )
                average_precision = average_precision_score(
                    y_test, predictions[model_name]
                )

                plt.figure()
                plt.step(recall, precision, color="b", alpha=0.2, where="post")
                plt.fill_between(recall, precision, step="post", alpha=0.2, color="b")
                plt.xlabel("Recall")
                plt.ylabel("Precision")
                plt.title(
                    f"Precision-Recall Curve for {model_name.upper()}: AP={average_precision:.2f}"
                )
                plt.show()

    if len(predictions) == 0:
        print("No models were active. Ensure at least one model is enabled.")
