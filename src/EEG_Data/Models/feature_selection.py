import numpy as np
import pandas as pd
from scipy.stats import t
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold


def calculate_robust_feature_importance(
    X, y, n_iterations=10, n_splits=5, n_estimators=100
):
    """
    Calculate feature importance across multiple cross-validation splits and iterations
    """
    # Initialize array to store feature importances
    all_importances = np.zeros((n_iterations * n_splits, X.shape[1]))

    # Perform multiple iterations
    for iteration in range(n_iterations):
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=iteration)

        # For each fold in cross-validation
        for fold_idx, (train_idx, val_idx) in enumerate(kf.split(X)):
            # Train Random Forest
            rf = RandomForestClassifier(
                n_estimators=n_estimators, random_state=iteration
            )
            rf.fit(X.iloc[train_idx], y.iloc[train_idx])

            # Store feature importances
            importance_idx = iteration * n_splits + fold_idx
            all_importances[importance_idx] = rf.feature_importances_

    # Calculate statistics
    mean_importance = np.mean(all_importances, axis=0)
    std_importance = np.std(all_importances, axis=0)

    # Calculate 95% confidence intervals manually
    alpha = 0.95
    n_samples = n_iterations * n_splits
    df = n_samples - 1
    sem = std_importance / np.sqrt(n_samples)
    t_crit = t.ppf(1 - (1 - alpha) / 2, df)
    CI_Lower = mean_importance - t_crit * sem
    CI_Upper = mean_importance + t_crit * sem

    # Calculate stability of rankings
    n_top_features = 10
    top_features_mask = np.zeros((len(X.columns), n_iterations * n_splits), dtype=bool)

    for i in range(n_iterations * n_splits):
        importance_ranking = np.argsort(all_importances[i])[::-1]
        top_features_mask[importance_ranking[:n_top_features], i] = True

    feature_stability = np.mean(top_features_mask, axis=1)

    # Get feature indices
    feature_indices = np.arange(len(X.columns))

    # Create results DataFrame
    results = pd.DataFrame(
        {
            "Feature_Index": feature_indices,
            "Feature": X.columns,
            "Mean_Importance": mean_importance,
            "Std_Importance": std_importance,
            "CI_Lower": CI_Lower,
            "CI_Upper": CI_Upper,
            "CV_Robustness": mean_importance
            / (std_importance + 1e-10),  # Higher is more robust
            "Ranking_Stability": feature_stability,
        }
    )

    # Sort by mean importance
    results = results.sort_values("Mean_Importance", ascending=False).reset_index(
        drop=True
    )

    return results, all_importances


# Load and prepare your data (using your existing code)
data = pd.read_json("../output.json", lines=True)
X = data.drop(columns=["filename", "direction"])
y = data["direction"]

# Calculate robust feature importance
results, all_importances = calculate_robust_feature_importance(X, y)

# Print detailed results for top 10 features, including feature index
print("\nTop 10 Most Important Features with Statistics:")
print(
    results.head(10).to_string(float_format=lambda x: "{:.4f}".format(x), index=False)
)

# Print summary of stability analysis, including feature index
print("\nFeature Stability Analysis:")
print(
    results[["Feature_Index", "Feature", "Mean_Importance", "Ranking_Stability"]]
    .head(10)
    .to_string(float_format=lambda x: "{:.4f}".format(x), index=False)
)
