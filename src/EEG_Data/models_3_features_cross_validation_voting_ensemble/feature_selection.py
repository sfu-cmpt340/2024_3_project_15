import numpy as np
import pandas as pd
from scipy.stats import spearmanr, t
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import mutual_info_classif
from sklearn.inspection import permutation_importance
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.preprocessing import StandardScaler


def calculate_robust_feature_importance(
    X, y, n_iterations=10, n_splits=5, n_estimators=100
):
    """
    Calculate feature importance across multiple methods and cross-validation splits
    """
    # Initialize arrays to store feature importances
    all_rf_importances = np.zeros((n_iterations * n_splits, X.shape[1]))
    all_permutation_importances = np.zeros((n_iterations * n_splits, X.shape[1]))

    # Calculate mutual information once (it's deterministic)
    mutual_info = mutual_info_classif(X, y)

    # Initialize scaler
    scaler = StandardScaler()

    # Perform multiple iterations
    for iteration in range(n_iterations):
        # Use stratified k-fold to maintain class distribution
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=iteration)

        # For each fold in cross-validation
        for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X, y)):
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

            # Scale features
            X_train_scaled = scaler.fit_transform(X_train)
            X_val_scaled = scaler.transform(X_val)

            # Train Random Forest
            rf = RandomForestClassifier(
                n_estimators=n_estimators,
                random_state=iteration,
                class_weight="balanced",
            )
            rf.fit(X_train_scaled, y_train)

            # Store random forest feature importances
            importance_idx = iteration * n_splits + fold_idx
            all_rf_importances[importance_idx] = rf.feature_importances_

            # Calculate permutation importance
            perm_importance = permutation_importance(
                rf, X_val_scaled, y_val, n_repeats=5, random_state=iteration
            )
            all_permutation_importances[importance_idx] = (
                perm_importance.importances_mean
            )

    # Calculate statistics for RF importances
    mean_rf_importance = np.mean(all_rf_importances, axis=0)
    std_rf_importance = np.std(all_rf_importances, axis=0)

    # Calculate statistics for permutation importances
    mean_perm_importance = np.mean(all_permutation_importances, axis=0)
    std_perm_importance = np.std(all_permutation_importances, axis=0)

    # Calculate 95% confidence intervals
    alpha = 0.95
    n_samples = n_iterations * n_splits
    df = n_samples - 1

    # For RF importances
    sem_rf = std_rf_importance / np.sqrt(n_samples)
    t_crit = t.ppf(1 - (1 - alpha) / 2, df)
    CI_Lower_rf = mean_rf_importance - t_crit * sem_rf
    CI_Upper_rf = mean_rf_importance + t_crit * sem_rf

    # For permutation importances
    sem_perm = std_perm_importance / np.sqrt(n_samples)
    CI_Lower_perm = mean_perm_importance - t_crit * sem_perm
    CI_Upper_perm = mean_perm_importance + t_crit * sem_perm

    # Calculate stability metrics
    def calculate_stability_metrics(importances, n_top_features=10):
        ranking_stability = np.zeros(len(X.columns))
        consistency_index = np.zeros(len(X.columns))

        for i in range(len(X.columns)):
            # Ranking stability (frequency in top features)
            top_k_masks = np.array(
                [np.argsort(imp)[::-1][:n_top_features] for imp in importances]
            )
            ranking_stability[i] = np.mean([i in mask for mask in top_k_masks])

            # Consistency index (variance of feature's rank)
            ranks = np.array(
                [list(np.argsort(imp)[::-1]).index(i) for imp in importances]
            )
            consistency_index[i] = 1 / (np.std(ranks) + 1)

        return ranking_stability, consistency_index

    rf_stability, rf_consistency = calculate_stability_metrics(all_rf_importances)
    perm_stability, perm_consistency = calculate_stability_metrics(
        all_permutation_importances
    )

    # Calculate feature type importance
    feature_types = {
        "time_domain": ["mean", "std", "var", "skewness", "kurtosis", "rms", "energy"],
        "frequency_domain": ["dominant_freq", "mean_freq", "spectral_entropy"]
        + ["delta_power", "theta_power", "alpha_power", "beta_power", "gamma_power"],
        "connectivity": ["corr_", "coherence_", "phase_sync_"],
        "nonlinear": ["sample_entropy", "approx_entropy", "lyapunov_exp", "hurst_exp"],
    }

    type_importance = {ftype: 0 for ftype in feature_types}
    for ftype, patterns in feature_types.items():
        type_mask = np.zeros(len(X.columns), dtype=bool)
        for pattern in patterns:
            type_mask |= np.array([pattern in col.lower() for col in X.columns])
        type_importance[ftype] = np.mean(mean_rf_importance[type_mask])

    # Create results DataFrame
    results = pd.DataFrame(
        {
            "Feature_Index": np.arange(len(X.columns)),
            "Feature": X.columns,
            "RF_Mean_Importance": mean_rf_importance,
            "RF_Std_Importance": std_rf_importance,
            "RF_CI_Lower": CI_Lower_rf,
            "RF_CI_Upper": CI_Upper_rf,
            "RF_CV_Robustness": mean_rf_importance / (std_rf_importance + 1e-10),
            "RF_Ranking_Stability": rf_stability,
            "RF_Consistency_Index": rf_consistency,
            "Perm_Mean_Importance": mean_perm_importance,
            "Perm_Std_Importance": std_perm_importance,
            "Perm_CI_Lower": CI_Lower_perm,
            "Perm_CI_Upper": CI_Upper_perm,
            "Perm_Ranking_Stability": perm_stability,
            "Perm_Consistency_Index": perm_consistency,
            "Mutual_Information": mutual_info,
        }
    )

    # Sort by RF mean importance
    results = results.sort_values("RF_Mean_Importance", ascending=False).reset_index(
        drop=True
    )

    # Calculate feature rankings correlation
    ranking_methods = {
        "RF_Importance": "RF_Mean_Importance",
        "Permutation_Importance": "Perm_Mean_Importance",
        "Mutual_Information": "Mutual_Information",
    }

    ranking_correlation = pd.DataFrame(
        index=ranking_methods.keys(), columns=ranking_methods.keys()
    )
    for method1, col1 in ranking_methods.items():
        for method2, col2 in ranking_methods.items():
            corr, _ = spearmanr(results[col1], results[col2])
            ranking_correlation.loc[method1, method2] = corr

    return (
        results,
        all_rf_importances,
        all_permutation_importances,
        type_importance,
        ranking_correlation,
    )


# Load and prepare your data
data = pd.read_csv("../output/output.csv")
X = data.drop(columns=["filename", "direction"])
y = data["direction"]

# Calculate robust feature importance
results, rf_importances, perm_importances, type_importance, ranking_correlation = (
    calculate_robust_feature_importance(X, y)
)

# Print detailed results for top 10 features
print("\nTop 10 Most Important Features with Statistics:")
print(
    results[
        [
            "Feature_Index",
            "Feature",
            "RF_Mean_Importance",
            "RF_Std_Importance",
            "RF_Ranking_Stability",
            "Perm_Mean_Importance",
            "Mutual_Information",
        ]
    ]
    .head(10)
    .to_string(float_format=lambda x: "{:.4f}".format(x), index=False)
)

# Print feature type importance
print("\nFeature Type Importance Analysis:")
for ftype, importance in type_importance.items():
    print(f"{ftype}: {importance:.4f}")

# Print ranking correlation
print("\nFeature Ranking Method Correlation:")
print(ranking_correlation.round(3))

# Save detailed results
results.to_csv("feature_importance_analysis.csv", index=False)
