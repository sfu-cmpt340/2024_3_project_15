import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import skew, kurtosis
from scipy.fft import fft
from sklearn.decomposition import PCA

# Load datasets 
scrolling_datasets = {
    'Scrolling Down': pd.read_csv('./Gilbert/museFiles/scrolling_down_60_1_gilbert.csv')[['timestamps', 'eeg_1', 'eeg_2', 'eeg_3']],
    'Scrolling Up': pd.read_csv('./Gilbert/museFiles/scrolling_up_60_1_gilbert.csv')[['timestamps', 'eeg_1', 'eeg_2', 'eeg_3']],
}

swiping_datasets = {
    'Swipe Left': pd.read_csv('./Gilbert/museFiles/swipe_left_60_1_gilbert.csv')[['timestamps', 'eeg_1', 'eeg_2', 'eeg_3']],
    'Swipe Right': pd.read_csv('./Gilbert/museFiles/swiping_right_60_1_gilbert.csv')[['timestamps', 'eeg_1', 'eeg_2', 'eeg_3']],
    'Swipe Up': pd.read_csv('./Gilbert/museFiles/swiping_up_60_1_gilbert.csv')[['timestamps', 'eeg_1', 'eeg_2', 'eeg_3']],
    'Swipe Down': pd.read_csv('./Gilbert/museFiles/swiping_down_60_1_gilbert.csv')[['timestamps', 'eeg_1', 'eeg_2', 'eeg_3']]
}

# Initialize dictionary to store extracted features
features = {feature: {'scrolling': {}, 'swiping': {}} for feature in ['mean', 'std', 'variance', 'skewness', 'kurtosis', 'fourier']}

# Function to apply PCA and reduce EEG signals to one component
def apply_pca(data):
    pca = PCA(n_components=1)
    return pca.fit_transform(data)

# Extract features from scrolling datasets
for name, dataset in scrolling_datasets.items():
    numeric_data = dataset[['eeg_1', 'eeg_2', 'eeg_3']].fillna(0)  # Ensure only numeric columns and fill NaN values with 0
    pca_data = apply_pca(numeric_data[['eeg_1', 'eeg_2', 'eeg_3']])  # Apply PCA to EEG1, EEG2, EEG3
    pca_data = pca_data.flatten()

    # Extract features
    features['mean']['scrolling'][name] = [np.mean(pca_data)]
    features['std']['scrolling'][name] = [np.std(pca_data)]
    features['variance']['scrolling'][name] = [np.var(pca_data)]
    features['skewness']['scrolling'][name] = [skew(pca_data)]
    features['kurtosis']['scrolling'][name] = [kurtosis(pca_data)]

    # Fourier Transform and feature extraction
    fourier_transform = fft(pca_data)
    features['fourier']['scrolling'][name] = [np.mean(np.abs(fourier_transform))]

# Extract features from swiping datasets
for name, dataset in swiping_datasets.items():
    numeric_data = dataset[['eeg_1', 'eeg_2', 'eeg_3']].fillna(0)  # Ensure only numeric columns and fill NaN values with 0
    pca_data = apply_pca(numeric_data[['eeg_1', 'eeg_2', 'eeg_3']])  # Apply PCA to EEG1, EEG2, EEG3
    pca_data = pca_data.flatten()

    # Extract features
    features['mean']['swiping'][name] = [np.mean(pca_data)]
    features['std']['swiping'][name] = [np.std(pca_data)]
    features['variance']['swiping'][name] = [np.var(pca_data)]
    features['skewness']['swiping'][name] = [skew(pca_data)]
    features['kurtosis']['swiping'][name] = [kurtosis(pca_data)]

    # Fourier Transform and feature extraction
    fourier_transform = fft(pca_data)
    features['fourier']['swiping'][name] = [np.mean(np.abs(fourier_transform))]

# Print extracted features to verify if they are meaningful
for feature_name in features.keys():
    print(f"\n{feature_name.capitalize()} Values for Scrolling:")
    for dataset_name, values in features[feature_name]['scrolling'].items():
        print(f"{dataset_name}: {values}")

    print(f"\n{feature_name.capitalize()} Values for Swiping:")
    for dataset_name, values in features[feature_name]['swiping'].items():
        print(f"{dataset_name}: {values}")

# Plot histograms to visualize if features are separable between different scrolling actions
feature_names = ['mean', 'std', 'variance', 'skewness', 'kurtosis', 'fourier']
for feature_name in feature_names:
    plt.figure(figsize=(10, 6))
    for dataset_name, feature_values in features[feature_name]['scrolling'].items():
        if len(feature_values) > 0:  # Ensure there is data to plot
            plt.hist(feature_values, bins=30, alpha=0.7, label=f'{dataset_name} - {feature_name}')
    plt.title(f'Histogram of {feature_name} for Scrolling Actions')
    plt.xlabel(f'{feature_name}')
    plt.ylabel('Frequency')
    plt.legend()
    plt.show()

# Plot histograms to visualize if features are separable between different swiping actions
for feature_name in feature_names:
    plt.figure(figsize=(10, 6))
    for dataset_name, feature_values in features[feature_name]['swiping'].items():
        if len(feature_values) > 0:  # Ensure there is data to plot
            plt.hist(feature_values, bins=30, alpha=0.7, label=f'{dataset_name} - {feature_name}')
    plt.title(f'Histogram of {feature_name} for Swiping Actions')
    plt.xlabel(f'{feature_name}')
    plt.ylabel('Frequency')
    plt.legend()
    plt.show()
