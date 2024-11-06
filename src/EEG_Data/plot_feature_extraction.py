import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import skew, kurtosis
from scipy.fft import fft

# Load datasets 
scrolling_datasets = {
    'Scrolling Down': pd.read_csv('./Jimmy/museFiles/scrolling_down_60_1_jimmy.csv')[['timestamps', 'eeg_1', 'eeg_2', 'eeg_3']],
    'Scrolling Up': pd.read_csv('./Jimmy/museFiles/scrolling_up_60_1_jimmy.csv')[['timestamps', 'eeg_1', 'eeg_2', 'eeg_3']],
}

swiping_datasets = {
    'Swipe Left': pd.read_csv('./Jimmy/museFiles/swiping_left_60_1_jimmy.csv')[['timestamps', 'eeg_1', 'eeg_2', 'eeg_3']],
    'Swipe Right': pd.read_csv('./Jimmy/museFiles/swiping_right_60_1_jimmy.csv')[['timestamps', 'eeg_1', 'eeg_2', 'eeg_3']],
    'Swipe Up': pd.read_csv('./Jimmy/museFiles/swiping_up_60_1_jimmy.csv')[['timestamps', 'eeg_1', 'eeg_2', 'eeg_3']],
    'Swipe Down': pd.read_csv('./Jimmy/museFiles/swiping_down_60_1_jimmy.csv')[['timestamps', 'eeg_1', 'eeg_2', 'eeg_3']]
}

# Initialize dictionary to store extracted features
features = {feature: {'scrolling': {}, 'swiping': {}} for feature in ['mean','std', 'variance', 'skewness', 'kurtosis', 'fourier']}

# Extract features from scrolling datasets
for name, dataset in scrolling_datasets.items():
    numeric_data = dataset[['eeg_1', 'eeg_2', 'eeg_3']].fillna(0) 
    for column in numeric_data.columns:
        data = numeric_data[column].values

        features['mean']['swiping'].setdefault(name, []).append(np.mean(data))
        features['std']['scrolling'].setdefault(name, []).append(np.std(data))
        features['variance']['scrolling'].setdefault(name, []).append(np.var(data))
        features['skewness']['scrolling'].setdefault(name, []).append(skew(data))
        features['kurtosis']['scrolling'].setdefault(name, []).append(kurtosis(data))

        fourier_transform = fft(data)
        features['fourier']['scrolling'].setdefault(name, []).append(np.mean(np.abs(fourier_transform)))

# Extract features from swiping datasets
for name, dataset in swiping_datasets.items():
    numeric_data = dataset[['eeg_1', 'eeg_2', 'eeg_3']].fillna(0)  
    for column in numeric_data.columns:
        data = numeric_data[column].values

        features['mean']['swiping'].setdefault(name, []).append(np.mean(data))
        features['std']['swiping'].setdefault(name, []).append(np.std(data))
        features['variance']['swiping'].setdefault(name, []).append(np.var(data))
        features['skewness']['swiping'].setdefault(name, []).append(skew(data))
        features['kurtosis']['swiping'].setdefault(name, []).append(kurtosis(data))

        fourier_transform = fft(data)
        features['fourier']['swiping'].setdefault(name, []).append(np.mean(np.abs(fourier_transform)))

# Plot histograms of different scrolling actions
feature_names = ['mean','std', 'variance', 'skewness', 'kurtosis', 'fourier']
for feature_name in feature_names:
    plt.figure(figsize=(10, 6))
    for dataset_name, feature_values in features[feature_name]['scrolling'].items():
        if len(feature_values) > 0:  
            plt.hist(feature_values, bins=30, alpha=0.7, label=f'{dataset_name} - {feature_name}')
    plt.title(f'Histogram of {feature_name} for Scrolling Actions')
    plt.xlabel(f'{feature_name}')
    plt.ylabel('Frequency')
    plt.legend()
    plt.show()

# Plot histograms of different swiping actions
for feature_name in feature_names:
    plt.figure(figsize=(10, 6))
    for dataset_name, feature_values in features[feature_name]['swiping'].items():
        if len(feature_values) > 0:  
            plt.hist(feature_values, bins=30, alpha=0.7, label=f'{dataset_name} - {feature_name}')
    plt.title(f'Histogram of {feature_name} for Swiping Actions')
    plt.xlabel(f'{feature_name}')
    plt.ylabel('Frequency')
    plt.legend()
    plt.show()

