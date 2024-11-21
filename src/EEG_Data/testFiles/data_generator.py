import pandas as pd
import numpy as np
from scipy.stats import skew, kurtosis
from scipy.fft import fft
import random
import matplotlib.pyplot as plt
import os

def extract_features(df):
    """
    Extracts statistical features from a DataFrame with EEG data.

    Parameters:
    df (DataFrame): DataFrame containing EEG data with columns ['eeg_1', 'eeg_2', 'eeg_3']

    Returns:
    dict: Dictionary containing feature values for each column
    """
    features = {}
    for col in ['eeg_1', 'eeg_2', 'eeg_3']:
        data = df[col].values
        features[f'{col}_std'] = np.std(data)
        features[f'{col}_variance'] = np.var(data)
        features[f'{col}_skewness'] = skew(data)
        features[f'{col}_kurtosis'] = kurtosis(data)
        features[f'{col}_fourier_mean'] = np.mean(np.abs(fft(data)))
    return features

def generate_symmetric_data(df):
    """
    Generate symmetric EEG data based on the input DataFrame.

    Parameters:
    df (DataFrame): Input DataFrame with EEG data

    Returns:
    DataFrame: Symmetrically transformed DataFrame
    """
    symmetric_df = df.copy()
    for col in ['eeg_1', 'eeg_2', 'eeg_3']:
        symmetric_df[col] = -df[col]  # Invert signal values for symmetry
    return symmetric_df

def add_random_noise(df, noise_level=0.05):
    """
    Add random noise to EEG data.

    Parameters:
    df (DataFrame): Input DataFrame with EEG data
    noise_level (float): Proportion of the signal's standard deviation to use as noise

    Returns:
    DataFrame: DataFrame with added noise
    """
    noisy_df = df.copy()
    for col in ['eeg_1', 'eeg_2', 'eeg_3']:
        noise = np.random.normal(0, noise_level * df[col].std(), size=len(df))
        noisy_df[col] += noise
    return noisy_df

def visualize_data(file_path):
    """
    Visualize EEG data from a file using appropriate visualization based on file name.

    Parameters:
    file_path (str): Path to the EEG data file
    """
    df = pd.read_csv(file_path)
    if 'scrolling' in file_path.lower():
        print("Visualizing scrolling EEG data...")
        visualize_scrolling_eeg(df)
    elif 'swiping' in file_path.lower():
        print("Visualizing swiping EEG data...")
        visualize_swiping_eeg(df)
    else:
        print("File type not recognized for visualization.")

def visualize_scrolling_eeg(df):
    """
    Visualize scrolling EEG data.

    Parameters:
    df (DataFrame): DataFrame containing scrolling EEG data
    """
    df['seconds_elapsed'] = (
        pd.to_datetime(df['timestamps'], unit='s') - pd.to_datetime(df['timestamps'], unit='s').iloc[0]
    ).dt.total_seconds()

    plt.figure(figsize=(12, 6))
    for col in ['eeg_1', 'eeg_2', 'eeg_3']:
        plt.plot(df['seconds_elapsed'], df[col], label=col)
    plt.title("EEG Signals Over Time (Scrolling)")
    plt.xlabel("Time (seconds)")
    plt.ylabel("EEG Signal")
    plt.legend()
    plt.grid(True)
    plt.show()

def visualize_swiping_eeg(df):
    """
    Visualize swiping EEG data.

    Parameters:
    df (DataFrame): DataFrame containing swiping EEG data
    """
    df['seconds_elapsed'] = (
        pd.to_datetime(df['timestamps'], unit='s') - pd.to_datetime(df['timestamps'], unit='s').iloc[0]
    ).dt.total_seconds()

    plt.figure(figsize=(12, 6))
    for col in ['eeg_1', 'eeg_2', 'eeg_3']:
        plt.plot(df['seconds_elapsed'], df[col], label=col)
    plt.title("EEG Signals Over Time (Swiping)")
    plt.xlabel("Time (seconds)")
    plt.ylabel("EEG Signal")
    plt.legend()
    plt.grid(True)
    plt.show()

def generate_synthetic_data_from_sample(input_file, noise_level=0.05):
    """
    Generate a synthetic EEG data sample based on an input sample file.

    Parameters:
    input_file (str): Path to the input sample file
    noise_level (float): Proportion of the signal's standard deviation to use as noise
    """
    # Load input data
    df = pd.read_csv(input_file)

    # Ensure required columns exist
    if not set(['eeg_1', 'eeg_2', 'eeg_3']).issubset(df.columns):
        raise ValueError("Input file must contain 'eeg_1', 'eeg_2', and 'eeg_3' columns.")

    # Generate symmetric data
    symmetric_df = generate_symmetric_data(df)

    # Add random noise
    noisy_symmetric_df = add_random_noise(symmetric_df, noise_level)

    # Generate output file name based on input file name
    file_name = os.path.basename(input_file)
    name_parts = file_name.split('_')
    name_parts[-2] = str(int(name_parts[-2]) + 1)  # Increment the trial number
    output_file = f"./testFiles/{'_'.join(name_parts).replace('.csv', '_test.csv')}"

    # Save the synthetic data to a new file
    noisy_symmetric_df.to_csv(output_file, index=False)
    print(f"Synthetic data saved to {output_file}")

    # Visualize the newly created synthetic data
    visualize_data(output_file)

# Example usage
input_file = '../EEG_Data/museFiles/scrolling_down_20_2_jimmy.csv'
generate_synthetic_data_from_sample(input_file, noise_level=0.1)
