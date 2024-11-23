import os

import numpy as np
import pandas as pd
from scipy.fft import fft
from scipy.stats import kurtosis, skew


def extract_fft_features(signal):
    """Extract relevant features from FFT result"""
    fft_vals = fft(signal)
    magnitudes = np.abs(fft_vals)

    return {
        "_dominant_freq": np.argmax(magnitudes),
        "_mean_freq": np.mean(magnitudes),
        "_median_freq": np.median(magnitudes),
        "_freq_std": np.std(magnitudes),
    }


def extract_features(directory, output_filepath):

    all_features = []

    for filename in os.listdir(directory):
        if filename.endswith(".csv"):
            filepath = os.path.join(directory, filename)

            df = pd.read_csv(filepath)
            df = df[["eeg_1", "eeg_2", "eeg_3", "eeg_4"]]

            print(f"Processing {filepath}, length: {len(df)}")

            file_features = {"filename": filename}

            # Determine direction label if present
            if "up" in filename.lower():
                file_features["direction"] = 0
            elif "down" in filename.lower():
                file_features["direction"] = 1

            # Extract features for each channel
            for column in df.columns:
                channel_features = {
                    f"{column}_mean": df[column].mean(),
                    f"{column}_std_dev": df[column].std(),
                    f"{column}_variance": df[column].var(),
                    f"{column}_skewness": skew(df[column]),
                    f"{column}_kurtosis": kurtosis(df[column]),
                }

                # Add FFT features
                channel_features.update(
                    {
                        f"{column}{key}": value
                        for key, value in extract_fft_features(df[column]).items()
                    }
                )

                file_features.update(channel_features)

            all_features.append(file_features)

    # Create the final dataframe
    features_df = pd.DataFrame(all_features)
    features_df.to_csv(output_filepath, index=False)
    print(f"\nFeatures extracted and saved to {output_filepath}")
