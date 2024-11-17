import os

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view
from scipy.fft import fft
from scipy.integrate import simpson
from scipy.signal import butter, filtfilt, hilbert, welch
from scipy.stats import kurtosis, skew


def extract_time_domain_features(signal):
    """Extract time domain features from the signal"""
    # Basic statistical features
    basic_stats = {
        "mean": np.mean(signal),
        "std": np.std(signal),
        "var": np.var(signal),
        "skewness": skew(signal),
        "kurtosis": kurtosis(signal),
        "max": np.max(signal),
        "min": np.min(signal),
        "range": np.ptp(signal),
        "rms": np.sqrt(np.mean(np.square(signal))),
        "zero_crossings": np.sum(np.diff(np.signbit(signal).astype(int))),
        "mean_abs": np.mean(np.abs(signal)),
        "median_abs": np.median(np.abs(signal)),
        "energy": np.sum(np.square(signal)),
        "iqr": np.percentile(signal, 75) - np.percentile(signal, 25),
        "percentile_25": np.percentile(signal, 25),
        "percentile_75": np.percentile(signal, 75),
    }

    # Hjorth parameters
    diff_first = np.diff(signal)
    diff_second = np.diff(diff_first)

    activity = np.var(signal)
    mobility = np.sqrt(np.var(diff_first) / np.var(signal))
    complexity = np.sqrt(np.var(diff_second) * np.var(signal)) / np.var(diff_first)

    hjorth = {"activity": activity, "mobility": mobility, "complexity": complexity}

    return {**basic_stats, **hjorth}


def extract_frequency_domain_features(signal, fs=250):  # Assuming 250Hz sampling rate
    """Extract frequency domain features from the signal"""
    # FFT features
    fft_vals = fft(signal)
    magnitudes = np.abs(fft_vals)[: len(signal) // 2]
    frequencies = np.fft.fftfreq(len(signal), 1 / fs)[: len(signal) // 2]

    # Power Spectral Density using Welch's method
    freqs, psd = welch(signal, fs=fs)

    # Calculate frequency bands power
    bands = {
        "delta": (0.5, 4),
        "theta": (4, 8),
        "alpha": (8, 13),
        "beta": (13, 30),
        "gamma": (30, 100),
    }

    band_powers = {}
    for band, (low, high) in bands.items():
        freq_mask = (freqs >= low) & (freqs <= high)
        band_power = simpson(y=psd[freq_mask], x=freqs[freq_mask])
        band_powers[f"{band}_power"] = band_power

        # Add band relative power
        total_power = simpson(y=psd, x=freqs)
        band_powers[f"{band}_relative_power"] = band_power / total_power

    # Spectral edge frequency (95% of power)
    total_power = np.cumsum(psd)
    total_power_norm = total_power / total_power[-1]
    sef_95 = freqs[np.where(total_power_norm >= 0.95)[0][0]]

    return {
        "dominant_freq": frequencies[np.argmax(magnitudes)],
        "mean_freq": np.average(frequencies, weights=magnitudes),
        "median_freq": np.median(frequencies),
        "freq_std": np.std(magnitudes),
        "spectral_edge_freq": sef_95,
        "spectral_entropy": -np.sum(psd * np.log2(psd + 1e-16)) / np.log2(len(psd)),
        **band_powers,
    }


def extract_connectivity_features(signals_dict):
    """Extract connectivity features between channels"""
    channels = list(signals_dict.keys())
    connectivity = {}

    for i, ch1 in enumerate(channels):
        for j, ch2 in enumerate(channels):
            if i < j:  # Avoid redundant calculations
                # Correlation
                correlation = np.corrcoef(signals_dict[ch1], signals_dict[ch2])[0, 1]
                connectivity[f"corr_{ch1}_{ch2}"] = correlation

                # Coherence (simplified)
                f1, psd1 = welch(signals_dict[ch1])
                f2, psd2 = welch(signals_dict[ch2])
                coherence = np.mean(np.abs(np.correlate(psd1, psd2)))
                connectivity[f"coherence_{ch1}_{ch2}"] = coherence

                # Phase synchronization
                analytic1 = hilbert(signals_dict[ch1])
                analytic2 = hilbert(signals_dict[ch2])
                phase_diff = np.angle(analytic1) - np.angle(analytic2)
                phase_sync = np.abs(np.mean(np.exp(1j * phase_diff)))
                connectivity[f"phase_sync_{ch1}_{ch2}"] = phase_sync

    return connectivity


def extract_nonlinear_features(signal, embed_dim=3, delay=1):
    """Extract nonlinear features from the signal"""

    # Sample Entropy (simplified version)
    def sample_entropy(x, m, r):
        N = len(x)
        templates = sliding_window_view(x, m)
        A = np.sum(
            [np.sum(np.max(np.abs(templates - t), axis=1) < r) for t in templates]
        )
        B = np.sum(
            [
                np.sum(np.max(np.abs(templates[:-1] - t), axis=1) < r)
                for t in templates[:-1]
            ]
        )
        return -np.log(A / B)

    # Approximate Entropy (simplified version)
    def approx_entropy(x, m, r):
        N = len(x)
        templates = sliding_window_view(x, m)
        phi = np.mean(
            [
                np.log(np.sum(np.max(np.abs(templates - t), axis=1) < r))
                for t in templates
            ]
        )
        return phi

    signal_std = np.std(signal)
    tolerance = 0.2 * signal_std  # Common choice for tolerance

    return {
        "sample_entropy": sample_entropy(signal, embed_dim, tolerance),
        "approx_entropy": approx_entropy(signal, embed_dim, tolerance),
        "lyapunov_exp": np.mean(np.diff(np.log(np.abs(np.diff(signal))))),
        "hurst_exp": np.mean(np.abs(signal - np.mean(signal))) / np.std(signal),
    }


def extract_all_features(df):
    """Extract all features from the dataframe"""
    all_features = {}

    # Extract features for each channel
    for column in df.columns:
        signal = df[column].values

        # Time domain features
        time_features = extract_time_domain_features(signal)
        all_features.update({f"{column}_{k}": v for k, v in time_features.items()})

        # Frequency domain features
        freq_features = extract_frequency_domain_features(signal)
        all_features.update({f"{column}_{k}": v for k, v in freq_features.items()})

        # Nonlinear features
        nonlinear_features = extract_nonlinear_features(signal)
        all_features.update({f"{column}_{k}": v for k, v in nonlinear_features.items()})

    # Connectivity features between channels
    signals_dict = {col: df[col].values for col in df.columns}
    connectivity_features = extract_connectivity_features(signals_dict)
    all_features.update(connectivity_features)

    return all_features


directory = "./Cleaned_Data"
output_filepath = "output.csv"
all_features = []

for filename in os.listdir(directory):
    if filename.endswith(".csv"):
        filepath = os.path.join(directory, filename)
        df = pd.read_csv(filepath)
        df = df[["eeg_1", "eeg_2", "eeg_3", "eeg_4"]]

        print(f"Processing {filepath}, length: {len(df)}")

        # Extract base features
        file_features = {"filename": filename}

        # Determine direction label if present
        if "up" in filename.lower():
            file_features["direction"] = 0
        elif "down" in filename.lower():
            file_features["direction"] = 1

        # Extract all features
        features = extract_all_features(df)
        file_features.update(features)

        all_features.append(file_features)

# Create the final dataframe
features_df = pd.DataFrame(all_features)
features_df.to_csv(output_filepath, index=False)
print(f"\nFeatures extracted and saved to {output_filepath}")
print("\nFirst few rows of extracted features:")
print(features_df.head())
