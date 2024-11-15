import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scipy.signal import butter, filtfilt, iirnotch
from scipy.stats import zscore

def clean_eeg_data(eeg_data, timestamps, sampling_rate=256):
    """
    Clean EEG data with enhanced artifact removal
    """
    def butter_bandpass(lowcut, highcut, fs, order=3):
        nyquist = 0.5 * fs
        low = lowcut / nyquist
        high = highcut / nyquist
        b, a = butter(order, [low, high], btype='band')
        return b, a

    # Adjust the values here to play around with cleaning.
    def remove_artifacts(data, threshold=2.5, min_window_size=int(0.1 * sampling_rate)):
        """
        Enhanced artifact removal with multi-scale detection
        """
        cleaned_data = data.copy()
        
        # Multi-scale window analysis
        windows = [int(0.1 * sampling_rate), int(0.5 * sampling_rate), int(1.0 * sampling_rate)]
        artifact_masks = []
        
        for window in windows:
            # Calculate rolling statistics
            rolling_mean = pd.Series(data).rolling(window=window, center=True).mean()
            rolling_std = pd.Series(data).rolling(window=window, center=True).std()
            rolling_std = rolling_std.replace(0, np.nan)
            
            # Z-score based detection
            z_scores = np.abs((data - rolling_mean) / rolling_std)
            z_scores = np.nan_to_num(z_scores, 0)
            
            # Create mask for this window size
            mask = z_scores > threshold
            artifact_masks.append(mask)
        
        # Combine masks - mark as artifact if detected in any window
        final_mask = np.any(artifact_masks, axis=0)
        
        # Expand mask to include surrounding samples
        expanded_mask = np.zeros_like(final_mask)
        for i in range(len(final_mask)):
            if final_mask[i]:
                start = max(0, i - min_window_size)
                end = min(len(final_mask), i + min_window_size)
                expanded_mask[start:end] = True
        
        # Advanced interpolation for artifacts
        artifact_segments = np.where(np.diff(expanded_mask.astype(int)))[0] + 1
        if expanded_mask[0]:
            artifact_segments = np.insert(artifact_segments, 0, 0)
        if expanded_mask[-1]:
            artifact_segments = np.append(artifact_segments, len(expanded_mask))
        
        for i in range(0, len(artifact_segments)-1, 2):
            start_idx = artifact_segments[i]
            end_idx = artifact_segments[i+1]
            
            # Get surrounding clean data points
            pre_window = slice(max(0, start_idx - min_window_size), start_idx)
            post_window = slice(end_idx, min(len(data), end_idx + min_window_size))
            
            pre_data = data[pre_window]
            post_data = data[post_window]
            
            if len(pre_data) > 0 and len(post_data) > 0:
                # Cubic spline interpolation
                x = np.concatenate([np.arange(pre_window.start, pre_window.stop),
                                  np.arange(post_window.start, post_window.stop)])
                y = np.concatenate([pre_data, post_data])
                
                # Generate interpolated values
                x_new = np.arange(start_idx, end_idx)
                cleaned_data[start_idx:end_idx] = np.interp(x_new, x, y)
        
        return cleaned_data

    # Initialize cleaned data array
    cleaned_data = np.zeros_like(eeg_data)
    
    for ch in range(eeg_data.shape[1]):
        channel_data = eeg_data[:, ch]
        
        # 1. Initial bandpass filter (1-45 Hz)
        b, a = butter_bandpass(1, 45, sampling_rate)
        filtered_data = filtfilt(b, a, channel_data)
        
        # 2. Apply notch filter at 50/60 Hz
        for freq in [50, 60]:
            b, a = iirnotch(freq, 30.0, sampling_rate)
            filtered_data = filtfilt(b, a, filtered_data)
        
        # 3. Remove artifacts with multi-scale detection
        cleaned_channel = remove_artifacts(filtered_data)
        
        # 4. Apply final bandpass filter (1-40 Hz)
        b, a = butter_bandpass(1, 40, sampling_rate)
        cleaned_channel = filtfilt(b, a, cleaned_channel)
        
        # 5. Scale the cleaned data to match original amplitude range
        p_orig = np.percentile(channel_data, [5, 95])
        p_cleaned = np.percentile(cleaned_channel, [5, 95])
        if p_cleaned[1] - p_cleaned[0] != 0:
            scale_factor = (p_orig[1] - p_orig[0]) / (p_cleaned[1] - p_cleaned[0])
            cleaned_channel = cleaned_channel * scale_factor
            cleaned_channel = cleaned_channel + (np.median(channel_data) - np.median(cleaned_channel))
        
        cleaned_data[:, ch] = cleaned_channel
    
    return cleaned_data

def load_and_prepare_data(file_path):
    """Load and prepare EEG data from CSV file with validation"""
    try:
        df = pd.read_csv(file_path)
        
        # Verify required columns exist
        required_columns = ['timestamps', 'eeg_1', 'eeg_2', 'eeg_3', 'eeg_4']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")
        
        # Convert timestamps and EEG data to numeric, replacing any non-numeric values with NaN
        df['timestamps'] = pd.to_numeric(df['timestamps'], errors='coerce')
        eeg_columns = ['eeg_1', 'eeg_2', 'eeg_3', 'eeg_4']
        for col in eeg_columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Check for NaN values
        nan_counts = df[required_columns].isna().sum()
        if nan_counts.any():
            print("Warning: Found NaN values in the following columns:")
            print(nan_counts[nan_counts > 0])
            
            # Fill NaN values with interpolation
            df[eeg_columns] = df[eeg_columns].interpolate(method='linear', limit_direction='both')
            
        timestamps = df['timestamps'].values
        eeg_data = df[eeg_columns].values
        
        # Verify data is not empty
        if len(timestamps) == 0 or eeg_data.size == 0:
            raise ValueError("No data found in the file after processing")
            
        # Check for remaining NaN values
        if np.isnan(eeg_data).any():
            raise ValueError("NaN values still present in EEG data after interpolation")
            
        return timestamps, eeg_data
        
    except Exception as e:
        print(f"Error in load_and_prepare_data: {str(e)}")
        raise

if __name__ == "__main__":
    # Specify your file path
    file_path = "./brian/museFiles/scrolling_down_20_1_brian.csv"  # Update this path
    
    try:
        # Load and prepare the data
        timestamps, eeg_data = load_and_prepare_data(file_path)
        
        # Verify data loaded correctly
        print("\nData Loading Statistics:")
        print(f"Number of samples: {len(timestamps)}")
        print(f"Number of channels: {eeg_data.shape[1]}")
        print("\nRaw Data Statistics:")
        for i in range(eeg_data.shape[1]):
            print(f"Channel {i+1} - Mean: {np.mean(eeg_data[:, i]):.2f}, Std: {np.std(eeg_data[:, i]):.2f}")
        
        # Clean the EEG data
        eeg_cleaned = clean_eeg_data(eeg_data, timestamps)
        
        # Plot original and cleaned EEG signals
        fig, axs = plt.subplots(4, 2, figsize=(15, 12))
        channel_labels = ["eeg_1", "eeg_2", "eeg_3", "eeg_4"]

        for i, label in enumerate(channel_labels):
            # Plot original signal
            axs[i, 0].plot(timestamps, eeg_data[:, i], label="Original")
            axs[i, 0].set_title(f"Original {label}")
            axs[i, 0].set_xlabel("Time (s)")
            axs[i, 0].set_ylabel("Amplitude")
            axs[i, 0].grid(True)

            # Plot cleaned signal
            axs[i, 1].plot(timestamps, eeg_cleaned[:, i], label="Cleaned", color='orange')
            axs[i, 1].set_title(f"Cleaned {label}")
            axs[i, 1].set_xlabel("Time (s)")
            axs[i, 1].set_ylabel("Amplitude")
            axs[i, 1].grid(True)

        plt.tight_layout()
        plt.show()
        
        # Print statistics to verify data integrity
        print("\nFinal Statistics:")
        for i, label in enumerate(channel_labels):
            print(f"\n{label} Statistics:")
            print(f"Original - Mean: {np.mean(eeg_data[:, i]):.2f}, Std: {np.std(eeg_data[:, i]):.2f}")
            print(f"Cleaned  - Mean: {np.mean(eeg_cleaned[:, i]):.2f}, Std: {np.std(eeg_cleaned[:, i]):.2f}")
        
    except FileNotFoundError:
        print(f"Error: Could not find the file at {file_path}")
    except Exception as e:
        print(f"An error occurred: {str(e)}")
