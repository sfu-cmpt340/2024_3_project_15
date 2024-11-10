import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

# Load and clean the EEG data
from clean_data import clean_df

scrolling_down_eeg_df = clean_df(
    pd.read_csv("./brian/museFiles/scrolling_down_20_1_brian.csv")
)
scrolling_down_eeg_df = scrolling_down_eeg_df[
    ["timestamps", "eeg_1", "eeg_2", "eeg_3", "eeg_4"]
]

# Extract data and sampling information
timestamps = scrolling_down_eeg_df["timestamps"]
eeg_data = scrolling_down_eeg_df[["eeg_1", "eeg_2", "eeg_3", "eeg_4"]].values

# Indices where artifacts are prominent (update these as needed)
artifact_indices = [5, 6, 7, 10, 11]  # Replace with actual artifact index values

# Initialize an empty array for the cleaned EEG data
eeg_cleaned = eeg_data.copy()

# Apply regression-based artifact removal for each EEG channel
for i in range(eeg_data.shape[1]):
    # Extract the EEG channel data for artifact and non-artifact parts
    channel_data = eeg_data[:, i]
    artifact_segment = channel_data[artifact_indices]

    # Build the regressor for the artifact model
    regressor = np.ones(
        (len(artifact_segment), 1)
    )  # Constant predictor (can expand with actual model)

    # Fit the regression model to the artifact data
    model = LinearRegression().fit(regressor, artifact_segment)
    predicted_artifact = model.predict(np.ones((len(channel_data), 1)))

    # Subtract the predicted artifact from the original EEG channel data
    eeg_cleaned[:, i] = channel_data - predicted_artifact

# Plot original and cleaned EEG signals for each channel
fig, axs = plt.subplots(4, 2, figsize=(15, 12))
channel_labels = ["eeg_1", "eeg_2", "eeg_3", "eeg_4"]

for i, label in enumerate(channel_labels):
    # Plot original signal
    axs[i, 0].plot(timestamps, eeg_data[:, i], label="Original")
    axs[i, 0].set_title(f"Original {label}")
    axs[i, 0].set_xlabel("Time (s)")
    axs[i, 0].set_ylabel("Amplitude")

    # Plot cleaned signal
    axs[i, 1].plot(timestamps, eeg_cleaned[:, i], label="Cleaned", color="orange")
    axs[i, 1].set_title(f"Cleaned {label}")
    axs[i, 1].set_xlabel("Time (s)")
    axs[i, 1].set_ylabel("Amplitude")

plt.tight_layout()
plt.show()
