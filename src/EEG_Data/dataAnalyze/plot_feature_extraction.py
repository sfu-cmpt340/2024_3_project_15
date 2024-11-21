import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import skew, kurtosis
from scipy.fft import fft
import os

# List of people to process
people = ["Brian", "Jimmy", "Gilbert", "Ethan"]

# Initialize a list to store feature vectors for all people
feature_vectors = []

# Define feature names
feature_names = ["std", "variance", "skewness", "kurtosis", "fourier"]
features = {feature: {"scrolling": {}, "swiping": {}} for feature in feature_names}

# Define action types and corresponding filenames
actions = {
    "scrolling": ["scrolling_down", "scrolling_up"],
    "swiping": ["swiping_left", "swiping_right", "swiping_up", "swiping_down"],
}

# Iterate over each person
for person in people:
    print(f"Processing data for {person}...")
    # Initialize dictionaries to store datasets
    datasets = {"scrolling": {}, "swiping": {}}

    # Load datasets for this person
    for action_type, action_list in actions.items():
        for action_name in action_list:
            # Construct the filename
            filename = f"./{person}/museFiles/{action_name}_60_1_{person.lower()}.csv"
            # Check if the file exists
            if os.path.exists(filename):
                # Read the CSV file
                dataset = pd.read_csv(filename)[
                    ["timestamps", "eeg_1", "eeg_2", "eeg_3"]
                ]
                # Store in datasets dictionary
                formatted_action_name = action_name.replace("_", " ").title()
                datasets[action_type][formatted_action_name] = dataset
            else:
                print(f"File not found: {filename}")
                continue  # Skip if file does not exist

    # Combine feature extraction and mean computation
    for action_type, action_datasets in datasets.items():
        for name, dataset in action_datasets.items():
            numeric_data = dataset[["eeg_1", "eeg_2", "eeg_3"]].fillna(0)

            # Initialize a dictionary to store features for this dataset
            feature_vector = {"person": person, "action": name, "type": action_type}

            # Collect features across EEG channels
            for feature_name in feature_names:
                feature_values = []
                for column in numeric_data.columns:
                    data = numeric_data[column].values

                    if feature_name == "std":
                        value = np.std(data)
                    elif feature_name == "variance":
                        value = np.var(data)
                    elif feature_name == "skewness":
                        value = skew(data)
                    elif feature_name == "kurtosis":
                        value = kurtosis(data)
                    elif feature_name == "fourier":
                        fourier_transform = fft(data)
                        value = np.mean(np.abs(fourier_transform))

                    feature_values.append(value)

                # Store individual feature values (optional)
                features[feature_name][action_type].setdefault(name, feature_values)

                # Compute the mean of the feature values across EEG channels
                feature_mean = np.mean(feature_values)
                feature_vector[feature_name] = feature_mean

            # Append the feature vector to the list
            feature_vectors.append(feature_vector)
            # Print the feature vector (optional)
            print(feature_vector)

# Convert the list of feature vectors into a DataFrame
df_features = pd.DataFrame(feature_vectors)

# Plot histograms for scrolling actions
scrolling_features = df_features[df_features["type"] == "scrolling"]
for feature_name in feature_names:
    plt.figure(figsize=(12, 8))
    for person in scrolling_features["person"].unique():
        person_data = scrolling_features[scrolling_features["person"] == person]
        for action in person_data["action"].unique():
            data = person_data[person_data["action"] == action][feature_name]
            plt.hist(data, bins=30, alpha=0.5, label=f"{person} - {action}")
    plt.title(f"Histogram of {feature_name} for Scrolling Actions")
    plt.xlabel(f"{feature_name}")
    plt.ylabel("Frequency")
    plt.legend()
    plt.show()

# Plot histograms for swiping actions
swiping_features = df_features[df_features["type"] == "swiping"]
for feature_name in feature_names:
    plt.figure(figsize=(12, 8))
    for person in swiping_features["person"].unique():
        person_data = swiping_features[swiping_features["person"] == person]
        for action in person_data["action"].unique():
            data = person_data[person_data["action"] == action][feature_name]
            plt.hist(data, bins=30, alpha=0.5, label=f"{person} - {action}")
    plt.title(f"Histogram of {feature_name} for Swiping Actions")
    plt.xlabel(f"{feature_name}")
    plt.ylabel("Frequency")
    plt.legend()
    plt.show()

print(df_features)
