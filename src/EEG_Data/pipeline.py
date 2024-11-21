"""
Used for cleaning and processing the data into a format suitable for Azure machine learning models.
"""

import os

import pandas as pd
from clean_data import get_pca

directory = "./cleaned_Data"
output_filepath = "output/output.csv"
header_written = False

for filename in os.listdir(directory):
    if filename.endswith(".csv"):

        filepath = os.path.join(directory, filename)

        df = pd.read_csv(filepath)
        df = df[["eeg_1", "eeg_2", "eeg_3", "eeg_4"]]

        print(filepath, len(df))

        # Flatten into a single column
        # df = pd.DataFrame(df.values.flatten(), columns=["value"])
        df = get_pca(df)
        df = df.T
        df = df.reset_index(drop=True)

        # Determine the value based on the filename
        if "up" in filename.lower():
            df["direction"] = 0
        elif "down" in filename.lower():
            df["direction"] = 1

        df.to_csv(output_filepath, mode="a", index=False, header=not header_written)
        header_written = True

final_df = pd.read_csv(output_filepath)
final_df = final_df.sample(frac=1).reset_index(drop=True)
final_df.to_csv(output_filepath, index=False)
