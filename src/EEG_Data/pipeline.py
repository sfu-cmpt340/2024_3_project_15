"""
Used for cleaning and processing the data into a format suitable for Azure machine learning models.
"""

import os
import pandas as pd
from clean_data import clean_df, get_pca


directory = "./museFiles"
output_filepath = "output.csv"
header_written = False

for filename in os.listdir(directory):
    if filename.endswith(".csv"):

        filepath = os.path.join(directory, filename)
        print(filepath)

        df = pd.read_csv(filepath)
        df = df[["timestamps", "eeg_1", "eeg_2", "eeg_3", "eeg_4"]]

        # Clean the data
        df = clean_df(df)
        df = get_pca(df)
        df = df.head(8000)
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
