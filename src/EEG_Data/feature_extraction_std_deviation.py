"""
Get std deviation of each signal and direction and compare:
- EEG1 up and down
- EEG2 up and down
- EEG3 up and down
"""

import pandas as pd

scrolling_down_eeg_df = pd.read_csv("./Brian/scrolling_down_60_1_Brian.csv")
scrolling_down_eeg_df = scrolling_down_eeg_df[["timestamps", "eeg_1", "eeg_2", "eeg_3"]]

scrolling_up_eeg_df = pd.read_csv("./Brian/scrolling_up_60_1_Brian.csv")
scrolling_up_eeg_df = scrolling_up_eeg_df[["timestamps", "eeg_1", "eeg_2", "eeg_3"]]

std_devs_table = pd.DataFrame(
    {
        "Up": [
            scrolling_up_eeg_df["eeg_1"].std(),
            scrolling_up_eeg_df["eeg_2"].std(),
            scrolling_up_eeg_df["eeg_3"].std(),
        ],
        "Down": [
            scrolling_down_eeg_df["eeg_1"].std(),
            scrolling_down_eeg_df["eeg_2"].std(),
            scrolling_down_eeg_df["eeg_3"].std(),
        ],
    },
    index=["EEG1", "EEG2", "EEG3"],
)
print(std_devs_table)
