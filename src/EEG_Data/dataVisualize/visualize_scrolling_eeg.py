import matplotlib.pyplot as plt
import pandas as pd
from clean_data import clean_df

# Set threshold based on observation and analysis of data
min_threshold = 0
max_threshold = 2000

# Read EEG data for scrolling down
scrolling_down_eeg_df = clean_df(
    pd.read_csv("./Jimmy/museFiles/scrolling_down_20_1_jimmy.csv")
)
scrolling_down_eeg_df = scrolling_down_eeg_df[["timestamps", "eeg_1", "eeg_2", "eeg_3"]]
scrolling_down_eeg_df["timestamps"] = pd.to_datetime(
    scrolling_down_eeg_df["timestamps"], unit="s"
)
scrolling_down_eeg_df["seconds_elapsed"] = (
    scrolling_down_eeg_df["timestamps"] - scrolling_down_eeg_df["timestamps"].iloc[0]
).dt.total_seconds()

# Read EEG data for scrolling up
scrolling_up_eeg_df = clean_df(
    pd.read_csv("./Jimmy/museFiles/scrolling_up_20_1_jimmy.csv")
)
scrolling_up_eeg_df = scrolling_up_eeg_df[["timestamps", "eeg_1", "eeg_2", "eeg_3"]]
scrolling_up_eeg_df["timestamps"] = pd.to_datetime(
    scrolling_up_eeg_df["timestamps"], unit="s"
)
scrolling_up_eeg_df["seconds_elapsed"] = (
    scrolling_up_eeg_df["timestamps"] - scrolling_up_eeg_df["timestamps"].iloc[0]
).dt.total_seconds()

# Filter for EEG values within threshold for each EEG column
scrolling_down_eeg_df = scrolling_down_eeg_df[
    (scrolling_down_eeg_df["eeg_1"].between(min_threshold, max_threshold))
    & (scrolling_down_eeg_df["eeg_2"].between(min_threshold, max_threshold))
    & (scrolling_down_eeg_df["eeg_3"].between(min_threshold, max_threshold))
]

# Apply threshold filter for scrolling up
scrolling_up_eeg_df = scrolling_up_eeg_df[
    (scrolling_up_eeg_df["eeg_1"].between(min_threshold, max_threshold))
    & (scrolling_up_eeg_df["eeg_2"].between(min_threshold, max_threshold))
    & (scrolling_up_eeg_df["eeg_3"].between(min_threshold, max_threshold))
]


fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)

ax1.plot(
    scrolling_down_eeg_df["seconds_elapsed"],
    scrolling_down_eeg_df["eeg_1"],
    label="EEG 1",
    color="blue",
)
ax1.plot(
    scrolling_down_eeg_df["seconds_elapsed"],
    scrolling_down_eeg_df["eeg_2"],
    label="EEG 2",
    color="orange",
)
ax1.plot(
    scrolling_down_eeg_df["seconds_elapsed"],
    scrolling_down_eeg_df["eeg_3"],
    label="EEG 3",
    color="green",
)
ax1.set_title("EEG Signals Over Time (Scrolling down)")
ax1.set_ylabel("EEG Signal")
ax1.legend()
ax1.grid(True)

ax2.plot(
    scrolling_up_eeg_df["seconds_elapsed"],
    scrolling_up_eeg_df["eeg_1"],
    color="blue",
    label="EEG 1",
)
ax2.plot(
    scrolling_up_eeg_df["seconds_elapsed"],
    scrolling_up_eeg_df["eeg_2"],
    color="orange",
    label="EEG 2",
)
ax2.plot(
    scrolling_up_eeg_df["seconds_elapsed"],
    scrolling_up_eeg_df["eeg_3"],
    color="green",
    label="EEG 3",
)
ax2.set_title("EEG Signals Over Time (Scrolling up)")
ax2.set_xlabel("Time (seconds)")
ax2.set_ylabel("EEG Signal")
ax2.legend()
ax2.grid(True)

plt.tight_layout()
plt.show()
