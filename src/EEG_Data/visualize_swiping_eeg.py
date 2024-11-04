import matplotlib.pyplot as plt
import pandas as pd


def load_eeg_data(filename):
    df = pd.read_csv(filename)
    df = df[["timestamps", "eeg_1", "eeg_2", "eeg_3"]]
    df["timestamps"] = pd.to_datetime(df["timestamps"], unit="s")
    df["seconds_elapsed"] = (
        df["timestamps"] - df["timestamps"].iloc[0]
    ).dt.total_seconds()
    return df


swiping_up_eeg_df = load_eeg_data("./Gilbert/museFiles/swiping_up_60_1_gilbert.csv")
swiping_left_eeg_df = load_eeg_data("./Gilbert/museFiles/swipe_left_60_1_gilbert.csv")
swiping_down_eeg_df = load_eeg_data("./Gilbert/museFiles/swiping_down_60_1_gilbert.csv")
swiping_right_eeg_df = load_eeg_data("./Gilbert/museFiles/swiping_right_60_1_gilbert.csv")

fig, axes = plt.subplots(2, 2, figsize=(14, 12), sharex=True)

titles = ["Swiping Up", "Swiping Left", "Swiping Down", "Swiping Right"]
dataframes = [
    swiping_up_eeg_df,
    swiping_left_eeg_df,
    swiping_down_eeg_df,
    swiping_right_eeg_df,
]


for ax, df, title in zip(axes.flatten(), dataframes, titles):
    ax.plot(df["seconds_elapsed"], df["eeg_1"], color = "blue", label="EEG 1")
    ax.plot(df["seconds_elapsed"], df["eeg_2"], color = "orange", label="EEG 2")
    ax.plot(df["seconds_elapsed"], df["eeg_3"], color = "green", label="EEG 3")
    ax.set_title(f"EEG Signals Over Time ({title})")
    ax.set_ylabel("EEG Signal")
    ax.legend()
    ax.grid(True)


fig.text(0.5, 0.04, "Time (seconds)", ha="center")
plt.tight_layout()
plt.show()
