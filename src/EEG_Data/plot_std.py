import pandas as pd
from pathlib import Path
from clean_data import clean_df
import matplotlib.pyplot as plt
import numpy as np

search_dir = Path("./Jimmy/museFiles")

# Get all scrolling down files
scrolling_down_files = []
for file_path in search_dir.glob("*scrolling_down_20*"):
    scrolling_down_files.append(str(file_path))

# Get all scrolling up files
scrolling_up_files = []
for file_path in search_dir.glob("*scrolling_up_20*"):
    scrolling_up_files.append(str(file_path))

# Load every scrolling down data and clean
scrolling_down_dfs = []
for file in scrolling_down_files:
    df = clean_df(pd.read_csv(file))
    df = df[["timestamps", "eeg_1", "eeg_2", "eeg_3"]]
    scrolling_down_dfs.append(df)

# Load every scrolling up data and clean
scrolling_up_dfs = []
for file in scrolling_up_files:
    df = clean_df(pd.read_csv(file))
    df = df[["timestamps", "eeg_1", "eeg_2", "eeg_3"]]
    scrolling_up_dfs.append(df)

# Calculate standard deviations for scrolling down
std_down = []
for df in scrolling_down_dfs:
    std_dict = {
        "direction": "Down",
        "eeg1_std": df["eeg_1"].std(),
        "eeg2_std": df["eeg_2"].std(),
        "eeg3_std": df["eeg_3"].std(),
    }
    std_down.append(std_dict)

# Calculate standard deviations for scrolling up
std_up = []
for df in scrolling_up_dfs:
    std_dict = {
        "direction": "Up",
        "eeg1_std": df["eeg_1"].std(),
        "eeg2_std": df["eeg_2"].std(),
        "eeg3_std": df["eeg_3"].std(),
    }
    std_up.append(std_dict)

# Convert to dataframes for easier analysis
std_down_df = pd.DataFrame(std_down)
std_up_df = pd.DataFrame(std_up)

# Calculate averages for each signal and direction
down_avgs = {
    "eeg1": std_down_df["eeg1_std"].mean(),
    "eeg2": std_down_df["eeg2_std"].mean(),
    "eeg3": std_down_df["eeg3_std"].mean(),
}

up_avgs = {
    "eeg1": std_up_df["eeg1_std"].mean(),
    "eeg2": std_up_df["eeg2_std"].mean(),
    "eeg3": std_up_df["eeg3_std"].mean(),
}

# Set up the plot style

fig, ax = plt.subplots(figsize=(14, 6))  # Increased figure width for labels

# Set the width of each bar and positions of the bars
width = 0.15
trial_positions = np.arange(len(std_down_df))

# Calculate x-axis spans for the dotted lines
down_span = [min(trial_positions) - width * 2, max(trial_positions) + width]
up_span = [
    min(trial_positions) + len(std_down_df),
    max(trial_positions) + len(std_down_df) + width * 4,
]

# Create bars for scrolling down
ax.bar(
    trial_positions - width * 1.5,
    std_down_df["eeg1_std"],
    width,
    label="EEG1 Down",
    color="royalblue",
    alpha=0.7,
)
ax.bar(
    trial_positions - width / 2,
    std_down_df["eeg2_std"],
    width,
    label="EEG2 Down",
    color="lightblue",
    alpha=0.7,
)
ax.bar(
    trial_positions + width / 2,
    std_down_df["eeg3_std"],
    width,
    label="EEG3 Down",
    color="navy",
    alpha=0.7,
)

# Create bars for scrolling up
ax.bar(
    trial_positions + len(std_down_df) + width * 1.5,
    std_up_df["eeg1_std"],
    width,
    label="EEG1 Up",
    color="indianred",
    alpha=0.7,
)
ax.bar(
    trial_positions + len(std_down_df) + width * 2.5,
    std_up_df["eeg2_std"],
    width,
    label="EEG2 Up",
    color="lightcoral",
    alpha=0.7,
)
ax.bar(
    trial_positions + len(std_down_df) + width * 3.5,
    std_up_df["eeg3_std"],
    width,
    label="EEG3 Up",
    color="darkred",
    alpha=0.7,
)


# Function to add labeled dotted lines
def add_labeled_hline(y, xmin, xmax, color, label, position="right"):
    ax.hlines(y=y, xmin=xmin, xmax=xmax, colors=color, linestyles="dotted", linewidth=2)
    if position == "right":
        ax.text(
            xmax + 0.1,
            y,
            f"{label}: {y:.2f}",
            color=color,
            verticalalignment="center",
            fontsize=9,
        )
    else:
        ax.text(
            xmin - 0.1,
            y,
            f"{label}: {y:.2f}",
            color=color,
            verticalalignment="center",
            horizontalalignment="right",
            fontsize=9,
        )


# Add labeled dotted lines for averages - Scrolling Down
add_labeled_hline(
    down_avgs["eeg1"], down_span[0], down_span[1], "royalblue", "EEG1 Down Avg", "left"
)
add_labeled_hline(
    down_avgs["eeg2"], down_span[0], down_span[1], "lightblue", "EEG2 Down Avg", "left"
)
add_labeled_hline(
    down_avgs["eeg3"], down_span[0], down_span[1], "navy", "EEG3 Down Avg", "left"
)

# Add labeled dotted lines for averages - Scrolling Up
add_labeled_hline(up_avgs["eeg1"], up_span[0], up_span[1], "indianred", "EEG1 Up Avg")
add_labeled_hline(up_avgs["eeg2"], up_span[0], up_span[1], "lightcoral", "EEG2 Up Avg")
add_labeled_hline(up_avgs["eeg3"], up_span[0], up_span[1], "darkred", "EEG3 Up Avg")

# Customize the plot
ax.set_ylabel("Standard Deviation (μV)")
ax.set_title("EEG Signal Standard Deviations: Scrolling Up vs Down")
ax.set_xticks(np.concatenate([trial_positions, trial_positions + len(std_down_df)]))
ax.set_xticklabels(
    [
        *[f"Down Trial {i+1}" for i in range(len(std_down_df))],
        *[f"Up Trial {i+1}" for i in range(len(std_up_df))],
    ]
)

# Add legend
ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left")

# Rotate x-axis labels for better readability
plt.xticks(rotation=45)

# Adjust plot margins to accommodate labels
plt.subplots_adjust(left=0.1, right=0.85)

# Show the plot
plt.show()

# Print the averages
print("\nAverage Standard Deviations for Scrolling Down:")
print(pd.DataFrame([down_avgs]).round(3))
print("\nAverage Standard Deviations for Scrolling Up:")
print(pd.DataFrame([up_avgs]).round(3))
