import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

# Load the dataset
df = pd.read_csv("../output/output.csv")
df = df.drop(
    columns=[
        "filename",
        "eeg_1_dominant_freq",
        "eeg_2_dominant_freq",
        "eeg_3_dominant_freq",
        "eeg_4_dominant_freq",
    ]
)
df = df[[col for col in df.columns if col != "direction"] + ["direction"]]

# Define features and target
x = df.iloc[:, :-1]
y = df.iloc[:, -1]

# Testing random states from 1 to 100
best_random_state = None
best_accuracy = 0
results = []

for random_state in range(1, 101):
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, train_size=0.8, test_size=0.2, random_state=random_state
    )

    clf = RandomForestClassifier(
        criterion="gini", max_depth=8, min_samples_split=10, random_state=random_state
    )

    clf.fit(x_train, y_train)
    y_pred = clf.predict(x_test)
    accuracy = accuracy_score(y_test, y_pred)

    # Store the results
    results.append((random_state, accuracy))

    if accuracy > best_accuracy:
        best_accuracy = accuracy
        best_random_state = random_state

# Display all results
results_df = pd.DataFrame(results, columns=["Random State", "Accuracy"])
print(results_df.sort_values(by="Accuracy", ascending=False))
average_accuracy = sum([result[1] for result in results]) / len(results)
print("Average Accuracy:", average_accuracy)
